# ==============================================================================
# Project PULSAR-ASM | tools/verify_gemma4_blob.py
# ------------------------------------------------------------------------------
# Independent verifier for a converted weight blob.
#
# In-line conversion checks are not evidence: the converter once omitted the
# safetensors header size from every range request, shifting all 9.26 GB by
# -263,960 bytes, and its own probe - repeating the same omission - reported
# "verified byte-for-byte". This script re-derives the source offsets from
# scratch (its own header parse, its own base) so a shared mistake is unlikely,
# and adds SEMANTIC checks that no byte comparison can make: RMSNorm weights
# must look like RMSNorm weights, not like someone else's bf16 mantissas.
#
# Usage:
#   python verify_gemma4_blob.py --blob /mnt/edge/pulsar/gemma4_e2b.bin [--sample 16]
# ==============================================================================

import argparse
import json
import os
import struct
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_gemma4_safetensors import HF, follow, hf_token, http_get  # noqa: E402


def bf16_to_f32(buf):
    u = np.frombuffer(buf[: len(buf) // 2 * 2], dtype=np.uint16)
    return (u.astype(np.uint32) << 16).view(np.float32).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blob", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--repo", default="google/gemma-4-E2B-it")
    ap.add_argument("--sample", type=int, default=16, help="large blobs to sample")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    man = json.load(open(a.manifest or a.blob + ".manifest.json"))
    lay = man["layout"]
    size = os.path.getsize(a.blob)
    if size != man["bytes"]:
        raise SystemExit(f"blob is {size} bytes, manifest says {man['bytes']}")
    print(f"blob {size/1e9:.3f} GB / {len(lay)} blobs  (manifest agrees)")

    tok = hf_token()
    url = man.get("source_url") or f"{HF}/{a.repo}/resolve/main/model.safetensors"
    if "resolve/main" in url:
        url, _ = follow(url, tok)
    head = http_get(url, (0, 1 << 20), tok)
    n_hdr = struct.unpack("<Q", head[:8])[0]
    base = 8 + n_hdr                      # tensor data starts here
    hdr = json.loads(head[8:8 + n_hdr].decode())
    tag = url.split("/resolve/")[1][:12] if "/resolve/" in url else url.split("/" * 2)[1][:40]
    print(f"source ...{tag}  header {base:,} B")

    fd = os.open(a.blob, os.O_RDONLY)
    jobs = []
    for name, e in lay.items():
        if e.get("kind") == "none" or "src" not in e:
            continue          # placeholders (KV-shared layers have no k/v tensors)
        s = hdr[e["src"]]
        lo, hi = s["data_offsets"]
        if e["kind"] == "raw":
            conv, nbytes = False, hi - lo
        else:
            conv = s["dtype"] == "BF16"
            nbytes = (hi - lo) if conv else (hi - lo) * 2
        span = hi - lo                     # source bytes
        full = span <= (1 << 20)
        take = min(span, 1 << 16) if not full else span
        jobs.append((name, lo + base, lo + base + take - 1, e["offset"], conv, take,
                     nbytes, full))
    small_full = sum(1 for j in jobs if j[7])
    big = [j for j in jobs if not j[7]]
    keep = [j for j in jobs if j[7]]
    step = max(1, len(big) // max(1, a.sample))
    keep += big[::step][:a.sample]
    print(f"verifying {len(keep)} blobs ({small_full:,} exhaustively, "
          f"{min(len(big), a.sample)} of {len(big)} large ones sampled)")

    bad = []

    def check(j):
        name, lo, hi, dst, conv, take, nbytes, full = j
        src = http_get(url, (lo, hi), tok)
        if conv:
            src = bf16_to_f32(src)
        got = os.pread(fd, len(src), dst)
        if len(src) != len(got) or src != got:
            bad.append(name)
            return False
        return True

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(check, keep))
    if bad:
        for b in bad[:12]:
            print("  MISMATCH", b, lay[b]["src"])
        raise SystemExit(f"{len(bad)} of {len(keep)} blobs disagree with the source")
    print("  byte-for-byte OK")

    # ---- semantic sanity: values must look like the tensors they claim to be --
    def stats(names, label, pred):
        vals = []
        for nm in names:
            e = lay[nm]
            if e["kind"] != "f32":
                continue
            b = os.pread(fd, int(np.prod(e["shape"])) * 4, e["offset"])
            a_ = np.frombuffer(b, dtype=np.float32)
            if not np.isfinite(a_).all():
                bad.append(f"{nm}: non-finite")
            vals.append(a_)
        if not vals:
            return
        allv = np.concatenate(vals)
        ok = pred(allv)
        print(f"  {label:22s} n={allv.size:>10,}  mean|w|={np.abs(allv).mean():8.4f} "
              f" min={allv.min():8.3f} max={allv.max():8.3f}  {'OK' if ok else 'SUSPECT'}")
        if not ok:
            bad.append(f"{label} looks wrong")

    norms = [k for k in lay if k.startswith("L") and "norm" in k]
    scal = [k for k in lay if k.endswith("layer_scalar")]
    # RMSNorm weights in a trained checkpoint are order 1 (this model's are large
    # but strictly positive-ish and finite); random bf16 from elsewhere in the
    # file would sit around zero with both signs and far smaller magnitude.
    stats(norms, "rmsnorm weights", lambda v: np.abs(v).mean() > 0.05 and np.isfinite(v).all())
    stats(scal, "layer_scalar", lambda v: np.abs(v).max() < 100 and np.isfinite(v).all())

    e = lay["embed"]
    row = np.frombuffer(os.pread(fd, e["shape"][1] * 2, e["offset"]), dtype=np.uint16)
    row = ((row.astype(np.uint32) << 16).view(np.float32)).astype(np.float64)
    print(f"  {'embed row 0':22s} std={row.std():.5f} max|.|={np.abs(row).max():.4f}")
    if not (0.001 < row.std() < 10.0):
        bad.append("embed row 0 has implausible statistics")

    os.close(fd)
    if bad:
        for b in bad:
            print("  FAIL", b)
        raise SystemExit("blob verification FAILED")
    print("VERIFIED: blob matches the checkpoint and its values are plausible")


if __name__ == "__main__":
    main()
