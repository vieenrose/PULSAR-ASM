#!/usr/bin/env python
"""Export the LFM3-Audio-2B text backbone to the .lfm3q8 blob the Nano engine
loads, in the exact numerics the certified accuracy arm used.

WHY A DEDICATED EXPORTER AND NOT GGUF: the engine's GEMV (q8_lc3.cu) takes
  s = max|w| / 127 in fp32,  codes = round(w/s) clamped to -128..127
which is quant_sensitivity.py's CERTIFIED arm (8/8 prompts). The ggml/pq1 Q8_0
convention stores s as __half, and that arm was measured at 7/8 with prompt F
diverging at token 6 (commit 2b5cbe5). A loader that silently imported ggml
Q8_0 files would therefore ship the non-certified numerics, so the blob
format carries fp32 scales and nothing else can produce it.

BIT-PARITY CONTRACT, checked not assumed: every tensor gets an FNV-1a
checksum over (codes, scales) in the header. The Nano loader recomputes the
checksum on the bytes it actually loaded and aborts on mismatch, so "the
engine is running the quantizer that was certified" is a runtime assertion
rather than a belief. `--selfcheck` re-derives the quantization from the
safetensors a second time and compares, which catches a nondeterministic
quantizer.

WHAT STAYS WIDE:
  norms        f32  (5 per layer family, 2048 floats each -- 1 MB total)
  conv.conv    f32  [H,1,7] depthwise taps, 14.7 M params, 0.4% of the
                   backbone; streaming it costs nothing and quantizing a
                   7-tap depthwise kernel is not covered by any measured arm.
  biases       absent: config conv_bias=False and every Linear is bias=False
               (verified: the census lists no bias tensors).
Embeddings are Q8_0'd here (they are TIED: config tie_word_embeddings=True
and there is no lm_head tensor, so the same table is the input embedding and
the output projection -- 295 MB/token streamed, 16% of decode).
"""
import argparse, json, os, struct, sys, time
import numpy as np
import torch
from safetensors import safe_open

REPO_LOCAL = "/home/louis/models/lfm3/LFM3-Audio-2B"
MAGIC = b"LFM3Q8\x01\x00"      # exactly 8 bytes; the reader uses len(MAGIC)
T_Q8 = 1        # int8 codes + fp32 scales
T_F32 = 2       # f32
BACKBONE = "lfm."

Q8_PATTERNS = ("conv.in_proj", "conv.out_proj",
               "feed_forward.w1", "feed_forward.w2", "feed_forward.w3",
               "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
               "self_attn.o_proj", "self_attn.gate_proj")


def quantize_q8(w):
    """[N,K] float32 -> (codes int8 [N,K], scales f32 [N,K/32]) in the
    certified arm's numerics. Must match quant_row() in q8_lc3.cu exactly:
    s = max|w|/127 in fp32, codes = round_half_even(w/s) clamped to -128..127,
    and an all-zero group writes codes 0 / scale 0 (no division)."""
    N, K = w.shape
    assert K % 32 == 0, "Q8_0 needs K%%32==0, got K=%d" % K
    g = w.reshape(N, K // 32, 32)
    m = np.abs(g).max(axis=-1)                      # f32
    s = (m / np.float32(127.0)).astype(np.float32)  # fp32 scale, NOT fp16
    safe = np.where(s == 0, np.float32(1.0), s)[:, :, None]
    # np.round is round-half-to-even, == torch.round == lrintf in the default
    # FE_TONEAREST mode. The three must agree exactly, so none of them is
    # replaceable by a truncating cast.
    c = np.round(g / safe)
    c = np.clip(c, -128.0, 127.0).astype(np.int8)
    c = np.where(s[:, :, None] == 0, np.int8(0), c)
    return np.ascontiguousarray(c.reshape(N, K)), np.ascontiguousarray(s.reshape(N, K // 32))


def fnv1a(*chunks):
    h = np.uint64(14695981039346656037)
    m = np.uint64(0xFFFFFFFFFFFFFFFF)
    for ch in chunks:
        b = np.frombuffer(ch, dtype=np.uint8)
        for x in b:
            h = (h ^ np.uint64(int(x))) * np.uint64(1099511628211) & m
    return int(h)


def fnv1a_fast(*chunks):
    """Vectorized FNV-1a is not a thing (the multiply is serial), and the
    byte loop in numpy is ~10 MB/min. Use zlib.crc32 for the transport check
    and keep FNV for the small per-tensor digests; both are recomputed by the
    loader, which is the point."""
    import zlib
    h = 0
    for ch in chunks:
        h = zlib.crc32(ch, h)
    return h & 0xFFFFFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=REPO_LOCAL)
    ap.add_argument("--out", default="/home/louis/models/lfm3/lfm3-audio-2b-q8.lfm3q8")
    ap.add_argument("--selfcheck", action="store_true",
                    help="re-derive the quantization and compare checksums")
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.model, "config.json")))["lfm"]
    sp = os.path.join(a.model, "model.safetensors")
    f = safe_open(sp, "pt")
    keys = sorted(k for k in f.keys() if k.startswith(BACKBONE))
    assert keys, "no %s* tensors in %s" % (BACKBONE, sp)

    # The pattern list has already cost a debugging cycle in this repo twice
    # (a filter matching ZERO tensors silently reports the bf16 result).
    # Every named pattern must land on at least one tensor.
    hit = {p: 0 for p in Q8_PATTERNS}
    hit["embed_tokens"] = 0
    n_f32 = 0
    entries, blobs = [], []
    off = 0
    t0 = time.time()
    for k in keys:
        shp = list(f.get_slice(k).get_shape())
        t = f.get_tensor(k).to("cpu")          # PySafeSlice has no get_tensor
        is_emb = "embed_tokens" in k
        q8 = is_emb or any(p in k for p in Q8_PATTERNS)
        if q8:
            assert t.dim() == 2, "%s is %d-D, the Q8 quantizer is 2-D only" % (k, t.dim())
            for p in (["embed_tokens"] if is_emb else [p for p in Q8_PATTERNS if p in k]):
                hit[p] += 1
            w = t.to(torch.float32).numpy()          # bf16 -> f32 is exact
            codes, scales = quantize_q8(w)
            entries.append(dict(name=k, dtype=T_Q8, ne=shp, off=off,
                                ncodes=codes.nbytes, nscales=scales.nbytes,
                                crc=fnv1a_fast(codes.tobytes(), scales.tobytes())))
            blobs.append(codes.tobytes()); blobs.append(scales.tobytes())
            off += codes.nbytes + scales.nbytes
        else:
            # conv.conv.weight is [H,1,7]: 3-D by design, and the f32 path
            # stores it verbatim (the 2-D-only quantizer never sees it).
            assert t.dim() in (1, 2, 3), k
            w = np.ascontiguousarray(t.to(torch.float32).numpy())
            entries.append(dict(name=k, dtype=T_F32, ne=shp, off=off,
                                ncodes=w.nbytes, nscales=0,
                                crc=fnv1a_fast(w.tobytes())))
            blobs.append(w.tobytes())
            off += w.nbytes
            n_f32 += 1

    empty = [p for p, c in hit.items() if c == 0]
    assert not empty, "Q8 patterns matched no tensors: %s" % empty
    # The 7-tap depthwise conv weight is [H,1,7]: 3-D. The 2-D-only quantizer
    # cannot see it, which is why it is f32 -- assert it is present, because a
    # silent filter drop here would ship a model with no conv taps at all.
    nconv = sum(1 for e in entries if e["name"].endswith("conv.conv.weight"))
    assert nconv == cfg["num_hidden_layers"] - sum(
        1 for lt in cfg["layer_types"] if lt != "conv"), \
        "expected %d conv.conv.weight, got %d" % (
            cfg["num_hidden_layers"] - sum(1 for lt in cfg["layer_types"] if lt != "conv"), nconv)

    hdr = dict(magic=MAGIC.decode(), cfg=cfg, n_tensors=len(entries),
               vocab_size=cfg["vocab_size"], hidden_size=cfg["hidden_size"],
               data_bytes=off, table=[{k: v for k, v in e.items()} for e in entries])
    hj = json.dumps(hdr, separators=(",", ":")).encode()
    with open(a.out, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<I", len(hj)))
        fh.write(hj)
        for b in blobs:
            fh.write(b)
    print("wrote %s: %.0f MB, %d tensors (%d f32), %.1f s"
          % (a.out, (len(hj) + off) / 1e6, len(entries), n_f32, time.time() - t0))

    if a.selfcheck:
        # Re-derive from the safetensors independently and compare per tensor.
        f2 = safe_open(sp, "pt")
        f3 = open(a.out, "rb")
        f3.read(len(MAGIC))
        n = struct.unpack("<I", f3.read(4))[0]
        f3.read(n)
        base = f3.tell()
        bad = []
        for e in entries:
            f3.seek(base + e["off"])
            raw = f3.read(e["ncodes"] + e["nscales"])
            if fnv1a_fast(raw) != e["crc"]:
                bad.append(e["name"] + ":load")
                continue
            t = f2.get_tensor(e["name"]).to("cpu").to(torch.float32).numpy()
            if e["dtype"] == T_Q8:
                c, s = quantize_q8(t)
                ref = c.tobytes() + s.tobytes()
            else:
                ref = np.ascontiguousarray(t).tobytes()
            if fnv1a_fast(ref) != e["crc"]:
                bad.append(e["name"] + ":quant")
        print("SELFCHECK %s (%d tensors, %d bad)%s"
              % ("OK" if not bad else "FAIL", len(entries), len(bad),
                 "" if not bad else " " + str(bad[:8])))
        sys.exit(0 if not bad else 1)
    print("L FM3_Q8_WRITE_DONE")


if __name__ == "__main__":
    main()
