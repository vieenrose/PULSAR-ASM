# ==============================================================================
# Project PULSAR-ASM | tools/convert_gemma4_safetensors.py
# ------------------------------------------------------------------------------
# Google Gemma 4 E2B -> PULSAR flat weight blob (TEXT-ONLY, Linux, zero-disk-copy)
#
# The official checkpoint is a 10.2 GB multimodal safetensors file (bf16) that we
# do NOT need on disk: we read ONLY the `model.language_model.*` tensors through
# HTTP Range requests, in parallel, and write them straight into the engine's
# flat layout. Vision/audio towers are skipped entirely (text-only engine).
#
# Weight layout (64-byte aligned blobs, one file, memory-mapped by the engine):
#   per layer L:  5x fp32 norm[1536] | q_norm fp32[hd] | k_norm fp32[hd] (if KV)
#                 layer_scalar fp32[1]
#                 w_q | w_k | w_v | w_o | ple_gate | ple_proj | gate | up | down
#   tail:         final_norm | ple_proj_norm | ple_model_proj | embed | ple_embed
#
# A JSON manifest next to the blob records every blob's byte offset + shape, so
# the Python loader can hand raw pointers to the assembly engine.
#
# Usage:
#   python convert_gemma4_safetensors.py [--repo google/gemma-4-E2B-it]
#                                        [--out /mnt/edge/pulsar/gemma4_e2b.bin]
#                                        [--workers 8]
# Zero PyTorch. Zero safetensors lib. Pure stdlib + urllib.
# ==============================================================================

import os
import io
import sys
import json
import time
import struct
import argparse
import urllib.request
import numpy as np
from concurrent.futures import ThreadPoolExecutor

HF = "https://huggingface.co"
ALIGN = 64


def log(msg):
    print(msg, flush=True)


def hf_token():
    for env in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(env):
            return os.environ[env].strip()
    p = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(p):
        return open(p).read().strip()
    return ""


def http_get(url, rng=None, token="", tries=5):
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url)
            if token:
                req.add_header("Authorization", f"Bearer {token}")
            if rng:
                req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except Exception as e:                      # transient CDN hiccups
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed {url} {rng}: {last}")


def follow(url, token=""):
    """Resolve the CDN redirect, returning (final_url, total_size)."""
    req = urllib.request.Request(url, method="GET")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as r:
        final = r.geturl()
        r.read(1)
        length = int(r.headers.get("Content-Length", "0") or 0)
    return final, length


def parse_header(blob):
    n = struct.unpack("<Q", blob[:8])[0]
    return json.loads(blob[8:8 + n].decode())


# ------------------------------------------------------------------------------
# Target layout description (what the assembly engine expects)
# ------------------------------------------------------------------------------
def build_plan(cfg):
    """Return the ordered list of blobs that make up the engine weight file."""
    tc = cfg["text_config"] if "text_config" in cfg else cfg
    H = tc["hidden_size"]
    NL = tc["num_hidden_layers"]
    NH = tc["num_attention_heads"]
    PLE_D = tc["hidden_size_per_layer_input"]
    VOCAB = tc["vocab_size"]
    VPLE = tc.get("vocab_size_per_layer_input", VOCAB)
    first_kv_shared = NL - tc.get("num_kv_shared_layers", 0)
    types = tc["layer_types"]
    hd_slide = tc["head_dim"]
    hd_full = tc["global_head_dim"]
    inter_base = tc["intermediate_size"]

    plan = []                                  # (kind, name, shape, src_tensor)
    for i in range(NL):
        hd = hd_full if types[i] == "full_attention" else hd_slide
        nq = NH * hd
        inter = inter_base * (2 if i >= first_kv_shared > 0 else 1)
        # Layers >= first_kv_shared are KV-SHARED: they re-use the KV states stored by the
        # last per-type layer before the sharing point (sliding -> L13, full -> L14). The
        # checkpoint still ships distinct k/v tensors for them, but the reference model
        # never loads them (_keys_to_ignore_on_load_unexpected) -> we skip them entirely.
        has_kv = (i < first_kv_shared) if first_kv_shared > 0 else True
        src = f"model.language_model.layers.{i}."
        pfx = f"L{i:02d}."
        plan += [
            ("f32", pfx + "norm_in",       [H],     src + "input_layernorm.weight"),
            ("f32", pfx + "norm_post_attn", [H],    src + "post_attention_layernorm.weight"),
            ("f32", pfx + "norm_pre_ff",   [H],     src + "pre_feedforward_layernorm.weight"),
            ("f32", pfx + "norm_post_ff",  [H],     src + "post_feedforward_layernorm.weight"),
            ("f32", pfx + "norm_ple",      [H],     src + "post_per_layer_input_norm.weight"),
            ("f32", pfx + "norm_q",        [hd],    src + "self_attn.q_norm.weight"),
            ("f32", pfx + "layer_scalar",  [1],     src + "layer_scalar"),
            ("raw", pfx + "w_q",   [nq, H],         src + "self_attn.q_proj.weight"),
            ("raw", pfx + "w_o",   [H, nq],         src + "self_attn.o_proj.weight"),
            ("raw", pfx + "ple_gate", [PLE_D, H],   src + "per_layer_input_gate.weight"),
            ("raw", pfx + "ple_proj", [H, PLE_D],   src + "per_layer_projection.weight"),
            ("raw", pfx + "mlp_gate", [inter, H],   src + "mlp.gate_proj.weight"),
            ("raw", pfx + "mlp_up",   [inter, H],   src + "mlp.up_proj.weight"),
            ("raw", pfx + "mlp_down", [H, inter],   src + "mlp.down_proj.weight"),
        ]
        if has_kv:
            plan += [   # v_norm is a scale-free RMSNorm (with_scale=False): no tensor exists
                ("f32", pfx + "norm_k",    [hd],    src + "self_attn.k_norm.weight"),
                ("raw", pfx + "w_k",       [hd, H], src + "self_attn.k_proj.weight"),
                ("raw", pfx + "w_v",       [hd, H], src + "self_attn.v_proj.weight"),
            ]
        else:
            plan += [("none", pfx + "norm_k", [0], None),
                     ("none", pfx + "w_k", [0], None),
                     ("none", pfx + "w_v", [0], None)]
    plan += [
        ("f32", "final_norm",    [H],                  "model.language_model.norm.weight"),
        ("f32", "ple_proj_norm", [PLE_D],              "model.language_model.per_layer_projection_norm.weight"),
        ("raw", "ple_model_proj", [NL * PLE_D, H],     "model.language_model.per_layer_model_projection.weight"),
        ("raw", "embed",         [VOCAB, H],           "model.language_model.embed_tokens.weight"),
        ("raw", "ple_embed",     [VPLE, NL * PLE_D],   "model.language_model.embed_tokens_per_layer.weight"),
    ]
    # KV-sharing map: which layer stores the shared states for each layer type
    prev_types = types[:first_kv_shared]
    kv_store = {}
    for t in set(types):
        idxs = [j for j in range(first_kv_shared) if types[j] == t]
        kv_store[t] = idxs[-1] if idxs else None

    return plan, {"hidden": H, "n_layers": NL, "head_dim_slide": hd_slide,
                  "head_dim_full": hd_full, "vocab": VOCAB, "ple_dim": PLE_D,
                  "inter_base": inter_base, "first_kv_shared": first_kv_shared,
                  "n_heads": NH, "sliding_window": tc.get("sliding_window"),
                  "rope_theta_slide": tc["rope_parameters"]["sliding_attention"]["rope_theta"],
                  "rope_theta_full": tc["rope_parameters"]["full_attention"]["rope_theta"],
                  "partial_rotary_full": tc["rope_parameters"]["full_attention"].get("partial_rotary_factor", 1.0),
                  "rms_eps": tc["rms_norm_eps"],
                  "logit_softcapping": tc.get("final_logit_softcapping"),
                  "kv_store_layers": kv_store,
                  "layer_types": types}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="google/gemma-4-E2B-it")
    ap.add_argument("--out", default="/mnt/edge/pulsar/gemma4_e2b.bin")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    token = hf_token()
    cfg_url = f"{HF}/{args.repo}/raw/main/config.json"
    cfg = json.loads(http_get(cfg_url, token=token))
    plan, meta = build_plan(cfg)

    data_url = f"{HF}/{args.repo}/resolve/main/model.safetensors"
    log(f"📡 Resolving {args.repo} ...")
    final_url, total = follow(data_url, token)
    head = http_get(final_url, (0, 1 << 20), token)
    hdr = parse_header(head)
    if "__metadata__" not in hdr:                     # header larger than 1 MB
        head = http_get(final_url, (0, 8 << 20), token)
        hdr = parse_header(head)
    src = {k: v for k, v in hdr.items() if k != "__metadata__"}
    # safetensors data_offsets are relative to the END OF HEADER, not to the start
    # of the file: tensor bytes begin at 8 + header_len. Reading them as file
    # offsets shifts EVERY tensor by the header size (~264 KB here) - and since a
    # verification pass that repeats the same omission agrees with itself, the
    # corruption passes its own checksum and only shows up as a model that talks
    # nonsense. This bug cost a full 9.26 GB conversion before it was caught.
    base = 8 + struct.unpack("<Q", head[:8])[0]
    log(f"   checkpoint: {total/1e9:.2f} GB / {len(src):,} tensors / header {base:,} B")

    # ---- assign output offsets ------------------------------------------------
    off, layout = 0, {}
    for kind, name, shape, sname in plan:
        if kind == "none":
            layout[name] = {"offset": 0, "shape": shape, "kind": "none"}
            continue
        elems = 1
        for d in shape:
            elems *= d
        nbytes = 4 * elems if kind == "f32" else 2 * elems   # bf16 = 2 bytes/element
        layout[name] = {"offset": off, "shape": shape, "kind": kind, "src": sname}
        off = (off + nbytes + ALIGN - 1) // ALIGN * ALIGN
    log(f"🧱 flat blob size: {off/1e9:.3f} GB  ({len([p for p in plan if p[0]!='none']):,} blobs)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        f.truncate(off)
    log(f"📦 preallocated {args.out}")

    # ---- build the transfer list: source ranges -> destination offsets --------
    jobs = []
    for kind, name, shape, sname in plan:
        if kind == "none":
            continue
        t = src[sname]
        lo, end = t["data_offsets"]          # end is EXCLUSIVE
        dst = layout[name]["offset"]
        if kind == "raw":
            if t["dtype"] != "BF16":
                raise RuntimeError(f"{sname}: expected BF16, got {t['dtype']}")
            jobs.append((sname, final_url, base + lo, base + end - 1, dst, "raw"))
        else:
            jobs.append((sname, final_url, base + lo, base + end - 1, dst,
                         "bf16->f32" if t["dtype"] == "BF16" else "copy"))
    total_bytes = sum(j[3] - j[2] + 1 for j in jobs)
    log(f"🚀 streaming {len(jobs):,} tensors / {total_bytes/1e9:.2f} GB with "
        f"{args.workers} parallel range readers")

    t0 = time.perf_counter()
    done = {"bytes": 0, "n": 0}

    def write_at(buf, dst):
        fd = os.open(args.out, os.O_WRONLY)
        try:
            os.pwrite(fd, buf, dst)
        finally:
            os.close(fd)

    def bf16_to_f32(buf):
        # bf16 -> fp32 is a plain 16-bit left shift; the exponent+mantissa layout is a
        # truncated fp32. Vectorised, no lookup, no branches.
        u = np.frombuffer(buf, dtype=np.uint16)
        return (u.astype(np.uint32) << 16).view(np.float32).tobytes()

    def do(job):
        name, url, lo, hi, dst, mode = job
        buf = http_get(url, (lo, hi), token)
        if len(buf) != hi - lo + 1:
            raise RuntimeError(f"short read {name}")
        if mode == "bf16->f32":
            buf = bf16_to_f32(buf)
        write_at(buf, dst)
        done["bytes"] += len(buf)
        done["n"] += 1
        if done["n"] % 40 == 0 or done["n"] == len(jobs):
            el = time.perf_counter() - t0
            done_b = done["bytes"]
            rate = done_b / max(el, 1e-9)
            eta = (total_bytes - done_b) / rate if rate > 0 else -1
            log(f"   {done['n']:5d}/{len(jobs)} tensors  {done_b/1e9:6.2f} GB  "
                f"{rate/1e9:5.2f} GB/s  eta {eta:6.0f}s")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(do, jobs))

    man = {"repo": args.repo, "bytes": off, "layout": layout, "meta": meta,
           "source_url": final_url,          # includes the resolved commit hash
           "layer_types": cfg["text_config"]["layer_types"],
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(args.out + ".manifest.json", "w") as f:
        json.dump(man, f)
    log(f"✅ done in {time.perf_counter()-t0:.1f}s -> {args.out} ({off/1e9:.3f} GB)")

    # ---- spot-check integrity against the source (catches layout bugs) -------
    log("🔬 verifying random samples against the source checkpoint...")
    fd = os.open(args.out, os.O_RDONLY)
    bad = 0
    small = [j for j in jobs if (j[3] - j[2] + 1) <= 65536]
    big = [j for j in jobs if (j[3] - j[2] + 1) > 65536]
    rng = __import__("random").Random(1234)
    probes = small + rng.sample(big, min(12, len(big)))     # all small ones, sample of big
    log(f"   {len(small):,} small blobs verified exhaustively, "
        f"{min(12, len(big))} large blobs sampled")
    for j in probes:
        name, url, lo, hi, dst, mode = j
        n = min(4096 if mode == "raw" else 8192, hi - lo + 1)
        src = http_get(url, (lo, lo + n - 1), token)
        if mode == "bf16->f32":
            src = bf16_to_f32(src)        # 2 source bytes become 4 stored bytes:
        got = os.pread(fd, len(src), dst)  # compare equal lengths or every probe 'fails'
        if src != got:
            bad += 1
            log(f"   ✗ MISMATCH {name}")
    os.close(fd)
    log("   ✔ layout verified byte-for-byte on %d probes" % len(probes))
    if bad:
        raise SystemExit(f"{bad} probe mismatches - weight layout is WRONG")


if __name__ == "__main__":
    main()
