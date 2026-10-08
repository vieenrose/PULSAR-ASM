#!/usr/bin/env python3
"""k2_horizon safetensors -> engine blob (own minimal format, bonsai_b3 precedent).

Blob layout (all little-endian):
  magic[4] = b'K2H1', nlayers u32, hidden u32, inter u32, nhead u32,
  nkv u32, hdim u32, vocab u32, wtype u32 (0=fp16, 1=fp32),
  npos u32 (rope table length; 0 = no tables yet)
  then per tensor in canonical order: raw bytes (fp16 weights, fp32 norms)
  then (if npos>0): cos[npos*32] fp32, sin[npos*32] fp32 (NeoX interleave,
  per-pair (j,j+32), matching apply_rotary_pos_emb on head_dim 64).

Canonical order per layer: input_layernorm, q, k, v, o,
post_attention_layernorm, gate, up, down; then embed, final norm, head.
Q4 comes later (same order, Q4_0 blocks); the engine reads wtype.

Usage: k2_convert.py <snapshot-dir> <out.blob> [--rope cos.npy sin.npy]
Rope tables are baked by a second pass (exact YaRN via transformers, CPU).
"""
import json
import os
import struct
import sys

import numpy as np

HID, INTER, NLAYER, NHEAD, NKV, HDIM, VOCAB = 1536, 5120, 28, 32, 8, 64, 64256
KINDS = ["model.layers.{}.{}"]
LAYER_TENSORS = ["input_layernorm.weight", "self_attn.q_proj.weight",
                 "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                 "self_attn.o_proj.weight", "post_attention_layernorm.weight",
                 "mlp.gate_proj.weight", "mlp.up_proj.weight",
                 "mlp.down_proj.weight"]
TOP_TENSORS = ["model.embed_tokens.weight", "model.norm.weight",
               "lm_head.weight"]
NORMS = {"input_layernorm.weight", "post_attention_layernorm.weight",
         "model.norm.weight"}


def read_sf(path):
    """Minimal safetensors reader (numpy only). Returns {name: array}."""
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            dtype = meta["dtype"]
            shape = tuple(meta["shape"])
            off0, off1 = meta["data_offsets"]
            f.seek(base + off0)
            raw = f.read(off1 - off0)
            dt = {"F16": np.float16, "BF16": None, "F32": np.float32}[dtype]
            if dt is None:  # BF16 -> fp32, all-unsigned (NumPy2 NEP50
                # promotes uint32|int32 to int64, which would double view)
                u32 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
                s = (u32 >> np.uint32(15)) << np.uint32(31)
                eraw = (u32 >> np.uint32(7)) & np.uint32(0xFF)
                m = (u32 & np.uint32(0x7F)) << np.uint32(16)
                z = eraw == np.uint32(0)
                f32 = (s | np.where(z, np.uint32(0), eraw << np.uint32(23))
                         | np.where(z, np.uint32(0), m)).astype(np.uint32)
                arr = f32.view(np.float32).reshape(shape)
            else:
                arr = np.frombuffer(raw, dtype=dt).reshape(shape).copy()
            out[name] = arr
    return out


def main():
    snap, blob, rope = sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None
    sf = None
    for fn in sorted(os.listdir(snap)):
        if fn.endswith(".safetensors"):
            print("reading", fn, flush=True)
            part = read_sf(os.path.join(snap, fn))
            sf = {** (sf or {}), **part}
    assert sf and len(sf) == 255, f"want 255 tensors, got {len(sf) if sf else 0}"
    order = []
    for il in range(NLAYER):
        for t in LAYER_TENSORS:
            order.append(f"model.layers.{il}.{t}")
    order += TOP_TENSORS
    assert all(k in sf for k in order), "missing tensors"
    cos = sin = None
    npos = 0
    if rope:
        r = np.load(rope)
        cos, sin = r["cos"], r["sin"]
        npos = cos.shape[0]
        assert cos.shape == sin.shape == (npos, 32)
    with open(blob, "wb") as f:
        f.write(struct.pack("<4s9I", b"K2H1", NLAYER, HID, INTER, NHEAD,
                            NKV, HDIM, VOCAB, 0, npos))
        total = 0
        for k in order:
            a = sf[k]
            short = k.split("model.layers.")[-1] if "model.layers." in k else k
            if short in NORMS or k == "model.norm.weight":
                b = a.astype(np.float32).tobytes()
            else:
                b = a.astype(np.float16).tobytes()
            f.write(b)
            total += len(b)
        if npos:
            f.write(cos.astype(np.float32).tobytes())
            f.write(sin.astype(np.float32).tobytes())
            total += 2 * npos * 32 * 4
    print(f"wrote {blob}: {total/1e9:.3f} GB, {len(order)} tensors, npos={npos}")


if __name__ == "__main__":
    main()
