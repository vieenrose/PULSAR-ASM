#!/usr/bin/env python3
"""M1 cross-check: Google's Q4_0 GGUF against the QAT bf16 safetensors.

For every Q4_0 tensor of the GGUF, take the matching bf16 tensor of
`gemma-4-E2B-it-qat-q4_0-unquantized`, quantize it with ggml's
quantize_row_q4_0_ref rule, and compare the bytes. Two variants of the rounding
step are tried, because ggml's C may be compiled with FMA contraction:
  sep : x0 = x*id (rounded to f32), then x0 + 8.5f (rounded)  -- strict C
  fma : x*id + 8.5f rounded once
A tensor passes when one variant reproduces every byte.

Usage: python3 -I g4_q4check.py <model.safetensors> <file.gguf>
"""
import json
import mmap
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g4_gguf import GGUF  # noqa: E402

HF2GG = {
    "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down", "per_layer_input_gate": "inp_gate",
    "per_layer_projection": "proj",
}


class SafeTensors:
    def __init__(self, path):
        self._fh = open(path, "rb")
        self.mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        n = struct.unpack_from("<Q", self.mm, 0)[0]
        self.h = json.loads(bytes(self.mm[8:8 + n]))
        self.h.pop("__metadata__", None)
        self.base = 8 + n

    def f32(self, name):
        e = self.h[name]
        assert e["dtype"] == "BF16", (name, e["dtype"])
        a, b = e["data_offsets"]
        u = np.frombuffer(self.mm, dtype=np.uint16, count=(b - a) // 2,
                          offset=self.base + a).astype(np.uint32)
        return (u << np.uint32(16)).view(np.float32).reshape(e["shape"])


def q4_0(x, fma):
    """ggml quantize_row_q4_0_ref, vectorized over blocks of 32."""
    blk = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, 32)
    i = np.argmax(np.abs(blk), axis=1)            # first max wins, as in the C loop
    mx = blk[np.arange(blk.shape[0]), i]
    d = (mx / np.float32(-8.0)).astype(np.float32)
    with np.errstate(divide="ignore"):
        idv = np.where(d != 0, np.float32(1.0) / d, np.float32(0.0)).astype(np.float32)
    if fma:
        v = (blk.astype(np.float64) * idv[:, None].astype(np.float64) + 8.5).astype(np.float32)
    else:
        v = (blk * idv[:, None]).astype(np.float32) + np.float32(8.5)
    q = np.minimum(15, np.trunc(v).astype(np.int32)).astype(np.uint8)
    nib = q[:, :16] | (q[:, 16:] << 4)
    out = np.empty((blk.shape[0], 18), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = nib
    return out.tobytes()


def main():
    st, gg = SafeTensors(sys.argv[1]), GGUF(sys.argv[2])
    ok = 0
    bad = []
    for name, e in sorted(gg.tensors.items()):
        if e["type"] != "Q4_0":
            continue
        parts = name.split(".")                     # blk.<i>.<kind>.weight
        il, kind = int(parts[1]), parts[2]
        hf = [k for k, v in HF2GG.items() if v == kind]
        assert len(hf) == 1, name
        src = f"model.language_model.layers.{il}.{hf[0]}.weight"
        w = st.f32(src)
        # GGUF dims are [ne0 = columns, ne1 = rows]; HF is [rows, columns]
        assert list(w.shape) == [e["dims"][1], e["dims"][0]], (name, w.shape, e["dims"])
        ref = bytes(gg.raw(name))
        hit = None
        for fma in (False, True):
            if q4_0(w, fma) == ref:
                hit = "fma" if fma else "sep"
                break
        if hit:
            ok += 1
        else:
            got = np.frombuffer(q4_0(w, False), dtype=np.uint8).reshape(-1, 18)
            exp = np.frombuffer(ref, dtype=np.uint8).reshape(-1, 18)
            nb = int((got != exp).any(axis=1).sum())
            bad.append((name, nb, got.shape[0]))
        print(f"{name:28s} {str(e['dims']):16s} {hit or 'MISMATCH'}", flush=True)
    print(f"\nQ4_0 tensors byte-identical: {ok}/{ok + len(bad)}")
    for n, nb, tot in bad:
        print(f"  {n}: {nb}/{tot} blocks differ")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
