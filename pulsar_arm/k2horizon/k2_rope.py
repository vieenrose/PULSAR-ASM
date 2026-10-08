#!/usr/bin/env python3
"""YaRN rope tables for K2-Horizon-0.9B (numpy float32, mirrors
transformers modeling_rope_utils._compute_yarn_parameters + the
K2HorizonRotaryEmbedding forward exactly).

Params: base=1e6, dim=64, factor=16, attention_factor=1.2772588722239782
(explicit), beta_fast=128, beta_slow=4, orig_max=8192, truncate=True
-> correction range lo=5, hi=14.
Forward: freqs = outer(positions, inv_freq); emb = cat(freqs, freqs);
cos = cos(emb)*attn_scale, sin likewise (NeoX pairs, so the unique
per-pair values are cos/sin(freqs): tables are (npos, 32)).

Usage: k2_rope.py <npos> <out.npz>
Verify against torch ASAP (1-ulp libm risk); engine validation uses
maxrel tolerance on the rope path regardless.
"""
import math
import sys

import numpy as np

BASE, DIM, FACTOR = 1000000.0, 64, 16.0
ATTN = 1.2772588722239782
BFAST, BSLOW, OMAX = 128.0, 4.0, 8192
F32 = np.float32


def yarn_inv_freq():
    pos = np.arange(0, DIM, 2, dtype=F32) / F32(DIM)
    pos_freqs = np.power(F32(BASE), pos).astype(F32)
    extra = (F32(1.0) / pos_freqs).astype(F32)
    inter = (F32(1.0) / (F32(FACTOR) * pos_freqs)).astype(F32)

    def corr_dim(nrot):
        return (DIM * math.log(OMAX / (nrot * 2 * math.pi))) / (2 * math.log(BASE))

    lo = max(math.floor(corr_dim(BFAST)), 0)
    hi = min(math.ceil(corr_dim(BSLOW)), DIM - 1)
    den = F32(hi - lo if hi != lo else 0.001)
    lin = (np.arange(DIM // 2, dtype=F32) - F32(lo)) / den
    ramp = np.clip(lin, F32(0), F32(1)).astype(F32)
    extra_factor = (F32(1.0) - ramp).astype(F32)
    return (inter * (F32(1.0) - extra_factor) + extra * extra_factor).astype(F32)


def tables(npos):
    inv = yarn_inv_freq()
    freqs = np.outer(np.arange(npos, dtype=F32), inv).astype(F32)
    cos = (np.cos(freqs) * F32(ATTN)).astype(F32)
    sin = (np.sin(freqs) * F32(ATTN)).astype(F32)
    return cos, sin


def main():
    npos, out = int(sys.argv[1]), sys.argv[2]
    cos, sin = tables(npos)
    assert cos.shape == sin.shape == (npos, 32)
    assert np.all(np.isfinite(cos)) and np.all(np.isfinite(sin))
    np.savez(out, cos=cos, sin=sin)
    print(f"rope tables: npos={npos} inv[0]={yarn_inv_freq()[0]:.6f} "
          f"cos00={cos[0,0]:.6f} (expect {ATTN:.6f})")


if __name__ == "__main__":
    main()
