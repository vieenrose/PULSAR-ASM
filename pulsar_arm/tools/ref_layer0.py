#!/usr/bin/env python3
"""Numpy-only layer-0 reference for the asm port (no torch/HF needed).

Reads bf16 weights straight from the safetensors file, mirrors the
verified architecture (input_norm -> QKV+qn/kn -> rope -> attn/16 ->
o -> post_norm -> residual -> pre_ff -> gate/up+gelu -> down ->
post_ff -> residual) and prints the SAME stage signature the asm
`fwd_layer0` prints: max abs + first 4 lanes.
Usage: ref_layer0.py [snap] [token=2]
"""
import glob
import json
import os
import struct
import sys

import numpy as np

EPS = 1e-6


def widen(u16):
    return ((np.asarray(u16, dtype=np.uint16).astype(np.uint32)) << 16).view(np.float32)


class Snap:
    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.hdr = json.loads(self.f.read(n))
        self.ds = 8 + n

    def rd(self, name):
        e = self.hdr[name]
        o = self.ds + e["data_offsets"][0]
        sh = tuple(e["shape"])
        self.f.seek(o)
        u = np.frombuffer(self.f.read(int(np.prod(sh)) * 2), dtype=np.uint16)
        return widen(u).reshape(sh)


def rms(x, w):
    return (x / np.sqrt((x ** 2).mean(-1, keepdims=True) + EPS) * (1 + w)).astype(np.float32)


def rope(v, c, s):
    h = len(v) // 2
    a, b = v[:h].copy(), v[h:].copy()
    return np.concatenate([a * c - b * s, b * c + a * s]).astype(np.float32)


def gelu_t(x):
    c1 = np.float32(0.7978845608028654)
    c3 = np.float32(0.044715) * c1
    return (0.5 * x * (1 + np.tanh(c1 * x + c3 * x ** 3))).astype(np.float32)


def sig(name, v):
    v = np.asarray(v, dtype=np.float32).ravel()
    print(f"L0 {name} max={float(np.abs(v).max()):.7f} "
          f"v=[{v[0]:.7f},{v[1]:.7f},{v[2]:.7f},{v[3]:.7f}]", flush=True)


def find_snap(explicit=None):
    if explicit and os.path.isfile(explicit):
        return explicit
    cands = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-3-270m-it-qat-q4_0-unquantized"
        "/snapshots/*/model.safetensors"))
    if not cands:
        raise SystemExit("checkpoint not found")
    return cands[0]


def main():
    snap = find_snap(sys.argv[1] if len(sys.argv) > 1 else None)
    tok = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    S = Snap(snap)
    p = "model.layers.0."
    E = S.rd("model.embed_tokens.weight")
    h = (E[tok] * np.sqrt(640)).astype(np.float32)
    sig("X", h)
    x = rms(h, S.rd(p + "input_layernorm.weight"))
    sig("H", x)
    Wq, Wk, Wv, Wo = (S.rd(p + "self_attn.q_proj.weight"),
                      S.rd(p + "self_attn.k_proj.weight"),
                      S.rd(p + "self_attn.v_proj.weight"),
                      S.rd(p + "self_attn.o_proj.weight"))
    nq, nk = S.rd(p + "self_attn.q_norm.weight"), S.rd(p + "self_attn.k_norm.weight")
    Q = (Wq @ x).reshape(4, 256)
    K = (Wk @ x).reshape(1, 256)
    V = (Wv @ x).reshape(1, 256)
    inv = 1.0 / (10000.0 ** (np.arange(0, 256, 2) / 256))
    cr, sr = np.cos(0 * inv).astype(np.float32), np.sin(0 * inv).astype(np.float32)
    Qn = np.stack([rms(Q[j], nq) for j in range(4)])
    Kn = rms(K[0], nk)
    sig("QN0", Qn[0])
    qr = np.stack([rope(Qn[j], cr, sr) for j in range(4)])
    kr = rope(Kn, cr, sr)
    avs = []
    for j in range(4):
        sc = np.array([kr @ (qr[j] / 16.0)])
        sc -= sc.max()
        w = np.exp(sc)
        w /= w.sum()
        avs.append((w[:, None] * V).sum(0))
    av = np.concatenate(avs).astype(np.float32)
    sig("AV", av)
    mo = Wo @ av
    mp = rms(mo, S.rd(p + "post_attention_layernorm.weight"))
    sig("POST", mp)
    x1 = h + mp
    sig("X1", x1)
    mh = rms(x1, S.rd(p + "pre_feedforward_layernorm.weight"))
    gg = gelu_t(S.rd(p + "mlp.gate_proj.weight") @ mh) * (S.rd(p + "mlp.up_proj.weight") @ mh)
    sig("GG", gg)
    dd = rms(S.rd(p + "mlp.down_proj.weight") @ gg,
             S.rd(p + "post_feedforward_layernorm.weight"))
    sig("D", dd)
    sig("X2", x1 + dd)


if __name__ == "__main__":
    main()
