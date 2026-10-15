#!/usr/bin/env python3
"""Float reference of Gemma 4 E2B on the dequantized weights of Google's Q4_0 GGUF.

Same equations, in the same order and precision, as pulsar_asm/tools/ref_gemma4_np.py
(the x86 port's reference, itself checked against HF transformers): float64 inside
norms, rope, GELU and softmax, float32 between stages, no activation quantization.
Only the weights differ: they come dequantized from the GGUF (Q4_0 layers, Q6_K
embedding/head and PLE table, F16 PLE projection), so this is the model the Nano
engine runs, in exact-ish arithmetic. The bit-exact engine reference is g4ref.c.

Usage: python3 -I g4_np.py <file.gguf> <ids...> [--gen N]
Prints one line per step: step, input id, argmax id, top logit (after soft-cap).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g4_dequant as dq  # noqa: E402
from g4_gguf import GGUF  # noqa: E402

H, NL, NH, PLE, V = 1536, 35, 8, 256, 262144
HD_S, HD_F = 256, 512
FIRST_SHARED = 15
WINDOW = 512
EPS = 1e-6
SOFTCAP = 30.0


def rms(x, w):
    xf = x.astype(np.float64)
    return (xf / np.sqrt((xf ** 2).mean() + EPS) * w.astype(np.float64)).astype(np.float32)


def rms_noscale(x):
    xf = x.astype(np.float64)
    return (xf / np.sqrt((xf ** 2).mean() + EPS)).astype(np.float32)


def gelu_tanh(x):
    c1, c3 = 1.5957691216057308, 0.07135481627462202
    x64 = x.astype(np.float64)
    return (x64 / (1.0 + np.exp(-(c1 * x64 + c3 * x64 ** 3)))).astype(np.float32)


def rope(x, cos, sin):
    h = x.size // 2
    a, b = x[:h].astype(np.float64), x[h:].astype(np.float64)
    c, s = cos.astype(np.float64), sin.astype(np.float64)
    return np.concatenate([a * c - b * s, b * c + a * s]).astype(np.float32)


def rope_tables(max_seq):
    """(cos, sin) per layer type, [max_seq, head_dim/2] fp32, HF default/proportional."""
    pos = np.arange(max_seq, dtype=np.float64)[:, None]
    out = {}
    for typ, theta, hd, part in (("slide", 1e4, HD_S, 1.0), ("full", 1e6, HD_F, 0.25)):
        n = hd // 2
        nrot = max(1, int(part * hd // 2))
        rot = 1.0 / (theta ** (np.arange(0, 2 * nrot, 2, dtype=np.float64) / hd))
        inv = np.concatenate([rot, np.zeros(n - nrot)])
        ang = pos * inv
        out[typ] = (np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32))
    return out


class Model:
    def __init__(self, path, max_seq=2048):
        self.g = GGUF(path)
        self.max_seq = max_seq
        self._w = {}
        sw = self.g.meta["gemma4.attention.sliding_window_pattern"]
        self.full = [not bool(s) for s in sw]          # pattern: True = sliding
        assert sum(self.full) == 7, sum(self.full)
        self.store = {"slide": max(i for i in range(FIRST_SHARED) if not self.full[i]),
                      "full": max(i for i in range(FIRST_SHARED) if self.full[i])}
        self.rope = rope_tables(max_seq)
        self.kc = {i: np.zeros((max_seq, HD_F if self.full[i] else HD_S), np.float32)
                   for i in range(FIRST_SHARED)}
        self.vc = {i: np.zeros_like(self.kc[i]) for i in range(FIRST_SHARED)}

    def w(self, name):
        if name not in self._w:
            self._w[name] = dq.tensor(self.g, name)
        return self._w[name]

    def ple_inputs(self, x, token):
        proj = (self.w("per_layer_model_proj.weight") @ x).astype(np.float64) * (1.0 / np.sqrt(H))
        proj = proj.reshape(NL, PLE)
        rowv = dq.row(self.g, "per_layer_token_embd.weight", token).astype(np.float64)
        nw = self.w("per_layer_proj_norm.weight").astype(np.float64)
        out = np.zeros((NL, PLE), np.float32)
        for l in range(NL):
            sl = proj[l]
            nrm = sl / np.sqrt((sl ** 2).mean() + EPS) * nw
            tok = rowv[l * PLE:(l + 1) * PLE] * np.sqrt(PLE)
            out[l] = ((nrm + tok) / np.sqrt(2.0)).astype(np.float32)
        return out

    def forward(self, token, pos):
        x = (dq.row(self.g, "token_embd.weight", token).astype(np.float64) * np.sqrt(H)).astype(np.float32)
        ple = self.ple_inputs(x, token)
        for i in range(NL):
            p = f"blk.{i}."
            full = self.full[i]
            typ = "full" if full else "slide"
            hd = HD_F if full else HD_S
            cos, sin = self.rope[typ][0][pos], self.rope[typ][1][pos]
            h = rms(x, self.w(p + "attn_norm.weight"))
            q = self.w(p + "attn_q.weight") @ h
            qn = self.w(p + "attn_q_norm.weight")
            q = np.concatenate([rope(rms(q[j * hd:(j + 1) * hd], qn), cos, sin) for j in range(NH)])
            src = i if i < FIRST_SHARED else self.store[typ]
            if i < FIRST_SHARED:
                k1 = rope(rms(self.w(p + "attn_k.weight") @ h, self.w(p + "attn_k_norm.weight")), cos, sin)
                v1 = rms_noscale(self.w(p + "attn_v.weight") @ h)
                self.kc[i][pos], self.vc[i][pos] = k1, v1
            lo = 0 if full else max(0, pos + 1 - WINDOW)
            k = self.kc[src][lo:pos + 1].astype(np.float64)
            v = self.vc[src][lo:pos + 1].astype(np.float64)
            att = []
            for j in range(NH):
                sc = k @ q[j * hd:(j + 1) * hd].astype(np.float64)   # scaling 1.0 in Gemma 4
                sc -= sc.max()
                e = np.exp(sc)
                e /= e.sum()
                att.append((e[:, None] * v).sum(0))
            a = np.concatenate(att).astype(np.float32)
            x = x + rms(self.w(p + "attn_output.weight") @ a, self.w(p + "post_attention_norm.weight"))
            h = rms(x, self.w(p + "ffn_norm.weight"))
            gu = gelu_tanh(self.w(p + "ffn_gate.weight") @ h) * (self.w(p + "ffn_up.weight") @ h)
            x = x + rms(self.w(p + "ffn_down.weight") @ gu, self.w(p + "post_ffw_norm.weight"))
            g = gelu_tanh(self.w(p + "inp_gate.weight") @ x) * ple[i]
            x = x + rms(self.w(p + "proj.weight") @ g, self.w(p + "post_norm.weight"))
            x = x * self.w(p + "layer_output_scale.weight")
        h = rms(x, self.w("output_norm.weight"))
        logits = self.w("token_embd.weight") @ h
        return (SOFTCAP * np.tanh(logits.astype(np.float64) / SOFTCAP)).astype(np.float32)


def main():
    args = sys.argv[1:]
    gen = 0
    if "--gen" in args:
        k = args.index("--gen")
        gen = int(args[k + 1])
        del args[k:k + 2]
    m = Model(args[0])
    ids = [int(a) for a in args[1:]]
    pos = 0
    n_prompt = len(ids)
    stop = {1, 106}
    while True:
        tok = ids[pos]
        lg = m.forward(tok, pos)
        top = int(np.argmax(lg))
        print(f"step {pos} id_in={tok} top={top} logit={lg[top]:.4f}", flush=True)
        pos += 1
        if pos >= len(ids):
            if len(ids) - n_prompt >= gen or top in stop:   # gen = tokens generated
                break
            ids.append(top)


if __name__ == "__main__":
    main()
