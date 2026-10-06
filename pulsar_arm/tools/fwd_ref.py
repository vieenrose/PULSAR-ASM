"""Full 18-layer numpy oracle for the asm engine (HF order, fp64).

Feeds a token sequence and prints, for every layer and position, the residual
signature in exactly the format `tools/perlayer_probe.patch` adds to the asm
engine (apply with `patch -p0 < tools/perlayer_probe.patch`, rebuild, run):

    DX L<i> p<p>
    DX max=.. v=[a,b,c,d]

plus the final top-5 logits after each position. Diff the two to localize the
first diverging layer/position in a single run - that is how the layer-0-only
q_norm bug was found (2026-10-06).

Usage: fwd_ref.py <ids,comma,separated> [n_top=5]
"""
import glob
import json
import math
import struct
import sys

import numpy as np

HID, NHEAD, HD, INTER, NLAY = 640, 4, 256, 2048, 18
FULL = {5, 11, 17}
SCALE = 1.0 / math.sqrt(HD)
EPS = 1e-6

F = glob.glob("/home/luigi/.cache/huggingface/hub/"
              "models--google--gemma-3-270m-it-qat-q4_0-unquantized/"
              "snapshots/*/model.safetensors")[0]
fh = open(F, "rb")
n = struct.unpack("<Q", fh.read(8))[0]
hdr = json.loads(fh.read(n))
BASE = 8 + n
_cache = {}


def T(name, nelem=None):
    if name in _cache:
        return _cache[name]
    e = hdr[name]
    nb = 2 if e["dtype"] in ("BF16", "F16") else 4
    nel = int(np.prod(e["shape"]))
    fh.seek(BASE + e["data_offsets"][0])
    raw = fh.read(nb * nel)
    if e["dtype"] == "BF16":
        us = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
        v = (us << 16).view(np.float32).astype(np.float64)
    else:
        v = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
    v = v.reshape(tuple(e["shape"]))
    _cache[name] = v
    return v


def M(p, nm, o, i):
    return T(p + nm).reshape(o, i)


def rms(x, w):
    return x / math.sqrt(float((x * x).mean()) + EPS) * (1.0 + w)


def sig(tag, v):
    v = np.asarray(v, dtype=np.float64).ravel()
    print(f"{tag} max={np.abs(v).max():.7f} "
          f"v=[{v[0]:.7f},{v[1]:.7f},{v[2]:.7f},{v[3]:.7f}]", flush=True)


def rope(v, cos, sin):
    a, b = v[:HD // 2].copy(), v[HD // 2:].copy()
    return np.concatenate([a * cos - b * sin, b * cos + a * sin])


def main():
    ids = [int(v) for v in sys.argv[1].split(",")]
    ntop = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    emb = T("model.embed_tokens.weight")
    winv = {}
    for th in (10000.0, 1000000.0):
        winv[th] = th ** (-(np.arange(HD // 2) / (HD // 2)))

    Kc = [None] * NLAY
    Vc = [None] * NLAY
    for p, tok in enumerate(ids):
        X = emb[tok] * math.sqrt(HID)
        for i in range(NLAY):
            pl = f"model.layers.{i}."
            H = rms(X, T(pl + "input_layernorm.weight").ravel())
            Q = M(pl, "self_attn.q_proj.weight", 1024, HID) @ H
            Kv = M(pl, "self_attn.k_proj.weight", HD, HID) @ H
            Vv = M(pl, "self_attn.v_proj.weight", HD, HID) @ H
            th = 1000000.0 if i in FULL else 10000.0
            ang = p * winv[th]
            cos, sin = np.cos(ang), np.sin(ang)
            Qn = np.empty_like(Q)
            for j in range(NHEAD):
                q = Q[j * HD:(j + 1) * HD]
                q = rms(q, T(pl + "self_attn.q_norm.weight").ravel())
                Qn[j * HD:(j + 1) * HD] = rope(q, cos, sin)
            Krop = rope(rms(Kv, T(pl + "self_attn.k_norm.weight").ravel()),
                        cos, sin)
            if Kc[i] is None:
                Kc[i], Vc[i] = [], []
            Kc[i].append(Krop)
            Vc[i].append(Vv)
            K = np.stack(Kc[i])
            V = np.stack(Vc[i])
            if i in FULL:
                lo, n = 0, p + 1
            else:
                lo, n = max(0, p + 1 - 512), min(p + 1, 512)
            out = np.empty(NHEAD * HD)
            for j in range(NHEAD):
                q = Qn[j * HD:(j + 1) * HD]
                s = (K[lo:lo + n] @ q) * SCALE
                s -= s.max()
                w = np.exp(s)
                w /= w.sum()
                out[j * HD:(j + 1) * HD] = w @ V[lo:lo + n]
            ao = M(pl, "self_attn.o_proj.weight", HID, NHEAD * HD) @ out
            Hpa = rms(ao, T(pl + "post_attention_layernorm.weight").ravel())
            X = X + Hpa
            pn = rms(X, T(pl + "pre_feedforward_layernorm.weight").ravel())
            g = M(pl, "mlp.gate_proj.weight", INTER, HID) @ pn
            u = M(pl, "mlp.up_proj.weight", INTER, HID) @ pn
            t = 0.7978845608028654 * (g + 0.044715 * g ** 3)
            gg = 0.5 * g * (1.0 + np.tanh(t)) * u
            d = M(pl, "mlp.down_proj.weight", HID, INTER) @ gg
            X = X + rms(d, T(pl + "post_feedforward_layernorm.weight").ravel())
            print(f"DX L{i} p{p}")
            sig("DX", X)
        Hf = rms(X, T("model.norm.weight").ravel())
        lg = emb @ Hf
        top = np.argsort(-lg)[:ntop]
        print(f"LOGITS p{p} " + " ".join(f"{int(k)}:{lg[k]:.4f}" for k in top),
              flush=True)
        print(f"ARGMAX p{p} {int(np.argmax(lg))}", flush=True)


if __name__ == "__main__":
    main()
