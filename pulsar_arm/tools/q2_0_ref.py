"""Qwen3 + ternary Q2_0 reference forward (numpy fp32) for the asm engine.

Reads the blob produced by tools/q2_0_gguf.py and runs the Qwen3 architecture
exactly as the asm engine will: plain-w RMSNorm, per-head q/k norm, GQA
(16 query heads / 8 kv heads x 128), rope theta 1e6, SwiGLU MLP, tied ternary
embeddings, no embedding scale.

Ternary tensors are dequantised block-wise: one fp16 scale per 128 weights,
w = (code - 1) * scale, codes LSB-first inside each byte (llama.cpp order).
Weights are dequantised one tensor at a time so the 457 MB blob never expands
into RAM; the tied LM head streams the embedding rows in chunks.

Prints, for every layer and position, the residual signature in the format
tools/perlayer_probe.patch makes the engine emit:

    DX L<i> p<p>
    DX max=.. v=[a,b,c,d]

plus top-5 logits and the argmax after each position, so `dxcmp2.py` can diff
engine against oracle.

Usage: q2_0_ref.py <blob.bin> <ids,comma,separated> [n_top=5]
"""
import json
import math
import struct
import sys

import numpy as np

GROUP = 128
GROUP_BYTES = 34            # Q2_0: fp16 scale + 32 B of 2-bit codes
TRITS_PER_BYTE = 5          # base-3 repack: 3**5 = 243 <= 256
B3_CODE_BYTES = -(-GROUP // TRITS_PER_BYTE)
B3_GROUP_BYTES = 2 + B3_CODE_BYTES
EPS = 1e-6
HEAD_CHUNK = 16384            # rows per chunk when streaming the tied head


class Blob:
    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.hdr = json.loads(self.f.read(n))
        self.base = 8 + n

    def info(self, name):
        return self.hdr[name]

    def raw(self, name):
        e = self.hdr[name]
        o, e2 = e["data_offsets"]
        self.f.seek(self.base + o)
        return self.f.read(e2 - o)

    def f32(self, name):
        e = self.hdr[name]
        assert e["dtype"] == "F32", e["dtype"]
        a = np.frombuffer(self.raw(name), dtype=np.float32)
        rows, cols = e["shape"]
        return a.reshape(rows, cols) if cols > 1 else a

    def groups(self, name, r0=0, r1=None):
        """-> (scale (n, ng) float32, codes (n, ng, GROUP) float32 in {-1,0,1})."""
        e = self.hdr[name]
        rows, cols = e["shape"]
        r1 = rows if r1 is None else r1
        ng = cols // GROUP
        n = r1 - r0
        if e["dtype"] == "Q2_0":
            b = np.frombuffer(self.raw(name), dtype=np.uint8)[: rows * ng * GROUP_BYTES]
            b = b.reshape(rows, ng, GROUP_BYTES)[r0:r1]
            scale = b[:, :, :2].copy().view(np.float16).astype(np.float32)
            scale = scale.reshape(n, ng)
            raw = b[:, :, 2:]
            codes = np.empty((n, ng, GROUP), dtype=np.float32)
            for k in range(4):
                codes[:, :, k::4] = ((raw >> (2 * k)) & 3).astype(np.float32) - 1.0
        elif e["dtype"] == "B3_128":
            b = np.frombuffer(self.raw(name), dtype=np.uint8)[: rows * ng * B3_GROUP_BYTES]
            b = b.reshape(rows, ng, B3_GROUP_BYTES)[r0:r1]
            scale = b[:, :, :2].copy().view(np.float16).astype(np.float32).reshape(n, ng)
            codes = np.empty((n, ng, GROUP), dtype=np.float32)
            for k in range(B3_CODE_BYTES):
                v = b[:, :, 2 + k].astype(np.uint16)
                for i in range(TRITS_PER_BYTE):
                    idx = k * TRITS_PER_BYTE + i
                    if idx < GROUP:
                        codes[:, :, idx] = (v % 3).astype(np.float32) - 1.0
                        v //= 3
        else:
            raise SystemExit(f"{name}: unexpected dtype {e['dtype']}")
        return scale, codes

    def deq(self, name, r0=0, r1=None):
        """Dequantise Q2_0 rows [r0, r1) to float32 (rows, cols)."""
        e = self.hdr[name]
        rows, cols = e["shape"]
        r1 = rows if r1 is None else r1
        scale, codes = self.groups(name, r0, r1)
        return (codes * scale[:, :, None]).reshape(r1 - r0, cols)


def rms(x, w):
    return (x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS)) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope(v, cos, sin):
    h = v.shape[-1] // 2
    a, b = v[..., :h], v[..., h:]
    return np.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1)


def sig(tag, v):
    v = np.asarray(v, dtype=np.float32).ravel()
    print(f"{tag} max={np.abs(v).max():.7f} "
          f"v=[{v[0]:.7f},{v[1]:.7f},{v[2]:.7f},{v[3]:.7f}]", flush=True)


def main():
    path, ids = sys.argv[1], [int(v) for v in sys.argv[2].split(",")]
    ntop = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    bl = Blob(path)
    emb_info = bl.info("model.embed_tokens.weight")
    VOC, HID = emb_info["shape"]
    INTER = bl.info("model.layers.0.mlp.gate_proj.weight")["shape"][0]
    HD = bl.info("model.layers.0.self_attn.q_norm.weight")["shape"][0]
    NHEAD = bl.info("model.layers.0.self_attn.q_proj.weight")["shape"][0] // HD
    NKV = bl.info("model.layers.0.self_attn.k_proj.weight")["shape"][0] // HD
    NLAY = 1 + max(int(k.split(".")[2]) for k in bl.hdr
                   if k.startswith("model.layers."))
    SCALE = 1.0 / math.sqrt(HD)
    print(f"# oracle: blob hidden {HID} inter {INTER} layers {NLAY} "
          f"q{NHEAD}/kv{NKV} x {HD} vocab {VOC}", flush=True)

    theta = 1000000.0
    inv = theta ** (-(np.arange(HD // 2) / (HD // 2)))

    def embed_rows(tok):
        scale, codes = bl.groups("model.embed_tokens.weight", tok, tok + 1)
        return (codes * scale[:, :, None]).reshape(HID)          # scale: (ng, 1)

    def head_logits(h):
        out = np.empty(VOC, dtype=np.float32)
        for r0 in range(0, VOC, HEAD_CHUNK):
            r1 = min(r0 + HEAD_CHUNK, VOC)
            out[r0:r1] = bl.deq("model.embed_tokens.weight", r0, r1) @ h
        return out

    Kc = [[] for _ in range(NLAY)]
    Vc = [[] for _ in range(NLAY)]
    for p, tok in enumerate(ids):
        X = embed_rows(tok)
        for i in range(NLAY):
            pl = f"model.layers.{i}."
            H = rms(X, bl.f32(pl + "input_layernorm.weight"))
            Q = bl.deq(pl + "self_attn.q_proj.weight") @ H
            K = bl.deq(pl + "self_attn.k_proj.weight") @ H
            V = bl.deq(pl + "self_attn.v_proj.weight") @ H
            qn = bl.f32(pl + "self_attn.q_norm.weight")
            kn = bl.f32(pl + "self_attn.k_norm.weight")
            Q = Q.reshape(NHEAD, HD)
            K = K.reshape(NKV, HD)
            Q = np.stack([rms(Q[j], qn) for j in range(NHEAD)])
            K = np.stack([rms(K[j], kn) for j in range(NKV)])
            ang = p * inv
            cos, sin = np.cos(ang), np.sin(ang)
            Q = rope(Q, cos, sin)
            K = rope(K, cos, sin)
            Kc[i].append(K)
            Vc[i].append(V.reshape(NKV, HD))
            Kk = np.stack(Kc[i])                 # (pos+1, NKV, HD)
            Vv = np.stack(Vc[i])
            lo, n = 0, p + 1
            out = np.empty(NHEAD * HD, dtype=np.float32)
            for j in range(NHEAD):
                kv = j // (NHEAD // NKV)
                s = (Kk[lo:lo + n, kv] @ Q[j]) * SCALE
                s -= s.max()
                w = np.exp(s)
                w /= w.sum()
                out[j * HD:(j + 1) * HD] = w @ Vv[lo:lo + n, kv]
            X = X + (bl.deq(pl + "self_attn.o_proj.weight") @ out)
            H = rms(X, bl.f32(pl + "post_attention_layernorm.weight"))
            g = bl.deq(pl + "mlp.gate_proj.weight") @ H
            u = bl.deq(pl + "mlp.up_proj.weight") @ H
            X = X + (bl.deq(pl + "mlp.down_proj.weight") @ (silu(g) * u))
            print(f"DX L{i} p{p}")
            sig("DX", X)
        lg = head_logits(rms(X, bl.f32("model.norm.weight")))
        top = np.argsort(-lg)[:ntop]
        print(f"LOGITS p{p} " + " ".join(f"{int(k)}:{lg[k]:.4f}" for k in top),
              flush=True)
        print(f"ARGMAX p{p} {int(np.argmax(lg))}", flush=True)


if __name__ == "__main__":
    main()
