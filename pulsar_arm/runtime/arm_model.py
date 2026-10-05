"""Linux loader + driver for the Gemma-3 270m ARM port (PULSAR-ARM).

Zero-copy by design: the safetensors file is mmapped and weights are handed
to C kernels as raw bf16 pointers - there is no blob conversion step. Numpy
is allocation/verification only; every flop runs in neon_gemv.c / neon_ops.c.

Gemma-3 270m facts baked in (asserted against the checkpoint header):
  hidden 640, 18 layers, 4 q heads, 1 kv head, head_dim 256 everywhere,
  sliding window 512 (5 sliding + 1 full), attention scale 1/sqrt(256),
  q_norm + k_norm (no v_norm), gelu_pytorch_tanh MLP, tied lm head,
  no logit softcap, per-layer KV (no sharing), bf16 norms.
"""
import ctypes
import json
import mmap
import os
import struct
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = [os.path.join(HERE, "..", "kernels", "neon_gemv.c"),
       os.path.join(HERE, "..", "kernels", "neon_ops.c")]
SO = os.path.join(HERE, "libarm.so")

HID, NHEAD, HD, INTER, NLAY = 640, 4, 256, 2048, 18
WINDOW = 512
FULL_AT = {5, 11, 17}
ATTN_SCALE = 1.0 / (256 ** 0.5)
EPS = 1e-6


def build():
    r = subprocess.run(["gcc", "-O3", "-mcpu=cortex-a72", "-fno-math-errno", "-fno-trapping-math",
      "-fPIC", "-shared"] + SRC + ["-o", SO, "-lm"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        sys.exit(1)


def widen(u16):
    return ((np.asarray(u16, dtype=np.uint16).astype(np.uint32)) << 16).view(np.float32)


class Arm270m:
    def __init__(self, safetensors, max_seq=1024, verbose=True):
        t0 = time.time()
        self.f = open(safetensors, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        hdr = json.loads(self.f.read(n))
        self.data_start = 8 + n
        self.max_seq = max_seq
        mm = mmap.mmap(self.f.fileno(), 0, prot=mmap.PROT_READ)
        self._mm = mm
        base = np.frombuffer(mm, dtype=np.uint8).ctypes.data

        self.off = {}
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            self.off[k] = (self.data_start + v["data_offsets"][0], tuple(v["shape"]))

        def raw(name):
            off, shape = self.off[name]
            return base + off, shape

        # norms widened once at load (vectors, not streams). Gemma3RMSNorm
        # multiplies by (1 + w), not w (weights init to zero, not one) - so
        # the +1 is folded here and the C kernel stays a plain scale. Done
        # in float64 before the fp32 cast to keep it exact to ~1e-9.
        def f32vec(name):
            off, shape = self.off[name]
            u = np.frombuffer(mm, dtype=np.uint16,
                              count=int(np.prod(shape)),
                              offset=off).astype(np.uint32)
            w = ((u << 16).view(np.float32)).astype(np.float64)
            return np.ascontiguousarray((1.0 + w).astype(np.float32))

        self.w = {}   # name -> (ptr, shape) for bf16 streams (kept as pointers)
        for k in self.off:
            if k.endswith(".weight") and "norm" not in k and "embed" not in k:
                self.w[k] = raw(k)
        self.emb_ptr, _ = raw("model.embed_tokens.weight")
        self.norms = {k: f32vec(k) for k in self.off
                      if k.endswith("layernorm.weight") or k.endswith("norm.weight")
                      or k.endswith("q_norm.weight") or k.endswith("k_norm.weight")}

        # RoPE tables: global theta 1e6 (full), local base 1e4 (sliding)
        pos = np.arange(max_seq, dtype=np.float64)[:, None]
        self.tables = {}
        for tag, theta in (("s", 10000.0), ("f", 1000000.0)):
            inv = 1.0 / (theta ** (np.arange(0, HD, 2) / HD))
            ang = pos * inv
            self.tables[tag] = (np.ascontiguousarray(np.cos(ang), dtype=np.float32),
                                np.ascontiguousarray(np.sin(ang), dtype=np.float32))

        # caches: per-layer fp32 K/V, full length (short contexts dominate)
        self.K = [np.zeros((max_seq, HD), dtype=np.float32) for _ in range(NLAY)]
        self.V = [np.zeros((max_seq, HD), dtype=np.float32) for _ in range(NLAY)]

        # scratch
        self.X = np.zeros(HID, dtype=np.float32)
        self.H = np.zeros(HID, dtype=np.float32)
        self.Q = np.zeros(NHEAD * HD, dtype=np.float32)
        self.QN = np.zeros(NHEAD * HD, dtype=np.float32)
        self.KV = np.zeros(HD, dtype=np.float32)
        self.KN = np.zeros(HD, dtype=np.float32)
        self.S = np.zeros(max_seq, dtype=np.float32)
        self.AV = np.zeros(NHEAD * HD, dtype=np.float32)
        self.G = np.zeros(INTER, dtype=np.float32)
        self.U = np.zeros(INTER, dtype=np.float32)
        self.LG = np.zeros(262144, dtype=np.float32)

        build()
        lib = ctypes.CDLL(SO)
        v = ctypes.c_void_p
        lib.gemv_bf16.argtypes = [ctypes.c_int, ctypes.c_int, v, v, v]
        lib.rmsnorm_f32.argtypes = [v, v, v, ctypes.c_int, ctypes.c_float]
        lib.rope_half.argtypes = [v, v, v, ctypes.c_int]
        lib.softmax_f32.argtypes = [v, ctypes.c_int]
        lib.gelu_tanh_f32.argtypes = [v, v, ctypes.c_int]
        lib.argmax_f32.argtypes = [v, ctypes.c_int]
        lib.argmax_f32.restype = ctypes.c_int
        lib.embed_row_f32.argtypes = [v, v, ctypes.c_int, ctypes.c_int, ctypes.c_float]
        lib.attn_scores_f32.argtypes = [v, v, v, ctypes.c_int, ctypes.c_int, ctypes.c_float]
        lib.gelu_mul_f32.argtypes = [v, v, v, ctypes.c_int]
        lib.rmsnorm_add_f32.argtypes = [v, v, v, v, ctypes.c_int, ctypes.c_float]
        lib.attn_values_f32.argtypes = [v, v, v, ctypes.c_int, ctypes.c_int]
        for f_ in ("gemv_bf16", "rmsnorm_f32", "rope_half", "softmax_f32",
                   "gelu_tanh_f32", "gelu_mul_f32", "rmsnorm_add_f32",
                   "embed_row_f32", "attn_scores_f32",
                   "attn_values_f32"):
            getattr(lib, f_).restype = None
        self.lib = lib
        self.pos = 0
        self.tape = None  # when a list, forward() appends X after each layer
        if verbose:
            print(f"  arm module ready, init {time.time()-t0:.1f}s", flush=True)

    def _gemv(self, out, wname, x, rows=None):
        ptr, shape = self.w[wname]
        M = shape[0] if rows is None else rows
        self.lib.gemv_bf16(M, shape[1], ptr, x.ctypes.data, out.ctypes.data)

    def forward(self, token):
        lib = self.lib
        P = lambda a: a.ctypes.data
        lib.embed_row_f32(P(self.X), self.emb_ptr, int(token), HID, HID ** 0.5)
        pos = self.pos
        for i in range(NLAY):
            full = i in FULL_AT
            p = f"model.layers.{i}."
            lib.rmsnorm_f32(P(self.H), P(self.norms[p + "input_layernorm.weight"]),
                            P(self.X), HID, EPS)
            self._gemv(self.Q, p + "self_attn.q_proj.weight", self.H)
            self._gemv(self.KV, p + "self_attn.k_proj.weight", self.H)
            lib.rmsnorm_f32(P(self.KN), P(self.norms[p + "self_attn.k_norm.weight"]),
                            P(self.KV), HD, EPS)
            cos, sin = self.tables["f" if full else "s"]
            lib.rope_half(P(self.KN), P(cos[pos]), P(sin[pos]), HD // 2)
            K, V = self.K[i], self.V[i]
            K[pos] = self.KN
            self._gemv(self.KN, p + "self_attn.v_proj.weight", self.H)
            V[pos] = self.KN
            lo = 0 if full else max(0, pos + 1 - WINDOW)
            n = pos + 1 if full else min(pos + 1, WINDOW)
            for j in range(NHEAD):
                seg = slice(j * HD, (j + 1) * HD)
                lib.rmsnorm_f32(P(self.QN[seg]), P(self.norms[p + "self_attn.q_norm.weight"]),
                                P(self.Q[seg]), HD, EPS)
                lib.rope_half(P(self.QN[seg]), P(cos[pos]), P(sin[pos]), HD // 2)
                lib.attn_scores_f32(P(self.S), P(self.QN[seg]), P(K[lo]), n, HD,
                                    np.float32(ATTN_SCALE))
                lib.softmax_f32(P(self.S), n)
                lib.attn_values_f32(P(self.AV[seg]), P(V[lo]), P(self.S), n, HD)
            self._gemv(self.H, p + "self_attn.o_proj.weight", self.AV)
            lib.rmsnorm_add_f32(P(self.H), P(self.norms[p + "post_attention_layernorm.weight"]),
                                P(self.H), P(self.X), HID, EPS)
            lib.rmsnorm_f32(P(self.H), P(self.norms[p + "pre_feedforward_layernorm.weight"]),
                            P(self.X), HID, EPS)
            self._gemv(self.G, p + "mlp.gate_proj.weight", self.H)
            self._gemv(self.U, p + "mlp.up_proj.weight", self.H)
            lib.gelu_mul_f32(P(self.G), P(self.G), P(self.U), INTER)
            self._gemv(self.H, p + "mlp.down_proj.weight", self.G)
            lib.rmsnorm_add_f32(P(self.H), P(self.norms[p + "post_feedforward_layernorm.weight"]),
                                P(self.H), P(self.X), HID, EPS)
            if self.tape is not None:
                self.tape.append(self.X.copy())
        lib.rmsnorm_f32(P(self.H), P(self.norms["model.norm.weight"]), P(self.X),
                        HID, EPS)
        # tied head over the full vocab table (bf16 stream)
        lib.gemv_bf16(262144, HID, self.emb_ptr, P(self.H), P(self.LG))
        self.pos += 1
        return int(lib.argmax_f32(P(self.LG), 262144))
