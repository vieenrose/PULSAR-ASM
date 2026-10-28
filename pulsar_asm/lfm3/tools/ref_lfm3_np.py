#!/usr/bin/env python3
"""NumPy reference for the LFM3 text backbone, reading the flat weights blob.

This is the parity oracle the fasm engine is built against: the asm kernels in
pulsar_asm/engine/lfm3_engine_flat.asm must reproduce what this file computes,
which in turn must reproduce HF. The chain is HF golden -> this reference ->
asm, and each link is checked separately (tests/test_lfm3_parity.py).

Math transcribed from the shipped modeling_lfm3.py (the authority, not from
memory), with the four details that are easy to get wrong and cost real time:

1. `embedding_norm` is the FINAL norm. It is applied AFTER the 24-layer loop
   (modeling_lfm3.py:760), despite the name. Applying it to the embedding
   produces plausible-looking garbage.
2. RoPE is asymmetric by layer type. `full_attention` is theta 1e6 with
   partial_rotary_factor 0.5; `sliding_attention` is theta 1e4 with NO partial
   factor, i.e. full rotation. Partial RoPE here is ZERO-TAIL, not a shortened
   rotated dim: inv_freq has rope_angles = int(partial*head_dim)//2 real
   frequencies padded with zeros to head_dim/2, so the tail gets cos=1, sin=0
   and is left unrotated, while rotate_half still spans the full 128 dims.
3. Sandwich norm: post_operator_norm wraps the attn/conv branch and
   post_ffn_norm wraps the FFN branch, each applied BEFORE the residual add.
4. Attention has a per-head sigmoid output gate: gate = sigmoid(gate_proj(h))
   with shape [T, n_head], multiplied into the attention output per head
   BEFORE o_proj -- not after.

Conv layers are LFM2 gated short convs: in_proj splits into B, C, xg in that
column order (BCx.chunk(3, dim=-2)), the conv runs on B*xg, and the result is
multiplied by C -- so C is the output gate and xg is the conv input gate.

Prefill-only by design: no K/V cache, no conv state. Each decode step re-runs
the whole prefix. That is O(T^2) and slow, which is fine for an oracle and
removes an entire class of cache-ordering bugs from the reference side.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BLOB = os.path.join(HERE, os.pardir, "weights.bin")


def bf16(x):
    """Round f32 -> bf16 -> f32 (round-to-nearest-even on the top 16 bits).

    The reference has to emulate bf16, not compute in fp32: HF runs every
    Linear, norm and elementwise product in bf16, and the asm engine's
    smp_bf16_gemb_avx2 does too. An fp32 reference is MORE accurate than the
    thing it is supposed to reproduce, so it can never be bit-exact against it
    -- and it drifts (measured: cosine 0.955 at layer 0, maxrel 0.17, from
    7680-term bf16 dots alone). Every stage of the conv was verified correct
    against HF's own sub-modules under this model.
    """
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u.astype(np.uint64) + r).astype(np.uint32) & 0xFFFF0000).view(np.float32)


class Ref:
    def __init__(self, weights=BLOB, use_bf16=True):
        self.bf16on = use_bf16
        self.man = json.load(open(weights + ".manifest.json"))
        self.m = self.man["meta"]
        self._f = open(weights, "rb")
        self._cache = {}

    def _blob(self, name, dtype):
        e = self.man["layout"][name]
        n = int(np.prod(e["shape"]))
        if n == 0:
            return np.array([])
        self._f.seek(e["offset"])
        data = self._f.read(n * np.dtype(dtype).itemsize)
        a = np.frombuffer(data, dtype=dtype)
        return a.reshape(e["shape"]) if len(e["shape"]) > 1 else a

    def f32(self, name):
        if name not in self._cache:
            self._cache[name] = np.ascontiguousarray(self._blob(name, np.float32))
        return self._cache[name]

    def w(self, name):
        """BF16 tensor, widened exactly (u16 << 16) to f32."""
        key = "w:" + name
        if key not in self._cache:
            u = self._blob(name, np.uint16).astype(np.uint32)
            f = (u << 16).view(np.float32)
            sh = self.man["layout"][name]["shape"]
            self._cache[key] = np.ascontiguousarray(
                f.reshape(sh) if len(sh) > 1 else f)
        return self._cache[key]

    # ---------------------------------------------------------------- parts
    def lin(self, x, W):
        """Linear: fp32 accumulate, bf16 output -- what cuBLAS/oneDNN do for a
        bf16 matmul, and what the asm GEMM does."""
        y = x @ W.T
        return bf16(y) if self.bf16on else y

    def mul(self, a, b):
        """Elementwise product of two bf16 tensors, rounded to bf16."""
        return bf16(a * b) if self.bf16on else a * b

    def add(self, a, b):
        return bf16(a + b) if self.bf16on else a + b

    def rms(self, x, g, eps):
        """Lfm3RMSNorm: mean-square in fp32, rsqrt, then weight * x.
        HF casts the normalized value back to the INPUT dtype (bf16) before
        multiplying by the bf16 weight, so round there too."""
        v = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
        y = x.astype(np.float32) * (1.0 / np.sqrt(v + eps))
        if self.bf16on:
            y = bf16(y)                      # .to(input_dtype)
            return bf16(g * y)               # bf16 weight * bf16 value
        return g * y

    def rope_tables(self, layer_type, T):
        """cos/sin [T, head_dim] for a layer type, per compute_default_rope_parameters."""
        rp = self.m["rope_parameters"][layer_type]
        dim = self.m["head_dim"]
        partial = rp.get("partial_rotary_factor", 1.0)
        base = rp["rope_theta"]
        angles = int(partial * dim) // 2
        ar = np.arange(0, 2 * angles, 2, dtype=np.float64) / dim
        inv = 1.0 / (base ** ar)
        inv = np.concatenate([inv, np.zeros(dim // 2 - angles, dtype=np.float64)])
        pos = np.arange(T, dtype=np.float64)[:, None]
        freqs = pos * inv[None, :]                      # [T, dim/2]
        emb = np.concatenate([freqs, freqs], axis=-1)    # [T, dim]
        return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)

    def rope(self, x, cos, sin):
        """x [n_head, T, head_dim]; rotate_half over the full head_dim."""
        h = x.shape[-1]
        x1, x2 = x[..., : h // 2], x[..., h // 2:]
        rot = np.concatenate([-x2, x1], axis=-1)
        return x * cos[None, :, :] + rot * sin[None, :, :]

    def conv_layer(self, h, L):
        """LFM2 gated short conv on [T,H]: B*xg -> depthwise causal conv -> * C.
        in_proj splits into B, C, xg in that column order (BCx.chunk(3,-2)): C
        is the OUTPUT gate, xg the conv INPUT gate."""
        T, H = h.shape
        bcx = self.lin(h, self.w("L%02d.conv_in_proj" % L))    # [T, 3H]
        B, C, xg = bcx[:, 0:H], bcx[:, H:2 * H], bcx[:, 2 * H:3 * H]
        g = self.mul(B, xg)                                     # conv input
        cw = self.w("L%02d.conv_w" % L).reshape(H, 7)            # [H, 1, 7] -> [H, 7]
        # left-pad 6 along TIME (axis 0): causal depthwise conv over positions,
        # one filter per channel.
        gp = np.concatenate([np.zeros((6, H), np.float32), g], axis=0)
        out = np.zeros((T, H), np.float32)
        for k in range(7):                                        # width 7
            # F.conv1d is CORRELATION, not convolution: no weight flip.
            out += gp[k:k + T] * cw[:, k][None, :]
        out = bf16(out) if self.bf16on else out
        return self.lin(self.mul(C, out), self.w("L%02d.conv_out_proj" % L))

    def attn_layer(self, h, L, cos, sin, win):
        T = h.shape[0]
        nh, nkv, hd = self.m["num_heads"], self.m["num_kv_heads"], self.m["head_dim"]
        q = (h @ self.w("L%02d.w_q" % L).T).reshape(T, nh, hd).transpose(1, 0, 2)
        k = (h @ self.w("L%02d.w_k" % L).T).reshape(T, nkv, hd).transpose(1, 0, 2)
        v = (h @ self.w("L%02d.w_v" % L).T).reshape(T, nkv, hd).transpose(1, 0, 2)
        q = self.rms(q, self.f32("L%02d.norm_q" % L), self.m["rms_norm_eps"])
        k = self.rms(k, self.f32("L%02d.norm_k" % L), self.m["rms_norm_eps"])
        q = self.rope(q, cos, sin)
        k = self.rope(k, cos, sin)
        rep = nh // nkv
        k = np.repeat(k, rep, axis=0)                    # GQA expand
        v = np.repeat(v, rep, axis=0)
        sc = (q @ k.transpose(0, 2, 1)) * (hd ** -0.5)   # [nh, T, T]
        i = np.arange(T)
        mask = (i[None, :, None] <= i[None, None, :])    # causal
        if win:
            mask = mask & (i[None, :, None] - i[None, None, :] < win)
        sc = np.where(mask, sc, np.float32(-np.inf))
        sc = sc - sc.max(axis=-1, keepdims=True)         # fp32 softmax
        e = np.exp(sc)
        a = (e / e.sum(axis=-1, keepdims=True)) @ v      # [nh, T, hd]
        gate = 1.0 / (1.0 + np.exp(-(h @ self.w("L%02d.gate_proj" % L).T)))
        a = a * gate.T[:, :, None]                       # per-head sigmoid gate
        return a.transpose(1, 0, 2).reshape(T, nh * hd) @ self.w("L%02d.w_o" % L).T

    def head(self):
        """lm_head is stored as {kind: none, shape: [0]}: the checkpoint is tied
        (tie_word_embeddings), so the blob deliberately does not duplicate the
        524 MB table. The engine must point CTR_LM_HEAD at embed_tokens too."""
        e = self.man["layout"]["lm_head"]
        if e["kind"] == "none" or list(e["shape"]) == [0] or int(np.prod(e["shape"])) == 0:
            assert self.m.get("tie_word_embeddings", True), \
                "lm_head missing but weights are not tied"
            return self.w("embed_tokens")
        return self.w("lm_head")

    def forward(self, ids):
        eps = self.m["rms_norm_eps"]
        types = self.m["layer_types"]
        T = len(ids)
        x = self.w("embed_tokens")[np.asarray(ids)]        # [T, H]
        cosp, sinp = {}, {}
        for lt in ("full_attention", "sliding_attention"):
            if lt in types:
                cosp[lt], sinp[lt] = self.rope_tables(lt, T)
        for L in range(self.m["n_layers"]):
            lt = types[L]
            res = x
            h = self.rms(x, self.f32("L%02d.norm_op" % L), eps)
            if lt == "conv":
                h = self.conv_layer(h, L)
            else:
                h = self.attn_layer(h, L, cosp[lt], sinp[lt],
                                    self.m["sliding_window"] if lt == "sliding_attention" else None)
            if self.m["use_sandwich_norm"]:
                h = self.rms(h, self.f32("L%02d.norm_post_op" % L), eps)
            x = res + h
            hn = self.rms(x, self.f32("L%02d.norm_ffn" % L), eps)
            u = hn @ self.w("L%02d.mlp_w1" % L).T
            g = hn @ self.w("L%02d.mlp_w3" % L).T
            d = (u * (g / (1.0 + np.exp(-g)))) @ self.w("L%02d.mlp_w2" % L).T
            if self.m["use_sandwich_norm"]:
                d = self.rms(d, self.f32("L%02d.norm_post_ffn" % L), eps)
            x = x + d
        x = self.rms(x, self.f32("embedding_norm"), eps)   # FINAL norm
        return x @ self.head().T                            # [T, vocab]

    def greedy(self, ids, n=48, eos=(124900, 125014)):
        """Re-runs the whole prefix each step: a reference, not an engine."""
        ids = list(ids)
        out = []
        for _ in range(n):
            lg = self.forward(np.asarray(ids))[-1]
            t = int(np.argmax(lg))
            out.append(t)
            ids.append(t)
            if t in eos:
                break
        return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="A")
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--blob", default=BLOB)
    a = ap.parse_args()
    gold = os.path.join(HERE, os.pardir, os.pardir, os.pardir,
                        ".auto", "lfm3", "golden_greedy_%s.json" % a.prompt)
    if not os.path.exists(gold):
        sys.exit("no golden at %s (run .auto/lfm3/oracle.py)" % gold)
    g = json.load(open(gold))
    r = Ref(a.blob)
    gen = r.greedy(g["prompt_ids"], a.tokens)
    ref = g["gen"]
    n = 0
    for x, y in zip(gen, ref):
        if x != y:
            break
        n += 1
    span = min(len(gen), len(ref), a.tokens)
    print("prompt %s: %d tokens, prefix %d/%d  ->  %s"
          % (a.prompt, len(gen), n, span,
             "MATCH" if n >= span else "DIVERGES at token %d (%d vs %d)"
             % (n, gen[n] if n < len(gen) else -1, ref[n] if n < len(ref) else -1)))
    return 0 if n >= span else 1


if __name__ == "__main__":
    sys.exit(main())
