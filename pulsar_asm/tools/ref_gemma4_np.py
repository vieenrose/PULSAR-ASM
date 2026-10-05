"""NumPy reference for the Gemma 4 E2B engine, reading the converted blob.

The authority for the math is transformers/models/gemma4/modeling_gemma4.py;
this module mirrors those same equations directly off the converted weights so a
divergence points at exactly one buffer. Used by tests/test_gemma4_engine.py and
for hunting down the first bad intermediate.
"""

import json
import numpy as np


class Ref:
    def __init__(self, weights, manifest=None):
        self.man = json.load(open(manifest or weights + ".manifest.json"))
        self.m = self.man["meta"]
        self._f = open(weights, "rb")

    def _blob(self, name, dtype):
        e = self.man["layout"][name]
        n = int(np.prod(e["shape"]))
        self._f.seek(e["offset"])
        data = self._f.read(n * np.dtype(dtype).itemsize)
        a = np.frombuffer(data, dtype=dtype)
        return a.reshape(e["shape"]) if len(e["shape"]) > 1 else a

    def f32(self, name):
        """fp32 tensors (norms, layer_scalar) are stored as fp32."""
        return self._blob(name, np.float32)

    def w(self, name):
        """bf16 weights. bf16 is a truncated fp32, so widen by shifting left 16."""
        u = self._blob(name, np.uint16).astype(np.uint32)
        f = (u << 16).view(np.float32)
        sh = self.man["layout"][name]["shape"]
        return f.reshape(sh) if len(sh) > 1 else f

    def w_row_block(self, name, start, n):
        """Rows [start, start+n) of a bf16 [rows, cols] tensor, as fp32."""
        e = self.man["layout"][name]
        cols = e["shape"][1]
        self._f.seek(e["offset"] + start * cols * 2)
        u = np.frombuffer(self._f.read(n * cols * 2), dtype=np.uint16).astype(np.uint32)
        return ((u << 16).view(np.float32)).reshape(n, cols)

    def row(self, name, i, cols=None):
        e = self.man["layout"][name]
        n = e["shape"][1] if cols is None else cols
        self._f.seek(e["offset"] + i * e["shape"][1] * 2)
        u = np.frombuffer(self._f.read(n * 2), dtype=np.uint16).astype(np.uint32)
        return (u << 16).view(np.float32)


def rms(x, w, eps=1e-6):
    xf = x.astype(np.float64)
    return (xf / np.sqrt((xf ** 2).mean() + eps) * w.astype(np.float64)).astype(np.float32)


def gelu_tanh(x):
    c1 = 1.5957691216057308
    c3 = 0.07135481627462202
    x64 = x.astype(np.float64)
    return (x64 / (1.0 + np.exp(-(c1 * x64 + c3 * x64 ** 3)))).astype(np.float32)


def rope(x, cos, sin):
    """rotate_half form; the unused tail of a partial-rotary table is cos=1/sin=0."""
    H = x.size // 2
    a, b = x[:H].astype(np.float64), x[H:].astype(np.float64)
    c, s = cos.astype(np.float64), sin.astype(np.float64)
    return np.concatenate([a * c - b * s, b * c + a * s]).astype(np.float32)


def layer(ref, i, x, ple_in, pos, cos, sin, kv=None):
    """One decoder layer, mirroring Gemma4TextDecoderLayer.forward().

    Returns (x_out, k_row, v_row) where k_row/v_row are the cache rows this layer
    would publish (None for KV-shared layers, which publish nothing).
    """
    m = ref.m
    p = f"L{i:02d}."
    full = ref.man["layer_types"][i] == "full_attention"
    hd = m["head_dim_full"] if full else m["head_dim_slide"]
    inter = m["inter_base"] * (2 if i >= m["first_kv_shared"] else 1)
    shared = i >= m["first_kv_shared"]
    nh = m["n_heads"]

    h = rms(x, ref.f32(p + "norm_in"), m["rms_eps"])
    q = (ref.w(p + "w_q") @ h)
    qn = ref.f32(p + "norm_q")
    qh = np.concatenate([rms(q[j * hd:(j + 1) * hd], qn, m["rms_eps"]) for j in range(nh)])
    qh = np.concatenate([rope(qh[j * hd:(j + 1) * hd], cos[:hd // 2], sin[:hd // 2])
                         for j in range(nh)])
    if shared:
        k, v, kr, vr = kv
    else:
        k = ref.w(p + "w_k") @ h
        k = rope(rms(k, ref.f32(p + "norm_k"), m["rms_eps"]), cos[:hd // 2], sin[:hd // 2])
        v = rms_scale(ref.w(p + "w_v") @ h, m["rms_eps"])
        kr, vr = k, v

    a = []
    n = pos + 1 if full else min(pos + 1, m["sliding_window"])
    lo = 0 if full else max(0, pos + 1 - m["sliding_window"])
    kv_, vv_ = k.reshape(-1, hd), v.reshape(-1, hd)     # one row per cached position
    for j in range(nh):
        qj = qh[j * hd:(j + 1) * hd].astype(np.float64)
        sc = kv_[lo:lo + n] @ qj                        # scaling is 1.0 in Gemma 4
        sc = sc - sc.max()
        w = np.exp(sc)
        w /= w.sum()
        a.append((w[:, None] * vv_[lo:lo + n].astype(np.float64)).sum(0))
    a = np.concatenate(a).astype(np.float32)

    h = rms(ref.w(p + "w_o") @ a, ref.f32(p + "norm_post_attn"), m["rms_eps"])
    x = x + h

    h = rms(x, ref.f32(p + "norm_pre_ff"), m["rms_eps"])
    g = gelu_tanh(ref.w(p + "mlp_gate") @ h)
    u = ref.w(p + "mlp_up") @ h
    h = rms(ref.w(p + "mlp_down") @ (g * u), ref.f32(p + "norm_post_ff"), m["rms_eps"])
    x = x + h

    g = gelu_tanh(ref.w(p + "ple_gate") @ x)
    g = g * ple_in
    h = rms(ref.w(p + "ple_proj") @ g, ref.f32(p + "norm_ple"), m["rms_eps"])
    x = x + h
    x = x * ref.f32(p + "layer_scalar")
    return x, kr, vr


def rms_scale(x, eps=1e-6):
    xf = x.astype(np.float64)
    return (xf / np.sqrt((xf ** 2).mean() + eps)).astype(np.float32)


class Model:
    """Whole-model greedy step, mirroring Gemma4TextModel + the tied lm head.

    Kept deliberately close to the HF source so a disagreement means one of the
    two is wrong; the engine is only ever "right" relative to this.
    """

    def __init__(self, ref, max_seq=4096, rope=None):
        self.ref = ref
        self.m = ref.m
        self.max_seq = max_seq
        m = self.m
        self.cache = {i: (np.zeros((max_seq, m["head_dim_full"] if ref.man["layer_types"][i] == "full_attention"
                                        else m["head_dim_slide"]), dtype=np.float32),
                          np.zeros((max_seq, m["head_dim_full"] if ref.man["layer_types"][i] == "full_attention"
                                        else m["head_dim_slide"]), dtype=np.float32))
                      for i in range(m["first_kv_shared"])}
        self.store = {t: self.cache[i] for t, i in m["kv_store_layers"].items()}
        if rope is None:
            from runtime.gemma4_model import rope_tables
            rope = rope_tables(m, max_seq)
        self.cos = {"sliding_attention": rope[0], "full_attention": rope[2]}
        self.sin = {"sliding_attention": rope[1], "full_attention": rope[3]}
        self.H = m["hidden"]
        self.PLE = m["ple_dim"]
        self.nl = m["n_layers"]

    def ple_inputs(self, x, token):
        """per_layer_inputs: (RMSNorm(proj(x)/sqrt(H)) + ple_emb[token][l]*sqrt(ple_dim))/sqrt(2)"""
        m = self.m
        proj = (self.ref.w("ple_model_proj") @ x).astype(np.float64) * (1.0 / np.sqrt(self.H))
        proj = proj.reshape(self.nl, self.PLE)
        e = self.ref.man["layout"]["ple_embed"]
        row = self.ref.row("ple_embed", token, cols=e["shape"][1]).astype(np.float64)
        scale = np.sqrt(self.PLE)
        inv2 = 1.0 / np.sqrt(2.0)
        w = self.ref.f32("ple_proj_norm").astype(np.float64)
        out = np.zeros((self.nl, self.PLE), dtype=np.float32)
        for l in range(self.nl):
            sl = proj[l]
            nrm = sl / np.sqrt((sl ** 2).mean() + m["rms_eps"]) * w
            tok = row[l * self.PLE:(l + 1) * self.PLE] * scale
            out[l] = ((nrm + tok) * inv2).astype(np.float32)
        return out

    def forward(self, token, pos, upto=None):
        """upto=k runs layers [0,k) and returns the stream: lets a test find the
        first layer where the engine and this reference part ways."""
        m, ref = self.m, self.ref
        eps = m["rms_eps"]
        x = (ref.row("embed", token).astype(np.float64) * np.sqrt(self.H)).astype(np.float32)
        ple = self.ple_inputs(x, token)
        for i in range(self.nl if upto is None else upto):
            p = f"L{i:02d}."
            typ = ref.man["layer_types"][i]
            full = typ == "full_attention"
            hd = m["head_dim_full"] if full else m["head_dim_slide"]
            shared = i >= m["first_kv_shared"]
            nh = m["n_heads"]
            cos = self.cos[typ][pos][:hd // 2]
            sin = self.sin[typ][pos][:hd // 2]

            h = rms(x, ref.f32(p + "norm_in"), eps)
            q = ref.w(p + "w_q") @ h
            qn = ref.f32(p + "norm_q")
            q = np.concatenate([rope(rms(q[j * hd:(j + 1) * hd], qn, eps), cos, sin)
                                for j in range(nh)])
            if shared:
                kc, vc = self.store[typ]
                k, v = kc[:pos + 1], vc[:pos + 1]
            else:
                k1 = rope(rms(ref.w(p + "w_k") @ h, ref.f32(p + "norm_k"), eps), cos, sin)
                v1 = rms_scale(ref.w(p + "w_v") @ h, eps)
                self.cache[i][0][pos], self.cache[i][1][pos] = k1, v1
                k, v = self.cache[i][0][:pos + 1], self.cache[i][1][:pos + 1]

            n = pos + 1 if full else min(pos + 1, m["sliding_window"])
            lo = (0 if full else max(0, pos + 1 - m["sliding_window"]))
            att = []
            for j in range(nh):
                qj = q[j * hd:(j + 1) * hd].astype(np.float64)
                sc = k[lo:lo + n].astype(np.float64) @ qj
                sc -= sc.max()
                w = np.exp(sc)
                w /= w.sum()
                att.append((w[:, None] * v[lo:lo + n].astype(np.float64)).sum(0))
            a = np.concatenate(att).astype(np.float32)

            x = x + rms(ref.w(p + "w_o") @ a, ref.f32(p + "norm_post_attn"), eps)
            h = rms(x, ref.f32(p + "norm_pre_ff"), eps)
            g = gelu_tanh(ref.w(p + "mlp_gate") @ h) * (ref.w(p + "mlp_up") @ h)
            inter = m["inter_base"] * (2 if shared else 1)
            assert ref.w(p + "mlp_gate").shape[0] == inter, (ref.w(p + "mlp_gate").shape, inter)
            x = x + rms(ref.w(p + "mlp_down") @ g, ref.f32(p + "norm_post_ff"), eps)

            g = gelu_tanh(ref.w(p + "ple_gate") @ x) * ple[i]
            x = x + rms(ref.w(p + "ple_proj") @ g, ref.f32(p + "norm_ple"), eps)
            x = x * ref.f32(p + "layer_scalar")

        if upto is not None:
            return x, x
        h = rms(x, ref.f32("final_norm"), eps)
        return h, x

    def logits(self, h, chunk=16384):
        """Tied lm head, in row blocks: the table is 262144 x 1536.

        Then final_logit_softcapping. HF returns cap*tanh(x/cap), not the raw dot
        product. Under argmax the two name the same token, which is how this
        reference AND the engine both missed it while still reproducing HF's
        greedy tokens; under temperature the capped distribution is materially
        flatter (E2B, first step of the zh prompt: p(top) 0.795 uncapped against
        0.572 for HF), so an uncapped sampler draws sharper than the model is.
        """
        V = self.m["vocab"]
        out = np.empty(V, dtype=np.float32)
        for s in range(0, V, chunk):
            blk = self.ref.w_row_block("embed", s, min(chunk, V - s))
            out[s:s + blk.shape[0]] = blk @ h
        cap = self.m.get("logit_softcapping")
        if cap:
            out = (cap * np.tanh(out.astype(np.float64) / cap)).astype(np.float32)
        return out
