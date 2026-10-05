"""NumPy mirror of Gemma4AssistantForCausalLM - the MTP drafter.

Reads the converted assistant blob and the target's KV cache, exactly as
Gemma4AssistantForCausalLM does when HF drives assisted decoding:

  inputs_embeds = cat(target_embedding(last_seen_token), h_prev)   # 1536 + 1536
  h_prev        = post_projection(assistant hidden), or the target's
                  pre-final-norm hidden for the first draft of a round
  position_ids  = CONSTANT for the whole drafting round
  K/V           = the target's shared KV, never written by the assistant

The logits come from the masked embedder: 2048 centroid dots, the top 32
clusters, and 32 * 128 = 4096 gathered lm_head rows. Nothing else in the 262144
vocab can win, so the argmax over those 4096 is the argmax over the vocab.
"""

import json

import numpy as np

from ref_gemma4_np import rms, rms_scale, gelu_tanh, rope


class Assist:
    def __init__(self, blob, target_ref, kv, meta=None):
        self.f = open(blob, "rb")
        self.man = json.load(open(blob + ".manifest.json"))
        self.m = self.man["meta"]
        self.t = target_ref                 # target Ref: embedding table lives there
        self.kv = kv                        # {"sliding_attention": (K, V), ...} rows
        self.pos_rows = None

    def off(self, name):
        return self.man["layout"][name]["offset"]

    def f32(self, name):
        e = self.man["layout"][name]
        self.f.seek(e["offset"])
        return np.frombuffer(self.f.read(e["bytes"]), dtype=np.float32).copy()

    def w(self, name):
        e = self.man["layout"][name]
        self.f.seek(e["offset"])
        u = np.frombuffer(self.f.read(e["bytes"]), dtype=np.uint16).astype(np.uint32)
        return ((u << 16).view(np.float32)).reshape(e["shape"])

    def rows(self, name, idx):
        """Gather rows by index out of a bf16 [rows, cols] table.

        Mapped, not read: the lm_head table is 262144 x 256 and the masked
        embedder only ever touches 4096 rows per step, so widening the whole
        table (268 MB of fp32) per draft token dominates the run time.
        """
        e = self.man["layout"][name]
        cols = e["shape"][1]
        if not hasattr(self, "_map"):
            self._map = {}
        if name not in self._map:
            self.f.seek(0, 2)
            import mmap
            self._map[name] = np.frombuffer(
                mmap.mmap(self.f.fileno(), 0, prot=mmap.PROT_READ), dtype=np.uint16)
        whole = self._map[name]
        n = e["bytes"] // 2 // cols
        u = whole[e["offset"] // 2: e["offset"] // 2 + n * cols].reshape(n, cols)[idx]
        return ((u.astype(np.uint32) << 16).view(np.float32))

    # -------------------------------------------------------------------------
    def layer(self, x, i, cos, sin, pos):   # cos/sin are [max_seq, hd/2] tables
        m = self.m
        eps = m["rms_eps"]
        full = m["layer_types"][i] == "full_attention"
        hd = m["head_dim_full"] if full else m["head_dim_slide"]
        nh = m["n_heads"]
        p = f"L{i}."
        h = rms(x, self.f32(p + "norm_in"), eps)
        q = self.w(p + "w_q") @ h
        qn = self.f32(p + "norm_q")
        K, V = self.kv["full_attention" if full else "sliding_attention"]
        n = pos + 1 if full else min(pos + 1, m["sliding_window"])
        lo = 0 if full else max(0, pos + 1 - m["sliding_window"])
        kr = K[lo:lo + n].astype(np.float64)
        vr = V[lo:lo + n].astype(np.float64)
        c, s = cos[pos, :hd // 2], sin[pos, :hd // 2]
        att = []
        for j in range(nh):
            qj = rope(rms(q[j * hd:(j + 1) * hd], qn, eps), c, s).astype(np.float64)
            sc = kr @ qj
            sc -= sc.max()
            w = np.exp(sc)
            w /= w.sum()
            att.append((w[:, None] * vr).sum(0))
        a = np.concatenate(att).astype(np.float32)
        x = x + rms(self.w(p + "w_o") @ a, self.f32(p + "norm_pa"), eps)
        h = rms(x, self.f32(p + "norm_pf"), eps)
        g = gelu_tanh(self.w(p + "mlp_gate") @ h) * (self.w(p + "mlp_up") @ h)
        x = x + rms(self.w(p + "mlp_down") @ g.astype(np.float32), self.f32(p + "norm_pff"), eps)
        return x * self.f32(p + "scalar")[0]

    def draft_logits(self, h):
        """Return (best_token, candidate tokens, candidate logits)."""
        m = self.m
        C, TK, VPC = m["num_centroids"], m["centroid_top_k"], m["vocab_per_centroid"]
        cl = self.w("centroids") @ h
        top = np.argpartition(-cl, TK - 1)[:TK]
        order = self._order()
        cands = order.reshape(C, VPC)[top].reshape(-1)
        lg = self.rows("embed", cands) @ h
        return int(cands[np.argmax(lg)]), cands, lg

    def _order(self):
        if not hasattr(self, "_ord"):
            e = self.man["layout"]["tok_order"]
            self.f.seek(e["offset"])
            self._ord = np.frombuffer(self.f.read(e["bytes"]), dtype=np.int32).copy()
        return self._ord

    def step(self, prev_tok, h_prev, pos, cos_s, sin_s, cos_f, sin_f):
        """One drafting step: (previous token, previous hidden) -> (token, hidden)."""
        m = self.m
        H = m["backbone_hidden"]
        emb = self.t.row("embed", prev_tok).astype(np.float32) * np.float32(np.sqrt(H))
        x = self.w("pre_proj") @ np.concatenate([emb, h_prev.astype(np.float32)])
        for i in range(m["n_layers"]):
            if m["layer_types"][i] == "full_attention":
                x = self.layer(x, i, cos_f, sin_f, pos)
            else:
                x = self.layer(x, i, cos_s, sin_s, pos)
        h = rms(x, self.f32("final_norm"), m["rms_eps"])
        tok, _, _ = self.draft_logits(h)
        return tok, (self.w("post_proj") @ h).astype(np.float32)

    def draft(self, prev_tok, h_prev, pos, n, tables):
        """The HF drafting loop: same constant position, stateless per step."""
        cos_s, sin_s, cos_f, sin_f = tables
        toks = []
        h = h_prev
        for _ in range(n):
            prev_tok, h = self.step(prev_tok, h, pos, cos_s, sin_s, cos_f, sin_f)
            toks.append(int(prev_tok))
        return toks
