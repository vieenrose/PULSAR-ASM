#!/usr/bin/env python3
"""Localize a divergence between the NumPy reference and HF, layer by layer.

Same idea as tools/perlayer_probe.patch for the Pi engine: compare the hidden
state after every decoder layer at the last prompt position, and the FIRST
layer that disagrees names the broken component. Layer types alternate
conv/attention, so the table also says whether conv, attention, or the shared
FFN/norm path is at fault.

Run with ~/venvs/lfm3/bin/python (needs torch).
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ref_lfm3_np import Ref, BLOB                                    # noqa: E402

REPO = "LiquidAI/LFM3-Audio-2B"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="A")
    ap.add_argument("--layers", type=int, default=6, help="how many layers to compare")
    a = ap.parse_args()

    gold = os.path.join(HERE, "..", "..", "..", ".auto", "lfm3",
                        "golden_greedy_%s.json" % a.prompt)
    ids = json.load(open(gold))["prompt_ids"]
    T = len(ids)

    from transformers import AutoModelForMultimodalLM
    hf = AutoModelForMultimodalLM.from_pretrained(
        REPO, trust_remote_code=True, dtype=torch.bfloat16).eval()
    body = hf.model.language_model if hasattr(hf.model, "language_model") else hf.model

    acts = {}

    def mk(i, mod):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            acts[i] = h[0, -1].detach().float().cpu().numpy().copy()
        return hook

    handles = [m.register_forward_hook(mk(i, m)) for i, m in enumerate(body.layers)]

    with torch.no_grad():
        out = hf.model(input_ids=torch.tensor([ids]))
    for h in handles:
        h.remove()
    hf_last = out.last_hidden_state[0, -1].detach().float().cpu().numpy().copy()

    # ---- reference, layer by layer
    r = Ref(BLOB)
    eps = r.m["rms_norm_eps"]
    types = r.m["layer_types"]
    x = r.w("embed_tokens")[np.asarray(ids)]
    cosp, sinp = {}, {}
    for lt in ("full_attention", "sliding_attention"):
        if lt in types:
            cosp[lt], sinp[lt] = r.rope_tables(lt, T)

    print("%-4s %-19s %10s %10s %10s" % ("L", "type", "maxabs", "maxrel", "cos"))
    nlayer = min(a.layers, r.m["n_layers"])
    first_bad = None
    for L in range(nlayer):
        lt = types[L]
        res = x
        h = r.rms(x, r.f32("L%02d.norm_op" % L), eps)
        if lt == "conv":
            h = r.conv_layer(h, L)
        else:
            h = r.attn_layer(h, L, cosp[lt], sinp[lt],
                             r.m["sliding_window"] if lt == "sliding_attention" else None)
        if r.m["use_sandwich_norm"]:
            h = r.rms(h, r.f32("L%02d.norm_post_op" % L), eps)
        x = res + h
        hn = r.rms(x, r.f32("L%02d.norm_ffn" % L), eps)
        u = hn @ r.w("L%02d.mlp_w1" % L).T
        g = hn @ r.w("L%02d.mlp_w3" % L).T
        d = (u * (g / (1 + np.exp(-g)))) @ r.w("L%02d.mlp_w2" % L).T
        if r.m["use_sandwich_norm"]:
            d = r.rms(d, r.f32("L%02d.norm_post_ffn" % L), eps)
        x = x + d

        ref = x[-1]
        got = acts[L]
        a_ = float(np.max(np.abs(ref - got)))
        r_ = a_ / (float(np.max(np.abs(got))) + 1e-9)
        c = float(ref @ got / (np.linalg.norm(ref) * np.linalg.norm(got) + 1e-12))
        bad = c < 0.9999
        if bad and first_bad is None:
            first_bad = L
        print("%-4d %-19s %10.3e %10.3e %10.6f %s"
              % (L, lt, a_, r_, c, "<-- DIVERGES" if bad else ""))
    x = r.rms(x, r.f32("embedding_norm"), eps)[-1]
    a_ = float(np.max(np.abs(x - hf_last)))
    print("final norm vs HF last_hidden_state: maxabs %.3e" % a_)
    print("first divergent layer: %s" % ("none" if first_bad is None else
          "L%02d (%s)" % (first_bad, types[first_bad])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
