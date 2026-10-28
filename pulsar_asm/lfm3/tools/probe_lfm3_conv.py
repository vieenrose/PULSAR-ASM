#!/usr/bin/env python3
"""Isolate LFM3 layer 0 (a conv layer): feed the NumPy reference's intermediate
into HF's own sub-modules and compare stage by stage, so the broken stage is
named rather than guessed.

Stages: in_proj -> (B, C, xg split) -> B*xg -> depthwise causal conv -> *C ->
out_proj.
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


def cmp(name, a, b):
    a = np.asarray(a).astype(np.float64).reshape(np.asarray(b).shape)
    b = np.asarray(b).astype(np.float64)
    ma = float(np.max(np.abs(a - b)))
    mr = ma / (float(np.max(np.abs(b))) + 1e-9)
    ok = ma < 2e-2
    print("  %-26s maxabs %10.3e  maxrel %9.3e  %s"
          % (name, ma, mr, "ok" if ok else "<-- WRONG" if mr > 1e-2 else "marginal"))
    return ok


def main():
    gold = os.path.join(HERE, "..", "..", "..", ".auto", "lfm3",
                        "golden_greedy_A.json")
    ids = json.load(open(gold))["prompt_ids"]
    T = len(ids)

    from transformers import AutoModelForMultimodalLM
    hf = AutoModelForMultimodalLM.from_pretrained(
        REPO, trust_remote_code=True, dtype=torch.bfloat16).eval()
    body = hf.model.language_model if hasattr(hf.model, "language_model") else hf.model
    conv = body.layers[0].conv

    r = Ref(BLOB)
    eps = r.m["rms_norm_eps"]
    x = r.w("embed_tokens")[np.asarray(ids)]
    h = r.rms(x, r.f32("L00.norm_op"), eps).astype(np.float32)
    H = h.shape[1]

    # HF on the same input, bf16 like the model
    ht = torch.tensor(h)[None].to(torch.bfloat16)
    with torch.no_grad():
        bcx_hf = conv.in_proj(ht)[0].float().numpy()        # [T, 3H]
        # HF's own split: transpose(-1,-2) then chunk(3, dim=-2)
        bcx_t = conv.in_proj(ht).transpose(-1, -2)
        B_hf, C_hf, x_hf = [t[0].float().numpy() for t in bcx_t.chunk(3, dim=-2)]
        gated = B_hf * x_hf
        conv_hf = conv._conv(torch.tensor(gated).unsqueeze(0).to(torch.bfloat16)
                             ).float().numpy()[0]          # [H, T]
        y_hf = C_hf * conv_hf                              # [H, T]
        out_hf = conv.out_proj(torch.tensor(y_hf.T).to(torch.bfloat16)
                               ).float().numpy()           # [T, H]

    # reference
    p = r.w("L00.conv_in_proj")
    bcx = h @ p.T
    B, C, xg = bcx[:, 0:H], bcx[:, H:2 * H], bcx[:, 2 * H:3 * H]
    cw = r.w("L00.conv_w").reshape(H, 7)
    gp = np.concatenate([np.zeros((6, H), np.float32), B * xg], axis=0)
    cout = np.zeros((T, H), np.float32)
    for k in range(7):
        cout += gp[k:k + T] * cw[:, k][None, :]

    print("L00 conv stage isolation (T=%d, H=%d):" % (T, H))
    cmp("in_proj (BCx)", bcx, bcx_hf)
    cmp("split B", B, B_hf.T)
    cmp("split C", C, C_hf.T)
    cmp("split xg", xg, x_hf.T)
    cmp("B*xg -> conv", cout, conv_hf.T)
    cmp("conv out_proj", (C * cout) @ r.w("L00.conv_out_proj").T, out_hf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
