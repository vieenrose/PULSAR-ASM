#!/usr/bin/env python3
"""Stage-by-stage: numpy mirror vs HF's own modules. First big gap wins."""
import os
import sys

import numpy as np
import torch

REPO = os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-3-270m-it-qat-q4_0-unquantized/snapshots/8f726c6a497fd439f0d6f726e52f8e3b439a26e5")

from transformers import AutoModelForCausalLM  # noqa: E402
from transformers.models.gemma3.modeling_gemma3 import (  # noqa: E402
    apply_rotary_pos_emb)


def rms(x, w, eps=1e-6):
    return (x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * (1 + w)).astype(np.float32)


def rope(v, c, s):
    h = len(v) // 2
    a, b = v[:h].copy(), v[h:].copy()
    return np.concatenate([a * c - b * s, b * c + a * s]).astype(np.float32)


def show(name, a, b):
    d = float(np.abs(np.asarray(a) - np.asarray(b)).max())
    print(f"  {name:14s} max|d|={d:.4e}", flush=True)


def main():
    hf = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32, device_map="cpu")
    hf.eval()
    ids = [2, 107, 1567]
    L = hf.model.layers[0]
    att = L.self_attn
    with torch.no_grad():
        h = hf.model.embed_tokens(torch.tensor([ids])).float().numpy()[0]
        sd = {k: v.float().numpy() for k, v in L.state_dict().items()}
        x = rms(h, sd["input_layernorm.weight"])
        Wq, Wk = sd["self_attn.q_proj.weight"], sd["self_attn.k_proj.weight"]
        nq, nk = sd["self_attn.q_norm.weight"], sd["self_attn.k_norm.weight"]
        Q = np.stack([(Wq @ x[t]).reshape(4, 256) for t in range(3)])
        K = np.stack([(Wk @ x[t]).reshape(1, 256) for t in range(3)])
        Qn = np.stack([[rms(Q[t, j], nq) for j in range(4)] for t in range(3)])
        Kn = np.stack([[rms(K[t, j], nk) for j in range(1)] for t in range(3)])
        # HF's own qn/kn
        q4 = torch.from_numpy(Q.reshape(1, 3, 4, 256)).transpose(1, 2)  # B,H,T,D
        k1 = torch.from_numpy(K.reshape(1, 3, 1, 256)).transpose(1, 2)
        hqn = att.q_norm(q4).numpy().transpose(0, 2, 1, 3).reshape(3, 4, 256)
        hkn = att.k_norm(k1).numpy().transpose(0, 2, 1, 3).reshape(3, 1, 256)
        show("qn", Qn, hqn)
        show("kn", Kn, hkn)
        cos, sin = hf.model.rotary_emb(torch.empty(1, 3, 640),
                                       torch.tensor([[0, 1, 2]]),
        # rotary_emb returns tuple already; handle both
        if isinstance(cos, tuple):
            cos, sin = cos
        qr, kr = apply_rotary_pos_emb(q4, k1, cos, sin, unsqueeze_dim=1)
        qr = qr.numpy().transpose(0, 2, 1, 3).reshape(3, 4, 256)
        kr = kr.numpy().transpose(0, 2, 1, 3).reshape(3, 1, 256)
        th = 10000.0
        inv = 1.0 / (th ** (np.arange(0, 256, 2) / 256))
        cr = [np.cos(p * inv).astype(np.float32) for p in range(3)]
        sr = [np.sin(p * inv).astype(np.float32) for p in range(3)]
        mq = np.stack([[rope(Qn[t, j], cr[t], sr[t]) for j in range(4)] for t in range(3)])
        mk = np.stack([[rope(Kn[t, j], cr[t], sr[t]) for j in range(1)] for t in range(3)])
        show("qrope", mq, qr)
        show("krope", mk, kr)


if __name__ == "__main__":
    main()
