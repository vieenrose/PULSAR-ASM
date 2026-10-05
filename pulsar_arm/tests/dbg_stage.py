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
        # rotary_emb returns a (cos, sin) tuple
        cos, sin = hf.model.rotary_emb(torch.empty(1, 3, 640),
                                       torch.tensor([[0, 1, 2]]),
                                       layer_type="sliding_attention")
        hqn4 = torch.from_numpy(Qn.reshape(1, 3, 4, 256)).transpose(1, 2)
        hkn1 = torch.from_numpy(Kn.reshape(1, 3, 1, 256)).transpose(1, 2)
        qr, kr = apply_rotary_pos_emb(hqn4, hkn1, cos, sin, unsqueeze_dim=1)
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
        sd_all = {k: v.float().numpy() for k, v in L.state_dict().items()}
        V = np.stack([(sd_all["self_attn.v_proj.weight"] @ x[t]).reshape(1, 256)
                      for t in range(3)])
        avs = []
        for j in range(4):
            sc = kr[0, :, 0, :] @ (qr[0, j, 2, :] / 16.0)
            sc = sc - sc.max()
            w = np.exp(sc).astype(np.float64)
            w /= w.sum()
            avs.append((w[:, None] * V[:, 0, :]).sum(0))
        mav = np.concatenate(avs).astype(np.float32)
        # my mirror's AV the same way from mq/mk
        savs = []
        for j in range(4):
            sc = mk[:, 0, :] @ (mq[2, j, :] / 16.0)
            sc = sc - sc.max()
            w = np.exp(sc).astype(np.float64)
            w /= w.sum()
            savs.append((w[:, None] * V[:, 0, :]).sum(0))
        sav = np.concatenate(savs).astype(np.float32)
        show("attn-out", sav, mav)
        mo = sd_all["self_attn.o_proj.weight"] @ sav
        ho = sd_all["self_attn.o_proj.weight"] @ mav
        show("o_proj", mo, ho)
        mp = rms(mo, sd_all["post_attention_layernorm.weight"])
        hp = rms(ho, sd_all["post_attention_layernorm.weight"])
        show("post_attn", mp, hp)
        # full HF layer-0 output for the end-to-end anchor
        with torch.no_grad():
            full = hf.model(input_ids=torch.tensor([ids]), use_cache=False,
                            output_hidden_states=True)
        ref1 = np.asarray(full.hidden_states[1].float())[0, 2]
        x1 = x[2] + mp
        mh = rms(x1, sd_all["pre_feedforward_layernorm.weight"])
        c1 = np.float32(0.7978845608028654)
        c3 = np.float32(0.044715) * c1
        gg = sd_all["mlp.gate_proj.weight"] @ mh
        uu = sd_all["mlp.up_proj.weight"] @ mh
        gg = (0.5 * gg * (1 + np.tanh(c1 * gg + c3 * gg ** 3))).astype(np.float32) * uu
        dd = rms(sd_all["mlp.down_proj.weight"] @ gg,
                 sd_all["post_feedforward_layernorm.weight"])
        show("layer0", x1 + dd, ref1)


if __name__ == "__main__":
    main()
