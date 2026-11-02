#!/usr/bin/env python3
"""Brute-force the form of HF's layer-16 attention against its own captured output.

Layer 16 is now pinned down to one block: operator_norm exact (implied w vs stored w:
corr 1.000000, max|diff| 0.00000), feed_forward exact (cos 1.0000), and self_attn at
cos 0.8483 (min 0.4594) on an exact input. Layers 2,5,8,11,14,15 are cos 1.000000 with a
32-token window, and a W sweep (32/64/96/128/192/256/full), an attention sink and HF's
inclusive window form did not fix layer 16 when applied to the whole encoder.

So: take HF's captured layer-16 input, run MY attention under a grid of plausible
conventions, and score per-row cos against HF's captured attention output. Whatever wins
is the convention the spec has wrong. Runs in seconds per variant because it is one
block on captured activations, not a 17-layer pass.
"""
import os, sys, itertools
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MODEL = '/home/louis/models/lfm3/LFM3-Audio-2B'
SPEECH = os.path.join(HERE, '..', 'speech', 'gate_60s_16k_mono.wav')
SAMPLES = 480000
L = int(os.environ.get("L16", "16"))

import wave
wf = wave.open(SPEECH, 'rb'); sr = wf.getframerate()
pcm = np.frombuffer(wf.readframes(SAMPLES), dtype='<i2').astype(np.float32) / 32768.0

from transformers import AutoModelForMultimodalLM, AutoProcessor
model = AutoModelForMultimodalLM.from_pretrained(MODEL, dtype=torch.float32,
                                                 trust_remote_code=True)
model.eval()
proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
enc = model.model.audio_encoder
for nm, m in enc.named_modules():
    if isinstance(m, torch.nn.ModuleList) and len(m) == 17:
        LAYERS = m; break
cap = {}
def hk(tag, mod):
    def _h(m, inp, kwargs, out):   # with_kwargs=True => (module, args, kwargs, output)
        # this implementation returns a ModelOutput-style dict from self_attn, so pick
        # the first tensor value rather than assuming tuple indexing
        o = out
        if isinstance(o, dict):
            o = next(v for v in o.values() if torch.is_tensor(v))
        elif isinstance(o, (tuple, list)):
            o = o[0]
        cap[tag] = o.detach().float().cpu().numpy()[0]
        cand = [a for a in list(inp) + list((kwargs or {}).values())
                if torch.is_tensor(a) and a.ndim == 3]
        cap[tag + '_in'] = cand[0].detach().float().cpu().numpy()[0]
    return mod.register_forward_hook(_h, with_kwargs=True)
hk('attn', LAYERS[L].self_attn)
feats = proc.feature_extractor([pcm], sampling_rate=sr, return_tensors='pt')
fam = feats.get('feature_attention_mask', feats.get('attention_mask'))
with torch.no_grad():
    model.model.get_audio_features(feats['input_features'], fam)

x = cap['attn_in']; y_hf = cap['attn']
T, HID = x.shape
print("layer %d: attn in %s out %s  rms(in) %.5f rms(out) %.5f" %
      (L, x.shape, y_hf.shape, float(np.sqrt((x ** 2).mean())), float(np.sqrt((y_hf ** 2).mean()))))

import enc_numpy as E
p = "encoder.lfm.lfm.layers.%d.self_attn." % L
Wq, Wk, Wv, Wo = [E.W(p + n) for n in ("q_proj.weight", "k_proj.weight",
                                        "v_proj.weight", "out_proj.weight")]
qn, kn = E.W(p + "q_layernorm.weight"), E.W(p + "k_layernorm.weight")
NH, NKV, HD = 8, 4, 96

def attn(x, win=0, mode="exc", sink=False, scale="rsqrt", norm="qk",
         rope=True, group="div", soft="maxsub"):
    q = (x @ Wq.T).reshape(T, NH, HD)
    k = (x @ Wk.T).reshape(T, NKV, HD)
    v = (x @ Wv.T).reshape(T, NKV, HD)
    if norm == "qk":
        q = E.rms(q, qn); k = E.rms(k, kn)
    if rope:
        cos, sin = E.rope_tables(T)
        c, s = cos[:, None, :], sin[:, None, :]
        q = q * c + E.rot_half(q) * s
        k = k * c + E.rot_half(k) * s
    sc_fac = {"rsqrt": HD ** -0.5, "lin": 1.0 / HD, "one": 1.0,
              "sqrt_hid": HID ** -0.5}[scale]
    i_idx = np.arange(T)[:, None]; j_idx = np.arange(T)[None, :]
    allowed = (j_idx <= i_idx)
    if win:
        allowed = allowed & ((j_idx >= i_idx - win) if mode == "inc"
                             else (j_idx > i_idx - win))
        if sink:
            allowed = allowed | (j_idx == 0)
    out = np.zeros((T, NH, HD), np.float64)
    for hq in range(NH):
        kv = (hq // (NH // NKV)) if group == "div" else (hq % NKV)
        sc = (q[:, hq, :].astype(np.float64) @ k[:, kv, :].astype(np.float64).T) * sc_fac
        sc = np.where(allowed, sc, -np.inf)
        if soft == "maxsub":
            sc = sc - sc.max(-1, keepdims=True)
        pr = np.exp(sc)
        pr = pr / pr.sum(-1, keepdims=True)
        out[:, hq, :] = pr @ v[:, kv, :].astype(np.float64)
    return out.reshape(T, NH * HD) @ Wo.T

def score(a, b=y_hf):
    a = a.astype(np.float64); b = b.astype(np.float64)
    d = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30)
    return float(d.mean()), float(d.min()), float(np.sqrt((a ** 2).mean()))

base = attn(x)
print("baseline (spec as shipped: W=0 full causal, qk-norm, rope, 1/sqrt(96), hq//2): "
      "cos %.4f min %.4f rms %.5f" % score(base))
print("HF's own: rms(out) %.5f\n" % float(np.sqrt((y_hf ** 2).mean())))

rows = []
for win, mode, sink, scale, norm, rope, group, soft in itertools.product(
        [0, 16, 32, 48, 64, 128, 256], ["exc", "inc"], [False, True],
        ["rsqrt", "lin", "one", "sqrt_hid"], ["qk", "none"], [True, False],
        ["div", "mod"], ["maxsub", "raw"]):
    m, mn, rms = score(attn(x, win, mode, sink, scale, norm, rope, group, soft))
    rows.append((m, mn, win, mode, sink, scale, norm, rope, group, soft, rms))
rows.sort(reverse=True)
print("top 12 of %d variants (W, mode, sink, scale, norm, rope, group, soft):" % len(rows))
for r in rows[:12]:
    print("  cos %.6f min %.4f rms %.5f | W=%-4s %-3s sink=%-5s %-8s %-4s rope=%-5s %-3s %s"
          % (r[0], r[1], r[10], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9]))
