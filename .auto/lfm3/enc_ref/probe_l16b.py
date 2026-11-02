#!/usr/bin/env python3
"""Attribute layer_16's uniform cos ~0.71 (section 39/40) to a SUB-STAGE.

What is known: layer_15 is cos 1.000000 at all 375 rows against HF's canonical dump, and
layer_16's output is cos 0.67-0.73 at EVERY row -- no position trend, invariant to a
32/64/96/128/192/256/full window, to an attention sink, and to HF's inclusive window
form. So it is not the mask. This script splits the layer into the pieces the C reference
and CUDA encoder implement separately, on HF's own captured activations:

    x_in -> op = attn(operator_norm(x_in)) -> x1 = x_in + op
         -> y  = mlp(ffn_norm(x1))          -> out = x1 + y

and reports per-row cos for h (the norm'd attention input), op (attention output), x1,
the norm'd mlp input, y (mlp output) and out. Whichever sub-stage first drops to ~0.7 is
the mechanism; if op is fine and y is fine, the layer's own norm weights are the suspect.

Usage: /home/louis/venvs/lfm3/bin/python -I probe_l16b.py
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MODEL = '/home/louis/models/lfm3/LFM3-Audio-2B'
SPEECH = os.path.join(HERE, '..', 'speech', 'gate_60s_16k_mono.wav')
L = 16
SAMPLES = 480000

import wave, struct
wf = wave.open(SPEECH, 'rb'); sr = wf.getframerate()
pcm = np.frombuffer(wf.readframes(SAMPLES), dtype='<i2').astype(np.float32) / 32768.0

from transformers import AutoModelForMultimodalLM, AutoProcessor
model = AutoModelForMultimodalLM.from_pretrained(MODEL, dtype=torch.float32,
                                                 trust_remote_code=True)
model.eval()
proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
enc = model.model.audio_encoder
LAYERS = enc.lfm.layers if hasattr(enc, 'lfm') else None
if LAYERS is None:
    for nm, m in enc.named_modules():
        if isinstance(m, torch.nn.ModuleList) and len(m) == 17:
            LAYERS = m; print("layer list at %r" % nm); break
lay = LAYERS[L]
print("layer %d children: %s" % (L, [(n, type(m).__name__) for n, m in lay.named_children()]))

cap = {}
def hk(name):
    def _h(mod, inp, out):
        def npz(x):
            if isinstance(x, (tuple, list)):        # self_attn returns (hidden, attn_weights)
                x = x[0]
            return x.detach().float().cpu().numpy() if torch.is_tensor(x) else x
        # some children are invoked with keyword-only args (the norms are, in this
        # implementation), so inp can be empty -- record None rather than crash.
        cap[name + '_in'] = npz(inp[0]) if len(inp) else None
        cap[name] = npz(out)
    return _h
hs = []
for nm, m in lay.named_children():
    hs.append(m.register_forward_hook(hk('l16.' + nm)))
feats = proc.feature_extractor([pcm], sampling_rate=sr, return_tensors='pt')
fam = feats.get('feature_attention_mask', feats.get('attention_mask'))
with torch.no_grad():
    model.model.get_audio_features(feats['input_features'], fam)
for h in hs:
    h.remove()
for k in sorted(cap):
    print("  captured %-28s %s" % (k, "None (kw-only call)" if cap[k] is None
                                               else cap[k].shape))

def sq(a):
    if a is None:
        return None
    a = np.asarray(a)
    return a[0] if a.ndim == 3 and a.shape[0] == 1 else a
T = sq(cap['l16.self_attn']).shape[0]
x_in = sq(cap.get('l16.operator_norm_in'))
if x_in is None:
    x_in = sq(cap.get('l16.self_attn_in'))
if x_in is None:
    sys.exit("no layer input captured; keys: %s" % sorted(cap))
hf = {}
for k in ('operator_norm', 'self_attn', 'ffn_norm', 'feed_forward'):
    for pref in ('l16.' + k, 'l16.' + k + '_in'):
        if pref in cap and cap[pref] is not None:
            hf[pref] = sq(cap[pref])
print("x_in %s  attn_out %s  mlp_out %s" % (x_in.shape,
      hf['l16.self_attn'].shape, hf['l16.feed_forward'].shape))

# our side, on HF's own x_in, using the spec's primitives and HF's weights
import enc_numpy as E
p = "encoder.lfm.lfm.layers.%d." % L
def cos(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    d = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30)
    return d
def rep(tag, mine, theirs):
    d = cos(mine, theirs)
    print("  %-22s cos mean %.4f  min %.4f  rows<0.999 %d/%d | rms mine %.5f theirs %.5f"
          % (tag, d.mean(), d.min(), int((d < 0.999).sum()), len(d),
             float(np.sqrt((mine.astype(np.float64) ** 2).mean())),
             float(np.sqrt((theirs.astype(np.float64) ** 2).mean()))))

mine_h = E.rms(x_in, E.W(p + "operator_norm.weight"))
rep("h = op_norm(x)", mine_h, hf['l16.operator_norm_in'])
mine_op = E.attn_block(mine_h, L, *E.rope_tables(T))
rep("op = attn(h)", mine_op, hf['l16.self_attn'])
x1 = x_in + hf['l16.self_attn']                       # continue from HF's residual, isolate stages
rep("ffn_norm(x1) [HF x1]", E.rms(x1, E.W(p + "ffn_norm.weight")), hf['l16.ffn_norm_in'])
rep("y = mlp(ffn_norm)", E.mlp(E.rms(x1, E.W(p + "ffn_norm.weight")), L), hf['l16.feed_forward'])
mine_x1 = x_in + mine_op
mine_out = mine_x1 + E.mlp(E.rms(mine_x1, E.W(p + "ffn_norm.weight")), L)
rep("out (all mine)", mine_out, sq(cap['l16.mlp_in']) if False else np.load(
    os.path.join(HERE, 'ref30', 'hidden_17.npy'))[0])
