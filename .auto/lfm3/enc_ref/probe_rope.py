#!/usr/bin/env python3
"""Pinpoint WHY layer_02 (the first attention layer) is where our derivation leaves HF.

probe30.py localized it exactly: stages ds_out, layer_00, layer_01 (mel -> downsample ->
2 conv layers + 2 FFN blocks) match HF at cos 1.000000 in EVERY position bucket, and
layer_02's INPUT is cos 1.000000 while its INCREMENT is 0.971. The sliding-window
hypothesis is dead: W=192/256/512 give the same cos as full causal.

What is left, in the attention block only, that grows with position:
  - RoPE (theta, partial rotary fraction, applied to which dims, norm-vs-rope order)
  - per-head RMSNorm placement (probe30 says the norm'd Q/K enter the diverging stage)
This script separates them by measuring HF's own tensors:
  q_pre_rope, k_pre_rope, q_post_rope, k_post_rope   for encoder layer LAYER (default 2)
captured with forward hooks, and the same four from the numpy spec. Per-bucket cos:
  q_pre matching + q_post diverging  => RoPE is the bug (theta/fraction/order)
  q_pre diverging                    => the projection/norm path is the bug
Usage: /home/louis/venvs/lfm3/bin/python -I probe_rope.py [--layer 2] [--samples 480000]
"""
import argparse, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MODEL = '/home/louis/models/lfm3/LFM3-Audio-2B'
AP = argparse.ArgumentParser()
AP.add_argument('--layer', type=int, default=2)
AP.add_argument('--samples', type=int, default=480000)
AP.add_argument('--speech', default=os.path.join(HERE, '..', 'speech',
                                                 'gate_60s_16k_mono.wav'))
A = AP.parse_args()

import wave
wf = wave.open(A.speech, 'rb')
import struct
raw = wf.readframes(A.samples); sr = wf.getframerate()
pcm = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0

from transformers import AutoModelForMultimodalLM, AutoProcessor
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL, dtype=torch.float32, trust_remote_code=True)
model.eval()
proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
enc = model.model.audio_encoder
def find_layers(m, path=''):
    # the encoder wraps an LFM2 model; locate the ModuleList of decoder layers wherever
    # it hangs (enc.layers, enc.model.layers, ...). Print the path so the provenance is
    # on the record, not assumed.
    best = []
    for nm, mod in m.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 8 and all(
                'DecoderLayer' in type(x).__name__ or 'Layer' in type(x).__name__
                for x in mod):
            best.append((len(nm.split('.')), nm, mod))
    if not best:
        raise SystemExit('no decoder-layer ModuleList found under %s' % type(m).__name__)
    best.sort()
    print('  layer list at %r (%d layers, %s)' % (best[0][1], len(best[0][2]),
                                                  type(best[0][2][0]).__name__))
    return best[0][2], best[0][1]
LAYERS, LPATH = find_layers(enc)
L = A.layer
lay = LAYERS[L]
print("layer %d: %s" % (L, type(lay).__name__))
print("  attn %s  norm %s  rotary attrs: %s" % (
    type(getattr(lay, 'attn', getattr(lay, 'self_attn', None))).__name__,
    type(getattr(lay, 'q_norm', None)).__name__,
    [a for a in dir(enc) if 'rotar' in a or 'rope' in a]))
cfg = enc.config
for k in ('rope_theta', 'partial_rotary_factor', 'head_dim', 'num_attention_heads',
          'num_key_value_heads', 'sliding_window', 'max_position_embeddings',
          'norm_eps', 'rope_scaling'):
    print("  cfg.%s = %s" % (k, getattr(cfg, k, '<absent>')))

cap = {}
def hook(name, fn):
    def _h(mod, inp, out):
        cap[name] = fn(mod, inp, out)
    return _h

def cat_in_out(mod, inp, out):
    x = inp[0].detach().float().cpu().numpy() if hasattr(inp[0], 'shape') else None
    return (x, np.asarray(out.detach().float().cpu()) if hasattr(out, 'shape')
            else [np.asarray(o.detach().float().cpu()) for o in out])

attn = getattr(lay, 'attn', None) or getattr(lay, 'self_attn', None)
handles = [attn.q_proj.register_forward_hook(hook('q_pre', cat_in_out)),
           attn.k_proj.register_forward_hook(hook('k_pre', cat_in_out))]
rot = None
for nm, m in enc.named_modules():
    if m.__class__.__name__.lower().find('rotary') >= 0 or nm.endswith('rotary'):
        rot = m; print("  rotary module: %s (%s)" % (nm, type(m).__name__)); break
if rot is not None:
    handles.append(rot.register_forward_hook(hook('rot', cat_in_out)))
feats = proc.feature_extractor([pcm], sampling_rate=sr, return_attention_mask=True,
                               padding='max_length', return_tensors='pt')
Tm = int(feats['num_frames'][0]); Tt = (Tm + 7) // 8
print("mel frames %d -> tokens %d" % (Tm, Tt))
with torch.no_grad():
    hf = model.model.get_audio_features(feats['input_features'],
                                        feats['attention_mask'], return_dict=True)
for h in handles:
    h.remove()

def dump(name, nheads=None):
    ent = cap.get(name)
    if ent is None:
        print("  %s: NOT CAPTURED" % name); return None
    x, y = ent
    arr = y if isinstance(y, np.ndarray) else np.stack(y)
    print("  %s: in %s out %s" % (name, None if x is None else x.shape, arr.shape))
    return x, arr

dump('q_pre'); dump('k_pre'); dump('rot')

# (our side is the numpy spec's own layer code; probe30.py compares stage outputs, this
# script's job is HF's structure + the four q/k tensors, and the comparison is done in
# probe_rope_cmp.py so a broken spec load cannot hide HF's numbers.)

json.dump({'Tm': Tm, 'Tt': Tt,
           'captured': {k: (None if v is None else
                            [None if v[0] is None else list(v[0].shape),
                             list(v[1].shape) if isinstance(v[1], np.ndarray)
                             else [list(a.shape) for a in v[1]]])
                        for k, v in cap.items()}},
          open('/tmp/probe_rope_cap.json', 'w'), indent=1)
print("[json] /tmp/probe_rope_cap.json")
