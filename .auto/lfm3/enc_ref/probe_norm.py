#!/usr/bin/env python3
"""Is HF's Lfm2RMSNorm `w * x/rms(x)` or `(1+w) * x/rms(x)`?

probe_l16b.py: at layer 16 my op_norm output has rms 0.42835, HF's 0.96839, on the SAME
captured input, while the FFN of the same layer is cos 1.0000. A norm's output rms is
rms(weight) if the weight multiplies, and rms(1+weight) if it is a Gemma-style
"1 + w". rms(w)=0.428 and rms(1+w)=0.968 is exactly the pair observed, so this script
measures HF's implied per-channel weight directly:  w_hf = norm_out / (x/rms(x)),
and prints its correlation with the stored weight, with 1+weight, and the layer's rms
numbers. Layer 15 (which matches HF at cos 1.000000) is the control.
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MODEL = '/home/louis/models/lfm3/LFM3-Audio-2B'
SPEECH = os.path.join(HERE, '..', 'speech', 'gate_60s_16k_mono.wav')
SAMPLES = 480000

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
print("Lfm2RMSNorm source (transformers ships it, so read it, don't guess):")
import inspect
from transformers.models.lfm2.modeling_lfm2 import Lfm2RMSNorm
print(inspect.getsource(Lfm2RMSNorm))

cap = {}
def hk(tag, mod):
    def _h(m, inp, out):
        o = out[0] if isinstance(out, (tuple, list)) else out
        cap[tag + '_in'] = inp[0].detach().float().cpu().numpy()[0]
        cap[tag] = o.detach().float().cpu().numpy()[0]
    return mod.register_forward_hook(_h)
for L in (15, 16):
    hk('L%d' % L, LAYERS[L].operator_norm)
# The live module's own weight is the authority for what HF multiplied by. If the
# checkpoint tensor for that name is not what the module holds, that IS the finding.
from safetensors import safe_open
sf = safe_open(os.path.join(MODEL, 'model.safetensors'), 'pt')
allk = list(sf.keys())
for L in (15, 16):
    live = LAYERS[L].operator_norm.weight.detach().float().cpu().numpy().astype(np.float64)
    name = "encoder.lfm.lfm.layers.%d.operator_norm.weight" % L
    ck = sf.get_tensor(name).float().numpy().astype(np.float64) if name in allk else None
    print("layer %d: live module rms %.5f | checkpoint %s rms %s | eps %s"
          % (L, float(np.sqrt((live ** 2).mean())), name,
             "ABSENT" if ck is None else "%.5f" % float(np.sqrt((ck ** 2).mean())),
             LAYERS[L].operator_norm.variance_epsilon))
    if ck is not None:
        print("   max|live - checkpoint| = %.6f  (live head: %s)"
              % (float(np.abs(live - ck).max()), np.round(live[:6], 5)))
    print("   keys mentioning layers.%d: %s" % (L, [k for k in allk
                                                    if 'layers.%d.' % L in k][:14]))

feats = proc.feature_extractor([pcm], sampling_rate=sr, return_tensors='pt')
fam = feats.get('feature_attention_mask', feats.get('attention_mask'))
with torch.no_grad():
    model.model.get_audio_features(feats['input_features'], fam)

import enc_numpy as E
for L in (15, 16):
    x, y = cap['L%d_in' % L], cap['L%d' % L]
    w = E.W("encoder.lfm.lfm.layers.%d.operator_norm.weight" % L).astype(np.float64)
    x64 = x.astype(np.float64)
    n = x64 / np.sqrt((x64 ** 2).mean(-1, keepdims=True) + 1e-5)
    imp = (y.astype(np.float64) / np.where(np.abs(n) > 1e-8, n, np.nan))
    m = np.isfinite(imp)
    a = imp[m]
    def cor(v):
        v = np.asarray(v, dtype=np.float64).ravel()
        # implied is per (row, channel); compare against the per-channel weight by
        # averaging implied over rows first, which is the estimator that is stable when
        # |x| is large (the norm is linear in the channel, so the channel mean of
        # implied/n * n recovers w exactly up to fp noise).
        iw = imp.mean(0)
        v = v[:iw.size]
        return float(np.corrcoef(iw, v)[0, 1]), float(np.abs(iw - v).max())
    print("\nlayer %d: rms(norm_out)=%.5f  rms(w)=%.5f  rms(1+w)=%.5f" %
          (L, float(np.sqrt((y ** 2).mean())), float(np.sqrt((w ** 2).mean())),
           float(np.sqrt(((1 + w) ** 2).mean()))))
    for tag, v in (("w", w), ("1+w", 1 + w)):
        c, mx = cor(v)
        print("   implied w vs %-4s : corr %.6f  max|diff| %.5f" % (tag, c, mx))
