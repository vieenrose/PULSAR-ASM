#!/usr/bin/env python3
"""Per-ROW profile of the layer_16 residual (section 39's open item).

Bucket averages hid it: layer_16 is cos ~0.68 over rows 200-374 under every window
(W=32, 64, 96, 128, 192, 256, full), so the mask is not the mechanism. This prints the
cos against position so the SHAPE of the failure is visible: a step at 200, a tail of the
last 32 rows (window/padding), every row, or an oscillation mod 32/256 (the layout
signatures this project has seen before).

Run:  /home/louis/venvs/lfm3/bin/python -I probe_l16.py <ref30> [W16]
"""
import sys, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'ref30')
W16 = int(sys.argv[2]) if len(sys.argv) > 2 else 0        # 0 = full causal for layer 16
os.environ.setdefault('OMP_NUM_THREADS', '4')
import enc_numpy as E
import probe30 as P
E.REF = OUT
feat = np.load(os.path.join(OUT, 'input.npy'))[0].astype(np.float32)
sched = {2: 32, 5: 32, 8: 32, 11: 32, 14: 32, 15: 32}
if W16:
    sched[16] = W16
st = P.spec_stages(E, feat, sched)
proj = E.projector(st['enc_out'])
T = st['layer_16'].shape[0]
print("T=%d  layer_16 window schedule %s" % (T, sched))

def cosrows(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    d = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30)
    return d

for name, mine in (("layer_15", st['layer_15']), ("layer_16", st['layer_16']),
                   ("enc_out", st['enc_out']), ("proj_out", proj)):
    # HF's dump is hidden_00..hidden_17 with hidden_00 == the encoder input (ds_out), so
    # layer_i pairs with hidden_(i+1). Getting this off by one compares a layer against
    # its PREDECESSOR and reads as a 0.8 "residual" everywhere -- probe30's stage_names()
    # table is the authority; this mirror of it must match.
    hf = np.load(os.path.join(OUT, {'layer_15': 'hidden_16.npy',
                                    'layer_16': 'hidden_17.npy',
                                    'enc_out': 'enc_out.npy',
                                    'proj_out': 'proj_out.npy'}[name]))[0]
    d = cosrows(mine, hf)
    print("\n== %s: rows with cos < 0.999 : %d / %d   min %.4f at row %d" %
          (name, int((d < 0.999).sum()), T, d.min(), int(d.argmin())))
    # 25-row blocks: where does it start?
    line = []
    for lo in range(0, T, 25):
        blk = d[lo:lo + 25]
        line.append("%d-%d:%.3f/%d" % (lo, min(lo + 24, T - 1), blk.mean(),
                                       int((blk < 0.999).sum())))
    print("   " + "  ".join(line))
    if name == "layer_16":
        print("   rows 150-260 by 10: " + " ".join("%d:%.4f" % (r, d[r])
                                                  for r in range(150, 261, 10)))
        print("   rows 336-374:      " + " ".join("%d:%.3f" % (r, d[r])
                                                  for r in range(336, T, 4)))
