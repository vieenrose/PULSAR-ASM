"""check_live.py -- live-loop gate: the M2 shell's records vs the HF oracle.

Usage: check_live.py <oracle_dir> <live_prefix>
Compares slot by slot, stopping at the first divergence with full context:
 (a) slots.txt line equality (mode machine + sampler decisions),
 (b) frames.i32 equality against the oracle's RETURNED codes (EOS override included),
 (c) text.i32 equality.
A live run with prescribed orders that passes here has reproduced HF's stream with
engine-sampled text and engine-driven MaskGIT on live hiddens -- the S6 end state.
"""
import os
import sys

import numpy as np

G, L = sys.argv[1], sys.argv[2]

want_slots = open(os.path.join(G, "slots.txt")).read().splitlines()
got_slots = open(L + ".slots.txt").read().splitlines()
ok = True
for i, (w, g) in enumerate(zip(want_slots, got_slots)):
    if w != g:
        print("FIRST DIVERGENCE at slot %d: oracle %s, live %s" % (i, w, g))
        print("  oracle slots around: %s" % want_slots[max(0, i - 3):i + 2])
        print("  live slots around:   %s" % got_slots[max(0, i - 3):i + 2])
        ok = False
        break
if len(want_slots) != len(got_slots):
    print("SLOT COUNT: oracle %d, live %d" % (len(want_slots), len(got_slots)))
    ok = False
else:
    print("slots: %d/%d exact" % (len(got_slots), len(want_slots)))

want_fr = np.fromfile(os.path.join(G, "audio_codes.i32"), dtype=np.int32).reshape(12, -1).T
got_fr = np.fromfile(L + ".frames.i32", dtype=np.int32)
got_fr = got_fr.reshape(-1, 12)
nf = min(len(want_fr), len(got_fr))
for i in range(nf):
    if not (want_fr[i] == got_fr[i]).all():
        d = np.argwhere(want_fr[i] != got_fr[i]).ravel().tolist()
        print("FIRST FRAME DIVERGENCE at frame %d, codebooks %s" % (i, d))
        print("  oracle: %s" % want_fr[i].tolist())
        print("  live:   %s" % got_fr[i].tolist())
        ok = False
        break
else:
    print("frames: compared %d, all exact" % nf)
if len(want_fr) != len(got_fr):
    print("FRAME COUNT: oracle %d, live %d" % (len(want_fr), len(got_fr)))
    ok = False

want_tx = np.fromfile(os.path.join(G, "text.i32"), dtype=np.int32)
got_tx = np.fromfile(L + ".text.i32", dtype=np.int32)
nt = min(len(want_tx), len(got_tx))
for i in range(nt):
    if want_tx[i] != got_tx[i]:
        print("FIRST TEXT DIVERGENCE at text slot %d: oracle %d, live %d"
              % (i, int(want_tx[i]), int(got_tx[i])))
        ok = False
        break
else:
    print("text: compared %d, all exact" % nt)
if len(want_tx) != len(got_tx):
    print("TEXT COUNT: oracle %d, live %d" % (len(want_tx), len(got_tx)))
    ok = False

print("LIVE GATE: %s" % ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
