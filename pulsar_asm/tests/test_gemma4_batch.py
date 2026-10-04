"""A batched pass must agree with the same tokens fed one at a time.

That equivalence is what speculative decoding spends its whole design on: the
verify pass hands the target B tokens and one weight read instead of B of each,
and it is only legitimate if the answers are the same. So this test does not
compare against NumPy - it compares the engine against itself, which is the
sharper check (any disagreement is a real ordering or cache bug, not a
tolerance).

STATUS: FAILING, and it is a diagnostic, not a gate. Row 0 of a batched pass
agrees with the sequential run's argmax and B=1 parity is exact, but rows >0
disagree and B=4 faults - the batched layer still has an index or stride bug in
it. Until this passes, the engine must only be run at B=1.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.gemma4_model import Gemma4  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/mnt/edge/pulsar/gemma4_e2b.bin")
    ap.add_argument("--b", type=int, default=4)
    ap.add_argument("--pos", type=int, default=6)
    a = ap.parse_args()

    g = Gemma4(a.weights, max_seq=256, verbose=False)
    # arbitrary token ids; positions start away from 0 so the sliding window and
    # the KV-shared caches are both in play
    toks = [2, 105, 729, 1538, 139, 2310, 153, 14955][:a.b]

    g.reset()
    g.pos = a.pos
    seq = [int(g.forward(t)) for t in toks]
    x_seq = g.buf["X"][:g.hidden].copy()
    kv_seq = [c[0].copy() for c in g._caches.values()]

    g.reset()
    bat = [int(v) for v in g.run(toks, pos=a.pos)]
    x_bat = g.buf["X"][:g.hidden].copy()

    ok = True
    same = seq == bat
    ok &= same
    print(f"tokens      {toks}")
    print(f"positions   {list(range(a.pos, a.pos + len(toks)))}")
    print(f"sequential  {seq}")
    print(f"batched     {bat}   {'match' if same else 'MISMATCH'}")

    rel = float(np.abs(x_bat - x_seq).max() / np.abs(x_seq).max())
    ok &= rel < 5e-6
    print(f"final stream rel {rel:.2e}")
    # every cache row the pass could have touched must land in the same place
    rows = slice(0, a.pos + len(toks))
    for i, (k, _) in g._caches.items():
        ks = kv_seq[i][rows]
        kb = k[rows]
        r = float(np.abs(kb - ks).max() / max(np.abs(ks).max(), 1e-30))
        if r > 1e-6:
            print(f"  cache {i} rows {rows.start}:{rows.stop}: rel {r:.2e}")
            ok = False
    print(f"\n{'BATCHED PASS AGREES WITH SEQUENTIAL' if ok else 'BATCHED DISAGREES'}")
    g.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
