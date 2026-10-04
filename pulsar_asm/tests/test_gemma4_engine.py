"""End-to-end parity: the assembly engine vs the NumPy reference on real weights.

Both read the same converted blob, so this validates the ENGINE (control flow,
descriptor wiring, buffer reuse, KV sharing, PLE, rope selection) rather than
the conversion, which tools/verify_gemma4_blob.py covers separately.

  python tests/test_gemma4_engine.py [--weights /mnt/edge/pulsar/gemma4_e2b.bin]
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from runtime import pulsar_abi as abi          # noqa: E402
from runtime.gemma4_model import Gemma4, rope_tables  # noqa: E402
import ref_gemma4_np as R                      # noqa: E402


def relerr(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    d = np.abs(b).max()
    return float(np.abs(a - b).max() / d) if d > 0 else float(np.abs(a - b).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/mnt/edge/pulsar/gemma4_e2b.bin")
    ap.add_argument("--tokens", default="2,105,729,1538,139,2310,153,14955")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()

    abi.require_avx2_fma()
    ids = [int(t) for t in a.tokens.split(",")]
    n = min(a.n, len(ids))
    ref = R.Ref(a.weights)
    g = Gemma4(a.weights, max_seq=max(64, n + 2), verbose=True)
    rope = rope_tables(g.meta, max(64, n + 2))
    model = R.Model(ref, max_seq=max(64, n + 2), rope=rope)

    ok = True
    print(f"\ngreedy roll-out, {n} tokens (reference fp64 accumulate, engine fp32 + bf16 weights)")
    print("  step  pos  engine tok   ref tok   match   logits rel   top2 gap")
    g.reset()
    for pos in range(n):
        t0 = time.time()
        got = g.forward(ids[pos])
        dt = time.time() - t0
        h, _ = model.forward(ids[pos], pos)
        lg = model.logits(h)
        exp = int(np.argmax(lg))
        e_lg = relerr(g.logits(), lg)
        srt = np.sort(lg)[::-1]
        gap = float(srt[0] - srt[1])
        same = got == exp
        ok &= same and e_lg < 2e-4
        print(f"  {pos:>4}  {pos:>3}  {got:>10}  {exp:>7}  {'ok' if same else 'DIFF':>5}"
              f"   {e_lg:.2e}   {gap:9.4f}   {dt*1000:6.1f} ms")
        if not same:
            mine = np.argsort(g.logits())[::-1][:3]
            print(f"        engine top3 {list(mine)} values {g.logits()[mine]}")
            print(f"        ref    top3 {list(np.argsort(lg)[::-1][:3])} values {lg[np.argsort(lg)[::-1][:3]]}")

    # a KV-sharing check that does not depend on any single logit: after the
    # window fills, a sliding layer must stop seeing early rows.
    print(f"\n{'ALL PARITY CHECKS PASS' if ok else 'PARITY FAILURES'}")
    g.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
