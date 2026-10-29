#!/usr/bin/env python
"""Long-context quality of the LFM_H2 arms (the audio-deployment question).

#151 re-verified H2's SPEED at 300 positions (-5.4%, same ratio as prompt A).
The user's pending purchase decision is made of the SHORT-prompt quality
record (#141: KL A/C/K 0.0016/0.0006/0.0005). Nothing scored H2 at 300-token
context -- the regime the raw arm's fp16 subnormal exposure (28% of one
layer's inputs) was flagged for, and the regime the engine will actually
serve for audio.

Design (divergence_floor.py, long):
  reference  = HF fp16 greedy from a fixed 300-id prompt, 16 new tokens
               (fp16 is the strictly-more-precise reference that fits the
               host GPUs, #114's argument verbatim; the arms are scored on
               the reference's own states -> on-manifold by construction)
  arm shipped = engine teacher-forced on that sequence, LFM_H2 unset
  arm H2=1    = same, LFM_H2=1

Reading: KL(ref||arm) over the last 16 states (context 300-316) vs the
short-prompt record. If H2's KL stays at its short-prompt level and top-1
agreement is unchanged, the -5.4% has no long-context quality caveat.
"""
import argparse, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import traj_drift, divergence_floor
from divergence_floor import load, logits_at, greedy, kl_curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=300)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    import random
    import numpy as np
    random.seed(a.seed)
    ids = [random.randrange(4, 120000) for _ in range(a.ctx)]
    print("== long-context quality: %d ids, %d gen, reference HF fp16 ==" % (a.ctx, a.gen))

    ref = load("float16")
    traj = greedy(ref, ids, a.gen)
    n = len(traj)
    R = logits_at(ref, traj, n)
    del ref
    import torch
    torch.cuda.empty_cache()
    print("  trajectory %d tokens; gen tail: %s" % (n, traj[a.ctx:][:8]))

    for name, env in [("shipped", ""), ("H2=1", "env LFM_H2=1 ")]:
        traj_drift.ENVQ = env
        toks, eng = traj_drift.engine_traj("LCX", n, feed=traj)
        if eng is None:
            print("  %s: NO DUMP (toks %d)" % (name, len(toks) if toks else -1))
            continue
        P = eng[:n]
        m = kl_curve(R, P)
        tail = slice(a.ctx - 4, n)                     # the 20 long-context states
        kls_t = [float(np.sum(divergence_floor.softmax(R[t]) *
                              (np.log(divergence_floor.softmax(R[t]) + 1e-30)
                               - np.log(divergence_floor.softmax(P[t]) + 1e-30))))
                 for t in range(*tail.indices(n))]
        agree_t = sum(int(R[t].argmax() == P[t].argmax()) for t in range(*tail.indices(n)))
        print("  %-8s KL mean %.4f | long-ctx (%d states) mean %.4f max %.4f | top-1 %d/%d"
              % (name, m[0], n - (a.ctx - 4), sum(kls_t) / len(kls_t), max(kls_t), agree_t, n - (a.ctx - 4)))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
