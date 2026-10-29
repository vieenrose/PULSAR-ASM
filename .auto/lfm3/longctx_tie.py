#!/usr/bin/env python
"""Tie landscape at long context (positions ctx-4..ctx+gen-1).

#195/#196 quantified argmax ties at short prompts: ~4.5% of teacher-forced
positions flip between the bf16 and fp16 HF references, and both engine arms
make identical choices on both references. This script extends that coverage
to audio-length contexts: same reference trajectory (fp16 greedy), same
teacher forcing, but scored at the 20 LONG states. Expectation (from #171/#201):
top-1 ~19-20/20 vs each reference, arms identical, ~1 fp16/bf16 disagreement.

  ~/venvs/lfm3/bin/python .auto/lfm3/longctx_tie.py [--ctx 300] [--gen 16]
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
from divergence_floor import load, greedy
import traj_drift


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
    print("== long-context ties: %d ids, %d gen, refs fp16 + bf16 ==" % (a.ctx, a.gen))

    ref16 = load("float16")
    traj = greedy(ref16, ids, a.gen)
    n = len(traj)
    import torch
    R16 = []
    for p in range(n):
        with torch.no_grad():
            R16.append(ref16(input_ids=torch.tensor([traj[:p + 1]], device="cuda")).logits[0, -1].float().cpu().numpy())
    del ref16
    torch.cuda.empty_cache()
    refb = load("bfloat16")
    RB = []
    for p in range(n):
        with torch.no_grad():
            RB.append(refb(input_ids=torch.tensor([traj[:p + 1]], device="cuda")).logits[0, -1].float().cpu().numpy())
    del refb
    torch.cuda.empty_cache()
    R16 = np.stack(R16); RB = np.stack(RB)
    tail = range(a.ctx - 4, n)
    dis = sum(int(R16[t].argmax() != RB[t].argmax()) for t in tail)
    print("  trajectory %d tokens; fp16/bf16 ref disagreements in long states: %d/%d"
          % (n, dis, len(list(tail))))

    for name, env in [("shipped", ""), ("H2=1", "env LFM_H2=1 ")]:
        traj_drift.ENVQ = env
        toks, eng = traj_drift.engine_traj("LCT", n, feed=traj)
        if eng is None:
            print("  %s: NO DUMP (toks %d)" % (name, len(toks) if toks else -1))
            continue
        P = eng[:n]
        t16 = sum(int(P[t].argmax() == R16[t].argmax()) for t in tail)
        tb = sum(int(P[t].argmax() == RB[t].argmax()) for t in tail)
        print("  %-8s long-state top-1: %d/%d vs fp16, %d/%d vs bf16"
              % (name, t16, len(list(tail)), tb, len(list(tail))))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
