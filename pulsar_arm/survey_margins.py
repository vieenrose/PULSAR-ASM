#!/usr/bin/env python3
"""Logit-margin survey: top1-top2 gap per decode step across varied prompts.

Informs whether an exact-argmax-preserving head prefilter is viable: the
prefilter's error bound must sit well below the typical margin.
Prints min/median margins overall and per prompt. Measurement only.
Usage: survey_margins.py [snap]
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runtime.arm_model import Arm270m  # noqa: E402


def find_snap(explicit=None):
    if explicit and os.path.isfile(explicit):
        return explicit
    cands = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-3-270m-it-qat-q4_0-unquantized"
        "/snapshots/*/model.safetensors"))
    if not cands:
        raise SystemExit("checkpoint not found")
    return cands[0]


PROMPTS = {
    "cjk": [2, 107, 235733, 120175, 107, 304, 2505, 9694],
    "nums": [2, 107, 3627, 4770, 5913, 7056, 8199, 9342],
    "code": [2, 107, 12475, 8912, 45102, 18770, 22034, 15007],
}


def main():
    arm = Arm270m(find_snap(sys.argv[1] if len(sys.argv) > 1 else None),
                  max_seq=256, verbose=False)
    allm = []
    for name, pr in PROMPTS.items():
        ms = []
        for t in pr:
            arm.forward(t)
        tok = pr[-1]
        for _ in range(24):
            tok = arm.forward(tok)
            lg = arm.LG
            i1 = int(np.argmax(lg))
            v1 = lg[i1]
            lg2 = lg.copy()
            lg2[i1] = -1e30
            ms.append(float(v1 - lg2.max()))
        allm += ms
        a = np.array(ms)
        print(f"{name}: min={a.min():.4f} p5={np.percentile(a,5):.4f} "
              f"median={np.median(a):.4f} max={a.max():.4f}", flush=True)
    a = np.array(allm)
    print(f"ALL: min={a.min():.4f} p5={np.percentile(a,5):.4f} "
          f"median={np.median(a):.4f}", flush=True)


if __name__ == "__main__":
    main()
