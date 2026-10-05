#!/usr/bin/env python3
"""CS-bound tightness probe for an exact-argmax head shortcut.

For real decode hidden states H: bound_i = ||H|| * ||E_i|| (>= true logit,
Cauchy-Schwarz). Count rows with bound_i >= top1_logit - eps: those are the
rows an exact bounded search must evaluate. Small fraction => implement it.
Usage: survey_bounds.py [snap]
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
    snap = find_snap(sys.argv[1] if len(sys.argv) > 1 else None)
    arm = Arm270m(snap, max_seq=256, verbose=False)
    # E row norms from the mmapped bf16 table (chunked, float64 accum)
    eoff = arm.off["model.embed_tokens.weight"][0]
    enorm = np.empty(262144, dtype=np.float64)
    CH = 16384
    for s in range(0, 262144, CH):
        E = np.frombuffer(arm._mm, dtype=np.uint16,
                          count=CH * 640,
                          offset=eoff + s * 640 * 2).reshape(CH, 640)
        Ef = (((E.astype(np.uint32)) << 16).view(np.float32)).astype(np.float64)
        enorm[s:s + CH] = np.sqrt((Ef * Ef).sum(axis=1))
    print(f"enorm: max={enorm.max():.3f} median={np.median(enorm):.3f}", flush=True)
    fracs = []
    arm.tape = []
    for name, pr in PROMPTS.items():
        for t in pr:
            arm.forward(t)
        tok = pr[-1]
        for _ in range(12):
            tok = arm.forward(tok)
            lg = arm.LG.astype(np.float64)
            top1 = lg.max()
            x = arm.tape[-1]  # X after L17
            h = x / np.sqrt((x.astype(np.float64) ** 2).mean() + 1e-6)
            hn = float(np.sqrt((h ** 2).sum()))
            bnd = hn * enorm
            surv = int((bnd >= top1 - 1e-3).sum())
            fracs.append(surv / 262144)
        arm.tape = []
    a = np.array(fracs)
    print(f"survivor frac: min={a.min():.4f} median={np.median(a):.4f} "
          f"max={a.max():.4f}", flush=True)
    print(f"METRIC survivor_median={np.median(a):.6f}", flush=True)


if __name__ == "__main__":
    main()
