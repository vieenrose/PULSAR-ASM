#!/usr/bin/env python3
"""A/B: old (dict-lookup) vs new (precomputed tables) driver, same process.

Alternates timed decode blocks to cancel thermal drift, and asserts
bit-identical logits between the two implementations.
Usage: ab_driver.py <snap> [tokens_per_block=16] [rounds=3]
"""
import importlib.util
import sys
import time

import numpy as np


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    snap = sys.argv[1]
    blk = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    nrounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    old = load("arm_old", "/tmp/arm_old.py")
    new = load("arm_new", "/tmp/arm_new.py")
    A = old.Arm270m(snap, max_seq=256, verbose=False)
    B = new.Arm270m(snap, max_seq=256, verbose=False)
    prompt = [2, 107, 1567, 236765, 107, 304, 2505, 9694]
    # separate streams would diverge; use ONE shared token stream per model
    toksA, toksB = list(prompt), list(prompt)
    for t in prompt:
        A.forward(t)
        B.forward(t)
    print("parity old-vs-new logits:",
          "IDENTICAL" if np.array_equal(A.LG, B.LG) else
          f"DIFFER max|d|={float(np.abs(A.LG - B.LG).max()):.3e}", flush=True)
    ta, tb = toksA[-1], toksB[-1]
    for r in range(nrounds):
        for tag, M in (("old", A), ("new", B)):
            t0 = time.perf_counter()
            tk = ta if M is A else tb
            for _ in range(blk):
                tk = M.forward(tk)
            if M is A:
                ta = tk
            else:
                tb = tk
            print(f"{tag} round{r}: {(time.perf_counter() - t0) / blk * 1000:.2f} ms/tok",
                  flush=True)


if __name__ == "__main__":
    main()
