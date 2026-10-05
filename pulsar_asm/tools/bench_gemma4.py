"""Numbers for the README, measured rather than claimed.

Three things matter for a CPU engine that reads 9.258 GB of weights per token:
how fast one token comes out, how much of that is the memory system rather than
the cores, and what speculative decoding could save. The last one is answered by
tools/bench_mtp.py, which measures acceptance against Google's own assistant
checkpoint; this script covers the first two.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.gemma4_model import Gemma4  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/mnt/edge/pulsar/gemma4_e2b.bin")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--prompt", type=int, default=64, help="prefill length")
    a = ap.parse_args()

    g = Gemma4(a.weights, max_seq=1024, n_threads=a.threads, verbose=False)
    wbytes = g.blob.size

    # ---- decode ------------------------------------------------------------
    g.reset()
    tok = 2
    g.forward(tok)                        # warm the caches, exclude first-touch
    t0 = time.perf_counter()
    for _ in range(a.steps):
        tok = g.forward(tok if tok != 1 else 2)
    dt = (time.perf_counter() - t0) / a.steps
    tps = 1.0 / dt
    print(f"decode        {tps:6.2f} tok/s   {dt*1e3:7.1f} ms/step   "
          f"({wbytes/dt/1e9:.1f} GB/s of weights streamed)")

    # ---- prefill: one token per step, so a prompt costs prompt*step --------
    ids = list(range(2, 2 + a.prompt))
    g.reset()
    t0 = time.perf_counter()
    for t in ids:
        g.forward(t)
    pf = time.perf_counter() - t0
    print(f"prefill       {pf*1e3:7.1f} ms for {a.prompt} tokens   "
          f"({a.prompt/(pf+dt):.2f} tok/s effective)")

    # ---- what speculative decoding is worth --------------------------------
    # Every candidate position needs the same weights, so a batched verify pass
    # costs one read instead of B. That is only true if the pass exists: the
    # batched path is still wrong, so the real number is 1.0x.
    print(f"weights       {wbytes/1e9:.3f} GB, {g.n_layers} layers, "
          f"hidden {g.hidden}, vocab {g.vocab}")
    print(f"engine        {g.mod.size} bytes of assembly, {a.threads} cores")

    # the check that this is a real measurement and not a stall
    ref = float(np.abs(g.buf["X"][:g.hidden]).max())
    assert ref == ref and 0.1 < ref < 1e3, f"stream looks wrong: {ref}"
    g.close()


if __name__ == "__main__":
    main()
