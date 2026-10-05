#!/usr/bin/env python3
"""C-gemv microbench: times libgemv directly (head + MLP sizes).

Usage: bench_gemv.py [snap] [iters=20]
Prints GB/s (weight bytes) for comparison with the asm microbench.
"""
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def main():
    snap = find_snap(sys.argv[1] if len(sys.argv) > 1 else None)
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    arm = Arm270m(snap, max_seq=256, verbose=False)
    rng = np.random.default_rng(0)
    arm.H[:] = rng.standard_normal(640).astype(np.float32)
    for M, tag in ((262144, "head"), (2048, "mlp")):
        t0 = time.perf_counter()
        for _ in range(iters):
            arm.lib.gemv_bf16(M, 640, arm.emb_ptr,
                              arm.H.ctypes.data, arm.LG.ctypes.data)
        dt = (time.perf_counter() - t0) / iters
        gbps = (M * 640 * 2 / 1e9) / dt
        print(f"C-GEMV {tag} {M}x640 x{iters}: {dt * 1000:.2f} ms "
              f"-> {gbps:.2f} GB/s", flush=True)
        print(f"METRIC c_gemv_{tag}_gbps={gbps:.4f}", flush=True)


if __name__ == "__main__":
    main()
