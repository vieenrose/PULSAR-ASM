#!/usr/bin/env python3
"""NEON bf16 GEMV vs numpy: correctness + first timing on the Pi."""
import ctypes
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "kernels", "neon_gemv.c")
SO = os.path.join(HERE, "libneon.so")


def build():
    r = subprocess.run(["gcc", "-O2", "-fPIC", "-shared", SRC, "-o", SO],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        sys.exit(1)


def main():
    build()
    lib = ctypes.CDLL(SO)
    lib.gemv_bf16.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                              ctypes.c_void_p, ctypes.c_void_p]
    lib.gemv_bf16.restype = None
    rng = np.random.default_rng(0)
    worst = 0.0
    for M, K in ((1024, 640), (2048, 640), (640, 1024), (262144, 640)):
        Wf = rng.standard_normal((M, K)).astype(np.float32)
        Wb = ((Wf.view(np.uint32)) >> 16).astype(np.uint16)
        x = rng.standard_normal(K).astype(np.float32)
        # widen the SAME bf16 bits both sides read: isolates kernel arithmetic
        # from input rounding (which production shares, reading one blob).
        Wref = ((Wb.astype(np.uint32)) << 16).view(np.float32)
        want = Wref @ x
        got = np.zeros(M, dtype=np.float32)
        t0 = time.perf_counter()
        lib.gemv_bf16(M, K, Wb.ctypes.data, x.ctypes.data, got.ctypes.data)
        dt = time.perf_counter() - t0
        rel = float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-9))
        worst = max(worst, rel)
        gflops = 2 * M * K / dt / 1e9
        print(f"  M={M:6d} K={K:4d} rel={rel:.2e} {dt*1000:8.1f}ms {gflops:5.1f} GFLOPS")
    print("PASS" if worst < 1e-3 else "FAIL", f"worst rel={worst:.2e}")
    return 0 if worst < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
