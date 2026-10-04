# ==============================================================================
# Project PULSAR-ASM | tests/test_bf16_gemb.py
# ------------------------------------------------------------------------------
# Numerical + SMP validation for the bf16 multi-RHS GEMM kernel.
#
#   python tests/test_bf16_gemb.py            # correctness
#   python tests/test_bf16_gemb.py --bench    # + DRAM bandwidth / tok/s estimate
#
# The reference is the same math done in NumPy on the *identical* bf16-quantised
# weights, so any failure here is a kernel bug, not a precision story.
# ==============================================================================

import os
import sys
import time
import ctypes
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime import pulsar_abi as abi                                   # noqa: E402


def to_bf16_exact(a):
    """Round-to-bf16 by truncating the fp32 mantissa (matches the converter)."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    u &= np.uint32(0xFFFF0000)
    return (u >> np.uint32(16)).astype(np.uint16), (u.view(np.float32))


def run_case(mod, M, K, B, smp=0, seed=0):
    rng = np.random.default_rng(seed)
    wb, wf = to_bf16_exact(rng.standard_normal((M, K), dtype=np.float32) * 0.022)
    xb = rng.standard_normal((B, K), dtype=np.float32)

    out_asm = np.zeros(B * M, dtype=np.float32)
    W = np.ascontiguousarray(wb)
    X = np.ascontiguousarray(xb)

    gemv = abi.make_reg_entry(mod.exports["bf16_gemb_avx2"], 6)
    gemv(out_asm.ctypes.data, W.ctypes.data, X.ctypes.data, K, M, B)

    ref = (X @ wf.T).astype(np.float32)
    got = out_asm.reshape(B, M)
    err = float(np.max(np.abs(got - ref)))
    scale = max(float(np.max(np.abs(ref))), 1e-30)
    rel = err / scale
    ok = rel < 2e-6
    print(f"  {'OK ' if ok else 'FAIL'} M={M:7d} K={K:5d} B={B}  "
          f"max|err|={err:.3e} rel={rel:.2e}")
    return ok


def run_smp_case(mod, M, K, B, seed=1):
    """Full MESI spin-worker pool: 3 pthreads + master, row-partitioned."""
    rng = np.random.default_rng(seed)
    wb, wf = to_bf16_exact(rng.standard_normal((M, K), dtype=np.float32) * 0.022)
    X = np.ascontiguousarray(rng.standard_normal((B, K), dtype=np.float32))

    smp = abi.exec_alloc(4096)
    ctypes.memset(smp, 0, 4096)
    ctypes.c_uint64.from_address(smp + 16).value = 4

    worker = mod.exports["smp_worker_proc4"]
    handles = []
    for i in range(1, 4):
        handles.append(abi.spawn_worker(worker, smp + 64 + i * 128))

    out_asm = np.zeros(B * M, dtype=np.float32)
    W = np.ascontiguousarray(wb)
    gsmp = abi.make_reg_entry(mod.exports["smp_bf16_gemb_avx2"], 7)
    gsmp(out_asm.ctypes.data, W.ctypes.data, X.ctypes.data, K, M, B, smp)

    ref = (X @ wf.T).astype(np.float32)
    got = out_asm.reshape(B, M)
    rel = float(np.max(np.abs(got - ref))) / max(float(np.max(np.abs(ref))), 1e-30)
    dc = ctypes.c_uint32.from_address(smp).value
    ok = (rel < 2e-6) and (dc == 3)      # dc==3 proves the 3 spin workers did the work

    # park and join the spinners
    ctypes.memset(smp + 8, 0, 4)
    for i in range(1, 4):
        ctypes.c_uint32.from_address(smp + 64 + i * 128 + 8).value = 1
    for h in handles:
        abi.join_worker(h)
    print(f"  {'OK ' if ok else 'FAIL'} SMP M={M:7d} K={K:5d} B={B}  rel={rel:.2e} "
          f"(done_counter={dc}/3 -> {'4 cores' if dc == 3 else 'DEGRADED TO 1 CORE'})")
    abi.exec_free(smp, 4096)
    return ok


def run_smp_repeat(mod, n=6):
    """Regression: the SAME spin-worker pool must accept repeated jobs.

    The job handshake is (job_id != last_job_id), so a dispatch that ever writes a
    non-incrementing job_id silently deadlocks - the master waits for a
    done_counter that can never move. One-shot pools cannot catch that.
    """
    smp = abi.exec_alloc(4096)
    ctypes.memset(smp, 0, 4096)
    worker = mod.exports["smp_worker_proc4"]
    handles = [abi.spawn_worker(worker, smp + 64 + i * 128) for i in range(1, 4)]
    gsmp = abi.make_reg_entry(mod.exports["smp_bf16_gemb_avx2"], 7)
    g1 = abi.make_reg_entry(mod.exports["bf16_gemb_avx2"], 6)
    ok = True
    try:
        for it in range(n):
            M, K, B = (4096, 1536, 1) if it % 2 == 0 else (6144, 1536, 3)
            rng = np.random.default_rng(100 + it)
            wb, wf = to_bf16_exact(rng.standard_normal((M, K), dtype=np.float32) * 0.02)
            X = np.ascontiguousarray(rng.standard_normal((B, K), dtype=np.float32))
            out = np.zeros(B * M, dtype=np.float32)
            W = np.ascontiguousarray(wb)
            gsmp(out.ctypes.data, W.ctypes.data, X.ctypes.data, K, M, B, smp)
            ref = (X @ wf.T).astype(np.float32)
            rel = float(np.max(np.abs(out.reshape(B, M) - ref))) / max(float(np.max(np.abs(ref))), 1e-30)
            dc = ctypes.c_uint32.from_address(smp).value
            good = rel < 2e-6 and dc == 3
            ok &= good
            print(f"  {'OK ' if good else 'FAIL'} repeat #{it} M={M} B={B} rel={rel:.2e} "
                  f"done={dc}/3 job_seq={ctypes.c_uint32.from_address(smp + 4).value}")
    finally:
        for i in range(1, 4):
            ctypes.c_uint32.from_address(smp + 64 + i * 128 + 8).value = 1
        for h in handles:
            abi.join_worker(h)
        abi.exec_free(smp, 4096)
    return ok


def bench(mod):
    print("\n-- DRAM bandwidth probe (4-Core SMP, bf16) --")
    K = 1536
    for M in (65536, 262144):
        W = np.zeros(M * K, dtype=np.uint16)
        X = np.zeros(K, dtype=np.float32)
        out = np.zeros(M, dtype=np.float32)
        smp = abi.exec_alloc(4096)
        ctypes.memset(smp, 0, 4096)
        worker = mod.exports["smp_worker_proc4"]
        handles = [abi.spawn_worker(worker, smp + 64 + i * 128) for i in range(1, 4)]
        gsmp = abi.make_reg_entry(mod.exports["smp_bf16_gemb_avx2"], 7)
        for _ in range(2):
            gsmp(out.ctypes.data, W.ctypes.data, X.ctypes.data, K, M, 1, smp)
        t0 = time.perf_counter()
        n = 3
        for _ in range(n):
            gsmp(out.ctypes.data, W.ctypes.data, X.ctypes.data, K, M, 1, smp)
        dt = (time.perf_counter() - t0) / n
        gbs = (M * K * 2) / dt / 1e9
        print(f"  M={M:7d} K={K} : {dt*1e3:7.2f} ms   {gbs:6.2f} GB/s streamed")
        for i in range(1, 4):
            ctypes.c_uint32.from_address(smp + 64 + i * 128 + 8).value = 1
        for h in handles:
            abi.join_worker(h)
        abi.exec_free(smp, 4096)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    print("=" * 74)
    print("PULSAR-ASM bf16 multi-RHS GEMM validation (Linux ABI bridge)")
    print("=" * 74)
    print(f"  cpu: {abi.cpu_report()}")

    asm = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "raw_materials", "mat_bf16_gemb_avx2_flat.asm")
    mod = abi.load_module(asm)
    print(f"  {mod}")
    print(f"  exports: {sorted(mod.exports)}")
    print()

    allok = True
    for (M, K) in [(1536, 1536), (256, 1536), (6144, 1536), (12288, 1536), (8192, 1536)]:
        for B in (1, 2, 5):
            allok &= run_case(mod, M, K, B, seed=M + K + B)
    print()
    allok &= run_smp_case(mod, 6144, 1536, 1)
    allok &= run_smp_case(mod, 262144, 1536, 1, seed=7)
    allok &= run_smp_case(mod, 4096, 1536, 3, seed=8)
    print()
    allok &= run_smp_repeat(mod)

    if args.bench:
        bench(mod)

    print("\n" + ("ALL KERNEL TESTS PASS" if allok else "*** FAILURES PRESENT ***"))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
