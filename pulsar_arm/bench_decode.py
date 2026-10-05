#!/usr/bin/env python3
"""Decode-speed benchmark for the ARM gemma-3-270m driver.

Prefills PROMPT tokens (untimed), then greedily decodes N tokens while
timing only the decode loop. Prints a METRIC line for the autoresearch
harness: ms per generated token (lower is better).

Usage: bench_decode.py [snap_dir] [n_decode=32]
"""
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runtime.arm_model import Arm270m  # noqa: E402


def find_snap(explicit=None):
    if explicit and os.path.isfile(explicit):
        return explicit
    cands = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-3-270m-it-qat-q4_0-unquantized"
        "/snapshots/*/model.safetensors"))
    if not cands:
        raise SystemExit("checkpoint model.safetensors not found")
    return cands[0]


def main():
    snap = find_snap(sys.argv[1] if len(sys.argv) > 1 else None)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    arm = Arm270m(snap, max_seq=256, verbose=True)
    prompt = [2, 107, 1567, 236765, 107, 304, 2505, 9694]  # fixed 8-token prefill
    for t in prompt:
        arm.forward(t)
    toks = []
    t0 = time.perf_counter()
    tok = prompt[-1]
    for _ in range(n):
        tok = arm.forward(tok)
        toks.append(tok)
    dt = time.perf_counter() - t0
    ms = dt / n * 1000.0
    print(f"decoded {n} tokens in {dt:.2f}s -> {1000.0 / ms:.2f} tok/s", flush=True)
    print(f"METRIC ms_per_token={ms:.4f}", flush=True)
    print(f"METRIC tok_per_sec={1000.0 / ms:.4f}", flush=True)
    print("tail:", toks[-8:])


if __name__ == "__main__":
    main()
