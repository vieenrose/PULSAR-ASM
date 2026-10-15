#!/usr/bin/env python3
"""Bake the Gemma 4 E2B rope tables once, so the C reference and the Nano engine
read the same bits instead of each calling a libm cos/sin.

Layout (little-endian float32), max_seq rows each:
  slide cos [max_seq][128], slide sin [max_seq][128],
  full  cos [max_seq][256], full  sin [max_seq][256]
Values: HF default rope (theta 1e4, head 256) and proportional rope (theta 1e6,
head 512, partial 0.25, zero-padded inv_freq), computed in float64, rounded to fp32,
exactly as g4_np.rope_tables.

Usage: python3 -I g4_rope.py <out.bin> [max_seq=2048]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g4_np import rope_tables  # noqa: E402


def main():
    out = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    t = rope_tables(n)
    with open(out, "wb") as f:
        for typ in ("slide", "full"):
            for a in t[typ]:
                f.write(np.ascontiguousarray(a, dtype="<f4").tobytes())
    print(f"wrote {out}: {os.path.getsize(out)} bytes, max_seq {n}")


if __name__ == "__main__":
    main()
