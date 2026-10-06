"""Synthetic PTGV fixtures with hand-checkable expectations.

The real-tensor fixtures cannot say *which* part of a ternary GEMV is wrong:
a mis-indexed x looks like a slightly wrong sum. These cases make each part
separately visible (x is all ones, so only trit count and scale matter):

  all1    every trit +1        -> scale * 128 per group
  idx0    trit at index 0 only -> scale
  idx127  trit at index 127    -> scale   (exercises the last, 3-trit byte)
  alt     every 5th trit       -> exercises every byte boundary

They caught a real bug: half_to_float ORed the mantissa in without shifting it
to bit 13, so 1.5 became 1.00003 and every scaled sum was off by ~1/3.

Usage: mk_syn_ternary.py <outdir>
"""
import os
import struct
import sys

import numpy as np

GROUP, GB = 128, 28
MAGIC = 0x50544756          # "PTGV"


def pack(path, trit_fn, scale, rows=2, ng=2, xval=1.0):
    x = np.full(ng * GROUP, xval, dtype=np.float32)
    w, ys = bytearray(), []
    for _ in range(rows):
        s = 0.0
        for _g in range(ng):
            w += struct.pack("<e", np.float16(scale))
            for k in range(26):
                v = 0
                for i in range(5):
                    idx = k * 5 + i
                    if idx >= GROUP:
                        break
                    v += (trit_fn(idx) + 1) * (3 ** i)
                w.append(v)
            s += scale * sum(trit_fn(idx) * xval for idx in range(GROUP))
        ys.append(s)
    hdr = struct.pack("<QIIIIII", MAGIC, 1, 1, rows, ng * GROUP, ng, GB)
    with open(path, "wb") as fh:
        fh.write(hdr + x.tobytes() + bytes(w) +
                 np.array(ys, dtype=np.float32).tobytes())
    return ys[0]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    cases = (("all1", lambda i: 1, 1.5),
             ("idx0", lambda i: 1 if i == 0 else 0, 1.5),
             ("idx127", lambda i: 1 if i == 127 else 0, 1.5),
             ("alt", lambda i: 1 if i % 5 == 4 else 0, 0.25))
    for name, fn, scale in cases:
        p = os.path.join(out, f"syn_{name}.tv")
        y0 = pack(p, fn, scale)
        print(f"{p}: expected y[0]={y0}")


if __name__ == "__main__":
    main()
