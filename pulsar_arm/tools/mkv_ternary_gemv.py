"""Test vectors for the asm ternary GEMV (B3_128 / Q2_0 blocks).

Emits one self-describing binary file per selected tensor slice so the kernel
can be unit-tested without the 4B blob in the loop:

    u64 magic 'PTGV', u32 version
    u32 fmt (0 = Q2_0 34 B/group, 1 = B3_128 28 B/group)
    u32 rows, u32 cols, u32 groups_per_row, u32 group_bytes
    f32 x[cols]                     input vector (deterministic)
    u8  w[rows][groups_per_row][group_bytes]      weight blocks, verbatim
    f32 y[rows]                     expected = W @ x, fp32

The comparison is against fp32 accumulation, so the kernel is expected to match
within a small relative tolerance (accumulation order differs), not bit-exactly.

Usage:
  mkv_ternary_gemv.py <blob.bin> <tensor> <out.tv> [row_start] [row_count] [seed]
"""
import struct
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
from q2_0_ref import Blob, GROUP, GROUP_BYTES, B3_GROUP_BYTES   # noqa: E402

MAGIC = 0x50544756      # "PTGV"


def main():
    blob, name, out = sys.argv[1], sys.argv[2], sys.argv[3]
    r0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    nrows = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 1234
    bl = Blob(blob)
    info = bl.info(name)
    rows, cols = info["shape"]
    r1 = rows if nrows == 0 else min(rows, r0 + nrows)
    fmt = 0 if info["dtype"] == "Q2_0" else 1
    gb = GROUP_BYTES if fmt == 0 else B3_GROUP_BYTES
    ng = cols // GROUP
    assert cols % GROUP == 0, (name, cols)

    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(cols).astype(np.float32) * 0.1)
    W = bl.deq(name, r0, r1)
    y = (W @ x).astype(np.float32)

    # weight blocks verbatim for the slice
    raw = bl.raw(name)
    per_row = ng * gb
    wbytes = raw[r0 * per_row: r1 * per_row]

    with open(out, "wb") as fh:
        fh.write(struct.pack("<QIIIIII", MAGIC, 1, fmt, r1 - r0, cols, ng, gb))
        fh.write(x.tobytes())
        fh.write(wbytes)
        fh.write(y.tobytes())
    print(f"{out}: {name}[{r0}:{r1}] rows={r1-r0} cols={cols} ng={ng} gb={gb} "
          f"fmt={'B3_128' if fmt else 'Q2_0'} "
          f"| |x|max={np.abs(x).max():.4f} |y|max={y.max():.4f} "
          f"size={(len(x)+len(wbytes)+len(y))*1 + (r1-r0)*0 + 40}B", flush=True)


if __name__ == "__main__":
    main()
