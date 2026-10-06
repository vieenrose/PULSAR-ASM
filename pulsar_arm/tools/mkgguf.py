#!/usr/bin/env python3
"""Build a GGUF tensor index for the pure-asm 1B loader (BUILD-TIME tool).

Reads the 1B GGUF, resolves the needed tensors by name, writes gidx.bin:
  u64 magic (0x4749445831323620), u64 data_start, u64 n,
  n x (u64 file_off, u64 dim0, u64 dim1, u32 type, u32 pad).
Order: token_embd, then per layer [attn_norm, q, q_norm, k, k_norm, v,
output, post_attn_norm, ffn_norm, gate, up, down], then output_norm.
Dims are GGUF order (as stored). Fails loudly on any missing tensor.
Usage: mkgguf.py <model.gguf> <gidx.bin> [layers=26]
"""
import json
import struct
import sys

MAGIC = 0x4749445831323620
LAYERS = int(sys.argv[3]) if len(sys.argv) > 3 else 26
PER_LAYER = [
    "attn_norm.weight",
    "attn_q.weight",
    "attn_q_norm.weight",
    "attn_k.weight",
    "attn_k_norm.weight",
    "attn_v.weight",
    "attn_output.weight",
    "post_attention_norm.weight",
    "ffn_norm.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
]


def main():
    src, dst = sys.argv[1], sys.argv[2]
    b = open(src, "rb")
    magic, ver, nt, na = struct.unpack("<IIQQ", b.read(24))
    assert magic == 0x46554747, hex(magic)

    def rd_str():
        n = struct.unpack("<Q", b.read(8))[0]
        return b.read(n).decode("utf-8", "replace")

    def skip_val(t):
        if t in (0, 1, 7):
            b.seek(1, 1)
        elif t in (2, 3):
            b.seek(2, 1)
        elif t in (4, 5, 6):
            b.seek(4, 1)
        elif t in (10, 11, 12):
            b.seek(8, 1)
        elif t == 8:
            rd_str()
        elif t == 9:
            at = struct.unpack("<I", b.read(4))[0]
            n = struct.unpack("<Q", b.read(8))[0]
            if at == 8:
                for _ in range(n):
                    rd_str()
            elif at in (4, 5, 6):
                b.seek(4 * n, 1)
            elif at == 7:
                b.seek(n, 1)
            else:
                raise ValueError(at)
        else:
            raise ValueError(t)

    for _ in range(na):
        rd_str()
        skip_val(struct.unpack("<I", b.read(4))[0])

    found = {}
    for _ in range(nt):
        nl = struct.unpack("<Q", b.read(8))[0]
        name = b.read(nl).decode()
        nd = struct.unpack("<I", b.read(4))[0]
        dims = struct.unpack("<" + "Q" * nd, b.read(8 * nd))
        ty = struct.unpack("<I", b.read(4))[0]
        off = struct.unpack("<Q", b.read(8))[0]
        found[name] = (off, dims, ty)

    names = ["token_embd.weight"]
    for i in range(LAYERS):
        names += [f"blk.{i}.{s}" for s in PER_LAYER]
    names.append("output_norm.weight")
    missing = [n for n in names if n not in found]
    if missing:
        raise SystemExit(f"missing {len(missing)}: {missing[:5]}")
    print(f"all {len(names)} tensors resolved", flush=True)

    data_start = b.tell()
    # tensor data follows, 32B-aligned; file offsets in table are relative to
    # data start? GGUF offsets are from start of tensor data area (after header
    # alignment). Record base + per-tensor offset; asm adds base.
    # GGUF spec: offset is from beginning of file? No: from tensor data start
    # (after alignment to GGUF_DEFAULT_ALIGNMENT=32). Compute data base:
    # current pos, aligned up to 32.
    base = (data_start + 31) & ~31
    # verify offset basis: first Q4_0 block's scale must be sane fp16
    # (nonzero, finite exponent). Catches wrong base/offset interpretation.
    q0 = next(n for n in names if found[n][2] in (2, 8))
    off0 = found[q0][0]
    b.seek(base + off0)
    sc = struct.unpack("<H", b.read(2))[0]
    exp = (sc >> 10) & 0x1F
    assert sc & 0x7FFF != 0 and exp not in (0, 31), f"bad scale {sc:#06x}"
    print(f"offset basis ok (first Q4_0 scale bits {sc:#06x})", flush=True)
    with open(dst, "wb") as f:
        f.write(struct.pack("<QQQ", MAGIC, base, len(names)))
        for n in names:
            off, dims, ty = found[n]
            d0 = dims[0] if len(dims) > 0 else 1
            d1 = dims[1] if len(dims) > 1 else 1
            f.write(struct.pack("<QQQII", off, d0, d1, ty, 0))
    print(f"wrote {dst} (base {base})", flush=True)


if __name__ == "__main__":
    main()
