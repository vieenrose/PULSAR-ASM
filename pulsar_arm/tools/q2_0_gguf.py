"""GGUF -> PULSAR blob converter for prism ternary (Q2_0) checkpoints.

Reads a prism-ml GGUF (Qwen3 + Q2_0 ternary) and writes a safetensors-shaped
blob that the asm engine already knows how to mmap and index, with two custom
dtypes:

  F32    plain fp32 tensor (the 113 norm tensors)
  Q2_0   ternary tensor, 128-weight groups, 34 bytes per group:
         fp16 scale + 32 bytes of 2-bit codes, w = (code - 1) * scale

Layout note: GGUF stores [dims0, dims1] with the quantisation groups running
along dims0, i.e. each of the dims1 rows is dims0 contiguous values. Our kernel
wants rows = dims1, cols = dims0, groups every 128 along cols - so the bytes are
copied verbatim and only the shape is swapped.

Usage:  q2_0_gguf.py <model.gguf> <out.bin> [vocab.bin] [bpe.bin]
        (vocab/bpe are delegated to the existing gguf_vocab/gguf_tok tools)
"""
import json
import struct
import sys

GGUF_MAGIC = 0x46554747
T_F32, T_F16, T_Q2_0 = 0, 1, 42
GROUP = 128                 # weights per ternary group
GROUP_BYTES = 2 + GROUP // 4    # fp16 scale + 2-bit codes = 34

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
            7: "<B", 10: "<Q", 11: "<q", 12: "<d"}


class GGUF:
    def __init__(self, path, keep_meta=True):
        self.path = path
        self.f = open(path, "rb")
        magic, self.ver, self.n_tensors, self.n_kv = \
            struct.unpack("<IIQQ", self.f.read(24))
        assert magic == GGUF_MAGIC, hex(magic)
        self.meta = {}
        for _ in range(self.n_kv):
            k = self._str()
            t = struct.unpack("<I", self.f.read(4))[0]
            v = self._val(t)
            if keep_meta:
                self.meta[k] = v
        self.tensors = {}
        self.order = []
        for _ in range(self.n_tensors):
            name = self._str()
            nd = struct.unpack("<I", self.f.read(4))[0]
            dims = [struct.unpack("<Q", self.f.read(8))[0] for _ in range(nd)]
            tt = struct.unpack("<I", self.f.read(4))[0]
            off = struct.unpack("<Q", self.f.read(8))[0]
            self.tensors[name] = (dims, tt, off)
            self.order.append(name)
        align = self.meta.get("general.alignment", 32)
        self.data_start = (self.f.tell() + align - 1) // align * align

    def _str(self):
        n = struct.unpack("<Q", self.f.read(8))[0]
        return self.f.read(n).decode("utf-8", "replace")

    def _val(self, t):
        if t == 8:
            return self._str()
        if t == 9:                       # array: skip payload, keep the length
            et = struct.unpack("<I", self.f.read(4))[0]
            n = struct.unpack("<Q", self.f.read(8))[0]
            for _ in range(n):
                self._val(et)
            return f"<array {n}>"
        return struct.unpack(_SCALARS[t], self.f.read(struct.calcsize(_SCALARS[t])))[0]

    def raw(self, name, offset=0, nbytes=None):
        dims, tt, off = self.tensors[name]
        self.f.seek(self.data_start + off + offset)
        return self.f.read(nbytes) if nbytes else self.f.read(self.nbytes(name))

    def nbytes(self, name):
        dims, tt, _ = self.tensors[name]
        n = 1
        for d in dims:
            n *= d
        if tt == T_F32:
            return n * 4
        if tt == T_F16:
            return n * 2
        if tt == T_Q2_0:
            assert dims[0] % GROUP == 0, (name, dims)
            return n // GROUP * GROUP_BYTES
        raise SystemExit(f"unsupported gguf type {tt} for {name}")


def convert(src, out):
    g = GGUF(src)
    hdr, payload, offs = {}, [], 0
    for name in g.order:
        dims, tt, _ = g.tensors[name]
        nb = g.nbytes(name)
        if tt == T_Q2_0:
            rows, cols, dt = dims[1], dims[0], "Q2_0"
        elif tt == T_F32:
            rows, cols, dt = (dims[1], dims[0]) if len(dims) == 2 else (dims[0], 1)
        else:
            raise SystemExit(f"{name}: unexpected dtype {tt}")
        hdr[name.replace("blk.", "model.layers.").replace(".attn_", ".self_attn.")
            if name.startswith("blk.") else name] = {
            "dtype": dt, "shape": [rows, cols], "data_offsets": [offs, offs + nb]}
        payload.append((name, offs, nb))
        offs += nb
    blob = bytearray()
    for name, o, nb in payload:
        blob += g.raw(name)
    with open(out, "wb") as fh:
        h = json.dumps(hdr).encode()
        fh.write(struct.pack("<Q", len(h)))
        fh.write(h)
        fh.write(blob)
    mb = (8 + len(h) + len(blob)) / 1e6
    print(f"{src} -> {out}: {len(hdr)} tensors, {mb:.1f} MB "
          f"(header {len(h)} B, data {len(blob)/1e6:.1f} MB)", flush=True)
    return g


def main():
    src, out = sys.argv[1], sys.argv[2]
    g = convert(src, out)
    m = g.meta
    print("meta:", {k: v for k, v in m.items() if not isinstance(v, str)
                    or len(v) < 40})


if __name__ == "__main__":
    main()
