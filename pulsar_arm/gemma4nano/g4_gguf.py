#!/usr/bin/env python3
"""Minimal GGUF v3 reader for the Gemma 4 E2B Nano port.

Reads the header of Google's `gemma-4-E2B_q4_0-it.gguf`, returns the metadata and
the tensor table (name, ggml type, dims, absolute byte offset, byte size), and
checks that the tensors tile the data section exactly. Tensor bytes are accessed
through a read-only mmap, so nothing is loaded until asked for.

Usage: python3 -I g4_gguf.py <file.gguf>     # prints the summary and the checks
"""
import mmap
import struct
import sys

# ggml types used by this file: block size (elements) and block bytes
GGML = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    8: ("Q8_0", 32, 34),
    14: ("Q6_K", 256, 210),
    30: ("BF16", 1, 2),
}
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
           7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class GGUF:
    def __init__(self, path):
        self.path = path
        self._fh = open(path, "rb")
        self.mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        self._o = 0
        assert self.mm[:4] == b"GGUF", "not a GGUF file"
        self._o = 4
        self.version = self._r("<I")
        assert self.version == 3, f"GGUF v{self.version}, want 3"
        n_tensors = self._r("<Q")
        n_kv = self._r("<Q")
        self.meta = {}
        for _ in range(n_kv):
            k = self._s()
            self.meta[k] = self._val(self._r("<I"))
        self.alignment = int(self.meta.get("general.alignment", 32))
        raw = []
        for _ in range(n_tensors):
            name = self._s()
            nd = self._r("<I")
            dims = [self._r("<Q") for _ in range(nd)]
            t = self._r("<I")
            off = self._r("<Q")
            raw.append((name, t, dims, off))
        self.data_start = -(-self._o // self.alignment) * self.alignment
        self.tensors = {}
        for name, t, dims, off in raw:
            assert t in GGML, f"{name}: unhandled ggml type {t}"
            tname, blk, bb = GGML[t]
            n = 1
            for d in dims:
                n *= d
            assert n % blk == 0, f"{name}: {n} elements not a multiple of {blk}"
            self.tensors[name] = dict(type=tname, dims=dims, n=n,
                                      offset=self.data_start + off,
                                      nbytes=n // blk * bb)

    # -- header primitives --------------------------------------------------
    def _r(self, fmt):
        v = struct.unpack_from(fmt, self.mm, self._o)[0]
        self._o += struct.calcsize(fmt)
        return v

    def _s(self):
        n = self._r("<Q")
        v = bytes(self.mm[self._o:self._o + n]).decode("utf-8")
        self._o += n
        return v

    def _val(self, t):
        if t == 8:
            return self._s()
        if t == 9:
            at = self._r("<I")
            n = self._r("<Q")
            return [self._val(at) for _ in range(n)]
        return self._r(_SCALAR[t])

    # -- data ----------------------------------------------------------------
    def raw(self, name):
        """The tensor's bytes exactly as stored (memoryview into the mmap)."""
        e = self.tensors[name]
        return memoryview(self.mm)[e["offset"]:e["offset"] + e["nbytes"]]

    def check_tiling(self):
        """Tensors must lie inside the file, not overlap, and each must start on
        the alignment boundary. Returns the number of bytes in alignment padding."""
        spans = sorted((e["offset"], e["offset"] + e["nbytes"], k)
                       for k, e in self.tensors.items())
        pad = 0
        prev_end = self.data_start
        for s, e, k in spans:
            assert (s - self.data_start) % self.alignment == 0, f"{k} misaligned"
            assert s >= prev_end, f"{k} overlaps the previous tensor"
            pad += s - prev_end
            prev_end = e
        assert prev_end <= len(self.mm), "last tensor runs past the end of file"
        pad += len(self.mm) - prev_end
        return pad


def main():
    g = GGUF(sys.argv[1])
    pad = g.check_tiling()
    by_type = {}
    for e in g.tensors.values():
        c = by_type.setdefault(e["type"], [0, 0])
        c[0] += 1
        c[1] += e["nbytes"]
    print(f"GGUF v{g.version}: {len(g.tensors)} tensors, {len(g.meta)} metadata keys, "
          f"data at {g.data_start}, alignment {g.alignment}, padding {pad} B")
    for t, (n, b) in sorted(by_type.items()):
        print(f"  {t:5s} {n:4d} tensors {b / 1e9:7.3f} GB")
    total = sum(e["nbytes"] for e in g.tensors.values())
    print(f"  tensor bytes {total} + header {g.data_start} + padding {pad} "
          f"= {total + g.data_start + pad} (file {len(g.mm)})")
    assert total + g.data_start + pad == len(g.mm)
    print("tiling OK")


if __name__ == "__main__":
    main()
