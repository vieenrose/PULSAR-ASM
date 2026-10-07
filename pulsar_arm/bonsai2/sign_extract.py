"""Extract prism.hadamard sign tables -> /tmp/sign{K}.bin (float32 +/-1).
Usage: python3 sign_extract.py <file.gguf>"""
import struct, sys

def rd(f, n):
    b = f.read(n)
    if len(b) < n:
        raise SystemExit("truncated")
    return b

def su(f):
    n = struct.unpack("<Q", rd(f, 8))[0]
    return rd(f, n).decode()

def get_arr(f, t):
    et = struct.unpack("<I", rd(f, 4))[0]
    n = struct.unpack("<Q", rd(f, 8))[0]
    if et == 8:
        return [su(f) for _ in range(n)]
    sz = {4:4, 5:4, 10:8, 11:8}.get(et)
    if sz is None:
        raise SystemExit("unsupported arr elem %d" % et)
    raw = rd(f, n * sz)
    fmt = {4:"<%di" % n, 5:"<%di" % n, 10:"<%dq" % n, 11:"<%dq" % n}[et]
    return list(struct.unpack(fmt, raw))

def main(path):
    f = open(path, "rb")
    assert rd(f, 4) == b"GGUF"
    rd(f, 4)
    nt, nkv = struct.unpack("<QQ", rd(f, 16))
    widths = values = None
    for _ in range(nkv):
        name = su(f)
        t = struct.unpack("<I", rd(f, 4))[0]
        if name == "prism.hadamard.sign_widths":
            assert t == 9
            widths = get_arr(f, t)
        elif name == "prism.hadamard.sign_values":
            assert t == 9
            values = get_arr(f, t)
        else:
            # skip
            if t in (0, 4, 10):
                rd(f, {0:1, 4:4, 10:8}[t])
            elif t in (1, 3, 5, 11):
                rd(f, {1:1, 3:2, 5:4, 11:8}[t])
            elif t == 2:
                rd(f, 2)
            elif t in (6, 12):
                rd(f, {6:4, 12:8}[t])
            elif t == 7:
                rd(f, 1)
            elif t == 8:
                rd(f, struct.unpack("<Q", rd(f, 8))[0])
            elif t == 9:
                get_arr(f, t)  # discard
            else:
                raise SystemExit("unknown kv type %d" % t)
    assert widths and values, "sign arrays not found"
    print("widths:", widths, "total values:", len(values))
    off = 0
    for w in widths:
        vec = values[off:off + w]
        assert len(vec) == w and all(v in (1, -1) for v in vec)
        out = "/tmp/sign%d.bin" % w
        with open(out, "wb") as g:
            import struct as S
            g.write(S.pack("<%df" % w, *[float(v) for v in vec]))
        print("wrote %s (%d floats)" % (out, w))
        off += w
    assert off == len(values)

main(sys.argv[1])
