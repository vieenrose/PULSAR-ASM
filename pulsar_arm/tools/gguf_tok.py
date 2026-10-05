import struct
import glob

f = glob.glob(
    "/home/luigi/.cache/huggingface/hub/models--google--gemma-3-1b-it-qat-q4_0-gguf"
    "/snapshots/*/gemma-3-1b-it-q4_0.gguf"
)[0]
b = open(f, "rb")
magic, ver, nt, na = struct.unpack("<IIQQ", b.read(24))
assert magic == 0x46554747


def rd_str():
    n = struct.unpack("<Q", b.read(8))[0]
    return b.read(n).decode("utf-8", "replace")


def rd_len(t):
    # returns element count for tokenizer arrays without reading all
    if t == 8:
        n = struct.unpack("<Q", b.read(8))[0]
        b.seek(n, 1)
        return 1
    if t in (4, 6):
        b.seek(4, 1)
        return 1
    if t == 9:
        at = struct.unpack("<I", b.read(4))[0]
        n = struct.unpack("<Q", b.read(8))[0]
        if at == 8:
            for _ in range(n):
                m = struct.unpack("<Q", b.read(8))[0]
                b.seek(m, 1)
        elif at in (4, 5, 6):
            b.seek(4 * n, 1)
        elif at == 7:
            b.seek(n, 1)
        else:
            raise ValueError((at, n))
        return n
    raise ValueError(t)


for _ in range(na):
    k = rd_str()
    t = struct.unpack("<I", b.read(4))[0]
    if k.startswith("tokenizer"):
        try:
            n = rd_len(t)
            print(f"{k} type={t} len={n}", flush=True)
        except Exception as e:
            print(f"{k} type={t} ERR {e}", flush=True)
            break
    else:
        # skip non-tokenizer KV (reuse minimal reader)
        if t == 8:
            rd_str()
        elif t in (0, 1, 7):
            b.seek(1, 1)
        elif t in (2, 3):
            b.seek(2, 1)
        elif t in (4, 5, 6):
            b.seek(4, 1)
        elif t in (10, 11, 12):
            b.seek(8, 1)
        elif t == 9:
            rd_len(t)
        else:
            raise ValueError(t)
