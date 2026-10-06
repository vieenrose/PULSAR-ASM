import struct
import glob
from collections import Counter
from huggingface_hub import snapshot_download

p = snapshot_download("unsloth/gemma-3-270m-it-GGUF",
                      allow_patterns=["gemma-3-270m-it-Q8_0.gguf"])
print("SNAP:", p, flush=True)
f = glob.glob(p + "/*.gguf")[0]
import os
print("SIZE-MB:", os.path.getsize(f) // 1048576, flush=True)
b = open(f, "rb")
magic, ver, nt, na = struct.unpack("<IIQQ", b.read(24))
assert magic == 0x46554747, hex(magic)
print(f"ver={ver} tensors={nt} kv={na}", flush=True)


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


WANT = ("general.architecture", "gemma3.embedding_length",
        "gemma3.block_count", "gemma3.feed_forward_length",
        "gemma3.attention.head_count", "gemma3.attention.head_count_kv",
        "gemma3.attention.key_length", "tokenizer.ggml.model")
for _ in range(na):
    k = rd_str()
    t = struct.unpack("<I", b.read(4))[0]
    if k in WANT:
        if t == 8:
            print(k, "=", rd_str(), flush=True)
        elif t == 4:
            print(k, "=", struct.unpack("<I", b.read(4))[0], flush=True)
        else:
            skip_val(t)
    else:
        skip_val(t)

TN = {0: "F32", 1: "F16", 2: "Q4_0", 8: "Q8_0", 12: "Q4_K", 14: "Q6_K", 15: "Q8_K"}
c = Counter()
shown = 0
for _ in range(nt):
    nl = struct.unpack("<Q", b.read(8))[0]
    assert nl < 200, nl
    name = b.read(nl).decode()
    nd = struct.unpack("<I", b.read(4))[0]
    dims = struct.unpack("<" + "Q" * nd, b.read(8 * nd))
    ty = struct.unpack("<I", b.read(4))[0]
    off = struct.unpack("<Q", b.read(8))[0]
    c[TN.get(ty, ty)] += 1
    if shown < 16 and ("blk.0." in name or "embd" in name or "output" in name):
        print(name, dims, TN.get(ty, ty), flush=True)
        shown += 1
print(dict(c), flush=True)
