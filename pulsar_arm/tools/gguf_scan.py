import struct
import glob
from collections import Counter

f = glob.glob(
    "/home/luigi/.cache/huggingface/hub/models--google--gemma-3-1b-it-qat-q4_0-gguf"
    "/snapshots/*/gemma-3-1b-it-q4_0.gguf"
)[0]
b = open(f, "rb")
magic, ver, nt, na = struct.unpack("<IIQQ", b.read(24))
assert magic == 0x46554747, hex(magic)
print(f"ver={ver} tensors={nt} kv={na}", flush=True)


def rd_str():
    n = struct.unpack("<Q", b.read(8))[0]
    return b.read(n).decode("utf-8", "replace")


def rd_val(t):
    if t == 0:
        return struct.unpack("<B", b.read(1))[0]
    if t == 1:
        return struct.unpack("<b", b.read(1))[0]
    if t == 2:
        return struct.unpack("<H", b.read(2))[0]
    if t == 3:
        return struct.unpack("<h", b.read(2))[0]
    if t == 4:
        return struct.unpack("<I", b.read(4))[0]
    if t == 5:
        return struct.unpack("<i", b.read(4))[0]
    if t == 6:
        return struct.unpack("<f", b.read(4))[0]
    if t == 7:
        return bool(struct.unpack("<B", b.read(1))[0])
    if t == 8:
        return rd_str()
    if t == 9:
        at = struct.unpack("<I", b.read(4))[0]
        n = struct.unpack("<Q", b.read(8))[0]
        return [rd_val(at) for _ in range(n)]
    if t == 10:
        return struct.unpack("<Q", b.read(8))[0]
    if t == 11:
        return struct.unpack("<q", b.read(8))[0]
    if t == 12:
        return struct.unpack("<d", b.read(8))[0]
    raise ValueError(t)


WANT = (
    "general.architecture", "gemma3.embedding_length", "gemma3.block_count",
    "gemma3.feed_forward_length", "gemma3.attention.head_count",
    "gemma3.attention.head_count_kv", "gemma3.attention.key_length",
    "gemma3.attention.value_length", "tokenizer.ggml.model",
    "general.quantization_version",
)
for _ in range(na):
    k = rd_str()
    t = struct.unpack("<I", b.read(4))[0]
    v = rd_val(t)
    if k in WANT:
        print(f"{k} = {v}", flush=True)

TN = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K",
}
c = Counter()
for _ in range(nt):
    nl = struct.unpack("<Q", b.read(8))[0]
    assert nl < 200, nl
    name = b.read(nl).decode("utf-8", "replace")
    nd = struct.unpack("<I", b.read(4))[0]
    dims = struct.unpack("<" + "Q" * nd, b.read(8 * nd))
    ty = struct.unpack("<I", b.read(4))[0]
    off = struct.unpack("<Q", b.read(8))[0]
    c[TN.get(ty, ty)] += 1
    if "blk.0." in name or "token_embd" in name or "output" in name:
        print(name, dims, TN.get(ty, ty), flush=True)
print(dict(c), flush=True)
