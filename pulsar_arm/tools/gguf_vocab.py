import struct
import glob
import json

f = glob.glob(
    "/home/luigi/.cache/huggingface/hub/models--google--gemma-3-1b-it-qat-q4_0-gguf"
    "/snapshots/*/gemma-3-1b-it-q4_0.gguf"
)[0]
b = open(f, "rb")
magic, ver, nt, na = struct.unpack("<IIQQ", b.read(24))


def rd_str():
    n = struct.unpack("<Q", b.read(8))[0]
    return b.read(n)


def rd_val(t):
    if t in (0, 1, 7):
        return b.read(1)
    if t in (2, 3):
        return b.read(2)
    if t in (4, 5, 6):
        return b.read(4)
    if t in (10, 11, 12):
        return b.read(8)
    if t == 8:
        return rd_str()
    if t == 9:
        at = struct.unpack("<I", b.read(4))[0]
        n = struct.unpack("<Q", b.read(8))[0]
        out = []
        for _ in range(n):
            if at == 8:
                out.append(rd_str())
            elif at in (4, 5, 6):
                out.append(struct.unpack("<f" if at == 6 else "<i", b.read(4))[0])
            elif at == 7:
                out.append(b.read(1))
            else:
                raise ValueError(at)
        return out
    raise ValueError(t)


toks = None
ids = {}
for _ in range(na):
    k = rd_str().decode()
    t = struct.unpack("<I", b.read(4))[0]
    v = rd_val(t)
    if k == "tokenizer.ggml.tokens":
        toks = [x.decode("utf-8", "replace") for x in v]
    if k in ("tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
             "tokenizer.ggml.unknown_token_id", "tokenizer.ggml.padding_token_id"):
        ids[k.split(".")[-1]] = struct.unpack("<I", v)[0]
    if k == "tokenizer.chat_template":
        print("CHAT-TEMPLATE:", v.decode()[:200], flush=True)
print("ntok:", len(toks), flush=True)
print("ids:", ids, flush=True)
j = glob.glob(
    "/home/luigi/.cache/huggingface/hub/models--google--gemma-3-270m-it-qat-q4_0-unquantized"
    "/snapshots/*/tokenizer.json"
)[0]
v270 = json.load(open(j))["model"]["vocab"]
# note: 270m vocab maps str->id; GGUF toks[i] should equal the str with vocab id i
match = sum(1 for s, i in v270.items() if i < len(toks) and toks[i] == s)
print(f"vocab match: {match}/{len(v270)}", flush=True)
# show first mismatch if any
for s, i in v270.items():
    if i >= len(toks) or toks[i] != s:
        print("FIRST-MISMATCH:", i, repr(s)[:40], repr(toks[i])[:40] if i < len(toks) else None, flush=True)
        break
