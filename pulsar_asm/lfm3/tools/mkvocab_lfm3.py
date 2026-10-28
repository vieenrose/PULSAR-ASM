#!/usr/bin/env python3
"""Build vocab blob for LFM3 text tokenizer."""
import json
import struct
import sys

MAGIC = 0x564F4341424C4242
NVOCAB = 262144


def main():
    src, dst = sys.argv[1], sys.argv[2]
    d = json.load(open(src, encoding="utf-8"))
    vocab = d["model"]["vocab"]
    assert d["model"]["type"] == "BPE"
    surf = [b""] * NVOCAB
    for s, i in vocab.items():
        if 0 <= i < NVOCAB:
            surf[i] = s.encode("utf-8")
    off = 16 + NVOCAB * 16
    with open(dst, "wb") as f:
        f.write(struct.pack("<QQ", MAGIC, NVOCAB))
        for s in surf:
            f.write(struct.pack("<QQ", off if s else 0, len(s)))
            off += len(s)
        for s in surf:
            f.write(s)
    print(f"wrote {dst}: {off} bytes", flush=True)


if __name__ == "__main__":
    main()
