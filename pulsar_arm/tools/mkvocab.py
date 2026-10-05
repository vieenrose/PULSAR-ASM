#!/usr/bin/env python3
"""Build a flat vocab blob for the pure-asm decoder (BUILD-TIME tool).

Reads tokenizer.json (BPE `model.vocab` string->id map), writes vocab.bin:
  u64 magic (0x564f4341424c4242), u64 count (=262144),
  then count x (u64 off, u64 len)  [off = file offset of surface bytes],
  then the concatenated UTF-8 surfaces (json-decoded).
Missing ids get (0, 0). The asm engine mmaps this read-only; no JSON
parsing, no escapes, no writes at runtime.
Usage: mkvocab.py <tokenizer.json> <vocab.bin>
"""
import json
import struct
import sys

MAGIC = 0x564F4341424C4242
NVOCAB = 262144


def main():
    src, dst = sys.argv[1], sys.argv[2]
    d = json.load(open(src, encoding="utf-8"))
    vocab = d["model"]["vocab"]
    assert d["model"]["type"] == "BPE", d["model"].get("type")
    surf = [b""] * NVOCAB
    for s, i in vocab.items():
        if 0 <= i < NVOCAB:
            surf[i] = s.encode("utf-8")
    marvel = [i for i, s in enumerate(surf) if not s]
    print(f"vocab entries: {sum(1 for s in surf if s)}/{NVOCAB} "
          f"(missing {len(marvel)}: {marvel[:8]})", flush=True)
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
