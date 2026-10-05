#!/usr/bin/env python3
"""Build a binary BPE table for the pure-asm encoder (BUILD-TIME tool).

Reads tokenizer.json, writes bpe.bin:
  u64 magic (0x4250455F424C4F42), u64 n_rules, u64 n_chars,
  rules sorted by (left,right): u32 left, u32 right, u32 result, u32 rank,
  charmap sorted by cp: u32 cp, u32 id   (single-codepoint pieces only),
  bytemap: 256 x u32 (byte -> <0xHH> id).
Verified recipe (18/18 vs HF): space->U+2581, char-level init with
byte fallback, single lowest-rank merge per pass, leftmost tie-break.
Usage: mkbpe.py <tokenizer.json> <bpe.bin>
"""
import json
import struct
import sys

MAGIC = 0x4250455F424C4F42


def main():
    src, dst = sys.argv[1], sys.argv[2]
    d = json.load(open(src, encoding="utf-8"))
    vocab = d["model"]["vocab"]
    merges = d["model"]["merges"]
    rules = []
    for i, (a, b) in enumerate(merges):
        rules.append((vocab[a], vocab[b], vocab[a + b], i))
    # Sort right-major to match the asm single-u64 key (left | right<<32,
    # little-endian load compares right first). Stable: same (l,r) keeps
    # rank order, so dedup keeps the lowest rank (mirrors dict-first-wins).
    rules.sort(key=lambda r: (r[1], r[0]))
    deduped = []
    prev = None
    for r in rules:
        if (r[0], r[1]) != prev:
            deduped.append(r)
            prev = (r[0], r[1])
    print(f"rules: {len(rules)} sorted, {len(rules)-len(deduped)} dupes dropped",
          flush=True)
    rules = deduped
    chars = sorted((ord(s), i) for s, i in vocab.items() if len(s) == 1)
    print(f"single-char pieces: {len(chars)}", flush=True)
    bmap = []
    for byte in range(256):
        for fmt in ("<0x%02X>", "<0x%02x>"):
            if fmt % byte in vocab:
                bmap.append(vocab[fmt % byte])
                break
        else:
            raise SystemExit(f"no byte token for {byte:#04x}")
    with open(dst, "wb") as f:
        f.write(struct.pack("<QQQ", MAGIC, len(rules), len(chars)))
        for left, right, res, rank in rules:
            f.write(struct.pack("<IIII", left, right, res, rank))
        for cp, i in chars:
            f.write(struct.pack("<II", cp, i))
        for i in bmap:
            f.write(struct.pack("<I", i))
    print(f"wrote {dst}", flush=True)


if __name__ == "__main__":
    main()
