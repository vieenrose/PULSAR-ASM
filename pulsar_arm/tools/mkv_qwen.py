#!/usr/bin/env python3
"""Build a vocab blob for the pure-asm decoder from a Qwen3 tokenizer.

Same container as tools/mkvocab.py (which does gemma): u64 magic
(0x564f4341424c4242), u64 count, count x (u64 off, u64 len), then the
concatenated surfaces. The asm engine mmaps it read-only and put_surf copies
bytes into OUTBUF - no JSON, no escapes, no BPE at runtime.

The one thing that differs from gemma is that Qwen's vocab strings are
GPT-2 byte-level encoded: 'ĠNGO' is " NGO" and 'à´¨' is two bytes of UTF-8 that
happen to look like those codepoints. The byte decoder is inverted HERE, at
build time, so what lands in the blob is the token's actual UTF-8 bytes and the
asm needs no per-token decoding logic at all. That is why an "asm detokenizer"
is a table lookup: the decode work moved to where it costs nothing.

Usage: mkv_qwen.py [--repo prism-ml/Ternary-Bonsai-1.7B-unpacked] <vocab.bin>
       mkv_qwen.py --from-tokenizer-json <tokenizer.json> <vocab.bin>
"""
import json
import struct
import sys

MAGIC = 0x564F4341424C4242
CAP = 262144                      # engine's VTAB capacity; qwen needs 151669


def byte_decoder():
    """Inverse of GPT-2's bytes_to_unicode: char -> byte value."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\u00a1"), ord("\u00ac") + 1))
          + list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def surfaces(vocab, dec):
    """id -> raw UTF-8 bytes, with the byte-level encoding undone."""
    n = max(vocab.values()) + 1
    out = [b""] * n
    literal = 0
    for tok, i in vocab.items():
        try:
            out[i] = bytes(dec[c] for c in tok)
        except KeyError:
            # Added tokens (<|im_start|>) are literal text, not byte-level
            # encoded, and may contain characters the table has no entry for.
            out[i] = tok.encode("utf-8")
            literal += 1
    return out, literal


def main():
    args = sys.argv[1:]
    if "--from-tokenizer-json" in args:
        i = args.index("--from-tokenizer-json")
        src = args[i + 1]
        del args[i:i + 2]
        vocab = json.load(open(src, encoding="utf-8"))["model"]["vocab"]
    else:
        repo = "prism-ml/Ternary-Bonsai-1.7B-unpacked"
        if "--repo" in args:
            i = args.index("--repo")
            repo = args[i + 1]
            del args[i:i + 2]
        from transformers import AutoTokenizer
        vocab = AutoTokenizer.from_pretrained(repo).get_vocab()
    dst = args[0]

    dec = byte_decoder()
    surf, literal = surfaces(vocab, dec)
    n = len(surf)
    if n > CAP:
        sys.exit(f"vocab {n} exceeds the engine's VTAB capacity {CAP}")
    missing = [i for i, s in enumerate(surf) if not s]
    print(f"ids {n}, {n - len(missing)} with bytes, {len(missing)} empty "
          f"{missing[:8]}, {literal} literal (added tokens)", flush=True)

    # The table is written at the engine's full capacity so VTAB indexing is
    # unconditional; ids past the vocab stay (0, 0) and put_surf skips them.
    off = 16 + CAP * 16
    with open(dst, "wb") as f:
        f.write(struct.pack("<QQ", MAGIC, n))
        for s in surf:
            f.write(struct.pack("<QQ", off if s else 0, len(s)))
            off += len(s)
        for _ in range(CAP - n):
            f.write(struct.pack("<QQ", 0, 0))
        for s in surf:
            f.write(s)
    print(f"wrote {dst}: {off} bytes (count {n})", flush=True)


if __name__ == "__main__":
    main()
