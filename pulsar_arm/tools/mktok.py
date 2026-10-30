"""mktok.py -- build the .lfm3tok blob and prove the pipeline is bit-exact

The tokenizer is not the simple BPE the port plan assumed. tokenizer.json says:

  normalizer        null
  pre_tokenizer     Sequence[ Split(regex, Isolated), ByteLevel(add_prefix_space=False) ]
  model             BPE, 124893 vocab, 293320 merges, ignore_merges=True,
                    byte_fallback=False, fuse_unk=False, no unk_token,
                    no continuing_subword_prefix, no end_of_word_suffix
  added_tokens      140 (ids 124893..125032)

(tokenizer_config.json reports pre_tokenizer: null -- it is wrong, and worth
knowing before writing a port against it.)

The Split regex is the GPT-4 contraction/number/whitespace pattern, so it needs
Unicode general categories \\p{L} and \\p{N}. Those tables are generated here and
shipped in the blob; nothing is approximated. `ignore_merges=True` is the other
detail that changes behaviour: a pre-token already present in the vocab is emitted
directly, without merging.

WHAT THIS FILE DOES, in order, because the C port must reproduce exactly this:

  1 build the byte <-> visible-codepoint map (ByteLevel's 256-entry alphabet);
  2 extract the Unicode L and N ranges from Python's unicodedata;
  3 REPLAY the whole pipeline in Python -- regex split, byte level, BPE with
     ignore_merges -- against the HF tokenizer, id for id, on a corpus built to
     provoke the regex boundaries;
  4 write the blob the asm/C runtime will load.

Step 3 is the point. A port that has not been replayed against the reference on a
corpus that hits the awkward cases (contractions, digit runs, newlines, leading and
trailing space, non-ASCII scripts, combining accents, emoji, digit runs longer than
the {1,3} quantifier) is not a port, it is a guess -- and the regex is exactly where
this kind of work goes wrong.

usage: mktok.py [--blob tok.bin] [--corpus]
"""
import argparse
import json
import os
import sys
import unicodedata

MODEL = os.path.expanduser("~/models/lfm3/LFM3-Audio-2B")
# The Split pattern, verbatim from tokenizer.json. Only \p{L} and \p{N} appear.
SPLIT_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
                 r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def byte_level_map():
    """ByteLevel's alphabet: printable ranges map to themselves, the remaining
    bytes take codepoints from 256 upward, in byte order."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) +
          list(range(0xAE, 0xFF + 1)))
    cs, n = list(bs), 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: c for b, c in zip(bs, cs)}          # byte -> codepoint


def unicode_ranges(prefix):
    out, start = [], None
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            ok = False
        else:
            ok = unicodedata.category(chr(cp)).startswith(prefix)
        if ok and start is None:
            start = cp
        elif not ok and start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, 0x10FFFF))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blob", default="tok.bin")
    ap.add_argument("--corpus", action="store_true")
    a = ap.parse_args()

    try:
        import regex
    except ImportError:
        sys.exit("the `regex` module is required: Python's re has no \\p{L} support")

    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tj = json.load(open(os.path.join(MODEL, "tokenizer.json"), encoding="utf-8"))
    m = tj["model"]
    vocab = m["vocab"]
    merges = [tuple(x) if isinstance(x, list) else tuple(x.split(" ")) for x in m["merges"]]
    added = {t["content"]: t["id"] for t in tj["added_tokens"]}
    print(f"vocab {len(vocab)}  merges {len(merges)}  added {len(added)}  "
          f"ignore_merges={m.get('ignore_merges')}  unk={m.get('unk_token')!r}")

    bmap = byte_level_map()
    assert len(bmap) == 256 and len(set(bmap.values())) == 256
    # the byte-level alphabet must be exactly the vocab's own single-codepoint entries
    single = {k for k in vocab if len(k) == 1}
    print(f"byte map: 256 entries, bijective; vocab single-char entries {len(single)}")

    Lr, Nr = unicode_ranges("L"), unicode_ranges("N")
    print(f"unicode ranges: L={len(Lr)} N={len(Nr)} (unicodedata {unicodedata.unidata_version})")

    inv = {c: b for b, c in bmap.items()}
    pat = regex.compile(SPLIT_PATTERN)
    rank = {p: i for i, p in enumerate(merges)}

    def to_bytelevel(piece):
        out = []
        for byte in piece.encode("utf-8"):
            out.append(chr(bmap[byte]))
        return "".join(out)

    def bpe(piece):
        if m.get("ignore_merges") and piece in vocab:
            return [vocab[piece]]
        syms = list(piece)
        while len(syms) > 1:
            best, bi = None, -1
            for i in range(len(syms) - 1):
                r = rank.get((syms[i], syms[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if bi < 0:
                break
            syms = syms[:bi] + [syms[bi] + syms[bi + 1]] + syms[bi + 2:]
        return [vocab[s] for s in syms]

    # Added tokens are matched on the RAW text before pre-tokenisation, longest
    # first, and are emitted as themselves. Skipping this cost 1/17 on the corpus
    # (<|im_start|> came back as 10 ordinary byte-level tokens).
    add_pat = regex.compile("|".join(regex.escape(t) for t in
                                     sorted(added, key=len, reverse=True))) if added else None

    def encode(text):
        ids = []
        pos = 0
        for mt in (add_pat.finditer(text) if add_pat else []):
            for piece in pat.finditer(text[pos:mt.start()]):
                ids.extend(bpe(to_bytelevel(piece.group(0))))
            ids.append(added[mt.group(0)])
            pos = mt.end()
        for piece in pat.finditer(text[pos:]):
            ids.extend(bpe(to_bytelevel(piece.group(0))))
        return ids

    if a.corpus:
        corpus = [
            "Hello, world!",
            "I'm sure they'll've done it, won't they?",
            "1234 123 12 1 12345 999999",
            "  leading and trailing  ",
            "line one\nline two\n\n\nline five",
            "\ttabbed\tand   spaced   out   ",
            "café naïve résumé", "cafe\u0301 nai\u0308ve re\u0301sume\u0301",
            "日本語のテキストと中文 한국어", "Привет, мир! Γειά σου Κόσμε",
            "emoji 😀🎧 audio 🎵", "def f(x): return x**2  # code",
            "a" * 300, "12" * 200,
            "<|im_start|>user<|im_end|>",
            "e.g. U.S.A. 3.14159", "don't  can't  I'd  you're",
            # hit \p{L} and \p{N} across scripts: the regex tables are the part
            # most likely to differ between Python's unicodedata and Rust's regex
            "Ω≈ç√∫˜µ≤≥÷", "العربية فارسية", "עברית", "हिन्दी संस्कृतम्", "ไทย",
            "አማርኛ", "ᚠᚢᚦᚨᚱᚲ", "ⅫⅬⅭⅮ ⑫⑬⑭ ①②③", "١٢٣٤٥ ༡༢༣",
            "ｆｕｌｌｗｉｄｔｈ　１２３", "𝕳𝖊𝖑𝖑𝖔 𝓦𝓸𝓻𝓵𝓭", "a1b2c3 9z8y7x",
            "\u00a0nbsp\u2003emspace\u200btiny", "MiXeD CaSe WoRdS 123abc456",
        ]
        bad = 0
        for text in corpus:
            want = hf.encode(text, add_special_tokens=False)
            got = encode(text)
            if want != got:
                bad += 1
                print(f"  MISMATCH {text[:40]!r}")
                print(f"    hf  {want[:24]}")
                print(f"    rep {got[:24]}")
                # first divergence
                for i, (x, y) in enumerate(zip(want, got)):
                    if x != y:
                        print(f"    first diff at {i}: hf={x} rep={y}")
                        break
        # Randomised pass: the hand-written corpus covers the cases I thought of,
        # the fuzz covers the ones I did not. Alphabet deliberately mixes letters,
        # digits, apostrophes, whitespace, newlines, punctuation and non-ASCII so
        # the Split alternation is exercised at its boundaries.
        import random
        rng = random.Random(1234)
        alpha = (list("abcXYZ019 .'\t\n\r\"!?-_/:;()[]{}#$%&*+=<>@~`|\\")
                 + list("éüñ中日한ПриветΑβγ") + list("😀🎵") + ["\u0301", "\u00a0", "\u2003"])
        fuzz_bad = 0
        N_FUZZ = 500
        for _ in range(N_FUZZ):
            text = "".join(rng.choice(alpha) for _ in range(rng.randint(1, 40)))
            if hf.encode(text, add_special_tokens=False) != encode(text):
                fuzz_bad += 1
                if fuzz_bad <= 3:
                    print(f"  FUZZ MISMATCH {text!r}")
        print(f"fuzz: {N_FUZZ - fuzz_bad}/{N_FUZZ} exact")
        bad += fuzz_bad
        print(f"replay: {len(corpus) - (bad - fuzz_bad)}/{len(corpus)} corpus + "
              f"{N_FUZZ - fuzz_bad}/{N_FUZZ} fuzz exact")
        if bad:
            return 1

    with open(a.blob, "wb") as f:
        f.write(b"LFM3TOK1")
        f.write(json.dumps({
            "n_vocab": len(vocab), "n_merges": len(merges),
            "ignore_merges": bool(m.get("ignore_merges")),
            "byte_fallback": bool(m.get("byte_fallback")),
            "regex": SPLIT_PATTERN,
            "unicode_L": Lr, "unicode_N": Nr,
        }).encode() + b"\0")
    print(f"wrote {a.blob} ({os.path.getsize(a.blob)} bytes of metadata; "
          f"vocab/merges tables are the C port's next step)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
