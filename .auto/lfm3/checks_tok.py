#!/usr/bin/env python3
"""checks_tok.py -- S1 gate: HF is the reference, the C port must match id-for-id.

Builds fixtures (31 hand-written + 500 seeded fuzz, the same corpus mktok.py replays),
dumps expected ids from the HF tokenizer, builds tok.bin, compiles the C driver and
compares. Exit 0 iff every fixture is exact.
"""
import json
import os
import random
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))  # .auto/lfm3
REPO = os.path.dirname(os.path.dirname(HERE))       # repo root
MODEL = os.path.expanduser("~/models/lfm3/LFM3-Audio-2B")
FIX = os.path.join(HERE, "tok_fix")
BLOB = os.path.join(FIX, "tok.bin")

CORPUS = [
    "Hello, world!",
    "I'm sure they'll've done it, won't they?",
    "1234 123 12 1 12345 999999",
    "  leading and trailing  ",
    "line one\nline two\n\n\nline five",
    "\ttabbed\tand   spaced   out   ",
    "café naïve résumé", "café naïve résumé",
    "日本語のテキストと中文 한국어", "Привет, мир! Γειά σου Κόσμε",
    "emoji 😀🎧 audio 🎵", "def f(x): return x**2  # code",
    "a" * 300, "12" * 200,
    "<|im_start|>user<|im_end|>",
    "e.g. U.S.A. 3.14159", "don't  can't  I'd  you're",
    "Ω≈ç√∫˜µ≤≥÷", "العربية فارسية", "עברית", "हिन्दी संस्कृतम्", "ไทย",
    "አማርኛ", "ᚠᚢᚦᚨᚱᚲ", "ⅫⅬⅭⅮ ⑫⑬⑭ ①②③", "١٢٣٤٥ ༡༢༣",
    "ｆｕｌｌｗｉｄｔｈ　１２３", "𝕳𝖊𝖑𝖑𝖔 𝓦𝓸𝓻𝓵𝓭", "a1b2c3 9z8y7x",
    " nbsp emspace​tiny", "MiXeD CaSe WoRdS 123abc456",
]


def main():
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    os.makedirs(FIX, exist_ok=True)
    rng = random.Random(1234)
    alpha = (list("abcXYZ019 .'\t\n\r\"!?-_/:;()[]{}#$%&*+=<>@~`|\\")
             + list("éüñ中日한ПриветΑβγ") + list("😀🎵") + ["́", " ", " "])
    texts = list(CORPUS)
    for _ in range(500):
        texts.append("".join(rng.choice(alpha) for _ in range(rng.randint(1, 40))))
    for i, text in enumerate(texts):
        with open(os.path.join(FIX, f"f{i:04d}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        ids = hf.encode(text, add_special_tokens=False)
        with open(os.path.join(FIX, f"f{i:04d}.ids"), "wb") as f:
            f.write(struct.pack(f"<{len(ids)}I", *ids))
    print(f"fixtures: {len(texts)} written", flush=True)

    r = subprocess.run([sys.executable, os.path.join(REPO, "pulsar_arm/tools/mktok.py"),
                        "--blob", BLOB], capture_output=True, text=True)
    print(r.stdout[-500:] if r.stdout else "", flush=True)
    if r.returncode:
        print(r.stderr[-2000:], flush=True)
        return 1
    src = os.path.join(REPO, "pulsar_arm/asm")
    drv = os.path.join(FIX, "tok_test")
    c = subprocess.run(["gcc", "-O2", "-Wall", "-o", drv,
                        os.path.join(src, "lfm3_tok_test.c"),
                        os.path.join(src, "lfm3_tok.c")],
                       capture_output=True, text=True)
    if c.returncode:
        print(c.stderr, flush=True)
        return 1
    r = subprocess.run([drv, BLOB, FIX], capture_output=True, text=True)
    print(r.stdout, flush=True)
    if r.stderr:
        print(r.stderr, flush=True)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
