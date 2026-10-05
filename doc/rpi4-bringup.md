# rpi4 bring-up: QAT-mobile E2B on Raspberry Pi 4

Goal (user-directed): run `google/gemma-4-E2B-it-qat-mobile-transformers` on
`luigi@raspberrypi.tailf63b31.ts.net`, then decide what a PULSAR-ASM ARM port
needs. This branch tracks Pi-side work; the x86 engine on `gemma-4` is untouched.

## Why not the bf16 blob

- Pi 4 Model B, 4× Cortex-A72 (ASIMD/NEON only: no SVE, no i8mm, no dotprod),
  **3.8 GB RAM, ~1.4 GB free**, 328 GB disk free, gcc 12.2, Python 3.11.
- The 9.258 GB bf16 blob cannot fit. Ever. No amount of NEON will fix that.
- The QAT-mobile checkpoint (wNa8o8 schema: 2-bit lm_head/embeds/most MLPs,
  4-bit PLE embed + early MLPs, `quant_method: gemma` handled inside
  transformers) is 2.46 GB on disk — the only E2B that fits this machine.

## Environment (on the Pi, `~/pulsar-rpi/`)

- `python3 -m venv v`, then `./v/bin/pip install -U pip torch transformers accelerate`
  → torch 2.14.1, transformers 5.18.0, accelerate 1.15.0. (PyPI torch drags
  CUDA wheels along; they are dead weight on ARM, CPU backend works.)
- Gated repo needs `HF_TOKEN` in the environment for the download; it is NOT
  stored on the Pi.
- First-light script: `rpi_firstlight.py` (text-only, greedy, one en + one zh
  prompt, prints tok/s + peak RSS). Run: `HF_TOKEN=... ./v/bin/python -u
  rpi_firstlight.py`.

## Open questions for the port decision

1. Does it even load in 3.8 GB (weights 2.46 GB + torch + KV + activations)?
2. What tok/s does the reference manage (expectation: well under 1 tok/s)?
3. Which kernels would an ARM port need for the 2/4-bit schema (dequant +
   GEMV in NEON, no AVX2 anywhere), and is the quality worth it?

## First light (2026-10-05)

- The non-IT `gemma-3-270m-qat-q4_0-unquantized` is a base model: it babbles
  (`Category` loops, prompt echoing). Its tokenizer ships no chat template.
  The `-it` variant (`google/gemma-3-270m-it-qat-q4_0-unquantized`) is the one
  that follows instructions.
- `-it` on Pi 4 CPU: loads in ~60 s at 1.4 GB peak, generates at **3.0 tok/s**
  within 1.6 GB. `en` answers correctly (`The answer is Paris.`); `zh`
  echoes the question and stops after 8 tokens - 270M-parameter reality.
- Raw E2B-QAT comparison: it loads (304 s, 2.76 GB) but immediately swap-
  thrashes (4+ GB in swap, 0.001 tok/s) because the box already runs ~700 MB
  of ruby/postgres before we start. Not a viable reference without freeing RAM.
- A `c10::Error` frame prints during generation on both models without stopping
  output; cause not yet chased (suspect cache-implementation fallback).
