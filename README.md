# PULSAR-ASM

**Pure-assembly Gemma-3 inference on a Raspberry Pi 4.** Static AArch64
binary, direct `svc` syscalls, no libc, no CRT, no third-party code. The
tokenizer, sampler, chat REPL and the whole forward pass live in one file:
`pulsar_arm/asm/core.S`.

Three checkpoints run unmodified, and the engine reads its dimensions from the
safetensors header, so one binary covers all of them:

| checkpoint | hidden | intermediate | layers | bytes/token | decode |
|---|---|---|---|---|---|
| `google/gemma-3-1b-it-qat-q4_0-unquantized` | 1152 | 6912 | 26 | 2.00 GB | **~543 ms/token** (1.8 tok/s) |
| `google/gemma-3-270m-it-qat-q4_0-unquantized` | 640 | 2048 | 18 | 536 MB | ~147 ms/token (6.8 tok/s) |
| `google/functiongemma-270m-it` | 640 | 2048 | 18 | 536 MB | same as 270m |

## At a glance

| | |
|---|---|
| Target | Raspberry Pi 4, Cortex-A72 / NEON only (no SVE, no dotprod), 3 cores |
| Engine | one static binary: `as` + `ld`, direct syscalls, zero dependencies |
| Bandwidth | ~3.6–3.9 GB/s of a measured 3.93 GB/s streaming ceiling (92–98 %) |
| Parity | greedy output identical to transformers, token for token (1B bf16, 270m fp32) |
| Determinism | fixed seed ⇒ byte-identical transcripts across runs |

## Quick start

```sh
# 1. build the engine
cd pulsar_arm/asm
as -o core.o core.S && ld -static -o core core.o

# 2. tokenizer blobs, once (the gemma-3 checkpoints share one tokenizer)
python3 ../tools/mkvocab.py <tokenizer.json> vocab.bin   # surfaces + byte fallback
python3 ../tools/mkbpe.py   <tokenizer.json> bpe.bin     # 514,906 merge rules

# 3. greedy bench: 8-token prompt, 32 decode steps, ids + ms/token
./core gemma-3-270m-it-qat-q4_0.safetensors vocab.bin
./core gemma-3-1b-it-qat-q4_0.safetensors   vocab.bin

# 4. chat REPL (needs bpe.bin)
./core gemma-3-1b-it-qat-q4_0.safetensors vocab.bin 2 c bpe.bin 1000 950 48

# 5. file mode: raw prompt on stdin, greedy, one shot (needs bpe.bin)
./core functiongemma-270m-it.safetensors fcvocab.bin 2 f fcbpe.bin 1000 950 16 < prompt.txt
```

`functiongemma-270m-it` uses its own `fcvocab.bin` / `fcbpe.bin`, built from its
own `tokenizer.json` with the same two tools.

## Modes and flags

```
./core <model.safetensors> <vocab.bin> [tok] [mode] [bpe.bin] [temp_milli] [topp_milli] [gencap]
```

| arg | meaning |
|---|---|
| `tok` | token id for the startup layer-0 signature self-test (the bench prompt itself is built in) |
| `mode` | `0` greedy bench (default) · `1` sampled bench · `c` chat REPL · `f` file mode |
| `bpe.bin` | required by `c` and `f` |
| `temp_milli` | sampling temperature ×1000 (default 1000, min 50) |
| `topp_milli` | top-p ×1000 (default 950) |
| `gencap` | max generated tokens (default 128, max 400) |

Sampling follows the model card: temperature 1.0, **top-k 64**, top-p 0.95
(min-p 0), which is what the sampler implements. Measured guidance beyond that:
keep temp ≥ 0.7 (0.5 degenerates into token loops), top-p is neutral here
(0.5 / 0.95 / 1.0 all stay diverse), and gencap bounds output cleanly. The chat template (SOT/EOT roles) is verified id-exact
against HF, and every run prints its prefill ids as `tpl:` for transparency.

## Demos

Real runs, nothing re-typed: prompt, `tpl:` ids and response are the engine's
own bytes, the status bar carries that session's measured rate, and the amber
`>` line is the user turn (in the FunctionGemma clips it is the user message
inside the prompt file — file mode does not echo it, so the status bar names
the file). The two FunctionGemma clips also show the system turn verbatim, i.e.
where the tool is defined. All frames share one font size (17) and one
typeface — DejaVu Sans Mono, with WenQuanYi Zen Hei used only for the CJK
glyphs DejaVu lacks, at the same size and line height. Only pacing is
libertied.

**gemma-3-1b-it** — each clip uses the most complex prompt the checkpoint
answers *correctly* (a four-item structured list, and a three-item one in
zh-TW), both complete and clean end to end:

![1B chat demo](doc/gemma3-1b-chat-en.gif)
![1B chat demo, Traditional Chinese](doc/gemma3-1b-chat-zh-tw.gif)

```sh
printf 'List the four seasons and one thing that changes in each.\n' | ./core gemma-3-1b-it-qat-q4_0.safetensors vocab.bin 2 c bpe.bin 1000 950 123
printf '請列出保持健康的三個要點。\n'                                | ./core gemma-3-1b-it-qat-q4_0.safetensors vocab.bin 2 c bpe.bin 1000 950 172
```

The gen caps above are tuned to end on the model's closing line (123 for the
English clip, 172 for the zh-TW one); the sampler is the checkpoint's
recommended configuration — temperature 1.0, top-k 64, top-p 0.95 — which is
what the engine implements.

**gemma-3-270m-it** — same binary, 18 layers:

![270m chat demo](doc/gemma3-270m-chat-en.gif)
![270m chat demo, Traditional Chinese](doc/gemma3-270m-chat-zh-tw.gif)

```sh
printf 'Explain gravity in two sentences for a child.\n' | ./core gemma-3-270m-it-qat-q4_0.safetensors vocab.bin 2 c bpe.bin 1000 950 48
printf '請用一句話解釋什麼是量子力學。\n'                 | ./core gemma-3-270m-it-qat-q4_0.safetensors vocab.bin 2 c bpe.bin 1000 950 48
```

**FunctionGemma-270m-it** — both an English and a Traditional-Chinese question
become the same tool call (one schema, only the user turn differs):

![FunctionGemma tool call, English prompt](doc/functiongemma-toolcall-en.gif)
![FunctionGemma tool call, zh-TW prompt](doc/functiongemma-toolcall-zh-tw.gif)

```sh
python3 pulsar_arm/tools/fc_write.py   # -> /tmp/tool_official.txt, /tmp/tool_official_zhtw.txt
./core functiongemma.safetensors fcvocab.bin 2 f fcbpe.bin 1000 950 16 < /tmp/tool_official.txt
./core functiongemma.safetensors fcvocab.bin 2 f fcbpe.bin 1000 950 16 < /tmp/tool_official_zhtw.txt
```

Re-render any subset with `python3 pulsar_arm/tools/make_chat_gif.py` (needs
PIL + ffmpeg); the transcripts are literals in that file, so a re-run can only
reproduce these frames, never invent them.

## Verified gates

- **HF-exact decode.** Greedy output is identical to transformers token for
  token: 32/32 bench tokens for both 270m (fp32) and 1B (bf16), and complete
  chat transcripts — 1B answers *"The capital of France is **Paris**."* exactly
  as HF does. The 270m residual also matches an fp64 oracle to ~1e-7 at every
  layer and every position.
- **One binary, header-driven dims.** `dims hid/inter/layers/vocab: 1152 6912
  26 262144` and `640 2048 18 262144` from the same build — hidden width,
  intermediate width, layer count, vocab, the folded-norm stride, every kernel
  dimension and the full-attention set (`i mod 6 == 5`) all come from the file.
- **BPE encode** 18/18 exact vs HF (char-level init, rank-order merges,
  leftmost tie-break, byte fallback).
- **Determinism.** Greedy, sampled and full chat transcripts reproduce
  bit-for-bit across runs at a fixed seed.
- **Robustness.** Empty / whitespace / CRLF input, 203-token prefill, 8-turn
  depth with automatic context reset, EOF and exit paths, gen-cap boundary.
- **asm at C parity.** GEMV reaches 3.88 GB/s using a single SHLL instruction
  for bf16 widening; the bit-identical wins the C port proved (fused
  elementwise passes, one layer body per call, precomputed tables) are present
  in the asm path too.

## Performance

- **The wall.** A numpy streaming-sum ceiling of 3.93 GB/s was measured on this
  Pi. The driver moves 536 MB/token for 270m and 2.00 GB/token for 1B, i.e.
  3.6–3.9 GB/s depending on the run — the same bus saturation at both sizes,
  so the floor is ~136 ms/token for 270m and ~510 ms/token for 1B here.
- **Multi-core is not the answer.** A 3-thread static-partition GEMV is 4.2 %
  *slower* (same-job A/B, twice): the bus is already saturated.
- **Thermal drift.** Absolute numbers move 5–18 % between sessions; compare
  same-job A/B ratios, never absolutes.

## Layout

```
pulsar_arm/asm/core.S        the engine: mmap stage → kernels → layer forward
                             → sampler → BPE → chat / bench / file drivers
pulsar_arm/tools/            build-time converters (mkvocab, mkbpe, fc_write),
                             oracles (fwd_ref, l0_any, hf_greedy), the per-layer
                             debug probe patch, demo renderer
pulsar_arm/kernels, runtime  C reference path — parity oracle only, NOT shipped
pulsar_arm/tests/            parity tests (Pi-side, need torch + HF cache)
doc/                         demo GIFs and write-ups
```

## Disclosures

- The 1B answers multi-item and explanatory prompts (see demos); the 270m
  manages one-line factual answers and degrades beyond that — checkpoint size,
  not the engine, whose greedy path is HF-identical for both.
- The chat path feeds `<bos>` separately on the first turn, so it is not part
  of the printed `tpl:` list; in file mode `<bos>` is literal text in the
  prompt file, so it is.
- Sampling above temp 0.7 stays coherent; greedy is the reference behaviour
  and what all parity claims use.
