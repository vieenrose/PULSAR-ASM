# PULSAR-ASM

**Pure-assembly Gemma-3-270m inference on a Raspberry Pi 4.** Static AArch64
binary, direct `svc` syscalls, no libc, no CRT, no third-party code. The
tokenizer, sampler, chat REPL and the whole forward pass live in one file:
`pulsar_arm/asm/core.S`.

Two checkpoints run unmodified: `google/gemma-3-270m-it-qat-q4_0-unquantized`
and `google/functiongemma-270m-it` (same architecture and tensor order).

## At a glance

| | |
|---|---|
| Target | Raspberry Pi 4, Cortex-A72 / NEON only (no SVE, no dotprod), 3 cores |
| Engine | one static binary: `as` + `ld`, direct syscalls, zero dependencies |
| Decode | **~143–150 ms/token** (greedy, 32 tokens) |
| Bandwidth | 3.84 GB/s of a measured 3.93 GB/s streaming ceiling → **98 %** |
| Parity | greedy output identical to transformers (fp32 + eager), token for token |
| Determinism | fixed seed ⇒ byte-identical transcripts across runs |

## Quick start

```sh
# 1. build the engine
cd pulsar_arm/asm
as -o core.o core.S && ld -static -o core core.o

# 2. tokenizer blobs, once per checkpoint (reads tokenizer.json)
python3 ../tools/mkvocab.py <tokenizer.json> vocab.bin   # surfaces + byte fallback
python3 ../tools/mkbpe.py   <tokenizer.json> bpe.bin     # 514,906 merge rules

# 3. greedy bench: 8-token prompt, 32 decode steps, ids + ms/token
./core <model.safetensors> vocab.bin

# 4. chat REPL (needs bpe.bin)
./core <model.safetensors> vocab.bin 2 c bpe.bin 1000 950 48

# 5. file mode: raw prompt on stdin, greedy, one shot (needs bpe.bin)
./core <model.safetensors> vocab.bin 2 f bpe.bin 1000 950 16 < prompt.txt
```

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

Measured sampling guidance: keep temp ≥ 0.7 (0.5 degenerates into token
loops), top-p is neutral here (0.5 / 0.95 / 1.0 all stay diverse), and gencap
bounds output cleanly. The chat template (SOT/EOT roles) is verified id-exact
against HF, and every run prints its prefill ids as `tpl:` for transparency.

## Demos

Real runs, nothing re-typed: prompt, `tpl:` ids and response are the engine's
own bytes, the status bar carries that session's measured rate, and the amber
`>` line is the user turn (in the FunctionGemma clips it is the user message
inside the prompt file — file mode does not echo it, so the status bar names
the file). Only pacing is libertied.

![270m chat demo](doc/gemma3-270m-chat-en.gif)
![270m chat demo, Traditional Chinese](doc/gemma3-270m-chat-zh-tw.gif)

```sh
printf 'Explain gravity in two sentences for a child.\n' | ./core model.safetensors vocab.bin 2 c bpe.bin 1000 950 48
printf '請用一句話解釋什麼是量子力學。\n'                 | ./core model.safetensors vocab.bin 2 c bpe.bin 1000 950 48
```

FunctionGemma turns both an English and a Traditional-Chinese question into
the same tool call — one schema, only the user turn differs:

![FunctionGemma tool call, English prompt](doc/functiongemma-toolcall-en.gif)
![FunctionGemma tool call, zh-TW prompt](doc/functiongemma-toolcall-zh-tw.gif)

```sh
python3 pulsar_arm/tools/fc_write.py   # -> /tmp/tool_official.txt, /tmp/tool_official_zhtw.txt
./core functiongemma.safetensors fcvocab.bin 2 f fcbpe.bin 1000 950 16 < /tmp/tool_official.txt
./core functiongemma.safetensors fcvocab.bin 2 f fcbpe.bin 1000 950 16 < /tmp/tool_official_zhtw.txt
```

`fcvocab.bin` / `fcbpe.bin` are built from the FunctionGemma `tokenizer.json`
with the same two tools as above.

Re-render any subset with `python3 pulsar_arm/tools/make_chat_gif.py`
(needs PIL + ffmpeg); the transcripts are literals in that file, so a re-run
can only reproduce these frames, never invent them.

## Verified gates

- **HF-exact decode.** Greedy output is identical to transformers (fp32 +
  eager) token for token: 32/32 bench tokens and complete chat transcripts,
  and the residual matches an fp64 oracle to ~1e-7 at every layer and every
  position. (What this replaced: `fwd_token` indexed `q_norm` with a constant
  layer-0 slot — invisible at position 0, wrong from position 1 on.)
- **BPE encode** 18/18 exact vs HF (char-level init, rank-order merges,
  leftmost tie-break, byte fallback).
- **Determinism.** Greedy, sampled and full chat transcripts reproduce
  bit-for-bit across runs at a fixed seed.
- **Robustness.** Empty / whitespace / CRLF input, 203-token prefill, 8-turn
  depth with automatic context reset, EOF and exit paths, gen-cap boundary.
- **asm at C parity.** GEMV reaches 3.88 GB/s using a single SHLL
  instruction for bf16 widening; the bit-identical wins the C port proved
  (fused elementwise passes, one layer body per call, precomputed tables) are
  present in the asm path too.

## Performance

- **The wall.** A numpy streaming-sum ceiling of 3.93 GB/s was measured on
  this Pi; the driver moves 536 MB/token at 3.84 GB/s, i.e. 98 % of it. The
  floor is ≈ 136 ms/token here, so the loop has converged.
- **Multi-core is not the answer.** A 3-thread static-partition GEMV is 4.2 %
  *slower* (same-job A/B, twice): the bus is already saturated.
- **Thermal drift.** Absolute numbers move 5–18 % between sessions; compare
  same-job A/B ratios, never absolutes.

## Layout

```
pulsar_arm/asm/core.S        the engine: mmap stage → kernels → layer forward
                             → sampler → BPE → chat / bench / file drivers
pulsar_arm/tools/            build-time converters (mkvocab, mkbpe, fc_write),
                             HF oracles (fwd_ref, ref_layer0), demo renderer
pulsar_arm/kernels, runtime  C reference path — parity oracle only, NOT shipped
pulsar_arm/tests/            parity tests (Pi-side, need torch + HF cache)
doc/                         demo GIFs and write-ups
```

## Disclosures

- The 270m answers simple factual and instructional prompts (see demos);
  harder requests degrade — that is checkpoint size, not the engine, whose
  greedy path is HF-identical.
- The chat path feeds `<bos>` separately on the first turn, so it is not part
  of the printed `tpl:` list; in file mode `<bos>` is literal text in the
  prompt file, so it is.
- Sampling above temp 0.7 stays coherent; greedy is the reference behaviour
  and what all parity claims use.
