# PULSAR-ASM — pure-asm gemma-3-270m engine (Pi 4)

SVC-only AArch64 inference: no libc, static link, direct syscalls.
Tokenizer, sampler, and chat live in `pulsar_arm/asm/core.S`.

## Current numbers (2026-10-06, Pi 4, 3-core target)

- Bench (greedy, 32 tok): **~143–150 ms/token** (baseline 158.83 → best 138.51 C-port, asm 143).
- Proven wall: numpy streaming ceiling 3.93 GB/s; driver moves 536 MB/token
  at 3.84 GB/s = **98% of ceiling**. Floor ≈ 136 ms/token here. Loop converged.
- Pi drifts ~5–18% across sessions — compare same-job A/B ratios, not absolutes.

## Build & run

```
cd pulsar_arm/asm && as -o core.o core.S && ld -static -o core core.o
./core <model.safetensors> <vocab.bin> [tok] [mode] [bpe.bin] [temp_milli] [topp_milli] [gencap]
```

- `tools/mkvocab.py`: tokenizer.json → vocab.bin (surfaces + byte fallback).
- `tools/mkbpe.py`: tokenizer.json → bpe.bin (514,906 rules, right-major for
  single-u64 binary search + 19,227 single chars + 256 byte fallbacks).
- Modes: default greedy bench · `1` sample bench · `c` chat REPL (needs bpe).
- Chat: `... 2 c bpe.bin [temp] [topp] [gencap]` (temp default 1000, min 50;
  topp default 950; gencap default 128, max 400). Template (SOT/EOT roles)
  verified id-exact vs HF. Sampling guidance (measured): prefer temp ≥ 0.7
  (0.5 degenerates into token loops, distinct4 0.09 vs 1.00); top-p neutral
  (0.5/0.95/1.0 all diverse); gen-cap bounds output cleanly.

## What was proven (26 experiments)

- Memory-wall bound: 3-thread GEMV +4.2% (twice, same-job A/B). Single thread saturates bus.
- Bit-identical wins kept: -O3/cortex-a72, fused elementwise, C layer_step, tables.
- asm GEMV at C parity (3.88 GB/s) via single-insn SHLL widening.
- BPE encode 18/18 exact vs HF (char init, rank-order merges, leftmost ties).
- **HF-exact decode**: greedy output is identical to transformers (fp32 +
  eager) token-for-token — 32/32 bench tokens and complete chat transcripts —
  and the residual matches an fp64 oracle to ~1e-7 at every layer and every
  position. (The mismatch this replaced: `fwd_token` read layer 0's `q_norm`
  for every layer — invisible at position 0, wrong from position 1 on.)
- Robustness proven: empty/whitespace/CRLF input, 203-token prefill, 8-turn
  depth with context-reset recovery, EOF/exit paths, gen-cap boundary.
- Deterministic (fixed seed): greedy, sampled, and full chat transcripts all
  reproduce bit-for-bit across runs (proven by diff, runs #38-40).

## Demos (270m-it-qat-q4_0 chat, temp 1.0, top-p 0.95, cap 48)

Every frame is a real run on the Pi: prompt, `tpl:` ids and response are the
engine's own bytes, and the status line carries that session's measured rate.
Lines marked `#` are labels (the engine never prints them). Only pacing is
libertied (fixed-cadence reveal, as in pulsar_asm's `make_demo_gif.py`).

![270m chat demo](doc/gemma3-270m-chat-en.gif)
![270m chat demo, Traditional Chinese](doc/gemma3-270m-chat-zh-tw.gif)

```
printf 'Explain gravity in two sentences for a child.\n' | ./core model.safetensors vocab.bin 2 c bpe.bin 1000 950 48
printf '請用一句話解釋什麼是量子力學。\n'                 | ./core model.safetensors vocab.bin 2 c bpe.bin 1000 950 48
```

FunctionGemma-270m-it turns both an English and a Traditional-Chinese question
into the same tool call (file mode, greedy, cap 16 — the same schema in both
prompt files, only the user turn changes):

![FunctionGemma tool call, English prompt](doc/functiongemma-toolcall-en.gif)
![FunctionGemma tool call, zh-TW prompt](doc/functiongemma-toolcall-zh-tw.gif)

```
python3 pulsar_arm/tools/fc_write.py    # -> /tmp/tool_official.txt, /tmp/tool_official_zhtw.txt
./core functiongemma.safetensors fcvocab.bin 2 f fcbpe.bin 1000 950 16 < /tmp/tool_official.txt
./core functiongemma.safetensors fcvocab.bin 2 f fcbpe.bin 1000 950 16 < /tmp/tool_official_zhtw.txt
```

Re-render all four: `python3 pulsar_arm/tools/make_chat_gif.py` (PIL + ffmpeg).
Sampling varies run to run (fixed seed reproduces exactly); temp < 0.7 tends to
loop — see guidance above. Prefill ids print as `tpl:` for transparency.

## Layout
`core.S`: file/mmap stage → compute kernels → layer forward → sampler →
BPE + chat → bench/chat drivers. Build-time converters in `tools/`.
