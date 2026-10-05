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
./core <model.safetensors> <vocab.bin> [tok] [mode] [bpe.bin] [temp_milli] [topp_milli]
```

- `tools/mkvocab.py`: tokenizer.json → vocab.bin (surfaces + byte fallback).
- `tools/mkbpe.py`: tokenizer.json → bpe.bin (514,906 rules, right-major for
  single-u64 binary search + 19,227 single chars + 256 byte fallbacks).
- Modes: default greedy bench · `1` sample bench · `c` chat REPL (needs bpe).
- Chat: `... 2 c bpe.bin [temp] [topp]` (temp milli, default 1000; topp default 950).
  Template (SOT/EOT roles) verified id-exact vs HF; prefer temp ≥ 0.7
  (0.5 degenerates into token loops, measured distinct4 0.09 vs 1.00).

## What was proven (26 experiments)

- Memory-wall bound: 3-thread GEMV +4.2% (twice, same-job A/B). Single thread saturates bus.
- Bit-identical wins kept: -O3/cortex-a72, fused elementwise, C layer_step, tables.
- asm GEMV at C parity (3.88 GB/s) via single-insn SHLL widening.
- BPE encode 18/18 exact vs HF (char init, rank-order merges, leftmost ties).
- Deterministic (fixed seed): greedy runs reproduce id-for-id.

## Layout
`core.S`: file/mmap stage → compute kernels → layer forward → sampler →
BPE + chat → bench/chat drivers. Build-time converters in `tools/`.
