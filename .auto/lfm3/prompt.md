# Autoresearch: LFM3-Audio-2B speech-to-text latency on Jetson Nano

## Objective
Reduce the end-to-end speech-to-text latency (ASR_RT = wall / audio_seconds) of the
ported LFM3-Audio-2B engine on the Jetson Nano Gen1, from the current ~1.33 best case
to <= 1.00 on >= 60 s audio. The functional port is complete (M1/M2/M3 all gated);
this loop optimizes the runtime, not the numerics.

## Metrics
- **Primary**: `asr_wall_s` (seconds, lower is better) — wall time for the full
  speech->text chain on the committed 60 s fixture, OR equivalently `asr_rt`
  (ratio, lower is better, target <= 1.00).
- **Secondary**: `asr_rt`, `decode_ms_per_tok`, `peak_rss_kb`, `emb_cos` (vs HF).

## How to Run
`./.auto/lfm3/measure.sh` — outputs `METRIC name=number` lines. ~10 min per run.

## Files in Scope
- `pulsar_arm/asm/lfm3_asr.c` — M1 shell (WAV -> text), argv-only execv of stages
- `pulsar_arm/asm/lfm3_nanode.cu` — backbone engine (--serve resident mode)
- `pulsar_arm/asm/lfm3_enc_gpu.cu` — audio encoder tower
- `pulsar_arm/asm/lfm3_mel.c` / `lfm3_mel.S` — log-mel front end
- `pulsar_arm/asm/lfm3_wav.c` — WAV reader
- `pulsar_arm/asm/lfm3_tok.c` — tokenizer
- `pulsar_arm/asm/lfm3_depth.cu` + `lfm3_depth_neon.c` — depth head (per-frame reload)
- `pulsar_arm/asm/lfm3_codec.cu` — codec (not on the ASR path)

## Off Limits
- The HF golden (`.auto/lfm3/speech/hf_asr_golden.json`) and the embedding matrix.
- The three accuracy gates in `.auto/lfm3/checks_asr.py`.
- Numerical results: any change must keep `emb_cos >= 0.85` and the greedy prefix
  matching HF.

## Constraints
- Board: picard@100.84.151.116 (Nano Gen1, sm_53, CUDA 10.2). No root clocks.
- Stale-binary discipline: build to /tmp, require RC=0 && test -s before mv.
- No Python/HF/llama.cpp at runtime.
- Build flags: nvcc -gencode arch=compute_53,code=sm_53 -O3 -fmad=false -Xptxas -O3,-dlcm=ca
- Every perf claim needs same-job A/B (ab_pair.sh) or cold-harness numbers.

## What's Been Tried
- Batched backbone prefill (S8): REFUTED — 4x slower at ASR shapes, flag-gated, not shipped.
- Resident backbone (--serve): kept (default), 4.85x speedup on the speech turn.
- Depth NEON GEMM: 13.4s -> 6.5s per forward, bit-identical.
- Codec batched GEMMs + iSTFT hoisting: 800s -> 95.9s for T=75.
- Open levers for realtime: S7 chunked input, resident depth server, faster load/prefill.

## Open Ideas
- S7: chunked live input with ring buffers (mel frames as they arrive, no full-clip prefill).
- Resident depth: make lfm3_depth a --serve-style server to eliminate 36 x 1.3s reloads.
- Backbone load: faster blob parsing / arena strategy (87s one-time cost).
- Front-end: overlap mel computation with encoder.
