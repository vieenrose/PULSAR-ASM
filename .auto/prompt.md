# Autoresearch: LFM3-S2S-realtime (speech turn on picard)

## Objective
Reduce end-to-end speech-turn realtime factor s2s_rtf = generation_time / audio_seconds
on the gated 47-slot interleaved turn (speak_gate_q8, 2.80 s audio), Jetson Nano Gen1
(picard@100.84.151.116, sm_53, CUDA 10.2). Baseline 3.44, current ~1.95 (Q5-mix default).
Target RTF <= 1.0. Lossy backbone allowed (user-approved Q4/Q5, 2026-11-11): Q5-mix q5c
(Q5 linears, layers 21-23 + attention Q8) is the shipped default; Q4 tried and FATAL
(WER 0.345). Exact Q8 path kept via LFM_SPK_BB.

## Metrics
- **Primary**: `s2s_rtf` (lower is better)
- **Secondary**: `s2s_gen_s`, `s2s_load_s`, `s2s_wall_s`, `s2s_text_ms`, `s2s_audio_ms`, `s2s_q8_rtf`

## How to Run
`./.auto/measure.sh` (= .auto/lfm3/measure_s2s.sh) — two cold turns L (default Q5-mix,
metrics) + X (exact Q8). Outputs METRIC lines. ~4 min bench + ~14 min checks.

## Files in Scope
- `pulsar_arm/asm/lfm3_nanode.cu` + `lfm3_qlow.cuh` — backbone (k_qlow Q4/Q5 GEMV, TLb Q8)
- `pulsar_arm/asm/lfm3_depth.cu` + `lfm3_depth_neon.c` + `lfm3_q8g.cuh` — depth head (MaskGIT)
- `pulsar_arm/asm/lfm3_speak.c` — turn controller (slot schedule, server spawn, codec timing)
- `pulsar_arm/asm/lfm3_codec.cu` — streaming codec
- `pulsar_arm/asm/tools_qlow_bench.cu` — standalone k_qlow timing (no gates needed)
- `.auto/lfm3/gptq_lfm3.py`, `export_qlow.py`, `quant_loss.py`, `speech_calib.py` — quant pipeline
- `.auto/lfm3/s2s_tf.sh --gate`, `s2s_intel.sh` — lossy loss gates

## Off Limits
- HF references: `speak_gate_q8/`, `s2s_ref.wav.md5`, `s2s_ref_q5c.wav.md5` (re-pin only with reason)
- ipho-nano (pesi@100.70.245.36) RESERVED. Never ssh/build/run there.
- Never load whole 1.95 GB blob into RAM (run 281 took picard offline).
- No Python/HF at runtime. Build flags: nvcc -gencode arch=compute_53,code=sm_53 -O3 -fmad=false -Xptxas -O3,-dlcm=ca
- Stale-binary discipline: build to /tmp, require RC=0 && test -s before mv.
- Exact X turn must stay byte-identical (slots/text/wav). L turn: text == HF, wav == self-decode.

## Constraints
- `.auto/checks.sh` runs text gates + s2s_tf --gate (hidden cos min >=0.99, rel-L2 mean <=0.05)
  + s2s_intel (free-turn ASR WER <=0.05, only when speech/ASR sources changed).
- Board drifts: single samples +-3%; claims need same-job A/B or the harness pair (L vs X).
- tools_qlow_bench.cu iterations are bench-only (no gates); engine changes need full harness.

## What's Been Tried (S2S; S2T history in log runs 1-213)
- #245 depth T=1 tiled kernel k_q8t1 KEPT (head + depth_linear, exact, kernel 1.7x, turn ~-90ms).
- Q5 attention DECLINED offline (no board time): RTN-Q5 attention-only = text KL 0.0025 (q8-eng 0.0013)
  + speech rel-L2 0.044 (q8-eng 0.012); full-Q5 RTN = KL 0.0131 top-1 45/48 + speech 0.093.
  GPTQ+speech roughly halves RTN error (q5c precedent) -> ~0.007 KL, still worse than q5c's
  0.0049 with top-1 risk at the logit-gate floors. The ~300MB/step saving is not worth it.
- T=12 tiled trunk kernel k_q12t MEASURED OUT (reverted, no repo trace): bit-exact (maxdiff 0)
  and 2.4x faster than k_q8g<12,2> on w1 [2816,1024] in a standalone bench, but 2x SLOWER
  in-engine (ffn phases 331 vs 158 ms/turn, same-binary A/B; harness 2.01 vs 1.96). RR=1 for
  N>2048 also slower in-engine (ffn 261 vs 158). The trunk stays on k_q8g<12,2>/<12,1,2>.
- METHODOLOGY CORRECTION (same root cause as #42): bursty microbenches run below the boost
  clock and flatter compute-bound kernels -- k_q12t looked 2.4x faster stood alone, 2x slower
  in the sustained engine. In-engine same-binary phase A/B (LFM_DEPTH_TIME totals, or the
  harness pair) is the ONLY valid GEMV comparison. Bench ratios are valid only within one
  binary back-to-back AND at matched clocks; when bench and engine disagree, engine wins.
- #214 baseline 3.44 (150-slot cap fix + Q8 default). #215 codec early -13.9%. #216 depth KB=2 -4%.
- #217 depth rms rows2. #220 tmpfs scratch. #222 iSTFT table. #223 codec OpenMP. #226 graphs repaired but slower (reverted default off).
- #227 skip lm_head on audio slots -7%. #228 depth head-select -2%. #229 depth last-layer row-select -2.9%.
- #231 depth warmup. #233 codec at EOS. #234 iSTFT skew. #236 codec FFN on device. #237 codec conv on device. #238 codec qkv one upload.
- #239 CORRECTNESS: codec got frame-major codes, fixed to codebook-major; streaming codec neutral.
- #240 Q4/Q5 infra (k_qlow, GPTQ, export, s2s_tf, s2s_intel). #241 Q5-mix q5c default (RTF 1.99), Q4 fatal later.
- #242 Q4 q4a FATAL (p3 WER 1.0, TF rel-L2 0.0655): Q4 out.
- #243 k_qlow PD2/MINB4 (+5-7% per shape): RTF 1.95.
- Depth nvprof (2026-11-11): k_q8g<12,2,1> 549ms/1089calls, <1,1,2> 204ms (head 4.5ms/call @5.6GB/s, 32 barrier rounds), <4,2,1> 167ms, <12,1,2> 101ms, attn_gate 82ms, HtoD 49ms, silu 28ms.
- Backbone nvprof: k_qlow 3.75s/5950, HtoD 1.39s/492 (!), TLb 1.27s/3990. HtoD 492x2.8ms is the next look (streaming?).
- Open: depth <12,2> wastes 2 warps/block (N/group 1024/2816 vs 128-thread blocks) -> split grid-y per group or fuse q|k|v/gate; depth-in-nanode merge (~6ms/frame IPC); qlow per-shape tuning; HtoD batching.
