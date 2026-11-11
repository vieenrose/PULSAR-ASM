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
- MaskGIT 3->2 passes: VIABLE (teacher-forced), #249's kill RETRACTED. #249's host WAVs were
  written from the codec's float output WITHOUT the x32767 scale (values 0/1 = near silence), so
  every host-decoded transcript was ASR hallucination ('well i think that's a good start' for
  very different code streams -- identical transcripts of different audio was the tell). With
  lfm3_wavout's scaling: gated fixture HF 3-pass and Q8 2-pass both say 'Hello? How can I assist
  you today?'; held-out free turns p1-p4 (q5c backbone, engine's own hidden states and orders):
  host Q8 3-pass == engine frames 239/239, 2-pass (6+6, same orders) codes 62-68% equal to
  3-pass, same EOS frame on all 4, ASR WER 0/47 for both. Not yet covered: the feedback loop
  (2-pass codes fed back into the backbone) -- needs the engine-side 2-pass through s2s_intel.
  RULE: any host-side audio check must reproduce the engine's own transcript on a control arm
  before its other arms are read.
- MaskGIT 2-pass ENGINE measurement (LFM_DEPTH_MGK=6, code kept as .auto/attic/depth_maskgit2.patch,
  default 4): bit-identical to the host 2-pass reference on the engine's own hidden states +
  orders (40/40 frames, p1), and depth GPU cost drops as predicted: 36.6 -> 26.6 ms/frame
  (-10 ms, p1/p2/p3 all agree), audio slot 110.4 -> 100.4 ms (-9%). QUALITY COST: held-out free
  turns p1-p4, 2-pass WER 1/47 words vs 3-pass 0/47 (p1 'One' -> 'Men', and p1 ends after 27
  frames vs 40), p2/p3/p4 transcripts identical. Worth ~-8% RTF on the gated turn (gen -0.35 s).
  NOT shipped: it is lossy, it changes the turn (frame count/WAV), so the X byte gates and the
  L pinned md5 cannot hold -- needs the same user decision as the 2026-11-11 Q4/Q5 approval,
  plus a re-derived L md5 and X split. Do not flip the default without that call.
- Depth has NO in-engine penalty (closes the merge idea): warm two-point slope of the depth
  server standalone (100 vs 300 frames in one process, load cancels) = 36.4 ms/frame == the
  in-engine 37-39 ms/frame from speak's STIME. Codec contention is 1.4 ms/frame (stream on/off,
  4 turns alternating). Per maskgit pass the depth is trunk 5.5 ms (45%, ~15 GB/s) + head 1.07
  + attn_gate 0.9 + silu 0.31 + qknorm 0.22, i.e. ~36 measured vs a ~17-20 floor. The trunk is
  the only material item and it is the shape #246 failed on (k_q12t, 2x slower in-engine) --
  any retry needs an in-engine phase A/B first, never a bench.
- Board hygiene: root nvprof leaks GPU mappings on picard (a 1728 MB Q8 arena cudaMalloc fails
  with nvgpu 'Failed to pin dma_buf' even after every CUDA process exits; only a reboot clears
  it). Use speak's in-engine STIME / LFM_DEPTH_TIME totals instead, and if nvprof is unavoidable
  expect to reboot afterwards. /usr/bin/time is absent on the board (use python3 time).
- fp16 depth trunk DECLINED (host range audit, no build): per-call |x| peaks (324 calls/family)
  qkv/g median 16.3 with 55% of calls > 16, w2 median 18.0 with 55% > 16 (max 48.3) -- a per-call
  fp16-partial guard (budget 65504/(32*127) = 16.1) would fall back on most calls, keeping fp16
  only for o/w1/w3 (~40% of trunk traffic, prize ~-0.03). Per-block exact rescale (H2=2 style)
  costs ~30-40% scan+scale overhead on an issue-bound kernel, netting ~zero (#153 precedent).
  The full-fp16 proxy was codes-exact 36/36 but used fp32 accumulation (PyTorch CPU half matmuls),
  so it did not cover HFMA2 fp16-accum rounding -- and the range data says the guard would eat
  the win anyway. The trunk stays Q8; fewer MACs (fewer passes) is the only direction and it
  is dead (previous row).
- #245 depth T=1 tiled kernel k_q8t1 KEPT (head + depth_linear, exact, kernel 1.7x, turn ~-90ms).
- Depth attn_gate fp32 expf (fw_expf port) MEASURED OUT (reverted, no repo trace): ported the
  certified fp32 glibc expf (fwd35 row 67) to k_attn_gate's softmax (inputs <= 0 by construction;
  sigmoid keeps gm_expf, out of cert). Exhaustive cert on the Nano: 0 mismatches over all 2^32
  inputs. But the win is below resolution: warmed kernel 3.22 vs 3.36 ms/frame (one sample each
  way, first sample's 4.04 was a cold outlier), turn-level 4v4 means 4.895 vs 4.878 (0.35%, pairs
  split). attn_gate's gm_expf is only ~10-15% of the kernel; Amdahl kills the swap. Lesson: even
  a 3x-faster primitive is worthless below ~15% kernel share on this board's noise floor.
- Q5 attention DECLINED offline (no board time): RTN-Q5 attention-only = text KL 0.0025 (q8-eng 0.0013)
  + speech rel-L2 0.044 (q8-eng 0.012); full-Q5 RTN = KL 0.0131 top-1 45/48 + speech 0.093.
  GPTQ+speech roughly halves RTN error (q5c precedent) -> ~0.007 KL, still worse than q5c's
  0.0049 with top-1 risk at the logit-gate floors. The saving is ~46 MB/step (attention is 130 MB at Q8; Q5 = 5.5/8.5 of that), ~2-3 ms/slot: not worth it.
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
