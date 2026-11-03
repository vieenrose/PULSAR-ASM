# Ideas backlog — RE-USE RTF

Dev host best: rtf_total 0.0917 (rtf_model 0.0877, enc 0.082 | tower 0.328 | dec 0.035 | heads 0.001).
Nano (deployment target): RTF_model 3.2384 @ 8 kHz, 6.3285 @ 16 kHz = 13.7x vs baseline.

Per-forward kernel table (nsys→sqlite, /tmp/p50.sqlite pattern; nsys report scripts
are missing in this install, query the sqlite directly with python3):
  in_conv (fused in_proj+causal conv) 140 ms  60 launches  — closed
  scan (fused dt_proj+scan)           103 ms  60 launches  — closed
  out_proj (8x8, #50)                  74 ms  60 launches  — closed
  convs (r8, 64 acc)                   72 ms               — closed
  x_proj                               35 ms               — 4 TFLOPS, near peak
  strided 1x3 patchify conv (naive)   4.7 ms   1 launch    — OPEN, see #1
  inorm_prelu                          10.5 ms 18 launches — at floor (#54)
  concat_f                             10.5 ms 60 launches — 91 GB/s, ~1.7x off
  outproj_ln_g                         9.6 ms  60 launches — 4x4, occupancy swept (#46)
  sp_up (tiled, #52)                    9.3→~6 ms
  resid scatters (tiled)                9.6 ms
  k_transpose                           4.0 ms  12 launches

## OPEN items (all small now)
1. **Strided 1x3 patchify conv** (encoder, C_in=64, H=1023,W=163 → 256x81, sH=4 sW=2,
   kH=1 kW=3, padW=0): naive k_conv2d, 4.7 ms for 0.5 GFlop = 109 GFLOPS. Each thread
   does 192 FMAs against 192 dependent scalar loads (1 FMA/load) → LSU/issue bound.
   TRIED AND NEUTRAL: 8/16/32 co per thread (weights become 96 broadcast loads per 96
   FMAs), and hoisting the per-tap bounds checks into an interior fast path.
   TO DO: full register+shared tiling — stage the (TR rows x 2TC+2 w) input strip and the
   PCOx3 weights per ci in shared; per thread RW output cols x PCO co with float4 window
   loads (target >=4 FMA/load). Floor = 42 MB read + 5 MB write = 0.3 ms; PCO=8 realistic
   ~2 ms => -2.5..-4 ms = -0.5% of rtf_total. Gate: bit-identical if the tap order per
   output is preserved.
2. **Decoder mag/phase branches on 2 CUDA streams** (~0.3%): reuse_gpu_decoder_pre
   resets the arena at entry (g->ws_off = 0), so the two branches must be called with the
   reset hoisted out (allocate both branches' temporaries in one arena pass). Needs a
   cudaStream_t threaded through decoder_pre -> launch_conv2d/inorm/sp_up (the #26 pattern).
   dec_s is 0.035 s, so the ceiling is small; do it last.
3. **Encoder skip-concat copies** (~0.2% + ~250 MB of Nano arena): dense_block_gpu does
   10 cudaMemcpyAsync per forward (skip=x_in, 2 per layer, out=cur). `skip = x_in` and
   `out = cur` are removable (init skip=x_in; last inorm writes out directly); the
   per-layer `[cur, skip]` concat needs contiguity, so removing it requires the conv to
   read two source pointers (a per-ci pointer select: ci < C1 ? x1 : x2 + (ci-C1)*H*W).
4. **concat_f** 10.5 ms at 91 GB/s (58% of the 158 GB/s ceiling): tile-shape sweeps only.
5. **k_transpose** 4.0 ms / 12 launches in the decoders: tile it like #36's gather.

## CLOSED chapters — do not retry (measured, with numbers in the experiment log)
- in_conv (140 ms): accumulator-shape sweep 5x4 (2.2 FMA/load) is OPTIMAL; 4x8 +8%,
  8x8 +23%, 256-row tile +48%; launch_bounds (256,3) taken in #45; KC sweep done (#18).
  It is latency-bound, not LSU-bound (the opposite of out_proj).
- convs (72 ms, r8 64-acc): issue-bound. Bank-conflict skew +50-74% (#48), float4 shared
  loads neutral (#21), hoisted loads neutral (#30), launch_bounds spills (#39/#46),
  Winograd adds instructions to an issue-bound kernel — skip.
- out_proj (74 ms): 4x4 -> 8x8 gave -4% (#50). K-split (#41), more warps (#41),
  fusion into the scan with atomics (#34), launch_bounds (#46) all neutral/worse.
- scan (103 ms): prefetch (#25), unroll sweep (#32), fused out_proj (#34), launch_bounds
  (#46), register->shared decay (#38 taken). L=256 serial chain at its floor.
- InstanceNorm: thread count 128/256/512/1024 neutral, and fusing sum+sumsq into one
  read pass is neutral (the 2nd read is L2-served), so the 3-pass structure costs nothing.
- STFT/iSTFT/heads: 0.002+0.002+0.001 s, i.e. 0.4% of the metric. GPU STFT is
  bit-identical on sm_53 and sm_61; on the Nano use REUSE_CPU_STFT=1 (0.091 vs 0.100 s).

## Deployment (Nano) rules learned
- Any kernel whose per-thread register demand rises must be Nano-validated: sm_53
  COMPILES 64-accumulator kernels and fails at LAUNCH ("too many resources requested").
  checks.sh cannot see that. Use gpu_is_maxwell() (honours REUSE_FORCE_MAXWELL_TILES=1).
- Fastest strong Nano check: run both machines with the fed-reference protocol +
  REUSE_DUMP_AUDIO and diff the waveforms (140 dB = bit-compatible). Binary is
  ./engine/reuse_nano_gpu, blob tests/golden/reuse.blob, needs REUSE_WS_MB=640 (8k) /
  1280 (16k); reps must be >=2 or "model best" prints 1e30 (rep 0 is warm-up).
- Memory-pattern wins transfer ~1:1 to the Nano; FMA-ratio/accumulator wins do not
  (Maxwell is latency-bound at 16 warps/SM). Score the dev host, deploy with the guards.
