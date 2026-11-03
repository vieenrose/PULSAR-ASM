# Ideas backlog — RE-USE RTF

Dev host best: rtf_total **0.0885** (rtf_model 0.0845 | enc 0.078 | tower 0.318 | dec 0.032 | heads 0.001 | stft 0.002 | istft 0.002).
Nano (deployment target, tegra210, CUDA_LAUNCH_BLOCKING=1, REUSE_CPU_STFT=1):
RTF_model **3.1346** @ 8 kHz (enc 3.285 | tower 11.566 | dec 1.113 | heads 0.011), **6.1243** @ 16 kHz. Cumulative 44.34 -> 3.13 = **14.1x**.

Per-forward kernel table (nsys -> sqlite, /tmp/p59.sqlite pattern; query the sqlite with
python3, the nsys report scripts are missing in this install):
  in_conv (fused in_proj+causal conv) 141 ms  60 launches  — CLOSED (acc-shape sweep #51, LBV=3 #45, KC #18)
  convs (r8 64-acc shapes)            ~110 ms             — CLOSED (issue-bound: #48 skew +50-74%, #21 float4, #30, #39)
  scan (fused dt_proj+scan)           104 ms  60 launches  — CLOSED (#25 prefetch, #32 unroll, #34 fusion, #38 shared-decay, #46 LB)
  out_proj_r8c (8x8, #50+#57)          76 ms  60 launches  — CLOSED (K-split #41, warps #41, LB #46)
  x_proj (register-tiled)              36 ms               — at ~4 TFLOPS, peak
  inorm_prelu                          10.5 ms 18 launches — CLOSED (#54: block width and 1-pass/3-pass both neutral)
  outproj_ln_g (4x4 + LN)               9.6 ms 30 launches — CLOSED (#59: 8x8 neutral; launch-quantisation bound)
  sp_up8t (tiled, #52)                  6.3 ms  2 launches — OPEN (shape sweeps only)
  resid scatters + gather              ~2 ms               — mostly gone (#57 fused out_proj, #58 tower fusion)
  encoder skip concays/copies          ~1.6 ms             — OPEN (idea #3)
  heads (3x conv1x1, atan2, crop, sweep) ~1 ms 7 launches  — OPEN, tiny

## OPEN items
1. **Nano 16 kHz uses the CPU iSTFT: 27.2 s = RTF 5.34 on top of model 6.12** (RTF_total 11.52).
   At 8 kHz the GPU iSTFT is used (0.017 s). Repro: `CUDA_LAUNCH_BLOCKING=1 REUSE_WS_MB=1280
   REUSE_CPU_STFT=1 ./engine/reuse_nano_gpu ... mic_test2_16000.wav` prints "iSTFT CPU".
   Find why the GPU path is skipped at nfft=640 (table alloc? workspace? silent fallback?) —
   halves the Nano's 16 kHz RTF. Zero dev-host risk (metric unchanged), high deployment value.
2. **sp_up8t shape sweep** (6.3 ms, 2 launches): #52 tested RW/RC/TW/NC on the pre-#55 kernel;
   the winner 4,8,32,64 tied with 8,8,32,64 and NC=128. Floor = write-bound ~2.5 ms => -2 ms = -0.4%.
3. **Encoder per-dense-layer copies** (~1.6 ms + ~250 MB of Nano arena): dense_block_gpu does
   10 cudaMemcpyAsync per forward (skip=x_in, 2 per layer, out=cur). `skip = x_in` and `out = cur`
   are removable; the per-layer [cur, skip] concat needs contiguity, so removing it means the conv
   reads two source pointers (per-ci select: ci < C1 ? x1 : x2 + (ci-C1)*H*W).
4. **Heads**: 3 conv1x1_to1 launches share inputs (dm/dp) — one batched kernel would take 7
   launches to 3. ~0.05 ms on the dev host, more on the Nano (launch-bound under CUDA_LAUNCH_BLOCKING).
5. Nano: REUSE_CPU_STFT=1 is the right setting (GPU STFT costs 0.100 s vs 0.091 s, Maxwell fp64
   is 1/64 rate). Dev host: GPU STFT is default (0.0020 s, and it killed the metric's noise).

## CLOSED chapters — do not retry (measured, numbers in the experiment log)
- in_conv (141 ms): 5x4 accumulators OPTIMAL; 4x8 +8%, 8x8 +23%, 256-row tile +48%; latency-bound.
- convs (~110 ms): issue-bound; bank-conflict skew +50-74% (#48), float4 shared neutral (#21).
- out_proj (76 ms): 8x8 = -4% (#50, taken). K-split, warps, LB, scan-fusion all neutral/worse.
- scan (104 ms): 5 levers refuted. L=256 serial chain at its floor.
- InstanceNorm (10.5 ms): block width and pass-count neutral (#54).
- outproj_ln (9.6 ms): 8x8 neutral (#59) — launch/quantisation bound, 81 blocks of 256.
- 2 CUDA streams for the independent mask/phase decoder branches: NEUTRAL and it races (the
  decoder arena resets to offset 0 and reuses the tower region where `enc` lives) (#56).
- STFT/iSTFT/heads = 0.005 s = 5.6% of rtf_total, all at their floors on the dev host.
- CPU STFT vectorisation (#43): 38% faster standalone, invisible in the metric (CPU noise 4x).
  Retry only on a quiet host, and it is a real Nano win (STFT = 2.7% of Nano RTF there).

## Deployment (Nano) rules learned
- Kernels with >=64 per-thread registers COMPILE for sm_53 and FAIL AT LAUNCH ("too many
  resources requested"); checks.sh cannot see it. Guard with gpu_is_maxwell() (honours
  REUSE_FORCE_MAXWELL_TILES=1). Guarded today: conv tile codes 644/645/646 (#47), REUSE_OP 3->2 (#53).
- Memory-pattern wins transfer ~1:1 (transpose tiling -5% decoders, tower fusion -2.7% tower,
  strided conv -4.3% encoder); FMA-ratio/accumulator wins do NOT (latency-bound at 16 warps/SM).
- Fast Nano A/B: same input + fed-reference protocol + REUSE_DUMP_AUDIO, diff waveforms
  (140 dB = bit-compatible). Binary ./engine/reuse_nano_gpu, blob tests/golden/reuse.blob,
  REUSE_WS_MB=640 (8k) / 1280 (16k), reps >= 2 (rep 0 is warm-up). Disk ~450 MB free, 3 GiB RAM.
