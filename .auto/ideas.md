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
1. CLOSED (#67, non-issue): the Nano's 16 kHz "CPU iSTFT 27.2 s" was an artifact of a manual
   invocation without REUSE_GPU_ISTFT - reuse_gpu_istft is opt-in via getenv and the deployment
   script run_bench.sh has exported it all along (8 kHz 3.1500, 16 kHz 6.2304 RTF_total, istft(gpu)
   0.35 s = RTF 0.069). Verified the GPU iSTFT at 16 kHz on Maxwell against the CPU reference:
   max|d| 2.24e-08, rel rms 9.5e-08, SNR 140.5 dB (the atomic floor). run_bench.sh now also exports
   REUSE_CPU_STFT=1 (the Nano's faster STFT path, #47: 0.091 s vs 0.100 s).
2. CLOSED (#64, sweep + a trap): the extra shapes that looked like decoders 0.031 -> 0.025 s
   (4,8,48,64 / 4,8,60,64 / 12,8,32,64 / 4,12,32,64 / 8,12,32,64 / 12,12,32,64 ...) were all
   REUSE_SPUP_T values with NO SPT() instantiation, i.e. the sub-pixel kernel simply never launched
   - the 0.025 s is the decoder chain without the sub-pixel conv, not a faster shape. Verified by
   the exact partition: every config that IS in the SPT list measures 0.031-0.035, every one that is
   not measures 0.025. Among real kernels the shipped 4,8,32,64 is the optimum (8,8,32,64 0.035,
   4,8,64,64 0.033, 4,16,32,64 0.034, 4,8,32,128 0.031 tie). sp_up's real cost is 0.031-0.025 = 6 ms
   with a ~2.5 ms write floor; the shape lever is exhausted.
   LESSON for every future sweep here: a dispatch-miss means "no work", not "fast" - check the
   dispatch table covers the swept tuple before believing a winner (same trap as #51's +41%).
1b. **Encoder per-dense-layer concat, REMOVED CODE EXISTS (reverted #61)**: dense_block_gpu's
   11 [cur,skip] cudaMemcpyAsync per forward (444 MB at 8 kHz) are removable: layer i's conv input
   is the concat [cur_{i-1},...,cur_0, x_in] = i+1 groups of exactly HID channels, so give
   k_conv2d_r/r2/r8/r9 two bases `(x, x2, ng)` and per channel group select
   `xin = (ci/HID < ng) ? x + (ng-1-ci/HID)*H*W : x2 + (ci/HID-ng)*H*W`; dense_block writes layer
   i's inorm output to slot i of one buffer (bufA, 7 slots) and passes x=bufA, x2=x_in, ng=i.
   Implemented fully and measured BIT-COMPATIBLE (max|d| 5.96e-08) with the materialising fallback
   for dilH=16/32 (the naive kernel cannot take two bases) - but the two-pointer k_conv2d_r8 faults
   with "misaligned address" in its float4 tile loader. The loader's guard
   `((ci*H*W + hh*W + ww) & 3) == 0` is provably equivalent mod 4 for the x2 planes (the offset
   differs by ng*HID*H*W, HID=64), so the suspect is nvcc's vectorisation across the two-base
   ternary. Next attempt: mark x2 `__restrict__`, hoist the base select into a per-ci variable
   computed before the loader loop, and force the scalar branch when the base is x2 (uniform per ci,
   zero cost in the inner loop). Ceiling: encoder 0.074 -> ~0.060 s = -0.6% rtf_total, plus 111 MB
   of Nano arena per dense block.
   LESSON from this run: after `git checkout` of a source file, delete the app binary -
   `make` compares .o mtimes and a hand-built engine/reuse_nano_gpu from the broken tree will be
   reused by checks.sh, which then reports the stale build's failure.
 (~1.6 ms + ~250 MB of Nano arena): dense_block_gpu does
   10 cudaMemcpyAsync per forward (skip=x_in, 2 per layer, out=cur). `skip = x_in` and `out = cur`
   are removable; the per-layer [cur, skip] concat needs contiguity, so removing it means the conv
   reads two source pointers (per-ci select: ci < C1 ? x1 : x2 + (ci-C1)*H*W).
4. CLOSED (#62, measured on the Nano): batching the three heads' conv1x1_to1 launches into one
   grid.z=3 kernel is bit-identical (SNR 140 dB) and dev-host neutral (0.0873-0.0879), but the Nano's
   heads phase got WORSE: 0.011 -> 0.014 s (the per-block dynamic select over a param struct puts the
   pointers in local memory, ~96 B/thread; a branch-dispatch version would fix that). More important:
   the heads' 11 ms on the Nano is the conv1x1 FLOPs (3 x ~3.4 ms of 336 MFlop each), not launch
   overhead, so removing launches cannot help. The heads are 0.07% of the Nano model - closed.
4b. CLOSED (#61 sweep, Maxwell branch on the dev host, REUSE_FORCE_MAXWELL_TILES=1, 8 kHz, 4 reps,
   model best): every tower/encoder knob is already at its optimum for Maxwell - conv tile 82 =
   0.4679 (64: 0.5952, 2.2x worse; 92/r9: 0.4704), REUSE_SCAN 2 = 0.4679 vs 1 = 0.4777 (+2.1%),
   k_in_conv_r __launch_bounds__ LB 3 = 0.4679 vs 2 = 0.4745, 4 = 0.4718, 6 = 1.77 (spills),
   REUSE_OP 2 = 0.4789 (the Nano's path, best) vs 1 = 0.4840, 3 = 0.4796 (neutral, i.e. the 8x8's
   sm_61 win does NOT transfer - FMA-ratio rule again), 0 = 36.4 s. A 32-accumulator 8x4 out_proj
   (retry #63, ~2h, reverted): the two-pointer plumbing (x, x2, ng) has three traps found the hard
   way - (1) all 12 k_conv2d_r8 launch sites must gain the two args (verify by grepping r8 launch
   lines for x2,ng: a missed one passes w as x2 = a wild pointer, "misaligned address" at conv2d);
   (2) an ordinary conv must NOT get ng = C_in/HID (that makes every group read the null x2) - use
   the sentinel ng = -1 meaning "plain tensor: xin = x + g_*H*W"; (3) with the correct sentinel the
   output goes NaN, i.e. the slot/concat index correspondence is still off somewhere (the DenseBlock
   writes layer i's inorm to slot i and layer i's conv reads slots 0..i-1 newest-first + x_in, and
   the dil=16/32 layers use the materialising fallback which allocates 6n from the arena and rewinds
   ws_off). A retry should start from the standalone-harness approach the #58 kernels used: verify
   one DenseBlock against the copies path in tests/gpu_block_bench.c before wiring it into the graph.
   (retry #65, harness-first with per-layer GPU_DEBUG dumps - the state of the art on this item):
   (a) the real bug was `plane = ci/HID` where `plane = (group base) + ci%HID` belongs; fixed, the
       encoder's layers 0,1,2 (dil 1,2,4) become BIT-EXACT and the block's inputs (x_in, slots) are
       verified bit-identical between the two paths;
   (b) what remains: layer 3 (dil=8, tile 645) and the decoders' layer 0 (tile 644, W=322) diverge in
       the conv's RAW output (rel rms 0.42/0.27) although every input plane is bit-identical and the
       analytic group->plane mapping is verified correct three times over. DEPTH=4 (dil 1,2,4,8), so
       the dil=16/32 materialising fallback is dead code. Suspect: nvcc's codegen for a runtime
       two-base ternary in the tile loader (vectorisation/aliasing assumptions that held for the
       single base x). Concrete next step: keep ONE base pointer in the kernel - pass a per-group
       plane table in __constant__ memory (8 ints, written with cudaMemcpyToSymbol per launch) so the
       loader stays a single-base affine index exactly as shipped; expected -0.6% on the dev host
       (encoder 0.074 -> ~0.060) plus 111 MB of Nano arena per dense block.
   (d) attempt #66 used the recommended design - ONE base pointer plus a per-group pointer table in
       __constant__ memory (cudaMemcpyToSymbolAsync from a static ring of staging buffers, so the H2D
       source outlives the copy). The kernel's device-side printf shows the table half-stale: for the
       ng=2 conv, c1 = x (exactly the right slot-0 pointer) while c0 is 36096 bytes off the slot-1
       base it should hold, i.e. the async symbol copy is not ordered with respect to the launch on
       this path, and outputs go rel-rms 1.0 from layer 0. Next design, if ever retried: write the
       table with a tiny setup kernel on the same stream immediately before the conv (stream-ordered
       by construction), or keep it in __device__ global memory read through __ldg. Cost of the item:
       ~5 h across 4 attempts, ceiling -0.6%; the encoder's dense layers 0,1,2 were bit-exact in
       attempt #65, which is the closest it got.
   (e) (4b) is therefore NOT worth building (its 2.7 FMA/load sits between OP=2's 2.0 and OP=3's
   4.0, and 4.0 is already neutral there). The instrumentation knob REUSE_INCONV_LB={2,3,4,6} is in
   the tree (default = the shipped LBV=3, bit-identical).
5. Nano: the Nano's remaining model time is the TOWER (RTF 11.88 of 16.0 s = 74% at 8 kHz).
   Its kernels are at their dev-host floors, so the Nano-specific lever is Maxwell occupancy
   (64 KB smem/SM, 24 blocks/SM, 255 regs/thread): sweep tile/LB/KC knobs under
   REUSE_FORCE_MAXWELL_TILES=1 on the dev host - REUSE_OP 2 vs 3, LBV 2/3/4 on k_in_conv_r
   (#45 picked 3 for sm_61), conv tile 82 vs 645, scan REUSE_SCAN 1 vs 2. Memory-pattern wins
   transfer ~1:1 (verified cross-arch: max|d| 1.2e-07 vs the dev host), FMA-ratio wins do not.
   Latest Nano validation: 8 kHz model best 15.96-16.12 s = RTF_model 3.13-3.17 (was 3.2384 at #53)
   = -3.4% transfer of #55/#56/#57/#58/#60. Cumulative Nano RTF_model 44.34 -> 3.13 = 14.2x.

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
