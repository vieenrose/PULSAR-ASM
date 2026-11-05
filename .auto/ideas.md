# Ideas backlog — RE-USE RTF

## Session state (2026-11-04, run #151)
Dev-host band **rtf_total 0.0803-0.0806** (rtf_model 0.0797-0.0800 | stft 0.0002 | istft 0.0004;
enc 0.071 | tower 0.305-0.307 | dec 0.030 | heads 0.001; held-outs 0.0811-0.0819 /
0.1203-0.1209 / 0.1639-0.1648). Noise floor 0.25% => only changes worth >=0.3% are scoreable.
Nano (deployment target, from /dev/shm/rp, run_bench2.sh): **RTF_total 3.0282-3.0352 @ 8 kHz**,
**5.9252-5.9855 @ 16 kHz**; RTF_model 44.34 -> 3.03 = 14.7x, 88.86 -> 5.94 = 14.9x.
Wins this session: #134 Kahan iSTFT (rtf_istft 0.0014->0.0004), #136 page-touch (rtf_stft
0.0014->0.0002), #137 DD STFT (0.0002), #141 **device allocation cache (-1.0%, rtf_model
0.0804->0.0797, Nano decoders -5.3%)**.
Closed this session (do not retry): #139 k_idft kernel sweep (11 tile shapes, split-k, multi-chain
Kahan, conjugate-symmetry pairing - all flat or worse; compensated floor 1.5-1.9 ms),
#143 chunked parallel scan (Hillis-Steele over the affine pair: +2.8%, the serial scan has 47
warps/SM so it is issue-bound not latency-bound), #144 in_conv row table (flat, LICM already
hoists the divide), #145 phase-sync removal (+0.2-0.4%, the syncs keep the launch queue from
stalling).
Only known lever above the noise floor: **Winograd F(2x2,3x3) on the dilation=1 dense convs**
(~0.5-1% of rtf_total; hours of work; gate risk - a register-allocation shuffle once cost
rel-rms 1.8e-4 against the 1e-4 gate). Everything else measured is sub-noise (STFT/iSTFT table
caching + memset removal 0.055%, float-polish STFT head 0.06%, iSTFT split-k 0.08%).


Dev host best: rtf_total **0.0831** (rtf_model 0.0802 | enc 0.071 | tower 0.306 | dec 0.030 | heads 0.001 | stft 0.0014 | istft 0.0015).
Nano (deployment target, tegra210, CUDA_LAUNCH_BLOCKING=1, REUSE_CPU_STFT=1):
RTF_model **3.1346** @ 8 kHz (enc 3.285 | tower 11.566 | dec 1.113 | heads 0.011), **6.1243** @ 16 kHz. Cumulative 44.34 -> 3.13 = **14.1x**.

Per-forward kernel table (nsys -> sqlite, /tmp/p59.sqlite pattern; query the sqlite with
python3, the nsys report scripts are missing in this install):
  in_conv (fused in_proj+causal conv) 141 ms  60 launches  — CLOSED (acc-shape sweep #51, LBV=3 #45, KC #18)
  convs (r8 64-acc shapes)            ~110 ms             — CLOSED (issue-bound: #48 skew +50-74%, #21 float4, #30, #39)
  scan (fused dt_proj+scan)           104 ms  60 launches  — CLOSED (#25 prefetch, #32 unroll, #34 fusion, #38 shared-decay, #46 LB)
  out_proj_r8c (8x8, #50+#57)          76 ms  60 launches  — CLOSED (K-split #41, warps #41, LB #46)
  x_proj k_x_proj_g<128,40,8>           ~27 ms  60 launches  — CLOSED (#68: 20 accs, 2.2 FMA/load,
     one 40-wide tile covers XP=36; the old (128,32) shape ran 2 blocks in x, the 2nd with 4 useful
     columns of 32. model -2.2%, bit-compatible. Shapes swept 12x; 128,48,16 0.4190 and 64,40,8/
     192,40,8 ~0.422 are the only near ties. KT=64 is not viable: shared 45.7 KB -> 1 block/SM.)
  inorm_prelu                          10.5 ms 18 launches — CLOSED (#54: block width and 1-pass/3-pass both neutral)
  outproj_ln_g (4x4 + LN)               9.6 ms 30 launches — CLOSED (#59: 8x8 neutral; launch-quantisation bound)
  sp_up8t (tiled, #52)                  6.3 ms  2 launches — OPEN (shape sweeps only)
  resid scatters + gather              ~2 ms               — mostly gone (#57 fused out_proj, #58 tower fusion)
  encoder skip concats/copies          0 ms (was 1.4)      — CLOSED #74: param-delivery table
     mode (REUSE_DENSE_SLOTS default 1 on non-Maxwell, 0 on Maxwell: the 32-acc shapes corrupt
     under sm_53 codegen, 0.234 signature, delivery-independent). 11 memcpys/444 MB gone.
  heads (3x conv1x1, atan2, crop, sweep) ~1 ms 7 launches  — CLOSED #76 analysis: heads_s=0.001
     is the WALL time between decoder-sync and final-sync (all 3 conv1x1 are inside it); 1 ms is
     already ~2x the naive-read floor; GEMM tiling ceiling = 1 ms = 0.25% of model. Not worth it.
  decoder tail pairing (sp_up2/inorm/transpose)  — CLOSED #76: tail-only pairing FLAT on Nano;
     paired TABLE dense convs = -9.3% Nano decoders but sm_53 TABLE corruption (8th confirmation).

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
   (g) CLOSED 6TH ATTEMPT (#73, +1 h): template<bool TABLE> on r2+r8 (runtime branch rejected:
   +5 instructions in the loader prologue inflated EVERY plain conv, enc 0.074 -> 0.122),
   k_set_src __device__ table, dense slots buffer, REUSE_DENSE_SLOTS env, default 0 verified
   bit-identical to HEAD (complex rel rms 2.855971e-06 / 6.0178e-03, exact). dev slots=1 = +10%
   model (64-acc TABLE instantiations). maxwell proxy (tile 82 = 32-acc) on sm_61: slots=1 =
   -0.7% model AND bit-compatible (max|d| 8.9e-08, all 4 conv configs incl. r2 via
   REUSE_DEC_TILE=64). Nano (real sm_53, same binary): -5.4% model at 8 kHz (14.78 vs 15.62 s),
   -3.0% at 16 kHz - the win is real - BUT audio corrupt: slots vs copies max|d| 0.234,
   rel rms 1.274, deterministic; volatile on g_src changed nothing and broke the sm_61
   maxwell-proxy identically (0.23425466/67 = one latent bug, two codegen paths). Verdict:
   the ~5% Nano win is real but pointer-table delivery is not codegen-robust; 6 attempts,
   ~4.5 h, CLOSED permanently. A 7th would need the loader byte-identical to shipped with the
   base from kernel PARAMS (by-value struct broken on CUDA 10.2, (f)(1)) - e.g. 16 separate
   pointer params; not worth further time.
   (f) 5TH ATTEMPT (#72, ~3.5 h, the design is now fully solved - item CLOSED for the dev host,
   Nano-only value): k_conv2d_r8 given (x2, ng) + a __device__ g_src[16] table written by a
   k_set_src<<<1,32>>> launched on the same stream immediately before the conv (stream-ordered,
   which fixes #66's cudaMemcpyToSymbolAsync race). Result: BIT-COMPATIBLE app-wide (slots vs
   copies max|d| 5.96e-08 = the atomic floor; all 4 dense blocks, encoder + decoders). Measured
   cost: the TABLE=true instantiations run the 16 dense convs ~5-10 ms slower than the 1.4 ms of
   cudaMemcpyAsync they remove (16 extra dependent loads: the loader's base pointer comes from a
   global table read per group) => NET NEGATIVE on the dev host, so slots_mode defaults to 0 and
   the tree keeps only the (bit-identical) scaffolding behind REUSE_DENSE_SLOTS=1. Nano value is
   real (-444 MB of copies ~ -18 ms, -44 launches/forward under CUDA_LAUNCH_BLOCKING, -111 MB
   arena per dense block) but the Nano runs k_conv2d_r2 (32-acc guard), which has NO table mode -
   porting the same (xin, pl) per-ci form into k_conv2d_r2 is the remaining step (~1 h, harness:
   compare slots vs copies audio as done here: max|d| ~6e-08 expected).
   TRAPS found the hard way (#72, all measured, do not repeat):
     (1) passing ConvSrcs {const float* p[16]} BY VALUE as a kernel param under CUDA 10.2 gave
         rel rms 6344 (garbage) - the by-value struct param path is broken/bitten here; use the
         __device__ table + setup kernel, which is verified correct;
     (2) a group-outer/channel-inner restructure of the ci loop (to keep the loader affine):
         brace surgery on that 90-line kernel put the epilogue INSIDE the loops - the model broke
         (gate rel rms 1.36) while timing stayed plausible (enc 0.065 = epilogue runs per-ci +
         sync skip artifacts). Braces around the compute-loop/epilogue boundary must be edited by
         replacing the WHOLE loop region, not by inserting closers at rindex;
     (3) `if (ci) __syncthreads()` inside a restructured loop becomes the outer loop variable -
         sync silently vanishes (enc 0.065, garbage output). Sync condition must be
         `(cij || g)` for a nested loop;
     (4) REUSE_CONV_TILE=64 is NOT a valid control for the dense convs: dense_block passes a
         per-call-site tilehint (645/644, #42) that OVERRIDES the env, so an "r2 control" run
         still measured r8.
     (5) the plain (TABLE=false) instantiations are codegen-identical to the shipped kernel
         (enc 0.074, gates bit-identical) - verified after every surgery; keep that invariant.
   (e) (4b) is therefore NOT worth building (its 2.7 FMA/load sits between OP=2's 2.0 and OP=3's
   4.0, and 4.0 is already neutral there). The instrumentation knob REUSE_INCONV_LB={2,3,4,6} is in
   the tree (default = the shipped LBV=3, bit-identical).
4c. Nano deployment ops (new, #142): the Nano's eMMC is 100% full (13 G/14 G, root-reserved
   blocks leave pesi 0 bytes), so `./build_nano.sh` in ~/reuse_port FAILS at link ("No space left
   on device", also rsync of engine/ fails). Workaround in use: `cp -a ~/reuse_port /dev/shm/rp`
   (tmpfs, 2.0 G) then rsync engine/ there, `rm -f engine/*.o engine/reuse_nano_gpu`,
   `NVCC=/usr/local/cuda-10.2/bin/nvcc ./build_nano.sh`, and run /dev/shm/rp/run_bench2.sh - a
   copy of run_bench.sh with `cd /dev/shm/rp` and REUSE_DUMP_* pointing at /dev/shm (the /tmp dumps
   would also fail on the full disk). tmpfs is wiped on reboot: re-create run_bench2.sh after a
   reboot. User data (b2 590 M, vc 410 M, exp 141 M, primetts 129 M) is NOT ours to delete.
4d. CLOSED: the #70 "fast CPU STFT" (parallel pre-pass + 4 double accumulator chains, -22% Nano
   stft(cpu)) no longer needs re-applying before an engine/ rsync - since #138 the Nano runs the
   GPU DD STFT, so the CPU path is only reached by REUSE_CPU_STFT=1 A/B runs. If such an A/B is
   ever run, expect the Nano's CPU STFT to read 0.087 s (serial) not 0.065 s (fast form).

5. Nano: the Nano's remaining model time is the TOWER (9.73 of 14.02 s = 69% at 8 kHz,
   #211). Its kernels are at their dev-host floors, so the Nano-specific lever is Maxwell
   occupancy (64 KB smem/SM, 24 blocks/SM, 255 regs/thread): sweep tile/LB/KC knobs under
   REUSE_FORCE_MAXWELL_TILES=1 on the dev host - REUSE_OP 2 vs 3, LBV 2/3/4 on k_in_conv_r
   (#45 picked 3 for sm_61), conv tile 82 vs 645, scan REUSE_SCAN 1 vs 2. Memory-pattern wins
   transfer ~1:1 (verified cross-arch: Nano-vs-dev amp BIT-IDENTICAL, #193/#196/#211),
   FMA-ratio wins do not. Cumulative Nano RTF_model 44.34 -> 2.75 = 16.1x (#211). The
   Nano-only backlog at the bottom of this file is the current list - this item's
   general advice stands, its old numbers (RTF 11.88/16.0, validation 15.96 s) are retired.

5. RETIRED (#211): REUSE_CPU_STFT=1 vs GPU STFT on Nano is moot - the Nano bench feeds
   the reference STFT (fed protocol), so no STFT runs at all in the timed region
   (stft(cpu) 0.0000 s). The DD GPU STFT (#137) stands as the faster live-input path.

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
- CLOSED FOR THE Nano (#70, implemented + validated, NOT LANDED - re-apply in ~2 min with the next
  Nano-validation keep): engine/stft_dft.cu - add a parallel pre-pass building the windowed frames in
  double (xw[t][n] = x[idx]*w[n], reflect-pad index computed once; the committed loop recomputes it F
  times per sample) and turn the DFT inner loop into a pure dot product with FOUR independent double
  accumulator chains (n mod 4), tree-combined (r0+r1)+(r2+r3), plus a tail loop for n_fft % 4.
  Nano 8 kHz: stft(cpu) 0.0893 -> 0.0686 s (median of 3 interleaved pairs) = -23% = -21 ms per
  forward = -0.7% of Nano RTF 3.05. Accuracy: mag BIT-IDENTICAL (max|d| = 0), pha max|d| 1.8e-12,
  rel rms 2.5e-15 -> float outputs identical, no atan2 sign flips. Env REUSE_STFT_FAST=0 restores the
  serial form (default ON). Dev host: the scored metric is untouched, the GPU STFT (0.0020 s) is the
  default there. Same patch adds REUSE_DUMP_STFT=<base> to reuse_host_gpu.c (writes <base>_mag.raw
  and <base>_pha.raw from the CPU STFT) - the tool for verifying any future STFT change; keep it.
  Not landed as a scored experiment because the dev metric is flat by construction; keeps are
  reserved for scored improvements.
  REVALIDATED #73 on a fresh Nano build: stft(cpu) 0.0636/0.0706 (fast) vs 0.0869 (serial) =
  -22% again; run_bench.sh has REUSE_CPU_STFT=1 and NO REUSE_DENSE_SLOTS (slots=0 on the Nano:
  the table mode is not codegen-robust, see item 1b(g)). IMPORTANT: the Nano's engine/stft_dft.cu
  now carries the fast form while the repo keeps the serial one - the next repo-side land must
  re-apply this patch (it is in git history via the #70/#73 working trees) BEFORE the next
  engine/ rsync, which would otherwise overwrite the Nano copy with the serial version.

## Deployment (Nano) rules learned
- Kernels with >=64 per-thread registers COMPILE for sm_53 and FAIL AT LAUNCH ("too many
  resources requested"); checks.sh cannot see it. Guard with gpu_is_maxwell() (honours
  REUSE_FORCE_MAXWELL_TILES=1). Guarded today: conv tile codes 644/645/646 (#47), REUSE_OP 3->2 (#53).
- Memory-pattern wins transfer ~1:1 (transpose tiling -5% decoders, tower fusion -2.7% tower,
  strided conv -4.3% encoder); FMA-ratio/accumulator wins do NOT (latency-bound at 16 warps/SM).
- Fast Nano A/B: same input + fed-reference protocol + REUSE_DUMP_AUDIO, diff waveforms
  (140 dB = bit-compatible). Binary ./engine/reuse_nano_gpu, blob tests/golden/reuse.blob,
  REUSE_WS_MB=640 (8k) / 1280 (16k), reps >= 2 (rep 0 is warm-up). Disk ~450 MB free, 3 GiB RAM.

## OPEN (new chapter: STFT/iSTFT fp64 -> compensated fp32, opened #134-#137)
The STFT/iSTFT were declared "at their floors" (#59) but were fp64 GEMMs on hardware with
1/32-1/64-rate fp64. Four keeps: rtf_total 0.0832 -> 0.0810.
1. RETIRED (#211): Nano GPU-vs-CPU STFT is moot - the Nano bench feeds the reference
   STFT (fed protocol), so no STFT runs in the timed region. The DD GPU STFT (#137)
   stands as the faster live-input path.
2. k_idft (~1.0 ms of the 2.0 ms iSTFT): grid is 4 x ceil(T/16) = 80 blocks on 15 SMs, 128
   threads each = 680 threads/SM. Split-k over 2-4 partials + a reduction, or a 2D tiling over
   n, raises occupancy. Ceiling -0.2% rtf_total. fp32-Kahan accumulation makes a split-k
   reduction cheap (add dd pairs, not fp64).
3. k_stft_head 0.28 ms: 161x1021 = 165k elements, double sqrt/log1p/atan2. log1p+atan2 in
   double are ~100-instruction paths; a dd/fp32 form with a double-polish only near the
   boundaries would cut it to ~0.1 ms. Ceiling -0.04%. Low value.
4. k_istft_tables regenerates the 2 x n_fft x F twiddle table on EVERY call (0.084 ms) - cache
   it per (n_fft, F) in the graph (it is a pure function of the shape). Ceiling -0.08 ms/call.
   Also k_stft_twiddles_dd (0.065 ms) and the window: same argument. Cheap, ~-0.03%.
5. Model side is unchanged and closed: tower 0.306 s (in_conv 288 ms/240, scan 209/240,
   out_proj 153/240, x_proj 58/240 per profile), enc 0.070, dec 0.030, heads 0.001.
6. CLOSED (#139 measurement, no code landed): k_idft_k (1.83 ms, 61% of the STFT+iSTFT) is
   latency-bound on Kahan's 4-op serial dependency, 5x off its 0.37 ms op roof. Standalone
   harness (/tmp/idft_bench.cu, shapes 320/161/1021, best of 7): plain fp32 0.53 ms
   (rel rms 8.5e-9 - NOT compensated, so rule 4 forbids it), shipped Kahan+exact-product-residual
   1.90 ms (1.0e-9), Kahan-on-rounded-product 3.67 ms (compiler quirk, slower), fp64 4.69 ms,
   register tiling 4x4/2x8/8x2/2x4/4x2/8x4/4x8/2x2/8x8/1x4/4x1 ALL 0.0022-0.0023 ms in-app
   (flat - not load-bound), split-k x2/x4/x8 = 1.57/1.50/1.53 ms (best, -21%), multi-chain
   (k mod 2/4/8) = 5.8-6.1 ms (register pressure kills occupancy). Best compensated form is
   -0.4 ms = 0.08% of rtf_total = below the 0.24% noise floor -> not landed.
7. CLOSED (#144): in_conv's per-item 64-bit m/L divide is NOT a lever - a shared
   row-table (one divide per block, identity map for rev=0, m-2*(m%L)-1 for rev=1,
   compute loop untouched so its register allocation is shipped-identical) measured
   0.4066 vs 0.4062 s model = flat. ptxas already LICMs the divide out of the k0
   loop, so the in-code note's "3% for hoisting" is unreachable from this direction
   (it came from a restructure that also moved register allocation, and failed the
   gate at 1.8e-4). Do not retry divide hoisting in k_in_conv_r/r2.

## Session state (2026-11-04, run #162)
Dev-host band **rtf_total 0.0790-0.0794** (rtf_model 0.0784-0.0788 | stft 0.0002 | istft 0.0004-0.0005;
enc 0.071-0.072 | tower 0.298-0.299 | dec 0.030 | heads 0.001; held-outs 0.0800-0.0804 /
0.1183-0.1185 / 0.1614-0.1619). Noise floor 0.25%.
Newest win: #157 vectorised scan's per-step xdbl reads (Bt/Ct/dr loaded as float4 once per
step instead of scalars inside the unrolled DSTATE loop) - tower 0.305 -> 0.298 (-2.3%),
rtf_total 0.0801 -> 0.0791 (-1.2%). Bit-compatible (max|d| 1.19e-07).
Remaining above-noise candidate: **Winograd F(2x2,3x3) on dilH=1 dense convs** (~0.5-1% of
rtf_total, hours, gate risk). Everything else measured is sub-noise or closed.

## Profile of the committed tree at rtf_total 0.0731 (#186, nsys, 2 forwards, per-forward kernel ms)
in_conv 110.8 (28%) | scan_dts 94.0 | conv2d_r8 family 84.5 (21%) | out_proj_r8c 76.4 (19%) |
x_proj 25.6 | inorm_prelu 5.3 | outproj_ln 4.9 | sp_up 3.2 | resid_freq 3.3 | resid_out 3.2 |
conv1x1 1.7 | transpose 1.4 | idft 0.9 | stft_dd 0.5 | misc ~2. GPU busy ~= wall (slight overlap)
=> NO launch-idle lever; #145's "syncs are free" holds.
NEW closures from this profile:
1. -Xptxas -dlcm=cg = +8% model, -dlcm=cs = +4%, default (ca) best; --extra-device-vectorization
   does not exist in CUDA 10.2 ptxas. Compile-flag lever CLOSED.
2. k_resid_freq_a + k_resid_out_a CANNOT be fused: the freq mamba block consumes xf between them.
   CLOSED TWICE OVER (#186 fusion-impossible; #257 both speed levers measured):
   (a) instruction count is NOT the limit - taking f/t/c from blockIdx/threadIdx cut SASS from
       ~300 to 54 inst/thread (5.6x) and was EXACTLY FLAT over 4 interleaved pairs (0.3521 vs
       0.3519 s). The per-element 64-bit index math is fully hidden under memory latency.
   (b) float4 width: 2 of 3 pairs better, mean -0.31% model, but the scored run read 0.0691 vs a
       0.0688 band top (opposite sign) => noise, not landed.
   The old "57% of the bandwidth roof / ~43% headroom" figure was wrong: it normalised against a
   2-stream (read+write) roof while the kernel moves 3 streams (2 reads + 1 write), i.e. it is
   already ~85% of a real 3-stream roof. Do not retry this pair.
3. Encoder dense convs are the largest single launches in the model: k_conv2d_r8p<48,8,4,2,true>
   (dil=8, C_in=384) = 29.5 ms for ONE conv; grid (3,32,8), 128 regs, 13 KB shared, 2 blocks/SM.
   FLOPs analysis: 70 GFlop => 2.4 TFLOPS = 41% of the 5.78 TFLOPS fp32 roof. The R shared tile is
   exactly TH+2*dilH (34/36/40/48 for dil 1/2/4/8) - the loader is optimal (every loaded row is
   used by some output row). Instruction mix per ci per thread = 576 FFMA + 72 weight-LDS
   (PCO=8 x 9 taps) + ~18 A-LDS + addressing = 87% FFMA -> issue-limited ceiling ~86% of roof,
   so the 41% gap is LSU/latency/register-bound, not multiply-bound (consistent with Winograd
   tying at #185). Shape space is nearly exhausted by shared+registers: accs=64 costs 128 regs
   -> only 2 blocks/SM; cutting PCO to 4 to halve weight loads needs R=66..80 rows = 18-22 KB
   shared (still 2 blocks/SM) but pushes regs past 136 -> 1 block/SM. Tile codes 644/645/646/81/82
   (PCO 4-16, NCB 1-4, NRA 1-4) already cover the feasible frontier. CLOSED for shapes.

## INCIDENT #193: Nano shipped garbage audio since the concat-fuse (#57/#58 era) - CAUGHT 2026-11-05
- Symptom: Nano-vs-dev amp diff rel_rms 0.95 (max|d| 1.88) on identical inputs+fed reference.
- Root cause (host protocol, not kernels): mamba_block_gpu's `cfuse` (REUSE_CONCAT_FUSE,
  default 1) skipped the separate concat whenever set, but only the tiled==3 kernels
  (k_out_proj_r8c[_t]) have the fused epilogue. OP=2 = k_out_proj_r writes `out`, so on the
  Nano (tiled forced 3->2 by the Maxwell guard) `cat` was NEVER filled - every Nano timing
  run since the fuse measured garbage at full speed. k_out_proj_r itself is FINE (gate
  rel_rms 1.34e-06 with the concat restored).
- Fix: shared `outproj_tiled()` helper (single home for the REUSE_OP default-3 +
  Maxwell->2 resolution); mamba_block_gpu forces cfuse=0 unless resolved tiled==3;
  mamba_dir_gpu uses the helper for its launch branch. tiled==3 paths byte-untouched.
- Verification: dev REUSE_OP=2 gate PASS (1.34e-06 / 117.9 dB); dev REUSE_FORCE_MAXWELL_TILES=1
  (ALL Maxwell paths: tile-82 convs, slots-0, dec-pair-0, OP=2) gate PASS (1.34e-06);
  Nano rebuilt: amp AND pha BIT-IDENTICAL to dev (max|d| 0.0), audio max|d| 1.79e-07
  (iSTFT atomic floor). Nano RTF_total 2.7541 @ 8 kHz / 5.3800 @ 16 kHz (wasTiming-only
  garbage 2.7432/5.3533; +0.4% is the now-running k_concat_f). Cumulative Nano
  RTF_model 44.34 -> 2.7444 = 16.2x.
- LESSONS: (1) periodic Nano VALIDATION must diff audio, not just time - timing-only
  runs are blind to correctness; (2) checks.sh now carries a REUSE_FORCE_MAXWELL_TILES=1
  model gate at 8 kHz (same thresholds) so the deployment path is gated on every
  iteration; (3) any kernel family with arch-dependent dispatch needs its non-default
  branch gated (OP=1/2, r2 convs were never gated before); (4) dev REUSE_FORCE_MAXWELL_TILES=1
  output == Nano output bit-exact (max|d| 0.0) - the proxy is perfect for correctness,
  use it before every Nano rsync.

## Closures session 2026-11-05 (#193-#198)
- #194 conv r8 weight staging (Ws[j][kh] float4, existing syncs): -2.7% rtf_total.
  Template for spotting the next one: SASS LDG(.E/.32) in the HUNDREDS on a kernel
  whose hot loop indexes weights by (output-group, tap) only = block-uniform
  traffic with 256x redundancy. Checked and clean since: x_proj (LDS128-only),
  outproj_ln_g (LDS128=66), scan (MUFU-bound, delivery hidden), in_conv Ws
  (compiler vectorizes k-direction), inorm/sp_up/conv1x1/heads (floors).
- #195 r8p WS: FLAT (decoders already ride r8 through fallbacks; r8p-exclusive
  remainder too small). Do not extend staging ideas to small-launch kernels.
- #197 encoder retile 645->644 post-WS (enc -6.6%, held16k -2.4% best): ALWAYS
  re-sweep tile/shape knobs after a memory-hierarchy change - the FMA/LSU balance
  moves the optimum. Decoder confirmed 644. 646 dead everywhere (168 regs).
- #198 KC re-sweep: 32 optimal (16 +2%, 8 +5%). Unchanged by tower reshaping.
- Winograd F(2x2,3x3) dilH=1: CLOSED PERMANENTLY (3rd verdict). Arithmetic:
  4 mults+~8 adds per output vs 9 mults direct - it only wins via the 16-way
  batched GEMM running at direct-conv efficiency, and #185 proved the batched
  form merely TIES at 41%-of-roof issue efficiency. No changed assumption can
  move this (WS helps direct and Winograd equally; tiles moved on). Do not retry.
- SASS-first rule (from #190/#192): check LDS mixes BEFORE building. k-direction
  groups in [row][k] shared are ALWAYS already vectorized by nvcc (manual
  interchange = no-op re-spelling, identical LDS counts). Only row-direction
  groups within one k iteration (strided, unvectorizable) reward layout
  transposes (#189). in_conv's 65 Xs scalars are the textbook unfixable case
  (stride-5 rows + halo-3 vs 16B alignment).

## Nano-only backlog (unscored on dev - build only with Nano A/B, never keep on dev metric)
1. TRIED #269, REVERTED - SLOWER ON THE NANO. Fused concat epilogue in k_out_proj_r
   was correct and -2.1% on the dev Maxwell proxy, but an interleaved ON-DEVICE A/B
   showed +4.8% model / +6.3% tower on the Nano (same regs/smem on sm_53, so likely
   the split 256 B half-row stores, one direction to time-flipped rows). Do not retry
   without a different store layout. Original note:
   Fuse the concat+residual epilogue into k_out_proj_r (OP=2, the Nano path):
   writes `out` today + separate k_concat_f (~0.6% Nano tower). r8c shows the
   epilogue pattern (coff/rev/xres/float4 stores); risk is regs (sm_53 launch
   ceiling 63/thread - the #73 TABLE saga). A/B on Nano only.
2. R8LAUNCH-WS twin for tile-82 (Maxwell dense): same 256x weight redundancy as
   #194, Nano encoder/decoder dense all run tile 82. A/B on dev with
   REUSE_CONV_TILE=82 for speed, confirm transfer on Nano.
3. Launch-count fusion for CUDA_LAUNCH_BLOCKING (Nano pays full launch latency):
   count launches per forward and fuse the cheapest boundaries first (heads
   conv1x1 batching was FLOP-bound #62 - pick glue kernels, not GEMMs).

## Session state (2026-11-05, run #200)
Dev-host band **rtf_total 0.0684-0.0688** (rtf_model 0.0678-0.0684 | stft 0.0002 |
istft 0.0005; enc 0.062 | tower 0.257-0.258 | dec 0.026 | heads 0.001;
held-outs 0.0701-0.0706 / 0.1027-0.1035 / 0.1392-0.1452). Noise floor ~0.25%.
Nano (deployment target, /dev/shm/rp, run_bench2.sh): **RTF_total 2.7574 @ 8 kHz**
(model 2.7477: enc 3.252 | tower 9.702 | dec 1.047 | heads 0.011),
**5.3707 @ 16 kHz** (model 5.3332); amp+pha BIT-IDENTICAL to dev every validation.
Cumulative: dev 0.0731 -> 0.0684 (-6.4% this session); Nano RTF_model 44.34 -> 2.75 = 16.1x.
Wins this session: #189 out_proj A-tile transpose (-3.0%), #193 Nano concat-fuse
protocol fix (deployment correctness, dev-neutral), #194 conv r8 weight staging
(-2.7%), #197 encoder retile 645->644 post-WS (-0.6%).
Discards: #188 OP-sweep (3<2<1 confirmed), #190 inconv-Ws (+3.5%: compiler already
vectorizes k-direction), #191 scan broadcast (+5%: MUFU-bound), #192 ck4 (noise:
manual k-interchange = compiler's own transform), #195 r8p-WS (flat: remainder too
small), #198 KC-sweep (32 optimal). Validations: #196 Nano (+dump-diff), #199/#200 band.
Standing rules learned: SASS-first (check LDS mixes before building - k-direction
groups are always compiler-vectorized; only strided row-groups reward transposes);
re-sweep shape knobs after every memory-hierarchy change; Nano validation must
diff audio (timing-only retired); checks.sh now gates Maxwell paths every iteration.
Scored levers: EXHAUSTED (see closures above). Remaining: Nano-only backlog (fused
OP2 epilogue, tile-82 WS, launch fusion) - unscored, build only with Nano A/B.

## Nano-only backlog, continued (unscored - see policy above)
4. REUSE_CONV_CHUNK sweep: CLOSED as no-op both places (#208: dev 64 0.3454/0.3472
   vs 128 0.3462/0.3467 flat vs 32 0.3515/0.3516 +1.6%). All model convs have
   C_out <= HID = 64 = one chunk at the default, so 128 changes nothing (same
   single iteration, including on Nano - no fewer launches there either); 32
   splits into 2 launches and loses. Default 64 confirmed. Do not retry.
5. Profiling calibration note (#201): nsys 2021.1 absolute kernel durations read
   ~2x below wall-implied values on this stack (GTX 1070, CUDA 10.2) - the #186
   profile matches 16 kHz rate-scaling, mine matches 8 kHz halves; gap fractions
   and rankings are scale-free and trustworthy, absolute ms are not. Do NOT use
   nsys ms for roof arithmetic; use ratios and counts only.

## Rejected structural idea (reasoned, no build, #220-turn)
- Share ONE in_proj GEMM across both scan directions (it is direction-independent
  math; today in_conv recomputes the 256->512 GEMM twice per block, 60 launches):
  REJECTED. The rev time-flip is baked into the fused kernel's row indexing
  (perm(m)), so sharing means unfusing (write 42 MB x0/z + 2x42 MB rereads per
  block = ~30 ms traffic) plus re-engineering the dual-direction halo/shuffle
  plumbing that carries three historical scars (rel-rms 0.57 shfl_up, NaN compact
  indexing, all-NaN padded-stride). Expected net: wash to negative at high risk.
  The fusion won for a reason (#58 era). Do not attempt without a new mechanism
  for the flipped-halo problem.

## WIN #255: weight staging reached the Nano (R8LAUNCH/tile-82) - Nano -7.7%
- Root cause of a long-standing blind spot: #194 added the WS twin ONLY to R64LAUNCH.
  Maxwell remaps 644/645/646 -> 82, and tile 82 dispatches through R8LAUNCH - so the
  Nano's dense convs kept doing 256x-redundant per-thread weight LDGs for the whole
  time the dev host was benefiting. R8LAUNCH now routes through an R8P helper that
  picks the WS=true twin via the same `cws` switch (REUSE_WST).
- Nano (MEASURED, /dev/shm/rp): 8 kHz model 14.0170 -> 12.9292 s (-7.8%;
  encoder 3.250 -> 2.375 = -26.9%, decoders 1.046 -> 0.829 = -20.7%),
  RTF_total 2.7581 -> 2.5448 (-7.7%). 16 kHz model 27.1978 -> 25.1555 (-7.5%),
  RTF_total 5.3692 -> 4.9689. Cumulative RTF_model 44.34 -> 2.5351 = 17.5x.
- Nano output BIT-IDENTICAL to the dev Maxwell proxy (amp max|d| 0.0).
- Dev scored primary FLAT (0.0689 vs 0.0683-0.0688) by construction (dev runs 644).
- GENERAL LESSON: after any optimization, grep ALL dispatch macros for the twin
  (R8ONE / R9LAUNCH / R64LAUNCH / R8LAUNCH / R8LAUNCH / LAUNCHP / R8P). A win that is
  wired into one macro silently misses every arch that dispatches through another.
  This single oversight cost the deployment target ~8% for several sessions.
- NANO AUDIT TODO: the same "which macro does Maxwell hit" question applies to the
  stepped kernels - k_conv2d_r8p (pair) got WS in #195 and was FLAT there, and
  REUSE_DEC_PAIR defaults to 0 on Maxwell so the pair path is dead on-device.

## Process: measurement hygiene (learned the hard way #229/#255)
Before trusting ANY reading, check BOTH:
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv   (foreign CUDA work)
  uptime                                                          (CPU load for the host-side launches)
A foreign CUDA process (1334 MiB, 96% util) and a CPU load of ~3 each inflated results by
30-100% and produced two bogus conclusions in a single iteration (a fake -30% dev win and
a fake +31% tower regression). Signature: bench wall time >55s when the normal is ~50s.

## #256 CLOSED: #189's A-tile transpose does NOT transfer to k_out_proj_r (OP=2 / Nano)
- Swept the same-class question raised by #255: #189's transpose landed only in
  k_out_proj_r8c_t (tiled==3), and Maxwell forces tiled=2 -> k_out_proj_r, so the
  Nano's out_proj never got it. Built k_out_proj_r_t (AsT[KC][MT+4], 272 B rows
  so every k-row is 16 B aligned; staging stays element-wise so it costs ZERO
  extra stores, unlike r8c which traded an STS.128 for 4 scalar STS).
- Result: FLAT. 3 interleaved pairs on the OP=2 path split 0-1-1, mean -0.06%
  (0.3867/0.3868/0.3851 off vs 0.3871/0.3863/0.3845 on), SASS byte-identical
  (64 LDS.U.128, 531 FFMA, 14 STS both). Bit-compatible (8.9e-08).
- WHY it differs from #189: r8c's a-group is 8 rows at an 80 B row stride; this
  one is 4 rows at 144 B. #189's win was an access-pattern/transaction effect on
  that specific group, not a general property of transposing A tiles. With only
  4 accumulators the per-k group is small enough that the pattern cost is gone.
- Transferable rule: an A-tile transpose is worth testing only when the group it
  makes contiguous is >= 8 elements wide. Do not auto-port #189 to other kernels.
- Dispatch audit (also from #255's lesson) now COMPLETE for convs: R64LAUNCH and
  R8LAUNCH both select the WS twin, R8ONE has no call site, R9LAUNCH (tile 92) is
  env-only and remapped to 82 on Maxwell. No further missed twins.

## #258 CLOSED: Maxwell dense-conv tile re-sweep AFTER the #255 weight staging
- Discipline check from #197 ("re-sweep shape knobs after a memory-hierarchy change"),
  applied to the Maxwell path #255 just changed. Swept the 32-accumulator tiles the
  hardcoded Maxwell fallback chooses from, encoder phase, WS staged, 2 pairs each:
      82  (8co x 2col x 2row)  0.064 / 0.064 s   <-- SHIPPED, best
      83  (16co x 2col x 1row) 0.079 / 0.080
      81  (4co x 4col x 2row)  0.084 / 0.085
      92  (r9, 16co)           0.098 / 0.098
- So the optimum is an INVERTED-U in the co dimension peaking at 8, not a monotone
  "more channels per block is better" trend (the initial read of 82 > 81 suggested a
  trend; 83 refuted it). #255 left no Maxwell shape on the table; tile 82 stands.
- Do not re-sweep this set again: 4 of the 4 Maxwell-launchable shapes are measured.
- sm_53 register note: all four 32-acc tiles compile to 92-96 regs (81: 92, 82: 96,
  83: 96-111), so STATIC register counts do not predict the documented 64-acc
  launch failure - that is a different mechanism. Do not use -Xptxas -v as a
  Maxwell launch-viability oracle; only an on-device launch is.

## WIN #259: scan recurrence - hoist the per-step multiplies (-0.79% model)
- The fused scan (k_scan_dts, 24% of the model) spends two FMULs per state element on
  hoistable work:
    __expf(d*a[n])  = FMUL(d*a) + FMUL(*log2e) + MUFU.EX2      -> FMUL(d*(a*log2e)) + MUFU
    d*Bt[n]*u       = FMUL(d*Bt) + FMUL(*u)                    -> FMUL(du*Bt),  du = d*u
  Folding log2(e) into the once-per-thread decay fill and hoisting du out of the n-loop
  is 32 fewer FMULs per step (SASS: FMUL 100->89, total 555->533, MUFU unchanged 33).
- Clean 4-pair A/B: model 0.34975 -> 0.34700 mean (-0.79%), 4/4 pairs; tower 0.262->0.259.
  Numerics: association only, end-to-end max|d| 1.6e-07 / rel_rms 3.9e-07 (floor ~9e-08),
  gate margin still 75x. REUSE_SCANF=0 restores the shipped form.
- CORRECTION to a long-standing note: "the scan is MUFU-bound" is only mostly true. MUFU
  count is unchanged here yet the model got 0.79% faster, so a slice of the loop IS
  issue-bound. Instruction-count reductions DO pay in the scan, unlike the resid kernels
  (#257) where a 5.6x cut was free. Always separate those two conclusions per kernel.
- NEXT IDEAS ON THIS THREAD (not yet tried, same "hoist what is loop-invariant" family):
  1. In the same loop, `u = xc[sl*DINNER + c]` is loaded per step; c is thread-fixed but
     sl advances by DINNER, so the address is already a running offset - probably fine.
  2. The dt path still computes dsoftplus = log1pf(__expf(v)) per step: __logf-based
     dsoftplus could drop a MUFU per step if the gate tolerates 1 ulp (v>20 branch already
     exists). That is a PRECISION-adjacent change - only attempt with the gate as arbiter.
  3. k_scan_dt / k_scan (the non-fused ts!=2 paths) never got #157's float4 row loads or
     this hoist; they are A/B-only paths, leave them.

## WIN #260: scan decay constants shared -> registers (-1.5% rtf_total, -1.6% Nano)
- The 16 per-thread decay constants (aAs[lane][n]) were re-read as 16 LDS.U.32 on EVERY
  step - 16 of ~90 instructions per step, for data that never changes.
- WHY they were in shared: __launch_bounds__(256,4) pins ptxas at exactly 64 regs/thread
  and k_scan_dts already used 64, so a 16-value per-thread array there would have spilled.
  The shared copy was forced by the launch bound, NOT chosen for speed. This is a general
  smell worth grepping for: "thread-private data staged in __shared__" almost always means
  an LB/register ceiling pushed it there, and relaxing the bound can be the cheaper trade.
- Relaxing to LB(256,2): 90 registers, no spills, shared 6272 -> 4096 B. Removes 16 LDS/step.
- Clean 3-group A/B (model): shared/LB4 0.34523 -> regs/LB2 0.33943 = -1.7%.
  regs/LB1 (0.33940) was a 0.01% tie, so LB2 was taken for the Nano's single SM headroom.
  Halving occupancy cost NOTHING on either arch: the 16 LDS were the binding constraint.
- Scored: rtf_total 0.0673 (band was 0.0683-0.0688) = -1.5%, tower 0.251, all 3 held-outs
  better. Nano: 8 kHz model 12.9292 -> 12.7217 (-1.6%), RTF_total 2.5041, 16 kHz 4.8867;
  output BIT-identical to the dev Maxwell proxy. Cumulative Nano 44.34 -> 2.4945 = 17.8x.
- Knob: REUSE_SCANR 0 = shipped shared/LB4, 1 = regs LB3, 2 = regs LB2 (default), 3 = LB1.
- NEXT on this thread: the same "thread-private data in shared" audit. Remaining candidates
  seen in SASS: k_inorm_prelu (LDS32=66, STS=34) and k_scan_dt/k_scan (non-fused A/B paths).
  k_scan_dts is now LDS-free in the hot loop - re-check its SASS mix before any further work.

## #261: x_proj occupancy lever CLOSED + 3 BROKEN instantiations found (LANDMINE)
- x_proj launches only 8 blocks (grid (1,8)) on a 15-SM GPU, which looked like the
  unclaimed lever. Swept everything via REUSE_XPROJ = MT*10000 + NT*100 + TX.
- Among CORRECT shapes the shipped default wins outright:
      128,40,8  (8 blocks)  0.3386 model   <- shipped
       64,40,8 (16 blocks)  +1.5%
       32,40,8 (32 blocks)  +5.7%
      256,40,8  (4 blocks)  +7.0%
  Mechanism: every row-block re-reads the whole 36x256 W tile (8 blocks = 288 KB of W
  traffic, 32 blocks = 1.15 MB), so extra blocks multiply weight traffic; below 8 blocks
  the 15 SMs go unfilled. 8 is the sweet spot. OCCUPANCY LEVER CLOSED - do not re-sweep.
- **LANDMINE**: all three TX=16 instantiations are SILENTLY WRONG.
  (128,32,16), (128,48,16), (128,64,16) each give rel_rms 0.89 vs the default on the same
  input (audio becomes near-noise, std 0.031 instead of speech-like).
  (128,64,16) presents as a **7.2% MODEL WIN** (0.3143 vs 0.3386) and is pure garbage -
  it "wins" by not doing the work. The default dispatches TX=8 so nothing shipped is
  broken, but REUSE_XPROJ can select them.
  => This is the SECOND occurrence of the #64 trap in this project's history. RULE (now
  mandatory): for any swept shape, run a dump-diff against the current default BEFORE
  comparing timings. A timing win larger than the kernel's entire share of the profile is
  proof of a bug, not a discovery. x_proj is ~6% of the model; a "7% model win" from it is
  arithmetically impossible.
- Verified-correct alternatives that lose: 64,40,8 and 32,40,8 (both bit-compatible).

## WIN #262: in_conv epilogue - hoist the depthwise-conv weights (-1.1% model, -2.2% Nano)
- k_in_conv_r's epilogue reloaded Wc[(c0+b4)*DCONV+k] and Bc[c0+b4] on EVERY one of the
  RA=5 row iterations, though both index only (c0, k) = thread constants:
  5*4 bias + 5*4*4 tap loads = 100 scalar LDGs per thread for 20 distinct values.
  Now loaded once into 20 registers before the loop. ptxas stayed at 80 registers (the
  values are live only in the epilogue, after the main loop's a[]/w[] die), so occupancy
  did not change. SASS scalar LDGs 100 -> 20, total 3282 -> 3222.
- Clean 3-pair A/B: model 0.33930 -> 0.33553 (-1.11%), tower 0.253 -> 0.2485. Bit-compat.
- Scored NEW BEST rtf_total 0.0663, tower 0.246, held12k under 0.10 for the first time.
- Nano: model 12.7217 -> 12.4421 (-2.2%), RTF_total 2.4492, 16 kHz 4.7812; bit-identical
  to the dev Maxwell proxy. Cumulative 44.34 -> 2.4396 = 18.2x.

## THE LENS THAT IS WORKING (apply it systematically - 3 wins in a row)
Search every hot kernel for **work re-done per iteration that depends only on thread
constants**, then hoist it. Concretely, grep the inner loops for:
  1. per-step multiplies of a scalar by a thread-constant table (#259: d*aAs, d*u -> -0.79%)
  2. thread-private data parked in __shared__ (usually forced there by a launch bound, so
     relaxing the bound is cheaper than the round-trip; #260 -> -1.5%)
  3. loads indexed only by (thread id, tap) INSIDE a loop over rows/steps (#262 -> -1.1%)
Each win was ~1%, each cost ~1 hour, and all three are still open for the kernels not yet
audited with this lens. NEXT CANDIDATES on the same lens:
  - k_conv2d_r8 / r8p: the weight tap loads `wv[j*C_in*9 + kh*3 + t]` sit inside the ci
    loop; j and kh are unrolled constants but C_in*9 strides over ci, so this is NOT the
    same pattern - verify with SASS before assuming.
  - k_outproj_ln_g: B[c0+b4] and LW/LB[c0+b4] are loaded once already; check the inner k.
  - k_inorm_prelu: 66 LDS32 / 34 STS - audit for thread-constant redundancy.
  - sp_up8w/sp_up8t: check the weight staging.
README: measure by interleaving a knob; dump-diff BEFORE timing (#261 rule); check
nvidia-smi compute-apps AND /proc/loadavg before trusting any absolute number.

## #263 CLOSED: dense-conv loader hoist - two hard constraints learned
Tried hoisting k_conv2d_r8's A-tile geometry (i/RS divide, q, hh/ww, 6 guards) out of the
ci loop (it runs up to C_in = 384 times). SASS: 929 XMAD + 374 integer + 105 ISETP against
only 535 FFMA, so it looked like the biggest remaining win in the #1-dense kernel.

1. **REGISTER CLIFF (new hard constraint).** k_conv2d_r8<80,8,2,4,true,true> = 126 registers.
   126*256 = 32256 of the 65536 per SM = exactly 2 blocks/SM with **2 registers of slack**.
   Storing the hoisted geometry per-thread took it to 159 -> 1 block/SM -> encoder +39%,
   model +8.7%. ANY change to this kernel that adds >2 registers/thread halves occupancy.
   Check ptxas register count BEFORE building anything here.
2. **NOT ADDRESS-ISSUE-BOUND.** The same geometry in SHARED keeps 128 registers and
   occupancy (21.5 KB/block) and genuinely cut total instructions 2966 -> 2580 (-13%,
   XMAD -20%) with bit-identical output - and measured FLAT-to-worse (+0.27%).
   The loader's address math is fully hidden under memory latency.

## Where "cut instructions" works vs does not (3 data points now)
  WORKS:   k_scan_dts (#259 -0.79% from removing 11 FMUL/step; the loop IS issue-bound)
  DOES NOT: resid relayouts (#257 - 5.6x fewer instructions, flat)
  DOES NOT: dense convs (#263 - 13% fewer, flat; they are memory/latency-bound)
Rule of thumb: instruction reductions pay only in kernels whose SASS shows the FMA/memory
mix already tight AND whose measured throughput is far below the memory roof. For the
convs, #186 already measured 41% of the fp32 roof with the op mix to match - despite the
inflated XMAD count, the kernel is waiting on memory, so integer work is free.

## #265 CLOSED: k_inorm_prelu Kahan-chain parallelism (4 chains) - negative
- Hypothesis: each thread walks its strided range with ONE serial Kahan chain, which
  serializes add latency AND memory-level parallelism. Built k_inorm_prelu4: 4 chains
  over the identical element set (chain r takes i = tid + r*NT, stride 4*NT), partials
  combined with a compensated 8-value Kahan pass so accuracy is NOT degraded to plain
  adds. Numerics held (max|d| 2.38e-07, rel_rms 5.13e-07 = 200x under the gate) and it
  used FEWER registers (28 vs 31).
- Result: flat-to-worse (+0.18%, 3 pairs). So the chain was not latency-bound: a rough
  per-thread instruction budget at ~4 IPC matches the measured 0.587 ms, i.e. the kernel
  is issue-bound at near-peak IPC. No ILP to unlock; the only remaining lever would be
  fewer instructions per element, and the Kahan adds are mandated by rule 4.
  The E[x^2]-mean^2 InstanceNorm formula (-25% traffic) would be a precision tradeoff,
  i.e. off-limits under "accuracy is a hard gate, not a tradeoff". DO NOT attempt it
  without an explicit accuracy-waiver decision.

## State of the levers after #263-#265 (three consecutive negative probes)
Every remaining kernel has now been probed for the three boundness classes:
  k_conv2d_r8    -> address-issue (#263, -13% inst, flat), barriers (#264, halved, flat),
                    shape (#197/#258, 644 optimal)
  k_inorm_prelu  -> latency/MLP (#265, flat)
  resid pair     -> instructions (#257, 5.6x, flat), float4 (#257, noise)
What remains is STRUCTURAL, not knob-sized:
  - k_conv2d_r8: split the C_in reduction into partial sums to expose ILP (a tile
    restructuring; needs a correctness harness first per trap #65(d)).
  - Winograd F(2,3): closed permanently (#185 arithmetic) unless someone produces a
    batched transform core that beats direct-conv issue efficiency.
Everything else measured this session sits at a documented floor.

## #266: conv profile revelation + a shape that produced garbage
- PER-LAUNCH conv timings (3 reps, nsys) instead of the old per-kernel AVERAGE:
    5.6 / 11.1 / 16.8 / 22.7 ms   <- encoder dense layers 0..3, linear in C_in (64/128/192/256)
    0.76 / 1.44 / 2.15 / 2.85 ms x2  <- the two decoders (same C_in ladder)
  The old "5.8 ms x 12 launches" view hid that ONE launch (dil=8, C_in=256) is
  22.7 ms = 6% of the entire model, and that all four encoder convs are the SAME
  lever at different C_in. Per-channel cost is constant.
  => Any future conv work must be evaluated against the dil-8 launch specifically,
     not against an average that dilutes it 4x.
- Weight-broadcast accounting: per ci per thread the compute does 24 LDS.128 (384 B)
  for 24 float4 that are IDENTICAL across all 256 threads of the block; the A-tile
  staging is 45 B. Weights are ~62% of the block's shared traffic and their GPU-wide
  total scales with grid_xy. Tried tile 647 = (PCO=4, NCB=2, NRA=8), R=130/132/136/144,
  which halves grid_xy at the same 64 accumulators.
- RESULT: garbage (rel_rms 1.256) + encoder 5.5x slower + 166 registers (past the
  2-blocks/SM cliff). Root cause of the garbage NOT found by inspection - R = TH+2*dil,
  hbase = h0-padH, grid (6,8,16) vs kernel steps, and shared = R*LS all check out.
  If anyone retries: FIRST isolate with tests/gpu_block_bench.c (trap #65(d)), and note
  the register cliff makes it unattractive anyway (cf. #263: crossing it cost 8.7%).

## #267 CLOSED: in_conv launch-bound sweep re-done on current code (+ the spill table)
Last swept at #61 on old code; in_conv has since gained halo skips (#154/#155), float4
stores (#180/#181) and the ICW hoist (#262). Three-way A/B, BIT-IDENTICAL output across
all three (max|d| 0.000e+00 - register allocation cannot move these fmaf chains):
    LB=2  90 regs, 0 spills,        2 blocks/SM (16 warps)  -> 0.34617 mean  (+3.7%)
    LB=3  80 regs, 16 B spill,      3 blocks/SM (24 warps)  -> 0.33393 mean  BEST
    LB=4  64 regs, 64 B spill (4x), 4 blocks/SM (32 warps)  -> 0.33623 mean  (+0.75%)
=> LB=3 confirmed; the 16-byte spill (32 stores/48 loads, in the epilogue) is NOT worth
removing. GENERAL LESSON: a small L1-cached spill is nearly free while a lost block/SM
costs 3-4% - check the blocks/SM count BEFORE chasing spills in an occupancy-limited
kernel. (#61's +0.8% for LB=4 reproduced almost exactly, so that old reading holds.)
Spill table for k_in_conv_r (useful when touching this kernel):
    KC=8   0 B (both ICW states)         KC=16 16 B spill when ICW=0, CLEAN when ICW=1
    KC=32  16 B spill (both ICW states) - the default, pre-existing

## Also noted (#267): k_x_proj_g<128,40,8> is SHARED-MEMORY-limited to 2 blocks/SM
24,192 B (As[128][36] + Bs[40][36]) x 3 = 72 KB > 64 KB. A shape with <=21.3 KB shared
(e.g. MT=96 -> 19,584 B) would allow 3 blocks/SM = 24 warps (+50%).
CAUTION: #68's shape data argues the opposite - MT=64 (4 blocks possible) was 0.7% WORSE
and MT=192 (1 block) also worse, so MT=128 looks like a local optimum where the extra
blocks are paid for with extra W-tile re-reads. KT TRIED #268: KT=16 -> 13,440 B, 3 blocks/SM
(ptxas-verified) but +1.2% SLOWER (3/3 pairs, bit-compatible): 2x chunks/barriers cost more
than +50% warps, because 20 accumulators/thread already hide latency. x_proj shape space
(MT, NT, TX, KT) is now EXHAUSTED - do not re-sweep.


## RULE (#269): the Maxwell proxy predicts CORRECTNESS, not PERFORMANCE
REUSE_FORCE_MAXWELL_TILES=1 on the dev host has been bit-exact against the Nano every
time (#193/#196/#211/#255/#260/#262) - keep using it for correctness. But in #269 its
timing had the WRONG SIGN (-2.1% on the proxy, +4.8% on the Nano). The proxy runs the
Maxwell code paths on a Pascal memory system, so memory-pattern changes do not transfer.
Any change that only affects Nano-path kernels must be timed with an interleaved A/B ON
THE NANO (same binary, knob only), e.g.:
  ssh ... 'cd /dev/shm/rp && for p in 1 2; do for k in 0 1; do CUDA_LAUNCH_BLOCKING=1 \
    REUSE_GPU_ISTFT=1 <KNOB>=$k REUSE_WS_MB=640 REUSE_MAG=... REUSE_PHA=... \
    ./engine/reuse_nano_gpu ... 3 | grep "model best"; done; done'
(#255 and #260 passed this test; #269 failed it.)
