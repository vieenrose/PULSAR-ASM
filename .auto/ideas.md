- 3-thread static-partition GEMV (OpenMP): rejected by memory wall (+4.2% slower, run #2). Revisit ONLY if per-token weight traffic drops substantially or on wider-bus hardware. Bit-identical design was proven (determinism held); patch lived in commit "rpi4: 3-thread OpenMP GEMV".
- Head-GEMV (336MB = 61% of per-token traffic) top-1 shortcut with exact recompute + error-bound candidate prefilter: needs rigorous max-miss bound proof before trying.
- Fuse elementwise passes: DONE (run #4, C-port era; kept, bits identical). Pruned 2026-10-06.
- INVESTIGATED+REJECTED (data): exact-argmax head shortcut via Cauchy-Schwarz
  row-norm bounds. Survivor fraction median 100% (bounds far too loose;
  inner products cancel, norms don't discriminate). Min margin 0.04 logits
  also rules out any lossy prefilter. Killed by survey_bounds.py, 2026-10-05.
- WALL (proven): numpy streaming-sum ceiling on this Pi = 3.93 GB/s;
  driver moves 536MB/token at 3.84 GB/s = 98% of ceiling. No identical-math
  implementation can go below ~136ms/token here. Loop converged 2026-10-05.
- Prior OMP (+4.2%) and prefetch (+19%) discards were measured cross-session
  under drift; magnitudes unreliable but directions stand (bus-bound theory +
  dual-locality failure). Retry only with same-job A/B if wall assumption changes.
- DONE 2026-10-06: HF-exact gate is now real. Root cause of every "model is
  bad" verdict was ONE asm line (`mov w0, #2` = layer 0's q_norm for all 18
  layers in fwd_token). Lesson: a pos-0 / layer-0 check cannot see q_norm at
  all (softmax over one key) - always verify with tools/fwd_ref.py over a
  multi-token sequence and diff every layer x position.
- Reference-tool trap (cost a whole session): numpy/torch refs must scale the
  embedding by sqrt(640) AND apply post_attention_layernorm to the attention
  output before the residual add. Wrong refs produced a phantom "MLP bug"
  (engine GG 184.6 vs "HF" 12.2; the true HF value is 184.6). Fixed in
  l0_full/mlp_slot/slot_brute/gg_id/gg_w/gelu_id/l0_ref.
- Working oracle: tools/fwd_ref.py (fp64, prints `DX L<i> p<p>` per-layer
  residuals + top-5 logits) reproduces transformers exactly; pair it with
  tools/perlayer_probe.patch (`patch -p0 < ...`, rebuild) to localize any
  future divergence to a layer+position in ONE run. Both are how the q_norm
  bug was found.
- MIGRATION RECIPE (2026-10-06, 270m -> 1B): the expensive part is not the maths
  but the 270m constants hiding in the forward path. Parameterized now: HID,
  INTER, NLAY, VOC, the FL entry stride, every rmsnorm n / gemv M,K / gelu n /
  argmax n, the embed scale sqrt(HID), and full attention = `i mod 6 == 5`
  (HF's default when config.json has no layer_types). Traps found the hard way:
  tab[] must hold 2+13*layers slots (236 was 270m-only), and the embed gather
  and rmsnorm_add take their length as an argument - both were literal 640.
  Always grep for the literal immediately before a call, not only in the index
  arithmetic.
- 1B numbers (Pi 4, 3 cores): 543 ms/token = 1.8 tok/s, 2.00 GB weights per
  token => ~3.7 GB/s, i.e. the same 3.93 GB/s wall as 270m. 1B is the first
  checkpoint here that answers multi-item prompts correctly (3 colors, lists,
  one-sentence explanations); 270m tops out at one-line facts.
- PARITY TOOLING that works: tools/fwd_ref.py (fp64, any checkpoint via
  FWD_MODEL), tools/l0_any.py (layer-0 only, fits any size), tools/hf_greedy.py
  (manual loop; HF_DTYPE=bfloat16 because 1B fp32 exceeds the Pi's 3.8 GB), and
  tools/perlayer_probe.patch to dump per-layer residuals from the asm.
- Next candidates: gemma-3-4b (hidden 2560, 34 layers, inter 10240) needs bigger
  bss and will not fit 3.8 GB in fp32 for the HF check; consider int8 weights
  or a 2-bit path only if the bus wall ever stops binding.
- 1B CEILING MAP (2026-10-06, temp 1.0, recommended sampling): correct and clean
  = 4-item lists with one fact each (seasons), 3-item advice (health tips),
  2-sentence child-level explanations (gravity), translation (good morning ->
  Bonjour!), one-sentence definitions (photosynthesis, minor stutters), a Python
  function with docstring, 17x3=51 (then it loops). NOT correct: word problems
  (train distance never states 120 km), sky-blue (stutter + cut), water cycle
  (stutters), zh-TW planet facts (Mercury called the farthest planet), zh-TW
  water cycle (invents a "negative cycle"). Practical rule: multi-item structured
  output is the sweet spot; numeric reasoning and multi-sentence prose are not.
- SAMPLING: the engine's sampler already matches the model card / Unsloth Gemma 3
  guidance (temperature 1.0, top-k 64, top-p 0.95, min-p 0). Stutters ("water
  water", "a a") are the checkpoint at temperature 1.0, not the engine - which is
  why demo transcripts are chosen for correctness rather than rerolled.
- CAP TUNING for demos: sampling continues past the closing line (newline tokens,
  then a hallucinated source list). Binaries search the gen cap until the last
  line is the model's own closing sentence, then diff the embedded transcript
  against a fresh run byte-for-byte.
- BONSAI TERNARY RUNTIME (state at 2026-10-06 late session):
  * tools/q2_0_gguf.py: GGUF Q2_0 -> PULSAR blob, lossless base-3 repack
    (5 trits/byte, 28 B/group) = 82.4% of Q2_0; blobs on the Pi:
    bonsai_b3.bin 376.8 MB (1.7B), bonsai4b_b3.bin 880.6 MB (4B). Runtime
    params live in the blob header (pulsar.arch_id/hidden/inter/layers/
    n_head/n_kv/head_dim/vocab/eps/rope_theta/ternary_group/pack).
  * tools/q2_0_ref.py: Qwen3 + Q2_0/base-3 oracle, VALIDATED (argmax ' Paris'
    = 12095 for "The capital of France is"; 1.7B: 140 DX signatures over 5
    positions; 4B reference runs the same way).
  * REPACK VERIFIED LOSSLESS (strongest evidence): the oracle run on the
    base-3 blob reproduces the Q2_0 run exactly - all 140 (layer, position) DX
    signatures identical (every relative difference 0.0, which is also why the
    reporter crashed: it assumed some difference would be non-zero) and both
    give ARGMAX 12095 = ' Paris'. The non-zero exit code in the task log was
    that reporter bug, not a data mismatch.
  * kernels/ternary_gemv.c: C reference, PASSES all four PTGV fixtures
    (scalar-vs-oracle 4.4e-06 .. 5.2e-05; NEON same order, ~1e-5).
  * asm/ternary_gemv.S: PASSES. Synthetic fixtures (x = all ones, so only trit
    count and scale matter) isolated the two bugs: (1) `bl half_to_float`
    without saving x30 -> ret into the row loop; (2) half_to_float ORed the
    mantissa in without `lsl #13`, so a scale of 1.5 became 1.00003 and every
    scaled sum was off by ~1/3. After both fixes the asm reproduces the C
    reference digit-for-digit on every fixture (real: 4.39e-06 / 5.22e-05 /
    8.91e-07; synthetic boundary cases: exactly 0). Harness: tests/test_ternary_asm.c
    + tests/mk_syn_ternary.py.
  * ORIGINAL DRAFT NOTE (kept for the record): First bug found and fixed: the
    kernel calls `bl half_to_float` but never saved x30, so its own ret jumped
    back into the loop (segfault). After that fix it returns but computes the
    wrong values (ASM-vs-oracle rel ~0.9-2.7 on the B3 fixtures, while the
    Q2_0 fixture passes because the harness routes fmt=0 to the C path). The
    decode loop mirrors the C reference line for line, so the next step is a
    row-0 asm-vs-C-vs-oracle dump to find where it diverges - prime suspects
    are the per-byte trit extraction (udiv/msub) and the xg indexing
    (`lsl x9, x25, #9` assumes 128 floats = 512 B per group).
  * Lesson worth keeping: any asm routine that calls a helper must save x30
    first; core.S kernels all do, this new file did not.
- QWEN3/TERNARY ENGINE PLAN for core.S (the next block; arch selected by the
  blob's pulsar.arch_id, so the Gemma path must stay byte-identical):
  1. loader: extend the dims cells with G_ARCH, G_FMT, G_NHEAD, G_NKV, G_HD,
     G_HDB, G_QDIM(=n_head*hd), G_KVDIM, G_ROPE_THETA, G_TOPK (all already in the
     blob header; pulsar.* scalars are numeric by design, no string parsing).
  2. norms: fold plain w for qwen3, (1+w) for gemma3 - one branch in fold_vec.
     q_norm/k_norm are head_dim wide (128), not HID, so FL needs two regions
     (or an entry-size table): big = HID (input/post-attn, final), small = HD.
  3. MLP: call silu_mul_f32 (done, PASS at 5.8e-08 vs double ref) instead of
     gelu_mul_f32 when arch == qwen3; same call shape, no layer-body change.
  4. attention: kv head = j / (n_head/n_kv); rope pairs = hd/2; scores over hd;
     no sliding window (lo = 0, n = pos+1); q dim = n_head*hd, k/v = n_kv*hd;
     both rope tables use the blob's theta (1e6 for 1.7B, 5e6 for 4B); no YaRN
     below 8192 (document that limit); attention scale = 1/sqrt(hd).
  5. embeddings/head: ternary GEMV row gather for the token embedding and the
     tied head (row blocks = hd/128 groups per row).
  6. bss for the 4B maxima: XB/FH 2560, FQ/FQN 4096, FKV/FKN/FV 1024, FAV 4096,
     FG/FU 9728, FL ~0.8 MB, KC/VC sized to a disclosed context cap (4096
     positions x 1024 x 4 x 2 = 33 MB; 32k positions would be 268 MB).
  7. modes: 'i' = raw ids on stdin (decimals, whitespace separated) so Bonsai
     can be driven without reimplementing Qwen's regex pretokenizer in asm;
     G_TOPK flag (the sampler hardcodes 64 today; the Bonsai card wants 20).
  8. gates: Gemma regression (bench ids byte-identical + 144-pair oracle diff)
     AND Bonsai layer/position diff vs /tmp/oracle_test.log (1.7B, 140 blocks)
     and /tmp/oracle4b_b3.log (4B, 180 blocks), then end-to-end ' Paris'.
- METRIC DRIFT RESOLVED (2026-10-07): the ~170 ms/token seen during the arch
  substeps is ambient drift, not a regression. Same-job A/B, both binaries
  interleaved: core_p5 (pre-arch, read 147-149 earlier today) 169.03 and 170.53,
  core_arch6 (arch substeps + bss resize) 170.04 and 170.89 => ratio 1.002-1.006.
  Lesson repeated: never compare absolutes across hours on this Pi; always
  interleave the two binaries in one job.
- DEMO PATH for Bonsai (what remains before a demo exists): (1) loader must read
  each tensor's dtype from the blob header ("B3_128" vs "F32") so ternary
  tensors are distinguishable from bf16 weights; (2) wire asm/ternary_gemv.S
  (already verified at C parity) for the q/k/v/o/gate/up/down projections, the
  embedding row gather and the tied head; (3) attention: kv head =
  j/(n_head/n_kv), rope pairs = hd/2, scores over hd, no sliding window,
  theta from the blob, scale 1/sqrt(hd); (4) FL two-region layout (HID-wide
  input/post-attn + final, HD-wide q/k norms); (5) ids mode ('i' = raw token ids
  on stdin) + G_TOPK so the card's temp 0.5 / top_p 0.85 / top_k 20 can be used;
  (6) then the layer/position diff against /tmp/oracle_test.log (1.7B) and
  /tmp/oracle4b_b3.log (4B), and only then the ceiling-prompt hunt + GIFs.

## Next session

Start at `.auto/HANDOFF.md` - it holds the verified state, the six remaining
steps with core.S line anchors, the Pi inventory, the build/regression commands
and the lessons. Objective and gates are unchanged.

## Spark decode session (2026-10-07, active)
- Thread the serial glue: FWHT blocks (17 max), GDN heads (48), Q-head loop (24),
  Q8-quant row-split, softmax — each is small alone; profiles first (gprof or
  timers in measure.sh) before touching. Row-split reductions only (bit-exact).
- Fused residual-add + norm epilogues (memory passes over 5120-vectors add up
  across 64 layers × several sites; measure first).
- Q8_K/Q8_0 quantize throughput (scalar now; NEON victims: max-search + scale).
- FWHT in fixed point? NO without bit-exact proof vs fork op (risky, low priority).
- Memoize transformed activations across GEMVs sharing K (fork does per-token
  memo; our engine recomputes signs+FWHT+Q8 per projection: 6+ projections share
  K=5120 inputs per token! cache transformed XQ per (layer-input) instead).
  Biggest structural win candidate after threading: ~6 repeated transforms/token/layer.
- Head GEMV (248320 rows) dominates single forwards; row-split already. Later:
  top-k preselect needs exactness proof (min margin unknown here - survey first).
- Prompt-parallel prefill is NOT applicable (causal incremental engine; batch
  prefill would change numerics - forbidden by accuracy gate).
- 8B line (fwd8/pq2) excluded from this loop; open a second session if ever.

## Q1_0 (true 1-bit) kernel — deferred ideas (Nano, 2.13 tps baseline)

Measured category map for this kernel so far:
- critical-path/occupancy changes worked on the 2-bit path (chunking +2.5%,
  fused staging+Au +4.0%, 5 blocks/SM +4.5%); instruction-count cuts gave 0.
- double-buffering the chunk tile on the 1-bit kernel REGRESSED -5.6%: the
  staging shuffle+reduction now overlaps compute, and its live ranges sit
  across the whole compute phase (register pressure / scheduling), while the
  single-buffer barrier was evidently not the bottleneck here.

Next candidates, in expected-value order:
1. Two groups per lane (GperThr=2 for K=4096): halves the per-lane epilogue
   (reductions, product stores, the popcount tail) per group and gives two
   independent dependency chains. Costs registers; watch the 48-reg/5-block
   budget.
2. Vectorize the act loads: the inner loop currently issues 32 LDS.32 per
   128-weight group; one LDS.128 per 4 nibbles cuts that to 8 and removes
   ~24 instructions per group.
3. Bank-conflict check on the act tile reads (lane sub reads its own group's
   128 bytes; 16 lanes stride 32 words apart -> likely 2-way conflicts).
4. Arena right-sizing: 1280MB default vs the measured need (~1.19GB + KV);
   consider growing the arena in chunks so the reserved GPU RAM tracks the
   model instead of a fixed cap.
5. The PQ1GPU exit stats line does not reach stderr in the 1-bit build (the
   2-bit build printed PQ2GPU), so gpu_used/peak are not on the dashboard for
   this workload. Register the atexit dump inside the glue TU instead of the
   shim's constructor.

## Measured-out micro-levers on the Q1_0 kernel (do not re-walk)

- Instruction-count cuts: 0% (the 2-bit E4 result; the 1-bit kernel is at
  ~29% of the ALU issue peak, so issue slots are not the constraint).
- Shared-memory bank conflicts: the act tile had a REAL 2-way conflict on
  every load (lanes 2m/2m+1 read words 32 apart = same bank since 32 % 32
  banks = 0). Padding the tile to 69 words per Q8_K block with the halves at
  words 0/33 removed it - and bought nothing (2.1349 -> 2.1071 tps, inside
  noise, discarded). Conclusion: LDS throughput has slack; shared memory is
  NOT the limiter, so further LDS/vectorisation work is low value.
- Overlap machinery (double-buffered staging): -5.6% (live ranges across the
  compute phase cost more than the overlap saves).
- So the remaining lever is the DEPENDENCY STRUCTURE / per-chunk fixed cost:
  two groups per lane (halves the per-chunk barriers and staging passes per
  unit of work, adds a second independent chain) is the untested candidate
  and needs a full work session - it touches the staging thread mapping
  (256 threads / 32 groups = 8 threads per group), sh_prod sizing and the
  ordered reduce, and it risks the 48-reg / 5-block occupancy budget.

## Closed direction: bigger blocks / more overlap (Q1_0 kernel)

Three structural attempts on the 32-row/512-thread amortisation idea and its
cousins, all measured on the same harness:
- 512-thread / 32 rows per block (same tile reused by 2x the rows, staging
  and barrier cost per unit of work halved, inner loop untouched): **-9.1%**
  (2.1349 -> 1.9404 tps). A 16-warp barrier converges worse than an 8-warp
  one, and this part is a single-SM (128-core) GPU, so coarser blocks just
  add arrival spread.
- double-buffered staging: -5.6% (live ranges across compute).
- bank-conflict-free padded tile: ~0% (LDS has slack).

Conclusion: the 256-thread / 16-row / 5-blocks-per-SM shape with one group per
lane is a local optimum for this kernel. What remains untested is the DEPENDENCY
STRUCTURE inside a lane (two groups per lane, i.e. 32 groups per chunk, with
the staging remapped to 8 threads per group): it amortises the per-chunk cost
without growing the block, but it needs the inner loop duplicated (I-cache risk
on this GPU) and it competes for the 48-register budget. Attempt it only with a
full session and a fallback plan (keep the current commit as the reference).

## Measurement method for this Nano harness (2026-10-13, run #22)

- A SINGLE harness run is a cold run and reads ~6% slower than the same binary
  run back-to-back in a warm sequence (413 vs 437 ms/tok on the same code).
  Board level also drifts across hours: the 2.3410 tps sample and a later 2.28
  sample of *identical* code differ by 2.2%, while a same-job interleaved A/B
  of the two binaries gives -0.10%. Conclusion: never compare a run against a
  number recorded in an earlier session; the only valid comparison for a change
  inside the noise band is an interleaved A/B in one job (2 runs per binary,
  alternating). Harness built for this: .auto/ab_stats.sh (builds the git HEAD
  glue next to the working-tree glue, links q1_old/q1_new from shared objects,
  runs 4 alternating runs, prints the ratio).
- The Q1_0 GPU-RAM baseline is now on the dashboard (was blank for every
  earlier run - the 1-bit TU printed the label 'PQ2GPU', the harness greps
  'PQ1GPU'): used = peak = 1014.8 MB, arena = 1280 MB. Right-sizing the arena
  default to ~1152MB (peak + 13%) is the next cheap secondary win; gates protect
  correctness (too small => CPU fallback => the goldens fail).
- gpu_free_mb reads 0.0 at exit. Verify what cudaMemGetInfo means on Tegra once
  the arena is allocated before trusting that number for anything.

## Arena right-sizing: MEASURED OUT (run #23, 2026-10-13)

Idea: the dashboard now shows gpu_used = gpu_peak = 1014.8MB of a 1280MB arena,
so drop the default to 1152MB (peak + 13.5%) for -128MB of reserved GPU RAM.

Result: primary neutral (+0.19% tps, inside noise) and startup -0.15s, but a
6-run interleaved A/B (.auto/ab_rss.sh) gives host peak RSS 342.0MB vs 262.3MB
= **+79.7MB**, reproducible 3/3 (new: 343/342/341, old: 278/229/280).

Why it backfires: on Tegra's unified memory a cudaMalloc comes out of the same
physical RAM as the page cache. Releasing 128MB of device reservation lets the
mmap'd GGUF pages become resident, so the GPU-RAM "saving" is paid out of
peak_rss_mb - the metric the objective names. Device reservation and host RSS
are ANTI-CORRELATED on this board, not additive.

Consequences to keep:
- Do not size the arena down to chase RSS. On a board with real discrete VRAM
  this becomes a genuine win; there, re-measure with ab_rss.sh.
- The RSS lever on this board is residency of the mmap'd GGUF (run #2's
  madvise(MADV_DONTNEED) after upload), not the arena.
- A bigger arena makes RSS *deterministic* (spread 2MB vs 51MB): it squeezes the
  page cache. Single-run RSS on this board swings by 50MB, so RSS changes must
  be A/B'd in one job, exactly like timing changes.

## Cache policy: MEASURED OUT (run #24, 2026-10-13)

Idea: the 1.06GB/token weight stream is read once, so __ldg (cache-all) evicts
the reused act/scale data from Tegra X1's 128KB L2. Switch the code + scale
loads to __ldcs (evict-first). Bit-identical by construction (parity green).

Result: interleaved A/B, 3 runs/side (.auto/ab_kernel.sh, new tool - swaps the
kernel .cu built from git HEAD vs the working tree against shared host objects):
decode 2.3285 vs 2.3367 tps = -0.35%, inside the new side's own spread. RSS
-18.7MB mean is a single 228MB outlier (median ~-4MB). No gain; reverted.

The L2-thrash theory does not apply here: the weight words are consumed by one
lane exactly once, so there is no reuse to protect, and __ldg was already
non-coherent. This closes the last resource axis on this kernel - see the
running list in "Measured-out micro-levers" above: issue slots 0, LDS/banks 0,
overlap -5.6%, block shape 0 (saturated at 64 rows), unroll -2.8%, cache policy
-0.35%. What is left is a different inner-loop ALGORITHM, not a tuning knob.

## Scheduling / tail-wave: MEASURED OUT (run #25, 2026-10-13)

Idea: this is a single-SM part with 5 resident blocks/SM, so any GEMV whose row
count is not a multiple of 5*64 = 320 ends in a partially-filled last wave.
Computed waste: 1.54% of work for N=4096 and N=12288, 20% for N=1024, ~2.3% of
GEMV time weighted by rows*K. Fix: grid = ceil(tiles/5)*5, each block takes a
contiguous ceil(N/G) row range and loops over 64-row tiles.

Result: bit-exact (parity green) and resource-neutral (ptxas: 40 registers,
0 spills, same 6336B smem) - and -7.65% tps in a 3-run interleaved A/B
(new 445/453/458 ms vs old 412/412/428 ms; every pair separated).

WHY IT FAILS - the arithmetic that kills it: all the large matrices have row
counts that are multiples of 64 (4096, 12288, 151669), so ceil(N / (5*ceil(N/320)))
= 64 = PQ1_BLOCK_ROWS, i.e. every block gets exactly ONE tile (ntiles = 1) and
the "even" mapping degenerates to the mapping it replaces. The scheme cannot
create fractional work: the work quantum per block is 256 threads / 4 lanes per
row = 64 rows, so a partial last wave is structural. All that remains is the
loop's cost (dynamic trip count, per-tile row/active/rrow re-derivation, the
64-bit row*K128 scale-address multiply).

Consequence: recovering that 1.54% requires a kernel whose work quantum per block
DIVIDES N (lanes splitting a row's 16 groups, so rows/block is not quantised by
256/lanes_per_row). That is the same structural rewrite as the open "different
inner-loop algorithm" item - one session-sized task, not a tuning knob. Small-N
matrices (k/v, N=1024, 20% tail) are 0.11% of per-token weight traffic, so they
do not justify a special case.

Note for future candidates: the tail-wave arithmetic (N vs 5*rows_per_block) is
worth checking on the calculator BEFORE writing the kernel - the whole experiment
was predictable from ceil() identities plus the model's row counts.

## Magnitude corrections from drift-free A/B (run #26, 2026-10-13)

The 64-rows-vs-32-rows win (run #19) was recorded as +7.6% from a cross-session
pair. Same-job interleaved A/B (3 runs/side, all pairs separated) says **+4.20%**
(2.3353 vs 2.2411 tps). The win is real; the number was not.

Rule adopted: a kept change's percentage is only trustworthy if measured with
the two binaries in ONE job. `GITREF=<commit> bash .auto/ab_kernel.sh 3` replays
any kernel-TU comparison that way.

Audited so far: #19 -> +4.2% (was +7.6%).
Presumed overstated (measured cross-session, not yet replayed): #7 (+4.5%),
#9 (+4.0%), #17 (+2.0%). Orderings are unaffected - #7's 5-block optimum was
independently reproduced by #8, and #17 was confirmed by #18.

Side observation from #26: the 64-row kernel is also more DETERMINISTIC than the
32-row one (sd 1.23ms vs 5.46ms over 3 runs), consistent with its better wave
fill. Run-to-run sd is itself a usable signal for launch-shape changes.

## Occupancy beyond 40 warps: MEASURED OUT (run #27, 2026-10-13)

ptxas says the kernel naturally uses 40 registers, and 6 blocks/SM needs <=42
(65536/(256*6)), so launch_bounds(256,6) buys 48 warps (62.5% -> 75% occupancy)
for FREE: confirmed 40 regs, 0 spills, identical 6336B smem.

Interleaved A/B: n=3 gave +1.28%, n=6 gave +0.47% (sd 5.22ms, ~0.7 sigma). Not
significant; the n=3 signal was one 417ms outlier. Reverted to 5 blocks.

Also CORRECTS run #8: its -0.7% rejection of 40 regs/6 blocks was drift, not a
regression. The supported claim is "occupancy beyond 40 warps does nothing", not
"40 is the sweet spot (48 is worse)".

Consequence: the kernel is latency-SATURATED, not latency-starved. Every added
latency-hiding mechanism now measures ~0 or negative - unroll x2 (-2.8%), double
buffering (-5.6%), cache hints (-0.35%), extra warps (+0.47% n.s.). Remaining
headroom must come from doing LESS work (the inner-loop rewrite), not from
scheduling the same work better.

BLOCKER for finding that work: GPU performance counters are denied on this Nano
(nvprof -> ERR_NVGPUCTRPERM, no stall-reason metrics in the Tegra metric set).
Asking the operator to enable profiling (root, or the Tegra perf access toggle)
is the highest-leverage request available: it is the only way to see where the
~40% issue-efficiency gap actually goes, and every structural idea so far has
had to be argued from first principles.

## Instruction sensitivity: the "not issue-bound" claim is OVER-STATED (run #28, 2026-10-13)

Measured with a deliberately numerically-wrong probe (inner loop w<4 -> w<2,
logged as checks_failed so it auto-reverts): halving the inner loop's
instructions cuts decode time 28.3% (296.8 vs 414.0 ms, interleaved A/B).

Slope ~= 0.6% time per 1% inner-loop instruction cut.

CORRECTION to "Instruction-count cuts: 0% ... issue slots are not the
constraint" (written after E4): that was true of the mask/keep ALU chain
specifically, which overlaps with stalls and is free to remove. It does NOT
generalise. The dependency-carrying part of the inner loop (the w-loop that
consumes code words) pays ~0.6% time per 1% of instructions.

CONSEQUENCE: the table-based inner loop is now a MEASURED bet, not a guess.
  - Enabling fact: the inner accumulation is INTEGER (pu0/pu1 int accumulators,
    isum = 2*Pu - Au + 16384 - 256*pc), so partial sums reassociate EXACTLY.
    A lookup table of integer partial sums over the int8 Q8_K activations is
    bit-identical to the bit-serial loop. A FLOAT table would not be - it would
    break the parity gate.
  - Shape: index by 2 bits -> 4 entries per 2-weight position, ~2 LDS + 2 IADD
    per 8 weights vs ~14 ops today (1.75 -> ~0.5 ops/weight).
  - Table build: once per token, in global/L2, shared by every block (it depends
    only on xq and the K range). K/2 positions x 4 ints = 2K floats: 32KB at
    K=4096, 96KB at K=12288 -> chunk K=12288 to stay inside the 128KB L2.
  - Upper bound from the probe: ~28-40% (the probe cut instructions AND byte
    traffic together; the rewrite cuts instructions only, so expect less).
  - Floor: 0, if time tracks weight bytes rather than instructions.
This is the highest-value remaining work, and it is a full-session task: build
pass + new inner loop + parity proof BEFORE any timing claim.

## Table-based inner loop: IMPLEMENTED, BIT-EXACT, AND 38% SLOWER (run #29, 2026-10-13)

The session's biggest structural bet, executed in full (kernel inner loop + host
build + glue upload + parity call site). Numerically PERFECT on the first try:
parity 256/256 @K=4096, 512/512 @K=12288, both goldens exact.

Performance: decode 1.4407 vs ~2.34 tps = ~-38%. REVERTED.

WHY: the table trades cheap ALU for expensive memory. Per lane per group it
issues 32 two-byte loads (vs 4 uint4 code loads + 32 tile-word reads), scattered
across ~16 distinct 32B sectors per warp step. The ops #28 showed as removable
were ALU ops largely OVERLAPPED WITH STALLS (that is exactly why E4's 30% cut
bought 0%), so removing them is nearly free; the ops added land on the LSU/latency
path, which is the actual bottleneck. #28's slope is real for ALU instructions
and DOES NOT convert into wins when the replacement is paid in memory ops.

CLOSED IN BOTH DIRECTIONS: 2-bit tables = 2 ops/weight (worse than the 1.75
baseline, and |value| fits int16); nibble tables = 4 ops per 4 weights (good ALU)
but 2 bytes of table traffic per weight and 32 loads/lane/group. Any table wide
enough to beat the bit-spread needs >=2 bytes of table traffic per weight, which
costs more than the ALU work it removes. Staging in shared makes it worse: the
chunk slice is 16KB (K=4096) -> 14.6KB/block -> 3 blocks/SM (32 warps), priced by
#7/#8 at ~-4.5% per block lost, and it still pays 2 bytes/weight of LDS.

REFRAMING FOR ANY FUTURE ATTEMPT: "fewer ALU instructions" is a NON-GOAL on this
part (proven twice: E4 = 0%, #29 = -38%). Remaining headroom must reduce MEMORY
OPERATIONS PER WEIGHT per unit of work, or shorten the dependency chain without
adding loads. Concretely that means keeping one wide load per lane serving many
weights (the current uint4 = 32 weights per load is already 8 bytes/weight of
ALU-facing work) - i.e. the burden of proof is on any design that increases the
load count per weight.

ASSET RECOVERABLE FROM THIS RUN (see ASI code_note for the exact index formula):
a bit-exact nibble-table build (Q8_K 73-word blocks, 0x80 bias) - if a future
architecture has a cheaper path (e.g. a hardware LUT/PRMT-based byte table on a
part with real LUT throughput), the numerics are already proven.

## Vectorized act loads (LDS.32 -> LDS.128): MEASURED OUT (run #30, 2026-10-13)

Backlog idea #2 - the last unimplemented item from the session's original
"known optimization directions" list. Implemented fully, bit-exact on the first
try (parity 256/256 @K=4096, 512/512 @K=12288, both goldens), chain assignment
preserved by construction.

Result: -0.46% tps in a 3-run interleaved A/B, all of it from one 424ms outlier
(medians 417.05 vs 417.47 ms = a wash). Reverted.

WHY: at 40 warps/SM the SM is idle-issuing, so shared-memory issue slots are not
scarce. Two independent experiments now bound this axis:
  #15  removed all bank-conflict STALLS (op count unchanged) -> 0%
  #30  removed 75% of LDS OP COUNT (conflicts irrelevant)   -> 0%
The old "LDS has slack" note is now proven from both directions.

## Cumulative bottleneck model (runs 13-30)

Everything the kernel does on-chip is measured as SLACK:
  ALU / instruction count ......... E4 -30% instructions -> 0% ; #29 -57% -> -38%
                                    (the removed ALU ops were already overlapped
                                     with stalls; adding memory ops is not free)
  Shared memory (conflicts) ....... #15 -> 0%
  Shared memory (op count) ........ #30 -75% LDS ops -> 0%
  Occupancy / warps ............... #27 +20% warps -> +0.47% (n.s.)
  ILP via unroll .................. #21 -2.8% ; #14 double-buffer -5.6%
  Block shape / amortisation ...... saturated at 64 rows (#20: +0.1%)
What is NOT slack: GLOBAL loads. #29's 32 extra 2-byte global loads per lane per
group cost 38% of total time. Cross-check: 1.06GB/token needs ~41ms of DRAM
transfer time, yet decode is ~427ms => ~10x the transfer time, i.e. the kernel
is exposed to global-load LATENCY, not bandwidth.

The only untested fix consistent with this model is putting more code loads in
flight WITHOUT paying #21's price (2x code + live ranges): hoist the second
group's uint4 load above the compute of the first (the half loop has 2 iterations,
g = sub and sub+8), so both LDGs issue ~2000 cycles before use. Costs ~4 registers
(ptxas reports 40 today; 51 is the 5-block ceiling), no code duplication.
Prior evidence is discouraging: the two latency-hiding attempts so far (#14, #21)
both regressed, because both bought latency by adding live ranges at the 40-reg
boundary. A 1-deep hoist is the cheapest remaining test of the model.

## Latency hiding: CLOSED (run #31, 2026-10-13)

1-deep hoisted prefetch of the code stream (issue group h+1's uint4 before
computing group h; last step's load skipped - warp-uniform predicate).
Register-free: 40 regs, 0 spills, no code duplication. Bit-exact.
Interleaved A/B 3 runs/side: -0.61% tps, inside noise. Reverted.

This is the THIRD latency-hiding mechanism to fail, and the first at zero
resource cost (#21 unroll -2.8% and #14 double-buffer -5.6% both paid registers
and live ranges, which plausibly explains their losses). A null with no cost
means the assumed ~700-cycle global-load latency is ALREADY hidden by the
hardware at 40 warps. So the ~10x gap between DRAM transfer time (~41ms for
1.06GB) and decode (~427ms) is NOT exposed load latency. The #30 model needs
this correction.

UNTESTED CONSEQUENCE (next candidate): DRAM REQUEST LOCALITY. At 4 lanes/row a
warp issues 8 rows x 64 contiguous bytes per load step, with a 512B (K=4096) row
stride. Giving each lane its TWO ADJACENT groups (g = 2sub, 2sub+1, 2sub+8,
2sub+9) concentrates each row's 128 bytes into 2 consecutive half-steps instead
of 4 spread across the chunk - a pure lane-to-group permutation: identical load
count, identical compute, identical code; bit-exact by construction because the
ordered reduce reads sh_prod by GROUP INDEX.

## DRAM-side reorganization of the same stream: CLOSED (runs #31-#32, 2026-10-13)

Two zero-cost mechanisms, both bit-exact, both null in interleaved A/B:
  #31  1-deep hoisted prefetch (issue next group's uint4 early)   -0.61%  (n.s.)
  #32  adjacent-pair lane mapping (row's 128B in 2 steps)        +0.65%  (0.35 sigma)

#31 removes the latency-hiding family (unroll -2.8%, double-buffer -5.6%,
prefetch -0.61%): with the register/code cost eliminated, latency hiding is
provably NOT the lever - the ~700-cycle global-load latency is already hidden at
40 warps. #32 removes DRAM page-locality: the mean gap was 2.7ms against ~8ms sd.

NET RESULT: every mechanism that reorganizes the SAME 1.06GB/token stream -
prefetch, request locality, LDS vectorization (#30), cache policy (#24),
occupancy (#27), instruction count (#29 in the cheap direction) - measures |effect|
< 1%. The only perturbation that ever moved decode time was #29 ADDING traffic
(-38%). Conclusion for the session: this kernel is at a LOCAL OPTIMUM with
respect to scheduling, memory ordering, cache policy, occupancy and instruction
count. Remaining headroom is algorithmic (reduce bytes touched per token) or
hardware-bound; it is not schedulable.

Note on #32's 2.3737 tps harness sample - the best single sample of the session.
It is NOT a win: the interleaved A/B of the same two binaries put it at +0.65%
at 0.35 sigma. This is the clearest illustration of why single-sample "wins" are
not admissible on this board.

## Cumulative gain audited (run #33, 2026-10-13)

GITREF=902dabc ab_kernel.sh 3 - HEAD kernel vs the run-#13 baseline kernel, one
interleaved job: +10.71% tps (2.4069 vs 2.1740), every pair separated by ~45ms
against ~1ms spread (t ~ 25).

The recorded headline (+9.7%, from #19's cross-session arithmetic) is CONFIRMED
and slightly UNDERSTATED. Consistent with #26: the isolated 64-vs-32-step was
overstated (+7.6% recorded, +4.2% real), but the cumulative chain was not - the
individual overstatements did not compound, because the baseline itself measured
slower in the same job than its own historical sample (2.1740 warm vs 2.1349 cold).

CAUTION FOR THE DASHBOARD: 2.4069 tps is a WARM-SEQUENCE number. The tool's
dashboard compares it against the COLD baseline 2.1349 and prints +12.7%; that
figure mixes conditions (cold-vs-warm is ~6% on this board) and should NOT be
quoted. The defensible statement is:

  decode is ~10.7% faster than the Q1_0 baseline kernel, measured drift-free.

Audit series status: #19 (isolated step) -> +4.2% real; cumulative chain -> +10.71%
real. Remaining unaudited kept claims: #7 (2-bit path, +4.5%) and #9 (2-bit path,
+4.0%) - different kernel TU, lower value, replay with GITREF if ever needed.

Kernel is at a LOCAL OPTIMUM (#32). Every zero-cost reorganization of the same
1.06GB/token traffic stream is null: prefetch -0.61%, locality +0.65%, LDS ops
-0.46%, cache hints -0.35%, occupancy +0.47%, instructions 0%. Only ADDING
traffic moves it (-38%, #29). Future headroom is algorithmic (fewer bytes per
token) or hardware-bound - not scheduling.

## gpu_free_mb was a FAILED QUERY, not a measurement (run #34, 2026-10-13)

The dashboard printed gpu_free_mb=0.0 on every Q1_0 (and PQ2) run. The binary's
own line also printed total_mb=0.0 - impossible for a working query - so
cudaMemGetInfo was FAILING: the stats printers run from atexit, after CUDA's own
atexit handler has destroyed the context, and the return code was never checked.
Both fields kept their `size_t x = 0` initialisers, which the dashboard then
reported as "0 MB free" on a 4GB board.

Fix: sample once at arena allocation (context provably alive), print NA when
unmeasured. Real values: free=412.5MB, total=3965.2MB == /proc/meminfo MemTotal,
i.e. CUDA's total on Tegra is the WHOLE unified pool, not a carveout.

NEW FACT for arena decisions: after the 1280MB arena cudaMalloc, 412.5MB of the
3965MB pool is free. This is the quantified form of #23's mechanism - the arena
and the page cache draw on one pool, so arena size trades against peak RSS, it
does not add to it. Do NOT size the arena down to chase RSS on this board.

Debug technique worth keeping: when a metric reads a physically impossible value
(total memory = 0), the API call failed and the value is the initialiser. Check
return codes before trusting printed diagnostics - and prefer a value that cannot
be a measurement (NA) over a default that looks real (0.0).

## ALU ops ARE a profitable currency when paid in ALU (run #35, 2026-10-13)

Packed-accumulator inner loop: the 4-byte masked sum is (ap & keep) summed into
the TOP 16 bits of the accumulator by ONE IMAD with 0x00010001, replacing the
9-op two-stage SWAR fold. 14 -> 11 ALU ops per 4 weights. Bit-exact by integer
associativity (parity green, both goldens, top=911). Resources IDENTICAL (40
regs, 0 spills, same smem). Interleaved A/B: **+10.01% decode** (2.5881 vs
2.3526 tps), every pair separated, t ~ 13. Largest single-step win of the
session; cold harness 2.5499 tps, prefill +11.7%.

SUPERSEDES #32's "local optimum" conclusion. The correct model is a currency
model, not an optimality claim:
  - #30 cut LDS ops  -> 0%  (LDS is not the binding currency)
  - #29 cut ALU ops, paid in LOADS -> -38% (paid in the scarcer currency)
  - #21/#14 bought ILP, paid in REGISTERS/LIVE RANGES -> -2.8% / -5.6%
  - #35 cuts ALU ops, paid in ALU (IMAD is full-rate, zero resource cost) -> +10%
Every earlier failure paid for the change in a resource the kernel was short of.
#28's slope (0.6% time per 1% inner-loop instruction cut) is confirmed in the
profitable direction: measured ~0.44% per 1% op cut (14 -> 11 ops = 9.1% time).

MINE IT FURTHER (untested): the keep-mask chain is still 3 ops (IMAD
b4*0x00204081, AND 0x01010101, IMAD *255). The *255 byte-expansion looks hard to
remove - the 0x01-per-lane form loses the activation byte values - so the floor
for this formulation is ~11 ops/4 weights, i.e. this loop is close to mined out.
The 2-bit kernel (pq2_swar_sm53.cu) has the same 9-op usum4 chain (E4 counted
539 ANDs) and has NEVER had this treatment: apply the packed accumulator there.

DO NOT re-walk from the pre-#35 mindset: #29 (tables) and #30 (LDS vectorisation)
are measured out - they failed for reasons this run CONFIRMS (wrong currency),
not reasons this run changes.

## Slope re-calibrated after #35 (run #36) + gate integrity bug (run #37)

SLOPE (diagnostic probe, kk<4 = half the inner loop, checks_failed by design):
386.45 -> 287.73 ms/tok = -25.5% decode time, every A/B pair separated
(sd 0.6-3.1ms). => 0.51% time per 1% op cut (was 0.57% pre-#35, #28).
The ALU currency is NOT saturated by #35. The per-nibble ALU chain is ~51% of
decode time, so the remaining 11 ops/4 weights are worth ~4.6% of decode time
per op removed. A primitive that removes 3 ops is worth ~14%.

WHY 11 OPS IS NEAR THE FLOOR (analysis, not measured - do not spend a session
re-deriving it): extract 2 (SHR+AND per nibble), mask spread 3 (IMAD
b4*0x00204081, AND 0x01010101, IMAD *255), AND 1, fold 4 (AND, SHR, AND, ADD),
accumulate 1 (IMAD). Examined and rejected: PRMT byte-select (cannot broadcast
bits), 64-bit 8-bit spread (costs more than it saves), LOP3-fused masks (equal,
already 1 op per AND), single-IMAD fold of unpre-scaled bytes (INEXACT: 4 biased
bytes sum to <=1020, so the top byte overflows mod 256 - the 16-bit lane room is
what makes #35's accumulator exact).

GATE INTEGRITY (the session's most important harness finding): checks.sh never
built anything. It ran ./pq1_parity and ./fwd8_q1 as the last build left them, so
any kernel-TU change measured through ab_kernel.sh (which builds its own
q1_old/q1_new objects) got a VACUOUS green gate. Proof: a kernel that ignores
half its weights passed all three gates, then failed 0/256 after a rebuild.
Fixed in #37: .auto/build_q1.sh (canonical copy of measure.sh's build recipe) is
run as checks.sh step 0; verified by re-applying the probe -> gate fails.

NOTE for future runs: gate 2's prompt-A check diffs .auto/last_run_log.txt,
which only measure.sh writes. A standalone checks.sh after a probe run therefore
fails prompt A legitimately. Run the harness, then checks.sh (what the tool does).

MOTIVATION FOR #35 THAT IS NOW MEASURED: #35's 14->11 op cut predicted ~12-13%
from the 0.57 slope and delivered 9.1% (0.44 slope) - the slope is real but
shallower for partial cuts than for the full-chain probe, consistent with the
loop containing non-reduced work (loads, addressing, loop arithmetic).

## 2-bit kernel: packed accumulator lands +12.38% (run #38, 2026-10-13)

#35's trick ports verbatim: pq2_swar_sm53 runs the 9-op usum4 fold FOUR times per
4 weights (two 1-bit planes x even/odd accumulators) -> AND + 4-op fold + 1 IMAD
each, ~29 -> ~21 ALU ops. Bit-exact at 256/256 @K=4096, 512/512 @K=12288 and
4096/4096 @K=4096 (maxrel 0.00e+00) on fresh binaries. 40 regs, 0 spills.

MEASURED AT KERNEL LEVEL (+12.38% speed, 1.6362 vs 1.8387 ms/launch, 3 runs/side,
spread <0.2%): the 2-bit GGUF (b8.gguf) is GONE from the Nano - it was deleted
when the objective retargeted to Q1_0 - so fwd8_engine cannot run end-to-end and
its decode_tps is permanently unmeasurable on this board. pq2_parity.c now carries
a PQ2_TIME=<iters> mode for exactly this situation (times N launches of the real
kernel on the real data; launch overhead included in both sides => conservative).

MODEL VALIDATION (the important part): the slope model (0.44-0.51% decode time per
1% inner-loop op cut, calibrated in #28/#36) PREDICTED ~12% for this ~28% chain cut
before any measurement; measured +12.38%. That is the first prospective prediction
of a speedup in this session, and it landed within 0.4 points. The slope model is
predictive, so future op-level changes can be sized on paper before burning board
time - and a change worth <2% on paper is not worth measuring at all.

BOTH GEMV KERNELS ARE NOW AT THE OP FLOOR for this algorithm family: 1-bit ~11
ops/4 weights (#35), 2-bit ~21 (#38). Remaining primary-metric headroom = ~4.6% of
decode time per inner-loop op removed, and the floor analysis (in the #36/#38 ASI
notes) says multiplication cannot SELECT bytes (cross terms between byte lanes) and
the 16-bit-lane fold is what makes the packed accumulator exact. A cheaper primitive
is a research question, not a tuning knob.

PITFALLS HIT AND CLOSED THIS SESSION (do not re-hit):
- C comments containing "*/" inside text like "pu*/mu*" silently close the comment.
- NEVER trust a parity line next to a failed build: rm -f the binary first, verify
  the object/binary exist, then read the gate (#37's lesson, re-learned in #38).
- A/B drivers must pass log PATHS to their parsers, not tags, and must link objects
  built with identical -D flags (the 2-bit path needs -DPQ2_HIST consistently).

## #35's win confirmed at n=5 (run #39, 2026-10-13)

GITREF=a3b3abe^ vs HEAD, one interleaved job, 5 runs/side:
new 2.6120 vs old 2.3393 tps = **+11.66%** (382.85 vs 427.48 ms), every pair
separated by >=31ms, Welch t~14, 1sigma ~0.8% absolute. #35's +10.01% (n=3) was
within ~2 sigma; RSS identical to 0.1MB (341.4 both sides) - third confirmation
the packed accumulator is resource-neutral.

SLOPE MODEL CLOSES PROSPECTIVELY ON BOTH KERNELS:
  1-bit: predicted -10.9% time (21.4% chain cut x ~51% chain share) -> measured
         -10.44%. 2-bit (#38): predicted ~12% -> measured +12.38%.
The model is accurate to ~5% relative, so op-level ideas can be sized on paper.

SIZING RULE for any future candidate: removing ONE op from the 11-op 1-bit chain
is worth ~4.6% of decode time. A candidate that cannot provably remove >=2 ops is
below the measurement floor - do not spend board time on it.

TWO IDEAS BOUNDED ANALYTICALLY (do not spend a session re-deriving these):
1. Bit-slicing the int8 activations (T-MAC/BitBLAS style: sum_j 2^j * popc(plane_j
   AND code_bits)). Int8 needs EIGHT planes, so per nibble it costs 8 LOP3 + 8 POPC
   = 16 ops before the weighted accumulation - versus 11 for the current form.
   Bit-slicing pays when the ACTIVATIONS are binary (1 plane); here the weights are
   the binary side (already 1 bit = the current AND-based selection) and the
   activations are the wide side, so the roles cannot be swapped profitably.
2. Offloading fold ops from the ALU pipe to the XU/VA pipe (VADD4/VAADDLE4, native
   on Maxwell). XU is 1/4 rate: moving 1 op per nibble costs ~59M warp-ops =
   ~236M XU cycles to save ~16M ALU cycles, and the XU pipe already carries ~32ms
   (7.4M warp-ops at 0.25/cycle) of the 394ms budget. Net loss.

Standing on the primary metric: ~2.53-2.61 tps, all scheduling/cache/occupancy/
instruction-count levers closed with data (#15/#24/#27/#29/#30/#31/#32/#35/#36/#38),
both GEMV kernels at their op floor for this algorithm family.

## The engine is DVFS-bound, not ALU-bound (2026-10-14, run #42)

Built the T-MAC bit-sliced 1-bit GEMV (run #41's plan) and it is **bit-exact at
every shape** (256/256 @K=4096, 512/512 @K=12288, 1024/1024, 151669/4096,
maxrel 0.00e+00) and **1.36-1.53x faster per launch in isolation** (PQ1_TIME in
pq1_parity, all launch-gap regimes 0/200µs/1ms/3ms). In the engine it is
**-1.8%**. That contradiction is the finding.

Measured mechanism: the GPU devfreq governor. Sampling
`/sys/devices/gpu.0/devfreq/57000000.gpu/cur_freq` during engine decode gives
**min 76.8 / median 460.8 / max 844.8 MHz**; a standalone kernel loop runs at
max. The engine issues ~253 GEMV launches per token, each ~0.35-2.5ms, separated
by CPU work (quantize, rope, softmax) — bursts too short for the governor to
boost. So in-engine the kernel is **launch/latency-bound at ~55% clock**, where
removing ALU ops buys nothing; standalone it is ALU-bound at 100% clock, where
T-MAC's 3.3x op cut shows up in full.

Per-shape in-engine time per call (SWAR -> T-MAC): 4096x4096 0.809->1.010ms,
1024x4096 0.348->0.604, 12288x4096 1.979->2.085, 4096x12288 1.945->2.399,
151669x4096 21.92->20.18 (the only win; 16% of decode). Net = -1.8%.
`pq1_fwd8_glue.c` now carries PQ1_KTIME=1 (per-shape device-path timing) and
PQ1_KTIME=2 (per-call phase timing: host plane build is 0.2µs — free), plus a
tm=/swar= call counter in the PQ1GPU stats line. The T-MAC kernel is kept behind
**PQ1_TM=1** (default off) so the kernel stays available for study.

THIS SUPERSEDES the "kernel is at a local optimum / is latency-bound" reading of
runs #15-#32: those experiments all measured in-engine, i.e. at ~460MHz. Any
kernel-level lever must be priced at the ENGINE clock, not at the standalone one.

ACTIONABLE (the first engine-level lever the session has found):
1. **Fewer, bigger launches.** Batch the projections that share an input vector
   into ONE launch: q+k+v (same K=4096 input), gate+up (same input). 7 GEMVs per
   layer -> 4, i.e. 253 -> ~150 launches/token, each ~2x longer. Longer bursts
   should hold the governor up AND cut per-call CPU overhead. Numerics unchanged
   (each output row's math is identical); needs a batched kernel taking a small
   array of {codes, scales, rows} plus a row->tensor map, and the arena may need
   to place grouped tensors adjacently.
2. **Overlap CPU work with the GPU** (double-buffer d_xq/d_y, async stream, sync
   once per layer instead of per GEMV). Keeps the GPU fed; the dependency chain
   (layer N+1 needs layer N's output) still allows one-deep pipelining.
3. Neither is a numerics change, so the three gates apply unchanged - and both
   are worth more than any remaining inner-loop op cut.

## Keep-warm / DVFS: REFUTED as the engine's bottleneck (2026-10-14, run #43)

Follow-up to #42's clock reading (461MHz median). Two measurements, both against
the engine binary that contains the T-MAC kernel:

1. **Cross-process keep-warm** (a second process launching 1ms dependent-IMAD
   spins): decode 714 ms/tok vs 370 control = **-93%**. CUDA contexts time-slice;
   the engine's ~253 launches/token each queue behind a 1ms spin.
2. **In-process keep-warm** (same context, side stream, 1 warp, dependent IMAD
   chain, 1ms spins + 200us gaps, PQ1_KEEPWARM=1): **376.5 -> 581.2 ms/tok =
   -35.2%** (3 runs/side, interleaved, sd 0.4-4.6ms). Same cause: GM20B has one
   SM, so the engine's short kernels serialise behind each spin launch.
   Code reverted; it is not a lever.

**And the DVFS premise itself is wrong**: sampling
`/sys/devices/57000000.gpu/cur_freq` through a full engine run gives median = max
= **921.6 MHz** (n=213-313 samples, 0.05s interval) in *both* arms. #42's
461MHz median was the sampler outliving the run and reading the parked clock of
an idle GPU (460.8MHz is a devfreq parking step; the engine exits in ~13s, so the
tail of a 20s sampling window is post-exit idle).

CONSEQUENCE for the record: #42's "engine is DVFS-bound" conclusion is VOID. The
correct picture of engine decode at run #43 is:
  - GPU at 921.6MHz (max) for the whole run.
  - Engine streams 1.06 GB/token in ~373ms = 2.84 GB/s = 13.3% of the measured
    21.4 GB/s single-SM ceiling.
  - Cutting the inner loop 3.3x in ALU ops (T-MAC) changes end-to-end decode by
    -1.8% => the engine's GEMV is NOT ALU-throughput-bound.
  - It is not DRAM-bandwidth-bound either (13% of ceiling).
  => It is bound by per-lane SERIAL LATENCY chains: ~828K weight-steps per lane
  with a dependent chain per step, on one SM with 40 warps. Levers that shorten
  the *per-lane chain* pay (#35 +10%, #40 +3.08%); levers that only reduce op
  *count* (T-MAC, E4, #29) do not. Any future inner loop must be designed for
  chain length: minimise the number of dependent ops from LDG to the accumulator,
  and maximise independent chains in flight (ILP x occupancy).

## Contention and measurement-window artifacts (2026-10-14, run #44)

Stream-benchmark ceiling measured twice in two window regimes, which is itself the
finding: the SAME grid=40 config reports 8.67 GB/s in a ~1s window (cold clocks)
and 21.4 GB/s at the tail of a multi-minute sweep (boosted clocks). Any "we are
at X% of the ceiling" claim must state the clock state of its baseline.

Contention test: stream benchmark run concurrently with a full engine decode
(OMP_NUM_THREADS=4 engine in the background) = 8.40 GB/s vs 8.67 GB/s with an
idle CPU. The CPU's per-token work steals no measurable GPU bandwidth, so the
engine's 2.84 GB/s is not DRAM contention with the CPU.

Engine-side per-token GEMV budget (PQ1_KTIME=1, --gen 4, per token, SWAR):
gate+up (12288x4096) 142.5ms = 38% | down (4096x12288) 70.0ms = 19% |
q+o (4096x4096) 58.2ms = 16% | head (151669x4096) 21.9ms = 6% | k+v 25.1ms = 7%.
The big matrices run at 3.2-3.8 GB/s in-engine and the same shapes measured
standalone agree (2.5-5.7 GB/s), so the engine is not paying a per-launch penalty
on them - the cost is inside the kernel's inner loop at 40 warps.

Blocked on: GPU performance counters (nvprof -> ERR_NVGPUCTRPERM, #27). Every
resource that can be measured from the outside (ALU ops, LDS ops, LDS latency,
DRAM bytes, DRAM contention, occupancy, instruction count, clock) has now been
bounded by A/B experiment, and every one of them is slack. The 17x gap between
the inner loop's ALU-issue estimate (~22ms/token) and measured decode (~373ms)
cannot be attributed without stall-reason counters. Requesting profiling access
(root, or the Tegra perf-access toggle) is the highest-value ask available.

## API-overhead family: DONE (2026-10-14, runs #53-#58)
The engine's per-GEMV API pattern (H2D + launch + D2H + sync) measured 46.6ms/token
= 12.8% of decode (PQ1_KTIME=3). Harvested with two structural changes, both
bit-exact (both greedy goldens byte-identical, parity green):
  * MERGED tensors: projections sharing an activation (q+k+v, gate+up) concatenated
    at repack time, one launch of the UNMODIFIED scalar kernel, one D2H into a host
    staging buffer + CPU scatter. +2.10% (gate/up, t=6.8) and +4.00% (q/k/v, t=22).
    Reverses #54's rejection of q/k/v batching: that loss was the multi-pointer
    batch KERNEL's cost (15 params, +49% on short launches), not the idea.
  * CUDA GRAPHS: capture {H2D, kernel, D2H} per tensor (stream capture, lazily on
    the 2nd call with a stable pointer key), replay with cudaGraphLaunch + sync.
    +3.4% total (tools/graph_probe.cu predicted 3.3% from 81us/call). Requires the
    staging buffers to be allocated ONCE at fixed size (256KB/1MB) - a graph that
    bakes in a pointer that is later reallocated reads freed memory silently.
Net: 2.7540 -> 3.0473 tps (+10.7%) in one session day; engine is kernel-bound again
(328ms/token, ~318ms of it inside the GEMV kernels).

## CLOSED: zero-copy outputs on Tegra (run #53)
cudaHostRegister is "operation not supported" on Tegra; cudaHostAlloc(Mapped) works
but is CPU-uncached, so the engine's CPU passes over GEMV outputs cost ~5x the DMA
copy they replace: 365 -> 1862 ms/token (-390%). The path stays behind PQ1_ZC=1,
off by default. Do not retry on this board; on discrete-VRAM targets it is a ~7% win.

## Remaining (kernel-level, unchanged conclusions)
* Inner-loop dependency chain: 4.6% of decode time per op removed from the 11-op
  1-bit chain (#36's slope); #36's analysis says 11 ops/4 weights is the floor for
  the SWAR formulation (PRMT cannot test bits; 64-bit spread costs more; LOP3 is
  already 1 op/AND; the 16-bit-lane fold is what makes the packed accumulator exact).
* Head GEMV device-side argmax: kills the 607KB D2H (0.41ms) + the CPU scan (~0.3ms)
  = ~0.2% of decode. Below the measurement floor; record only.
* Launch+sync floor: ~150 graph launches/token x 133us (measured in tools/graph_probe.cu)
  = 20ms/token = 6% - irreducible while each GEMV's output is consumed by the CPU
  before the next one's input exists.

---

# SESSION HANDOFF — Q1_0 true-1-bit 8B on Jetson Nano (branch `bonsai-jetson-nano`)

Final state 2026-10-14: **decode 3.53 tps vs the 2.13 tps baseline (+65%)**, prefill 3.66 tps,
RSS 341 MB, startup 3.0 s, all three gates green. 68 runs logged; ~20 kept changes.
Full detail: `.auto/HANDOFF_NANO_Q1_0.md`.

## Review: bonsai8b-jetson-nano-research-notes.md (2026-10-14)

**Confirmed by measurement (docs agree with the board):**
- One 128-core Maxwell SM = 4 partition x 32 lanes; 65536 regs/SM. The doc hedges
  ("if regsPerMultiprocessor = 32768, halve the block counts") - RESOLVED empirically:
  5 blocks x 256 thr x 40 regs = 51200 registers resident, and 6 blocks at 42 regs
  (64512) also fits (#7/#27), so the file is the full 65536.
- LOP3 / POPC / BFE / PRMT / SHFL all present; 128-bit LDG; sm_53 + CUDA 10.2 frozen.
- The doc's "XNOR+POPC core" (LOP3 0xC3, POPC, IMAD) is exactly our T-MAC kernel, which
  is now the default for the long shapes (#60 head +0.9%, #62 gate/up +5.5%).
- The doc's occupancy table (42 regs = 6 blocks = 48 warps = 75%) matches #27's
  measurement that extra occupancy beyond 40 warps buys nothing (~+0.5%, n.s.).
- Register/launch_bounds discipline, uint4 coalescing, shuffle reductions, and fused
  QKV/gate+up launches are all things this session arrived at independently (merged
  tensors #55/#56 + CUDA graphs #57/#58 replaced the "fused kernel" form: same effect,
  zero kernel-side cost).

**Where the doc does NOT apply to this engine (important for the next session):**
- Its whole GEMV design assumes **binary activations** (sign(x) -> bit-packed bitmap,
  1 LOP3 + 1 POPC per 32 weights). Our engine mirrors ggml `Q1_0` semantics:
  1-bit weights x **int8 Q8_K activations**, so the inner loop needs the SWAR
  mask-and-fold (11 ALU + 2 LDS per 4 weights) or T-MAC's 8 bit-planes.
- A binary-activation engine would be a NEW OBJECTIVE, not a tuning step: it would
  cut the inner loop ~4x (the doc's roofline says the kernel then becomes DRAM-bound
  at 1.06 GB/token = ~44 ms/token floor, i.e. ~15-20 tps). It cannot be validated
  against the current frozen goldens (they encode the Q8_K pipeline's token stream)
  - it needs gates derived from the true model (MLX/paper) semantics first.
- KV quantization/paging/64K-context sections: N/A while prompts are tens of tokens.

**Instruments worth building (cheap, not yet built):**
- §6.1 POPC throughput probe (15 lines): settles 1/4 vs 1/8 XU rate, which decides the
  T-MAC-vs-SWAR crossover and how far the bit-sliced path can be pushed.
- §6.2 read-only DRAM probe: already covered by `tools/stream_ceiling.cu`
  (21.4 GB/s boosted from 1 SM, 8.7 GB/s in a 1-s window).

**Operator-level levers (§4.4) never tested here because they need root - HIGHEST VALUE ASK:**
- `sudo nvpmodel -m 0 && sudo jetson_clocks` (10 W MAXN + lock CPU/GPU/EMC clocks).
  The GPU is already at 921.6 MHz during runs (#43), but the CPU governor was never
  checked, and #68 found **~28% of decode is CPU-side** (79 ms of 283 ms/token:
  attention softmax, silu expf, Q8_K quantize, rmsnorms, argmax). If the A57s are
  parked by schedutil, that share is inflated - lock the clocks, then RE-BASELINE.
- `isolcpus=2,3` + pinning the engine to cores 2-3, zram, `nvm-map=3264M` headless.

**Prefill batching (doc §5.4) - recorded, not scheduled:** our engine prefills one token
at a time (prefill_tps 3.66 ~= decode 3.53), so a 20-token prompt costs like 20 decode
steps. A token-tiled prefill kernel (M=16/64) would make prompt processing ~free and
speed up every harness run, but it does not move `decode_tps` (the primary metric), and
it is a large kernel+KV-API task. Bit-exactness is plausible (per-token dot order
unchanged) but must be proven with the same parity+golden gates.

## #74 CPU attention/silu/quant second pass (keep): decode 6.4104 -> 6.7476 cold
K cache tiled-transposed (16 positions per 8 KB tile), V cache kv-head major, p.v with
8 named accumulators, silu 2 chains, Q8_K quant 16-lane amax probe + uzp1 + bsums tree.
Bit-exact (logits identical 44/44, goldens A/B exact, parity 256/256 512/512). On-board
alternating A/B vs HEAD binary, 2 rounds x 3 pairs, ms/tok (new/head): 156.65/160.94,
150.85/160.97, 150.73/166.77, 151.09/161.22, 150.60/160.96, 151.88/160.95 = -6.0%.
Measured OUT in the same run: `[i][t]` K^T at 2 KB stride (L1 2-way thrash, attn 17.7 ms),
`float32x4_t a[16]` accumulators (kept in memory by gcc 7, p.v 8.8 ms), OpenMP on heads
and silu (passive/active: attn 11.6/9.7, silu 9.4/10.5 vs 8.3/7.4 serial), T-MAC
latency-hiding variants (pipe/unroll/launch_bounds: 1.429-1.645 vs 1.4255 ms).

## #75 Hybrid CPU/GPU row split (measured OUT; code kept, default off `PQ1_HYB=0`)
A tail of each tensor's rows (multiple of 128, env `PQ1_HYB` = percent) stays in host RAM
and is computed by a NEON Q8_K x 1-bit kernel (`pq1_hyb_rows`, tbl+tst mask, vpadalq, the
reference per-row chain) on `PQ1_HYB_THREADS` spinning workers while the GPU streams the
head rows. Bit-exact at every setting (logits identical 44/44, goldens A/B exact, parity
256/256 512/512; hybtest 0 mismatches vs pq1_gemv_cpu at 4 shapes; 9.4-9.9 Gw/s on 3-4
threads standalone). Two findings killed it:
1. `cudaGraphLaunch` BLOCKS on this platform (CUDA 10.2 Tegra, spin scheduling): the
   cudaDeviceSynchronize after it costs 7.8 us/call with 120 ms of GPU work per token.
   The caller's thread cannot compute while the GPU runs; the first version (slice on the
   caller + workers after the launch) serialised fully: HYB=4 5.54, 8 4.82, 12 3.55 tps
   vs 6.73 off (+8 ms/tok per 1% share, i.e. the CPU kernel at its standalone speed, zero
   overlap). Fixed by `pq1_hyb_start` (workers) before the launch, `pq1_hyb_wait` after.
2. With overlap the two sides starve each other on the shared DRAM/cluster path: at 8% on
   3 workers the GPU phase on 92% of the rows took 126 ms/tok (121 ms for 100% off) and
   the CPU slice ~145 ms (4.3 Gw/s, less than half standalone); 3 workers + the spinning
   caller also starve the CUDA driver (HYB=4/3 thr: gpu 129 ms, hwait 21 ms/tok, 5.28 tps).
   Two workers: HYB=2 6.13, 4 6.64, 6 6.70, 8 **6.79** (gemv 119.9 vs 122.3 ms/tok,
   gpu_ms 5230 vs 5384 per 44 tokens = -2.9% for -8% rows), 10 6.36, 12 5.82 (CPU slice
   overruns: hwait 505/1192 ms). 3 register-only spinning cores do NOT slow the GPU
   (6.726 vs 6.73): it is memory contention, not power/clock capping (MAXN, GPU 921.6 MHz).
ab_pair "PQ1_HYB=8 PQ1_HYB_THREADS=2" 3 reps: 6.7089/6.7942 (+1.27%), 6.8223/6.7298
(+1.37%), 6.7108/6.4187 (-4.4%) -> not resolved, <=+1.3% at best, +47 MB RSS, and two
cores burned. Not worth a default. If revisited: the CPU share only pays if its DRAM
traffic stops stalling the GPU (e.g. prefetch the tail rows into L2 between tokens, or a
bit-transposed code layout so the CPU kernel is compute-light and issues fewer requests).

## #76 fused FFN tail: one CUDA graph per layer, silu*u + Q8_K quant on the device (KEPT, +10.4%)
Per layer {H2D Q8_K(ffn_norm x), prep, T-MAC gate+up -> MDY, `pq1_silu_q8k` -> d_xq, prep,
T-MAC down -> MDY+512KB, D2H -> FO} is one graph (`pq1_ffn_fused`, glue). Removes per token:
CPU silu (7.4 ms), the CPU quant of the down activation, 36 D2H of 98 KB, 36 H2D, 36 syncs.
No new allocation: reuses the merged gate+up and scalar down tensors, d_xq/MDY/d_pl/d_au.
Exactness: `pq1_silu_q8k` replays glibc-2.27 expf in double (+ glibc's |x|>=88 branches
the NEON port hands to libm), 1/(1+e), (g*sg)*u, then q8_K_quant with the fmla that gcc
emits in fwd8.c (`fmla` seen in the fwd8_q1.o disassembly) -> fma.rn.f32. Every f32 op is
inline PTX without .ftz (the .cu is --use_fast_math = ftz; the A57 keeps denormals).
- `pq1_ffn_test`: device v vs host pq1_silu_mul over ALL 2^32 g (u=1) + 2^26 random (g,u)
  incl. denormal u: 0 mismatches (NaN pairs only where both NaN).
- `PQ1_FFN_CHECK=1` in the engine: 1548 fused layers vs the host path, xq_blocks_bad=0,
  fo_bad=0. Logits of prompt A identical (diff 0). Gates 3/3 green on a fresh build.
- cold: decode 7.3212 tps (136.59 ms/tok), prefill 7.5251, peak RSS 345 MB (baseline 6.7582
  / 344). ab_pair "PQ1_FFN=0" 1 pair: 7.3164/6.4948 base-first, 6.7476/7.3039 new-first ->
  off/on = 0.9056 (-9.44%), i.e. fused +10.4%. PQ1GPU ffn=36/1548 proves the path ran.

## #77 T-MAC: pc folded into plane 7 (KEPT, +2.2%); carry-save across code words (OUT)
2pu - 256pc = 2*(S7 - 128*popc(~P7&m)) since popc(P7&m) = pc - popc(~P7&m): the 4 POPC +
3 adds of pc per row-group go (POPC 72 -> 64 per 2-row iteration; ~P7 is free in LOP).
Same integer isum -> same fp. Kernel (PQ1_TIME 200, 2 alternations): 24576x4096 1.430/1.423
-> 1.373/1.373 ms; 4096x12288 0.809/0.805 -> 0.788/0.794; 6144x4096 0.417/0.416 ->
0.411/0.414; 4096x4096 0.299/0.297 -> 0.284/0.295. All bit-exact (pq1_parity all shapes).
Engine: cold 7.4572 tps decode / 7.6816 prefill, RSS 340; ab_pair "PQ1_TMV=0" 1 pair:
7.4594/7.3217 base-first, 7.2956/7.4800 old-first -> old/new 0.9784 (new +2.2%).
OUT: carry-save across code words 0..2 per plane (popc(x0)+popc(x1)+popc(x2) =
popc(x0^x1^x2)+2popc(maj)), POPC 64 -> 48 at equal instruction count. Needs plane-major
order: spills at 48 regs (1.64 ms), 64 regs/4 blocks no spill still 1.44 ms > 1.373. The
kernel is issue-bound; POPC removal only pays when instructions go with it.

## #78 Attention block as one CUDA graph per layer — KEPT
q+k+v GEMV, head norms, rope, KV store (device cache in arena), attention, Q8_K and attn_output fused per layer. PQ1_ATT_CHECK: 1548 layer-steps bit-exact. ab_pair PQ1_ATT=0: -7.9% when off. Cold 7.46 -> 7.89 decode tps, RSS 343. Arena 1158.9/1280 MB. Next: rms/residual/FFN quant on device (whole layer in one graph); T-MAC epilogue lane split; head GEMV at ~48% of ceiling.

## #79 T-MAC prep one warp per group (ballot) — KEPT
nvprof showed pq1_tm_prep at 36 us x 4/layer (one thread per group). Ballot version: planes/au identical, A/B PQ1_TMPW=0 -3.2%. Cold 7.89 -> 8.16 decode tps, RSS 339.
Measured out (iter 5): T-MAC epilogue lane split + sh_prod pad, kernel flat/-1%. Kernel is POPC-throughput bound (sm_5x POPC 32/clk/SM).
Remaining small kernels per layer (nvprof): attn_pre 100 us, silu_q8k 62 us, attn_core 55 us.

## #80 attn_pre per-thread RMS chains — KEPT
One-lane DP instructions cost a full warp of DP issue on sm_53; packing 40 head chains into lanes: A/B PQ1_APRE=0 -1.75%. Cold 8.29 decode tps, RSS 342.

## Measured out (loop iter 8): T-MAC plane skipping
Activation word bit widths (prompt A): K=4096 inputs avg 7.2 signed bits (55% need 8), down input avg 6.7. Kernel V=2 skipping sign-extension planes with warp-uniform branches: 1.371 -> 1.695 ms at 24576x4096 (+23%), overhead >> the 10-16% POPC it could save. Patch in autoresearch/loop-261014-2216/iter8_tm_planeskip.patch.

## Measured out (loop iter 9): silu in the gate+up T-MAC epilogue
Thread row0=gate i, row1=up N+i, epilogue silu(g)*u then q8k<0>: bit-exact (FFNCHK 1548/0) but ab_pair PQ1_TMSILU=0 +2.04% (old path faster): the DP expf tail serializes on lane sub 0 at block end instead of overlapping. Patch iter9_tm_silu_epilogue.patch.

## Loop 261014-2216 rows 11-35 (8.27 -> 16.11 decode tps): summary, details in HANDOFF §0
Kept: flash-attn f16 KV (c8b2bff), Harley-Seal hs GEMV (1e1a1bb, +30.7%), sign-magnitude
hs + load ring (2b1c440), whole token as one graph (8fa3f38, +11.9%), plane build fused
into Q8_K (8a1bad7), SEG reduction at K=12288 (5d2dad4), fa_sm shuffle scan (f2cb48e),
certified fp32 expf/RMS family (0c70c2d, d59c852, ba9f43b, 8a3e10c), warp-per-block Q8_K
(1bf9aac), device argmax (9dd569f), occupancy x prefetch-depth hs configs (0eae2d3 +2.7%,
11806d3 +9.8%). Rule: PD = RB (whole row batch in flight) AND >= 24 resident warps;
either alone loses. Measured out: cache-policy loads, magic I2F, pinned staging,
kernel-only token graph, NR=2 tile (504 us vs 435), block-wide reduction (592 vs 435),
batched expf q8k_w<3> (35.2 vs 31.0 us in-engine), K=12288 re-sweep at PD=RB (all >= 437
vs 406 us). Peak RSS drifts 596 -> 770 MB on identical binaries over one session.

## Loop 261014-2216 rows 36-44 (16.11 -> 16.51 decode tps)
Kept: hs K=4096 deferred row reduction (a645bbd), out-of-line silu/expf fallback in q8k_w<2>
(2e685da, SASS 1836 -> 762, 31.8 -> 22.6 us), attn_pre3 one q/k head per warp over 5 blocks
(bfabc8a, 25.0 -> 10.25 us). Measured out: rms_q8k serial fallback __noinline__ (launch failure,
not debugged) or unroll 1 (no change, 15.2 us); fa_pv register prefetch ring PD=4 (18.1 -> 21.6
us, fa_pv is conversion-pipe bound, ~2.2 us per 8 positions); PQ1_PV_ALU mix (noise).
Both hs GEMVs are issue-bound (IPC 3.89, 77% issue slots; gu ~24.3, down ~28.9 thread-instr
per weight word vs <= ~21.5 needed for the 21.9 GB/s ceiling at 921 MHz).
Open levers: fuse rms_q8k/q8k_w into the next GEMV prologue (~1.1 ms); attention mega-fusion or
qk4 wave-quantization fix (~2 ms total); down SEG reduction lane use (4/32); CPU-shares-rows
co-execution to break the GPU issue bound (48.4 ms bus floor).

## Feature backlog (user, 2026-10-15), beyond decode speed
1. 3/4-bit TurboQuant KV cache (replace the f16 K/V cache; cuts the attention bytes and arena,
   needs its own accuracy gate since it is not bit-exact against the f16 reference).
2. Bonsai-8B ternary checkpoint support on the Nano (pq2/TQ path exists for Spark CPU and the
   earlier exact 2-bit Nano path; needs the current hs/graph chain ported).
3. N > 1 concurrent users: batched decode over N sequences (weights streamed once per step for
   all N, per-user KV caches); the batch GEMV variant in pq1_swar_sm53.cu is the starting point.

## Note: why the out-of-line silu fallback (2e685da) won (nvprof metrics, pq1_parity fpl, 215 calls)
q8k_w<2> (SNI=1) vs q8k_w<3> (SNI=0, old): inst_executed avg 53658 vs 57961 (-7.4%), max 80674 vs
83026; IPC 2.98 vs 3.20; warp exec eff 91.0 vs 92.8%. The 31.8 -> 22.6 us win is mostly not
executed instructions but code size (SASS 1836 -> 762: the inlined, never-taken fallback was
evicting the hot loop from the instruction cache). Engine-side metrics are not obtainable:
nvprof --metrics does not see kernels inside the token graph, and with PQ1_GRAPH=0 the engine
launches no q8k_w at all (only tm_prep_w and the two hs GEMVs). The earlier job bvf8gopr1's
"engine" numbers were in fact the par_fl (parity) half.

## Measured out (row 45): down GEMV reduction lane use (private tile rings + NT batches per SEG step)
Down and gate+up are equally issue-bound (77% issue slots, IPC 3.89); down executes 19% more
instructions per weight word (0.90 vs 0.76 warp-instr/word). Upper bound for the 4-lane SEG
reduction: cutting it to 1 of 8 steps (wrong output) takes the kernel 405 -> 374 us, ~1.1 ms/token.
pq1_gemv_hs3n (each warp keeps only its own wt+1 tile slots, so 2 batches fit per slot: 8 reduction
lanes, half the barriers), all bit-exact 4096/4096, nvprof parity: old 405-408 us, NT=1 442-447,
NT=2 423-427, NT=2 with uniform per-team batch counts 506. NT=2 runs 3.7% fewer instructions but IPC
falls 3.90 -> 3.65 as memory-dependency stalls rise 5 -> 18%: the per-batch reduction gap was also
covering load latency. Patch kept as scratchpad row45_hs3n_private_ring.patch (not in the repo).

Row 46 (discard): pq1_fa_sm fused into pq1_fa_qk4. The last y-block of each kv head (threadfence +
atomic counter, re-armed by that block so graph replay stays valid) runs the softmax for its 4 query
heads, one warp each, reading the scores through L2 (__ldcg). It is bit-exact: fa parity
4096/4096 at T = 1..4096, r16 OK. ab_pair over 3 reps gave new/base 0.9971 (-0.29%). The serial
tail (atomic, fence, L2 reread of sc, expf chain on 8 blocks) costs more than the ~1.5 us launch
gap it removes. Conclusion: launch gaps between attention kernels are not worth fusing on this
board, so the same goes for fusing pv or q8k_w<0>.

## Feature: TurboQuant K/V cache (PQ1_KVTQ=3|4|43, opt-in, lossy)

Random sign flip + 128-point Walsh-Hadamard rotation, then each coordinate snapped to a
Lloyd-Max codebook for the d=128 sphere marginal (3 or 4 bits), plus one f32 norm per row
(MSE variant, no QJL residual). 43 = 4-bit K + 3-bit V. Kernels pq1_tq_{store,qk,sm,pv,out};
pq1_parity tq 3|4 checks device codes vs the host encoder (0 differ) and the kernel vs fp64
attention on the decoded cache (max err 1.2e-5). K relative MSE 0.0307 / 0.0085 (theory 0.034 / 0.0093).
Measured on the real 8B (teacher-forced on 2048 f16-generated tokens, top-1 vs f16):
TQ3 90.8%, K4V3 95.0%, TQ4 95.9%. Prompt A: all three generate the same 24 tokens as f16.
ms/tok at ~100 / ~1050 / ~1970 positions: f16 62.1 / 88.4 / 117.0; TQ3 64.0 / 78.7 / 92.5;
K4V3 65.8 / 92.0 / 116.2; TQ4 65.7 / 91.3 / 114.8 (qk<4> is the slow kernel).
GPU used at ctx 2112: f16 1314.6 MB, TQ3 1077.9, K4V3 1087.2, TQ4 1096.5.
Host RSS follows the arena size, not the cache: a 1280 MB arena gave 795 MB RSS vs 694 MB at
1426 with the same used_mb, so TQ keeps the f16 arena sizing (RSS 684 vs f16 665 at 2k).
Open: qk<4> speed (fp16 LUT), and a larger default arena may cut default-path RSS by ~100 MB.
TQ follow-up (qk magnitude table): pq1_tq_qk keeps only the 2^(B-1) magnitudes in its smem
table and applies the code's sign bit as fmaf(+-1, e, a) (exactly a +- e, outputs bit-identical).
4-bit table 32 KB -> 16 KB (1 -> 2 blocks/SM, 16-entry rows had 2-way bank conflicts):
parity-timed tq4 per layer at T=1024 0.920 -> 0.437 ms, tq3 0.489 -> 0.472. Engine, TF-2048,
ms/tok @~1970: f16 117.1, TQ3 91.0, K4V3 90.3, TQ4 88.8 (top-1 90.8 / 95.0 / 96.0%, unchanged).
TQ4 is now the best arm: fastest at long context and most accurate.
Row 50 (measured out): larger GPU arena to cut peak RSS. Prompt A, PQ1_ARENA_MB sweep: max RSS
drops ~1 MB per arena MB (TQ4: 1280 -> 716-724 MB, 1426 -> 583, 1536 -> 469, 1792 -> 214; f16
same slope), speed and PQ1GPU used_mb unchanged. Mid-run /proc: RssAnon 604 MB (1280) vs 85 MB
(1792), RssFile 32 vs 34, but system MemAvailable 1426 vs 1421 MB and Cached 1.29 vs 1.26 GB:
the board uses the same memory, the arena size only moves ~520 MB between the process's anon
RSS and the nvmap/GPU-allocator accounting. Not a saving. Peak RSS on this board is therefore
an accounting metric; compare memory claims with MemAvailable at the same arena size.
Row 51 (kept): TQ short-context path. At ctx <= 256 (one V chunk; the graph is captured per
ctx, prompt A runs at ctx 64) pq1_attn_tq_launch runs pq1_tq_qks + pq1_tq_tail instead of
store + qk + sm + pv + out. nvprof in the engine, us per layer call at ctx 64: old chain qk 43.3
+ pv 21.5 + sm 15.4 + store 7.4 + out 6.3 = 93.8; new qks 19.8 + tail 28.2 = 48.0; f16 qk4 18.4
+ pv 18.3 + sm 9.7 = 46.3. What the time was: not arithmetic but serial latency. A one-position
pq1_tq_sm costs 13 us, and the first tail (4 heads' softmax one after the other, 16 barriers)
took 48 us at T=1; running the 4 heads' trees side by side (3 barriers) halved it. The fully
unrolled 128-term qk body (~14 KB of code, fetched cold every layer since the weight stream
evicts L2) cost ~2x vs a rolled word loop (T=44: 36.8 -> 19.4 us). qks forms the table entry
fl(q_r c) inline and folds the K/V row store of npos in; tail keeps every reduction tree of
sm/pv/out. Bit-identical: pq1_parity tq 3|4|43 compares the short path with store + long chain
word for word at ctx 64 and 256 (0 differ), TQ4 goldens A 44/44 B 26/26 unchanged. The parity
harness gained K4/V3 support (it had one code width) and a correctly sized partials buffer
(nh*ctx floats was short at ctx < 256: the ctx-64 FAIL was the harness, the engine's region is
2*64*ctx floats). Order-balanced A/B on one binary, prompt A, PQ1_KVTQ=4 vs +PQ1_TQSHORT=0:
R = 1.0304 / 1.0261 / 1.0292 (bias 1.001 / 0.997 / 0.999); TQ4 vs f16 same rep 1.003 / 1.000 /
1.003. Short-context TQ4 overhead is gone. Open: ctx > 256 runs still use the long chain while
T is small (needs a host-side per-step graph choice).

52. Context beyond 4096 positions (keep, capability; short-ctx decode untouched). Compile-time
CTX 4096 -> 16384, run cap PQ1_CTX (default 4096, 64-aligned, <= 16384). The host f16 K16/V16
arrays (BSS [36][8][CTX][128], 2.4 GB virtual at 16384) are now calloc'd at KVCTX stride on first
use by the host attention only (CPU fallback, PQ1_ATT_CHECK); the fused GPU path never touched
them. Arena: unchanged at ctx <= 4096 (f16-sized, the RSS lesson of row 50); above 4096 it takes
max(f16 bytes at 4096, TQ bytes at ctx) so it stays monotone. The f16 cache above 4096 is refused
(147 KB/position: 2.3 GB at 16k does not fit beside 1.19 GB of weights) instead of falling into
the CPU path. pq1_parity tq 3|4|43 at PQ1_TQ_CTX=16384, T up to 16384: PASS (the harness had a
4096-entry stack score array that segfaulted at 8192; heap now). Engine runs, TQ4, TF on repeated
ids2k text: 6144 positions: arena 1717 MB, cache 231.9 MB, max RSS 224 MB, ms/token 110 (pos 1),
92 (2048), 120 (4096), 145 (6144). 16384 positions: arena 1761 MB, cache 612 MB, PQ1GPU used
1640 MB, free 301 MB, max RSS 230 MB, ms/token 172 (8192), 223 (12288), 275 (16383). Gates green
on a fresh build, TQ4 goldens unchanged, prompt A 16.44 tps (single cold sample; the hot path is
unchanged so no A/B). The per-token cost grows ~13 ms per 1k positions: the TQ qk/pv kernels are
far from the cache-streaming floor (612 MB at 16k = ~24 ms), next lever.

53. TQ pv without the shared code copy (keep, long ctx). pq1_tq_pv staged the chunk's 256
positions of V code words in 16 KB of shared memory (26 KB per block, 86 registers: 2 blocks,
8 warps on the one SM). pq1_tq_pv2 reads each position's words straight from global (one
transaction per warp), 10 KB smem, 52-57 registers: 6 blocks. Same fmaf chains and reduction
tree, bit-identical: pq1_parity tq 3|4 compares the variant with the old kernels bitwise at
every T to 16384 (0 differ). PQ1_TQV=0 is the old path. Layer time (parity harness, store
excluded): T=16384 5.905 -> 4.630 ms, T=4096 1.638 -> 1.320 ms. Engine, 2040-id prompt + 16
generated, order v1 v0 v0 v1: decode 84.53 / 90.50 / 90.60 / 84.49 ms (-6.7%, R = 1.071), greedy
stream identical. Profile at T=16384 before the change (nvprof, per layer): pv 2.95, qk 2.65,
sm 0.36, out 0.06 ms; the K or V code stream alone is 0.33 ms. qk is shared-bandwidth bound
(lds.128 of a per-lane table entry = 4 wavefronts per term; floor ~2.3 ms): a bit-identical
rewrite (inline fl(q c), several positions per thread) is worth <= ~15%; a 2x needs fp16
arithmetic (HFMA2), which moves the TQ bits.

54. TQ short pair per step (keep, TQ ctx > 256). The qks + tail pair (row 51) only ran when the
whole run fit in 256 positions; ctx is just the cache row stride for it, so it is valid for any
step with T <= 256. pq1_token_fused now keeps two graph slots (PQ1_TOK_OKV/EXV[2]) when TQ is on,
ctx > 256 and PQ1_TQSHORT is set: slot 1 is captured with pq1_tq_short_next(1) for the steps with
npos < 256, slot 0 with the long chain after that. qks grid y is sized from min(ctx, 256).
pq1_parity tq now checks the short pair against the long chain at its own ctx (T <= 256 at
stride 2048): tq3/tq4/tq43 0 words differ. Engine, TQ4, prompt A + 300 generated (ctx 320),
order v1 v0 v0 v1: steps 20..255 61.59 / 63.03 / 63.16 / 61.80 ms (-2.2%); steps >= 257 64.93 vs
64.96 (equal; step 256 carries the one-time 2nd capture, 108.7 ms). Greedy stream md5 identical.
Gates green on a fresh build, 16.53 tps (f16 gate run, path unaffected).

55. TQ qk2: no product table, two positions per thread (keep, TQ ctx > 256). pq1_tq_qk read a
float4 table entry (the 4 heads' fl(q_r[j] c[v])) per term: lds.128 at a per-lane address = 4
wavefronts, shared-bandwidth bound. pq1_tq_qk2 forms each product inline: a broadcast float4 of
q_r[j] (1 wavefront) and a signed 16-entry centroid table (1 wavefront), a = fl(a + fl(q (+-c))),
which is exactly fmaf(+-1, fl(q c), a), the old chain (sign flip exact, signed zeros equal).
One position per thread: -17% (T=4096) / -10% (16384) per layer; the sign-float build dropped by
the signed table is part of that. Two positions per thread (t, t+256 of a block's 512-span,
grid y = min(ctx/512, 16)) share the q load: per layer T=700 0.264 -> 0.233, T=4096 1.336 ->
1.011, T=16384 4.68 -> 3.90 ms (order v1 v3 v3 v1). A first P=2 attempt paired t with t + grid
stride, all duplicates when T <= 4096 (1.45 ms at 4096): pair inside the block. More blocks per
kv head (32, 64) are slower; 8 is within 2%. pq1_parity tq 3/4/43: variant == old kernels
bitwise at every T to 16384. Engine, TQ4, 2040 ids + 16 generated, order v3 v1 v1 v3: 78.82 /
84.54 / 84.68 / 78.74 ms (-6.9%, R = 1.074), stream identical. Default PQ1_TQV 1 -> 3. Gates
green on a fresh build, 16.44 tps (f16 gate run, path unaffected).

56. N>1 users, batched decode GEMV (measured out) + down GEMV unsegmented (keep, +1.83%).
Prototype pq1_gemv_hsn<NB,...> (.auto/hsn_batch_proto.patch): one weight pass serves NB
activation plane sets; warp 0 lanes < NB*RB run the row chains, the lane's set is lane / RB.
Bit-identical per set to pq1_gemv_hs (pq1_parity hsn, 0 words differ on every shape below).
Time of one NB-set call vs NB single calls (20 iterations, cudaEvent): gate+up 24576x4096 NB=2
0.72-0.93x (best tile RB 2/PD 2/MINB 3), NB=3 0.74x, NB=4 0.76x; qkv 6144x4096 NB=2..4
0.75-0.80x; down 4096x12288 NB=2 0.76x. Slower than separate calls at every NB: hs is
issue-bound at N=1 (IPC 3.89), sharing the weight load saves bus time the kernel is not waiting
on, and the NB-wide plane registers cut residency. Batched decode would lower aggregate tokens/s
on this GPU, so N>1 is not built; concurrent users can only time-slice the N=1 engine.
The probe exposed that hsn<1> at K=12288 (unsegmented chain) beat the default segmented hs<3>:
375 vs 400 us per call, holding with the timing order reversed; PQ1_HSG=0 (hs<3,...,SEG=false>)
gives the same 375. The SEG win (row 41-era, 69 us of 472) predates the RB 4/PD 4/MINB 4 retune;
at that occupancy warp 0 no longer straggles. Engine A/B PQ1_HSG=0, ab_pair 1 pair: 16.75/16.74
vs 16.45/16.44 tps (+1.83% both orders). nvprof in the engine: hs<3> 411.7 -> 384.9 us per call
(576 calls, SEG=1 -> SEG=0 instance). Same chain, same order: default PQ1_HSG 1 -> 0. Gates green
on a fresh build, 16.72 tps.

57. Ternary-Bonsai-4B phase 2: the pq1 GPU engine under -DPQ1_TERN (keep; 4B 10.1 -> 14.5 tps).
PQ2_0 rows repack to two 1-bit planes (b0 | b1, K/32 words each); pq1t_gemv_hs<G,WR,TPB,MINB,RB,PD>
runs one Harley-Seal sign count per plane against the Q8_K sign-magnitude planes, isum =
2*S(b1) + S(b0) - S - S_N, then (ws*db)*isum per group, summed in group order with __fadd_rn: the
fork's pq2 vec_dot order. pq1t_parity: BIT-EXACT at qkv 6144x2560, o 2560x4096, gate+up
19456x2560, down 2560x9728, head 151669x2560, 1001x2560, 37x9728. Engine A/B (PQ1_KVTQ=0) equal
the fork goldens 44/44 and 26/26.
(a) rms_q8k at n=2560: lane-0 tail fell to the serial fp64 chain on every call for n not a power
of two (230 us x 73 calls = 17 ms/token). Bracket S = H + L in fp64 with directed rounding
(S(1 -/+ 2^-40)/n), accept when both ends round to the same float, else serial. pq1_parity rms
2560/4096/1280 bit-exact (fallback on ~20% of the steered-to-midpoint vectors only); nvprof
230 -> 13.9 us/call; decode 99 -> 83.7 ms. Q1 n=4096 path unchanged (gates green).
(b) Team shapes per K (pq1t_parity, 50-launch mean, us): K=2560 <20,5,1,4,2,2> -> <20,2,2,4,4,2>
qkv 296 -> 235, gate+up 900 -> 726, head 6979 -> 5556; K=4096 <32,1,4,6,4,2> -> <..,3,3> 187 ->
149; K=9728 <76,5,1,4,4,2> -> <..,3,3> 525 -> 429. Tried and slower: RB 1/PD 1, RB 3/PD 1,
RB 5/PD 5, RB 6/PD 3, TPB 8/MINB 3, WR 4. Engine, order HST 1 0 0 1: 83.76 / 68.71 / 69.15 /
83.75 ms per token (-17.7%), streams identical. TQ4 default: 68.4 ms (14.6 tps), 5/44 tokens
differ from the f16 stream (lossy cache, no 4B TQ goldens frozen). Q1 gates green, 16.72 tps.

58. N>1 users as time-sliced sessions (keep; feature, solo speed unchanged).
fwd8.c: prompts separated by --next become sessions (ids, npos, step, last top, sampler RNG copy),
stepped round-robin; the next id is drawn right after a step while HEAD still holds that
session's logits. Glue: pq1_sess_init(n) makes the K/V region n caches long; pq1_sess_select(s)
repoints d_kv and swaps PQ1KV_NEXT/LIVE, the 36 attention graphs and the two token graphs per
session (each bakes its cache pointer; captured once per session, kept). First run: under the
f16 cache all 4 streams equal their goldens, under TQ4 sessions 1-3 diverged from step 0 (logit
8.8495 vs 8.9102): a solo run's first token is always the host attention on exact f16 K/V
(merged tensors do not exist yet), while a later session's went to the device graph, which reads
position 0 back from the TurboQuant cache. Fix: NPOS == 0 always takes the host attention.
Results (8B): A B A B, each stream == solo golden under PQ1_KVTQ=0 and 4; greedy B full 40
steps == solo B --gen 24; --sample 0.7 0.9 40 1234 0, A B B: sessions == their solo runs.
Throughput, TQ4, 4 sessions, steps 9-128: 59.54 ms/step = 16.79 tok/s aggregate, 4.20 per user
(solo 16.73). Capacity (TQ4): 4 sessions x 4032 positions = 602 MB of cache allocates (arena
grows). 4B: 4 sessions == fork goldens (f16), TQ4 68.24 ms/step, 14.65 aggregate, 3.66 per user.
Gates: checks.sh sections 1-4 green, measure 16.73 tps; checks_t4 green.

59. Bonsai-2 (qwen35 hybrid: Gated DeltaNet + full attention, PTQ1_0 ternary) GPU engine on the
Nano (keep; new model family, fwd35_sm53.cu, gates .auto/checks_b2.sh, fork goldens golden_b2_*).
Bit-exact vs the PrismML fork: greedy 48 tokens, prompts A and B, 2B and 4B, on the Nano (sm_53)
and a GTX 1070 (sm_61), batched prefill on and off. Decode (jetson_clocks, prompt A, 48 tok):
2B 21.6 -> 21.97 tps, 4B 8.16 -> 9.94 tps from the tuning below; prefill 25.4 / 10.8 tps.
Kept: (a) GEMV k_gemv<G,WR,TPB,RB,PD,AD> per-K shape table (G16 <16,1,8,2,2>, G20 <20,5,1,3,3>
AD1, G32 <32,1,8,4,2> AD1, G48/G96 <..,3,2,4,2>, G72 <72,9,1,2,1>); AD=1 = immediate-offset
addressing over a tail-padded arena (+2 MB codes, +1 MB scales), helps G20/G32, hurts G48/G72.
(b) norm_xq with BS/2 threads: 512-thread x5 blocks exceeded 2048 threads/SM on the 4B, 190 ->
110 us. (c) k_gdn: glibc-exact expf/logf gate scalars once per block into shared memory (fp64
runs at 1/32 rate), 330 -> 282 us. (d) k_ab: 8-wide loads ahead of the in-order fma chain.
Measured out, do not re-walk:
- MTP / DFlash speculative decoding: the ternary GEMV is issue-bound per (row, token) (~286 SASS
  per 128-weight group, ~180 of them the POPC/LOP3 core; ~13 GB/s ceiling vs 12.4 achieved), so
  verifying k tokens costs about k decode steps. Batched prefill on sm_53, per-token cost vs
  chunk 1: chunk 2 1.81x, 3 1.62x, 8 1.36x (4B 8.9 -> 6.56 tps); 2B chunk 8 19.4 vs 25.1 tps.
  The same batching is 3.0-3.7x faster on the GTX 1070 (2B 151 -> 560, 4B 83 -> 252 tps), hence
  E_PCHUNK defaults to 8 on cc >= 6.1 and 1 on sm_53. DFlash drafts (~1.57 GB) do not fit anyway.
- The old pq1t_gemv_hs 4-word tree needs Q8_K (one scale per 256); Q8_0 per-32 scales block it.
- Multi-element exact rms sum (wsum_sqv): slower on the Nano. RW interleave in k_gdn: no gain.
- GEMV shape sweeps (E_V16 0-7, E_V48 0-6, E_V20/72/32): the table above is the sweep winner.
Remaining 4B per-token GPU profile: G20 (in + ffu + head) ~44 ms, G72 16.7, G32 7.2, norm_xq 7,
gdn 6.8, xq 2.4, fa 1.9 ms.
9B on the Nano (fits: 1890.5 MB planes + 118.2 MB scales + 178 MB state; repack 21 s): checks_b2 9B
4/4 PASS, prefill 6.2 tps, decode 5.78 tps (48 tok), 5.72 over 99 steps (~11.5 GB/s of weights).
Shape sweep at K=12288 (G96, down) and K=4096 (G32: qkv/gate/up/head/o), 2 reps, tps (default
5.71/5.68): V96 <96,3,2,4,2,AD1> 5.58, <96,3,2,3,3> 5.50/5.57, <96,6,1,2,2> 5.52/5.50, <96,3,1,4,2>
5.12, <96,3,4,2,2> 5.62/5.61; V32 <32,1,8,4,2,AD0> 5.41/5.43, <32,1,8,3,3> 5.75/5.73, <32,1,4,4,2>
5.75/5.74, <32,2,4,4,2> 5.40/5.37, <32,1,8,6,3> 4.89/4.92. Best is +0.8%, inside the +-1% of one
sample and on the 4B's G32 too: defaults kept, variants stay behind E_V96 / E_V32.
Demos (doc/bonsai2-{2b,4b,9b}-nano-chat-{en,zh-tw}.gif): greedy to <|im_end|> with E_STOP=1 and
E_STEPMS=1; every stream equals fork_greedy over its full length (2B 101/351, 4B 83/405, 9B
100/93 ids). Decode 2B 22.11 / 21.06, 4B 9.90 / 9.25, 9B 5.72 / 5.74 tps (en / zh-tw).

60. Bonsai-2 4B long-context loop (autoresearch loop-261016-1216, 15 iterations, Nano pesi):
score = sqrt(prefill_tps x decode_tps) on the 3968-token zh-TW prompt + 64 greedy tokens, f16
K/V at ctx 4096, every kept change bit-identical to the fork golden (checks_b2.sh 4B ALL PASS).
4.4182 -> 9.6182 (+117.7%); prefill 5.53 -> 10.67, decode 3.53 -> 8.67 tps; peak RSS 201800 ->
198328 KB (noise band 198-203 MB). Kept: k_fa smem flag packing, 16-row V prefetch, 4-position
score passes (it 1-3); split attention k_fs/k_fw/k_fv (it 5, the big one at 4k ctx); k_fv FVD=4
(it 6); exact rms sum overlapped with the residual GEMV via k_ssq on a second stream, then
fence-free sentinel hand-off (it 7-8); k_gdn 2 rows x 2 warps (it 10); k_fvg 2 heads x 2 dims per
thread (it 11); ping-pong prompt staging buffers (it 12).
Measured out (all bit-identical unless noted): k_fag grouped attention (it 4, slower); fused
k_gnorm->k_xq (neutral); Veltkamp f16 rounding in k_fv with exact fallback (2.29 -> 2.64 ms) and
unchecked (2.08 -> 1.95, not exact); conversion-free V via the h2f bit trick in k_fvg (2.083 ->
2.025 ms, NOT taken: inexact if a V cache entry is inf/NaN; a sticky nonfinite flag from k_qk
would make it safe); k_fvg shapes (2x2 best at FVU 4; 4x2 2.097, 2x4 2.269, 1x4 2.825, FVU 2/8
2.205/2.118 ms); smem-tree k_fs reduction (0.82 -> 0.98 ms); conversion-free K in k_fs (0.89);
parallel prefix-max k_fw (0.286 -> 0.283; k_fw is bound by the FP64 glibc-exact expf); on-GPU
embed / pipelined decode (removing every per-step host sync gains 0.2-0.5%, not built); k_ab with
2-10 chunks of loads in flight (no change: nvprof's 133 us/call is not exposed latency); k_gdn
software-pipelined state loads (-0.6%). Batched prefill (E_PCHUNK 2/4/8) on sm_53 is 5.2/6.2/7.0
vs 11.9 tps at chunk 1: the single-token GEMV is popc-pipe and issue bound (~10 popc + ~35 ALU per
32 weights per token, both ~0.33 clk/word on the one SM), so sharing weight loads buys nothing; an
HFMA2/FFMA decode path costs ~1.7 issue per weight-token. nvprof at 2000 tokens: GEMVs 59% of GPU
time (k_ssq's 22% runs concurrently), k_gdn 5.6%, k_fvg 3.2%, k_ab 2.9%, k_xq 2.2%, k_fs 1.7%.
Open: a cheaper exact ternary x int8 inner product (fewer popc per word), the sticky-flag
conversion-free V, k_gdn closer to the streaming ceiling (16 GB/s now).

61. 8-POPC ternary x Q8_0 identity in k_gemv / k_gemvN (keep, bit-identical). With c0 = b0 (w = 0)
and c1 = lop3(b0, b1, N, 0x46) = (w = +1, x >= 0) | (w = -1, x < 0) (exclusive masks), each lane's
v = (c0 + 2 c1)|x| = w x + |x| < 256 has carry-free bit planes D_0 = c0 Q_0, D_q = c0 Q_q | c1 Q_{q-1}
(one LOP3 0xEA each), D_7 = c1 Q_6, so sum w x = sum_q 2^q popc(D_q) - S and the activation
correction word becomes cg = -S (was -(S + S_N)). Same integer as the 9-POPC carry-save ripple, so
the float chain is untouched; per-lane identity checked exhaustively (3 codes x 255 q). SASS: POPC
2916 -> 2592 in the binary, GEMV body 900 -> 720 instructions at the same 96 registers. Nano pesi,
same job: 4k harness prefill 11.37 / decode 9.25 tps, score 10.2554 (HEAD 9.51-9.56 the same day),
GOLDEN PASS, RSS 200368 KB (noise band 198-203 MB); ABBA x2 4B 47-token decode 11.58/11.61/11.61/
11.62 vs 10.75/10.69/10.75/10.76 (+8.1%), 400-token prefill 12.81/12.85/12.87/12.87 vs 11.92/11.92/
12.01/12.02 (+7.4%). checks_b2 2B+4B ALL PASS (chunk 1 and 8); 2B decode 24.9 tps (was 22.1).
Instructions -20% but time -9%: nvprof (E_GRAPH=0; graphs hide kernels from --metrics) on the
GEMVs: K=2560 occupancy 31% / issue 57% / stalls mem 36% exec 28% sync 22%; K=9216 42% / 52% /
mem 23% exec 19% sync 42%; K=4096 37% / 50% / mem 51% exec 28% sync 0%. The body's issue floor
(~42 instr per word at ~125/clk on the one SM, ~48 ms/token) now meets the 50 ms weight-streaming
floor; what is left is latency hiding (the tile barrier, too few bytes in flight).

62. P2 batched prefill GEMV on the fp16 pipe, k_hg (keep, bit-identical; prefill +37%). Lane = row,
RB = 2 rows per lane, W = 4 warps, NT = 8 tokens. Each 32-weight word pair is decoded once per chunk
to 16 half2 of +-2 / 0 (nz = ~b0, neg = ~(b0 | b1); even/odd elements interleaved with 2 LOP3 so one
shift + mask yields the half2 of elements (i, i + 16)). Activations are q/2 in fp16, written by
q8word when gridDim.y > 1 into a tail of each token's d_pl stride (PLS += kmax / 2 words), staged 4
groups at a time in shared memory. Each half lane sums 16 products 2 w q / 2 = w q, integers <= 2032
(fp16 exact to 2048), so lo + hi through 2 F2F + FADD is exactly the integer s and the fork chain
(si = fma(d8, s, si), sumf = fma(scale, si, sumf)) is k_gemv's. Standalone bench vs a verbatim copy
of k_gemv x 8 tokens, 0 mismatches on every shape: v1 (RB 1, all loops unrolled) 1.05-1.33x;
nvprof showed 34% inst-fetch and 31% memory-dependency stalls -> RB 2, the 4-word loop rolled
(registers rotated), next-group weight prefetch: 1.49x K=2560 / 1.67x K=9216 / 1.55x K=4096, issue
75%, 155 registers, occupancy 18%; capping registers (launch bounds 4-12 blocks) or W 2/8, GC 2/8
all lose. Floor of this form ~0.8 issue per weight-token (16 HFMA2 + 4 LDS.128 + 2 F2F + FADD +
FFMA per word-row-token at RB 2, decode amortized over 8 tokens), so not the ~0.5 hoped. Engine:
default chunk is now NTM = 8 on every arch; k_gemvN is kept for sm_61 (fp16 1/64 rate), E_HG=0
forces it (8.91 tps on the 400-token prefill vs 18.27, same stream md5). Nano pesi: 4k harness
prefill 11.37 -> 15.54 tps, decode 9.23 (unchanged), score 10.2554 -> 11.9764, GOLDEN PASS, RSS
197576 KB; checks_b2 2B+4B ALL PASS (chunk 1 and 8); 4B prompt A prefill 12.57 (chunk 1) / 17.68
(chunk 8), 2B 29.6 / 43.7.

63. P1 lane = row decode GEMV on a 32-row tiled weight layout, k_li (keep, bit-identical; decode
+7%, 4k score +4.9%). First prototype (lane = row on the row-major planes, activation planes of the
whole K in shared memory as broadcast LDS.128, each lane running its row's chain s_k = tw + cg,
si = fma(d8, s, si), sumf = fma(sc, si, sumf)) was bit-exact but 2.0-2.5x SLOWER than k_gemv
(7-8 GB/s): a warp's 32 loads hit 32 row-strided sectors, half of each used. The fix is the layout,
not the arithmetic: weights repacked at load into tiles of 32 rows, per group the 32 rows' c0
uint4s then their c1 uint4s (64 G uint4 per tile) and the 32 scales, so each warp load is one
contiguous 512 B line. Standalone bench vs a verbatim k_gemv, 0 mismatches: K=2560 N=18432 19.0 ->
20.4 GB/s (x1.07, W 4 PD 2), K=9216 N=2560 14.7 -> 20.0 GB/s (x1.36, W 4 PD 4), K=4096 N=2560
16.7 -> 19.1 GB/s (x1.14); W 1 loses (too few warps at N = 2560), W 8 neutral. k_hg (prefill) reads
the same tiles (group stride 64, c1 offset 32, scale stride 32). Tensors are padded to 32 rows; a
merged tensor that is not tile aligned is refused at load. E_LI=0 restores row-major + k_gemv;
default off on sm_61 (k_gemvN, row-major only). Nano pesi, same binary ABBA prompt A: 4B decode
12.40/12.39 vs 11.60/11.57 (+7.0%), prefill chunk 8 18.65 vs 17.76 (+5%); 2B decode 25.16 vs
24.78 (+1.5%), prefill 45.5 vs 43.9. 4k harness: prefill 15.54 -> 16.16, decode 9.23 -> 9.77,
score 11.9764 -> 12.5652, GOLDEN PASS, RSS 200832 KB (198-203 MB run spread); checks_b2 2B+4B ALL
PASS. Note on storage: PTQ1_0 in the GGUF is already base-3 (1.75 bpw); the GPU planes are 2.125 bpw
because the POPC identity needs bit planes, and the decode GEMVs now run at 19-20 GB/s of a ~22 GB/s
ceiling, so a denser on-GPU encoding only pays if its unpack is nearly free.

64. P3 chunked GDN prefill k_gdnc + batched tiny-order k_abt (keep, bit-identical; 4k prefill +9.9%).
k_gdnc: for nt > 1 each warp loads its state rows of M once, runs them through the chunk's tokens in
order (m = m * eg_t, the k / q dot chains, delta, update; k_gdn's per-row per-token arithmetic) and
stores them once; q / k of the chunk staged in shared memory, the double-precision exp/log of the
gates one thread per token. Rows 4 per warp chain x 4 warps (E_GCRW / E_GCNW; 1x2 18.77, 1x4 19.86,
2x4 20.22, 2x8 20.34, 1x16 20.12, 4x8 20.52, 4x4 20.53 tps on the 400-token prefill vs 19.19 with
E_GDNC=0). k_abt: the prompt batch's tiny-order F32 alpha / beta projections (8 lanes, dott_reduce)
ran 4x underfilled warps per token; now 4 tokens per warp, lanes 8t'..8t'+7 with the same chains
and the same width-8 tree. Same-binary 400-token prefill, all streams md5 b1591925: both off 19.22 /
19.20, k_gdnc only 20.60, both 21.46 / 21.44 (+11.7%). 4k harness: prefill 16.16 -> 17.76, decode
9.77, score 12.5652 -> 13.1725, GOLDEN PASS, RSS 202552 KB; checks_b2 2B+4B ALL PASS (4B prompt A
chunk 8 prefill 18.70 -> 21.15, 2B 46.1 -> 49.8).

65. Batch rms sums once per token, k_ssqn (keep, bit-identical; 4k prefill +5.3%). A fresh 4k nvprof
split (after rows 63-64) showed k_norm_xq at 4.40 ms/token in prefill vs 1.12 in decode: with
nt > 1 every one of the E / BS Hadamard blocks of every token re-ran the exact serial warp sum
wsum_sq (decode reads k_ssq's precomputed sum). Now one warp per token computes it once (k_ssqn,
same wsum_sq) and k_norm_xq reads sp[blockIdx.y]. E_SSQN=0 restores the old path. Same binary
400-token prefill ABBA, md5 b1591925: 23.11 / 21.45 / 21.42 / 22.84 tps (+7.2%). 4k harness:
prefill 17.76 -> 18.70, decode 9.76, score 13.1725 -> 13.5097, GOLDEN PASS, RSS 200936 KB;
checks_b2 2B+4B ALL PASS. Same split at 4k, decode ms/token: k_li K=2560 42.3, k_fvg 14.0,
k_li K=9216 11.5, k_fs 7.2, k_gdn 6.4, k_li K=4096 5.4, k_xq 2.5, k_fw 2.1, k_ab 1.4, k_norm_xq 1.1
(wall 103.7); prefill: k_hg K=2560 22.6, K=9216 7.6, K=4096 3.5, k_fvg 6.8, k_norm_xq 4.4 (now
fixed), k_fs 3.4, k_gdnc 2.4, k_xq 2.3, k_fw 1.0, k_abt 0.9 (wall 59.6).

66. k_fvg mixed ALU / F2F f16 rounding (measured out, not shipped). The exhaustively certified ALU
rounding (add/sub of an exponent-derived constant, 0 mismatches over all 2^32 floats bar the sign
of zero) was put on NA of the 4 chains per thread, with the per-step overflow check replaced by a
running FMNMX max and an exact f16r recompute pass if it ever exceeds 65504. Standalone bench
($SP/p6/fv.cu: n=4000, 16 heads, 4 KV, k_fvg<2,2> shape, random V and online-softmax weights with
rescales), bits identical to base in every arm: NA=1 0.90x, NA=2 0.85x, NA=3 0.46x, NA=4 0.45x of
the all-F2F kernel (1.68 ms). The kernel is issue bound, not F2F bound: the 3-4 extra ALU ops per
update cost more than the quarter-rate conversion they replace. With the earlier all-ALU (row 60)
and Veltkamp variants (loop rows 9, 11), the k_fvg rounding lever is closed.

67. Certified FP32 expf in k_fw, fw_expf (keep, bit-identical; 4k score +0.45%). The softmax
weights only take x <= 0. fw_expf evaluates glibc's own formula in float-float on the FP32 pipe:
k = rne(x * 32/ln2) with 32/ln2 split exactly into three floats, r with two-products, the same
degree-3 polynomial and the same 2^(i/32) table as hi/lo floats, ~2^-45 relative. The FP64
gm_expf runs instead when |r| is within 2^-20 of 0.5 (k tie), when the value is within 2^-38 of
a float rounding midpoint, or when x < -87 (subnormal results). Exhaustive check on the Nano
against gm_expf over every negative float and +0: 0 mismatches; throughput loop 70 vs 205 ms.
E_FEXP=0 restores gm_expf. 400-token prefill md5 b1591925 unchanged. 4k harness ABBA (E_FEXP
1/0/0/1): score 13.5765 / 13.5272 / 13.5202 / 13.5837, decode 9.82 / 9.78 / 9.77 / 9.82, prefill
18.77 / 18.71 / 18.71 / 18.79, GOLDEN PASS all four; checks_b2 2B+4B ALL PASS.

68. PQ2_0 instead of PTQ1_0 for the 2B and 4B (measured out for speed; kept as an input format)
and a table-driven PTQ1_0 repack (keep: load -6 s on the 4B, bit-identical planes). The question
was whether the 2.125 bpw PQ2_0 GGUFs run faster on the Nano than the 1.75 bpw PTQ1_0 ones.
(1) Same weights: every ternary tensor of the PrismML PQ2_0 files has exactly the PTQ1_0 file's
codes and fp16 scales (2B 151 tensors, 4B 201; code 3 never used; all F32 tensors byte-equal).
(2) Same GPU bytes: the engine already re-encodes PTQ1_0 into two bit planes, 34 bytes per 128,
the size of a PQ2_0 block. (3) The only compute difference would be the fork's PQ2_0 numerics,
Q8_K activations (one scale per 256) and one integer sum per 128-weight group. Standalone k_li
bench with that epilogue against today's 4-FMA Q8_0 epilogue, Nano, jetson_clocks, 4B and 2B
shapes, two runs: 0.997x to 1.016x (gate+up 19456x2560 0.642 vs 0.642 ms, down 2560x9728 0.365
vs 0.360, head 248320x2560 7.920 vs 7.875): k_li runs at 18-21.4 GB/s against the 21.9 GB/s
ceiling, so the epilogue no longer matters; not worth a second activation path and new goldens.
(4) Load. The engine now also reads PQ2_0 tensors (type 142; same planes, same stream: 351- and
405-token zh prompts, 64 ids identical). Cold ABBA on the Nano, before the repack change: 2B
repack 5.2 s PTQ1_0 vs 2.3 s PQ2_0 (wall 15.14 / 15.13 vs 12.23 / 12.25 s), 4B 11.9 vs 5.3 s
(wall 37.20 / 37.26 vs 30.61 / 30.65 s); prefill and decode equal within 0.3 % (4B 22.71 /
12.15-12.24 vs 22.70 / 12.14). Cold `cat` of the 2B files: 1.36 s vs 1.65 s, so the gap was
the CPU base-3 decode (per-weight LUT into t[128], then a 128-step bit loop), not the card.
Fix: per-byte plane patterns (TPL/TPH for qs[0..15], TP2 for qs[16..23], TP3 for qh) OR-ed in
shifted by the byte's position, 26 lookups per plane per group. E_RPCHK=1 re-runs the old
decoder on every chunk: 0 chunks differ on the 2B and 4B. 4B cold ABBA after: PTQ1_0 repack 5.6
/ 5.1 s vs PQ2_0 5.2 / 5.2 s, wall 30.83 / 30.37 vs 30.55 / 30.50 s, stream identical. PTQ1_0
keeps the smaller file (429 vs 517 MB, 950 vs 1146 MB) at the same speed. checks_b2 2B+4B ALL
PASS. The 9B stays on PTQ1_0 (2.0 GB vs ~2.4 GB on a 14 GB card).
69. 2B 4k divergence root-caused and fixed: RoPE sin from the fork's fused sincosf (keep; fixes the 2B,
no speed change). The 2B's 4k stream left the fork's at step 21 of 64. Bisecting the prompt length
on the desktop (fork_greedy, one thread, ubatch 32, vs the engine and fwd35_ref): step-0 logits
bit-equal at 256-1664 tokens, different from 1728 on; at 1700 the engine equals the host
reference (0 of 248320 differ) and both differ from the fork (max 0.138), so the reference
missed a fork behaviour. Per-tensor dumps put the first difference at position 1699, layer 3,
one Q element (head 0, index 54 = second half of RoPE pair i=22) and the K at the same position.
Cause: ggml's rope_yarn calls cosf(theta) and sinf(theta) on one argument and GCC fuses the pair
into sincosf. x86-64 glibc 2.27's __sincosf_fma differs from sinf only for sin at 2^-27 <= |x| <
2^-5 (783266 floats, exhaustive; cos and every other range identical). Disassembly: the
vectorised small-range code evaluates the inner term as fma(x^3, SS1, SS0) where sinf uses x^2,
then sin = fma(x^2 cx, x, x). gm_sincos_sinf in glibc_mathf.h transcribes that; exhaustive check
over all floats: 0 sin and 0 cos differences against libm sincosf. Before the change, sincosf plus
the existing dst1 form reproduced all 61440 Q/K RoPE outputs of the fork's last ubatch in the
six attention layers. 217 of 131072 (pos, i) angles below 4096 differ, and most of those round
away, which is why only long prompts showed it. After: 2B step-0 logits bit-equal to the fork at
1700, 1728, 1792 and 2048 tokens (engine and reference); 2B 4k 64-token stream equals the fork's
on the desktop and on the Nano; 9B 4k stream equals the fork's (unchanged); 4B 4k golden PASS at
18.77 / 9.82 (same as before); checks_b2 2B+4B ALL PASS. The fork streams are committed as
golden_b2_2b_4k.txt / golden_b2_9b_4k.txt. checks_b2.sh and measure_b2_4k.sh now also copy
glibc_mathf.h to the board (they copied only the .cu and built against a stale header).
70. P8 decode GDN fusions k_convl2 + k_gdnf (keep, bit-identical; 4k decode +0.36%). Per GDN layer at one
token, k_ab + k_conv + k_l2 + k_gdn + k_gnorm (5 launches) became 2: k_convl2 (conv + silu + state
shift + the q/k l2 norm, one 128-thread block per 128-channel head, sum from shared) and k_gdnf
(block h computes its own beta/alpha rows in k_ab's warp order, both the AVX2 and the tinyBLAS
order, runs k_gdn's rows with outputs kept in shared, then k_gnorm's gated rms norm of its 128
outputs). k_conv stays out of k_gdnf because the q/k channels are shared by NV/NK blocks and the
conv state update would race. E_GF=0 restores the old chain. Local step-0 logits bit-equal (1-token
prompt), streams identical (tiny and non-tiny order); checks_b2 2B+4B ALL PASS; 4B 4k ABBA
(GF 1/0/0/1): decode 9.89 / 9.84 / 9.83 / 9.85, prefill 18.77-18.78 all four, golden PASS. The win
is far below the ~2 ms the handoff estimated: in a CUDA graph on the Nano the small kernels cost
little beyond their memory traffic, which the fusion keeps.
71. Warp-form FWHT + Q8_0, k_xqw / k_norm_xqw (keep, bit-identical; 4k prefill +2.7%, decode +1.1%).
A Hadamard block of 512 runs on 16 / Q warps with Q elements per lane: stages len < 32 by shuffle
(lane ^ len), the next log2 Q in registers, the rest through shared (2 barriers each); every
butterfly is the shared-memory form's (u + w low, u - w high, same operands, increasing len), then
q8word per word. E_XQW=0 restores k_xq / k_norm_xq; E_XQWD (one token, default 2 = 8 warps per
Hadamard block) and E_XQWP (token batch, default 4) pick Q. Only BS == 512 uses it. Nano 4B,
96-token decode: Q 16 / 8 / 4 / 2 / 1 = 11.86 / 12.36 / 12.58 / 12.67 / 12.58 vs 12.49 for k_xq
(ABBA); 390-token chunk-8 prefill: 23.19 / 23.47 / 23.57 / 23.47 vs 22.81. Local logits and
streams bit-equal for every Q. 4k ABBA (XQW 0/1/1/0): prefill 18.80 / 19.30 / 19.30 / 18.78,
decode 9.86 / 9.95 / 9.97 / 9.85, golden PASS; checks_b2 2B+4B ALL PASS.
72. P4 k_fs part, k_fsb: chunk-batched scores with a transposed reduce (keep, bit-identical; 4k prefill
+3.9%). One block per KV group and 256-position tile scores all chunk tokens (<= 8), each K row
loaded and converted once per chunk. With 32 / GQ positions per warp pass, the pass's 32
dotg_reduce trees run transposed (stages 16, 8, 4, 1, 2: a lane keeps half its sets and swaps
the other half, 31 shuffles per 32 scores instead of 160; every add is lower + upper, which
commutes bitwise). nvprof, 1500-token prompt: k_fs 2.12 s, k_fsb with plain dotg_reduce 1.73 s,
transposed 0.875 s. The transposed reduce alone (k_fst, one token per block) was 2.20 s, slower,
which matches loop row 9 (P5 measured out): it pays only once the K loads are shared. E_FSB=0
restores k_fs, 1 = k_fsb with dotg_reduce. 4k ABBA (FSB 0/2/2/0): prefill 19.27 / 20.04 / 20.04 /
19.29, decode 9.95 / 9.95 / 9.94 / 9.94, golden PASS; checks ALL PASS.
73. k_fvg straight-line groups + k_fvg<2, 4> for chunks (keep, bit-identical; 4k prefill +3.7%, decode
+2.3%). When none of a group's FVU x HG weights carries the rescale sign (warp-uniform, one OR of
the sign bits), the group runs without the per-weight branch; SASS showed the branch/SYNC
skeleton beside the F2F chains. E_FVF=0 restores the old loop. For token batches, 4 heads share
each V conversion (k_fvg<2, 4>, E_FVGP, default 2); at one token it has only NH / 4 blocks and is
0.4% slower at 4k decode (10.09 / 10.05 / 10.04 / 10.08), so decode keeps k_fvg<2, 2> (E_FVGV).
nvprof 1500 tokens: k_fvg 2.76 s (FVF=0) -> 2.40 (FVF) -> 2.13 (FVF + <2, 4>). 4k ABBA
(old/new/new/old): prefill 19.88 / 20.63 / 20.61 / 19.88, decode 9.89 / 10.12 / 10.11 / 9.89, golden
PASS; checks ALL PASS. Score 13.60 (row 70) -> 14.44.
74. k_fvg load prefetch (E_FVP) + lean FD=2 loop k_fvh (E_FVH) (keep, bit-identical; 4k score
+0.4%), with two measured-out k_fvg ideas. nvprof, decode k_fvg<2, 2>: IPC 3.15, issue slots 65%,
stall exec-dependency 71%, memory dependency 0.4%: the kernel is issue-bound, and the loop
around a group (predicated loads, 64-bit addresses, __half array shuffles) costs about as much
as the ~60-instruction chain body. E_FVP=1 issues the next group's loads before the current
chain: 4k ABBA (0/1/1/0) decode 10.11 / 10.17 / 10.18 / 10.11, score 14.45 -> 14.48. k_fvh
(FD 2, HG 4 or 2) runs full groups unpredicated, weights as one float4 per head (ctx % 4 == 0,
group start % 4 == 0), V as one half2 word per position (__low2float / __high2float); groups
with a rescale weight and the tail keep k_fvg's per-position form, same arithmetic and order.
4k ABBA (FVH 0/1/2/2/1/0): prefill 20.60 / 20.74 / 20.61 / 20.62 / 20.75 / 20.62, decode 10.16 /
10.13 / 10.17 / 10.17 / 10.19 / 10.17, score 14.47 / 14.49 / 14.48 / 14.48 / 14.54 / 14.48; groups
of 8 (E_FVH=2) give nothing. Local 2B logits + streams equal for FVH 0/1/2 x chunk 8/5/1;
checks_b2 2B+4B ALL PASS.
Measured out: (a) f16r on the FP32 pipe (C = 1.5 * 2^(e + 13), (y + C) - C, sign OR; exact for
|y| < 2^15, checked exhaustively) to offload F2F pairs: longer dependent chain, more issues;
4k ABBA E_FVF 1/3/5: score 14.26 / 13.41 / 12.94. (b) larger decode groups E_FVU 4/16/32 (with
prefetch): decode 10.18 / 9.55 / 8.65 (register pressure, occupancy), not memory latency.
75. k_fs register cap: k_fs<4, FK, MB> with FK = 2 K rows per warp pass and __launch_bounds__(256, 1)
(keep, bit-identical; 4k decode +0.7%). nvprof, decode k_fs<4> at FAK: 97 registers, achieved
occupancy 0.24, stall exec-dependency 58%, memory dependency 17%, IPC 3.17. E_FSK = 10 FK + MB
picks the variant (0 = old FAK form). 4k sweep: 0 10.08, 24 9.98, 23 10.13, 21 10.15, 81 10.05, 43
10.11, 0 10.09 decode. ABBA (21/0/0/21): decode 10.14 / 10.08 / 10.08 / 10.17, score 14.50 / 14.46
/ 14.46 / 14.52, golden PASS. Same per-row chain and dotg_reduce order, only how many rows a pass
holds. Measured out alongside: k_gdnf next-step prefetch (E_GPF) and 4 warps (E_GNW=4): ABBA
decode 10.14 / 10.12 / 10.14 / 10.14 / 10.14 / 10.15, k_gdnf is at its memory rate. E_PROF=1 calls
cudaProfilerStart after prefill, so `nvprof --profile-from-start off` profiles decode only.
76. One-F2F exact f16 rounding in the k_fvh V chain (E_FVR, default 3; keep, bit-identical; 4k score
+0.7%). SASS of k_fvh: ~95 instructions per 4-position group, 40 of them F2F (f16r is two
conversions), ~21 cycles per warp chain step per scheduler, consistent with conversions at 16/clk/SM.
f16rr: h = F2F.F16.F32(|y|), f = float(h << 13) * 2^112 (the shift is one BFI), then y's sign OR'd
back. Equal to __half2float(__float2half_rn(y)) for every |y| < 65520 including f16 denormals
(build has no ftz), checked over all 2^32 inputs on the host and on the Nano's sm_53 (0
mismatches). Overflow / NaN inputs give f >= 65536, so the chain keeps om = max(om, f) and a block
whose om >= 65536 reruns its chain with exact f16r: no stream can differ. Microbench (1024 threads x
4 chains x 4096 steps): 2xF2F 1.60 ms, shift+scale 1.58, +max 1.71; it helps in the kernel because
the F2F pipe is the shared limit there, not the dependent latency. 4k ABBA (FVR 0/1/3/3/1/0):
prefill 20.80 / 20.73 / 20.89 / 20.88 / 20.71 / 20.73, decode 10.12 / 10.20 / 10.22 / 10.22 / 10.22 /
10.12, score 14.51 / 14.54 / 14.61 / 14.61 / 14.55 / 14.48, golden PASS. Locally (2B, 700 ids,
chunks 8 / 5 / 1) streams and logits dumps equal with FVR 0 / 3 and with the fallback forced.
77. Block-parallel exact rms sum k_ssqb (E_SSQ, default 2; keep, bit-identical; 4k decode +3.4%).
The one-warp wsum_sq of a 2560 row took 83 us on its own and sat on the critical path after the o-proj
and down GEMVs (the st2 k_ssq fork could not overlap: the rows land at the GEMV's end). Counted per
sum (gaussian): ~92 warp passes, ~10 binade exits (s doubles from ~1 to 2560) and ~2 ties. A
one-warp V-terms-per-lane scan (wsum_sqv) cut passes to 22 but each costs ~2.6 us: 52-57 us. A block
iteration per event (256 threads) was 12.8 iterations x ~5000 cycles: 60-68 us. Kept form, bsum_sq:
the terms are squares, so s is monotone; a float prefix scan predicts the binade of s before every
term, each term is rounded to its predicted ulp (events: ties, r >= 2^52, a predicted binade change
after the term), one unsigned 64-bit block scan gives the segment sums (event terms contribute 0; r
< 2^52 and n < 4096 rule out wrap), and thread 0 walks the events with real double adds, accepting a
segment only if s lies in the predicted binade and the segment sum keeps it below 2^53. Any
rejection, a non-finite float prefix or more than 64 events resumes the per-event block iteration
from the last accepted point. Bench (Nano): 0 mismatches of 32768 rows over 8 distributions (n
100 / 2304 / 2560 / 4096, ties, huge range, leading zeros) against the host sequential sum, also with
the chain forced to break at events 0-7 and before the final segment; 15.1 us per 2560 sum (256
threads, 106 registers). E_SSQ=2 drops the fork: every one-token norm_xq runs the one-block k_ssqb
first (also the embedding norm, which did the warp sum in every block). 4k ABBA (1/2/2/1): decode
10.22 / 10.54 / 10.58 / 10.20, score 14.60 / 14.83 / 14.87 / 14.59, golden PASS. Measured out: the
same speculation in one warp per chunk (wsum_sq2, exact): 51-60 us alone, 206 vs 270 us per
64-token gaussian grid, ~0.4% of prefill for k_ssqn; not taken.
78. Compact-code k_ssqb (keep, bit-identical; 4k decode +1.4%). k_ssqb cost 15 us warm in the bench but
35.7 us per call in the engine (nvprof, E_GRAPH=0, 4B decode): it runs once per norm between big
GEMVs, so every call starts with a cold instruction cache, and the fully unrolled TM-term register
arrays made ~4800 SASS instructions (thread-0 phase clocks on the 64 real attn_residual / l_out rows
of a 4k dump: 4.5 events per row, fast path always, but ~4900 cycles to round 10 terms). Steps,
engine us per call: branch-free bs_round + smem-staged coalesced loads (warm 8.8 us) 28.3; rolled
loops re-reading the row from smem in three passes (P0 float sum, P1 rounded sum + event count, P2
event records; 1614 SASS, warm 13.8) 21.6; the first two events of a thread kept in registers in P1
so P2 is a few stores (re-walk only for > 2 events; warm 10.7) 19.3; 512 threads x TM 5 (TM 8 for
E 4096) 16.7. 1024 x 3 is slower (warm 19-24 us: the NW = 32 cross-warp loops); #pragma unroll 2 in
P1 gains nothing. Bench: 0 mismatches of 32768 rows (8 distributions) at 256/512/1024 threads, also
with the event chain forced to break (BS_TESTFB), and 0 of 64 real rows. 4k ABBA (1/2/2/1, prebuilt
binaries): decode 10.57 / 10.72 / 10.72 / 10.58, score 14.85 / 14.96 / 14.96 / 14.86, golden PASS.
Cold: 20.86 / 10.71 / 14.95. Short decode 2B 27.3 -> 28.1, 4B 13.2 -> 13.4 tok/s.
79. Prompt-batch rms sums on k_ssqb + float4 alpha/beta GEMV k_abt2 (keep, bit-identical; 4k prefill
+2.0%). (a) E_SSQN=2: the per-token rms sums of a prompt batch (norm_xq, nt > 1) run as one 512-thread
k_ssqb block per token (TM 5, TM 8 for E 4096) instead of one k_ssqn warp per token: 102 -> 58 us per
call (nvprof, 480-token 4B prefill). ABBA vs E_SSQN=1 (1/2/2/1, one build): prefill 20.89 / 21.05 /
21.03 / 20.86, decode equal. (b) E_ABT=2: k_abt (tinyBLAS-order fp32 alpha/beta GEMV, 2 NV = 64 rows x
E 2560 per chunk, 0.93 ms per prefill token) keeps its 8 chains per (row, token) and its tree. The 8
chains now sit on 2 lanes (4 each), so one LDG.128 of w and one LDS.128 of x feed 4 FMAs. The 8
tokens of a block are staged in smem 256 columns at a time (stride 66 float4: conflict-free) with a
fully unrolled 16-load chunk loop. Per call (nvprof): k_abt 312 us; smem-staged scalar form 232;
float4 with 512-column chunks, unroll 4: 156 (memory-dependency stall 79%, occupancy 0.27);
256 / unroll 8: 125; 256 / 16: 115; 128 / 16: 116; 256 / 32: 115; 512 / 64: 135 (registers). The
480-token md5 is unchanged in every variant. ABBA vs E_ABT=1 (1/2/2/1, one build): prefill 21.06 /
21.31 / 21.30 / 21.05, decode 10.71 / 10.70 / 10.72 / 10.72, score 15.02 / 15.10 / 15.11 / 15.02,
golden PASS. Cold: 21.39 / 10.71 / 15.14.
Measured out:
- 16-token prefill chunks (NTM 16, k_hg NT 16): bit-exact but prefill 18.52 / 18.52 vs 20.86 / 20.88
  (ABBA of prebuilt binaries).
- k_hg RB 1 / 3 / 4 at NT 8 and 16: all slower than RB 2 / NT 8.
- k_hg sits near the issue floor of its form. Per k-iteration SASS is 455 instructions (256 HFMA2,
  ~90 decode / MOV, 40 LDS, 32 F2F, 16 FADD, 16 FFMA); issue ~78%, not_selected 27%. Larger tiles
  lose registers and occupancy.
NTM is now #ifndef-overridable (k_gemvN's template parameter is renamed NTN to avoid the clash).
80. Transposed warp reduce in prefill k_gdnc + 8 rows per warp + v staged in smem (keep, bit-identical;
4k prefill +2.2%). k_gdnc's per-row dt and output sums ran RW independent dotg_reduce trees (5 shuffles
each). dotg_reduceT reduces RW rows together: at stage k with offset st (16, 8, 4, 1, 2: dotg_reduce's own
order) each lane keeps half its values and adds the partner's other half, so RW rows cost RW-1 + the
remaining stages of shuffles, and every row's sum has exactly dotg_reduce's association (IEEE add is
commutative, only the tree shape matters). The row sum lands on lane gtr_base(u); dt is broadcast from it,
outputs are stored by the lanes with !(lane & gtr_rest). k_gdnc is now <RW, NT, TR>; E_GTR=0 restores the
old reduce, E_GCRW the rows per warp (default 8, 4 warps). nvprof per call (480-token 4B prefill): old
RW4 NW4 806 us; TR RW4 NW4 665; RW2 NW4 675; RW8 NW4 544-555; RW4 NW8 777; RW8 NW2 614; RW2 NW8 653; RW16
NW2 563; RW16 NW4 580; RW8 NW4 + the token's v row staged in smem 463 (-42%; issue 69%, IPC 3.15, occ
0.29, 93 regs). RW16 x NW8 and RW8 x NW16 fail to launch. All variants md5-identical on the 480-token
32-gen run. Measured out: k_gdnq, a row per lane quad (E_GDNC=2 while it existed), bit-identical but 960 us
(IPC 1.46, occ 0.24, 113 regs); __launch_bounds__ caps worse (6 blocks 971, 8 blocks 1441 us with spills).
4k ABBA A = E_GTR=0 E_GCRW=4 vs B = default (1/2/2/1, one build): prefill 21.32 / 21.80 / 21.80 / 21.33,
decode 10.72 / 10.71 / 10.71 / 10.76, score 15.12 / 15.28 / 15.28 / 15.15, golden PASS. Cold: 21.88 /
10.73 / 15.32. checks_b2 2B 4B ALL PASS. Decode k_gdnf has the same shape but is memory-bound (row 75): not
applied there.
81. Transposed amax / |q|-sum reduce in the Q8_0 activation quantizer (keep, bit-identical; 4k prefill
+0.5%). q8word ran two 5-shuffle all-reduces (fmaxf amax, integer |q| sum) per 32-element word; a lane of
fwht_q8w holds Q words. q8words<Q> reduces the Q words together (offsets 1..Q/2 transposed, lane r ends
with word r, then the remaining xor stages): Q-1 + 5-log2 Q shuffles per reduce instead of 5 Q, plus Q
broadcasts for amax. fmaxf of non-negative floats and int adds are exact in any order. Used by k_xqw and
k_norm_xqw (E_Q8T=0 keeps q8word). k_xqw was issue-bound (issue 70%, IPC 3.35, ~690 inst per warp at Q 4).
nvprof 480-token 4B prefill totals: k_xqw 525 -> 502 ms, k_norm_xqw 195 -> 175 ms. md5-identical.
4k ABBA E_Q8T=0 vs 1 (1/2/2/1, one build): prefill 21.75 / 21.86 / 21.85 / 21.74, decode 10.75 / 10.74 /
10.75 / 10.75, golden PASS. Cold: 21.85 / 10.76 / 15.33. checks_b2 2B 4B ALL PASS.
82. Exact integer fast path for wsum_sq (keep, bit-identical; 4k prefill +1.0%, decode +0.6%). wsum_sq
(the sequential double sum of float squares x^2 on one warp) walked 32-term chunks with a 64-bit
prefix scan and event handling: k_qk was exec-dependency bound (72% stalls, ~1900 inst per warp).
wsum_sq_fast: warp min of every nonzero term's lowest set bit lo, max of its top bit hi; if hi - lo
<= 52 and the integer total in units of 2^lo is < 2^53 (lane sums saturate at 2^53 before the warp
add), no prefix of the sequential sum can round (terms >= 0, prefix <= total), so ldexp(total, lo)
is the same double; inf/NaN or a wider range falls back to the walk. E_WSF=0 restores. Used by
k_qk, k_l2, k_gnorm and the decode q/k/gated norms. nvprof 480-token 4B prefill totals: k_gnorm 200
-> 110 ms, k_l2 175 -> 82, k_qk 134 -> 107. md5-identical. 4k ABBA E_WSF=0 vs 1 (1/2/2/1, one
build): prefill 21.85 / 22.08 / 22.05 / 21.85, decode 10.76 / 10.82 / 10.83 / 10.75, golden PASS.
(The residual-stream rms sums of k_ssqb, n 2560 with outliers, would mostly fail the range test.)
83. k_conv2: prefill causal conv with the nt inputs of a channel loaded up front (keep, bit-identical;
4k prefill +0.3%). k_conv's per-token chain re-issued each qkv load after the cv store (possible
alias); k_conv2<NTM> (restrict pointers, x[NT] preloaded, 128 threads) runs the same chain.
k_conv 151 -> k_conv2 87 ms (480-token 4B prefill total). Measured out: a thread per (channel,
token) reading its window (157 us vs 105 per call: 4x the loads, no L1 for global on sm_53), and
l2 fused into k_conv2 for the q/k heads (155 ms vs 86 + 83: -14 ms, below ABBA resolution; removed).
4k ABBA E_CONV=0 vs 1: prefill 22.07 / 22.14 / 22.13 / 22.06, decode 10.80 / 10.82 / 10.82 /
10.77, golden PASS. Cold: 22.13 / 10.81 / 15.47. checks_b2 2B 4B ALL PASS (both changes).
84. Mixed-pipe V chain rounding in k_fvh (E_FVX / E_FVXP, defaults 1 / 2; keep, bit-identical; 4k
prefill +0.8%, decode +0.8%). Since row 76 every V chain of a k_fvh thread rounds with f16rr (one F2F +
shift, scale, max, sign OR); the exact f16r is two F2F and nothing else. With all chains on f16rr the
kernel is issue bound (the converse test, E_FVA: V halves converted on the ALU by the same bit form,
exhaustively exact for finite halves with a nonfinite flag forcing the exact rerun, md5-identical but
4k decode 10.46 vs 10.81 and prefill 21.75 vs 22.13). So the first NX chains of a thread now round with
f16r and the rest with f16rr: f16r chains are the reference op itself (inf / NaN propagate natively and
need no om check), the f16rr chains keep the om >= 65536 rerun of the whole thread with f16r. Sweep
(4k prompt + 64 gen, all md5 0b84b6abb1e2): decode HG 2 (4 chains) NX 0 / 1 / 2 / 3 = 10.82 / 10.88 /
10.77 / 10.44; prefill HG 4 (8 chains) NX 0 / 1 / 2 / 3 / 4 / 6 = 22.14 / 22.20 / 22.29-22.31 / 22.23 /
22.16 / 22.02. 4k ABBA A = E_FVX=0 E_FVXP=0 vs B = default (1/2/2/1, one build): prefill 22.14 / 22.30 /
22.31 / 22.12, decode 10.81 / 10.87 / 10.88 / 10.77, golden PASS. Cold: 22.32 / 10.86 / 15.57.
Measured out this round (all md5-identical):
- Decode k_fvh with one dim per thread (E_FVD1, HG 1): 4k decode 10.62 vs 10.81.
- Decode k_gdnf: E_GRW=4 10.81-10.83 vs 10.81 (noise); the k_gdnc transposed reduce in k_gdnf (RW 2 /
  4 / 8): 10.74 / 10.75 / 10.68 vs 10.81. k_gdnf is memory-latency bound (row 75), and the transposed
  form's selects only add issue.

85. Lean decode score kernel k_fsl (E_FSL, default 1; keep, bit-identical; 4k decode +0.4%). k_fs
recomputed each K row address per row (base + min(i0+u, cnt-1)) * NKV * 256 + g * 256 with 64-bit
multiplies, stored each score with its own (h0+h)*ctx+base+i0+u address under a per-head lane==0
predicate, and ran the full predicated dotg_reduce (with broadcast) though only lane 0 uses the sum.
k_fsl steps one row pointer by 8*FK rows per pass (lane folded in), clamps the tail row to a
precomputed last-row pointer, keeps one per-head store pointer and reduces with dotg_reduce0: the
same 5 shuffle-adds without predicates or broadcast, exact for lane 0 (a lane read at a stage is one
the predicated form also updates, read before that stage's add). Same chains, tree, scale: SASS loop
240 vs 301 instructions per 2 positions. Sweep (4k prompt + 64 gen, all md5 0b84b6abb1e2): E_FSL 0 /
1 / 1 / 0 decode 10.86 / 10.91 / 10.97 / 10.87; FK/MB 41 / 22 / 23 / 21 (default) = 10.85 / 10.93 /
10.94 / 10.91-10.97, so FK 2 MB 1 stays. 4k ABBA A = E_FSL=0 vs B = default (1/2/2/1, one build):
prefill 22.31 / 22.31 / 22.31 / 22.30, decode 10.88 / 10.92 / 10.93 / 10.89, golden PASS. Cold:
22.33 / 10.89 / 15.59. checks_b2 2B 4B ALL PASS.

86. Lane-major K cache rows (E_KPERM, default 1; keep, bit-identical; 4k decode +0.6%, prefill +0.6%).
k_qk now stores logical K element 32 * c + lane of a row at lane * 8 + c, so the 8 elements of a
lane's score chain (c = 0..7, the same chain as before) are one 16-byte LDG.128 instead of 8 LDG.U16.
The warp's row read is 512 contiguous bytes either way; what drops is the load instruction count. The
decode score kernel k_fsl and the prefill batch score kernel k_fsb (TR form) take a KP template arm.
k_fs, k_fa and k_fsb's non-TR form index through kix() on a __constant__ c_kperm, so every reader
stays correct under either layout. Chains, trees and scale are unchanged. Decode profile before
(E_GRAPH=0 nvprof, 4k prompt + 64 gen): per token k_li 58 ms, k_fvh 11.0 ms, k_gdnf 8.1 ms,
k_fsl 6.1 ms (8 calls of 764 us for ~8 MB of K each, ~10.5 GB/s). Sweep (all md5 0b84b6abb1e2):
E_KPERM 0 / 1 / 1 / 0 decode 10.93 / 11.03 / 11.01 / 10.91, prefill 22.28 / 22.33 / 22.33 / 22.19;
with KP, FSK 21 / 41 / 42 / 22 / 81 = 11.01 / 11.00 / 10.98 / 11.04 / 10.92, so FK 2 MB 1 stays.
4k ABBA A = E_KPERM=0 vs B = default (1/2/2/1, one build): prefill 22.20 / 22.33 / 22.33 / 22.20,
decode 10.98 / 11.02 / 11.01 / 10.93, golden PASS. Cold: 22.34 / 11.02 / 15.69. checks_b2 2B 4B ALL PASS.

87. Coalesced weight pass k_fwv (E_FWV, default 1; keep, bit-identical; 4k prefill +0.2%, decode
+0.3%). k_fw gave each thread a contiguous chunk of (n + 255) / 256 positions (16 at 4k): every
warp load touched 32 cache lines with no L1 for global loads, every score was read twice (max pass,
exp pass), and thread 0 ran the 256-step chunk-max prefix serially. Each output only depends on its
own score and the max of all earlier scores (prefix max), so any partition gives the same values:
k_fwv gives thread t the float4 groups t, t + 256, ... (one coalesced LDG.128 per group, kept in
registers, at most 8 tiles of 1024 so CTX <= 8192, CTX % 4 == 0), a shuffle scan per tile plus an
8-entry warp table gives each group its exclusive prefix max (carried across tiles), then the same
per-position branch (s > run: -(exp(run - s)), run = s; else exp(s - run)) writes float4 groups
back. fmaxf is order-free on these values; a prefix that differs only in a zero's sign gives the
same comparisons and exp arguments up to a zero's sign, exp(+-0) = 1. Sweep (4k + 64 gen, all md5
0b84b6abb1e2): E_FWV 0 / 1 / 1 / 0 prefill 22.34 / 22.38 / 22.39 / 22.34, decode 11.04 / 11.06 /
11.10 / 11.04. 4k ABBA A = E_FWV=0 vs B = default (1/2/2/1, one build): prefill 22.35 / 22.39 /
22.39 / 22.33, decode 11.02 / 11.06 / 11.06 / 11.04, golden PASS. Cold: 22.41 / 11.06 / 15.74.
checks_b2 2B 4B ALL PASS. Prefill profile at this point (nvprof, E_GRAPH=0, 4k): k_hg 75.2%,
k_fvh 10.0%, k_gdnc 3.1%, k_fsb 3.1%, k_xqw 2.3%, k_fw 1.6% (before k_fwv).

88. k_fvh 8 positions per group (E_FVU8 decode / E_FVUP prefill, default 8, was 4; keep, bit-identical,
md5 0b84b6abb1e2). Same-job ABBA (A = 4/4, B = 8/8): decode 11.07 / 11.07 vs 11.30 / 11.27 (+2.0%),
prefill 22.40 / 22.40 vs 22.48 / 22.50 (+0.4%); fresh score 15.95 (15.74 before). Sweep: decode U=16
11.05 (no gain), prefill U=16 21.63 (slower). Measured out on k_fsl with KP: FSK 24 (MB 4) noise
(11.07 / 11.10 vs 11.07 / 11.06), 25 spills, 14 slower (10.99). checks_b2 2B 4B ALL PASS.

89. Re-sweep of the k_fvh mixed-rounding split under U=8 (measured out; defaults stay). Decode E_FVX
0/1/2/3/4: 11.00 / 11.30 / 11.20 / 11.07 / 10.96 (1 twice: 11.29, 11.31). Prefill E_FVXP 1/2/3/4/6:
22.43 / 22.51 / 22.47 / 22.37 / 22.16. All md5 0b84b6abb1e2. Also measured: RMW of the GDN state
(2 MB per layer, in place) peaks at 18.8 GB/s on the Nano (read only 21.3), so k_gdnf's floor is
~5.4 ms per token against ~8.1 ms; k_li runs at 18-21 GB/s of the 21.9 ceiling.

90. k_li pipeline depth PD 8 for the K=9728 down GEMV (E_LIPD76; measured out, code reverted). Same-job
ABBA (A = PD 4, B = PD 8, 1/2/2/1): decode 11.27 / 11.29 vs 11.28 / 11.27, prefill 22.49 / 22.48 vs
22.49 / 22.48, all golden PASS. CORRECTION: this A/B tested nothing. The 4B has FF=9216, so its down
GEMV is K=9216, G=72 (k_li<72,4,4>); G=76 (K=9728) never runs on the 4B. PD 8 on G=72 is untested.

91. Investigation: decode and prefill are largely clock (compute) bound, not DRAM bound (research,
no code change). GPU clock sweep, 4B prompt A, 64 gen, EMC unchanged: decode 921.6 / 768 / 614.4 /
460.8 MHz = 13.52 / 12.36 / 10.37 / 8.04 tok/s, while the streaming read bench (rmw.cu) gives 21.34 /
20.92 / 20.84 / 19.83 GB/s and the in-place RMW 18.41 / 18.30 / 17.85 / 16.61. A 33% clock cut
costs the bus 2.3% but decode 23%. Per kernel (nvprof summary, E_GRAPH=0, 921.6 vs 614.4 MHz):
k_li<20> 641 -> 866 us (+35%), k_li<72> 342 -> 446 (+30%), k_li<32> 166 -> 210 (+26%), k_gdnf 338 ->
374 (+10%, memory), prefill k_hg<20> 2.80 -> 4.09 ms (+46% of a possible +50%, compute bound),
k_hg<72> 1.88 -> 2.71 ms. Budget per 4B token at 921.6 MHz: 31.3M groups x 4 words x 8 POPC =
1.0e9 POPC at 32/clk/SM (sm_5x quarter rate) = 34 ms of POPC pipe; ~30 issued instructions per
word = ~37-40 ms of issue; 1064 MB at 21.3 GB/s = 50 ms of DRAM; measured k_li ~57 ms. So k_li
sits on a joint compute + memory floor with partial overlap; row 68's "18-21 of 21.9 GB/s, at the
ceiling" was right about bandwidth but not about why the gap stays. Consequences: (1) a denser
base-3 GPU encoding (34 -> 28 B per 128) would add unpack ALU to a POPC/issue-bound kernel: not
worth building. (2) A T-MAC nibble LUT costs ~5 instructions per 4 weights per plane (~80 per 32
weights vs ~30 + 8 POPC now) and the fp16 HFMA2 path costs ~48 per word for one token: both worse
for decode; a POPC / HFMA2 split balances at ~4% HFMA (issue then binds). (3) The levers left are
fewer issued instructions per word in k_li (activation LDS amortized over 2 rows per lane, ~5% of
issue) and better compute/memory overlap; prefill k_hg is fp16-pipe bound (NT 16 measured out,
line 1553). nvprof --metrics profiled no engine kernel after k_feinit (also with --kernels on the
mangled name); the clock sweep was the substitute. GPU devfreq: set max then min, restore 921600000.

92. k_li / k_gdnf geometry levers after row 91 (all measured out, bit-exact, code reverted). 4B prompt A
64-gen screens at 921.6 MHz, ids md5 af77ae304be1 on every arm. (a) k_li2, two row tiles per warp
sharing one activation LDS (E_LIRR=2): decode 12.52 vs 13.53; with PD halved 13.06 vs 13.51 (the
amortized LDS is lost to register pressure / fewer loads in flight). (b) word-major activation
layout in smem with a 1-word tw1 (E_LITR): 13.33 vs 13.51. (c) 8-warp k_li blocks (E_LIW8): 13.46
(K >= 6144 only) / 13.37 (all shapes) vs 13.52. (d) k_gdnf<2> capped at 32 registers
(__launch_bounds__(64,32), 0 spills) so all 32 head blocks are resident in one wave instead of ~25
+ 7: 13.64/13.66 vs 13.65/13.66, i.e. the k_gdnf gap to the RMW bench is not a wave tail. The k_li
launch geometry is exhausted.

93. k_gdnf software prefetch of the next RW rows' state ahead of the two dotg reductions (E_GPF; measured
out, bit-exact, code reverted): decode 13.52 / 13.50 vs 13.52 / 13.51 (also with 4 warps per head:
13.53). With rows 92(d) and 93 the k_gdnf 0.338 ms per layer vs 0.253 ms RMW bench gap is neither a
wave tail nor load latency; what is left is the read-to-write turnaround of a 4 MB in-place RMW.
nvprof GPU summary at 921.6 MHz (E_GRAPH=0, 4B prompt A, 64 gen) for reference: k_li<20> 638 us,
k_li<72> 340, k_gdnf 338, k_li<32> 166; the small decode kernels (k_ssqb 13-23 us, k_xqw 12-24,
k_norm_xqw 12, k_convl2 33, k_argmax1 164 per token) together are ~4% of a decode token.

94. One-token decode scores on k_fsb (transposed reduce, 31 shuffles per 32 scores) instead of k_fsl
(E_FSB1; measured out, golden PASS, code reverted). 4k harness: decode 11.25 / 11.27 (FSB1) vs
11.31 / 11.32 (k_fsl). At 4k, per decode token, attention is 8 x (k_fsl 0.66 ms + k_fvh<2,8> 1.15 ms)
= 14.5 ms; k_fsl moves 8 MB of K per layer at 12.4 GB/s but is not shuffle-bound (this row, and
row 72's k_fst), and k_fvh is the fork-ordered sequential f16 chain, issue-bound at ~10
instructions per position step (4096 chains x ~4000 steps). Prefill k_hg is at its issue floor
(row 79: 455 instructions per 256 HFMA2; HFMA2 alone is ~50% of its time).

95. Structural review (research). Prompt-lookup speculative decoding (no draft model; n-gram 3/2/1
suffix match over prompt + output, exact greedy verify) simulated offline on the frozen streams:
4B 4k golden 1.14 / 1.21 / 1.23 tokens per step for draft length 1 / 2 / 3; prompt A (output-only
lookup) 1.07, prompt B 1.00. A 2-token POPC verify costs ~1.35-1.4 decode steps (k_li issue ~37-40
ms per token doubles against a shared 50 ms weight stream), so the break-even is ~1.4 tokens per
step: measured out on paper. Exact levers left are CPU co-processing (the 4 A57s idle while the GPU
runs the captured graphs): prefill is compute bound (weights at ~3 GB/s), so a row split of the k_hg
GEMVs onto NEON adds compute without contending for the bus; decode is near the bus ceiling, so a
CPU share there is capped by the shared 25.6 GB/s. Lossy levers (f16 GDN state: 192 MB of the
~1.26 GB per decode token; TurboQuant K/V; MTP) are excluded by the 2026-10-16 decision.

96. CPU co-processed prefill (E_CPUF, per mille of each prefill GEMV's rows on the A57s) measured
out. The NEON kernel (MUL.16B + SADALP word sums, 2 rows sharing q loads, lanes = tokens, k_hg's
fma chain) runs 16.4 GMAC/s on 4 A57s and is bit-exact (prompt A md5 af77ae304be1 at every share).
Wiring is in the captured graph: st forks st2, which runs a D2H of the chunk's d8 words and q/2,
a cudaLaunchHostFunc node, an H2D of the rows, then a join and a store kernel. Prompt A prefill
(83 tokens): 24.37 (off) vs 20.95 / 20.20 / 13.01 tok/s at 30 / 60 / 120 per mille. The fixed cost
is ~0.4 ms per split GEMV (host-node wake-up plus 4 copies on Tegra), against a ~2.6 ms GEMV per
8-token chunk whose ideal CPU saving is ~0.26 ms. Below break-even at NTM=8, so the code was
reverted (kernel kept at .auto/attic/fwd35_cpu.c). It would need chunks of more than 64 tokens or a
persistent CPU worker polling a device flag; the latter cannot be expressed in a CUDA graph
without a spin kernel holding the single SM.

97. Larger prefill chunks (NTM 8 -> 16, the prerequisite for amortizing row 96's CPU split, or for
any 64-token tile) measured out. At NTM=64, k_hg needs 68 KB of smem (> 48 KB) and does not
build. At NTM=16 it builds without spills (182 vs 158 registers, 17 KB smem; the sm_61 k_gemvN tile
had to be capped at 8) and is bit-exact: prompt A md5 af77ae304be1, 4k stream 0b84b6abb1e2 =
golden. But it is slower: prompt A prefill 20.30 vs 24.36 tok/s; 4k prefill 20.22 vs 22.68 tok/s
(same job, decode 11.31 both). nvprof (E_GRAPH=0, 3k-token prefill), per token:
- k_hg<20> 0.400 vs 0.350 ms;
- k_hg<72> 0.248 vs 0.235 ms;
- k_gdnc 0.086 vs 0.057 ms.
k_hg is issue-bound per token-MAC (row 79), not per weight expansion, so sharing each expanded
weight word across 16 tokens buys nothing, and the larger register file and smem tile cost ILP.
Bigger chunks are out on the GPU side, and with them the amortized CPU split of row 96.

98. 2B decode: k_li pipeline depth PD 2 for K=6144 (the 2B down GEMV; E_LI48, default 2, was 4;
keep, bit-identical). The new knob E_LI16 sweeps PD for K=2048, and its default of 2 stays (1 / 4 /
8 give 27.06 / 27.57 / 22.46 vs 28.37 tok/s). Same-job ABBA, 2B prompt A decode: PD2 28.64 / 28.71
/ 28.65 vs PD4 28.44 / 28.43 / 28.40 tok/s (+0.8%), prefill equal. 2B 4k golden PASS: decode 24.19
vs 23.98, prefill 57.27 vs 57.46 (one sample each). checks_b2 2B 4B ALL PASS. K=6144 occurs only in
the 2B, so the 4B and 9B are unchanged.

2B decode nvprof per token (~35 ms):
- head k_li: 6.68 ms at 20.2 GB/s;
- 12288-row K=2048 k_li: 0.35 ms each at 19.3 GB/s;
- 8192-row: 18.5 GB/s;
- down: 17.4 GB/s;
- 2048 x 2048: 14.3 GB/s;
- k_gdnf: 18 x 0.16 ms.

Smaller k_li blocks for tensors of at most 2048 rows were measured out (1 or 2 warps: 26.33 /
27.95 vs 28.75 tok/s), because every block refills the shared activation planes.

99. 9B shape tuning (measured out; no code change). The 9B is back on the Nano and the 4B is off
it (SD card space). Prompt A today: prefill 13.3 tok/s, decode 7.8 tok/s (the port started at 6.2
/ 5.7). Decode nvprof per shape:
- 24576 x 4096 k_li: 19.9 GB/s;
- head: 20.7 GB/s;
- 12288 x 4096: 19.5 GB/s;
- 10240 x 4096: 19.4 GB/s;
- down 4096 x 12288: 18.1 GB/s;
- 4096 x 4096: 18.0 GB/s;
- k_gdnf: 0.38 ms per layer;
- the quantizers (k_xq / k_norm_xq) and k_ssqb: ~3%.

Results, all bit-identical (md5 6a03581e518b):
- **k_li PD for K=12288 (E_LI96):** 2 / 3 / 6 / 8 give 7.61 / 7.79 / 7.63 / 7.53 vs 7.74 tok/s at
  4 in the 32-token screen. The 64-token ABBA reverses PD3: 7.72 / 7.77 / 7.74 vs 7.82 / 7.76 /
  7.82 at PD4.
- **k_li PD for K=4096 (E_LI32):** 1 / 3 / 4 give 7.66 / 7.69 / 7.72 vs 7.74 at 2.
- **Warp-form quantizer k_xqw / k_norm_xqw at BS 4096 (P=128):** the 9B is the only model with a
  4096-wide Hadamard block and runs the old k_xq. Q 8 / 16 are bit-identical but give decode 7.67
  / 7.51 vs 7.71 / 7.76 and prefill +1%. Q 4 (1024 threads) fails to launch.

The 4B-tuned defaults stand for the 9B.

## 100. MTP / DFlash speculative decoding for Bonsai-2: estimated at most +4..23% (2B), -4..+14% (9B), not built

PrismML ships MTP (one Q8_0 layer) and DFlash drafts. Fork acceptance (4B MTP, CPU, greedy, 300
tokens, 4 prompts) at draft n=1 is 65-95% (zh-TW 65%, EN 83%, code/list 95-97%); n=2/3 add little
(per-draft acceptance falls to 50-75%).

Verify cost on fwd35: the batched k_hg path costs ~4-4.7 decode steps for any 2-8 token chunk
(E_PCHUNK: 2B 107.6 ms at c2 vs 26.7 ms/token), so a dedicated multi-token decode GEMV was probed:
k_liv (k_li with NV per-token chains from the same weight planes, token 0 exact, md5 unchanged).
Decode with every GEMV at NV tokens, prompt A 64 tokens, same job:
- 2B: 28.61 (NV1) / 20.02 (NV2, 1.43x) / 13.62 (NV3, 2.10x).
- 9B: 7.79 / 4.82 (NV2, 1.61x) / 2.45 (NV3, 3.18x; per-token MAC issue bound, not bandwidth).

With n=1 (pass = 1.43-1.61 steps + ~0.1-0.15 for the MTP layer and a pruned head; GDN state
rollback and nt=2 attention not counted): 2B 1.04-1.23x, 9B 0.96-1.14x, prompt dependent. DFlash
verifies ~16-token blocks (>= 4.7 steps here) and needs ~1.6 GB drafts on a 4 GB board: out.
The sm_53 GEMV is MAC-issue bound once a second token is added, so speculation buys little. The
probe source is in `.auto/attic/fwd35_liv_probe.cu`.
- [LFM3/Nano] Engine runs end-to-end at 5.84 tok/s (171.1 ms/token) on the Nano, logits NOT yet
  matching HF (corr -0.046 vs logits_last_A.npy, hf-top1 ranked 31005). Validated so far: blob CRCs,
  embedding layout row-major (cos 0.99997; col-major 0.008), lm_head host-staging bit-exact vs
  resident (all 128000 logits, all row-slab sizes), logit SCALE correct after adding the missing
  embedding_norm (std 2.167 vs 2.093; was 0.41). So the residual stream has the right magnitude and
  the wrong direction => divergence is INSIDE the 24 layers, not in lm_head/norm/blob.
  NEXT (localization order, cheapest first, repo-standard perlayer_probe style):
   1. position 0 only (--steps 1 --prompt 124894): exercises embedding -> 24 layers with a 1-position
      attention and a ZERO conv ring -> norm -> lm_head. If pos 0 matches HF, the bug is in
      attention/conv STATE at pos>0 (RoPE invf, sliding window, ring indexing, GQA head map).
   2. If pos 0 already fails: check the conv branch math first (13 of 24 layers, B*xg -> 7-tap -> C*y,
      ring layout), then the sandwich-norm placement (operator_norm / post_operator_norm /
      ffn_norm / post_ffn_norm), then attention (head gates: sigmoid(gate_proj) applied to attn
      BEFORE o_proj, per-head q_norm/k_norm, GQA 16q/4kv).
   3. Dump x per layer (--trace hook is the place: max|x|, NaN count, and a 8-float fingerprint) and
      compare to oracle.py intermediates at the same layer.
  Do not trust a self-test that compares two paths built from the same assumption -- the lm_head
  selftest is bit-exact by construction and proved nothing about layout; the independent HF
  comparison is what found the real issues (5fd7e79's lesson, again).
- [LFM3/Nano] Localization of the logits divergence, layer-0 table (engine vs HF, pos 0, prompt
  [124894]). Engine: ./lfm3_nanode <blob> --emb-host --steps 1 --prompt 124894 --trace
  (the --trace hook prints rms of the residual stream after every layer). HF: /tmp/probe_rms.py
  (venvs/lfm3) saves /tmp/hf_pos0_rms.npy, 25 values, embedding through embedding_norm.
      layer type  engine  HF        L0 conv 0.0280/0.0247  L1 conv 0.0929/0.1564
      L2 attn 0.0678/0.1125  L3 conv 0.0707/0.1194  L4 attn 0.0701/0.0764
      L5 conv 0.0871/0.1956  ... L23 conv 0.4938/0.5386  final 2.8949/3.6474
    Already 13% off at L0 and 41% at L1 (both conv layers), monotonically diverging. No state is
    involved at pos 0 (1-position attention, zero conv ring), so this is arithmetic, not cache/RoPE.
  RULED OUT by reading HF's source line by line, not by a self-test (each of these was a candidate):
    conv in_proj split order (B,C,x = BCx.chunk(3,-2) -> my chunk0=B, chunk1=C, chunk2=x matches),
    conv gating order (B*x -> conv -> C*y -> out_proj), conv tap alignment (the size-7 kernel pairs
    the NEWEST input with tap 6; my ring[6]=newest, taps[k]*ring[k] matches), sandwich-norm
    arrangement (post_operator_norm on the branch output before the add, post_ffn_norm on ffn_out
    before the add -- matches), SwiGLU (silu(w1)*w3, matches), Q8 scale/block pairing (dequantized
    the blob in Python: cos 0.99986 vs HF for w1/w2/in_proj, transposed-cos ~0.000), embedding
    layout (row-major cos 0.99997, col-major 0.008), all 286 CRCs, lm_head bit-exact, logit scale.
  NEXT: instrument INSIDE layer 0 on both sides (pre-norm, in_proj, B*x, conv, C*y, out_proj,
  post_operator_norm, add, ffn_norm, FFN, post_ffn_norm, add) -- probe_lfm3_conv.py is the template
  for the HF side. Top suspect: k_rmsnorm (first op of every layer, so it explains an L0 error;
  check mean-count, eps placement, rsqrt, and the weight multiply against Lfm3RMSNorm). Second:
  the Q8 GEMV's row->output-index mapping, which the dequant test cannot see (it validates the
  matrix, not the index map -- a permutation survives every gate that shares the convention).

## Trajectory drift (runs #109-#113): the engine is faithful on-manifold, divergent off-manifold

`.auto/lfm3/traj_drift.py` measures KL(HF || engine) at every state the engine actually
visits, plus `--on-manifold`, which feeds HF's own golden continuation and scores both sides
on it. Measured on the shipped arm (48 steps, 6 prompts):

| mode | states | mean KL | top-1 = HF's | slope (nats/token) |
|---|---|---|---|---|
| on-manifold (golden fed) | 275 | **0.0012** | **273/275 (99.3%)** | ~0 (+/-3e-5) |
| free-running (engine's own tokens) | 288 | 2.60 | 151/288 (52%) | **+0.147 median** |

Read together: the arithmetic is faithful wherever HF's own distribution puts weight (KL
1.2e-3, zero slope over 48 steps, top-1 at 273 of 275 states), and the greedy stream
divergence is the feedback loop leaving that manifold -- after the first near-tie flip, the
engine and HF are answering different questions, and KL grows ~0.15 nats/token from there.
This is what the "0/7 full stream matches" line really means, and teacher-forced gates
structurally cannot see it (they re-anchor on HF's tokens every position).

- **Open control that decides how to read this: the generic divergence floor.** Run HF at
  fp32 free-running and score HF-bf16 on the fp32 trajectory. If a *stronger* HF also
  diverges at ~0.15 nats/token, then +0.147 is what any imperfect decoder does off-manifold
  and the engine's number is normal, not a defect. Without it we are reading a ratio with no
  denominator.
- Gate-design lesson: a growth *ratio* test on near-zero KLs is a false-positive generator
  (prompt K "failed" at x3.3 growth from 0.0005 to 0.0015). The growth test now requires an
  absolute late-window floor too (default 0.02; the free-running signal is at 4.3-5.2, 200x
  above it, so nothing real is masked).
- Alignment is the whole bug surface for this gate: at free-running step t the engine has
  consumed prompt_ids + *generated* tokens; the tokens it sampled while consuming the prompt
  are not context. Feeding HF `toks[:t]` measures a state that never existed (KL 6.6 nats at
  t=0 against a teacher-forced 0.0014 was the tell).
- Do not relax the free-running floors to make the gate green. It is a diagnostic that fails
  on a real property of the shipped engine, disclosed in the README, not a gate to satisfy.

## The divergence floor (run #114): what a known perturbation does, and what the engine does

`.auto/lfm3/divergence_floor.py` scores three models on ONE token sequence (the reference's
own greedy trajectory), so all arms see identical states:

reference = HF **fp16** (chosen because fp32 does not fit: the host's GPUs have 1.9 GiB and
1.3 GiB free and the fp32 model needs 5.7 GiB -- it OOMs. fp16 has a 10-bit mantissa vs
bf16's 7, so it is the strictly more precise reference that fits, which makes the bf16 arm a
quantization perturbation of *known size* -- the denominator this experiment exists for).

| prompt | bf16 HF (quantization floor) | engine Q8 (same states) | ratio | top-1 |
|---|---|---|---|---|
| A | KL 0.0004, slope -1e-5 | KL 0.0013, slope +1e-5 | 3.3x | 47/48 both |
| C | KL 0.0002, slope -1e-5 | KL 0.0005, slope -2e-5 | 2.5x | 47/48 both |
| K | KL 0.0003, slope  0     | KL 0.0006, slope +2e-5 | 2.0x | 47/48 both |

Three conclusions, and they close the quality investigation:

1. **On a shared trajectory nothing grows.** Slopes are +-1e-5 nats/token for both the
   pure-quantization arm and the engine. So the +0.147 nats/token measured by traj_drift in
   free-running mode is NOT quantization drift -- it is trajectory divergence: once each
   model picks its own tokens, the states themselves differ, and state divergence dominates
   per-state logit perturbation. That is a measurement effect of comparing free-running
   streams, and it is the right reading of "0/7 full stream matches".
2. **The engine's per-state error is 2-3x the bf16 rounding of the same weights**, same order
   of magnitude, identical top-1 agreement (47/48). That is the quantitative cost of Q8
   linears + fp32 scales + ggml summation order, against the best perturbation achievable by
   merely rounding the same model.
3. **Top-1 agreement is 47/48 for both arms on all three prompts** -- the engine loses the
   same single near-tie position bf16 loses. It is not losing positions the bf16 model wins.

Open follow-up, if the quantization ratio is ever worth pushing: the ratio's biggest
suspect is the Q8 embedding (the lm_head streams 278 MB/token of Q8 rows at 14 GB/s); the
certified 8/8 arm (bf16 embeddings) is the variant that isolates it, and it measured
161.67 ms/token, i.e. +18% -- a quality/latency trade to state, not to chase silently.

## Non-GEMV kernels are NOT a lever (run #115): the KTIME table's 17 ms was an artifact

The serialized per-kernel table (`LFM_KTIME=1`) shows the four per-layer rmsnorms at
0.125-0.20 ms per call, 96 calls/token, i.e. ~17 ms = 12.5% of the step. That number is
bogus: every timed launch carries a sync tax, and the same table's GEMV rows alone sum to
~139 ms against a 136.3 ms step, so ~+20 us per launch is instrumentation, not work.

Priced directly with a timing-only ablation (`LFM_ABLATE=1`, skips the rmsnorm launches;
**output is invalid** - attribution probe, never a runnable configuration):

    arm1 (shipping)          median 137.44 ms/token
    arm2 (LFM_ABLATE=1)      median 136.70 ms/token
    R = 1.0027  ->  all 96 rmsnorm launches cost 0.74 ms/token = 0.54%

So norm fusion (the natural move: fuse `post_op_norm`+`add1` and `post_ffn_norm`+`add2`)
recovers at most ~0.5%, and the obvious pair `post_op_norm`+`ffn_norm` is NOT fusable at
all - `ffn_norm` reads `x` *after* `add1` has consumed `post_op_norm`'s output, so they are
dependent, not the independent adjacent pair the call sites look like. Not worth touching
bit-exact normalization code for.

Consequence for the search: the GEMVs are ~97% of the step and are a characterized optimum
under ten structural probes. The small-kernel surface is closed too. Remaining levers are
environmental (the Nano reboot to de-bias the config absolutes) and quality-side, not
latency-side.

## The Q8-embedding is NOT the divergence source (run #116): the +18% arm is priced and declined

`divergence_floor.py --prompt A,C` against the certified blob (`lfm3-audio-2b-cert.lfm3q8`,
bf16 embeddings, Q8 linears) vs the shipping blob (Q8 embeddings, Q8 linears), same fp16-HF
trajectory, same bf16 floor:

| prompt | bf16 floor | Q8-emb (shipping) | bf16-emb (cert arm) |
|---|---|---|---|
| A | 0.0004 | 0.0013 (3.3x) | 0.0012 (3.0x) |
| C | 0.0002 | 0.0005 (2.5x) | 0.0003 (1.5x) |

Swapping the embedding quantization changes the ratio by ~8% on A and ~40% on C - small,
prompt-dependent, and the two arms are within noise of each other on A. So the engine's
excess over the pure-quantization floor is carried by the **Q8 linears and the ggml
summation order**, which are identical in both blobs. The embedding hypothesis is dead.

Priced decision: the certified arm costs 161.67 vs 136.30 ms/token (+18.6%) for a top-1 of
48/48 vs 47/48 and this marginal KL change. Declined; shipping default stays the Q8-embedding
blob. If a future objective wants the last token of fidelity, this is the lever and it costs
one sixth of the step - and the remaining 2-3x ratio is NOT reachable by dequantizing the
embedding, which is where it would have been cheapest.

## The 2-3x divergence ratio is the Q8 weights, not the summation (run #117, closed analytically + empirically)

Candidate levers for the engine's excess over the bf16 floor were scale granularity, and
compensated (Kahan/2-sum) accumulation. Both are dead:

* Scale granularity: the blobs are Q8_0 -- one fp32 scale per 32 codes, already the
  fine end of the family. Nothing left to buy.
* Accumulation, analytically: float32 summation of K=2048 products carries relative
  error ~ eps*sqrt(K) = 1.2e-7*45 ~ 5e-6, while per-weight Q8 error is (scale/2)/|w|
  ~ 0.4% which averages over K to ~9e-4. **The accumulation is ~200x below the
  quantization term**, so perfect summation could not move the ratio.
* Accumulation, empirically, from data already on the books (#112): LFM_NOFMA=1 doubles
  the per-product rounding (mul-then-add instead of FMA, 2 roundings vs 1) and KL moved
  0.0014 -> 0.0015 on prompt A, i.e. ~8% of the 0.0013 excess for a 2x change in
  accumulation error. Extrapolating, summation accounts for ~8% of the excess and
  quantization for the remaining ~92%.

A Kahan accumulation would have cost +6-12% FP work on an FP-pipe-bound kernel (the
mechanism of the -2.0% FMA win, inverted) to chase the smaller term. Not built, and the
reason is now on the record: **the 2-3x ratio is the price of Q8 weights, and reducing it
means buying bytes back (fp16 linears = 2x the stream = ~2x the step), not better arithmetic.**

## Stuck-process CPU contention is NOT inflating decode (run #118): the reboot re-sweep is dropped

The three un-reapable `lfm3_conv_test` processes spin cores 0/2/3 (load ~3.9). Paired,
order-balanced A/B of the shipping binary, unpinned vs `taskset -c 1` (the idle core):
free 137.50/137.67/137.73/137.64 (median 137.66), pinned 137.60/137.56/137.66/137.40
(median 137.58), R = 1.0006, 2/4 pairs each way -> null. The launch thread is off the
critical path (same mechanism as CUDA graphs -0.55%, #99). My earlier "absolutes are
inflated, re-sweep after a reboot" caveat was an untested assumption and is withdrawn; the
README is corrected. The stuck processes remain worth clearing for hygiene, not for latency.

## The step lines print SAMPLED tokens, not fed tokens (run #133): #113's free-running number is retracted

Re-reading the step loop while scoping the lm_head skip: `printf(s, nt, ms)` prints nt,
the argmax JUST sampled -- not the fed token. Three consequences:

1. The `LFM_SKIP` idea (skip lm_head on prompt steps, ~13% TTFT) is INVALID and was not
   built: every step's argmax is pushed and steps >= n consume the pushed tokens starting
   with nt_0, so every step's logits are load-bearing. (Whether that shifted continuation
   is the *right* decoding is a separate harness question; the gates certify it as-is.)
2. traj_drift's free-running alignment was wrong for t >= n: it built
   seq = prompt + parsed[n:] (skipping the first n predictions) while the engine actually
   consumed prompt + parsed[0:]. HF was scored on states the engine never stood in, so the
   +0.147 nats/token growth and 151/288 top-1 were misalignment artifacts, not engine
   drift. Corrected (seq = prompt + parsed[:steps]): 6-prompt free-running KL is
   0.0023-0.0072 mean, top-1 45-48/48 per prompt (282/288 total), slopes ~1e-4/tok, TRAJ
   DRIFT OK. The engine stays in-distribution on its own trajectory; 0/7 stream matches is
   pure argmax/tie amplification. The on-manifold control (KL 0.0012, teacher-forced,
   always aligned) and divergence_floor (teacher-forced, always aligned) are unaffected.
3. The gate comment claiming "the token column is the token FED at that step" was wrong and
   is fixed; the stream gate's slicing (parsed[n-1:] starts with the first TRUE token,
   prediction after the full prompt) was already correct, which is why the gates never
   caught this.
4. TTFT baseline corrected: the first TRUE output token completes at step n-1, so prompt-A
   TTFT is sum(steps 0..22) = 2642.89 ms, not 2755.41 (which included the step-23 decode
   of the shifted continuation). -112ms of definition, not performance.

## Batched prefill GEMV, first bring-up (run #135): bit-exact, but 2x SLOWER than sequential

The `q8_prefill<bf16,CT=4>` kernel reuses each Q8 weight block across a tile of 4 prompt positions,
keeping the reduction order identical to `q8_partial`. `--prefill-selftest` compares it on the
embedding tensor (N=128000, K=2048, T=23) against 23 sequential `q8_partial` launches:
**BIT-EXACT (0/2,944,000 partials differ)**, T=1 also bit-exact, but it takes 775 ms against
400 ms sequential (ratio 0.52). The code is not kept, but the design lessons are:

1. Per-lane partials must be **registers**. A shared `acc[r][c]` array is a race, because all 32
   lanes of a warp reduce the same row, so they read-modify-write the same slot. It was
   deterministic and plausible-looking (-0.965 vs -0.110). Bisect kernels found it: a verbatim
   copy of `q8_partial` (v2) and the staging rewrite (v3a) were exact; adding only a shared
   accumulator (v3c) broke it.
2. **x must be staged in shared memory.** Reading x from global was exact but 8x slower
   (3498 ms), because the serial FMA chain waits ~400 cycles on each x load.
3. Staged at CT=4, shared memory is 33.8 KB per block, so only **one block fits per SM** (vs 6 for
   `q8_partial`). It is then latency-bound, and that is the 2x loss. Weight reuse only pays off
   once occupancy is restored. Next design: CT=2 (16.9 KB, 2 blocks/SM) or unpadded bf16 x staging
   (halves shared memory), then a TTFT A/B through the full layer loop. The prize is still about
   17x (#134), so this remains the e2e lever.
- #136: CT=2 (16.9 KB shared, 2 blocks/SM) is BIT-EXACT again (0/2,944,000) and much closer to sequential: 379 vs 334 ms (0.88x, two runs agree). CT=4 was 0.52x. Moving from 1 to 2 blocks/SM recovered most of the occupancy loss, but weight reuse still doesn't beat 6-block sequential. Next: unpadded bf16 x staging (halves shared memory again, giving 4+ blocks/SM at CT=2) before wiring the kernel into the layer loop.
- #137: bf16 x staging (8.7 KB at CT=2) is BIT-EXACT and reaches **1.02x** (325.6 vs 333.3 ms, two runs).
  CT=4 at 17.4 KB gives 0.71x. Occupancy is fixed, but weight reuse still buys only ~2%.

## Correction to #134's "~17x prefill prize": the GEMVs are FP-bound, so weight reuse cannot deliver it

#134 assumed prefill was bandwidth-bound: read the weights once instead of 23 times and get ~17x.
But #94 had already shown these GEMVs are **FP-pipe-bound**, and the launch-bounds win (#121) was a
scheduling win, not a bandwidth one. Batching removes weight *bytes* but not a single FMA: each
(row, position) still needs its 32 int8->float conversions and 32 FMAs. Three tile sizes, all
bit-exact, agree: CT=4 0.52x / 0.71x, CT=2 0.88x, CT=2 + bf16 staging 1.02x. The ceiling for
"same arithmetic, fewer bytes" on this board is about 1.0x, not 17x.

What can still shrink prefill is **less FP work per product**, and every option changes numerics:
- sm_53 has native half2 (HFMA2, 2 products per instruction): about 2x on the FP pipe, but fp16
  accumulation would need the logit/divergence gates to decide whether that's acceptable.
- No dp4a on sm_53 (it's sm_61+), so an int8 x int8 dot product means emulating it, or quantizing x
  to int8 (a Q8_1-style activation quant, as ggml does), which is a reference-order change.
Both are quality trades that must go through the 6-prompt logit gate + divergence_floor, not
through the stream gate. Prefill is still 96% of TTFT; batching alone is not the lever.

## The feeding bug (run #139): "0/7 stream matches" was a harness bug, not numerics. Now 6/7.

The step loop pushed EVERY step's argmax into `toks`, including the n-1 predictions made
mid-prompt. The first generation step therefore consumed toks[n] = step 0's prediction (made
after ONE prompt token), and each later step consumed a stale mid-prompt guess. Only the first
output (step n-1) was ever right. That is exactly why every prompt scored "prefix 1-2", and why
the streams read like garbage ("11)1 four seasons obvious the the the").

Fixed: only predictions from step n-1 onward are appended. With the fix, the lm_head on prompt
steps 0..n-2 feeds nothing, so it is skipped (except under --dump-logits-all, which the
teacher-forced logit gate needs). `LFM_LEGACYFEED=1` keeps the old loop as a control.

| | legacy feed | fixed feed |
|---|---|---|
| streams vs HF goldens (16 tok) | 0/7 full, prefixes 1-2 | **6/7 full**, F = 11/16 (its known near-tie) |
| teacher-forced gates | unchanged to the digit | unchanged to the digit |
| TTFT, prompt A, warm, paired 4/4 | 2593.4 ms | **2271.1 ms (-322 ms, -12.4%)** |
| first output token | 207 | 207 |

6/7 exact is the plan's certified figure for this arm ("q8_0_embq8: 6/8 exact"), now reached.

**Retractions this forces:**
- #113/#133's reading that 0/7 stream matches were "argmax/tie amplification on agreeing
  distributions" is wrong. The distributions did agree (the teacher-forced gates were always
  right), but the streams diverged because the harness fed the wrong tokens.
- #87's "the plan's 8/8 goldens gate is not reachable by a from-scratch GEMV (cuBLAS-order
  dependent)" was built on the same broken streams and is withdrawn. With correct feeding the
  Q8 arm reaches 6/7, matching its certification. The bf16-embedding arm should be re-scored.
- traj_drift's free-running numbers (#133) measured the legacy trajectory; the script now
  follows the fixed feeding (LFM_LEGACYFEED in LFM3_ENV selects the old alignment).
- measure.sh's steady-state "decode ms/token" averages steps 2..47, which includes prompt steps
  that no longer run the lm_head. It is not comparable across this commit; a decode-only number
  needs steps >= n.

## half2 GEMV probe (run #140): -5.6% TTFT at gate-indistinguishable quality, NOT bit-exact, default OFF

`LFM_H2=1` swaps the layer GEMVs' inner loop to exact int8->half2 conversion (byte-permute under
0x64, then HSUB2 1152) plus HFMA2, two products per instruction, with an fp16 accumulator inside
each 32-code block. The accumulation across blocks stays fp32.

- Quality (full gate suite, LFM3_ENV="LFM_H2=1"): matrix 23x23 within 3%, logits0 unchanged, logit gate
  top-1 130/131, KL 0.0006-0.0017 (shipping 0.0007-0.0019), streams 6/7 exact (same as shipping).
  The numbers differ from shipping, so the path ran. Indistinguishable within prompt-to-prompt spread.
- Speed (paired 4/4, order-balanced, cold page cache, prompt A): TTFT 2267.7 -> 2141.8 ms (-125.9 ms,
  -5.6%); steady decode 112.8 -> 107.0 ms/token (-5.2%); same first token.
- The gain is ~5%, not ~2x, so after launch-bounds the GEMVs are no longer cleanly FP-bound; bytes
  and FP now share the limit.
- Not adopted by default because it trades bit-reproducibility with the fp32 path for 5.6%. That's the
  user's call. Before any adoption: run divergence_floor.py on it (shared-trajectory KL vs the bf16
  floor), and watch for fp16 range limits (|x| > 65504 overflows, |x| < 6e-5 goes subnormal). The
  gates passed on 7 prompts, but activations on long or unusual audio inputs were not covered.
- #141, the checks owed on the half2 probe:
  - divergence_floor (fp32 reference, same trajectory, shipped arm reproduces #128 exactly as a control):
    shipped KL A/C/K = 0.0013/0.0005/0.0006; H2 = 0.0016/0.0006/0.0005; top-1 47/48 in both on all
    three. A small, mixed cost (+20% on two prompts, -17% on one).
  - fp16 range on the 7 gate prompts (LFM_XRANGE=1, ~200M GEMV inputs): no input overflow (max |x|
    45.5 in ffn.w2), but conv.out_proj has 28.3% of its nonzero inputs in fp16's subnormal range
    (max |x| only 5.0), and ffn.w2/o_proj ~1%. The in-block fp16 accumulator has only theoretical
    headroom: 45.5 x 127 x 32 = 185k > 65504. It never overflowed here, but nothing bounds it on
    unseen audio.
  - Verdict: H2 stays default OFF. If adopted, first add exact power-of-two rescaling of each
    32-input block (multiply x by 2^k before conversion, multiply the block dot by 2^-k after; both
    exact), which moves conv.out_proj out of the subnormal range and bounds the accumulator.
    Re-gate after that. -5.6% TTFT is the prize.
- #142: `LFM_H2=2` (per-32-input-block exact 2^k rescale into [8,16) before half conversion):
  - Gates: everything passes, and compared with the unscaled H2 probe it recovers prompt E's lost
    top-1 (23/23 vs 22/23) and prompt C's worst KL (0.0240 -> 0.0084 is pre-#139; now E 22->23).
    Streams are **7/7 exact** (F 11/16 before) -- the first 7/7 on any arm. The rescale removed
    exactly the fp16-induced penalty the diagnostic predicted, so the design is confirmed.
  - Cost: all speed gone. TTFT 2267.7 (shipped) vs 2258.1 (median of 4, -0.4%); decode 112.8 vs
    112.5 (-0.3%). The fmaxf/frexpfldexpf staging pass costs what the faster dot loop saves:
    +1 staging pass over x per block plus per-thread max reduction. Raw H2 stays at -5.6% TTFT /
    -5.2% decode. Measured null, keep raw + scaled both behind the flag.
  - Lesson: rescaling arithmetically perfect work back into fp16's good range does not make it
    free -- the scan that finds the scale factor is itself a full pass over x. A free version would
    need the scale from statistics the prefill already computes (e.g. the norm's rms, which is
    max-adjacent, not max -- not a bound). Until such a source exists, H2 stays a measured
    quality/speed option, not the default.
  - Side effect for the log: the feeding fix's stream scores moved the floor -- with 7/7 exact on
    the H2=2 arm, the stream gate now discriminates between arms that differ only in the fourth
    decimal of KL. Keep floor=1 (it is the arm's certified floor); do not chase the gate upward.

## Cert (bf16-embedding) arm re-scored with fixed feeding (run #143): declined harder

`LFM3_BLOB=lfm3-audio-2b-cert.lfm3q8` through the full gate suite (fixed feeding, default FMA path):

- Streams: **6/7 exact** (F 11/16), up from 2/1 -- the feeding fix repaired the cert arm exactly as
  much as the Q8 arm. So the +18.6% latency (161.67 vs 136.30 ms/token decode-era; TTFT gap scales
  the same way) buys ZERO stream advantage: both arms 6/7 with the same F miss.
- Logit gate: 130/131 overall, but **prompt E drops to 21/23 top-1 (floor 22) -> CHECKS FAILED**.
  The worst-KL position (0.0063 at pos 10) shows engine and HF agreeing on the SAME token (12008),
  so the two misses are near-tie flips elsewhere; mean KL is unchanged (0.0017, same as Q8). The
  bf16 embedding is not systematically better -- ties flip either way -- and the floor is the floor.
- Verdict: the #116 decline stands, strengthened. The cert arm now costs +18.6% latency for
  (a) identical streams, (b) a gate failure where Q8 passes. Do not revisit unless the embedding
  changes (different quant, or the tied-embedding lm_head path specifically).
- Side note: E 21/23 vs Q8's 23/23 is a 2-position tie-flip difference at KL 0.0017 in both arms.
  It is evidence FOR the "top-1 counts are cliff-like, KL is the signal" position, not evidence
  the cert arm is worse in any distributional sense.

## Unpadded bf16 staging is dead: 0.09x via 16-way bank conflicts (run #145)

Tried dropping the x-staging pad (stride 32 bf16 instead of 34) to halve shared memory again,
with R=128 to double block count. Result: BIT-EXACT but 0.09x (3540 vs 334 ms), R-independent.
Mechanism, verified by arithmetic: lanes in a warp read different 32-code blocks simultaneously;
with stride-32 halfwords the per-lane addresses step 64 bytes = 16 banks, so 16 lanes pile onto
each of 2 banks -- a 16-way conflict on every x read in the serial dot chain. The pad exists to
avoid exactly this, and the constraints pin it down: stride S must satisfy S >= 32 (blocks must
not overlap) and S/2 coprime to 32 (consecutive lanes must land on distinct banks, i.e. S = 2
mod 4). S = 34 is minimal: 8.7 KB at CT=2, which is #137's design (1.02x). There is no smaller
conflict-free layout for warp-strided-b x reads, so same-arithmetic prefill batching is closed
at ~1.0x from both sides (occupancy via padding, conflicts via stride).

## Norm skip on lm_head-skipped prompt steps: exact, but unresolvable from noise (run #146, discarded)

With fixed feeding, prompt steps 0..n-2 skip the lm_head, which makes the final embedding_norm dead
too (its output feeds only the lm_head). Skipping it as well is exact by construction -- and the
40-step generated streams are identical with/without the skip. But the paired A/B (same session,
order-balanced, cold page cache) is null: keep-norm median 2266.94 vs skip-norm 2267.86 ms TTFT
(R=0.9998, skip faster in 3/4, delta 0.9 ms = 0.04%). The ~8 ms gap seen the day before was
board drift between sessions, not the norm: lesson re-learned -- never compare across sessions.
Reverted: the tree keeps measured wins only, and ~1 ms expected savings (22 launches + 22 x 16 KB
passes) is below what this board resolves. The 10-line patch is trivial to resurrect if a future
pass ever needs the launch count down.

## H2=2 divergence floor completes its quality record (run #147): best-or-tied everywhere

`LFM_H2=2` (exact per-block 2^k rescale) on identical fp32 trajectories, bf16 floor as control:

| prompt | bf16 floor | shipped | H2 raw (#141) | H2 rescaled (#147) |
|---|---|---|---|---|
| A | 0.0005 | 0.0013 | 0.0016 | **0.0012** |
| C | 0.0001 | 0.0005 | 0.0006 | 0.0007 |
| K | 0.0002 | 0.0006 | 0.0005 | **0.0005** |

Top-1 47/48 in every arm on every prompt; slopes ~1e-5 (nothing grows). The rescaled arm is
best-or-tied on all three prompts, and it is also the arm with 7/7 exact streams. Combined with
#142 (gates pass, E top-1 recovered) and the range analysis (no overflow possible by
construction, no subnormals by construction), H2=2 has the most complete quality record of any
arm -- at ~0% latency change. The -5.6% raw H2 keeps the speed record with a slightly worse A
(0.0016) and unmitigated range exposure. Both stay behind the flag; the documented trade for the
user is now: exactness (shipped) vs speed (raw H2) vs evidence-of-fidelity (rescaled H2).

## MAXTOK=64 silently truncates any audio-length prompt (run #149): the engine cannot serve audio inputs as configured

A 100-token synthetic prompt (prompt A ids cycled) with --steps 110 prints steps 0..63 and
stops: `if (pos >= MAXTOK) break` with MAXTOK=64. The remaining 36 prompt tokens are never
consumed, zero generation happens, and nothing warns -- the run just prints steady-state over
62 steps. Per-step cost is flat at ~98 ms out to the cap (prompt steps without lm_head).

Implications for e2e: TTFT for a real audio prompt (hundreds to thousands of tokens) is not
"170 s", it is UNSUPPORTED. The comment above the define says it was sized to the benchmark,
and raising it is priced as follows: KV cache is 2 x 24 layers x MAXTOK x 4 kv-heads x 128 dim
x 2 bytes = 3 MB at 64, 192 MB at 4096. Device free after load is 1.31 GB, so 64 -> 4096 costs
~192 MB and fits with ~1.1 GB to spare. (The in-code comment's "1.0 GB at 4096" is stale; the
arithmetic gives 192 MB.) The sliding window is 256, so full-attention layers are what need
the depth; the 25 MB figure in the comment is for 256.

Next: raise MAXTOK to 4096 (matching the plan's 4k context), re-run the full gate suite (the
binary changes, so gates must re-verify even though the cache size cannot change numerics),
confirm the 100-id prompt runs past 64, and check steady-state at long positions for the
attention O(t) growth the linear model predicts (~3-5% per step at t=1500).

## H2's win holds at 300 tokens: -5.4%, 4/4 (run #151)

`LFM_H2=1` vs shipped on the 300-id prompt (sum steps 0..299, paired, both orders, cold page
cache): shipped median 29879.03 ms vs H2 28243.83 ms, R = 1.0285, H2 faster 4/4. Same ratio
as prompt A (-5.6%), so the attention/KV growth at long positions (+14%/step at t=300) does
not dilute it -- the Q/K/V/O GEMV savings dominate the attention-dot growth. Absolute terms:
1.6 s saved on 300 tokens (~8 s projected at 1500 audio tokens) vs 0.13 s on prompt A. For the
pending user decision: the H2 prize scales WITH context length, so the audio case (where TTFT
is minutes) is where -5.6% matters most. Still default OFF pending the call.

## Full e2e pipeline map (run #153): the engine is the text backbone only

Measured against the HF repo (host, for scoping -- numbers do not transfer to the Nano):

- Audio encodes at ~12 tokens/s: 1 s -> 18 ids, 5 s -> 68 ids (mel 128x499), 15 s -> 193 ids.
  A 30 s query is ~360 prefill tokens, i.e. ~36 s on the Nano engine today at ~100 ms/step.
- Upstream of the engine and unported: mel frontend (~10 ms/5 s on host CPU, measured) and the
  audio tower (audio_encoder 109.5M + audio_projector 5.8M). Direct encoder timing failed on an
  internal mask-format mismatch; end-to-end host generate (1.55 s for 68+4) is polluted by a
  missing-kernel fallback, so the encoder's GPU cost is still unmeasured anywhere.
- "First speech output" runs through the depth decoder (83.9M) + codebook embeddings (49.6M),
  which predict speech codebooks from LM hidden states, plus the liquid codec -- all present in
  the HF tree, all unported to the Nano engine, which covers only the 1.7B text backbone (1729M)
  + tied head (262M).
- Consequence: in-engine TTFT work (prefill batching, H2) applies to ~360-token audio prefills
  where it matters ~15x more than on the 23-token benchmark; but complete audio e2e needs the
  encoder + depth decoder ported first, which is a separate project, not an iteration of this one.

## Audio-length TTFT, measured on the engine (run #154): 36.0 s for 360 tokens

360-id synthetic prompt (prompt A cycled, all ids valid), --steps 380: all steps run, TTFT_360
(sum steps 0..359) = 36027.5 ms = 100.08 ms/step. Per-step cost 97.9 ms (positions 0-9) ->
103.3 ms (positions 350-359): +5.5%, still GEMV-dominated, linear model holds to 360.

- No cliff at the 256 sliding-window boundary and no failure anywhere to 380 steps; MAXTOK=4096
  path is sound.
- The garbage cyclic prompt makes the model emit EOS (125014) repeatedly from the first
  generated token -- clean termination, no runaway, no hang. (Quality gates do not apply; this
  is a timing/scaling probe, not a benchmark.)
- E2e meaning: 30 s of audio (~360 tokens at 12 tok/s) costs ~36 s of prefill on this engine
  before the first real output token, plus answer decode, plus the unported encoder (109.5M)
  and depth decoder (83.9M + codec). Prefill batching's prize at audio scale is tens of
  seconds, not the 0.13 s visible on the 23-token benchmark.

## Fused H2 rescale killed by construction, without building (run #153 analysis)

Considered fusing H2 == 2's rescale scan into the staging/dot loop to recover H2 == 1's -5.6%
while keeping exactness. It cannot work, for three structural reasons:

1. Same global reads: H2 == 2's scan reads x once from global -- exactly like any staging.
   There is no second global pass to eliminate; a fused version reads the same bytes.
2. Irreducible per-element scale cost: applying 2^k needs a multiply per converted element
   (or HMUL2, which needs k in [-24,15] and breaks on the tiny inputs this exists for).
   H2 == 1 has no per-element scale at all.
3. The max itself is a serial fmaxf chain (loop-carried dependency, ~4 cycles each); an
   exponent-OR variant saves the frexp but not the chain, and OR can only underestimate the
   range (safe direction, but then it is just a clumsier H2 == 1).
H2 == 2 as landed IS the fused-optimal form (single global read, cooperative scan). The -0.4%
vs -5.6% gap is the scan's ALU cost, and no restructuring removes it. If the safe version must
be faster, that needs hardware counters on the scan loop, not another variant.

## Audio tower architecture: a 17-layer LFM, i.e. portable with existing kernels (run #157 scoping)

`audio_encoder` = downsample convs (3.3M) + `lfm`: 17-layer transformer, hidden 768, 8 heads
(106.2M), consuming spectrogram features directly (no token embedding registered). Plus a
5.8M projector. Total audio path ~115M -- same kernel family as the text backbone (Q8 GEMVs,
attention, norms), different shapes. A port is mechanical, not research: repack the new
shapes, reuse q8_partial/k_attn_head/k_rmsnorm, gate against HF's tower.

- Downsample convs need a shape check (stride/channel structure unknown from the outside).
- The depth decoder is a different pattern: bidirectional masked model over codebook slots
  PER speech frame (non-causal within a frame), then the liquid codec. It needs its own
  kernel design, not the decode loop.
- Direct HF timing of the encoder failed three ways (internal mask/length calling convention,
  plus a missing-kernel fallback that pollutes host numbers anyway); host timings would not
  transfer to the Nano, so no number is claimed. The structural finding (same kernel family)
  is the result.
- Mel frontend measured: ~10 ms per 5 s audio on host CPU (negligible next to seconds of
  prefill).

## Sliding-window RoPE was half-missing (run #158): the long-range drift, found and fixed

`.auto/lfm3/matrix_long.py` (new gate: per-layer x per-position rms vs HF at 300 teacher-forced
positions, same 3% rule) showed 243/7200 cells outside 3%, all engine-LOW, growing smoothly from
1.000 at pos 0 to ~0.98 at pos 300, originating in the first sliding-attention layer (L4; L0-L3
clean everywhere). Root cause: k_rope was launched with rot=HD/2 for ALL layers, but sliding
layers need full rotation (partial_rotary_factor 1.0 vs 0.5). Their slow half (dims 32..63) went
unrotated -- invisible at pos 0 (RoPE is the identity), under 3% at pos <= 22, then a
position-growing phase error. Fix: rot = is_slide[L] ? HD : HD/2.

- Long gate after fix: 243/7200 -> 1/7200 (one cell at pos 186/L13, ratio 0.97, a near-tie).
- Short gates unchanged-or-better: matrix green, logits0 identical, every logit metric improved
  on every prompt (e.g. A corr 0.99964 -> 0.99973, KL 0.0014 -> 0.0009; all six prompts likewise),
  still 130/131 (F 20/21).
- Streams: F 11/16 -> 16/16 (!), L 16/16 -> 5/16. Still 6/7. The L regression is proven tie luck:
  traj_drift on L's own free-running states gives top-1 = HF's at 47/48 with KL 0.0007 flat
  (F 46/48, KL 0.0009) -- the distributions agree closely and greedy flips on ties.
- TTFT A/B: base 2271.94 vs fixed 2270.46, R=1.0003, null (RoPE is negligible either way).
- Lesson recorded earlier stands: a serialized per-kernel timer would have shown nothing here;
  only a same-state comparison across positions could see a smooth 2% drift, and only a gate
  that EXCEEDS the benchmark length could see it at all. The 23-pos suite was blind by design.

## H2=2 re-scored post-RoPE-fix: 7/7 was tie luck, and it now fails the F floor (run #159)

`LFM_H2=2` through the full gate suite on the fixed tree: streams 6/7 (F 16/16, L 5/16 --
the same L tie-flip the shipped arm shows), and the logit gate FALLS to 128/131 with F at
19/21 (floor 20): A 22/23, C 20/21, E 22/23, F 19/21, K 22/23, L 22/23 -- one top-1 lost per
prompt vs its own pre-fix record, all near-ties (KL 0.0006-0.0012, worst-KL positions agree
with HF on the same token). The rescale is still exact-arithmetic-safe by construction, but
empirically it now fails a ratchet the shipped arm passes, so the #142 "7/7, best-or-tied"
record is retracted as tie luck on both sides.

What this means for the pending H2 call: raw H2 (-5.6%, needs re-timing post-fix but the
RoPE delta is arm-independent and negligible) keeps the speed record with an unmitigated
range exposure; rescaled H2 keeps the range guarantee but now fails the F floor on ties
while costing the whole speedup. Neither dominates; shipped stays default. Do not soften
the floor to accommodate H2=2 -- the floor is the arm's certified worst case, and H2=2's
misses are the same near-ties the floor exists to surface without failing on.

## Thermal vs attention, disambiguated (run #157): the long-range growth is positional

Back-to-back 300-id runs (hot board throughout run 2): run1 first-10 = 97.94, run2 first-10 =
97.83 -- identical. If thermal throttling caused the position-dependent slowdown, run 2 would
start slow. It does not. The +5% growth over 300 positions here (+14% on the day of #150 --
magnitude varies with board state) is attention/KV growth, not heat. Either way the GEMV
phase dominates and no lever follows; this only prevents misattribution.

## H2 raw re-verified post RoPE-fix + MAXTOK (run #161): -5.4%, both orders

One paired A/B each order on the current tree: shipped 2265.88/2269.00 vs H2=1 2141.60/2148.13
(R = 1.057/1.056, H2 faster 2/2). The -5.6% survives the intervening changes, as expected: the
RoPE fix adds identical work to both arms and MAXTOK only enlarges an allocation. Same first
token. This was the last un-re-measured number in the H2 decision table; it is now current.

## LFM3 S2T latency (session 2026-11-06)
- REJECTED w/o run: overlap encoder (GPU, 33 s) with backbone load (48 s). Encoder peak = ~1.1 GB
  (MemAvailable 3189 -> 2074 MB) + backbone 2.15 GB device + ~0.3 GB host > board. OOM hung picard once (run 281).
  Revisit only if encoder activation memory is cut (e.g. free x1/x2 buffers early, bf16 activations).

## S2S breakthrough search (2026-11-11): no exact breakthrough exists on this board
Audio slot 138 ms = backbone headless ~90 ms + depth server ~42 ms (read 1.7, maskgit 40, write 0.6) + ~6 ms IPC.
- Backbone layers stream 1.65 GB Q8 per step: >= 77 ms at the 21.4 GB/s ceiling, i.e. ~96% of the
  80 ms of audio a frame produces. The chain is strictly serial (step t+1's layer-0 rmsnorm needs the
  full embedding of frame t's codes, which need all 3 MaskGIT passes over step t's hidden).
- Lossless weight compression: Q8 code entropy 7.66 bits/code over 1728 M backbone codes
  (7.61-7.68 per tensor) -> <= 4% fewer bytes at ideal coding. Dead.
- Depth GEMMs (nvprof, 36 frames): ~34 ms/frame of ~38 ms kernel time. Microbench (warm, exact):
  k_q8g<12,2,1> reaches 42-45% of FP32 peak on 1024x1024 / 2816x1024 / 1024x2816; register prefetch
  variant is slower everywhere (25-27%); small k/v/gate run at 2-25% (1-2 blocks). Fusing q|k|v|gate
  (N 1568: 457 vs 544 us) and w1|w3 (N 5632: 1329 vs 1398 us) is exact but worth ~1.2 ms/frame (<1%).
  A rewrite to ~65% of peak would save ~11 ms/frame (~8%).
- Exact floor estimate: backbone 77-86 + depth ~12-15 + IPC 0 => ~95-100 ms per audio frame,
  RTF ~1.6-1.7 with the text slots. Realtime (RTF <= 1) needs fewer backbone bytes, i.e. a lossy
  backbone format (excluded by the 2026-11 lossless decision) or a faster bus.
- [LFM3/Nano S2S] Depth-in-backbone single process (UNRESOLVED, needs a clean measurement
  first): in-engine the depth server costs 37-39 ms/frame (speak's own STIME per call, 97
  frames over p3) while the ideal from its kernels is ~28. Codec contention is ruled out
  (streaming on/off: 3.693 vs 3.552 s over 97 frames = 1.4 ms/frame). The standalone-vs-
  in-engine comparison is NOT usable: the depth blob (243 MB) load varies 1.2-4.8 s with the
  page cache (identical 60-frame workload measured 2.89, 3.25 and 6.05 s), which swamps a
  10 ms/frame effect. Method that would settle it: warm both blobs, then run N=200 frames
  in one depth process (load amortized to <1%) and compare against the STIME per-call figure
  from the same session. ANSWERED 2026-11-11: warm two-point slope (100 vs 300 frames, one
  process, load cancels) = 36.4 ms/frame == the in-engine 37-39. There is NO in-engine penalty,
  so the merge refactor is CLOSED as unnecessary. Depth's own composition per maskgit pass
  (nvprof): trunk k_q8g<12,2> 5.5 ms (45%, ~15 GB/s of the 21 GB/s bus), head 1.07, attn_gate
  0.9, silu 0.31, qknorm 0.22; floor ~17 ms/frame vs 36 measured. The trunk lever is the same
  one #246 failed at (k_q12t), so treat with care: only an in-engine phase A/B counts.
  ORIGINAL NOTE: If the gap is real (~-0.35 s/turn, ~-7% RTF), the fix is folding the
  depth trunk into the backbone process (removes the second GPU context and the per-frame
  file hand-off); it is a multi-hour refactor of two monolith TUs with gate risk, so only
  worth it with the gap measured, not assumed.

**The prefill's CVT cost, measured in the engine: a null AND not bit-exact (run #268).** The
wide-K tensors already build code floats from bits (`tlg_leaf`), but the TL/TLf leaf that
serves the smaller prefill tensors (`tlb_leaf`, `tlf_leaf`) still did
`(float)(int)(signed char)b` -- one quarter-rate CVT chain PER CODE, two per fma pair. The
exact fold: the x staging XORs 0x80 into every byte, so the staged byte is exactly 128+b
(0..255), `__uint_as_float(0x4B000000 | (128+b))` is that integer as a float with no CVT, and
`sum (128+b)x = dot - 128*sum x` with both parts exact (the dot is an integer; sum x over a
block is at most nb*127 integers, so 128*sx is exactly representable). Implemented as
LFM_PF_TLBC (tlb_leaf + tlf_leaf, `fma(sb, dot - 128*sx, acc)` vs `fma(sb, dot, acc)`), so
the only change per block is where the +128 bias lands.
  Result: prefill 3897/3899/3900/3898/3899/3915 (TLBC=1) vs 3900/3902/3900/3902/3916/3920
  (TLBC=0) -- **null**, and the generated ids md5 (7bdc9104/1bf47b8a) differ from the frozen
  goldens (049346af/830f6f64), i.e. NOT bit-exact, which would fail the gate regardless.
  Reverted.
WHY IT FAILED -- two independent facts, both worth keeping:
1. **The arithmetic was right but the fold is not bit-exact.** `fma(sb, dot - 128*sx, acc)`
   rounds `dot - 128*sx` to float BEFORE the scale multiply; the cvt form folds sb into the
   unrounded dot. `dot` is an exact integer only while |dot| < 2^24 (it is 32 codes x 127 x
   |x| <= 127, so up to 512k -- usually fine but not always), and `fma(sb, d, acc)` rounds
   once where `fma(sb, d', acc)` rounds twice. The gate sees the difference. A bit-exact
   version must fold the bias into `acc` per block with the SAME rounding sequence as the cvt
   form, which the cvt form does not have (it has no explicit bias term at all).
2. **The win was never there.** Even if bit-exact, the engine-level A/B says removing the
   CVTs from the TL path buys <= 0.1% of the turn. The conversion-pipe model was wrong: the
   prefill's real limiter is not the cvt pipe (the bench said 14% ceiling on that op mix, and
   the engine shows ~0%), so the "quarter-rate cvt is the wall" story is closed in both
   directions.
CONSEQUENCE: the batched prefill is at its floor. 3.89 s = 24 real layers x 118 positions at
~29 ms/position + the last layer's final position; every knob (geometry, op mix, conversion
pipe, ILP shape, dead-output elimination) is measured. What remains on the ASR side is lossy
math (the raw half2 arm, -5.6% TTFT, excluded by the lossless decision) or a different
algorithm (fp16 accumulation), both of which change the goldens.
- [LFM3/Nano S2T] ASR prefill CLOSED at the exact floor (2026-11-12, runs #268-#269): the last
  three untested axes were measured and all three are null or worse. (1) --prefill-run T (the
  chunk size, the one knob #260 flagged and #267's sweep omitted): T=32 vs 128 prefill 3895/3896
  vs 3891 ms, turn identical, and both goldens byte-identical at T=128, which independently
  confirms the chunk-invariance the batched prefill rests on. (2) The int8->float conversion in
  the TL/TLf leaf: SASS shows nvcc already emits I2F.F32.S8 with BYTE SELECTORS (R2, R4.B1/B2)
  = one instruction per code, and it is HOISTED across the CT=4 positions (1 I2F per 4 FFMA).
  Any PRMT/magic-constant rewrite is therefore strictly more instructions, and the bias-folding
  variant is not bit-exact (double rounding through the scale multiply, #268). (3) Occupancy:
  q8_partial_TLg<9> runs 64 regs + 32 KB shared = exactly 2 blocks/SM = 32 warps, while its op
  mix (0.25 I2F + 1 FFMA + 0.25 LDS.128 per MAC) could issue ~66% of the fp32 peak and it
  measures 44%. Forcing 3 blocks/SM (q8_partial_TLg3<8>, __launch_bounds__(512,3): 32 regs,
  16 KB shared, STACK 40 -> 144 = spills) is +1.3% SLOWER on the prefill, 4/4 turns both orders
  (3938-3942 vs 3887-3895 ms). Probe: .auto/attic/row269_tlg3_3block_probe.patch.
  CONSEQUENCE: the prefill (3.89 s = 61% of the turn) is not issue-, occupancy-, geometry-,
  chunk- or conversion-bound; the remaining 44%-of-peak efficiency is the kernel's dependency
  structure, and the only levers past it change the numerics (half2/fp16 accumulation), which
  the lossless decision excludes. Do not re-sweep these knobs or re-try the conversion tricks.
- [LFM3/Nano S2T] The prefill is NOT weight-bandwidth-bound (2026-11-12, run #271, refutes the
  last structural hypothesis). q8_partial_TLg reads its whole 512-row weight slab itself, so the
  tensor is streamed once per ceil(T/CT) position-tiles: at CT=4 the ASR prefill moves
  1.946 GB x 118/4 ~= 57 GB, which at the 19.3 GB/s the decode demonstrably achieves is a 2.97 s
  floor against 3.89 s measured -- i.e. 76% bus vs 44% of the fp32 peak, which had been read as
  "the bus is the nearer limit". Built q8_partial_TLh (CT=8, NS=4, 32 KB dynamic shared, 512
  threads, LFM_PF_TLGS=98; patch .auto/attic/row271_tlh_ct8_probe.patch): the m-loop and the
  per-(block,k) weight loads halve while the FMA count is unchanged. Prefill 3879/3879/3880/3882
  vs 3887/3886/3888/3892 = a wash. CONCLUSION: the ~30 tensor passes are free (the unified-memory
  path absorbs them or the SM stalls elsewhere); halving the weight stream does nothing, so the
  44%-of-peak efficiency is a dependency/latency property of the inner chain, not a resource
  ceiling -- closing bus, issue-op-mix (#267), conversion (#268/#269 SASS), occupancy (#269),
  chunk size (#269), staging waste (#270) and dead work (#266) as explanations.
  TRAP THIS RUN ALMOST SET: the CT=8 kernel was not bit-exact (its zh transcript was one EOS id),
  so the same A/B showed a fake -15% turn (4.59 vs 5.43 s) purely because the transcript was 1
  token instead of 9. When a probe changes output, ONLY the stage timer (PF4-RUN prefill ms) is a
  valid read; turn walls track the generated length. Always print the id count/md5 in a probe A/B.
- [LFM3/Nano S2T] Held-out generalization gate + a real defect in the resident encoder service
  (2026-11-12, run #272). `.auto/combined/generalize.sh` runs 7 clips (the 2 harness clips, four
  5 s/9 s TTS-derived clips, and the 60 s real-gate clip = 11003 mel frames / 1376 audio rows)
  through BOTH process models and requires the same id stream: cold one-shot CLI vs the resident
  services. Result: all six short clips IDENTICAL (resident is 4.6-8.2x faster per turn), which is
  the first evidence that the resident architecture (#259-#266) generalizes beyond the two goldens
  it was certified on.
  DEFECT FOUND BY IT (still open, root cause unknown): a served encoder pass breaks once the
  service has history. Repro, no backbone involved: `encorder.sh 0 "big big big"` -> requests 0
  and 1 answer `nonzero=2818048`, request 2 never answers (no ENC_DONE); LFM_ENC_POOL=0 instead
  hangs inside the pass; the generalize sequence (6 small clips then the 60 s one) came back with
  `dump proj_out n=2818048 nonzero=0` and no OOM / LAUNCH ERROR / EXEC ERROR line anywhere. The
  same 60 s request SUCCEEDS as request 0 of a fresh service (2818048 nonzero, rc=0) and as a
  one-shot child beside the live backbone service, so it is per-request state in the service, not
  memory coexistence and not the pool's growth path alone. Suspect the static non-pooled device
  buffers that a pass frees and reallocates (g_part_buf at lfm3_enc_gpu.cu:1043, g_ss at :1321,
  the tile768 scratch at :1093/:1116, g_rps at :649) being freed while work from an earlier pass
  is still queued -- the pass ends with chk_final() but the service loops without a fresh
  context. Next step for whoever picks this up: `cuda-memcheck --tool memcheck` around a 3-request
  sequence, or a cudaDeviceSynchronize at the top of run_pass behind an env to test the
  queued-work theory in one line.
  SHIPPED NOW (safety net, metric-neutral 0.5076, all four gates green): the service answers a
  dead pass instead of vanishing (setjmp/longjmp + pass_fail() on the OOM / LAUNCH / EXEC /
  PLAUSIBILITY paths, then `ENC_DONE n rc=3` and it retires rather than serve on a suspect
  context), and lfm3_asr falls back to the one-shot encoder for that turn on a missing or rc!=0
  reply. The client's wait is 180 s (LFM_ENC_WAIT), not the 900 s it used to stall on a dead
  service. Verified: the 60 s clip through the resident path now returns ids identical to cold
  (108 s vs cold 64.5 s -- recovery is slower than cold for oversized audio, so a deployment
  that routinely sees >40 s clips should use the cold arm or fix the service).
- [LFM3/Nano S2T] ROOT CAUSE of the served-encoder zeros, and the rule it teaches (2026-11-12,
  run #273). `static double* mv` in the projector (lfm3_enc_gpu.cu, LFM_ENC_LNM) was allocated
  `if (lnm && !mv && cudaMalloc(&mv, (size_t)Tt * 16))` -- sized by the FIRST request. A resident
  service that had served a 9 s clip (Tt=113 -> 1.8 KB) then served the 60 s clip (Tt=1376 ->
  22 KB) and wrote 12x past the buffer: a device-heap overflow that corrupted the neighbouring
  pooled buffers and came out as `dump proj_out nonzero=0` with NO OOM / LAUNCH ERROR / EXEC
  ERROR line anywhere in the log. Fixed to grow-only with a cap (the pattern g_ss at :1321
  already used). Proven: `encstage.sh 1` (one small served request, then the 60 s one) goes from
  nonzero=0/rc=3 to nonzero=2818048 max=1.0471/rc=0, and the value matches a fresh-service run
  to the digit. The one-shot process could never hit it: its first request is its only request.
  RULE, worth applying before any new service is trusted: **a resident service turns every
  "allocate once, size from the first request" static into a latent buffer overflow.** Audit
  `!ptr && cudaMalloc` and any static sized by a request-dependent length; the same class in the
  backbone was already caught once (#264, the layer-major xall buffer sized by the first clip).
  Still OPEN (next iteration): with the encoder fixed, the 60 s clip now fails one stage later --
  the resident BACKBONE produces nothing for 900 s on a 1396-position clip (generalize.sh:
  `the resident backbone produced nothing for 900 s`, MEMLOW 346 MB), while the same clip works
  end-to-end cold (64.8 s, 96 ids) and the encoder answers rc=0 beside it. Suspect the same
  first-request-sized pattern in lfm3_nanode's service path, or the arena growing past what is
  left next to 2.15 GB + the encoder's ~800 MB for a long clip. `bash encmix.sh both` isolates
  it against `enc_only`, and the client's LFM_ENC_WAIT/900 s waits make it slow to probe: lower
  them while iterating. Also from this run: my first hand-rolled fifo poll loop reported a fake
  "third request hangs" by racing its own writer -- trust only the real client (lfm3_asr) or a
  per-request proj_out existence check, and say so in the diagnostic's header.
- [LFM3/Nano S2T] The 60 s resident failure after #273 was two things, neither a backbone bug
  (2026-11-12, run #274). (1) generalize.sh booted the backbone service with --prefill-run 128,
  a chunk the harness never ships (measure.sh: PFRUN=32). At 128 a 1396-id prompt prefills at
  67.3 ms/pos (94.0 s) against ~33 ms/pos at 32; ids are identical at both. Alone the clip then
  completes (154 s), so "nothing for 900 s" in the 7-clip order was this slowness plus history,
  not a wedge; not reproduced once the gate ran at 32. Why 128 is 2x slower only on long prompts
  is NOT explained (on 133 ids it is +1.3%, #269). (2) A client race: lfm3_asr opened the service
  fifos O_WRONLY|O_NONBLOCK and died on ENXIO when it arrived between the service's EOF and its
  reopen. Back-to-back turns hit it (clip_en right after clip_zh); the harness never did because
  a TTS request sits between ASR turns. Fixed with a bounded ENXIO retry (fifo_open_w,
  LFM_FIFO_WAIT default 10 s). Result: GENERALIZE OK, 7/7 clips identical cold vs resident.
  STILL OPEN: the resident 60 s turn is SLOWER than cold (84.6-120 s vs 64.9 s over three runs).
  The served encoder reply is 31.7 s vs 7.4 s cold while its GPU time is 0.66 s, with
  MemAvailable down to ~340-390 MB beside the 2.15 GB backbone: suspect allocation/page reclaim
  under memory pressure, unmeasured. Until fixed, audio over ~40 s is better served cold.
  New knobs: LFM_SRV_WAIT (backbone reply wait, default 900 s), generalize.sh PFRUN / CLIPS.
- [LFM3/Nano S2T] Why the resident 60 s turn is slower than cold, and the pool trim (2026-11-12,
  run #275). enc60.sh, same served 60 s encoder request after one small one: encoder alone 5.93 s
  (MemAvailable 3175 -> 1921 MB), backbone resident 19.14 s (1069 -> 382 MB). The pass needs
  ~1.25 GB and only ~1.07 GB is available beside the backbone, so it runs under reclaim: memory
  pressure, not a code path (GPU time 0.55 vs 0.71 s). Only cutting the encoder's activation
  memory for long audio fixes the turn itself (not done). Shipped: LFM_ENC_POOL_MAX_MB (default
  512, 0 = off): a pass that leaves the grow-only pool above the cap frees it (772 MB after a
  60 s clip), so one long clip no longer pins the board at ~380 MB for the session. Encoder-only
  sequence small/60 s/small/small, MemAvailable after each: trim 2805/2685/2871/2975 MB, no trim
  2773/1908/2259/2260 MB; first small request after a trim 0.62 s vs 0.55 s, then 0.54 s.
  Gate with the 60 s clip FIRST then 4 short clips: 5/5 identical cold vs resident, post-turn
  MemAvailable low 1078 MB (was ~350; sampled after turns, not inside the 60 s turn). Resident
  60 s turn in that order: 71.6 s vs cold 64.8 s. 9 s harness turns stay under the cap: 0.5086.
- [LFM3/Nano S2T] Encoder downsampler buffers aliased into x1 (2026-11-12, run #276, keep).
  LFM_ENC_MEM=1 (new) lists a 60 s pass: x1 343.9 MB, x2 / x2b / perm 86.0 each, x3 / x3b / flat
  21.5 each, all 17 layers + projector ~100 MB; pool 772 MB. x1 is read only by conv_mid, so
  perm, x2b, x3, x3b, flat (11/16 of x1) now live inside it (LFM_ENC_ALIAS, default 1), refilled
  with zero / NaN-poison once conv_mid has run: pool 536 MB. proj_out byte-identical alias 0 / 1 /
  1+POISON on clip_zh, clip_en, cv_en5 and the 60 s clip. Served 60 s encoder reply beside the
  resident backbone, order 0/1/1/0: 20.4 / 5.6 / 5.5 / 31.8 s (encoder alone: 5.9 s) -- the pass
  no longer runs under reclaim. Resident 60 s turn 62.9-63.6 s (4 runs) vs cold 64.5-64.9;
  in-turn MemAvailable low 545 MB. 9 s harness turns: asr_wall 5.877 s, unchanged.
  OPEN, NOT EXPLAINED: one run of the 7-clip gate stalled on the 60 s clip in the backbone
  (CLIP START + AUDIO-EMB armed, then no output for 900 s; encoder had replied in 8.8 s). That
  run overlapped a harness run I started in the same tool call, so it is contaminated; 3 lean
  repros (stall.sh, captures process state at 150 s) and a clean full gate all passed. The
  pre-#274 "nothing for 900 s" had the same signature, so treat an intermittent backbone stall
  on long clips as possible until stall.sh has caught one. Also: tts_wall is bimodal across
  harness runs (0.54 / 0.65 s; #273 0.547, #274 0.649, #275 0.541, #276 0.648), untouched code.
- [LFM3+PrimeTTS/Nano] The TTS number was set by the GPU clock at request time, and the harness
  was sampling it badly (2026-11-12, run #277). ttsq2.sh (one fresh resident TTS, blocking wait,
  cur_freq read before each request), wall@MHz: back to back 1.67@537 .71@384 .54@691 .48@844
  .48@921 .48@921; 2 s gaps .63@460 then 1.14 / 1.22 / 1.05 / 1.34 / 1.01 all @76; 6 s gaps .69@460
  then 1.24-1.69 @76. So: 0.48 s at full clock, 1.0-1.7 s once nvhost_podgov has parked the GPU
  (within 2 s of idle). The old harness protocol booted the TTS service after the ASR turns,
  polled readiness at 0.5 s and the reply at 0.1 s: it caught the clock partway down by an amount
  set by its own poll phase -> 0.54 / 0.65 s bimodal (#273-#276), and a 5 ms poll loop is worse
  (grep spawns steal a core: 0.78-0.95 s). PROTOCOL CHANGE, not an engine change: PT_TURN=1
  (default) keeps the TTS service resident beside the ASR services and issues one request right
  after each ASR turn with a blocking wait; PT_TURN=0 is the old protocol. Same build, order
  1/0/1 + the logged run: tts 0.492 / 0.537 / 0.490 / 0.483 s, asr 5.897 / 5.885 / 5.896 / 5.868 s,
  combined 0.5065 / 0.5090 / 0.5061 / 0.5034; served waveform byte-identical on every turn;
  MemAvailable low 759 vs 1049 MB (third resident service, ~290 MB). Numbers before #277 used
  the old protocol: compare across it only with PT_TURN=0.
  DEPLOYMENT RISK THIS EXPOSES (not in the metric): the reply ids are a fixture here, so TTS
  follows ASR with no gap. If whatever produces the reply takes > ~1-2 s without using the GPU,
  the TTS runs on a parked clock and costs 2-3x (1.0-1.7 s). A fix belongs in the product path
  (e.g. the TTS service re-ramping the clock when the reply text starts arriving); an in-process
  keep-warm during ASR is NOT the answer (#43: -35%, one SM). Needs root to pin the governor.
- [LFM3/Nano S2T] The long-clip backbone stall: not reproducible on the current build, and two
  earlier readings corrected (2026-11-12, run #278, diagnostic). stall.sh (resident arm only: 7
  short turns then the 60 s clip; now with BALLOON_MB and PFRUN): 8/8 passes at 62.87-63.90 s,
  prefill 46.28-46.63 s, ids md5 eca77bbe (= cold), in-turn MemAvailable low 534-540 MB. With
  touched anonymous memory held during the turn (chunk 32): 150 MB -> low 453, prefill 46.33 s;
  300 -> 415, 46.34; 450 -> 369, 46.32 s, encoder reply 15.4 s. Chunk 128: no balloon 46.36 s,
  450 MB balloon 46.37 s (low 371), encoder reply 10.3 s.
  CORRECTIONS: (1) #274's "chunk 128 prefills at 67 ms/pos on long prompts" was not a chunk
  effect -- on this build 128 and 32 are equal (33.2 ms/pos). (2) Memory pressure down to ~370 MB
  slows the ENCODER reply (allocation under reclaim: 5.6 -> 10-15 s) but not the backbone
  prefill, so "low memory stalls the prefill" is not supported at that level either.
  What the evidence does say: every slow or stalled long turn (72-154 s, and the 900 s stalls)
  happened while the encoder pass was 772 MB (before #276's aliasing) or, once, with a second
  2.15 GB backbone loading beside the resident one (my concurrent harness run). 15 long turns
  since #276: none slow. Mechanism still not observed directly; stall.sh's capture is armed.
  The balloon holds anonymous pages, not nvmap memory, so it is not the same pressure as a
  larger device allocation -- that is the untested difference. Not pushed below ~370 MB: the
  board hard-hangs at zero and cannot be rebooted remotely.
  Harness spread under the PT_TURN=1 protocol, 4 samples of unchanged engines: combined_rtf
  0.5034 / 0.5061 / 0.5065 / 0.5078 (asr 5.868-5.904 s, tts 0.483-0.504 s) -> ~+-0.4%.
- [PrimeTTS/Nano, single stream] The in-turn TTS is at the runtime's warm floor; three N=1 probes
  null (2026-11-12, run #279, diagnostic, no code change). Resident service, back-to-back T81
  requests at full GPU clock: 0.464-0.482 s steady (first request of a fresh idle service 0.58-1.9
  s: cold clock + first touch). The harness turn reads 0.46-0.52 s, i.e. already there.
  (1) PT_V21_DEVADD=1 (device residency; closed in prime-tts at N=4 as +261 ms on a saturated
  GPU) retested at N=1 where the GPU has slack, order 0/1/1/0: 0.465-0.477 vs 0.502-0.524 s =
  ~+6% SLOWER, waveform byte-identical (md5 e6092967 = one-shot reference). Closed at N=1 too.
  (2) CPU frequency: all four A57 at 1479 MHz (max) mid-request under schedutil. Not a factor.
  (3) taskset -c 3: 0.468-0.482 vs 0.464-0.481 unpinned. Null.
  The service thread is 100% busy for the whole request (user ~350 + sys ~110 ms of 0.47 s wall,
  2 threads) because it yield-polls the GPU, so CPU accounting cannot split host work from waiting.
  One-shot PT_V21_EKIND=1 profile (T81, 590 ms total): CONV 408 ms / 124, CONVT 35, ADD 26 / 107,
  MATMUL 19, TANH 16, EINSUM 10, LEAKYRELU 9, SOFTMAX 8, SIGMOID 7, MUL 6, rest < 5 each; the
  profiler prints nothing in --serve-fifo mode. Untested and not small: moving the ~100 ms of
  host elementwise ops to a helper thread (3 cores idle in a single-stream turn); the prime-tts
  history says host work overlaps GPU work already, so the gain is unknown, not ~100 ms.
- [LFM3/Nano S2T] CORRECTION + encoder memory preflight (2026-11-13, run #280, keep).
  CORRECTION: gate_60s_zh_16k_mono.wav is 110.04 s of audio (16 kHz mono; 11003 mel frames at
  100/s, 1376 audio rows at 12.5/s), NOT 60 s. Every "60 s clip" in the #272-#279 notes above
  means this 110 s clip; I took the length from the filename. Measurements are unaffected, the
  scaling is: encoder device memory ~5 MB per audio second (542 MB at 110 s), not ~9.
  THREE services resident (what the harness runs since #277) + the 110 s clip, stall.sh TTS=1,
  5 runs: ids == cold every time, turn 64-70 s, MemAvailable low 252-310 MB, encoder reply
  6.1-12.2 s (under reclaim again; 5.6 s with two services). TTS straight after the long turn:
  0.57-0.61 s, the next 0.48 s. (A first reading of 0.99-1.26 s was the script's own 1 s poll
  letting the GPU clock park, not memory.)
  So the long-clip margin with three services is ~250 MB and a longer clip would not fit; this
  board hard-hangs when the pool runs out. Shipped: a preflight in the encoder pass
  (LFM_ENC_MEMFLOOR_MB, default 128, 0 = off): estimate the pass's device bytes, subtract what
  the pool already holds, and if MemAvailable minus that is below the floor, refuse BEFORE
  allocating (rc = 4; the service keeps serving -- it only retires on other failures).
  guard.sh: default floor -> 110 s clip allowed (pass ~542 MB, needs ~498 more, 762 available),
  65.1 s, ids == cold. Floor forced to 600 -> service answers ENC_DONE rc=4, the client's one-shot
  fallback refuses too, lfm3_asr exits 2 in 0.76 s with the MEMGUARD line, the next short turn is
  served and correct. Limits: the estimate is from sizes measured on this build (reported 542 vs
  536 MB allocated), MemAvailable is the kernel's own estimate, and the real exhaustion edge was
  not probed (cannot be, remotely). Rough capacity with three services: ~135 s of audio.
- [LFM3+PrimeTTS/Nano] Held-out gate now runs the measured configuration (2026-11-13, run #281,
  keep, gate only). generalize.sh TTS=1 (default): three resident services, one TTS request right
  after every resident ASR turn, rotating over all five golden reply cases (T81 T20 T120 T40
  sid1_T29; the harness only ever served T81), each byte-compared with a one-shot reference made
  before the services boot. Result: 7/7 ASR id streams identical cold vs resident, 7/7 served
  waveforms byte-identical to one-shot (first evidence the resident TTS is exact on a case other
  than T81), 0 guard trips, 110 s clip 68.1 s resident vs 65.0 s cold (three services: the
  encoder pass is under reclaim, #280). Served TTS walls: T20 0.17, T29 0.23, T40 0.27, T81
  0.47-0.50, T120 0.71 s; T20 right after the 110 s turn 0.50 s (3x its usual: one request after
  a memory-pressure turn is slow, the #280 observation again). MEMLOW 766 MB is sampled after
  turns, not inside them (in-turn low for the 110 s clip is ~250-310 MB, #280).
- [LFM3/Nano S2T] Backbone context cap enforced; spec parser hardened (2026-11-13, run #282, keep,
  robustness only). The K/V caches hold MAXTOK = 4096 positions and the batched prefill writes
  one row per prompt position with no bound: a prompt over 4096 ids would have written past them
  (device-heap overflow, the #273 class). lfm3_asr already refused n_ids + max_gen > 4096, but
  only before stage 5, i.e. after the encoder pass; lfm3_nanode itself had no check. Now:
  (1) lfm3_asr refuses as soon as the mel frame count is known (330 s clip: rc 2 in 1.2 s, no
  encoder pass); (2) lfm3_nanode refuses a prompt > MAXTOK (service: CONTEXT REFUSED + DONE
  steps=0 and keeps serving; one-shot: exit 2); (3) a non-numeric id line in a clip spec no
  longer loops forever pushing ids until bad_alloc kills the service (strtol that consumed
  nothing; found when my own test spec had its lines swapped): SPEC REFUSED, service stays up.
  ctxcap.sh proves all three and that the next turn returns the golden ids. Max audio at this
  prompt with max_gen 96: 318 s (two services; with three the encoder memory guard trips first,
  ~135 s). Not covered: prompt + generation crossing 4096 inside the engine still ends the step
  loop silently (the client's check prevents it for lfm3_asr callers).
- [LFM3/Nano S2T] Context-full is announced; full-context soak; held-out gate on the hardened
  build (2026-11-13, run #283, keep, robustness only). lfm3_nanode prints "CONTEXT FULL: stopped
  at position 4096 of N requested steps" where the step loop used to end silently. ctxfull.sh:
  a 4090-id prompt + 20 requested ids through the resident service (the longest prompt ever run
  through it; 152 s) stops with that line after 6 ids, service alive, next turn golden,
  MemAvailable 1229 MB after. generalize.sh on this build (three services): 7/7 ASR identical,
  7/7 TTS bytes identical, 0 guard trips; 110 s clip 64.1 s resident vs 64.7 s cold this run
  (68.1 vs 65.0 in #281: the long resident turn varies 63-70 s with three services).
- [LFM3/Nano S2T] Long-conversation soak: 24 turns flat, no leak, no drift (2026-11-13, run #284,
  keep, gate/tooling only; no engine change). .auto/combined/soak.sh runs N alternating zh/en
  turns through the three resident services (the configuration the harness measures) and records
  per turn: ASR wall, TTS wall, transcript-vs-golden, SNDR, MemAvailable, and the three services'
  VmRSS. It exists because both resident defects found so far (#273's first-request-sized static,
  #275's pool pinning the board) were turn-count or history dependent and a 2-turn harness cannot
  see that class. Result at N=24: ids == golden on all 48 transcript checks (24 turns x 2 clips
  alternating), SNDR 70.09 on all 24 TTS requests, zh asr 5.416-5.439 s (0.4% spread), en
  6.308-6.372 s (1.0%), tts 0.463-0.482 s, MemAvailable 788-790 MB flat, RSS bb 31 / enc 35-36 /
  tts 243-244 MB flat. The zh/en gap is 0.92 s = the 9 extra decode steps of the en transcript at
  ~101 ms each, which independently confirms the per-turn cost model. Backbone RSS is 31 MB
  because its 2.15 GB arena is nvmap-accounted, not anonymous: read device residency from
  MemAvailable, not from RSS. Harness on the current tree 0.5053, all gates + checks.sh green.
- [LFM3/Nano S2T] Mixed-length soak: the trim/regrow path is exact, and a long turn taxes its
  neighbours (2026-11-13, run #284b, keep, gate extension + findings). soak.sh CLIPS="..." now
  takes any clip list (bash array; the first version's eval indirection silently produced the
  literal "${1}" as a filename and every turn exited 2 without being flagged -- a soak that
  cannot fail is not a gate, so a nonzero client exit now prints CLIENT_FAIL and a missing clip
  aborts). 2 rounds x the 7-clip held-out set (6 short + the 110 s gate clip) through one service
  session: every turn rc=0, SNDR 70.09, ids STABLE against the first pass of the same clip and
  GOLDEN where a frozen golden exists (including the 110 s clip on its second pass -- the pool
  trim/regrow path of #275 produces identical ids). What a long turn costs its neighbours: the
  TTS service's RSS drops 243 -> 49 MB across the 110 s turn and stays there (the encoder's
  ~500 MB pass evicts the TTS's resident pages; its next request is still 0.46-0.48 s, so the
  fault-back is cheap); the TTS request issued right after a long turn is 0.564-0.576 s vs the
  usual 0.463-0.485 (+0.09 s); the first short ASR turn after it 5.676 s vs its 5.42-5.44 s
  steady (+0.25 s, one turn only); encoder RSS 35 -> 46 MB and backbone RSS 31 -> 25 MB.
  Limitation: the mixed arm seeds each clip's reference from its own first pass, so a defect that
  appeared on the FIRST pass of a clip in a session would be missed here -- generalize.sh
  (cold vs resident, frozen goldens) covers that; the two together are the gate.
- [LFM3+PrimeTTS/Nano] Arrival clock ramp: the TTS no longer runs on a parked GPU clock
  (2026-11-13, run #285, keep; prime-tts 8e2f025; metric-neutral by construction). #277 found the
  risk and left it as "needs root": nvhost_podgov parks the GPU at 76.8 MHz within ~2 s of idle,
  and a T81 request then costs 1.0-1.7 s against 0.48 s at the 921.6 MHz max. Root is not needed:
  the resident driver can read cur_freq/max_freq (world-readable devfreq sysfs) and, when a
  request arrives parked, hold the SM busy on the startup ramp's own ALU kernel
  (pt_v21_cuda_keepwarm, private stream + private sink) until the clock reaches max or
  PT_V21_RAMP_MS (default 500) expires. When the clock is already at max it returns without
  launching anything -- so the harness, where TTS follows ASR with no gap, never enters the path
  (proof: zero "clock parked" lines in /tmp/cmb.pt.log across harness runs, and tts_wall 0.483 s
  unchanged). Paired order-balanced A/B with 6 s gaps, 3 requests/boot, wall including the ramp
  (.auto/combined/tts_ramp_ab.sh): ramp on 0.728-0.813 s vs ramp off 1.125-2.259 s (11-26 bursts,
  192-333 ms each ramp) -> worst case ~2.8x better, median ~1.6x. Back-to-back at max:
  0.4759/0.4772 (on) vs 0.4926/0.4927 (off), no ramp line. First request after boot with a cold
  clock: +43 ms (0.631 vs 0.588, one sample each, inside that sample's own spread; the harness
  pays it in its untimed warm-up turn). Bit-exact by construction and gated: served waveform
  byte-identical to one-shot, SNDR 70.09, harness 0.5063 (band 0.5034-0.5078). LESSON: the
  parked-clock penalty is invisible to any benchmark that keeps the GPU busy or measures one
  request after another -- a per-request latency feature must be A/B'd with idle gaps that
  reproduce the deployment's dead time, and the ramp's own cost must be inside the timed window.
- [LFM3/Nano S2T] The ASR does NOT need the arrival clock ramp (2026-11-13, run #286, measured out;
  tool .auto/combined/asr_gap_ab.sh). Extending #285's fix to the ASR turn looked like the bigger
  prize (5.9 s of GPU work arriving after the user's silence, clock parked at 76.8 MHz). Measured,
  3 rounds x {10 s gap then turn} vs {turn immediately} in one service session: gap
  5.510/5.556/5.574 s vs immediate 5.426/5.432/5.434 s = **+0.09 s (1.7%)**, ids GOLDEN in every
  turn. WHY it differs from the TTS (where the same regime cost 2-3x): the ASR turn begins with
  ~1.5 s of CPU-only work (WAV decode, mel, prompt ids, tokenizer join) before the encoder's first
  launch, so the governor is ramped by the time the GPU is needed -- the stage's own front-end
  pays the ramp. The TTS has no such phase (its first launch is immediate), which is exactly why
  the ramp belongs there and not here. RULE: a parked-clock penalty is only worth an arrival ramp
  in a stage whose first GPU launch is immediate; measure the stage's CPU lead time first. Not
  built; PT_V21_RAMP_MS stays the only ramp. Harness 0.5054, all gates + checks.sh green.
- [PrimeTTS/Nano, resident] The TTS service no longer dies on a bad request (2026-11-13, run #287,
  keep; prime-tts f7001bc). It was the last resident stage with no recovery: the --serve-fifo loop
  did `if (sndr < 0.0) break;`, so ONE unloadable case dir ended the speech stage for the rest of
  the session (encoder: answers rc=3 + client fallback, #272; backbone: refuses and keeps serving,
  #282). Now rc distinguishes the failure classes -- 0 rendered and matches the case golden, 1
  rendered but SNDR < 60 dB (engine/context problem), 2 the case could not be loaded or the run
  failed (a bad request) -- and a bad request is answered so the session continues. Safe to
  continue because each request self-checks SNDR >= 60 dB against its own golden, so a corrupt
  context surfaces as rc=1 rather than as silence; PT_V21_SERVE_MAXFAIL (default 3, <=0 = never)
  ends a service that fails that many in a row. ttsrecovery.sh: badpath rc=2 alive, T81 rc=0
  byte-identical to a one-shot reference made before boot, empty case dir rc=2 alive, T81 again OK,
  3 bad in a row exits with its reason. Also published run_case's feeds as they load (ownership
  hygiene; the claimed "partial-load leak" is NOT real -- 20 partial failures moved VmRSS < 1 MB in
  both the fixed and pre-fix binaries, the buffers before the failing one are a few KB). Two
  harness traps hit while measuring it: the leak arm needed the give-up cap disabled (the new binary
  exited at request 3 of 20), and pgrep on the fifo path matched the `timeout sh -c` request writers,
  so the RSS was read from an exited pid (delta printed as -216). Harness 0.5058, all four gates +
  checks.sh green; the metric is untouched by design (a good request never enters this path).
- [LFM3/Nano S2T] Concurrent ASR clients were broken two ways; both fixed (2026-11-13, run #288,
  keep; .auto/combined/concasr.sh). Nobody had ever run two lfm3_asr clients against the resident
  services -- the harness runs one turn at a time -- and the N>1 case is on the feature backlog.
  Both service clients assume serialised traffic in code: the encoder client derives its request
  index by COUNTING the ENC_DONE lines already in the log ("turns are serialised, so that count is
  this turn's index"), and the backbone client waits for the first "CLIP n DONE steps=" after the
  log offset it noted before writing its spec. Measured with 2 concurrent clients: the loser
  returned on the winner's reply and read a proj_out.bin that did not exist
  ("lfm3_enc_gpu produced -1 bytes at /tmp/lfm3_asr.<pid>.d/proj_out.bin (expected >= 925696)",
  rc=2), 3 of 4 trials failing. FIX 1 (client): turn_lock()/turn_unlock(), an advisory flock on
  /tmp/lfm3_asr.turn.lock held from the first service touch (the encoder request WRITE, which
  happens in stage 1 because the pass overlaps the CPU work -- a lock taken where the reply is
  awaited is too late, the first attempt failed 3/4 exactly that way) until the backbone clip is
  done; the kernel drops it on exit so no error path unwinds it; LFM_ASR_SERIALIZE=0 control,
  LFM_ASR_LOCK moves the file. FIX 2 (service, found after fix 1 left one 400 s hang): the clip
  fifo's read end was closed at EOF and reopened with freopen(), which deadlocks when a client
  writes and closes while the service is busy -- the line sits in the pipe while the service
  blocks in open() waiting for a writer that has already left. Now opened ONCE O_RDWR|O_NONBLOCK
  for the life of the service (a writer always exists, so clients never get ENXIO and the service
  never sees EOF), with clearerr+20 ms on an empty read; a piped-in batch keeps stdin/EOF
  semantics. Result: 4 trials x 2 clients and 3 trials x 4 clients, every client rc=0 with its own
  frozen golden, and CLIPS_DONE/ENC_DONE exactly one per request (9 and 13) -- no lost or doubled
  specs. Queueing is the right model here anyway: one SM, and the backbone serialises clips itself.
  Harness 0.5060 all gates green; generalize.sh 7/7 identical + 7/7 TTS bytes + 0 guard trips.
  Harness trap while measuring: concasr.sh's bare `wait` also waited for the two long-lived
  services, so the first version looked like a 25-minute hang after the clients had exited.
- [LFM3/Nano S2T] Multi-user latency with the resident services, measured (2026-11-13, run #289,
  tool .auto/combined/concasr.sh wall column). Two clients launched together, 3 trials, ids ==
  frozen golden for both every time: walls 5.46 + 11.68 s, 6.38 + 11.73 s, 5.45 + 11.68 s against
  solo turns of 5.42 (zh) / 6.34 (en) s. So the second user waits exactly one full turn: aggregate
  throughput is the single-stream rate split between users, and per-user latency is additive. That
  matches the batched-decode finding (row 56: N>1 aggregate == N=1) and is now measured for the
  session/queue path as well, with the correctness gate from #288. Consequence for a deployment:
  capacity planning is N_users x turn_time <= 1 s of audio budget, and no further concurrency work
  helps throughput on this board (one SM); the only lever that could is the lossy one.
  Also checked: the encoder service's request reader needs no equivalent of #288's fifo fix -- it
  opens O_RDONLY|O_NONBLOCK and never blocks in open(), which is what the backbone's freopen() did.
- [LFM3/Nano S2T] CPU share of the batched ASR prefill: the CPU half is priced, the hand-off is the
  open question (2026-11-13, run #293, diagnostic; .auto/combined/cpu_leaf_bench.c). Context: after
  #290-#292 every other block of the turn is at a measured floor -- the decode lm_head streams its
  ~295 MB at ~20 GB/s (14.7 ms/step by the prompt-vs-generated step difference, 85.8 vs 100.5 ms;
  the old "14 GB/s" note is stale), layers at ~19 GB/s, encoder GEMMs 36-47 GMAC/s (~35% of the 118
  GMAC/s fp32 peak, family swept in #267), client fully accounted (stage sum == process total within
  1 ms). The 3.89 s prefill runs ~195 GMAC at ~50 GMAC/s while all four A57s are idle.
  Bench: the GPU leaf's per-output chain (per 32-code block: dot = sequential fma over the codes from
  0, then acc = fma(scale, dot, acc)) on NEON with positions as lanes (vfmaq_f32 is a fused IEEE fma),
  8 positions per code broadcast, x staged transposed. BITWISE equal to a scalar fmaf chain on every
  output (57344 checked). 1 row/pass: 1.68 GMAC/s on 1 thread, 4.4 on 4 (x re-streamed from L2:
  64 KB per row at K=2048). 2 rows/pass sharing each x load: 2.06-2.24 on 1 thread, 7.2-7.8 on 4.
  Bound: (50 + 7.5) / 50 -> prefill -13% (~0.5 s/turn) with zero hand-off cost and no contention;
  ~-10% with 3 workers (one core left for the launch thread). Not reproduced in the bench: the GPU's
  pairwise st[] tree across leaves (O(K/32) adds per output, negligible cost, must be matched exactly).
  Why Bonsai's two attempts do not settle it: #75 was decode (bus-bound; here the CPU share reuses its
  rows across a 32-position chunk, ~0.3 GB/s of weight reads); #96 was 8-token chunks with ~0.4 ms
  fixed cost per split vs 2.6 ms GEMVs (here 32 positions, ~4x longer GEMVs). The open cost: ~576
  split points per turn (24 layers x ~6 GEMVs x 4 chunks; fewer if the q/k/v and w1/w3 groups that
  share an input split once). At 0.4 ms each that is ~0.23 s, half the prize. NEXT: a CUDA probe of
  the per-split hand-off on this board (D2H of the chunk's x, CPU rows on 3 workers concurrently with
  the GPU's rows, H2D of the CPU's y, join) inside the real prefill chunk; build only if it is
  <~0.15 ms per split. Mapped memory is CPU-uncached on Tegra (Bonsai #53) and managed memory blocks
  concurrent access, so the hand-off has to be real copies.
- [LFM3/Nano S2T] CPU share of the prefill: the hand-off and the contention are measured, both
  favourable -> BUILD (2026-11-13, run #294, diagnostic; handoff_probe.cu, contend.sh).
  (1) Hand-off latency L (handoff_probe.cu, real chunk sizes: x 32x2048 fp32 D2H pinned, host sees it,
  y 32x640 fp32 H2D, consumer kernel; median of 200, two runs): GPU idle 0.277-0.323 ms; with a spin
  kernel occupying the SM on another stream 1.056-1.062 ms. A split GEMV of unsplit GPU time G whose
  CPU share is balanced to finish with the GPU's rows saves (G - L)/(1 + r), r = GPU/CPU rate, so L
  costs only L/(1+r) per split: it does not halve the prize as I estimated in #293.
  (2) Contention (contend.sh, real resident zh turns, cpu_leaf_bench loading 3 cores for the whole
  turn): GPU prefill 3895/3894/3898 ms quiet vs 3902/3905/3908 loaded (+0.25%), ids identical; the
  CPU kernel 6.06 GMAC/s alone vs 5.15-5.24 beside the prefill (-14%). Opposite of Bonsai #75: this
  prefill is compute-bound, and the CPU share reuses its rows across the 32 positions.
  Revised bound, 3 workers at 5.2 GMAC/s vs GPU ~50 (r ~ 9.6), L = 1.06 ms on all 576 splits:
  (3890 - 576 x 1.06)/10.6 ~ 0.31 s/turn (~5% of asr_wall); more if short GEMVs are not split.
  BUILD PLAN (exactness first): (a) reproduce the GPU leaf's EXACT order for the split rows -- the
  per-32-block dot + fma(scale) chain is matched bitwise already (#293); the cross-leaf pairwise st[]
  tree of q8_partial_TLg (k-order permutation + __fadd_rn tree) must be transcribed and checked on
  real rows against the GPU's own output (bitwise, every row and position of a chunk) before any
  timing; (b) weights for the CPU rows repacked row-major int8 + f32 scales at load (host RAM, ~share
  x 1.95 GB); (c) split per GEMV: GPU runs rows [0, N-n_cpu), D2H of x, 3 spinning workers run the
  tail rows, H2D into the same output buffer, the next kernel waits on that copy; share per shape
  from G and the measured rates; only GEMVs with G >> L; (d) gate: ids == goldens + LFM_DUMPX
  last-position residual md5 (70b78a7b zh / 9591fe6c en) + generalize.sh.
- [LFM3/Nano S2T] CPU prefill share, step (a) DONE: the host transcription of the prefill GEMV is
  bit-exact on real data (2026-11-13, run #295; lfm3_nanode LFM_CPUCHK=R, default 0). Order, as read
  from q8_partial_TLf (tlf_leaf2b / tlf_leaf) and q8_partial_TLg (tlg_leaf): per (row, position),
  leaves k = 0..31 with i = bit-reverse5(k); leaf i walks blocks b = i, i+32, ... < nb with
  acc = fma(scale_b, dot_b, acc) from 0, dot_b = sequential fma over the block's 32 codes in byte
  order from 0; leaves combine as a binary counter (bit l of k set: pv = st[l] + pv, else st[l] = pv
  and stop); y = f2bf(0 + pv) with the device's RN-even. Weights read back from the tiled layout.
  One real zh turn through the resident backbone (last 640 rows of EVERY tiled single-chunk prefill
  GEMV, 24 layers, 5 chunks): 10,352,512 outputs, 0 bitwise differences, over w1/w3 (N 7680 K 2048),
  w2 (N 2048 K 7680, TLg), conv.in_proj (6144), q/o (2048), k/v (512); ids == golden.
  conv.out_proj is not on the tiled path (not checked, would stay unsplit). NEXT: (b) host repack of
  the tail rows at load (row-major int8 + scales), NEON kernel in this exact order (2 rows/pass from
  #293, plus the leaf permutation and tree -- the tree is per (row, position) scalar adds, cheap),
  checked against this same CPUCHK reference; (c) wire the split.
- [LFM3/Nano S2T] CPU prefill share, step (b) first cut: NEON rows bit-exact, but 3.5x too slow
  (2026-11-13, run #296; lfm3_cpurows.cpp, LFM_CPUCHK_NEON=TH). Positions as lanes (float32x4,
  x transposed [k][t]), vfmaq_f32 + vaddq_f32 in the TLf/TLg order: one real zh turn, last 1024
  rows of every tiled prefill GEMV, 15,665,152 outputs, 0 bitwise differences vs the GPU, ids golden,
  at 3 and 4 threads. Rate on the real shapes: 2.02 GMAC/s (3 thr) / 2.35 (4 thr), vs 7.2-7.8 in
  cpu_leaf_bench (#293). Differences from the bench: codes read from the 32-row TILED layout (two
  16-byte pieces per row per block, 512 B apart), one row per pass (x re-streamed per row), a scalar
  int8 -> float per code broadcast, thread spawn per call, runtime G loop. At 2.35 GMAC/s the share
  is ~4.5% of the GPU's rate: minus hand-off, ~0.1 s/turn, not worth wiring. NEXT: row-major repack
  of the CPU rows at load, 2 rows/pass sharing each x vector, codes widened 8 at a time (vmovl),
  persistent spinning workers; target >= 5 GMAC/s before step (c).
  BUILD NOTE: nvcc 10.2 cannot parse gcc-8's arm_neon.h (undefined __builtin_aarch64_* in the
  host pass), so NEON code lives in its own g++ TU linked into lfm3_nanode (measure.sh and
  measure_s2t.sh build it).
- [LFM3/Nano S2T] CPU prefill share, step (b) v2: 4.22 GMAC/s (3 thr) / 4.37 (4 thr), still bit-exact
  (2026-11-13, run #297; LFM_CPUCHK_V2=1). Row-major repack (load-time in a real split, untimed here),
  2 rows per pass sharing each x vector, codes widened 8 at a time (vmovl_s8/s16 + vcvtq), products by
  vfmaq_laneq_f32 (one fused fma each: same chain per lane). Real zh turn, last 1024 rows of every
  tiled prefill GEMV: 15,665,152 outputs, 0 bitwise differences, ids golden. 2.1x over v1 (2.02/2.35).
  3 -> 4 threads adds only 4%: the x tile is K x 32 floats (256 KB at K=2048, 960 KB at K=7680),
  re-streamed per 2-row pass, so it is cache/bandwidth-limited, and the timer includes per-call thread
  spawn. Revised split estimate with the #294 contention factor (-14% -> ~3.6 GMAC/s, r ~ 14) and
  L = 1.06 ms on every split: (3890 - 576 x 1.06)/15 ~ 0.22 s/turn. Remaining kernel levers: 4 rows
  per pass (halves x traffic, 32 accumulators -> needs G<=4 halves), x blocked so a block's 32 x 32
  floats (4 KB) are reused across many rows (loop rows inside blocks: needs the per-row leaf/tree
  state kept in memory), persistent workers.
- [LFM3/Nano S2T] CPU prefill share, v3 (4 rows/pass over 16-position halves, x traffic per row
  halved): bit-exact (15.7M outputs, 0 differ) but SLOWER: 3.91 / 3.84 GMAC/s vs v2 4.16 in the
  same session (3 threads, order v3/v2/v3). So the x re-stream is NOT v2's limit (refutes #297's
  reading); the extra passes over the codes and stack-held leaf/tree state cost more. v2 runs ~24%
  of the 3-core fp32 FMLA peak (~17.7 GFMA/s at 1479 MHz) with 16 independent chains: the next
  suspects are the per-call thread spawn (k/v calls are tiny) and the vcvt/vmovl issue share.
  Patch: .auto/attic/row298_neon_rows4_probe.patch (reverted, 2026-11-13, run #298).
- [LFM3/Nano S2T] CPU row share of the batched prefill SHIPPED (2026-11-13, run #299, LFM_CPUSPLIT
  default 1, 0 = control). w1/w3 (N 7680 = 15 x 512): the GPU kernel launches one 512-row block
  fewer (kernel unchanged), 3 std::threads run the last 512 rows with lfm3_neon_rows2 (row-major
  repack of the tail at first use, ~50 MB host), hand-off via a second stream (event before the GPU
  launch -> D2H x, CPU, f2bf, 2-D H2D into y, g_launch waits). ab_split.sh ABBA x 3 reps x zh/en,
  one build: prefill 3895-3904 (off) vs 3777-3793 ms (on) = -113 ms (-2.9%), turn -0.11 s, every
  pair separated, ids identical; LFM_DUMPX=132 residual md5 70b78a7b / 9591fe6c in both arms
  (= #264's). Per split: x wait 7.8 ms (preceding GPU work), CPU compute+put 7.4 ms.
  Levers left on it: persistent workers, also w2 / in_proj (need a non-512 share or a kernel row
  bound), q/k/v grouping; the share (6.7%) is ~ balanced with the rate ratio, so a faster CPU kernel
  is what moves it further.
- [LFM3/Nano S2T] #299's CPU split broke the held-out gate on the 110 s clip; fixed by mapping the
  CPU rows from the blob (2026-11-13, run #300). generalize.sh (cold arm now pinned LFM_CPUSPLIT=0,
  resident 1) on #299's build: 6 short clips IDENTICAL, 7/7 TTS bytes OK, but the 110 s clip's
  resident turn FAILED with an encoder MEMGUARD trip (post-turn MemAvailable ~560 MB vs ~770 before).
  stall.sh TTS=1, 110 s clip, 3 reps: split off encoder reply 6.8-10.7 s / turn 65-69 s; split on
  14.8-39.5 s / 72-98 s, ids eca77bbe every time -- the split's ~70 MB of ANONYMOUS repacked rows
  pushes the encoder pass into reclaim. Fix: the blob already stores each tensor as row-major codes
  then [N][nb] f32 scales, so the tail rows are mmap'd read-only from it (LFM_CPUSPLIT_MMAP default 1;
  verified once per tensor against the tiled device copy, any difference -> anonymous repack +
  printed line; 0 fallbacks). Clean file pages count as MemAvailable and are dropped under pressure.
  RssAnon with split: 85.8 MB (repack) -> 15.4 MB (= split off), RssFile 91 MB; ids + LFM_DUMPX md5
  exact. generalize.sh after: 7/7 IDENTICAL (unsplit cold vs split resident), 110 s clip 66.1 s
  resident vs 64.4 s cold, GUARD_TRIPS 0, MEMLOW 800 MB, GENERALIZE OK.
  LESSON: a resident-service speedup that adds anonymous memory must be gated on the long clip with
  three services -- the harness's 9 s turns cannot see the encoder's memory edge.
- [LFM3/Nano S2T] CPU prefill split: DEFAULT OFF again (2026-11-13, run #300). Three configurations,
  all bit-exact, none acceptable as a default:
  (1) anonymous rows (#299): harness asr 5.683 s (best), but generalize.sh FAILS the 110 s clip with
      three services (encoder MEMGUARD trip; stall.sh turns 72-98 s vs 65-69 s): +70 MB anonymous
      tips the encoder's long pass into reclaim. Shipping it would buy the 9 s benchmark with a real
      long-audio regression.
  (2) rows mmap'd from the blob (clean, reclaimable): generalize.sh OK 7/7 (110 s 66.1 s, 0 trips,
      MEMLOW 800 MB), equal speed in the 2-service ab_split.sh, but the harness's three-service
      configuration reclaims the pages between turns: asr 6.633 / 6.572 s (first timed turn prefill
      5440 vs 3785 ms, 4 KB refaults from eMMC).
  (3) (2) + madvise(WILLNEED) at clip start: asr 5.950 s, still worse than no split (~5.77): the
      ~50 MB re-read at ~90 MB/s is ~0.55 s against a ~0.11 s saving.
  Root constraint: on this 4 GB board with three resident services, the CPU share needs ~50-70 MB
  that is either unreclaimable (costs long-clip capacity) or reclaimable (re-read per turn). Revisit
  only if memory frees up (e.g. encoder activation cut for long clips, or two services instead of
  three) -- then (1) is a measured -2.9% prefill. Code kept behind LFM_CPUSPLIT=1
  (LFM_CPUSPLIT_MMAP=0|1).
- [LFM3/Nano S2T] CPU split + long-turn drop, mixed-length soak (2026-11-13, run #302, no engine
  change). soak.sh 12 turns, CLIPS = zh zh 110s zh zh zh (x2), three services, LFM_CPUSPLIT 1 vs 0
  in separate sessions: every turn rc=0, ids STABLE/GOLDEN, SNDR 70.09, 2 drops acked in the split
  arm. Short steady turns 5.212-5.230 s (split) vs 5.324-5.356 s (off) = -0.12 s (-2.3%); prefill
  3774-3802 vs 3903-3927 ms. First short turn after a long one: 5.447 / 5.488 (split) vs 5.580 /
  5.553 (off), i.e. the same +0.22-0.27 s post-long tax in both arms (#284b): the row rebuild is not
  visible there, because it happens inside the long turn's own prefill (first split call after the
  encoder pass). Long turn 63.77 / 62.92 s (split) vs 63.31 / 64.58 s (off), prefill 46.19-46.29 vs
  46.40-46.67 s: no regression, gain only ~0.2-0.4 s (the rebuild is paid there). Cost: MemAvailable
  701-745 MB (split) vs 767-806 MB (off), backbone RSS 89-102 vs 28-31 MB.
- [LFM3/Nano S2T] CPU split thread sweep; the caller is now a worker (2026-11-13, run #303,
  LFM_CPUSPLIT_TH default 3 = 2 spawned threads + the launching thread, which only blocked in join
  before). ab_split.sh split-on sessions, 2 zh + 2 en turns each, order 3/4/2/1/4/3, prefill ms:
  TH=3 3747-3763 and 3753-3767; TH=4 3759-3762 and 3760-3763; TH=2 3795-3819; TH=1 4958-5206.
  Before (3 spawned, caller idle): 3777-3793 -> about -25 ms; ids identical in every turn. Reading:
  the CPU side is marginally the critical path at 512 rows (2 threads lose 45 ms, a 4th gains
  nothing: the kernel is memory-bound past 3), so the split sits at its balance point. A larger
  share needs a faster CPU kernel, not more rows; sub-512 shares for w2 / in_proj would need a
  kernel row bound and a block's time does not shrink with fewer live rows (est. < 25 ms).
- [LFM3/Nano S2T] Split build: concurrency + 24-turn soak (2026-11-13, run #305, no source change).
  concasr.sh 3 trials x 4 clients: every client rc=0, golden where one exists, CLIPS_DONE=13
  ENC_DONE=13 (one per request). soak.sh 24 alternating zh/en turns, three services: all ids
  GOLDEN, SNDR 70.09, zh 5.181-5.199 s (was 5.416-5.439 unsplit in #284), en 6.088-6.109 s (was
  6.308-6.372), tts 0.458-0.470 s, MemAvailable 725-728 MB flat (788-790 unsplit), backbone RSS
  101-102 MB flat, 6034 splits. No leak or drift from the split. NOT exercised: a long (>30 s)
  turn among concurrent clients, i.e. the CPUROWS DROP line under concurrency (it is sent inside
  the turn lock, so it should serialise, but that is by construction, not measured).
- [LFM3/Nano S2T] CPUROWS DROP under concurrency (2026-11-13, run #306, tool change only:
  concasr.sh CLIPS= override). 3 trials x 4 clients launched together (zh, en, the 110 s clip,
  cv_zh9), backbone + encoder services (no TTS service in this tool): every client rc=0, zh/en
  GOLDEN, the long clip's ids md5 eca77bbe (= cold) in 3/3, drop acked 3/3, CLIPS_DONE=13
  ENC_DONE=13. The long turn landed 4th, 2nd and 4th in the queue (walls 78.0 / 68.0 / 78.1 s), so
  the drop was sent both behind and ahead of short turns; short turns queued behind it waited the
  full ~62 s (per-user latency is additive, #289).
- [LFM3/Nano S2T] Long turn among concurrent clients with THREE services resident (2026-11-13,
  run #307, tool only: concasr.sh TTS=1 + MemAvailable watcher). 3 trials x 4 clients (zh, en,
  110 s clip, cv_zh9), TTS service resident with one request served before and one after: every
  client rc=0, zh/en GOLDEN, long ids eca77bbe (= cold) 3/3, drop acked 3/3, GUARD_TRIPS 0,
  CLIPS_DONE=13 ENC_DONE=13, TTS after rc=0 SNDR 70.09. In-turn MemAvailable low 265 MB (same
  250-310 MB margin as the unsplit build in #280, i.e. the drop gives the split's memory back);
  the long turn's encoder reply 10.1 / 13.6 / 5.4 s (under reclaim in 2 of 3, as in #280), long
  turn 78-81 s when queued behind short ones. The ~250 MB margin on a 110 s clip is the board's
  limit with three services, not something the split changed.
- [LFM3/Nano S2T] NEON CPU rows v4, leaf-major (2026-11-13, run #308; LFM_CPUROWS_V=4, default 2).
  Outer loop = leaf, so a leaf's x blocks stay in L1 while all rows are visited; per-row tree state
  st[5] in memory. Bit-exact: LFM_CPUCHK=1024 over every tiled prefill GEMV of a zh + en turn,
  57,361,408 outputs, 0 bitwise differences vs the GPU. Check-harness rate (all shapes incl.
  K=7680) 5.19 GMAC/s on 3 threads vs v2's 4.22 (#297, other session). In the shipped split
  (w1/w3 only, K=2048), order V 2/4/4/2: prefill 3749-3757 / 3750-3760 / 3757-3761 / 3759-3769 ms
  = null; compute+put 10.29 / 9.79 / 10.20 / 10.43 s per 1493 splits (~-3%). So at K=2048 the x
  re-stream was not v2's limit (agrees with #298), and a ~3% faster CPU side does not move the
  prefill: with #303's thread sweep, the split's end time is not set by the NEON kernel's speed
  alone. v4's gain is at K=7680 (w2), which is not split. Kept opt-in, default unchanged.
- [LFM3/Nano S2T] Is the harness number representative? Speed on held-out audio (2026-11-13, run
  #309, analysis of #304's generalize.sh walls on the same engine code; no code change). Resident
  ASR wall / audio length: harness clips zh 5.180/9.00 = 0.576, en 6.100/9.00 = 0.678 (mean 0.627);
  held-out cv_zh9 0.576, cv_en9 0.676, cv_zh5 3.308/5.00 = 0.662, cv_en5 4.113/5.00 = 0.823,
  110 s clip 63.66/110.04 = 0.579; the four held-out short clips pooled 0.668. So the 9 s numbers
  transfer exactly to unseen 9 s audio, long audio is no worse per second (while it fits memory),
  and SHORT utterances are worse per second (5 s: 0.66-0.82): the turn has ~0.6 s of fixed cost
  (encoder reply ~0.45 s + client) and the transcript's decode steps (~100 ms each) do not shrink
  with the clip the way the prefill does. English is ~0.1 worse than Chinese at equal length
  (more ids per second of speech). The harness's s2t_rtf 0.63 is therefore a fair figure for ~9 s
  utterances and optimistic by ~5-30% for 5 s ones; quote it with the clip length.
- [LFM3/Nano S2T] Long-turn drop threshold 30 s -> 75 s (2026-11-13, run #310, LFM_SPLIT_LONG_S).
  30 s was a guess. Measured with 45 s and 75 s cuts of the 110 s clip, three services, soak.sh
  (zh, 45, zh, 75, zh), order drop/no/no/drop: 45 s turn 31.25 / 30.39 / 30.50 / 31.07 s, 75 s turn
  45.36 / 44.94 / 45.10 / 45.96 s -> keeping the rows is 0.6-0.7 s faster (prefill 18.0 vs 18.7 s,
  30.2 vs 30.9 s: the drop's rebuild from the device copy costs ~0.7 s, more than the split saves
  on that turn). Whole-session MemAvailable low 322-331 MB without the drop vs 346-377 MB with it,
  0 guard trips in every arm; ids of both cuts == the cold unsplit one-shot (5fe1c0a9 / 1389fad3).
  New default 75 s: turns up to 75 s keep the rows (measured), longer ones drop (110 s measured:
  generalize.sh 7/7, drop acked, 62.58 s resident vs 64.32 cold, 0 trips). 75-110 s without the
  drop is untested apart from #299's memguard trip at 110 s, so the threshold is not pushed higher.
  Note the ~0.7 s rebuild: for >75 s turns the drop still costs more time than the split returns;
  it is there for memory, not speed.
- [LFM3/Nano S2T] CPU-row rebuild priced (2026-11-13, run #311; timer added to the CPUSPLIT exit
  line, no behaviour change). 61 tensors are split (68.6 MB of rows). One build of all 61: boot
  622 ms, of which 400 ms is the cudaDeviceSynchronize before each read-back (queued GPU work that
  has to run anyway) and ~220 ms the D2H + repack; after a long-turn drop 1751 - 622 = ~1130 ms
  accounted (sync ~890 ms, repack ~240 ms) inside a 30.8 s prefill that is ~0.7 s slower than the
  no-drop arm (#310). So the recoverable part for a > 75 s turn is the ~0.24 s repack plus whatever
  of the sync is pipeline stall: at most ~0.7 s of a 45-63 s turn (~1%), long turns only. Not built.
  Also retracts #310's hint: skipping the split on the long turn and rebuilding afterwards would
  lose more (the split saves ~1.3 s of a 46 s prefill) than the rebuild costs.
- [LFM3/Nano S2T] CPU split backs off under CPU contention (2026-11-13, run #312; LFM_CPUSPLIT_MINRATE
  default 3.0 GMAC/s, LFM_CPUSPLIT_BACKOFF default 32, MINRATE=0 = old). Found with decont.sh
  (memstream.c: N threads streaming 96 MB buffers during a real resident en turn): the launch stream
  waits for the CPU rows, so a busy core makes the split the critical path. Prefill ms, quiet ->
  loaded: split off 3900 -> 3925 (1 thr) / 3945 (3 thr); split on, no back-off 3760 -> 4750-4975 /
  7326-7461 (the turn 6.07 -> 9.8-10.0 s). Each split now times its NEON share; two consecutive
  splits below MINRATE send the next 32 eligible GEMVs to the GPU only, then one probes again. Both
  paths are bit-identical (ids md5 identical in every turn of every arm). After: 4027-4045 (1 thr),
  4257-4283 (3 thr); quiet 3760-3783 (8 trips / 224 of ~1730 GEMVs skipped over 6 quiet turns, not
  resolvable in the prefill timer). Rate histogram on idle cores: 4-6.5 GMAC/s, tail down to 1.
  A single-sample rule at 4.0 re-tripped on quiet turns (prefill 3966-4051): the first split after
  a back-off is slow by itself. Residual under load (+270 / +500 ms vs split off) is the probes.
  Also measured: CPU memory streaming slows the bus-bound DECODE step with the split irrelevant:
  100.4 -> 111.7 ms/step (1 thread, 3.4 GB/s) / 118-119 (3 threads, 5.4 GB/s). A co-located
  memory-heavy process (e.g. the text->ids front end, an LLM for the reply) costs the ASR 11-18%
  of decode regardless; the harness cannot see it (nothing else runs on the board).
- [LFM3/Nano S2T] Twin x-tile reuse across the w1/w3 split pair: MEASURED OUT; the split's host
  buffers were also unsynchronized (2026-11-13, run #313). w1 and w3 are split back to back on the
  SAME ffn_norm output (same K, same Tn), so the second call can reuse the first's staged x tile:
  implemented (one-entry cache keyed on (x,K,Tn) + an explicit cpusplit_twin() adjacency token set
  between the two calls) and A/B'd same-build, ORDER 1 2 2 1 1 2 x 3 reps x zh/en: prefill 3766-3790
  ms with 640-658 hits vs 3757-3792 ms with the reuse off -- null, 24/24 turns ids md5 identical.
  So the D2H of x and the single-threaded bf16->f32 transpose of the whole 32x2048 tile are NOT on
  the critical path (they hide behind the GPU's own rows); only the NEON part is (#303). Reverted.
  TWO THINGS KEPT:
  (1) The rotating host buffers. The old single out/hy pair let the next split overwrite them while
      the previous call's ASYNC H2D was still reading hy -- safe only because ~6 ms of NEON sits in
      between, never synchronized. Now out[2]/hy[2] rotate and the slot being reused is waited on
      (cudaEventQuery on a never-recorded event reports complete, so it costs one query per split).
  (2) A reusable technique: ship every reuse cache with a debug arm that re-reads the source and
      memcmps it against the cached copy. Mine fired immediately (stale=2) and found a REAL bug in
      my own change: the adjacency token was cleared inside the split path, so a w3 call that did
      NOT split (the #312 back-off, or an ineligible shape) leaked the token to the NEXT split --
      which can be the next layer's w1, i.e. the same A.h1 pointer, same K, same Tn, stale content
      -> silently wrong rows. Consuming the token at the pf_gemv_T entry (whether or not that call
      splits) fixed it: 630 hits, stale=0, ids golden. RULE: any "reuse the previous call's staged
      input" cache must have its token consumed by the call site, not by the path that uses it.
- [LFM3+PrimeTTS/Nano] The TTS's device footprint does NOT limit the long turn (2026-11-13, run
  #314, measured out, reverted). Hypothesis: with three resident services the 110 s clip's encoder
  pass runs under reclaim because the TTS holds the largest device side after the backbone (110 MB
  of pooled weights + its activation arena), so release it for the long turn (the same protocol as
  the CPUROWS DROP) and take it back on the next request. Built: pt_v21_nano --serve-fifo answers a
  "FREE" line by pt_v21_close(c) and re-opens via pt_v21_open() on the next request; lfm3_asr sends
  it with the long-turn drop when given --tts-in. Result, 2x2 A/B (soak.sh CLIPS=zh,110s,en,75s x
  LFM_TTS_FREE 1/0/1/0): long turn asr 62.86 / 62.60 (free) vs 62.78 / 62.82 s (no free), 75 s turn
  44.80 / 44.54 vs 44.50 / 44.58, post-turn MemAvailable 692-732 vs 698-738 MB -- NULL in every
  column, ids and SNDR unchanged, and the served waveform after the FREE was byte-identical.
  AND the implementation cost is disqualifying: closing and re-opening the ctx makes EVERY later
  request pay ~27 s (27.42 / 26.98 / 27.07 s for what costs 0.46 s warm, TTS RSS 49 -> 152 MB), not
  the ~0.3 s of a weight re-upload -- the per-request cost persists across requests, so whatever
  pt_v21_open() rebuilds is being rebuilt every time. If anyone revisits a device-release protocol,
  it must free only the device side through the existing weights_release()/arena path and keep the
  host-side ctx (container records, graph tables) alive; a full close+reopen is 60x too slow.
  CONSEQUENCE: the long-turn memory pressure is not the third service. What is left on the board
  besides the encoder's own pass (#276 cut it 772 -> 536 MB) is the backbone's 2.15 GB arena, whose
  size trades against RSS rather than adding to it (Bonsai #23, row 50: MemAvailable is unchanged
  by arena size, pages just move between anon RSS and nvmap). So long-clip capacity now needs the
  encoder's activation memory cut (time-windowed downsampler), which is the only remaining option
  in that direction and is a project, not an iteration.
- [LFM3/Nano S2T] Long-clip encoder reply: cause relocated (2026-11-13, run #315). The served 110 s
  encoder reply is 5.4-6.6 s with TWO services (12 turns, one session, ids eca77bbe, 0 guard trips)
  and 15-18 s with THREE -- so the slow mode is memory pressure from the third resident stage, not
  the CPUROWS DROP timing (end-of-turn vs pre-encoder drop: equal, A/B'd 3 order alternations).
  #314 already refuted "release the TTS's device side". Next candidate for that 9 s: the third
  service's HOST side (243 MB RSS, evicted by the pass per #284b) -- i.e. general reclaim, fixable
  only by the encoder's own footprint: the time-windowed downsampler (x1 344 MB at 110 s; conv
  locality in time means a window of x2 rows [s0,s1) needs x1 rows [2s0-1, 2s1+1] -> ~32 MB per
  512-row window, peak 536 -> ~220 MB, exact by construction IF no time-axis reduction sits between
  the two convs (verify before building).
  HARNESS TRAPS from this iteration (both cost a whole A/B): (1) an 'A/B' whose binary failed to
  compile silently measured the stale board binary -- fail the run on ANY compiler diagnostic; an
  `echo BUILT` after a pipe hides gcc's exit status, and a helper inserted before its callees'
  definitions builds only with implicit declarations (wrong widths). (2) A stage-time grep with the
  delta on the wrong side of the name returned nothing; '?' in a results column is a parse failure,
  not a free stage.
- [PrimeTTS/Nano] Payload release REFUTED, and a metric-facing lead opened (2026-11-13, run #317).
  Protecting the TTS's 111 MB fp32 host payload arena PROT_NONE faults, and releasing every arena
  page that the first served request did not touch frees 110.9 MB but makes EVERY later request
  garbage (SNDR sentinel 1e9, wrong output lengths for all 5 golden cases) -- the arena is live
  memory. The PROT_NONE backtrace (sigsegv_trace <- op_gather <- pt_v21_run_graph) named only
  op_gather because gather is the FIRST host read in graph order; a fault site is the first reader,
  not the only one (my tracker at the dispatch sites said 302/336 records, and it was right).
  THE LEAD THIS OPENS (metric-facing, unlike everything else left): the host runner reads the
  weights of essentially every conv/convT/layernorm node once per request. If the CUDA conv is
  re-deriving something from those host bytes per request (a repack, a layout conversion, a lazy
  upload), that is ~110 MB of CPU traffic per request on the critical path -- at the A57's ~2 GB/s
  that is ~55 ms of the 470 ms served request (~11% of tts_wall, ~5% of combined_rtf). Test: count
  bytes the CUDA conv path reads from host weights per request, and check whether a repacked device
  copy could be built once at open (cf. PT_V21_DEVADD, measured ~6% SLOWER at N=1 in #279, which is
  evidence against a naive variant of this but not against a per-call repack existing).
