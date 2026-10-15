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
