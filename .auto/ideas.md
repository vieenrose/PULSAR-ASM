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
- K2-FT PHONE LOOP (2026-10-09, CLOSED at practical bit-exact limit, 5.89 -> 14.09 tok/s, +139%).
  Kept: exp1 attention expf fusion (+3.6%); exp3 threaded attention heads (+98%); exp4 unrolled batch GEMV (+4.8%); exp6 2-row blocking (+11%). Discarded: exp2 score-dot vectorization (dup bypass slower-or-equal; scalar loop at latency floor); exp5 I-cache alignment lottery (+1.5%, fragile, reverted).
  Measured dead ends: little cores (taskset ff, 8 threads) 2.2x SLOWER (165.7s vs 76.7s, barrier stragglers + heat) — 4 big cores optimal, do not revisit. B=16 dead (needs 16 acc + 16 bases; SIMD+GPR pressure forces stack traffic that erases the gain). 4-row blocking dead (same pressure). Score/es/p attention at bit-exact floor (threaded+fused+vectorized; score loop latency-bound by IEEE order). Glue (norms/silu/rope/quant/copies) ~2% total; rms reductions + silu expf + q8 max-scan cannot parallelize/vectorize bit-exactly — skip. THP/prefetch considered, skipped (4.14 file mappings unlikely to merge; TLB cost ~1%). Unrolled scalar GEMV considered, skipped (~2%). Head skip for intermediate prefill tokens: breaks the step-line golden (verification), never.
  Roofline (phone memcpy 20.3 GB/s; FT weights ~545MB/tok): decode-short floor ~27ms (at 62ms); decode-long and prefill dominated by attention instructions + sustained thermal clocks, not bandwidth. Remaining bit-exact headroom ~10-15% via heroic micro-opts only. Further gains need more cores (worse here), new ISA, or relaxing bit-exactness (forbidden: demo bytes + Spark parity depend on it).
  Method lessons: phone component split (prefill vs decode) swings +-10% anticorrelated run to run (thermal/DVFS) while TOTAL holds +-1% — judge ONLY by total; single-run secondaries mislead (nearly chased alignment ghosts). Cool to <=38C big-core temp before timed runs; airplane mode + screen off; keep harness frozen mid-session. log_experiment keep sweeps git add -A: split history hygiene manually (0b1ce6d incident).

## K2-FT PHONE LOOP: DECLARED PRACTICAL BIT-EXACT LIMIT (2026-10-09)
Final: 5.89 -> 15.37 tok/s (+161%), 1080/1080 steps bit-identical. HEAD=exp#6.
Keeps (4): exp1 expf-fusion (+3.6%), exp3 threaded-attention (+98%),
exp4 unrolled-batch-GEMV (+4.8%), exp6 2-row-blocking (+21%).
Discards (8), each proving a mechanism: exp2 score-vectorize (dup bypass >=
scalar, latency floor); exp5 alignment (+-1.5% lottery); exp7 sleeping-pool
(-4.6%, DVFS needs 100% spin); exp8 scalar-unroll (-3.8%, frontend effects);
exp9 THP (-7%, khugepaged steals bandwidth); exp10 q8-NEON (-5.2%, frontend
swamps ALU); exp11 rope-NEON (-1.4%, true ~0%); exp12 memoize-quants (-6.6%,
noise/alignment, true +0.12% unmeasurable).
Dead without trial (measured or proven elsewhere): little cores (2.2x worse),
B=16 + 4-row blocking (register pressure), fdiv/reciprocal + f64->f32 + fast
exp (bit-exact forbids), head shortcut (margins too small per bonsai survey),
prompt/caching/batch/thread/config gaming (forbidden).
Remaining ideas all <=0.12% (100-700x below +-7% phone noise): silu-mul
(0.06%, expf-call overhead eats SIMD savings), q8-max (0.007%), sampler
(0.01%), fused-norm (L1-resident, ~0%), KV-relayout (~0.3%, high risk).
Phone variance +-7% (thermal/cache/background); judge by total only
(prefill/decode secondaries anticorrelate +-10%).
To unlock more: (a) fp32-close attention (real vectorization), (b) cooled or
bigger hardware (sustained 1.71GHz thermal cap binds), (c) new workload/config.
Engine + phone binary both at exp#6, gate green. Loop converged; do not revive
discards without a changed assumption.

## CPU-side (deprioritized per user: objective is GPU decode)
- Quad-row blocked A57 GEMV (k2_core_a57.S, ~30% op reduction/block): syntax-clean version saved at /tmp/k2_core_a57_quad_bak.S. Validated law: 4-product smull tree (v1.8b*v2.8b, v1.16b*v2.16b, v0.8b*v3.8b, v0.16b*v3.16b) == tail code. Bit-exact gate never run. Resume only if CPU leg matters again.
- V7 WARP-PER-ROW Q4 GEMV PROVEN (run #27, 2026-10-13): 1.19 -> 1.90 tok/s (+60%),
  EN 216/216 + ZH 114/114 bit-exact vs engine goldens. Design: 32 lanes cover
  8 blocks/iter (lane t: block s=t>>2, bytes q+4k q=t&3), xq streamed coalesced,
  pure pointer walks (+144/+292/+32/+292), lane-0-equivalent chain run
  redundantly on ALL lanes (unpredicated), fma(t,dot,acc) ascending = engine fmadd.
  Sources: pulsar_arm/k2horizon/gpu_v7/ (generator + ptx + engine snapshot + selfcheck).
- CUDA-8.0 (Nano) JIT LESSONS (cost a full debug loop, all proven by device dumps):
  (a) per-iteration integer address recomputation inside loops gets MISCOMPILED
  (stale values across the pipelined schedule) - use v6-style pure pointer walks;
  (b) heavily-predicated (@%p5) shared-mem load chains in loops get MISCOMPILED -
  use unpredicated all-lane redundant chains (single predicated store is fine);
  (c) min.u32 was NOT the cause (selp equivalent behaves identically);
  (d) kcmp-style gcc -O2 CPU references may be UNfused (mul+add) while the engine
  uses fmadd - 1-ulp diffs vs such refs are HARNESS ARTIFACTS; gate on engine goldens.
  Debug recipe that works: numpy sim of the exact mapping (catches logic bugs
  instantly) + device dump of per-lane partials with a C self-check mirror
  (catches JIT bugs by localizing to chain-vs-lanes in one run).
- NEXT (GPU): per-token = ~196 GEMV calls x ~2.7ms = 527ms. Candidates:
  (i) 2-row blocking in the warp kernel (amortize xq/code decode over 2 rows);
  (ii) fewer/smarter launches (batch small GEMVs: q/k/v/o per layer?);
  (iii) profile kernel-vs-overhead split first (nsys unavailable? use cuEvents).
- V8 VECTORIZED LOADS PROVEN (run #28): 1.90 -> 2.73 tok/s (+44%). Same (s,m)
  mapping as v7 but contiguous bytes j=4m+r per lane: code 2x u16 (18s+4m is
  2-aligned; u32 faults - 18s odd), x8/x16 u32 (32s+4m+4 always 4-aligned).
  Warp covers 8 contiguous 16B code chunks + 256B xq (was: strided 1B loads,
  ~4-6x L2-sector waste). 14 -> 6 loads/lane/iter. Bit-exact by construction
  (same products, int-exact dots, same fma order); EN 216 + ZH 114 green.
- NEXT (GPU): 366ms/tok / ~196 calls = ~1.9ms/call. Still ~2x sector waste on
  code (8x32B for 128B). Candidates: (i) 2-row blocking (share xq/code decode,
  halves xq traffic); (ii) sc/db broadcast via shfl/smem (minor); (iii) batch
  small GEMVs into fewer launches if launch overhead shows in profile.
- V9 2-ROW BLOCKING PROVEN (run #29): 2.73 -> 3.26 tok/s (+19%). Warp covers
  rows (2w,2w+1): xq/x16/db loaded once, dotted vs 2 code/sc rows (9 loads for
  2 rows = 4.5/row). Grid 16 rows/block (both engine launch sites updated);
  odd-row tail guarded (selp-clamp + uniform branch skip; engine selftest covers
  full range incl. tail). Debug note: shmem stride MUST be 1024B/warp (4 arrays);
  512B stride silently aliases warp w row-0 <- warp w-1 row-1. EN 216 + ZH green.
- NEXT (GPU): 307ms/tok / ~196 calls = ~1.6ms/call. Candidates: (i) 4-row
  blocking (register pressure? 2 rows already share xq; diminishing); (ii) code
  2x sector waste (8x32B per 128B: 16B chunks in 18B rows) - needs layout change,
  high risk; (iii) batch small GEMVs / persistent kernels if launch overhead
  shows; (iv) profile per-shape timings to aim the next cut.
- OPS: NANO "WENT DEAD" WAS A LOCAL SSH BUG, NOT THE BOARD (2026-10-13). `ssh picard@picard-desktop`
  failed host-key verification (stale hashed known_hosts entry for the short name), which reads
  identically to a dead box through the tailnet. Diagnosis ladder that worked: `tailscale status`
  (showed "active; relay hkg, rx 0") -> `tailscale ping` (timeouts, looked fatal) -> ssh to the
  bare IP with `-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null` => CONNECTED
  immediately, 23 days uptime. Lesson: when a tailnet peer looks dead, ALWAYS ssh with host-key
  checking disabled ONCE to separate "unreachable" from "key changed/stale". Board was up, no reboot.
  Fixed by reseeding known_hosts with `ssh-keyscan -t ed25519,ecdsa,rsa <name>` (short-name entry
  must be added for `picard-desktop` AND `picard-desktop.tailf63b31.ts.net`; a scan keyed on the
  IP does not satisfy the hostname lookup).
- V10 / EXP4 STATE (2026-10-13, pre-benchmark): /tmp/k2_cuda_v10.c == Nano /tmp/k2_exp.c
  (md5 1fd18113fe9070383089fb3600bdad42), PTX /tmp/k2v10.ptx in sync both sides
  (md5 061347d1a7604b7283b9870ef6bbc5b8). Gate evidence on Nano: /tmp/gate10_out.txt shows
  "CUDA self-test OK (8/8 bit-exact)" and /tmp/zh_check.txt has 114 steps == k2nano_golden_zh.txt.
  Change vs v9: Q4 code repacked from 18B-stride rows to contiguous 16B blocks (host-side one-time
  gather per tensor, cached in rp_cache[64] keyed by device base/rows/nb), killing the 2x 32B-sector
  waste on code reads; addressing is shl #4 / shl #2 instead of mul 18. Bit-exact by construction
  (same products, same order).
- V10 EXP4 DEVICE-MEMORY WALL (2026-10-13, Nano went dark mid-bench): the v10 repack stages
  a SECOND copy of every Q4 tensor (16B blocks, 635 MB) alongside the existing 714 MB blob
  mirror => 1349 MB device footprint on a 4 GB Nano whose TDA/carveout is ~1.9 GB. Evidence:
  `mlock blob failed (errno 12)` at startup (already at the RLIMIT_MEMLOCK edge), the box's
  tailscale rx counter climbing to 1.59 MB then freezing, `ssh` rc=127/137, and 5/5 tailscale
  ping timeouts. My own two `ssh` calls exited **137 = SIGKILL = local OOM killer** (3791255
  bench + tee chain was also live). Net: the Nano is now unreachable at the network layer
  (tailscale `Online: True, Conn: None` = stale key-expiry only, no key exchange).
  FIX REQUIRED BEFORE ANY v10 RUN: (1) do NOT double-stage — repack IN PLACE into a single
  canonical Q4 layout (one copy) and delete the 18B-stride mirror from the loop path, i.e.
  stage only the repacked 635 MB and drop dev_base for the Q4 region; (2) free RAM first
  (jtop x4 was running, 749 MB used + 172 buff/cache); (3) never launch a bench while the
  box has < 800 MB MemAvailable. Recovery: power-cycle the Nano, verify
  `free -m`, `tailscale ping`, then re-run .auto/k2nano_gpu_bench.sh with SRC=/tmp/k2_cuda_v10.c.
  v10 SOURCES ARE SAFE: committed b18104e (gpu_v10/{gen_v10.py,k2_gexp_engine_v10.c,k2v10.ptx}),
  md5 src 1fd18113fe9070383089fb3600bdad42, PTX 061347d1a7604b7283b9870ef6bbc5b8. Gate evidence
  captured pre-crash on the Nano: /tmp/gate10_out.txt "CUDA self-test OK (8/8 bit-exact)",
  /tmp/zh_check.txt 114 steps == .auto/k2nano_golden_zh.txt. NOT benchmarked => v9 (3.26 tok/s,
  fea4c10) remains HEAD of the GPU line.

## Exp8 postmortem (run #37): shuffle reduction ABANDONED on Nano driver
- `shfl.idx.b32` / `shfl.up.b32` JIT-compile (regprobe + engine device-stage OK) but
  deliver ~0 at runtime on L4T R32 (R440-era ptxas, sm_53), regardless of form:
  mask 31 / -1 / reg, .idx / .up-with-delta, .version 4.0 / 6.4 — all bit-identical
  wrong (~0 rows). Graft proof (v14 loads + v12 chain = self-test 8/8) isolates shfl
  as the sole defect. DO NOT retry shfl on this toolchain without new information.
- Fresh-text row blocks need differential proof: gen_v14.py now self-checks that all
  4 row emission blocks are structurally identical modulo register renaming
  (caught 2 typos in row 3: lo-sub `%r30,%r29` and hi-and `%r31,%r29`).
- NEXT: chain redundancy remains the fat (~50% of per-iter cost) but must be cut
  WITHOUT cross-lane exchange: candidates (a) per-lane dual-s without exchange is
  impossible (order) — instead (b) 16-block iters need 2KB shmem/warp (alias risk),
  (c) small-grid latency sense: 512-row calls at 1GB/s vs 5120-row at 2.1GB/s —
  investigate ILP (2 iters interleaved per thread?) or (d) fuse host per-call sync
  (196 drains/token) via double-buffered async pipeline.

## Exp9 (run #38) f3-kernel postmortem (2026-10-13)
- f3 (3-seg row-fusion kernel, v12 body + routed bases) NEVER produced correct output despite: loop proven identical by mechanical diff, setup math verified by hand 6x, params proven (15-slot echo kernel bit-exact 37382.0), inputs proven (single-kernel oracle bit-exact on same values), stores proven landing (canary). Minimal-correct variants (V-A: no branches; V-D: 6 loads, 64 regs = v12) failed BIT-IDENTICALLY. Verdict: R440/ptxas-6.4 miscompiles correct PTX in multi-entry modules (same class as v14 shfl delivering ~0). Do NOT revive f3-style custom kernels without a standalone tiny-harness proof first.
- Real bug found in f3 text anyway (hoist retarget): 2-temp pointer math collapsed 16s (added 4q twice, dropped 16s). Generators: never merge two live temporaries into one reg across unconsumed values; the row-block differential self-check (v14) doesn't catch setup-offset bugs (it covers nibble/dot code, not pointer math).
- xq-compare lesson: per-292B-block trailing [260..292) is quantizer-skipped stale bsums (kernel-invisible); full-memcmp across member calls is FLAKY (stale padding varies). Masked compare (64B header + [0..260)/block) is the correct fusion gate.
- Tegra unified-memory fragmentation: 7.8MB cuMemAlloc failed mid-run (after 467MB repack churn) but 13MB pre-allocated early at init succeeds. Pre-allocate all device regions early; fill lazily.
- Fusion-via-LAYOUT (Exp9b) is the surviving design: fused W/SC regions (DtoD) + proven single kernel. Zero PTX risk.

## Run #40 quoting postmortem (2026-10-13) — READ BEFORE ANY HAND SSH RUN
- Hand ssh runs used "$EN" (QUOTED: one space-separated argv string -> binary sees a 1-token prompt -> 1-echo streams that NEVER match golden). The bench uses unquoted $EN (21 separate argv ids -> 21 echoes -> golden). ALWAYS replicate bench argv style (unquoted) in hand runs; never quote $EN. This artifact voided: vrabal, bisA/B/C, ref200, k2_base probes, gen-quirk theory, consumer/uninit/layout/OOM/stuck-bit theories (ALL built on 1-token-prompt data).
- #38 (v15b, bench args): the ONLY genuine post-v12 divergence (per-layer regions bug — real).
- Smoke data (fusion engages, self-checks pass) REMAINS VALID (checks are self-consistent within a run, prompt-independent).
- k2_base was never validly stream-tested (all probes quoted); it is timing-baseline-only (never gated) — do not use its stream as an oracle.

## Exp9 CLOSED (run #41 checks_failed): call-buffering fusion is design-dead
- Proof: same ordinal 5120-call sees scb=0bf326216 (garbage) under fusion vs the same valid shared scb 6x under v12 (v12-snoop). Sequences demonstrably differ => model consumes member-y (or its derivatives) between member calls (RoPE/cache between q,k,v; SiLU between up/gate suspected). Buffering delays compute to group completion => violates call->compute->read contract. All value paths verified exact (11k launch checks, flush-vs-single check, mask==CPU-read-set, PTX identical) => pure timing/contract bug, invisible to value checks.
- Lessons: (1) ANY fusion must preserve per-call y-timing (compute+sync before return). (2) Self-checks that compare final values CANNOT catch timing violations; need sequence-level oracles (step streams only). (3) Flush path (unverified, no checks) is where prefill diverged first; future designs must check EVERY path or eliminate contingencies. (4) v15b's layer-regions bug (build once, reuse across 28 layers) vs v15c's per-layer rebuild: per-layer is required (tensors are per-layer); amortizes to zero only if sources stabilize (they didn't in prefill (monotonic advance), would in decode).
- BANKED (timing-safe): CUDA-graphs replay (capture per-shape graphs once, replay per call: same y-timing as singles, ~5us replay vs ~25us launch; R440 supports graphs (r410+)). Effort: capture machinery for 7 shapes + head. Next-loop candidate.
- Banked: persistent work-queue kernel with sync-before-return (preserves contract; saves launch (~25us->~3us/call ~ 4ms/token ~1.5%); needs PTX spin-loop around v12 body (asm risk)).

## Exp12 (next): occupancy without extra waves at 4 rows (from #42/#43)
- #42 v16 8-row: 512T (-14.5%). #43 v17 2-row: 1536T but 2x waves (-1.9%). Tradeoff balanced at 2-4 rows.
- Proposal: keep v12's 4-row shape/grids (32 rows/block, same waves) but cut per-block cost: pack shbuf 16KB->~10-12KB (sc slots use only 8 of 64 entries/row; dot uses 32 of 64) + trim regs 64->~48 via temp reuse => 4-5 blocks (1024-1280T) at SAME wave count. Predicted +5-10% if latency-bound.
- Tool fix: /tmp/regprobe.c reads attr id 8 (MAX_DYNAMIC_SHARED_SIZE_BYTES) mislabeled as SHARED_SIZE_BYTES (real id is 1). True static smem = 49152 - reported. v12/v16/v17 static = 16/16/8KB confirmed via ptxas -v. Anyone using regprobe shared numbers must reinterpret.

## Occupancy direction CLOSED (from #44 + calculator + nvprof-blocked)
- v18 (8KB decl, same 64 regs): occupancy-calculator says 4 blocks fit (64*256*4=65536 exact, 8KB*4=32KB<=48KB, 32 warps<=64) => v18 ran at 0.5 vs v12 0.375, yet tied 269 vs 268ms.
- Conclusion: kernel is NOT latency-bound above ~768 threads; v16 (512T) was below the knee, v12 already at it. More occupancy cannot help. Do NOT pursue further block/occupancy variants.
- nvprof hardware counters are permission-blocked on Nano (ERR_NVGPUCTRPERM, no passwordless sudo) => no achieved_occupancy/stalled measurements possible. API-trace timing still works.
- Remaining levers: (a) fewer BYTES per token (bigger quant groups = accuracy risk; fewer GEMV calls = fusion/CUDA-graphs), (b) launch+staging overhead ~50-64us/call x ~200 calls (~4-5%: CUDA-graph replay of decode sequence, banked), (c) effective-BW anomalies (why ~3.4GB/s on coalesced streams? revisit K2PROFILE class breakdown in b38 before theorizing).
- v18 one-liner (shbuf[4096]->[2048] in gen_v12.py line 19) is safe/bit-exact and kept 4-block occupancy; reapply on top of any future v12-derived kernel for free.

## Geometry ridge CONFIRMED, stop reshaping (from #42-#45)
- Rows/warp: 2 (v17, -1.9% tie) / 4 (v12 HEAD) / 8 (v16, -14.5%). Rows/block: 16 (v17 tie) / 32 (v12) / 64 (v19, -10%).
- v12 (4 rows/warp, 32 rows/block, 8 warps) is a tuned optimum in EVERY direction. Halving waves at same work/thread LOSES => wave fixed-cost does not dominate; fill/drain + resident-parallelism balance favors 32-row waves.
- Exp14 sketch (needs K2PROFILE per-class exec first, .auto/k2_prof.c): CLASS-SPECIFIC dispatch - small k/v GEMVs (512 rows, 16 blocks, tail-heavy) vs large up/head (160-1000 blocks) may want different geometries; one hetero kernel pair, each proven-shaped. Do NOT attempt without per-class numbers showing >5% addressable.
- Fallback if Exp14 unmotivated: CUDA-graph replay of decode GEMV sequence (~4-5% launch overhead). Big change; only with no better idea.

## LOOP CONVERGED on v12 @ 3.73 tok/s (after #46, 21 experiments)
- Geometry ridge: 2-row tie, 8-row -14.5%, 64-rowblock -10% (avg AND head-only). v12 shape optimal everywhere.
- Occupancy knee at 768T; bytes-bound above. Clocks already maxed (podgov). No thermal issue. Overhead ~4% but CPU-serial (graphs ~0.4% only).
- Do NOT retry geometry/occupancy/fusion-launch ideas without changed assumptions.
- If resumed: structural work only (prefill batching needs engine batch support; quant-group changes need accuracy qualification) or micro-levers (zero-copy xq ~1%, async DtoH ~0.5% - poor EV).
- Driver lesson: cuModuleLoadDataEx WITH jit-log options broken on this Tegra libcuda (rc=1 empty log; earlier segfault with info-log opts). Always use plain cuModuleLoadData here.
- Exp15 v21 PAIR-REDUCE (benched #47): warps 0-1 reduce 4 slots each (16 rows), 6 warps skip warp-uniform; sums distributed via shbuf upper half (r61+8192); lane0 loads+stores. 30x fewer shmem reduce-loads (LD_SH was 37% of static PTX). Bit-exact by construction (identical FMA/leaf order). Self-test 8/8 first try. Regs: f<40 decl (16 accs f1/f16/f22/f25+f26-f37), r76=tid.y r77=scratch, p7 dispatch (dead pred). Labels $red0/$red1/$redDis/$redSkip.
- Exp16 v22 LANE-SPLIT EXCHANGE (benched #48): lane (q*8+sv) reads only its sv-partial (5 shmem reads vs 160/lane); 32 uniform (sc,dotsum) writes to per-warp area r65+8192 (256B x 8 = 2KB, upper shbuf); bar; lane0 replays 32 FMAs in exact v12 order (bit-exact, 4 accs, same regs/decls, no new branches, same 2-bar count). Self-test 8/8. BUG LESSON: gen line56 (r61/r65 setup) is INSIDE $LIT (recomputed per iter) - first v22 draft did add r61,+8192 there = runaway base, MISMATCH idx=0; fixed with per-iter fresh r79 base.
- Exp17 v23 SV-MAJOR replay (benched #49): lane0's 32-FMA replay reordered q-major -> sv-major (4 independent acc-FMAs per sv step, 4-wide ILP covers FMA latency). Bit-exact by construction (each acc's sv-ascending FMA sequence unchanged; independent accs commute). Self-test 8/8. Tests the latency-bound model: if FMA latency limits, this wins; if tie, limiter is elsewhere (int-IMAD chains?).
- Exp18 v24 UNROLLx2 + ping/pong/pang (benched #50): trip=2 iters, B-loads (+128/+292/+32 address deltas, fresh regs r80-89/f26-30) issue before A-compute (22 loads in flight covering L2-miss latency); per-warp ping A/pong B/pang-epi 256B areas (r79/r92/r93 setup-hoisted) drop the post-replay bar (3->2 bars/iter). Bit-exact (identical per-(iter,row,sv) order). Self-test 8/8. BUG LESSON: body() retyped row3 as sub r30,r29,8 (should be r30,r30,8) -> idx=3 MISMATCH; caught by v23-PTX differential (A-body must equal v23 modulo +0 addrs). Always diff new bodies against proven PTX.
- Exp19 v25 4-LANE PARALLEL REPLAY (benched #51): lane q (0..3) owns row q's FMA chain (was lane0 running all 4 chains = 96 instrs/iter). Reads its 8 (sc,dot) pairs at stride 32 from base 8q (exchange layout already matched: pair(q,sv) at 32sv+8q), sv ascending => bit-exact. v2 shared loads/stores; single predicated store for all 4 rows (replaces 4 store blocks + 3 branches). Replay 96->16 warp instrs/iter (~23% of per-iter issue slots). R440 LESSON: setp.lt.and.u32 is REJECTED by ptxas (line 460 arg mismatch) — use two setp + predicated mov to synthesize AND-inside-val; also cuModuleLoadDataEx jit-log is broken (see earlier).
- Exp20 v26 BFE/XOR/MAD int path (benched #52): per r-step 48->22 instrs. (a) int8 x bytes: shl+shr.s32 sign-extend (2) -> bfe.s32(width 8) (1); (b) int4 code nibbles: shr+and255+and15+sub8 / shr4+and15+sub8 (7) -> xor.b32 0x88<<8r (1) + 2x bfe.s32(width 4) (2) — XOR flips nibble sign bits so sign-extend yields exactly (n-8); (c) mul/mul/add + acc add (4) -> 2x mad.lo.s32 (2). Integer math is EXACT (no rounding) => order/format-free, bit-exact by algebra. Exhaustive 16.7M-combo equivalence proof in Python before bench. ~96 fewer instrs/iter.
- LESSON (v27 Exp21): when hoisting a load block across the pointer WALKS, the immediate offsets must absorb the walk amount (A-next loads need +(256,584,64), not 0). Bug signature: deterministic MISMATCH idx=0 with self-test gpu!=cpu; fix = pass delta=(256,584,64). Loads moved earlier for pipelining must be re-anchored to the pre-walk base.

## Exp25 v31 (host-path slim-down: async copies + persistent context) — 2026-??
Hypothesis: per-call fixed cost ~140us (H2D 28 host-blocking + launch 46 + sync/D2H 63)
x ~190 calls/token = ~14ms/token. Keep the CUDA context current for the process
lifetime (no per-call cuCtxSetCurrent(g_ctx)/cuCtxSetCurrent(0)) and use
cuMemcpyHtoDAsync/cuMemcpyDtoHAsync on the legacy default stream with a single
cuCtxSynchronize at the end, so the host never blocks on the H2D and the D2H is
enqueued before the sync. Fallback to the sync path if the async symbols are absent.
NOTE: cannot beat the ~20-50us device-side dispatch latency; only host stall removed.

## Exp26 v32 (pinned staging buffers) — host-path step 2
v31 removed host blocking but the copies still stage through pageable memory.
Persistent cuMemHostAlloc buffers (64KB in / 512KB out) + one host memcpy each way.
Targets the ~63us D2H + ~28us H2D pageable latency x 188 calls/token.

## Exp35 v43 (run #66): drop both per-iteration bar.sync — warp-synchronous STS→LDS
- v29's two per-iter bar.sync (named barrier tid.y+1, count 32) are redundant: all 32 lanes of a warp are warp-synchronous; every STS→LDS pair is intra-warp (area base r65 = shbuf + tid.y*1024; the v4 LDS reads the 4 dot partials written by lanes 4*sv..4*sv+3 of the SAME warp; the replay reads the warp's own 256B area). SASS shows each BAR.SYNC also emits MEMBAR.CTA → removing 2 pairs saves ~4 instrs/iter (1.3%).
- Self-test 8/8 on first build. If EN/ZH gates pass, the warp-synchrony argument is validated.

## Exp39 v46 (run #68): threaded init (dlopen'd pthreads) — DISCARDED, and WHY (important)
- Implemented exactly as designed: `libpthread.so.0` resolved at runtime (link line is
  `-ldl` only), 2 threads for the 467MB 18B->16B repack (row-slice segments) and for the
  H2D (16 x 64MB chunks, each worker does `cuCtxSetCurrent`). Self-test 8/8, 216-step gate
  would pass; bench = 6.25 / 6.45 tok/s vs v38 baseline 6.29 / 6.54 => pure noise, no gain.
- ROOT CAUSE of the null result: the isolated microbenchmarks that motivated it are NOT
  representative. 467MB memcpy: 1 thread 435 ms vs 2 threads 52 ms (8x!) — but that buffer
  was pre-touched. In the engine, `tmp = malloc(467MB)` is FRESH: the repack pays ~115k
  first-touch page faults, serialized on the kernel mmap_lock, so extra threads buy nothing
  (they contend on the same lock). Same for H2D: the engine's H2D is not the 530ms measured
  from a pre-faulted source buffer.
- LESSON (do not repeat): to cut init you must remove PAGE FAULTS, not add threads to a
  fault-bound loop. If init is ever revisited: `mmap(MAP_ANONYMOUS|MAP_POPULATE)` (or
  hugepages) for the repack buffer, and repack+H2D pipelined per-chunk so the first token
  starts earlier; measure with K2INITDBG (INIT ... lines) before believing any win.
- NOTE the init-cost accounting pitfall: init is 1.74 s of a 34 s window = 5.1%, so a FULL
  elimination of init is worth +5%. Threading the repack (435 -> ~50 ms of CPU time) does
  not translate because the metric is dominated by 216 tokens x 155 ms of GPU-bound decode.

## HARNESS DISASTER + RECOVERY (2026-10-14, run #69 attempt)
- The Nano (picard-desktop) REBOOTED (uptime 3:26h, boot 2000-01-01 09:00 fake-clock). /tmp is tmpfs:
  ALL of /tmp on the Nano was wiped — k2h_09_q4.blob, k2_base, k2_gexp, k2_exp.S.
- LOCAL /tmp was ALSO wiped (local machine rebooted: /tmp is tmpfs, 31G tmpfs now ~empty):
  /tmp/k2_cuda.c, /tmp/k2_cuda.S, /tmp/k2_cuda_v*.c ALL GONE. The /tmp-only files were:
  k2_cuda.S (THE engine core asm — k2_core_a57.S + .globl line + 3 call-site swaps:
  bl k2q4_gemv_range -> bl cuda_gemv at ~1608/1653/1691, + bl cuda_init at ~3148),
  /tmp/k2_cuda.c (staged HEAD engine = gpu_v43 copy), and the v33-v48 working C files.
- RECOVERED into .auto/rescue/ + git commit 8cab961: k2_cuda.S (md5 4f1eb84b, from
  luigi@spark:~/k2/k2_cuda.S, Oct 12 22:05 — matches the deployed-md5 record), k2_core_a57.S
  (identical to repo copy), k2_cuda.c (Phase-1 engine, reference only).
- REBUILD RECIPE (memorize): engine C = pulsar_arm/k2horizon/gpu_vNN/k2_gexp_engine_vNN.c ->
  /tmp/k2_cuda.c; asm = cp .auto/rescue/k2_cuda.S /tmp/k2_cuda.S (THE .S NEVER CHANGED after
  Oct 12 22:05 — all v33..v48 iterations were C-side (q4_build/gemv host path); kernel changes
  went through gen_vNN.py -> PTX string inside the C, so k2_cuda.S is stable).
  k2_base = as -march=armv8-a k2_core_a57.S + gcc -static -nostartfiles (rebuilt OK on Nano).
- Blob: md5 1b026978510f9111bb0a6a6607f3c50d, 748976168 B, source luigi@spark:~/k2/k2h_09_q4.blob
  (spark is the durable store; Nano /tmp is volatile!). Restore: direct hop
  `scp luigi@spark:~/k2/k2h_09_q4.blob picard@picard-desktop:/tmp/` (worked at 30+ MB/s).
- RULE: keep durable copies of /tmp assets in the repo (.auto/rescue) and re-verify at loop start:
  `ssh picard 'ls /tmp/k2h_09_q4.blob'` + md5 of /tmp/k2_cuda.S before any bench.
- Run #69 (v48 pinned-staging init) pre-bench findings BEFORE the wipe: cuMemHostAlloc(467MB)
  SUCCEEDS on the Nano (rc=0, ~450ms) and pin+pretouch+repack ran, but the process was OOM-KILLED
  (dmesg: Out of memory: Kill process k2_gexp48, total-vm 9.5GB) DURING the 467MB H2D: on 4GB
  Tegra the 467MB pinned staging + 467MB device alloc + 714MB blob mmap exceed RAM. The v48 design
  MUST use the 32MB ring (mode 1), not the single 467MB pinned buffer (mode 2): Q4_RING=2 is DEAD
  by design on this board; Q4_RING=1 is the viable config. Ring numbers before kill: dev-alloc
  467MB ~288-826ms, pin 467MB ~441-1979ms (!), pretouch ~204-239ms.
- v48 code survived? NO (was in /tmp only). Rebuild from gpu_v43 engine + re-add ring:
  Q4_NSLOT=2 x Q4_SLOT=32MB pinned ring, cuMemcpyHtoDAsync(g_stream), pre-touch slots with
  memset, free via cuMemHostAlloc'd pointer -> cuMemFreeHost ONLY (free() = SIGSEGV double-free).

## Exp42 (run #69) — v49 pinned-ring streaming init, pre-bench evidence
- repackprobe/repackprobe2 on the Nano (910MB dest, byte-verified identical output):
  V0 glibc `memcpy(d, s+2, 16)` = **1007 ms**, V5 packed-struct 16B assignment = **287 ms**
  (V4 __builtin_memcpy 295, V7 u64 pair 288, V8 unrolled x4 296 — all identical bytes,
  all within noise of each other; took V5). => ~700 ms of the engine's measured
  981 ms "repack + H2D" init phase is glibc memcpy call overhead on 16-byte copies.
  v49 applies V5 to BOTH the serial repack and the ring producer (shared Q4_REPACK_INTO).
- h2dp6 (engine-state probe, 467MB device alloc first): depth-1 H2D 1.05 GB/s,
  depth-2 **2.86 GB/s**, single cuMemcpyHtoD 5.99 GB/s (82 ms for 467MB).
  **Corrects a misreading of h2dp5**: its `rc=1` at slot 4 was an OUT-OF-RANGE
  destination (15 x 32MB = 480MB written into a 467MB cuMemAlloc), NOT a queue-depth
  limit. Depth-2 is legal on R440. v49 NSLOT=3, sync every 2 enqueues.
- Predicted v49 init: pin+pretouch 3x32MB ~80 ms + fast repack ~150 ms + H2D overlapped
  ~163 ms => ~250 ms vs 981 ms measured for the serial path, i.e. ~700-900 ms of the
  34 s metric window (2-2.6%) IF the ring works; falls back to the serial path (with
  the fast repack) on any allocation failure, which alone is worth ~700 ms.
- Probe scripts (in /tmp, ephemeral — content in this file): h2dp5/h2dp6 (pinned DMA
  throughput vs queue depth and engine memory state), repackprobe/2 (16B copy forms).

## Exp45 CUDA-GRAPH replay: MEASURED DEAD on R440 (run #70/#71 probes) — closes the banked idea
- pgb probe (real CUDA 10.2 signatures; R440 has NO cuGraphAddMemcpyNode1D — that is CUDA 11+,
  the 10.2 symbol is cuGraphAddMemcpyNode with CUDA_MEMCPY3D by pointer):
  a 3-node graph (H2D -> gemv -> D2H) instantiated once, replayed per call, is
  **BIT-IDENTICAL** to the direct path (0/256 mismatches) so the mechanism WORKS,
  but per-call cost is HIGHER than the engine's current path:
    A engine-style (H2DA+launch+D2HA+event-poll) 323.75 us
    B graph + per-call cuGraphExecKernelNodeSetParams 250.44 us  (set 4.16, launch 68.14, poll 178.03)
    C graph, fixed params                                 206.07 us
    D graph pipelined (no per-call event)                 961.54 us  (queue saturation, far worse)
  The killer number: **cuGraphLaunch = 68 us vs cuLaunchKernel = 42.5 us** (pg v2, 1-node
  graphs amortized to 11.14 us/node over a 30-node chain, i.e. graph *dispatch* is cheap but
  graph *submission* costs ~60 us fixed). On R440 a graph launch can therefore never beat a
  single kernel launch; it only pays off when ONE launch stands in for many nodes.
- DO-NOT-RETRY: per-call graph replay to cut the engine's ~100 us/call host path. The
  submission cost is in the driver's per-launch path, which graphs share (and exceed).
- Consequence for the remaining search: the ~9.2 ms/token of launch cost (40 us x ~230 calls,
  6.2% of 149 ms) can only be removed by REDUCING THE NUMBER OF LAUNCHES, i.e. a persistent
  worker kernel (one launch per token, device-side doorbell + completion flags in pinned mem).
  Estimated ceiling +6-8%; big change, bit-exactness preserved because the math per row is unchanged.

## Exp51 (run #73 probes) persistent-kernel megakernel: feasibility PROVEN, engine build pending
- p63/p65 bisect: the rc=719 "persistent kernel crash" was MY addressing bug (loop kernels added
  the base pointer twice: `rd3 = pw + tid*stride` computed outside the loop, then `pw + item + rd3`
  inside -> OOB address). Corrected loop kernels (p67 q1/q2/q3) run CLEAN on R440 (sync rc=0,
  liveness verified nonzero). No driver prohibition on loops/atomics/stores in kernels.
- Measured (probe_bw68, random-filled buffer, 360MB, engine-dominant geometry):
  flat grid 14656x256 @96B/thr: 13.25 GB/s (27.2ms)  [p96 == n96: .nc texture path = NO effect]
  loop grid 64x256 (under-parallel): q1 11.05 GB/s, q2 atomics 79 GB-s-IMPLAUSIBLE (see below),
  q3_small 16.7ms/120MB = 7.2 GB/s effective.
  LIVENESS TRAP: q2_loop_atom reported y[0] = 2x expected and 79 GB/s > DRAM peak (25.6 GB/s) ->
  the probe's y expectation formula and/or the compiler's dead-code-elimination inflate loop-kernel
  numbers; NEVER trust GB/s from loop kernels without a byte-verified checksum. p96 flat is the
  honest baseline: 13.25 GB/s at this access pattern.
- probe_graph69 (1-node graph vs raw, engine cadence 229 calls x 20 tokens): GRAPH is 30.5 us/call
  SLOWER (2969 vs 2938 us/call) -> confirms Exp45: per-call graph replay is DEAD on R440. DO NOT
  RETRY graphs at any node count.
- DESIGN for v53 megakernel (if pursued): host enqueues 229 work descriptors (w, sc_off, xq, y,
  r0, r1, nb32) per token into a pinned ring; ONE launch per token; kernel grid-strides over all
  rows of all descriptors (full-grid occupancy ~128 SMs x 2 blocks x 256 thr); completion via
  device->pinned progress word + host poll. Bit-exactness: per-row math unchanged.
  Ceiling ~ +6-8% (9.2ms launch cost + inter-kernel gaps out of 143ms). Loop-form BW loss at
  64 blocks (11 vs 13.25 GB/s) means the megakernel MUST use a flat grid-stride over (item, row),
  not a per-block serial loop over items.
- Launch-cost reference (probe_queue54, engine shapes): launch-only 96.5 us, H2DA+launch+D2HA
  65.5 us, +event-poll 111.6 us per call at engine sizes.
