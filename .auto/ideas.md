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
