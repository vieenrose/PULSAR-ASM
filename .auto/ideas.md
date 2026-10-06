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
