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
