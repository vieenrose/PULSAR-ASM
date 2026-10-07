# Autoresearch: Spark Bonsai-27B prefill + decode speed at exact accuracy

## Objective
Make the 27B engine (`pulsar_arm/bonsai2`: `fwd.c` + `tq_gemv_neon.c` +
`fwht.S`) faster per token on the Spark (20× Cortex-X925, CPU-only) —
both prefill (prompt processing) and decode (generation). Every kept run
must produce **bit-identical greedy streams** on fixed prompts
(`checks.sh` enforces this; it cannot be overridden).

Current state: ~0.71 s/token threaded (20 cores), ~7.6 s/token single-core.
GEMV kernel does 0.29 ns/weight single-core (BIT-EXACT). The engine is
compute-bound (unlike the Pi gen1 loop, which converged at the 3.93 GB/s
bus wall — that session is closed, see git log + `.auto/log.jsonl`
runs ≤224).

## Metrics
- **Primary**: `decode_ms` (ms per generated token, lower is better) —
  steady-state incremental decode on prompt A + 20 gen tokens.
- **Secondary**: `prefill_ms` (ms per prompt token, same kernel, reported
  separately), `load1` (Spark 1-min load during the run — contention
  witness, never optimized).

## How to Run
`./.auto/measure.sh` — syncs tree sources to the Spark, builds, times
prefill (21 prompt forwards) and decode (20 gen forwards), prints
`METRIC` lines, saves step logs to `.auto/last_*.txt`.
`./.auto/checks.sh` — accuracy gate, runs automatically after passing runs.

## Prompts (fixed, never tune to them)
- **A (measured)**: seasons en templated, 21 ids:
  `248045 846 198 826 279 2943 15127 321 799 3065 421 4203 303 1754 13 248046 198 248045 74455 198 248068` + 20 greedy gen.
- **B (held-out, checks only)**: haiku templated, 17 ids:
  `248045 846 198 7734 264 6185 36974 883 279 9117 13 248046 198 248045 74455 198 248068` + 10 greedy gen.
- Golden streams in `.auto/golden_A.txt`, `.auto/golden_B.txt` (tops only,
  from the proven engine). Final acceptance additionally requires a FRESH
  third prompt validated against server temp-0 (manual, not in the loop).

## Files in Scope
- `pulsar_arm/bonsai2/fwd.c` — forward, norms, attention, GDN wiring, sampler
- `pulsar_arm/bonsai2/tq_gemv_neon.c` — NEON integer GEMV (the hot loop)
- `pulsar_arm/bonsai2/fwht.S` — FWHT asm
- `pulsar_arm/bonsai2/tq_xgemv.c`, `tq_gemv_int.c` — scalar references (change only with cause)
- `.auto/measure.sh`, `.auto/checks.sh` — may gain instrumentation as needed

## Off Limits
- `pulsar_arm/asm/*`, `pulsar_arm/tools/*` (Pi engine + gemma tooling).
- `pulsar_arm/bonsai2/fwd8.c`, `pq2_gemv.c` (8B line — separate session if ever).
- Fork server, GGUF weights, tokenizer files, Spark `~/bonsai2/asm/*` dev tree
  (loop builds in `~/bonsai2/exp/` from synced tree sources only).
- `.auto/golden_*.txt` (frozen oracles), `.auto/log.jsonl` (append-only).

## Constraints (hard — violations are cheating, discard + document)
1. **Accuracy**: `checks.sh` must pass — exact greedy streams on A and B;
   kernel GEMVs must be bit-exact OR fp32-close (maxrel<1e-6), so fp
   reassociation is allowed but formulation drift is not.
   No precision reduction anywhere without a bit-exact (or argmax-preserving
   + distribution-checked) proof first.
2. **No benchmark gaming**: fixed prompts, fixed gen counts, fixed threads
   (20, taskset 0-19), fixed flags (-O3 -fopenmp -march=armv8.2-a+dotprod+fp16 -mtune=native as of run #232; no -ffast-math ever, fp order preserved).
   No prompt-dependent branches, no answer caching, no skipped layers/norms,
   no EOS-handling changes that alter measured work, no accuracy-for-speed
   trades hidden in the sampler (sampler is out of scope unless bit-identical
   streams preserved).
3. **No overfit**: the primary metric lives on prompt A; B is an unseen
   correctness tripwire; victory lap uses a third prompt + server diff.
   Ideas that only help A-specific shapes (e.g. tuned to 21/20 lengths) are
   discarded even if faster.
4. **Same-job discipline**: one benchmark at a time on the Spark; record
   `load1`; don't run while other heavy jobs hold cores (check `uptime`
   first; idle llama-servers are fine).
5. **Keep the tree green**: builds must be warning-clean-ish (no new
   warnings), binaries stay bit-deterministic threaded-vs-single (row-split
   reductions only — verify on any threading change).

## What's Been Tried
- (this session) scalar-int 139 ms → NEON SDOT 26.0 ms per 89M-weight
  projection (5.3×), BIT-EXACT; OpenMP rows → 0.71 s/token on 20 cores
  (10.7× over single-core, tops bit-identical).
- (prior Pi session, closed): bus wall 3.93 GB/s, OMP +4.2% discarded,
  prefetch +19% discarded, converged — see ideas.md wall notes.
