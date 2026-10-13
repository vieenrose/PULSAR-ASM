# Autoresearch: Jetson Nano 8B exact-PQ2 — prefill + decode speed, peak RSS

## Objective
Speed up the Nano GPU engine on branch `bonsai-jetson-nano`
(`pulsar_arm/bonsai2/pq2_swar_sm53.cu` + `pq2_fwd8_glue.c` + the PQ2 path):
**prefill and decode**, at frozen accuracy, while **reducing peak RSS**.
Every kept run must keep three gates green (`.auto/checks.sh` enforces them
and they cannot be overridden):
  1. `pq2_parity` BIT-EXACT at K=4096 (256/256) and K=12288 (512/512) vs the
     fork-faithful host reference (mirrors `pq2_gemv.c` /
     `ggml_vec_dot_pq2_0_q8_K` accumulation order).
  2. greedy stream on prompt A (measured) == frozen golden (44 tokens).
  3. greedy stream on held-out prompt B == frozen golden (26 tokens).

Context: 8B PQ2_0 streams 2.18 GB/token, so the board's 25.6 GB/s bus caps
this at 11.6 tok/s at 100% EMC (~7 realistic); the kernel is compute-bound
today, not bus-bound. 13-15 tok/s is above the theoretical bus for this
model size and is not a target.

## Metrics
- **Primary**: `decode_ms_per_tok` (ms per generated token, lower better) —
  steady-state per-step forward time from in-binary timers.
- **Secondary**: `prefill_ms_per_tok` (steady-state prompt steps),
  `peak_rss_mb` (maximum resident set size), `startup_s` (one-time
  split-repack + 2.3GB arena upload), `build_s`.
- Contention witness: GPU clock + `gpu_clk_hz` echoed in the METRIC line
  block. Bus/clock drift is real on this board: judge by ratios from
  same-invocation A/B whenever a change is within ~3%.

## How to run
`bash .auto/measure.sh` — syncs sources to the Nano, builds, runs the fixed
prompt, prints METRIC lines, saves logs to `.auto/last_*.txt`.
`bash .auto/checks.sh` — the three accuracy gates (runs automatically after
each passing benchmark).

## Fixed prompts (never tune to them)
- **A (measured)**: chatml, "List the four seasons and one thing that changes
  in each." — 20 ids + 24 greedy steps.
- **B (held-out, checks only)**: chatml, "Write a haiku about the ocean." —
  16 ids + 10 greedy steps.

## Files in scope
- `pulsar_arm/bonsai2/pq2_swar_sm53.cu` — the sm_53 SWAR GEMV (hot loop)
- `pulsar_arm/bonsai2/pq2_fwd8_glue.c` — repack/cache/arena/launch glue
- `pulsar_arm/bonsai2/pq2_host.c`, `pq2_wrap.h` — repack + host reference
- `pulsar_arm/bonsai2/pq2_parity.c` — the parity gate itself
- `.auto/measure.sh`, `.auto/checks.sh` — harness (may gain instrumentation)

## Known optimization directions (from the perf analysis)
- Occupancy is capped at 3 blocks/SM by the 13KB act tile (register cap
  allows 4). Chunking K would shrink the tile.
- Mask work (split4 + keep4, 11 ops per 4 weights) is *row-independent*: it
  can be computed once for 2+ rows per lane, amortizing it.
- Latency hiding: software-pipeline the next group's code loads.
- Peak RSS: the transient host split buffers (up to ~165MB for the head)
  and the mmap'd GGUF pages are the resident mass; `madvise(MADV_DONTNEED)`
  after each tensor is uploaded drops the latter.

## Record: superseded objective
The previous session on this repo targeted the Spark 27B CPU engine
(`fwd.c` + `tq_gemv_neon.c` + `fwht.S`), primary `decode_ms`, ~0.71 s/token
threaded. Its harness is kept at `.auto/measure_spark_27b.sh` and its
directives/goldens (`golden_A.txt`/`golden_B.txt` were re-frozen for the
Nano prompts above) are in the git history. Do not mix the two.
