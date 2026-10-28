# LFM3-Audio-2B on Jetson Nano (sm_53) — session ledger

Continues `.auto/HANDOFF_LFM3.md` (previous session: Pi CPU path, GEMV kernel tuning,
quantization certification). Metric: `decode_ms_per_tok`, lower is better. Gate: HF-derived
(see `.auto/checks_lfm3.py` docstring). Baseline established run #81: **141.04 ms/token**,
best so far **139.56 ms/token (7.17 tok/s)** at 306 MB RSS.

## BOARD STATE — read this before trusting any timing number

Three `lfm3_conv_test` processes from the 2026-10-28 session (the warp-shuffle conv
deadlock that killed it) are stuck in state `R` inside the CUDA driver, 10+ hours each:

* `kill -9` does not reap them; `taskset`/`renice` do not take.
* `systemctl reboot` is swallowed — systemd's shutdown waits on those tasks, so **the box
  cannot be rebooted from ssh**. Needs a power cycle.
* Consequence: 3 of the 4 A57 cores spin permanently. Absolute ms/token is inflated
  (first cold run read 213 ms/token, steady state 141.0; run-to-run spread 0.15% once warm).
  Same-session paired A/B is still trustworthy (`ab.sh`), which is what runs #82-#84 used.
* Suspected side effect: the certified (bf16-embedding) blob at 2.18 GB device-resident dies
  with `unspecified launch failure` while the 1.95 GB Q8 blob runs. Device reports 1.11 GB
  free after the allocation, so the nvmap backing pool is the suspect, and the three zombie
  contexts are the suspect for that. Re-measure on a clean board before concluding.

## What was wrong with the CUDA path (it had never been run end-to-end)

The engine compiled and printed plausible tok/s while computing the wrong model. Five root
causes, each found by a layer×position rms cross-check against HF, not by reading code:

1. **f32 payloads read through the float\* arena** — `M.df + of` is float\* arithmetic on a
   BYTE offset, so it advanced `4*of`. Every "UPLOAD FAIL … invalid argument" traced here.
2. **The input embedding row was memcpy'd raw bf16** while the Q8 blob stores codes+scales;
   needs `bf16(f32(code)*scale)`. The original code was written for the *certified* arm
   (bf16 embedding) — which is why it looked self-consistent.
3. **The conv branch gated on `x` (the pre-norm residual) instead of the `operator_norm`
   output.** This is why the 7-tap conv "looked rms-dead" at 0.00016 in the previous
   session's notes: gating on x, tap 6's contribution is a small residual, not the
   0.105-rms signal it is on the normalized stream.
4. **Attention's V-weighted sum covered only dimensions 0..31** (indexed by `e = lane`),
   leaving `segred[seg][32..127]` as uninitialized shared memory — 3/4 of every head's
   output. Signature: attention rms 3.25× HF, head 0's first four values exact.
5. **`M.x = M.h1`** (to feed the final embedding_norm) leaked into later steps because `x`
   is a `Model` member, so from step 1 on `operator_norm`'s in-place write destroyed the
   residual. Signature: *first token right, everything after garbage* — the most
   expensive signature to recognize because a single-token smoke test passes.

Plus one shape bug in the harness: **`rope_parameters` is per layer type** (config
`/lfm/rope_parameters`: `full_attention` theta 1e6 + `partial_rotary_factor` 0.5,
`sliding_attention` theta 1e4 full rotation + a 256 sliding window). Applying one variant
to all 11 attention layers is invisible at position 0 (RoPE is the identity at pos 0:
cos 1, sin 0) and wrong everywhere else.

After the fixes: all 23 layers × 23 positions within 3% of HF, pos-0 logits corr 0.99998,
top-1 equal to HF's.

## Method notes worth keeping

* **Cross-check against HF per layer AND per position.** `.auto/lfm3/probe_lfm3_stages.py`
  (HF side) + `--trace --probe-layer N --probe-step P` + the `POS p L n rms` matrix (engine
  side, one run gives the whole grid). This is the Pi engine's
  `tools/perlayer_probe.patch` role; it localizes a divergence to layer+position and is
  what turned four multi-hour hunts into four short ones.
* Beware probe scripts that look right: `B*x` is `(1,H,1)`, so `W[:,k] * xvec` with
  `squeeze(0).transpose(0,1)` broadcasts to an **(H,H) outer product** and
  `.pow(2).mean()` reports mean-of-rank-1, not rms. Use `reshape(-1)`.
* `rms_host`-style probes need an allocated 4 KB staging buffer per tensor — a probe copy
  returning `invalid value` is the probe's bug, not the model's (that cost a cycle blaming
  the conv).
* **Assert every `str.replace` patch, then grep for the new identifier.** Five patches this
  session silently no-op'd on invented anchors, including two (the dtype-3 input row, the
  bf16 lm_head insertion) that each cost a debugging cycle chasing a phantom.
* Per-launch `cudaGetLastError()` (+ optional sync), env-gated, at all 28 launch sites, is
  what distinguishes "this kernel faults" from "the fault surfaced two kernels later".
  Async launch failures point at innocent victims; two hypotheses were built on victims.

## GEMV config: measured local optimum

`pick()` = KCHUNK 2048, TPB 256, R 256-for-N>=1024 (run #82/#85, +1.0% baked).
Env knobs for A/B: `LFM_R`, `LFM_KCHUNK`, `LFM_TPB`, `LFM_ROWS` (via `PQ1_EXTRA="..."`).

* R=256 for all shapes: **+0.5% kept** (paired, 4/4, bit-identical — R only remaps rows→blocks).
* `LM_ROWS` 8192→4096: R = 1.0000, saturated (matches the plan's Q8 slab tuning 2048-8192).
* `LFM_TPB=512`: **2.3% slower** (4/4) — 512-thread blocks halve the block count on a 4-SM
  part; the 8 KB shared x-stage + `__syncthreads` cost more than the extra ILP.
* `LFM_KCHUNK=1024`: **4.4% slower** (4/4) — more partials and more x re-staging. Note it
  also changes the combine summation order, so it is not a bit-only change.
* Load vectorization is **already optimal**: `q8_partial` has each lane read a whole 32-byte
  Q8 block via two `int4`s, i.e. 1024 B per warp-iteration, fully coalesced. The
  "32 B/warp against Maxwell's 128 B transactions" hypothesis is dead.

Bytes/token are pinned by the quality gate: Q8 linears + fp32 scales = 1.2 GB + 183 MB
scales + 278 MB lm_head ≈ 1.34 GB/token ⇒ 9.6 GB/s achieved at 139.6 ms. Q4 linears would
halve that (≈2× faster) and are **measured out on quality** (plan: `q4_0_lmhead_f16act`
5/8, worst prefix 0), so they are not a latency lever here — do not re-try without a
re-certification plan.

## Open / deferred

* **Certified 8/8 arm (the plan's gate) — one localized bug left.** Blob
  `lfm3-audio-2b-cert.lfm3q8` (export with `--emb-bf16`). Resident form (`--lm-selftest`)
  runs bit-exact; full-model mode 0 (2.18 GB) dies with `unspecified launch failure` at
  step 0 with every launch reporting clean under `LFM_DBG_LAUNCH_SYNC`; mode 1
  (`--emb-host`, bf16 pinned-host row slabs) fails in the loader: the codes arena is sized
  1,466,269,696 while the Q8 uploads need 1,471,381,504 — **4,875 KiB unaccounted** between
  the sizing pass and the upload loop in the dtype-3 + emb-host combination.
  `LFM_DBG_SIZE=1` now dumps per-tensor routing (the first dump came back with 0 SIZE
  lines — check that the print is in the loop that actually runs before chasing arithmetic).
* **CUDA graphs** for the decode step (the Bonsai Nano objective's kept lever on this exact
  board, `.auto/HANDOFF_NANO_Q1_0.md`). ~200-300 launches/token; on a 1-core-available board
  the host cost is real. Estimated 1.5-3%. Needs `tok` and `pos` moved into device memory
  (they are kernel args today, and a graph bakes args in): `k_dequant_row`, `k_rope`, the KV
  write index, `k_attn_head`'s `np`. Arithmetic is untouched, so graph mode is bit-identical.
* **Fuse `k_bf2f` (bf16→f32 of x) into the GEMV prologue**: ~120 launches/token, ~0.7%.
* **fp16 scales** would cut 7.5% of the streamed bytes (scales are fp32: 183 MB per 1.2 GB
  of codes). Changes the quantization arm, so it requires a `quant_sensitivity.py` re-run
  before it can be kept — the plan's 8/8 configuration is explicitly "fp32 scales".
* The 26% gap to the streaming ceiling (9.6 GB/s achieved) is unexplained; the plan's own
  Nano ceiling measurement is the place to look for what a single kernel can reach.
* Long-chain attention (ctx > 256) is untested; the sliding-window path (`SLIDE_WIN 256`)
  has only been exercised at ctx ≤ 48.
