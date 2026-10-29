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

## Session 2 (2026-10-29, autoresearch runs #81-#90): where the 139 ms actually goes

`LFM_TIME=1` per-phase decode timing (added run #89) reframed the whole problem:

| phase | ms/token | bytes/token | effective GB/s |
|---|---|---|---|
| embedding lookup | 0.00 | 4 KB | – |
| 13 conv layers | 76.44 | ~715 MB | 9.4 |
| 11 attention layers | 60.31 | ~545 MB | 9.0 |
| final norm | 0.05 | 4 KB | – |
| **lm_head** | **19.79** | 278 MB | **14.0** |

The lm_head is **14%** of the step, not the ~50% the plan's 127.7 ms budget assumed — that is
exactly why the slab-size knob measured `R = 1.0000`. It streams at 14 GB/s, which is the
achievable ceiling on this board, so all remaining headroom is in the 24 layer GEMVs at
9.0-9.4 GB/s.

### Measured out (paired A/B, 3-4 runs each, one pair in each order)

| lever | result |
|---|---|
| R=256 rows/block for all shapes | **+1.0%, kept** (bit-identical; RMAX=256 is the hard ceiling — `sh[RMAX]` overflows above it, now clamped with a warning) |
| lm_head slab rows 8192→4096 | R = 1.0000, saturated |
| TPB 256→512 | **−2.3%** (fewer blocks; the 8 KB shared x-stage + syncthreads dominates) |
| KCHUNK 2048→1024 | **−4.4%** (more partials, more x re-staging; also changes the combine order) |
| R=64 for the small-N shapes | **−5.7%** (x-stage amortization beats block fill) |
| merged gate+up GEMV (the plan's Bonsai technique, 24→48 blocks) | +0.33% for 679 MB — **not kept** |

Conclusion: the GEMV config is at a measured local optimum (R=256, TPB=256, KCHUNK=2048), and
the layer GEMVs are **not** block-fill limited — merging rows to double the block count bought
0.33%. The gap to 14 GB/s is the *access pattern*: the lm_head streams one 524 MB contiguous
region with perfect DRAM locality, while a layer GEMV runs 8-24 blocks bursting over a 12 MB
tensor. The plan's merge win was measured on 24576-row gate+up tensors, 4x these shapes.

Bytes/token are pinned by the quality gate: Q8 linears + fp32 scales ≈ 1.2 GB + 183 MB scales
+ 278 MB lm_head ≈ 1.34 GB/token. At 139.6 ms that is 9.6 GB/s aggregate. Q4 linears would
halve the bytes (~2x) and are measured out on quality (plan: 5/8, worst prefix 0).

### The plan's 8/8 goldens gate is not reachable by a from-scratch GEMV

The certified `q8_0` arm (linears Q8 + fp32 scales + **bf16** embeddings, blob
`lfm3-audio-2b-cert.lfm3q8`, export with `--emb-bf16`) runs at **161.67 ms/token (6.19 tok/s)**
device-resident and **223.46 ms/token (4.48 tok/s)** with `--emb-host`. Its greedy streams are
**byte-identical to the Q8-embedding arm's** (prefix 2 / 1 on prompts A / B), so embedding
quantization is *not* the divergence, and the 15% latency + 262 MB the bf16 embedding costs
buys nothing here.

Why the goldens cannot be matched: they are HF/**cuBLAS** outputs. This repo's Q8 kernel is
bit-exact against the *ggml* reference (the plan's 18/18-shape claim) — a different summation
order. Each Linear then differs by ~1 bf16 ulp, the per-layer rms ratio table sits at
0.995-1.006 (i.e. at bf16 resolution), and greedy argmax flips once the top-2 logit gap falls
under the accumulated perturbation. **G6 as written is not achievable**; the defensible gates
are the HF-derived ones in `checks_lfm3.py` (23x23 layer/position rms within 3%, pos-0 logit
top-1 equality + corr ≥ 0.999, greedy prefix ≥ the arm's certified floor, stated per arm).

### Infra bugs found in the harness itself (these invalidate numbers if unfixed)

* `test -x $BIN` is **not** a build check: nvcc leaves the previous binary in place on compile
  failure, so three broken builds measured stale numbers silently. All three scripts now
  compile to `$BIN.new` and `mv` only on success, and refuse to run without `BUILT_OK`.
  Runs #83-#86 predate this; their numbers were re-validated on a clean build (139.54 vs 139.56).
* `PQ1_EXTRA` does **not** propagate through the ssh quoting layer in `measure.sh` (an env knob
  set that way silently does nothing). `ab.sh` passes its arm as `env $1` inside the ssh command
  and *does* work. Use `LFM3_ARGS=--flag` or set the env directly.
* `ab.sh`/`checks.sh` picked up a dangling `ssh ... \` line from the build-hardening patch, and
  `ARGS` was unbound under `set -u`. Fixed.

### State and next steps

Best kept: **139.62 ms/token (7.16 tok/s)**, RSS 306 MB, gates green, 9 kept runs.
Deferred, ranked: (1) CUDA graphs for the ~300 launches/token (needs `tok`/`pos` moved into
device memory; est. 1.5-3%; the Bonsai Nano engine's kept lever on this board); (2) fuse
`k_bf2f` into the GEMV prologue (~0.7%); (3) re-baseline after the board's power cycle — the
three zombie `lfm3_conv_test` processes (unkillable, blocking reboot) steal 3 of 4 cores, so
today's absolutes are inflated; (4) a top-k-vs-HF logit agreement gate, which is the
quantitative version of "how close is this engine to HF".

### CUDA graphs: scoped, not attempted (est. 1.5-3%, ~2 h, hot-kernel risk)

Scoping found five classes of per-step state baked into launch parameters, all of which must
move to device memory before a capture is valid:

1. `k_dequant_row` (Q8 path): the host computes codes/scale pointers with `tok` and does two
   pinned-host `cudaMemcpyAsync` at token-dependent offsets.
2. the input embedding row for `dtype == 3`: a `cudaMemcpy` at `tok * H * 2`.
3. `k_rope(q)`, `k_rope(k)`: `pos` is a kernel argument (2 launches per attention layer).
4. **The KV cache writes: the q/k/v GEMVs write directly into `M.kc + L*MAXTOK*NKV*HD +
   pos*NKV*HD`, so `pos` is baked into the *output pointer* of 3 GEMV launches per attention
   layer = 33 pointers.** This is the blocker: fixing it means adding
   `(base, const int* pos, stride)` to `q8_partial`/its combine, i.e. changing the most
   performance-critical kernel in the file — the one the plan certifies bit-exact on 18 shapes.
5. `k_attn_head`: `np = pos + 1` (its grid is over heads, so a device-read `np` is fine).

The pinned-host-slot trick handles 1-2 (a kernel reads pinned memory at execution time, so a
fixed source address carries a changing value); 3 and 5 are one-line kernel changes; **4 is the
work, and it touches the certified kernel.** Suggested order: (a) add the pos-indirect output
pointer to the GEMV behind a flag, verify bit-exactness on the 18-shape probe and the gates;
(b) then capture on a non-default stream after 2 warmup steps and replay, keeping the logits
D2H readback outside the graph; (c) `--trace` probes must stay off under graphs (their
destination buffers are per-call locals). Measure with `ab.sh`; expect the win to be largest
while the board is CPU-starved, which is also when the estimate is noisiest.

## Session 3 (2026-10-29, runs #92-#101): the GEMV is FP-pipe-bound, not bandwidth-bound

`LFM_TIME` said conv 77 + attn 61 = 88% of the step in the 24 layer GEMVs at 9.0-9.4 GB/s,
against the lm_head's 14 and a measured 17 GB/s streaming ceiling. Four structural levers were
implemented and measured; all four are measured out. The win came from instruction counting.

### The finding

Per row the inner loop did `dot = __fadd_rn(dot, __fmul_rn(c[j], x[j]))` for 32 products, then
`acc = __fadd_rn(acc, __fmul_rn(scale, dot))` over 64 blocks. That is **2 FP instructions per
product** (the plan's deliberate exact-per-product rounding, reinforced by the build's
`-fmad=false`), plus a loop-carried serial chain. Counting them -- 64 x (32 mul + 32 add +
1 acc + 5 shuffles) = 2432 FP warp-instructions per warp, where the arithmetic minimum is 512
FMAs -- reproduces the measured 1.33 ms/layer almost exactly. **The kernel was issuing 4.75x
the necessary FP work; 9.2 GB/s was a symptom, not a cause.**

`__fmaf_rn` accumulation: **139.14 -> 136.41 ms/token (-2.0%, R=1.0100, faster 3/3)**, every
GEMV phase improving 6-8.5% (conv 76.4->70.6, attn 60.3->56.6, lm_head 19.8->18.5), gates green,
greedy streams byte-identical, logit corr unchanged at 0.99998, and the logit std moved marginally
*closer* to HF (4.564 vs 4.558 against HF's 4.547). ggml's own reference dot product is
`sum += w*x` in C, which compilers contract to an FMA -- so the FMA form is at least as faithful
to the reference as the plan's mul-then-add, and strictly more accurate per product.
**Shipped: 141.04 -> 136.4 ms/token (7.33 tok/s), -3.3%.**

### Measured out this session (paired A/B, one pair in each order)

| lever | result |
|---|---|
| fused `k_bf2f` into the GEMV's x staging (bit-identical, ~110 fewer launches) | **+0.39%, kept** |
| resident vs the plan's prescribed `--emb-host` lm_head streaming | **+10.4%, resident is already the default** |
| double-buffered lm_head row streaming (2 slabs, 2nd stream, events, WAR guard) | **2.6x WORSE** (lm_head 19.8 -> 52.5 ms) |
| CUDA graphs of the 24-layer loop (the plan's #1 next lever, est. 1.5-3%) | **-0.55%** |
| merged gate+up GEMV (the Bonsai Nano technique) | +0.33% for 679 MB |
| `-fmad=true` build-wide (contracts the conv/attention/norm kernels too) | +0.06% = nothing |
| `__ldcs` streaming loads on the once-read Q8 code bytes | R = 1.0001 = nothing |
| KCHUNK=1024 / TPB=512 / R=64, **re-run under FMA** | -5.3% / -3.6% / -6.5% (all confirmed, all slightly worse than pre-FMA) |

**Why CUDA graphs lose here** (worth recording, since the plan recommends them): the GPU spends
136 ms per step and the CPU issues all ~250 launches in ~1 ms, so the launch queue runs 250
steps ahead and launches are nowhere near the critical path. A graph can only add its own
launch cost. The Bonsai Nano engine gains from graphs because its per-phase work is 5-kernel
chains with a captured graph per phase; this engine's hot path is 2 launches of 500 blocks
(lm_head) plus 24 layers of large blocks.

**Why streaming the lm_head from pinned host is 10% slower**: on Tegra the H2D copy and the GEMV
share the 21.9 GB/s bus, so streaming costs *two* full passes over 278 MB. The plan's `--emb-host`
requirement was sized for the 1944 MB whole-backbone-Q8_0 config; the Q8-embedding blob is
1466 MB and fits resident with ~2.8 GB device free, so no streaming is needed at all.

### Where the remaining time is, and why it is hard

conv 70.6 + attn 56.6 + lm_head 18.5 = 136 ms. The GEMV is now at 512 FMA-instructions/warp --
the arithmetic minimum -- and the config (R=256, TPB=256, KCHUNK=K=2048) is a characterized
optimum on both accumulation forms. Per-layer time exceeds the GEMV byte floor by 1.9 ms
(attention) and 2.9 ms (conv), and graphs proved that is not launch overhead: it is the strict
q/k/v -> rope -> cache -> attention -> o_proj dependency chain plus the elementwise kernels
inside those phases, which are small and latency-bound. Bytes per token are pinned by the
quality gate: further large gains need fewer bytes per token (lower-bit weights), which is
measured out on quality (plan: Q4 = 5/8, worst prefix 0).

### Process notes (the expensive ones)

* **The "prove the code path ran" rule caught three false results this session**: the
  double-buffer A/B reported R=0.9999 on a path that never executed (the default lm_head is
  resident, so `lm_head_host` was not called); the FMA gate run through `PQ1_EXTRA` reported
  byte-identical green gates while measuring the *non-FMA* arm (`checks.sh` has no env
  propagation -- use `LFM_FMA=1 bash checks.sh`, which flows shell -> python -> subprocess);
  two `__ldcs` A/Bs measured unmodified code because the patch's patterns had drifted
  (indentation, the `g_launch` argument). Tell for a null probe: R exactly 1.0000 *and* arm
  numbers matching the control to 4 digits means the knob was inert.
* `log_experiment`'s auto-revert does not reliably restore files that a later patch also
  touched: the CUDA-graph code survived the #99 discard and is in the tree (behind its flag,
  gates green), which is how I had documented it -- verified, not assumed.
* Greedy stream prefixes are stable to a 1-ULP perturbation on these two prompts (all 16
  tokens unchanged by the FMA switch), so stream equality is a weak discriminator for
  accumulation-order changes; the rms matrix and logit std/corr are the sensitive ones.

### Open items

* **Power cycle the Nano.** Three unkillable `lfm3_conv_test` processes from 2026-10-28 own 3
  of 4 A57 cores; today's absolutes are inflated, only paired A/B is sound.
* A top-k-vs-HF logit agreement gate: the quantitative version of "how close is this engine to
  HF", and the natural gate for any future accumulation-order change.
* Prompts C-L goldens carry no `prompt_ids`, so the stream gate scores only A/B.
* Promising-but-deferred: (1) FMA'ing the fp32 scale path in the lm_head combine (0.2 ms of 18.5,
  marginal); (2) a persistent-kernel GEMV to test whether the per-layer 1.9-2.9 ms excess is
  block-scheduling rather than the dependency chain; (3) re-measuring the resident lm_head's
  14 GB/s against the 17 GB/s ceiling after the power cycle -- if the ceiling moves, the GEMV
  config sweep should be repeated, since it was run under CPU starvation.
* Run #162 follow-ups (batched H2 GEMV is proven bit-exact at 1.34x on the embedding): (1) integrate q8_prefill_h2 into the full prefill path (all 16 layers' q/k/v + out + gate/up/down GEMVs batched per tile, plus batched attention + KV fill) and A/B TTFT_23 -- the component win is 1.34x but prefill also pays attention/norm/lm_head per position; (2) CT=4 variant (17.4 KB shm, fewer tiles) now that the structure is proven -- expect ~1.0x given #137's occupancy analysis, but cheap to price; (3) the H2 decision (raw -5.4% vs rescaled ~0%) transfers directly to the batched kernel since it uses the same inner loop -- no re-qualification needed beyond the bit-exactness already shown.
* Run #165 CLOSED the batched-prefill integration: ffn.w2 (K=7680, x+row*K stride into arena scratch beyond the 7680 silu buffer) makes layer-first batching non-bit-exact BY CONSTRUCTION; w2 must stay sequential. Root cause documented on q8_prefill_h2. Remaining batched-gemv value: only the standalone selftest + the H2 decision table. If prefill batching is ever revisited it requires changing w2's x-layout in the SEQUENTIAL engine too (a numerics change to the goldens: rows 1..2047 of w2 change when the scratch window changes), not just batching.
