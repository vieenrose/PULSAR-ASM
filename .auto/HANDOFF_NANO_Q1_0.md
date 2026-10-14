# HANDOFF — Bonsai-8B Q1_0 (true 1-bit) on Jetson Nano, branch `bonsai-jetson-nano`

Written 2026-10-14 at the end of an autoresearch session (68 logged runs).
**Read this first; `.auto/ideas.md` has the append-only history, `.auto/log.jsonl` the raw runs.**

---

## 1. Objective and gates (unchanged, `.auto/prompt.md` is authoritative)

Speed up the Nano GPU engine for **Bonsai-8B-Q1_0.gguf** (true 1-bit, ~1.06 GB/token
streamed per token, 24 tps is the theoretical bus bound) at **frozen accuracy**, while
peaking down RSS. Primary metric `decode_tps` (higher better).

Three gates, enforced by `.auto/checks.sh`, never overridable:
1. `pq1_parity`: **all four kernels** (swar/r128/r256/tm) bit-exact vs the fork-faithful
   host reference at K=4096 (256/256) and K=12288 (512/512).
2. greedy stream on prompt A == frozen golden (44 tokens).
3. greedy stream on held-out prompt B == frozen golden (26 tokens).

## 2. Current state (all committed; tree clean at handoff)

| metric | baseline (#13) | now | notes |
|---|---|---|---|
| decode_tps | 2.1349 | **3.9900** (+87%) | cold harness #69; board clocks locked (jetson_clocks + c7 off, see §5.2) |
| prefill_tps | 2.2256 | **4.1333** | same code path; no prompt batching |
| peak_rss_mb | 206 | 342 | grew with merges+graphs; arena unchanged |
| startup_s | 2.8 | 2.5 | step 0 = repack + upload + first forward |
| gpu_used / arena | - | 1014.8 / 1280 MB | the arena is NOT to be shrunk (see #23) |

Per-token budget (PQ1_KTIME=1 + arithmetic): **283 ms = 204 ms device + ~79 ms CPU (28%)**
before #69; after #69 (pass-level profiler `PQ1_CPUTIME=1`, ms/tok over decode steps):
total 251.2 = gemv 214.2 (GPU + glue) + attn 13.6 + silu 12.3 + quant 4.7 + rms 2.1 +
headnorm 1.4 + sample 1.1 + resid 0.8 + rope 0.5 + argmax 0.4 + kvcopy 0.1.
HEAD-before-#69 under the same clocks: gemv 217.6, attn 26.7, quant 12.7, silu 11.2, rope 5.9.

| shape | calls/token | ms/call | ms/token | kernel |
|---|---|---|---|---|
| gate+up merged 24576x4096 | 36 | 2.97 | 106.8 | **T-MAC** |
| down 4096x12288 | 36 | 1.60 | 57.5 | SWAR r128 |
| q/k/v merged 6144x4096 | 36 | ~0.45 | ~16 | SWAR r128 (inside a graph) |
| o 4096x4096 | 36 | 0.67 | 24.2 | SWAR r128 |
| head 151669x4096 | 1 | 15.4 | 15.4 | **T-MAC** |

## 3. What the engine looks like now (the kept chain)

* **Merged tensors** (#55/#56): q+k+v and gate+up are concatenated at repack time into
  one arena tensor; the **unmodified scalar kernel** runs over the concatenation, then a
  host `memcpy` scatters the staging buffer into the engine's own vectors. Zero
  kernel-side cost, 253 -> 150 launches/token. Cache cap `PQ1_MERGED_CAP` (default 128;
  the engine needs 72 entries, a cap of 64 silently left the last 4 layers unmerged — #64).
* **CUDA graphs** (#57/#58): every GEMV's {H2D, kernel, D2H} is captured once (lazily,
  2nd call with a stable pointer key) and replayed with one `cudaGraphLaunch` + sync.
  **The staging buffers are allocated once at fixed size (256 KB activation, 1 MB output)
  and must never be reallocated** — a graph that bakes in a freed pointer fails silently.
* **Shape-gated T-MAC** (#60/#62): the bit-sliced popcount kernel runs only for
  `rows >= 20000` (gate+up) and `rows >= 100000` (head); `PQ1_TM` modes 0..6 control it.
  Default **PQ1_TM=4**. T-MAC is bit-exact and *wins* on these two shapes; it **loses**
  on 6144 rows (q/k/v, -0.73%), 4096 rows x K=12288 (down, -3.05%) and everything smaller.
  The discriminator is wave fill / block count, NOT loop length (down has the longest K
  and loses). Do not re-litigate without a new mechanism.
* **SWAR kernels** (pq1_swar_sm53.cu): 64-row (`swar`), 128-row (`r128`, 2 lanes/row),
  256-row (`r256`, 1 lane/row). 1-bit weights x **int8 Q8_K** activations, packed
  16-bit-lane accumulator (#35), LOP3-fused byte fold (#40), and a **16-entry shared
  keep-mask table (#67, +8%)** that replaced `((b4*0x00204081)&0x01010101)*255`
  (ptxas expands each constant multiply into 3 XMAD-family ops).
  The r256 kernel appears to *dislike* the table (inferred from #67 vs #62 arithmetic —
  latent, since the default never uses r256 for a big shape).

## 4. What is measured OUT (do not re-walk; each has data in ideas.md/log.jsonl)

Instruction-count cuts as such (#28/#29: -57% ops paid in GLOBAL loads = **-38%**; but
op cuts paid **in ALU** pay ~0.44-0.51% time per 1% — #35 +10%, #38 +12%, #40 +3%, #67 +8%).
LDS bank conflicts (#15) and LDS op count (#30): zero. Occupancy beyond 40 warps (#27):
+0.47% n.s. Unroll x2 (#21): -2.8%. Double-buffered staging (#14): -5.6%. Cache hints
(#24): -0.35%. DRAM request locality (#32): +0.65% n.s. Prefetch (#31): -0.61%. Bigger
blocks (#16: -9.1%; #20 flat at 128 rows). Tail-wave scheduling (#25): -7.65% — and it is
arithmetically impossible to help, since every N is a multiple of 64. Arena shrink (#23):
primary-neutral but **raises** RSS by 79.7 MB (Tegra unified memory). Zero-copy outputs
(#53): **-390%** (cudaHostAlloc memory is CPU-uncached on Tegra). GPU keep-warm (#43):
-35% (1 SM, kernels serialise). T-MAC on all shapes (#50): -3%.

## 5. Open levers, ranked

1. **CPU passes: now ~37 ms/token (15%), was ~79.** Done in #69 (all bit-exact, logits
   identical on every step, goldens exact): rope tables once per position (5.9 -> 0.5),
   expf once per score + 8-way interleaved fmaf q.k chains + 4-lane vfmaq p.v (26.7 ->
   13.6), NEON Q8_K quant with the nearest_int magic-add done per lane (12.7 -> 4.7).
   What is left and how to take it:
   - **silu 12.3 ms** = 36 x 12288 scalar `expf` + divide. The only bit-exact route is
     re-implementing glibc 2.27's `expf` (Szabolcs Nagy table+poly, computed in double)
     in 2-lane float64 NEON, then `vdivq_f32` (IEEE). Est. gain 5-8 ms. Moving silu to
     the GPU breaks bit-exactness (different expf).
   - **attn 13.6 ms** at NPOS<=44: dominated by the p.v pass (128-wide, memory-bound
     on VCA) and the 8-chain q.k; a further 2x would need K/V in a head-major layout
     (one contiguous 128-float row per (t, kh) already, so little left).
   - **OpenMP is a measured LOSS here, do not retry naively:** with passive waits the
     ~390 us worker wake-up after each GPU wait made quant 12.7 -> 55 ms; with
     `OMP_WAIT_POLICY=active` / `GOMP_SPINCOUNT` the CPU passes drop (quant 7, attn 9,
     silu 4) but the gemv bucket rises 217 -> ~270 ms (spinning workers slow the CUDA
     sync path; GPU clock stayed 921.6 MHz, cause not isolated). Net: 3.28-3.31 tps vs
     3.57 single-threaded. A custom pool that spins only inside CPU phases and parks
     (futex) before each GPU wait is the untested variant.
2. **Board clocks (operator, needs root, NON-PERSISTENT across reboot):** governor was
   already `performance` @1.479 GHz. In #69 the board was put under `sudo jetson_clocks`
   (EMC 1.6 GHz, GPU 921.6 MHz) and cpuidle c7 disabled on all 4 CPUs
   (`echo 1 | sudo tee /sys/devices/system/cpu/cpu*/cpuidle/state1/disable`). Effect on
   HEAD-before-#69 code: 280 -> ~273 ms/tok (~2%). Re-apply after a reboot before
   comparing against #69's numbers; the #69 A/B (3 pairs, HEAD vs NEON, same clocks):
   275.1/269.5/275.4 vs 246.4/246.6/247.1 ms/tok = +10.5%, no overlap.
3. **POPC-rate probe** (15 lines, doc §6.1 of the research notes): settles whether POPC
   is 1/4 or 1/8 rate on this die; decides how far the bit-sliced family can be pushed.
4. **Token-tiled prefill** (M=16/64 GEMM): our prefill = decode speed per token, so the
   20-token prompt costs ~20 decode steps. Batching would make prompts nearly free and
   shorten every harness run, but it does **not** move `decode_tps`. Large task; needs
   its own parity proof (per-token dot order must be unchanged).
5. **Binary-activation engine = a NEW objective, not a tuning step.** The research notes
   assume sign-binarized activations (1 LOP3 + 1 POPC per 32 weights => DRAM-bound at
   ~44 ms/token, ~15-20 tps). Our engine mirrors ggml Q1_0 semantics (int8 Q8_K
   activations), which is what the frozen goldens encode. Going binary would require
   re-deriving the accuracy gates from the true model (MLX/paper) semantics first.
6. r256 + keep-table (latent): if r256 is ever re-enabled for a big shape, revert the
   table in that kernel only.

## 6. How to run it

```bash
# from /home/louis/PULSAR-ASM
bash .auto/measure.sh      # sync -> build -> run fixed prompt -> METRIC lines
bash .auto/checks.sh       # the three gates (builds from the working tree first)
```

* Board: `pesi@100.70.245.36`, engine dir `/home/pesi/exp`, model `/home/pesi/b8q1.gguf`.
* Build recipes: `.auto/build_q1.sh` (canonical; checks.sh runs it as step 0),
  `.auto/measure.sh` (same recipe + run). Kernel TU is built with
  `nvcc -gencode arch=compute_53,code=sm_53 -O3 -maxrregcount=48 --use_fast_math
  -Xptxas -O3,-dlcm=ca`; host TUs with `gcc -O2 -fopenmp -mcpu=cortex-a57`;
  **links need `-fopenmp`** (pq1_host.o uses OpenMP now).
* Engine env vars reach the remote only through **`PQ1_EXTRA="VAR=1 VAR2=0"`** —
  a bare `VAR=1 bash .auto/measure.sh` does NOT propagate.
* Useful knobs: `PQ1_TM` (0=SWAR everywhere, 1=all shapes, 2=head only, 4=default,
  5=+long-K, 6=+all merged), `PQ1_GRAPH=0` (no graphs), `PQ1_MERGED_CAP`, `PQ1_SWAR=1`
  (force the 64-row kernel), `PQ1_R128`, `PQ1_KTIME=1|2|3` (per-shape / per-phase / API
  timing), `PQ1_TRACE=1` (per-call trace + the PQ1GPU counter line).

## 7. Measurement discipline (learned the hard way — follow it)

* **Never compare a run against a number from another hour.** The board drifts 2-6%.
  Use an **order-balanced interleaved A/B in one job**: `.auto/ab_pair.sh "<ENV>" N`
  (engine, env-switched) — the second arm carries a ~1.2-1.6% penalty, so one pair in
  each order and `R = sqrt(o1/o2)`.
* `.auto/ab_kernel.sh` / `ab_kernel_pair.sh` (kernel-TU A/B) **must not be given
  `GITREF=HEAD`**: they append `.auto/ab_delegates.inc`, which redefines the
  r128/r256/tm entry points that HEAD already has, the link fails, and the script
  silently measures a **stale binary** left by an earlier audit (this produced a
  meaningless "+62%" in this session). Only valid for pre-#42 refs.
* A single cold harness sample is worth ~±1%; the dashboard's "+X%" mixes cold and warm
  conditions — quote the order-balanced number.
* **Prove the code path ran** before believing a timing (grep the `PQ1GPU` counter line;
  #42's in-engine T-MAC number was void because a `-DPQ1_REAL_CUDA` omission made the
  engine run SWAR in both arms, and #61's was void because a new branch returned before
  the scatter that delivers merged outputs).
* `checks.sh` needs `measure.sh` to have run first (gate 2 diffs `.auto/last_run_log.txt`).
* Anything that fails the gates is auto-reverted by the tooling: never "keep" a gate
  failure, and treat a gate failure as information (the parity gate caught a wrong
  one-multiply algebra in seconds in #66; the goldens caught the glue bug in #61).

## 8. File map (what to touch)

| file | role |
|---|---|
| `pulsar_arm/bonsai2/pq1_swar_sm53.cu` | the 4 GEMV kernels (swar/r128/r256/tm/batch) |
| `pulsar_arm/bonsai2/pq1_fwd8_glue.c` | repack/arena/merges/graphs/T-MAC gate/timers |
| `pulsar_arm/bonsai2/fwd8.c` | the engine (layer loop, attention, sampler) — where the CPU 28% lives |
| `pulsar_arm/bonsai2/pq1_host.c` | split repack (OpenMP) + host reference |
| `pulsar_arm/bonsai2/pq1_parity.c` | the parity gate (also takes `<rows> <K>` to probe any shape) |
| `.auto/*.sh` | harness, gates, A/B tools |
| `bonsai8b-jetson-nano-research-notes.md` | the reference study reviewed at handoff (see ideas.md) |
| `.auto/wip_glue_merged_tm.c` | OBSOLETE WIP from #61 — the fix is committed; delete freely |
