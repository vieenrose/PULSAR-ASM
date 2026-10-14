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
| decode_tps | 2.1349 | **6.7476** (+216%) | cold harness #74; board clocks locked (jetson_clocks + c7 off, see §5.2) |
| prefill_tps | 2.2256 | **6.8938** | same code path; no prompt batching |
| peak_rss_mb | 206 | 340 | grew with merges+graphs; arena unchanged |
| startup_s | 2.8 | 2.4 | step 0 = repack + upload + first forward |
| gpu_used / arena | - | 1014.8 / 1280 MB | the arena is NOT to be shrunk (see #23) |

Per-token budget (PQ1_KTIME=1 + arithmetic): **283 ms = 204 ms device + ~79 ms CPU (28%)**
before #69; after #69 (pass-level profiler `PQ1_CPUTIME=1`, ms/tok over decode steps):
total 251.2 = gemv 214.2 (GPU + glue) + attn 13.6 + silu 12.3 + quant 4.7 + rms 2.1 +
headnorm 1.4 + sample 1.1 + resid 0.8 + rope 0.5 + argmax 0.4 + kvcopy 0.1.
HEAD-before-#69 under the same clocks: gemv 217.6, attn 26.7, quant 12.7, silu 11.2, rope 5.9.
After #74 (same profiler, 24 decode steps): total 151.5 = gemv 127.8 (GPU + API) + attn 8.3 +
silu 7.4 + quant 3.8 + rms 2.0 + headnorm 1.4 + sample 1.1 + kvcopy 0.8 + resid 0.7 +
rope 0.5 + argmax 0.4. CPU side is now ~24 ms/token (16%).

| shape | calls/token | ms/call | ms/token | kernel |
|---|---|---|---|---|
| gate+up merged 24576x4096 | 36 | 1.49 (#73) | 54 | **T-MAC** 2 rows/thread, graph |
| down 4096x12288 | 36 | 0.84 (#73) | 30 | **T-MAC** (was r128 1.60) |
| q/k/v merged 6144x4096 | 36 | 0.48 (#73) | 17 | **T-MAC** (was r128 ~0.45-0.86) |
| o 4096x4096 | 36 | 0.35 (#73) | 12.7 | **T-MAC** (was r128 0.67) |
| head 151669x4096 | 1 | 8.7 (#73) | 8.7 | **T-MAC** 2 rows/thread, graph |

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
* **T-MAC on every shape** (#72; default **PQ1_TM=1**, `PQ1_TM=4` is the old gate and
  `0` pure SWAR, for A/B). Two things made the one-row kernel lose on the narrow shapes
  (#60/#62: -0.73% on q/k/v, -3.05% on down): it was LDS-bound (fixed by #71, 2 rows per
  thread, now the fastest kernel at every decode shape), and the tm path built the bit
  planes on the HOST (0.21 ms at K=4096, 0.64 ms at K=12288 per call) and ran five API
  calls while the SWAR path ran one graph. #72 moved the plane/group-sum build into a
  tiny device kernel (`pq1_tm_prep`, K/128 threads) and captures {H2D raw xq, prep, tm,
  D2H} as one graph per tensor, so the tm path now costs what the SWAR path cost. The
  parity gate checks the device-built planes byte-for-byte against the host build and
  runs the tm gate on them. The SWAR r128/r256 kernels no longer run in decode
  (`PQ1GPU ... tm=6380 r128=0 r256=0`).
* **Spin scheduling** (#73): `cudaSetDeviceFlags(cudaDeviceScheduleSpin)` before the
  context is created (first CUDA call = arena allocation). The default heuristic blocks
  on this 4-core board and every one of the ~145 syncs per token paid a wake-up; the
  per-call gap between the engine (PQ1_KTIME=1) and the hot loop fell from ~0.11-0.16 ms
  to ~0.03-0.09 ms. ab_pair 2 reps: 6.0074/6.4092, 6.4160/5.9276 = **+7.46%**, bias 0.993.
  `PQ1_SPIN=0` is the A/B arm. (A first attempt whose patch had not applied measured
  +0.19% on identical binaries: that is the harness's noise floor with locked clocks.)
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
-35% (1 SM, kernels serialise). T-MAC on all shapes with the HOST plane build (#50 -3%, #72 -5.5% even with the 2-row kernel): the host build + 5 API calls per call were the cost, see §3. Hybrid CPU/GPU row split (#75): <=+1.3% unresolved at its best point (8% rows, 2 workers), the two sides starve each other on DRAM; see §5.1.

## 5. Open levers, ranked

1. **CPU passes: now ~24 ms/token (16%), was ~79.** #74 (all bit-exact, logits identical
   on all 44 steps): K cache transposed into 16-position tiles `KTC[l][kh][t/16][i][t%16]`
   so q.k is one vfmaq chain per lane (lane = position, i order unchanged; the first
   attempt with a plain `[i][t]` layout at 2 KB stride THRASHED the 2-way L1 and was
   slower, 17.7 ms); V cache kv-head major `VCA[l][kh][t][128]` so p.v streams a 22 KB
   slab that stays in L1 across the 4 passes and the 4 heads of a kv group; p.v with 8
   NAMED accumulators (gcc 7 -O2 keeps a `float32x4_t a[16]` in memory: 8.8 ms); silu
   with two independent 4-lane chains per iteration; Q8_K quant with a 16-lane amax-index
   probe, uzp1 narrowing and an in-register pairwise bsums tree (12.5 vs 15.8 us/call at
   K=4096, bytes identical). attn 13.6 -> 8.3, silu 9.1 -> 7.4, quant 4.5 -> 3.8.
   OpenMP retried on the head loop and silu (4 threads, static): attn 11.6/9.7 ms and
   silu 9.4/10.5 ms (passive/active) vs 8.3/7.4 serial - still a loss, see below.
   Previously done in #69 (all bit-exact, logits
   identical on every step, goldens exact): rope tables once per position (5.9 -> 0.5),
   expf once per score + 8-way interleaved fmaf q.k chains + 4-lane vfmaq p.v (26.7 ->
   13.6), NEON Q8_K quant with the nearest_int magic-add done per lane (12.7 -> 4.7).
   What is left and how to take it:
   - **silu 12.3 -> 9.0 ms (#70), 7.4 (#74):** glibc 2.27 `expf` re-implemented in 2-lane float64
     NEON (`pq1_expf_neon.h`, proven by exhaustive test over all 4.28e9 non-NaN floats
     with `pq1_expf_test.c`, 0 mismatches). It MUST live in a `-ffp-contract=off` unit
     (pq1_host.c): under default contraction 2 inputs came out 1 ulp off. The remaining
     9 ms is ~30 cycles/element: the table gather crosses NEON<->GP twice per lane
     (vgetq_lane + scalar ldr + vcreate); `vld1q_lane_u64` straight from the table and
     `vdivq_f32` latency are the next targets (worth maybe 3-4 ms = ~1.5%). Moving silu
     to the GPU breaks bit-exactness (different expf).
   - **attn 8.3 ms (#74)** at NPOS<=44: q.k ~3.7 (1.77M fmla/token, near 1 fmla/clk),
     softmax ~1.2 (expf + fdiv per score), p.v ~3.8. What is left is instruction floor
     plus the 32x per-head overhead; the next real step is the head loop on the idle
     cores with a custom spin pool (see OpenMP note).
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
   - **T-MAC kernel knockouts (#74, hot loop 24576x4096, base 1.434 ms):** no weight
     LDG 1.323 (-8%), no plane LDS 1.198 (-16%), both 1.067 (-26%). 1.07 ms IS the POPC
     floor (28.3M POPC/call at 32/clk/SM, 921.6 MHz = 0.96 ms) and the issue floor (~30
     instrs per 32-weight row step), so the kernel sits at ~70% of its instruction
     floor. Latency-hiding variants, all bit-exact, all measured OUT: software-pipelined
     weight prefetch 1.435, full unroll 1.645 (68 B spill), `__launch_bounds__(256,4)`
     at 64 regs 1.481, pipe+lb4 1.429, unroll+lb4 1.574 vs 1.4255 base. LUT, HFMA2 and
     byte-SWAR formulations were analysed and are no better than POPC on sm_53.
   - **Hybrid CPU/GPU row split: MEASURED OUT (#75).** Code kept (`PQ1_HYB=<percent>`,
     `PQ1_HYB_THREADS`, `pq1_hyb_*` in pq1_host.c, bit-exact at every setting), default
     off. `cudaGraphLaunch` blocks the caller until the GPU is done on this platform, so
     the slice must run on workers started before the launch; even so the GPU and CPU
     starve each other on DRAM (8%/3 thr: GPU phase 126 ms on 92% rows vs 121 ms on 100%,
     CPU slice at 4.3 Gw/s vs 9.9 standalone). Best setting 8% on 2 workers: ab_pair
     3 reps +1.27/+1.37/-4.4%, not resolved, +47 MB RSS. Details in ideas.md #75.
3. **POPC-rate probe: DONE (#71, `pulsar_arm/bonsai2/tools/popc_probe.cu`).** POPC runs
   at ~32/clk/SM (quarter rate; popc+iadd chains 31.4 steps/clk, lop+popc+iadd the same).
   But the T-MAC kernel is NOT POPC-bound: knockouts at 24576x4096 (PQ1_TIME hot loop,
   2.27 ms) gave no-weight-LDG -2.5%, POPC->LOP -1.5%, no-plane-LDS **-35%**. It was
   LDS-bound (8 LDS.128 per 128 weights per row). **#71 made it 2 rows/thread** (one
   plane load feeds two rows' code words): tm 2.27 -> 1.43 ms at gate+up, head 13.5 ->
   8.2 ms, engine A/B +14.5/+17.9/+18.1%. **#72 then put T-MAC on every shape** with a
   device-side plane build inside the graph (see §3): A/B vs #71 173.10/222.22,
   171.20/222.26, 170.66/219.75 ms/tok (-22%, +29% tps), cold harness 5.9961 tps.
   Next on this kernel: 4 rows/thread (registers: 40 used + 16 B spill at 2 rows under
   -maxrregcount=48; the cap may need raising for the tm kernel only via
   __launch_bounds__). The GPU is now ~133 ms of the 167 ms token (tm per-call sums above);
   the CPU passes (~35-40 ms: attention 13.6, silu 9, norms, Q8_K quant) are the other
   lever. The r128/r256/swar kernels are now dead code in decode; keep them for the gate.
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
* Useful knobs: `PQ1_TM` (1=all shapes = default since #72, 0=SWAR everywhere, 2=head only, 4=old gate,
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
