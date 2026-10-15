# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Dependency-free LLM inference engines whose definition of done is **bit-exactness**: same math, same order, same bits as the reference (HF transformers, or the PrismML llama.cpp fork for Bonsai). Speed is judged by achieved bandwidth vs a measured streaming ceiling, never by absolutes. Three engine families coexist and must not be mixed:

| tree | target | language | status |
|---|---|---|---|
| `pulsar_arm/asm/core.S` | Raspberry Pi 4, Gemma-3 270m/1B/FunctionGemma | pure AArch64 asm, `as` + `ld -static`, svc syscalls only | the "vision" deliverable; `pulsar_arm/kernels`, `runtime` are its C/Python parity oracle, NOT shipped |
| `pulsar_arm/bonsai2/` | Bonsai ternary/1-bit checkpoints (8B, 27B) | C + NEON + OpenMP (Spark CPU), C + CUDA sm_53 (Jetson Nano) | active work; this branch (`bonsai-jetson-nano`) is the Nano GPU path |
| `pulsar_asm/` | x86-64 Gemma-2B / Gemma-4 (AVX2 + F16C, fasm flat binaries driven by Python via ctypes) | flat x86 asm | original lineage, dormant on this branch |

Standing directives from `.auto/vision.md`: no C/Python in the asm inference path, no mocks, no edited demo text, disclose divergences with root causes, decisions via same-job A/B only.

## Current objective (branch `bonsai-jetson-nano`)

Speed up **Bonsai-8B Q1_0** (true 1-bit, ~1.06 GB streamed per token) decode on a Jetson Nano Gen1 (Tegra X1: 4x A57 + Maxwell sm_53, 4 GB unified, 25.6 GB/s bus) at frozen accuracy. Primary metric `decode_tps`. Read `.auto/HANDOFF_NANO_Q1_0.md` first: it lists the kept chain, everything measured out (do not re-walk those), and the ranked open levers. `.auto/ideas.md` is the append-only history, `.auto/log.jsonl` the raw run log. `.auto/prompt.md` still describes the older PQ2 objective in places; the handoff and `checks.sh`/`measure.sh` are authoritative for Q1_0.

Files in scope for this objective:

- `pulsar_arm/bonsai2/pq1_swar_sm53.cu`: the four GEMV kernels (`swar` 64-row, `r128`, `r256`, `tm` T-MAC bit-sliced) plus the batch variant.
- `pulsar_arm/bonsai2/pq1_fwd8_glue.c`: repack, GPU arena, merged q+k+v / gate+up tensors, CUDA graph capture, shape-gated T-MAC dispatch, timers.
- `pulsar_arm/bonsai2/fwd8.c`: the engine (layer loop, attention, Q8_K quant, sampler). Compiled unmodified with `-include pq1_fwd8_shim.h`, which `#define`s `pq2_gemv` to the CUDA compat entry. The CPU side is ~28% of decode time and is the top open lever.
- `pulsar_arm/bonsai2/pq1_host.c`, `pq1_wrap.h`: split repack (OpenMP) and the fork-faithful host reference the kernels must match bit-for-bit.
- `pulsar_arm/bonsai2/pq1_parity.c`: the parity gate; `./pq1_parity <rows> <K> all` probes any shape against all kernels.

The `bq1_*` and `pq2_*` files are earlier Nano paths (sign-projected binary, exact 2-bit) kept for reference; `pq1_*` is live.

## Commands (Nano objective)

Everything builds and runs **on the Nano** over ssh (`pesi@100.70.245.36`, engine dir `/home/pesi/exp`, model `/home/pesi/b8q1.gguf`). The scripts scp the seven `pq1_*`/`fwd8.c` sources, build with `nvcc` (CUDA 10.2, `-gencode arch=compute_53,code=sm_53 -O3 -maxrregcount=48 --use_fast_math -Xptxas -O3,-dlcm=ca`) and `gcc -O2 -fopenmp -mcpu=cortex-a57`, then run. Links need `-fopenmp`.

```bash
bash .auto/measure.sh          # sync -> build -> fixed prompt A (20 ids + 24 greedy) -> METRIC lines, logs in .auto/last_*.txt
bash .auto/checks.sh           # the three accuracy gates; rebuilds from the working tree first (build_q1.sh), needs a prior measure.sh
bash .auto/ab_pair.sh "PQ1_TM=0" 2   # order-balanced A/B of an env switch, N reps, prints R = sqrt(o1/o2)
bash .auto/ab_env.sh PQ1_R128 3      # simpler interleaved A/B (fixed order; ~1.6% second-arm bias, use ab_pair for <2% effects)
```

Engine env vars reach the remote **only** through `PQ1_EXTRA="VAR=1 VAR2=0" bash .auto/measure.sh`; a bare `VAR=1 bash .auto/measure.sh` does nothing. Knobs: `PQ1_KVTQ` (K/V cache: **4 = TurboQuant 4-bit, the engine default**, 3, 43 = 4-bit K + 3-bit V, 0 = exact f16; every `.auto` script pins 0, so `PQ1_EXTRA="PQ1_KVTQ=4"` measures the shipped default), `PQ1_CTX` (run context cap, default 4096, up to the compile-time 16384; above 4096 only with a TurboQuant cache, the f16 cache is refused), `PQ1_TQV` (TQ long-chain kernel variants, bit mask, default 1 = pv2; 0 = the old kernels, bit-identical, `pq1_parity tq` compares them), `PQ1_TQSHORT=0` (TQ: force the 5-kernel chain at ctx <= 256 instead of the bit-identical qks+tail pair; `pq1_parity tq 3|4|43`, `PQ1_TQ_CTX=64` checks both), `PQ1_TM` (0 SWAR everywhere, 4 default shape gate, 1 all shapes), `PQ1_GRAPH=0`, `PQ1_MERGED_CAP`, `PQ1_SWAR=1`, `PQ1_R128`, `PQ1_KTIME=1|2|3` (per-shape / per-phase / API timing), `PQ1_TRACE=1`, `PQ1_STATS=1` (GPU memory line `PQ1GPU ...` at exit).

`build_q1.sh` and the inline recipe in `measure.sh` are byte-identical by design; change compiler flags in both.

The three gates, never overridable: (1) `pq1_parity` all four kernels bit-exact at K=4096 (256/256) and K=12288 (512/512); (2) greedy stream on prompt A equals `.auto/golden_q1_A.txt` (44 tokens); (3) held-out prompt B equals `.auto/golden_q1_B.txt` (26 tokens). Gates 2-3 run with the exact f16 cache (`PQ1_KVTQ=0`). `checks.sh` also checks the lossy TQ4 default against `.auto/golden_q1_{A,B}_tq4.txt`; regenerate those only with a recorded reason. A gate failure is information, not something to work around; the tooling auto-reverts it.

## Measurement discipline (the expensive lessons)

- The board drifts 2-6% between hours; never compare against a number from another run. Use `ab_pair.sh` (one pair in each order). A single cold harness sample is worth about ±1%.
- Prove the code path ran before believing a timing: check the `PQ1_KTIME=1` per-shape table or the `PQ1GPU` counter line. Two sessions lost results to a kernel silently running the old path (a missing `-DPQ1_REAL_CUDA`, and a branch that returned before the merged-output scatter).
- `ab_kernel.sh` / `ab_kernel_pair.sh` must not be given `GITREF=HEAD`: they append `ab_delegates.inc`, the link fails, and the script measures a stale binary. Valid only for pre-#42 refs.
- CUDA-graph staging buffers (256 KB activation, 1 MB output) are allocated once and must never be reallocated; a graph that baked in a freed pointer fails silently.
- The 1280 MB GPU arena is not to be shrunk (raises RSS on Tegra unified memory). Growing it lowers max RSS ~1 MB per MB (1792 MB: 214 MB RSS) without saving anything: MemAvailable is unchanged, anon pages just move to nvmap accounting. Compare memory only at equal arena size, or via MemAvailable. Zero-copy host outputs are catastrophic on Tegra (CPU-uncached).
- T-MAC wins only on gate+up (24576 rows) and the head (151669 rows); it loses on everything smaller. The discriminator is wave fill, not K length.
- Any NEON or reordering rewrite of the CPU passes (attention, rmsnorm, Q8_K quant) must preserve IEEE evaluation order and tie-breaks or the goldens break.

## Other engines: build and run

**Pi asm engine** (`pulsar_arm/asm`): `as -o core.o core.S && ld -static -o core core.o`; vocab/BPE blobs from `pulsar_arm/tools/mkvocab.py` and `mkbpe.py`; usage `./core <model.safetensors> <vocab.bin> [tok] [mode 0|1|c|f] [bpe.bin] [temp_milli] [topp_milli] [gencap]`. Parity tests in `pulsar_arm/tests/test_arm_*.py` run on the Pi (need torch + HF cache); `tools/fwd_ref.py` is the fp64 per-layer oracle, pair it with `tools/perlayer_probe.patch` to localize a divergence to layer+position.

**Spark CPU Bonsai** (`pulsar_arm/bonsai2`, needs ARMv8.2+dotprod and the fork's `libggml-base`/`libggml-cpu` for quant helpers and oracles): build lines are in `pulsar_arm/bonsai2/README.md`; `tokdetok` converts prompt text to ids and back; `--sample temp top_p top_k seed min_p` follows the fork chain order (top-k, top-p, min-p on raw logits, temperature last). Old 27B harness: `.auto/measure_spark_27b.sh`.

**Nano Makefile** (`pulsar_arm/bonsai2/Makefile.sm53`): the earlier `bq1` sign-projected path; `make -f pulsar_arm/bonsai2/Makefile.sm53 host_test` runs on any host, `fwd8_sm53` and `check-sass` on the Nano. `NANO_BRINGUP.md` is the chronological ledger of the Nano effort.

**x86 lineage** (`pulsar_asm`): `fasm` flat `.asm` sources assembled to `.bin` blobs (gitignored, rebuilt by the tests) loaded into executable mmap by `runtime/pulsar_abi.py`; tests are plain scripts, e.g. `python pulsar_asm/tests/test_gemma4_engine.py --weights <blob>`.

## Conventions

- Commit messages are long, data-bearing records: what changed, the A/B numbers in both orders, the cold-harness numbers, and that all three gates were green on a fresh build. Keep that format.
- Performance claims in docs cite the measurement conditions (clocks, threads, cold vs warm, same-job ratio).
- `.gitignore` drops `*.bin`, `.auto/log.jsonl` and `.auto/last_*.txt`; goldens and handoffs in `.auto/` are committed.
