# Nano Gen1 bringup — Bonsai-8B binary (pesi@ipho-nano)

Target: Tegra X1 (A57 + Maxwell sm_53), 4GB LPDDR4 shared, 25.6 GB/s ceiling.
Goal: 12–15 tok/s decode, DRAM-bound (baseline llama.cpp Q1_0: 4–6 tok/s).

## Files (this dir)

| file | role |
|---|---|
| `bq1_swar_sm53.cu` | SM53 SWAR GEMV. Replaces `tq_gemv_neon.c:dec8/dec16+vdot` + `pq2_gemv.c:dec64n+vdot` (no SDOT on Maxwell) with `AND + IADD3`. Keeps `fwd8.c` quant semantics + fork-faithful fp order |
| `bq1_wrap.h` / `bq1_cpu.c` | host ref (bit-exact vs GPU), `bq1_quant_row_i8`, `bq1_repack_from_q1_0`, `bq1_repack_from_pq2_0` (sign-projection of the shipped 8B PQ2_0 for roofline runs) |
| `bq1_host_test.c` | parity gate, runs anywhere: `gcc … && /tmp/bq1_host_test` → `PASS` |
| `bq1_ggml_wrap.cpp` | dispatch snippet for `ggml-cuda.cu` |
| `Makefile.sm53` | Nano build: strict `-gencode arch=compute_53,code=sm_53`, no PTX |

Host gate (this machine): PASS — `maxrel 0.00e+00`, `lsb-map PASS`.

## Nano runbook (pesi@ipho-nano, JetPack 4 / CUDA 10.2)

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
git pull  # branch with bq1_* files
make -f pulsar_arm/bonsai2/Makefile.sm53 host_test   # must PASS
make -f pulsar_arm/bonsai2/Makefile.sm53 fwd8_sm53    # CUDA binary
make -f pulsar_arm/bonsai2/Makefile.sm53 check-sass
# expect: IADD3 >0, IMUL/HMMA =0 in gemv, LDG.E.128 >0

# roofline first (no model needed): 1.06GB/tok -> 25.6/1.06 = 24.1 TPS peak
# standalone engine (preferred, no llama overhead):
OMP_NUM_THREADS=4 PQ2_THREADS=4 ./fwd8_sm53 b8.gguf 151644 872 --gen 200 --sample 0.6 0.95 20 7

# llama.cpp path (4GB-safe):
./build/bin/llama-server -m ./models/bonsai-8b-q1_0.gguf \
  --n-gpu-layers 99 -c 1024 --batch-size 1 --ubatch-size 1 \
  --mlock --no-mmap --cache-type-k q8_0 --cache-type-v q8_0 \
  --temp 0.6 --top-p 0.95 --top-k 20 --threads 4 --port 8080
# -c 1024 (not 4096: display + unified RAM OOMs above ~1536)
# --mlock --no-mmap mandatory (kill migration stalls), --batch-size 1 = mul_mat_vec path
```

## Experiment log (fill on Nano)

- [ ] `host_test` PASS on Nano
- [ ] `check-sass` counts
- [ ] `fwd8_sm53` tok/s (4 threads, 200 gen) vs CPU `fwd8` baseline
- [ ] `nsight-compute`: `dram__throughput ~80%`, `sm ~25%` = memory-bound proof
- [ ] greedy 5/5 parity vs fork server temp-0 (existing Bonsai gate)

## Notes

- 8B ships as PQ2_0 (2-bit). `bq1_repack_from_pq2_0` projects sign for 1-bit
  roofline runs; true Q1_0 checkpoint uses `bq1_repack_from_q1_0`.
- BQ1 superblock = 256 weights / 1 fp16 scale (8×u32). 8B/4096: bits 0.5MB +
  scales 8KB per 4096×4096 matrix; full model ~1.06GB/tok streamed.
# BQ1 Nano ledger (pesi@ipho-nano, MAXN, jetson_clocks: GPU 921MHz EMC 1600MHz CPU 1.479GHz)
- host_test (A57): PASS maxrel 0.00e+00, lsb-map PASS
- nvcc sm_53: OK (needed apt cuda-cudart-dev-10-2 + cuda-toolkit-10-2 for headers)
- SASS: IADD3 x12, IMUL/HMMA/DP4A x0 (v1), LDG.E.CI.128 present, 26 regs, 1KB shmem
- v1 1024x4096 kernel: 783us = 0.68 GB/s | parity 1012/1024 exact, maxrel 1.75e-07 CLOSE
- v2 (LUT+dual-acc): 1.06ms = REGRESSED (32 regs, constant-load latency) -> dropped
- v3 (IMUL spread mask): 770us = 0.72 GB/s, parity holds -> IMUL pipe not saturated, mask not the bottleneck
- cpu scalar x4 (A57): 15.11ms 4096x4096 = 0.15 GB/s -> GPU 5x scalar CPU
- 4096x4096: 3.08ms = 0.73 GB/s; 12288x4096: 10.03ms = 0.67 GB/s (linear, compute-bound)
- target needs ~13-16 GB/s. gap = scalar instr/weight (20+/4w) on 128 cores, not clocks/mem/IMUL.
- next: A57 NEON port (SMULL/SADALP+UZP, no SDOT) per k2 f6b8cfe playbook, and/or 8B end-to-end baseline.

## Measured ledger (pesi@ipho-nano, MAXN, GPU 921MHz / EMC 1600MHz / A57 1.479GHz)

- v4 half-warp-row + single-tile + int4 + 16 rows/block: **1.87-1.90 GB/s**
  (4096²: 1.20ms, 12288-wide: 3.53ms), SASS IADD3 x260 / IMUL x0.
  Parity CLOSE (fp order + cancellation; int partials exact).
- v5 popc diet: 1.81 GB/s, no gain -> reverted (kept v4 IADD3 core).
- K=12288 relerr 3.7e-05 root-caused as summation-order under cancellation
  (48 terms ~500 cancelling to ~20), not a bug.
- A57 NEON v2 (vshl-mask + vand + vaddlv, no SDOT, k2-f6b8cfe playbook):
  **BIT-EXACT 1024/1024**, 0.37 GB/s x4 (2.2x over v1 stack-tmp, 2.5x scalar).
- Stack: GPU 1.9 + NEON 0.37 ~= 2.3 GB/s heterogeneous headroom (~2.1 tok/s).
- Physics: 13 tok/s x 1.06 GB/tok = 13.8 GB/s > Nano EMC max 12.8 GB/s.
  13-15 unreachable on this board for 8B/1.06GB; ceiling ~= 8 at 65% EMC.
  Scalar SWAR realistic top ~= 3 GB/s here (~2.8 tok/s).

## Diet attempts (all on Nano, all vs v4r 1.87 GB/s baseline)
- 2-way even/odd-k accumulator split: 1.84 GB/s, wash (-1.6%).
- v6 block-cooperative S_all precompute: 1.65 GB/s, REGRESS (separate pass
  costs more than the fused hsum it removed; +1 sync on critical path).
- Reverted to v4r single fused loop: 1.86 GB/s confirmed.
- Standing: CUDA 1.86 + NEON-v2 0.37 ~= 2.2 GB/s hetero headroom (~2.1 tok/s
  equiv @1.06GB/tok). Scalar SWAR at local optimum; next gains are
  end-to-end (cudaGraph over ~250 GEMVs/tok, GPU||NEON overlap, KV/ctx
  discipline) and the 4B-binary option for the 10 zone (~5.5GB/s = 45% EMC).

## End-to-end (pesi@ipho-nano, Bonsai-8B PQ2_0 sign-projected to BQ1 binary)
- fwd8_nano via -include shim (zero fwd8.c edits) + lazy PQ2->BQ1 repack.
- Landmines fixed: BQ1C[192] vs MAXT-300 bounds smash (deterministic death
  at tensor 193 with wandering signatures); cache-hit re-repack (181/step
  thrash -> 253 total); Tegra fixed-VA mapping table (arena, 1 mapping);
  libcuda late-init tcache interaction (constructor warmup); /tmp ramfs
  wiped on reboot (weights in $HOME survive; resync + rebuild needed).
- E2E: 8B binary decode RUNS, sane logits (179), clean exit, 253 repacks.
- gen30 timed: mixed (big NEON) 98s, pure-CUDA 67s (incl ~45s one-time
  repack) -> decode ~0.73s/tok ~= **1.4 tok/s**; mixed ~0.57 tok/s.
  Effective 1.45GB/s vs 1.9 microbench: launch+attn+quant overhead visible.
- Quality: greedy loops (2987/1007); needs --sample + template pass.
  Sign-projection fidelity vs true Q1 TBD.

## Exact PQ2 path (branch bonsai-jetson-nano) — CORRECTNESS SOLVED

BQ1 sign-projection was the quality killer (logit ~179, single-token loop).
Exact PQ2 with SWAR decomposition: v = c-1 = 2*b1 + (b0-1)
  isum = 2*mpos_b1 + S_pos(b0) - S_all   (int-exact, |isum| <= 32512)
  acc += (ws*db)*isum, reduced in GROUP ORDER in fp32 (ordered shared
  reduce, __fmul_rn/__fadd_rn) -> bit-identical to the serial reference.

**Gate: BIT-EXACT.** `pq2_parity`: 256/256 @ K=4096, 512/512 @ K=12288,
plus 2048-row run; independent fp32 numpy oracle agrees with the host ref.
The host ref mirrors pq2_gemv.c, which the repo verified bit-exact vs
ggml_vec_dot_pq2_0_q8_K.

**End-to-end on the real 8B (`fwd8_exact`, engine's own Q8_K activations):**
logits sane (17-25 vs 179), sampled tokens varied, output is a coherent
Qwen3 reasoning trace on the four-seasons chatml prompt:
  "<think> Okay, the user is asking to list the four seasons and one thing
   that changes in each. Let me start by recalling the four seasons: spring,
   summer, fall, and winter. ..."
Arena 2300MB ok (Tegra fixed-VA: single cudaMalloc); 253 tensors cached.

**Speed (honest):** gen10 54s, gen60 132s -> marginal 1.56 s/token
(~0.64 tok/s), one-time repack ~45s. Compute-bound: ~10.5 instr/weight
(2 planes x split4/keep4/dot4 + hsum4). Next optimization targets
(unsigned-biased activations + per-group precomputed corrections,
LUT for the plane masks) to reach ~4.5 instr/weight.

**Roofline note:** exact 8B PQ2_0 streams 2.18GB/token; at the Nano's
25.6 GB/s peak that is 11.6 tok/s at 100% EMC (~7 at a realistic 65-70%).
13-15 tok/s is above the *theoretical* bus for this model size.
