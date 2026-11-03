# Autoresearch: RE-USE GPU port RTF (Jetson Nano target)

## Objective
Reduce the runtime of the self-contained C/CUDA port of NVIDIA NemotronLabs
RE-USE (speech enhancement: DFT STFT -> DenseEncoder -> 30x bidirectional
TFMamba -> mag/phase decoders -> atan2 head -> sweep mask -> DFT iSTFT)
**while preserving numerical accuracy** against the PyTorch reference.

Target hardware is a Jetson Nano Gen1 (tegra210, **1 SM / 128 cores @ 921 MHz**,
CUDA 10.2, `sm_53`, 3 GiB shared memory). The optimization loop runs on a dev
host (GTX 1070, 15 SMs, CUDA 10.2, `sm_61`) because it is ~23x faster; the
Nano is validated by periodic manual checks (see "Nano validation").

Current dev-host state (8 kHz fixture, 5.100 s audio):
`rtf_total ~= 1.93` = model 1.91 + STFT 0.016 + iSTFT 0.002.
Phase split: tower 8.55 s (97%), encoder 0.85 s, decoders 0.33 s, STFT 0.08 s,
iSTFT 0.010 s. On the Nano: RTF_model 44.34 (8 kHz) / 88.86 (16 kHz).

## Metrics
- **Primary**: `rtf_total` (ratio, **lower is better**) = (model best + STFT +
  iSTFT) wall time / audio duration, 8 kHz fixture, full pipeline with **no**
  reference fed into the model.
- **Secondary** (monitors, not scored): `rtf_model`, `rtf_stft`, `rtf_istft`,
  `enc_s`, `tower_s`, `dec_s`, `heads_s`, and three **held-out shapes**:
  `rtf_held8k` (3.90 s clip, 8 kHz), `rtf_held12k` (3.30 s, 12 kHz, nfft 480),
  `rtf_held16k` (4.40 s, 16 kHz). These exist to catch fixture-specific tuning:
  a real optimization improves them too. If `rtf_total` improves while the
  held-out RTFs do not, treat it as overfitting and discard.

## How to run
- `./.auto/measure.sh` (~66 s) — builds, runs the benchmark + held-outs, emits
  `METRIC name=value` lines.
- `./.auto/checks.sh` (~32 s) — accuracy gates + `sm_53` compile. Runs
  automatically after a passing benchmark; a failure blocks `keep`.

## Files in scope
- `pulsar_asm/nemotron_reuse/engine/reuse_gpu.cu` — all GPU kernels + the model
  graph + the GPU iSTFT. This is where nearly all work happens.
- `pulsar_asm/nemotron_reuse/engine/reuse_gpu.h` — GPU API.
- `pulsar_asm/nemotron_reuse/engine/gemm_tiled.cuh` — tiled GEMM helpers.
- `pulsar_asm/nemotron_reuse/engine/stft_dft.cu`, `istft_dft.cu` — CPU DFT-GEMM
  STFT/iSTFT (the STFT is still on the CPU at 0.081 s; a GPU STFT is fair game).
- `pulsar_asm/nemotron_reuse/engine/blob.c`/`blob.h` — weight blob loader.
  Changing the on-disk/in-memory **layout** of weights (e.g. pre-transposed for
  coalesced B-tile loads) is a legitimate optimization; regenerate with
  `python3 pulsar_asm/nemotron_reuse/tools/mkblob.py` if the format changes.
- `pulsar_asm/nemotron_reuse/engine/reuse_host_gpu.c` — driver. **Timing
  protocol off limits** (see below); you may add instrumentation.
- `pulsar_asm/nemotron_reuse/tests/gpu_*.c` — block/encoder/decoder gate
  harnesses, `tests/gpu_block_bench.c` for single-block timing. Use them to
  iterate quickly (a single TFMamba block gate runs in ~2 s).

## Off limits
- `pulsar_asm/nemotron_reuse/tests/golden/**` (PyTorch references),
  `tests/gate_*.py`, `ref/**`, `.auto/fixtures/**`.
- The **measurement protocol** in `reuse_host_gpu.c`: what is timed, the
  reps/warm-up rule (rep 0 is warm-up, best of the rest), the RTF formulas, and
  the `REUSE_MAG`/`REUSE_PHA` reference-STFT-feeding path. Making the benchmark
  look faster by changing any of these is cheating.
- Gate thresholds in `.auto/checks.sh` and `.auto/measure.sh`. Instrumentation
  may be added; thresholds may never be relaxed.
- The CPU reference model used for cross-checks: `engine/model_full.c`,
  `engine/tfmamba.c`, `engine/conv2d_full.c`, `engine/instancenorm.c`.

## Constraints
1. Gates must PASS at **both** 8 kHz and 16 kHz: model complex-spectrum
   rel-rms <= 1e-4 and waveform SNR >= 50 dB (reference STFT fed).
2. `sm_53` with CUDA 10.2 must keep compiling (checks.sh). The Nano is the
   deployment target, so prefer work that fits **Maxwell limits**: 64 KB
   shared memory per SM, <= 48 KB/block, 24 blocks/SM, no `mma`/tensor-core
   paths, fp64 at 1/64 rate.
3. No cuBLAS / cuFFT / cuDNN / new dependencies / network. Self-contained
   kernels only (that is the project premise: no Python, no BLAS at runtime).
4. Keep fp32 storage and the reference's accumulation semantics: places where
   the PyTorch/CPU reference accumulates in double (InstanceNorm sums over
   ~169k terms, the selective scan, the inverse DFT) must stay accurate — use
   double (CPU/dev host) or compensated fp32 (Kahan / two-float) where Nano
   fp64 throughput is prohibitive, and verify the gate.
5. No fixture-specific behaviour: no branching on T/F/frame counts, no caching
   outputs, no skipping frames/blocks, no early exit.
6. Keep the naive reference paths behind env switches where they exist
   (`REUSE_NAIVE_OP=1`) so A/B is possible.

## Anti-cheat / anti-overfit rules (checked every iteration)
- Primary metric is the **full pipeline**; never feed the reference STFT in the
  measured run (measure.sh unsets `REUSE_MAG`/`REUSE_PHA` for a reason).
- All three phases are reported; a "win" that moves cost from the model into
  STFT/iSTFT shows up as `rtf_stft`/`rtf_istft` rising and `rtf_total` not
  improving.
- Held-out clips at 3 unseen shapes must not regress materially (they are the
  overfit detector; ~1-2% noise is normal).
- Accuracy is a hard gate, not a tradeoff: no precision loss, no truncation of
  the 30 blocks, no reduced frame/binary counts, no looser epsilon in exp/softplus.
- `git diff` must show real kernel work, not measurement edits.

## What's Been Tried
Wins so far (all committed on `nemotron-re-use-jetson-nano`, which this branch
started from):
- **out_proj was the hotspot**: `k_gate_outproj` (naive SGEMV, 256 threads,
  shared reduce) cost 0.31-0.33 s per scan direction = ~81% of a TFMamba block.
  A 32x32 shared-memory tiled kernel with the elementwise gate fused into the
  A-tile load (`out[m][o] = sum_k y[m][k]*silu(z[m][k])*W[o][k]`, output layout
  `[SL,HID]`, gate = **first** DINNER columns of z, applied **inside** the
  K-reduction) runs in 0.00060 s (~590x). Tower 47.5 s -> 8.6 s, RTF 9.53 ->
  1.91. Beware: an earlier version applied `z` post-hoc as `[HID,SL]` and gave
  rel-rms 2.57 — the gate caught it.
- **GPU iSTFT** as two real GEMMs against a host-built float64 DFT table
  (m_k = 1 at DC/Nyquist, 2 for interior bins), double accumulation, atomicAdd
  overlap-add, `sum(w^2)` normalisation: 1.27 s -> 0.0103 s (8 kHz).
- Per-kernel timings per scan direction (S=81, L=256, dev host, sync-based
  `REUSE_KPROF=1`; the concat/resid numbers are *windows* that include the
  un-instrumented output_proj GEMV + LayerNorm):
  in_proj 0.0332 s, x_proj 0.0214 s, conv 0.0064 s, dt_proj 0.0048 s,
  scan 0.0011 s, out_proj 0.0006 s, concat window 0.138 s, resid windows
  0.141/0.147 s.
- `in_proj` runs at ~163 GFLOPS (5.4 GFlop per launch) — i.e. a plain tiled GEMM
  is not enough; it needs register blocking (>=4 accumulators/thread) plus
  double buffering. Measured earlier: a 32x32 tiled GEMM without register
  blocking gave **no** speedup over the naive kernel on the projections.
- Nano: the async launch path dies with a Tegra poisoned-page MMU fault, so the
  Nano runs `CUDA_LAUNCH_BLOCKING=1`. Consequence: **kernel launch count matters
  much more on the Nano than on the dev host**; fusion-driven wins that do not
  move the dev-host metric may still be the highest-value Nano work. Record
  launch counts in ASI when a change reduces them.

## Nano validation (periodic, not per iteration)
`ssh pesi@100.70.245.36` (hostname `iPhO`), repo at `~/reuse_port`:
`rsync -a --exclude '*.o' --exclude 'reuse_nano_gpu' engine/ pesi@100.70.245.36:~/reuse_port/engine/`
then `cd ~/reuse_port && NVCC=/usr/local/cuda-10.2/bin/nvcc ./build_nano.sh`
and `./run_bench.sh` (it exports `CUDA_LAUNCH_BLOCKING=1 REUSE_GPU_ISTFT=1 REUSE_CPU_STFT=1` and
runs both rates with the fed-reference protocol; always drive the Nano through this script - manual
invocations without `REUSE_GPU_ISTFT` fall back to the CPU iSTFT and produced a bogus 16 kHz RTF of
11.5 in an earlier note). Current: 8 kHz RTF_total 3.1500 (model 3.1326 + istft(gpu) 0.0174),
16 kHz RTF_total 6.2304 (model 6.1615 + istft(gpu) 0.0689); cumulative 44.34 -> 3.15 = 14.1x.
Only ~3 GiB device memory: the encoder arena peak is ~560 MiB at 8 kHz, ~1.1 GiB
at 16 kHz. Disk is tight (~450 MB). Do not stop services on it.
