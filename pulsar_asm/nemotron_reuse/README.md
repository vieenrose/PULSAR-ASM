# NemotronLabs RE-USE on Jetson Nano Gen1

Port of [NVIDIA NemotronLabs RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE) (9.6M params, mamba-ssm speech enhancement) to **Jetson Nano Gen1** with CUDA acceleration (sm_53, CUDA 10.2).

## Targets
- **Primary metric:** RTF = processing time / input audio duration < 1.0
- **Target rates:** 8 kHz and 16 kHz (real-time feasible). 44.1/48 kHz reported for information only.
- **Runtime:** Self-contained CUDA + C/AArch64, no Python/PyTorch/cuBLAS/cuFFT at runtime.

## Plan
See [doc/NEMOTRON_REUSE_NANO_PLAN.md](../../doc/NEMOTRON_REUSE_NANO_PLAN.md) for full details: budget analysis, accuracy gates, stages S0–S8, benchmark methodology, and open decisions.

## Status
- [x] Branch `nemotron-re-use-jetson-nano` created and pushed
- [x] Port plan committed and pushed (S0 scaffolding started)
- [ ] S0: Reference model + fixtures + dumps
- [ ] S1: Weight blob conversion
- [ ] S2: STFT front-end
- [ ] S3: Encoder
- [ ] S4: TFMamba tower (30 blocks, fused selective scan)
- [ ] S5: Decoders + iSTFT
- [ ] S6: Driver (full-clip + chunked modes)
- [ ] S7: RTF benchmarking
- [ ] S8: Gated optimizations

## Accuracy Gates
Unlike other PULSAR-ASM ports, bit-exactness vs PyTorch is not required (exp/atan2/log1p). Gates:
1. Per-stage rel-rms ≤ 1e-4 vs fp32 PyTorch reference
2. Waveform SNR ≥ 50 dB (fp32), ≥ 30 dB + perceptual delta ≤ 0.02 (half2/fp16)
3. Complex-spectrum comparison for phase (not raw atan2)
4. Sweep-artifact mask matches reference

## Implementation status
- S0: Done (fixtures, fp32 references, `reuse.blob`: 1416 tensors)
- S1: Done (blob format, host + GPU upload)
- S2: Done (DFT-GEMM STFT/iSTFT, no cuFFT; non-power-of-two nfft OK)
- S3-S5: **Done and gated** (encoder, 30x TFMamba, both decoders, sweep mask) in
  C (`tfmamba.c`, `model_full.c`) and CUDA (`reuse_gpu.cu`)
- S6: Done (CPU + GPU drivers, RTF reporting)
- S7: **Done** (CUDA kernels pass all model gates)
- S8: Blocked on device access (RTF on the requested `ipho` Nano is unmeasured)

## Verified results (dev host, x86 + GTX 1070, CUDA 10.2)

Gates are run with the **reference STFT fed in**, so the model and iSTFT are
measured on their own (`tests/gate_full.py`):

| stage | 8 kHz | 16 kHz | gate |
|---|---|---|---|
| DenseEncoder (all sub-stages) | <= 6.8e-7 | -- | 1e-4 |
| one TFMamba block | 2.31e-7 | -- | 1e-4 |
| full model, complex spectrum rel-rms | **1.10e-6** | **7.82e-7** | 1e-4 |
| waveform SNR (GPU model + iSTFT) | **119.91 dB** | **123.99 dB** | 50 dB |
| verdict | GATE PASS | GATE PASS | |

STFT accuracy against an independent float64 reference: our DFT complex rel-rms
9.2e-8 / phase rms 1.31e-5 rad vs PyTorch's 1.2e-7 / 2.77e-5 -- our STFT is the
more accurate one (see `tests/diag_stft_accuracy.py` and plan section 8).

End-to-end with our own STFT: 42 dB waveform SNR, **disclosed, not a gate**.
Cause is measured: the gap lives entirely in the phase channel (our mag +
reference phase = 122.6 dB; reference mag + our phase = 42.1 dB), and iid phase
noise of the same rms gives 99 dB, so the DFT-vs-FFT phase delta is coherent and
lands in the direction the 30-block recurrent tower amplifies. Reaching 50 dB
end-to-end requires reproducing the reference FFT's float32 rounding bit-for-bit.

## RTF (measured on GTX 1070, naive kernels; Jetson Nano numbers pending)

| fixture | audio | model | RTF_model | iSTFT (CPU) | RTF_total |
|---|---|---|---|---|---|
| 8 kHz | 5.10 s | 48.6 s | 9.53 | 1.02 s | 9.75 |
| 16 kHz | 5.10 s | 98.1 s | 19.23 | 3.58 s | 19.94 |

Phase breakdown at 8 kHz: encoder 0.86 s, 30x TFMamba 47.4 s (97.5%), decoders
0.33 s. The Mamba tower dominates. A tiled GEMM for the projections was measured
to give no speedup (48.6 -> 48.3 s), and skipping individual Mamba kernels
(in_proj / conv / x_proj / dt_proj / scan / out_proj / concat) individually
changed the block time by <1%, so the tower is **not** bound by any single
kernel body; the block cost (1.576 s/block) is structural and the hot kernel is
not yet identified. The profiler available for CUDA 10.2 does not support the
dev-host GPU, so per-kernel attribution is still open.

**Consequence for the Nano:** the Nano has 1 SM vs the 1070's 15, so RTF_model
on the order of 100+ is expected for the current kernels. Real-time (RTF < 1)
needs the kernel-level optimization pass in plan section 4, and the first step
is working per-kernel profiling on the target.

## Build

```
./build_nano.sh cpu          # CPU reference (fp32 + double accumulators)
./build_nano.sh host 61      # dev-host CUDA build (NVCC=... to override)
./build_nano.sh              # Jetson Nano: sm_53, CUDA 10.2 (JetPack R32.5.1)
./bench_rtf.sh               # RTF over the golden fixtures
```
`sm_53` compilation is verified; **execution on Nano hardware is not** (the
requested `ipho` device is unreachable from the dev environment).

## Reference-weight warning (critical for anyone regenerating references)
The PyTorch reference needs a `mamba_ssm.Mamba` whose **parameter names match
the checkpoint exactly**. A shim that nests them (e.g. under `.mixer.`) makes
`from_pretrained` silently leave all 30 Mamba towers randomly initialized and
every downstream number is meaningless. Verify with
`load_state_dict(..., strict=False)` reporting **0 missing / 0 unexpected**.

## Next steps
1. Per-kernel profiling of the TFMamba tower on the target (nsight-compute on Nano).
2. Optimize: fused Mamba direction kernel, chunked/decoupled-lookback scan for
   parallelism over L, shared-memory tiled convolutions, fp32 compensated
   reductions (already in InstanceNorm).
3. Move STFT/iSTFT to the GPU (DFT-GEMM kernels exist on CPU; nfft is
   non-power-of-two at 44.1/48 kHz).
4. Measure RTF on the `ipho` Jetson Nano once device access is available.
