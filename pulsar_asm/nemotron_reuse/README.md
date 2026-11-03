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
- [x] Port plan (`doc/NEMOTRON_REUSE_NANO_PLAN.md`)
- [x] S0: fixtures + fp32 PyTorch references (4 rates) + stage dumps
- [x] S1: weight blob (`reuse.blob`, 1416 tensors)
- [x] S2: STFT / iSTFT, DFT-GEMM, no cuFFT, non-power-of-two nfft
- [x] S3: DenseEncoder (gated <= 6.8e-7)
- [x] S4: 30x TFMamba - conv1d, selective scan, bidirectional, LayerNorm
- [x] S5: magnitude + phase decoders, atan2 head, sweep mask, iSTFT
- [x] S6: drivers (CPU `reuse_host.c`, GPU `reuse_host_gpu.c`) + RTF reporting
- [x] S7: CUDA kernels; **verified on the Jetson Nano itself** (sm_53):
      complex rel-rms 1.104e-06, waveform SNR 119.91 dB, gate PASS
- [x] S8 (partial): out_proj tiled+fused gate -> tower 47.5 s -> 8.6 s on the
      dev host; GPU iSTFT (123x); Nano RTF_model 44.34 (8 kHz) / 88.86 (16 kHz)
- [ ] S8 (open): real-time (RTF < 1) is NOT achieved (~45x short at 8 kHz);
      the profile says the 45x has to come from the tower (74%) and the
      encoder convolutions (19%). See next steps.

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

## RTF (RTF = processing wall time / audio duration; <1 = faster than real time)

Jetson Nano Gen1 (tegra210, 1 SM / 128 cores @ 921 MHz, CUDA 10.2, L4T R32.5.1,
`sm_53`, `CUDA_LAUNCH_BLOCKING=1`, GPU model + GPU iSTFT, CPU STFT):

| fixture | audio | model | **RTF_model** | STFT | iSTFT | **RTF_total** | gate on device |
|---|---|---|---|---|---|---|---|
| 8 kHz | 5.100 s | 226.12 s | **44.34** | 0.44 s | 0.09 s | **44.36** | 1.104e-06 / 119.88 dB PASS |
| 16 kHz | 5.100 s | 453.16 s | **88.86** | 1.66 s | 0.35 s | **88.92** | 7.818e-07 / 123.99 dB PASS |

8 kHz phase breakdown on the Nano: encoder 43.84 s (19%), 30x TFMamba
167.42 s (74%), decoders 14.83 s (7%), heads 0.01 s.

Real time (RTF < 1) is **not** achieved: the Nano is ~23x slower than the
GTX 1070 (1 SM vs 15, 0.55x clock), and the naive-kernel model needs ~44x
real time at 8 kHz. Getting to RTF < 1 needs ~45x more throughput; the
profile says where it is: 74% tower, 19% encoder convolutions.

Dev host (GTX 1070, 15 SMs, `sm_61`), same code:

| fixture | model | RTF_model | iSTFT CPU -> GPU | RTF_total (GPU iSTFT) |
|---|---|---|---|---|
| 8 kHz | 9.73 s | 1.91 | 1.27 s -> 0.0103 s | **1.92** |
| 16 kHz | 19.9 s | 3.91 | 4.29 s -> 0.0365 s | **3.90** |

Nano is ~23x the 1070 model time (1 SM vs 15, 0.55x clock).

## Per-kernel profile of the tower (dev host, `REUSE_KPROF=1`)

Time between instrumentation marks, one Mamba scan direction
(S=81, L=256, 8 kHz):

| kernel | per call | note |
|---|---|---|
| in_proj | 0.0332 s | 5.4 GFlop -> 163 GFLOPS (already at tiled-GEMM rate) |
| x_proj | 0.0214 s | |
| conv1d (causal, depthwise) | 0.0064 s | |
| dt_proj | 0.0048 s | |
| selective scan | 0.0011 s | |
| **out_proj + gate (tiled)** | **0.00060 s** | was 0.32 s as an SGEMV -> ~590x |
| concat window | 0.138 s | includes the un-instrumented output_proj GEMV + LayerNorm |
| residual windows | 0.141 / 0.147 s | frequency/output residual scatter |

The original hotspot was **out_proj**: 0.31-0.33 s per direction, ~81% of a
block. That is why skipping any single kernel moved the block time <1%, and
why the earlier tiled-GEMM experiment (which left out_proj alone) showed no
gain. Replacing it with a 32x32 shared-memory tiled kernel that fuses
`y * silu(z)` into the A-tile load took the tower from 47.5 s to 8.6 s.

### iSTFT moved to the GPU (`REUSE_GPU_ISTFT=1`)

The iSTFT is a DFT-GEMM: `Y[n][t] = sum_k Cr[k][t]*Cc[n][k] + Ci[k][t]*Cs[n][k]`
with the DFT table built in float64 on the host and accumulation in double
(Tegra fp64 is 1/64 rate, so this is the accuracy/speed trade; the CPU
reference also accumulates in double). Then a windowed overlap-add with
`atomicAdd` and a `sum(w^2)` normalisation.

| | CPU iSTFT | GPU iSTFT | speedup | waveform SNR |
|---|---|---|---|---|
| 8 kHz (dev host) | 1.27 s | 0.0103 s | 123x | 119.87 dB (CPU: 119.91) |
| 16 kHz (dev host) | 4.29 s | 0.0365 s | 117x | 123.96 dB (CPU: 123.99) |

On the Nano the CPU iSTFT cost 6.91 s (RTF 1.36) on 4x A57 - it was the
largest non-model cost; GPU iSTFT is ~0.1 s there.

Dev-host total with GPU iSTFT: **RTF_total 1.92 at 8 kHz** (was 2.13),
**3.90 at 16 kHz** (was 4.71).

Next targets, in order: the encoder convolutions (43.5 s = 19% of Nano model
time, naive), the `in_proj` projection (tiled, register-blocked), the
concat/residual transpose-scatter kernels (pure memory motion), and the STFT
front-end (still CPU, 0.44 s on the Nano).

## Nano-only blockers found (all fixed)
1. **Arena unit bug.** `REUSE_WS_MB` was megabytes-of-*floats*, so the default
   "512 MB" workspace was 2 GiB. On the Nano's shared-memory GPU a 2 GiB
   `cudaMalloc` succeeds as an over-commit and then a kernel touching an
   unbacked page gets a poisoned-page MMU fault, which CUDA reports as
   `the launch timed out and was terminated` - indistinguishable from a
   watchdog kill. MB now means bytes, with a halving probe and a clean
   `workspace exhausted` error. Encoder peak is ~560 MiB, so the Nano scripts
   default to `REUSE_WS_MB=640`.
2. **Async launch queues fault on Nano.** With a 640 MiB arena and ~1.5 GiB
   free, the async path dies with `nvgpu: fake mmu fault ... type 15
   (poisoned)`; `CUDA_LAUNCH_BLOCKING=1` completes. Launch overhead is ~0.1 s
   against a 225 s model, so the bench/gate scripts set it on Tegra. The async
   path is an open issue.
3. **Weight-name lifetime.** Freeing the host blob after the weight upload
   broke lookups because the entry name strings pointed into it; names are now
   copied at create time (this also fixed a latent use-after-free).

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
Ordered by expected RTF impact on the Nano:
1. **(done) iSTFT on the GPU** - see the iSTFT section above; `REUSE_GPU_ISTFT=1`.
   The STFT front-end is still on the CPU (0.44 s on the Nano) and is the same
   DFT-GEMM shape, so it is a small follow-on.
2. **Encoder convolutions.** 43.5 s of Nano time (19%) in naive convs; tiled
   shared-memory convolutions, and the dense-block concat in shared memory.
3. **in_proj + the concat/residual scatters.** in_proj is 163 GFLOPS (at the
   tiled-GEMM rate, so it needs register blocking + double buffering, not just
   tiling); concat/resid are pure transposes (0.14 s windows each on the host).
4. **Fusion / launch reduction.** 30 blocks x ~12 launches; CUDA graph capture
   per block would cut launch overhead, which matters most on the Nano where
   the async path is currently forced to serialize (blocker 2 above).
5. fp16/half2 storage for weights (38.6 MB -> 19 MB) to cut memory traffic;
   gate at SNR >= 30 dB + perceptual delta <= 0.02.
6. 44.1 / 48 kHz: gates not run (GPU cost ~44x the 8 kHz model); the STFT
   scales with the sample rate, so all rates run ~200 frames/s.
7. Chunked/streaming mode for true streaming RTF (plan S6, not started).
