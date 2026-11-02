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
- S0: Done (fixtures, references, blob)
- S1: Done (blob format)
- S2: Done (DFT-GEMM STFT/iSTFT, matches torch)
- S3-S5: In progress (kernels + model structure; full forward pass being wired)
- S6: Driver loads blob/wav, runs pipeline
- S7-S8: Pending

## Recent progress
- Full pipeline wired: blob → STFT → encoder (dense conv + IN+PReLU + stride conv) → forward → iSTFT
- CPU engine compiles and runs; outputs generated
- CUDA build script ready for Nano deployment

## Status (current)
- Engine builds and runs on CPU: STFT → (model pass-through) → iSTFT
- CUDA build script ready for Nano
- Full model forward pass in progress

## Next steps
- Implement full model forward pass (encoder, 30×TFMamba, decoders)
- Wire iSTFT round-trip to match reference
- Port kernels to CUDA (sm_53)
- Benchmark RTF on Nano
