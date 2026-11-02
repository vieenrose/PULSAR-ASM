# PULSAR-ASM: NemotronLabs RE-USE on Jetson Nano Gen1

[NVIDIA NemotronLabs RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE) is a 9.6M-parameter multilingual universal speech enhancement (USE) model based on SEMamba. This branch ports it to **Jetson Nano Gen1** (Tegra X1, 4× Cortex-A57 + 128-core Maxwell sm_53, CUDA 10.2) with CUDA acceleration, targeting **RTF < 1.0** on 8 kHz and 16 kHz inputs.

## Branch
[`nemotron-re-use-jetson-nano`](https://github.com/vieenrose/PULSAR-ASM/tree/nemotron-re-use-jetson-nano)

## Port details
- **Runtime:** Self-contained CUDA + C/AArch64, no Python/PyTorch/cuBLAS/cuFFT at runtime.
- **Architecture:** STFT (200 frames/s, SFI) → DenseEncoder → 30× TFMamba blocks (bidirectional time+freq Mamba, d_inner=256, d_state=16) → Mag/Phase decoders → iSTFT with 50% overlap chunking.
- **Accuracy gates:** Stage-wise rel-rms ≤ 1e-4 vs fp32 PyTorch reference; waveform SNR ≥ 50 dB (fp32), ≥ 30 dB + perceptual delta ≤ 0.02 (half2/fp16). Phase compared on complex spectrum.
- **Targets:** 8 kHz and 16 kHz (primary). 44.1/48 kHz reported for information only.

## Documentation
- [`pulsar_asm/nemotron_reuse/README.md`](pulsar_asm/nemotron_reuse/README.md) — Status, structure, quickstart.
- [`doc/NEMOTRON_REUSE_NANO_PLAN.md`](doc/NEMOTRON_REUSE_NANO_PLAN.md) — Budget, stages S0–S8, RTF definition, benchmark plan, open decisions.

## Repository layout (relevant)
```
pulsar_asm/nemotron_reuse/
  ref/      Pure-PyTorch reference + fixture generation
  tools/    Weight blob conversion
  engine/   CUDA kernels + driver
  tests/    Accuracy gates
  bench/    RTF benchmarks
```

## License
OpenMDW-1.1 (see [LICENSE](LICENSE)).
