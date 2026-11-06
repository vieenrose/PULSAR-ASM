# PULSAR-ASM: NemotronLabs RE-USE on Jetson Nano Gen1

[NVIDIA NemotronLabs RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE) is a 9.6M-parameter multilingual universal speech enhancement (USE) model based on SEMamba. This branch ports it to **Jetson Nano Gen1** (Tegra X1, 4× Cortex-A57 + 128-core Maxwell sm_53, CUDA 10.2) with CUDA acceleration, targeting **RTF < 1.0** on 8 kHz and 16 kHz inputs.

## Branch
[`nemotron-re-use-jetson-nano`](https://github.com/vieenrose/PULSAR-ASM/tree/nemotron-re-use-jetson-nano)

## Results (final)

RTF = processing time / audio duration (< 1 = faster than real time), full
pipeline, 5.1 s test clip, output gated against fp32 PyTorch (rel-rms <= 1e-4,
SNR >= 50 dB) at both rates.

| device | 8 kHz RTF | 16 kHz RTF | vs. first run |
|---|---|---|---|
| **Jetson Nano Gen1** (1 SM, sm_53) | **1.91** | **3.77** | 44.36 -> 1.91 = **23.2x** faster |
| Dev host, GTX 1070 (15 SMs, sm_61) | 0.066 | - | ~15x faster than real time |

**The Nano does not reach real time:** it takes ~1.9 s to enhance 1 s of 8 kHz
audio. The model has only 9.6M parameters but runs over the full time-frequency
grid (30 bidirectional Mamba blocks plus full-resolution dense convs), and the
Nano's GPU (~0.24 TFLOPS fp32) is ~25x smaller than the dev host's. Profiling on
the device shows ~99% of wall time is inside GPU kernels, so launch overhead is
not the bottleneck.

## Conclusion: IP-PBX / live VoIP

Running this model on a Jetson-Nano-based IP-PBX to enhance every VoIP call in
real time is **not realistic**:

- **Throughput:** enhancing both legs of N calls needs RTF <= 1/(2N), i.e. 0.5 for
  one call (~4x short) and 0.05 for ten calls (~40x short).
- **Latency:** RE-USE scans time in both directions over the whole utterance, so
  output depends on future audio; live calls need each 10-20 ms packet back within
  tens of ms. That requires look-ahead chunking or a retrained causal model.
- **PBX side:** every call must be decoded and re-encoded (no direct media), and
  enhancement does not fix packet loss, jitter or echo.

Realistic alternatives: a streaming-native enhancer (RNNoise, DeepFilterNet) for
live calls, and RE-USE offline on the Nano (recordings, voicemail, speech-to-text
pre-processing: ~31 min of 8 kHz audio per hour). Details in
[`pulsar_asm/nemotron_reuse/README.md`](pulsar_asm/nemotron_reuse/README.md#conclusion-ip-pbx--live-voip).

## Port details
- **Runtime:** Self-contained CUDA + C/AArch64, no Python/PyTorch/cuBLAS/cuFFT at runtime.
- **Architecture:** STFT (200 frames/s, SFI) → DenseEncoder → 30× TFMamba blocks (bidirectional time+freq Mamba, d_inner=256, d_state=16) → Mag/Phase decoders → iSTFT with 50% overlap chunking.
- **Accuracy gates:** Stage-wise rel-rms ≤ 1e-4 vs fp32 PyTorch reference; waveform SNR ≥ 50 dB (fp32), ≥ 30 dB + perceptual delta ≤ 0.02 (half2/fp16). Phase compared on complex spectrum.
- **Targets:** 8 kHz and 16 kHz (primary). 44.1/48 kHz reported for information only.

## Documentation
- [`pulsar_asm/nemotron_reuse/README.md`](pulsar_asm/nemotron_reuse/README.md) — Status, results, Nano optimisation notes, IP-PBX conclusion, build.
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
