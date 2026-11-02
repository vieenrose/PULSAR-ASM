# NemotronLabs RE-USE on Jetson Nano Gen1: port plan

Target: the full RE-USE speech-enhancement pipeline (WAV → STFT → SEMamba → iSTFT → WAV)
as a self-contained CUDA sm_53 + C engine on the Nano (CUDA 10.2, JetPack R32.5.1).
No Python, PyTorch, cuBLAS or cuFFT at runtime. Metric: **RTF = processing wall time / input audio duration**.

## 1. What the model is (from `models/*.py`, `model.safetensors`)

- 9.61 M params, 1416 tensors, all F32, 38.6 MB.
- SFI STFT: the 8 kHz config (n_fft 320, hop 40, win 320) scales with the input rate, so every
  rate runs at **200 frames/s** and only the frequency axis F = n_fft/2+1 grows (161 / 321 / 883 / 961
  at 8 / 16 / 44.1 / 48 kHz). Magnitude is `log1p(|X|)`, phase is `angle(X)`.
- DenseEncoder at full T×F: 1×1 conv 2→64, a 4-layer dilated 3×3 DenseBlock (in 64/128/192/256 → 64,
  dilation 1/2/4/8 in time), (1,3) conv with stride (4,2). Every conv is followed by InstanceNorm2d(affine) + PReLU.
- 30 TFMamba blocks at T'×F' (time ÷4, freq ÷2). Each block runs a bidirectional Mamba along time
  (F' sequences of length T') and then along frequency (T' sequences of length F'). Each Mamba has
  d_model 64, d_inner 256, d_state 16, d_conv 4, dt_rank 4, plus output_proj 128→64, LayerNorm and a residual.
- Mag and phase decoders: a DenseBlock, then two SPConvTranspose (sub-pixel) upsamplers; the mag head ends in a
  1×1 conv, the phase head in atan2(conv_i, conv_r).
- Post-processing: zero out frames where more than 50% of bins have `expm1(relu(mag)) == 0`, then
  iSTFT (`expm1(relu(mag))`, hann, center).
- `inference_chunk.py` runs 5 s chunks with a 50% hop and a hann-clamped crossfade. **The 50% overlap doubles the compute.**

## 2. Budget (5 s chunk, counted from the shapes above, MACs per audio-second)

| rate | T'×F' | encoder | 30-block tower | decoders | total | exp/s | RTF floor fp32 | RTF floor half2 | one 64-ch fp32 plane |
|---|---|---|---|---|---|---|---|---|---|
| 8 kHz | 251×81 | 12.1 G | 37.5 G | 4.0 G | **53.6 G** | 2.0 G | 0.45 | 0.23 | 40 MB |
| 16 kHz | 251×161 | 24.0 G | 74.5 G | 7.9 G | **106.5 G** | 4.0 G | 0.90 | 0.45 | 79 MB |
| 44.1 kHz | 252×442 | 65.9 G | 205.3 G | 21.9 G | **293.7 G** | 10.9 G | 2.49 | 1.24 | 217 MB |
| 48 kHz | 251×481 | 71.5 G | 222.5 G | 23.7 G | **318.5 G** | 11.9 G | 2.70 | 1.35 | 236 MB |

The floors assume the Nano's 128 CUDA cores at 921.6 MHz run at 100% of peak: 118 G fp32 FMA/s, or 236 G FMA/s with
half2. They do **not** include the 50% chunk overlap; the reference chunked mode doubles every number.
The model is compute-bound: weights are 38 MB, so bandwidth is not the limit and, unlike the LLM ports, the
bandwidth-ceiling playbook does not apply. Realistic kernels reach 50–70% of peak.

Expected RTF (non-overlapped / reference 50% overlap):

- **8 kHz**: ~0.7 / ~1.4 in fp32; ~0.4 / ~0.8 with half2.
- **16 kHz**: ~1.4 / ~2.8 in fp32; ~0.8 / ~1.6 with half2.
- **44.1/48 kHz**: well above 1. The 5.1 s fixture (`mic_test2.wav`) is 44.1 kHz, so its RTF will be above 1 unless it is processed at a lower rate.

So real time is reachable at 8 kHz, borderline at 16 kHz, and not reachable at full-band rates
without changing what is computed. The available levers are half2, smaller chunk overlap, and
resampling before enhancement. Every one of them is a gated deviation, not a free win.

## 3. Accuracy gates (replacing bit-exactness)

Bit-exactness against PyTorch is not a meaningful goal here: torch's own CUDA and CPU results differ, and the
model contains exp, softplus, atan2 and log1p. The gates are:

1. **Per-stage**: rel-rms ≤ 1e-4 (fp32 engine) vs an fp32 PyTorch reference at each boundary: STFT mag/phase,
   encoder out, after each of the 30 blocks (a per-block × time/freq rms matrix, to localize drift), decoder mag/phase.
2. **Waveform**: SNR of the engine output vs the reference output ≥ 50 dB for fp32, and ≥ 30 dB plus a PESQ/DNSMOS
   delta ≤ 0.02 for any half2/fp16 variant.
3. **Phase**: compare on the complex spectrum, not raw atan2, which wraps at ±π on near-zero bins.
4. **Sweep-artifact mask**: frame mask identical to the reference's.

The reference is a pure-PyTorch mirror of `mamba_ssm.Mamba.forward`: in_proj, causal depthwise conv + SiLU, x_proj,
softplus(dt_proj + bias), selective scan with D skip and SiLU(z) gating, out_proj. It also runs in fp64 as a referee.
If possible, it is cross-checked once against the official `mamba_ssm` fused kernel on the dev box's GTX 1070.
`mamba_ssm` needs CUDA ≥ 11.6, so it cannot run on the Nano, and there is no official on-device baseline to compare against.

## 4. Stages

| # | stage | deliverable | gate |
|---|---|---|---|
| S0 | Reference + fixtures (dev host) | venv (torch, numpy, soundfile, librosa), pure-torch Mamba mirror, `dump_ref.py` dumping every boundary tensor; fixtures: `mic_test2` at native 44.1 k plus 8/16/48 k resamples, one 60 s concatenation for chunked mode; exact MAC counter | mirror vs official `mamba_ssm` ≤ 1e-5 rel (if buildable on GTX 1070); fp32 vs fp64 referee |
| S1 | Blob | `mkblob.py`: safetensors → `.reuse` (HF names verbatim, config in header, fp32 + optional fp16 copy, precomputed `A = -exp(A_log)`) | round-trip byte check, 1416/1416 tensors |
| S2 | Front end | WAV reader (PCM16/f32, multichannel), reflect pad, STFT **as a DFT-GEMM** (handles 1764 = 2²3²7² without cuFFT; ~0.3 GMAC/s), log1p / atan2 | vs `torch.stft` mag/phase |
| S3 | Encoder | kernels: 1×1 conv, dilated 3×3 conv over the dense concat **without materializing the concat** (reads separate 64-ch planes), strided (1,3) conv, InstanceNorm (fp32 sum/sumsq in a fused epilogue, then a normalize + PReLU pass) | encoder-out rel-rms |
| S4 | Mamba tower | one fused per-direction pipeline: in_proj GEMM → conv1d+SiLU → x_proj → dt (softplus) → **selective scan** (one thread per (sequence, d_inner) channel, 16-state h in registers, parallel over 81–481 × 256 × 251 lanes) → D + gate → out_proj. Backward direction via reversed indexing (no flip copies); time/freq orientation via strides (no permute copies); output_proj + LN + residual fused | 1 block, then all 30 with the per-block rms matrix |
| S5 | Decoders + iSTFT | dense block, SPConvTranspose = conv + sub-pixel shuffle in the store, heads, atan2, sweep mask, inverse DFT-GEMM + overlap-add + window-sum normalization matching `torch.istft(center=True)` and `pad_or_trim_to_match` (reusing the LFM3 iSTFT spike's approach) | waveform SNR ≥ 50 dB |
| S6 | Driver `reuse_nano` | full-clip mode and a chunked mode identical to `inference_chunk.py` (5 s, hop 0.5, clamped hann, last chunk end-aligned); fixed GPU arena sized per (rate, chunk); one CUDA graph per chunk shape; WAV out | both modes vs reference |
| S7 | Benchmark | `bench_rtf.sh`: scp → nvcc build on the Nano → warm-up → N=10 reps → median / p90 RTF, per-stage `cudaEvent` timings, peak RSS + MemAvailable, `tegrastats` clocks/temperature log | — |
| S8 | Optimize (gated) | GEMM tiling for the 64/256-wide shapes, scan fusion, half2 arithmetic, fp16 plane storage, overlap tuning (0.5 → 0.25 → 0.1), streaming/early start of the next chunk's STFT; every lever measured with order-balanced same-job A/B pairs, as the repo already does | S3–S6 gates re-run |

## 5. RTF definition (what the benchmark reports)

- `RTF_compute`: H2D of the PCM through the output PCM on the host (STFT → model → iSTFT, plus chunk crossfade) ÷ audio duration.
- `RTF_e2e`: from WAV read to WAV written, same divisor.
- Reported separately: model load and CUDA init (one time) and the first-chunk latency.
- Matrix: {8, 16, 44.1, 48 kHz} × {full-clip, chunked 50%} × {fp32, half2} on the 5.1 s fixture and the 60 s
  concatenation. Every row is labeled with the clocks (nvpmodel mode, jetson_clocks on/off) and whether the board's
  other services were running.

## 6. Risks / decisions needed

1. **Memory.** The Nano currently has **~860 MB available**: it also runs Asterisk, a 500 MB Python server,
   `pt_v21_serve` and MySQL. At 44.1 kHz one 64-ch fp32 plane of a 5 s chunk is 217 MB, and the encoder DenseBlock
   needs 4 of them live (~870 MB). Options:
   - fp16 plane storage with fp32 stats;
   - shorter chunks at high rates (changes the outputs: InstanceNorm and Mamba see the whole chunk);
   - tile-and-recompute (~2.5× encoder cost).

   Recommendation: fp16 storage at ≥ 44.1 kHz, gated by SNR.
2. **Benchmark contention.** The board is in production use. Can those services be paused and `jetson_clocks` set
   for benchmark runs, or should numbers be taken as-is and labeled?
3. **Which rate is "the" target.** Real time is realistic at 8 kHz, a stretch at 16 kHz, and not achievable at
   44.1/48 kHz with the reference's 5 s / 50% chunking on this GPU. A resample-to-16 k front option would trade
   bandwidth for RTF.
4. **The GPU is a single 128-core SM** (Nano, not TX1), so all occupancy planning uses one SM.
   The scan's exp throughput (SFU, 32/clk) is ~7% of the time at 8 kHz and ~37% at 48 kHz.

## 7. Layout on the branch

```
pulsar_asm/nemotron_reuse/
  ref/        dump_ref.py, mamba_ref.py, mkfixtures.py, count_macs.py   (dev-host only)
  tools/      mkblob.py
  engine/     reuse.cu (single TU: kernels + arena + graphs), reuse_host.c (WAV, driver, chunker)
  tests/      gate_stages.py, gate_wave.py, golden/ (SHA-pinned fixtures + reference outputs)
  bench_rtf.sh, build_nano.sh, README.md (results table)
```
