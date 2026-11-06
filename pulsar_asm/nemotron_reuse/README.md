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
- [x] S8: out_proj tiled+fused gate -> tower 47.5 s -> 8.6 s on the dev host;
      GPU iSTFT (123x); autoresearch session (~270 kept experiments)
- [x] Dev host real time: RTF_total 0.0656 at 8 kHz (~15x faster than real time)
- [ ] Nano real time (RTF < 1) NOT achieved: RTF_total 1.91 at 8 kHz,
      3.77 at 16 kHz (was 44.36 / 92.9 at the first on-device run; 23.2x)
- [x] Nano profiling via an audited nvprof spool server (see Profiling)
- [x] Final verdict for live IP-PBX / VoIP use: **not viable on a Jetson Nano**
      with this model (see [Conclusion: IP-PBX / live VoIP](#conclusion-ip-pbx--live-voip))

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
- S7: **Done** (CUDA kernels pass all model gates, verified on the Nano)
- S8: Done for the dev host; Nano real-time still open (see Status)

## Device
Target: **NVIDIA Jetson Nano Developer Kit** (tegra210, 4x A57, 1 SM / 128 cores,
3 GiB shared memory, L4T R32.5.1, CUDA 10.2, `sm_53`). The development host is
an x86 box with a GTX 1070 (`sm_61`, 15 SMs).

## Nano kernel shares (8 kHz, 2 forwards, `mode=summary`)

Measured on the device with the audited nvprof spool server (see Profiling):

| kernel | share of GPU time | calls per forward |
|---|---|---|
| `k_in_conv_r` | 33.6% | 60 |
| `k_scan_dts` | 15.2% | 60 |
| `k_out_proj_r84` | 10.9% | 60 |
| `k_conv2d_r8` (dilation 8) | 10.0% | ~1.5 |
| `k_x_proj_g` | 6.8% | 60 |
| `k_conv2d_r8` (dilation 4) | 7.3% | ~1.5 |
| `k_conv2d_r8` (dilation 2) | 4.8% | ~1.5 |
| `k_outproj_ln_g` | 2.6% | 30 |
| `k_sp_up8t` | 1.8% | ~2 |
| `k_conv2d_r8` (dilation 1) | 2.7% | ~1.5 |
| `k_concat_f`, `k_inorm_prelu`, `k_resid_*` | < 1% each | -- |

GPU kernels are about 99% of the wall time on the Nano (`CUDA_LAUNCH_BLOCKING=1`),
so these shares are directly the model-time shares. `in_conv` + `out_proj` together
are ~44% of the device time; the scan is 15%.

## Verified results (dev host, x86 + GTX 1070, CUDA 10.2)

Gates are run with the **reference STFT fed in**, so the model and iSTFT are
measured on their own (`tests/gate_full.py`). Current code (commit `645df5f` and later):

| stage | 8 kHz | 16 kHz | gate |
|---|---|---|---|
| full model, complex spectrum rel-rms | **1.23e-6** | **7.68e-7** | 1e-4 |
| waveform SNR (GPU model + iSTFT) | **117.9 dB** | **~124 dB** | 50 dB |
| verdict | GATE PASS | GATE PASS | |

The same gates also run on the Maxwell code path (`REUSE_FORCE_MAXWELL_TILES=1`),
which is what the Nano executes: identical numbers. `checks.sh` runs both.

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

### Reading the numbers

RTF is **processing time divided by audio duration**. The test clip is 5.1 s of
audio, so:

- RTF 0.5 means it takes 0.5 s to enhance 1 s of audio. That is 2x real time.
- RTF 2.0 means it takes 2 s per 1 s of audio, so it can't keep up with a live stream.
- The inverse, 1 / RTF, is the speed in multiples of real time.

The numbers reported here use the same clip for both machines:

| term | meaning |
|---|---|
| `RTF_model` | GPU model time only (the 30 Mamba blocks, encoder, decoders, heads) |
| `RTF_total` | model + STFT + iSTFT, the full pipeline; this is the headline number |
| 8 kHz / 16 kHz | sample rate of the fixture. Same 5.1 s clip, twice the samples at 16 kHz |

Worked example, dev host at commit `645df5f`:

- Model: 0.331 s for 5.1 s of audio, so `RTF_model` = 0.331 / 5.1 = 0.0649.
- STFT and iSTFT add about 0.004 s, so `RTF_total` = 0.0656.
- 1 / 0.0656 = **15.2x faster than real time**.

Nano at commit `0fa348c`, 8 kHz:

- Model: 9.72 s for 5.1 s of audio, so `RTF_model` = 1.905.
- `RTF_total` = 9.76 / 5.1 = **1.91**, which is 1.9x **slower** than real time.
- To reach real time (RTF 1.0) the Nano needs another 1.91x speedup.

Taking the first on-device measurement as the baseline (`RTF_total` 44.36 at 8 kHz),
the Nano's speedup so far is 44.36 / 1.91 = **23.2x**.

The 16 kHz fixture has the same duration but twice the samples, so it costs
roughly twice as much (Nano `RTF_total` 3.77, about 3.8x slower than real time).

### Jetson Nano (tegra210, 1 SM, CUDA 10.2, `sm_53`, `CUDA_LAUNCH_BLOCKING=1`)

Current code (commit `0fa348c`), 5.100 s audio, `./run_bench.sh` protocol:

| | model | **RTF_model** | **RTF_total** |
|---|---|---|---|
| 8 kHz | 9.72 s | **1.905** | **1.91** |
| 16 kHz | - | - | **3.77** (19.23 s total) |

8 kHz phase breakdown: encoder 1.67 s, 30x TFMamba 7.37 s, decoders 0.66 s,
heads 0.01 s. Still **~1.9x slower than real time**; the tower is ~76% of the model.
Nano output is bit-identical to the dev-host Maxwell proxy (amp max|d| 0.0) and
passes the accuracy gates at both rates.

The last stretch was scored **on the Nano itself** (`rtf_nano8k`, 2.258 -> 1.915,
-15.2%), because dev-host timing proved unable to predict it. What paid there:

| change | Nano effect |
|---|---|
| conv launch bound for the 32-acc shapes (2 -> 3 -> 4 blocks/SM) | encoder -21%, -5% |
| in_conv k-chunk 32 -> 16 (removes a 16 B spill) | model -2.5% |
| out_proj (`k_out_proj_r84`) and x_proj launch bounds / KT=16 | model -2.0%, -1.9%, -0.3% |
| dense conv double buffer by input-channel parity (one barrier, not two) | encoder -6.5%, decoders -3.5% |
| 32-bit `m % L` in the in_conv epilogue (sm_53 has no integer divide) | model -0.6% |

Rule of thumb on the single Maxwell SM: an extra block/SM is usually worth a small
(< ~30 B) register spill, not a large one (~150 B).

History on the same device: the first on-device run measured RTF_model 44.34
(8 kHz) / 88.86 (16 kHz), before the dev-host optimizations were ported. The
recent gains (19.9x overall) came from the Nano-path kernels: the 32-accumulator
out_proj (`REUSE_OP=4`, the largest single step: -10.8% RTF), weight staging in
shared memory for the dense convolutions, the scan's register-resident decay
constants, and the `in_conv` halo and dead-work removal.

### Dev host (GTX 1070, 15 SMs, `sm_61`)

The scored autoresearch metric (`rtf_total`, 8 kHz, full pipeline, no reference
fed) is **0.0656** at this commit: model 0.33 s for 5.1 s of audio, held-out
clips at 0.0674 / 0.0991 / 0.1327 (8 / 12 / 16 kHz).

### Nano-only gotchas that shaped the code
- `REUSE_OP=4` (32-accumulator out_proj) is the Maxwell default: the 64-accumulator
  kernels cannot launch on sm_53.
- Dev-host timing does not predict Nano timing. Several dev-host wins were
  slower on the Nano (for example a fused concat, and an ftz `silu`), so every
  change that only touches Nano-path kernels is timed on the device before it
  lands.
- The dev-host Maxwell proxy (`REUSE_FORCE_MAXWELL_TILES=1`) is bit-exact with the
  Nano for correctness, but not for speed.
- The reverse also happens: ideas rejected as flat on the dev host can win on the
  Nano (the conv double buffer was flat on the GTX 1070, -6.5% encoder on the Nano).
- Closed on the device: `-Xptxas -dlcm=ca` (+3.7%; the sm_53 default is already
  L2-only), a 64-row in_conv tile (+25%), launch bound 2 for out_proj (+2.3%).

## Profiling

**Nano (the target):** `nvprof` needs root on L4T, so `tools/nvprof_server.py`
provides an audited spool server. It runs one fixed binary under nvprof when a
request file appears, and it only accepts whitelisted env switches.

- Install and start it: `sudo sh tools/install_nvprof_server.sh` (Ctrl-C stops it).
- Request: write `{"rate": 8000, "mode": "summary"}` to `/dev/shm/reuse_nvprof/req/<id>.json`.
- Result: `/dev/shm/reuse_nvprof/out/<id>.txt` plus `<id>.status`.
- Audit: every binary run as root is logged with its sha256.

Security: the profiled binary runs as root. While the server is running, anyone
who can write the binary can run code as root, so stop it when you're done.

Nano kernel shares, 8 kHz, four forwards (`mode=summary`):

| kernel | share of GPU time | calls per forward |
|---|---|---|
| `k_in_conv_r` | 33.6% | 60 |
| `k_scan_dts` | 15.2% | 60 |
| `k_out_proj_r84` | 10.9% | 60 |
| `k_conv2d_r8` (dilation 8) | 10.0% | ~1.5 |
| `k_x_proj_g` | 6.8% | 60 |

These shares are of GPU kernel time, which is ~99% of the wall time on the Nano
(the profile covers 2 forwards: 22.8 s of kernels vs 2 x 11.4 s of wall). Launch
overhead is negligible, so the Nano is **kernel-bound**; `in_conv` alone is ~34%
of the model.

**Dev host:** `REUSE_KPROF=1` gives a per-stage breakdown; the old table below is
kept for the record.

### Historical per-kernel profile of the tower (dev host, `REUSE_KPROF=1`)

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
`sm_53` compilation is verified, and so is **execution on the Nano itself**
(`iPhO`, 100.70.245.36, reached over SSH; the build runs from `/dev/shm`, because
the device's root filesystem is nearly full). See `tools/install_nvprof_server.sh`.

## Reference-weight warning (critical for anyone regenerating references)
The PyTorch reference needs a `mamba_ssm.Mamba` whose **parameter names match
the checkpoint exactly**. A shim that nests them (e.g. under `.mixer.`) makes
`from_pretrained` silently leave all 30 Mamba towers randomly initialized and
every downstream number is meaningless. Verify with
`load_state_dict(..., strict=False)` reporting **0 missing / 0 unexpected**.

## Conclusion: IP-PBX / live VoIP

Question asked at the end of the work: can this model run on a Jetson-Nano-based
IP-PBX and enhance every VoIP call in real time? **No, not realistically.**

1. **Throughput.** One audio stream needs RTF <= 1; the Nano is at 1.91 (8 kHz).
   A PBX enhancing both legs of N calls needs about RTF <= 1/(2N):

   | load | streams | RTF needed | gap from 1.91 |
   |---|---|---|---|
   | 1 call | 2 | 0.5 | ~4x |
   | 10 calls | 20 | 0.05 | ~40x |

   Under this port's rules (fp32, PyTorch-matched to rel-rms 1e-4, no cuBLAS/cuDNN)
   the remaining levers are worth a few percent each. fp16 (Maxwell has 2x-rate
   fp16), TensorRT, pruning/distillation or a coarser frequency grid could plausibly
   reach about one call, not a PBX's worth.
2. **Latency / causality.** RE-USE scans time in both directions and processes the
   whole utterance, so each output frame depends on future audio. A call needs each
   10-20 ms packet back within tens of ms. Live use would need chunked processing
   with look-ahead (latency plus a quality change) or a retrained causal model.
3. **PBX side.** Enhancement forces RTP decode/re-encode of every call (no direct
   media, extra CPU on the Nano's 4 ARM cores), and it only removes noise and
   reverberation. It does not fix packet loss, jitter or echo, which are often the
   real VoIP quality problems. (The 8 kHz path fits G.711 narrowband, which helps.)

**What is realistic:**
- **Live calls:** a streaming-native enhancer such as RNNoise (very light, CPU, many
  streams per core) or DeepFilterNet (causal, low latency, higher quality). Measure
  streams per device on the target hardware before committing.
- **RE-USE on the Nano:** offline / near-line jobs where delay is acceptable, such as
  cleaning call recordings, voicemail, or audio before speech-to-text. At RTF 1.91 a
  Nano processes about 31 minutes of 8 kHz audio per hour.
- **RE-USE live** would take a causal or distilled model, fp16/TensorRT and an
  Orin-class Jetson, then a measured streams-per-device figure.

## Next steps
Only relevant if offline Nano throughput matters (currently RTF_total 1.91 at 8 kHz):
1. **`in_conv`** (`k_in_conv_r`, ~34% of the Nano GPU time). Closed on the device:
   tile height (64 and 128 rows lose to 80), loader index math (latency-hidden),
   launch bound 4 (spills). A double buffer at KC=8 recovers 4% over KC=8 but is
   still ~0.9% behind KC=16; KC=16 with a double buffer (2 blocks/SM) is untested.
2. **The rest of the tower:** the scan (~15%), out_proj (~11%) and x_proj (~7%).
   Launch overhead is NOT a lever: kernels are ~99% of the wall time.
3. fp16/half2 compute and storage. This is the largest remaining lever on Maxwell,
   but it needs an explicit accuracy decision (the fp32 gates would not hold as is).
4. 44.1 / 48 kHz: gates not run; cost scales with the sample rate.
5. Chunked/streaming mode for true streaming RTF (plan S6, not started); see the
   IP-PBX conclusion for why it matters.

Open decisions and the full experiment log live in `.auto/ideas.md` and
`.auto/log.jsonl` at the repository root.
