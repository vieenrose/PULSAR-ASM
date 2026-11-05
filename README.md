# PULSAR-ASM: porting LFM3-Audio-2B to a Jetson Nano

**LiquidAI's LFM3-Audio-2B** is a 2 B-parameter speech model: a log-mel front end, a
17-layer LFM2 audio encoder, a 24-layer LFM3 text backbone (mixed short-conv and
attention layers), a MaskGIT depth head over 12 audio codebooks, and a LiquidCodec
decoder that turns tokens back into a 24 kHz waveform.

This branch ports it to a **Jetson Nano Gen1** — the 2019 4 GB board, 4× Cortex-A57 +
4 SM Maxwell (sm_53, CUDA 10.2) — as a single-translation-unit CUDA engine plus a
complement of AArch64 NEON kernels, with **no Python, cuBLAS or llama.cpp at runtime**.
The port proceeds one component at a time, and a component is only called ported when
the HF reference's own outputs certify it.

```
PORTED AND GATED    text backbone — 24 layers, 286 tensors, 1946 MB at Q8
                    113.8 ms/token (8.8 tok/s) · 307 MB RSS on a 4 GB board
                    17.1 GB/s achieved against a 21.8 GB/s streaming roof I measured
                    on this board with a pure-read kernel — 78% of it
                    log-mel front end — NEON, 0.554 ms/frame, 19 fixtures;
                    as accurate as HF against an exact-f64 referee
                    liquid7.1 tokenizer — 531/531 fixtures id-for-id vs HF
                    AND byte-exact round-trip (encode+decode, 5.4 MB blob)
                    audio encoder tower — 17 layers + projector, GPU, 130 MB blob,
                    176/176 dtype decisions cross-checked against Liquid's own GGUF
                    MaskGIT depth head — 82/82 frames exact vs HF-on-Q8, 5 fixtures
                    LiquidCodec decoder — 45.26 dB vs HF-on-Q8, = the Q8 floor
                    iSTFT — SNR 103–130 dB vs torch on 9 cases
                    interleaved text/audio controller — mode machine PASS on HF traces

END TO END          speech in -> text out (M1)   260/262 tokens vs HF, token-WER 0.008
                    text in -> speech out (M2)   60-slot turn in 5m34s, 2.80 s of audio,
                                                 records byte-identical across builds
                    speech in -> speech out (M3) cascaded path runs end to end:
                                                 lfm3_asr --out-ids -> lfm3_m3prompt ->
                                                 lfm3_speak; 100-slot turn, 36 text + 64 audio
                                                 frames, 5.12 s of 24 kHz audio, exit 0.
                                                 Native M3 (audio rows into the speech prompt)
                                                 is not wired yet.
REFUTED             batched backbone prefill — built, bit-exact, and ~4x SLOWER than
                    the shipped T=1 path at ASR shapes; kept flag-gated, not shipped
NOT MET             realtime ASR (ASR_RT <= 1.00): 1.33 best case, engine alone 0.65
NOT STARTED         S7 shell polish (streaming mic) · S9 packaging
```

The staged plan for the rest of the port — every remaining component with its gate, its
derived cost and its risks, ending at a self-contained asm + C runtime — is
[`doc/LFM3_PORT_STAGES.md`](doc/LFM3_PORT_STAGES.md).

The engine document is
[`pulsar_arm/asm/README-lfm3-audio.md`](pulsar_arm/asm/README-lfm3-audio.md):
per-component port status with tensor and byte budgets, the checkpoint → `.lfm3q8`
blob → engine contract, the reuse matrix and the traps for each component still to
port, the gate protocol a port is certified by, and the memory + bandwidth budget for
the whole model. What follows is the summary.

## Port status

Bytes are measured from the checkpoint headers and the exported blob, not estimated,
except where marked.

| component | tensors | Q8 bytes | status |
|---|---|---|---|
| text backbone (LFM3, 24 layers) | 286 | **1946 MB** | **ported, 4 gates green** |
| log-mel front end | *(no weights)* | — | **ported, gated** (bit-exact vs its C mirror; HF tolerance 8.3e-6 live-cell) |
| liquid7.1 tokenizer | — | 5.4 MB blob | **ported, gated** (531/531 id parity + byte round-trip; `tokenizer_config.json`'s `pre_tokenizer:null` is wrong — see plan §12) |
| iSTFT + overlap-add (S5 spike) | *(no weights)* | — | **gated** (SNR 103–130 dB; `__sincosf` arm) |
| audio encoder (LFM2, 17 layers) | 158 | ~119 MB | **ported, gated** (all 17 layers + projector, GPU) |
| downsample front end + projector | 18 | fp32 + ~6 MB | **ported**; 176/176 GGUF cross-check |
| MaskGIT head, depth embeddings, depth_linear, codebook | 66 | 243 MB blob | **ported, gated** — 82/82 frames exact vs HF-on-Q8 across 5 fixtures |
| LiquidCodec decoder transformer | 100 | 41 MB blob | **ported, gated** — 45.26 dB vs HF-on-Q8, 32.04 dB vs fp32 (= the Q8 floor, so no code gap) |
| interleaved controller + shell | — | — | **ported** — `lfm3_ctrl.c` mode machine PASS on 2 HF traces; `lfm3_speak` runs the live loop |
| batched prefill (backbone, audio path) | — | — | **REFUTED**: bit-exact but ~4x slower than T=1 at ASR shapes; flag-gated, not shipped |

Every row above is certified by a gate that scores the engine against HF **on the same
Q8 weights**, not against fp32 — the fp32 comparison is kept as the quantization-error
record, because an engine more accurate than its reference can never match it.

**Where the time goes on the speech path.** A 60-slot interleaved turn (36 audio frames,
2.80 s of audio) runs in **5m34s**, down from 27 min, and its token stream, frame codes
and WAV are byte-identical across that change. The split: 87 s one-time backbone load +
prefill, 200 s over the frames (3.2 s depth MaskGIT, 1.3 s depth blob reload, 1.4 s
backbone at 12 tokens/frame), 47 s codec. Two things got it there and both were found by
reading disassembly after a timing table pointed at them, not by guessing: the depth and
codec GEMMs were one dependent fp32 chain per output (the depth engine's 12 positions are
now one batched pass through a NEON kernel — 13.4 s -> 6.5 s per forward, bit-exact), and
the codec iSTFT was recomputing k-only work for all 1280 output samples (~736 M
transcendentals per turn), which had become the whole codec cost once the GEMMs were
batched. **Two data races surfaced only in the byte comparison, not as crashes** — a
parallel overlap-add over a hop-320/frame-1280 layout, and magnitude tables shared across
threads. That is why the rule in this repository is that a speed change ships with a byte
comparison against the pre-change binary.

**Why the port is ordered this way — and the correction that changed the budget.**
Time on this board follows bytes at 11–17 GB/s, so a component's cost can be budgeted
before it is written — but bytes are only half the roofline. The first version of this
paragraph was bandwidth-only and understated the codec 2.7×; the corrected per-component
floors (max(bytes/17 GB/s, MACs/236 GFMA/s), plan §2/§13): the encoder is compute-bound
at **10–20 ms per 0.96 s chunk**, the depth head ~13 ms minimum / 25–40 ms realistic per
frame, the codec transformer 11.6 ms minimum / 25–50 ms realistic — 4.3 ms was bandwidth
alone and needs an unachievable 1100 GFLOP/s. The backbone prefill still dominates an
audio query and the mel front end is free. **The active target is speech-in/text-out
realtime** (plan §14–18). The encoder emits **12.5 tokens per second of audio** (measured
with HF; the earlier "25/s" misread the downsampler, plan §18), so prefill is 18.4 GMAC
per audio-second. At the measured batched rates (30–33 GMAC/s, bit-exact) the budget is
prefill 0.56–0.61 s + encoder ~0.06 s (estimate) + text decode 0.31–0.44 s ≈ **0.9–1.07×
realtime: borderline, not yet demonstrated end to end**.

## What makes the port mechanical

- **The blob carries the architecture, not the C++.** `config.json` travels in the
  `.lfm3q8` header and the engine reads `layer_types`, `sliding_window`, `conv_L_cache`,
  `head_dim`, `norm_eps`, `use_sandwich_norm` and the rope parameters out of it.
- **Tensor names are the HF names, verbatim.** No rename table; a new component keeps its
  own prefix and lands in the same dictionary.
- **One dtype switch.** Q8 codes + fp32 scales (1.125 B/weight) for 2-D linears, fp32
  verbatim for norms and 3-D conv taps, bf16 verbatim for the embedding arm.
- **The gates are the same for every component**: HF teacher-forced references dumped at
  the exact boundary being scored, a per-layer × per-position rms matrix (within 3%), a
  position-0 logits check, top-1/KL agreement over every generated position, and greedy
  streams. Long-context and tie-zone variants ride on top.
- **Reproduce the reference's rounding points first, then optimize only where the gate
  says you may.** Activations are stored bf16 and arithmetic is fp32 with an explicit
  bf16 round at each op — an fp32 engine is *more* accurate than its reference and can
  never match it.

## Why the backbone is fast, in one paragraph

Per decode step the engine streams every linear it owns plus the tied 128 000-row head:
1946 MB. At 113.8 ms that is ~17.1 GB/s. A pure-read probe on this board tops out at
**21.8 GB/s**, so the backbone sits at 78% of the streaming roof — 1.26x of headroom left
in the kernel, no more. Arithmetic levers are capped at ~2%: doubling GEMV FLOPs costs
+1.8%, measured, not assumed. The wins that got here
were the ones that removed *time not bytes*: `__launch_bounds__(256,6)` on the Q8 GEMV
(−17.3%, a resource hint that changed no value), `__fmaf_rn` accumulation (−2.0%, one
rounding per product, as ggml's own reference dot product compiles to), fusing the
bf16→f32 staging into the GEMV (~110 launches), plus a load-time warm-up and a
feeding bug that had been costing 318 ms of every first token. Everything else in the
config space — R, TPB, KCHUNK, cache hints, ILP, CUDA graphs, per-lane rows — is
measured out with numbers in the document's appendix. **Do not re-walk them.**

## Gates, and how to run them

```sh
bash .auto/lfm3/checks.sh A B C E F K L                    # the four reference gates
python .auto/lfm3/checks_lfm3.py --long A B C E F K L      # + long-context and tie-zone
bash .auto/lfm3/measure.sh                                 # sync, build, fixed job, METRIC lines

# component gates, one per ported stage:
python .auto/lfm3/checks_mel.py                            # S2 log-mel (fixtures + HF + f64 referee)
python .auto/lfm3/checks_tok.py                             # S1 tokenizer (531 fixtures + round-trip)
python .auto/lfm3/checks_istft.py                           # S5 iSTFT spike (9 cases, SNR >= 60 dB)
bash  .auto/lfm3/checks_gemm.sh                            # batched Q8 GEMM (fp64 gate, audio shapes)
```

```
matrix, 23 layers x 23 positions vs HF        all within 3%
pos-0 logits                                  top-1 = HF's, corr 0.99998
131 positions x 128k logits, 6 prompts        top-1 130/131, KL 0.0007-0.0019
greedy streams, 7 prompts x 16 tokens         6/7 exact (the 7th is a known near-tie)
long context, 1000 positions vs fp16 ref      0 / 24000 cells outside 3%
tie zone, 134 positions                       top-1 133/134 vs bf16, 127/134 vs fp16
```

Two reference-precision facts the gates encode: bf16 and fp16 HF references disagree with
each other at ~4.5% of teacher-forced positions (argmax ties where precision flips the
answer), and the engine tracks the fp16 reference on magnitudes while matching the bf16
one on argmax. Both are true; a gate says which question it asks.

The depth and codec engines share one NEON kernel file, because nvcc 10.2's front end
cannot parse GCC's `arm_neon.h` even in the host pass — it is compiled by gcc and linked:

```sh
gcc -O2 -fopenmp -c pulsar_arm/asm/lfm3_depth_neon.c -o lfm3_depth_neon.o
nvcc -gencode arch=compute_53,code=sm_53 -O2 -Xcompiler -fopenmp \
     pulsar_arm/asm/lfm3_depth.cu lfm3_depth_neon.o -lgomp
```

Engine: `nvcc -gencode arch=compute_53,code=sm_53 -O3 -fmad=false -Xptxas -O3,-dlcm=ca
lfm3_nanode.cu -lz`. The engine itself takes and prints token ids; the tokenizer is now
a separately gated component (`pulsar_arm/asm/lfm3_tok.c`, id parity with HF proven) not
yet linked into the engine binary. Test-only knobs for the prefill experiment:
`LFM_PF_CT` (batch width), `LFM_PF_ARM`, `LFM_PF_T`; harness `--prefill-h2batched`.

## Layout

```
pulsar_arm/asm/lfm3_nanode.cu          LFM3 text backbone: Q8 GEMV, attention, short conv, head
pulsar_arm/asm/lfm3_conv.S             gated short-conv decode kernel, AArch64 NEON, bit-exact
pulsar_arm/asm/lfm3_elem.S             rmsnorm / mul / add on bf16, AArch64 NEON, bit-exact
pulsar_arm/asm/lfm3_mel.S/.c           S2 log-mel front end, AArch64 NEON (gated)
pulsar_arm/asm/lfm3_tok.c/.h           S1 liquid7.1 tokenizer, C (gated, encode+decode)
pulsar_arm/asm/lfm3_istft.cu           S5 iSTFT spike, CUDA (gated, SNR 103–130 dB)
pulsar_arm/asm/lfm3_enc_gpu.cu         S3 audio encoder tower, CUDA (gated)
pulsar_arm/asm/lfm3_depth.cu           S4 depth head + in-engine MaskGIT (gated, 82/82 frames)
pulsar_arm/asm/lfm3_depth_neon.c       shared Q8 GEMV/GEMM kernels, NEON + OpenMP
pulsar_arm/asm/lfm3_codec.cu           S5 LiquidCodec decoder (gated, 45.26 dB)
pulsar_arm/asm/lfm3_ctrl.c             S6 interleaved mode-machine checker
pulsar_arm/asm/lfm3_asr.c              M1 shell: WAV -> text
pulsar_arm/asm/lfm3_speak.c            M2 shell: text -> 24 kHz WAV (resident backbone)
pulsar_arm/asm/lfm3_wav.c/.wavout.c    WAV reader (no resampling) / 24 kHz writer
pulsar_arm/asm/lfm3_gemm.cu            batched Q8 GEMM instrument, audio shapes (fp64-gated)
pulsar_arm/tools/mkmel.py/.mktok.py/.mkaudio.py  build-time: mel tables, tokenizer blob, audio blob
pulsar_arm/asm/README-lfm3-audio.md    the porting document (status, contract, recipes, budgets)
doc/LFM3_PORT_STAGES.md                the staged plan + measurement record (§13 GGUF, §14 challenge)
pulsar_asm/lfm3/tools/export_lfm3_q8.py  checkpoint -> .lfm3q8 blob exporter
.auto/lfm3/                            gate suite, HF reference dumpers, A/B harness
doc/LFM3_AUDIO_PLAN.md                 the original scoping study and measured facts
```

## Other engines in this repository

This repository started as pure-assembly inference engines, and they are all still here
and working:

- a single static AArch64 binary that runs Gemma-3 on a Raspberry Pi 4 with no libc;
- the C + NEON Bonsai port that runs Ternary-Bonsai 8B and 27B on a DGX Spark CPU;
- the original x86-64 Gemma engines;
- the previous-generation Jetson Nano engines: **Ternary-Bonsai-2**
  ([2B/4B/9B, 4k context](doc/BONSAI2_NANO_JETSON.md) — 24.2 / 11.3 / 7.0 tok/s decode,
  bit-exact against the PrismML llama.cpp fork) and the 1-bit **Bonsai-8B Q1_0**
  ([16.5 tok/s](doc/BONSAI_8B_Q1_NANO.md)).

See [doc/OTHER_ENGINES.md](doc/OTHER_ENGINES.md) and
[pulsar_arm/bonsai2/README.md](pulsar_arm/bonsai2/README.md).
