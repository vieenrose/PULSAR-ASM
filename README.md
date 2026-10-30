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
                    112.8 ms/token (8.9 tok/s) · 2280 ms to first token on a 23-token
                    prompt · 307 MB RSS on a 4 GB board
                    at the board's ~17 GB/s streaming ceiling — bandwidth-saturated
NOT STARTED         log-mel front end · 17-layer audio encoder · audio→text projector
                    MaskGIT depth head + 12 codebook heads · LiquidCodec decoder
                    liquid7.1 tokenizer · interleaved text/audio controller
```

The full document is
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
| audio encoder (LFM2, 17 layers) | 170 | ~119 MB | not started |
| downsample front end + projector | 19 | ~11 MB | not started |
| MaskGIT head, depth embeddings, depth_linear, codebook | 77 | ~169 MB | not started |
| LiquidCodec decoder | — | 147 MB | not started |
| log-mel front end | *(no weights)* | — | not started |

**Why the port is ordered this way.** Time on this board follows bytes at 11–17 GB/s, so
a component's cost can be budgeted before it is written. The encoder streams its ~119 MB
once per audio frame (100 frames/s) ≈ 0.7–1.0 s of GPU per second of audio; the backbone
costs ~25 tokens per second of audio × ~98 ms/step of sequential prefill ≈ 2.4 s per
second of audio. **The backbone prefill dominates an audio query, the encoder is second,
and the mel front end is free.** Both figures are byte/rate derivations from measured
per-step costs, flagged as such in the porting document — the encoder is not ported yet.

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
1946 MB. At 112.8 ms that is ~17.3 GB/s, at the ceiling a pure-read probe measures on
this GPU, so the backbone is bandwidth-saturated and arithmetic levers are capped at
~2% — measured, not assumed: doubling GEMV FLOPs costs +1.8%. The wins that got here
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

Engine: `nvcc -gencode arch=compute_53,code=sm_53 -O3 -fmad=false -Xptxas -O3,-dlcm=ca
lfm3_nanode.cu -lz`. It takes and prints token ids; there is no tokenizer on the board.

## Layout

```
pulsar_arm/asm/lfm3_nanode.cu          LFM3 text backbone: Q8 GEMV, attention, short conv, head
pulsar_arm/asm/lfm3_conv.S             gated short-conv decode kernel, AArch64 NEON, bit-exact
pulsar_arm/asm/lfm3_elem.S             rmsnorm / mul / add on bf16, AArch64 NEON, bit-exact
pulsar_arm/asm/README-lfm3-audio.md    the porting document (status, contract, recipes, budgets)
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
