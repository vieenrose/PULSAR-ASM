# PULSAR-ASM: Bonsai-8B 1-bit on a Jetson Nano

An 8-billion-parameter language model running at **8.3 tokens/s on a
Jetson Nano Gen1**, the 2019 4 GB board, using its 128-core Maxwell GPU.
The model is PrismML's **Bonsai-8B** (Qwen3-8B architecture) in the
`Q1_0` format: true 1-bit weights, one sign bit per weight plus one f16
scale per 128-weight block.

| | |
|---|---|
| File | `Bonsai-8B-Q1_0.gguf`, 1,158,654,496 bytes, sha256 `284a335aa3fb2ced3b1b01fcb40b08aa783e3b70832767f0dd2e3fdfa134bd54` |
| Streamed per token | ~1.06 GB of weights |
| Engine | C + CUDA (sm_53), written for this board. No llama.cpp, no cuBLAS, no Python at runtime; the binary links only `libcudart`, `libgomp`, libc/libm/libpthread and libstdc++ |
| Accuracy | every GEMV kernel bit-exact against a host reference of the PrismML llama.cpp fork's Q1_0 arithmetic; greedy streams frozen as goldens and enforced by a gate on every change |

## Demo

Both clips play in real time: each token appears at the moment the engine
produced it, using the per-step times from the run's own log. Loading
(weight repack + upload) and prompt processing are shown at their measured
durations too.

**English** (seed 7, 90 tokens, decode 7.7 tok/s)

![Bonsai-8B Q1_0 on a Jetson Nano, English prompt](doc/bonsai8b-q1-nano-chat-en.gif)

**Traditional Chinese** (seed 123, 211 tokens, decode 7.4 tok/s)

![Bonsai-8B Q1_0 on a Jetson Nano, Traditional Chinese prompt](doc/bonsai8b-q1-nano-chat-zh-tw.gif)

How the clips were made:

- Engine code is the committed `bonsai-jetson-nano` HEAD (4df60ed),
  built with `.auto/build_q1.sh`, on a Nano with `jetson_clocks`
  (GPU 921.6 MHz, CPU 1.479 GHz), L4T R32.5.1, CUDA 10.2.
- The prompts are "List the four seasons and one thing that changes in
  each." and "請列出四季，以及每一季會改變的一件事。", wrapped in the
  model's chat template with an empty think block. Prompt ids came from
  `llama-tokenize` against the same GGUF.
- Sampling uses the GGUF's own recommended settings: temperature 0.5,
  top-p 0.85, top-k 20.
- Both runs stopped on their own at end-of-sequence.
- The commands, as shown in the clips:

  ```sh
  OMP_NUM_THREADS=4 ./fwd8_q1 b8q1.gguf 151644 872 198 852 279 3040 15584 323 825 3166 429 4344 304 1817 13 151645 198 151644 77091 198 151667 271 151668 271 --gen 400 --sample 0.5 0.85 20 7
  OMP_NUM_THREADS=4 ./fwd8_q1 b8q1.gguf 151644 872 198 100792 114116 109080 3837 101034 104588 99377 99496 110946 99774 108008 1773 151645 198 151644 77091 198 151667 271 151668 271 --gen 400 --sample 0.5 0.85 20 123
  ```

- The text is the model's output, detokenized and not edited. The
  Chinese answer has 1-bit-model slips, kept as generated: "氣溫低於
  freezing", "雪fall", and two invented flower names.
- 7.7 and 7.4 tok/s are lower than the 8.29 headline below because these
  runs go deeper into the context (up to position 235, against 44 in the
  harness) and ran in a different hour on the same board, which drifts
  2–6 %. The headline is the greedy harness number.
- The transcripts and per-token times are literals in
  `pulsar_arm/tools/make_chat_gif.py`, so re-rendering can only reproduce
  these frames: `python3 pulsar_arm/tools/make_chat_gif.py
  bonsai8b-q1-nano-en bonsai8b-q1-nano-zh-tw` (needs Pillow ≥ 8.2 and
  ffmpeg).

## At a glance

| | |
|---|---|
| Board | Jetson Nano Gen1: Tegra X1, 4× Cortex-A57, one Maxwell SM (sm_53, 128 cores), 4 GB LPDDR4 shared by CPU and GPU; measured streaming ceiling 21.4 GB/s |
| Decode | **8.29 tok/s** (~121 ms/token), up from 2.13 at the first working build |
| Prefill | 8.27 tok/s (token by token through the decode path) |
| Startup | ~2.3–2.4 s (repack + upload + first forward) |
| Peak RSS | 342 MB; GGUF pages are dropped once each tensor is on the GPU |
| GPU memory | 1280 MB arena, 1158.9 MB used (weights, K/V cache, staging) |
| Bandwidth | ~8.8 GB/s of 21.4 (41 %): the GEMV is bound by POPC issue on the single SM, not by the bus |
| Conditions | cold harness run, prompt A (20 prompt ids + 24 greedy tokens), `jetson_clocks`, cpuidle c7 off |

## How it works

At load, the 1-bit weights are repacked into a 1280 MB GPU arena. After
that, a decode step runs almost entirely on the GPU: per layer, one CUDA
graph for the attention block and one for the FFN, then a GEMV over the
151,669-row output head. The CPU keeps the embedding lookup, a few small
passes and sampling (~5 ms/token together).

- **GEMV kernels** (`pq1_swar_sm53.cu`) multiply 1-bit weights by int8
  activations. The main one is a **T-MAC-style bit-sliced kernel**: the
  activations are split into 8 bit planes, each plane is ANDed with 32
  weight sign bits and counted with POPC, two rows per thread. The
  activation planes are built on the GPU, one warp per 128-element group.
  Three SWAR kernels (64, 128 and 256 rows per block) are kept as
  alternatives and parity references.
- **Merged tensors.** q, k and v are one 6144-row tensor; gate and up are
  one 24576-row tensor. Fewer launches, fuller waves on the one SM.
- **CUDA graphs** with fixed staging buffers, and spin scheduling for the
  host/device syncs.
- **Attention on the GPU:** per-head RMS norm, YaRN rope, K/V cache store,
  q·k, softmax, p·v and the o projection in one graph. The K/V cache lives
  in the arena.
- **FFN on the GPU:** gate+up, silu(g)·u with an `expf` that reproduces
  glibc's bit for bit, activation quantization and the down projection in
  one graph.

Time per token (`PQ1_KTIME=1`): attention graph 0.90 ms × 36 layers, FFN
graph 2.20 ms × 36, head 8.4 ms. The T-MAC kernel is ~85 % of decode, and
it is near its POPC floor (~68 ms/token of ~120).

Every step above was kept only after an order-balanced A/B on the same
board in the same hour, with all three accuracy gates green on a fresh
build. Each has a switch to turn it off (`PQ1_FFN=0`, `PQ1_ATT=0`,
`PQ1_TM=0`, `PQ1_GRAPH=0`, ...). The full record, including every idea
that was measured and rejected, is `.auto/HANDOFF_NANO_Q1_0.md`.

## Accuracy

`.auto/checks.sh` rebuilds from the working tree and runs three gates.
They are never overridden:

1. `pq1_parity`: all four GEMV kernels bit-exact against the host
   reference (`pq1_host.c`) at K = 4096 (256/256 rows) and K = 12288
   (512/512 rows).
2. The 44-token greedy stream on prompt A equals `.auto/golden_q1_A.txt`.
3. The 26-token greedy stream on held-out prompt B equals
   `.auto/golden_q1_B.txt`.

`PQ1_ATT_CHECK=1` and `PQ1_FFN_CHECK=1` rerun the host attention or FFN
after every GPU layer and compare bitwise (0 mismatches over 1548
layer-steps).

**Divergence from the fork, disclosed.** The engine quantizes activations
to **Q8_K** (256-wide blocks, fp32 scale), a choice inherited from the
earlier 2-bit path. The fork's Q1_0 dot product uses **Q8_0** (32-wide
blocks, f16 scale). The engine is therefore not logit-identical to the
fork. On prompt A it produces the same greedy stream as the fork for all
25 generated tokens, but the logits differ slightly: at step 19 the top
logit is 19.8211 here and 19.8919 in the fork. The goldens freeze the
engine's own exact path. Note that the fork's own greedy stream is not
unique either: on a long Traditional Chinese prompt, flash attention on vs
off first differs at token 36, and 20 threads vs 1 thread at token 32.

## Build and run

Everything builds on the Nano (CUDA 10.2, gcc 7.5). The exact recipe is
`.auto/build_q1.sh`; in short:

```sh
cd pulsar_arm/bonsai2
nvcc -gencode arch=compute_53,code=sm_53 -O3 -maxrregcount=48 --use_fast_math \
     -Xptxas -O3,-dlcm=ca -std=c++11 -c pq1_swar_sm53.cu -o pq1_swar_sm53.o
CF="-O2 -fopenmp -mcpu=cortex-a57 -DSTANDALONE"
INC=-I/usr/local/cuda-10.2/targets/aarch64-linux/include
gcc $CF -ffp-contract=off -DPQ1_REAL_CUDA -c pq1_host.c -o pq1_host.o
gcc $CF -DPQ1_USE_CUDA -DBQ1_TIME -include pq1_fwd8_shim.h -c fwd8.c -o fwd8_q1.o
gcc $CF -DPQ1_USE_CUDA $INC -c pq1_fwd8_glue.c -o pq1_glue.o
g++ fwd8_q1.o pq1_host.o pq1_glue.o pq1_swar_sm53.o -o fwd8_q1 \
    -L/usr/local/cuda-10.2/targets/aarch64-linux/lib -lcudart -lm -lpthread -fopenmp
```

Run:

```sh
sudo jetson_clocks                     # fixed clocks; numbers above assume this
OMP_NUM_THREADS=4 ./fwd8_q1 Bonsai-8B-Q1_0.gguf <prompt ids...> --gen N [--sample temp top_p top_k seed [min_p]]
```

Without `--sample` the engine decodes greedily. The engine takes and
prints token ids; there is no tokenizer on the board. Tokenize the chat
template on another machine with `llama-tokenize` from a llama.cpp build
that knows Q1_0 (the PrismML fork), or with
`pulsar_arm/bonsai2/tokdetok.cpp` built against it, which also turns the
printed ids back into text.

## Measurement harness

Run from a host with ssh access to the board (the scripts default to
`pesi@100.70.245.36`; set `NANO=` to change it). They copy the sources,
build on the board and run there.

```sh
bash .auto/measure.sh                 # build -> prompt A -> METRIC decode_tps / prefill_tps / peak_rss_mb lines
bash .auto/checks.sh                  # the three accuracy gates (needs a prior measure.sh)
bash .auto/ab_pair.sh "PQ1_TM=0" 2    # order-balanced A/B of an engine switch, 2 pairs
PQ1_EXTRA="PQ1_KTIME=1" bash .auto/measure.sh   # engine env vars reach the board only through PQ1_EXTRA
```

Useful engine switches: `PQ1_KTIME=1|2|3` (per-shape / per-phase / API
timing), `PQ1_CPUTIME=1` (per-pass CPU budget), `PQ1_STATS=1` (GPU memory
and graph counters at exit, the proof that a code path actually ran),
`PQ1_TRACE=1`.

The board drifts 2–6 % from hour to hour, so a number is only compared
with another from the same A/B job, never with one from another run.

## Limitations and work in progress

- Context is capped at 512 tokens, and the prompt is processed one token
  at a time through the decode path (no batched prefill).
- One sequence at a time.
- **In progress:** a 4096-token context with an f16 K/V cache whose
  attention arithmetic follows the fork's flash-attention decode path
  (`.auto/FORK_F16_FA_SPEC.md`). Decode speed will then also be reported
  at depth, on a 3197-token Traditional Chinese prompt
  (`.auto/prompt_zh_tw.txt`).

## Layout

```
pulsar_arm/bonsai2/pq1_swar_sm53.cu   GPU kernels: T-MAC + SWAR GEMVs, attention, FFN, quant
pulsar_arm/bonsai2/pq1_fwd8_glue.c    repack, GPU arena, merged tensors, CUDA graphs, dispatch, timers
pulsar_arm/bonsai2/fwd8.c             engine: layer loop, CPU passes, sampler (compiled with pq1_fwd8_shim.h)
pulsar_arm/bonsai2/pq1_host.c         host reference the kernels must match bit for bit; OpenMP repack
pulsar_arm/bonsai2/pq1_parity.c       parity gate: ./pq1_parity <rows> <K> all
pulsar_arm/bonsai2/bq1_*, pq2_*       earlier Nano paths (sign-projected, 2-bit), kept for reference
pulsar_arm/tools/make_chat_gif.py     demo renderer; transcripts and timings are literals
.auto/                                harness scripts, goldens, HANDOFF_NANO_Q1_0.md, ideas.md (history)
autoresearch/                         per-loop results tables and discarded patches
doc/                                  demo GIFs; OTHER_ENGINES.md
```

## Other engines in this repository

This repository started as pure-assembly inference engines. They are
still here and still work: a single static AArch64 binary that runs
Gemma-3 on a Raspberry Pi 4 with no libc, the C + NEON Bonsai port that
runs Ternary-Bonsai 8B and 27B on a DGX Spark CPU, and the original x86-64
Gemma engines. See [doc/OTHER_ENGINES.md](doc/OTHER_ENGINES.md) and
[pulsar_arm/bonsai2/README.md](pulsar_arm/bonsai2/README.md).
