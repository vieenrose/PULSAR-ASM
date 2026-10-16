# PULSAR-ASM: Ternary-Bonsai-2 on a Jetson Nano

PrismML's **Ternary-Bonsai-2** family (2B, 4B, 9B) running on a **Jetson
Nano Gen1**, the 2019 4 GB board, on its 128-core Maxwell GPU. The
models are the `PTQ1_0` GGUFs: ternary weights packed 128 to a 28-byte
block with one f16 scale, 1.75 bits per weight. The architecture is `qwen35`,
a hybrid that puts one full-attention layer after every three Gated
DeltaNet (linear attention) layers.

| model | GGUF bytes | on the GPU | prefill | decode |
|---|---|---|---|---|
| Ternary-Bonsai-2-2B | 429,183,456 | 476 MB | **44.1 tok/s** | **25.1 tok/s** |
| Ternary-Bonsai-2-4B | 949,549,920 | 1064 MB | **17.9 tok/s** | **11.6 tok/s** |
| Ternary-Bonsai-2-9B | 1,997,486,976 | 2009 MB | 6.2 tok/s | 5.7 tok/s |

Prefill and decode are measured on English prompt A (24 prompt tokens),
running to the model's own end of turn: 100 and 82 decode steps for the
2B and 4B, from one run each at commit 1767589 (batched prefill, 8-token
chunks). The 9B row is from the earlier build (before the 8-POPC change
and the batched prefill) and is not re-measured: the board's SD card
holds only one of the 4B and 9B at a time. Conditions are `jetson_clocks`
(GPU 921.6 MHz, CPU 1.479 GHz), L4T R32.5.1, CUDA 10.2, greedy decoding,
f16 K/V cache. The 2B and 4B decode streams 10.8 and 11.0 GB/s of weights
(GGUF bytes times tok/s); the 9B streams 11.5 GB/s. The GEMV is
issue-bound at about 13 GB/s on this GPU, below the 21.9 GB/s bus ceiling,
because ternary dot products cost more ALU work per byte than the 1-bit
ones below.

| | |
|---|---|
| Engine | `pulsar_arm/bonsai2/fwd35_sm53.cu`, one CUDA file, written for this board. No llama.cpp, no cuBLAS, no Python at runtime |
| Accuracy | **bit-exact against the PrismML llama.cpp fork**: every demo stream below equals the fork's own greedy stream over its full length, and `.auto/checks_b2.sh` gates 48 tokens on two prompts per model against fork goldens |
| Sources | [prism-ml/bonsai-2-small](https://huggingface.co/collections/prism-ml/bonsai-2-small) |

sha256: 2B `36061ad406df8a4903a6db005be350c50fabfaf717f9ef0acf4b76adcd9929fa`,
4B `56711263ec9a9ffeaa74a3a0e58dece8127cf9268404746c721113ca76994d60`,
9B `3561df7ac7d26a71ffa991ac2db740b10841c3a350aa2676ccd3ff5d428b4e8b`.

## Demo

The same two prompts on each model, one English and one Traditional
Chinese. The clips were recorded before the 8-POPC and batched-prefill
changes; their status lines show that build's numbers (the table above has
the current 2B and 4B figures). Re-shooting them is still to do. The clips play in real time: each token appears at the moment
the engine produced it, using the per-step times from the run's own log.
Loading (weight repack + upload) and prompt processing are shown at their
measured durations too.

### Ternary-Bonsai-2-2B

**English** (100 tokens, decode 22.1 tok/s)

![Ternary-Bonsai-2-2B on a Jetson Nano, English prompt](doc/bonsai2-2b-nano-chat-en.gif)

**Traditional Chinese** (350 tokens, decode 21.1 tok/s)

![Ternary-Bonsai-2-2B on a Jetson Nano, Traditional Chinese prompt](doc/bonsai2-2b-nano-chat-zh-tw.gif)

### Ternary-Bonsai-2-4B

**English** (82 tokens, decode 9.9 tok/s)

![Ternary-Bonsai-2-4B on a Jetson Nano, English prompt](doc/bonsai2-4b-nano-chat-en.gif)

**Traditional Chinese** (404 tokens, decode 9.25 tok/s)

![Ternary-Bonsai-2-4B on a Jetson Nano, Traditional Chinese prompt](doc/bonsai2-4b-nano-chat-zh-tw.gif)

### Ternary-Bonsai-2-9B

**English** (99 tokens, decode 5.7 tok/s)

![Ternary-Bonsai-2-9B on a Jetson Nano, English prompt](doc/bonsai2-9b-nano-chat-en.gif)

**Traditional Chinese** (92 tokens, decode 5.7 tok/s)

![Ternary-Bonsai-2-9B on a Jetson Nano, Traditional Chinese prompt](doc/bonsai2-9b-nano-chat-zh-tw.gif)

How the clips were made:

- The engine is `fwd35_sm53.cu` at the commit that adds these clips,
  built on the Nano with
  `nvcc -O3 -gencode arch=compute_53,code=sm_53 --fmad=false -Xcompiler -fopenmp,-ffp-contract=off`.
- The prompts are "List the four seasons and one thing that changes in
  each." and "請列出四季，以及每一季會改變的一件事。", wrapped in the
  model's chat template with an empty think block. Prompt ids came from
  the fork's `llama-tokenize` on the same GGUF.
- Decoding is greedy. `E_STOP=1` ends the run at `<|im_end|>`, which
  every run reached on its own. `E_STEPMS=1` logs the step times the
  clips replay.
- **Every stream equals the fork's.** `fork_greedy.cpp` (the fork's
  own CPU graph, f16 K/V cache, flash attention) was run for each clip's
  full length: 2B 101 / 351, 4B 83 / 405 and 9B 100 / 93 ids are
  identical. The text is decoded with each GGUF's own vocabulary and is
  not edited.
- Decode steps: 2B 44–68 ms, 4B 99–125 ms, 9B 171–210 ms. Steps grow
  with depth only through the full-attention layers (one in four), so
  the long Chinese answers lose 5–7 %. Loading is 5.1 s, 11.1 s and
  23.5 s: the ternary planes are repacked on the CPU before upload.
- The 2B's Chinese answer has slips, kept as generated, for example
  "花朵開墜" and summer "植物葉片變黃". The 4B writes "收获" (simplified
  获) once.
- The board's 14 GB card holds the 2B plus one of the 4B and 9B at a
  time, so the 4B clips were recorded before the 9B was copied in.
- Re-render with `python3 pulsar_arm/tools/make_chat_gif.py
  bonsai2-2b-nano-en bonsai2-2b-nano-zh-tw bonsai2-4b-nano-en
  bonsai2-4b-nano-zh-tw bonsai2-9b-nano-en bonsai2-9b-nano-zh-tw` (needs
  Pillow ≥ 8.2 and ffmpeg). Transcripts and step times are literals.

### Speculative decoding: measured out on this board

The Bonsai-2 releases come with MTP heads and DFlash draft models. On the
Nano neither can help. The ternary GEMV is bound by instruction issue per
(row, token): ~286 SASS instructions per 128-weight row group, ~180 of
them the POPC/LOP3 core. Checking k drafted tokens in one batched pass
therefore costs nearly k single-token steps. Measured with the engine's
own batched forward on the 4B (`E_PCHUNK`), the per-token cost of a
k-token pass against single tokens is 1.81× at k = 2, 1.62× at 3 and
1.36× at 8. A draft would need near-perfect acceptance just to break
even. Those figures are for the POPC batch kernel. The fp16 batch kernel that
prefill now uses (`k_hg`, ideas row 62) runs 8 tokens 1.5–1.7× faster per
token than single tokens on the 4B's GEMV shapes, so the k-token verify
cost is not re-measured with it. The DFlash drafts (~1.57 GB) would not
fit next to the 4B or 9B anyway. `E_PCHUNK` defaults to 8 on sm_53 and
sm_61; `E_HG=0` falls back to the POPC batch kernel.

### Build, run and gates

```sh
nvcc -O3 -gencode arch=compute_53,code=sm_53 --fmad=false \
     -Xcompiler -fopenmp,-ffp-contract=off -o fwd35 fwd35_sm53.cu -lgomp
sudo jetson_clocks
./fwd35 Ternary-Bonsai-2-4B-PTQ1_0.gguf 400 <prompt ids...>   # prints greedy ids; tok/s on stderr
bash .auto/checks_b2.sh 2B 4B 9B    # from the host: sync, build on the Nano, fork-golden gates
```

Switches: `E_STOP=1`, `E_STEPMS=1`, `E_PCHUNK=n` (prefill tokens per
batched pass, default 8), `E_HG=0` (POPC batch kernel instead of the fp16 one), `E_GRAPH=0` (no CUDA graphs), `E_CTX` (default 4096).
The host reference `fwd35_ref.c` (plain C, same bits on x86 and the
Nano's A57) and the per-kernel record are in `.auto/ideas.md` row 59.

## Bonsai-8B Q1_0 engine (previous generation)

The pages below document the engine this repository was built around:
PrismML's **Bonsai-8B** (Qwen3-8B architecture) in the `Q1_0` format,
true 1-bit weights (one sign bit per weight plus one f16 scale per
128-weight block), at **16.5 tokens/s** on the same board. It also runs
the first-generation Ternary-Bonsai-4B (`-DPQ1_TERN -DBONSAI_4B`,
14.5 tok/s). Its demo clips were replaced by the Bonsai-2 ones above and
remain in git history (commit d2b7ed9).

| | |
|---|---|
| File | `Bonsai-8B-Q1_0.gguf`, 1,158,654,496 bytes, sha256 `284a335aa3fb2ced3b1b01fcb40b08aa783e3b70832767f0dd2e3fdfa134bd54` |
| Streamed per token | ~1.06 GB of weights |
| Engine | C + CUDA (sm_53), written for this board. No llama.cpp, no cuBLAS, no Python at runtime; the binary links only `libcudart`, `libgomp`, libc/libm/libpthread and libstdc++ |
| Accuracy | every GEMV kernel bit-exact against a host reference of the PrismML llama.cpp fork's Q1_0 arithmetic; greedy streams frozen as goldens and enforced by a gate on every change |

### At a glance

| | |
|---|---|
| Board | Jetson Nano Gen1: Tegra X1, 4× Cortex-A57, one Maxwell SM (sm_53, 128 cores), 4 GB LPDDR4 shared by CPU and GPU; measured streaming ceiling 21.9 GB/s |
| Decode | **16.51 tok/s** (60.6 ms/token), up from 2.13 at the first working build |
| Prefill | 16.19 tok/s (token by token through the decode path) |
| Startup | ~2.4–2.5 s (repack + upload + first forward) |
| Peak RSS | 325–773 MB, set by the board's page-cache state rather than the code: unchanged binaries read 596, 680, 730 and 770 MB over one session, and in the A/B for 11806d3 both arms read ~770 MB. GGUF pages are dropped once each tensor is on the GPU |
| GPU memory | 1280 MB arena, 1025.1 MB used (weights, K/V cache, staging) |
| Bandwidth | ~17.1 GB/s of 21.9 (78 %); the GEMV kernels alone run at 16.5–20.6 GB/s, up to 94 % of the ceiling |
| Conditions | cold harness run, prompt A (20 prompt ids + 24 greedy tokens), `jetson_clocks`, cpuidle c7 off |

### How it works

At load, the 1-bit weights are repacked into a 1280 MB GPU arena. After
the first token, a whole decode step is **one CUDA graph**: per layer,
residual add + RMS norm + Q8_K activation quantization, the merged q+k+v
GEMV, attention, the o projection, the FFN, and finally the 151,669-row
output head and, for greedy decoding, the argmax over its logits, with
the hidden state kept on the GPU throughout. The CPU copies in the
embedding row and the rope table; greedy decoding copies out only the
winning token id and logit, sampling copies out the full logits and draws
on the host.

- **GEMV kernel** (`pq1_gemv_hs` in `pq1_swar_sm53.cu`) multiplies 1-bit
  weights by int8 activations. It is **group-stationary**: each lane keeps
  one 128-wide K group's activation bit planes in registers (sign plus 7
  magnitude planes) and streams weight rows past them. It reduces each
  row-group product with a Harley-Seal carry-save tree of LOP3 full adders
  and POPC, tuned to sm_53's quarter-rate POPC pipe. The integer sums are
  exact, and the fp epilogue adds the per-group products in the host
  reference's order; the K=4096 shapes defer
  the row sums over three batches so 18 lanes reduce at once instead of 6.
  The earlier bit-serial T-MAC kernel and three SWAR
  kernels are kept as fallbacks and parity references.
- **Merged tensors.** q, k and v are one 6144-row tensor; gate and up are
  one 24576-row tensor. Fewer launches, fuller waves on the one SM.
- **Fused activation prep.** The RMS-norm and silu kernels quantize to
  Q8_K and build the GEMV's bit planes in the same launch. The RMS sum is
  computed as a parallel double tree, and it is used only when it rounds
  to the same float as fwd8's serial chain; otherwise the kernel runs the
  serial chain.
- **Attention on the GPU:** per-head RMS norm, YaRN rope, K/V cache store,
  q·k, an online softmax (prefix maxima by warp shuffle scan) and p·v. The
  K/V cache lives in the arena. The per-head norm and rope run one head
  per warp over five blocks (10 µs per layer).
- **FFN on the GPU:** gate+up, silu(g)·u with an `expf` that reproduces
  glibc's bit for bit, activation quantization and the down projection.

GEMV time per token (nvprof, prompt A): q+k+v 190 µs, o 138 µs, gate+up
681 µs, down 406 µs, each × 36 layers, plus the head at 4.2 ms (bfabc8a).
That is ~55 ms of the ~61 ms step (~91 %), at 16.5–20.6 GB/s per kernel
against the 21.9 GB/s streaming ceiling; the big shapes keep their whole
row batch's loads in flight at 24 resident warps. The rest is attention, norms and the silu `expf`.

Every step above was kept only after an order-balanced A/B on the same
board in the same hour, with all three accuracy gates green on a fresh
build. Each has a switch to turn it off (`PQ1_FFN=0`, `PQ1_ATT=0`,
`PQ1_TM=0`, `PQ1_HS=0`, `PQ1_TOK=0`,
`PQ1_GRAPH=0`, ...). The full record, including every idea
that was measured and rejected, is `.auto/HANDOFF_NANO_Q1_0.md`.

### Accuracy

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

### Build and run

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

### Measurement harness

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

### Limitations and work in progress

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
pulsar_arm/bonsai2/fwd35_sm53.cu      Bonsai-2 (qwen35, PTQ1_0) GPU engine: GEMV, GDN, attention, graphs
pulsar_arm/bonsai2/fwd35_ref.c        Bonsai-2 host reference, bit-exact vs the fork; glibc_mathf.h
pulsar_arm/bonsai2/fork_greedy.cpp    the fork's own greedy stream (goldens, full-length demo checks)
pulsar_arm/bonsai2/pq1_swar_sm53.cu   GPU kernels: hs, T-MAC + SWAR GEMVs, attention, FFN, rms/quant
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
