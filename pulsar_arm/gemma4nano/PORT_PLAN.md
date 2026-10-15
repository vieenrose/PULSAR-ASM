# Gemma 4 E2B (text only) on the Jetson Nano: port plan

Branch `gemma-4-jetson-nano`, started from `k2-horizon-jetson-nano` (4f98ea0), which already
contains the x86-64 Gemma 4 E2B engine and oracles (`pulsar_asm/`) and the Nano
asm + C + CUDA engine for K2-Horizon-0.9B (`.auto/rescue/k2_cuda.S`,
`pulsar_arm/k2horizon/gpu_v52/`). Status: **plan and measurements only, no engine code yet.**

## Source of truth

- Checkpoint: `google/gemma-4-E2B-it` (not gated, one 10,246,621,918-byte `model.safetensors`
  that also carries the audio and vision towers, which this port ignores). Downloaded to
  `/home/user/g4/` with `config.json`, tokenizer and chat template.
- Math: `pulsar_asm/tools/ref_gemma4_np.py` (float64 NumPy mirror of HF
  `modeling_gemma4.py`, already checked against HF). It is the spec for every stage below.

## Architecture, from config.json and the reference

35 layers, hidden 1536, 8 q-heads, **1 kv-head**, vocab 262144, tied head, GELU(tanh).

| item | value |
|---|---|
| layer pattern | 4 sliding (window 512, head_dim 256) then 1 full (head_dim 512); 28 sliding + 7 full |
| KV sharing | layers 15..34 reuse the cache of layer 13 (sliding) / 14 (full); only 15 layers store KV (12 sliding, 3 full) |
| MLP | GeGLU, intermediate 6144; **12288** on the 20 KV-shared layers (double-wide) |
| attention | q/k RMS norm per head (with weight), v RMS norm without weight, scaling 1.0, no biases |
| RoPE | sliding theta 1e4 full rotary; full layers theta 1e6, "proportional", partial 0.25 (zero-padded inv_freq, pairs `(i, i+hd/2)`) |
| per-layer inputs (PLE) | `ple_in[l] = (RMSNorm(proj(x)/sqrt(1536)) + ple_embed[token][l]*sqrt(256))/sqrt(2)`; per layer `x += RMSNorm(ple_proj(gelu(ple_gate(x)) * ple_in[l]))`, then `x *= layer_scalar` |
| head | final RMS norm, tied embedding dot, then `30*tanh(logit/30)` soft-capping |
| embedding scale | `x0 = embed[token] * sqrt(1536)` |

## Size and speed budget on the Nano (computed from the config)

| | params | Q4_0 (18 B / 32 weights) |
|---|---|---|
| transformer layers (attention 278M, MLP 1557M, PLE gates 42M) | 1.877 B | 1.06 GB |
| tied embedding / output head | 0.403 B | 0.23 GB |
| **streamed per token** | **2.28 B** | **1.28 GB** |
| PLE token table (262144 x 8960) | 2.349 B | 1.32 GB (bf16: 4.70 GB) |

- Only one 8960-value row of the PLE table is read per token (5,040 B in Q4), so it can sit
  in a memory-mapped file on the CPU side and never enters the GPU or the streaming budget.
- GEMV shapes: rows 256..12288 and the 262144-row head; columns 1536, 2048, 4096, 6144, 12288,
  all multiples of 256, so the K2 kernel's Q8_K activation layout (292 B per 256) applies as is.
- KV cache in f32: 25 MB at 1024 context, 38 MB at 2048. Negligible.
- Decode upper bounds from bandwidth alone, 1.28 GB per token: 5.6 tok/s at the 7.2 GB/s the K2
  kernel reaches, 8.3 tok/s at 10.7 GB/s (best probe), 13 tok/s at 17 GB/s (the Bonsai kernel's
  rate). **Expect roughly 4-6 tok/s with the current kernel design**; this is an estimate.
- Memory: the board shows 3,964 MB total, ~3,400 MB available, 8 GB swap file, 20 GB free disk.
  GPU-resident weights 1.3 GB plus a 1.3 GB mapped PLE table fit; the bf16 blob (9.3 GB) does not.

## Design decisions

1. **Reuse, don't rewrite**: the K2 Nano engine's driver path (dlopen `libcuda`, JIT PTX, mapped
   pinned staging, timing-free event, 4-stream readback) and its Q4 x Q8_K GEMV kernel carry over.
   The kernel needs a variable-shape launch (any rows, cols multiple of 256).
2. **Arithmetic is defined by our own Q4 reference**, as for K2: Q4_0 weights, Q8_K activations,
   exact int32 dots, fixed-order float accumulation. Bit-exactness is claimed against that
   reference (a scalar C file) and frozen as goldens, not against HF bf16. Agreement with HF is
   reported separately (argmax agreement and logit distance).
3. **Transcendentals**: the reference uses float64 `exp`/`tanh` for GELU, softmax and the
   logit cap. glibc's `expf` is already mirrored in the K2 asm; GELU(tanh), the f64 softmax
   `exp` and `tanh` need a deterministic implementation shared by the C reference and the asm,
   defined in float32/double with fixed operation order, not by calling libm.
4. **GPU does the GEMVs; the host asm does the rest** (norms, RoPE, QK/V norms, attention over
   the sliding window or full cache, GeGLU, PLE, soft-cap, argmax). Attention is cheap here:
   one kv-head, and sliding layers read at most 512 positions.
5. **Precision of the head and PLE table** is a quality decision for the user (see below).

## Milestones

| # | deliverable | check |
|---|---|---|
| M0 | checkpoint, config, tokenizer on disk; budget (this file) | done except the 10.2 GB download |
| M1 | `convert_g4_q4.py`: safetensors -> Q4 blob + manifest, PLE table in a separate mmap file | tensor-by-tensor round trip against the checkpoint |
| M2 | scalar C reference of the Q4 forward (`ref_g4_q4.c`) + golden greedy streams on 2 prompts (en, zh-TW) | matches `ref_gemma4_np.py` fed the same dequantized weights |
| M3 | GPU GEMV for every shape, bit-identical to the C reference per layer | per-layer parity test, like K2's startup self-test |
| M4 | asm host: all non-GEMV stages, PLE, soft-cap, sampler | golden stream bit-identical |
| M5 | end-to-end on the Nano: bench, ceiling-prompt GIFs, README | gate + A/B timing |

## Open decisions for the user

- **Output head**: Q4 (0.23 GB, fastest) or Q8 (0.43 GB, safer logits; the head decides the argmax).
- **PLE table**: Q4 (1.3 GB mapped) or Q8 (2.5 GB, tight next to the GPU arena).
- **Context cap** (1024 or 2048 positions; KV is cheap, but prefill is token by token).
- Whether the K2 asm engine is forked (copy and edit) or generalized into one engine with a
  per-model table; forking is faster to a first result and is the plan unless told otherwise.

## Update 2026-10-15: QAT checkpoint and Google's own Q4_0 file

User decisions: Q4, using the **QAT** release, context cap 2048.

- `google/gemma-4-E2B-it` (non-QAT) was downloaded and verified (sha256 `2db5482b…c550`),
  then deleted for disk space; it is no longer the source.
- Source now: `google/gemma-4-E2B-it-qat-q4_0-unquantized` (bf16 weights trained for Q4_0,
  identical text_config) and Google's quantized release `google/gemma-4-E2B-it-qat-q4_0-gguf`
  (`gemma-4-E2B_q4_0-it.gguf`, 3.35 GB). Both ungated, in `/home/user/g4qat/`.
- Header of Google's "Q4_0" GGUF (read by HTTP range): **only the layer projections are Q4_0**
  (q, k, v, o, gate, up, down, PLE gate and proj). The token embedding / tied head and the PLE
  table are **Q6_K**, `per_layer_model_proj` is F16, norms and layer scalars F32.
  So "all Q4_0" would quantize the head and the PLE table more coarsely than Google does.

Revised M1: take the weights from Google's GGUF as they are (this *is* the official Q4 model),
and use the QAT bf16 file only as the reference: re-quantizing its layer tensors with ggml's
Q4_0 rule must reproduce the GGUF's Q4_0 bytes exactly, which checks both files and our quantizer.

Revised budget: GPU 1.06 GB (Q4_0 layers) + 0.33 GB (Q6_K head) + 0.03 GB (F16) = 1.42 GB;
PLE table 1.93 GB Q6_K, memory-mapped, one 7.4 KB row per token. Fits the 3.4 GB available.
Cost: one more GPU kernel (Q6_K GEMV for the 262144-row head) besides the K2 Q4_0 kernel.

## M1 result (2026-10-15)

- Files verified against Hugging Face's LFS sha256:
  `model.safetensors` (QAT bf16) 10,208,852,878 B, `33fe0cec…0137`;
  `gemma-4-E2B_q4_0-it.gguf` 3,349,516,256 B, `fa401b55…6634`.
- `g4_gguf.py`: 541 tensors tile the GGUF exactly (header 15,815,552 B + tensors +
  980 B alignment padding = file size).
- `g4_q4check.py`: **275/275 Q4_0 tensors are byte-identical** to ggml's
  `quantize_row_q4_0_ref` applied to the QAT bf16 weights, all with strict C rounding
  (`x*id` rounded to f32, then `+8.5f`), none needing FMA contraction. Runs in 13 s.
  So the GGUF's Q4_0 layers are exactly "the QAT weights in Q4_0", and our quantizer
  reproduces Google's.
- Not yet cross-checked: the two Q6_K tensors (head, PLE table) and the F16 projection;
  they will be checked by dequantizing against the bf16 source (error bound) in M2.

## M2 result (2026-10-15)

- `g4_dequant.py` (Q4_0 / Q6_K / F16 dequantization, ggml order) and the remaining M1 checks:
  F16 `per_layer_model_proj` is the bf16 source exactly; the two Q6_K tensors are the bf16
  source at 1.4 % relative RMS error (2000 random rows each), as expected for 6-bit.
  GGUF norms are stored as is (no +1 offset); `rope_freqs` (64 ones, then 1e30) is the
  proportional rope of HF.
- `g4_np.py`: float reference on the dequantized GGUF weights (same equations and precision
  as the x86 port's HF-checked `ref_gemma4_np.py`). English prompt: a correct four-season
  answer that ends cleanly on `<turn|>`.
- `g4ref.c`: the **bit-exact reference** that defines the engine arithmetic (contract in its
  header: Q8_K activations, K2's Q4_0 GEMV, integer Q6_K head, F16 fmaf chains, float64 stages
  with an own `g4_exp`, baked rope tables from `g4_rope.py`, greedy argmax on raw logits).
  `g4_exp` is within 1 ulp of libm over the whole normal range (mean 0.10 ulp, 19.5 M points).
  Deterministic: identical output at 1, 3 and 20 OpenMP threads.
- Agreement with the float reference: English, all generated tokens identical (the one
  differing argmax is a prompt position, a near-tie 26.40 vs 26.42); Traditional Chinese,
  identical up to generated step 32, then a different but equally correct answer: the
  expected effect of quantizing activations to Q8_K.
- Goldens (from `g4ref`): `golden_en.txt` (97 steps, ends on `<turn|>`),
  `golden_zh.txt` (222 steps, `--gen 200`); prompts and ids in `prompts.json`.

## M3 result (2026-10-15): GPU GEMVs bit-identical to g4ref.c

`g4gemv_test` (on the Nano, driver API through dlopen, no toolkit at runtime) compares
every GPU GEMV with the g4ref.c function itself (included, main compiled out), on the real
weights, with three activation vectors each (two Gaussian-like, one with 1 % outliers x40):

- Q4_0, K2's `q4_gemv` kernel (`g4_ptx_q4.h`): **275/275 tensors bit-identical**.
- Q6_K tied head, all 262144 rows: v1..v4 all **bit-identical** (`g4_kernels.cu`).
- F16 `per_layer_model_proj` (`f16_gemv_t`, weights stored transposed): 8960/8960 identical.

Kernel times, warm GPU at 921.6 MHz (clocks not pinned: no sudo for jetson_clocks; the first
run of a session starts at 76.8 MHz and must be discarded):

| kernel | time | effective |
|---|---|---|
| Q4_0 layers, per shape | 0.03-1.10 ms | 8.6-11.6 GB/s |
| head Q6_K v1 (byte loads) | 99.4 ms | 3.3 GB/s |
| head v2 (212-byte padded blocks, u32 loads, activations in registers) | 57.1 ms | 5.8 GB/s |
| head v3 (v2 + interleaved block reductions) | 56.0 ms | 6.0 GB/s |
| **head v4** (4 values per u32 decoded with byte masks, -32 offset hoisted) | **41.4 ms** | **8.1 GB/s** |

The head was issue-bound, not memory-bound (one SM, ~100 instructions per block per lane in
v1-v3); v4 cut the decode work. Per-token GEMV estimate from the layer schedule: layers
108.7 ms + head 41.4 ms = **150 ms, a 6.7 tok/s ceiling from GEMVs alone**, before the CPU
stages and driver overhead. The engine will use `q6k_gemv4`; v1-v3 stay as parity records.

## M4 first light (2026-10-15): engine output = goldens, bit for bit

`g4eng.c` runs g4ref.c's own forward() (shared code) with its three GEMV hooks pointed
at the M3 kernels. On the Nano: **golden_en.txt (97 steps) and golden_zh.txt (222 steps)
reproduced byte for byte**, i.e. the aarch64 build and the GPU give exactly the x86
reference's tokens and printed logits.

First version (all CPU stages in C, one synchronous H2D + launch + sync + D2H per GEMV):
load 4.6-6.4 s, 1,456 MB on the GPU, peak RSS 700 MB; decode 3.22 tok/s (en, 76 steps),
2.96 tok/s (zh, 200 steps); prefill 2.76 tok/s. Next: profile and optimize.

Tried and dropped (2026-10-15): head v5, splitting the six-bit values into 16-bit halves
with byte permutes and mad.wide.s16 (meant to map to one XMAD per pair). Bit-identical,
but 44.3 ms vs v4's 41.9 ms: nvcc kept the byte extracts and added shifts. v4 stays.

Tried and dropped: q4i with an 8- or 16-block load ring (64 / 116 registers). Fewer
resident warps made most shapes slower (e.g. ffn_gate 12288x1536: 0.78 -> 0.94 / 1.11 ms);
only down at 12288 columns improved (16-deep: 1.018 ms vs K2's 1.048), ~0.6 ms per token,
not worth a third kernel.

## Status at the end of the night (2026-10-16)

Engine complete for single-sequence text generation, greedy and sampled, every stage
bit-identical to g4ref.c (gates.sh: host asm test, GPU kernel test, three goldens).

| step | en decode tok/s |
|---|---|
| first light (C host, synchronous GEMV round trips) | 3.23 |
| merged launches | 3.49 |
| K2 driver path | 3.97 |
| NEON exp / GeGLU / attention | 4.44 |
| MLP chain on the GPU | 4.98 |
| 8-head attention pass, NEON RMS norm and Q8_K | 5.1-5.3 |
| q4i row-per-lane Q4_0 kernel (220 of 275 tensors) | 5.7-5.9 |
| q6i row-per-lane Q6_K head (41.7 -> 18.0 ms) | 6.8-6.9 |

Final measurement (en gate prompt): load 4.05 s, 1,474 MB on the GPU, peak RSS 729 MB,
prefill 6.54 tok/s, decode 6.83 tok/s. Demo prompts: 6.17 (291 steps) and 6.56 tok/s
(117 steps); 5.5-5.7 tok/s over 500 steps. Per token ~146 ms: Q4_0 GEMVs + MLP chain ~103,
head ~19, F16 projection ~4, CPU ~20.

Open, measured or estimated:
- down at 12288 columns stays on K2's kernel (~21 ms/token over 20 layers); a split-K
  q4i needs the ordered float chain moved to an epilogue.
- ~5 GPU synchronizations per layer (~14 ms/token of overhead); the per-layer-input chain
  could join the GPU like the MLP (~2-4 ms, estimate). Moving norms to the GPU needs a
  parallel-friendly sum order, i.e. a change of the reference arithmetic and new goldens.
- Prefill is token by token.
