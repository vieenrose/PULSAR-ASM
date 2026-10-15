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
