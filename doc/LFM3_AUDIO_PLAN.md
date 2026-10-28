# LFM3-Audio-2B on a PULSAR asm engine — scope, facts, and plan

Status: **scoping only. No engine code exists for this model.** This document is
the factual basis for a go/no-go decision. Every number below was read from
`LiquidAI/LFM3-Audio-2B` (`config.json`, the safetensors header, the file
listing) on 2026-10-28, not estimated.

## 0. Two open questions that gate everything

1. **Which engine is "pulsar-sam"?** There is no `pulsar-sam` in this repo. The
   two candidates are materially different targets:
   - `pulsar_asm/` — literal match for the name. x86-64, fasm flat binaries,
     AVX2 + F16C, driven by Python/ctypes. Gemma-2B and Gemma-4. ~10.6k lines.
   - `pulsar_arm/asm/core.S` — the "vision" deliverable per `CLAUDE.md`.
     AArch64, `as` + `ld -static`, svc syscalls only. Gemma-3 270m/1B. 6.6k lines.

   This document assumes **`pulsar_asm/`** (x86-64). If the intent is the AArch64
   asm engine, the kernel list below is the same but every implementation is
   different code, and the NEON/F16C availability differs.

2. **Scope.** "Support LFM3-Audio-2B speech to speech" is not one feature. See §2.

## 1. What the checkpoint actually contains

`model.safetensors` = 4,004,729,320 B, **539 tensors, BF16**. Separate
`audio_tokenizer/model.safetensors` = 147,143,540 B (LiquidCodec).
Custom code ships with the repo: `modeling_lfm3.py` (42 KB),
`modeling_lfm3_audio.py` (91 KB), `modeling_liquid_codec.py` (18 KB),
`lfm3_conv.py` (8.5 KB), `feature_extraction_lfm3_audio.py` (13 KB),
`processing_lfm3_audio.py` (19 KB), `chat_template.jinja` (9 KB),
`tokenizer.json` (18 MB, `LiquidAI/liquid7.1_tokenizer`).

Tensor census by top-level module:

| prefix | tensors | role |
|---|---|---|
| `lfm.` | 286 | LFM3 text backbone |
| `encoder.` | 170 | causal audio encoder (+ `downsample` front end) |
| `depthformer.` | 37 | MaskGIT parallel speech head |
| `depth_embeddings.` | 36 | 12 codebook input embeds + norms + logits |
| `mlp.` | 6 | audio→backbone projection (3 × 768→768) |
| `depth_linear.` | 2 | backbone → 12 depth streams |
| `codebook_embedding.` | 1 | `[24204, 2048]` (= 12 × 2017, exact) |
| `mask_embedding.` | 1 | `[1024]` MaskGIT mask token |

### Backbone (`lfm`) — LFM3, 24 layers
hidden 2048, intermediate 7680, vocab 128000 (tied), 16 heads / 4 KV heads
(head_dim 128), `sliding_window` 256, `conv_L_cache` 7, `use_sandwich_norm` true,
norm_eps 1e-5, rope theta 1e6 full-attn / 1e4 sliding-attn,
**`partial_rotary_factor` 0.5**.

`layer_types` is **not** uniform: 13 `conv`, 6 `sliding_attention`,
5 `full_attention`. Measured from tensors: 13 × `conv.{conv,in_proj,out_proj}`,
11 × `self_attn.*`, 24 × FFN.
Conv layer = `in_proj [6144,2048]` (= 3×d: x, v, z gate) → depthwise
`conv [2048,1,7]` → `out_proj [2048,2048]`.

### Audio encoder (`encoder`) — LFM2, 17 layers
hidden 768, intermediate **1792** (2560 auto-adjusted down by
`block_multiple_of` 256), 8 heads / 4 KV heads, **head_dim 96**,
`sliding_window` 32, `conv_L_cache` 3. 10 conv + 7 attention layers.
`downsample` front end: three `[256,1,3,3]` convs, two `[256,256,1,1]`, and
`lin_out [768,4096,1]` (4096 = 4 × 256 frame concat).
Front-end spec: 16 kHz, `n_mels` 128, `n_fft` **400**, `win_length` 400,
`hop_length` 160, hann periodic, center false, power 2.0, slaney mel + slaney
norm, f_min 0 / f_max 8000, log base 10 with floor −12, then affine
`(x − 4) / 4`. Padding: right, constant, min 400 samples, alignment offset 400 /
multiple 160.

### Speech head (`depthformer`) — MaskGIT, 3 layers
hidden 1024, intermediate 2816, 32 heads / 8 KV heads, **head_dim 32**,
`max_position_embeddings` 12, **`is_causal` false** (bidirectional),
`sliding_window` 256. `maskgit_steps` **3**, `codebooks` **12**,
`audio_vocab_size` 2017, `depth_linear [12288, 2048]` (= 12 × 1024).
12 independent `depth_embeddings[N]`: embed `[2017,1024]`, norm,
to_logits `[2017,1024]`.

### Interleaving
`interleaved_n_text` 6, `interleaved_n_audio` 12. LiquidCodec: 12 tokens per
80 ms → 24 kHz output. Recommended speech sampling:
`audio_temperature` 1.0, `audio_top_k` 128, `audio_repetition_penalty` 1.1,
`audio_repetition_window` 4096. Text-only modes set `suppress_mixed` /
`suppress_audio_out`. Control tokens `<|mixed_start|>` / `<|mixed_end|>`.

## 2. What "speech to speech" decomposes into

Five networks, one front end, one tokenizer, one controller:

1. Log-Mel front end (streaming/causal)
2. Audio encoder: 17 LFM2 layers + `downsample` (≈ 0.15 GB of the 4.00 GB)
3. LFM3 backbone: 24 mixed conv/attn layers (the bulk of the 4.00 GB)
4. MaskGIT speech head: `depth_linear` + 3 bidirectional layers + 12 logits heads
5. LiquidCodec decoder: 147 MB, tokens → 24 kHz waveform, streaming
6. `liquid7.1` BPE tokenizer, 128k vocab (18 MB `tokenizer.json`)
7. Interleaved text/audio generation controller

Per the standing directive (`CLAUDE.md`: *no C/Python in the asm inference
path*), all of 1–7 must be asm. The Python `runtime/` layer may only map memory
and hand over an address block, as `pulsar_abi.py` / `gemma4_model.py` do today.

## 3. Kernel gap — what `pulsar_asm/` does not have

`pulsar_asm/raw_materials/` today: f16c gemm, f16c gemv, bf16 gemb, smp
variants, geglu, gemma rmsnorm, exp/softmax LUT. Gemma-specific.
`grep -i conv pulsar_asm/` hits **only Python tooling** (the word "convert") —
there is no conv kernel of any kind in the tree.

New primitives required (✔ = exists and is reusable, ✘ = does not exist):

| # | primitive | state |
|---|---|---|
| 1 | Log-Mel: FFT-400 (**not** a power of two), hann periodic, 128-band slaney filterbank, log10 floor −12, affine | ✘ |
| 2 | streaming/causal mel with feature cache | ✘ |
| 3 | `[C,1,3,3]` 2-D conv ×3 + `[C,C,1,1]` in the downsample stack | ✘ |
| 4 | causal depthwise 1-D conv **L=7** with per-layer conv state cache | ✘ |
| 5 | causal depthwise 1-D conv **L=3** (encoder) | ✘ |
| 6 | conv `in_proj` 3d → x/v/z split + `out_proj` | ✘ |
| 7 | GQA attention 16q/4kv, head_dim 128 | ~✔ (Gemma GQA, adaptable) |
| 8 | **partial rotary** — 64 of 128 dims | ✘ |
| 9 | **two rope thetas** (1e6 full, 1e4 sliding) selected per layer type | ✘ |
| 10 | sliding-window attention, window 256, with window-bounded mask | ✘ |
| 11 | per-head scalar attention **gate** (`gate_proj [16,2048]`) | ✘ |
| 12 | per-head `q_norm`/`k_norm` RMSNorm on head_dim 128 / 96 / 32 | ~✔ |
| 13 | **sandwich norm** (`post_operator_norm`, `post_ffn_norm`) | ✘ |
| 14 | SwiGLU FFN w1/w2/w3 | ~✔ (Gemma geglu, adaptable) |
| 15 | `codebook_embedding` gather-sum over 12 codebooks | ✘ |
| 16 | `depth_linear [12288,2048]` GEMM | ✔ (gemm, new shape) |
| 17 | **MaskGIT loop**: 3 steps, bidirectional attn, confidence-based re-masking | ✘ (whole algorithm) |
| 18 | depthformer: 32 heads × head_dim 32, 8 KV, `is_causal=false` | ✘ |
| 19 | 12 × `to_logits [2017,1024]` + top-k 128 + **repetition penalty over a 4096 window** | ✘ (sampler lacks rep-penalty) |
| 20 | LiquidCodec decoder (VQ conv-transformer, 12 codebooks → 24 kHz, streaming) | ✘ (a second full model) |
| 21 | `liquid7.1` BPE: new `mkvocab`/`mkbpe` blobs, 128k vocab | ✘ |
| 22 | interleaved text/audio controller (6 text / 12 audio, mixed tokens, suppress flags) | ✘ |

~17 of 22 primitives do not exist in any form. The three largest are the mel
front end + streaming, the MaskGIT speech head, and LiquidCodec — and LiquidCodec
alone is a 147 MB generative model, i.e. comparable in size to the entire
existing Gemma-2B engine effort.

## 4. Exactness gates (this repo's definition of done)

`CLAUDE.md`: definition of done is bit-exactness; parity gates are never
overridable; a gate failure is information, not something to work around.
Gates needed here, in build order:

| gate | reference |
|---|---|
| G1 mel | `processor.feature_extractor` on fixed wav, max abs diff on fp32 |
| G2 encoder | per-layer hidden states, first divergence = (layer, position) |
| G3 backbone prefill | logits at last prompt position, fp32/bf16 tolerance declared |
| G4 greedy text stream | HF greedy tokens on frozen prompt A + held-out B |
| G5 audio codes | `audio_codes` for fixed prompt, all 12 codebooks |
| G6 waveform | `codec.decode` output, per-sample max abs diff |
| G7 end-to-end | interleaved text+speech stream == reference |

Gates G5–G7 are sampling-dependent (top-k 128, temp 1.0, rep-penalty 1.1), so
they need a frozen seed **and** a documented tie-break order before they can be
bit-exact. `fwd8.c`'s sampler chain order (top-k, top-p, min-p on raw logits,
temperature last) is the precedent to follow.

## 5. Environment blockers (measured today, on this machine)

These block *verification*, so nothing in §4 is currently checkable:

| check | result |
|---|---|
| `fasm` | **absent** — the existing x86 flat binaries cannot be assembled here (`nasm`, `as`, `gcc` are present; fasm is not) |
| `torch` | **absent** (`ModuleNotFoundError`) |
| `transformers` | **4.38.2** — model requires **≥ 5.20.2** and `AutoModelForMultimodalLM` + remote code |
| Python | **3.8.0** — transformers 5.x needs a newer interpreter |
| disk | 141 GB free — fine for 4.00 GB + 147 MB + quantized blobs |

So the reference oracle (the thing every gate compares against) cannot be built
in this environment as it stands. That is the first thing to fix, before any asm.

## 6. Staged plan, each stage gated

| stage | deliverable | gate |
|---|---|---|
| **S0** | toolchain: install `fasm`, a Python ≥3.10 + torch + transformers ≥5.20.2 env, download weights | oracle emits reference tensors |
| **S1** | text-only LFM3 backbone engine (24 mixed layers, 2B): conv+cache, GQA, partial rope, dual theta, sliding window, sandwich norm, gate, SwiGLU | G3, G4 |
| **S2** | `liquid7.1` tokenizer in asm (BPE blobs via new `mkvocab`/`mkbpe`) | token-id parity on a fixed corpus |
| **S3** | mel front end + 17-layer encoder | G1, G2 → **audio-input text-output works (ASR / voice chat, text reply)** |
| **S4** | MaskGIT head: `depth_linear`, 3 bidirectional layers, 12 logits heads, rep-penalty sampler | G5 |
| **S5** | LiquidCodec decoder in asm, streaming | G6, G7 → **speech to speech** |

S1 is the foundation and is independently useful: it is the engine for
LFM3 text models generally. S3 gives a genuinely shippable milestone
(audio in, text out) well before S5. Full speech-to-speech is S5.

Realistically this is a multi-week project dominated by S1 and S5, not a
single-session change. S0 is the only thing to do immediately.

## 7. Recommendation

Do S0 now (toolchain + oracle), then S1. Do not start from S5: the codec is the
largest single item and is useless without S1–S4 producing valid code tokens.
Confirm the target engine (§0.1) before any kernel is written.
