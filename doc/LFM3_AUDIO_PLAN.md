# LFM3-Audio-2B on a PULSAR asm engine — scope, facts, and plan

Status: **measured; verdict in §11 — no-go for full speech-to-speech on a 4 GB
Jetson Nano, and S1 fits only on a dedicated board.** Not an opinion: the
accuracy floor is measured (§11, Q8_0 is bit-identical to BF16 and nothing
below it survives), the incumbent it must beat is measured (§10.1, a 27 tok/s
ASR+LLM+TTS stack in 1357 MB running 6 days), and the kernel gap is measured
(§8.4b, the Nano has no Q8 GEMV and K=7680 is unsupported). Every number below
was read from `LiquidAI/LFM3-Audio-2B` (`config.json`, the safetensors headers,
the file listing), from `.auto/HANDOFF_NANO_Q1_0.md`, or measured on the two
Jetson Nanos, on 2026-10-28 — not estimated.

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

   **Resolved 2026-10-28: the target is `pulsar_arm/bonsai2/` on branch
   `lfm3-audio-jetson-nano`** (the Jetson Nano C + NEON + CUDA sm_53 path, the
   one that runs Bonsai-8B Q1_0 at 16.11 tok/s). The x86 `pulsar_asm/lfm3/`
   scaffold stays as a preparation exercise / parity oracle, not the deliverable.
   See §8 for the measured Nano feasibility, which supersedes §5.

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

S0 is done (§8.1). The next thing to build is **not** a kernel: it is the
pipeline comparison in §10, because the accuracy floor (§9) says LFM3-Audio's
backbone alone is 1.72 GB minimum, 1.8x the 950 MB Bonsai-4B LLM already
shipping in the Nano voice-chat stack. Confirm what `pulsar-sam` means (§0.1)
and which board is the target (§9.1: only picard-desktop has the RAM) before
writing S1.

Do not start from S5: the codec is the largest single item and is useless
without S1–S4 producing valid code tokens.

---

## 8. Jetson Nano target — measured 2026-10-28 (supersedes §5)

All figures below are measured: tensor bytes from the safetensors headers, the
21.9 GB/s streaming ceiling and the Bonsai-8B reference from
`.auto/HANDOFF_NANO_Q1_0.md`, `MemAvailable` from `free -m` on the live board
(`pesi@100.70.245.36`).

### 8.1 §5 is stale — S0 is done

| check | §5 said | actual |
|---|---|---|
| `fasm` | absent | `~/.local/bin/fasm` 1.73.02 |
| `torch` | absent | 2.6.0+cu124 in `~/venvs/lfm3` (Python 3.11.15) |
| `transformers` | 4.38.2 | 5.21.0 (that 4.38.2 is the system py3.8, not the venv) |
| weights | to download | cached: 4.005 GB main + 0.147 GB LiquidCodec = **4.152 GB** |

The HF oracle exists and loads `Lfm3AudioProcessor` / `AutoModelForMultimodalLM`,
so every gate in §4 is now checkable. The x86 scaffold runs: `pulsar_asm/lfm3/
run_lfm3_chat.py` loads 24 layers / 3.458 GB and reports `SMOKE_OK`.

### 8.2 Exact module sizes (safetensors headers)

| module | bytes BF16 |
|---|---|
| `lfm` (text backbone, 1728.8 M weights) | 3.458 GB |
| `encoder` (17-layer LFM2 audio encoder) | 0.219 GB |
| speech head (`depth_embeddings`+`codebook_embedding`+`depthformer`+`depth_linear`+`mlp`) | 0.328 GB |
| `audio_tokenizer` (LiquidCodec) | 0.147 GB |
| **total** | **4.152 GB** |

The backbone breaks down as: FFN w1/w2/w3 1132.6 M (66%), `embed_tokens` 262.1 M
(15%, tied), conv 218.2 M (13%), attention 115.5 M (7%).

### 8.3 Memory: the text backbone fits; the earlier "won't fit" read was wrong

The board is 4 GB unified shared with the userspace. **Measured `MemAvailable`
right now: 1476 MB** (`free -m`: 3965 total, 2432 used, 1286 buff/cache).

The reference point is what already runs: Bonsai-8B Q1_0 = 1.06 GB weights +
1280 MB device arena + ~304 MB RSS.

| LFM3 backbone | weights | f16 K/V @4k | total | decode bus floor |
|---|---|---|---|---|
| BF16 | 3.458 GB | 0.053 GB | 3.511 GB | 157.9 ms/tok (6.3 tok/s) |
| 8-bit | 1.729 GB | 0.053 GB | 1.782 GB | 78.9 ms/tok (12.7 tok/s) |
| **4-bit** | **0.864 GB** | **0.053 GB** | **0.917 GB** | **39.5 ms/tok (25.3 tok/s)** |

Two findings that invert the §5/§0 pessimism (both about structure; **the
accuracy floor in §9 then reverses finding 1 — read §9 before quoting 0.864 GB**):

1. **At 4-bit the backbone is 0.864 GB — smaller than Bonsai-8B Q1_0's 1.06 GB,
   which runs on this board today.** Size-wise it is in the feasible class.
   **But 4-bit does not survive the accuracy gate** (§9): the greedy stream
   collapses at Q4. Treat this row as a memory bound, not a candidate.
2. **The K/V cache is 53 MB, not gigabytes** (11 attention layers × 4 KV heads ×
   head_dim 128; 6 of the 11 are sliding-window-256 bounded). Bonsai-8B pays
   **1.208 GB** for the same 4k context — 36 layers × 8 KV heads. That is what
   the 1280 MB arena is sized for, so **LFM3 needs a far smaller arena**. The
   CLAUDE.md "do not shrink the 1280 MB arena" rule is Bonsai-shaped (it is
   about Tegra nvmap RSS accounting at a given arena size), not a physics law;
   re-size it for LFM3's 53 MB cache and re-measure RSS at equal arena size.

Full speech-to-speech adds encoder 0.219 + speech head 0.328 + codec 0.147 =
**0.694 GB more** on top of the accuracy floor — that is the memory wall for
S4/S5.

### 8.4 The real blocker is kernel shape specialization, not RAM

`pq1_swar_sm53.cu:26` documents the tuned kernels as `K in {4096, 12288}; N
arbitrary`, and the dispatch gate at `pq1_swar_sm53.cu:1340` is literally

```c
return N < 16 || (K != 4096 && K != 12288);
```

**Every LFM3 GEMV has K = 2048 (conv in/out proj, q, k, v, o, gate, w1, w3) or
K = 7680 (w2). Not one of them is 4096 or 12288** — so all of them take the
fallback path and none of the 16.11 tok/s work applies. The 16.11 figure came
from per-shape sweeps (TPB/MINB/RB/PD, §0.3) run at exactly K=4096 and K=12288;
K=2048 needs its own sweep.

Good news: 2048 = 8 × 256 and 7680 = 30 × 256, so both are exact multiples of
the Q8_K 256-block and the 128-group structure the repack and bit-plane kernels
assume. K=2048 is a clean 8-block Q8_K activation — structurally compatible,
untuned.

**This optimism was wrong, and `pq1_parity` is what disproved it (§8.4b).**
Probing the real kernels at the LFM3 shapes returned bit-exact 7680/7680,
6144/6144, 2048/2048, 512/512 at K=2048 — which looked like the answer. It is
not: those kernels are a **2-bit** format (`pq1_swar_sm53.cu:1-30`: "EXACT
PQ1_0 GEMV", `code c = b0 + 2*b1, value v = c-1`, layout `w_bits [N, K/32] u32`
= two bit planes + fp16 scales per 128-group ≈ 2.125 bits/weight). LFM3 cannot
use that format — §9 scores Q2_0 at 0/48. So K=2048 being clean says nothing
about the Q8 GEMV LFM3 actually needs, and **a Q8_0 GEMV does not exist in the
Nano tree**. The K=2048 question stays open, and it is bigger than tuning.

So on Nano the work is **not** "port a model"; it is "re-specialize the GEMV
family for K=2048/7680" plus the §3 kernel gap (17 of 22 primitives absent —
depthwise conv + conv state cache, partial rotary, dual rope theta, sliding
window, sandwich norm, per-head gate — all new, in NEON/CUDA rather than AVX2).

### 8.4b What `pq1_parity` does say (measured, ipho-nano, EMC pinned 1600 MHz)

`PQ1_TIME=15 ./pq1_parity <rows> <K> all`, ms/call, hot loop (gap=0 so GPU
 clocks boost — the engine's real pattern has CPU work between GEMVs, so these
 are best-case). `hs` is the production kernel (default since loop-261014-2216
 row 12).

| shape | rows | K | swar | r128 | r256 | tm | **hs** | bit-exact |
|---|---|---|---|---|---|---|---|---|
| w1/w3 | 7680 | 2048 | 0.822 | 0.605 | 0.581 | 0.361 | **0.398** | 7680/7680 ✓ |
| conv_in_proj | 6144 | 2048 | 0.561 | 0.414 | 0.454 | 0.255 | **0.258** | 6144/6144 ✓ |
| w_q / w_o | 2048 | 2048 | 0.220 | 0.232 | 0.195 | 0.125 | **0.119** | 2048/2048 ✓ |
| w_k / w_v | 512 | 2048 | 0.108 | 0.088 | 0.097 | 0.061 | **0.056** | 512/512 ✓ |
| **w2 (FFN down)** | 2048 | **7680** | 0.507 | 0.415 | 0.685 | 0.259 | 0.259 | **0/2048 MISMATCH maxrel 1.47e+02** |
| ref (Bonsai) | 4096 | 4096 | 0.632 | 0.508 | 0.543 | 0.308 | **0.205** | 4096/4096 ✓ |
| ref (Bonsai) | 4096 | 12288 | 1.700 | 1.378 | 2.454 | 0.838 | **0.485** | 4096/4096 ✓ |

Two findings:

1. **K=7680 is not supported at all** — every kernel returns 0/2048 bit-exact
   with maxrel 1.47e+02 (total mismatch, not rounding). That is the FFN
   down-projection, 377.5 M weights = 22% of the backbone, so an LFM3 GEMV port
   has to add a third K to a family whose tile/warp decomposition is built for
   K = 2^12·{1,3}. The parity gate catches it loudly, which is the gate working
   as intended.
2. **K=2048 needs no retuning for correctness**, and `hs` already beats `tm`
   there (ratios 1.04–1.09x on the small shapes) — unlike K=4096/12288 where
   `hs` wins by 1.50x/1.73x. The big K=2048 shape (w1/w3, 15.7 MB) is the
   exception: `hs` 0.398 ms is **24% slower than `tm` 0.361 ms**, because RB/PD
   /NT were swept for the two Bonsai K only. That is the one place §8.4's
   "needs its own sweep" is directly confirmed.

### 8.5 Ranked order for the Nano

| stage | Nano deliverable | feasibility at measured numbers |
|---|---|---|
| S1 text backbone | 24 mixed conv/attn layers, 4-bit | **fits** (0.917 GB + arena); bus floor 25.3 tok/s, so a tuned K=2048 kernel is the gate |
| S2 tokenizer | `liquid7.1` 128k BPE blobs | independent, small |
| S3 encoder + mel | audio-in → text-out | +0.219 GB; **the shippable milestone** |
| S4 MaskGIT head | audio codes out | +0.328 GB, tight; 3 bidirectional layers × 3 steps per segment; needs rep-penalty sampler (absent in `fwd8.c`, `sub_sampler_*`) |
| S5 LiquidCodec | waveform out | +0.147 GB; a second full model, streaming conv-transformer — weeks |

Build S1 first: it is the single item that de-risks everything, and unlike the
x86 scaffold it must be written in NEON/CUDA sm_53 (no SDOT on Maxwell — use
the `AND + IADD3` / SMULL+SADALP playbook from `NANO_BRINGUP.md`), not AVX2.
Do not start from S5.

---

## 9. Accuracy floor: 4-bit does not survive the gate (measured 2026-10-28)

`.auto/lfm3/quant_sensitivity.py` fake-quantizes → dequantizes the 153 backbone
2-D tensors (same numerics an int-dot GEMV has) and greedy-decodes against the
frozen HF goldens. Metric = longest common prefix, the repo's metric (`n/48`).
Group = weights per f16 scale; Q8_0/Q4_0 style is group 32.

| scheme | bits/w | GB backbone | prompt A | prompt B |
|---|---|---|---|---|
| bf16 | 16.00 | 3.457 | **48/48** | **48/48** |
| **q8_0** | 9.64 | **2.082** | **48/48** | **48/48** |
| q6_0 | 7.94 | 1.716 | 48/48 | 32/48 |
| q5_0 | 7.09 | 1.532 | 48/48 | 14/48 |
| q4_1 | 6.67 | 1.441 | 48/48 | 12/48 |
| q4_g128 | 6.03 | 1.303 | 6/48 | 24/48 |
| q4_0 | 6.24 | 1.349 | 9/48 | 0/48 |
| q3_0 | 5.39 | 1.166 | 0/48 | 0/48 |
| q2_0 | 4.55 | 0.982 | 0/48 | 0/48 |
| q6_0 + q6 emb | 6.57 | 1.421 | 48/48 | 32/48 |
| q5_0 + q5 emb | 5.57 | 1.205 | 48/48 | 15/48 |
| q4_1 + q4 emb | 5.00 | 1.080 | 27/48 | 23/48 |

**Q8_0 is the only scheme that reproduces the BF16 greedy stream on both
prompts.** Everything below Q8 diverges, prompt B first (it is the harder
prompt: Q6 already loses it at token 32). Quantizing the embedding table buys
memory but rescues nothing — sensitivity lives in the FFN (1132.6 M = 66% of
the backbone). So:

- **Accuracy floor ≈ 1.72 GB** (FFN+conv+attn at Q8_0 = 1.558 GB, embeddings
  Q4_1 = 0.164 GB). Below that the stream breaks.
- **§8.3's "4-bit is feasible" was wrong as a proposal.** 0.864 GB is a memory
  bound, not a candidate: Q4_0 gives 9/48 and 0/48. Corrected above.
- Caveats: 2 prompts x 48 tokens, greedy only, torch fake-quant (group 32
  matches llama.cpp Q8_0/Q4_0). Prompt A tolerates down to Q4_1 while B breaks
  at Q6 — the spread means the gate needs more prompts before any tolerance is
  frozen. **The 8-prompt deterministic replication is §11 and it confirms this
  table**; the prefix columns here are valid, only the `exact` boolean in the
  harness was broken at the time (see §11's note on the metric).

**Validation that the quantizer really ran** (the failure mode is a filter that
matches nothing, which reads as "4-bit is lossless"). Relative Frobenius error
on `layers.20.feed_forward.w2` (2048x7680, the deepest dot in the model):

| scheme | max abs delta | rel. Frobenius | greedy A / B |
|---|---|---|---|
| q8_0 | 0.00229 | **0.57%** | 48/48, 48/48 |
| q6_0 | 0.00928 | 2.23% | 48/48, 32/48 |
| q5_0 | 0.02026 | 4.61% | 48/48, 14/48 |
| q4_1 | 0.02151 | 7.90% | 48/48, 12/48 |
| q4_0 | 0.04150 | 9.88% | 9/48, 0/48 |
| q2_0 | 0.20215 | 65.9% | 0/48, 0/48 |

Monotone in the right places: 0.57% noise is small enough not to flip any argmax
across 48 greedy steps, 2.2% already flips prompt B at token 32. That is the
shape a genuine quantization study has, so the Q8 floor is not a no-op artifact.

### 9.1 Memory verdict against the two boards

| board | role | MemAvailable | passwordless sudo | EMC clock |
|---|---|---|---|---|
| `picard@100.84.151.116` (picard-desktop) | clean / dedicated | **2542 MB** | no | idle (unpinned) |
| `pesi@100.70.245.36` (ipho-nano) | **production voice-chat** | **1457 MB** | yes | pinned 1600 MHz |

Both are Tegra X1, cc 5.3, **1 SM**, 4.16 GB, CUDA 10.2; sm_53 builds and runs
on both (verified with a deviceQuery probe).

- **ipho-nano (1457 MB): S1 at the Q8 floor (1.72–2.08 GB) does not fit.** No-go
  there. It is also not free to use — mysql, asterisk, erlang, mysqld, gdb and
  the voice-chat server are resident (~1.1 GB), and it is headless already
  (lightdm inactive, no Xorg), so the 1457 MB is *not* a display tax.
- **picard-desktop (2542 MB): S1 at the Q8 floor fits with ~0.4–0.8 GB
  headroom** — tight, and it needs the small LFM3 arena (53 MB K/V), not
  Bonsai's 1280 MB. This is the only board where S1 is plausibly runnable.
- Full speech-to-speech (floor 1.72 + 0.694 GB ≈ 2.4 GB + CUDA context + codec
  buffers) is **over budget on both** 4 GB boards.

### 9.2 Bandwidth probe: reported as inconclusive, not as a board number

`.auto/lfm3/probe_q8_gemv.cu` measures Q8_0 GEMV at the LFM3 shapes. It read
0.81 GB/s for the GEMV and ~10 GB/s for its "read ceiling" — **below the repo's
own *achieved* 17.1 GB/s**, so the probe kernel is the limit, not the board.
Same result on both boards (EMC pinned 1600 MHz on ipho-nano), so it is not a
clock artifact. Do not quote 10 GB/s as a Nano ceiling.

The K=2048 question from §8.4 therefore stays open and needs the repo's real
instrument, not this probe: time the existing `hs` kernel at K=2048/7680 via
`pq1_parity <rows> <K> all`. Note CLAUDE.md's trap: `ab_kernel.sh` with
`GITREF=HEAD` appends `ab_delegates.inc`, fails to link, and silently measures
a stale binary — pre-#42 refs only.

Two bugs found on the way, both worth remembering: an `int` Q8 load indexed
`+ lane` (stride 1) instead of `+ lane*4` is misaligned and Maxwell kills the
whole launch with "unspecified launch failure"; and an unchecked launch makes
that read as `0.000 ms` and an absurd GB/s. `cudaGetLastError()` first, always.

## 10. The deployment this is actually for (found on the board)

`/home/pesi/vc/voicechat-jetson/` on ipho-nano is a **running LAN speech-to-
speech assistant** (`run.sh`, port 8443) that already does the whole LFM3-Audio
job as a pipeline of separate models:

| stage | model | size |
|---|---|---|
| VAD | `silero_vad.onnx` | small |
| turn detection | `models/smart-turn` | — |
| ASR | `~/xasr-new/480` | — |
| **LLM** | **`Ternary-Bonsai-2-4B-PTQ1_0.gguf`** (the bonsai2 engine) | **950 MB** |
| TTS | `~/primetts_v2/primetts_v21_3voice.onnx` + `g2pw` | — |

It vendors a copy of this repo's engine at `pulsar/pulsar_arm/bonsai2/
{glibc_mathf.h, fwd35_sm53.cu}`. This is the strongest `pulsar-sam` candidate
found so far — a Pulsar-engine voice assistant on the Nano. **Still unconfirmed;
§0.1 remains open.**

Why it decides the go/no-go: the LLM slot is 950 MB inside a 1457 MB budget, and
LFM3-Audio's accuracy floor is 1.72 GB for the backbone alone — 1.8x the whole
incumbent LLM, with ASR and TTS on the same board. LFM3-Audio replaces
ASR+LLM+TTS with one model, so the fair comparison is the *whole pipeline*, not
the LLM slot. That comparison is the next thing to measure.

### 10.1 The comparison, measured (2026-10-28, ipho-nano production log)

Incumbent, from `ps` RSS, `du` on the model files, and `decode N tok T s = X`
lines in `voicechat-jetson/server.log` (service up 5 d 22 h):

| component | bytes on disk | RSS |
|---|---|---|
| `app/server.py` (ASR + TTS + VAD + G2P + turn) | — | **498 MB** |
| `fwd35_serve` (LLM engine host side) | — | 37 MB |
| Ternary-Bonsai-2-4B-PTQ1_0.gguf | 906 MB | (in nvmap arena) |
| xasr ASR `~/xasr-new/480` | 162 MB | in the 498 |
| g2pw | 153 MB | in the 498 |
| PrimeTTS v2 `primetts_v21_3voice.onnx` | 112 MB | in the 498 |
| smart-turn / tok | 11 / 13 MB | in the 498 |
| **whole speech-to-speech pipeline** | **1357 MB** | **~535 MB + 906 MB arena** |

**Incumbent LLM decode: 25.2–28.3 tok/s** (`decode 11 tok 0.397 s = 27.73`).

LFM3-Audio-2B at the §9 accuracy floor, same function:

| | incumbent | LFM3-Audio @ Q8 floor |
|---|---|---|
| weights resident | 1.357 GB | 1.72 (backbone) + 0.219 encoder + 0.328 speech head + 0.147 codec = **2.41 GB** |
| LLM decode | **27 tok/s** | 2.082 GB/tok ÷ 17.1 GB/s = **8.2 tok/s** (10.5 at the 21.9 ceiling) |
| fits ipho-nano (1457 MB) | yes, 6 days stable | **no** (1.7x over) |
| fits picard-desktop (2542 MB) | yes | **no** (2.41 GB + CUDA context > 2542) |

So LFM3-Audio on a 4 GB Nano is **~1.8x the incumbent pipeline's resident bytes
and ~3x slower on the LLM**, while doing the same job — and it fits on neither
board. The 1-bit ternary Bonsai pipeline is the reason this is a no-go: it is
not a slow reference to be beaten, it is a very strong incumbent that this
model's accuracy floor cannot approach.

What would change the verdict: (a) a quantization tier that preserves the
stream below Q8 — none found in §9, and the FFN at 66% of weights is the
sensitivity driver, so it needs more than a better group size; (b) a smaller
LFM3-Audio checkpoint (2B is the smallest published); (c) accepting a stream
that diverges from BF16, i.e. redefining "frozen accuracy" — the §9 table is
the input to that decision, and the deterministic 8-prompt extension is in
`.auto/lfm3/quant_sensitivity_prompts8.jsonl` (results, §11).

---

## 11. Verdict (definitive numbers, deterministic harness)

`.auto/lfm3/quant_sensitivity_prompts8.jsonl` — 8 prompts (factual, arithmetic,
code, translation, chemistry, date formats, explanation), 24 greedy tokens,
1 thread, bf16 control row passing 8/8. `exact` = agreement over the compared
span `min(len(golden), --tokens)`; `worst` = the minimum prefix over the 8.

| scheme | bits/w | GB backbone | exact | mean prefix | worst |
|---|---|---|---|---|---|
| bf16 (control) | 16.00 | 3.457 | **8/8** | 23.1 | 17 |
| **q8_0** | 9.64 | **2.082** | **8/8** | **23.1** | 17 |
| q6_0 | 7.94 | 1.716 | 6/8 | 19.4 | **0** |
| q5_0 | 7.09 | 1.532 | 3/8 | 16.1 | 6 |
| q4_1 | 6.67 | 1.441 | 2/8 | 10.4 | 0 |
| q4_0 | 6.24 | 1.349 | 1/8 | 9.1 | 0 |
| q2_0 | 4.55 | 0.982 | **0/8** | **0.0** | 0 |

**Q8_0 is bit-identical to BF16 on all 8 prompts** (same 8/8, same 23.1 mean
prefix — quantization noise at 0.57% rel. Frobenius never flips an argmax).
Below it the collapse is steep and monotone: Q6 already loses 2 prompts
entirely, Q4_0 survives 1 of 8, Q2_0 diverges at the **first token of every
prompt** (mean prefix 0.0).

The verdict, per stage, against measured budgets (ipho-nano 1457 MB available,
picard-desktop 2542 MB, Tegra X1 1 SM, 21.9 GB/s ceiling / 17.1 GB/s achieved):

| stage | needs | verdict |
|---|---|---|
| S1 text backbone | 2.082 GB (Q8_0) + 53 MB K/V + arena + ~300 MB RSS | **only on picard-desktop**, tight; **no** on ipho-nano. ~8 tok/s (2.082 GB ÷ 17.1 GB/s), vs the 27 tok/s incumbent |
| S1+S2+S3 (audio-in → text-out) | + 0.219 GB encoder | marginal on picard-desktop only |
| S4 MaskGIT head | + 0.328 GB, 3 bidirectional layers × 3 steps, rep-penalty sampler (absent) | **no** on either board |
| S5 full speech-to-speech | ≈ 2.41 GB weights + CUDA context + codec buffers | **no** on 4 GB, and ~1.8x the incumbent pipeline's 1357 MB at ~1/3 its speed |

The reason this is a no-go rather than a "slow start" is the incumbent: the
Nano already runs ASR + 4B-ternary-LLM + TTS + VAD + turn-detection in
1357 MB at 27 tok/s, stably for 6 days (§10.1). LFM3-Audio's accuracy floor is
1.72 GB for the backbone alone — it cannot approach that footprint, and the
kernel story is worse than the memory story: the Nano has **no Q8 GEMV** (the
`pq1_*` family is 2-bit PQ1_0, §8.4b), K=7680 is unsupported outright, and the
only Q8 measurement available is 0.81 GB/s, ~20x short of the 17.1 GB/s the
estimate above assumes.

If the goal is speech-to-speech on a Jetson Nano, the measured evidence says
keep the existing pipeline. If the goal is specifically LFM3-Audio's quality,
the target needs more RAM than 4 GB. The one piece of S1 worth doing on the
Nano regardless is a **Q8_0 GEMV at K=2048** — it is the missing primitive for
*any* 8-bit model on this tree (LFM3, and the Q8-embedding GGUFs already sitting
on picard-desktop), and it is self-contained: `pq1_parity`-style bit-exactness
against a host reference, then a TPB/MINB/RB/PD sweep like K=4096 got.

---

## 12. That Q8_0 GEMV exists now: 13.0 GB/s at K=2048 (`.auto/lfm3/q8_gemv_sm53.cu`)

Built and measured on ipho-nano (Tegra X1, 1 SM, EMC pinned 1600 MHz, CUDA
10.2, `-O3 --use_fast_math -Xptxas -O3,-dlcm=ca`). Bytes counted as codes +
f16 scale per 32-code block (codes-only flatters the number by 6%).

| rows | shape (K=2048) | MB | ms/call | GB/s | % of 21.9 ceiling |
|---|---|---|---|---|---|
| 7680 | w1/w3 | 16.7 | 1.282 | **13.04** | 60% |
| 6144 | conv_in_proj | 13.4 | 1.006 | **13.29** | 61% |
| 2048 | w_q / w_o | 4.5 | 0.350 | **12.73** | 58% |

R (rows per block) sweeps 6.4 -> 10.5 -> 12.9 -> 13.0 GB/s at 4/8/16/32/64,
flat past 16; that curve is the amortization of staging `x` into shared memory.

The path there is the useful part, because every step was a silent-performance
bug of the kind this repo's measurement discipline exists to catch:

| cut | GB/s | what was wrong |
|---|---|---|
| throwaway probe (§9.2) | 0.81 | 1 warp/row, per-row `atomicAdd`, 4 separate byte loads |
| thread-per-Q8_0-block, int4 loads | 3.10 | `x` re-staged into shared **per block**: 15.4 MB of `x` traffic on top of 15.7 MB of weights, so half the bus was moving activations |
| naive row-loop | 0.79 | dropped the `tid / nb` term, so only 64 of 256 threads' block indices were used: **4x redundant work** (3.10/4 = 0.79 exactly — the arithmetic named the bug) |
| thread mapping fixed | 3.40 | — |
| **padded shared row stride 32 -> 33** | **12.50** | 32-way shared **bank conflict**: lanes 0..31 read `xs[0*32+j] … xs[31*32+j]`, all stride-32 floats = the same bank, on all 32 shared loads |
| R = 16…64, scales counted | **13.0** | honest byte count |

So the §9.2 answer is: **the 0.81 GB/s was kernel quality, not physics** — Q8
at K=2048 reaches 13.0 GB/s, 76% of what this engine achieves with its tuned
2-bit kernels. Two lessons that generalize to the rest of the tree: count the
activation traffic you stage into shared, and check the shared-index stride of
any kernel where a warp reads a column of a `[block][32]` tile.

**Correctness: bit-exact against a device reference.** The first cut reported
`differs` at maxabs 1.2e-3 / maxrel 2.9e-4 against a sequential reference, for
two known reasons: the 64 block-partials of a row are combined by a warp-shuffle
**tree**, not the reference's sequential `acc += p[b]` (fp add is not
associative), and `--use_fast_math` contracts the 32-term sums into FMAs. The
repo already owns the fix — `pq1_swar_sm53.cu:19-22`, "the fp accumulation is
made bit-exact too by reducing per-group products in GROUP ORDER in fp32
(`__fmul_rn`/`__fadd_rn`, no contraction)". Applied here: `q8_ref_tree_kernel`
mirrors the kernel's arithmetic operation for operation (index-order
`__fmul_rn`/`__fadd_rn` per Q8_0 block, then the same butterfly32 at
o = 16,8,4,2,1, then `part[0] + part[1]`). Result, all shapes, all R:

```
tree BIT-EXACT   seq maxrel 0.00029   tree maxrel 0
```

**BIT-EXACT in both builds**, `-fmad=false` and `--use_fast_math` alike, at
12.35–12.45 GB/s in the principled `-fmad=false` build (only ~4% under the fast
build, so contraction is not what buys the bandwidth). The `__fadd_rn`/
`__fmul_rn` intrinsics are `__device__`-only — nvcc rejects calling them from a
host function ("calling a __device__ function from a __host__ function is not
allowed"), which is why the bit-exactness reference runs as a GPU kernel, one
thread per row. Caveat worth stating: bit-exactness in the `--use_fast_math`
build is verified on this data only — fma-vs-mul-then-add differs on rare
inputs, so `-fmad=false` is the build to gate on.

So this is now a shippable primitive, not a bandwidth probe: parity-checkable
against a reference the way `pq1_parity` requires, at 12.4 GB/s.

**And it does not change the §11 verdict.** At the measured 13.0 GB/s the Q8_0
backbone is 2.082 GB/token = 160 ms/token = **6.2 tok/s** (the §11 estimate of
8.2 assumed the engine's 17.1 GB/s, so this is worse, not better). Against the
incumbent's 27 tok/s. The memory wall is unchanged: 2.08 GB of weights will not
fit ipho-nano's 1457 MB available, so this kernel cannot even be loaded there.

## 13. The GEMV is solved; memory and the embedding are what is left (measured 2026-10-29, clean Nano)

Everything in §12 was priced with a kernel that could not read K=7680. The
kernel work is now done and it moved decode a lot; the two open items are
device memory and the tied embedding. Numbers below are one SM on the clean
Nano, `-fmad=false`, bit-exact per config, coverage asserted.

### 13.1 The rate, per tensor (`q8_lc3.cu`)

Lane-strided chunked Q8_0: **one warp per row with lanes striding the row's
blocks** (kg's structure, so `nb` may be any integer) plus **x staged in
shared and amortized over R rows** (split-K's actual win). Best config each:

| tensor | shape | calls/tok | ms/call | GB/s | ms/token |
|---|---|---|---|---|---|
| lm_head (tied embed, Q8) | 128000x2048 | 1 | 17.893 | 16.51 | 17.9 |
| gate+up w1,w3 | 7680x2048 | 48 | 1.090 | 16.28 | 52.3 |
| down w2 | 2048x7680 | 24 | 1.184 | 14.98 | 28.4 |
| conv_in_proj | 6144x2048 | 13 | 0.875 | 16.22 | 11.4 |
| conv_out / w_q / gate_proj / w_o | 2048x2048 | 46 | 0.313 | 15.12 | 14.4 |
| w_k, w_v | 2048x512 | 22 | 0.149 | 8.00 | 3.3 |
| **total (Q8 embedding)** | | | | | **127.7 ms = 7.83 tok/s** |

Three results that took a falsification each to reach:

1. **`split-K` was the wrong variable.** `KCHUNK=7680` is `nchunk=1`, i.e. no
   splitting at all, and it is the *fastest* down-w2 config (14.31 GB/s vs kg's
   5.96). The lever is rows-amortized-per-x-stage: R=1 → 5.96, R=64 → 12.96,
   R=256 → 14.31 GB/s. Occupancy is not a lever here (the 1536/2048
   threads-per-SM cap binds before shared).
2. **5fd7e79's 12.88 GB/s on down w2 is withdrawn**: `7680 % 2048 = 1536`, so
   `nchunk = K/KCHUNK = 3` read 6144 of 7680 columns. No multiple of 1024
   divides 7680 (7680 = 32×240, 240 % 32 = 16), so a 32-lane-group butterfly
   cannot cover that K at all. The bit-exact gate printed YES because
   `ref_chunked` computed `nchunk` the same way — **a reference that mirrors
   the kernel's structure mirrors its structural bugs.** Coverage
   (`nchunk*KCHUNK == K`) is now asserted independently.
3. **fp32 scales are faster than fp16 scales** (16.51 vs 16.33 GB/s), because
   reading 4 bytes removes a `__half2float` per block while the scale is 1/32
   of the codes traffic. They are also the *certified* numerics.

14-16.5 GB/s is 64-75% of the 21.9 GB/s ceiling measured in §9.2. The FFN is
74% of decode.

### 13.2 Accuracy: three ways to shrink the model, each measured (`quant_sensitivity.py`)

| arm | exact | mean prefix | weights |
|---|---|---|---|
| `q8_0` — linears Q8, **embed bf16**, fp32 scales | **8/8** | 24.0 | **2174 MB** |
| `q8_0h` — fp16-rounded scales, ±127 clamp | 7/8 | 20.9 | 2082 MB |
| `q8_0_embq8` — linears Q8, **embed Q8** | 6/8 | 19.6 | 1836 MB |

The engine's `__half` scale (the pq1/ggml Q8_0 convention) is **not** the arm
that was certified: prompt F dies at token 6. Quantizing the tied embedding
costs 2 more prompts and collapses the worst case to prefix 1. Prompt F is the
canary in every arm (the bf16 control itself reaches only 17/24 there).

So the byte floor at fixed accuracy is **2174 MB**: linears at Q8_0 with fp32
scales, embedding bf16. §9 closed the per-group direction (Pareto-dominated),
this closes the scale-dtype and embedding directions. The bytes are not
reducible at 8/8.

### 13.3 The tied lm_head was missing from every budget

`tie_word_embeddings=true` and the census lists no `lm_head`, so every token
streams the whole 128000×2048 table: 524 MB/token bf16 = **35.0 ms** at
15 GB/s, 24% of decode. The "1.558 GB/token" figure counted only the 1466 M
Linear params. At the certified config the honest GEMV budget is

  127.7 − 17.9 (Q8 lm_head) + 35.0 (bf16 lm_head) = **144.8 ms → 6.90 tok/s**

### 13.4 Memory, measured on the board — not by accounting

`cudaMemGetInfo` on the clean Nano: **2742 MB free at idle, 1867-1872 MB with
a co-resident service up.** The certified config needs ~2184 MB (2174 weights
+ ~10 arena). So it **runs on a quiet Nano and does not fit a contended one**.
Supporting observations:

* All allocations succeed at 2.88 GB free, including first-touch writes into a
  1728 MB block. The first "invalid argument" failures were contention, not a
  hard wall — the initial 1.77 GB reading was the service, not the board.
* A progressive 256 MB-step probe ssh-dropped its own session. Do not probe
  Nano memory in big steps.
* The harness now checks every `cudaMalloc`, prints free memory in MB, and
  waits up to 30 min for the board to drain, because a transient is not a
  result. `MAXTOK` (the KV cache) is benchmark-sized at 64 (6 MB); a
  256-token context costs 25 MB and needs the headroom to match.

### 13.5 Engine state

`pulsar_asm/lfm3/tools/export_lfm3_q8.py` writes `.lfm3q8` (1946 MB, 286
tensors) with a per-tensor crc32 and `--selfcheck` → **SELFCHECK OK, 286/286**.
`pulsar_arm/asm/lfm3_nanode.cu` loads it, re-asserts every crc32 on the bytes
it loaded (so "the engine runs the certified quantizer" is a load-time
assertion), and runs the real `Lfm3DecoderLayer.forward` from
`modeling_lfm3.py`: sandwich norms, per-head QK norms over head_dim=128, GQA
16q/4kv, sigmoid head gate before o_proj, the conv op's `B*xg → 7-tap depthwise
→ C*y` ring, zero-tail partial RoPE (θ=1e6 pr=0.5 full / θ=1e4 pr=1.0 sliding),
SwiGLU, tied logits, greedy argmax. Activations are bf16 rounded at torch's
points — the §ref_lfm3_np lesson. CUDA 10.2 gates the bf16 intrinsics behind
`__CUDA_ARCH__ >= 800`, so bf16 is raw `unsigned short` plus hand-written
RN-even conversions.

**Open:** end-to-end ms/token (needs a quiet board), the non-GEMV elementwise
cost (unpriced; ~28% of decode in the Bonsai engine), and NEON ports of the
elementwise ops — the "mostly assembly" half of the target, for which
`lfm3_conv.S` (bit-exact, 40.2 µs/step) is the template.

### 13.4 The "mostly assembly" side: NEON elementwise kernels, bit-exact (commit pending)

`pulsar_arm/asm/lfm3_elem.S` + `lfm3_elem_test.c`, built and run on the Nano:
`lfm3_rmsnorm_bf16`, `lfm3_mul_bf16`, `lfm3_add_bf16`, the ops the decode step
runs 96 + 26 + 48 times per token.

    $ gcc -O0 -ffp-contract=off -o lfm3_elem_t0 lfm3_elem_test.c lfm3_elem.o -lm && ./lfm3_elem_t0
      rmsnorm  H=8,16,24,32,64,128,512,1024,2048,4096   all ok (bit-exact)
      rmsnorm  40 seeds at H=2048: bit-exact (0 / 81920 elements)
      mul  n=2048,7680  bit-exact        add  n=2048,7680  bit-exact
      AAPCS64 d8-d15 contract: ok
      rmsnorm H=2048   9.38 us   x96/token = 0.90 ms/token
      mul   n=7680    14.47 us   x26/token = 0.38 ms/token
      add   n=7680    14.28 us   x48/token = 0.69 ms/token
      elementwise total (these 3 ops) = 1.96 ms/token
    LFM3_ELEM_OK                (identical at -O0 and -O2)

**Answer to the question the plan posed.** These three ops cost **1.96
ms/token** on one A57 core -- 1.5% of the 127.7 ms GEMV budget, not the ~28%
the Bonsai engine pays (there it is the 4000-position attention, not
rmsnorm/mul/add). The elementwise side is not where the remaining decode time
is. mul/add are bandwidth-bound at ~3.8 GB/s, near the A57 single-core
ceiling, so they are close to optimal as written. What remains unmeasured on
the non-GEMV side is the **attention** (softmax over positions) and the
`exp()`-bearing ops, which stay on the GPU: this path is libc-free (svc
syscalls only, house style), so a polynomial `exp()` is not bit-comparable to
CUDA's `expf`. Disclosure, not omission.

bf16 arithmetic is hand-built (`shll` to widen, RN-even integer sequence to
narrow) because FEAT_BF16 is Armv8.4 and A57 is ARMv8.0. `rmsnorm` rounds to
bf16 **before** multiplying by the weight, because `modeling_lfm3.py` says
`self.weight * hidden_states.to(input_dtype)` -- rounding after is the common
way this gate fails. The variance reduction order is pinned in the header
comment (8 accumulators, chunks ascending, `((a0+a1)+(a2+a3))+((a4+a5)+(a6+a7))`)
and the reference mirrors it, so the gate asserts equality and never closeness.

Four bugs, all found by the gate, all instructive:

1. **`fmov w26, s21` clobbered x26** -- x19-x28 are callee-saved. The `-O0`
   build ran clean; the `-O2` build died on `ldr w20,[x26]` in `main` with x26
   = `0x3f086752`, a float's bits. The d8-d15 checker is blind to integer
   registers, so the `-O2` build is the real detector for this class and is
   now run as part of the gate.
2. **The RN-even tie bit was read from the wrong position.** `(u>>16)>>31`
   reads bit 47 of u, not bit 16, so every exact tie rounded down. Caught on
   `add`: `a+b = 2.1484375` exactly, asm `0x4009` (odd) where IEEE RN-even
   gives `0x400a`. It was also the cause of rmsnorm's "1 ulp at the ~0.2% tie
   rate" failures -- one macro, three symptoms, and the 1-ulp signature looked
   like a `sqrt` mismatch, which sent the investigation there first.
3. **`ldr q28, [x6], #32`** advanced the weight cursor by 32 instead of 16, so
   lanes 4-7 multiplied by `w[8..11]` and the cursor ran 32 B/chunk off the end
   of `w` -- the source of the rmsnorm `i>=4` failures *and* of the heap
   corruption that faulted the optimized test.
4. **`shll2 v25.4s, v22.8h`** read v22's stale upper half-lanes (`shrn` writes
   only the low 4h): each narrowed half must be widened from its own register.

Harness bugs found on the way, recorded because they were the misleading
symptoms: the parity list tests `H=4096` against buffers sized for `H=2048`;
the failure diagnostic printed the reference's float *value* next to the
kernel's *bits*, which made a mismatch print as `asm=407c ref=407c`; and mul
and add shared one output buffer, so mul's diagnostic printed add's results.

binutils 2.30 (the Nano's `as`) also rejected the bare vector `lsr`, the
indexed `fadd Vd.2s, Vn.2s, Vn.2s[1]`, `dup Vd.4s, Sn`, and `fmov Sd, Vn.s[n]`
(wants plain `mov`) -- the same gap class the repo already documents for
`smovl`/`vcvt`/`vext`, extended here.
