# LFM3-Audio-2B on a PULSAR asm engine — scope, facts, and plan

Status: **scoped and partly measured. No engine code exists for this model on
any target.** Every number below was read from `LiquidAI/LFM3-Audio-2B`
(`config.json`, the safetensors headers, the file listing), from
`.auto/HANDOFF_NANO_Q1_0.md`, or measured on the two Jetson Nanos, on
2026-10-28 — not estimated. §8–§10 are the Nano verdict; §9 is the one that
decides it.

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

So on Nano the work is **not** "port a model"; it is "re-specialize the GEMV
family for K=2048/7680" plus the §3 kernel gap (17 of 22 primitives absent —
depthwise conv + conv state cache, partial rotary, dual rope theta, sliding
window, sandwich norm, per-head gate — all new, in NEON/CUDA rather than AVX2).

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
  frozen.

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
