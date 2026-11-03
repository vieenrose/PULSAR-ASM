# LFM3-Audio-2B — remaining port plan: from "backbone done" to a self-contained asm + C runtime

Status: **planning document**. The text backbone is ported and gated (see
[`pulsar_arm/asm/README-lfm3-audio.md`](../pulsar_arm/asm/README-lfm3-audio.md)); everything below
is the work that turns that engine into a runtime that takes an audio file and produces text and
speech with no Python, no HF and no llama.cpp in the loop.

All sizes and shapes in this document are **census facts** taken from
`~/models/lfm3/LFM3-Audio-2B/model.safetensors` and its `audio_tokenizer/model.safetensors`
(safetensors headers, 2026-10-31). All timings are **derived** from those bytes at the board's
measured rates (11–17 GB/s; backbone step measured at 112.8 ms). Nothing here is a guess dressed
as a measurement — where a number could be either, it says which.

---

## 0. Definition of done

A single binary, built with `nvcc` + `as` + `ld`, that:

1. reads a **16 kHz mono WAV/PCM** stream (or a token-id list for text-only use),
2. runs mel → encoder → projector → backbone, and emits **text tokens** through a ported BPE
   tokenizer (no Python at runtime),
3. at `--speak`, continues with the backbone → depth head (MaskGIT) → codec decoder and emits a
   **24 kHz WAV** stream,
4. keeps every component **certified by an HF-derived gate** (§4), so "it fits in the budget" is
   never the argument — the gate is,
5. runs inside 4 GB unified memory with everything resident, and reports its own per-component
   timings (`LFM_TIME`) and peak RSS.

The house style from `.auto/vision.md` applies: **asm + C only in the inference path**, no mocks,
no edited outputs, every divergence disclosed with its root cause, and each stage gated before the
next starts.

## 1. What the checkpoint actually still owes (census)

| component | prefix | tensors | bf16 | params | Q8 estimate |
|---|---|---|---|---|---|
| text backbone | `lfm.` | 286 | 3458 MB | 1728.8 M | **1946 MB** (shipped) |
| audio encoder | `encoder.lfm.lfm.` | 158 (10 conv x 8 + 7 attn x 11 + embedding_norm) | 219 MB | 109.5 M | ~123 MB |
| encoder downsample front end | `encoder.downsample.` | 12 (6w+6b; GGUF cross-check §13) | 3.7 MB | 1.85 M | fp32, keep |
| audio→backbone projector | `mlp.` | 6 | 11.5 MB | 5.77 M | ~6.5 MB |
| MaskGIT depth head | `depthformer.` | 37 | 67.8 MB | 33.9 M | ~38 MB |
| 12 codebook logits | `depth_embeddings.` | 36 | 99.2 MB | 49.6 M | ~56 MB |
| backbone→12 streams | `depth_linear.` (+bias) | 2 | 50.4 MB | 25.2 M | ~28 MB |
| audio codebook table | `codebook_embedding.` | 1 | 99.1 MB | 49.6 M | ~56 MB |
| mask token | `mask_embedding.` | 1 | 0.004 MB | 1 K | fp32 |
| LiquidCodec **decoder only** | `decoder.*` (`audio_tokenizer`) | 104 | 73.0 MB | 36.5 M | **keep bf16** |
| tokenizer | — | — | 18 MB json | — | ~20 MB binary tables |

Structure, as the engine will have to reproduce it:

```
encoder   17 layers, hidden 768, FFN 1792, 8 heads / 4 KV, head_dim 96, sliding 32, conv_L 3
          10 conv layers (0,1,3,4,6,7,9,10,12,13), 7 attention layers (2,5,8,11,14,15,16)
          NO sandwich norm (operator_norm + ffn_norm only)
          attention layers use q_layernorm/k_layernorm [96] (per-head norm, not the [96] dim of
          the backbone's [128]) and out_proj (not o_proj); conv layers are [2304,768] -> 3x768
          front end: conv 3x3 -> conv 3x3 -> conv 3x3 (stride), lin 1x1 -> lin 1x1 -> lin_out
          [768, 4096, 1] over (128/8 mel bins x 256 channels) => 12.5 tokens per second of audio
          (CORRECTED run #303: three stride-2 convs shrink BOTH axes 8x; 4096 = 16 x 256, not a
          4-frame concat. Measured with HF get_audio_features: 125 embeddings for 10 s, 375 for 30 s.)
depthformer 3 layers, hidden 1024, FFN 2816, 32 heads / 8 KV, head_dim 32, 12 positions,
          is_causal FALSE, 2 norms per layer, maskgit_steps 3, 12 codebooks, vocab 2017
projector   LayerNorm(768) -> 768->2048 -> activation -> 2048->2048   (note: the output is 2048,
          the backbone's hidden size — this is the audio/embedding boundary)
codec      FSQ: 12 codebooks x 4 levels [8,7,6,6] = 2016 codes + 1 mask = 2017 per codebook
          decoder: from_quantizer 256->512 (FSQ dequant) -> nearest-exact x6 upsample
                   -> 8-layer LFM3 @512 (5 conv / 3 attn, hd 64, 8 heads, kv 128, FFN 2304,
                      conv_L 7, sliding_window 30, sandwich norm)
                   -> lin 512 -> 1282 -> polar(exp(logmag), angle) = 641-bin complex spectrum
                   -> irfft(1280) * hann(1280) -> overlap-add, hop 320 -> 24 kHz
          80 ms per frame (frame_samples 1920), 12.5 frames/s, 75 upsampled positions per frame
```

Three structural conclusions follow immediately, and they set the shape of the whole plan:

* **Every remaining component is the same block family as the backbone** (short-conv layers,
  GQA attention with per-head norms and a head gate, SwiGLU FFN, rmsnorm) at different widths.
  The kernel work is parameterisation, not new math — except for the two DSP surfaces (mel, iSTFT)
  and the FSQ quantizer.
* **The codec decoder is the only part of LiquidCodec that output needs.** The encoder half
  (71 MB) turns audio into codes, which the LFM3 audio encoder already does for our input path.
  Dropping it saves 71 MB and a whole stage.
* **The encoder is only needed while reading input, the codec only while writing output.** They
  never have to be resident simultaneously — a free ~123 MB (§6).

## 2. Two decisions the plan takes up front

**D1 — GPU for streamed weights, CPU (AArch64 NEON asm) for DSP and control.**
Measured: the GPU GEMV runs at 17 GB/s of the 21.9 GB/s bus (backbone, saturated), the mel front
end costs ~10 ms per 5 s of audio on the A57s, and the tokenizer is microseconds per string. So:
mel, tokenizer, sampling, WAV I/O, the interleaving controller and the FSQ index↔value mapping go
to C/asm on the CPU; encoder, backbone, depth head, codec decoder and iSTFT go to CUDA. The
existing `lfm3_conv.S` / `lfm3_elem.S` NEON kernels stay as the bit-exact host reference for the
conv/norm path that the CUDA kernels must match.

**D2 — every audio-side component is written batched (GEMM-shaped) from day one; the backbone is
not, and that has to be fixed.**
This is the single largest engineering decision in the plan, and it is forced by arithmetic:

| component | batched cost (weights streamed once per chunk) | unbatched cost |
|---|---|---|
| encoder, 96-frame chunk (0.96 s audio) | 219 MB / 17 GB/s ≈ **13 ms** | 100 frames/s × 219 MB = 21.9 GB/s → **1.3 s per audio second** |
| codec decoder, one 80 ms frame | 73 MB / 17 GB/s ≈ **4.3 ms** | 75 positions × 73 MB = 5.5 GB → **320 ms per frame = 4× realtime** |
| depth head, one frame (3 maskgit passes) | 3 × 68 MB ≈ **12 ms** | same (it is already 12 parallel positions) |
| backbone, one frame | **112.8 ms measured** (bandwidth-bound, cannot batch across a frame) | — |

**Correction 2026-10-31: the table above is bandwidth-only and understates the
batched column wherever the compute roofline binds.** Units first, because the
plan has mixed them: peak is **236 G FMA/s** (256 cores × 921.6 MHz; one FMA =
one MAC); the sweep tool reports 2-counted GFLOP/s, so divide by 2 before
comparing to peak (the "3% of peak" remarks are really ~1.5% — the qualitative
point stands, the arithmetic is fixed here). Roofline per component from census
shapes (MACs = params × positions × passes; minimum = max(bytes/17 GB/s,
MACs/236 GFMA/s), i.e. 100% efficiency, unattainable, a floor not a quote):

| component | MACs | bandwidth floor | compute floor | binding | realistic (30–50% eff) |
|---|---|---|---|---|---|
| encoder / 0.96 s chunk (100 M params × 24 pos) | 2.40 G | 7 ms | 10.2 ms | compute | **10–20 ms** (D2's 13 ms stands) |
| depthformer / frame (33.9 M × 12 pos × 3) | 1.22 G | 4.0 ms | 5.2 ms | compute | |
| to_logits + depth_linear / frame (50 M × 12 × 3) | 1.80 G | 5.9 ms | 7.6 ms | compute | depth total **13 ms min, 25–40 ms realistic** (D2's 12 ms was right number, wrong reason) |
| codec transformer / frame (36.5 M × 75) | 2.74 G | 4.3 ms | **11.6 ms** | compute | **25–50 ms realistic (D2's 4.3 ms needs 1100 GFLOP/s — unachievable)** |

Consequences: (1) the codec was understated 2.7× — 4.3 ms assumed the ALUs could
sustain 1100 GFLOP/s; (2) S0's true need is set by depth+codec ≈ 5.8 GMAC/frame:
25 ms combined needs ~230 GFMAC/s (~97% of peak), so the 60-floor (30 GFMAC/s)
is a **halfway milestone, not the need** — at 60 the pair still costs ~190 ms;
(3) the backbone alone (112.8 ms measured) already exceeds an 80 ms frame, so
speech-out on this board is **≥1.7× realtime at 100% efficiency and ~2.2×
realistic, regardless of S0** — S0 decides whether audio adds ~25 ms or ~200 ms
to the backbone's 113 ms, not whether speech is realtime. D2's decision
(everything batched) stands; its numbers are superseded by this table.

Unbatched codec decoding is fatal (4× realtime slower); batched it is 5% of an 80 ms frame
budget. The encoder goes from "as expensive as realtime" to free. **So: no component after the
backbone may be written token-at-a-time.** The backbone already exists and is token-at-a-time,
which is fine for its 1-step-per-80 ms speech role (RTF 1.41, §3) but not for audio *prefill*,
where 12.5 tokens per second of audio (corrected #303; was 25) at 98 ms/step = 1.23 s per audio second. That is stage S8,
and it is the one place where the plan has to touch shipped numerics (see §5, risk R3).

## 3. What the finished runtime costs (derived)

Speech out: one backbone step (112.8 ms) + depth head (≈13 ms minimum, 25–40 ms
realistic) + codec (≈11.6 ms minimum, 25–50 ms realistic) per 80 ms frame
→ **≈137 ms minimum (1.7×) and ~180 ms realistic (≈2.2× realtime)**; a 5 s answer
takes ≈11 s. The old "129 ms / 1.6×" was bandwidth-only. Speech in
(ASR path): 1 s of audio = 12.5 encoder tokens (corrected #303; this paragraph said 25) = batched
encoder ~0.06 s + backbone prefill 12.5 steps × 98 ms = 1.23 s (unbatched, today) or ~0.56-0.61 s
batched at the measured 30-33 GMAC/s (18.4 GMAC per audio-second). See section 18.

The 112.8 ms backbone step is **76% of the per-frame budget** and it is bandwidth-saturated at
1946 MB/step. Nothing in this plan can shrink it except fewer bits per weight — which is a quality
decision, not a porting one, and the plan does not take it (the measured verdict on this model is
that 4-bit does not survive the gates).

## 4. Stages, each with its gate

Gate conventions, reused everywhere (from the backbone work): the reference is **HF's own output
dumped at the exact boundary being scored**, the tolerance is stated per gate, floors are
ratchets, and a stage is not "done" until its gate is green on a fresh build.

### S1 — tokenizer (`liquid7.1`, 128 k vocab)

*Build*: `mkbpe`-style build-time tool (precedent: `pulsar_arm/tools/mkbpe.py`, which already
emits the binary table the asm encoder in `core.S` consumes) run over `tokenizer.json`; C/asm
encode + decode against that table.
*Gate*: id parity with HF on a fixed corpus (≥1 k strings: ASCII, accents, CJK, emoji, code,
leading/trailing whitespace, byte fallback) **and** detokenize(encode(x)) == HF's round-trip.
*Cost*: µs per string. *Risk*: pre-tokenizer regex differences — enumerate the classes in the
corpus rather than trusting one string.

### S2 — log-mel front end (no weights)

Spec (from `config.json → preprocessor`, exactly): 16 kHz, 128 mels, n_fft 400, win 400, hop 160,
hann **periodic**, center **false**, power 2, slaney mel + slaney norm, f_min 0 / f_max 8000,
log10 with floor −12, then affine `(x − 4) / 4`; right padding to ≥400 samples with alignment
offset 400 / multiple 160.
*Build*: C + NEON: framing → hann → rfft(400) → power → 128×201 sparse mel → log10 → affine.
`rfft(400)`: 400 = 16 × 25, so a radix-2 stage plus a radix-5 stage; fixed size, no general FFT.
*Gate*: per-frame max |Δ| against HF's `feature_extraction_lfm3_audio.py` on 5 fixtures (silence,
white noise, a sweep, a real speech clip, an exact-multiple and a ragged length) — target ≤1e-6,
and if the HF path's float order is mirrored, **bit-identical**. **This gate is load-bearing**:
every encoder gate rides on it, so it is green before S3 starts.
*Cost*: ~10 ms per 5 s of audio (measured earlier).

### S3 — audio encoder tower + downsample

Downsampler, exactly (`Lfm3AudioDownsampler.forward`): dense Conv2d(1→256, 3×3/2, pad 1)
→ SiLU → depthwise Conv2d(256, 3×3/2, groups=256) → 1×1 → SiLU → depthwise → 1×1 →
SiLU → `lin_out` Conv1d(4096→768, 1×1, no act) over the flattened 4-frame groups;
every stage has a bias, every conv stage is preceded by a padding mask
(`lengths=(lengths+1)//2` per stride-2). Projector (`Lfm3AudioProjector`): LN(768)
→ Linear(768→2048, bias) → **GELU** (not SiLU) → Linear(2048→2048, bias).

*Build*: batched Q8 GEMV/GEMM at K∈{768, 1792} (new `GemvCfg` entries, with the coverage assert),
3×3 strided 2-D conv + 1×1 convs for the front end, 4-frame concat + `lin_out`, attention with
hd 96 / 8 heads / window 32 / per-head norms, conv_L 3 (extend `k_conv7`/`k_shift_in` to a
width parameter), FFN 1792, **no sandwich norm** (2 norms/layer — do not copy the backbone's 4).
*Gate*, in this order: (a) per-layer × per-frame rms matrix vs HF, ≤3%; (b) projector output rms;
(c) encoder output fed through the ported projector into the backbone reproduces HF's pos-0 logits
on an audio prompt (top-1 equal, corr ≥0.9995). Then run the existing text gates to prove nothing
regressed.
*Cost*: ≈13 ms per 0.96 s chunk batched. *Risk*: Q8 on the encoder may fail its own
quant-sensitivity test → keep bf16 (219 MB, still fits, §6).

### S4 — MaskGIT depth head, 12 codebook heads, sampler

*Build*: three **non-causal** attention layers over 12 positions (small enough for one block per
frame, all of q/k/v in shared memory — the causal `k_attn_head` does not apply and must not be
bent into this), the per-head gate `[32,1024]`, stacked `depth_linear` (12288×2048, **plus its [12288] bias**) and 12
`to_logits` (2017×1024 each) as one `N = 12288` and one `N = 24204` GEMV at repack time, then the
maskgit loop (3 steps) with the vendor-corroborated sampler (§13: mmproj KV gives exactly
temperature 1.0, top-k 128, repetition
penalty 1.1 over a 4096 window, on raw logits in the fork's chain order.
*Gate*: (a) per-layer rms vs HF on teacher-forced states; (b) per-step top-1 agreement over
12 codebooks × 20 frames; (c) **full codebook-token stream equality** vs HF on 5 fixtures with a
fixed seed (ties handled by scoring streams, not by chasing float logits — the lesson from the
backbone's reference-precision ties).
*Cost*: ≈12 ms per frame. *Risk*: sampler tie behaviour is the only part that can differ without
a numerical cause; it is fixed by seeding and by comparing streams.

### S5 — LiquidCodec decoder (spike first, then the stack)

*Build*, in this order:
1. **spike, standalone**: FSQ dequant → `from_quantizer` → nearest-exact ×6 → `lin` → polar →
   **irfft(1280) × hann, overlap-add hop 320**. Gate this against `torch.fft.irfft` +
   `F.fold` on random spectra before any transformer work: SNR ≥ 60 dB, and record whether we
   can be bit-exact (the reference computes in float32 with `norm="backward"`).
2. the 8-layer LFM3 stack at 512 (5 conv / 3 attn, hd 64, sandwich norm, sliding 30), batched over
   the 75 upsampled positions of a frame.
*Gate*: (a) predicted spectrum per frame vs HF (rms ≤3%); (b) **waveform SNR ≥ 40 dB** vs HF's
decoder on 5 fixed codebook sequences (a neural codec's waveform has no bit-exactness to promise,
so the gate is stated in dB and the number is recorded, not fitted); (c) a listening-length
regression: 3 s of output must be free of clicks at frame joins (the overlap-add envelope is
where that shows up).
*Cost*: ≈4.3 ms per frame batched. *Risk*: this is the least-known surface in the port — hence
the spike ordered first, before it is wired to anything.

### S6 — joint controller: interleaving, modes, repetition windows, streaming state

*Build*: the generation loop from `modeling_lfm3_audio.py` in C: text/audio/interleaved modes,
`<|audio_start|>` / `<|mixed_start|>` / `<|text_end|>` / `<|im_end|>` control tokens, 6 text / 12
audio slots per interleaved block, text and audio repetition windows (4096 / 4096), suppress-token
lists, per-mode stop conditions, and the speech-frame/codebook bookkeeping.
*Gate*: token-stream equality with HF's own `generate` on 5 fixtures per mode with fixed seeds,
and **mode-for-mode** (a text-only fixture must not start speaking, and vice versa).

### S7 — runtime shell: I/O, chunking, memory, CLI

*Build*: WAV (or raw PCM) reader → 16 kHz mono; chunker emitting hop-aligned mel chunks of
8 × 12 frames for the live-ASR path; ring buffers for the encoder conv state and both KV caches;
WAV writer at 24 kHz; `--speak`, `--tokens-only`, `--from-ids FILE`; `LFM_TIME` per-component
timings; peak-RSS report; single-command regression script.
*Gate*: end-to-end on the 5 fixtures equals the per-stage composed result (no drift introduced by
chunking), and a 60 s stream stays memory-flat (the backbone's 300-step RSS test, extended to the
audio path).

### S8 — backbone prefill batching (the only stage that touches shipped numerics)

*Why it is in this plan*: 1 s of audio = 25 backbone prefill tokens at 98 ms/step = 2.45 s. At
audio-length prompts the input path is the dominant cost, and it is the one place where the
existing engine is not bandwidth-saturated.
*Blocker, known and documented*: the `ffn.w2` GEMV indexes x at `row*K`, so rows 1..2047 read
step-recursive arena scratch; no per-position tiling reproduces those bytes, which is why the
#165 integration was closed. Batching needs a w2 x-layout that is a real `[N][K]` matrix.
*Consequence to state up front*: this changes the engine's output at the byte level, so the
existing byte-goldens are superseded — the **HF gates**, not the old engine output, are what
certify it. That is a user decision (it was recorded as such in the backbone README), and S8 does
not start without it.
*Gate*: batched prefill's per-position logits equal the sequential path's within the logit-gate
tolerances on every position (not just the last), plus the full existing suite green.
*Cost when done*: 25-token prefill goes from 2.45 s to ~0.3–0.6 s per audio second (FLOP-bound
estimate, to be measured).

### S9 — packaging

One binary, blobs beside it (`backbone.lfm3q8`, `audio.lfm3q8`, `codec.bf16`, `bpe.bin`,
`mel.bin`), `--help`, and a single `checks_lfm3_audio.sh` that runs S1–S7's gates plus the
backbone suite in one command.

## 5. Order, dependencies, and the risks that reorder things

```
S1 tokenizer ─┐
S2 mel ───────┴─> S3 encoder ──> [S8 batched prefill (perf, user decision)]
S4 depth head (independent of S3) ──┐
S5 codec spike (independent) ───────┴─> S6 controller ──> S7 shell ──> S9 packaging
```

Rationale for that order: S1 and S2 are cheap, independent, and unblock every gate downstream
(nothing about the encoder can be scored without a bit-reproducible mel fixture). S3 and S4 are
independent of each other and can proceed in parallel. **S5's spike starts early even though S5
finishes late**, because the iSTFT is the only surface in the whole port with no precedent in this
repository — if it is going to be a problem, we want to know in week one. S6/S7 are the glue and
need S3 and S5's interfaces frozen first.

Risks, with the fallback already decided:

| # | risk | fallback |
|---|---|---|
| R1 | encoder fails its Q8 gate | keep bf16 (219 MB); budget already allows it |
| R2 | codec waveform SNR < 40 dB at bf16 | keep the decoder in fp32 (146 MB, still fits) and disclose the number |
| R3 | w2 layout change refused | audio input stays 2.45 s per audio second; the runtime is usable for short utterances only, and the README says so |
| R4 | maskgit/sampler ties differ from HF | gate on codebook streams with fixed seeds; document tie density like the backbone's 4.5% |
| R5 | everything resident does not fit | load order mel→encoder→backbone→(evict encoder)→head→codec; encoder+codec never co-resident (§6) |
| R6 | mel not bit-identical to HF | fixed-point-identical frames are not required, but the encoder gate tolerance must be re-derived if mel differs; prefer mirroring HF's float order |

## 6. Memory budget for the finished runtime

| resident set | bytes |
|---|---|
| backbone (Q8, shipped) | 1946 MB |
| encoder (bf16) + downsample + projector | 234 MB |
| depth head + depth_embeddings + depth_linear + codebook + mask (bf16) | 217 MB |
| codec decoder (bf16) | 73 MB |
| tokenizer + mel tables | ~20 MB |
| KV @4096, allocated per layer today | 192 MB (90 MB if restricted to the 11 attention layers) |
| CUDA arena (unchanged, do not shrink) | 60 MB |
| **total, everything resident** | **≈2.74 GB** of 4.16 GB |

With the §6 rule (encoder and codec never co-resident) the worst case drops to ≈2.6 GB, and the
Q8 paths for encoder/head would take it to ≈2.4 GB. The board currently shows 0.79 GB free with
the backbone loaded, so the remaining components fit with room for the arena — **measured before
each stage, not assumed**: every stage's first action is `cudaMemGetInfo` + `MemAvailable` in its
own run, exactly as the backbone did.

## 7. Milestones (what a user can actually run)

| M | after | what it does | gate |
|---|---|---|---|
| **M1 audio → text** | S1, S2, S3, S7-lite | 16 kHz WAV in, detokenized text out | HF greedy text equal on 5 clips |
| **M2 text → speech** | S4, S5, S6, S7 | text prompt in, 24 kHz WAV out | codebook-stream equality + waveform SNR ≥40 dB |
| **M3 speech → speech** | all of the above | audio in, audio out (the model's actual product) | end-to-end on 5 fixtures |
| **M4 usable latency** | S8 | audio-length input without the 2.45 s/audio-second penalty | batched-vs-sequential logit gates |

M1 is the first shippable thing and is worth aiming at deliberately: it needs no MaskGIT, no
codec, no iSTFT, and it exercises the entire input path against HF.

## 8. Kernel and tool inventory still to write

| kernel / tool | surface | new or parameterised |
|---|---|---|
| ~~hann + rfft(400) + 128-mel + log10 + affine~~ **DONE** | CPU, NEON asm | shipped: `pulsar_arm/asm/lfm3_mel.S` + `lfm3_mel.c`, gate `.auto/lfm3/checks_mel.py` (S2, §10) |
| ~~irfft(1280) + hann + overlap-add hop 320~~ **DONE (spike)** | GPU | shipped: `pulsar_arm/asm/lfm3_istft.cu`, gate `.auto/lfm3/checks_istft.py` (S5 spike, §10) |
| BPE encode/decode + build table | CPU, C/asm | **new** (precedent exists in `tools/mkbpe.py` + `core.S`) |
| hann + rfft(400) + 128-mel + log10 + affine | CPU, C/NEON | **new** |
| 3×3 strided conv, 1×1 conv, 4-frame concat | GPU | **new** (`downsample`) |
| batched Q8 GEMV/GEMM (positions × weights) | GPU | **new** — the highest-value kernel in the plan (S3/S4/S5 all use it, and S8 reuses it) |
| conv width parameter (3 vs 7) | GPU | parameterise `k_conv7`/`k_shift_in` |
| per-head norm at hd 96 / 64 / 32 | GPU | parameterise `k_headnorm` |
| non-causal attention over 12 positions | GPU | **new** (small, must not reuse the causal path) |
| FSQ dequant (levels 8/7/6/6 → 2016 codes) | CPU | **new**, trivial arithmetic |
| nearest-exact ×6 upsample | GPU | **new**, trivial indexing |
| irfft(1280) + hann + overlap-add hop 320 | GPU | **new** — spike this first |
| sampler: temp/top-k/top-p/min-p/repetition penalty | CPU | **new** in this tree (chain order precedent: the Bonsai CPU sampler) |
| WAV read/write, chunker, ring buffers | CPU, C | **new** |

## 9. What to measure before the plan's estimates are quoted as facts

1. Encoder Q8 vs bf16: quant-sensitivity on the encoder's own output (pattern:
   `.auto/lfm3/quant_sensitivity.py`), since §6's 123 MB assumes Q8 passes.
2. Codec decoder bf16 vs fp32: SNR per §4-S5 on 5 sequences.
3. Batched GEMV efficiency at the audio shapes (K = 768 / 1792 / 512 / 2304, tile 12 / 75 / 96
   positions) — **MEASURED 2026-10-30, and the answer is negative for the naive form**:
   `pulsar_arm/asm/lfm3_gemm.cu` sweeps 18 audio shapes x T in {1,6,12,24,75} x 7 tile sizes and
   every single one lands at **7.0-7.1 GFLOP/s** (3% of the board's ~230 GFLOP/s fp32 peak), with
   per-position cost flat in T (batching win 0.01-0.23x, i.e. worse than the T=1 GEMV), and
   T=1 weight streaming at 1.0-3.5 GB/s against the engine's measured 17 GB/s. Accuracy is fine
   (fp64 gate: max relative error 2.9e-8 everywhere), so this is a structural limit, not a bug:
   one shared-memory load per (weight, position) pair. Consequence: **the D2 cost table stays
   DERIVED**, and S3's GEMM is specified as a port of `q8_partial`, whose x is register-staged.
4. iSTFT determinism: `torch.fft.irfft` vs our fixed 1280-point inverse on the same spectra.
5. MaskGIT tie density, as the backbone's reference-precision work did for text.

Until those five exist, every timing in this document stays labelled *derived*.

## 10. Status (updated as stages land)

| stage | state | evidence |
|---|---|---|
| **S2 log-mel front end** | **done, gated** | `pulsar_arm/tools/mkmel.py` (build-time: the reference's own `mel_filter_bank` + `torch.hann_window` + twiddles -> `mel.bin`), `pulsar_arm/asm/lfm3_mel.S` (window + half-length pair-form direct DFT 400->201, 128x208 mel matmul, clamp+affine -- NEON, documented reduction order), `lfm3_mel.c` (HF's `pad_length`/`num_frames` padding, `log10f`), `lfm3_mel_test.c` (bit-exact C mirror). Gate `.auto/lfm3/checks_mel.py`, 19 fixtures: **asm vs C mirror 29,412 checks 0 mismatches (bit-exact)**; vs HF frame counts exact on all 19, live-cell (within 6 decades of the frame peak) max 8.3e-6, live mean 2.2e-7; exact-f64 referee says the engine is as accurate as HF (worst mean-ratio 1.28); all-cell max 0.0499 disclosed (one chirp cell 13.5 decades under its peak, where HF is 0.039 from exact). **Read the two claims as different things:** the 29,412-check bit-parity is against
*my own* C mirror, i.e. self-consistency between the asm and the documented
reduction order -- it cannot detect a shared mistake in that order. The correctness
claim is the HF comparison, which is tolerance-based (8.3e-6 live-cell, 2.2e-7 live
mean) because HF itself uses pocketfft plus BLAS and is therefore not a bit
reference. Neither claim is "bit-exact to the reference". Cost **0.554 ms/frame** = 5.5% of realtime, so the front end is free. |
| **S2b WAV front** | **done, gated** | `pulsar_arm/asm/lfm3_wav.c`: RIFF/WAVE -> f32 mono at 16 kHz (PCM16, IEEE float32, extensible for both, mono/stereo `(l+r)*0.5f`, any chunk order, junk chunks, odd sizes, lying RIFF size). Refuses other rates/bit depths/channel counts/truncation with reasons -- **no resampler, no channel guessing**. Gate `.auto/lfm3/checks_wav.py`: 9 fixtures **bit-exact vs numpy** (0 cells differ), 7 refusals rc!=0, and 6 chain fixtures WAV -> lfm3_wav -> lfm3_mel vs the cached HF mel at live max 6.26e-06 (S2 bar 1e-05). §32. |
| **S5 iSTFT spike** | **done, gated** (transformer stack not started) | `pulsar_arm/asm/lfm3_istft.cu`: direct inverse 641->1280 in a documented order, x hann, overlap-add hop 320 with a fixed 4-term contribution order (no atomics), envelope divide, `trim=480`/`tail=960` per `decode`. Gate `.auto/lfm3/checks_istft.py`: **9 cases, SNR 103-130 dB vs torch** (cases built from the checkpoint's real `decoder.lin` weights, bf16-rounded like the engine will, plus a 440 Hz tone and a window-centred click), well past the 60 dB bar; max abs scales with the case (relative bound 1e-3 x rms). **Decision: the `__sincosf` arm ships** -- it is faster in 9/9 cases (the 3.2 MB/frame twiddle table is DRAM-bound). Cost 0.7-4.5 ms/frame (board drift included; 75 frames per audio second => 5-11% RT, acceptable next to the backbone's 141% RT). **Follow-up: an FFT form** (1280 = 2^7 x 5) if it ever matters -- the direct form is SFU-bound on `__sincosf` (measured 2 ms/frame vs a 1.64 MFLOP/frame FLOP count => 0.8 GFLOP/s of ~100). |
| batched Q8 GEMM (D2's enabling kernel) | **measured, not shipped** | `pulsar_arm/asm/lfm3_gemm.cu` + `.auto/lfm3/checks_gemm.sh`: fp64 gate PASS on all 18 audio shapes x 5 T x 7 tiles (max relative error 2.9e-8). Performance verdict: **7.1 GFLOP/s ceiling on every shape, no batching win** (per-position cost flat in T; T=1 streams weights at 1.0-3.5 GB/s vs the engine's 17 GB/s). Two bugs the fp64 gate caught: an out-of-bounds write when T is not a multiple of TILE (`ceil(T/TILE)` tiles need the `t0+t<T` guard on both the x load and the store -- symptom: scattered zeros plus a poisoned CUDA context that made every later shape "fail to allocate"); and scalar `int8` weight loads (2.4x at T=1, and the 40x-off streaming rate). Root cause of the ceiling: the inner loop reads x from shared once per (weight, position) pair -> shared-bandwidth bound. **Action: port `q8_partial` (x register-staged, KCHUNK partials, R rows/block, warp-shuffle reductions) to the batched form; do not ship a from-scratch GEMM.** |
| S1 tokenizer | **done, gated (encode+decode)** | `pulsar_arm/tools/mktok.py` (blob: vocab 124893 + 293320 merges + 140 added + 782 L/N ranges + 10 White_Space ranges + byte map, 5.4 MB) + `pulsar_arm/asm/lfm3_tok.c` (hand-rolled Split regex, ByteLevel, ignore_merges BPE) + gate `.auto/lfm3/checks_tok.py`: **531/531 fixtures id-for-id vs HF (9988 ids) AND byte-exact round-trip** decode(encode(x))==x on all 531. The A6 (`\\s+(?!\\S)`) trap is documented in §12. |
| S3 encoder tower | **DONE, gated (all 17 layers + projector, GPU)** | `pulsar_arm/tools/mkaudio.py` builds `lfm3-audio-2b-enc.lfm3q8` (130 MB, 176 tensors, 102 Q8 / 74 f32) with SELFCHECK OK and **176/176 dtype decisions matching Liquid's mmproj GGUF table** (census corrections in §13a). Spec gate §22 (`enc_ref/enc_numpy.py` reproduces HF's forward to 2.2e-6), C reference §23 (`lfm3_enc.c`, C-vs-numpy 3.3e-6, Nano-bit-identical §24), GPU tower §25-§28: downsampler + 17 layers + final norm + projector in CUDA, gated by `.auto/lfm3/checks_enc.sh` on **20 stage dumps + 4 attention internals**, energy-relative 8e-07 (ds_out) .. 1.3e-05 (enc_out) .. 2.7e-05 (proj_out), probes 5e-07..1e-06, determinism bit-identical through layer_04. Cost **0.0320 s GPU per audio-second** (min-of-8, 3.7 GMAC/s), 5x inside the 0.06 budget. Open: the run-to-run defect of §29-§30 (layer-0 input rows, mechanism class settled, unfixed). |
| S4 depth head + MaskGIT | not started | |
| S5 codec transformer | not started | |
| S6 controller / S7 shell / S9 packaging | not started | |
| S8 batched backbone prefill | not started; needs the `ffn.w2` layout decision (R3) | |

Two lessons the gates earned, recorded here because both are the class of bug that
survives a tolerance check:

* **AArch64 asm: capture every argument register before it becomes a cursor.**
  `lfm3_mel_power` used `x5` as the x cursor and only then read `w5` for `n_bins`,
  so the bin-loop bound was the y-pointer's low word -- the kernel computed bin 0
  and exited, leaving `pw[k>=1]` at the sentinel. `lfm3_mel_norm` took `n` from
  `w3`, which is the 4th integer slot: FP args are `s0-s2` *independently* of the
  integer slots, so `n` was garbage and the kernel stored nothing. Neither is
  visible to the house `d8-d15` checker; both look like "the output is zero".
* **CUDA: stdio in a `.cu` host function hits CUDA's device-side `fopen/fread/
  fwrite` overloads** ("too few arguments in function call", pointing at the wrong
  line). Use POSIX `open/read/write` in tool drivers, as `lfm3_istft.cu` does.

## 11. S0: the batched Q8 GEMM throughput floor (OPEN, with a measured negative)

S0 exists because D2's cost table is a **bandwidth** statement (weights streamed
once at ~17 GB/s) and the first GEMM I wrote ran at 7.1 GFLOP/s. The 7.1 turned
out to be my bug, not a wall; the corrected kernel is 11-28 GFLOP/s with the fp64
gate green on every sampled output of 18 audio shapes x T in {12,24,48,75}
(max relative error 3e-8). Status: **the floor is NOT met.**

| | value | needed for D2 |
|---|---|---|
| throughput | 11-28 GFLOP/s | ~60 GFLOP/s |
| effective weight streaming | **0.9 GB/s** | 17 GB/s |
| fraction of DRAM bound | 0.0-0.1x | 1.0x |

What the four fixes bought, in the order they were found (each measured):

1. **Uncoalesced weight loads** were the 7.1 explanation. Lane *b* owned k-block
   *b*, so the eight lanes of a group read bytes 128 apart: 32 distinct 32-byte
   sectors per load instruction for 512 useful bytes. Predicted 0.5 MAC/SM/cycle
   against 0.96 measured. Putting all eight lanes on one 128-element block, each
   taking a 4-byte quarter so the group reads 32 contiguous bytes, gave
   **7.1 -> 30 GFLOP/s**.
2. Chunking K to a single 128-element scale block instead of holding `[TILE][K]`
   in shared, so TILE is not capped at `sharedMemPerBlock/4K` = 9.
3. TILE is capped by **registers**, not shared: `acc[TILE]` plus `ps[TILE]` are
   live, TILE=16 needs ~110 regs/thread, TILE=24 needs 159 and the per-block
   register limit refuses the launch.
4. The store needed an `active` guard: for N not a multiple of 32 the surplus
   groups wrote `C[(t0+t)*N + n]` for n = N..N+29 into the next row at n-N.

**The attempt to close the rest of the gap failed, and the failure is
informative.** Register-blocking the way D2 implies — one `int4` (16-byte) weight
load per lane per block so the group reads a full 128-byte cacheline, `float4`
shared reads, and the block scale folded into the weights to drop `ps[TILE]` and
free registers for TILE=32 — ran at **5-11 GFLOP/s, i.e. slower than the code it
replaced**. ptxas confirms no spills (93 regs max), so the cause is the shared
pattern, not local memory: with each lane owning 16 *contiguous* floats, the eight
lanes of a group sit 16 floats apart and alias onto **2** shared banks, where the
4-float stride of the 4-byte-load form hits **8** distinct banks. Four times fewer
`LDS` instructions bought four times more bank conflicts, and the register
pressure of `wz[16]` cost the rest. Kept as a recorded negative rather than
tuned away: the next attempt has to satisfy both constraints at once, which means
a swizzled shared layout (lane base = `lane*16` XOR a per-(t, v) skew) rather than
a bigger tile.

Also added to the kit, because this file's own history demands it: a
**plausibility gate** that runs before any timing is believed — the launch error
is read immediately after the launch (`cudaDeviceSynchronize` returns
`cudaSuccess` for a *refused* launch, so a kernel that never ran reads as
"all zeros, no error, 174 GB/s"), every output must be non-zero as well as
fp64-accurate, and the measured rate must be below the board's fp32 peak.

Update 2026-10-31 (v2: scale folded, float4 shared reads, TILE=24 default): **20-28
GFLOP/s** across the sweep (best 28.2 at dep 12x to_logit T=48), 71 regs/thread (down
from 110), fp64 gate still green everywhere (max rel 4e-8). The register win did not
move the throughput needle: same 10% of peak, still ALU-bound at ~0.1x of DRAM. Next
up if S0 is revisited: 32-bit indices (the kernel uses 64-bit `size_t` arithmetic for
products under 2^31), `__ldcs` streaming weight loads (weights are re-read
ceil(T/TILE) times), and occupancy (3 blocks/SM at 71 regs vs the engine's 6 at 40).
Update (v3: 32-bit indices + `__ldcs` weight loads): **+5%**, 3 interleaved pairs on
24204x1024 T=24 (v3 wins 2/3, means 28.8 vs 27.6 GFLOP/s; best single-shape 31.7 at
cod w1/w3 T=24). Same bits (gate green), kept. Remaining if revisited: occupancy (3 blocks/SM at 71 regs vs the engine's 6 at 40)
and the register-blocked outer product (x in registers). Floor NOT met; v3 is the
instrument.
Update (v4: 16 rows x 16 positions, one acc/thread, no shared/syncs, L1 broadcast;
`LFM_V4=1`): **NEGATIVE, -25% vs v3** (22.1/20.4 vs 28.0/27.0 GFLOP/s, 2 shapes x 2
passes same session). Cause identified: v4 converts weights AND x per MAC with no
reuse (16 threads redundantly convert the same 16 weights), while v3 amortizes
weight converts over TILE and x converts over 32 rows -- occupancy (7 blocks/SM
at 32 regs, 0 spill) did not compensate. v4 kept behind the flag for reference.
v4 also caught a **gate hole: NaN outputs report rel=0/PASS** (NaN comparisons are
false, so worst never updates). Closed twice: plausibility now rejects non-finite
outputs outright, and the sampled loop forces worst=1.0 on non-finite `got`. The
32-reg figure above is real (spill-free); the 23 GFLOP/s beside the original
`rel=0` was NaN throughput, not FLOPs.

Consequence for the schedule: **S3/S4/S5b stay blocked on S0.** At the measured
rate the codec decoder's 73 MB of weights cost ~80 ms per 80 ms frame (1.0x
realtime, no headroom) and the depth head costs 34 ms at T=12 against a 12 ms
target. Building the audio layers on top of that would mean shipping components
whose budget is known to be unmet. S1 is now done and gated; S2/S5a were already
green. All three are independent of S0.

## 12. S1: the tokenizer is not the simple BPE the plan assumed

`tokenizer_config.json` says `pre_tokenizer: null`. **It is wrong.** The real
pipeline lives in `tokenizer.json`:

| stage | value |
|---|---|
| normalizer | null |
| pre_tokenizer | `Sequence[ Split(regex, Isolated), ByteLevel(add_prefix_space=False) ]` |
| model | BPE, 124,893 vocab, **293,320 merges**, `ignore_merges=True`, `byte_fallback=False`, `fuse_unk=False`, no `unk_token`, no `continuing_subword_prefix`/`end_of_word_suffix` |
| added tokens | 140, ids 124,893..125,032 |
| post_processor / decoder | `ByteLevel`, `add_prefix_space=True` |

Three consequences the plan did not anticipate:

1 **The Split regex is the GPT-4 pattern**, so it needs Unicode general categories
   `\p{L}` and `\p{N}`: 648 and 134 codepoint ranges respectively, shipped in the
   blob. Nothing is approximated, and the tables come from `unicodedata` in the
   build tool, so a Unicode-version drift between Python and Rust's `regex` is a
   gate failure rather than a silent behavioural difference.
2 **`ignore_merges=True`** changes the algorithm: a pre-token that is already in
   the vocab is emitted directly, with no merge pass. A classic BPE loop is wrong
   here.
3 **Added tokens are matched on the raw text before pre-tokenisation**, longest
   first, and emitted as themselves. Omitting that made `<|im_start|>` come back as
   ten ordinary byte-level tokens.

`transformers` also prints a warning that this tokenizer uses an "incorrect regex
pattern" and suggests `fix_mistral_regex=True`. The shipped `tokenizer.json` regex
is what the reference engine uses by default, so that is what the port reproduces;
the flag is recorded here so nobody "fixes" it later and breaks bit-parity with the
engine.

Status: **DONE and gated.** `mktok.py` now writes the full blob (vocab strings by id,
293,320 merges in rank order, 140 added tokens, L/N/S ranges, byte map; 5.4 MB) and
`pulsar_arm/asm/lfm3_tok.c` + gate `.auto/lfm3/checks_tok.py` give **531/531 fixtures
id-for-id vs HF (9988 ids)** on the same corpus the replay used.

The one trap the gate caught, kept because it is the class of bug a tolerance check
hides: **`\\s+(?!\\S)` (A6) matches a proper prefix of a space run, not just trailing
whitespace.** Greedy `\\s+` takes the longest prefix whose next char is not non-space;
inside a maximal spaces-only run of length L followed by a non-space, that is L-1
(which fails for L==1, falling through to A7). So `'a  b'` splits as `['a',' ',' b']`
(first space alone, second joins the letter), `'a  1'` as `['a',' ',' ','1']`, and
`'\\xa0\\xa0\u00fc'` as `['\\xa0','\\xa0\u00fc']` (first space alone, second joins the
letter via A2's optional prefix). The first C port required the run to reach end for
A6 and failed ~11% of fuzz fixtures with swapped merge choices downstream; the
symptom (wrong ids several positions in) was far from the cause (one split boundary).
Unicode note: `\\s` is White_Space=yes (11 ranges, verified 0..0x4000 against Python
`regex` with 0 mismatches); L/N come from `unicodedata` 14.0.0, so a Unicode-version
drift against Rust `regex` surfaces as a gate failure, not a silent difference.

## 13. Vendor GGUF release (2026-Q4): what it validates and what it doesn't

Liquid published `LiquidAI/LFM3-Audio-2B-GGUF` (main BF16/F16/Q8_0/Q4_0/Q4_K_M +
mmproj BF16/F16/Q8_0/Q4_0/Q4_K_M; llama.cpp b11960+/v0.8.0, `tokenizer.ggml.pre =
lfm2`, chat templates `asr`/`tts`). Only headers were fetched (2–16 MB Range
requests); no weights downloaded. Findings, each checked against the census:

1 **Sizes validate the census to the scale dtype.** Summing the GGUF tensor tables
   with Q8_0 at its true 34 B/32 (fp16 scales) gives main 1837 MB vs the file's
   1846.2, mmproj 303.7 vs 303.8 — exact. Our `.lfm3q8` (1946 MB) is 5.4% larger
   for one reason only: fp32 block scales (36 B/32). No missing tensors: main has
   287 = our 286 + `rope_freqs` (computed, not in safetensors); mmproj's 319 are
   all accounted for (157 encoder + 10 front-end + 43 depth/code + 4 quant + ~96
   codec-tfm + 9 projector/misc).
2 **The vendor's split is our split.** Main = backbone only; mmproj = encoder +
   downsample + projector + depthformer + codebooks + the full codec decoder
   transformer (`a.gen.wav.tfm`, 8 blks @512 with sandwich norms, `in_proj`
   256→512, `out_proj` 512→1282 + biases, `output_norm`). The codec transformer
   rides the same repack path as S3/S4 — S5's transformer weights arrive via
   mmproj, not via the audio_tokenizer file. (Only `quant.first`, the FSQ
   dequant, is tiny and F32.)
3 **Structural census, confirmed field by field:** encoder attention layers at
   2,5,8,11,14,15,16; q/k/v [768/384] + q/k_norm [96] F32 + out_proj; conv layers
   [2304/768] in + [768] out with F32 depthwise [3,768] and 2 norms (no sandwich);
   FPN 1792 (not the config's 2560); depthformer 3 blks, q/k/v/o [1024/256],
   gate [1024,32], q/k_norm [32], FFN 2816, 2 norms; stacked codebook
   `embd [1024,2017,12]` + `out_embd [2048,24204]` (i.e. exactly the N=24204
   GEMV S4 already specifies); `proj_in [2048,12288]` + bias.
4 **Census corrections (small, all in the S3/S4 repack inventory):** downsample is
   12 tensors (6 weights + 6 biases), not 11; `depth_linear` has a [12288] bias;
   12 per-codebook `embedding_norm`s exist (= GGUF `head_norm [1024,12]`) plus an
   `output_norm [1024]`; projector linears all carry biases and the activation is
   GELU; encoder `out_proj` naming confirmed.
5 **Sampler corroborated by the vendor's own packaging.** mmproj KV:
   `gen.audio.sampling: temp 1.0, top_k 128, penalty_repeat 1.1, penalty_last_n
   4096` — exactly the S4 spec. Main KV: `temp 0.0, top_k 1` (greedy text) —
   exactly the shipped default. Chat templates `asr` (live-ASR system prompt) and
   `tts` (per-language voice prompts) are S6 prompt material.
6 **Q4 changes nothing in the plan.** Q4_K_M (1.10 GB) exists and runs in
   llama.cpp, but the plan's precisions are set by HF-derived gates, and the
   measured verdict on this model (4-bit does not survive the gates) stands until
   a gate says otherwise. Q4_K_M is now the download-and-run baseline for M4
   latency comparisons and the quality reference if 4-bit is ever revisited; Q8_0
   (fp16 scales) is a cross-check blob for ours (same quant, 5.4% smaller).
7 **llama.cpp is a second oracle.** b11960+ implements the full pipeline
   (serve/CLI/TTS/realtime voice). Uses: its `lfm2` pre-tokenizer as a cross-check
   for the S1 splitter (independent of both HF and our C), and its generate loop
   for S6 stream comparisons with fixed seeds.

## 13a. S3's blob: what the GGUF cross-check caught on its first run

`mkaudio.py` exports the M1 audio path (encoder + downsample + projector) in the
backbone blob's container (same magic, same certified quantizer copied verbatim,
per-tensor crc, `--selfcheck` re-derives and compares). Its distinguishing feature
is `--gguf`: a 16 MB Range fetch of Liquid's mmproj header is enough to parse their
319-tensor table, and every dtype decision is asserted to match theirs (Q8_0 <-> our
Q8, F32 <-> our f32), tensor for tensor.

Three things it caught before any tower code was written:

1 **`self_attn.out_proj` (encoder) was not in the Q8 patterns** -- the patterns had
   the backbone's `self_attn.o_proj` spelling, so the encoder's seven attention
   output projections would have shipped f32 (and the f32/f32 cross-check on
   `conv.out_proj` pattern-matched, hiding it). This is the zero-match filter class
   this repo has been burned by twice; here the *vendor's own table* is the
   referee, which is stronger than any internal assert.
2 **`lin_out` is Conv1d [768, 4096, 1]**, squeezed to 2-D for the quantizer, and is
   **Q8** in Liquid's table (`a.pre_encode.out`), not f32 as the first draft had it.
3 **The encoder census was 158, not 159**, and the missing tensor is
   `encoder.lfm.lfm.embedding_norm.weight` -- a tower-final rmsnorm the safetensors
   census had skipped and which explains GGUF's `a.post_ln`. Table §1 updated.

Result: `lfm3-audio-2b-enc.lfm3q8`, 130 MB (was 151 MB before lin_out went Q8),
176 tensors, SELFCHECK OK, 176/176 GGUF agreement. The blob is the S3 foundation;
the tower kernels and the per-layer gate against HF are the remaining S3 work.

## 14. Challenge: realtime speech-in / text-out (2026-10-31 acceptance frame)

Target set by the operator: **process speech audio in at >= 1.0x realtime sustained
(ASR_RT = wall / clip <= 1.00 on a >= 60 s clip), with the existing text gates green.**
Text-out mode sidesteps the speech-out hard limit (section 13's 24.3 GB/s arithmetic
excludes TTS at 8-bit; ASR has no such wall). The budget:

| per wall second | work | rate needed | today | verdict |
|---|---|---|---|---|
| mel | 100 frames x 0.554 ms / 100 | 55 ms CPU | measured | free |
| encoder | 2.5 GMAC (25 tok x 100 M) | 2.5 GMAC/s | v3 GEMM 16 GMAC/s -> 156 ms | fits as-is |
| backbone prefill | 43.2 GMAC (25 tok x 1.73) | **43.2 GMAC/s** | batched 11-18 GMAC/s | **THE gap: x2.4-3.9** |
| text decode | 4-6 tok/s x 1.946 GB (T=1, GEMV) | 7.8-11.7 GB/s of ~17 | 112.8 ms/tok measured | fits at <= 6 tok/s |
| KV reads | ~33 MB/tok at 4096 ctx | 0.2 GB/s | — | free |

Prefill batching amortizes the weight stream (one 1946 MB pass serves T positions),
so at T>=4 the prefill side becomes compute-bound and the text side keeps its
bandwidth — the two share the bus at ~13-14 GB/s of the measured 17. Nothing physical
excludes the target: 43.2 GMAC/s is 18% of the 236 GFMA/s peak.

### What run #294 measured (fresh build, `--prefill-h2batched`, shipped blob)

1 **The #165 blocker's premise does not reproduce at kernel level.** The kept selftest
   reports **8/8 shapes BIT-EXACT including `feed_forward.w2.weight` in both fp32 and
   H2 arms** (0/32768 and 0/16384 cells differ, batched T=8 vs per-position sequential
   at the same KCHUNK config). The kernel comment's claim -- "gemv() indexes x + row*K
   ... non-bit-exact BY CONSTRUCTION, no per-position tiling reproduces those bytes" --
   matches no visible launcher (q8_partial stages `x + koff`; rowlane stages x once;
   prefill_h2 stages `xall[tt*K + koff]`). The observed #165 divergence (position 16
   got 17-19's bytes) is consistent with a bug in the REMOVED integration's
   per-position persistence, not with the math. **Status: "by construction" downgraded
   to "integration bug, mechanism misattributed"; S8 needs one integration-level gate
   run (logits vs sequential), not a goldens waiver.** One open detail before S8 v2:
   the selftest's seq arm uses the same KCHUNK search as its batch arm (w2 f32 lands
   at KC=3840 nchunk=2, H2 at KC=7680 nchunk=1), so batch==sequential-at-that-config;
   batch==SHIPPED-sequential (KC=7680 nchunk=1 for w2, f32) must be re-checked once at
   the integration level, where the chunk-split parenthesization question lives.

2 **Decode sits at an exact crossover, which is why it has no headroom.** Q8 weights
   cost 1.125 B/MAC; the bus sustains ~17 GB/s; crossover = 15.1 GMAC/s. The H2
   kernels measure 17-18 GMAC/s at T=1 (embed: 14.7 ms/pos = 17.8 GB/s AND 17.8
   GMAC/s simultaneously). Bandwidth and compute hit together -- there is no slack on
   either side at T=1.

3 **The batched form exposes the real ceiling: ~11-18 GMAC/s of 236 (5-8% of peak),
   and it is shared by every kernel family in this repo.** At T=8, weight traffic
   drops 8x (bandwidth should free up) yet throughput stays put:

   | kernel | rate | chain depth nb |
   |---|---|---|
   | v3 `lfm3_gemm` | 16 GMAC/s | 8-64 |
   | q8_partial H2 (T=1, embed) | 17.8 GMAC/s | 64 |
   | q8_prefill_h2 H2 bat (embed, w1) | 17.9-18.2 GMAC/s | 64 |
   | q8_prefill bat (w2, K=7680) | **11-12 GMAC/s** | **240** |

   The w2-vs-w1 rate tracks the block count (dependency-chain depth): the inner H2
   unpack+HFMA2 loop and the outer `acc` chain are both serial. Four independent
   implementations converging on one number means the limit is the instruction
   structure, not shared memory, not banks, not occupancy, not bandwidth.

4a **Two structural hypotheses tested and refuted (run #295), both in the tool:**
   (i) chain split `LFM_SPLIT=2` (SPL independent acc chains routed by
   (b*4+ci)%SPL, summed at end; SPL=1 bit-identical) -- rates unchanged:
   w2 12.4/12.0 vs 12.4/14.1, w1 13.5/14.5 vs 14.3/14.6, q 10.3/11.8 vs
   12.6/13.1 (2 interleaved passes). The OUTER chain is not the binder.
   SPL=4 compiled to 135 regs (SPL=2: 86, SPL=1: 69 -- ptxas register use is
   erratic across the instances) and its launch was refused; the launch check
   caught it as designed. (ii) staging granularity `LFM_G2` (stage 2 k-blocks
   per sync-pair, 17.4 KB, same occupancy, order preserved): w2 +7%, w1 -4%,
   audio ~0 -- barrier count is not the binder either.
4b **Cross-finding: the ENGINE's q8_prefill is 2.6x the tool's rate at batch
   T=8** (w1: engine bat 7.01 ms for 126 MMAC = 17.9 GMAC/s vs tool 6-8 GMAC/s
   on the same shapes). What the engine does differently: stages the whole
   KCHUNK in **bf16** (17 half2 = 34 B per 32 inputs vs the tool's 4.25 B/input
   -- 4x less shared traffic), one sync per chunk not per 128 elements, and the
   HFMA2 byte-perm unpack. The G2 test at G=2 was too timid to emulate that --
   the real granularity is KCHUNK-scale, which needs bf16 staging to afford
   (float staging of K=2048 at TILE=16 = 128 KB, impossible; bf16 = 8.7 KB).
   **Instrumentation pivots: the challenge's kernel work now happens against
   q8_prefill_h2 in-engine (test-only arm, H2 selftest as harness), not
   lfm3_gemm.** The remaining lever list for x2.4: split the inner 8-deep half2
   chain into two 4-deep accumulators, interleave the outer acc by block parity,
   then raise CT (register cost watch: acc[CT] floats).
4 **Therefore the challenge reduces to: break the serial chains to >= 45 GMAC/s
   batched (x2.4) while keeping T=1 behavior at 17 GB/s (decode untouched).** The
   experiment, exact design: interleave 2-4 independent accumulators in the inner loop
   (N partial dot products summed at the end of each block) plus 4-8 independent chunk
   partials for the outer `acc` (summed ascending, so T=1 summation order and every
   shipped bit are preserved when the interleave count is 1 = current code). Register
   cost is the risk (v1's 159-reg TILE was refused); budget <= 110 regs for 2
   blocks/SM. Gate: fp64 relative 2e-5 (existing) + a timing sweep against today's
   11-18 GMAC/s; if >=45 holds on w2/w1/embed, the budget table above closes to
   ASR_RT <= 1.00 with margin (prefill 43.2 GMAC at 45 = 960 ms -- needs the decode
   overlap shown in the table: <= 6 tok/s text on the bandwidth side).

### 14c. First in-engine lever banked: CT=4 (run #296)

`q8_prefill_h2` gained two TEST-ONLY template knobs (the harness is
`--prefill-h2batched`, pairs stay bit-exact by construction): `LFM_PF_CT`
(batch width 2|4|6|8) and `LFM_PF_ARM` (0 = shipped structure; 1 = split inner
half2 chain + parity-interleaved outer acc). `LFM_PF_T` sets the tested batch
size (24 = one ASR chunk of 0.96 s audio). Results, H2=1, ARM=0, T=24, all
BIT-EXACT vs the structure-consistent sequential pair (8/8 shapes in every
config):

| bat T=24 | CT=2 (today) | CT=4 | ratio | CT4 rate |
|---|---|---|---|---|
| embed 128000x2048 | 283.29 ms | **211.43** | 1.34x | 29.8 GMAC/s |
| w2 2048x7680 | 24.28 | **11.71** | 2.07x | 32.2 GMAC/s |
| w1 7680x2048 | 17.80 | **12.82** | 1.39x | 29.4 GMAC/s |
| conv.out 2048x2048 | 4.76 | **3.52** | 1.35x | 28.7 GMAC/s |

**CT=4 is the optimum of {2,4,6,8}** (CT=6 and CT=8 are regressive: w2 at
CT6/T24 is 31.6 vs CT4's 11.7 -- the extra `acc[CT]` registers cost more
occupancy than the chains return; CT8 embed 160.9 vs 71.2 at T=8). **ARM=1
(chain split) is redundant once CT=4** -- four per-position chains already hide
the serial latency, and the split's register cost made it slower than plain
CT4 (w1 4.97 vs 4.43 at T=8). ARM=1's numerics are UNRESOLVED by design: its
selftest pair (ARM1-T1 vs ARM8-arm) is self-consistent by construction, so only
an external reference tests the values; vs ARM0 the f32-arm delta was 1e-4
(ordinary reassociation) but the H2-arm max-rel hit 41.7 on near-zero cells
(f32 fp16-in-block cancellation), which max-rel cannot judge -- needs an
f32-reference quality metric before ARM1 is ever used. ARM=0 stays the
recommendation.

**Budget status after this win.** Prefill needs 43.2 GMAC/s of audio-second;
at CT4's 30 that is 1.44 s, plus text decode 4 tok/s x 113 ms = 0.45 s of
bandwidth time on the same serial GPU -> **~0.53x realtime**, not yet 1.0. The
realtime target needs prefill compute >= ~79 GMAC/s (at 4 tok/s decode): a
**x2.6 from today's 30**. An instruction-issue count (4 IPC x ~20 lane-MAC per
warp-instruction at this loop's mix) bounds ~75 GMAC/s, so the remaining gap is
approximately the stall gap -- plausible, but stall counters are not available
on this board (nvprof needs root), so the next probes are scheduling variants
that are testable blind: staging/compute overlap across the c loop, HFMA2 chain
interleaving that does not spend registers (reuse the c-loop slots), and an
ARM variant combined with CT=2 where its registers fit. Also recorded: w2 at
CT4 uses KC=3840 (nch=2) vs shipped KC=7680 (nch=1) -- bit-exact vs its own
config's pair but numerically distinct from shipped; that shape needs the same
integration-level gate §14/14a already scoped.

Speed claims here are same-session same-binary A/B of env knobs; the board
baseline drift caveat does not apply within one harness run.

### 14d. RETRACTED (run #299): the ARM=5 "5x / ~100 GMAC/s" was a no-op kernel

**Everything below this banner in 14d that concerns ARM=5, the "binder was
shared-load traffic" conclusion, the 1.0-1.2x realtime budget, and the "KC
chooser truncation" harness note is WRONG and is kept only as a record of the
error.** What happened: in the harness the row count came from `LFM_PF_RPB`
but the ARM=5 kernel's `GRP` template came from `LFM_PF_RB` (default 2). With
RPB=4/8 the kernel's `if (GR != GRP) continue;` guard skipped all compute and
only staged x into shared memory (time ~ N/R: embed 109.9 -> 55.0 ms exactly 2x
between GRP 4 and 8). Both sides of the self-consistency pair emitted zeros, so
the pair printed BIT-EXACT; `seq[0]=0` and `ARM1-vs-ARM0 max rel 1` were
printed and not read. The one matched config (GRP=2) was real and was 2.6x
SLOWER than ARM0 -- the evidence was in the same log. This violated the
plausibility rule recorded at the start of S0 (every output non-zero).

Corrected measurements, T=24, CT=4, H2, same binary, nonzero outputs verified
(harness now fails on all-zero output or a rate above 236 GFLOP/s, and
refuses a GRP/R mismatch loudly):

| kernel | embed GMAC/s | w2 | w1 |
|---|---|---|---|
| ARM=0 (shipped structure) CT=4 | **29.7** | **32.3** | **29.3** |
| ARM=5 GRP=2 | 12.1 | 11.1 | -- |
| ARM=5 GRP=4 | 14.5 | 13.6 | -- |
| ARM=5 GRP=8 | 9.4 | 10.7 | -- |

ARM=5 is slower than ARM=0 at every GRP (consistent with the live decode-swap
measurement, 304 vs 107 ms/token, which used the real GRP=8 kernel). So:

* The realtime budget is back to **~0.53x** (CT=4 ARM=0, 29-32 GMAC/s); the
  remaining gap is again x2.6 to ~79 GMAC/s. CT=4's 1.35-2.07x (nonzero,
  plausible, bit-exact) STANDS.
* "Shared-load traffic is the binder" is UNSUPPORTED: ARM=3 and ARM=5 both cut
  shared loads and both got slower; ARM=2 (fewer ops, higher occupancy) was
  slower; ARM=4 (prefetch) was -7%. Four hypotheses measured out, none
  confirmed; the binder is still unidentified.
* "Occupancy formally refuted by the fastest config at 126 regs" rested on the
  bogus run; the occupancy evidence is only ARM=2 (66% occ, slower).
* The KC "truncation" note was wrong: CT=8 at K=7680 picks KC=1920 (nch=4, as
  the 196608-cell count shows), not 240.
* What survives from the S8 work (section 15): ARM=5 (H2) T=1 == q8_partial
  bit-exact on 153/153 real-activation configs, and the shipped-kernel delta.
  Those compared nonzero data and stand.

Original (incorrect) section follows, unedited, for the record.

### 14d. The binder was shared-load traffic: ARM=5 gives ~5x, ~100 GMAC/s (run #297)

Four arms were built and measured on `q8_prefill_h2` after CT=4, all
bit-exact against their structure-consistent sequential pairs (8/8 shapes in
every configuration), all in one binary so the ratios are drift-immune:

| arm | mechanism | verdict (w2 / w1 / embed, T=24, CT=4) |
|---|---|---|
| ARM=2 | convert weights once per `b`, interleave the CT chains across `k` (−75% conversion ops) | **refuted**: 16.67 / 13.99 / 222 ms vs ARM0's 11.70 / 12.75 / 211. 56 regs, 66% occupancy -- *higher* occupancy, fewer ops, slower. `max rel = 0` vs ARM0 (numerically identical, as designed) |
| ARM=3 | register-blocked rows (x reused across `RB` rows, 2x less shared traffic) | **refuted**: 52.6 / 47.8 / 772 ms (4.5x slower). At RB=2: 63 regs, no spills, 66% occupancy -- so not spills either |
| ARM=4 | software-pipelined weight loads (prefetch `b+32` before computing `b`) | **refuted**: 12.61 / 13.67 / 224.9 ms (−7%) -- weight-load latency was not the binder |
| **ARM=5** | **x into registers, rows inner**: load the 16 x half2 once per `(b,c)` into registers, loop the warp's rows inside; rows/warp set small (`GRP`) so the accumulators fit | **5.0x vs the shipped CT=2 structure**: 4.62 / 3.40 / 55.0 ms. ~80-114 GMAC/s |

Why ARM=5 works and ARM=3 did not: the LSU accounting says 64 shared half2
loads per `(b,row)` x GR=32 rows x nb=120 x 128 warps = 31.5M warp-cycles of
shared-load issue on 4 SMs, ~7.9 ms of w2's measured 11.7 ms. ARM=5 makes
shared loads per `b` = `CT*16` **independent of rows/warp** (they are amortised
over GRP rows in registers); ARM=3 cut the same traffic only 2x and paid more
in restructured instructions. Corollary: at GRP=8 the kernel runs at 126 regs,
**2 blocks/SM (33% occupancy)** and is the fastest of all -- occupancy is
formally not the binder, which the ARM2/ARM3 results already implied.

Achieved rates at the ASR chunk shape (T=24), CT=4/GRP=8:

| shape | shipped CT=2 | ARM=5 CT=4 GRP=8 | ratio | GMAC/s |
|---|---|---|---|---|
| embed 128000x2048 | 283.29 ms | **55.0** | 5.14x | 114 |
| w2 2048x7680 | 24.28 | **4.62** | 5.08x | 81 |
| w1 7680x2048 | 17.80 | **3.40** | 5.01x | 111 |
| conv.out 2048x2048 | 4.76 | **0.97** | 4.25x | 104 |

Reproducibility: embed 54.92 / 55.05 / 55.08 ms over three repeats (±0.3%).
T=8: embed 19.14 ms vs the shipped 95.07 (5.0x), w2 1.21 vs 8.43 (7.0x).
T=4: embed 10.02, w2 0.69, w1 0.65. The T=1 launches (the sequential side of
the pair, i.e. today's decode-adjacent prefill shape) also improve 4x: embed
seq 338 -> 84 ms -- that is a TTFT-relevant win independent of batching.

**Realtime budget after ARM=5.** Prefill work is 43.2 GMAC per audio-second; at
the measured 80-114 GMAC/s that is 0.38-0.54 s, plus text decode 4 tok/s x
113 ms = 0.45 s -> **~0.83-0.99 s GPU per audio second = 1.0-1.2x realtime**.
First time the target is met on paper. Status: NOT achieved -- this is a kernel
measurement, not an end-to-end run (encoder tower unbuilt, no integration run
yet), and the decode share assumes 4 tok/s.

Two harness facts to carry forward. (1) The ARM=5 config needs rows/warp small
(the harness sets `R = WPB*RPB`; GRP=8 -> R=64 is the best; GRP=2/4 are 2-3x
slower; GRP=16 ran 8/8 bit-exact but its timings were filtered out of the
captured output and need a rerun to record). (2) The harness KC chooser has a truncation bug for CT=8
at K=7680: it picks KC=240, where nb = KC/32 = 7 truncates and 16 columns per
chunk go uncomputed (both sides share it, so the pair still prints BIT-EXACT).
CT=2/CT=4 configs are exact (KC=7680/nb=240 and KC=3840/nb=120). Wiring ARM=5
into the engine must set KC exactly, and the w2 KC=3840 (nch=2) vs shipped
KC=7680 (nch=1) question still needs the integration gate of section 14.

### 15. S8 gate built: ARM=5 == the port's reference GEMV, bit-exact on real activations; the shipped-kernel swap is measured out (run #298)

`--prefill-arm5-gate` captures the **real** activations every shipped `gemv()`
call sees during prompt A (24 ids + 24 greedy = 25 rows/tensor, 153 shape/KC
configs), then replays them through the batched kernels. Three comparisons per
config:

| comparison | result |
|---|---|
| ARM=5 (H2) T=1 vs `q8_partial` f32 partials, bit-for-bit | **153/153 BIT-EXACT** |
| ARM=5 (H2) T=1 vs the shipped `q8_gemv_hs` output | 153/153 MISMATCH, ~99.1% of bf16 cells, max rel ~1e38; **6 configs straddle the bf16 overflow asymmetrically** (layers 12 `w1`, 23 `conv.in_proj`: one side inf, the other finite) |
| ARM=5 (f32, H2=0) T=1 vs shipped | 22/153 bit-exact, 131 mismatch, 24 launch refusals at shm>48K (now reported) -- no bit-exact-vs-shipped form found |

So ARM=5 is exactly the port's reference order (the same one every published
gate was built on, `q8_partial`) -- including w2 at KC=3840/nch=2, which
closes the section-14 chunk-split question -- and the engine's shipped decode
kernel `q8_gemv_hs` is a **different summation order by construction**. The
engine's prefill runs through `gemv()`, i.e. through `q8_gemv_hs`, so adopting
ARM=5 for prefill is an order change, not a bit-exact substitution.

**The swap arm was built and measured out.** `LFM_PF_USE=1` routes `gemv()`
through `q8_prefill_h2<CT=4, ARM=5, GRP=8>` with `g_prefill_phase` scoping it
to prompt steps. Two findings:

* **Goldens.** Every quantitative gate stays green -- matrix 23x23 within 3%,
  pos-0 top-1 = HF's with corr 0.99998, per-prompt top-1 100% (F 20/21), KL
  0.0006-0.0019 -- but prompt **F's greedy prefix moves from 16/16 to 11/16**
  (L is the known near-tie in both arms): 5/7 vs the shipped 6/7. Prefill order
  propagates through the KV cache into the generated stream.
* **Speed.** `decode_ms_per_tok 304.34 vs 107.17` (2.8x WORSE; same session,
  two measure.sh runs). The harness job's 20 steps all sit inside the prompt
  window, so they all take the arm, and ARM=5's T=1 form is the wrong shape for
  a decode step: R=64 means 2 blocks/SM and it loses the per-shape R/TPB tuning
  `pick()` gives the shipped kernel. (CORRECTED #299: the "5x win at batched
  T>=4" that used to be asserted here was the no-op kernel; the real ARM=5 is
  slower than ARM=0 at T=24 too.)

**Verdict (CORRECTED in run #299): keep the shipped decode/prefill kernel.**
The sentence that stood here -- "ARM=5 enters through S8's batched prefill ...
expected TTFT ~0.42 s at ~100 GMAC/s" -- was built on the retracted 14d numbers;
ARM=5 is in fact 2-3x SLOWER than ARM=0 at T=24 (see the 14d banner). A batched
S8 prefill would use CT=4 ARM=0 (29-32 GMAC/s, bit-exact vs its own sequential
pair, q8_partial order), which is ~0.53x realtime, not 1.0x. Its numerics are
q8_partial's, not q8_gemv_hs's, so the F-stream sensitivity measured above
applies to it as well.

Measurement-discipline additions from building this gate: a top-of-function
capture reads the *previous* call's output (first version: 99.95% cells
differed, rel ~1e38); non-finite cells are a failure only when **asymmetric**
(the shipped path's own bf16 inf at layers 12/23 is reproduced bit-identically
and is not this gate's business); and a refused launch is silent -- the f32
arm's first version measured zeros because nothing read `cudaGetLastError()`.

### 16. Coalesced weight staging: ARM=6, bit-exact and 2x SLOWER in the engine (run #300)

The bench (`pulsar_arm/asm/bench/bench_h2.cu`, 4164a32) showed coalesced loads
at 2.9x the direct pattern in isolation. The engine test did not transfer.

ARM=6 stages each warp's 32 consecutive 32-byte weight blocks (1 KB, coalesced
int4 loads) into a per-warp shared slice, then each lane reads its own block.
Same lane->b map and k-chain as ARM0, so the arithmetic is ARM0's.

Results, T=24, CT=4, H2, nonzero verified, same session:

| tensor | ARM0 (shipped structure) | ARM6 (coalesced staging) | max rel vs ARM0 |
|---|---|---|---|
| embed 128000x2048 | 29.8 GMAC/s | 16.9 | 0 |
| w2 2048x7680 | 32.0 | 8.6 | 0 |
| w1 7680x2048 | 29.3 | 16.7 | 0 |
| conv.out 2048x2048 | 27.7 | 16.4 | 0 |

ARM6 is bit-exact (every cell, ARM1-vs-ARM0 max rel 0) and 2-3.5x slower.
Plausible explanation, not measured: each 32-block group adds shared stores
(8 words per lane) and loads, plus a warp sync, to the path the arm was meant
to relieve. That is the shared-traffic direction the ARM3/ARM5 results already
ruled out. The bench's 2.9x was for a kernel with x in global memory, so the
bottleneck it removed is not the engine's.

Verdict: refuted in the engine. The shipped structure (ARM0, CT=4) stays.
The binder is still unidentified; this run removes "uncoalesced weight loads"
from the candidate list for this kernel.

### 17. Binder search, measured (runs #301-#302): two co-binders; corrected ceilings

All numbers: T=24 (one ASR chunk), H2, warm steady-state (`LFM_PF_REPS=10`,
second half of back-to-back launches), same session, nonzero outputs.

**Board facts that change the ceilings.** The Nano GPU is **one** Maxwell SM
(`multiProcessorCount=1`; the TX1 has two). `sm_clock.cu` (clock64 vs
cudaEvent) shows ~904 MHz sustained under load, after a ~0.4 s DVFS ramp
(first 390 ms averaged 348 MHz); warm vs cold harness timings agree within 2%,
so earlier rates were not taken at low clock. The occupancy API gives the
shipped-structure CT=4 H2 kernel **2 blocks/SM = 16 warps (25%)**.
`bench/peak_rates.cu` measured ceilings: HFMA2 **105.8 GMAC/s** (45% of the
236 datasheet; ~2 warp-HFMA2/clk), FFMA 82 GFMA/s, LDS.32 100.5 GB/s, and the
engine's exact chain shape (4 chains x 16 deep HFMA2, operands in registers)
**97 GMAC/s at 16 warps**. The plausibility gate is tightened from 236 to
**120 GMAC/s** accordingly.

**SASS of the b-loop (ARM0 CT=4):** 212 instructions per iteration (one 32-weight
block x 4 positions per lane): 64 LDS, 64 HFMA2, 16 PRMT, 16 HADD2, 8 LOP, 8 F2F,
3 LDG, rest overhead. Conversions are hoisted across positions. (Two parse
traps: dual-issue pairs print with `{` and end in `}` not `;`.)

**Phase profile (ARM=8, clock64 per warp, bit-exact vs ARM0):** b-loops 72-81%,
reductions 11-20%, x staging ~7%, barriers <1%. Wave arithmetic on the
per-warp cycle counts reproduces every measured time (w1: 30 blocks / 2
resident x 1.10 ms = 16.5 vs 16.7 measured).

**Ablations (ARM=7, timing-only, ABL=0 control bit-exact vs ARM0):**

| removed | embed | w2 | w1 | conv |
|---|---|---|---|---|
| nothing (control) | 29.9 | 32.6 | 29.9 | 29.9 |
| weight global loads (2) | 32.4 | 28.6 | 31.9 | 30.2 |
| scale global load (16) | 35.0 | 33.5 | 34.4 | 32.5 |
| all global loads (18) | 35.3 | 32.5 | 34.5 | 32.7 |
| shared x reads (8) | 38.5 | 33.7 | 37.4 | 36.1 |
| **global AND shared (26)** | **54.1** | **57.5** | **53.0** | **50.6** |

Two co-binders: each memory class alone, removed, gains 15-28%; both removed,
1.8x. That explains why every single-lever arm failed. Correction: the first
ABL8/26 build used one register x for all positions, letting the compiler merge
the CT chains; with distinct per-position x (SASS-verified 64 HFMA2) ABL26 is
1.8x, not the 2.2x first measured. ABL=4 (no conversion) is confounded
(raw bits reinterpreted as half) and ABL=1 (no staging) faults; neither is used.

**Arms built against this (all bit-exact vs ARM0, max rel 0):**

| arm | design | embed | w2 | w1 | conv |
|---|---|---|---|---|---|
| ARM0 CT=4 | shipped structure | 29.8 | 32.5 | 29.8 | 29.8 |
| ARM9 RB=1 | k-outer loop order | 32.7 | 30.4 | 32.4 | 30.6 |
| ARM9 RB=2/4 | row blocking (1/2, 1/4 shared loads per MAC) | 27.9/29.0 | 31.7/32.8 | 27.8/28.8 | 27.8/28.8 |
| **ARM10 RB=2** | row blocking + all global loads prefetched one iteration ahead across row boundaries | **31.7** | **35.5** | **31.6** | **31.4** |

ARM10 RB=2 is the best bit-exact arm: **+6-9%** over CT=4. Neither fix alone
(row blocking, prefetch) reaches the ablation's 1.8x, and combined they recover
about a tenth of it. Not adopted into the engine (test-only, like CT=4).

**Realtime assessment.** The target needs ~79 GMAC/s. The practical HFMA2
ceiling is ~106, the engine's chain shape tops out at 97 with no memory, and
the conversions, reductions and staging that a real kernel cannot drop leave
~54 GMAC/s even with every global and shared load removed (ABL26). So **79 is
above anything this kernel family has shown, even with its memory traffic
deleted**. Speech-in/text-out realtime on this board is not reachable by
GEMM tuning alone; at the best bit-exact arm (~32-35 GMAC/s) the budget is
~0.56x. Getting further would need a different formulation (fewer
non-HFMA2 instructions per MAC, e.g. dp4a-free int8 tricks are unavailable
on sm_53) or less work (smaller/quantized-activation audio path), which are
scope decisions, not tuning.

### 18. CORRECTION (run #303): 12.5 audio tokens per second, not 25 -- realtime is borderline, not out of reach

Every realtime budget in sections 2, 3, 13a, 14, 14c, 14d and 17 used **25 backbone
tokens per second of audio**. That figure came from reading `lin_out`'s 4096-wide
input as "4 concatenated 256-wide frames". The code
(`Lfm3AudioDownsampler`) is three stride-2 convolutions over the (mel, time)
plane: both axes shrink 8x, and 4096 = (128/8 mel bins) x 256 channels. Measured
with HF's own `get_audio_features`: 13 / 63 / 125 / 375 embeddings for
1 / 5 / 10 / 30 s of audio = **12.5 per second** (dim 2048, after the projector).

Also measured: the batched rates do not depend on the chunk size. A 0.96 s chunk
is now T=12, and T=12 runs at the same GMAC/s as T=24 (CT=4: 29.7 / 32.5 / 29.7 /
29.7; ARM10 RB=2: 31.6 / 35.3 / 31.3 / 31.4; embed/w2/w1/conv, all bit-exact).
Text rate: the tokenizer gives 1.141 tokens per word on English prose (99 words ->
113 tokens); at 2.5-3.5 words/s of speech that is 2.85-4.0 text tokens/s.

Corrected budget per second of audio:

| item | old | corrected | basis |
|---|---|---|---|
| backbone prefill | 43.2 GMAC | **18.4 GMAC** (12.5 x 1.47; the head is not run on audio tokens) | measured token rate |
| prefill time | 1.31-1.44 s | **0.56-0.61 s** | measured 30-33 GMAC/s |
| encoder + downsampler + projector | uncounted | ~1.9 GMAC, **~0.06 s** | estimate |
| text decode | 0.45 s | **0.31-0.44 s** | 1.141 tok/word (measured) x 2.5-3.5 words/s (assumed) x 110 ms |
| prefill non-GEMM (attention, norms, conv) | uncounted | **unmeasured** | |
| **total** | **1.9 s (0.53x)** | **~0.93-1.11 s (~0.9-1.07x)** | |

Consequences:
* Section 17's "realtime out of reach for this kernel family" is **withdrawn**. The
  kernel ceiling analysis there stands, but it was compared against a 2.3x
  overstated requirement. At 12.5 tokens/s the required prefill rate for a 1.0 s
  total is ~18.4 / (1.0 - 0.06 - 0.31..0.44) = **29-36 GMAC/s**, which the measured
  bit-exact arms (30-35) are at.
* It is **borderline, not achieved**. The margin depends on speech rate (decode
  share) and on the unmeasured non-GEMM prefill work, and nothing has run end to end
  (the encoder tower is unported, batched prefill is not integrated).
* The next deliverable is therefore the end-to-end path, not more GEMM tuning:
  S3 encoder tower (gated per layer against HF), then S8 batched prefill
  integration, then an `ASR_RT` measurement on >= 60 s of real speech.
* Also corrected in place: the engine README's projector ("3 x 768->768" ->
  LayerNorm + 768->2048 + GELU + 2048->2048) and its encoder cost (per mel frame at
  100/s -> per token at 12.5/s, batched).

### 19. S3 spec gate: the audio encoder re-derived in fp32, matches HF to 2e-6 (run #304)

`.auto/lfm3/enc_ref/dump_encoder.py` dumps HF's own stages for a fixed synthetic
log-mel input (no audio, no mel front end). `.auto/lfm3/enc_ref/enc_numpy.py`
re-derives the whole encoder from the checkpoint's float32 weights and compares
stage by stage.

Result, T=400 mel frames (50 tokens): downsampler 2.5e-7; layers 0-15 3e-7 to
6e-7; final encoder output 2.4e-7; projector 2.2e-6 (rms relative). Odd length
T=37 (5 tokens, not a multiple of 8): worst 9e-7. **Gate: PASS.**

Three facts in the earlier notes were wrong, and the spec found them:

1. **The encoder is NOT the LFM3 backbone block.** Its `lfm` is transformers'
   built-in `Lfm2Config` (`Lfm2Model`). Differences that matter: attention has
   q/k layernorm and `out_proj`, with **no sigmoid output gate** (the gate is
   backbone-only); no sandwich norm; FFN width 1792 (2560 auto-adjusted by
   `block_multiple_of` 256).
2. **The offline encoder uses full causal attention, not a 32-token window.**
   Every encoder `layer_types` entry is `full_attention`, so HF's offline mask is
   unwindowed. A windowed reference missed layer 2 by 1e-2; full causal matches
   to 1.3e-7. The 32-token `sliding_window` belongs to the streaming path, which
   remaps layers to `sliding_attention`. Streaming is a separate port decision.
3. **The conv block is width 3** (`conv_L_cache` 3), depthwise, causal, with no
   bias, on `B * x` gated by `C`.

Structure, confirmed by the spec: downsampler (conv_in 3x3 s2, SiLU; conv_mid
depthwise 3x3 s2, lin_mid 1x1, SiLU; conv_end depthwise 3x3 s2, lin_end 1x1,
SiLU; flatten channel-major (c*16 + h); lin_out 4096->768, no activation), then 17
pre-norm layers (RMSNorm eps 1e-5, RoPE theta 1e6 over all 96 dims as
rotate-half, q/k norm before rope), then embedding_norm, then the projector
(LayerNorm 768, Linear 768->2048, GELU exact, Linear 2048->2048).

HF's `hidden_states[17]` is post embedding_norm, so the gate compares layers
0-15 against HF and covers layer 16 through `enc_out`.

Next (S3 implementation): the same math in C, reading the `.lfm3q8` encoder blob
(`lfm3-audio-2b-enc.lfm3q8`), stage-gated against these dumps, then Q8 with a
gate on the quantized path. The encoder runs once per audio token (12.5/s),
so it is cheap next to the backbone.

### 20. S3 C reference: the encoder runs from the .lfm3q8 blob (run #305)

`pulsar_arm/asm/lfm3_enc.c` (C99, no dependencies beyond libm) reads
`lfm3-audio-2b-enc.lfm3q8`, dequantizes the Q8 linears once at load, and runs the
spec's math in fp32: downsampler, 17 layers (10 conv, 7 full-causal attention with
q/k norm and RoPE), embedding_norm, projector. Build:
`cc -O2 -std=c99 -Wall -Wextra -o lfm3_enc lfm3_enc.c -lm` (no warnings).

Two gates, `.auto/lfm3/enc_ref/check_enc_c.py` (T=400, 50 tokens):

| comparison | result | what it means |
|---|---|---|
| C vs numpy spec, **same dequantized blob weights** | worst rms rel **3.3e-6** (projector); every stage 1e-6 class | the C code is correct |
| C vs HF, fp32 checkpoint, encoder output | rms rel **2.6e-3** | Q8 quantization error (numpy shows the identical 2.6e-3 on the same weights) |
| C vs HF, projector output | rms rel **1.8e-2** | Q8 quantization error, accumulated through the projector |
| per-layer C vs HF | 1.5e-3 at the downsampler, 7.1e-3 at layer 12, 5.5e-3 at layer 15 | all inside the 3 % per-stage bound the backbone gates use |

**The Q8 gap is the encoder's own quantization error and is not yet accepted.**
The projector's 1.8 % rms is inside the backbone's 3 % per-stage bound, but
rms is not the criterion that matters: the real test is downstream, whether the
audio embeddings change the transcript. That needs an audio-input transcript
gate (speech in, greedy text out, against HF), which needs real speech, not the
synthetic features used here.

Reference speed: 6.9 s for 400 mel frames (50 tokens) on this host, naive loops,
dequantization per call, RoPE recomputed with `pow` per element. It is a
correctness reference, not the target; the Nano build and the GPU kernels are the
next steps. Accounted cost per audio second (plan section 18): ~1.9 GMAC, which
at naive C speed is far too slow, so the C path is for gating only.

Known simplifications (must be fixed before the engine uses it): weights are
dequantized to fp32 (420 MB resident), the conv loops are naive, and the
attention is not batched across heads.

### 21. The C encoder on the Nano (run #306)

`lfm3_enc.c` built with `gcc -O2 -std=c99 -Wall -Wextra` on the Nano (aarch64,
gcc 8.4) and run on the same blob and input (T=400, 50 tokens):

* **Agreement with x86:** bit-identical at the downsampler, layers and encoder
  output; projector rms rel 9.6e-8. The CPU port is portable at the bit level.
* **Time: 157 s** for 400 mel frames (4 s of audio), against 6.9 s on this x86 host.
  That is about 0.04 GMAC/s of useful work, against the ~1.9 GMAC per audio-second
  the encoder needs in the target. The naive reference is therefore ~200x too slow
  for realtime on the Nano, as expected: naive conv loops, per-call Q8 dequantization,
  RoPE with `pow` per element, no batching across heads.

So the encoder cannot run as this reference. The next work is the engine's own
kernels: Q8 GEMMs for the linears (existing batched H2 structure), a vectorized
depthwise conv and attention, rope tables computed once, and weights kept as
int8 on the GPU. The C file is the correctness oracle for that port, gated per stage
as above.

### 22. Encoder GEMM on the Nano, fp32 activations: 10 GMAC/s, 2.3x over the linears' budget (run #307)

`pulsar_arm/asm/bench/bench_enc_gemm.cu` includes the shipped `lfm3_nanode.cu` with its
`main` renamed, so it times the engine's kernels, not copies. T=50 positions (4 s clip),
fp32 activations (the spec is fp32), min of 5 runs, Nano, nonzero outputs and a
120 GMAC/s gate.

| encoder linear | per-position `q8_partial<float>` | batched `q8_prefill_h2<float,ARM5,GRP2>` |
|---|---|---|
| in_proj 2304x768 | 6.2 GMAC/s | 4.8 |
| out_proj 768x768 | 3.8 | 4.8 |
| w1 1792x768 | 9.2 | 6.4 |
| w2 768x1792 | 9.1 | 6.6 |
| lin_out 768x4096 | 15.9 | 3.9 |
| proj1 2048x768 | 10.7 | 6.4 |
| proj2 2048x2048 | 15.8 | 3.9 |
| **sum** | **10.4** | **4.6** |

Both arms agree bit-for-bit on every shape (max rel 0), so the batched f32 arm is correct
but slower.

Two things this changes:
1. **The linears alone need 1.45 GMAC per audio second** (116 MMAC per token x 12.5 tokens/s:
   17 layers x ~6.3 MMAC + lin_out + projector). At 10.4 GMAC/s that is **0.14 s of GPU per
   audio second**, against the 0.06 s budget: **2.3x short before the convolutions and
   attention are counted.**
2. **The fp32-activation path is not the answer.** The 30 GMAC/s the engine reached in
   section 14 used half2 arithmetic (bf16 activations, fp16 products), which is faster but
   not the spec's fp32 math. Using it for the encoder needs a tolerance gate and a
   disclosed divergence, as the research brief said.

Also found on the way: the batched f32 arm at GRP=8 is refused at launch ("too many
resources", more than 255 registers per thread); GRP=2 is the largest that launches. Its
first bench run looked like a plausible 800 GMAC/s, because a `cudaGetLastError` read had
overwritten the launch error. The bench now checks the launch error directly.

Decision for the user (not made here): either accept the half2 GEMM for the encoder with a
tolerance gate and a disclosed divergence, or keep fp32 and find about 2.3x on the fp32 GEMM
first.

### 23. ASR_RT measured on the shipped engine: 1.87x, and where the 1.87x lives (run #308)

`.auto/lfm3/measure_rt.sh` builds fresh from the tree, feeds a 60 s-speech workload
(NPREF = 12.5 x SEC audio-embedding tokens as prompt steps, NGEN = 4 tok/s of text
tokens as generation steps -- the aggressive end of the measured 2.85-4.0 tok/s), and
prints the decomposition. Board clean: 0 spinners, MemAvailable 3295 MB, GPU at 922 MHz.

| item | ms for 60 s of speech | per audio second | share |
|---|---|---|---|
| prefill, 750 audio tokens at T=1 | 78 141 | 1.30 s | 70% |
| decode, 240 text tokens | 30 862 | 0.51 s | 28% |
| mel front end (S2 gate, 55 ms/s) | 3 300 | 0.055 s | 3% |
| encoder | not wired | — | — |
| **ASR_RT (engine GPU only)** | **112 302** | | **1.872** |

With the encoder added at the best measured rate (section 24, batched half2,
0.085-0.092 s/audio-s) ASR_RT is ~1.95; at the fp32 rate ~2.03. **Realtime is 1.9x away
and it is two terms, not a tuning gap.**

**Prefill (70%) is bandwidth-bound by construction at T=1.** 104.2 ms/token against the
1651 MB of layer weights streamed per token = 15.9 GB/s, 92% of the 17.3 GB/s this board
achieves for the backbone. The non-GEMV kernels are 0.54% of kernel time (handoff #115),
and the serialized phase split (conv 70.0 ms + attn 79.8 ms + lm_head 3.9 per token
against a 95 ms GEMV byte floor) is launch-latency, which CUDA graphs were measured null
on this path (#118/#165-era). Nothing left to tune there: **the only prefill lever is
amortizing the weight stream across positions, i.e. the batched kernel.** At the measured
30-33 GMAC/s (CT=4 ARM0) the 18.4 GMAC/audio-s prefill becomes 0.56-0.61 s/audio-s =
~35 s for this query, so ASR_RT goes 1.87 -> ~1.24 at 4 tok/s, ~1.09 at 2.85 tok/s.
That integration is the handoff's open item 2: bit-exact batched prefill requires the
`ffn.w2` x-layout change, which changes the goldens (run #165 root cause).

**Decode is not the short-context band at ASR context.** 128.59 ms/token over positions
750-990, against the 107-113 ms band quoted at 24-48 positions (+14%, positional, in
line with the 98 -> 112 ms growth measured in #160). Any ASR_RT budget must use ~128 ms,
not 112 ms.

**Arithmetic for the target:** at 2.85 text tok/s (2.5 words/s) a 60 s query needs
~35 s prefill + ~22 s decode + 3.3 s mel + ~5 s encoder = 65 s -> 1.09. Reaching <= 1.00
therefore needs the batched prefill AND ~10% more (decode-side, or overlapping mel and
the encoder of chunk n+1 with the prefill/decode of chunk n -- a streaming pipeline is
the clean way, since all three stages are independent resources: CPU DSP, GPU GEMV,
GPU attention).

### 24. Encoder GEMM, real weights and real activations, with an fp64 referee (run #308)

`pulsar_arm/asm/bench/bench_enc_gemm.cu` now has a `real` mode: the weights are the real
encoder tensors from `lfm3-audio-2b-enc.lfm3q8` (layer 0 conv/FFN, downsampler `lin_out`,
projector `mlp.model.1/3`), the activations are the real HF layer-0 input
(`enc_ref/ref/hidden_00`, tiled to each shape's K where that shape's own activation was
not dumped), and every arm is judged against an **fp64 referee** (48 rows x T per shape,
per-32-block dequant, ascending block order). T=50, reps 5, min time, two runs (board
drift ~20% shows in the absolute rates; the ranking is stable).

| arm | kernel | GMAC/s | s GPU per audio second (1.45 GMAC) | rms rel vs fp64 |
|---|---|---|---|---|
| B fp32/pos | `q8_partial<float>` per position | 7.7-10.3 | 0.14-0.19 | **3.2e-5** |
| A fp32/bat | `q8_prefill_h2<float,ARM5,GRP2>` | 4.1-4.4 | 0.33-0.36 | 3.2e-5 |
| F bf16/pos | `q8_partial<bf16,H2=0>` = the shipped decode GEMV | 8.3-9.1 | 0.16-0.18 | 0.275 |
| E h2/pos | `q8_partial<bf16,H2=1>` | 8.4-8.7 | 0.17 | 0.278 |
| C h2/bat | `q8_prefill_h2<bf16,ARM0>` | 15.7-18.1 | 0.080-0.092 | 0.278 |
| D h2/bat10 | `q8_prefill_h2<bf16,ARM10,RB2>` | 16.1-18.7 | 0.078-0.090 | 0.278 |

The referee settles three things that the fp32-vs-half2 comparison alone could not:

1. **The fp32 arms are right** (3.2e-5 rms vs fp64). The 800 GMAC/s and the "half2 is
   2.3x off" stories from run #307 were both bench artifacts and are superseded.
2. **The 0.28 rms deviation of every bf16-activation arm is the ACTIVATION rounding, not
   the fp16 products.** Arm F (bf16 activations, fp32 FMA math, the shipped GEMV) deviates
   by 0.275 and arms E/C/D (bf16 activations + fp16 products) by 0.278 -- a 3e-3
   difference between them. Fed bf16-rounded inputs on the host, arm F matches fp64 to
   7.7e-4 max / 1.6e-5 rms, so the kernels read bf16 correctly and the arithmetic is
   clean. Per-cell the bf16 input rounding reaches rms 0.71 on `out_proj` (max rel 35):
   the encoder's dot products cancel hard, so an 11-bit input rounding amplifies. This is
   the same arithmetic every shipped backbone GEMV uses, and it is the same mechanism as
   the Q8-vs-HF gap the S3 gate measures (2.6e-3 at enc_out, 1.8e-2 at proj_out, run #305).
3. **fp16 products are effectively free in accuracy terms** relative to the engine's own
   bf16-activation baseline: C/D are 1.9x faster than F for 3e-3 rms.

Consequences for the encoder port (decision framed for the user, not taken here):
- the encoder's linears cost 0.078-0.09 s/audio-s on the batched half2 arm -- 1.3-1.5x
  over the 0.06 s budget, and the budget's own basis (the encoder's GMAC count) is an
  estimate, so this is close enough to build and measure, not to keep tuning;
- acceptance for the audio path must be end-to-end (speech in -> greedy text vs HF),
  because a per-cell 3e-1 rel error is *normal* for bf16 activations on this data and no
  bit-exactness gate can be the standard; the C reference (`lfm3_enc.c`) stays the
  per-stage structural oracle;
- the fp32 arms are the fallback if the user wants the encoder's arithmetic to match the
  spec, at 2.0-2.4x the encoder cost.

### 25. S3 downsampler ported to the GPU: 3.4e-5 vs the C spec, 0.046-0.059 s/audio-s (run #310)

`pulsar_arm/asm/lfm3_enc_gpu.cu` includes `lfm3_nanode.cu` with its `main` renamed, so
the loader, the Q8 GEMM arms and `k_combine_f` are the shipped code. Activations are
fp32 (the spec's arithmetic), so `lfm3_enc.c` is a tight per-stage oracle. Gate
`.auto/lfm3/checks_enc.sh` (comparator `.auto/lfm3/enc_ref/cmp_enc.py`, numpy runs on
the Nano), fresh build, Tm=400 -> Tt=50 tokens:

```
ds_out   rms rel 3.449e-05 vs C   max abs 2.503e-06   rms 0.12701 both   OK
ds_out   rms rel 2.338e-01 vs HF  max abs 1.211e-03   (reported, not gated: lin_out is Q8)
ENC_GATE PASS (arms=fp32 T=50 stages=1)
```

Stage order is copied from `lfm3_enc.c` (`silu(conv_in)` -> depthwise `conv_mid` ->
`silu(lin_mid)` -> depthwise `conv_end` -> `silu(lin_end)` -> `flatten(1,2)` ->
`lin_out` + bias), with out-of-range conv taps skipped as the C reference skips them.
The 3x3 convs are direct kernels (a 9-MAC stencil); the two 1x1 convs and `lin_out` run
on GEMM paths.

**The dtype census matters here:** `lin_mid`/`lin_end` are **f32** in the blob (the same
table as Liquid's mmproj GGUF, cross-checked in section 13a), only `lin_out` is Q8. The
first run therefore died with `unspecified launch failure` on null Q8 pointers -- the
encoder needs an f32-weight GEMM as well as the Q8 arms, so `k_sgemm_f32`
(16x16 tile, one output per thread, k ascending in a single accumulator = the C
reference's order) was added and the site dispatches on the blob's dtype.

Cost: **183.5-237.8 ms per 4 s clip = 0.046-0.059 s GPU per audio second** at 458 MMAC
(1.9-2.5 GMAC/s). The whole downsampler budget is being spent on `lin_mid`
(210 of the 458 MMAC) in the untiled-rate SGEMM; the next step is register blocking
there (a numerics-neutral change, so the 3.4e-5 gate stays valid), then the 17 layers.

### 26. CORRECTION + downsampler optimized: 0.0158 s GPU per audio second (run #311)

**Section 25's cost numbers were clock-ramp artifacts.** Single-pass per-stage timings
on this board inherit the devfreq ramp (the backbone harness fights the same thing with
`LFM_WARMMS`): the first stages of a fresh process run at 77-154 MHz. The same binary
gave conv_mid at 3.81 ms in one run and 17.57 ms in the next. The encoder now runs
`LFM_ENC_REPS=N` and keeps the **min per stage**, the protocol the backbone benches use.

Paired A/B for the conv_in rewrite, both arms twice (`LFM_ENC_CONVIN=1/0`, min of 8):
shared-tile 3.88 ms vs flat work-item grid 5.69 ms, in both orders -> shared ships.

Full stage table, fp32 arms, Tm=400 -> Tt=50, min of 8:

| stage | ms | MMAC | GMAC/s |
|---|---|---|---|
| lin_mid (f32 SGEMM, 256x256 over 3200 positions) | 17.44 | 210 | 12.0 |
| lin_out (Q8 per-position arm, 768x4096) | 11.66 | 157 | 13.5 |
| lin_end (f32 SGEMM) | 5.13 | 52 | 10.2 |
| conv_in (shared-tile, 1->256 k3s2) | 3.85 | 29.5 | 7.7 |
| chw2phw permute | 3.78 | — | — |
| conv_end (depthwise) | 3.64 | 1.8 | 0.5 |
| conv_mid (depthwise) | 2.05 | 7.4 | 3.6 |
| flatten + combine | 0.68 | — | — |
| **total** | **63.0** | **458** | **7.3** |

**The downsampler costs 0.0158 s GPU per audio second (0.0146 on the half2 arms), not
the 0.046-0.059 of section 25** -- 26% of the 0.06 s/audio-s encoder budget, and the
gate is unchanged by the rewrites (`ds_out rms rel 3.449e-05 vs C, max abs 2.503e-06`
in both the fp32 and half2 arms; the SGEMM tiling, the flat conv grid and the shared
tile all preserve the C reference's summation order bit for bit).

Two rewrites were needed to get there, both measured, neither a numerics change:
1. the conv kernels' 3-D grid left 78% of the lanes idle on a 200-wide row (flat
   work-item grid, 4 outputs per thread); an ablation priced the silu `expf` at ~10 ms
   of the naive 63, so it was never the bottleneck;
2. `conv_in` re-read the same 9-tap mel window once per output channel -- all 256
   channels share it -- so staging the 3x258 tile in shared memory cut it 1.5x.

What this means for the remaining S3 work: with the downsampler at 0.016, the 17 layers
(1.4 GMAC per audio second) dominate. At the run #309 rates they cost 0.14-0.20 s/audio-s
on the fp32 arms and 0.078-0.088 on the batched half2 arms, so the encoder lands at
~0.10-0.11 s/audio-s -- 1.6-1.8x over the 0.06 budget. The layer loop is next:
`in_proj -> B*X -> gated depthwise conv3 -> C*y -> out_proj`, the attention layers
(prefill attention + head-norm + RoPE), and the FFN, gated per layer against
`lfm3_enc.c`'s existing `layer_XX` dumps.

### 27. S3 layer path: the conv block + FFN ported, gated on layers 0-1; the gate caught a dropped residual branch (run #312)

The encoder's layer loop now runs on the GPU for the conv-block layers
(`types[L]==0` in the C reference: in_proj -> B*X -> causal depthwise conv width 3 ->
C*y -> out_proj -> residual -> ffn_norm -> w1/w3 -> silu*mul -> w2 -> residual), with
`k_rmsnorm_f32` keeping the C reference's **fp64 sum-of-squares**, `k_bcx_split`,
`k_conv3_causal` (its `t-2+k < 0` taps skipped, as the reference skips them) and
`k_silu_mul`. Layers 0 and 1 gate against `lfm3_enc.c`'s existing `layer_XX` dumps:

```
ds_out     rms rel 3.449e-05 vs C   max abs 2.503e-06   rms 0.12701 both   OK
layer_00   rms rel 2.055e-04 vs C   max abs 2.384e-06   rms 0.12758 both   OK
layer_01   rms rel 1.887e-04 vs C   max abs 2.861e-06   rms 0.13835 both   OK
ENC_GATE PASS (arms=fp32 T=50 stages=3)
```

Three things this increment established:

1. **A real bug, caught by the gate on its first run.** The FFN's w2 output was written
   into the same buffer that already held `x + conv`, so the conv branch silently
   dropped out of the residual (`rms rel 24 vs C`, aggregate rms 0.1312 vs 0.1276). The
   C reference accumulates `out = conv_out; out += x; out += ffn`, so the FFN needs its
   own buffer. It is why the gate checks three independent things per stage: rms rel,
   max abs, and the aggregate rms agreeing to 4 decimals -- all three flagged it.
2. **Per-stage gates, with the reason written down.** `ds_out` is gated at 1e-4 rms
   (only `lin_out` is Q8 there, so both sides run the same fp32 arithmetic). The
   `layer_*` stages have five Q8 linears, and the C reference *dequantizes to float and
   dots in ascending k* while the engine's Q8 arm dots the int8 codes per 32-code block
   and scales after: two legitimate fp32 algorithms that differ by ~1e-3 on cancelling
   768-term dots (run #309 measured exactly this). So `layer_*` is gated at rms rel
   2e-2 AND max abs 1e-3 AND aggregate-rms agreement. Measured: 1.9-2.1e-4, 2.4-2.9e-6,
   agreement in the 4th decimal.
3. **Determinism verified** (3 repeats, max abs difference 0.000e+00), so a gate reading
   is a property of the code, not of the run.

Cost, min over 6 passes of the whole pipeline, Tm=400 -> Tt=50:
**176.5 ms = 0.0441 s GPU per audio second** for the downsampler + 2 conv layers, i.e.
~55 ms per conv layer. Extrapolated over 17 layers that is ~0.23-0.28 s/audio-s on the
fp32 arms -- 4x over the 0.06 budget, and the batched half2 arm (16-18 GMAC/s on these
shapes, run #309) is the path that brings it to ~0.10-0.12. Per-stage attribution inside
the layer loop is not trustworthy yet (the dumps' host writes sit inside the timed
windows), so only the total is quoted; the instrument needs the dumps moved out.

Next: the prefill attention kernel (the engine ships decode-only `k_attn_head`; the
encoder needs full causal attention over T=50 positions with per-head RMSNorm and
RoPE rotate-half, theta 1e6), then layers 2/5/8/11/14-16 and the projector.
Backbone gates re-run green on a fresh build (`CHECKS OK`).

### 28. S3 COMPLETE: the whole encoder tower runs on the Nano GPU, 0.032 s per audio second

All 17 layers plus the final norm and the projector are ported (`lfm3_enc_gpu.cu`), and
the gate covers 20 stage dumps plus 4 attention internals (`probe_q/k/v/ao_02`, from
`LFM_ENC_PROBE_LAYER=2` on both binaries). Fresh-build result, fp32 arms, Tm=400 -> Tt=50:

```
ds_out  energy rel 8.4e-07   layer_00 9.6e-07   layer_08 4.0e-05   layer_16 2.6e-05
enc_out 1.3e-05              proj_out 2.7e-05   probes 5e-07..1e-06
ENC_GATE PASS (arms=fp32 T=50 stages=20)          backbone gates: CHECKS OK
```

Whole pipeline, min of 8 passes: **127.9 ms per 4 s of audio = 0.0320 s GPU per audio
second** -- the downsampler's 0.0158 plus ~6.6 ms per layer, so the 17 layers cost
~0.016 s/audio-s on the fp32 arms (3.7 GMAC/s). The encoder is now 0.032 + mel 0.0055
= ~0.038 s/audio-s: inside the 0.06 budget, so ASR_RT is set by prefill and decode
(1.30 + 0.51 + 0.038 = ~1.85) and the encoder is no longer the binding constraint.

**Three real bugs, each found by a gate check and each localized with a probe, not a guess:**

1. `k_headnorm` used a `__shared__ double ps[32]` tree with blockDim 96: eight to
   twelve threads alias every slot, last writer wins, 7/8 of the fp64 partials vanish.
   Gate showed layer_02 at rms rel 40 vs C. Serial ascending sum in thread 0 fixes it
   and is bit-identical to the C loop; `k_layernorm_f32` (blockDim 256, 8-way aliasing)
   had the same defect and was fixed the same way.
2. `k_attn_prefill` had every thread write `sc[j] = expf(...)` and read it back with no
   barrier between. 96 threads is three warps, warps are not synchronized with each
   other, so an ahead warp divided by the **pre-exp score**. Scores matched fp64 to
   3e-06, which is why the score path was not the suspect; the fp64 referee of the
   C algorithm gave C max abs 6.5e-08 and the GPU 1.0e-01. Fix: thread 0 writes the
   weights, barrier before every read.
3. The residual stream rotated `xr <-> hout` with ONE scratch buffer, so odd layers
   wrote into `ds` -- every rep after the first started from layer 16's output
   (rep 2's `ds_out max` was 5683, layer 16's, not 1.78). Now a two-buffer pool with
   `ds` read-only.

**Gate design, since the per-element rms-rel measure proved misleading.** Projector and
FFN outputs come from cancelling 768/1792-term dots, so near-zero cells show a large
relative error from a tiny absolute one (`proj_out`: per-element 1.9e-02 with max abs
2.3e-04). The primary bar is now the **energy-relative** error
`sqrt(mean((a-b)^2)) / sqrt(mean(b^2)) <= 5e-3`, plus max abs, non-finite, all-zero,
aggregate-rms agreement, and per-element rms rel gated only where it is meaningful
(`ds_out`, `probe_ao_02`). Every bug above was caught by at least two of them.

### 29. OPEN DEFECT (recorded, bounded, not hidden): encoder run-to-run spread

Six runs of one binary, same input, same board: `ds_out` bit-identical across all six,
`layer_00`..`layer_04` within 6e-05 energy, then **layer_05 onwards spread up to
1.85e-03** (layer_07 1.85e-03, layer_08 1.76e-03, layer_16 9.7e-04, proj_out 1.0e-03),
while at every layer at least one run is exact to ~1.4e-06. Two runs launched
back-to-back inside one ssh session agree to 2-3e-05. So: history-dependent, two stable
states, onset at layer 5, self-consistent -- the signature of reading the previous
occupant of a device buffer, not of random garbage.

What has been excluded by measurement, not by inspection:
* `dalloc` buffers: `cudaMemset` 0 added, and `LFM_ENC_POISON=1` fills them with 0xff
  bytes (NaN) -- every stage stays finite, so no output reads an unwritten dalloc slot.
* `Gemm::part`: zeroing it made the encoder WORSE (layer_01 vs C 2.86e-06 -> 2.12e-04,
  spread 1e-4 -> 1.7e-3), so the slots the combine kernel reads are all written by that
  GEMM's launches; the zeroing was reverted, with the measurement in the comment.
* an under-filled weight arena: `LFM_ENC_POISON_ARENA=1` NaNs the whole f32+scales
  arenas, every output goes NaN (as it must -- every GEMM reads scales), so it cannot
  isolate an over-read; in-arena over-reads land in a neighbour's valid data anyway.

The gate carries the defect openly: the bar is 5e-03 (2.7x the observed worst; 1e-03
sat INSIDE the spread and made the gate flaky rather than strict), and
`checks_enc.sh` now runs the encoder twice and prints the run-to-run energy rel per
stage against a 3e-03 defect budget. Next hunt: per-tensor arena poison (NaN one
tensor's range at a time) to find which tensor's neighbourhood is being read.

### 30. The section 29 hunt: three hypotheses tested, all refuted; gate ships with two bar sets

`q8_partial` now checks `cudaGetLastError()` after **every** launch (run #307's lesson: a
refused launch is silent, and a silent refusal leaves that position's partial slots
holding whatever the previous occupant of that memory had -- precisely the
history-dependent signature we are chasing). Eight runs: **zero refusals**, so that
hypothesis is dead, and now falsifiable rather than unfalsifiable.

The same eight runs also kill section 29's "two stable states" reading. `layer_08` energy
rel vs the C reference across eight separate invocations of one binary:

```
run     21      22      23      24      25      26      27      28
rel   1.4e-06 1.1e-04 4.2e-04 1.2e-04 1.4e-06 1.9e-06 1.9e-05 2.9e-03
abs   2.0e-05 6.0e-04 2.6e-03 3.9e-04 2.0e-05 2.0e-05 8.6e-05 1.5e-02
```

The amplitude is **continuous**, not bimodal, and some runs are exact to 1.4e-06. That is
reading memory this process never wrote, with content inherited from earlier processes
(Tegra unified memory, driver-recycled pages) -- not a self-referential fixed point.

Excluded so far, each by an experiment rather than by reading code:
`dalloc` buffers (zeroed, and NaN-filled with `LFM_ENC_POISON=1`: every stage stays
finite), `Gemm::part` (zeroing it makes accuracy WORSE -- those slots are load-bearing,
so the combine's inputs are all written), refused launches (checked per launch, none),
and an under-filled weight arena (`LFM_ENC_POISON_ARENA=1` NaNs every output, so every
byte the GEMMs consume is written; in-arena over-reads land in a neighbour's valid data
and would be deterministic anyway). Still open: a read that runs past the end of the
scales/codes arenas into the engine's 48 MB runtime arena, which the encoder never
writes -- the one region that would show exactly this behaviour. Next probe: cudaMemset
that runtime arena with a NaN pattern at startup and see which stage's output goes
non-finite.

**Gate ships with two bar sets, because one bar set cannot be strict against a moving
target.** Default = the tight bars (energy 1e-3, max abs 2e-3, probes 1e-4/5e-5,
determinism budget 3e-03): green on clean runs (1 of the last 3, and 10-20x margin when
clean), RED whenever the defect touches the run -- the honest verdict, since that run is
not reproducible. `DEFECT_OK=1` = defect-tolerant bars (measured worst x2-3: energy
5e-3, max abs 2e-2, probes 1e-3/2e-2, budget 5e-03; h2 arms 1e-2/5e-2/budget 1e-02) for
regression work on the rest of the port, and it prints `** DEFECT-TOLERANT MODE **` so a
green in that mode is never read as a clean green.

### 31. The defect gets a location and a mechanism class; the warm-up-rep "fix" fails

`LFM_ENC_DUMP_REP=1` writes one dump directory per timed rep, which turns one process
into a determinism experiment on its own. Four reps of the same pipeline, same process:

```
layer_00  rep-to-rep max abs vs rep0: 0, 6.47e-05, 6.47e-05, 6.47e-05
layer_03                                       0, 6.79e-05, ...
layer_08                                       0, 8.59e-04, 1.97e-02, 8.59e-04
```

**Rep 0 is not the same computation as reps 1-3**, and reps 1-3 match the C reference
exactly (`layer_00` max abs 2.38e-06, energy rel 9.61e-07). The differing elements are
2299 of 38400 in `layer_00`, and they are *all of rows t=35,36,37, every one of the 768
channels, uniform in c mod 32 and mod 256* -- a whole-input-row perturbation, not a
channel- or block-indexed one. That localizes the first-order site: some input row of
layer 0's path (op_norm output `hn`, in_proj input, `hbx`/`hcg`, `hcy`, `hffn` -- the
per-position chain) is read at positions 35-37 with content this pass did not write.
`ds_out` and the downsampler are bit-identical everywhere, so it is not in the
downsampler.

So the mechanism class is settled: **a read of device memory this pass never wrote**, in
the layer path, amplified by a 17-layer stack (1 ulp at layer 0's input becomes 1e-3
energy rel at layer 8). Every candidate buffer I could name has been excluded by
experiment: `dalloc` (zeroed, and NaN-filled -> all stages finite), `Gemm::part` (zeroing
it makes accuracy WORSE, and `q8_partial`'s guards are complete -- every q/s read sits
inside `if (row < N)`, `sh[]` is written only under `lane == 0 && row < N`, and the
write-back is guarded by `tid < R && base + tid < N`, so partial blocks read nothing),
refused launches (checked per launch, zero in eight runs), and under-filled arenas (the
loader CRC-verifies every tensor's file bytes and aborts on any cudaMemcpy failure, so
what the GEMMs consume is what the file said).

**The warm-up pass is not the fix.** `LFM_ENC_REPS=2` across six processes still spreads:

```
ds_out    cross-run max abs 0.00e+00
layer_00  cross-run max abs 0.00e+00
layer_03  cross-run max abs 1.84e-03  energy 5.16e-04
layer_16  cross-run max abs 1.12e-01  energy 2.62e-03
proj_out  cross-run max abs 3.54e-03  energy 2.97e-03
```

Reps 1-3 agree inside a process because they read what rep 0 wrote; across processes the
steady state moves too, so the stale content comes from outside the encoder's own buffer
cycle. Layers 0-2 are reproducible (0.00e+00) and the amplification starts at layer 3.

Open next probes, in the order that cheapest rules most out: (1) NaN-fill the device
pages the *loader* leaves behind (the loader's host-side `buf` is reused across 176
tensors, so the pages past the last copy are the frontier the encoder's `dalloc` calls
get); (2) allocate the layer-path buffers once, outside the reps loop, and zero them
there -- if the spread collapses, the site is one of them and the per-rep `cudaMalloc`
frontier is the amplifier; (3) single-position bisect: run with T=36 vs T=38 and see
whether the deviating rows move with T, which distinguishes a fixed-index bug from a
scheduling-dependent read.

Until then the gate's strict bars stay as they are, and a run the defect touched is
reported RED on purpose: a result that is not reproducible is not a measurement.

### 32. WAV in, audio embeddings out: the front end is closed, and every link is gated

`pulsar_arm/asm/lfm3_wav.c` is the runtime's only container parser, and it is the last
piece between "a recording" and "the mel file the encoder already takes":

```
lfm3_wav in.wav out.f32 | lfm3_mel mel.bin out.f32 mel.f32 | lfm3_enc_gpu blob mel.f32 400 dir
```

Accepted forms: 16 kHz, mono or stereo (`(l+r)*0.5f`), PCM 16-bit, IEEE float32, and
WAVE_FORMAT_EXTENSIBLE for both, chunks in any order with junk chunks and odd sizes
honoured. Everything else is REFUSED with a reason: other rates, 24-bit, 3 channels,
truncated data, non-RIFF, an extensible header whose SubFormat is PCM24. No resampling,
no channel guessing, no "assume 16 kHz" -- a resampler inside a gated path is DSP nobody
has validated, and the failure mode of guessing here is a transcript that is subtly
wrong while every front-end gate stays green.

`.auto/lfm3/checks_wav.py` gates it two ways, and the second is the reason the first
matters:

```
  pcm16_mono/stereo, f32_mono/stereo, ext_pcm16, ext_f32, chunks_junk (lying RIFF size,
  LIST+fact before fmt, odd-size pad byte), pcm16_extremes (-32768/32767), empty
      -> BIT-EXACT vs numpy, 0 cells differ, 9 fixtures
  rate_44k, rate_8k, pcm24, three_ch, truncated, ext_pcm24, not_riff
      -> REFUSED with rc != 0, 7 fixtures
  WAV -> lfm3_wav -> lfm3_mel -> mel.f32  vs the CACHED HF mel, 6 fixtures:
      chirp T=99 live max 6.26e-06 | clicks 5.96e-08 | dc 4.77e-07 | len1 0.0
      len159 1.19e-07 | len15999 5.96e-07                     (S2 bar is 1e-05)
WAV_GATE PASS (16 fixtures, 6 chain)
```

The chain fixtures reuse the mel gate's own PCM signals, so no new audio enters the
suite, and they exercise the reader *through* the validated mel stage: a WAV bug shows
up as mel drift against HF, not as a green parser.

`lfm3_mel.c` writes channel-major `[128][frames]`, which is exactly the `[128][Tm]`
layout the encoder reads, so the three stages concatenate with no transpose. End to end
on a synthesized 4 s chirp (400 frames -> Tt=50 tokens, the 12.5 tokens/s law):

```
lfm3_wav: 64240 samples | MEL frames=400 mels=128 | layers ported: 17 of 17
WAV -> proj_out: 102400 cells, finite True, rms 0.0567, max 1.1620 (ds_out rms 0.0229)
ENCODER total 232 ms = 0.058 s/audio-s cold single pass (warm min-of-8: 0.0320)
```

What is left for the realtime gate is now narrow and specific: feed `proj_out`'s rows
into the backbone prefill as the audio-embedding prompt (an embedding-stream override at
the audio positions), and run `.auto/lfm3/measure_rt.sh` on >= 60 s of real speech,
which is the one input I do not have.

### 33. First real speech on the board: a TTS fixture, and the encoder at real length

`.auto/lfm3/mk_speech.py` synthesizes the gate's speech clip with **gTTS 2.5.4**
(`lang=en, tld=com, slow=False`) from a transcript authored in that file, then ffmpeg's
it to the exact contract `lfm3_wav` accepts (16 kHz, mono, `pcm_s16le`, no dithering
mode beyond rectangular for reproducibility). The script **fails** if the decoded clip
is under 60 s, so the realtime gate's precondition is enforced by the tool and not by
eyeball. Computed at synthesis time: 129.720 s, 2,075,520 samples, rms 0.1098, sha256
recorded in `speech/manifest.json` together with the verbatim transcript.

**Provenance, stated with the fixture:** this is a synthetic voice, not a recording. It
measures timing and greedy token-stream agreement on speech-shaped audio -- it does NOT
certify speaker, accent, noise, channel, or disfluency robustness, and no doc claim
should read that way. The transcript is authored (not corpus-sourced) precisely because
the gate needs `transcript == exactly what was spoken`, which a corpus clip couples to a
license and an alignment we do not have. Its text is deliberately numeric: decimals,
percentages, ordinals, dates, thousands separators, units, so a transcript gate has
something to get wrong.

The chain on the Nano, on real speech:

| stage | result |
|---|---|
| `lfm3_wav` | 2,075,520 samples, **BIT-EXACT vs numpy** (0 diffs) -- the 9-fixture gate is on synthesized signals; this is the real-speech confirmation |
| `lfm3_mel` | **12,971 frames** for 129.720 s = 100.0 frames/s exactly, and 12971/8 = 1622 tokens = 12.5 tokens/audio-s, confirming run #303's rate on speech |
| `lfm3_enc_gpu` (80 s sub-clip, 7999 frames -> Tt=1000) | all stages finite; rms ds_out 0.0257 -> layer_16 1.718 -> enc_out 0.849 -> **proj_out rms 0.0306, max 0.9157** |

The encoder number **improves at realistic length**: 80 s costs 1430.64 ms (min of 3)
= **0.0179 s GPU per audio second**, versus 0.0320 measured on the 4 s fixture. The
difference is per-launch fixed cost: at Tt=50 the 33 launches per pass are amortized
over 8x less work. So the encoder is **1.8% of realtime**, 3.3x inside the 0.06 budget,
and ASR_RT decomposes (per audio second) as prefill 1.302 + decode 0.514 + mel 0.055 +
encoder 0.018 = **1.889**, consistent with the 1.872 measured before the encoder was
counted.

Two facts this run surfaced, both recorded rather than patched around:

* `lfm3_enc_gpu` caps `Tm` at **8192 mel frames = 81.9 s of audio** (`lfm3_enc_gpu.cu:582`),
  so the 129.7 s fixture needs the cap raised (arena sizing, not an algorithmic limit)
  before the full-clip gate can run. The 80 s sub-clip is a legitimate sub-fixture.
* `lfm3_mel` emitted **7999** frames for an 80.000 s clip, not 8000 -- the HF
  `pad_length`/`num_frames` rule, which `checks_mel.py` already pins as exact. Feeding
  the encoder a Tm that doesn't match the file is caught ("input short"), and it caught
  this on the first try.

### 34. The run-to-run defect, root-caused and fixed: a missing barrier in `k_rmsnorm_f32`

Six runs of hunting (§29, §30, §31) ended at line 373 of `lfm3_enc_gpu.cu`:

```cuda
__global__ void k_rmsnorm_f32(float* x, const float* w, int D) {
  float* r = x + blockIdx.x * D;
  double v = 0;
  for (int i = 0; i < D; i++) v += (double)r[i] * r[i];   // every thread reads the WHOLE row
  /* <-- NO BARRIER */
  const float inv = 1.0f / sqrtf((float)(v / D) + EEPS);
  for (int i = threadIdx.x; i < D; i += blockDim.x) r[i] = w[i]*r[i]*inv;   // writes IN PLACE
}
```

Every thread builds its own fp64 sum of squares over the row, then all threads write the
row back **in place**, with nothing between the two phases. An ahead warp scales `r[i]`
while a lagging warp is still inside its sum loop; the lagging warp's sum is then built
from a mix of pre-scale and post-scale values, its `inv` is wrong, and the row ends up
with elements scaled by several different inversions. Launch is `<<<Tt, 256>>>` = 8
warps, D = 768, so the sum loop is long enough to collide. It is used at three sites:
per-layer `ffn_norm`, `op_norm`, and the tower-final `embedding_norm` -- all in place on
`hn`. Layer 0 is a **conv** block, which is why run #313's attention-race fix never
touched it; `k_headnorm` (96 threads) has its barrier and `k_layernorm_f32` writes to a
separate output, so neither has the bug.

The control that proved it, same binary, same input, three runs, Tt=1000:

```
ds_out    identical        (the downsampler has no in-place norm)
layer_00  max 2.202e-01    814 of 1000 rows, rows 40,44,68,72-78,96,... ALL 768 channels
layer_16  max 2.943e+01    rows 0-39 bit-identical, then sparse
```

Sparse rows = collision-dependent, which is exactly why the amplitude looked
*continuous* (1.4e-06 .. 2.9e-03) rather than bimodal, why it was rarer at Tt=50 (50
blocks collide less often than 1000), and why "reps 1-3 agree with each other and with
C" while rep 0 did not. Zeroing/poisoning, launch refusals, `Gemm::part` and the loader
were all cleared earlier because the mechanism was never stale memory.

**After the barrier**: 3 runs of the same binary at Tt=1000, 20 stage dumps, **all bit-
identical**; `ENC_GATE PASS` with run-to-run worst energy rel **0.000e+00**, and accuracy
*improved* -- layer_08 energy rel 1.387e-06 where §28 recorded 4.0e-05 (29x), layer_16
1.381e-06 vs 2.6e-05. The gate was absorbing the race into its bars.

Two enabling changes, both gated by the same runs:

* **`Tm` cap 8192 -> 32768** (`= 327.7 s` of audio). The old cap was a bring-up guard,
  not a structural limit: every per-position buffer is a dynamic `cudaMalloc`, and the
  loader's free/need gate refuses before any kernel runs if memory is short.
* **`k_attn_prefill` no longer stages scores**: it recomputes them from q/k on each of
  the three passes (max, denominator, weighted sum) and keeps one shared slot, so there
  is no 1024-token score buffer. This is what let the full clip run at Tt=1622; the
  full-clip control is 3 runs, 20 dumps, all bit-identical. Cost is 3 QK passes instead
  of 1; the encoder measured 0.0132-0.0147 s GPU per audio second on the 129.7 s clip.

### 35. External check on the audio embeddings, and a gap I am not going to average away

The HF golden for the fixture records `audio_embeddings.hf_shape = [1622, 2048]` and
`mean_row_l2_norm = 1.60280`. The engine's `proj_out` for the same clip:

```
ENGINE  mean row L2 1.2791   (min 0.911, max 9.398)   -> engine/HF = 0.7981
```

A 20% norm deficit is far too large to read as Q8 noise, and it sits next to two
observations that do not line up with it: the 4 s fixture's aggregate rms matched HF to
0.3% (0.05424 vs 0.05406), and the gate against the C reference is at energy rel 1e-06
across all 20 stages -- so this is not the GPU drifting from its own reference, it is the
Q8 path (or the tail/padding rows of a long clip) differing from HF's fp-bf16 reference.
The HF comparison stays "reported, not gated" for exactly this reason (§305 disclosed the
Q8 gap the same way), and the row-norm mean is dominated by a few large rows (max 9.4 vs
mean 1.28), so a distribution comparison is required before this number means anything.
The data needed is HF's full `[1622, 2048]` embedding matrix, which the golden did not
save; it is 13 MB and is the next thing to collect.

### 36. The embedding gap is HF's bf16 tower, not our encoder: three-way test

Run #318 disclosed a 20% row-norm deficit and mean per-row cosine 0.851 between the
engine's `proj_out` and HF's audio embeddings, degrading with position. The
three-way test settles it. Implementations: **numpy-f32** (§22's spec re-derivation,
running HF's own weights in fp32, no quantization anywhere), **C reference** (the Q8
blob, x86), and HF (its audio tower as shipped). On the 30 s sub-clip, per-row cosine:

| pair | rows 0-3 | 3-21 | 21-100 | 100-200 | 200-300 | 300-375 |
|---|---|---|---|---|---|---|
| numpy-f32 vs HF | 1.0000 | 1.0000 | 0.9755 | 0.9398 | 0.9099 | **0.8775** |
| C reference (Q8) vs HF | 1.0000 | 0.9998 | 0.9753 | 0.9396 | 0.9098 | **0.8773** |
| C reference (Q8) vs numpy-f32 | 1.0000 | 0.9998 | **0.9998** | **0.9997** | **0.9997** | **0.9997** |

Mean row L2: numpy-f32 **1.4725**, C reference **1.4726**, HF **1.6508**.

Read the last row first: our Q8 paths agree with the **unquantized fp32 spec** to 0.9997
at every position, so neither the Q8 weights nor the GPU code is losing anything. And the
drift away from HF is the *same number from both sides* -- 0.8775 for the fp32 spec,
0.8773 for Q8, agreeing to four digits -- which is what an HF-side effect looks like: our
quantization cannot produce a deviation that is independent of quantization. HF's tower
is bf16, and bf16 rounding accumulates exactly the way this curve does (perfect for the
first ~20 rows, then monotonically worse), so **HF is the outlier** and its mean row norm
is the one that is off, by +12%.

Consequences, stated as decisions:

* The encoder's acceptance gate stays what §30 set it to be -- the C reference at energy
  rel ~1e-6 plus **end-to-end** greedy speech->text against the HF golden. An
  embedding-level bar against HF's bf16 tower would be gating on HF's rounding, and at
  long positions it is unpassable by construction: a *correct* fp32 implementation fails
  it, as this table shows.
* "Reported, not gated" for the HF embedding comparison (§30) was the right call, and is
  now justified by a controlled experiment rather than by the part-Q8 caveat alone.
* The remaining external arbiter for the audio path is the transcript: `WAV -> text`
  against `speech/hf_asr_golden.json`, which the `--audio-emb` prefill override now makes
  runnable.

### 37. First end-to-end speech->text, and the encoder's wall is launch-bound

`.auto/lfm3/checks_asr.py` runs the whole chain from a fresh build on the Nano and grades
the greedy token stream against the HF golden (never the authored transcript -- HF's own
WER on this fixture is 36%, so grading against it would measure HF):

```
wav 3.1s | mel 7.8s -> 12971 frames | encoder -> proj_out rows 1622 (HF's law: 1622)
AUDIO-EMB armed: 1622 rows x 2048, overriding prompt positions 15..1636 (13.3 MB)
AUDIO-EMB fed 1622 of 1622 rows
greedy prefix vs HF golden: 126 / 262 tokens
  first divergence at index 126: engine id 20, HF id 22   (inside a numeral: "...at 3.4 degrees")
  engine text : 'The weather service issued its seasonal outlook on the 9th of March, and
                 the numbers were unusual. In the nort'
  HF golden   : 'The weather service issued its seasonal outlook on the 9th of March, and
                 the numbers were unusual. In the nort'
ASR_RT = 3.866   (prefill+decode 3.160, front end 0.706)
```

126 identical tokens is a coherent ~90-word transcription, and the divergence is at a
numeral, which is the hardest class of token in the fixture by design. The gate's bar is
deliberately a floor (prefix >= 20, plus rows/Tm/proof-line checks): a bar invented before
the first measurement is a guess with a gate attached. The prompt ids come from
`.auto/lfm3/asr_prompt_ids.py`, which rebuilds HF's prompt from the golden's head+tail
splice, checks the sha256, and *expands the single `<|audio|>` placeholder into 1622
copies* of `audio_token_id=125031` -- the count, contiguity and position checks are all
enforced, because an id list one token off shifts every audio embedding.

`ASR_RT 3.866` is 2x my §33 decomposition (1.889), and the difference is **wall-clock vs
kernel time in the encoder**. With `LFM_ENC_WALL=1`:

```
WALL blob read+CRC+upload      5.75 s
WALL mel read+validated        0.05 s
WALL 17 layers                61.52 s        <--  ENCODER total GPU = 1.68 s
WALL layers+projector+dumps     1.11 s
```

Two hypotheses tested and rejected, then the answer:
* **gate dumps** (103 MB of device-to-host + eMMC writes): 80.4 -> 76.5 s, i.e. 3.9 s.
* **allocator churn** (per-GEMM `cudaMalloc`/`cudaMemset`/`cudaFree` of an 11 MB partial
  buffer, ~100 per pass): pooling it into a grow-only pool changed the layer loop by 1 s.
* **per-kernel `cudaDeviceSynchronize()`** in `chk()` (~200 sites): gating it behind
  `LFM_ENC_SYNC=1` took 67.5 -> 61.5 s. Real, small.
* **the remaining 61.5 s is launch count**: `gemm_run`'s fp32 path is
  `for (t = 0; t < T; t++) q8_partial<float><<<...>>>` -- **one launch per position per
  GEMM**. At Tt=1622 and ~100 Q8 GEMMs per pass that is ~162,000 launches, each with its
  immediate `cudaGetLastError()` check: 0.38 ms each, host-side, and invisible to every
  GPU timer in the project. The 4 s fixture (Tt=50) hides it: 5,000 launches.

The fix is already validated: run #307 measured the batched fp32 arm
`q8_prefill_h2<float, CT=4, ARM5, GRP=2>` as **bit-identical** to the per-position arm at
4.6 vs 10.4 GMAC/s. It cuts launches 4x (162k -> 40k): GPU 1.68 -> ~3.7 s, host 61 -> ~15 s,
encoder wall ~22 s = 0.17 s per audio second, and ASR_RT 3.87 -> ~3.3. The half2 arm is
not the answer (it changes numerics and §29 showed it is not faster).

Kept from this round as instrumentation, all env-gated and inert by default:
`LFM_ENC_NODUMP=1` (shipping path, skips the gate's dumps), `LFM_ENC_WALL=1` (phase
breakdown), `LFM_ENC_SYNC=1` (per-kernel syncs, what the accuracy gate uses), the pooled
grow-only partial buffer, and `chk_final()` -- one process-wide sync so an async fault is
still always caught when per-kernel syncs are off.

### 38. The batched fp32 arm is faster and WRONG; it stays off, and the 40 s sync is open

`LFM_ENC_BAT32=1` (batched `q8_prefill_h2<float,CT4,ARM5,RB2,GRP2>` replacing the
per-position `q8_partial` loop) on the 129.7 s clip:

```
BAT32=0 SYNC=0  wall 69.0 s      BAT32=1 SYNC=0  wall 51.5 s
BAT32=0 SYNC=1  wall 73.2 s      BAT32=1 SYNC=1  wall 51.9 s
phase split with BAT32=1:  blob 5.7 | mel 0.05 | 17 layers 4.95 | proj 0.00 | final sync 40.2
```

The launch-count diagnosis is confirmed by the phase split: the layer loop went
**61.5 s -> 4.95 s** (12x) when the per-position launches went away.

But the arm **fails the accuracy gate**, and the gate is the acceptance bar, not timing:

```
layer_01   energy rel 6.310e-01  max abs 1.365e+00  rms(GPU) 0.12701 vs rms(C) 0.13835  FAIL
probe_q_02 energy rel 1.000e+00  rms 0.0000 vs 0.9994  FAIL          (q_proj output all zeros)
ENC_GATE FAIL (arms=fp32 T=50 stages=3)     <- bailed at layer_01 on the all-zero check
```

So run #307/309's "bit-identical batched fp32" result **does not transfer** to this call
configuration: the arm's launch contract (R/RB/GRP/grid/shared, and the `KC` it was
sized for) is coupled to the bench harness that validated it, and my 25-line port got it
wrong -- the signature is a GEMM whose whole output is zeros, which the plausibility
check catches immediately. `LFM_ENC_BAT32` therefore **defaults to 0**; the encoder ships
on the per-position arm at 69 s wall. Re-verified after the change: `ENC_GATE PASS`
(20 stages, run-to-run 0.000e+00) and backbone `CHECKS OK`.

Also corrected while doing this: `gemm_setup` sized `KC` from an `h2` argument that
`gemm_run` ignores (the arm is chosen from `LFM_ENC_H2`), so the downsampler's `K=4096`
`lin_out` kept `KC=4096` (nb=128) and its batched launch was refused for 67 KB of shared,
silently falling back to 1622 per-position launches. The bound now follows the arm that
will actually run.

Open, and honestly labelled: with per-kernel syncs off, a single
`cudaDeviceSynchronize()` at the end of the pass costs ~40 s in all four matrix cells,
against 1.57 s of event-timed GPU work. It is not GPU work, not the dumps (3.9 s), not
the allocator (1 s) and not the K=4096 fallback (0.6 s of launches). It is the dominant
term in the encoder's wall and it is not explained. Next probe: time the sync itself at
incremental queue depths, to separate "draining a deep launch queue" from "waking an idle
Jetson GPU".

### 39. ROOT CAUSE of the encoder/HF divergence: the encoder's attention is windowed at 32

The open item from §36 (my three implementations agreeing with each other at cos 0.9997
and drifting from HF in a position-growing way) is localized and largely fixed, with the
checkpoint's own config as the evidence.

`.auto/lfm3/enc_ref/probe30.py` compares, per stage and per position bucket, the numpy
fp32 spec against HF's **canonical** encoder forward (provenance pinned: the dump equals
`audio_encoder(feats, lengths)` and `get_audio_features(...)` both to `max_abs_diff 0.0`,
`hf_meta.json`; HF weights fp32, sdpa).

```
ds_out, layer_00, layer_01 (mel -> downsample -> 2 conv layers -> 2 FFN blocks)
      cos 1.000000 in EVERY position bucket 0-2 ... 300-374
layer_02  = the FIRST ATTENTION LAYER:  input cos 1.000000, increment cos 0.971037,
                                           output cos 0.996537 (200-299), 0.996428 (300-374)
```

So the divergence is *introduced by the attention block*, not inherited, not the mel, not
the downsampler, not conv, not FFN, not RoPE-offset (killed in §36), not HF precision
(killed: HF fp32 vs bf16 cos 0.99996).

The config says why, and I had read the wrong config level:

```
encoder.sliding_window = 32
encoder.lfm.layer_types = [conv, conv, full_attention, conv, conv, full_attention,
                           conv, conv, full_attention, conv, conv, full_attention,
                           conv, conv, full_attention, full_attention, full_attention]
encoder.lfm.rope_parameters.rope_theta = 1000000.0
```

§25/§31's "layer_types are all full attention, no sliding" reading came from the top-level
LM config (`lfm.layer_types` does contain `sliding_attention` with window 256, and
`encoder.lfm.layer_types` does not) — and it made the spec full-causal everywhere. HF's
encoder forward builds its mask with `create_sliding_window_causal_mask(...,
sliding_window=enc.config.sliding_window)` = **32** (`probe30.py` lines 215-224 reproduce
the call with checkpoint code, and `hf_meta.json` records
`hf_encoder_sliding_window: 32`).

Mechanism test, with a per-layer window schedule added to the probe
(`--window-test 32 --window-layers 2,5,8,11,14,15`):

| spec | first stage with cos(rows 200+) < 0.999 |
|---|---|
| full causal (as written) | **layer_02** 0.996537 / 0.996428, increment 0.971 |
| W=32 on attention layers 2,5,8,11,14,15 | **layer_16** only — layers 02-15 all **1.000000** |

W=192/256/512 were tested first and are null (0.9965, identical to full causal), which is
why the window hypothesis looked dead; only the checkpoint's own value 32 works.

**Remaining, open:** layer_16 (the last encoder layer) does not match under either
treatment — full causal 0.7238, windowed at 32 0.6836, increment cos 0.214/0.161, with
its input at 1.000000. So HF treats the last attention layer differently again, and the
next probe is per-layer internals of layer 16 (q/k pre/post rotary via forward hooks,
`.auto/lfm3/enc_ref/probe_rope.py`, which also records that the encoder's decoder layers
live at `audio_encoder.lfm.layers`, class `Lfm2DecoderLayer`, rotary at `lfm.rotary_emb`,
and no `q_norm` module on the layer).

This is the mechanism behind the 126/262 transcript prefix in §37: the encoder's
activations are exact through layer 15 at every position and start drifting at the first
attention layer, growing to cos 0.877 by row 300+ at full length.

### 40. The window fix shipped, and the prefix metric is retired as the quality bar

Ported `encoder.sliding_window = 32` to all three implementations, with the same rule
(the attention layers HF windows = `layer_types` attention minus the last layer, so
2,5,8,11,14,15; layer 16 stays full causal):

* `enc_numpy.py` (spec of record) — `WIN`/`WINDOWED`, env `LFM_ENC_WIN=0` reproduces the
  old spec. Its own comment claimed "the 32-token window exists only in the streaming
  path"; that claim is what let the bug survive, and §39's measurement kills it.
* `lfm3_enc.c` — `jlo = i - win + 1` on the three softmax loops.
* `lfm3_enc_gpu.cu` — `k_attn_prefill(..., int win)`, `g_enc_win` from `LFM_ENC_WIN`,
  and a start-up proof line (`ENC-ATTN layers=2,5,8,11,14,15 window=32; layer 16 full
  causal`) so no timing or gate number can be read without knowing which mask ran.

`ENC_GATE PASS` (20 stages, run-to-run worst energy rel 0.000e+00) — the C reference and
the CUDA encoder moved together, so this gate cannot detect a *shared* spec error, which
is exactly how §39's bug stayed invisible: the §22 dumps come from
`dump_encoder.py`, which passes `attention_mask={"full_attention": None, "conv": None}` —
a mask **dict**, HF's unwindowed path — while the golden's real forward goes through
`create_sliding_window_causal_mask(sliding_window=32)`. Two HF paths, two masks, one gate.

Quality, measured on what the LM consumes (`proj_out` vs the embedding matrix HF fed the
LM at the same positions, 1622 rows x 2048, the 129.7 s clip):

| arm | mean per-row cos | energy rel | per-200-row cos |
|---|---|---|---|
| full causal (pre-fix spec) | **0.303** | 1.42 | 0.278-0.327, drifting |
| W=32 on 2,5,8,11,14,15 | **0.892** | 0.454 | **0.888-0.899, flat** |

and the transcript, with metrics that respond monotonically (added to `checks_asr.py`,
gated at `ASR_MIN_EMB_COS=0.85`):

```
greedy prefix vs HF golden: 69 / 262 tokens          <- CHAOTIC, informational only
stable metrics: all-position match 143/262 (54.6%), LCS 233, token-WER 0.111 (sub 29 ins 0 del 0)
EMB vs HF golden: mean per-row cos 0.891976 (min 0.7076), energy rel 4.540e-01
ASR_RT = 3.848  (prefill+decode 3.390, front end 0.458)
```

**Why the prefix went 126 -> 69 while the encoder got ~3x better.** Greedy decoding is
chaotic in logit margins: one flipped token ends the prefix, so prefix length measures
"where the first coin-flip landed", not quality. The divergence is at a numeral (index 69,
engine id 22 vs HF id 20) in a numeric-dense stretch ("...at 3.4 degrees"), the classic
near-tie region. LCS 233/262 (88.9% of content tokens agree) and token-WER 0.111 with
**zero insertions and zero deletions** are the numbers that describe the transcript, and
the LM-input cos is the upstream metric that predicts them. The gate now reports all of
them, gates the emb-cos, and keeps the prefix printed for continuity.

Encoder wall also dropped to 47 s from 80.7 s: the gate now runs with per-kernel syncs
off by default (`LFM_ENC_SYNC`, §38) and `chk_final()` catches async faults.

Open: layer_16's uniform cos 0.67-0.73 vs HF (every row, no position trend, unchanged by
window/sink/window-form — so not a mask effect), removed by the final norm (`enc_out`
1.000 at 351/375 rows). The 0.892 emb-cos plateau and the remaining ~0.11 WER are the
size of what that leaves on the table.

### 41. A/B of the two arms through the same gate, and a disclosure about the 126/262

Run through `checks_asr.py` with `ASR_ENC_ENV` selecting the arm (same build, same prompt,
same golden, same metric code):

| arm | prefix | all-position match | LCS | token-WER | EMB cos | gate |
|---|---|---|---|---|---|---|
| `LFM_ENC_WIN=32` (section 39/40) | 69 | 143/262 (54.6%) | 233 | 0.111 | 0.891976 | PASS |
| `LFM_ENC_WIN=0` (old full causal) | 0 | 1/262 (0.4%) | 1 | 0.996 | 0.303083 | **FAIL** (cos < 0.85) |

The pre-fix arm is not "slightly worse": with 1 token of 262 in common and WER 0.996 the
transcript shares nothing with HF's. `ASR_MIN_EMB_COS=0.85` accepts the good arm at 0.892
and rejects the bad one at 0.303, so the bar is calibrated by two points on either side of
the defect it is meant to catch, which is the standard the rest of this project uses.

**Disclosure, because §37's headline is no longer reproducible.** §37 reported prefix
126/262 for the pre-fix encoder. Running the pre-fix arm today gives prefix 0, LCS 1. Two
reasons, and both are stated rather than smoothed:
1. Prefix length on a greedy stream is chaotic, so 126 vs 0 is not a measurement of a
   126-fold difference in anything; the monotonic metrics (LCS, WER, emb-cos) are the ones
   that are stable, and the emb-cos for §37's build was never taken.
2. The tree changed since §37: the GEMM partial buffer is now a pooled grow-only allocation
   (§38) and the gate runs the encoder async with `chk_final()` instead of per-kernel
   syncs. The C-vs-CUDA gate is measured at T=50, so neither change is verified at
   Tt=1622. The 126 figure should be read as a historical observation from a superseded
   build, not as a current result. The current, reproducible pair is the table above.

### 42. The rule is ALL attention layers: transcript 260/262 tokens, EMB cos 0.999774

Section 39/40's "attention layers minus the last" exception was wrong, and the way it
died is the lesson. Layer 16 was pinned to a single block (`probe_l16b.py`: operator_norm
exact with `implied w vs stored w: corr 1.000000, max|diff| 0.00000`; `feed_forward` cos
1.0000; `self_attn` cos 0.8483 on an exact input), then `probe_l16attn.py` swept 1792
plausible conventions -- window {0,16,32,48,64,128,256} x {exclusive, inclusive} x
{sink, no sink} x scale {1/sqrt(96), 1/96, 1, 1/sqrt(768)} x {qk-norm, none} x {rope, no
rope} x {GQA hq//2, hq%4} x {max-subtract, raw} -- on HF's captured input, scoring
against HF's captured output. The winner, out of 1792:

```
cos 1.000000 min 1.0000 rms 0.51026 | W=32 exc sink=False rsqrt qk rope=True div maxsub   <- HF rms 0.51026
```

W=32, the identical convention as layers 2-15. With all seven attention layers windowed,
the numpy spec reproduces HF's `enc_out` AND `proj_out` at **cos 1.0000 at every one of
the 375 rows**. The layer_16-vs-`hidden_17` 0.68 that motivated the exception is a
capture artifact of `dump_encoder.py`'s last hidden file: that layer's attention is
cos 1.000000 and its FFN cos 1.0000 individually, so the pair cannot sum to 0.68.

Shipped in all three implementations (`enc_numpy.WINDOWED`, `lfm3_enc.c`
`win = (types[L]==1) ? ENC_WIN : 0`, `k_attn_prefill(..., g_enc_win)`), proof line
`ENC-ATTN layers=2,5,8,11,14,15,16 window=32 (all attention layers; section 42)`, and
`ENC_GATE PASS` (20 stages, run-to-run 0.000e+00).

End-to-end speech -> text on the 129.7 s fixture, same gate, same golden:

| metric | pre-fix (W=0) | §40 (last full) | §42 (all windowed) |
|---|---|---|---|
| greedy prefix | 0 | 69 | **207** |
| all-position match | 1/262 (0.4%) | 143/262 (54.6%) | **260/262 (99.2%)** |
| LCS | 1 | 233 | **260** |
| token-WER | 0.996 | 0.111 | **0.008** (2 subs, 0 ins, 0 del) |
| EMB mean per-row cos | 0.303 | 0.892 | **0.999774** (min 0.9996, flat in all nine 200-row buckets) |
| EMB energy rel | 1.42 | 0.454 | **0.0206** |

The remaining difference is two substituted tokens starting at gen index 207
(engine id 368 vs HF 5385, then the streams re-converge: LCS 260 of 262), which is a
greedy near-tie in the LM's decode, not an encoder gap: the LM's input is cos 0.9998.
`ASR_RT = 3.866` (prefill+decode 3.399, front end 0.466) is unchanged, as expected --
this was an accuracy fix; the wall is the LM's prefill and is the `w2` decision.

### 43. ASR_RT decomposed, and the realtime target restated against the roofline

`ASR_RT = 3.866` as a single number hid where the time is. Measured on the 129.7 s clip:

| phase | wall | /audio-s | evidence |
|---|---|---|---|
| decode (262 tokens) | 207 s | **1.60** | engine prints `steady-state 108.68 ms/token ... over 1904 steps`; 1904 x 108.68 ms = 207 s |
| engine one-time startup | 83 s | 0.64 | a 20-id prompt with `--steps 24` took 85.3 s wall against 2.6 s of steps; blob is 1.946 GB |
| prefill (1642 ids incl. 1622 audio) | 151 s | 1.16 | remainder of the 441 s engine wall |
| front end: wav 3.9 + mel 8.4 + encoder 48.2 | 60 s | 0.47 | gate timers; encoder is 0.0125 s GPU per audio-second |

The decode term, not prefill, is the largest single contributor -- a consequence of the
workload definition (4 text tok/s charged against the clip), and the gate now prints the
split plus a `warm ASR_RT` that excludes the one-time startup.

What the hardware allows, from the same measured constants (Q8 weights 1.06 GB streamed
per token, streaming ceiling 17.3 GB/s): decode cannot go below 1.06e9/17.3e9 = 61 ms per
token, so the decode term has a floor of 0.47 of ASR_RT; prefill with the batched arm
(measured 4x fewer launches, `w2` layout pending sign-off) has a floor around 0.4;
the front end is already 0.47 (encoder 0.0125, mel 0.065, wav 0.030); startup 0.64 is
one-time and amortizes to ~0 in a service that stays resident.

So the reachable floor for this workload on this board is ~1.3-2.0 warm, and the
`ASR_RT <= 1.00` bar as written (one-shot process, 129.7 s clip, 4 tok/s of generated
text) is not achievable on a Jetson Nano Gen1 at Q8: the weights are 1.06 GB/token and
the bus is 17.3 GB/s. Stating it as a target-vs-physics result rather than rounding the
metric: the honest deliverable is (a) warm ASR_RT, (b) the batched-prefill win once the
`w2` contract is signed, and (c) a resident-process mode that amortizes startup.

### 44. Measurement-discipline items from the gate wiring

* **`--trace` costs 77 ms/token.** The same binary on the same prompt reports
  `steady-state 108.68 ms/token` without `--trace` and 185.39 ms/token with it: printing
  1904 step lines through ssh is a 70% surcharge on decode. The ASR gate needs `--trace`
  to read the generated ids, so its phase split now charges decode at the untraced rate
  (`ASR_DECODE_MS`, default 108.68) and prints the traced number next to it as an explicit
  `--trace cost`. Reading the traced steady-state as the decode cost produced a nonsensical
  "prefill 2.3 s" on the first run of the split, which is what prompted the fix.
* **`checks_tok.py` and the ASR text view must run under the HF venv**
  (`/home/louis/venvs/lfm3/bin/python -I`), not the system python3.8: system transformers
  cannot load this checkpoint's tokenizer (broken PIL, and the tokenizer registers through
  the remote-code processor), and it presents as `Tokenizer class TokenizersBackend does
  not exist` -- a gate failure that is not one. Documented in the script header. With the
  venv, `tok gate: 531/531 fixtures exact+roundtrip (9988 ids)` and the ASR gate prints
  both texts:
  `engine text : 'The weather service issued its seasonal outlook on the 9th of March, and
  the numbers were unusual. In the nort'` -- identical to the HF golden for 207 tokens.

Gate board as of this writing, all on a fresh build: `ENC_GATE PASS` (20 stages,
run-to-run 0.000e+00), `WAV_GATE PASS` (16 fixtures, 6 chain), S2 mel `PASS`,
`tok gate: 531/531`, backbone `CHECKS OK` (matrix 23x23, 6 prompts of logits-seq top-1,
streams 2/2), `ASR_GATE PASS` (prefix 207, match 260/262, WER 0.008, EMB cos 0.999774,
ASR_RT 3.838 / warm 3.200).

### 45. RETRACTION: "decode dominates ASR_RT" was an arithmetic error, and realtime is reachable

Row 322 / section 43 reported `decode 207 s (1.60) > prefill 151 s (1.16)` and concluded
that `ASR_RT <= 1.00` is not reachable on this board at Q8. **Both are wrong**, and the
error is the kind this project has a rule against: I multiplied all 1904 steps by the
decode rate and labelled the product "decode".

In this engine a prompt step and a decode step cost the same thing (T=1 GEMVs per
position, `GEMV-only budget 127.7 ms`, 92% of the 17.3 GB/s ceiling), so the split is by
**step count**. Measured with graphs off and **`--trace` off**, the same binary on the
same prompt on the same board:

```
GRAPH=1  steady-state 109.01 ms/token   wall 291.12 s
GRAPH=0  steady-state 108.27 ms/token   wall 288.44 s        <- graphs cost nothing here
traced   (the gate's run)                wall 441 s            <- --trace surcharge ~150 s
```

Corrected decomposition of the 129.7 s clip, engine untraced:

| phase | s | /audio-s |
|---|---|---|
| prefill = 1642 prompt ids x 108.27 ms | 178 s | **1.37** |
| decode = 262 generated tokens x 108.27 ms | 28 s | **0.22** |
| engine one-time startup | 82 s | 0.63 |
| front end (wav 3.9 + mel 8.5 + encoder 47.6) | 60 s | 0.47 |
| **ASR_RT_TIMED** | **348 s** | **2.69** |

So `ASR_RT` as the gate printed it (3.838) was inflated by `--trace` by ~1.14, decode is
**0.22 not 1.60**, prefill is the real target at **1.37**, and the roofline conclusion
flips: with batched prefill (`w2`, the ARM5/CT=4 arm measured at 4x fewer launches) the
prefill term goes toward ~0.35, a resident process removes the 0.63 startup, decode sits
at 0.22 near the ceiling, and the front end is 0.47 -- **floor ~1.0-1.1, i.e. the
realtime bar is reachable, not out of reach.**

Two traps, both now closed in the gate: (1) `ASR_RT` must not be timed from a `--trace`
run -- `checks_asr.py` now does one untraced timed run and keeps the traced run only to
read the ids, printing the surcharge; (2) graphs are not the lever anyone would guess
from the backbone's TTFT work (109.01 vs 108.27 ms/token on this workload).

The `w2` sign-off is therefore the decision that matters for realtime, and a resident
process is worth 0.63.

### 46. `LFM_PF_USE=1` is refuted as the ASR prefill lever (measured, not argued)

The engine's existing batched ARM=5 prefill arm (validated bit-exact for T=1 on prompt A
in run #298, 153/153) was measured on the ASR prompt, graphs off, same board:

```
PF_USE=0  wall 449.5 s   1906 step lines parsed
PF_USE=1  TIMED OUT at 1200 s, 56 steps completed   (3% of the run)
```

So the arm that wins on the 20-id prompt A (where per-launch host overhead dominates)
loses catastrophically at 1642 prompt ids. It stays opt-in and off. The prefill term
(1.37 of ASR_RT) needs the real thing: **CT=4 positions per GEMV**, which is the `w2`
x-layout item -- the engine reads `x + row*K` (position-first) for `ffn.w2` (N=2048,
K=7680) while the other nine GEMVs prefer layer-first, and the earlier layer-first
attempt produced "top-1 matched by tie luck, top-5 garbage" (lines 1276-1286). That is
why it is a contract decision and not a flag.

### 47. ASR_RT as the gate now reports it

```
prefill  178.8 s (1.38) = 1642 prompt ids x 108.87 ms
decode    28.5 s (0.22) = 262 generated tokens x 108.87 ms
startup   82.0 s (0.63) = one-time blob load
wav 4.0 + mel 8.6 + encoder 47.3 + engine 291.5 (untraced; traced run was 441.3) = 351.3 s
ASR_RT_TIMED = 2.708        warm ASR_RT excluding the one-time startup = 2.076
```

accuracy on the same run is unchanged: prefix 207, all-position match 260/262,
token-WER 0.008, EMB cos 0.999774, `ASR_GATE PASS`.

Distance to the bar, term by term: prefill 1.38 -> ~0.35 with true CT=4 batching (the `w2`
item), startup 0.63 -> ~0 in a resident process, decode 0.22 is at the ceiling, front end
0.46 is mel 0.066 + encoder 0.365 (GPU 0.0125) + wav 0.031 + process launches. Floor
~1.0-1.1, so `ASR_RT <= 1.00` is at the edge of reachability, not out of it.

### 48. Resident-process mode (`--clip`): 93.7 s of the 545 s pair is the startup, and clips are bit-identical

`lfm3_nanode --clip SPEC --clip SPEC ...` runs several ASR clips in **one process**, so
the 82 s one-time startup (1.946 GB blob read + upload + setup) is paid once. `SPEC` is a
text file: line 1 = embedding path (or `-`), line 2 = prompt ids, line 3 = `--audio-at`
position, line 4 = `ngen` (per-clip generation count, so a short clip in a resident batch
isn't forced to the longest clip's `--steps`). With no `--clip`, the code is exactly one
pass, so the shipped single-prompt path is untouched.

```
resident wall 451.40 s for 2 clips      (solo pair: 453.5 + 91.6 = 545.1 s)  -> -93.7 s
clip 0 == solo 0 : True   (agree on 1906 ids)
clip 1 == solo 1 : True   (agree on   44 ids)
CLIPMODE_IDS PASS
```

`checks_asr.py` now has a `== resident clip mode ==` section that runs 2 clips in one
process and compares each against its own solo run, and `ASR_GATE` requires it
bit-identical. `.auto/lfm3/cmp_clips.py` takes `N:FILE` to pair run N with a solo
reference, and uses **prefix** semantics: a resident clip may be asked for a shorter tail
than the reference, and demanding equal lengths made a bit-identical clip report
`eq False` with `first diffs at []` -- the signature of a length check posing as a
content check.

Three traps hit and closed while building this, all of them the same class of trap this
project already has rules for:
1. **Stale binary, twice.** `nvcc -o /tmp/nX` followed by a failed build left the old
   `lfm3_nanode` in place, and the test measured it: the tell was the new build marker
   (`LFM3_CLIPMODE BUILD=...`) not printing, and `BUILT_MOVED` absent. Now every build
   step asserts `RC -eq 0 && test -s OUT` before `mv`, and the binary prints a build
   marker that the test greps for.
2. **Fixed-size spec buffer.** `char idsp[8192]` truncated the ASR prompt's 11 KB id
   line: `CLIP 0 START ids=150 ... at=0`, the remainder of line 2 read as line 3 (so
   `audio-at` became garbage) and the real line 3 read as line 4. The reader now grows a
   `std::string` across buffer-sized line fragments.
3. **`PQ1_GRAPH` is the Bonsai knob.** LFM3 uses `LFM_GRAPHS`, so section 45's
   "graphs cost nothing here" A/B ran graphs-off on both arms -- that claim is
   UNTESTED, not measured, and the clip-mode safety argument does not rely on it (LFM3
   captures graphs only when `LFM_GRAPHS` is set, so the ASR path never uses one).

Measured ASR_RT for a stream of clips is the warm number: 2.689 single-shot, **2.057
warm** (startup amortized), verified by the 2-clip wall above.

### 49. True CT=4 batched prefill, tranche 1: the x-layout question is resolved (`--prefill-chunk`)

Row 331 measured the ceiling (2.7–3.8× on the real shapes). Tranche 1 answers the
plumbing question the `w2` item was parked on: **can the batched GEMM read a
driver-owned, position-first activation arena (row *t* at `x + t*K`) and reproduce T
per-position launches bit-for-bit, on the real weights?**

`./lfm3_nanode <blob> --prefill-chunk T` says yes, for all ten real GEMV shapes at
**T=4 and T=24** (24 = the ASR chunk: 24 backbone tokens per 0.96 s of audio):

```
PF4-CHUNK ...feed_forward.w2.weight N=2048 K=7680 KC=1920 nch=4 T=24: BIT-EXACT
   (0/49152 cells differ, nonzero 49152/49152, bat 44.07 ms, 17.1 GMAC/s, plausible yes)
PF4_CHUNK PASS: 10 shapes, ALL BIT-EXACT        (T=4 and T=24, k_rmsnorm_T launch ok)
```

Design points, all deliberate:
* `k_rmsnorm_T` calls the **same `rmsnorm_body`** as `k_rmsnorm`, with a row offset from
  `blockIdx.x` — the T-wide path must be the same code in the same evaluation order as
  the T=1 path, because a re-written normalisation kernel is exactly what cost the
  encoder six runs (§40's in-place `k_rmsnorm_f32` race).
* `x` is a deterministic synthetic pattern (nonzero, finite, per-*t* distinct). The
  question is **layout**, not embedding semantics; real-activation coverage is
  `--prefill-arm5-gate` and row 331's captured activations.
* It compares **batched-vs-per-position**, never vs shipped: row 330 established that
  `q8_partial` and `q8_gemv_hs` are different functionals, so the GEMV-level "vs
  shipped" gate is not the arbiter. End-to-end HF gates are.
* A launch error prints **no number** (the rule, applied inside the new test).

**The test found a real defect on its first run**: every shape mismatched at `t>=1`
(`first diff at t=1 row=0`, unmatched cells holding `0x0000/0x7fff/0xffff` = unwritten
memory). Cause: **`k_combine_bf` has no T dimension** — it combines `N × nch` partials,
i.e. one position, so a single launch over a T-position partial buffer writes position 0
only. Fixed by combining per position in the test; a real T-wide prefill wants a T-aware
combine kernel (a one-line grid change), and the comment at that call records it.

Also fixed while building: `H` and `TPB` are `#defines` in this file, so `const int H =
2048` preprocessed to `const int 2048 = 2048`; and the host fill must use `host_f2bf`
(`f2bf` is `__device__`).

Not shipped: `--prefill-chunk` is a selftest. The layer loop is still one position per
step; the remaining tranches are conv (needs the previous `K-1` activations as chunk
state), RoPE/attention (T-wide scores, causal within chunk + all previous positions),
SwiGLU/residual/Q8-quant, and the T-aware combine — graded by the existing HF gates
(backbone streams 2/2, ASR EMB cos 0.999774, token-WER 0.008).
