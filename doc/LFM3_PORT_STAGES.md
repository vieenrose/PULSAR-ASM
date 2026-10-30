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
          [768, 4096, 1] over 4 concatenated 256-wide frames  => 25 tokens per second of audio
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
where 25 tokens per second of audio at 98 ms/step = 2.45 s per audio second. That is stage S8,
and it is the one place where the plan has to touch shipped numerics (see §5, risk R3).

## 3. What the finished runtime costs (derived)

Speech out: one backbone step (112.8 ms) + depth head (≈13 ms minimum, 25–40 ms
realistic) + codec (≈11.6 ms minimum, 25–50 ms realistic) per 80 ms frame
→ **≈137 ms minimum (1.7×) and ~180 ms realistic (≈2.2× realtime)**; a 5 s answer
takes ≈11 s. The old "129 ms / 1.6×" was bandwidth-only. Speech in
(ASR path): 1 s of audio = 25 encoder tokens = batched encoder ≈13 ms + backbone prefill 25 steps
× 98 ms = 2.45 s (unbatched, today) or ≈0.3–0.6 s once batched (FLOP-bound: 25 tokens × 3.5 GFLOP
= 88 GFLOP against ≈236 GFLOP/s fp32 peak) → the input side is where batching pays 5–8×.

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
| **S5 iSTFT spike** | **done, gated** (transformer stack not started) | `pulsar_arm/asm/lfm3_istft.cu`: direct inverse 641->1280 in a documented order, x hann, overlap-add hop 320 with a fixed 4-term contribution order (no atomics), envelope divide, `trim=480`/`tail=960` per `decode`. Gate `.auto/lfm3/checks_istft.py`: **9 cases, SNR 103-130 dB vs torch** (cases built from the checkpoint's real `decoder.lin` weights, bf16-rounded like the engine will, plus a 440 Hz tone and a window-centred click), well past the 60 dB bar; max abs scales with the case (relative bound 1e-3 x rms). **Decision: the `__sincosf` arm ships** -- it is faster in 9/9 cases (the 3.2 MB/frame twiddle table is DRAM-bound). Cost 0.7-4.5 ms/frame (board drift included; 75 frames per audio second => 5-11% RT, acceptable next to the backbone's 141% RT). **Follow-up: an FFT form** (1280 = 2^7 x 5) if it ever matters -- the direct form is SFU-bound on `__sincosf` (measured 2 ms/frame vs a 1.64 MFLOP/frame FLOP count => 0.8 GFLOP/s of ~100). |
| batched Q8 GEMM (D2's enabling kernel) | **measured, not shipped** | `pulsar_arm/asm/lfm3_gemm.cu` + `.auto/lfm3/checks_gemm.sh`: fp64 gate PASS on all 18 audio shapes x 5 T x 7 tiles (max relative error 2.9e-8). Performance verdict: **7.1 GFLOP/s ceiling on every shape, no batching win** (per-position cost flat in T; T=1 streams weights at 1.0-3.5 GB/s vs the engine's 17 GB/s). Two bugs the fp64 gate caught: an out-of-bounds write when T is not a multiple of TILE (`ceil(T/TILE)` tiles need the `t0+t<T` guard on both the x load and the store -- symptom: scattered zeros plus a poisoned CUDA context that made every later shape "fail to allocate"); and scalar `int8` weight loads (2.4x at T=1, and the 40x-off streaming rate). Root cause of the ceiling: the inner loop reads x from shared once per (weight, position) pair -> shared-bandwidth bound. **Action: port `q8_partial` (x register-staged, KCHUNK partials, R rows/block, warp-shuffle reductions) to the batched form; do not ship a from-scratch GEMM.** |
| S1 tokenizer | **done, gated (encode+decode)** | `pulsar_arm/tools/mktok.py` (blob: vocab 124893 + 293320 merges + 140 added + 782 L/N ranges + 10 White_Space ranges + byte map, 5.4 MB) + `pulsar_arm/asm/lfm3_tok.c` (hand-rolled Split regex, ByteLevel, ignore_merges BPE) + gate `.auto/lfm3/checks_tok.py`: **531/531 fixtures id-for-id vs HF (9988 ids) AND byte-exact round-trip** decode(encode(x))==x on all 531. The A6 (`\\s+(?!\\S)`) trap is documented in §12. |
| S3 encoder tower | **blob done; tower not started** | `pulsar_arm/tools/mkaudio.py` builds `lfm3-audio-2b-enc.lfm3q8` (130 MB, 176 tensors, 102 Q8 / 74 f32) with SELFCHECK OK and **176/176 dtype decisions matching Liquid's mmproj GGUF table**. Census corrections found by the cross-check (§13a): encoder is **158** tensors (not 159), tower-final `embedding_norm` exists (= GGUF `a.post_ln`), `self_attn.out_proj` needed its own Q8 pattern (the backbone's `o_proj` name didn't match it -- caught in the tool's first run), `lin_out` is Conv1d [768,4096,1] squeezed to 2-D and Q8 (GGUF `a.pre_encode.out`). Tower implementation + per-layer gate still to come. |
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
