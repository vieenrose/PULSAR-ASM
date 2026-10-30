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
| audio encoder | `encoder.lfm.lfm.` | 159 | 219 MB | 109.5 M | ~123 MB |
| encoder downsample front end | `encoder.downsample.` | 11 | 3.7 MB | 1.85 M | fp32, keep |
| audio→backbone projector | `mlp.` | 6 | 11.5 MB | 5.77 M | ~6.5 MB |
| MaskGIT depth head | `depthformer.` | 37 | 67.8 MB | 33.9 M | ~38 MB |
| 12 codebook logits | `depth_embeddings.` | 36 | 99.2 MB | 49.6 M | ~56 MB |
| backbone→12 streams | `depth_linear.` | 2 | 50.4 MB | 25.2 M | ~28 MB |
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

Unbatched codec decoding is fatal (4× realtime slower); batched it is 5% of an 80 ms frame
budget. The encoder goes from "as expensive as realtime" to free. **So: no component after the
backbone may be written token-at-a-time.** The backbone already exists and is token-at-a-time,
which is fine for its 1-step-per-80 ms speech role (RTF 1.41, §3) but not for audio *prefill*,
where 25 tokens per second of audio at 98 ms/step = 2.45 s per audio second. That is stage S8,
and it is the one place where the plan has to touch shipped numerics (see §5, risk R3).

## 3. What the finished runtime costs (derived)

Speech out: one backbone step (112.8 ms) + depth head (≈12 ms) + codec (≈4.3 ms) per 80 ms frame
→ **≈129 ms per 80 ms of speech ≈ 1.6× realtime**; a 5 s answer takes ≈8 s. Speech in
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
bent into this), the per-head gate `[32,1024]`, stacked `depth_linear` (12288×2048) and 12
`to_logits` (2017×1024 each) as one `N = 12288` and one `N = 24204` GEMV at repack time, then the
maskgit loop (3 steps) with the model card's sampler: temperature 1.0, top-k 128, repetition
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
   positions) — the 1.15–1.45× standalone figure from the backbone's CT=2 kernel is the precedent,
   but the audio shapes are wider-K and should do better.
4. iSTFT determinism: `torch.fft.irfft` vs our fixed 1280-point inverse on the same spectra.
5. MaskGIT tie density, as the backbone's reference-precision work did for text.

Until those five exist, every timing in this document stays labelled *derived*.

## 10. Status (updated as stages land)

| stage | state | evidence |
|---|---|---|
| **S2 log-mel front end** | **done, gated** | `pulsar_arm/tools/mkmel.py` (build-time: the reference's own `mel_filter_bank` + `torch.hann_window` + twiddles -> `mel.bin`), `pulsar_arm/asm/lfm3_mel.S` (window + half-length pair-form direct DFT 400->201, 128x208 mel matmul, clamp+affine -- NEON, documented reduction order), `lfm3_mel.c` (HF's `pad_length`/`num_frames` padding, `log10f`), `lfm3_mel_test.c` (bit-exact C mirror). Gate `.auto/lfm3/checks_mel.py`, 19 fixtures: **asm vs C mirror 29,412 checks 0 mismatches (bit-exact)**; vs HF frame counts exact on all 19, live-cell (within 6 decades of the frame peak) max 8.3e-6, live mean 2.2e-7; exact-f64 referee says the engine is as accurate as HF (worst mean-ratio 1.28); all-cell max 0.0499 disclosed (one chirp cell 13.5 decades under its peak, where HF is 0.039 from exact). Cost **0.554 ms/frame** = 5.5% of realtime, so the front end is free. |
| **S5 iSTFT spike** | **done, gated** (transformer stack not started) | `pulsar_arm/asm/lfm3_istft.cu`: direct inverse 641->1280 in a documented order, x hann, overlap-add hop 320 with a fixed 4-term contribution order (no atomics), envelope divide, `trim=480`/`tail=960` per `decode`. Gate `.auto/lfm3/checks_istft.py`: **9 cases, SNR 103-130 dB vs torch** (cases built from the checkpoint's real `decoder.lin` weights, bf16-rounded like the engine will, plus a 440 Hz tone and a window-centred click), well past the 60 dB bar; max abs scales with the case (relative bound 1e-3 x rms). **Decision: the `__sincosf` arm ships** -- it is faster in 9/9 cases (the 3.2 MB/frame twiddle table is DRAM-bound). Cost 0.7-4.5 ms/frame (board drift included; 75 frames per audio second => 5-11% RT, acceptable next to the backbone's 141% RT). **Follow-up: an FFT form** (1280 = 2^7 x 5) if it ever matters -- the direct form is SFU-bound on `__sincosf` (measured 2 ms/frame vs a 1.64 MFLOP/frame FLOP count => 0.8 GFLOP/s of ~100). |
| S1 tokenizer | not started | |
| S3 encoder tower | not started (unblocked: S2's gate is green) | |
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
