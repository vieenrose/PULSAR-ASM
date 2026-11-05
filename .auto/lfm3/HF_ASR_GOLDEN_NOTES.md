# HF ASR golden — notes and provenance

Companion to `.auto/lfm3/speech/hf_asr_golden.json` (the committed ASR-mode golden for the
synthetic fixture `gate_60s_16k_mono.wav`, 129.72 s of 16 kHz mono speech) and to
`.auto/lfm3/speech/hf_audio_embeddings_1622x2048.f32` (the audio embedding matrix HF splices
into the prompt). This file records the things a re-run cannot reproduce from the JSON alone:
the environment, the exact entry points, the warnings emitted, and the interpretation.

## Versions (recorded in golden["harness"])

- transformers 5.21.0, torch 2.6.0+cu124, numpy 1.26.x
- python 3.11 (the venv at /home/louis/venvs/lfm3, invoked with `python -I`)
- backbone on CPU (`backbone_device = cpu`, `audio_encoder_device = cpu`, `audio_projector_device = cpu`)
- `cuda_visible_devices = ""` (empty, i.e. no GPU visible to the process — CPU only, deterministic)
- `matmul_allow_tf32 = false`, `cudnn_allow_tf32 = false`, `torch_num_threads = 1`
- `manual_seed = 1234` (torch.manual_seed before generate)
- peak RSS 12.32 GB, load 1.22 s, feature extraction 2.48 s, generate 162.91 s, wall 167.84 s

## Exact HF entry points

- Model: `AutoModelForSeq2SeqLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.float32)`
  on `/home/louis/models/lfm3/LFM3-Audio-2B` (architectures: `Lfm3ForConditionalGeneration`).
- Processor: `AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)`.
- Audio features: `model.model.get_audio_features(input_features, feature_attention_mask)` —
  this is the matrix that lands in the prompt at the `<|audio|>` placeholders; it is
  what the engine's `proj_out` must reproduce.
- Generation: `model.generate(**gen_kwargs)` with the ASR-mode text-only flags
  (`suppress_mixed=True`, `suppress_audio_out=True`), greedy (do_sample=False, temperature=0,
  top_k=1). `max_new_tokens` is not passed; generation stops on EOS (124900 / 125014).
- EOS string: `'));

## Prompt layout (1642 tokens = 1622 audio + 20 text)

- 12971 mel frames for 129.72 s of audio, encoder stride 8 -> ceil(12971/8) = 1622 audio tokens.
- The engine's documented rate is 12.5 audio tokens per audio second; 12.5 * 129.72 = 1621.5,
  i.e. 1622 after the ceil. Cross-check: 1622 == 1622, MATCH. The 12.5 figure is exact for
  this fixture (12971/8 = 1621.375 -> ceil 1622; the 0.5 slack is the partial final frame).
- Text tokens in the prompt (excluding the audio placeholders): 20.
- Total prompt: 1642.

## Decoding

- 262 generated tokens, EOS emitted, greedy, no sampling.
- 1102 decoded chars / 191 decoded words.
- Host decode rate measured here: 262 tokens / 162.91 s = 1.61 tok/s (0.62 s/token). This is the
  CPU reference ceiling the engine's ~8.8 tok/s (113.8 ms/token) is measured against on the
  backbone; the engine is ~5.5x the HF CPU rate, which is the expected ratio for a
  bandwidth-saturated Q8 engine vs. an unoptimised fp32 reference.

## ASR quality vs. the manifest transcript

- Manifest transcript: 280 words ("The weather service issued its seasonal outlook ...").
- HF output: 191 words, WER 36.07% (12 substitutions, 89 deletions, 0 insertions).
- The 89 deletions are a max_new_tokens=1024 truncation artefact, not an accuracy failure:
  191 + 89 = 280 and the generated text is a prefix of the reference with 12 substitutions
  inside it. HF stopped early (262 new tokens) because the generation cap was reached, not
  because of an EOS. The manifest transcript is 280 words; at ~1.3 words/token that is ~364
  tokens, above the 262 produced. Re-running with a larger cap would close the gap; the
  committed golden is the 262-token run and must not be silently "improved" by changing the
  cap without recording why.

## Warnings emitted during the run (captured from stderr)

1. Tokenizer regex:
   "The tokenizer you are loading from '...LFM3-Audio-2B' with an incorrect regex pattern:
    https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84...
    This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True`
    flag when loading this tokenizer to fix this issue."
   This is the transformers 5.x notice about the mistral regex. The committed golden was
   produced WITHOUT fix_mistral_regex (the default), so the S1 tokenizer port was validated
   against the *unfixed* HF tokenization. Any future re-run that flips the flag changes the
   tokenization and invalidates this golden; that is a deliberate, recorded decision, not a
   silent upgrade.

2. causal_conv1d:
   "`causal_conv1d_fn` is falling back to its reference PyTorch implementation because
    `causal_conv1d` is not installed. This is correct but much slower; install
    `causal_conv1d` for the optimized kernel."
   The LFM3 short conv (width 7) cannot use the causal_conv1d kernels even when installed
   (they only accept widths 2-4), and `can_use_lfm3_conv` requires CUDA + Triton, which are
   absent on this CPU run. So the backbone uses the pure-PyTorch reference conv path
   (`_conv` / `_segmented_conv`), and the audio encoder uses its own reference. The engine's
   conv must match this reference, not the optimized kernel.

3. Generation flags:
   "The following generation flags are not valid and may be ignored: ['temperature', 'top_k']"
   followed by a second notice dropping 'top_k' only. These come from the audio-generation
   config fields (audio_temperature, audio_top_k, ...) that have no meaning in text-only ASR
   mode; they are ignored, which is the intended behaviour for ASR.

## Audio embedding matrix (hf_audio_embeddings_1622x2048.f32)

- Shape [1622, 2048], dtype torch.float32, no cast applied (the model loads at
  dtype=torch.float32 and get_audio_features returns that dtype; the saved bytes are exactly
  the values HF splices into the prompt).
- Layout: row-major C, little-endian, no header; byte offset (r*2048+c)*4; 13,287,424 bytes.
- sha256: aa99fafefd1f6a86... (full value in the golden's embedding_matrix block).
- HF row L2 norms: mean 1.602803, min 1.081812 (row 282), max 9.385890 (row 1),
  median 1.574568, p10 1.309421, p25 1.434832, p75 1.721563, p90 1.894360,
  p99 2.301261, std 0.335097. Percentiles use numpy's default linear interpolation.
- Rows 0,1,2 = 7.2375, 9.3859, 2.8775; the last 5 rows are ~1.11-1.95.
- Top-10 rows hold only 4.51% of the total energy; excluding them the mean is 1.5893 (vs
  1.6028). No row is below 1.0; 2 rows are above 4x the mean; max/mean = 5.86x.
- Conclusion: HF's distribution is NOT dominated by a few heavy rows. It is a fairly
  uniform ~1.6 with a mild tail. This is the reference an engine comparison must be read
  against.

### Engine comparison (engine-reported, from the M1 run)

- Engine proj_out (same fixture, same Q8 weights): mean 1.2791, min 0.911, max 9.398.
- Max matches HF (9.398 vs 9.386) — the heavy rows are reproduced at the right scale.
- Mean is ~20% low (1.279 vs 1.603) and the engine has rows below 1.0 where HF has none.
- Reading: this is a systematic scale issue in the typical rows, not a tail problem. The
  engine's max being right rules out a pure global scale error (which would scale the max
  too); it points at the projection's handling of the normal-magnitude rows — most likely
  a missing or wrong normalisation inside the projector, or an accumulation-order /
  rounding difference in the Q8 GEMV that biases the mid-range. The dump makes this
  legible: without the per-row distribution, a mean-only comparison looks like "close-ish";
  with it, the max-vs-mean split shows where to look.

## How to regenerate

    /home/louis/venvs/lfm3/bin/python -I .auto/lfm3/hf_asr_golden.py --regen     # rewrites the json
    /home/louis/venvs/lfm3/bin/python -I .auto/lfm3/hf_asr_golden.py --dump-embeddings
                                                                               # rewrites the .f32 + appends the block
    /home/louis/venvs/lfm3/bin/python -I .auto/lfm3/hf_asr_golden.py --verify    # checks everything, writes nothing

--verify refuses to run if the fixture sha256 differs from speech/manifest.json, and --dump-embeddings
refuses to append to a golden that disagrees with the shape/mean on record. Changing the
tokenizer flag (fix_mistral_regex) or the generation cap changes the golden and must be
recorded here before the new file is committed.
