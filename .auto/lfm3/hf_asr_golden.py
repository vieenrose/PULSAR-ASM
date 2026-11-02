#!/usr/bin/env python3
"""HF-derived speech->text golden for the committed ASR fixture.

The arbiter for the audio-input path of the hand-written engine is HF, never the
engine itself: this script runs the checkpoint's own audio path (transformers
5.21.0 + the checkpoint's trust_remote_code processor/feature-extractor/model) on
.auto/lfm3/speech/gate_60s_16k_mono.wav and records what HF produces.

    python .auto/lfm3/hf_asr_golden.py             # run HF, print the report, write nothing
    python .auto/lfm3/hf_asr_golden.py --regen     # run HF, write speech/hf_asr_golden.json
    python .auto/lfm3/hf_asr_golden.py --verify    # re-hash the fixture, re-run HF, compare
                                                   # against the committed golden; writes nothing

--verify (and every other mode) refuses to run when the fixture's sha256 differs
from speech/manifest.json, so a golden can never be attached to a swapped clip.

HF entry points used (see speech/HF_ASR_GOLDEN_NOTES.md for the full list):
  AutoProcessor.from_pretrained(trust_remote_code=True)      -> Lfm3AudioProcessor
  processor.apply_chat_template(..., tokenize=True, return_dict=True)
  AutoModelForMultimodalLM.from_pretrained(dtype=float32)    -> Lfm3AudioForConditionalGeneration
  model.model.get_audio_features(input_features, feature_attention_mask)  # token-count probe
  model.generate(**inputs, do_sample=False, suppress_mixed=True, suppress_audio_out=True)
  processor.decode(...)                                      # the README's text-out path

The task is the checkpoint's documented ASR mode: system prompt
"Perform ASR in English.", one audio clip in the user turn, text-only output.
Nothing here is our engine, whisper, or a cloud ASR.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import platform
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPEECH = os.path.join(REPO, ".auto", "lfm3", "speech")
FIXTURE = os.path.join(SPEECH, "gate_60s_16k_mono.wav")
MANIFEST = os.path.join(SPEECH, "manifest.json")
GOLDEN = os.path.join(SPEECH, "hf_asr_golden.json")
MODEL = "/home/louis/models/lfm3/LFM3-Audio-2B"

# The checkpoint files the golden is pinned to. A change to any of these (weights,
# mel front end, chat template, sampling defaults) invalidates the golden.
PINNED_FILES = [
    "config.json",
    "model.safetensors",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "feature_extraction_lfm3_audio.py",
    "processing_lfm3_audio.py",
    "modeling_lfm3_audio.py",
    "configuration_lfm3_audio.py",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "audio_tokenizer/config.json",
    "audio_tokenizer/model.safetensors",
]

SYSTEM_PROMPT = "Perform ASR in English."
MAX_NEW_TOKENS = 1024
SAMPLING_RATE = 16000
# The engine's measured audio-token rate (run #303): 12.5 tokens per audio second.
ENGINE_TOKENS_PER_AUDIO_SECOND = 12.5


def sha256_file(path: str, block: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(block)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def load_manifest() -> dict:
    with open(MANIFEST) as fh:
        return json.load(fh)


def check_fixture(manifest: dict, mode: str) -> tuple[str, int]:
    """Refuses to proceed when the fixture on disk is not the fixture in the manifest."""
    digest = sha256_file(FIXTURE)
    want = manifest["wav_sha256"]
    if digest != want:
        print(f"FATAL [{mode}]: fixture sha256 {digest} != manifest wav_sha256 {want}", file=sys.stderr)
        print("The fixture changed; refusing to write or certify a golden against it.", file=sys.stderr)
        sys.exit(2)
    with open(FIXTURE, "rb") as fh:
        header = fh.read(44)
    if header[0:4] != b"RIFF" or header[8:12] != b"WAVE":
        print("FATAL: fixture is not a RIFF/WAVE file", file=sys.stderr)
        sys.exit(2)
    fmt = int.from_bytes(header[20:22], "little")
    channels = int.from_bytes(header[22:24], "little")
    rate = int.from_bytes(header[24:28], "little")
    bits = int.from_bytes(header[34:36], "little")
    if (fmt, channels, rate, bits) != (1, manifest["channels"], manifest["rate"], manifest["bits"]):
        print(f"FATAL: fixture header pcm_{bits}s{bits}le {rate} Hz {channels} ch "
              f"({fmt=}), manifest says pcm_s{manifest['bits']}le {manifest['rate']} Hz "
              f"{manifest['channels']} ch", file=sys.stderr)
        sys.exit(2)
    return digest, int(manifest["samples"])


def normalize_words(text: str) -> list[str]:
    """Lowercase, digits stay digits, punctuation and paragraph breaks drop out."""
    keep = []
    for word in text.lower().split():
        word = word.strip(".,;:!?\"'()[]{}“”‘’")
        if word:
            keep.append(word)
    return keep


def wer(ref: list[str], hyp: list[str]) -> dict:
    """Word error rate by edit distance (substitutions + deletions + insertions) / len(ref)."""
    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
        prev = cur
    # trace back for the split
    i, j, sub, dele, ins = n, m, 0, 0, 0
    prev_rows = [list(range(m + 1))]
    prev = prev_rows[0]
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
        prev_rows.append(cur)
        prev = cur
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and prev_rows[i][j] == prev_rows[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            sub += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and prev_rows[i][j] == prev_rows[i - 1][j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return {
        "ref_words": n,
        "hyp_words": m,
        "errors": prev_rows[n][m],
        "substitutions": sub,
        "deletions": dele,
        "insertions": ins,
        "wer": prev_rows[n][m] / n if n else 0.0,
    }


def run_hf(device: str, max_new_tokens: int) -> dict:
    import numpy as np
    import soundfile as sf
    import torch
    import transformers
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    t_start = time.time()
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    tokenizer = proc.tokenizer
    feature_extractor = proc.feature_extractor

    # Read the fixture ourselves (pcm_s16le -> float32) so the samples that reach HF
    # are exactly the committed bytes; the extractor is then told the rate explicitly.
    waveform, file_rate = sf.read(FIXTURE, dtype="float32")
    if file_rate != SAMPLING_RATE or waveform.ndim != 1:
        print(f"FATAL: fixture is {file_rate} Hz shape {waveform.shape}, expected {SAMPLING_RATE} Hz mono",
              file=sys.stderr)
        sys.exit(2)
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    num_samples = int(waveform.shape[0])

    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "audio", "audio": waveform}]},
    ]
    prompt_text = proc.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = proc.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"sampling_rate": SAMPLING_RATE},
    )
    input_ids = inputs["input_ids"]
    input_features = inputs["input_features"]
    feature_attention_mask = inputs["feature_attention_mask"]
    audio_token_id = tokenizer.convert_tokens_to_ids(proc.audio_token)
    placeholder_tokens = int((input_ids == audio_token_id).sum())
    mel_frames = int(input_features.shape[-1])
    feature_lengths = feature_attention_mask.sum(-1)

    t_load = time.time()
    model = AutoModelForMultimodalLM.from_pretrained(MODEL, dtype=torch.float32, trust_remote_code=True)
    model.eval()
    backbone_device = torch.device(device)
    model.language_model.to(backbone_device)
    model.lm_head.to(backbone_device)
    # The audio encoder and projector stay on CPU: fp32 there is deterministic (no cuDNN
    # TF32, no kernel-selection drift) and the 12971-frame downsampler does not fit
    # alongside the backbone on an 8 GB card. Both are HF modules of this checkpoint.
    model.model.audio_encoder.to("cpu")
    model.model.audio_projector.to("cpu")
    load_seconds = time.time() - t_load

    with torch.no_grad():
        t_fe = time.time()
        # Count-only probe of the exact call the model's forward makes: this is what HF
        # feeds the backbone, one projected vector per eight mel frames.
        audio_features = model.model.get_audio_features(input_features, feature_attention_mask)
        feature_seconds = time.time() - t_fe
        audio_feature_tokens = int(audio_features.shape[0])
        audio_feature_norm = float(audio_features.norm(dim=-1).mean())

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "suppress_mixed": True,
            "suppress_audio_out": True,
            "return_dict_in_generate": True,
        }
        # Penalties the checkpoint's generate does not implement are pinned at their inert
        # values in the checkpoint's generation_config.json; presence/frequency/no-repeat/ngram
        # penalties are not accepted as kwargs at all (`_generate` builds its own warpers, so the
        # standard LogitsProcessor list is never used). Record them, do not pass them.
        inert_fields = (
            "presence_penalty", "frequency_penalty", "no_repeat_ngram_size", "early_stopping",
            "penalty_alpha", "typical_p", "epsilon_cutoff", "eta_cutoff", "diversity_penalty",
            "length_penalty", "min_length", "min_new_tokens", "stop_strings",
        )
        penalty_state = {k: getattr(model.generation_config, k, "<absent>") for k in inert_fields}
        not_inert = sorted(
            k for k in model._generation_fields_unsupported
            if getattr(model.generation_config, k, model._generation_fields_unsupported[k])
            not in (model._generation_fields_unsupported[k], None)
        )
        t_gen = time.time()
        outputs = model.generate(
            input_ids=input_ids.to(backbone_device),
            attention_mask=inputs["attention_mask"].to(backbone_device),
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            **gen_kwargs,
        )
        generate_seconds = time.time() - t_gen

    sequences = outputs.sequences
    prompt_len = int(input_ids.shape[1])
    generated_ids = sequences[0, prompt_len:]
    # sequences returned by this model start at the prompt; mirror the README slice
    generated_list = [int(t) for t in generated_ids]
    text = proc.decode(generated_ids, skip_special_tokens=True)
    text_raw = proc.decode(generated_ids, skip_special_tokens=False)
    eos_ids = model.generation_config.eos_token_id
    eos_ids = [int(e) for e in (eos_ids if isinstance(eos_ids, (list, tuple, set)) else [eos_ids])]
    eos_token = tokenizer.eos_token
    eos_emitted = any(t in eos_ids for t in generated_list)

    import resource

    return {
        "harness": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            "backbone_device": str(backbone_device),
            "audio_encoder_device": "cpu",
            "audio_projector_device": "cpu",
            "model_dtype": str(model.dtype),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "manual_seed": 0,
            "max_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
            "load_seconds": round(load_seconds, 2),
            "feature_seconds": round(feature_seconds, 2),
            "generate_seconds": round(generate_seconds, 2),
            "wall_seconds": round(time.time() - t_start, 2),
        },
        "prompt": {
            "system_prompt": SYSTEM_PROMPT,
            "chat_template_add_generation_prompt": True,
            "prompt_text_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
            "prompt_text_len": len(prompt_text),
            "prompt_text_head": prompt_text[:96],
            "prompt_text_tail": prompt_text[-64:],
            "audio_placeholder_string": proc.audio_token,
            "audio_placeholder_occurrences_in_text": prompt_text.count(proc.audio_token),
            "prompt_tokens_total": prompt_len,
            "prompt_audio_placeholder_tokens": placeholder_tokens,
            "prompt_text_tokens_excluding_audio": prompt_len - placeholder_tokens,
            "audio_token_id": audio_token_id,
            "audio_out_token_id": tokenizer.convert_tokens_to_ids(proc.audio_out_token),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "tokenizer_padding_side": tokenizer.padding_side,
            "feature_extractor_padding_side": feature_extractor.padding_side,
        },
        "feature_extractor": {
            "class": type(feature_extractor).__name__,
            "sampling_rate_requested": SAMPLING_RATE,
            "sampling_rate_of_extractor": int(feature_extractor.sampling_rate),
            "num_mel_bins": int(feature_extractor.feature_size),
            "n_fft": int(feature_extractor.n_fft),
            "hop_length": int(feature_extractor.hop_length),
            "win_length": int(feature_extractor.win_length),
            "encoder_stride": int(feature_extractor.encoder_stride),
            "log_floor": float(feature_extractor.log_floor),
            "normalization_offset": float(feature_extractor.normalization_offset),
            "normalization_divisor": float(feature_extractor.normalization_divisor),
            "input_audio_samples": num_samples,
            "mel_frames_from_samples_formula": feature_extractor.num_frames(num_samples),
            "mel_frames_in_tensor": mel_frames,
            "feature_attention_mask_ones": int(feature_lengths.sum()),
            "audio_tokens_from_frames_formula": feature_extractor.num_encoder_tokens(mel_frames),
            "input_features_shape": [int(x) for x in input_features.shape],
            "input_features_dtype": str(input_features.dtype),
        },
        "audio_embeddings": {
            "hf_entry_point": "model.model.get_audio_features(input_features, feature_attention_mask)",
            "hf_shape": [int(x) for x in audio_features.shape],
            "hf_audio_embedding_tokens_fed_to_backbone": audio_feature_tokens,
            "embedding_dim": int(audio_features.shape[1]),
            "mean_row_l2_norm": audio_feature_norm,
            "matches_placeholder_count": audio_feature_tokens == placeholder_tokens,
            "matches_frames_over_stride": audio_feature_tokens == feature_extractor.num_encoder_tokens(mel_frames),
            "engine_tokens_per_audio_second": ENGINE_TOKENS_PER_AUDIO_SECOND,
            "engine_expected_tokens": 1622,
            "hf_minus_engine": audio_feature_tokens - 1622,
        },
        "decoding": {
            "hf_entry_point": "model.generate(**inputs, **gen_kwargs)",
            "generate_kwargs": gen_kwargs,
            "generation_config_default": {
                k: v for k, v in model.generation_config.to_dict().items() if not k.startswith("_")
            },
            "effective_text_sampling": "do_sample=False forces temperature/top_k/top_p/min_p to None in "
            "modeling_lfm3_audio._generate (lines 1468-1471), so _build_warpers returns [] and "
            "_sample takes the argmax branch (line 1663): pure greedy",
            "eos_token_ids": eos_ids,
            "eos_token_string": eos_token,
            "pad_token_id": int(model.generation_config.pad_token_id),
            "penalty_fields_not_passed_as_kwargs": penalty_state,
            "generation_config_fields_outside_the_inert_defaults": not_inert,
            "eos_emitted": eos_emitted,
            "generated_tokens": len(generated_list),
            "generated_token_ids": generated_list,
        },
        "output": {
            "decoded_text_skip_special_tokens": text,
            "decoded_text_with_special_tokens": text_raw,
            "decoded_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "decoded_words": len(normalize_words(text)),
            "decoded_chars": len(text),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regen", action="store_true", help="write speech/hf_asr_golden.json")
    ap.add_argument("--verify", action="store_true",
                    help="re-hash the fixture, re-run HF, compare with the committed golden; never writes")
    ap.add_argument("--device", default="cpu",
                    help="backbone device (default cpu); the audio encoder/projector stay on CPU")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--golden", default=GOLDEN, help="golden path (for experiments)")
    args = ap.parse_args()

    if args.regen and args.verify:
        print("FATAL: --regen and --verify are mutually exclusive", file=sys.stderr)
        return 2
    mode = "regen" if args.regen else ("verify" if args.verify else "report")

    manifest = load_manifest()
    fixture_sha, _samples = check_fixture(manifest, mode)
    transcript = manifest["transcript"]
    transcript_sha = hashlib.sha256(transcript.encode()).hexdigest()
    if transcript_sha != manifest["transcript_sha256"]:
        print(f"FATAL: manifest transcript sha256 {transcript_sha} != recorded "
              f"{manifest['transcript_sha256']}", file=sys.stderr)
        return 2
    if mode == "verify" and not os.path.exists(args.golden):
        print(f"FATAL: --verify needs an existing golden at {args.golden}", file=sys.stderr)
        return 2

    print(f"[{mode}] fixture {os.path.relpath(FIXTURE, REPO)} sha256 {fixture_sha} (matches manifest)")
    missing = [rel for rel in PINNED_FILES if not os.path.isfile(os.path.join(MODEL, rel))]
    if missing:
        print(f"FATAL: model files missing under {MODEL}: {missing}", file=sys.stderr)
        return 2
    t_sha = time.time()
    # Hash the checkpoint before running it: a swapped weight file must not get as far as producing a golden.
    pinned = {rel: sha256_file(os.path.join(MODEL, rel)) for rel in PINNED_FILES}
    print(f"[{mode}] hashed {len(PINNED_FILES)} checkpoint files in {time.time() - t_sha:.1f}s "
          f"(model.safetensors {pinned['model.safetensors'][:16]}...)")
    result = run_hf(args.device, args.max_new_tokens)

    audio = result["audio_embeddings"]
    gen = result["decoding"]
    text = result["output"]["decoded_text_skip_special_tokens"]
    ref_words, hyp_words = normalize_words(transcript), normalize_words(text)
    err = wer(ref_words, hyp_words)
    sm = difflib.SequenceMatcher(a=ref_words, b=hyp_words)
    opcodes = [{"op": op, "ref": [i1, i2], "hyp": [j1, j2]} for op, i1, i2, j1, j2 in sm.get_opcodes()
               if op != "equal"]

    golden = {
        "kind": "HF-derived speech->text golden (ASR mode) for the committed synthetic fixture",
        "produced_by": "transformers %s on the checkpoint's own trust_remote_code model; no non-HF ASR "
                       "(whisper, cloud) was used" % result["harness"]["transformers"],
        "fixture": {
            "path": os.path.relpath(FIXTURE, REPO),
            "sha256": fixture_sha,
            "manifest": os.path.relpath(MANIFEST, REPO),
            "manifest_sha256": sha256_file(MANIFEST),
            "manifest_wav_sha256": manifest["wav_sha256"],
            "rate": manifest["rate"],
            "channels": manifest["channels"],
            "bits": manifest["bits"],
            "samples": manifest["samples"],
            "duration_s": manifest["duration_s"],
            "kind": manifest["kind"],
            "tool": manifest["tool"],
            "transcript_sha256": transcript_sha,
            "transcript_words": manifest["words"],
            "transcript": transcript,
        },
        "model": {
            "path": MODEL,
            "architectures": json.load(open(os.path.join(MODEL, "config.json")))["architectures"],
            "pinned_sha256": pinned,
        },
        "task": {
            "name": "ASR (speech->text), the checkpoint's documented transcription mode",
            "system_prompt": SYSTEM_PROMPT,
            "text_only_output_flags": {"suppress_mixed": True, "suppress_audio_out": True},
            "asr_mode_available": True,
        },
        "crosscheck_vs_engine": {
            "engine_audio_tokens_per_audio_second": ENGINE_TOKENS_PER_AUDIO_SECOND,
            "engine_expected_tokens_for_this_clip": audio["engine_expected_tokens"],
            "hf_audio_embedding_tokens_fed_to_backbone": audio["hf_audio_embedding_tokens_fed_to_backbone"],
            "hf_minus_engine": audio["hf_minus_engine"],
            "match": audio["hf_minus_engine"] == 0,
            "mel_frames": result["feature_extractor"]["mel_frames_in_tensor"],
            "mel_frames_per_audio_second": round(
                result["feature_extractor"]["mel_frames_in_tensor"] / manifest["duration_s"], 4),
            "encoder_stride": result["feature_extractor"]["encoder_stride"],
            "formula": "ceil(mel_frames / encoder_stride) = ceil(%d / 8) = %d"
            % (result["feature_extractor"]["mel_frames_in_tensor"],
               audio["hf_audio_embedding_tokens_fed_to_backbone"]),
        },
        "asr_quality": {
            "wer_vs_manifest_transcript": err,
            "normalized_ref_words": ref_words,
            "normalized_hyp_words": hyp_words,
            "diff_opcodes": opcodes,
            "note": "word error rate of HF's own transcription against the manifest ground truth; "
                    "lowercase, punctuation-stripped, digits compared as written",
        },
        **result,
    }

    print("[crosscheck] HF audio-embedding tokens fed to the backbone: %d   (engine: %d)  -> %s"
          % (audio["hf_audio_embedding_tokens_fed_to_backbone"], audio["engine_expected_tokens"],
             "MATCH" if audio["hf_minus_engine"] == 0 else "MISMATCH %d" % audio["hf_minus_engine"]))
    print("[prompt] %d tokens total = %d audio placeholders + %d text tokens; mel frames %d"
          % (result["prompt"]["prompt_tokens_total"], result["prompt"]["prompt_audio_placeholder_tokens"],
             result["prompt"]["prompt_text_tokens_excluding_audio"],
             result["feature_extractor"]["mel_frames_in_tensor"]))
    print("[decode] %d generated tokens, eos_emitted=%s, %.1fs (%.2f s/token)"
          % (gen["generated_tokens"], gen["eos_emitted"], result["harness"]["generate_seconds"],
             result["harness"]["generate_seconds"] / max(1, gen["generated_tokens"])))
    print("[text] %.200s%s" % (text, "..." if len(text) > 200 else ""))
    print("[WER] %.2f%% over %d ref words (S %d / D %d / I %d)"
          % (100 * err["wer"], err["ref_words"], err["substitutions"], err["deletions"], err["insertions"]))

    if mode == "verify":
        with open(args.golden) as fh:
            want = json.load(fh)
        checks = [
            ("fixture sha256", want["fixture"]["sha256"], fixture_sha),
            ("manifest sha256", want["fixture"]["manifest_sha256"], golden["fixture"]["manifest_sha256"]),
            ("model pinned sha256", want["model"]["pinned_sha256"], pinned),
            ("generated token ids", want["decoding"]["generated_token_ids"], gen["generated_token_ids"]),
            ("decoded text sha256", want["output"]["decoded_text_sha256"],
             golden["output"]["decoded_text_sha256"]),
            ("hf audio token count", want["audio_embeddings"]["hf_audio_embedding_tokens_fed_to_backbone"],
             audio["hf_audio_embedding_tokens_fed_to_backbone"]),
            ("prompt tokens", want["prompt"]["prompt_tokens_total"], golden["prompt"]["prompt_tokens_total"]),
        ]
        bad = 0
        for name, a, b in checks:
            ok = a == b
            bad += not ok
            print("  %-24s %s" % (name, "OK" if ok else "DIFFERS"))
            if not ok:
                print("      golden: %s\n" % json.dumps(a)[:400], "      this run: %s" % json.dumps(b)[:400])
        print("[verify] %s" % ("PASS: HF reproduced the committed golden bit-for-bit" if not bad
                               else "FAIL: %d field(s) differ" % bad))
        return 0 if not bad else 1

    if mode == "report":
        print("[report] no file written (use --regen)")
        return 0

    with open(args.golden, "w") as fh:
        json.dump(golden, fh, indent=1, ensure_ascii=False, sort_keys=False)
        fh.write("\n")
    print("[regen] wrote %s (%d bytes)" % (args.golden, os.path.getsize(args.golden)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
