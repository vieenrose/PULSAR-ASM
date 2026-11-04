"""speak_hf_oracle.py -- S6/M2 oracle: text prompt -> speech, straight from HF's generate.

This is the contract for the native controller (`lfm3_speak`). It runs HF's own
`Lfm3AudioForConditionalGeneration.generate`, not a reimplementation, and records the
three things the native side has to reproduce:

  prompt_ids.i32   the tokenized prompt (chat template + the audio_start marker HF appends)
  text.i32         the generated text slots, in order (post-frame trailing text)
  codes.<i>.i32    the 12 raw sampled codes of speech frame i (pre-EOS-override)
  hidden.<i>.f32   the LM last_hidden_state fed to the depth head for frame i (post final norm)
  frames.meta.json slot bookkeeping: how many frames, where the eos frame is, the text slots
  audio.f32        the codec-decoded 24 kHz waveform of the frames (the M2 listening target)

Wrapped rather than reimplemented on purpose: the mode machine (which slot is text and which
is a frame, when the eos frame flips the mode back to text) is exactly what S6 has to get
right, so reading it out of HF's own loop is the only way the gate means anything.

Env:
  SPEAK_MODEL   model dir (default /home/louis/models/lfm3/LFM3-Audio-2B)
  SPEAK_PROMPT  fixture index 0..4 (default 0) -- see PROMPTS
  SPEAK_FRAMES  max_new_tokens (slots), default 40
  SPEAK_MODE    audio | interleaved | text (default audio)
  SPEAK_OUT     output dir (default speech/speak_gate[_p<k>])
  SPEAK_CODEC   1 to also decode the frames with the codec (24 kHz WAV target)
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("SPEAK_MODEL", "/home/louis/models/lfm3/LFM3-Audio-2B")
FIXTURE = int(os.environ.get("SPEAK_PROMPT", "0"))
MAXSLOTS = int(os.environ.get("SPEAK_FRAMES", "40"))
MODE = os.environ.get("SPEAK_MODE", "audio")
OUT = os.environ.get("SPEAK_OUT", os.path.join(
    HERE, "speech", "speak_gate" if FIXTURE == 0 else "speak_gate/p%d" % FIXTURE))

# The chat template ends the generation prompt with "<|im_start|>assistant\n"; HF's generate
# appends <|audio_start|> itself when generation_mode="audio", so it must NOT be in the text.
PROMPTS = [
    ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
     "<|im_start|>user\nSay hello to the user.<|im_end|>\n<|im_start|>assistant\n"),
    ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
     "<|im_start|>user\nCount slowly from one to five.<|im_end|>\n<|im_start|>assistant\n"),
    ("<|im_start|>system\nYou are a concise narrator.<|im_end|>\n"
     "<|im_start|>user\nRead the weather report aloud.<|im_end|>\n<|im_start|>assistant\n"),
    ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
     "<|im_start|>user\nTell me a short fact about the sea.<|im_end|>\n<|im_start|>assistant\n"),
    ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
     "<|im_start|>user\nGood morning, how are you today?<|im_end|>\n<|im_start|>assistant\n"),
]
PROMPT_TEXT = PROMPTS[FIXTURE]


def w32(path, a):
    np.ascontiguousarray(np.asarray(a, dtype=np.float32)).tofile(path)


def wi32(path, a):
    np.ascontiguousarray(np.asarray(a, dtype=np.int32)).tofile(path)


def load_model():
    try:
        from transformers import AutoModelForMultimodalLM as Loader
    except ImportError:
        from transformers import AutoModelForCausalLM as Loader
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = Loader.from_pretrained(MODEL, dtype=torch.float32, trust_remote_code=True).eval()
    return tok, model


def main():
    os.makedirs(OUT, exist_ok=True)
    tok, model = load_model()
    cfg = model.config
    # where a field lives is not uniform (top level vs a sub-config); resolve once and report
    raw = json.load(open(os.path.join(MODEL, "config.json")))
    def cf(name, default=None):
        for holder in (cfg, getattr(cfg, "depthformer", None), getattr(cfg, "lfm", None)):
            if holder is not None and hasattr(holder, name):
                return getattr(holder, name)
        # the released config.json carries these at top level (codebooks, maskgit_steps, ...)
        if name in raw:
            return raw[name]
        if default is not None:
            return default
        raise AttributeError("no config holder has %r" % name)
    ids = tok(PROMPT_TEXT, return_tensors="pt").input_ids
    # apply_chat_template would re-render; the strings above are already the rendered prompt,
    # so tokenize them directly and record what the model actually sees.
    ids_list = ids[0].tolist()
    print("prompt: %d ids" % len(ids_list))

    rec = {"hidden": [], "codes": [], "orders": []}
    orig_sample = model.sample_audio_frame

    def wrapped(hidden_state, *a, **k):
        # single-row calls arrive 1-D (sample_audio_frame unsqueezes internally); a batch
        # row is already 2-D. Taking [0] unconditionally recorded one float per frame.
        h = hidden_state.float().cpu().numpy()
        rec["hidden"].append((h if h.ndim == 1 else h[0]).copy())
        out = orig_sample(hidden_state, *a, **k)
        rec["codes"].append(out[0].cpu().numpy().copy() if out.ndim > 1 else out.cpu().numpy().copy())
        return out

    model.sample_audio_frame = wrapped

    class TextStreamer:
        def __init__(self):
            self.ids = []
            self.skip = len(ids_list)

        def put(self, value):
            for t in value.reshape(-1).tolist():
                if self.skip > 0:
                    self.skip -= 1
                else:
                    self.ids.append(int(t))

        def end(self):
            pass

    class AudioStreamer:
        def __init__(self):
            self.frames = []

        def put(self, value):
            self.frames.append(np.asarray(value.cpu().numpy() if hasattr(value, "cpu")
                                          else value, dtype=np.int32))

        def end(self):
            pass

    ts, asr = TextStreamer(), AudioStreamer()
    with torch.no_grad():
        out = model.generate(
            ids,
            generation_mode=MODE,
            audio_temperature=0.0,
            max_new_tokens=MAXSLOTS,
            do_sample=False,
            return_audio_eos=True,
            streamer=ts,
            audio_streamer=asr,
        )

    codes = out.audio_codes[0].numpy()          # [12, n_frames] incl. the eos frame
    frames = [np.asarray(f, dtype=np.int32) for f in asr.frames]
    eos_code = int(cf("audio_eos_code", 2016))
    print("config: codebooks=%s vocab=%s maskgit=%s eos=%d"
          % (cf("codebooks"), cf("audio_vocab_size"), cf("maskgit_steps"), eos_code))

    wi32(os.path.join(OUT, "prompt_ids.i32"), ids_list)
    wi32(os.path.join(OUT, "text.i32"), ts.ids)
    for i, h in enumerate(rec["hidden"]):
        w32(os.path.join(OUT, "hidden.%d.f32" % i), h)
    for i, c in enumerate(rec["codes"]):
        wi32(os.path.join(OUT, "codes.%d.i32" % i), c)
    wi32(os.path.join(OUT, "audio_codes.i32"), codes)
    wi32(os.path.join(OUT, "frames.i32"), np.asarray(frames, dtype=np.int32).reshape(-1)
         if frames else np.zeros(0, dtype=np.int32))

    meta = {
        "fixture": FIXTURE, "mode": MODE, "max_new_tokens": MAXSLOTS,
        "prompt_text": PROMPT_TEXT, "prompt_ids": ids_list, "n_prompt": len(ids_list),
        "audio_start_token_id": int(cf("audio_start_token_id")),
        "mixed_start_token_id": int(cf("mixed_start_token_id")),
        "audio_out_token_id": int(cf("audio_out_token_id")),
        "text_end_token_id": int(cf("text_end_token_id")),
        "eos_code": eos_code, "num_codebooks": int(cf("codebooks")),
        "audio_vocab_size": int(cf("audio_vocab_size")),
        "maskgit_steps": int(cf("maskgit_steps")),
        "interleaved_n_text": int(cf("interleaved_n_text")),
        "interleaved_n_audio": int(cf("interleaved_n_audio")),
        "text_tokens": ts.ids,
        "n_frames_sampled": len(rec["codes"]),
        "n_frames_returned": int(codes.shape[1]),
        "frames_are_eos": [bool((c == eos_code).any()) for c in rec["codes"]],
        "audio_repetition_penalty": 1.1, "audio_repetition_window": 4096,
        "text_repetition_penalty": 1.0, "text_repetition_window": 32,
        "generation_mode_note": "HF appends audio_start to the prompt when mode=audio",
    }
    with open(os.path.join(OUT, "frames.meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("frames sampled %d (eos at %s), text slots %d"
          % (len(rec["codes"]),
             next((i for i, e in enumerate(meta["frames_are_eos"]) if e), None),
             len(ts.ids)))
    print("text ids: %s" % ts.ids[:40])
    for i, c in enumerate(rec["codes"][:4]):
        print("  frame %d: %s" % (i, c.tolist()))

    if os.environ.get("SPEAK_CODEC"):
        codec_oracle(OUT, codes[:, :len(rec["codes"])])
    print("wrote %s" % OUT)


def codec_oracle(outdir, codes):
    """Decode the frames with HF's codec into the 24 kHz target waveform (the M2 ear check)."""
    import importlib.util
    sys.path.insert(0, os.path.join(
        "/home/louis/.cache/huggingface/modules/transformers_modules",
        "LFM3_hyphen_Audio_hyphen_2B", "228cd04f9ca7b43e"))
    from modeling_liquid_codec import LiquidCodecModel
    from transformers import AutoConfig
    cc = AutoConfig.from_pretrained(os.path.join(MODEL, "audio_tokenizer"), trust_remote_code=True)
    codec = LiquidCodecModel.from_pretrained(
        os.path.join(MODEL, "audio_tokenizer"), config=cc, trust_remote_code=True).eval()
    with torch.no_grad():
        audio = codec.decode(torch.tensor(codes)[None])
    a = audio[0].float().cpu().numpy()
    w32(os.path.join(outdir, "audio.f32"), a)
    print("codec audio: %d samples (%.3f s at 24 kHz)" % (a.size, a.size / 24000.0))


if __name__ == "__main__":
    main()
