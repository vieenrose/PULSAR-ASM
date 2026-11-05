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
SEED = 1234


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
            self.primed = False

        def put(self, value):
            # HF's first put is the whole prompt, INCLUDING any marker generate() appended
            # (audio mode adds <|audio_start|>), so drop the entire first call. Skipping only
            # len(prompt) recorded the marker as generated text (run 1).
            if not self.primed:
                self.primed = True
                return
            for t in value.reshape(-1).tolist():
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
    # randperm drives the MaskGIT reveal order and is the only RNG in greedy temp-0 decoding;
    # unseeded, two runs gave 4 and 17 frames. Seed it so the recording is reproducible.
    # record each frame's reveal order: sample_audio_frame draws one randperm(num_codebooks)
    # per frame, and the depth engine takes that order as data. Wrap randperm only for the
    # call, then restore it.
    orders = []
    real_randperm = torch.randperm
    def rec_randperm(n, *a, **k):
        p = real_randperm(n, *a, **k)
        if int(n) == int(cf("codebooks")):
            orders.append(p.cpu().numpy().astype(np.int32).copy())
        return p
    torch.randperm = rec_randperm
    torch.manual_seed(SEED)
    try:
      with torch.no_grad():
        out = model.generate(
            ids,
            # "none" infers the mode from the prompt (text, with the model free to switch to speech)
            generation_mode=None if MODE == "none" else MODE,
            audio_temperature=0.0,
            max_new_tokens=MAXSLOTS,
            do_sample=False,
            return_audio_eos=True,
            streamer=ts,
            audio_streamer=asr,
        )
    finally:
        torch.randperm = real_randperm
    # slots.txt: one line per generated slot, in HF's own order: "T <id>" for a text token,
    # "A <eos 0|1>" for a speech frame. Taken from the returned sequences, so the marker
    # generate() appends is excluded by construction, not by counting.
    seq = out.sequences[0].tolist()
    marker = {"audio": [int(cf("audio_start_token_id"))],
              "interleaved": [int(cf("mixed_start_token_id"))]}.get(MODE, [])
    gen_start = len(ids_list) + (1 if marker else 0)
    if marker:
        assert seq[len(ids_list)] == marker[0], "appended marker not where expected"
    gen = seq[gen_start:]
    lines, a = [], 0
    for t in gen:
        if t == int(cf("audio_out_token_id")):
            is_eos = int(bool((out.audio_codes[0][:, a] == int(cf("audio_eos_code", 2016))).any()))
            lines.append("A %d" % is_eos)
            a += 1
        else:
            lines.append("T %d" % t)
    with open(os.path.join(OUT, "slots.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    for i, o in enumerate(orders):
        wi32(os.path.join(OUT, "order.%d.i32" % i), o)

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
    # HF's own code -> backbone-input path: codebook_embedding(codes + arange*codebook_size).sum.
    # Recorded from HF's function (not re-derived here) so the native gather+sum can be gated
    # against it, offset convention included.
    with torch.no_grad():
        e = model.model.get_audio_out_embeds(torch.tensor(codes.T, dtype=torch.long))
    w32(os.path.join(OUT, "audio_embeds.f32"), e[0].float().numpy() if e.ndim == 3 else e.float().numpy())
    print("audio_embeds", tuple(e.shape))
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
    import glob
    import importlib
    import importlib.util
    # same importlib-as-package trick as codec_hf_oracle.py: the cached remote module
    # uses relative imports, so a bare sys.path insert dies (seen: ImportError at the
    # end of an otherwise complete oracle run).
    cands = sorted(glob.glob("/home/louis/.cache/huggingface/modules/transformers_modules/"
                             "LFM3_hyphen_Audio_hyphen_2B/*/modeling_liquid_codec.py"),
                   key=os.path.getmtime)
    if not cands:
        raise SystemExit("no cached modeling_liquid_codec.py")
    base = os.path.dirname(cands[-1])
    pkg = "lfm3_remote_codec"
    if pkg not in sys.modules:
        spec = importlib.util.spec_from_file_location(pkg, os.path.join(base, "__init__.py"),
                                                      submodule_search_locations=[base])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[pkg] = mod
        spec.loader.exec_module(mod)
    m = importlib.import_module(pkg + ".modeling_liquid_codec")
    from transformers import AutoConfig
    cc = AutoConfig.from_pretrained(os.path.join(MODEL, "audio_tokenizer"), trust_remote_code=True)
    codec = m.LiquidCodecModel.from_pretrained(
        os.path.join(MODEL, "audio_tokenizer"), config=cc, trust_remote_code=True).eval()
    with torch.no_grad():
        audio = codec.decode(torch.tensor(codes)[None])
    a = audio[0].float().cpu().numpy()
    w32(os.path.join(outdir, "audio.f32"), a)
    print("codec audio: %d samples (%.3f s at 24 kHz)" % (a.size, a.size / 24000.0))


if __name__ == "__main__":
    main()
