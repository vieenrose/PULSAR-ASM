#!/usr/bin/env python
"""HF reference oracle for LFM3-Audio-2B (gate G4 / G3 of doc/LFM3_AUDIO_PLAN.md).

Run with ~/venvs/lfm3/bin/python (torch 2.6.0, transformers 5.21.0).

PROTOCOL TRAP (found 2026-10-28, cost a whole debugging cycle): the chat
template ends at "<|im_start|>assistant\n" with NO modality tag. The model's
first greedy token from that state is 125014 = <|text_end|>, which is in
generation_config.eos_token_id = [124900, 125014], so generate() stops
immediately and returns ZERO new tokens. That is not a bug and not a
divergence -- the model requires an explicit modality request:

    text out   -> append "<|text_start|>"  (125013)
    mixed out  -> append "<|mixed_start|>" (125015)   (MaskGIT / speech out)
    audio out  -> append "<|audio_out|>"   (125032)
    audio in   -> "<|audio|>"              (125031) in the user turn

Every gate prompt MUST append the modality tag, and every gate MUST assert a
non-zero token count, or the gate passes vacuously on an empty stream.
"""
import argparse, json, os, sys, time

import numpy as np
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

REPO = "LiquidAI/LFM3-Audio-2B"

# Control tokens (tokenizer.added_vocab; verified against the checkpoint).
TOK = {
    "pad": 124893, "startoftext": 124894, "im_start": 124899, "im_end": 124900,
    "think": 124901, "think_end": 124902, "audio_end": 125010, "quar": 125012,
    "text_start": 125013, "text_end": 125014, "mixed_start": 125015,
    "mixed_end": 125016, "audio": 125031, "audio_out": 125032,
}
EOS = [124900, 125014]

# The gate prompt set. Keep these frozen: the goldens are keyed on them.
PROMPTS = {
    "A": "Name the four seasons, and one thing that changes in each.",
    "B": "Why does a metal spoon feel colder than a wooden one at room temperature?",
}


def build_ids(proc, text, mode="text"):
    """Chat-template ids with the modality tag appended (see PROTOCOL TRAP)."""
    s = proc.apply_chat_template([{"role": "user", "content": text}],
                                 add_generation_prompt=True, return_tensors="pt")
    if not s.endswith("<|im_start|>assistant\n"):
        sys.exit("FATAL: chat template tail changed; goldens are keyed on it: %r" % s[-40:])
    s += {"text": "<|text_start|>", "mixed": "<|mixed_start|>",
          "audio_out": "<|audio_out|>"}[mode]
    return proc.tokenizer(s, return_tensors="pt", add_special_tokens=False)["input_ids"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--logits", action="store_true", help="also dump first-step logits (G3)")
    a = ap.parse_args()

    t0 = time.time()
    proc = AutoProcessor.from_pretrained(REPO, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        REPO, trust_remote_code=True, dtype=torch.bfloat16).eval()
    print("loaded in %.1fs; %.1f M params" % (time.time() - t0,
          sum(p.numel() for p in model.parameters()) / 1e6), flush=True)

    for key, text in PROMPTS.items():
        ids = build_ids(proc, text, "text")
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=a.tokens, do_sample=False)
        seq = out.sequences[0] if hasattr(out, "sequences") else out[0]
        gen = seq[ids.shape[1]:].tolist()

        # Vacuous-gate guard: the empty-stream failure mode is silent.
        assert len(gen) > 0, (
            "prompt %r produced 0 tokens -- modality tag missing/wrong" % key)
        assert gen[0] not in EOS, (
            "prompt %r: first token is EOS (%d); modality tag missing?" % (key, gen[0]))

        gold = {"prompt": text, "prompt_ids": ids[0].tolist(), "gen": gen,
                "text": proc.tokenizer.decode(gen), "n": len(gen)}
        p = os.path.join(a.out, "golden_greedy_%s.json" % key)
        json.dump(gold, open(p, "w"), indent=1)
        print("G4 %s: %d tokens -> %s\n   %r" % (key, len(gen), p, gold["text"][:110]))

        if a.logits:
            with torch.no_grad():
                lg = model(input_ids=ids).logits[0, -1].float().cpu().numpy()
            np.save(os.path.join(a.out, "logits_last_%s.npy" % key), lg)
            print("G3 %s: logits_last %s finite=%d argmax=%d"
                  % (key, lg.shape, np.isfinite(lg).sum(), int(lg.argmax())))

    print("ORACLE_OK")


if __name__ == "__main__":
    main()
