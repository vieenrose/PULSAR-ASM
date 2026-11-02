#!/usr/bin/env python3
"""asr_prompt_ids.py -- reconstruct HF's ASR prompt token ids for the fixture.

The golden stores the prompt as TEXT plus a sha256, and the counts
(prompt_tokens_total, prompt_audio_placeholder_tokens, audio_token_id). The engine takes
ids, so this rebuilds the id list from the same text with the same tokenizer and then
REFUSES to write it unless every stored count and the sha256 agree. The reason for the
checks: an id list that is one token off shifts every audio embedding's position, which
shows up as a transcript that is subtly wrong -- the class of bug a text-comparison gate
catches late and an id-level check catches here.

  /home/louis/venvs/lfm3/bin/python .auto/lfm3/asr_prompt_ids.py            # write
  ... --check                                                                # verify only

Writes .auto/lfm3/speech/asr_prompt_ids.txt as one comma-separated list.
"""
import argparse, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "speech", "hf_asr_golden.json")
OUT = os.path.join(HERE, "speech", "asr_prompt_ids.txt")
MODEL = os.environ.get("LFM3_MODEL", "/home/louis/models/lfm3/LFM3-Audio-2B")

ap = argparse.ArgumentParser()
ap.add_argument("--check", action="store_true")
a = ap.parse_args()

g = json.load(open(GOLDEN))
p = g["prompt"]
text = p["prompt_text_head"]
if len(text) != p["prompt_text_len"]:
    # The golden stores a head and a tail view. Splice them on their overlap and let the
    # sha256 be the judge -- if the overlap is ambiguous the hash will not match, and a
    # silently mis-spliced prompt would shift every audio embedding by a token.
    head, tail = p["prompt_text_head"], p["prompt_text_tail"]
    best = 0
    for k in range(min(len(head), len(tail)), 0, -1):
        if head[-k:] == tail[:k]:
            best = k
            break
    text = head + tail[best:]
    if len(text) != p["prompt_text_len"]:
        sys.exit("head+tail splice gives %d chars, golden says %d (overlap %d chars found) "
                 "-- extend the golden to store the full prompt text" % (len(text), p["prompt_text_len"], best))
h = hashlib.sha256(text.encode()).hexdigest()
if h != p["prompt_text_sha256"]:
    sys.exit("prompt text sha256 mismatch: %s vs golden %s" % (h[:16], p["prompt_text_sha256"][:16]))
print("prompt text reconstructed (%d chars), sha256 matches the golden" % len(text))

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
ids = list(tok(text, add_special_tokens=False).input_ids)

n_total, n_ph, aid = int(p["prompt_tokens_total"]), int(p["prompt_audio_placeholder_tokens"]), int(p["audio_token_id"])
occ = text.count(p["audio_placeholder_string"])

# HF's chat template puts ONE placeholder token in the prompt text and the model
# scatters get_audio_features()'s rows into that slot, so the ids the backbone actually
# sees are the placeholder repeated n_ph times. The engine takes the expanded form, so
# expand here -- and keep every stored count as a check, because an id list that is one
# token off shifts every audio embedding's position.
ph = [i for i, t in enumerate(ids) if t == aid]
if len(ph) == 1 and n_ph > 1:
    i0 = ph[0]
    ids = ids[:i0] + [aid] * n_ph + ids[i0 + 1:]
    print("expanded the single %s placeholder at index %d into %d copies of audio_token_id %d"
          % (p["audio_placeholder_string"], i0, n_ph, aid))

fails = []
if len(ids) != n_total:
    fails.append("token count %d != golden %d" % (len(ids), n_total))
if occ != 1 and occ != n_ph:
    fails.append("placeholder occurrences %d, expected 1 (expanded) or %d (pre-expanded)" % (occ, n_ph))
pos = [i for i, t in enumerate(ids) if t == aid]
if len(pos) != n_ph:
    fails.append("ids equal to audio_token_id %d: %d, expected %d" % (aid, len(pos), n_ph))
if pos and pos != list(range(pos[0], pos[0] + n_ph)):
    fails.append("audio placeholder ids are not contiguous: first %d, count %d, span %d..%d"
                 % (pos[0], n_ph, pos[0], pos[-1]))
print("prompt ids: %d total, audio_token_id=%d appears %d times at positions %d..%d, text tokens %d"
      % (len(ids), aid, len(pos), pos[0] if pos else -1, pos[-1] if pos else -1, len(ids) - len(pos)))
for f in fails:
    print("FAIL:", f)
if fails:
    sys.exit(1)
if not a.check:
    open(OUT, "w").write(",".join(str(i) for i in ids))
    print("wrote %s (audio segment at positions %d..%d)" % (OUT, pos[0], pos[-1]))
