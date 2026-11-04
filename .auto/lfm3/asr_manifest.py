#!/usr/bin/env python3
"""asr_manifest.py -- emit the ASR prompt text + manifest the native shell (lfm3_asr) needs.

The shell must build the SAME 1642-id prompt the Python gates build, without HF. It gets
the prompt as TEXT (with a sha256 the shell verifies at runtime), the placeholder string,
the audio token id, the expansion count, the audio start position and the eos id -- all
taken from the HF golden, so there is one source of truth and the shell cannot drift
silently.

  /home/louis/venvs/lfm3/bin/python .auto/lfm3/asr_manifest.py

Writes speech/asr_prompt.txt and speech/asr_manifest.txt.
"""
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "speech", "hf_asr_golden.json")
OUT_TXT = os.path.join(HERE, "speech", "asr_prompt.txt")
OUT_MAN = os.path.join(HERE, "speech", "asr_manifest.txt")

g = json.load(open(GOLDEN))
p = g["prompt"]
head, tail = p["prompt_text_head"], p["prompt_text_tail"]
best = 0
for k in range(min(len(head), len(tail)), 0, -1):
    if head[-k:] == tail[:k]:
        best = k
        break
text = head + tail[best:]
if len(text) != int(p["prompt_text_len"]):
    sys.exit("splice gives %d chars, golden says %d" % (len(text), p["prompt_text_len"]))
h = hashlib.sha256(text.encode()).hexdigest()
if h != p["prompt_text_sha256"]:
    sys.exit("sha256 mismatch %s vs %s" % (h[:16], p["prompt_text_sha256"][:16]))

# The audio start position is where the placeholder tokens begin in the id list. Recompute
# it the way asr_prompt_ids.py does -- ids before the placeholder -- so the shell's
# --audio-at cannot drift from the id layout.
sys.path.insert(0, HERE)
from transformers import AutoTokenizer                      # noqa: E402
MODEL = os.environ.get("LMF3_MODEL", os.environ.get("LFM3_MODEL",
                                                   "/home/louis/models/lfm3/LFM3-Audio-2B"))
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
ph = p["audio_placeholder_string"]
pre, post = text.split(ph)
pre_ids = list(tok(pre, add_special_tokens=False).input_ids)
post_ids = list(tok(post, add_special_tokens=False).input_ids)
n_ph = int(p["prompt_audio_placeholder_tokens"])
full = pre_ids + [int(p["audio_token_id"])] * n_ph + post_ids
if len(full) != int(p["prompt_tokens_total"]):
    sys.exit("rebuilt prompt is %d ids, golden says %d" % (len(full), p["prompt_tokens_total"]))
ref = [int(x) for x in open(os.path.join(HERE, "speech", "asr_prompt_ids.txt")).read().strip().split(",")]
if full != ref:
    d = next(i for i in range(len(ref)) if ref[i] != full[i])
    sys.exit("rebuilt ids differ from the gated asr_prompt_ids.txt at %d (%d vs %d)" % (d, full[d], ref[d]))
print("rebuilt ids match the gated file 1642/1642; audio starts at position %d" % len(pre_ids))

open(OUT_TXT, "wb").write(text.encode())
man = [
    ("sha256", p["prompt_text_sha256"]),
    ("prompt_text_len", str(p["prompt_text_len"])),
    ("audio_placeholder", ph),
    ("audio_token_id", str(p["audio_token_id"])),
    ("prompt_audio_placeholder_tokens", str(n_ph)),
    ("prompt_tokens_total", str(p["prompt_tokens_total"])),
    ("audio_at", str(len(pre_ids))),
    ("eos_token_id", str(p["eos_token_id"])),
]
with open(OUT_MAN, "w") as f:
    f.write("\n".join("%s=%s" % kv for kv in man) + "\n")
print("wrote %s (%d bytes) and %s" % (OUT_TXT, len(text.encode()), OUT_MAN))
