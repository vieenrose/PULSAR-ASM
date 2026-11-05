"""gate_speak.py -- S6 backbone-in-the-loop gate: teacher-forced speak replay vs HF.

Usage: gate_speak.py <oracle_dir> <nanode_bin> <blob>
Feeds the oracle's exact slot sequence (prompt + mode marker + text ids with
audio_out placeholders at speech slots) and the oracle's frame embeddings, then checks:
 (a) SPEAK-EMB served exactly nframes rows (the counter line proves the path ran),
 (b) per frame i: the engine's post-final-norm hidden at the step that consumed the
     last pre-frame token matches the oracle's hidden.i (the quantity HF's
     sample_audio_frame received) -- cos and energy-relative error,
 (c) per text slot: argmax of the engine's logits row equals the recorded text id
     (HF's text repetition penalty is 1.0, a no-op, so raw argmax must match).
"""
import json
import os
import subprocess
import sys

import numpy as np

G, ENG, BLOB = sys.argv[1], sys.argv[2], sys.argv[3]
W = "/tmp/speakgate"
os.makedirs(W, exist_ok=True)
NCB, H = 12, 2048

meta = json.load(open(os.path.join(G, "frames.meta.json")))
mode = meta["mode"]
marker = {"audio": meta["audio_start_token_id"],
          "interleaved": meta["mixed_start_token_id"]}[mode]
AUDIO_OUT = meta["audio_out_token_id"]
prompt = np.fromfile(os.path.join(G, "prompt_ids.i32"), dtype=np.int32).tolist()
full = prompt + [marker]
slots, frame_slots, text_at = [], [], {}
for line in open(os.path.join(G, "slots.txt")):
    k, v = line.split()
    if k == "A":
        frame_slots.append(len(slots))  # slot index of this frame
        full.append(AUDIO_OUT)
        slots.append(("A", int(v)))
    else:
        text_at[len(full)] = int(v)
        full.append(int(v))
        slots.append(("T", int(v)))
P, S, N = len(prompt) + 1, len(slots), len(full)

emb_path = os.path.join(G, "audio_embeds.f32")
emb = np.fromfile(emb_path, dtype=np.float32)
nf = emb.size // H
assert emb.size == nf * H and nf == meta["n_frames_sampled"], (emb.size, nf)

csv = ",".join(str(t) for t in full)
lg_path = os.path.join(W, "logits.f32")
hd_path = os.path.join(W, "hidden.f32")
for p in (lg_path, hd_path):
    if os.path.exists(p):
        os.remove(p)
r = subprocess.run([ENG, BLOB, "--prompt", csv, "--steps", str(N),
                    "--speak-emb", emb_path,
                    "--dump-logits-all", lg_path, "--dump-hidden-all", hd_path],
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
print("\n".join(l for l in r.stdout.splitlines()
                if "SPEAK-EMB" in l or "steady-state" in l or "ERROR" in l or "FAIL" in l))
if r.returncode != 0:
    sys.exit("engine failed rc=%d\n%s" % (r.returncode, r.stderr[-500:]))
fed = [l for l in r.stdout.splitlines() if "SPEAK-EMB fed" in l]
assert fed and "fed %d of %d" % (nf, nf) in fed[0], fed

lg = np.fromfile(lg_path, dtype=np.float32)
V = lg.size // N
assert lg.size == N * V, (lg.size, N)
lg = lg.reshape(N, V)
hd = np.fromfile(hd_path, dtype=np.float32).reshape(N, H)

# (c) text slots. Causal LM: the logits at row q-1 predict token q, so the text id at
# seq pos q must equal argmax of row q-1 (comparing row q is the classic off-by-one:
# every row then shows the NEXT slot's token, found on the first run: pos29 showed 358).
bad_t = [(q, tid, int(lg[q - 1].argmax())) for q, tid in text_at.items()
         if int(lg[q - 1].argmax()) != tid]
print("text slots: %d/%d argmax-exact" % (len(text_at) - len(bad_t), len(text_at)))
for q, tid, got in bad_t[:8]:
    print("  pos %d: trace %d engine-argmax %d" % (q, tid, got))

# (b) frame hiddens: oracle hidden.i is the output for toks[framepos-1]
cos_min, rel_max, bad_f = 1.0, 0.0, []
for i in range(nf):
    slot_idx = next(j for j, s in enumerate(slots) if s[0] == "A" and
                    sum(1 for t in slots[:j + 1] if t[0] == "A") == i + 1)
    framepos = P + slot_idx
    ref = np.fromfile(os.path.join(G, "hidden.%d.f32" % i), dtype=np.float32)
    got = hd[framepos - 1]
    cos = float(got.dot(ref) / np.sqrt((got ** 2).sum() * (ref ** 2).sum()))
    rel = float(np.sqrt(((got - ref) ** 2).mean()) / np.sqrt((ref ** 2).mean()))
    cos_min = min(cos_min, cos)
    rel_max = max(rel_max, rel)
    if cos < 0.999 or rel > 5e-2:
        bad_f.append((i, cos, rel))
# Bar rationale (recorded, not tuned to fit): the backbone runs Q8 weights with bf16
# activations against an fp32 oracle; the S4/M1 precedent accepts this regime via stream
# equality (ASR: 260/262 tokens) rather than bit equality. The hidden check proves the
# engine tracks HF's trajectory (neighbors sit at cos<0, so the gate has teeth); the
# text check proves decisions match, with margins >> any perturbation (min margin 4.1).
# The LIVE sampling loop, not this replay, is the final arbiter.
bar_ok = cos_min >= 0.999 and rel_max <= 5e-2
print("frame hiddens: %d/%d within bar (cos min %.6f, rel max %.3e)" %
      (nf - len(bad_f), nf, cos_min, rel_max))
for i, c, rr in bad_f[:8]:
    print("  frame %d: cos %.6f rel %.3e" % (i, c, rr))

ok = not bad_t and bar_ok
print("SPEAK GATE: %s" % ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
