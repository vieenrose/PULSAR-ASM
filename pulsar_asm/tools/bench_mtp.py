#!/usr/bin/env python3
"""How many tokens does the official drafter actually get accepted?

Measuring this before building batched verification: the draft/verify accept
logic is identical whether the target checks its candidates one step at a time
or in one batched pass, so the acceptance rate here is exactly the multiplier a
batched verifier would cash in.

Greedy chain: the target's own prediction for the current position is the first
thing a draft has to match; each accepted draft then costs one more target step.

  python tools/bench_mtp.py --k 4
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from runtime.gemma4_model import Gemma4, rope_tables      # noqa: E402
import ref_gemma4_np as R                                 # noqa: E402
from ref_gemma4_assist_np import Assist                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="/mnt/edge/pulsar/gemma4_e2b.bin")
ap.add_argument("--assist", default="/mnt/edge/pulsar/gemma4_e2b_mtp.bin")
ap.add_argument("--tok", default="/mnt/edge/pulsar/tok")
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--prompt", default="Explain why the sky is blue, in two sentences.")
a = ap.parse_args()

from transformers import AutoTokenizer                    # noqa: E402
tk = AutoTokenizer.from_pretrained(a.tok)
ids = list(tk.apply_chat_template([{"role": "user", "content": a.prompt}],
                                  add_generation_prompt=True, tokenize=True, return_dict=False))

g = Gemma4(a.weights, max_seq=1024, verbose=False)
ref = R.Ref(a.weights)
kv = {}
for t, i in g.man["meta"]["kv_store_layers"].items():
    hd = g.man["meta"]["head_dim_full"] if t == "full_attention" else g.man["meta"]["head_dim_slide"]
    kk, vv = g._caches[i]
    kv[t] = (kk.reshape(g.max_seq, hd), vv.reshape(g.max_seq, hd))
A = Assist(a.assist, ref, kv)
tables = rope_tables(g.man["meta"], g.max_seq)

for t in ids:
    g.forward(t)
cur = int(g.logits().argmax())
accepted = steps = drafts_made = 0
draft_cost = tgt_cost = 0.0

print(f"{len(ids)} prompt tokens · drafting {a.k}/round · {a.steps} verify passes\n")
print("  pass    pos  accept  target says     drafts                accepted")
for s in range(a.steps):
    p = g.pos                       # position `cur` occupies; constant for the round
    t0 = time.perf_counter()
    tgt = int(g.forward(cur))       # target's prediction for position p+1
    tgt_cost += time.perf_counter() - t0
    h = g.buf["X"].copy()           # pre-final-norm hidden of position p
    t0 = time.perf_counter()
    drafts = A.draft(cur, h, p, a.k, tables)
    draft_cost += time.perf_counter() - t0
    drafts_made += a.k

    n_ok, prev, chain = 0, tgt, []
    for d in drafts:
        if d != prev:
            break
        n_ok += 1
        chain.append(d)
        prev = int(g.forward(d))    # verified drafts really enter the sequence
    accepted += n_ok
    steps += 1
    cur = prev
    print(f"  {s:>4}  {p:>6}  {n_ok:>6}     {tk.decode([tgt])!r:14s} "
          f"{[tk.decode([d]) for d in drafts[:3]]!s:26s} {tk.decode(chain)!r}")

per_pass = accepted / max(steps, 1)
print(f"\nmean accepted drafts/pass = {per_pass:.2f}")
print(f"tokens per verify pass    = {per_pass + 1:.2f}   (accepted + 1 bonus)")
print(f"draft cost                = {draft_cost/max(drafts_made,1)*1000:.1f} ms/draft token "
      f"(numpy; the asm engine will be ~100x cheaper)")
print(f"target cost               = {tgt_cost/max(steps,1)*1000:.0f} ms/pass")
g.close()
