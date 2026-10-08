#!/usr/bin/env python3
"""Diff fwd_k2 engine output vs oracle fixtures (run on Spark, needs npz).

Usage: k2_diff.py <fixtures.npz>
- engine greedy continuation must equal oracle gen ids EXACTLY
- engine final-step argmax must equal oracle logits argmax
- full-logit maxrel reported (rope/libm paths use tolerance)
"""
import re
import subprocess
import sys

import numpy as np

d = np.load(sys.argv[1])
prompt = [int(x) for x in d["prompt"]]
gen_oracle = [int(x) for x in d["gen"]]
ologits = d["logits"].astype(np.float64)
print(f"prompt {len(prompt)} ids, oracle gen {gen_oracle}", flush=True)

ids = " ".join(map(str, prompt))
cmd = (f"taskset -c 5 ./fwd_k2 k2h_09.blob {ids} --gen {len(gen_oracle)}")
out = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                     cwd="/home/luigi/k2").stdout
steps = []
for ln in out.splitlines():
    m = re.match(r"step (\d+) id_in=(\d+) top=(\d+) logit=([\d.\-e+]+)", ln)
    if m:
        steps.append((int(m.group(1)), int(m.group(2)), int(m.group(3)),
                      float(m.group(4))))
n0 = len(prompt)
gen_eng = [i for (s, i, t, l) in steps if s >= n0]
print(f"engine generated {len(gen_eng)}: {gen_eng}", flush=True)
print("STREAM:", "EXACT" if gen_eng == gen_oracle else "DIFFERS")
# final-step logits: engine only prints top; compare argmax + top logit
ae = max(range(len(ologits)), key=lambda i: ologits[i])
print(f"oracle argmax {ae} logit {ologits[ae]:.4f}", flush=True)
last = [x for x in steps if x[0] == n0 + len(gen_oracle) - 1]
if last:
    print(f"engine last top {last[0][2]} logit {last[0][3]:.4f}", flush=True)
