#!/usr/bin/env python3
# Per-token kernel breakdown of an fwd35 run traced with
#   sudo -n env E_GRAPH=0 nvprof --print-gpu-trace --csv --normalized-time-unit us --log-file tr.csv \
#     ./fwd35 model.gguf G <prompt ids>
# The run ends with one DtoH copy per generated token, so the last G DtoH rows split the trace
# into prefill (everything before the first of them) and decode (G - 1 steps after it).
# Usage: python3 nvprof_split.py tr.csv G NPROMPT
import csv, re, sys, collections
path, G, NP = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = []
with open(path) as f:
    for line in f:
        if line.startswith('"Start"'): break
    next(f)
    for r in csv.reader(f):
        if len(r) < 19: continue
        try: rows.append((float(r[0]), float(r[1]), r[18], r[2]))
        except ValueError: pass
dth = [i for i, r in enumerate(rows) if 'DtoH' in r[2]]
cut = dth[-G]
def agg(lo, hi, ntok, label):
    tot = collections.Counter(); cnt = collections.Counter()
    for s, d, n, gx in rows[lo:hi]:
        key = n if 'CUDA m' in n else re.sub(r'\(.*', '', n).replace('void ', '')
        if 'k_gemv' in key: key += ' gx' + gx
        tot[key] += d; cnt[key] += 1
    wall = rows[hi - 1][0] + rows[hi - 1][1] - rows[lo][0]
    print(f'== {label}: {ntok} tok, wall {wall / 1e3 / ntok:.2f} ms/tok, kernel sum {sum(tot.values()) / 1e3 / ntok:.2f} '
          f'(k_ssq overlaps the residual GEMV on a second stream)')
    for k, v in tot.most_common(): print(f'  {v / 1e3 / ntok:8.3f} ms/tok {cnt[k] / ntok:7.2f}/tok  {k}')
agg(cut + 1, len(rows), G - 1, 'decode')
agg(0, cut + 1, NP, 'prefill')
