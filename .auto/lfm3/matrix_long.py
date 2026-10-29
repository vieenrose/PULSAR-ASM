#!/usr/bin/env python
"""Long-context matrix gate: per-layer x per-position rms vs HF at 300 positions.

The shipped gates compare 23 teacher-forced positions (prompt A). Audio inputs
are hundreds of tokens, where the sliding window (256), RoPE at large pos and
the KV cache all engage for the first time -- and nothing checks them. This
gate teachers-forces a 300-id prompt on BOTH sides and compares rms cell by
cell with the same 3% rule as gate_matrix.

Engine side: --steps 300 consumes the 300 prompt ids (no generation), --trace
prints POS/L/rms per step per layer. HF side: one forward with hidden states,
rms over hidden dim per layer per position (row L = after layer L).

Usage:  python .auto/lfm3/matrix_long.py            # dump HF ref, run engine, compare
        python .auto/lfm3/matrix_long.py --hf-only   # just (re)dump the reference
"""
import argparse, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from checks_lfm3 import NANO, DIR, BLOB, BIN, ENVQ, run_remote  # noqa: E402

NT = 300


def prompt_ids(nt=NT):
    import json
    a = json.load(open(os.path.join(HERE, "golden_greedy_A.json")))["prompt_ids"]
    return (a * ((nt // len(a)) + 1))[:nt]


def dump_hf(ids, path, dtype=None):
    import numpy as np
    import torch
    dtype = dtype or torch.bfloat16
    from transformers import AutoModelForMultimodalLM
    model = AutoModelForMultimodalLM.from_pretrained(
        "LiquidAI/LFM3-Audio-2B", dtype=dtype, trust_remote_code=True).to("cuda").eval()
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], device="cuda"), output_hidden_states=True)
    hs = out.hidden_states            # [embed, L0..L23]
    print("hidden states: %d tensors of %s" % (len(hs), tuple(hs[0].shape)))
    rows = []
    for h in hs[1:]:
        v = h[0].float().cpu().numpy()
        rows.append(np.sqrt((v.astype(np.float64) ** 2).mean(-1)))
    arr = np.stack(rows).astype(np.float64)     # (24, T): row L = after layer L
    np.save(path, arr)
    print("wrote %s %s" % (path, arr.shape))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-only", action="store_true")
    ap.add_argument("--ref", default=os.path.join(HERE, "hf_matrix_long.npy"))
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--nt", type=int, default=NT)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    a = ap.parse_args()
    import numpy as np
    ids = prompt_ids(a.nt)
    if (a.nt != NT or a.dtype != "bf16") and a.ref == os.path.join(HERE, "hf_matrix_long.npy"):
        a.ref = os.path.join(HERE, "hf_matrix_long_%d_%s.npy" % (a.nt, a.dtype))
    if a.hf_only or not os.path.exists(a.ref):
        import torch
        dump_hf(ids, a.ref, getattr(torch, {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[a.dtype]))
        if a.hf_only:
            return 0
    hf = np.load(a.ref)
    NL, Lp = hf.shape
    out = run_remote("cd %s && timeout 1800 %s./%s %s --steps %d --prompt %s --trace 2>&1"
                     % (DIR, ENVQ, BIN, BLOB, len(ids), ",".join(map(str, ids))))
    eng, typ = {}, {}
    for l in out.splitlines():
        m = re.match(r"POS\s+(\d+)\s+L(\d+)\s+(\S+)\s+rms\s+([-\d.]+)", l)
        if m:
            eng[(int(m.group(1)), int(m.group(2)))] = float(m.group(4))
            typ[(int(m.group(1)), int(m.group(2)))] = m.group(3)
    print("parsed %d cells (need %d)" % (len(eng), NL * Lp))
    bad, worst = [], None
    for p in range(Lp):
        for L in range(NL - 1):           # same convention as gate_matrix
            if (p, L) not in eng:
                continue
            r = eng[(p, L)] / hf[L, p] if hf[L, p] else float("nan")
            if not (1 - a.tol <= r <= 1 + a.tol):
                bad.append((p, L, eng[(p, L)], hf[L, p], r))
    if not bad:
        print("matrix-long: %d layers x %d positions all within %.0f%% of HF" % (NL - 1, Lp, a.tol * 100))
        return 0
    import collections
    bypos = collections.Counter(p // 50 for p, _, _, _, _ in bad)
    byd = collections.Counter()
    for p, L, e, h, r in bad:
        import json
        byd["L%d" % L] += 1
    bytype = collections.Counter(typ.get((p, L), "?") for p, L, _, _, _ in bad)
    loside = sum(1 for _, _, _, _, r in bad if r < 1.0)
    print("by layer type:", dict(bytype), " engine-low fraction: %.2f" % (loside / max(1, len(bad))))
    print("by position block:", dict(sorted(bypos.items())))
    print("worst layer counts:", dict(sorted(byd.items(), key=lambda kv: -kv[1])[:8]))
    import json
    # full per-layer table: fails, first-fail pos, and mean ratio over ALL pos
    import collections
    perL = collections.defaultdict(list)
    for p in range(Lp):
        for L in range(NL - 1):
            if (p, L) in eng and hf[L, p]:
                perL[L].append((p, eng[(p, L)] / hf[L, p]))
    print("layer fails firstpos meanratio:")
    for L in range(NL - 1):
        v = perL[L]
        f = [(p, r) for p, r in v if abs(1 - r) > a.tol]
        m = sum(r for _, r in v) / len(v)
        print("  L%-2d fails=%3d first=%s meanratio=%.5f" % (L, len(f), (f[0][0] if f else "-"), m))
    json.dump([[p, L, r] for p, L, _, _, r in bad],
              open(os.path.join(HERE, "matrix_long_bad.json"), "w"))
    # full ratio curves for the four most-missed layers (all positions, not just failures)
    import collections
    curveL = collections.Counter(L for _, L, _, _, _ in bad).most_common(4)
    allr = {}
    for p in range(Lp):
        for L in range(NL - 1):
            if (p, L) in eng and hf[L, p]:
                allr.setdefault(L, []).append(eng[(p, L)] / hf[L, p])
    for L, _ in curveL:
        v = allr[L]
        print("layer %d ratio curve (mean per 30-pos block): %s" % (
            L, [round(sum(v[i:i+30])/len(v[i:i+30]), 4) for i in range(0, 300, 30)]))
    # shape analysis: mean deficit by position decile, and worst layer's curve
    import collections
    dec = collections.defaultdict(list)
    for p_, L_, _, _, r in bad:
        dec[p_ // 30].append(1 - r)
    print("mean deficit by pos-decile:", {k: round(sum(v)/len(v), 4) for k, v in sorted(dec.items())})
    wl = collections.Counter(L for _, L, _, _, _ in bad).most_common(1)[0][0]
    curve = sorted((p, round(1 - r, 4)) for p, L, _, _, r in bad if L == wl)
    print("worst layer L%d deficit curve (pos,def):" % wl, curve[:40])
    bad.sort(key=lambda t: abs(1 - t[4]), reverse=True)
    print("matrix-long: %d/%d cells outside %.0f%% -- worst 5:" % (len(bad), NL * Lp, a.tol * 100))
    for p, L, e, h, r in bad[:5]:
        print("  pos %d layer %d: engine %.5f vs HF %.5f (ratio %.4f)" % (p, L, e, h, r))
    return 1


if __name__ == "__main__":
    sys.exit(main())
