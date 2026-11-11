#!/usr/bin/env python
"""Loss of low-bit backbone weight formats, measured BEFORE any kernel exists.

Fake-quantizes the backbone linears (model.language_model.layers.*: conv in/out_proj, attention
q/k/v/o, ffn w1/w2/w3) of the HF model from the ORIGINAL bf16 weights, runs the model in
fp16, and scores it on the fp16 reference's own greedy trajectory -- the
divergence_floor.py protocol, so the numbers sit next to the engine's (Q8: KL
0.0013 / 0.0005 / 0.0006 on A / C / K, top-1 47/48).

Embeddings / lm_head (tied) and everything outside the backbone layers stay as they are.

  python .auto/lfm3/quant_loss.py --prompts A,C,K --arms q8,q5_0,q5_1,q4_0,q4_1,q4_1o
"""
import argparse, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from checks_lfm3 import ids_of                      # noqa: E402
from divergence_floor import load, logits_at, greedy, kl_curve  # noqa: E402


def fq(W, arm, g=32):
    """[N, K] fp32 -> dequantized fp32 under format `arm` (groups of g along K).
    Scales/mins are rounded to fp16 (what a 4/5-bit blob would store), except q8,
    which mirrors the shipped engine (fp32 scale, amax/127)."""
    import torch
    N, K = W.shape
    x = W.reshape(N, K // g, g)
    base = arm.rstrip("o")
    opt = arm.endswith("o")
    h = lambda t: t.half().float()
    if base == "q8":
        d = x.abs().amax(-1, keepdim=True) / 127.0
        d = torch.where(d == 0, torch.ones_like(d), d)
        return (torch.round(x / d).clamp(-127, 127) * d).reshape(N, K)
    bits = int(base[1]); L = 1 << bits
    if base.endswith("_0"):                          # symmetric, ggml Q4_0 / Q5_0 style
        idx = x.abs().argmax(-1, keepdim=True)
        m = torch.gather(x, -1, idx)                 # signed value of max magnitude
        d = h(m / -(L // 2))
        d = torch.where(d == 0, torch.ones_like(d), d)
        def deq(d):
            return torch.round(x / d).clamp(-(L // 2), L // 2 - 1) * d
        if opt:                                       # search the scale that minimizes SSE
            best = deq(d); be = ((best - x) ** 2).sum(-1, keepdim=True)
            for f in torch.linspace(0.80, 1.10, 31):
                dd = h(d * f); dd = torch.where(dd == 0, torch.ones_like(dd), dd)
                y = deq(dd); e = ((y - x) ** 2).sum(-1, keepdim=True)
                sel = e < be; best = torch.where(sel, y, best); be = torch.where(sel, e, be)
            return best.reshape(N, K)
        return deq(d).reshape(N, K)
    # asymmetric: w = d * q + mn, q in [0, L-1] (ggml Q4_1 / Q5_1 style)
    mn = x.amin(-1, keepdim=True); mx = x.amax(-1, keepdim=True)
    d = (mx - mn) / (L - 1); d = torch.where(d == 0, torch.ones_like(d), d)
    d, mn = h(d), h(mn)
    q = torch.round((x - mn) / d).clamp(0, L - 1)
    if opt:                                           # alternate: q | (d, mn) least squares
        for _ in range(4):
            qm = q.mean(-1, keepdim=True); xm = x.mean(-1, keepdim=True)
            var = ((q - qm) ** 2).sum(-1, keepdim=True)
            dn = ((q - qm) * (x - xm)).sum(-1, keepdim=True) / torch.where(var == 0, torch.ones_like(var), var)
            dn = torch.where(var == 0, d, dn)
            mnn = xm - dn * qm
            dn, mnn = h(dn), h(mnn)
            dn = torch.where(dn == 0, torch.ones_like(dn), dn)
            qn = torch.round((x - mnn) / dn).clamp(0, L - 1)
            eo = ((q * d + mn - x) ** 2).sum(-1, keepdim=True)
            en = ((qn * dn + mnn - x) ** 2).sum(-1, keepdim=True)
            sel = en < eo
            d = torch.where(sel, dn, d); mn = torch.where(sel, mnn, mn); q = torch.where(sel, qn, q)
    return (q * d + mn).reshape(N, K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="A,C,K")
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--arms", default="q8,q5_0,q5_1,q4_0,q4_1")
    ap.add_argument("--speech-eval", default="", help="held-out speech turns: rel-L2 of the hidden fed to the depth head")
    ap.add_argument("--layer-sweep", default="", help="arm: quantize ONE layer at a time with it, report speech rel-L2")
    ap.add_argument("--load", default="", help="score a gptq_lfm3.py result instead of --arms")
    ap.add_argument("--only", default="", help="comma list: quantize only names containing one (others stay fp16)")
    ap.add_argument("--keep", default="", help="comma list of name substrings kept at q8 (mixed)")
    a = ap.parse_args()
    import numpy as np
    import torch

    model = load(torch.float16)
    lin = [(n, m) for n, m in model.named_modules()
           if isinstance(m, torch.nn.Linear) and n.startswith("model.language_model.layers.")
           and m.weight.shape[1] % 32 == 0]
    print("backbone linears: %d (%.1f M weights)" % (len(lin), sum(m.weight.numel() for _, m in lin) / 1e6))
    orig = {n: m.weight.detach().float().cpu() for n, m in lin}   # the bf16 values, exact in fp32

    refs = {}
    for p in a.prompts.split(","):
        ids = ids_of(p)["prompt_ids"]
        traj = greedy(model, ids, a.steps)
        n = min(a.steps, len(traj))
        refs[p] = (traj, n, logits_at(model, traj, n))

    keep = [k for k in a.keep.split(",") if k]
    only = [k for k in a.only.split(",") if k]
    sp = torch.load(a.speech_eval) if a.speech_eval else []
    aid = model.config.audio_out_token_id
    def hid(seq, codes):
        with torch.no_grad():
            o = model(input_ids=torch.tensor([seq], device="cuda"), audio_out_codes=codes.cuda(), output_hidden_states=True)
        h = o.hidden_states[-1][0].float()
        pos = [p for p in range(len(seq) - 1) if seq[p + 1] == aid]
        return h[pos]
    sref = [hid(s_, c_) for s_, c_ in sp]
    def speech_err():
        e = [((hid(s_, c_) - r).norm(dim=-1) / r.norm(dim=-1)) for (s_, c_), r in zip(sp, sref)]
        e = torch.cat(e); return float(e.mean()), float(e.max())
    if a.layer_sweep:
        for L in range(24):
            with torch.no_grad():
                for n, m in lin:
                    W = orig[n].cuda()
                    q = n.startswith("model.language_model.layers.%d." % L) and not any(k in n for k in keep)
                    m.weight.copy_((fq(W, a.layer_sweep) if q else W).half())
            print("layer %2d alone at %s: speech rel-L2 mean %.4f max %.4f" % ((L, a.layer_sweep) + speech_err()), flush=True)
        return
    arms = a.arms.split(",")
    if a.load:
        res = torch.load(a.load)
        arms = ["gptq:" + os.path.basename(a.load)]
    for arm in arms:
        with torch.no_grad():
          if a.load:
            for n, m in lin:
                b, pk = res[n]
                if pk is None: m.weight.copy_(fq(orig[n].cuda(), "q8").half()); continue
                Q, d, mn = (t.cuda() for t in pk)
                N, K = Q.shape
                m.weight.copy_((Q.float().reshape(N, -1, 32) * d.float().unsqueeze(-1) + mn.float().unsqueeze(-1)).reshape(N, K).half())
          else:
            for n, m in lin:
                W = orig[n].cuda()
                use = "q8" if any(k in n for k in keep) else arm
                if only and not any(k in n for k in only): m.weight.copy_(W.half()); continue
                m.weight.copy_(fq(W, use).half())
        line = []
        for p, (traj, n, R) in refs.items():
            mkl, early, late, slope, agree, nn = kl_curve(R, logits_at(model, traj, n))
            line.append("%s KL %.4f top1 %d/%d" % (p, mkl, agree, nn))
        if sp: line.append("speech rel-L2 mean %.4f max %.4f" % speech_err())
        print("%-8s %s%s" % (arm, " | ".join(line), ("  (q8 kept: %s)" % a.keep) if keep else ("  (only: %s)" % a.only if only else "")), flush=True)


if __name__ == "__main__":
    sys.exit(main())
