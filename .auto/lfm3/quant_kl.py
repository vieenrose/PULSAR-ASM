#!/usr/bin/env python
"""quant_kl.py -- distribution-level quality of backbone weight quantization schemes.

quant_sensitivity.py scored schemes by exact greedy-prefix match against bf16 goldens.
That metric is cliff-like: one near-tie flip ends the prefix, and even the shipped Q8
engine misses 1 of 7 streams that way (README gate history). This script measures what a
lossy-but-acceptable scheme actually needs to preserve -- the next-token distribution --
on a fixed set of states, so every scheme sees identical inputs:

  reference = the checkpoint in fp32 (its bf16 weights upcast exactly)
  sequences = prompt_ids + golden continuation of each golden_greedy_<id>.json
  per scheme: KL(ref || scheme) and top-1 agreement at every continuation position
              (teacher-forced: one forward per sequence, states never diverge)

Dequantized weights stay fp32 (the engine dequantizes into fp32 too), unlike the
bf16-model arms of quant_sensitivity.py, which re-rounded them to bf16.

A scheme is a list of (group, bits, group_size, asym) applied to the ORIGINAL weights;
tensors in no listed group stay at checkpoint precision. 'emb' is the tied embedding.

Usage: ~/venvs/lfm3/bin/python .auto/lfm3/quant_kl.py [--schemes a,b,...] [--prompts A,C,...]
"""
import argparse, json, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from quant_sensitivity import fq, GROUPS, BACKBONE, EMB, build_ids   # same quantizer, groups, prompt ids

MODEL = "LiquidAI/LFM3-Audio-2B"   # HF cache id, as the oracle scripts load it
LIN = "all"   # every 2-D linear of the backbone (ffn + attn + conv)

# name -> list of (group, bits, group_size, asym); group in GROUPS or "emb"
SCHEMES = {
    "q8_0_embq8": [("all", 8, 32, False), ("emb", 8, 32, False)],   # the shipped Nano arm
    "q6_0":       [("all", 6, 32, False), ("emb", 8, 32, False)],
    "q5_0":       [("all", 5, 32, False), ("emb", 8, 32, False)],
    "q5_1":       [("all", 5, 32, True),  ("emb", 8, 32, False)],
    "q4_1":       [("all", 4, 32, True),  ("emb", 8, 32, False)],
    "q4_0":       [("all", 4, 32, False), ("emb", 8, 32, False)],
    "q4_0_g16":   [("all", 4, 16, False), ("emb", 8, 32, False)],
    "ffn4_1":     [("attn", 8, 32, False), ("conv", 8, 32, False), ("ffn", 4, 32, True), ("emb", 8, 32, False)],
    "ffn5_0":     [("attn", 8, 32, False), ("conv", 8, 32, False), ("ffn", 5, 32, False), ("emb", 8, 32, False)],
    "q4_1_emb4":  [("all", 4, 32, True),  ("emb", 4, 32, True)],
}


def bytes_per_w(bits, gs, asym):
    return (bits + 16.0 / gs + (16.0 / gs if asym else 0)) / 8   # codes + f16 scale (+ f16 min)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", default="q8_0_embq8,q6_0,q5_1,q5_0,ffn5_0,q4_1,ffn4_1,q4_0_g16,q4_0")
    ap.add_argument("--prompts", default="A,B,C,D,E,F,K,L")
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForMultimodalLM.from_pretrained(MODEL, trust_remote_code=True,
                                                      dtype=torch.float32).eval()
    params = {n: p for n, p in model.named_parameters()
              if p.dim() == 2 and n.startswith(BACKBONE)}
    orig = {n: p.data.clone() for n, p in params.items()}      # CPU copy of the checkpoint
    model.to(dev)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    def members(group):
        if group == "emb":
            return [EMB]
        return [n for n in params if n != EMB and any(g in n for g in GROUPS[group])]

    seqs = []
    for k in a.prompts.split(","):
        g = json.load(open(os.path.join(HERE, "golden_greedy_%s.json" % k)))
        pid = build_ids(proc, g["prompt"])[0].tolist()
        if "prompt_ids" in g:
            assert pid == g["prompt_ids"], "prompt ids differ from the golden's for %s" % k
        seqs.append((k, torch.tensor([pid + g["gen"]]).to(dev), len(pid)))

    def run():
        out = []
        with torch.no_grad():
            for k, ids, npr in seqs:
                lg = model(input_ids=ids).logits[0].float()
                out.append(lg[npr - 1:-1].cpu())   # predictions of the continuation tokens
        return out

    t0 = time.time()
    ref = run()
    ref_lp = [torch.log_softmax(r.double(), -1) for r in ref]
    print("reference fp32: %d sequences, %d positions, %.0f s" %
          (len(seqs), sum(r.shape[0] for r in ref), time.time() - t0), flush=True)
    nparam = {n: p.numel() for n, p in params.items()}
    print("%-12s %7s %9s %9s %9s %8s   per-prompt mean KL" %
          ("scheme", "GB", "meanKL", "p95KL", "maxKL", "top1"), flush=True)
    for s in a.schemes.split(","):
        spec = SCHEMES[s]
        for n, p in params.items():
            p.data.copy_(orig[n].to(dev))
        nbytes, seen = 0.0, set()
        for group, bits, gs, asym in spec:
            ms = members(group)
            assert ms, "group %s matched no tensors" % group
            for n in ms:
                params[n].data.copy_(fq(orig[n].to(dev), bits, gs, asym))
                nbytes += nparam[n] * bytes_per_w(bits, gs, asym)
                seen.add(n)
        nbytes += sum(nparam[n] * 2 for n in params if n not in seen)   # untouched: bf16
        t0 = time.time()
        got = run()
        kls, agree, per = [], 0, []
        for lp, g in zip(ref_lp, got):
            lq = torch.log_softmax(g.double(), -1)
            kl = (lp.exp() * (lp - lq)).sum(-1)
            kls.append(kl)
            per.append(kl.mean().item())
            agree += (lp.argmax(-1) == lq.argmax(-1)).sum().item()
        kl = torch.cat(kls)
        n = kl.numel()
        print("%-12s %7.3f %9.5f %9.5f %9.4f %4d/%-4d  %s  (%.0f s)" %
              (s, nbytes / 1e9, kl.mean().item(), kl.quantile(0.95).item(), kl.max().item(),
               agree, n, " ".join("%.4f" % x for x in per), time.time() - t0), flush=True)
        with open(os.path.join(HERE, "quant_kl.jsonl"), "a") as f:
            f.write(json.dumps({"scheme": s, "spec": spec, "gb": nbytes / 1e9,
                                "mean_kl": kl.mean().item(), "p95_kl": kl.quantile(0.95).item(),
                                "max_kl": kl.max().item(), "top1": agree, "n": n,
                                "prompts": a.prompts, "per_prompt_kl": per}) + "\n")


if __name__ == "__main__":
    main()
