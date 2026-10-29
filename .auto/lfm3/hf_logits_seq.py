#!/usr/bin/env python
"""Dump HF's logits at every teacher-forced position of a prompt.

Partner to the engine's --dump-logits-all, which appends the logits of every
decode step. Comparing the two position by position gives a quantitative
quality gate: greedy stream prefixes are 1-2 tokens of 16 (0/7 prompts fully
match), so the stream gate cannot see a quality regression from a change in
summation order -- an FMA switch left all 16 stream tokens unchanged. Position
wise top-1 agreement, top-k coverage, Pearson correlation and KL are
continuous, so they can.

Teacher forcing matches --trace's convention (checks_lfm3.py's matrix gate):
HF sees ids[:p+1] and its logits at the LAST position are position p's.

  ~/venvs/lfm3/bin/python .auto/lfm3/hf_logits_seq.py --prompt A --n 23
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "LiquidAI/LFM3-Audio-2B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="A", help="tag, or comma-separated tags")
    ap.add_argument("--n", type=int, default=23, help="positions to dump (0..n-1)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForMultimodalLM
    model = AutoModelForMultimodalLM.from_pretrained(
        REPO, dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    for tag in a.prompt.split(","):
        g = json.load(open(os.path.join(HERE, "golden_greedy_%s.json" % tag)))
        ids = list(g["prompt_ids"])
        n = min(a.n, len(ids))
        lg = []
        for p in range(n):
            with torch.no_grad():
                o = model(input_ids=torch.tensor([ids[:p + 1]], device="cuda")).logits[0, -1]
            lg.append(o.float().cpu().numpy())
        arr = np.stack(lg).astype(np.float32)
        out = a.out if (a.out and len(a.prompt.split(",")) == 1) else \
            os.path.join(HERE, "hf_logits_%s_seq.npy" % tag)
        np.save(out, arr)
        print("wrote %s: %s (position p = logits after ids[:p+1], teacher forced)" % (out, arr.shape))


if __name__ == "__main__":
    sys.exit(main())
