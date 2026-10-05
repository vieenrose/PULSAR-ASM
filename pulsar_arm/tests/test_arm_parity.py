#!/usr/bin/env python3
"""ARM port vs transformers: greedy parity on gemma-3-270m-it.

Loads the same safetensors through both paths and compares an 8-token greedy
rollout token-for-token, plus final-step top-100 logit agreement. Runs on the
Pi (needs torch + the HF cache); the x86 box only holds the source.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.arm_model import Arm270m  # noqa: E402

CACHE = os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-3-270m-it-qat-q4_0-unquantized")
falls = []


def case(name, ok, detail=""):
    print(f"  {'ok' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        falls.append(name)


def main():
    snap = None
    for root, dirs, files in os.walk(os.path.join(CACHE, "snapshots")):
        for f in files:
            if f == "model.safetensors":
                snap = os.path.join(root, f)
    assert snap, "weights not found"
    repo = os.path.dirname(snap)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tk = AutoTokenizer.from_pretrained(repo, trust_remote_code=False)
    hf = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16,
                                              device_map="cpu")
    hf.eval()

    arm = Arm270m(snap, max_seq=128, verbose=False)
    ids = tk("The capital of France is", return_tensors="pt")["input_ids"][0].tolist()

    # prefill both
    with torch.no_grad():
        hx = torch.tensor([ids], dtype=torch.long)
        out = hf(input_ids=hx, use_cache=False)
    for t in ids:
        arm.forward(t)
    href = np.asarray(out.logits[0, -1].float())
    arow = np.asarray(arm.LG)
    top = np.argsort(-href)[:100]
    d = float(np.max(np.abs(arow - href)[top]))
    case("prefill top-100 logits", d < 0.5, f"max|d|={d:.4f}")

    # --- rollout with KV carried on the HF side ---
    from transformers import DynamicCache
    past = DynamicCache()
    with torch.no_grad():
        out = hf(input_ids=torch.tensor([ids]), use_cache=True, past_key_values=past)
    ht = int(out.logits[0, -1].argmax())
    at = int(np.argmax(arm.LG))
    case("first token equal", ht == at, f"hf={ht} arm={at}")
    ok = (ht == at)
    for s in range(7):
        arm.forward(at)
        at = int(np.argmax(arm.LG))
        with torch.no_grad():
            out = hf(input_ids=torch.tensor([[ht]]), use_cache=True, past_key_values=past)
        ht = int(out.logits[0, -1].argmax())
        if ht != at:
            ok = False
            print(f"    step {s}: hf={ht} arm={at}")
    case("8-token rollout identical", ok)
    print("ALL ARM CHECKS PASS" if not falls else f"FAILURES: {falls}")
    return 1 if falls else 0


if __name__ == "__main__":
    sys.exit(main())
