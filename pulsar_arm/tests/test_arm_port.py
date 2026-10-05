#!/usr/bin/env python3
"""Determinism + module-level agreement: the port's correctness contract.

Full-pass greedy parity with HF is unachievable on this checkpoint by ANY
independent implementation (see doc/rpi4-bringup.md: layer 17 amplifies
last-bit rounding differences by ~3e11, verified inside HF's own code).
So the gates that actually bind the speed loop are:

1. determinism: same input twice -> bit-identical logits (required so
   optimizations are verifiably behavior-preserving);
2. module-level agreement: HF's own layer fed our exact input matches our
   output (proves faithful math, independent of rounding chaos);
3. kernel checks (test_arm_kernels.py) stay green.
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
    repo = os.path.dirname(snap)
    from transformers import AutoModelForCausalLM, DynamicCache  # noqa: E402
    hf = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32,
                                              device_map="cpu")
    hf.eval()

    arm = Arm270m(snap, max_seq=64, verbose=False)
    for t in (2, 107):
        arm.forward(t)
    a = arm.forward(1567)
    lg1 = arm.LG.copy()
    # reset and replay identically
    arm2 = Arm270m(snap, max_seq=64, verbose=False)
    for t in (2, 107, 1567):
        arm2.forward(t)
    lg2 = arm2.LG.copy()
    case("determinism: replay bit-identical", np.array_equal(lg1, lg2),
         f"max|d|={float(np.abs(lg1 - lg2).max()):.2e}")

    # module-level: HF layer 17 on our exact L16 output matches our L17 output
    arm.tape = []
    arm.forward(1567)
    with torch.no_grad():
        L = hf.model.layers[17]
        cos, sin = hf.model.rotary_emb(torch.empty(1, 1, 640),
                                       torch.tensor([[arm.pos - 1]]),
                                       layer_type="full_attention")
        y = L(hidden_states=torch.from_numpy(arm.tape[16]).unsqueeze(0).unsqueeze(0),
              position_embeddings=(cos, sin), attention_mask=None,
              past_key_values=DynamicCache())
        y = np.asarray(y[0, 0] if not isinstance(y, tuple) else y[0][0, 0])
    d = float(np.abs(y - arm.tape[17]).max())
    case("module L17 agreement", d < 0.05, f"max|d|={d:.4f} (values O(1e4))")
    print("ALL PORT CHECKS PASS" if not falls else f"FAILURES: {falls}")
    return 1 if falls else 0


if __name__ == "__main__":
    sys.exit(main())
