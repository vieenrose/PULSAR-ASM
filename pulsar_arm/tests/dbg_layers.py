#!/usr/bin/env python3
"""Bisect the 38-logit gap: embed, layer-0, layer-5, final norm vs HF."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.arm_model import Arm270m  # noqa: E402

CACHE = os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-3-270m-it-qat-q4_0-unquantized")
snap = None
for root, dirs, files in os.walk(os.path.join(CACHE, "snapshots")):
    for f in files:
        if f == "model.safetensors":
            snap = os.path.join(root, f)
repo = os.path.dirname(snap)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
tk = AutoTokenizer.from_pretrained(repo, trust_remote_code=False)
hf = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32, device_map="cpu")
hf.eval()

ids = [2, 107, 1567]
with torch.no_grad():
    emb_hf = hf.model.embed_tokens(torch.tensor([ids])).float().numpy()[0]
    out = hf.model(input_ids=torch.tensor([ids]), use_cache=False,
                   output_hidden_states=True)

arm = Arm270m(snap, max_seq=64, verbose=False)
for t in ids[:-1]:
    arm.forward(t)
arm.tape = []
arm.forward(ids[-1])

print("prompt ids:", ids, flush=True)
# embed check: fresh single-token embed through the C kernel
lib = arm.lib
x = np.zeros(640, dtype=np.float32)
lib.embed_row_f32(x.ctypes.data, arm.emb_ptr, ids[0], 640, 640 ** 0.5)
print(f"embed max|d|={float(np.abs(x - emb_hf[0]).max()):.4f} "
      f"(zero here means scale is right)", flush=True)
hh = [np.asarray(h.float()) for h in out.hidden_states]
print(f"HF hidden_states: {len(hh)} (0=input emb)", flush=True)
for i, (a, b) in enumerate(zip(arm.tape, hh[1:])):
    d = float(np.abs(a - b).max())
    print(f"  layer {i:2d} max|d|={d:.4f}", flush=True)
    if i >= 6:
        print("  ... (truncated)", flush=True)
        break
