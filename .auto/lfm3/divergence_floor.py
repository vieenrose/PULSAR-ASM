#!/usr/bin/env python
"""The generic divergence floor: how fast does a KNOWN perturbation diverge
off-manifold?

traj_drift.py established that the Nano engine is faithful on-manifold (KL 0.0012,
top-1 at 273/275 of HF's own states, zero slope) and diverges on its own greedy
trajectory at ~+0.15 nats/token. That number has no denominator on its own:
off-manifold KL growth is what happens to ANY decoder that is not bit-identical
to the reference, so +0.15 is only interpretable against a perturbation of known
size.

This measures that reference. Take HF in fp32 as the reference, let IT decode
greedy (so the trajectory is on-manifold for the reference by construction),
then score two perturbed models on exactly those states:

  bf16 HF     -- the pure quantization floor, same weights rounded to bf16
  the engine  -- Q8 linears + fp32 scales + Q8 embedding, ggml summation order

Reading:
  engine slope ~= bf16 slope   -> the engine's off-manifold divergence is the
                                  generic quantization effect; +0.15 nats/token
                                  is the floor, not a defect of this engine.
  engine slope >> bf16 slope   -> the engine carries error beyond quantization
                                  (a GEMV/kernel problem), and the ratio says
                                  how much.

Everything is measured on the SAME token sequence, so the three models see
identical states and the comparison is apples-to-apples.

  python .auto/lfm3/divergence_floor.py --prompt A --steps 48
"""
import argparse, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from checks_lfm3 import ids_of                      # noqa: E402
from traj_drift import engine_traj                  # noqa: E402

REPO = "LiquidAI/LFM3-Audio-2B"


def softmax(v):
    import numpy as np
    m = v.max(-1, keepdims=True)
    e = np.exp(v - m)
    return e / e.sum(-1, keepdims=True)


def load(dtype):
    import torch
    from transformers import AutoModelForMultimodalLM
    m = AutoModelForMultimodalLM.from_pretrained(
        REPO, dtype=dtype, trust_remote_code=True).to("cuda").eval()
    return m


def logits_at(model, seq, n):
    """Logits at seq[:t+1] for t in 0..n-1 (the state the reference stood in)."""
    import numpy as np
    import torch
    rows = []
    for t in range(n):
        with torch.no_grad():
            o = model(input_ids=torch.tensor([seq[:t + 1]], device="cuda")).logits[0, -1]
        rows.append(o.float().cpu().numpy())
    return np.stack(rows).astype(np.float32)


def greedy(model, ids, steps):
    import torch
    with torch.no_grad():
        out = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=steps,
                             do_sample=False)
    seq = out.sequences[0] if hasattr(out, "sequences") else out[0]
    return seq.tolist()


def kl_curve(ref, pert):
    import numpy as np
    n = ref.shape[0]
    kls = [float(np.sum(softmax(ref[t]) * (np.log(softmax(ref[t]) + 1e-30)
                                          - np.log(softmax(pert[t]) + 1e-30))))
           for t in range(n)]
    third = max(1, n // 3)
    slope = float(np.polyfit(np.arange(n), np.array(kls), 1)[0]) if n > 3 else 0.0
    agree = sum(int(ref[t].argmax() == pert[t].argmax()) for t in range(n))
    return (sum(kls) / n, sum(kls[:third]) / third, sum(kls[-third:]) / third, slope, agree, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="A")
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--skip-engine", action="store_true")
    ap.add_argument("--ref-dtype", default="float16",
                    help="reference precision. fp32 was the first choice and does not fit: the "
                         "host's two GPUs have 1.9 GiB and 1.3 GiB free and the fp32 model needs "
                         "5.7 GiB. float16 has a 10-bit mantissa vs bf16's 7, so it is the "
                         "strictly more precise reference that fits, and the bf16 arm is then a "
                         "quantization perturbation of KNOWN size -- which is the denominator "
                         "this whole experiment exists to produce.")
    a = ap.parse_args()
    import numpy as np
    import torch

    ids = ids_of(a.prompt)["prompt_ids"]
    refdt = getattr(torch, a.ref_dtype)
    print("== divergence floor: prompt %s, %d steps, reference = HF %s =="
          % (a.prompt, a.steps, a.ref_dtype))

    ref32 = load(refdt)
    traj = greedy(ref32, ids, a.steps)                  # the reference's own trajectory
    n = min(a.steps, len(traj))
    R = logits_at(ref32, traj, n)
    del ref32
    torch.cuda.empty_cache()
    print("  reference trajectory (%d tokens): %s" % (len(traj) - len(ids), traj[len(ids):][:16]))
    print("  (all arms are scored on THIS token sequence, so they see identical states)")

    for name, fn in [("bf16 HF (quantization floor)", lambda: logits_at(load(torch.bfloat16), traj, n)),
                     ("engine Q8 (same states)", None)]:
        if fn is None:
            if a.skip_engine:
                continue
            toks, eng = engine_traj(a.prompt, len(traj), feed=traj)   # teacher-forced on fp32's tokens
            if eng is None:
                print("  %s: NO DUMP (skipped)" % name)
                continue
            P = eng[:n]
        else:
            P = fn()
        mkl, early, late, slope, agree, nn = kl_curve(R, P)
        print("  %-30s KL mean %.4f (early %.4f -> late %.4f)  slope %+.5f/tok  top-1 %d/%d"
              % (name, mkl, early, late, slope, agree, nn))
        print("  %-30s median-KL %.4f  p90 %.4f" % ("", float(np.median([
            float(np.sum(softmax(R[t]) * (np.log(softmax(R[t]) + 1e-30)
                                          - np.log(softmax(P[t]) + 1e-30))))
            for t in range(nn)])), float(np.percentile([
            float(np.sum(softmax(R[t]) * (np.log(softmax(R[t]) + 1e-30)
                                          - np.log(softmax(P[t]) + 1e-30))))
            for t in range(nn)], 90))))


if __name__ == "__main__":
    sys.exit(main())
