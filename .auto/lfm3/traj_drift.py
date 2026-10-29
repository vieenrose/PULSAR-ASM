#!/usr/bin/env python
"""Free-running trajectory gate: does the engine's distribution agree with HF's
at the states the ENGINE actually visits?

The existing logit gate (checks_lfm3.gate_logits_seq) is teacher forced: HF and
the engine both see ids[:p+1]. That is the right gate for "is the math right",
and it is strong (top-1 equals HF's on 130/131 positions, KL < 2e-3). It cannot
see the thing the streams report: the engine's greedy stream diverges from HF's
after 1-2 tokens on every prompt, and teacher forcing hides precisely that,
because every position is re-anchored on HF's tokens.

This gate drops the anchor. Run the engine free-running, take ITS consumed
token sequence, and ask HF for logits on that state. Two shapes to look for:

  KL(t) flat          the engine stays in-distribution on its own trajectory;
                      stream divergence is argmax/tie amplification, and the
                      "0/7 full matches" line means what the logit gate says.
  KL(t) growing       compounding drift: every self-sampled token pushes the
                      state away from where the engine was validated. That is a
                      real quality problem teacher-forced gates structurally
                      cannot detect.

--on-manifold is the control that makes the two separable: feed HF's OWN golden
continuation (teacher-forced, --prompt ids,gen) and evaluate both sides on it.
Any KL there is pure numerics on a state HF itself reaches. Comparing the two
runs localises the failure: tiny on-manifold + growing off-manifold is a
feedback-loop / off-manifold sensitivity; growing on-manifold is the engine.

Alignment is the entire bug surface, so it lives in one place: `seq` is the
token sequence the engine consumed, step t's state is seq[:t+1], and its
logits predict seq[t+1]. In free-running mode seq = prompt_ids + generated
tokens; the tokens the engine sampled while consuming the prompt are NOT
context. Feeding HF toks[:t] instead measures a state that never existed --
that bug reported KL 6.6 nats at t=0 where the teacher-forced gate says 0.0014.

Reference cost: one HF forward per step per prompt. This is not a latency
measurement; use .auto/lfm3/measure.sh for the metric.

  python .auto/lfm3/traj_drift.py                        # off-manifold, all prompts
  python .auto/lfm3/traj_drift.py --on-manifold          # the control
  LFM3_ENV="LFM_NOFMA=1" python .auto/lfm3/traj_drift.py # other engine arm
"""
import argparse, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from checks_lfm3 import NANO, DIR, BLOB, BIN, ENVQ, ids_of, run_remote  # noqa: E402

PROMPTS = ["A", "C", "E", "F", "K", "L"]      # prompts with verified prompt_ids (D is refused)


def engine_traj(tag, steps, feed=None):
    """Run the engine free-running (feed=None) or teacher-forced on `feed`;
    return (sampled_tokens, logits[n,V]) from its own --dump-logits-all."""
    ids = feed if feed is not None else ids_of(tag)["prompt_ids"]
    dlog, dtok = "traj_%s.bin" % tag, "traj_%s.txt" % tag
    run_remote("cd %s && rm -f %s %s" % (DIR, dlog, dtok))
    out = run_remote("cd %s && timeout 1800 %s./%s %s --steps %d --prompt %s "
                     "--dump-logits-all %s > %s 2>&1; ls -l %s 2>&1"
                     % (DIR, ENVQ, BIN, BLOB, steps, ",".join(map(str, ids)), dlog, dtok, dlog))
    if dlog not in out:
        return None, None
    for f in (dlog, dtok):
        subprocess.run(["scp", "-o", "ConnectTimeout=20", "-q", "%s:%s/%s" % (NANO, DIR, f),
                        "/tmp/%s" % f], check=False, timeout=600)
    toks = []
    for l in open("/tmp/%s" % dtok):
        p = l.split()
        if len(p) == 4 and all(x.replace(".", "").replace("-", "").isdigit() for x in p):
            toks.append(int(p[1]))
    import numpy as np
    nrow = os.path.getsize("/tmp/%s" % dlog) // 4
    if not toks or nrow % len(toks):
        return toks, None
    eng = np.fromfile("/tmp/%s" % dlog, dtype=np.float32).reshape(len(toks), nrow // len(toks))
    return toks, eng


def hf_on_trajectory(seq, steps):
    """HF logits on the state the engine stood in at each step: seq[:t+1]."""
    import numpy as np
    import torch
    from transformers import AutoModelForMultimodalLM
    model = AutoModelForMultimodalLM.from_pretrained(
        "LiquidAI/LFM3-Audio-2B", dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    n = min(steps, len(seq))
    rows, top = [], []
    for t in range(n):
        with torch.no_grad():
            o = model(input_ids=torch.tensor([seq[:t + 1]], device="cuda")).logits[0, -1]
        v = o.float().cpu().numpy()
        rows.append(v)
        top.append(int(v.argmax()))
    return np.stack(rows).astype(np.float32), top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=",".join(PROMPTS))
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--on-manifold", action="store_true", help="control: feed the golden gen")
    ap.add_argument("--kl-max", type=float, default=0.05, help="mean KL floor (ratchet)")
    ap.add_argument("--grow-min", type=float, default=0.02,
                    help="absolute late-window KL below which the growth test cannot fire")
    ap.add_argument("--grow", type=float, default=3.0,
                    help="FAIL if mean KL of the last third exceeds this multiple of the first third")
    a = ap.parse_args()
    import numpy as np

    def softmax(v):
        m = v.max(-1, keepdims=True)
        e = np.exp(v - m)
        return e / e.sum(-1, keepdims=True)

    fails, rows = [], []
    mode = "on-manifold (golden fed)" if a.on_manifold else "free-running"
    print("== trajectory drift: %s / %s / %s%s ==" % (BLOB, BIN, mode,
                                                      "  env: " + ENVQ if ENVQ else ""))
    for tag in a.prompts.split(","):
        gids = ids_of(tag)["prompt_ids"]
        if a.on_manifold:
            fed = gids + list(ids_of(tag)["gen"])
            toks, eng = engine_traj(tag, len(fed), feed=fed)
            seq = fed
        else:
            toks, eng = engine_traj(tag, a.steps)
            # toks[0:len(gids)] were sampled while the prompt was consumed and
            # are not context; the generated tail starts at len(gids).
            seq = gids + toks[len(gids):]
        if eng is None:
            fails.append("traj[%s]: engine produced no usable logit dump" % tag)
            print("  traj[%s]: NO DUMP" % tag)
            continue
        hf, hf_top = hf_on_trajectory(seq, a.steps)
        n = hf.shape[0]
        kls, agree = [], 0
        for t in range(n):
            pe, ph = softmax(eng[t]), softmax(hf[t])
            kls.append(float(np.sum(ph * (np.log(ph + 1e-30) - np.log(pe + 1e-30)))))
            agree += int(int(eng[t].argmax()) == hf_top[t])
        third = max(1, n // 3)
        early, late = sum(kls[:third]) / third, sum(kls[-third:]) / third
        mkl = sum(kls) / n
        slope = float(np.polyfit(np.arange(n), np.array(kls), 1)[0]) if n > 3 else 0.0
        print("  traj[%s]: n=%d  KL mean %.4f (early %.4f -> late %.4f, x%.1f, slope %+.5f/tok)  "
              "top-1 vs HF at these states %d/%d"
              % (tag, n, mkl, early, late, (late / early if early > 0 else float("inf")),
                 slope, agree, n))
        rows.append((tag, mkl, early, late, slope, agree, n))
        if mkl > a.kl_max:
            fails.append("traj[%s]: mean KL %.4f > %.4f" % (tag, mkl, a.kl_max))
        # The growth test needs an absolute floor as well as a ratio: on the
        # on-manifold control, prompt K "failed" at x3.3 growth from 0.0005 to
        # 0.0015 -- a 3x ratio of numbers 30x below the mean-KL floor and 3000x
        # below the free-running signal (late KL 4.3-5.2). A ratio test on
        # near-zero values is a false-positive generator.
        if early > 0 and late > a.grow * early and late > a.grow_min:
            fails.append("traj[%s]: KL grows x%.1f over the trajectory (%.4f -> %.4f) -- "
                         "compounding drift on the engine's own states"
                         % (tag, late / early, early, late))
    if rows:
        print("  summary %s: mean KL %.4f, top-1 %d/%d, median slope %+.5f/tok"
              % (mode, sum(r[1] for r in rows) / len(rows),
                 sum(r[5] for r in rows), sum(r[6] for r in rows),
                 sorted(r[4] for r in rows)[len(rows) // 2]))
    if fails:
        print("TRAJ DRIFT FAILED (%d):" % len(fails))
        for f in fails:
            print("  -", f)
        return 1
    print("TRAJ DRIFT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
