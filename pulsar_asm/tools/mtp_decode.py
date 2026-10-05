#!/usr/bin/env python3
"""Greedy MTP decode: assembly drafter + batched target verify, proven equal.

Each round costs two target passes no matter how many drafts hit: one single
forward for the current token (which also produces the fresh h the assistant
is seeded from, exactly like the NumPy oracle in bench_mtp.py), then ONE
batched pass over k drafts. The emitted sequence is target argmaxes by
construction - every kept token either equaled one or was one - so comparing
against a plain greedy rollout guards the plumbing (positions, caches, h),
which is the only thing that can be wrong here.

  python tools/mtp_decode.py --k 4 --tokens 24
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.gemma4_model import Gemma4      # noqa: E402
from runtime.mtp_model import AssistAsm      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="/mnt/edge/pulsar/gemma4_e2b.bin")
ap.add_argument("--assist", default="/mnt/edge/pulsar/gemma4_e2b_mtp.bin")
ap.add_argument("--tok", default="/mnt/edge/pulsar/tok")
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--tokens", type=int, default=24)
ap.add_argument("--prompt", default="Explain why the sky is blue, in two sentences.")
a = ap.parse_args()

from transformers import AutoTokenizer  # noqa: E402
tk = AutoTokenizer.from_pretrained(a.tok)
prompt = list(tk.apply_chat_template([{"role": "user", "content": a.prompt}],
                                     add_generation_prompt=True, tokenize=True,
                                     return_dict=False))


def plain_greedy(n):
    g = Gemma4(a.weights, max_seq=1024, n_threads=4, verbose=False)
    g.set_sampling(temp=0)
    for t in prompt:
        g.forward(t)
    cur = int(g.logits().argmax())
    out = []
    for _ in range(n):
        out.append(cur)
        cur = int(g.forward(cur))
    g.close()
    return out


def mtp_greedy(n, k):
    g = Gemma4(a.weights, max_seq=1024, n_threads=4, verbose=False)
    g.set_sampling(temp=0)
    A = AssistAsm(a.assist, g, verbose=False)
    for t in prompt:
        g.forward(t)
    cur = int(g.logits().argmax())
    out, accepted, rounds = [cur], 0, 0
    t_tgt = t_dft = 0.0
    h = np.zeros(1536, dtype=np.float32)
    hn = np.zeros(1536, dtype=np.float32)
    while len(out) < n:
        p = g.pos
        t0 = time.perf_counter()
        tgt = int(g.forward(cur))           # cur evaluated at p; h is fresh
        t_tgt += time.perf_counter() - t0
        h[:] = np.asarray(g.buf["X"][:1536])
        t0 = time.perf_counter()
        drafts, prev, hh = [], cur, h
        for _ in range(k):
            d = A.step(prev, hh, hn, p)
            drafts.append(d)
            prev, hh = d, hn.copy()
        t_dft += time.perf_counter() - t0
        t0 = time.perf_counter()
        # one batched verify pass over the k drafts at p+1..p+k
        g.run(drafts, pos=p + 1)
        t_tgt += time.perf_counter() - t0
        # row j evaluates d_{j+1} at p+1+j, so it predicts p+2+j.
        Brow = [int(np.argmax(g.logits(row=j))) for j in range(k)]
        want = [tgt] + Brow[:-1]
        nok = 0
        for d, w in zip(drafts, want):
            if d != w:
                break
            nok += 1
        accepted += nok
        rounds += 1
        chain = drafts[:nok]
        bonus = want[nok] if nok < k else Brow[-1]
        out.extend(chain + [bonus])
        # caches past p+nok are speculative rows; later passes overwrite them.
        # the bonus was never evaluated, so it becomes next round's cur.
        cur = bonus
        g.pos = p + nok + 1
    g.close()
    return out[:n], accepted / max(rounds, 1), t_tgt, t_dft, rounds


ref = plain_greedy(a.tokens)
got, acc, t_tgt, t_dft, rounds = mtp_greedy(a.tokens, a.k)
print(f"rounds={rounds} k={a.k} accept/pass={acc:.2f} "
      f"target={t_tgt/rounds*1000:.0f}ms/round drafter={t_dft/(rounds*a.k)*1000:.2f}ms/draft")
print("ref :", ref)
print("mtp :", got)
print("IDENTICAL" if ref == got else "MISMATCH")
sys.exit(0 if ref == got else 1)
