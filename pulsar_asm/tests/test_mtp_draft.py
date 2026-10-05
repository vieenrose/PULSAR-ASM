#!/usr/bin/env python3
"""Assembly MTP drafter vs the NumPy oracle (tools/ref_gemma4_assist_np.py).

Three levels, cheapest first: the top-32 select (set equality - order is
allowed to differ, the argmax over the set cannot), the row gather
(bit-exact, it is a copy), then full draft steps (exact token, tight hidden).
The NumPy side does attention in float64, so the hidden comparison is a
tolerance, not bits; the TOKEN must be identical or the drafter is wrong.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from runtime.gemma4_model import Gemma4, rope_tables
from runtime.mtp_model import AssistAsm
import ref_gemma4_np as R
from ref_gemma4_assist_np import Assist

BLOB = "/mnt/edge/pulsar/gemma4_e2b.bin"
ASSIST = "/mnt/edge/pulsar/gemma4_e2b_mtp.bin"
TOK = "/mnt/edge/pulsar/tok"

fails = []


def case(name, ok, detail=""):
    print(f"  {'ok' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        fails.append(name)


def main():
    g = Gemma4(BLOB, max_seq=256, n_threads=4, verbose=False)
    ref = R.Ref(BLOB)
    kv = {}
    for t, i in g.man["meta"]["kv_store_layers"].items():
        hd = (g.man["meta"]["head_dim_full"] if t == "full_attention"
              else g.man["meta"]["head_dim_slide"])
        kk, vv = g._caches[i]
        kv[t] = (kk.reshape(g.max_seq, hd), vv.reshape(g.max_seq, hd))
    A = Assist(ASSIST, ref, kv)
    ASM = AssistAsm(ASSIST, g, verbose=False)
    tables = rope_tables(g.man["meta"], g.max_seq)
    cos_s, sin_s, cos_f, sin_f = (t.astype(np.float32) for t in tables)

    # ---- unit: top32 select (set equality) --------------------------------
    rng = np.random.default_rng(7)
    v = rng.standard_normal(2048).astype(np.float32)
    # want FIRST: the asm call mutilates v (picked entries become -inf)
    want = set(int(i) for i in np.argpartition(-v, 31)[:32])
    idx = np.zeros(32, dtype=np.uint32)
    ASM.top32_fn(v.ctypes.data, idx.ctypes.data)
    case("top32 set == argpartition set", set(int(i) for i in idx) == want,
         f"got {sorted(int(i) for i in idx)[:4]}...")

    # ---- unit: gather (bit-exact copy) -------------------------------------
    import ctypes
    import json
    import mmap
    import os
    man = json.load(open(ASSIST + ".manifest.json"))
    lay = man["layout"]
    # expected rows straight from the blob via the reference helper
    cands = A._order().reshape(2048, 128)[np.argsort(-v)[:32]].reshape(-1)
    exp = A.rows("embed", cands)
    top = np.argsort(-v, kind="stable")[:32].astype(np.uint32)
    dst = np.zeros(4096 * 256 * 2, dtype=np.uint8)
    size = os.path.getsize(ASSIST)
    f = open(ASSIST, "rb")
    mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    base = np.frombuffer(mm, dtype=np.uint8).ctypes.data
    ASM.gather_fn(dst.ctypes.data, base + lay["embed"]["offset"],
                    base + lay["tok_order"]["offset"], top.ctypes.data)
    got = ((dst.view(np.uint16).astype(np.uint32)) << 16).view(np.float32).reshape(4096, 256)
    d = np.abs(got - exp)
    case("gather bit-exact", d.max() == 0.0, f"max|d|={d.max():.2e}")

    # ---- full draft steps ----------------------------------------------------
    import run_gemma4_chat as cli
    tk, eos = cli.load_tokenizer(TOK)
    prompt = [105, 2364, 107, 237105, 122100, 95202]
    for t in prompt:
        g.forward(t)
    cur = int(g.logits().argmax())
    worst = 0.0
    alltok = True
    for step in range(4):
        p = g.pos
        h = np.asarray(g.buf["X"][:1536]).astype(np.float32).copy()
        etok, ehh = A.step(cur, h, p, cos_s, sin_s, cos_f, sin_f)
        hn = np.zeros(1536, dtype=np.float32)
        atok = ASM.step(cur, h, hn, p)
        tok_ok = (atok == int(etok))
        alltok &= tok_ok
        r = float(np.abs(hn - ehh).max() / max(np.abs(ehh).max(), 1e-9))
        worst = max(worst, r)
        case(f"draft step {step} token", tok_ok, f"asm={atok} ref={int(etok)}")
        cur = int(g.forward(cur))
    case("all 4 draft tokens exact", alltok)
    case("h_next rel err < 2e-4 (ref is fp64 inside)", worst < 2e-4, f"worst={worst:.2e}")
    g.close()
    print("ALL MTP CHECKS PASS" if not fails else f"{len(fails)} FAILURES: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
