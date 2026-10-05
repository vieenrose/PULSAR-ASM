"""Bisect the batched pass by stage.

Builds a variant of the engine that returns immediately after a chosen step, so
a buffer can be inspected in the state the batched pass actually left it in.
Compares row b>0 against the same stage run one token at a time.
"""
import ctypes
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime import pulsar_abi as abi                      # noqa: E402
from runtime.gemma4_model import Gemma4      # noqa: E402

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "engine", "gemma4_engine_flat.asm")
MARK_RE = re.compile(r"mov\s+qword \[rbx \+ CTR_MARK\], (\d+)")


def patched_src(after_mark):
    """Engine source with a ret inserted after `mov [rbx+CTR_MARK], after_mark`."""
    out = []
    done = False
    for line in open(SRC):
        out.append(line)
        m = MARK_RE.search(line)
        if not done and m and int(m.group(1)) == after_mark:
            out.append("    add     rsp, 136\n"
                       "    pop     r15\n    pop     r14\n    pop     r13\n"
                       "    pop     r12\n    pop     rbp\n    pop     rbx\n    ret\n")
            done = True
    if not done:
        raise SystemExit(f"no CTR_MARK = {after_mark} in the engine source")
    return "".join(out)


def stage_engine(after_mark, tokens, pos, threads=4):
    """One batched pass, stopped after `after_mark`; returns the engine + row view."""
    # the patched source has to live next to the engine: the sub-assemblies are
    # %include'd by relative path
    src = patched_src(after_mark)
    tmp = os.path.join(os.path.dirname(SRC), f"_probe_stage{after_mark}.asm")
    open(tmp, "w").write(src)
    g = Gemma4("/mnt/edge/pulsar/gemma4_e2b.bin", max_seq=512, n_threads=threads, verbose=False)
    entry = abi.make_reg_entry(abi.load_module(tmp).exports["gemma4_step"], 1)
    g._step = entry
    return g, tmp


def rows(g, name, b, dim):
    a = np.asarray(g.buf[name]).reshape(-1)[:b * dim].reshape(b, dim).copy()
    return a


def main():
    b = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    toks = [2, 105, 2364, 107, 237105, 122100, 238360, 238458][:b]
    marks = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["2", "3"])]

    # reference: one token at a time through the same stages
    ref = {}
    for mark in marks:
        g, tmp = stage_engine(mark, toks, 0)
        seq = []
        for i, t in enumerate(toks):
            g._set("B", 1)
            g._set("POS", i)
            g.tokens[0] = t
            g._step(g.ctx_mem.ctypes.data)
            seq.append({k: rows(g, k, 1, d)[0].copy() for k, d in
                        (("X", g.hidden), ("PLE_NEXT", g.n_layers * g.ple_dim))})
        g.close()
        os.unlink(tmp)
        ref[mark] = seq

    for mark in marks:
        g, tmp = stage_engine(mark, toks, 0)
        g._set("B", b)
        g._set("POS", 0)
        for i, t in enumerate(toks):
            g.tokens[i] = t
        g._step(g.ctx_mem.ctypes.data)
        print(f"--- after CTR_MARK == {mark}")
        for name, dim in (("X", g.hidden), ("PLE_NEXT", g.n_layers * g.ple_dim)):
            bat = rows(g, name, b, dim)
            for i in range(b):
                r = ref[mark][i][name]
                d = float(np.max(np.abs(bat[i] - r))) / max(float(np.abs(r).max()), 1e-30)
                print(f"    {name:9s} row {i}  max rel {d:.3e}")
        g.close()
        os.unlink(tmp)


if __name__ == "__main__":
    main()
