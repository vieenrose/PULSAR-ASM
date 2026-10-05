"""Stop inside one decoder layer, before a chosen label, and compare rows.

MAXLAYER=1 keeps it to layer 0 (sliding: head_dim 256, inter 6144 - the narrow
case, which is where a stride derived from the layer rather than the buffer
would show). For each label the buffers are compared between a batched pass and
the same stage run one token at a time.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime.gemma4_model import Gemma4                        # noqa: E402

LAYER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "sub_assemblies", "sub_gemma4_layer_flat.asm")
ORIG = open(LAYER).read()
EPI = ("    add     rsp, 192\n    pop     r14\n    pop     r13\n"
       "    pop     r12\n    pop     rbp\n    ret\n")
LABELS = [".ly_nin", ".ly_qrow", ".ly_qhead", ".ly_kvrow", ".ly_window", ".ly_att",
          ".ly_att_head", ".ly_postr", ".ly_preffr", ".ly_postrff", ".ly_plemul",
          ".ly_plenorm"]
BUFS = ["X", "T1", "T2", "T3", "Q", "A", "G", "U", "M", "TMP256", "PLE_CUR",
        "PLE_NEXT", "PLE_IN"]


def install(label):
    """Layer source that returns just before `label`; written over the real file."""
    if ORIG.count(f"\n{label}:") != 1:
        raise SystemExit(f"label {label} not unique in the layer source")
    return ORIG.replace(f"\n{label}:", "\n" + EPI + f"\n{label}:")


def rows(g, name, b, dim):
    return np.asarray(g.buf[name]).reshape(-1)[:b * dim].reshape(b, dim).copy()


def run(g, toks, batched):
    dims = {"X": g.hidden, "T1": g.hidden, "T2": g.hidden, "T3": g.hidden,
            "Q": g.n_head * g.max_hd, "A": g.n_head * g.max_hd,
            "G": g.max_inter, "U": g.max_inter, "M": g.max_inter,
            "TMP256": g.ple_dim, "PLE_CUR": g.n_layers * g.ple_dim,
            "PLE_NEXT": g.n_layers * g.ple_dim, "PLE_IN": g.n_layers * g.ple_dim}
    g._set("MAXLAYER", 1)
    out = {}
    if batched:
        g._set("B", len(toks))
        g._set("POS", 6)
        for i, t in enumerate(toks):
            g.tokens[i] = t
        g.run(toks, pos=6)
        for k, d in dims.items():
            out[k] = rows(g, k, len(toks), d)
    else:
        g._set("B", 1)
        for i, t in enumerate(toks):
            g.run([t], pos=6 + i)
            for k, d in dims.items():
                out.setdefault(k, []).append(rows(g, k, 1, d)[0].copy())
        for k in out:
            out[k] = np.stack(out[k])
    return out


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else ".ly_plenorm"
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    toks = [2, 105]
    open(LAYER, "w").write(install(label))
    try:
        g = Gemma4("/mnt/edge/pulsar/gemma4_e2b.bin", max_seq=512, n_threads=4, verbose=False)
        ref = run(g, toks, False)
        g.reset()
        got = run(g, toks, True)
        print(f"--- layer returns before {label}")
        for k in BUFS:
            if k not in got:
                continue
            line = [f"    {k:9s}"]
            for i in range(b):
                r = ref[k][i]
                d = float(np.max(np.abs(got[k][i] - r))) / max(float(np.abs(r).max()), 1e-30)
                line.append(f"row{i} {d:.2e}")
            print("  ".join(line))
        g.close()
    finally:
        open(LAYER, "w").write(ORIG)


if __name__ == "__main__":
    main()
