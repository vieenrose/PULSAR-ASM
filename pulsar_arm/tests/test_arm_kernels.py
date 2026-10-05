#!/usr/bin/env python3
"""Unit-test every NEON kernel against torch/numpy on the Pi."""
import ctypes
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SO = os.path.join(HERE, "libarmk.so")
SRC = [os.path.join(HERE, "..", "kernels", "neon_gemv.c"),
       os.path.join(HERE, "..", "kernels", "neon_ops.c")]
fails = []


def case(name, ok, detail=""):
    print(f"  {'ok' if ok else 'FAIL'}  {name}  {detail}", flush=True)
    if not ok:
        fails.append(name)


def main():
    r = subprocess.run(["gcc", "-O2", "-fPIC", "-shared"] + SRC + ["-o", SO, "-lm"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lib = ctypes.CDLL(SO)
    v = ctypes.c_void_p
    for f_, n in (("rmsnorm_f32", 5), ("rope_half", 4), ("softmax_f32", 2),
                  ("gelu_tanh_f32", 3), ("argmax_f32", 2), ("embed_row_f32", 5),
                  ("attn_scores_f32", 5), ("attn_values_f32", 5)):
        fn = getattr(lib, f_)
        fn.argtypes = [v] * n + ([ctypes.c_int] if False else [])
        import ctypes as C
        ints = {"rmsnorm_f32": [3], "rope_half": [3], "softmax_f32": [1],
                "gelu_tanh_f32": [2], "argmax_f32": [1], "embed_row_f32": [2, 3],
                "attn_scores_f32": [3, 4], "attn_values_f32": [3, 4]}
        at = [v] * n
        for i in ints[f_]:
            at[i] = C.c_int
        if f_ in ("rmsnorm_f32", "embed_row_f32"):
            at[4 if f_ == "rmsnorm_f32" else 4] = C.c_float
        fn.argtypes = at
        fn.restype = C.c_int if f_ == "argmax_f32" else None
    import torch
    rng = np.random.default_rng(0)

    x = rng.standard_normal(640).astype(np.float32)
    w = rng.standard_normal(640).astype(np.float32)
    o = np.zeros(640, dtype=np.float32)
    lib.rmsnorm_f32(o.ctypes.data, w.ctypes.data, x.ctypes.data, 640, 1e-6)
    ref = (torch.from_numpy(x) / torch.sqrt((torch.from_numpy(x) ** 2).mean() + 1e-6)
           * torch.from_numpy(w)).numpy()
    case("rmsnorm", float(np.abs(o - ref).max()) < 1e-5, f"{float(np.abs(o-ref).max()):.2e}")

    h = 128
    vv = rng.standard_normal(256).astype(np.float32)
    cc = rng.standard_normal(128).astype(np.float32)
    ss = rng.standard_normal(128).astype(np.float32)
    o = vv.copy()
    lib.rope_half(o.ctypes.data, cc.ctypes.data, ss.ctypes.data, h)
    a, b = vv[:h], vv[h:]
    ref = np.concatenate([a * cc - b * ss, b * cc + a * ss])
    case("rope", float(np.abs(o - ref).max()) < 1e-5, f"{float(np.abs(o-ref).max()):.2e}")

    z = (rng.standard_normal(37).astype(np.float32)) * 3
    o = z.copy()
    lib.softmax_f32(o.ctypes.data, 37)
    ref = torch.softmax(torch.from_numpy(z), -1).numpy()
    case("softmax", float(np.abs(o - ref).max()) < 1e-5, f"{float(np.abs(o-ref).max()):.2e}")

    g = (rng.standard_normal(2048).astype(np.float32)) * 2
    o = np.zeros_like(g)
    lib.gelu_tanh_f32(o.ctypes.data, g.ctypes.data, 2048)
    ref = (torch.nn.functional.gelu(torch.from_numpy(g), approximate="tanh")).numpy()
    case("gelu", float(np.abs(o - ref).max()) < 1e-4, f"{float(np.abs(o-ref).max()):.2e}")

    q = rng.standard_normal(256).astype(np.float32)
    K = rng.standard_normal((9, 256)).astype(np.float32)
    s = np.zeros(9, dtype=np.float32)
    lib.attn_scores_f32(s.ctypes.data, q.ctypes.data, K.ctypes.data, 9, 256)
    case("scores", float(np.abs(s - K @ q).max()) < 1e-3, f"{float(np.abs(s-K@q).max()):.2e}")
    lib.softmax_f32(s.ctypes.data, 9)
    V = rng.standard_normal((9, 256)).astype(np.float32)
    o = np.zeros(256, dtype=np.float32)
    lib.attn_values_f32(o.ctypes.data, V.ctypes.data, s.ctypes.data, 9, 256)
    case("values", float(np.abs(o - s @ V).max()) < 1e-3, f"{float(np.abs(o-s@V).max()):.2e}")

    print("ALL KERNEL CHECKS PASS" if not fails else f"FAILURES: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
