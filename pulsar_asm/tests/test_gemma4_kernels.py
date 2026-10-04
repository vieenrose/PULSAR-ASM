"""Numerical tests for the Gemma 4 pointwise/attention kernels (numpy as truth).

Every kernel is compared against the exact torch/numpy reference for the op it
implements, using fp32 activations and (where relevant) bf16-rounded weights.
Run:  python3 tests/test_gemma4_kernels.py
"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runtime import pulsar_abi as abi  # noqa: E402

MOD = None


def f32bits(x):
    return struct.unpack("<I", struct.pack("<f", np.float32(x)))[0]


def to_bf16_exact(a):
    w = a.astype(np.float32).view(np.uint32)
    w &= np.uint32(0xFFFF0000)
    return np.ascontiguousarray((w >> 16).astype(np.uint16)), (w.view(np.float32))


def relerr(got, ref):
    d = float(np.max(np.abs(got - ref)))
    s = float(np.max(np.abs(ref)))
    return d if s == 0 else d / s


def case(name, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'} {name:34s} {detail}")
    return bool(ok)


def main():
    global MOD
    allok = True
    mod = abi.load_module(os.path.join(os.path.dirname(__file__), "kernels_test_mod.asm"))
    MOD = mod
    X = lambda *a, n=0: abi.make_reg_entry(mod.exports[a[0]], a[1] if len(a) > 1 else n)
    rng = np.random.default_rng(7)
    eps = f32bits(1e-6)

    # ---- rmsnorm (plain weight, no Gemma-2 1+w) ------------------------------
    g_rms = X("rmsnorm_avx2", 5)
    for N in (256, 1536, 2048, 6144):
        x = rng.standard_normal(N).astype(np.float32)
        w = (rng.standard_normal(N).astype(np.float32) * 0.1 + 1.0).astype(np.float32)
        ref = (x * (1.0 / np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-6))).astype(np.float32) * w
        out = np.zeros(N, np.float32)
        g_rms(out.ctypes.data, w.ctypes.data, x.ctypes.data, N, eps)
        allok &= case(f"rmsnorm N={N}", relerr(out, ref) < 3e-7, f"rel={relerr(out, ref):.2e}")

    # ---- rmsnorm without weight (v_norm), scale applied after norming --------
    g_rs = X("rmsnorm_scale_avx2", 5)
    for N, sc in ((256, 1.0), (512, 0.5), (2048, 0.25)):
        x = rng.standard_normal(N).astype(np.float32)
        ref = (x * (1.0 / np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-6)) * sc).astype(np.float32)
        out = np.zeros(N, np.float32)
        g_rs(out.ctypes.data, x.ctypes.data, N, f32bits(sc), eps)
        allok &= case(f"rmsnorm_scale N={N} s={sc}", relerr(out, ref) < 3e-7, f"rel={relerr(out, ref):.2e}")

    # ---- gelu_pytorch_tanh --------------------------------------------------
    g_gelu = X("gelu_tanh_avx2", 3)
    x = np.concatenate([rng.standard_normal(4096), rng.uniform(-12, 12, 4096)]).astype(np.float32)
    c = np.sqrt(2.0 / np.pi).astype(np.float32)
    ref = 0.5 * x * (1.0 + np.tanh(c * (x + np.float32(0.044715) * x ** 3)))
    out = np.zeros_like(x)
    g_gelu(out.ctypes.data, x.ctypes.data, x.size)
    allok &= case("gelu_tanh N=8192", relerr(out, ref) < 2e-5, f"rel={relerr(out, ref):.2e}")

    # ---- geglu (tanh-gated MLP) ---------------------------------------------
    g_geglu = X("geglu_avx2", 4)
    N = 6144
    gate = rng.standard_normal(N).astype(np.float32)
    up = rng.standard_normal(N).astype(np.float32)
    ref = (0.5 * gate * (1 + np.tanh(c * (gate + np.float32(0.044715) * gate ** 3))) * up).astype(np.float32)
    out = np.zeros(N, np.float32)
    g_geglu(out.ctypes.data, gate.ctypes.data, up.ctypes.data, N)
    allok &= case(f"geglu N={N}", relerr(out, ref) < 2e-5, f"rel={relerr(out, ref):.2e}")

    # ---- add_scaled (residual + layer_scalar) -------------------------------
    g_add = X("add_scaled_avx2", 6)
    N = 1536
    a = rng.standard_normal(N).astype(np.float32)
    b = rng.standard_normal(N).astype(np.float32)
    sa, sb = 1.0, 0.7071
    ref = (a * sa + b * sb).astype(np.float32)
    out = np.zeros(N, np.float32)
    g_add(out.ctypes.data, a.ctypes.data, b.ctypes.data, N, f32bits(sa), f32bits(sb))
    allok &= case("add_scaled N=1536", relerr(out, ref) < 1e-7, f"rel={relerr(out, ref):.2e}")

    # ---- scale / zero -------------------------------------------------------
    g_sc, g_z = X("scale_avx2", 3), X("zero_avx2", 2)
    a = rng.standard_normal(1536).astype(np.float32)
    ref = (a * 0.7071).astype(np.float32)
    out = a.copy()
    g_sc(out.ctypes.data, out.size, f32bits(0.7071))
    allok &= case("scale N=1536", relerr(out, ref) < 1e-7)
    out = np.full(1536, 3.0, np.float32)
    g_z(out.ctypes.data, out.size)
    allok &= case("zero N=1536", not out.any())

    # ---- embed_bf16 (row gather + sqrt(dim) scale) --------------------------
    g_emb = X("embed_bf16_avx2", 5)
    V, D, tok = 512, 1536, 377
    emb_b, emb_f = to_bf16_exact(rng.standard_normal((V, D)).astype(np.float32))
    scale = np.float32(np.sqrt(D))
    ref = (emb_f[tok] * scale).astype(np.float32)  # fp32 mul, matches the kernel exactly
    out = np.zeros(D, np.float32)
    g_emb(out.ctypes.data, emb_b.ctypes.data, tok, D, f32bits(scale))
    allok &= case(f"embed_bf16 row={tok}", np.array_equal(out, ref), "exact bf16->f32")

    # ---- rope_apply ---------------------------------------------------------
    # The table comes from the loader so the test cannot drift from what the
    # engine is handed. partial < 1 is the full-attention case: HF zero-pads
    # inv_freq, so the tail is cos=1/sin=0 and STILL pairs (i, i+hd/2) - a
    # narrower window there is the wrong function.
    g_rope = X("rope_apply_avx2", 5)
    from runtime.gemma4_model import inv_freq as hf_inv_freq
    for hd, theta, part in ((256, 10000.0, 1.0), (512, 1000000.0, 1.0),
                            (512, 1000000.0, 0.25)):
        H = hd // 2
        pos = 77
        inv = hf_inv_freq({"partial_rotary_full": 1.0}, hd, theta, part)
        ang = pos * inv
        cos = np.cos(ang).astype(np.float32)
        sin = np.sin(ang).astype(np.float32)
        x = rng.standard_normal(hd).astype(np.float32)
        o = x.copy()
        o[:H] = x[:H] * cos - x[H:] * sin
        o[H:] = x[H:] * cos + x[:H] * sin
        buf = x.copy()
        rp = H
        g_rope(buf.ctypes.data, cos.ctypes.data, sin.ctypes.data, hd, rp)
        allok &= case(f"rope head_dim={hd} partial={part}", relerr(buf, o) < 1e-6,
                      f"rel={relerr(buf, o):.2e}")

    # ---- softmax ------------------------------------------------------------
    g_sm = X("softmax_avx2", 2)
    for n in (1, 7, 512, 2050):
        x = (rng.standard_normal(n) * 8).astype(np.float32)
        if n > 8:
            x[:8] = -1e4  # exercise the clamped tail
        z = x - x.max()
        ref = np.exp(z.astype(np.float64))
        ref /= ref.sum()
        buf = x.copy()
        g_sm(buf.ctypes.data, n)
        allok &= case(f"softmax n={n}", relerr(buf, ref) < 2e-5, f"rel={relerr(buf, ref):.2e}")

    # ---- attn_scores / attn_values ------------------------------------------
    g_sc2 = X("attn_scores_avx2", 6)
    g_vv = X("attn_values_avx2", 6)
    for hd in (256, 512):
        kv_len, kv_start, rows = 129, 11, 200
        q = rng.standard_normal(hd).astype(np.float32)
        K = rng.standard_normal((rows, hd)).astype(np.float32)
        Vm = rng.standard_normal((rows, hd)).astype(np.float32)
        scores = np.zeros(kv_len, np.float32)
        g_sc2(scores.ctypes.data, q.ctypes.data, K.ctypes.data, kv_len, hd, kv_start)
        ref = np.einsum("d,pd->p", q, K[kv_start:kv_start + kv_len])
        allok &= case(f"attn_scores hd={hd}", relerr(scores, ref) < 1e-6, f"rel={relerr(scores, ref):.2e}")
        w = (rng.random(kv_len) * 0.01).astype(np.float32)
        out = np.zeros(hd, np.float32)
        g_vv(out.ctypes.data, Vm.ctypes.data, w.ctypes.data, kv_len, hd, kv_start)
        refv = np.einsum("p,pd->d", w, Vm[kv_start:kv_start + kv_len])
        allok &= case(f"attn_values hd={hd}", relerr(out, refv) < 1e-6, f"rel={relerr(out, refv):.2e}")

    # ---- softcap_tanh -------------------------------------------------------
    g_cap = X("softcap_tanh_avx2", 3)
    cap = 30.0
    x = (rng.standard_normal(4096) * 25).astype(np.float32)
    ref = (cap * np.tanh(x / cap)).astype(np.float32)
    buf = x.copy()
    g_cap(buf.ctypes.data, buf.size, f32bits(cap))
    allok &= case("softcap_tanh N=4096", relerr(buf, ref) < 2e-5, f"rel={relerr(buf, ref):.2e}")
    # monotonicity guard: argmax must survive the cap untouched
    g_ax = X("sampler_argmax_avx2", 2)
    raw = (rng.standard_normal(65536) * 40).astype(np.float32)
    capped = (cap * np.tanh(raw / cap)).astype(np.float32)
    a1 = g_ax(raw.ctypes.data, raw.size)
    a2 = g_ax(capped.ctypes.data, capped.size)
    allok &= case("softcap is argmax-invariant", a1 == a2 == int(np.argmax(capped)), f"asm={a1}/{a2}")

    # ---- argmax edge cases --------------------------------------------------
    for label, row in (("all negative", -np.abs(rng.standard_normal(50000))) ,
                       ("tie first wins", np.r_[np.zeros(1000), np.ones(10)]),
                       ("tail max", np.r_[np.zeros(40000), np.full(40000, -0.5)])):
        row = row.astype(np.float32)
        got = g_ax(row.ctypes.data, row.size)
        exp = int(np.argmax(row))
        allok &= case(f"argmax {label}", got == exp, f"asm={got} np={exp}")

    # ---- ple_combine --------------------------------------------------------
    g_ple = X("ple_combine_avx2", 8)
    D = 256
    proj = rng.standard_normal(D).astype(np.float32)
    # per_layer_embeddings stays bf16 in the blob (262144 x 8960 fp32 would cost
    # another 9 GB), so the token branch arrives as bf16 - the kernel must widen
    # it, and a test that hands it fp32 will never notice.
    tok, tokf = to_bf16_exact(rng.standard_normal(D) / 16)
    # NON-ones weight on purpose: an all-ones per_layer_projection_norm weight
    # hides a kernel that forgets to apply it (it did), and hides a dropped 8th
    # argument, because the kernel then reads a pointer it never multiplies.
    w = (1 + 0.05 * rng.standard_normal(D)).astype(np.float32)
    ps, ts = 0.0255351, 16.0
    v = proj * ps
    ref = ((v / np.sqrt(np.mean(v.astype(np.float64) ** 2) + 1e-6) * w) + tokf * ts) / np.sqrt(2.0)
    out = np.zeros(D, np.float32)
    g_ple(out.ctypes.data, proj.ctypes.data, tok.ctypes.data, D, f32bits(ps), f32bits(ts),
          f32bits(1 / np.sqrt(2)), w.ctypes.data)
    allok &= case("ple_combine D=256", relerr(out, ref) < 3e-6, f"rel={relerr(out, ref):.2e}")

    allok &= lint_avx2_purity()
    print("\n" + ("ALL GEMMA-4 KERNEL TESTS PASS" if allok else "FAILURES PRESENT"))
    return 0 if allok else 1




def lint_avx2_purity():
    """No AVX-512-only encodings may appear in any PULSAR source.

    VBROADCASTSS with a *register* source exists only in AVX-512 (EVEX). FASM
    happily emits it on an AVX2 target, so it assembles, runs on this AVX-512
    dev box, and would #UD on the AVX2-only machines the project promises.
    The memory form (or vinsertf128 + vshufps) is the AVX2-legal way.
    """
    import os, re, glob
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bad = []
    for f in glob.glob(os.path.join(root, "**", "*.asm"), recursive=True):
        for i, line in enumerate(open(f, errors="replace"), 1):
            if re.match(r"\s*vbroadcasts[sd]\s+ymm\d+,\s*xmm", line) or \
               re.search(r"\b(?:ymm|xmm)(?:1[6-9]|[2-9][0-9])\b", line.split(';')[0]):
                bad.append(f"{os.path.relpath(f, root)}:{i}: {line.strip()}")
    print(("\n" + "\n".join("  AVX512-ONLY " + b for b in bad)) if bad else
          "  OK   all .asm sources are AVX2-only (no register-source vbroadcastss)")
    return not bad


if __name__ == "__main__":
    sys.exit(main())
    main()
