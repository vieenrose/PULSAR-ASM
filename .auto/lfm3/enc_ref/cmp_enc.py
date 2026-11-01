#!/usr/bin/env python3
"""cmp_enc.py -- on-Nano comparator for the S3 encoder gate (checks_enc.sh).

Compares the GPU encoder's stage dumps (encgpu/<stage>.bin, raw float32) against the C
reference's dumps (encC/<stage>, raw float32 from lfm3_enc.c). Prints rms/max relative
deviation, an all-zero check and a non-finite check per stage, and exits non-zero if a
ported stage misses its gate.

Gate values:
  fp32 arms   : rms rel <= 1e-4 (fp32 rounding; measured 3.4e-5 for the downsampler)
  half2 arms  : no rms gate -- those are graded by the end-to-end transcript gate,
                because bf16 ACTIVATION rounding (not the fp16 products) sets a ~0.28
                per-cell rms on encoder data (run #309), which every shipped backbone
                GEMV already does too.
"""
import numpy as np
import os
import sys

TOK = int(sys.argv[1])
H2 = int(os.environ.get("LFM_ENC_H2", "0"))
# Per-stage gates, and WHY they differ.
#  ds_out  (only lin_out is Q8, the two 1x1 convs are f32): the C reference and the GPU
#          run the same fp32 arithmetic, so 1e-4 rms is the right bar (measured 3.4e-5).
#  layer_* (in_proj/out_proj/w1/w2/w3 are Q8): the C reference DEQUANTIZES the codes to
#          float and dots in ascending k; the engine's Q8 arm dots the int8 codes per
#          32-code block and scales afterwards. Both are legitimate fp32 algorithms, and
#          on cancelling 768-term dots they differ by ~1e-3 (run #309 measured exactly
#          this on real activations), so the C file is not a bit reference for Q8 dots.
#          The bar is rms rel 2e-2 AND max abs 1e-3 AND the aggregate rms matching to
#          4 decimals -- three independent checks that all caught the dropped-conv-branch
#          bug (rms rel 24, rms 0.1312 vs 0.1276) on the first run.
#  half2 arms: no rms gate at all (bf16 ACTIVATION rounding dominates, run #309); the
#          non-zero, finite and aggregate-rms checks still apply, and shipping them is
#          graded by the end-to-end transcript gate.
Q8_STAGES = set(["layer_%02d" % L for L in range(17)] + ["enc_out", "proj_out"])
STAGES = [("ds_out", 768), ("enc_out", 768), ("proj_out", 2048)] + \
         [("layer_%02d" % L, 768) for L in range(17)]
fails = 0
nport = 0
for name, cols in STAGES:
    gp, cp = os.path.join("encgpu", name + ".bin"), os.path.join("encC", name)
    if not os.path.exists(gp):
        continue                                     # stage not ported yet
    nport += 1
    if not os.path.exists(cp):
        print("  %-10s MISSING C REFERENCE DUMP" % name)
        fails += 1
        continue
    a = np.fromfile(gp, dtype=np.float32).reshape(TOK, cols)
    b = np.fromfile(cp, dtype=np.float32).reshape(TOK, cols)
    if not np.isfinite(a).all():
        print("  %-10s NON-FINITE OUTPUT (fail)" % name)
        fails += 1
        continue
    nz = int((a != 0).sum())
    d = np.abs(a - b).astype(np.float64)
    m = np.abs(b) > 1e-6
    rms = float(np.sqrt(((d[m] / np.abs(b[m].astype(np.float64))) ** 2).mean())) if m.any() else 0.0
    gate = None if H2 else (2e-2 if name in Q8_STAGES else 1e-4)
    # aggregate rms must agree to 4 decimals (catches scale/branch bugs the relative
    # measures can hide when a whole branch is missing)
    rms_agree = abs(float(np.sqrt((a ** 2).mean())) - float(np.sqrt((b ** 2).mean()))) < 1e-4
    maxabs_gate = 1e-3 if name in Q8_STAGES else 1e-5
    ok = nz > 0 and rms_agree and float(d.max()) <= maxabs_gate and (gate is None or rms <= gate)
    if not ok:
        fails += 1
    print("  %-10s rms rel %.3e vs C  max abs %.3e  rms(GPU) %.5f rms(C) %.5f  nonzero %d/%d  %s"
          % (name, rms, d.max(), np.sqrt((a ** 2).mean()), np.sqrt((b ** 2).mean()), nz, a.size,
             "OK" if ok else "FAIL (gates: rms %s maxabs %s rmsagree %s)"
             % (rms <= (gate or 0), float(d.max()) <= (1e-3 if name in Q8_STAGES else 1e-5), rms_agree)))
print("ENC_GATE %s (arms=%s T=%d stages=%d)" % ("PASS" if fails == 0 and nport else "FAIL",
                                                "half2" if H2 else "fp32", TOK, nport))
sys.exit(0 if (fails == 0 and nport) else 1)
