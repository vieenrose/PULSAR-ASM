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
    gate = None if H2 else 1e-4
    ok = nz > 0 and (gate is None or rms <= gate)
    if not ok:
        fails += 1
    print("  %-10s rms rel %.3e vs C  max abs %.3e  rms(GPU) %.5f rms(C) %.5f  nonzero %d/%d  %s"
          % (name, rms, d.max(), np.sqrt((a ** 2).mean()), np.sqrt((b ** 2).mean()), nz, a.size,
             "OK" if ok else "FAIL"))
print("ENC_GATE %s (arms=%s T=%d stages=%d)" % ("PASS" if fails == 0 and nport else "FAIL",
                                                "half2" if H2 else "fp32", TOK, nport))
sys.exit(0 if (fails == 0 and nport) else 1)
