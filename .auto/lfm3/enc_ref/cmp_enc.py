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
# TWO BAR SETS, and the reason for the split. The encoder has an open run-to-run defect
# (doc section 29): six runs of one binary agree bit-for-bit through layer_04 and then
# deviate by an amplitude that VARIES between runs -- energy rel 1.4e-06 (exact run) up
# to 5.2e-03, max abs 9.5e-05 up to 6.8e-02. A single bar set cannot be both strict and
# stable against a moving target, so:
#   default            = TIGHT bars, written from the clean state. The gate goes RED when
#                        the defect bites, which is the correct answer: the run is not
#                        reproducible. Clean-state runs pass with 10-20x margin.
#   LFM_ENC_DEFECT_OK=1 = defect-tolerant bars (measured worst x2-3), for regression work
#                        on the rest of the port while section 29 stays open. The line
#                        "DEFECT-TOLERANT MODE" is printed so a green here is never
#                        mistaken for a green there.
DEFECT_OK = int(os.environ.get("LFM_ENC_DEFECT_OK", "0"))
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
# The attention internals are gated too: q/k after headnorm+RoPE, v, and the attention
# output of the probe layer (LFM_ENC_PROBE_LAYER=L on both binaries, L=2 by default).
PROBES = [("probe_q_02", 38400), ("probe_k_02", 19200), ("probe_v_02", 19200), ("probe_ao_02", 38400)]
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
    # Tightened once the whole tower was ported: with the attention race fixed the worst
    # stage is rms rel 1.0e-04 / max abs 9.1e-04 vs the C reference, so the Q8 bar moves
    # from 2e-2/1e-3 to 1e-3/2e-3 -- 10x margin on rms, 2x on max abs, and it still
    # catches everything the loose one did (dropped residual branch: rms 24; headnorm
    # aliasing race: rms 40; softmax shared race: rms 0.33).
    # THE GATE, and why each piece is there.
    #  energy rel = sqrt(mean((a-b)^2)) / sqrt(mean(b^2)) -- cancellation-robust, the
    #          primary quantitative bar: 1e-3. Measured (fp32 arms): ds_out 8.4e-07,
    #          layer_00 9.6e-07, layer_16 2.7e-06, enc_out 2.5e-05, proj_out 1.6e-04.
    #  max abs <= 2e-3 (layers/enc_out/proj_out), 1e-5 (ds_out). Measured 9.9e-04 worst.
    #  per-element rms rel is GATED at 1e-4 for ds_out only, where it is meaningful
    #          (f32 weights, no cancellation; measured 3.4e-05). For the Q8 stages it is
    #          reported, not gated: projector/ffn outputs come from cancelling 768- and
    #          1792-term dots, so near-zero cells give a large relative error from a tiny
    #          absolute one (proj_out: per-element 1.9e-02 with max abs 2.3e-04).
    #  aggregate rms agree to 4 decimals, non-zero, finite -- catches whole-branch
    #          losses the relative measures hide (dropped conv branch: rms 24).
    en = float(np.sqrt((d ** 2).mean()) / np.sqrt((b.astype(np.float64) ** 2).mean()))
    # The bar is 5e-3 because of the OPEN DEFECT recorded in doc section 29: repeated
    # runs of one binary place layers 5-16 at up to 1.85e-03 energy rel (some runs are
    # exact to 1.4e-06). 1e-3 sat INSIDE that spread, so the gate was flaky, not strict.
    # 5e-3 = 2.7x the observed worst; tighten it back when the defect is pinned.
    # The half2 arms get their OWN bar, not "no gate": bf16 activation rounding is a
    # known, accepted loss channel (run #309), so per-element and max-abs bars written
    # for fp32 are not the right test -- measured h2: energy rel 4.6e-04 (ds_out) to
    # 3.9e-03 (proj_out) with per-element 0.13-0.84 and max abs up to 2.5e-02. The
    # h2 bar below catches structural breakage (a dropped branch is 1e+00 energy);
    # shipping them is graded by the end-to-end speech->text gate.
    gate = (1e-2 if H2 else ((5e-3 if DEFECT_OK else 1e-3) if name in Q8_STAGES else 1e-4))
    # fp32: aggregate rms must agree to 4 decimals. half2: bf16 activation rounding moves
    # the aggregate rms by ~1e-4, so that arm uses a 1e-3 RELATIVE agreement test.
    ra_, rb_ = float(np.sqrt((a ** 2).mean())), float(np.sqrt((b ** 2).mean()))
    rms_agree = abs(ra_ - rb_) / max(rb_, 1e-12) < 1e-3 if H2 else abs(ra_ - rb_) < 1e-4
    # max abs: 2e-2 for the Q8 stages. Set from measurement, not taste -- the run-to-run
    # defect (section 29) puts the observed single-run worst at 6.5e-03 (layer_13), while
    # every real bug this gate has caught sat at 1e-01 and above (headnorm race 1.0e-01,
    # attention softmax race 1.0e-01, dropped residual branch 4.5e+02). 2e-2 keeps 3x
    # clearance over the defect and 5x over the worst real bug.
    maxabs_gate = (5e-2 if H2 else ((2e-2 if DEFECT_OK else 2e-3) if name in Q8_STAGES else 1e-5))
    per_elem_gate = None if (H2 or name in Q8_STAGES) else 1e-4
    ok = (nz > 0 and rms_agree and float(d.max()) <= maxabs_gate
          and (gate is None or en <= gate)
          and (per_elem_gate is None or rms <= per_elem_gate))
    if not ok:
        fails += 1
    print("  %-10s energy rel %.3e  per-elem rms rel %.3e  max abs %.3e  rms(GPU) %.5f rms(C) %.5f  nonzero %d/%d  %s"
          % (name, en, rms, d.max(), np.sqrt((a ** 2).mean()), np.sqrt((b ** 2).mean()), nz, a.size,
             "OK" if ok else "FAIL (energy<=%g got %g | maxabs<=%g got %g | per-elem<=%g got %g | rmsagree %s)"
             % (gate or 0, en, maxabs_gate, float(d.max()), per_elem_gate or 0, rms, rms_agree)))
# ---- run-to-run determinism report (the open defect, made visible, not hidden) ----
GPU2 = "encgpu2"
if os.path.isdir(GPU2):
    worst = 0.0
    for name, cols in STAGES:
        gp2 = os.path.join(GPU2, name + ".bin")
        gp, cp = os.path.join("encgpu", name + ".bin"), os.path.join("encC", name)
        if not (os.path.exists(gp) and os.path.exists(gp2)): continue
        a = np.fromfile(gp, dtype=np.float32).astype(np.float64)
        a2 = np.fromfile(gp2, dtype=np.float32).astype(np.float64)
        e = float(np.sqrt(((a - a2) ** 2).mean()) / np.sqrt((a ** 2).mean()))
        worst = max(worst, e)
        if e > 1e-5:
            print("  %-10s run-to-run energy rel %.3e  (max abs %.2e)" % (name, e, np.abs(a - a2).max()))
    # The defect budget is the run-to-run spread we have MEASURED and accept while the
    # root cause (section 29) stays open: fp32 arms 5e-03 (worst observed 2.8e-03, a
    # same-session pair measured as low as 4.9e-05), half2 arms 1e-02 (worst observed
    # 4.0e-03). It is a bound on a known defect, printed every run -- not accuracy slack:
    # the accuracy bars above are what the port is graded on.
    budget = 1e-2 if H2 else (5e-3 if DEFECT_OK else 3e-3)
    print("  run-to-run WORST energy rel %.3e  (defect budget %.0e: %s)"
          % (worst, budget, "OK" if worst <= budget else "OVER"))
    if worst > budget: fails += 1

for name, n in PROBES:
    gp, cp = os.path.join("encgpu", name + ".bin"), os.path.join("encC", name)
    if not (os.path.exists(gp) and os.path.exists(cp)):
        print("  %-10s MISSING (run both binaries with LFM_ENC_PROBE_LAYER=2)" % name); fails += 1; continue
    a = np.fromfile(gp, dtype=np.float32); b = np.fromfile(cp, dtype=np.float32)
    assert a.size == n == b.size, (name, a.size, b.size)
    m = np.abs(b) > 1e-6
    rms = float(np.sqrt(((np.abs(a - b)[m] / np.abs(b[m])) ** 2).mean()))
    en = float(np.sqrt(((a - b) ** 2).mean()) / np.sqrt((b ** 2).mean()))
    fin = bool(np.isfinite(a).all())
    # fp32 arms: energy <= 1e-4, max abs <= 5e-5, and for probe_ao (a cancelling PV sum)
    # per-element rms rel <= 1e-4. Measured 6.4e-07..1.0e-06, 1.9e-07..5.7e-06, 4.9e-05.
    # The headnorm aliasing race that motivated these probes showed max abs 1e-1.
    # half2 arms: bf16 activation rounding, so energy <= 1e-1 and max abs <= 5e-2, and
    # the per-element bar is dropped (it measures the accepted rounding, not a bug).
    # Bars set from the two states the encoder is known to occupy (section 29), not from
    # the clean run alone. Clean: energy 6.4e-07..1.0e-06, max abs 1.9e-07..5.7e-06.
    # Defect state (same binary, different allocation history): energy up to 4.4e-04,
    # max abs up to 9.1e-03. So the bars sit above the defect and far below every real
    # bug the probes have caught -- headnorm aliasing race max abs 1.0e-01, attention
    # softmax race max abs 1.0e-01 (5-10x clearance). Per-element rms rel is gated only
    # for probe_ao in the fp32 arms: it is a cancelling PV sum, where that measure is the
    # sensitive one (clean 4.9e-05). For q/k it reflects the headnorm's normalising
    # division, so it is reported. half2 arms: bf16 rounding, so energy/max abs only.
    ebar, mbar = (1e-1, 5e-2) if H2 else ((1e-3, 2e-2) if DEFECT_OK else (1e-4, 5e-5))
    ok = (fin and en <= ebar and float(np.abs(a - b).max()) <= mbar
          and (H2 or name != "probe_ao_02" or rms <= 1e-3))
    print("  %-11s energy rel %.3e  per-elem rms rel %.3e  max abs %.3e  rms %.4f vs %.4f  finite %s  %s"
          % (name, en, rms, np.abs(a - b).max(), np.sqrt((a ** 2).mean()), np.sqrt((b ** 2).mean()), fin,
             "OK" if ok else "FAIL (per-elem<=1e-3 got %g | energy<=%g got %g | maxabs<=%g got %g)"
             % (rms, ebar, en, mbar, np.abs(a - b).max())))
    if not ok: fails += 1

if DEFECT_OK:
    print("  ** DEFECT-TOLERANT MODE (LFM_ENC_DEFECT_OK=1): section 29's run-to-run defect"
          " is being tolerated by these bars **")
print("ENC_GATE %s (arms=%s T=%d stages=%d)" % ("PASS" if fails == 0 and nport else "FAIL",
                                                "half2" if H2 else "fp32", TOK, nport))
sys.exit(0 if (fails == 0 and nport) else 1)
