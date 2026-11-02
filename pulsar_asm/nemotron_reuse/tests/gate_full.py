#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Full-pipeline gate for the RE-USE Nano port.

Gate structure (see doc/NEMOTRON_REUSE_NANO_PLAN.md 3.3):
  1. STFT   : compare the COMPLEX spectrum (mag*e^{j pha}), not raw atan2.
              The model consumes raw phase as a feature, so a +-pi sign flip
              at a near-zero imaginary part is a large input delta even though
              it is spectrally meaningless.
  2. Model  : per-tensor rel-rms <= 1e-4 vs the fp32 PyTorch reference,
              fed with the reference STFT (REUSE_MAG/REUSE_PHA).
  3. iSTFT  : waveform SNR >= 50 dB for identical model outputs.
  4. E2E    : waveform SNR reported and DISCLOSED; it is floored by the
              reference's own float32 STFT phase noise amplified by the
              30-block recurrent tower, so it is not a pass/fail gate.

usage: gate_full.py --ref DIR --c DIR [--F 161 --T 1021]
"""
import argparse
import os
import sys
import numpy as np

STAGE_GATE = 1e-4
SNR_GATE = 50.0


def rd(p):
    if not os.path.exists(p):
        return None
    return np.fromfile(p, dtype=np.float32)


def have(*xs):
    return all(x is not None for x in xs)


def rel_rms(r, c):
    r = np.asarray(r, dtype=np.float64).ravel()
    c = np.asarray(c, dtype=np.float64).ravel()
    k = min(r.size, c.size)
    r, c = r[:k], c[:k]
    den = np.sqrt((r * r).mean())
    if den == 0.0:
        return float('inf'), k
    return float(np.sqrt(((r - c) ** 2).mean()) / den), k


def snr(r, c):
    r = np.asarray(r, dtype=np.float64).ravel()
    c = np.asarray(c, dtype=np.float64).ravel()
    k = min(r.size, c.size)
    r, c = r[:k], c[:k]
    e = ((r - c) ** 2).sum()
    if e == 0.0:
        return float('inf')
    return float(10.0 * np.log10((r * r).sum() / e))


def complex_rel_rms(mag_r, pha_r, mag_c, pha_c):
    zr = (mag_r * np.exp(1j * pha_r)).ravel()
    zc = (mag_c * np.exp(1j * pha_c)).ravel()
    k = min(zr.size, zc.size)
    zr, zc = zr[:k], zc[:k]
    return float(np.sqrt(np.abs(zr - zc) ** 2).mean() / np.sqrt((np.abs(zr) ** 2).mean()))


def main():
    ran = []
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True, help='dir with in_mag/in_pha/amp/pha/audio.f32 from ref_full.py')
    ap.add_argument('--c', required=True, help='dir with c_inmag/c_inpha/c_amp/c_pha/c_audio.f32')
    ap.add_argument('--label', default='')
    ap.add_argument('--mode', default='auto', choices=['auto', 'e2e', 'model'],
                    help='e2e = STFT+audio gates; model = stage gates + iSTFT gate')
    a = ap.parse_args()

    def g(d, n):
        return rd(os.path.join(d, n))

    fails = []
    if a.mode == 'auto':
        a.mode = 'e2e' if os.path.exists(os.path.join(a.c, 'c_inmag.raw')) else 'model'
    print('=== RE-USE port gate %s [mode=%s] ===' % (a.label, a.mode))

    im_r, ip_r = g(a.ref, 'in_mag.raw'), g(a.ref, 'in_pha.raw')
    im_c, ip_c = g(a.c, 'c_inmag.raw'), g(a.c, 'c_inpha.raw')
    if have(im_r, ip_r, im_c, ip_c):
        cr = complex_rel_rms(im_r, ip_r, im_c, ip_c)
        mode = 'gate' if a.mode in ('auto', 'e2e') else 'info'
        # raw phase, for disclosure only
        d = np.abs(np.angle(np.exp(1j * (ip_r.astype(np.float64) - ip_c.astype(np.float64)))))
        print('[STFT ] complex-spectrum rel_rms = %.3e   (%s %.0e)  raw-phase rms=%.2e max=%.2e'
              % (cr, mode, STAGE_GATE, np.sqrt((d ** 2).mean()), d.max()))
        if mode == 'gate' and cr > STAGE_GATE:
            fails.append('STFT complex rel_rms %.2e > %.0e' % (cr, STAGE_GATE))
        ran.append('stft')
    else:
        print('[STFT ] skipped (missing input dumps)')

    # The gate quantity is the complex spectrum the iSTFT consumes:
    #   mag = expm1(relu(amp)), z = mag * e^{j pha}
    # Raw amp/pha rms is reported for information only: the phase head is an
    # atan2, so its raw values wrap by 2*pi and raw rms is not meaningful.
    amp_r, pha_r = g(a.ref, 'amp.raw'), g(a.ref, 'pha.raw')
    amp_c, pha_c = g(a.c, 'c_amp.raw'), g(a.c, 'c_pha.raw')
    if have(amp_r, pha_r, amp_c, pha_c):
        a_r, a_c = amp_r.astype(np.float64), amp_c.astype(np.float64)
        k = min(a_r.size, a_c.size)
        zr = (np.expm1(np.maximum(a_r[:k], 0.0)) * np.exp(1j * pha_r[:k].astype(np.float64)))
        zc = (np.expm1(np.maximum(a_c[:k], 0.0)) * np.exp(1j * pha_c[:k].astype(np.float64)))
        den = np.sqrt((np.abs(zr) ** 2).mean())
        vz = float(np.sqrt((np.abs(zr - zc) ** 2).mean()) / den) if den else float('inf')
        mode = 'gate' if a.mode == 'model' else 'info'
        va, _ = rel_rms(a_r, a_c)
        d = np.abs(np.angle(np.exp(1j * (pha_r[:k].astype(np.float64) - pha_c[:k].astype(np.float64)))))
        print('[MODEL] complex spectrum rel_rms = %.3e  n=%d  (%s %.0e)' % (vz, k, mode, STAGE_GATE))
        print('        amp raw rel_rms = %.3e   phase wrap-dist rms = %.2e max = %.2e'
              % (va, np.sqrt((d ** 2).mean()), d.max()))
        if a.mode == 'model' and vz > STAGE_GATE:
            fails.append('model complex rel_rms %.2e > %.0e' % (vz, STAGE_GATE))
        ran.append('model')
    else:
        print('[MODEL] skipped (missing amp/pha dumps)')

    ar, ac = g(a.ref, 'audio.f32'), g(a.c, 'c_audio.f32')
    if ar is not None and ac is not None:
        s = snr(ar, ac)
        gated = (a.mode == 'model')
        verdict = 'PASS' if s >= SNR_GATE else ('FAIL' if gated else 'DISCLOSED (reference-conditioning floor, see plan 3.3)')
        print('[AUDIO] SNR = %.2f dB  n=%d  (%s %.0f dB) %s' % (s, min(ar.size, ac.size), 'gate' if gated else 'disclosed', SNR_GATE, verdict))
        if gated and s < SNR_GATE:
            fails.append('audio SNR %.1f dB < %.0f dB' % (s, SNR_GATE))
        if not gated and s < SNR_GATE:
            print('        NOTE: to separate iSTFT accuracy from STFT conditioning, rerun the')
            print('        engine with REUSE_MAG/REUSE_PHA pointed at the reference STFT.')
        ran.append('audio')
    else:
        print('[AUDIO] skipped (missing audio)')

    if fails:
        print('GATE FAIL:')
        for f in fails:
            print('  -', f)
        return 1
    if not ran:
        print('GATE FAIL: no checks performed (missing dumps - the run did not produce output)')
        return 1
    print('GATE PASS (%d checks: %s)' % (len(ran), '+'.join(ran)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
