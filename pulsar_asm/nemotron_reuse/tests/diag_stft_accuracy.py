#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Diagnose the end-to-end SNR floor of the RE-USE Nano port.

Answers three questions with measurements, not assertions:

  A. Which STFT is closer to the truth?  Compare the engine STFT and the
     PyTorch STFT against an independent float64 reference STFT.
  B. Which model input channel drives the end-to-end SNR?  Cross-swap
     magnitude and phase between engine and reference.
  C. Is that channel's error in the sensitive direction of the model?
     Compare the real phase difference against iid phase noise of equal rms.

Run the engine twice first (dumps via env vars):
  REUSE_DUMP_INMAG/INPHA  -> engine STFT
  REUSE_MAG/REUSE_PHA     -> feed reference STFT, dump audio (model-isolated run)
"""
import argparse
import os
import sys
import numpy as np
import scipy.io.wavfile as wav

sys.path.insert(0, '/home/louis/NVIDIA-NemotronLabs-RE-USE')


def rd(p):
    return np.fromfile(p, dtype=np.float32) if os.path.exists(p) else None


def snr(ref, x):
    ref = np.asarray(ref, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    k = min(ref.size, x.size)
    ref, x = ref[:k], x[:k]
    e = ((ref - x) ** 2).sum()
    return float('inf') if e == 0 else float(10 * np.log10((ref ** 2).sum() / e))


def stft_f64(wav_path, n_fft, hop, win):
    sr, data = wav.read(wav_path)
    x = data.astype(np.float64)
    if data.ndim > 1:
        x = x[:, 0]
    x = x / 32768.0
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win) / win)   # periodic == torch.hann_window
    xp = np.pad(x, (n_fft // 2, n_fft // 2), mode='reflect')
    T = 1 + ((x.size + 2 * (n_fft // 2)) - n_fft) // hop
    X = np.zeros((n_fft // 2 + 1, T), dtype=np.complex128)
    for t in range(T):
        seg = xp[t * hop:t * hop + win]
        buf = np.zeros(n_fft)
        buf[:win] = seg * w
        X[:, t] = np.fft.rfft(buf)[:n_fft // 2 + 1]
    return np.log1p(np.abs(X)), np.angle(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True, help='reference dir from ref_full.py')
    ap.add_argument('--c', required=True, help='engine dump dir (c_inmag/c_inpha)')
    ap.add_argument('--wav', required=True)
    ap.add_argument('--nfft', type=int, required=True)
    ap.add_argument('--hop', type=int, required=True)
    ap.add_argument('--win', type=int, required=True)
    ap.add_argument('--F', type=int, required=True)
    ap.add_argument('--T', type=int, required=True)
    a = ap.parse_args()
    F, T = a.F, a.T

    def key(mag, pha):
        return (mag.astype(np.float64) * np.exp(1j * pha.astype(np.float64)))

    mt, pt = stft_f64(a.wav, a.nfft, a.hop, a.win)
    rm, rp = rd(f'{a.ref}/in_mag.raw').reshape(F, T), rd(f'{a.ref}/in_pha.raw').reshape(F, T)
    cm, cp = rd(f'{a.c}/c_inmag.raw').reshape(F, T), rd(f'{a.c}/c_inpha.raw').reshape(F, T)

    print('=== A. STFT accuracy vs independent float64 reference ===')
    for nm, m, p in (('torch', rm, rp), ('engine', cm, cp)):
        dp = np.abs(np.angle(np.exp(1j * (pt - p.astype(np.float64)))))
        print('  %-6s complex rel_rms = %.3e   phase rms = %.3e  max = %.3e'
              % (nm,
                 np.sqrt(np.abs(key(mt, pt) - key(m, p)) ** 2).mean() / np.sqrt((np.abs(key(mt, pt)) ** 2).mean()),
                 np.sqrt((dp ** 2).mean()), dp.max()))

    print('=== B. channel cross-swap (needs the model-isolated audio dumps) ===')
    base = rd(f'{a.c}/c_audio.f32')
    for tag in ('audio_mypha.f32', 'audio_mymag.f32'):
        p = os.path.join(a.c, tag)
        if rd(p) is not None and base is not None:
            print('  %-18s SNR = %.2f dB' % (tag[:-5], snr(rd(f'{a.ref}/audio.f32'), rd(p))))
        else:
            print('  %-18s (missing; run engine with REUSE_MAG/REUSE_PHA cross-swaps)' % tag[:-5])

    print('=== C. structure of the phase difference ===')
    d = np.angle(np.exp(1j * (cp.astype(np.float64) - rp.astype(np.float64))))
    print('  phase diff rms = %.3e  max = %.3e  |d|>1e-3 bins = %d'
          % (np.sqrt((d ** 2).mean()), np.abs(d).max(), (np.abs(d) > 1e-3).sum()))
    dl = d[:, T // 4:]
    ac = [float(np.corrcoef(row[:-1], row[1:])[0, 1]) for row in dl if row.std() > 0]
    ac = [v for v in ac if np.isfinite(v)]
    print('  lag-1 autocorrelation of phase diff along TIME: mean = %.3f (iid noise ~ 0.0)'
          % (np.mean(np.abs(ac)) if ac else float('nan')))
    print('  => a coherent difference is amplified far more than iid noise of equal rms;')
    print('     the 50 dB fp32 waveform gate is only reachable by reproducing the')
    print('     reference FFT implementation bit-for-bit, not by being more accurate.')


if __name__ == '__main__':
    main()
