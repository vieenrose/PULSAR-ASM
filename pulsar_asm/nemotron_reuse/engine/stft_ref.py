#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch
import numpy as np


def make_even(v):
    v = int(round(v))
    return v if v % 2 == 0 else v + 1


def stft_dft_matrix(n_fft):
    n = np.arange(n_fft)[:, None]
    k = np.arange(n_fft)[None, :]
    W = np.exp(-2j * np.pi * n * k / n_fft)
    return W.astype(np.complex64)


def stft_mag_pha(x, n_fft, hop_size, win_size, compress_factor='relu_log1p', center=True):
    # x: [B, T] or [T]
    if x.dim() == 1:
        x = x.unsqueeze(0)
    B, T = x.shape
    window = torch.hann_window(win_size, periodic=True).float()
    if center:
        pad = n_fft // 2
        x = torch.nn.functional.pad(x, (pad, pad), mode='reflect')
        T2 = T + 2 * pad
    else:
        T2 = T
    # frames
    num_frames = 1 + (T2 - n_fft) // hop_size
    # build frames
    W = stft_dft_matrix(n_fft)
    W_real = torch.from_numpy(W.real).float()
    W_imag = torch.from_numpy(W.imag).float()
    win = window
    mags = []
    phas = []
    for i in range(num_frames):
        start = i * hop_size
        frame = x[:, start:start + n_fft]
        frame = frame * win.unsqueeze(0)
        # DFT
        real = torch.matmul(frame, W_real.T)
        imag = torch.matmul(frame, W_imag.T)
        spec = torch.complex(real, imag)
        mag = torch.abs(spec)
        pha = torch.angle(spec)
        mags.append(mag)
        phas.append(pha)
    mag = torch.stack(mags, dim=2)  # [B, F, T]
    pha = torch.stack(phas, dim=2)
    if compress_factor in ['log1p', 'relu_log1p', 'signed_log1p']:
        mag_c = torch.log1p(mag)
    else:
        mag_c = mag ** compress_factor
    return mag_c, pha, mag
