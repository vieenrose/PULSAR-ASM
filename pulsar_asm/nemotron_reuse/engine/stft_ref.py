#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch
import numpy as np


def stft_mag_pha(x, n_fft, hop_size, win_size, compress_factor='relu_log1p', center=True):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    B, T = x.shape
    window = torch.hann_window(win_size, periodic=True).float()
    if center:
        pad = n_fft // 2
        x = torch.nn.functional.pad(x, (pad, pad), mode='reflect')
    n = np.arange(n_fft)[:, None]
    k = np.arange(n_fft)[None, :]
    W = np.exp(-2j * np.pi * n * k / n_fft)
    W_real = torch.from_numpy(W.real).float()
    W_imag = torch.from_numpy(W.imag).float()
    L = x.shape[-1]
    num_frames = 1 + (L - n_fft) // hop_size
    frames = []
    for i in range(num_frames):
        start = i * hop_size
        frame = x[:, start:start + n_fft]
        frames.append(frame)
    frames = torch.stack(frames, dim=1)  # [B, Fm, N]
    frames = frames * window.unsqueeze(0).unsqueeze(0)
    frames = frames.reshape(-1, n_fft)  # [B*Fm, N]
    real = frames @ W_real
    imag = frames @ W_imag
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    pha = torch.angle(spec)
    mag = mag.reshape(B, num_frames, n_fft // 2 + 1).permute(0, 2, 1)
    pha = pha.reshape(B, num_frames, n_fft // 2 + 1).permute(0, 2, 1)
    if compress_factor in ['log1p', 'relu_log1p', 'signed_log1p']:
        mag_c = torch.log1p(mag)
    else:
        mag_c = mag ** compress_factor
    return mag_c, pha
