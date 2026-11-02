#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import os, sys, argparse
import torch
import numpy as np


def snr(ref, est, eps=1e-12):
    ref = ref.float().reshape(-1)
    est = est.float().reshape(-1)
    n = min(ref.numel(), est.numel())
    ref = ref[:n]
    est = est[:n]
    noise = ref - est
    p_s = (ref ** 2).sum()
    p_n = (noise ** 2).sum()
    if p_n < eps:
        return 1e18
    return 10 * np.log10((p_s / p_n).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', required=True)
    parser.add_argument('--cand', required=True)
    parser.add_argument('--snr_min', type=float, default=50.0)
    args = parser.parse_args()
    ref = torch.load(args.ref, map_location='cpu')
    cand = torch.load(args.cand, map_location='cpu')
    s = snr(ref['audio_g'], cand['audio_g'])
    print(f'SNR={s:.2f} dB')
    sys.exit(0 if s >= args.snr_min else 1)


if __name__ == '__main__':
    main()
