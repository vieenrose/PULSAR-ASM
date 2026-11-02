#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Dump the DenseEncoder output for a WAV so the C engine can be gated against it."""
import os, sys, argparse
import numpy as np
import scipy.io.wavfile as wav

sys.path.insert(0, os.environ.get('REUSE_REPO', '/home/louis/NVIDIA-NemotronLabs-RE-USE'))
import torch
from models.stfts import mag_phase_stft
from models.generator_SEMamba_time_d4 import SEMamba
from utils.util import load_config

REPO = os.environ.get('REUSE_REPO', '/home/louis/NVIDIA-NemotronLabs-RE-USE')


def make_even(v):
    v = int(round(v))
    return v if v % 2 == 0 else v + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wav', required=True)
    ap.add_argument('--config', default=os.path.join(REPO, 'recipes',
        'USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml'))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = SEMamba.from_pretrained('nvidia/NVIDIA-NemotronLabs-RE-USE', cfg=cfg).eval()

    sr, data = wav.read(args.wav)
    if data.ndim > 1:
        data = data[:, 0]
    x = torch.from_numpy(data.astype(np.float32) / 32768.0).unsqueeze(0)

    s = cfg['stft_cfg']
    nfft = make_even(s['n_fft'] * sr // s['sampling_rate'])
    hop = make_even(s['hop_size'] * sr // s['sampling_rate'])
    win = make_even(s['win_size'] * sr // s['sampling_rate'])

    with torch.no_grad():
        mag, pha, _ = mag_phase_stft(x, n_fft=nfft, hop_size=hop, win_size=win,
                                     compress_factor=cfg['model_cfg']['compress_factor'],
                                     center=True, addeps=False)
        # replicate SEMamba.forward's front padding exactly
        nm = mag.permute(0, 2, 1).unsqueeze(1)
        np_ = pha.permute(0, 2, 1).unsqueeze(1)
        B, C, T, F = nm.shape
        nm = torch.cat((nm, torch.zeros(B, C, T, 2)), dim=-1)
        np_ = torch.cat((np_, torch.zeros(B, C, T, 2)), dim=-1)
        xc = torch.cat((nm, np_), dim=1)
        xc = torch.cat((xc, torch.zeros(B, 2, 2, F + 2)), dim=-2)
        enc = model.dense_encoder(xc)
    print('encoder', tuple(enc.shape), 'nfft', nfft, 'hop', hop)
    e = enc[0].numpy().astype(np.float32)
    with open(args.out, 'wb') as f:
        f.write(e.tobytes())
    with open(args.out + '.shape', 'w') as f:
        f.write(json_dumps(tuple(e.shape)) if False else repr(tuple(int(v) for v in e.shape)))


json_dumps = None

if __name__ == '__main__':
    main()
