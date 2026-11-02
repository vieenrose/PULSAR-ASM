#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Dump the input and output of TFMamba block index N so the C engine can be gated."""
import os, sys, argparse
import numpy as np
import scipy.io.wavfile as wav

sys.path.insert(0, os.environ.get('REUSE_REPO', '/home/louis/NVIDIA-NemotronLabs-RE-USE'))
import torch
from models.stfts import mag_phase_stft
from models.generator_SEMamba_time_d4 import SEMamba
from utils.util import load_config

REPO = os.environ.get('REUSE_REPO', '/home/louis/NVIDIA-NemotronLabs-RE-USE')
me = lambda v: (lambda q: q if q % 2 == 0 else q + 1)(int(round(v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wav', required=True)
    ap.add_argument('--block', type=int, default=0)
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--config', default=os.path.join(REPO, 'recipes',
        'USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml'))
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = SEMamba.from_pretrained('nvidia/NVIDIA-NemotronLabs-RE-USE', cfg=cfg).eval()
    sr, data = wav.read(args.wav)
    if data.ndim > 1:
        data = data[:, 0]
    x = torch.from_numpy(data.astype(np.float32) / 32768.0).unsqueeze(0)
    s = cfg['stft_cfg']
    nfft, hop, win = me(s['n_fft']*sr//s['sampling_rate']), me(s['hop_size']*sr//s['sampling_rate']), me(s['win_size']*sr//s['sampling_rate'])

    with torch.no_grad():
        mag, pha, _ = mag_phase_stft(x, n_fft=nfft, hop_size=hop, win_size=win,
                                     compress_factor=cfg['model_cfg']['compress_factor'],
                                     center=True, addeps=False)
        nm = mag.permute(0, 2, 1).unsqueeze(1)
        np_ = pha.permute(0, 2, 1).unsqueeze(1)
        B, C, T, F = nm.shape
        nm = torch.cat((nm, torch.zeros(B, C, T, 2)), -1)
        np_ = torch.cat((np_, torch.zeros(B, C, T, 2)), -1)
        xc = torch.cat((nm, np_), 1)
        xc = torch.cat((xc, torch.zeros(B, 2, 2, F + 2)), -2)
        h = model.dense_encoder(xc)
        print('enc', tuple(h.shape))
        for i, blk in enumerate(model.TSMamba):
            if i == args.block:
                h[0].numpy().astype(np.float32).tofile(args.prefix + '_in.raw')
                out = blk(h)
                out[0].numpy().astype(np.float32).tofile(args.prefix + '_out.raw')
                print('block', i, 'in/out', tuple(h.shape), tuple(out.shape))
                break
            h = blk(h)


if __name__ == '__main__':
    main()
