#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from models.stfts import mag_phase_stft, mag_phase_istft
from models.generator_SEMamba_time_d4 import SEMamba
from utils.util import load_config, pad_or_trim_to_match
from huggingface_hub import hf_hub_download
from mamba_ref import MambaRef

RELU = nn.ReLU()


def make_even(value):
    value = int(round(value))
    return value if value % 2 == 0 else value + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--chunk', action='store_true')
    parser.add_argument('--chunk_size_in_seconds', type=float, default=5.0)
    parser.add_argument('--hop_length_portion', type=float, default=0.5)
    parser.add_argument('--dump_tensors', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = load_config(args.config)
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    sampling_rate = cfg['stft_cfg']['sampling_rate']

    device = torch.device('cpu')
    model = SEMamba.from_pretrained('nvidia/NVIDIA-NemotronLabs-RE-USE', cfg=cfg).to(device)
    model.eval()

    noisy_wav, noisy_sr = torchaudio.load(args.wav)
    if noisy_wav.shape[0] > 1:
        noisy_wav = noisy_wav[:1, :]
    noisy_wav = noisy_wav.to(device)

    n_fft_scaled = make_even(n_fft * noisy_sr // sampling_rate)
    hop_size_scaled = make_even(hop_size * noisy_sr // sampling_rate)
    win_size_scaled = make_even(win_size * noisy_sr // sampling_rate)

    meta = {
        'src': args.wav,
        'sr': noisy_sr,
        'n_fft': n_fft_scaled,
        'hop_size': hop_size_scaled,
        'win_size': win_size_scaled,
        'compress_factor': compress_factor,
        'sampling_rate_cfg': sampling_rate,
    }
    with open(os.path.join(args.outdir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    if args.dump_tensors:
        torch.save({'noisy_wav': noisy_wav.cpu()}, os.path.join(args.outdir, 'noisy_wav.pt'))

    with torch.no_grad():
        if not args.chunk:
            noisy_mag, noisy_pha, _ = mag_phase_stft(
                noisy_wav,
                n_fft=n_fft_scaled,
                hop_size=hop_size_scaled,
                win_size=win_size_scaled,
                compress_factor=compress_factor,
                center=True,
                addeps=False,
            )
            if args.dump_tensors:
                torch.save({'mag': noisy_mag.cpu(), 'pha': noisy_pha.cpu()}, os.path.join(args.outdir, 'stft.pt'))
            amp_g, pha_g, _ = model(noisy_mag, noisy_pha)
            mag = torch.expm1(RELU(amp_g))
            zero_portion = torch.sum(mag == 0, 1) / mag.shape[1]
            amp_g_masked = amp_g.clone()
            amp_g_masked[:, :, (zero_portion > 0.5)[0]] = 0
            mag_m = torch.expm1(RELU(amp_g_masked))
            audio_g = mag_phase_istft(amp_g_masked, pha_g, n_fft_scaled, hop_size_scaled, win_size_scaled, compress_factor)
            audio_g = pad_or_trim_to_match(noisy_wav.detach(), audio_g, pad_value=1e-8)
            torchaudio.save(os.path.join(args.outdir, 'enhanced.flac'), audio_g.cpu(), noisy_sr)
            torch.save({'audio_g': audio_g.cpu(), 'mag': mag_m.cpu(), 'pha': pha_g.cpu()}, os.path.join(args.outdir, 'output.pt'))
        else:
            import math
            chunk_size = int(args.chunk_size_in_seconds * noisy_sr)
            hop_length = int(args.hop_length_portion * chunk_size)
            window = torch.clamp(torch.hann_window(chunk_size), min=1e-7)
            audio_enhanced = torch.zeros_like(noisy_wav)
            window_sum = torch.zeros_like(noisy_wav)
            num_chunks = max(1, math.ceil((noisy_wav.shape[1] - chunk_size) / hop_length) + 1)
            for c in range(noisy_wav.shape[0]):
                noisy_wav_ch = noisy_wav[c:c+1, :]
                for i in range(num_chunks):
                    if i < num_chunks - 1:
                        noisy_wav_chunk = noisy_wav_ch[:, i * hop_length: i * hop_length + chunk_size]
                    else:
                        noisy_wav_chunk = noisy_wav_ch[:, -chunk_size:]
                    noisy_mag, noisy_pha, _ = mag_phase_stft(
                        noisy_wav_chunk,
                        n_fft=n_fft_scaled,
                        hop_size=hop_size_scaled,
                        win_size=win_size_scaled,
                        compress_factor=compress_factor,
                        center=True,
                        addeps=False,
                    )
                    amp_g, pha_g, _ = model(noisy_mag, noisy_pha)
                    mag = torch.expm1(RELU(amp_g))
                    zero_portion = torch.sum(mag == 0, 1) / mag.shape[1]
                    amp_g_masked = amp_g.clone()
                    amp_g_masked[:, :, (zero_portion > 0.5)[0]] = 0
                    audio_g = mag_phase_istft(amp_g_masked, pha_g, n_fft_scaled, hop_size_scaled, win_size_scaled, compress_factor)
                    audio_g = pad_or_trim_to_match(noisy_wav_chunk.detach(), audio_g, pad_value=1e-8)
                    if i < num_chunks - 1:
                        audio_enhanced[c:c+1, i * hop_length:i * hop_length + chunk_size] += audio_g * window[0:audio_g.shape[1]]
                        window_sum[c:c+1, i * hop_length:i * hop_length + chunk_size] += window[0:audio_g.shape[1]]
                    else:
                        audio_enhanced[c:c+1, -chunk_size:] += audio_g * window[0:audio_g.shape[1]]
                        window_sum[c:c+1, -chunk_size:] += window[0:audio_g.shape[1]]
            audio_enhanced = audio_enhanced / window_sum
            torchaudio.save(os.path.join(args.outdir, 'enhanced.flac'), audio_enhanced.cpu(), noisy_sr)
            torch.save({'audio_g': audio_enhanced.cpu()}, os.path.join(args.outdir, 'output.pt'))


if __name__ == '__main__':
    main()
