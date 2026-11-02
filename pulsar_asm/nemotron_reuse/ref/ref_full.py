#!/usr/bin/env python3
# Self-contained reference: mirrors inference.py without torchaudio.
import os, sys, argparse
import numpy as np, torch, torch.nn as nn, scipy.io.wavfile as wav
sys.path.insert(0, '/home/louis/NVIDIA-NemotronLabs-RE-USE')
from models.stfts import mag_phase_stft, mag_phase_istft
from models.generator_SEMamba_time_d4 import SEMamba
from utils.util import load_config, pad_or_trim_to_match
REPO = '/home/louis/NVIDIA-NemotronLabs-RE-USE'
RECIPE = 'USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml'
RELU = nn.ReLU()
def make_even(v):
    v = int(round(v)); return v if v % 2 == 0 else v + 1
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wav', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = load_config(os.path.join(REPO, 'recipes', RECIPE))
    n_fft, hop_size, win_size = (cfg['stft_cfg'][k] for k in ('n_fft', 'hop_size', 'win_size'))
    cf = cfg['model_cfg']['compress_factor']; base_sr = cfg['stft_cfg']['sampling_rate']
    m = SEMamba.from_pretrained('nvidia/NVIDIA-NemotronLabs-RE-USE', cfg=cfg).eval()
    sr, data = wav.read(a.wav)
    if data.ndim > 1: data = data[:, 0]
    x = torch.from_numpy(data.astype(np.float32) / 32768.0).unsqueeze(0)
    nf, nh, nw = make_even(n_fft*sr//base_sr), make_even(hop_size*sr//base_sr), make_even(win_size*sr//base_sr)
    with torch.no_grad():
        mag, pha, _ = mag_phase_stft(x, n_fft=nf, hop_size=nh, win_size=nw, compress_factor=cf,
                                     center=True, addeps=False)
        amp_g, pha_g, _ = m(mag, pha)
        mg = torch.expm1(RELU(amp_g))
        zp = torch.sum(mg == 0, 1) / mg.shape[1]
        amp_g[:, :, (zp > 0.5)[0]] = 0
        audio = mag_phase_istft(amp_g, pha_g, nf, nh, nw, cf)
        audio = pad_or_trim_to_match(x, audio, pad_value=1e-8)
    os.makedirs(a.out, exist_ok=True)
    amp_g.numpy().astype(np.float32).tofile(f'{a.out}/amp.raw')
    pha_g.numpy().astype(np.float32).tofile(f'{a.out}/pha.raw')
    mag.numpy().astype(np.float32).tofile(f'{a.out}/in_mag.raw')
    pha.numpy().astype(np.float32).tofile(f'{a.out}/in_pha.raw')
    au = audio.numpy().astype(np.float32).ravel()
    au.tofile(f'{a.out}/audio.f32')
    pcm = (np.clip(au, -1.0, 1.0) * 32767.0).astype(np.int16)
    wav.write(f'{a.out}/enhanced.wav', sr, pcm)
    np.save(f'{a.out}/meta.npy', np.array([sr, nf, nh, nw, mag.shape[1], mag.shape[2], au.size]))
    print('sr', sr, 'nfft', nf, 'hop', nh, 'F', mag.shape[1], 'T', mag.shape[2], 'n', au.size)
main()
