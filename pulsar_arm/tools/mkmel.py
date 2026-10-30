#!/usr/bin/env python3
"""mkmel.py -- build the log-mel front-end tables the LFM3-Audio runtime loads.

Emits `mel.bin`: the exact float32 constants the runtime's AArch64 kernels
(pulsar_arm/asm/lfm3_mel.S) consume, produced from the SAME code the reference
feature extractor uses, so the runtime cannot drift from HF by construction:

  * the 128 x 201 Slaney mel filterbank -- `transformers.audio_utils.mel_filter_bank`,
    the exact call in feature_extraction_lfm3_audio.py:Lfm3AudioFeatureExtractor.__init__
    (num_frequency_bins=n_fft//2+1, min 0, max sampling_rate/2, norm="slaney",
    mel_scale="slaney"),
  * the window -- `torch.hann_window(win_length)`, the exact call in `_log_mel`
    (periodic, float32, torch's own rounding),
  * the DFT twiddles -- 201 bins x 200 taps of cos/sin(2*pi*k*n/400), precomputed
    so the kernel's inner loop is sequential vector loads (no lane-wise gather,
    which NEON cannot do, and no per-tap modulo).

The precomputation is also why the kernel can use the half-length pair form:
for a real frame y of length N=400,

    X[k] = sum_{n<200} (y[n] + (-1)^k y[n+200]) * exp(-2i*pi*k*n/400)

so one 200-tap dot per bin and two 200-float scratch arrays per frame.

Run inside the model's venv (needs torch + transformers):
  ~/venvs/lfm3/bin/python pulsar_arm/tools/mkmel.py --out mel.bin
  ~/venvs/lfm3/bin/python pulsar_arm/tools/mkmel.py --config ~/models/lfm3/LFM3-Audio-2B

Everything the runtime needs from `config.json -> preprocessor` is validated
against the values this tool was written for; a config that differs aborts
rather than silently producing a mel.bin the engine misreads.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np
import torch
from transformers.audio_utils import mel_filter_bank

MAGIC = b"LFM3MEL1"

# the exact expectations of lfm3_mel.S / lfm3_mel.c
EXPECT = dict(sample_rate=16000, n_fft=400, hop_length=160, win_length=400, n_mels=128,
              logmel_floor=-12.0, offset=4.0, divisor=4.0, f_min=0.0, f_max=8000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.expanduser("~/models/lfm3/LFM3-Audio-2B"),
                    help="model directory (reads config.json) or a config.json path")
    ap.add_argument("--out", default="mel.bin")
    a = ap.parse_args()

    cfg_path = a.config
    if os.path.isdir(cfg_path):
        cfg_path = os.path.join(cfg_path, "config.json")
    cfg = json.load(open(cfg_path))
    pp = cfg.get("preprocessor", cfg)          # accept a bare preprocessor block too

    def get(key, want):
        got = pp.get(key, want)
        if got != want:
            sys.exit("preprocessor.%s = %r, this build tool only supports %r" % (key, got, want))
        return want

    E = {k: get(k, v) for k, v in EXPECT.items()}
    norm = pp.get("normalization", {})
    if norm.get("type", "affine") != "affine":
        sys.exit("only affine log-mel normalization is supported")
    E["offset"] = norm.get("offset", 4.0)
    E["divisor"] = norm.get("divisor", 4.0)
    for key, want in (("window", "hann"), ("window_periodic", True), ("center", False),
                      ("normalized", False), ("onesided", True), ("power", 2.0),
                      ("mel_scale", "slaney"), ("mel_norm", "slaney"), ("log_base", 10),
                      ("feature_type", "log_mel")):
        got = pp.get(key, want)
        if got != want:
            sys.exit("preprocessor.%s = %r, unsupported by this build tool" % (key, got))
    pad = pp.get("padding", {})
    if pad.get("side", "right") != "right" or pad.get("value", 0.0) != 0.0:
        sys.exit("only right zero padding is supported (padding=%r)" % pad)

    n_fft, hop, win, n_mels, sr = E["n_fft"], E["hop_length"], E["win_length"], E["n_mels"], E["sample_rate"]
    n_bins = n_fft // 2 + 1
    bins_pad = (n_bins + 7) & ~7                  # 208: zero-padded so the kernel needs no tail

    # --- the reference's own constants, in the reference's own dtypes ---------
    filters = mel_filter_bank(num_frequency_bins=n_fft // 2 + 1, num_mel_filters=n_mels,
                              min_frequency=E["f_min"], max_frequency=E["f_max"],
                              sampling_rate=sr, norm="slaney", mel_scale="slaney")
    filters = np.asarray(filters, dtype=np.float32)
    assert filters.shape == (n_bins, n_mels), filters.shape       # mel_filter_bank returns (bins, mels)
    window = torch.hann_window(win, dtype=torch.float32).numpy().astype(np.float32)

    # --- twiddles: cos/sin(2*pi*k*n/400), k < 201, n < 200 --------------------
    k = np.arange(n_bins, dtype=np.int64)[:, None]
    n = np.arange(n_fft // 2, dtype=np.int64)[None, :]
    ang = 2.0 * np.pi * (k * n % n_fft) / n_fft   # exact integer index, then one transcendental
    twc = np.cos(ang).astype(np.float32)
    tws = np.sin(ang).astype(np.float32)

    filt = np.zeros((n_mels, bins_pad), dtype=np.float32)
    filt[:, :n_bins] = filters.T                  # row m = filter m, columns = bins, zero-padded

    with open(a.out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<8i4f", n_fft, hop, win, n_mels, n_bins, bins_pad, 0, 0,
                            float(E["logmel_floor"]), float(E["offset"]), float(E["divisor"]), 0.0))
        f.write(window.tobytes())
        f.write(np.ascontiguousarray(twc).tobytes())
        f.write(np.ascontiguousarray(tws).tobytes())
        f.write(filt.tobytes())
    sz = os.path.getsize(a.out)
    print("wrote %s: %d B -- n_fft %d hop %d win %d mels %d bins %d(pad %d) "
          "floor %g offset %g divisor %g" %
          (a.out, sz, n_fft, hop, win, n_mels, n_bins, bins_pad,
           E["logmel_floor"], E["offset"], E["divisor"]))
    print("  window[0]=%.9g window[200]=%.9g  filters row0 nnz=%d  filt sum=%.9g" %
          (window[0], window[200], int((filters[:, 0] != 0).sum()), float(filt.sum())))


if __name__ == "__main__":
    main()
