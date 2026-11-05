"""decode_frames_codec.py -- standalone HF codec decode of recorded frames (M2 listening target).

Usage: decode_frames_codec.py <fixture_dir>
Reads audio_codes.i32 [12,nframes] (the RETURNED codes incl. the EOS frame) and writes
audio.f32 (24 kHz mono). Split out of speak_hf_oracle because generate and decode have
different costs: generate is 47-150 autoregressive backbone steps, decode is one codec
forward. Re-running generate just to reach the codec step wastes an HF run.
"""
import os
import sys

import numpy as np
import torch

G = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from codec_hf_oracle import load_codec  # noqa: E402  (the importlib-as-package trick lives here)

MODEL = os.environ.get("SPEAK_MODEL", "/home/louis/models/lfm3/LFM3-Audio-2B")


def main():
    codes = np.fromfile(os.path.join(G, "audio_codes.i32"), dtype=np.int32)
    n = codes.size // 12
    assert codes.size == n * 12, codes.size
    codes = codes.reshape(12, n)
    # the EOS frame (all-2016 by HF's override rule) carries no audio: HF's decode
    # validates codes in [0, 2016), so trailing all-EOS frames are dropped, not decoded
    # (first attempt died here with ValueError on the EOS frame).
    while n > 0 and bool((codes[:, n - 1] == 2016).all()):
        n -= 1
    codes = codes[:, :n]
    print("kept %d audio frames (dropped trailing EOS)" % n)
    codec = load_codec()
    with torch.no_grad():
        audio = codec.decode(torch.tensor(codes.reshape(12, n))[None])
    a = audio[0].float().cpu().numpy()
    a.tofile(os.path.join(G, "audio.f32"))
    print("decoded %d frames -> %d samples (%.3f s at 24 kHz) -> %s/audio.f32"
          % (n, a.size, a.size / 24000.0, G))


if __name__ == "__main__":
    main()
