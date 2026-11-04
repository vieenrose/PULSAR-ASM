#!/usr/bin/env python
"""mkcodec.py -- export LiquidCodec decoder weights to .lfm3q8 blob."""
import argparse
import json
import os
import struct
import time
import numpy as np
import torch
from safetensors import safe_open

MAGIC = b"LFM3Q8\x01\x00"
T_Q8 = 1
T_F32 = 2
T_BF16 = 3

def quantize_q8(w):
    N, K = w.shape
    assert K % 32 == 0
    g = w.reshape(N, K // 32, 32)
    m = np.abs(g).max(axis=-1)
    s = (m / np.float32(127.0)).astype(np.float32)
    safe = np.where(s == 0, np.float32(1.0), s)[:, :, None]
    c = np.round(g / safe)
    c = np.clip(c, -128.0, 127.0).astype(np.int8)
    c = np.where(s[:, :, None] == 0, np.int8(0), c)
    return np.ascontiguousarray(c.reshape(N, K)), np.ascontiguousarray(s.reshape(N, K // 32))

def fnv1a_fast(*chunks):
    import zlib
    h = 0
    for ch in chunks:
        h = zlib.crc32(ch, h)
    return h & 0xFFFFFFFF

REPO_LOCAL = os.path.expanduser("~/models/lfm3/LFM3-Audio-2B")

Q8_PATTERNS = (
    "feed_forward.w1", "feed_forward.w2", "feed_forward.w3",
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "in_proj.weight", "out_proj.weight",  # conv in/out proj
    "lin.weight",
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=REPO_LOCAL)
    ap.add_argument("--out", default="/home/louis/models/lfm3/lfm3-audio-2b-codec.lfm3q8")
    a = ap.parse_args()

    sp = os.path.join(a.model, "audio_tokenizer", "model.safetensors")
    f = safe_open(sp, "pt")
    # Only export decoder + istft window we need
    keys = sorted(k for k in f.keys() if k.startswith('decoder.'))
    print(f"Found {len(keys)} codec decoder tensors")

    entries, blobs, off = [], [], 0
    n_f32 = 0
    t0 = time.time()
    for k in keys:
        t = f.get_tensor(k).to("cpu")
        is_w = k.endswith(".weight")
        q8 = is_w and any(p in k for p in Q8_PATTERNS) and t.dim() == 2
        if q8:
            assert t.shape[1] % 32 == 0, "%s K=%d not %%32" % (k, t.shape[1])
            shp = list(t.shape)
            w = t.to(torch.float32).numpy()
            codes, scales = quantize_q8(w)
            entries.append(dict(name=k, dtype=T_Q8, ne=shp, off=off,
                                ncodes=codes.nbytes, nscales=scales.nbytes,
                                crc=fnv1a_fast(codes.tobytes(), scales.tobytes())))
            blobs += [codes.tobytes(), scales.tobytes()]
            off += codes.nbytes + scales.nbytes
        else:
            w = np.ascontiguousarray(t.to(torch.float32).numpy())
            entries.append(dict(name=k, dtype=T_F32, ne=list(w.shape), off=off,
                                ncodes=w.nbytes, nscales=0,
                                crc=fnv1a_fast(w.tobytes())))
            blobs.append(w.tobytes()); off += w.nbytes
            n_f32 += 1

    cfg = dict(component="lfm3-audio-codec", n_fft=1280, hop_length=320, win_length=1280,
               frame_size=1920, num_codebooks=12, codebook_size=2017, encoder_dim=512,
               decoder_window=30)
    hdr = dict(magic=MAGIC.decode(), cfg=cfg, n_tensors=len(entries),
               data_bytes=off, table=entries)
    hj = json.dumps(hdr, separators=(",", ":")).encode()
    with open(a.out, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<I", len(hj)))
        fh.write(hj)
        for b in blobs:
            fh.write(b)
    print("wrote %s: %.0f MB, %d tensors (%d f32 / %d Q8), %.1f s"
          % (a.out, (len(hj) + off) / 1e6, len(entries), n_f32,
             len(entries) - n_f32, time.time() - t0))

if __name__ == "__main__":
    main()
