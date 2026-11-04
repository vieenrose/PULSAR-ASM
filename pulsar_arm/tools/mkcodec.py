#!/usr/bin/env python
"""mkcodec.py -- export the LiquidCodec DECODER chain to a .lfm3q8 blob (S5).

Chain (modeling_liquid_codec.py): codes [12,T] -> FSQ dequant -> quantizer.project_out
-> from_quantizer -> nearest x6 upsample -> 8-layer LFM3 (conv at 0,1,3,5,7 / sliding
attention at 2,4,6, window 30, sandwich norms) -> decoder.lin -> polar -> iSTFT.
The ENCODER side (frontend/LFM/to_quantizer/project_in) is a separate blob later (M3).

Export gates (README-lfm3-audio.md "Export gates"): Q8 tensors selfcheck (dequant vs
source fp32, rel err + top1 argmax on the fp32 GEMV), tensor-coverage gate (every name
the C engine needs), tie gate (duplicated rows), and a manifest of the derived
constants (FSQ levels/basis, layer types, rope). SELFCHECK OK is printed only when all
pass; the C engine refuses a blob without it."""
import argparse
import json
import math
import os
import struct
import time
import numpy as np
import torch
from safetensors import safe_open

MAGIC = b"LFM3Q8\x01\x00"
T_Q8 = 1
T_F32 = 2

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

def dequant_q8(codes, scales):
    return codes.astype(np.float32) * np.repeat(scales, 32, axis=1).astype(np.float32)

def crc32(*chunks):
    import zlib
    h = 0
    for ch in chunks:
        h = zlib.crc32(ch, h)
    return h & 0xFFFFFFFF

REPO_LOCAL = os.path.expanduser("~/models/lfm3/LFM3-Audio-2B")

Q8_PATTERNS = (
    "feed_forward.w1", "feed_forward.w2", "feed_forward.w3",
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "self_attn.gate_proj",
    "conv.in_proj.weight", "conv.out_proj.weight",
    "from_quantizer.weight", "decoder.lin.weight",
)

# the exact tensor list the S5 engine reads (coverage gate)
def required_names(nq=12):
    names = ["quantizer.project_out.weight", "quantizer.project_out.bias",
             "from_quantizer.weight", "from_quantizer.bias",
             "decoder.lfm.embedding_norm.weight", "decoder.lin.weight", "decoder.lin.bias",
             "decoder.istft.window"]
    for L in range(8):
        p = f"decoder.lfm.layers.{L}"
        names += [f"{p}.operator_norm.weight", f"{p}.post_operator_norm.weight",
                  f"{p}.ffn_norm.weight", f"{p}.post_ffn_norm.weight",
                  f"{p}.feed_forward.w1.weight", f"{p}.feed_forward.w2.weight", f"{p}.feed_forward.w3.weight"]
        if L in (2, 4, 6):
            names += [f"{p}.self_attn.q_proj.weight", f"{p}.self_attn.k_proj.weight",
                      f"{p}.self_attn.v_proj.weight", f"{p}.self_attn.o_proj.weight",
                      f"{p}.self_attn.gate_proj.weight",
                      f"{p}.self_attn.q_norm.weight", f"{p}.self_attn.k_norm.weight"]
        else:
            names += [f"{p}.conv.conv.weight", f"{p}.conv.in_proj.weight", f"{p}.conv.out_proj.weight"]
    return names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=REPO_LOCAL)
    ap.add_argument("--out", default="/home/louis/models/lfm3/lfm3-audio-2b-codec.lfm3q8")
    a = ap.parse_args()

    cfg_src = json.load(open(os.path.join(a.model, "audio_tokenizer", "config.json")))
    levels = list(cfg_src["fsq_levels"])
    n_q = int(cfg_src["n_q"])
    assert math.prod(levels) == 2016, "FSQ levels product changed"
    basis = [math.prod(levels[:i]) for i in range(len(levels))]

    need = set(required_names(n_q))
    f = safe_open(os.path.join(a.model, "audio_tokenizer", "model.safetensors"), "pt")
    have = set(f.keys())
    missing = sorted(need - have)
    if missing:
        raise SystemExit(f"TENSOR-COVERAGE GATE FAIL: missing {missing}")

    entries, blobs, off = [], [], 0
    n_f32 = 0
    selfcheck = []
    t0 = time.time()
    for k in sorted(need):
        t = f.get_tensor(k).to("cpu").to(torch.float32).numpy()
        is_w = k.endswith(".weight")
        q8 = is_w and any(p in k for p in Q8_PATTERNS) and t.ndim == 2
        if q8:
            assert t.shape[1] % 32 == 0, "%s K=%d not %%32" % (k, t.shape[1])
            codes, scales = quantize_q8(t)
            dq = dequant_q8(codes, scales)
            rel = float(np.sqrt(((dq - t) ** 2).mean()) / np.sqrt((t ** 2).mean()))
            x = np.random.RandomState(7).randn(t.shape[1]).astype(np.float32)
            am_ok = int((dq @ x).argmax() == int((t @ x).argmax()))
            selfcheck.append((k, "q8", rel, am_ok))
            entries.append(dict(name=k, dtype=T_Q8, ne=list(t.shape), off=off,
                                ncodes=codes.nbytes, nscales=scales.size,   # nscales counts FLOATS (matches the C loader)
                                crc=crc32(codes.tobytes(), scales.tobytes())))
            blobs += [codes.tobytes(), scales.tobytes()]
            off += codes.nbytes + scales.nbytes
        else:
            if is_w and t.ndim == 2:
                dup = t.shape[0] - len(np.unique(t, axis=0))
                if dup:
                    raise SystemExit(f"TIE GATE FAIL: {k} has {dup} duplicated rows")
            w = np.ascontiguousarray(t)
            entries.append(dict(name=k, dtype=T_F32, ne=list(w.shape), off=off,
                                ncodes=w.nbytes, nscales=0, crc=crc32(w.tobytes())))
            blobs.append(w.tobytes()); off += w.nbytes
            n_f32 += 1

    cfg = dict(component="lfm3-audio-codec-decoder",
               n_fft=cfg_src["stft_n_fft"], hop_length=cfg_src["stft_hop_length"],
               win_length=cfg_src["stft_win_length"], frame_stack=cfg_src["stft_frame_stack"],
               frame_samples=cfg_src["frame_samples"], sampling_rate=cfg_src["sampling_rate"],
               n_q=n_q, codebook_size=math.prod(levels), fsq_levels=levels, fsq_basis=basis,
               quantizer_dim=cfg_src["quantizer_dim"], encoder_dim=cfg_src["encoder_dim"],
               hidden_size=512, intermediate_size=2304, n_head=8, n_kv=2, head_dim=64,
               n_layers=8, layer_types=["conv" if i not in (2, 4, 6) else "sliding_attention" for i in range(8)],
               sliding_window=cfg_src["decoder_window"], conv_L_cache=7, norm_eps=1e-5,
               rope_theta=1_000_000.0, eos_code=2016)
    hdr = dict(magic=MAGIC.decode(), cfg=cfg, n_tensors=len(entries), data_bytes=off, table=entries)
    hj = json.dumps(hdr, separators=(",", ":")).encode()
    with open(a.out, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<I", len(hj)))
        fh.write(hj)
        for b in blobs:
            fh.write(b)

    worst = max((r for _, kind, r, _ in selfcheck if kind == "q8"), default=0.0)
    am_bad = [k for k, kind, _, ok in selfcheck if kind == "q8" and not ok]
    print("selfcheck: %d q8 tensors, worst rel %.2e, argmax mismatches %d"
          % (len(selfcheck), worst, len(am_bad)))
    for k, kind, r, ok in selfcheck:
        print(f"  {kind} {k:56s} rel {r:.2e} argmax {'OK' if ok else 'FAIL'}")
    ok = worst < 5e-2 and not am_bad
    print("SELFCHECK OK" if ok else "SELFCHECK FAIL")
    print("wrote %s: %.1f MB, %d tensors (%d f32 / %d Q8), %.1f s"
          % (a.out, (len(hj) + off) / 1e6, len(entries), n_f32, len(entries) - n_f32, time.time() - t0))
    if not ok:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
