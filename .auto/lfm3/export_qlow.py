#!/usr/bin/env python
"""Write a low-bit backbone blob: the shipped Q8 blob with every backbone linear that
gptq_lfm3.py quantized to 4 or 5 bits replaced by a pre-tiled dtype 6 (Q4) / 7 (Q5)
payload in the layout of pulsar_arm/asm/lfm3_qlow.cuh. Tensors GPTQ kept at q8 (and all
others: embedding, norms, conv taps, attention if kept) are copied byte for byte.

  python .auto/lfm3/export_qlow.py --q8 lfm3-audio-2b-q8.lfm3q8 --gptq /tmp/gptq5b.pt --out q5.lfm3q8
"""
import argparse, json, struct, zlib
import numpy as np


def pack(Q, d, mn, B):
    """Q [N, K] uint8 codes, d / mn [N, K/32] fp16 -> (codes bytes, scales bytes)"""
    N, K = Q.shape
    G, T = K // 32, N // 32
    q = Q.reshape(N, G, 32).astype(np.uint32)
    lo = [8 * i + b for i in range(4) for b in range(4)]
    hi = [k + 4 for k in lo]
    by = ((q[:, :, lo] & 15) | ((q[:, :, hi] & 15) << 4)).astype(np.uint8)         # [N, G, 16]
    by = by.reshape(T, 32, G, 16).transpose(0, 2, 1, 3).reshape(T, G, 512)
    if B == 5:
        h = np.zeros((N, G), np.uint32)
        for p in range(8):
            for b in range(4):
                k = 8 * (p >> 1) + 4 * (p & 1) + b
                h |= ((q[:, :, k] >> 4) & 1) << np.uint32(8 * b + p)
        hb = np.ascontiguousarray(h.reshape(T, 32, G).transpose(0, 2, 1)).astype("<u4").view(np.uint8).reshape(T, G, 128)
        by = np.concatenate([by, hb], axis=2)
    dd = (d.astype(np.float32) * 128.0).astype(np.float16)
    assert np.isfinite(dd).all()
    assert (dd.astype(np.float32) == d.astype(np.float32) * 128.0).all(), "d * 128 not exact in fp16"
    sc = np.stack([dd, mn.astype(np.float16)], -1)                                   # [N, G, 2]
    sc = sc.reshape(T, 32, G, 2).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(by).tobytes(), np.ascontiguousarray(sc).astype("<f2").tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q8", required=True)
    ap.add_argument("--gptq", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    res = torch.load(a.gptq)
    src = open(a.q8, "rb")
    magic = src.read(8)
    n = struct.unpack("<I", src.read(4))[0]
    hdr = json.loads(src.read(n))
    base = 12 + n
    out_tab, blobs, off, nlow, saved = [], [], 0, 0, 0
    for e in hdr["table"]:
        e = dict(e)
        hf = None
        if e["name"].startswith("lfm.model.layers.") and e["name"].endswith(".weight"):
            hf = "model.language_model.layers." + e["name"][len("lfm.model.layers."):-len(".weight")]
        if hf in res and res[hf][1] is not None:
            B, (Q, d, mn) = res[hf]
            Q, d, mn = Q.numpy(), d.numpy(), mn.numpy()
            assert list(Q.shape) == e["ne"], (e["name"], Q.shape, e["ne"])
            c, s = pack(Q, d, mn, B)
            saved += e["ncodes"] + e["nscales"] - len(c) - len(s)
            e.update(dtype=6 if B == 4 else 7, ncodes=len(c), nscales=len(s),
                     crc=zlib.crc32(s, zlib.crc32(c)) & 0xffffffff)
            payload = [c, s]; nlow += 1
        else:
            src.seek(base + e["off"])
            raw = src.read(e["ncodes"] + e["nscales"])
            payload = [raw]
        e["off"] = off
        off += sum(len(x) for x in payload)
        out_tab.append(e); blobs.extend(payload)
    hdr["table"] = out_tab; hdr["data_bytes"] = off
    hj = json.dumps(hdr, separators=(",", ":")).encode()
    with open(a.out, "wb") as f:
        f.write(magic); f.write(struct.pack("<I", len(hj))); f.write(hj)
        for b in blobs: f.write(b)
    print("wrote %s: %d low-bit tensors, %.1f MB data (%.1f MB smaller than the Q8 blob)"
          % (a.out, nlow, off / 1e6, saved / 1e6))


if __name__ == "__main__":
    main()
