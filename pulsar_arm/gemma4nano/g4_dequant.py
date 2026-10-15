"""Dequantization of the GGUF block formats used by Gemma 4 E2B (Q4_0, Q6_K, F16, F32).

Mirrors ggml's dequantize_row_q4_0 / dequantize_row_q6_K in float32, operation by
operation (d * (q - 8) for Q4_0; (d * sc) * q for Q6_K), so the values are the
ones ggml itself would produce.
"""
import numpy as np


def q4_0(raw, n):
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 18)
    assert b.shape[0] * 32 == n
    d = b[:, :2].copy().view(np.float16).astype(np.float32)          # (nb, 1)
    q = b[:, 2:]
    lo = (q & 0x0F).astype(np.int32) - 8
    hi = (q >> 4).astype(np.int32) - 8
    v = np.concatenate([lo, hi], axis=1).astype(np.float32)          # x[0..15], x[16..31]
    return (v * d).astype(np.float32).ravel()


def q6_k(raw, n):
    """block_q6_K: ql[128] qh[64] scales[16] (int8) d (f16) = 210 bytes / 256 values."""
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 210)
    assert b.shape[0] * 256 == n
    ql = b[:, 0:128].astype(np.int32)
    qh = b[:, 128:192].astype(np.int32)
    sc = b[:, 192:208].view(np.int8).astype(np.float32)
    d = b[:, 208:210].copy().view(np.float16).astype(np.float32)     # (nb, 1)
    y = np.empty((b.shape[0], 256), dtype=np.float32)
    for half in range(2):                       # two 128-value halves per block
        L = ql[:, 64 * half:64 * half + 64]
        H = qh[:, 32 * half:32 * half + 32]
        S = sc[:, 8 * half:8 * half + 8]
        q1 = (L[:, 0:32] & 0xF) | (((H >> 0) & 3) << 4)
        q2 = (L[:, 32:64] & 0xF) | (((H >> 2) & 3) << 4)
        q3 = (L[:, 0:32] >> 4) | (((H >> 4) & 3) << 4)
        q4 = (L[:, 32:64] >> 4) | (((H >> 6) & 3) << 4)
        for k, q in enumerate((q1, q2, q3, q4)):
            # sub-block scale index: is = l/16 (+0,+2,+4,+6 per quarter)
            s = np.repeat(S[:, 2 * k:2 * k + 2], 16, axis=1)
            ds = (d * s).astype(np.float32)                           # d * sc first
            y[:, 128 * half + 32 * k:128 * half + 32 * k + 32] = \
                (ds * (q - 32).astype(np.float32)).astype(np.float32)
    return y.ravel()


def tensor(gguf, name):
    """Whole tensor as float32, shaped [rows, cols] (GGUF dims reversed)."""
    e = gguf.tensors[name]
    raw = gguf.raw(name)
    t = e["type"]
    if t == "F32":
        v = np.frombuffer(raw, dtype=np.float32).copy()
    elif t == "F16":
        v = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif t == "Q4_0":
        v = q4_0(raw, e["n"])
    elif t == "Q6_K":
        v = q6_k(raw, e["n"])
    else:
        raise ValueError(t)
    return v.reshape(list(reversed(e["dims"]))) if len(e["dims"]) > 1 else v


def row(gguf, name, i):
    """Row i of a 2-D tensor (one token of an embedding table), float32."""
    e = gguf.tensors[name]
    cols = e["dims"][0]
    t = e["type"]
    if t == "Q6_K":
        rb = cols // 256 * 210
        raw = gguf.raw(name)[i * rb:(i + 1) * rb]
        return q6_k(raw, cols)
    if t == "Q4_0":
        rb = cols // 32 * 18
        return q4_0(gguf.raw(name)[i * rb:(i + 1) * rb], cols)
    raise ValueError(t)
