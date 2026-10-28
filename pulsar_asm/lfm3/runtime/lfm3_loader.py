#!/usr/bin/env python3
import re
"""Build LFM3 engine context from manifest - mirrors gemma4_model pattern."""
import json
import os
import struct
import ctypes
import numpy as np

from runtime import pulsar_abi as abi


def _fbits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def build_context(man, weights_path, CTR, D, DESC_SIZE, max_seq=1024, n_threads=4, maxb=1):
    m = man["meta"]
    blob = abi.MappedFile(weights_path)
    base = blob.addr

    def wp(name):
        e = man["layout"].get(name)
        if not e or e.get("kind") == "none" or e.get("offset", 0) == 0:
            return 0
        return base + e["offset"]

    H = m["hidden"]
    V = m["vocab_size"]
    NL = m["n_layers"]
    NH = m["num_heads"]
    NKV = m["num_kv_heads"]
    HD = m["head_dim"]
    inter = m["intermediate_size"]
    types = m["layer_types"]
    sliding_window = m.get("sliding_window", 256)
    conv_L = m.get("conv_L_cache", 7)
    use_sandwich = m.get("use_sandwich_norm", True)
    rope_params = m.get("rope_parameters", {})

    # Build rope tables
    pos = np.arange(max_seq, dtype=np.float64)[:, None]
    slide_p = rope_params.get("sliding_attention", {"rope_theta": 10000.0})
    full_p = rope_params.get("full_attention", {"rope_theta": 1000000.0, "partial_rotary_factor": 0.5})

    def inv_freq_partial(hd, theta, partial=1.0):
        n = hd // 2
        nrot = max(1, int(partial * hd // 2))
        rot = 1.0 / (theta ** (np.arange(0, 2 * nrot, 2, dtype=np.float64) / hd))
        return np.concatenate([rot, np.zeros(n - nrot)])

    ang_s = pos * inv_freq_partial(HD, slide_p.get("rope_theta", 10000.0), 1.0)
    ang_f = pos * inv_freq_partial(HD, full_p.get("rope_theta", 1000000.0), full_p.get("partial_rotary_factor", 0.5))
    cos_s = np.ascontiguousarray(np.cos(ang_s), dtype=np.float32)
    sin_s = np.ascontiguousarray(np.sin(ang_s), dtype=np.float32)
    cos_f = np.ascontiguousarray(np.cos(ang_f), dtype=np.float32)
    sin_f = np.ascontiguousarray(np.sin(ang_f), dtype=np.float32)

    ctx_size = CTR.get("SIZE", 512)
    ctx_mem = np.zeros(ctx_size, dtype=np.uint8)
    ctx_mem = np.ascontiguousarray(ctx_mem)

    def put(name, val):
        off = CTR[name]
        if isinstance(val, (int, np.integer)):
            struct.pack_into("<Q", ctx_mem, off, int(val))
        else:
            struct.pack_into("<I", ctx_mem, off, _fbits(val))

    put("WBASE", base)
    put("NLAYER", NL)
    put("VOCAB", V)
    put("HIDDEN", H)
    put("NHEAD", NH)
    put("NKVHEAD", NKV)
    put("HEADDIM", HD)
    put("SLIDING_WINDOW", sliding_window)
    put("INTER", inter)
    put("MAXB", maxb)
    put("EMB", wp("embed_tokens"))
    put("FINAL_NORM", wp("embedding_norm"))
    put("RMS_EPS", _fbits(m.get("rms_norm_eps", 1e-5)))
    put("B", 1)
    put("XSTRIDE", H * 4)

    # buffers
    _keep = [ctx_mem, cos_s, sin_s, cos_f, sin_f]
    buf = {}
    for k, n in (("X", maxb * H), ("T1", maxb * H), ("T2", maxb * H), ("T3", maxb * H),
                 ("Q", maxb * NH * HD), ("K", maxb * NKV * HD), ("V", maxb * NKV * HD),
                 ("A", maxb * NH * HD), ("SCORE", max_seq), ("G", maxb * inter), ("U", maxb * inter),
                 ("LOGITS", maxb * V)):
        buf[k] = np.zeros(n, dtype=np.float32)
        _keep.append(buf[k])

    # conv states: for each conv layer, cache of shape [maxb, H, conv_L_cache]
    conv_state = np.zeros((maxb, H * NL * conv_L), dtype=np.float32)
    _keep.append(conv_state)

    for k, a in buf.items():
        put(f"BUF_{k}", a.ctypes.data)
    put("BUF_CONV_STATE", conv_state.ctypes.data)
    put("COS_SLIDE", cos_s.ctypes.data)
    put("SIN_SLIDE", sin_s.ctypes.data)
    put("COS_FULL", cos_f.ctypes.data)
    put("SIN_FULL", cos_f.ctypes.data)
    put("ROPE_STR_SLIDE", cos_s.shape[1] * 4)
    put("ROPE_STR_FULL", cos_f.shape[1] * 4)

    # descriptors
    desc_count = NL
    desc = np.zeros((desc_count, DESC_SIZE // 8), dtype=np.uint64)
    DSL_map = {name: i for i, name in enumerate([k for k in D.keys()])}
    # Build by index based on order
    for i in range(NL):
        p = f"L{i:02d}."
        lt = types[i]
        if lt == "conv":
            ltype = 0
            flags = 0
        elif lt == "full_attention":
            ltype = 1
            flags = 1  # FLAG_FULL
        elif lt == "sliding_attention":
            ltype = 2
            flags = 2  # FLAG_SLIDING
        else:
            ltype = 0
            flags = 0

        # fill descriptor slots
        desc[i, D.get('QW', 0)//8] = wp(p + "w_q") if lt != "conv" else 0
        desc[i, D.get('KW', 1)//8] = wp(p + "w_k") if lt != "conv" else 0
        desc[i, D.get('VW', 2)//8] = wp(p + "w_v") if lt != "conv" else 0
        desc[i, D.get('OW', 3)//8] = wp(p + "w_o") if lt != "conv" else 0
        desc[i, D.get('GATE', 4)//8] = wp(p + "gate_proj") if lt != "conv" else 0
        desc[i, D.get('UP', 5)//8] = wp(p + "mlp_w3")
        desc[i, D.get('DOWN', 6)//8] = wp(p + "mlp_w2")
        desc[i, D.get('QNORM', 7)//8] = wp(p + "norm_q") if lt != "conv" else 0
        desc[i, D.get('KNORM', 8)//8] = wp(p + "norm_k") if lt != "conv" else 0
        desc[i, D.get('NORM_OP', 9)//8] = wp(p + "norm_op")
        desc[i, D.get('NORM_FFN', 10)//8] = wp(p + "norm_ffn")
        desc[i, D.get('NORM_POST_OP', 11)//8] = wp(p + "norm_post_op") if use_sandwich else 0
        desc[i, D.get('NORM_POST_FFN', 12)//8] = wp(p + "norm_post_ffn") if use_sandwich else 0
        desc[i, D.get('CONV_IN_PROJ', 13)//8] = wp(p + "conv_in_proj") if lt == "conv" else 0
        desc[i, D.get('CONV_W', 14)//8] = wp(p + "conv_w") if lt == "conv" else 0
        desc[i, D.get('CONV_OUT_PROJ', 15)//8] = wp(p + "conv_out_proj") if lt == "conv" else 0
        desc[i, D.get('INTER', 16)//8] = inter
        desc[i, D.get('HEADDIM', 17)//8] = HD
        desc[i, D.get('LAYER_TYPE', 18)//8] = ltype
        desc[i, D.get('FLAGS', 19)//8] = flags

    put("DESC", desc.ctypes.data)

    # tokens
    tokens = np.zeros(maxb, dtype=np.int64)
    outtok = np.zeros(maxb, dtype=np.int64)
    _keep.extend([tokens, outtok, desc, blob])
    put("TOKENS", tokens.ctypes.data)
    put("OUTTOK", outtok.ctypes.data)

    # smp
    if n_threads != 1:
        smp = abi.exec_alloc(4096)
        ctypes.memset(smp, 0, 4096)
        ctypes.c_uint64.from_address(smp + 16).value = n_threads
        put("SMP", smp)
        put("NTHR", n_threads)
    else:
        put("SMP", 0)
        put("NTHR", 1)

    return {
        "ctx_mem": ctx_mem, "blob": blob, "_keep": _keep,
        "tokens": tokens, "outtok": outtok, "buf": buf,
        "man": man, "max_seq": max_seq
    }


def parse_abi(path, prefix="CTR_", desc_prefix="D_", extras=()):
    if not os.path.exists(path):
        return {}, {}, {}
    txt = open(path).read()

    def grab(prefix):
        out, seen = {}, {}
        for m in re.finditer(rf"^{prefix}([A-Z0-9_]+)\s+equ\s+(\S+)", txt, re.M):
            name, val = m.group(1), int(m.group(2), 0)
            out[name] = val
            if not name.endswith("SIZE"):
                seen.setdefault(val, []).append(name)
        dup = {v: names for v, names in seen.items() if len(names) > 1}
        if dup:
            raise ValueError(f"{prefix} offsets collide")
        return out

    import re as _re
    ctr = grab(prefix)
    d = grab(desc_prefix) if desc_prefix else {}
    extra = {}
    for name in extras:
        m = _re.search(rf"^{name}\s+equ\s+(\S+)", txt, _re.M)
        if m:
            extra[name] = int(m.group(1), 0)
    return ctr, d, extra
