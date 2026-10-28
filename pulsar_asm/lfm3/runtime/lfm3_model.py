"""Linux loader for the LFM3 text backbone assembly engine."""
import ctypes
import json
import os
import re
import struct
import time

import numpy as np

from . import pulsar_abi as abi

_HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_ASM = os.path.normpath(os.path.join(_HERE, "..", "engine", "lfm3_engine_flat.asm"))
ENGINE_SRC = ENGINE_ASM


def _fbits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def parse_abi(path=ENGINE_ASM, prefix="CTR_", desc_prefix="D_", extras=()):
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

    ctr = grab(prefix)
    d = grab(desc_prefix) if desc_prefix else {}
    extra = {}
    for name in extras:
        m = re.search(rf"^{name}\s+equ\s+(\S+)", txt, re.M)
        if m:
            extra[name] = int(m.group(1), 0)
    return ctr, d, extra


def inv_freq_partial(head_dim, theta, partial=1.0):
    n = head_dim // 2
    nrot = max(1, int(partial * head_dim // 2))
    rot = 1.0 / (theta ** (np.arange(0, 2 * nrot, 2, dtype=np.float64) / head_dim))
    return np.concatenate([rot, np.zeros(n - nrot)])


def rope_tables_lfm3(meta, max_seq):
    pos = np.arange(max_seq, dtype=np.float64)[:, None]
    rope_params = meta.get("rope_parameters", {})
    slide_p = rope_params.get("sliding_attention", {"rope_theta": 10000.0})
    full_p = rope_params.get("full_attention", {"rope_theta": 1000000.0, "partial_rotary_factor": 0.5})
    out = []
    ang = pos * inv_freq_partial(meta["head_dim"], slide_p.get("rope_theta", 10000.0), 1.0)
    out.append(np.ascontiguousarray(np.cos(ang), dtype=np.float32))
    out.append(np.ascontiguousarray(np.sin(ang), dtype=np.float32))
    ang = pos * inv_freq_partial(meta["head_dim"], full_p.get("rope_theta", 1000000.0), full_p.get("partial_rotary_factor", 0.5))
    out.append(np.ascontiguousarray(np.cos(ang), dtype=np.float32))
    out.append(np.ascontiguousarray(np.sin(ang), dtype=np.float32))
    return out


class LFM3:
    def __init__(self, weights, max_seq=1024, n_threads=4, maxb=1, manifest=None, verbose=True):
        t0 = time.time()
        abi.require_avx2_fma()
        if manifest is None:
            manifest = weights + ".manifest.json"
        man = json.load(open(manifest))
        self.man = man
        m = man["meta"]
        self.meta = m
        self.max_seq = max_seq
        self.n_layers = m["n_layers"]
        self.hidden = m["hidden"]
        self.vocab = m["vocab_size"]
        self.head_dim = m["head_dim"]
        self.num_heads = m["num_heads"]
        self.num_kv_heads = m["num_kv_heads"]
        self.intermediate_size = m["intermediate_size"]
        self.conv_L_cache = m.get("conv_L_cache", 7)
        self.sliding_window = m.get("sliding_window", 256)
        self.layer_types = m["layer_types"]
        self.maxb = maxb

        if verbose:
            print(f"LFM3: {self.n_layers} layers, hidden={self.hidden}, vocab={self.vocab}, heads={self.num_heads}/{self.num_kv_heads}, head_dim={self.head_dim}")
            print(f"  layer_types: {self.layer_types}")
            print(f"  conv_L_cache={self.conv_L_cache}, sliding_window={self.sliding_window}, sandwich_norm={m.get('use_sandwich_norm')}")

        self.blob = abi.MappedFile(weights)
        self.weights_ready = True
        self.pos = 0
        try:
            self.mod = abi.load_module(ENGINE_SRC)
            if verbose:
                print(f"  engine loaded: {self.mod}")
        except Exception as e:
            if verbose:
                print(f"  engine load failed: {e}")
            self.mod = None
        if verbose:
            print(f"  weights {self.blob.size/1e9:.3f} GB loaded in {time.time()-t0:.1f}s")

    def reset(self):
        self.pos = 0

    def forward(self, token):
        if self.mod is None:
            raise RuntimeError("engine not loaded")
        return 0

    def generate(self, tokens, max_new=64, eos=None, each=None):
        out = list(tokens)
        cur = out[-1]
        for _ in range(max_new):
            nxt = self.forward(cur)
            out.append(nxt)
            if each:
                each(nxt, len(out) - len(tokens))
            if eos is not None and nxt == eos:
                break
            cur = nxt
        return out

    def close(self):
        pass
