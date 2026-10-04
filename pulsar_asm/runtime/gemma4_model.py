"""Linux loader for the Gemma 4 E2B assembly engine.

Everything the engine needs is one block of absolute addresses. This module
builds that block - parsing the context layout out of the assembly source (so
the assembly stays the single source of truth for offsets), memory-mapping the
converted weight blob, allocating fp32 buffers and KV caches, and generating the
RoPE tables. No torch, no transformers, no HF at runtime: the engine is the
inference runtime.

Weight layout comes from the manifest written by tools/convert_gemma4_safetensors.py.
"""

import ctypes
import json
import os
import re
import struct
import time

import numpy as np

from . import pulsar_abi as abi

_HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_ASM = os.path.normpath(os.path.join(_HERE, "..", "engine", "gemma4_engine_flat.asm"))
ENGINE_SRC = os.path.normpath(os.path.join(_HERE, "..", "engine", "gemma4_engine_flat.asm"))


def _fbits(v):
    """float32 bit pattern of v, as the integer the engine reads."""
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def parse_abi(path=ENGINE_ASM):
    """Read the equate tables that define the context/descriptor ABI.

    The assembly declares them; the loader reads them. If they ever disagree the
    engine reads the wrong buffer, which shows up as plausible-looking garbage,
    so the layout is never duplicated in Python.
    """
    txt = open(path).read()

    def grab(prefix):
        """Collect one equate table, rejecting offset collisions.

        FASM lets two names share a value without complaining: CTR_B and
        CTR_MAXLAYER once both sat at 368, so writing MAXLAYER from a test
        quietly changed the batch size and every symptom pointed at the wrong
        stage. The assembly is the layout's source of truth, so the check lives
        where the layout is read.
        """
        out, seen = {}, {}
        for m in re.finditer(rf"^{prefix}([A-Z0-9_]+)\s+equ\s+(\S+)", txt, re.M):
            name, val = m.group(1), int(m.group(2), 0)
            out[name] = val
            if not name.endswith("SIZE"):
                seen.setdefault(val, []).append(name)
        dup = {v: names for v, names in seen.items() if len(names) > 1}
        if dup:
            raise ValueError(f"{prefix} offsets collide: " +
                             "; ".join(f"{v}: {n}" for v, n in dup.items()))
        return out

    ctr = grab("CTR_")
    d = grab("D_")
    extra = {}
    for name in ("DESC_SIZE", "FLAG_FULL", "FLAG_KVSHARED", "FLAG_KVSTORE"):
        m = re.search(rf"^{name}\s+equ\s+(\S+)", txt, re.M)
        extra[name] = int(m.group(1), 0)
    return ctr, d, extra


def rot_pairs(meta, head_dim):
    """Table length per head = head_dim/2 pairs.

    HF proportional_rope computes inv_freq for the first
    `int(partial * head_dim // 2)` frequencies and then PADS the rest with zeros
    up to head_dim/2, so cos/sin always cover the whole head (pairs (i, i+hd/2))
    and the unrotated tail is expressed as cos=1/sin=0 rather than a narrow
    window. That is why this returns head_dim//2 and the *proportion* only picks
    where the table stops changing.
    """
    return head_dim // 2


def inv_freq(meta, head_dim, theta, partial=1.0):
    """HF default/proportional inv_freq, zero-padded to head_dim/2 entries."""
    n = head_dim // 2
    nrot = max(1, int(partial * head_dim // 2))
    rot = 1.0 / (theta ** (np.arange(0, 2 * nrot, 2, dtype=np.float64) / head_dim))
    return np.concatenate([rot, np.zeros(n - nrot)])


def rope_tables(meta, max_seq):
    """fp32 cos/sin tables, one row per position, per layer type.

    Row length is that type's rot_pairs - HF uses cat(freqs, freqs) so a single
    set of values serves both halves of the rotary window, and dimensions past
    the window are left untouched by the kernel rather than padded with cos=1.
    """
    pos = np.arange(max_seq, dtype=np.float64)[:, None]
    out = []
    for th, hd, part in ((meta["rope_theta_slide"], meta["head_dim_slide"], 1.0),
                         (meta["rope_theta_full"], meta["head_dim_full"],
                          meta["partial_rotary_full"])):
        ang = pos * inv_freq(meta, hd, th, part)
        out.append(np.ascontiguousarray(np.cos(ang), dtype=np.float32))
        out.append(np.ascontiguousarray(np.sin(ang), dtype=np.float32))
    return out


class Gemma4:
    """Owns the mapping, the buffers, the worker pool, and the engine module."""

    def __init__(self, weights, max_seq=1024, n_threads=4, maxb=8, manifest=None, verbose=True):
        t0 = time.time()
        abi.require_avx2_fma()
        self.CTR, self.D, self.FLAG = parse_abi()
        CTR, D, FL = self.CTR, self.D, self.FLAG

        if manifest is None:
            manifest = weights + ".manifest.json"
        man = json.load(open(manifest))
        self.man = man
        m = man["meta"]
        self.meta = m
        self.max_seq = max_seq
        self.n_layers = m["n_layers"]
        self.hidden = m["hidden"]
        self.vocab = m["vocab"]
        self.ple_dim = m["ple_dim"]
        self.n_head = m["n_heads"]
        self.window = m["sliding_window"]
        self.n_share = m["first_kv_shared"]
        self.max_hd = max(m["head_dim_slide"], m["head_dim_full"])
        self.maxb = maxb                        # rows a verify pass can hold
        self.max_inter = m["inter_base"] * (2 if m.get("double_wide", True) else 1)

        self.blob = abi.MappedFile(weights)
        base = self.blob.addr

        def wp(name):
            e = man["layout"][name]
            return base + e["offset"]

        self._keep = []                       # buffers referenced by the engine

        def f32(*shape):
            a = np.zeros(shape, dtype=np.float32)
            self._keep.append(a)
            return a

        def ptr(addr):
            return addr

        # ---- activation buffers ---------------------------------------------
        H, V = self.hidden, self.vocab
        # Row-major [MAXB][dim]: a batched pass indexes row b at b*dim, and with
        # B = 1 this is exactly the single-token layout. LOGITS is the big one -
        # 8 x vocab - because the tied head is the one GEMM that must cover every
        # row in a single pass over the embedding table.
        B = self.maxb
        buf = {}
        for k, n in (("X", B * H), ("T1", B * H), ("T2", B * H), ("T3", B * H),
                     ("Q", B * self.n_head * self.max_hd), ("A", B * self.n_head * self.max_hd),
                     ("K", self.max_hd), ("V", self.max_hd),
                     ("SCORE", max_seq), ("G", B * self.max_inter), ("U", B * self.max_inter),
                     ("M", B * self.max_inter),
                     ("PLE_CUR", B * self.n_layers * self.ple_dim),
                     ("PLE_NEXT", B * self.n_layers * self.ple_dim),
                     ("PLE_IN", B * self.n_layers * self.ple_dim),
                     ("LOGITS", B * V), ("TMP256", B * self.ple_dim)):
            buf[k] = f32(n)
        self.buf = buf

        # ---- KV caches ------------------------------------------------------
        # Only the non-shared prefix allocates: layers 15..34 read the rows
        # published by kv_store_layers (13 sliding / 14 full). A sliding layer
        # keeps `window` rows conceptually but we keep them all - the engine
        # limits the range it reads, which costs RAM and buys nothing else, so
        # the window is enforced in the engine's kv_start.
        caches = {}
        store = m["kv_store_layers"]
        for i in range(self.n_share):
            hd = self.head_dim(i)
            caches[i] = (f32(max_seq * hd), f32(max_seq * hd))
        self._caches = caches
        self._store = store

        # ---- per-layer descriptors -----------------------------------------
        DESC = self.FLAG["DESC_SIZE"] // 8
        self.DSL = {k: v // 8 for k, v in self.D.items()}      # slot index per field
        self.desc = np.zeros((self.n_layers, DESC), dtype=np.uint64)
        for i in range(self.n_layers):
            p = f"L{i:02d}."
            full = man["layer_types"][i] == "full_attention"
            hd = self.max_hd if full else m["head_dim_slide"]
            inter = m["inter_base"] * (2 if i >= self.n_share else 1)
            shared = i >= self.n_share
            src = store[man["layer_types"][i]]
            kbase, vbase = caches[src] if shared else caches[i]
            d = self.desc[i]
            # descriptor offsets are BYTE offsets in the assembly; the row is a
            # uint64 array, so index by offset/8 (getting this wrong reads the
            # wrong field and looks like a model that simply disagrees).
            d[self.DSL["QW"]] = wp(p + "w_q")
            d[self.DSL["KW"]] = 0 if shared else wp(p + "w_k")
            d[self.DSL["VW"]] = 0 if shared else wp(p + "w_v")
            d[self.DSL["OW"]] = wp(p + "w_o")
            d[self.DSL["GATE"]] = wp(p + "mlp_gate")
            d[self.DSL["UP"]] = wp(p + "mlp_up")
            d[self.DSL["DOWN"]] = wp(p + "mlp_down")
            d[self.DSL["QNORM"]] = wp(p + "norm_q")
            d[self.DSL["KNORM"]] = 0 if shared else wp(p + "norm_k")
            d[self.DSL["NIN"]] = wp(p + "norm_in")
            d[self.DSL["NPOSTATT"]] = wp(p + "norm_post_attn")
            d[self.DSL["NPREFF"]] = wp(p + "norm_pre_ff")
            d[self.DSL["NPOSTFF"]] = wp(p + "norm_post_ff")
            d[self.DSL["PLEGATE"]] = wp(p + "ple_gate")
            d[self.DSL["PLEPROJ"]] = wp(p + "ple_proj")
            d[self.DSL["PLENORM"]] = wp(p + "norm_ple")
            d[self.DSL["SCALAR"]] = wp(p + "layer_scalar")
            d[self.DSL["KV"]] = kbase.ctypes.data
            d[self.DSL["VV"]] = vbase.ctypes.data
            d[self.DSL["INTER"]] = inter
            d[self.DSL["HEADDIM"]] = hd
            d[self.DSL["KVSTRIDE"]] = hd * 4
            flags = (self.FLAG["FLAG_FULL"] if full else 0) | \
                    (self.FLAG["FLAG_KVSHARED"] if shared else 0)
            if not shared and src == i:
                flags |= self.FLAG["FLAG_KVSTORE"]
            d[self.DSL["FLAGS"]] = flags
        self._keep.append(self.desc)

        # ---- rope tables ----------------------------------------------------
        cs, ss, cf, sf = rope_tables(m, max_seq)
        self._keep += [cs, ss, cf, sf]

        # ---- context block --------------------------------------------------
        self.ctx_mem = np.zeros(max(CTR["SIZE"], 4096), dtype=np.uint8)
        self._keep.append(self.ctx_mem)
        c = self.ctx_mem.ctypes.data
        put = lambda off, val: ctypes.c_uint64.from_address(c + off).__setattr__("value", val)
        putf = lambda off, val: struct.pack_into("<f", self.ctx_mem, off, val)

        put(CTR["WBASE"], base)
        put(CTR["DESC"], self.desc.ctypes.data)
        put(CTR["NLAYER"], self.n_layers)
        put(CTR["B"], 1)
        self.tokens = np.zeros(maxb, dtype=np.int64)
        self.outtok = np.zeros(maxb, dtype=np.int64)
        self._keep += [self.tokens, self.outtok]
        put(CTR["TOKENS"], self.tokens.ctypes.data)
        put(CTR["OUTTOK"], self.outtok.ctypes.data)
        put(CTR["XSTRIDE"], self.hidden * 4)
        put(CTR["PLEROW"], self.n_layers * self.ple_dim * 4)
        put(CTR["MAXLAYER"], self.n_layers)
        put(CTR["POS"], 0)
        put(CTR["TOKEN"], 0)
        put(CTR["VOCAB"], self.vocab)
        put(CTR["HIDDEN"], self.hidden)
        put(CTR["NTHR"], n_threads)
        put(CTR["PLE_DIM"], self.ple_dim)
        put(CTR["WINDOW"], self.window)
        put(CTR["NSHARE"], self.n_share)
        put(CTR["NHEAD"], self.n_head)
        put(CTR["EMB"], wp("embed"))
        put(CTR["PLE_EMB"], wp("ple_embed"))
        put(CTR["PLE_PROJ"], wp("ple_model_proj"))
        put(CTR["PLE_PROJ_NORM"], wp("ple_proj_norm"))
        put(CTR["FINAL_NORM"], wp("final_norm"))
        putf(CTR["PROJ_SCALE"], 1.0 / (self.hidden ** 0.5))
        putf(CTR["INV_SQRT2"], 1.0 / (2.0 ** 0.5))
        putf(CTR["EMB_SCALE"], self.hidden ** 0.5)
        putf(CTR["PLE_TOK_SCALE"], self.ple_dim ** 0.5)
        putf(CTR["RMS_EPS"], m["rms_eps"])
        putf(CTR["SOFTCAP"], m["logit_softcapping"])

        # Sampling stays off until asked for: TEMP == 0 keeps the argmax head,
        # which is what the parity tests pin. set_sampling() turns on the
        # checkpoint's own temperature/top_k/top_p.
        self.rng = np.zeros(1, dtype=np.uint64)
        self.rng[0] = 0x9E3779B97F4A7C15
        put(CTR["RNG"], self.rng.ctypes.data)
        put(CTR["TEMP"], 0)
        put(CTR["TOPK"], 64)
        put(CTR["TOPP"], _fbits(0.95))

        # spin-worker pool: master + (n_threads-1) pthreads
        self.smp = abi.exec_alloc(4096)
        ctypes.memset(self.smp, 0, 4096)
        ctypes.c_uint64.from_address(self.smp + 16).value = n_threads
        put(CTR["SMP"], self.smp)

        for k, a in buf.items():
            key = "BUF_" + k
            if key not in CTR:
                raise KeyError(f"buffer {k} has no CTR_BUF_{k} equate in the engine")
            put(CTR[key], a.ctypes.data)
        put(CTR["COS_SLIDE"], cs.ctypes.data)
        put(CTR["SIN_SLIDE"], ss.ctypes.data)
        put(CTR["COS_FULL"], cf.ctypes.data)
        put(CTR["SIN_FULL"], sf.ctypes.data)
        put(CTR["ROPE_STR_SLIDE"], cs.shape[1] * 4)
        put(CTR["ROPE_STR_FULL"], cf.shape[1] * 4)

        self.mod = abi.load_module(ENGINE_SRC)
        self.worker = self.mod.exports["smp_worker_proc4"]
        self.handles = [abi.spawn_worker(self.worker, self.smp + 64 + i * 128)
                        for i in range(1, n_threads)]
        self.step = abi.make_reg_entry(self.mod.exports["gemma4_step"], 1)
        self.pos = 0
        if verbose:
            print(f"  engine {self.mod.size} bytes, "
                  f"{n_threads} cores, weights {self.blob.size/1e9:.3f} GB, "
                  f"init {time.time()-t0:.1f}s")

    # -------------------------------------------------------------------------
    def head_dim(self, i):
        return (self.meta["head_dim_full"] if self.man["layer_types"][i] == "full_attention"
                else self.meta["head_dim_slide"])

    def _set(self, field, val):
        ctypes.c_uint64.from_address(self.ctx_mem.ctypes.data + self.CTR[field]).value = val

    def _get(self, field):
        return ctypes.c_uint64.from_address(self.ctx_mem.ctypes.data + self.CTR[field]).value

    def set_sampling(self, temp=1.0, top_k=64, top_p=0.95, seed=None):
        """temperature -> top-k -> nucleus, the way generation_config.json asks.

        Not usable yet: sampler_nucleus_avx2 faults, so turning TEMP on would
        take the process down with it. The plumbing (context fields, the head's
        branch, this API) is finished and the kernel is the only gap; raising
        here keeps a future CLI flag from finding that out the hard way.
        """
        raise NotImplementedError(
            "sampler_nucleus_avx2 faults; see sub_assemblies/"
            "sub_sampler_nucleus_flat.asm - the engine stays on argmax")
        self._set("TEMP", _fbits(temp) if temp else 0)
        self._set("TOPK", int(top_k))
        self._set("TOPP", _fbits(top_p))
        if seed is not None:
            self.rng[0] = int(seed) or 1

    def reset(self):
        for k, (kk, vv) in self._caches.items():
            kk[:] = 0.0
            vv[:] = 0.0
        self.pos = 0
        self._set("POS", 0)

    def forward(self, token):
        """Feed one token at the current position, return the argmax for the next."""
        return int(self.run([token])[0])       # run() owns the position update

    def run(self, tokens, pos=None):
        """One pass over len(tokens) consecutive positions; returns each argmax.

        With len == 1 this is plain decode and takes the identical code path it
        always did (B = 1). With len > 1 it is a speculative verify pass: the
        weights are streamed once for the whole block, which is the only reason
        draft-then-verify can beat one-token-at-a-time on a bandwidth-bound CPU.
        The caller owns acceptance; this just fills in the target's own answers.
        """
        b = len(tokens)
        if b > self.maxb:
            raise ValueError(f"batch {b} exceeds maxb {self.maxb}")
        p0 = self.pos if pos is None else pos
        if p0 + b > self.max_seq:
            raise RuntimeError(f"positions {p0}..{p0 + b - 1} exceed max_seq {self.max_seq}")
        self.tokens[:b] = tokens
        self._set("TOKEN", int(tokens[0]))     # TOKENS[0] and TOKEN stay equal
        self._set("B", b)
        self._set("POS", p0)
        self.step(self.ctx_mem.ctypes.data)
        self._set("B", 1)
        if pos is None:
            self.pos = p0 + b
        return self.outtok[:b].copy()

    def logits(self, row=0):
        """Row `row` of the logits block. The buffer is [maxb][vocab]: the tied
        head computes every batched row in one pass over the embedding table, so
        a batched run leaves B rows behind, not one."""
        V = self.vocab
        return self.buf["LOGITS"][row * V:(row + 1) * V]

    def generate(self, tokens, max_new=64, eos=None, each=None):
        out, t = list(tokens), time.time()
        cur = out[-1]
        for _ in range(max_new):
            nxt = self.forward(cur)
            out.append(nxt)
            if each:
                each(nxt, len(out) - len(tokens))
            if eos is not None and nxt == eos:
                break
            cur = nxt
        return out, (len(out) - len(tokens)) / max(time.time() - t, 1e-9)

    def close(self):
        for h in getattr(self, "handles", []):
            abi.join_worker(h)
        self.blob.close()
