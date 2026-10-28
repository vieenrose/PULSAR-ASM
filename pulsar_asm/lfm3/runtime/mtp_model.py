"""Linux loader for the Gemma 4 MTP assistant (assembly drafter).

Owns the assistant blob mapping, the draft context block, and the MTP engine
module. The assistant reads the TARGET's weights and caches - its embedding
row, the published KV rows, the RoPE tables - so constructing one needs a
live target engine next to the assistant blob. Nothing here does token math;
every flop happens behind `mtp_draft_step`.

No torch, no transformers, no HF at runtime: numpy here is allocation and
pointer setup only, exactly like the target loader.
"""

import ctypes
import json
import os
import struct
import time

import numpy as np

from . import pulsar_abi as abi
from .gemma4_model import parse_abi, rope_tables

_HERE = os.path.dirname(os.path.abspath(__file__))
MTP_SRC = os.path.normpath(os.path.join(_HERE, "..", "engine", "mtp_draft_flat.asm"))


class AssistAsm:
    """Assembly MTP drafter bound to a live target engine."""

    def __init__(self, assist_blob, target, manifest=None, verbose=True):
        t0 = time.time()
        abi.require_avx2_fma()
        MTP, _, _ = parse_abi(MTP_SRC, prefix="MTP_", desc_prefix="", extras=())
        self.MTP = MTP

        if manifest is None:
            manifest = assist_blob + ".manifest.json"
        man = json.load(open(manifest))
        self.man = man
        m = man["meta"]
        am = target.man["meta"]

        # Manifest-locked geometry the .asm was written against. Anything else
        # is a different assistant and must fail here, not as silent garbage.
        assert m["n_layers"] == 4, m["n_layers"]
        assert m["hidden"] == 256, m["hidden"]
        assert m["backbone_hidden"] == 1536 == am["hidden"], (m["backbone_hidden"], am["hidden"])
        assert m["intermediate"] == 2048, m["intermediate"]
        assert m["n_heads"] == 4, m["n_heads"]
        assert m["head_dim_slide"] == 256 and m["head_dim_full"] == 512
        assert m["layer_types"] == ["sliding_attention"] * 3 + ["full_attention"]
        assert m["sliding_window"] == 512 == am["sliding_window"]
        assert m["vocab"] == 262144 == am["vocab"]
        assert (m["num_centroids"], m["centroid_top_k"], m["vocab_per_centroid"]) == (2048, 32, 128)

        self.blob = abi.MappedFile(assist_blob)
        base = self.blob.addr
        lay = man["layout"]

        def wp(name):
            return base + lay[name]["offset"]

        self.ctx_mem = np.zeros(max(MTP["SIZE"], 4096), dtype=np.uint8)
        c = self.ctx_mem.ctypes.data

        def put(off, val):
            ctypes.c_uint64.from_address(c + off).value = val

        def putf(off, val):
            struct.pack_into("<f", self.ctx_mem, off, val)

        put(MTP["W_PRE"], wp("pre_proj"))
        put(MTP["W_POST"], wp("post_proj"))
        put(MTP["W_CENT"], wp("centroids"))
        put(MTP["TOK_ORDER"], wp("tok_order"))
        put(MTP["W_EMB_A"], wp("embed"))
        put(MTP["W_FIN"], wp("final_norm"))
        for i in range(4):
            p = f"L{i}."
            put(MTP[f"W_Q{i}"], wp(p + "w_q"))
            put(MTP[f"W_O{i}"], wp(p + "w_o"))
            put(MTP[f"N_IN{i}"], wp(p + "norm_in"))
            put(MTP[f"N_PA{i}"], wp(p + "norm_pa"))
            put(MTP[f"N_PF{i}"], wp(p + "norm_pf"))
            put(MTP[f"N_PFF{i}"], wp(p + "norm_pff"))
            put(MTP[f"N_Q{i}"], wp(p + "norm_q"))
            put(MTP[f"MG{i}"], wp(p + "mlp_gate"))
            put(MTP[f"MU{i}"], wp(p + "mlp_up"))
            put(MTP[f"MD{i}"], wp(p + "mlp_down"))
            put(MTP[f"SC{i}"], wp(p + "scalar"))

        # Target side: embedding table, published KV rows, RoPE tables.
        tbase = target.blob.addr
        put(MTP["TGT_EMB"], tbase + target.man["layout"]["embed"]["offset"])
        for t, i in am["kv_store_layers"].items():
            kk, vv = target._caches[i]
            if t == "sliding_attention":
                put(MTP["K_S"], kk.ctypes.data)
                put(MTP["V_S"], vv.ctypes.data)
            else:
                put(MTP["K_F"], kk.ctypes.data)
                put(MTP["V_F"], vv.ctypes.data)
        cs, ss, cf, sf = rope_tables(am, target.max_seq)
        self._keep_tables = (cs, ss, cf, sf)
        put(MTP["COS_S"], cs.ctypes.data)
        put(MTP["SIN_S"], ss.ctypes.data)
        put(MTP["COS_F"], cf.ctypes.data)
        put(MTP["SIN_F"], sf.ctypes.data)
        put(MTP["STRIDE_S"], cs.shape[1] * 4)
        put(MTP["STRIDE_F"], cf.shape[1] * 4)
        putf(MTP["EMB_SCALE"], am["hidden"] ** 0.5)
        putf(MTP["EPS"], am["rms_eps"])
        put(MTP["WINDOW"], am["sliding_window"])

        self.mod = abi.load_module(MTP_SRC)
        self.step_fn = abi.make_reg_entry(self.mod.exports["mtp_draft_step"], 1)
        self.top32_fn = abi.make_reg_entry(self.mod.exports["mtp_top32_avx2"], 2)
        self.gather_fn = abi.make_reg_entry(self.mod.exports["mtp_gather_rows_avx2"], 4)
        if verbose:
            print(f"  mtp module {self.mod.size} bytes, "
                  f"weights {self.blob.size/1e6:.1f} MB, init {time.time()-t0:.1f}s")

    def step(self, prev_tok, h_prev, h_next, pos):
        """One draft step. h_prev/h_next are 1536-fp32 arrays (caller-owned,
        zero-copy). Returns the draft token id."""
        MTP = self.MTP
        c = self.ctx_mem.ctypes.data
        ctypes.c_uint64.from_address(c + MTP["POS"]).value = int(pos)
        ctypes.c_uint64.from_address(c + MTP["PREV_TOK"]).value = int(prev_tok)
        ctypes.c_uint64.from_address(c + MTP["H_PREV"]).value = \
            np.asarray(h_prev, dtype=np.float32).ctypes.data
        ctypes.c_uint64.from_address(c + MTP["H_NEXT"]).value = \
            np.asarray(h_next, dtype=np.float32).ctypes.data
        return int(self.step_fn(c))
