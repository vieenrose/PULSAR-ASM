#!/usr/bin/env python
"""mkaudio.py -- export the M1 audio path (encoder + downsample + projector) to
an .lfm3q8 audio blob, and cross-check every quant decision against Liquid's own
mmproj GGUF tensor table.

POLICY, and why it is not invented: linears Q8 (fp32 128-block scales, the
certified quantizer from export_lfm3_q8.py), everything else f32 verbatim.
Liquid's mmproj makes every one of those decisions itself -- Q8_0 for ffn/conv_pw
attn linears and the projector's two linears, F32 for conv_dw taps, norms and
biases -- so this exporter's table is CHECKED against theirs tensor-for-tensor
(§13 of the port plan). Where our f32 tensors differ from their F32 by count, the
census is what is inspected and corrected, not the policy.

Container: same magic + JSON-header + FNV/crc per tensor as the backbone blob, so
the engine's loader reads it unchanged (component field in cfg selects the tower).

usage: mkaudio.py [--model DIR] [--out FILE] [--selfcheck]
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

# quantize_q8 and fnv1a_fast are copied VERBATIM from
# pulsar_asm/lfm3/tools/export_lfm3_q8.py (the tool that built the shipped
# backbone blob). They are the CERTIFIED numerics (quant_sensitivity.py's 8/8
# arm): s = max|w|/127 in fp32, round-half-even codes, all-zero group -> 0/0.
# A cross-tree import would couple the live pulsar_arm tree to the dormant
# pulsar_asm one, so the code is duplicated here with this pointer; changing
# EITHER copy requires re-certifying the quantizer, not an edit.
MAGIC = b"LFM3Q8\x01\x00"      # same magic: the engine loader reads either blob
T_Q8 = 1        # int8 codes + fp32 scales
T_F32 = 2       # f32 payload
T_BF16 = 3      # raw bf16 payload


def quantize_q8(w):
    """[N,K] float32 -> (codes int8 [N,K], scales f32 [N,K/32]) in the
    certified arm's numerics. Must match quant_row() in q8_lc3.cu exactly:
    s = max|w|/127 in fp32, codes = round_half_even(w/s) clamped to -128..127,
    and an all-zero group writes codes 0 / scale 0 (no division)."""
    N, K = w.shape
    assert K % 32 == 0, "Q8_0 needs K%%32==0, got K=%d" % K
    g = w.reshape(N, K // 32, 32)
    m = np.abs(g).max(axis=-1)                      # f32
    s = (m / np.float32(127.0)).astype(np.float32)  # fp32 scale, NOT fp16
    safe = np.where(s == 0, np.float32(1.0), s)[:, :, None]
    # np.round is round-half-to-even, == torch.round == lrintf in the default
    # FE_TONEAREST mode. The three must agree exactly, so none of them is
    # replaceable by a truncating cast.
    c = np.round(g / safe)
    c = np.clip(c, -128.0, 127.0).astype(np.int8)
    c = np.where(s[:, :, None] == 0, np.int8(0), c)
    return np.ascontiguousarray(c.reshape(N, K)), np.ascontiguousarray(s.reshape(N, K // 32))


def fnv1a_fast(*chunks):
    """Transport digest: loader recomputes on the bytes it loaded."""
    import zlib
    h = 0
    for ch in chunks:
        h = zlib.crc32(ch, h)
    return h & 0xFFFFFFFF

REPO_LOCAL = os.path.expanduser("~/models/lfm3/LFM3-Audio-2B")

# M1 components (prefixes in model.safetensors)
ENC = "encoder.lfm.lfm."     # 17-layer tower, hidden 768
DOWN = "encoder.downsample." # front end, all f32 per GGUF
PROJ = "mlp."                # LayerNorm(768) -> 768->2048 -> GELU -> 2048->2048

# Linears that go Q8. The GGUF cross-check asserts this list equals theirs:
# a.blk.*.{ffn_gate,ffn_up,ffn_down,attn_q,attn_k,attn_v,attn_out,conv_pw1,
# conv_pw2} and mm.a.mlp.{1,3}.weight are Q8_0 there.
Q8_PATTERNS = (
    "conv.in_proj", "conv.out_proj",              # encoder conv layers
    "feed_forward.w1", "feed_forward.w2", "feed_forward.w3",
    "self_attn.q_proj", "self_attn.k_proj",
    "self_attn.v_proj", "self_attn.o_proj",
    "self_attn.out_proj",                        # ENCODER naming (backbone is o_proj);
                                                 # the GGUF cross-check caught the miss
    "mlp.model.1", "mlp.model.3",                 # projector linears (with .weight/.bias)
    "lin_out",                                    # GGUF a.pre_encode.out is Q8_0; the
                                                  # first draft left it f32 and the
                                                  # cross-check caught it
)


GGUF_DTYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 8: "Q8_0", 12: "Q4_K",
               24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "BOOL"}


def parse_gguf_tensors(path):
    """Tensor table from a GGUF header prefix: returns {name: dtype_name}.
    Only the table is needed; a 16 MB Range fetch covers the 319-tensor mmproj."""
    d = open(path, "rb").read()
    assert d[:4] == b"GGUF", "not a GGUF file"
    ver, nt, nkv = struct.unpack_from("<IQQ", d, 4)
    p = 4 + 4 + 8 + 8        # magic + version + n_tensors + n_kv = 24, not 20
    def sread():
        nonlocal p
        n = struct.unpack_from("<Q", d, p)[0]; p += 8
        x = d[p:p + n].decode(); p += n
        return x
    def tval():
        nonlocal p
        ty = struct.unpack_from("<I", d, p)[0]; p += 4
        if ty == 8: v = sread()
        elif ty == 9:
            at = struct.unpack_from("<I", d, p)[0]; p += 4
            n = struct.unpack_from("<Q", d, p)[0]; p += 8
            v = []
            for _ in range(n):
                if at == 8: v.append(sread())
                elif at == 4: v.append(struct.unpack_from("<I", d, p)[0]); p += 4
                elif at == 5: v.append(struct.unpack_from("<i", d, p)[0]); p += 4
                elif at == 7: v.append(bool(d[p])); p += 1
        elif ty in (0, 7): v = bool(d[p]); p += 1
        elif ty == 1: p += 1; v = "i8"
        elif ty == 2: p += 2; v = "u16"
        elif ty == 3: p += 2; v = "i16"
        elif ty == 4: p += 4; v = "u32"
        elif ty == 5: p += 4; v = "i32"
        elif ty == 6: p += 4; v = "f32"
        elif ty == 10: p += 8; v = "u64"
        elif ty == 11: p += 8; v = "i64"
        elif ty == 12: p += 8; v = "f64"
        else: raise ValueError("gguf kv type %d" % ty)
        return v
    for _ in range(nkv):
        sread(); tval()
    out = {}
    for _ in range(nt):
        nm = sread()
        nd = struct.unpack_from("<I", d, p)[0]; p += 4 + 8 * nd
        ty = struct.unpack_from("<I", d, p)[0]; p += 4 + 8
        out[nm] = GGUF_DTYPES.get(ty, "T%d" % ty)
    return out


def safetensors_to_gguf(name):
    """Map our safetensors name to Liquid's mmproj tensor name. GGUF stores
    linears [in,out] where safetensors stores [out,in]; shapes were checked
    field-by-field (plan section 13). w1/w3 <-> ffn_gate/ffn_up are unordered
    by shape and interchangeable for the dtype check."""
    if name.startswith("encoder.lfm.lfm.layers."):
        rest = name[len("encoder.lfm.lfm.layers."):]
        i, _, field = rest.partition(".")
        n = "a.blk.%s." % i
        m = {"conv.in_proj.weight": "conv_pw1.weight", "conv.out_proj.weight": "conv_pw2.weight",
             "conv.conv.weight": "conv_dw.weight",
             "feed_forward.w1.weight": "ffn_gate.weight", "feed_forward.w2.weight": "ffn_down.weight",
             "feed_forward.w3.weight": "ffn_up.weight",
             "ffn_norm.weight": "ln1.weight", "operator_norm.weight": "ln2.weight",
             "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight",
             "self_attn.v_proj.weight": "attn_v.weight", "self_attn.out_proj.weight": "attn_out.weight",
             "self_attn.q_layernorm.weight": "attn_q_norm.weight",
             "self_attn.k_layernorm.weight": "attn_k_norm.weight"}
        return n + m.get(field, "MISSING:" + field)
    if name == "encoder.lfm.lfm.embedding_norm.weight":
        return "a.post_ln.weight"     # the tower-final norm the census missed
    dm = {"encoder.downsample.conv_in.weight": "a.conv1d.0.weight",
          "encoder.downsample.conv_in.bias": "a.conv1d.0.bias",
          "encoder.downsample.conv_mid.weight": "a.conv1d.2.weight",
          "encoder.downsample.conv_mid.bias": "a.conv1d.2.bias",
          "encoder.downsample.lin_mid.weight": "a.conv1d.3.weight",
          "encoder.downsample.lin_mid.bias": "a.conv1d.3.bias",
          "encoder.downsample.conv_end.weight": "a.conv1d.5.weight",
          "encoder.downsample.conv_end.bias": "a.conv1d.5.bias",
          "encoder.downsample.lin_end.weight": "a.conv1d.6.weight",
          "encoder.downsample.lin_end.bias": "a.conv1d.6.bias",
          "encoder.downsample.lin_out.weight": "a.pre_encode.out.weight",
          "encoder.downsample.lin_out.bias": "a.pre_encode.out.bias",
          "mlp.model.0.weight": "mm.a.mlp.0.weight", "mlp.model.0.bias": "mm.a.mlp.0.bias",
          "mlp.model.1.weight": "mm.a.mlp.1.weight", "mlp.model.1.bias": "mm.a.mlp.1.bias",
          "mlp.model.3.weight": "mm.a.mlp.3.weight", "mlp.model.3.bias": "mm.a.mlp.3.bias"}
    return dm.get(name, "MISSING:" + name)


def cross_check_gguf(entries, path):
    """Every tensor we ship must exist in Liquid's table with the same dtype
    decision (Q8_0 <-> our Q8, F32 <-> our f32). Disagreements are printed and
    fail: our table is not 'a' policy, it is THE policy, corroborated."""
    g = parse_gguf_tensors(path)
    bad = []
    for e in entries:
        gname = safetensors_to_gguf(e["name"])
        if gname.startswith("MISSING:"):
            bad.append((e["name"], "no GGUF map")); continue
        ours = "Q8_0" if e["dtype"] == T_Q8 else "F32"
        theirs = g.get(gname)
        if theirs is None:
            bad.append((e["name"], gname + " absent from GGUF")); continue
        if theirs != ours:
            bad.append((e["name"], "%s: ours %s vs GGUF %s" % (gname, ours, theirs)))
    print("GGUF cross-check: %d/%d decisions agree with Liquid's mmproj table"
          % (len(entries) - len(bad), len(entries)))
    for n, why in bad[:12]:
        print("  DISAGREE %s: %s" % (n, why))
    if bad:
        sys.exit(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=REPO_LOCAL)
    ap.add_argument("--out", default="/home/louis/models/lfm3/lfm3-audio-2b-enc.lfm3q8")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--gguf", metavar="HEADER",
                    help="cross-check every dtype decision against Liquid's mmproj GGUF "
                         "tensor table (a 2-16 MB Range fetch of the header is enough)")
    a = ap.parse_args()

    sp = os.path.join(a.model, "model.safetensors")
    f = safe_open(sp, "pt")
    keys = sorted(k for k in f.keys()
                  if k.startswith(ENC) or k.startswith(DOWN) or k.startswith(PROJ))
    assert keys, "no audio tensors in %s" % sp

    # count assertions from the census: 17 encoder layers, 12 downsample tensors
    # (6w+6b), 6 projector tensors (3w+3b). The census correction (11 -> 12) came
    # from the GGUF cross-check; these asserts keep it from regressing silently.
    n_enc = len(keys) if False else sum(1 for k in keys if k.startswith(ENC))
    n_down = sum(1 for k in keys if k.startswith(DOWN))
    n_proj = sum(1 for k in keys if k.startswith(PROJ))
    # 158 = 10 conv layers x 8 + 7 attn layers x 11 + embedding_norm. The plan's
    # census said 159; the GGUF cross-check (a.post_ln) is what exposed the
    # tower-final norm, so 158 is the corrected count.
    assert n_enc == 158, "encoder: expected 158 tensors, got %d" % n_enc
    assert n_down == 12, "downsample: expected 12 tensors (6w+6b), got %d" % n_down
    assert n_proj == 6, "projector: expected 6 tensors (3w+3b), got %d" % n_proj

    # mlp.model.1.weight is Q8, its .bias is f32 -- match on the WEIGHT name only.
    entries, blobs, off, n_f32 = [], [], 0, 0
    t0 = time.time()
    for k in keys:
        shp = list(f.get_slice(k).get_shape())
        t = f.get_tensor(k).to("cpu")
        is_w = k.endswith(".weight")
        q8 = is_w and any(p in k for p in Q8_PATTERNS)
        if q8:
            # lin_out is Conv1d [768,4096,1]: kernel-1, squeezed to [768,4096]
            # before quantizing (Liquid's GGUF stores it 2-D Q8_0 as well).
            if t.dim() == 3 and shp[2] == 1:
                t = t.squeeze(-1)
            assert t.dim() == 2, "%s is %d-D, the Q8 quantizer is 2-D only" % (k, t.dim())
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
            # f32 verbatim: norms, biases, depthwise taps, downsample (GGUF: all F32)
            w = np.ascontiguousarray(t.to(torch.float32).numpy())
            entries.append(dict(name=k, dtype=T_F32, ne=shp, off=off,
                                ncodes=w.nbytes, nscales=0,
                                crc=fnv1a_fast(w.tobytes())))
            blobs.append(w.tobytes()); off += w.nbytes
            n_f32 += 1

    cfg = dict(component="lfm3-audio-enc", hidden=768, ffn=1792, heads=8, kv=4,
               head_dim=96, conv_L=3, layers=17, attn_layers=[2, 5, 8, 11, 14, 15, 16],
               n_enc=n_enc, n_down=n_down, n_proj=n_proj,
               proj_in=768, proj_mid=2048, proj_out=2048)
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

    if a.gguf:
        cross_check_gguf(entries, a.gguf)

    if a.selfcheck:
        f2 = safe_open(sp, "pt")
        fh = open(a.out, "rb"); fh.read(len(MAGIC))
        n = struct.unpack("<I", fh.read(4))[0]; fh.read(n)
        base = fh.tell()
        bad = []
        for e in entries:
            fh.seek(base + e["off"])
            raw = fh.read(e["ncodes"] + e["nscales"])
            if fnv1a_fast(raw) != e["crc"]:
                bad.append(e["name"] + ":load"); continue
            t = f2.get_tensor(e["name"]).to("cpu")
            # same squeeze as the write path: lin_out is Conv1d [768,4096,1]
            # in safetensors but stored 2-D, so re-deriving must match.
            if e["dtype"] == T_Q8 and t.dim() == 3 and list(t.shape)[2] == 1:
                t = t.squeeze(-1)
            t = t.to(torch.float32).numpy()
            if e["dtype"] == T_Q8:
                c, s = quantize_q8(t)
                ref = c.tobytes() + s.tobytes()
            else:
                ref = np.ascontiguousarray(t).tobytes()
            if fnv1a_fast(ref) != e["crc"]:
                bad.append(e["name"] + ":quant")
        print("SELFCHECK %s (%d tensors, %d bad)%s"
              % ("OK" if not bad else "FAIL", len(entries), len(bad),
                 "" if not bad else " " + str(bad[:8])))
        sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()

