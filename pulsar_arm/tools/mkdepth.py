#!/usr/bin/env python
"""mkdepth.py -- S4 export: depth decoder (depthformer + depth_linear + 12 codebook
heads + codebook_embedding + mask_embedding) to an .lfm3q8 blob.

POLICY (same certified quantizer as the backbone/encoder exporters, copied verbatim
with the pointer comment -- changing either copy requires re-certifying, not an edit):
  Q8  (int8 + fp32 128-block scales): every GEMV weight -- depth_linear [12288,2048],
      the trunk's q/k/v/o/w1/w2/w3, and the STACKED 24204x1024 logits head (12 tied
      [2017,1024] heads flattened codebook-major: row c*2017+v = codebook c, vocab v).
  BF16 (dtype 3, verbatim): the codebook embeddings [2017,1024] x12 and
      codebook_embedding [24204,2048]. Both are LOOKUP-ONLY tensors; keeping them
      bf16 makes each lookup bit-identical to the checkpoint instead of carrying Q8
      dequant noise into the trunk input / the backbone's residual stream.
  F32: norms, depth_linear.bias, mask_embedding.

The 12 to_logits.weight tensors are NOT exported: they are bitwise-tied to the
embedding weights in the checkpoint (verified 12/12, see the tie check) and HF's
loader drops them (_keys_to_ignore_on_load_unexpected), with logits computed against
embedding.weight. One tensor, two roles: the lookup uses the bf16 copy, the logits
GEMV uses the Q8 copy derived from the same bytes.

The stacked head layout means logits[c, v] = GEMV row c*2017 + v -- the engine does
the index math, the blob stays a plain [24204,1024] matrix.

usage: mkdepth.py [--model DIR] [--out FILE] [--selfcheck]
"""
import argparse, json, os, struct, sys, time
import numpy as np
import torch
from safetensors import safe_open

MAGIC = b"LFM3Q8\x01\x00"
T_Q8, T_F32, T_BF16 = 1, 2, 3

def quantize_q8(w):
    """Certified numerics (export_lfm3_q8.py quant_row == q8_lc3.cu quant_row):
    s = max|w|/127 fp32, round-half-even codes, all-zero group -> 0/0."""
    N, K = w.shape
    assert K % 32 == 0, "Q8 needs K%%32==0, got K=%d" % K
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
NCodebooks, CodebookVocab, DHid, SrcHid = 12, 2017, 1024, 2048

Q8_PATTERNS = (
    "depth_linear.weight",
    "feed_forward.w1", "feed_forward.w2", "feed_forward.w3",
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "self_attn.gate_proj",
)
BF16_PATTERNS = ("depth_embeddings.", "codebook_embedding")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=REPO_LOCAL)
    ap.add_argument("--out", default="/home/louis/models/lfm3/lfm3-audio-2b-depth.lfm3q8")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()

    sp = os.path.join(a.model, "model.safetensors")
    f = safe_open(sp, "pt")
    keys = sorted(k for k in f.keys()
                  if k.startswith(("depthformer.", "depth_linear.", "mask_embedding",
                                   "depth_embeddings.", "codebook_embedding")))
    # census: 3 trunk layers x 12 (w1/w2/w3 + 2 norms + attn q/k/v/o/gate + q/k_norm)
    # + trunk embedding_norm + depth_linear w+b + mask_embedding + 12 x (embedding +
    # embedding_norm) + codebook_embedding. The trunk has NO sandwich norm (config
    # use_sandwich_norm false) but DOES have the attention gate_proj [32,1024].
    n_trunk = sum(1 for k in keys if k.startswith("depthformer.model.layers."))
    n_head = sum(1 for k in keys if ".embedding.weight" in k)
    print("census: %d trunk-layer tensors, %d codebook embeddings, %d total keys"
          % (n_trunk, n_head, len(keys)))
    assert n_trunk == 36, "expected 36 trunk-layer tensors (3 layers x 12), got %d" % n_trunk
    assert n_head == 12, "expected 12 codebook embeddings, got %d" % n_head
    assert "depth_linear.weight" in keys and "depth_linear.bias" in keys
    assert "mask_embedding" in keys and "codebook_embedding.weight" in keys
    assert "depthformer.model.embedding_norm.weight" in keys
    # tie gate: refuse to ship unless every to_logits equals its embedding
    for i in range(NCodebooks):
        if not torch.equal(f.get_tensor(f"depth_embeddings.{i}.embedding.weight"),
                           f.get_tensor(f"depth_embeddings.{i}.to_logits.weight")):
            sys.exit(f"depth_embeddings.{i}: to_logits is NOT tied -- exporter assumption broken")

    entries, blobs, off, n_q8, n_bf16, n_f32 = [], [], 0, 0, 0, 0
    t0 = time.time()
    for k in keys:
        if ".to_logits." in k:
            continue                      # tied: see header comment
        t = f.get_tensor(k).to("cpu")
        is_w = k.endswith(".weight") and "norm" not in k
        q8 = is_w and any(p in k for p in Q8_PATTERNS) and t.dim() == 2
        bf16 = (not q8) and is_w and any(p in k for p in BF16_PATTERNS)
        if q8:
            if k == "depth_linear.weight":
                pass                      # [12288, 2048] as-is
            assert t.shape[1] % 32 == 0, "%s K=%d not %%32" % (k, t.shape[1])
            shp = list(t.shape)
            w = t.to(torch.float32).numpy()
            codes, scales = quantize_q8(w)
            entries.append(dict(name=k, dtype=T_Q8, ne=shp, off=off,
                                ncodes=codes.nbytes, nscales=scales.nbytes,
                                crc=fnv1a_fast(codes.tobytes(), scales.tobytes())))
            blobs += [codes.tobytes(), scales.tobytes()]
            off += codes.nbytes + scales.nbytes
            n_q8 += 1
        elif bf16:
            w = t.to(torch.bfloat16).view(torch.uint8).numpy()
            w = np.ascontiguousarray(w)
            entries.append(dict(name=k, dtype=T_BF16, ne=list(t.shape), off=off,
                                ncodes=w.nbytes, nscales=0,
                                crc=fnv1a_fast(w.tobytes())))
            blobs.append(w.tobytes()); off += w.nbytes
            n_bf16 += 1
        else:
            w = np.ascontiguousarray(t.to(torch.float32).numpy())
            entries.append(dict(name=k, dtype=T_F32, ne=list(w.shape), off=off,
                                ncodes=w.nbytes, nscales=0,
                                crc=fnv1a_fast(w.tobytes())))
            blobs.append(w.tobytes()); off += w.nbytes
            n_f32 += 1

    # the stacked logits head: one [24204,1024] Q8 tensor, codebook-major
    stacked = torch.stack([f.get_tensor(f"depth_embeddings.{i}.embedding.weight")
                           for i in range(NCodebooks)]).to(torch.float32).numpy()
    stacked = np.ascontiguousarray(stacked.reshape(NCodebooks * CodebookVocab, DHid))
    codes, scales = quantize_q8(stacked)
    entries.append(dict(name="depth_head_stacked.weight", dtype=T_Q8,
                        ne=[NCodebooks * CodebookVocab, DHid], off=off,
                        ncodes=codes.nbytes, nscales=scales.nbytes,
                        crc=fnv1a_fast(codes.tobytes(), scales.tobytes())))
    blobs += [codes.tobytes(), scales.tobytes()]
    off += codes.nbytes + scales.nbytes
    n_q8 += 1

    cfg = dict(component="lfm3-audio-depth", num_codebooks=NCodebooks,
               codebook_size=CodebookVocab, hidden=DHid, src_hidden=SrcHid,
               layers=3, heads=32, kv_heads=8, head_dim=32, ffn=2816,
               norm_eps=1e-5, rope_theta=1e6, rotary_dim=DHid,
               maskgit_steps=3, maskgit_schedule=[4, 4, 4],
               audio_eos_code=CodebookVocab - 1)
    hdr = dict(magic=MAGIC.decode(), cfg=cfg, n_tensors=len(entries),
               data_bytes=off, table=entries)
    hj = json.dumps(hdr, separators=(",", ":")).encode()
    with open(a.out, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<I", len(hj)))
        fh.write(hj)
        for b in blobs:
            fh.write(b)
    print("wrote %s: %.0f MB, %d tensors (%d Q8 / %d bf16 / %d f32), %.1f s"
          % (a.out, (len(hj) + off) / 1e6, len(entries), n_q8, n_bf16, n_f32,
             time.time() - t0))

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
            if e["name"] == "depth_head_stacked.weight":
                t = torch.stack([f2.get_tensor(f"depth_embeddings.{i}.embedding.weight")
                                 for i in range(NCodebooks)]).to(torch.float32).numpy()
                t = np.ascontiguousarray(t.reshape(NCodebooks * CodebookVocab, DHid))
                c, s = quantize_q8(t)
                ref = c.tobytes() + s.tobytes()
            elif ".to_logits." in e["name"]:
                continue
            else:
                t = f2.get_tensor(e["name"]).to("cpu")
                if e["dtype"] == T_Q8:
                    if t.dim() == 3: t = t.squeeze(-1) if t.shape[2] == 1 else t.reshape(t.shape[0], -1)
                    c, s = quantize_q8(t.to(torch.float32).numpy())
                    ref = c.tobytes() + s.tobytes()
                elif e["dtype"] == T_BF16:
                    ref = np.ascontiguousarray(t.to(torch.bfloat16).view(torch.uint8).numpy()).tobytes()
                else:
                    ref = np.ascontiguousarray(t.to(torch.float32).numpy()).tobytes()
            if fnv1a_fast(ref) != e["crc"]:
                bad.append(e["name"] + ":quant")
        print("SELFCHECK %s (%d tensors, %d bad)%s"
              % ("OK" if not bad else "FAIL", len(entries), len(bad),
                 "" if not bad else " " + str(bad[:8])))
        sys.exit(0 if not bad else 1)

if __name__ == "__main__":
    main()
