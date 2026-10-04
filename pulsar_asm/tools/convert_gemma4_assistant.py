#!/usr/bin/env python3
"""Convert google/gemma-4-E2B-it-assistant into PULSAR's flat blob layout.

Small enough (150 MB) to read the checkpoint straight off disk, unlike the 10 GB
target. Two structural notes the layout encodes:

  * The drafter has NO k_proj / v_proj: every layer is KV-shared with the target,
    so its attention reads the target's published KV rows and it never writes a
    cache of its own (HF runs it with use_cache=False).
  * model.embed_tokens.weight doubles as the lm_head (tie_word_embeddings=true),
    and is read row-wise by the masked embedder, so it stays bf16.

  python tools/convert_gemma4_assistant.py \
      --src /mnt/edge/pulsar/assist/model.safetensors \
      --out /mnt/edge/pulsar/gemma4_e2b_mtp.bin
"""

import argparse
import json
import os
import struct

import numpy as np

PER_LAYER = {                                   # blob key -> checkpoint suffix
    "w_q":      ("self_attn.q_proj.weight", "bf16"),
    "w_o":      ("self_attn.o_proj.weight", "bf16"),
    "norm_q":   ("self_attn.q_norm.weight", "bf16->f32"),
    "norm_in":  ("input_layernorm.weight", "bf16->f32"),
    "norm_pa":  ("post_attention_layernorm.weight", "bf16->f32"),
    "norm_pf":  ("pre_feedforward_layernorm.weight", "bf16->f32"),
    "norm_pff": ("post_feedforward_layernorm.weight", "bf16->f32"),
    "mlp_gate": ("mlp.gate_proj.weight", "bf16"),
    "mlp_up":   ("mlp.up_proj.weight", "bf16"),
    "mlp_down": ("mlp.down_proj.weight", "bf16"),
    "scalar":   ("layer_scalar", "bf16->f32"),   # not ".weight" here, unlike the target
}
GLOBALS = {
    "pre_proj":   ("pre_projection.weight", "bf16"),
    "post_proj":  ("post_projection.weight", "bf16"),
    "centroids":  ("masked_embedding.centroids.weight", "bf16"),
    "tok_order":  ("masked_embedding.token_ordering", "i64->i32"),
    "embed":      ("model.embed_tokens.weight", "bf16"),      # tied lm_head
    "final_norm": ("model.norm.weight", "bf16->f32"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/edge/pulsar/assist/model.safetensors")
    ap.add_argument("--out", default="/mnt/edge/pulsar/gemma4_e2b_mtp.bin")
    a = ap.parse_args()

    with open(a.src, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(hlen))
        base = 8 + hlen                       # tensor data begins after the header
        cfgp = os.path.join(os.path.dirname(a.src), "config.json")
        cfg = json.load(open(cfgp)) if os.path.exists(cfgp) else {}
    tc = cfg.get("text_config", {})

    def read(f, off, n):
        f.seek(off)
        return f.read(n)

    want = []
    for k, (src, kind) in GLOBALS.items():
        want.append((k, src, kind))
    nlayer = tc.get("num_hidden_layers", 4)
    for i in range(nlayer):
        for k, (suffix, kind) in PER_LAYER.items():
            want.append((f"L{i}.{k}", f"model.layers.{i}.{suffix}", kind))

    layout, jobs, total = {}, [], 0
    for key, src, kind in want:
        if src not in head:
            layout[key] = {"kind": "none", "offset": 0, "bytes": 0, "shape": [0]}
            continue
        e = head[src]
        lo, hi = e["data_offsets"]
        n = hi - lo
        nbytes = {"bf16": n, "bf16->f32": 2 * n, "i64->i32": n // 2}[kind]
        layout[key] = {"src": src, "kind": kind, "offset": total,
                       "bytes": nbytes, "shape": list(e["shape"])}
        jobs.append((key, src, kind, lo, hi, nbytes))
        total += (nbytes + 63) & ~63          # 64B alignment for the kernels

    missing = [k for k, v in layout.items() if v["kind"] == "none"]
    with open(a.out, "wb") as out:
        out.seek(total - 1)
        out.write(b"\0")                      # materialize once, then patch in place
        out.seek(0)
        with open(a.src, "rb") as f:
            for key, src, kind, lo, hi, nbytes in jobs:
                raw = read(f, base + lo, hi - lo)   # offsets are header-relative
                if kind == "bf16":
                    blob = raw
                elif kind == "bf16->f32":
                    blob = ((np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16)
                            .view(np.float32).tobytes())
                else:                                          # i64 -> i32
                    blob = np.frombuffer(raw, dtype="<i8").astype("<i4").tobytes()
                assert len(blob) == nbytes, (key, len(blob), nbytes)
                out.seek(layout[key]["offset"])
                out.write(blob)
                print(f"  {key:14s} {str(layout[key]['shape']):16s} {kind:9s} "
                      f"{nbytes/1e6:8.2f} MB @ {layout[key]['offset']:>11,d}", flush=True)

    man = {"source": a.src, "config": cfg, "meta": {
        "n_layers": nlayer,
        "hidden": tc.get("hidden_size", 256),
        "backbone_hidden": cfg.get("backbone_hidden_size", 1536),
        "intermediate": tc.get("intermediate_size", 2048),
        "n_heads": tc.get("num_attention_heads", 4),
        "head_dim_slide": tc.get("head_dim", 256),
        "head_dim_full": tc.get("global_head_dim", 512),
        "sliding_window": tc.get("sliding_window", 512),
        "layer_types": tc.get("layer_types", ["sliding_attention"] * 3 + ["full_attention"]),
        "vocab": tc.get("vocab_size", 262144),
        "rms_eps": tc.get("rms_norm_eps", 1e-6),
        "rope_theta_slide": 10000.0,
        "rope_theta_full": 1000000.0,
        "partial_rotary_full": 0.25,
        "num_centroids": cfg.get("num_centroids", 2048),
        "centroid_top_k": cfg.get("centroid_intermediate_top_k", 32),
        "vocab_per_centroid": tc.get("vocab_size", 262144) // cfg.get("num_centroids", 2048),
        "no_kv_proj": True,
        "logit_softcapping": None,
    }, "layout": layout, "bytes": total, "missing": missing}
    json.dump(man, open(a.out + ".manifest.json", "w"), indent=1)
    print(f"\nwrote {a.out}  {total/1e6:.1f} MB  ({len(jobs)} tensors, "
          f"{len(missing)} absent from the checkpoint: {missing})")


if __name__ == "__main__":
    main()
