#!/usr/bin/env python3
# ==============================================================================
# Project PULSAR-ASM | lfm3/tools/convert_lfm3_safetensors.py
# ------------------------------------------------------------------------------
# LiquidAI LFM3 text backbone -> PULSAR flat weight blob
# ==============================================================================
import os
import sys
import json
import time
import struct
import argparse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

HF = "https://huggingface.co"
ALIGN = 64


def log(msg):
    print(msg, flush=True)


def hf_token():
    for env in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(env):
            return os.environ[env].strip()
    p = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(p):
        try:
            return open(p).read().strip()
        except:
            pass
    return ""


def http_get(url, rng=None, token="", tries=5):
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "pulsar-asm/lfm3-converter")
            if token:
                req.add_header("Authorization", f"Bearer {token}")
            if rng is not None:
                lo, hi = rng
                req.add_header("Range", f"bytes={lo}-{hi}")
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except Exception as e:
            last = e
            time.sleep(0.5 * (attempt + 1))
    raise last


def parse_header(head):
    n = struct.unpack("<Q", head[:8])[0]
    if n > len(head) - 8:
        raise RuntimeError(f"safetensors header too large")
    return json.loads(head[8:8 + n])


def build_plan(cfg_text):
    H = cfg_text["hidden_size"]
    NL = cfg_text["num_hidden_layers"]
    NH = cfg_text["num_attention_heads"]
    NKV = cfg_text["num_key_value_heads"]
    HD = cfg_text.get("head_dim") or (H // NH)
    inter = cfg_text["intermediate_size"]
    VOCAB = cfg_text["vocab_size"]
    types = cfg_text["layer_types"]
    conv_L = cfg_text.get("conv_L_cache", 7)
    use_sandwich = cfg_text.get("use_sandwich_norm", True)
    sliding_window = cfg_text.get("sliding_window", 256)
    rope_params = cfg_text.get("rope_parameters", {})
    rms_eps = cfg_text.get("norm_eps", 1e-5)
    tie = cfg_text.get("tie_word_embeddings", True)

    plan = []
    for i in range(NL):
        lt = types[i]
        is_attn = lt in ("full_attention", "sliding_attention")
        is_conv = lt == "conv"
        src = f"lfm.model.layers.{i}."
        pfx = f"L{i:02d}."

        plan += [
            ("f32", pfx + "norm_op", [H], src + "operator_norm.weight"),
            ("f32", pfx + "norm_ffn", [H], src + "ffn_norm.weight"),
        ]
        if use_sandwich:
            plan += [
                ("f32", pfx + "norm_post_op", [H], src + "post_operator_norm.weight"),
                ("f32", pfx + "norm_post_ffn", [H], src + "post_ffn_norm.weight"),
            ]
        else:
            plan += [
                ("none", pfx + "norm_post_op", [0], None),
                ("none", pfx + "norm_post_ffn", [0], None),
            ]

        if is_conv:
            plan += [
                ("raw", pfx + "conv_in_proj", [3*H, H], src + "conv.in_proj.weight"),
                ("raw", pfx + "conv_w", [H, 1, conv_L], src + "conv.conv.weight"),
                ("raw", pfx + "conv_out_proj", [H, H], src + "conv.out_proj.weight"),
                ("none", pfx + "w_q", [0], None), ("none", pfx + "w_k", [0], None),
                ("none", pfx + "w_v", [0], None), ("none", pfx + "w_o", [0], None),
                ("none", pfx + "norm_q", [0], None), ("none", pfx + "norm_k", [0], None),
                ("none", pfx + "gate_proj", [0], None),
            ]
        elif is_attn:
            plan += [
                ("raw", pfx + "w_q", [NH * HD, H], src + "self_attn.q_proj.weight"),
                ("raw", pfx + "w_k", [NKV * HD, H], src + "self_attn.k_proj.weight"),
                ("raw", pfx + "w_v", [NKV * HD, H], src + "self_attn.v_proj.weight"),
                ("raw", pfx + "w_o", [H, NH * HD], src + "self_attn.o_proj.weight"),
                ("f32", pfx + "norm_q", [HD], src + "self_attn.q_norm.weight"),
                ("f32", pfx + "norm_k", [HD], src + "self_attn.k_norm.weight"),
                ("raw", pfx + "gate_proj", [NH, H], src + "self_attn.gate_proj.weight"),
                ("none", pfx + "conv_in_proj", [0], None), ("none", pfx + "conv_w", [0], None),
                ("none", pfx + "conv_out_proj", [0], None),
            ]
        else:
            raise ValueError(f"Unknown layer type {lt}")

        plan += [
            ("raw", pfx + "mlp_w1", [inter, H], src + "feed_forward.w1.weight"),
            ("raw", pfx + "mlp_w2", [H, inter], src + "feed_forward.w2.weight"),
            ("raw", pfx + "mlp_w3", [inter, H], src + "feed_forward.w3.weight"),
        ]

    plan += [
        ("raw", "embed_tokens", [VOCAB, H], "lfm.model.embed_tokens.weight"),
        ("f32", "embedding_norm", [H], "lfm.model.embedding_norm.weight"),
    ]
    if tie:
        plan += [("none", "lm_head", [0], None)]
    else:
        plan += [("raw", "lm_head", [VOCAB, H], "lfm.lm_head.weight")]

    return plan, {
        "hidden": H, "n_layers": NL, "num_heads": NH, "num_kv_heads": NKV,
        "head_dim": HD, "intermediate_size": inter, "vocab_size": VOCAB,
        "layer_types": types, "conv_L_cache": conv_L,
        "use_sandwich_norm": use_sandwich, "sliding_window": sliding_window,
        "rope_parameters": rope_params, "rms_norm_eps": rms_eps, "tie_word_embeddings": tie,
    }


def read_from_cache_or_download(repo):
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    import glob
    pattern = os.path.join(cache_dir, f"models--{repo.replace('/', '--')}", "snapshots", "*", "model.safetensors")
    for m in glob.glob(pattern):
        if os.path.islink(m) or os.path.isfile(m):
            log(f"   found in cache: {m}")
            return m
    return None


def convert_from_safetensors_file(st_path, out_path, plan, meta):
    from safetensors import safe_open
    log(f"   reading {st_path}")
    with safe_open(st_path, framework='pt') as f:
        keys = list(f.keys())
        log(f"   checkpoint has {len(keys)} tensors")

        off, layout = 0, {}
        for kind, name, shape, sname in plan:
            if kind == "none":
                layout[name] = {"offset": 0, "shape": shape, "kind": "none"}
                continue
            elems = 1
            for d in shape:
                elems *= d
            nbytes = 4 * elems if kind == "f32" else 2 * elems
            layout[name] = {"offset": off, "shape": shape, "kind": kind, "src": sname}
            off = (off + nbytes + ALIGN - 1) // ALIGN * ALIGN
        log(f"   flat blob size: {off/1e9:.3f} GB")

        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        with open(out_path, "wb") as fout:
            fout.truncate(off)

        def write_at(buf, dst):
            fd = os.open(out_path, os.O_WRONLY)
            try:
                os.pwrite(fd, buf, dst)
            finally:
                os.close(fd)

        for kind, name, shape, sname in plan:
            if kind == "none":
                continue
            if sname not in keys:
                raise RuntimeError(f"Missing tensor {sname} in checkpoint")
            t = f.get_tensor(sname)
            dst = layout[name]["offset"]
            if kind == "raw":
                if t.dtype == torch.bfloat16:
                    buf = t.view(torch.uint16).numpy().tobytes()
                elif t.dtype == torch.float32:
                    u32 = t.view(torch.uint32)
                    lsb = (u32 >> 16) & 1
                    u32 = u32 + 0x7fff + lsb
                    buf = (u32 >> 16).to(torch.uint16).numpy().tobytes()
                elif t.dtype == torch.float16:
                    f32 = t.float()
                    u32 = f32.view(torch.uint32)
                    lsb = (u32 >> 16) & 1
                    u32 = u32 + 0x7fff + lsb
                    buf = (u32 >> 16).to(torch.uint16).numpy().tobytes()
                else:
                    raise RuntimeError(f"bad dtype for raw {sname} {t.dtype}")
                write_at(buf, dst)
            elif kind == "f32":
                if t.dtype == torch.bfloat16:
                    buf = t.float().numpy().astype(np.float32).tobytes()
                elif t.dtype == torch.float16:
                    buf = t.float().numpy().astype(np.float32).tobytes()
                elif t.dtype == torch.float32:
                    buf = t.numpy().astype(np.float32).tobytes()
                else:
                    buf = t.float().numpy().astype(np.float32).tobytes()
                write_at(buf, dst)
            log(f"   wrote {name} {tuple(t.shape)} {t.dtype}")

        return layout, off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="LiquidAI/LFM3-Audio-2B")
    ap.add_argument("--out", default="/mnt/edge/pulsar/lfm3_2b.bin")
    args = ap.parse_args()

    token = hf_token()
    log(f"Resolving {args.repo} ...")
    cfg_url = f"{HF}/{args.repo}/raw/main/config.json"
    cfg_data = http_get(cfg_url, token=token)
    cfg = json.loads(cfg_data)
    if "text_config" in cfg:
        cfg_text = cfg["text_config"]
    elif "lfm" in cfg:
        cfg_text = cfg["lfm"]
    else:
        cfg_text = cfg

    plan, meta = build_plan(cfg_text)
    log(f"   backbone: {meta['n_layers']} layers, hidden {meta['hidden']}, vocab {meta['vocab_size']}")
    log(f"   layer types: {meta['layer_types']}")

    st_path = read_from_cache_or_download(args.repo)
    if not st_path:
        log("   no cache found")
        raise SystemExit(1)
    layout, blob_size = convert_from_safetensors_file(st_path, args.out, plan, meta)

    man = {"repo": args.repo, "model_type": "lfm3", "bytes": blob_size, "layout": layout, "meta": meta, "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    man_path = args.out + ".manifest.json"
    with open(man_path, "w") as f:
        json.dump(man, f)
    log(f"✅ done -> {args.out} ({blob_size/1e9:.3f} GB)")
    log(f"   manifest -> {man_path}")


if __name__ == "__main__":
    main()
