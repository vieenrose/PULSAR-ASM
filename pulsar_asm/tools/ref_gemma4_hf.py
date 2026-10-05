#!/usr/bin/env python3
"""Run HuggingFace's own gemma4 modeling code over the converted blob.

This is the ground-truth check: no re-download, and no reliance on this repo's
NumPy reference, which shares this repo's reading of the architecture and so
cannot adjudicate a disagreement about it. Every tensor in the blob knows the
checkpoint name it came from (the manifest's `src` field), so HF's module tree
can be pointed straight at the blob as zero-copy views.

    python3 tools/ref_gemma4_hf.py --prompt "用一句話說：..." --compare

With --compare the same prompt is run through the assembly engine and the two
greedy trajectories are printed side by side. Expect them to agree for a while
and then diverge at one token - that is bf16 rounding flipping a near-tie, not
a disagreement.

Two things the binding needs that a naive load_state_dict misses: embed_scale
and the rope inv_freq tables are non-persistent buffers computed in __init__, so
they must be rebuilt rather than loaded; and lm_head is tied to the embedding
table, so replacing that Parameter leaves the head pointing at a meta tensor.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bind_blob(model, layout, blob):
    """Point model's parameters at the blob, returning (bound, unbound)."""
    by_dst = {}
    for e in layout.values():
        if e["kind"] == "none":
            continue
        by_dst[e["src"].replace("model.language_model.", "model.")] = e

    def view(e):
        n = int(np.prod(e["shape"]))
        dt = np.uint16 if e["kind"] == "raw" else np.float32
        raw = memoryview(blob)[e["offset"]:e["offset"] + n * (2 if dt is np.uint16 else 4)]
        t = torch.from_numpy(np.frombuffer(raw, dtype=dt).reshape(e["shape"]))
        return t.view(torch.bfloat16) if dt is np.uint16 else t

    bound = unbound = 0
    for full, p in list(model.named_parameters()):
        e = by_dst.get(full)
        if e is None:
            unbound += 1
            continue
        path, leaf = full.rsplit(".", 1) if "." in full else (None, full)
        parent = model.get_submodule(path) if path else model
        parent._parameters[leaf] = torch.nn.Parameter(view(e), requires_grad=False)
        bound += 1
    for full, b in list(model.named_buffers()):
        e = by_dst.get(full)                      # layer_scalar lives here
        if e is None:
            continue
        path, leaf = full.rsplit(".", 1)
        parent = model.get_submodule(path) if path else model
        parent._buffers[leaf] = view(e).to(b.dtype)
        bound += 1
    return bound, unbound


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blob", default="/mnt/edge/pulsar/gemma4_e2b.bin")
    ap.add_argument("--config", default="/mnt/edge/gemma4-hf")
    ap.add_argument("--tok", default="/mnt/edge/pulsar/tok")
    ap.add_argument("--prompt", default="用一句話說：為什麼用組合語言寫語言模型很奇怪？")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--compare", action="store_true", help="also run the assembly engine")
    a = ap.parse_args()

    from transformers import AutoConfig, AutoTokenizer, Gemma4ForCausalLM
    from run_gemma4_chat import decode_text, render

    tk = AutoTokenizer.from_pretrained(a.tok)
    ids = tk.encode(render([{"role": "user", "content": a.prompt}]), add_special_tokens=False)
    print(f"prompt: {len(ids)} tokens")

    layout = json.load(open(a.blob + ".manifest.json"))["layout"]
    blob = np.memmap(a.blob, dtype=np.uint8, mode="r")
    cfg = AutoConfig.from_pretrained(a.config)
    text_cfg = getattr(cfg, "text_config", cfg)
    with torch.device("meta"):
        model = Gemma4ForCausalLM(text_cfg)
    bound, unbound = bind_blob(model, layout, blob)
    print(f"bound {bound} tensors from the blob, {unbound} unresolved")

    model.model.embed_tokens.embed_scale = torch.tensor(float(text_cfg.hidden_size ** 0.5))
    model.model.embed_tokens_per_layer.embed_scale = torch.tensor(
        float(text_cfg.hidden_size_per_layer_input ** 0.5))
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding
    model.model.rotary_emb = Gemma4TextRotaryEmbedding(text_cfg)
    model.lm_head._parameters["weight"] = model.model.embed_tokens._parameters["weight"]
    left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
            if t.device.type == "meta"]
    if left:
        raise SystemExit(f"still on meta, refusing to generate: {left}")
    model.eval()

    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=x, use_cache=False).logits[0, -1].float()
        gen = [int(t) for t in model.generate(x, max_new_tokens=a.steps,
                                              do_sample=False, use_cache=False)[0, len(ids):]]
    p = torch.softmax(logits, dim=-1)
    print("first-step top-5 (HF):")
    for prob, idx in zip(*torch.topk(p, 5)):
        print(f"   id {int(idx):7d}  p={float(prob):.4f}  {decode_text(tk, [int(idx)])!r}")
    print("HF greedy:", gen)
    print("HF text  :", repr(decode_text(tk, gen)))

    if a.compare:
        from runtime.gemma4_model import Gemma4
        import run_gemma4_chat as cli

        eng = Gemma4(a.blob, max_seq=1024, n_threads=4, verbose=False)

        class A:
            greedy, seed, temp, top_k, top_p = True, None, None, None, None

        cli.sampler_from_config(eng, a.tok, A)
        mine = [int(t) for t in cli.Chat(eng, tk, {1, 106, 50}, max_new=a.steps).turn(a.prompt,
                                                                                      quiet=True)]
        eng.close()
        print("engine   :", mine)
        d = next((i for i, (u, v) in enumerate(zip(gen, mine)) if u != v), None)
        print(f"identical for {len(gen) if d is None else d} of {len(gen)} tokens"
              + ("" if d is None else f", first divergence at {d}"))
        print("engine   :", repr(decode_text(tk, mine)))


if __name__ == "__main__":
    main()
