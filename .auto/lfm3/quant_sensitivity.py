#!/usr/bin/env python
"""Quantization sensitivity of the LFM3-Audio-2B text backbone, measured on the
greedy token stream against the HF BF16 goldens (gate G4 of the plan).

Why this exists: on Jetson Nano the BF16 backbone needs 3.511 GB resident
against a measured MemAvailable of 1476 MB, so BF16 is not an option and some
weight quantization is forced. The repo's definition of done is bit-exactness,
which quantization makes unachievable by construction -- so the gate has to
become a measured tolerance, and that needs a number, not an opinion.

Method: fake-quantize -> dequantize the 2-D backbone weights in torch (the same
numerics an int-dot GEMV has: dot over dequantized codes), then greedy-decode
and report the longest common prefix with the frozen golden. Prefix length is
the repo's metric (golden_q1_A.txt style: first divergence = fail).

Group = weights sharing one scale. Q8_0/Q4_0 style is group 32, Q4_K style 128.
Embeddings (262.1 M = 15% of the backbone) are an explicit arm: whether they
tolerate quantization moves the memory number a lot.

NOTE on the parameter filter (this cost a debugging cycle): the HF parameter
names are model.language_model.*, NOT lfm.* -- "lfm." is the safetensors
prefix, not the module path. A filter on "lfm." matches ZERO tensors and
silently reports the BF16 result for every scheme. n_q is asserted > 0 for
that reason.
"""
import argparse, json, os, time
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

REPO = "LiquidAI/LFM3-Audio-2B"
HERE = os.path.dirname(os.path.abspath(__file__))
BACKBONE = "model.language_model."          # NOT "lfm." -- see module docstring


def fq(w, bits, group, asym):
    """Quantize -> dequantize a [N, K] weight with per-group scales."""
    N, K = w.shape
    if K % group:
        group = 32 if K % 32 == 0 else K
    wf = w.float().reshape(N, K // group, group)
    if asym:
        lo = wf.min(-1, keepdim=True).values
        hi = wf.max(-1, keepdim=True).values
        s = (hi - lo) / (2 ** bits - 1)
        s = torch.where(s == 0, torch.ones_like(s), s)
        d = torch.round((wf - lo) / s).clamp(0, 2 ** bits - 1) * s + lo
    else:
        m = wf.abs().amax(-1, keepdim=True)
        s = m / ((1 << (bits - 1)) - 1)
        s = torch.where(s == 0, torch.ones_like(s), s)
        d = torch.round(wf / s).clamp(-(1 << (bits - 1)), (1 << (bits - 1)) - 1) * s
    return d.reshape(N, K).to(w.dtype)


def build_ids(proc, text):
    """Chat ids + the mandatory modality tag (see oracle.py PROTOCOL TRAP)."""
    s = proc.apply_chat_template([{"role": "user", "content": text}],
                                 add_generation_prompt=True, return_tensors="pt")
    assert s.endswith("<|im_start|>assistant\n"), s[-40:]
    s += "<|text_start|>"
    return proc.tokenizer(s, return_tensors="pt", add_special_tokens=False)["input_ids"]


def score(gen, gold):
    n = 0
    for a, b in zip(gen, gold):
        if a != b:
            break
        n += 1
    return n


# name -> (bits, group, asym). `emb` decides the embedding table separately.
SCHEMES = {
    "bf16":    dict(bits=16, group=32, asym=False, emb_bits=16),
    "q8_0":    dict(bits=8,  group=32, asym=False, emb_bits=16),
    "q6_0":    dict(bits=6,  group=32, asym=False, emb_bits=16),
    "q5_0":    dict(bits=5,  group=32, asym=False, emb_bits=16),
    "q4_0":    dict(bits=4,  group=32, asym=False, emb_bits=16),
    "q4_1":    dict(bits=4,  group=32, asym=True,  emb_bits=16),
    "q4_g128": dict(bits=4,  group=128, asym=True, emb_bits=16),
    "q3_0":    dict(bits=3,  group=32, asym=False, emb_bits=16),
    "q2_0":    dict(bits=2,  group=32, asym=False, emb_bits=16),
    "q4_1_emb": dict(bits=4, group=32, asym=True,  emb_bits=4),
    "q5_0_emb": dict(bits=5, group=32, asym=False, emb_bits=5),
    "q6_0_emb": dict(bits=6, group=32, asym=False, emb_bits=6),
}
EMB = "model.language_model.embed_tokens.weight"


def apply_scheme(model, sc):
    n_q, bytes_total = 0, 0
    for name, p in model.named_parameters():
        if p.dim() != 2 or not name.startswith(BACKBONE):
            continue
        if name == EMB:
            bits, group, asym = sc["emb_bits"], 32, True
            if bits >= 16:
                bytes_total += p.numel() * 2
                continue
        else:
            bits, group, asym = sc["bits"], sc["group"], sc["asym"]
            if bits >= 16:
                bytes_total += p.numel() * 2
                continue
        with torch.no_grad():
            p.data.copy_(fq(p.data, bits, group, asym))
        n_q += 1
        # codes + one f16 scale per group (+ f16 min for asymmetric)
        b = bits + 16.0 / group + (16.0 / group if asym else 0)
        bytes_total += p.numel() * b / 8
    return n_q, bytes_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--schemes", default="bf16,q8_0,q6_0,q5_0,q4_1,q4_0,q4_g128,q3_0,q2_0")
    a = ap.parse_args()

    gold = {k: json.load(open(os.path.join(HERE, "golden_greedy_%s.json" % k)))
            for k in ("A", "B")}
    proc = AutoProcessor.from_pretrained(REPO, trust_remote_code=True)
    ids = {k: build_ids(proc, gold[k]["prompt"]) for k in gold}

    print("scheme       bits/w  GB(bone)  A-prefix  B-prefix   n_q  (of %d)" % a.tokens,
          flush=True)
    rows = []
    for s in a.schemes.split(","):
        sc = SCHEMES[s]
        t0 = time.time()
        model = AutoModelForMultimodalLM.from_pretrained(
            REPO, trust_remote_code=True, dtype=torch.bfloat16).eval()
        n_q, nbytes = apply_scheme(model, sc)
        # Prove the quantizer ran. A filter that matches nothing would report
        # the BF16 result for every scheme and look like "4-bit is lossless".
        assert n_q > 0 or sc["bits"] >= 16, "quantizer matched zero tensors"
        res = {}
        with torch.no_grad():
            for k in gold:
                out = model.generate(ids[k], max_new_tokens=a.tokens, do_sample=False)
                seq = out.sequences[0] if hasattr(out, "sequences") else out[0]
                gen = seq[ids[k].shape[1]:].tolist()
                assert len(gen) > 0, "empty stream: modality tag lost"
                res[k] = score(gen, gold[k]["gen"])
        bw = 16.0 if sc["bits"] >= 16 else (nbytes * 8) / 1728.8e6
        print("%-12s  %5.2f   %6.3f    %3d/%-3d   %3d/%-3d   %4d  (%.0fs)"
              % (s, bw, nbytes / 1e9, res["A"], a.tokens, res["B"], a.tokens,
                 n_q, time.time() - t0), flush=True)
        rows.append(dict(scheme=s, bits_w=round(bw, 2), gb=round(nbytes / 1e9, 3),
                         prefix=res, n_q=n_q, n=a.tokens))
        del model
        import gc; gc.collect()
    json.dump(rows, open(os.path.join(HERE, "quant_sensitivity.json"), "w"), indent=1)
    print("SENSITIVITY_DONE")


if __name__ == "__main__":
    main()
