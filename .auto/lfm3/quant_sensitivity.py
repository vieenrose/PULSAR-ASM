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
import argparse, json, os, sys, time
import torch
# Pin the CPU reduction order BEFORE any torch op that matters. The default
# thread count makes bf16 CPU matmuls reduce in a varying order, which flips
# near-tie argmaxes: the bf16 reference then fails to reproduce its OWN goldens
# (measured: 8/12 prompts exact, worst divergence at token 15 of 40). Greedy
# goldens are only meaningful if the oracle is deterministic, so threads are
# pinned here and the bf16 pass is asserted to be self-reproducing.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
from transformers import AutoModelForMultimodalLM, AutoProcessor

REPO = "LiquidAI/LFM3-Audio-2B"
HERE = os.path.dirname(os.path.abspath(__file__))
BACKBONE = "model.language_model."          # NOT "lfm." -- see module docstring


def fq(w, bits, group, asym, scale16=False):
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
        # scale16 = the numerics the CUDA kernel ACTUALLY runs. The scale is
        # stored as __half in the GEMV (pq1/ggml Q8_0 convention), so the
        # device sees s16 = fp16(m/127), not m/127, and computes codes
        # against s16 with a +-127 clamp (not +-128, which the plain q8_0
        # arm allows). Without this arm the 8/8 accuracy result certified a
        # quantizer the engine does not implement.
        if scale16:
            s = s.half().float()
            hi = (1 << (bits - 1)) - 1
            s = torch.where(s == 0, torch.ones_like(s), s)
            d = torch.round(wf / s).clamp(-hi, hi) * s
            return d.reshape(N, K).to(w.dtype)
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
    # The arm the Nano engine actually runs: fp16-rounded scale, +-127 clamp.
    "q8_0h":   dict(bits=8,  group=32, asym=False, emb_bits=16, scale16=True),
    # The Nano blob Q8_0's the TIED embedding too (it is 524 MB bf16, 24% of
    # decode, and the table is streamed every token as lm_head). Every
    # existing arm keeps emb at bf16, so that saving was never certified --
    # and the emb branch below is ASYMMETRIC, which is q8_1, not the
    # symmetric fp32-scale q8_0 the exporter writes. emb_asym=False makes
    # this arm the numerics export_lfm3_q8.py actually produces.
    "q8_0_embq8": dict(bits=8, group=32, asym=False, emb_bits=8, emb_asym=False),
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

# Per-group arms. Why these exist: the FFN (gate/up/down) is 108.8 of the
# 143.7 ms/token decode GEMV (measured in q8_gemv_splitk.cu), i.e. 76% of the
# cost, and 1132 M of the 1466 M streamed params. Every prior arm in this file
# quantized the WHOLE backbone at one scheme, which measures the floor of the
# most sensitive group and hides which group it was. Only ffn-only quantization
# can buy the 2x between 6.96 tok/s and the 14 tok/s ceiling at 21.9 GB/s, so
# the sensitivity curve has to be separable per group.
GROUPS = {
    "ffn":  ("feed_forward.w1", "feed_forward.w2", "feed_forward.w3"),
    # The attention projections are self_attn.{q,k,v,o}_proj -- NOT
    # attention.wq and NOT attention.w_q. Both guesses matched zero tensors
    # and the assert below fired both times, which is the whole reason the
    # assert exists. gate_proj is included: it is the per-head sigmoid gate,
    # a real [2048,2048] GEMV per attention layer, and it was MISSING from the
    # per-token GEMV budget in q8_gemv_splitk.cu (see that commit's note).
    "attn": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
             "self_attn.o_proj", "self_attn.gate_proj"),
    # conv.conv.weight is [H, 1, 7] -- 3-D, so the 2-D quantizer never sees it
    # and listing it here made the assert fire. The short conv weight is 7*H
    # floats per layer (14.7 M total, 0.4% of the backbone), far too small to
    # matter for the memory number, so it stays bf16.
    "conv": ("conv.in_proj", "conv.out_proj"),
    "all":  ("feed_forward.w1", "feed_forward.w2", "feed_forward.w3",
             "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
             "self_attn.o_proj", "self_attn.gate_proj",
             "conv.in_proj", "conv.out_proj"),
}


def apply_scheme(model, sc, groups="all"):
    n_q, bytes_total = 0, 0
    pats = GROUPS[groups]
    matched = {g: 0 for g in pats}
    for name, p in model.named_parameters():
        if p.dim() != 2 or not name.startswith(BACKBONE):
            continue
        if name == EMB:
            pass
        elif not any(g in name for g in pats):
            # not in this arm -> stays bf16, and counts as bf16 bytes
            bytes_total += p.numel() * 2
            continue
        if name == EMB:
            bits, group, asym = sc["emb_bits"], 32, sc.get("emb_asym", True)
            scale16 = sc.get("emb_scale16", False)
            if bits >= 16:
                bytes_total += p.numel() * 2
                continue
        else:
            bits, group, asym = sc["bits"], sc["group"], sc["asym"]
            scale16 = sc.get("scale16", False)
            if bits >= 16:
                bytes_total += p.numel() * 2
                continue
        with torch.no_grad():
            p.data.copy_(fq(p.data, bits, group, asym, scale16))
        n_q += 1
        for g in pats:
            if g in name:
                matched[g] += 1
        # codes + one f16 scale per group (+ f16 min for asymmetric)
        b = bits + 16.0 / group + (16.0 / group if asym else 0)
        bytes_total += p.numel() * b / 8
    # The parameter filter has already cost one debugging cycle at the group
    # level (a filter matching ZERO tensors silently reports the bf16 result
    # for every scheme). Each named pattern in the arm must land on tensors.
    empty = [g for g, c in matched.items() if c == 0]
    assert not empty, "group patterns matched no tensors: %s" % empty
    return n_q, bytes_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--prompts", default=None,
                    help="JSON [{id,text},...]; default = the two gate prompts A,B")
    ap.add_argument("--schemes", default="bf16,q8_0,q6_0,q5_0,q4_1,q4_0,q4_g128,q3_0,q2_0")
    ap.add_argument("--groups", default="all",
                    help="comma list of %s: which weight groups get quantized,"
                         " the rest stay bf16" % sorted(GROUPS))
    a = ap.parse_args()

    if a.prompts:
        plist = json.load(open(a.prompts))
        keys, gold = [], {}
        for p in plist:
            gp = os.path.join(HERE, "golden_greedy_%s.json" % p["id"])
            gold[p["id"]] = json.load(open(gp)) if os.path.exists(gp) \
                else {"prompt": p["text"], "gen": None}
            keys.append(p["id"])
    else:
        keys = ["A", "B"]
        gold = {k: json.load(open(os.path.join(HERE, "golden_greedy_%s.json" % k)))
                for k in keys}
    proc = AutoProcessor.from_pretrained(REPO, trust_remote_code=True)
    ids = {k: build_ids(proc, gold[k]["prompt"]) for k in keys}

    # The bf16 pass mints goldens for any prompt lacking one, so a new prompt
    # set needs no separate oracle run.
    out_jsonl = os.path.join(HERE, "quant_sensitivity_%s.jsonl"
                             % (os.path.basename(a.prompts or "gate").split(".")[0]))
    done = set()
    if os.path.exists(out_jsonl):
        for line in open(out_jsonl):
            try:
                rec = json.loads(line)
                done.add((rec["scheme"], rec.get("n")))
            except Exception:
                pass

    print("scheme       bits/w  GB(bone)  exact  mean_pre  worst  n_q  (of %d tok, %d prompts)"
          % (a.tokens, len(keys)), flush=True)
    for s in a.schemes.split(","):
        sc = SCHEMES[s]
        label = s if a.groups == "all" else "%s:%s" % (a.groups, s)
        # stored scheme string is `label`, so resume keys are per group+scheme
        if (label, a.tokens) in done:
            print("%-16s  (already recorded, skipping)" % label, flush=True)
            continue
        t0 = time.time()
        model = AutoModelForMultimodalLM.from_pretrained(
            REPO, trust_remote_code=True, dtype=torch.bfloat16).eval()
        n_q, nbytes = apply_scheme(model, sc, a.groups)
        # Prove the quantizer ran. A filter that matches nothing would report
        # the BF16 result for every scheme and look like "4-bit is lossless".
        assert n_q > 0 or sc["bits"] >= 16, "quantizer matched zero tensors"
        res = {}
        eos_early = []
        with torch.no_grad():
            for k in keys:
                out = model.generate(ids[k], max_new_tokens=a.tokens, do_sample=False)
                seq = out.sequences[0] if hasattr(out, "sequences") else out[0]
                gen = seq[ids[k].shape[1]:].tolist()
                assert len(gen) > 0, "empty stream: modality tag lost"
                if gold[k]["gen"] is None:
                    gold[k]["gen"] = gen
                    json.dump({"prompt": gold[k]["prompt"], "gen": gen,
                               "text": proc.tokenizer.decode(gen), "n": len(gen)},
                              open(os.path.join(HERE, "golden_greedy_%s.json" % k), "w"),
                              indent=1)
                res[k] = score(gen, gold[k]["gen"])
                # "exact" = agreement over the whole COMPARED SPAN, which is
                # min(len(golden), --tokens). Two conflated cases cost two
                # false FATALs and one bogus "oracle is nondeterministic" call:
                #   golden shorter than --tokens (prompt hit EOS, e.g. the
                #     "one sentence" prompts) -> comparing against --tokens
                #     makes it permanently non-exact: the 12-prompt run scored
                #     bf16 8/12, worst 15/40, for this reason alone.
                #   golden LONGER than --tokens (A,B minted at 48, run caps 24)
                #     -> the run is truncated, not divergent, so it is exact.
                span = min(len(gold[k]["gen"]), a.tokens)
                res[k] = min(res[k], span)
                if len(gold[k]["gen"]) < a.tokens:
                    eos_early.append(k)
        exact = sum(1 for k in keys
                    if res[k] >= min(len(gold[k]["gen"]), a.tokens))
        bw = 16.0 if sc["bits"] >= 16 else (nbytes * 8) / 1728.8e6
        print("%-12s  %5.2f   %6.3f   %2d/%-3d  %7.1f   %4d   %4d  (%.0fs)%s"
              % (label, bw, nbytes / 1e9, exact, len(keys),
                 sum(res.values()) / len(res), min(res.values()), n_q, time.time() - t0,
                 "  early-EOS:" + ",".join(eos_early) if eos_early else ""),
              flush=True)
        # The bf16 row is the harness's own control: it compares the oracle
        # against itself. Anything short of all-exact means the reference is
        # not reproducible and EVERY scheme below it is measured against noise.
        if s == "bf16" and exact != len(keys):
            bad = {k: (res[k], len(gold[k]["gen"])) for k in keys
                   if res[k] < min(len(gold[k]["gen"]), a.tokens)}
            print("FATAL: bf16 did not reproduce its own goldens on %d prompts: %s\n"
                  "       threads=%d -- goldens are not reproducible; delete them\n"
                  "       and re-mint under 1 thread before trusting any scheme."
                  % (len(bad), bad, torch.get_num_threads()), flush=True)
            sys.exit(3)
        with open(out_jsonl, "a") as f:
            f.write(json.dumps(dict(scheme=label, groups=a.groups, bits_w=round(bw, 2),
                                    gb=round(nbytes / 1e9, 3), exact=exact,
                                    mean_prefix=round(sum(res.values()) / len(res), 2),
                                    worst=min(res.values()), prefix=res, n_q=n_q,
                                    n=a.tokens, prompts=keys)) + "\n")
        del model
        import gc; gc.collect()
    print("SENSITIVITY_DONE -> %s" % out_jsonl)


if __name__ == "__main__":
    main()
