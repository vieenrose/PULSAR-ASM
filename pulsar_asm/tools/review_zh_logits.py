"""Compare the assembly engine's logits against HF's and against float32.

The sharpest statement of fidelity is the float32 comparison: this engine widens
bf16 weights and accumulates in fp32, so it reproduces a float32 NumPy reference
bit for bit, while HF's bf16 path drifts up to ~4 logits on top-100 tokens and
flattens the distribution (p 0.79 -> 0.57 on the leader). Both pick the same
argmax; they disagree on confidence. Trajectory tests therefore split at near-ties
for a reason that has nothing to do with a bug here.

Trajectory matching only says the argmax agrees. This asks the sharper question:
is the engine's logit for every token id within bf16 noise of HF's, or is some
set of ids systematically inflated? A bias toward high-frequency, low-information
tokens would explain a reply that reads like it was assembled from filler
characters, and would be an engine bug rather than a property of the checkpoint.

Also asks whether the repetition loops seen at temperature 1.0 belong to the
model: HF's own logits are put through the same top-k / top-p cut and compared
token-for-token with what the engine's sampler picks.
"""
import json
import os
import struct
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

BLOB = "/mnt/edge/pulsar/gemma4_e2b.bin"
PROMPT = "用一句話說：為什麼用組合語言寫語言模型很奇怪？"
TOK = "/mnt/edge/pulsar/tok"


def fbits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def hf_logits(prompt, want_tokens=None):
    """Last-position logits from HF's own gemma4 code over the converted blob."""
    from transformers import AutoConfig, Gemma4ForCausalLM
    from tools.ref_gemma4_hf import bind_blob

    layout = json.load(open(BLOB + ".manifest.json"))["layout"]
    blob = np.memmap(BLOB, dtype=np.uint8, mode="r")
    cfg = AutoConfig.from_pretrained("/mnt/edge/gemma4-hf")
    text_cfg = getattr(cfg, "text_config", cfg)
    with torch.device("meta"):
        model = Gemma4ForCausalLM(text_cfg)
    bind_blob(model, layout, blob)
    model.model.embed_tokens.embed_scale = torch.tensor(float(text_cfg.hidden_size ** 0.5))
    model.model.embed_tokens_per_layer.embed_scale = torch.tensor(
        float(text_cfg.hidden_size_per_layer_input ** 0.5))
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding
    model.model.rotary_emb = Gemma4TextRotaryEmbedding(text_cfg)
    model.lm_head._parameters["weight"] = model.model.embed_tokens._parameters["weight"]
    assert not [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
                if t.device.type == "meta"]
    model.eval()
    x = torch.tensor([prompt], dtype=torch.long)
    with torch.no_grad():
        lg = model(input_ids=x, use_cache=False).logits[0, -1].float().numpy()
    if want_tokens:
        return lg
    return lg


def engine_logits(prompt):
    """The same row from the assembly engine."""
    import run_gemma4_chat as cli
    from runtime.gemma4_model import Gemma4

    eng = Gemma4(BLOB, max_seq=1024, n_threads=4, verbose=False)

    class A:
        greedy, seed, temp, top_k, top_p = True, None, None, None, None

    cli.sampler_from_config(eng, TOK, A)
    tk, eos = cli.load_tokenizer(TOK)
    c = cli.Chat(eng, tk, eos, max_new=1)
    c.reset_state()
    # the BARE question: prefill() appends a user turn and renders the template
    # itself, so handing it an already-rendered template doubles every turn mark
    c.prefill(PROMPT)
    lg = np.asarray(eng.logits()).reshape(-1)[:eng.vocab].copy()
    eng.close()
    return lg


def nucleus(lg, temp, top_k, top_p):
    """Reference top-k/top-p over a logit row: (kept ids, renormalized probs)."""
    z = lg.astype(np.float64) / temp
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    order = np.argsort(-p, kind="stable")
    keep = order[:top_k]
    q = p[keep] / p[keep].sum()
    c = np.cumsum(q)
    n = int(np.searchsorted(c, top_p, "left")) + 1
    keep, q = keep[:n], q[:n]
    return keep, q / q.sum()


def main():
    ids = json.load(open("/tmp/zh_prompt_ids.json"))
    tk, _ = __import__("run_gemma4_chat").load_tokenizer(TOK)
    from ref_gemma4_np import Model, Ref
    hf = hf_logits(ids)
    eng = engine_logits(ids)

    ref = Ref(BLOB)
    m = Model(ref, max_seq=64)
    h = None
    for pos, t in enumerate(ids):
        h, _ = m.forward(int(t), pos=pos)
    fp32 = np.asarray(m.logits(h), dtype=np.float64).reshape(-1)

    print("=== Q0: against a float32 reference (the real fidelity statement)")
    top100 = np.argsort(-fp32)[:100]
    for name, row in (("numpy fp32", fp32), ("engine", eng), ("HF bf16", hf)):
        p = np.exp(row - row.max())
        p /= p.sum()
        print(f"    {name:10s} argmax {int(np.argmax(row)):7d}  p(top) {p.max():.4f}  "
              f"corr {float(np.corrcoef(row, fp32)[0,1]):.6f}  "
              f"max|dlogit| top-100 {float(np.max(np.abs(row - fp32)[top100])):.4f}")

    print("\n=== Q1: per-token logit difference, same prompt, same state")
    d = eng - hf
    scale = float(np.abs(hf).max())
    print(f"    max |hf|            {scale:9.4f}")
    print(f"    max |engine - hf|   {float(np.abs(d).max()):9.4f}   "
          f"({float(np.abs(d).max())/scale:.2e} of range)")
    print(f"    mean |difference|   {float(np.abs(d).mean()):9.5f}")
    print(f"    corr                {float(np.corrcoef(hf, eng)[0,1]):9.6f}")
    top = np.argsort(-hf)[:200]
    print(f"    max |difference| over HF's top-200 ids: {float(np.abs(d[top]).max()):.5f}")

    print("\n    the four characters in question:")
    for name in ("預期", "預意", "錯誤", "錯乎", "難", "細"):
        # single characters only: '預期' is two tokens and asking the tokenizer
        # for it hands back id 3, which says nothing about either character
        tid = tk.convert_tokens_to_ids(name[0])
        if tid is None or tid == tk.unk_token_id:
            tid = tk.encode(name[0], add_special_tokens=False)[0]
        hp = np.exp(hf[tid] - hf.max())
        hp = float(hp / np.exp(hf - hf.max()).sum())
        ep = np.exp(eng[tid] - eng.max())
        ep = float(ep / np.exp(eng - eng.max()).sum())
        print(f"      {name}  id {tid:7d}  hf p={hp:.6f}   engine p={ep:.6f}   "
              f"dlogit={d[tid]:+.4f}")

    worst = np.argsort(-np.abs(d))[:8]
    print("\n    largest per-token differences:")
    for i in worst:
        from run_gemma4_chat import decode_text
        print(f"      id {int(i):7d}  dlogit {d[i]:+8.4f}  "
              f"hf {decode_text(tk, [int(i)])!r}")

    print("\n=== Q2: does the model's own distribution produce the same loops?")
    for temp in (1.0,):
        keep, q = nucleus(hf, temp, 64, 0.95)
        rng = np.random.default_rng(1234)
        draw = rng.choice(len(keep), size=40, p=q)
        from run_gemma4_chat import decode_text
        print(f"    temp {temp}: nucleus keeps {len(keep)}, "
              f"HF-side 40 draws -> {decode_text(tk, [int(keep[j]) for j in draw])!r}")
        print(f"    top-8 under HF's logits: "
              f"{[(int(i), decode_text(tk, [int(i)]), round(float(np.exp(hf[i]-hf.max())/np.exp(hf-hf.max()).sum()), 4)) for i in np.argsort(-hf)[:8]]}")


if __name__ == "__main__":
    main()