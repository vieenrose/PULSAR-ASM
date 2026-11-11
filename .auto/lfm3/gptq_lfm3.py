#!/usr/bin/env python
"""GPTQ for the LFM3 backbone linears, in the format a Q4_1 / Q5_1-style kernel reads:
groups of 32 along K, w = d * q + mn, q in [0, 2^b - 1], d and mn stored as fp16.
Group parameters are fixed up front from the original weights (least-squares min/max
refinement, quant_loss.py's q*_1o), then GPTQ chooses the codes column by column,
compensating each column's rounding error on the not-yet-quantized columns with the
inverse Hessian of the layer's calibration inputs. Layers are processed SEQUENTIALLY:
layer L's calibration inputs come from the model whose layers < L are already quantized.

Calibration = the model's own greedy answers to prompts disjoint from the gate prompts
(A..L), so the gate numbers are held-out.

  python .auto/lfm3/gptq_lfm3.py --bits 4 --out /tmp/gptq4.pt [--keep self_attn]
  then: python .auto/lfm3/quant_loss.py --load /tmp/gptq4.pt   (scores A, C, K)
"""
import argparse, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from divergence_floor import load            # noqa: E402
from quant_loss import fq                    # noqa: E402

CALIB = [
    "Explain how a refrigerator keeps food cold.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "What are the main differences between Python and C?",
    "Give me a recipe for a simple vegetable soup.",
    "Summarize the causes of the French Revolution in a few sentences.",
    "How do I change a flat tire on a bicycle?",
    "Describe the water cycle to a ten-year-old.",
    "Translate into French: The weather is nice today and we are going to the park.",
    "List five tips for sleeping better.",
    "Why is the ocean salty?",
    "Write a polite email asking my manager for a day off next Friday.",
    "What is the Pythagorean theorem and how is it used?",
    "Tell me a joke about computers.",
    "Compare electric cars and gasoline cars.",
    "How does the immune system fight a virus?",
    "Plan a three-day trip to Kyoto.",
    "用中文介紹一下台灣的夜市文化。",
    "What should I consider when adopting a dog?",
    "Explain recursion with a simple example.",
    "Describe your ideal weekend.",
]


TOPICS = ["volcanoes", "the stock market", "photosynthesis", "medieval castles", "machine learning",
          "coffee", "the moon landing", "jazz music", "climate change", "basketball", "the human heart",
          "ancient Rome", "smartphones", "gardening", "black holes", "chess", "the internet", "bees",
          "electric cars", "Japanese cuisine", "earthquakes", "the printing press", "yoga", "rainforests",
          "cryptography", "dinosaurs", "public speaking", "solar panels", "the Olympic games", "taxes",
          "台灣的夜市", "中國的茶文化", "學習程式設計", "颱風", "高鐵"]
TEMPLATES = ["Explain {} to a beginner.", "Write a short story involving {}.", "What are three surprising facts about {}?",
             "Give me a step-by-step guide related to {}.", "Argue for and against: {} matters more than people think.",
             "Write a dialogue between two friends talking about {}.", "Summarize the history of {} in one paragraph.",
             "請用中文簡單說明{}。", "Answer briefly: why do people care about {}?"]


def calib_seqs(model, tok, gen, n=160, batch=16, cache="/tmp/lfm3_calib.pt"):
    """the model's own sampled answers to n prompts (disjoint from the gate prompts A..L),
    teacher-forced later; cached so every arm sees the same calibration set"""
    import random, torch
    if cache and os.path.exists(cache):
        return torch.load(cache)
    rng = random.Random(1234)
    prompts = [rng.choice(TEMPLATES).format(rng.choice(TOPICS)) for _ in range(n)]
    bad = {getattr(model.config, "audio_out_token_id", None), getattr(model.config, "audio_token_id", None)}
    bad.discard(None)
    tok.padding_side = "left"; pad = tok.pad_token_id or 0
    seqs = []
    torch.manual_seed(1234)
    for i in range(0, n, batch):
        enc = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
               for p in prompts[i:i + batch]]
        enc = [e if isinstance(e, list) else e["input_ids"] for e in enc]
        L = max(map(len, enc))
        ids = torch.tensor([[pad] * (L - len(e)) + e for e in enc], device="cuda")
        am = torch.tensor([[0] * (L - len(e)) + [1] * len(e) for e in enc], device="cuda")
        with torch.no_grad():
            o = model.generate(ids, attention_mask=am, max_new_tokens=gen, do_sample=True, temperature=0.8, top_p=0.95)
        o = o.sequences if hasattr(o, "sequences") else o
        for j, e in enumerate(enc):
            out = [t for t in o[j, L:].tolist()]
            cut = next((k for k, t in enumerate(out) if t in bad or t == pad), len(out))
            seqs.append(e + out[:cut])
    if cache:
        torch.save(seqs, cache)
    return seqs


def group_params(W, bits, g=32):
    """fixed per-group (d, mn) of the q*_1o format, from the original weights"""
    import torch
    N, K = W.shape
    L = (1 << bits) - 1
    x = W.reshape(N, K // g, g)
    mn = x.amin(-1, keepdim=True); mx = x.amax(-1, keepdim=True)
    d = (mx - mn) / L; d = torch.where(d == 0, torch.ones_like(d), d)
    d, mn = d.half().float(), mn.half().float()
    q = torch.round((x - mn) / d).clamp(0, L)
    for _ in range(4):
        qm = q.mean(-1, keepdim=True); xm = x.mean(-1, keepdim=True)
        var = ((q - qm) ** 2).sum(-1, keepdim=True)
        dn = ((q - qm) * (x - xm)).sum(-1, keepdim=True) / torch.where(var == 0, torch.ones_like(var), var)
        dn = torch.where(var == 0, d, dn); mnn = xm - dn * qm
        dn, mnn = dn.half().float(), mnn.half().float()
        dn = torch.where(dn == 0, torch.ones_like(dn), dn)
        qn = torch.round((x - mnn) / dn).clamp(0, L)
        eo = ((q * d + mn - x) ** 2).sum(-1, keepdim=True); en = ((qn * dn + mnn - x) ** 2).sum(-1, keepdim=True)
        sel = en < eo
        d = torch.where(sel, dn, d); mn = torch.where(sel, mnn, mn); q = torch.where(sel, qn, q)
    return d.squeeze(-1), mn.squeeze(-1)          # [N, K/g]


def gptq(W, H, bits, damp=0.01, blk=128, g=32):
    """W [N, K] fp32 (cuda), H [K, K] fp32 -> (codes uint8 [N, K], d [N, K/g], mn [N, K/g])"""
    import torch
    N, K = W.shape
    L = (1 << bits) - 1
    d, mn = group_params(W, bits, g)
    W = W.clone()
    dead = torch.diag(H) == 0
    H = H.clone(); H[dead, dead] = 1; W[:, dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(K, device=H.device)
    Hi = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros(N, K, dtype=torch.uint8, device=W.device)
    for i1 in range(0, K, blk):
        i2 = min(i1 + blk, K)
        W1 = W[:, i1:i2].clone(); E1 = torch.zeros_like(W1); Hi1 = Hi[i1:i2, i1:i2]
        for i in range(i2 - i1):
            c = i1 + i; gi = c // g
            w = W1[:, i]
            q = torch.round((w - mn[:, gi]) / d[:, gi]).clamp(0, L)
            Q[:, c] = q.to(torch.uint8)
            e = (w - (q * d[:, gi] + mn[:, gi])) / Hi1[i, i]
            W1[:, i:] -= e.unsqueeze(1) * Hi1[i, i:].unsqueeze(0)
            E1[:, i] = e
        W[:, i2:] -= E1 @ Hi[i1:i2, i2:]
    return Q, d, mn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--gen", type=int, default=384)
    ap.add_argument("--keep", default="", help="name substrings left at q8 (fake-quant)")
    ap.add_argument("--bits-for", default="", help="per-name override, e.g. 'feed_forward.w2:5,conv.out_proj:5'")
    ap.add_argument("--speech", default="", help="speech_calib.py turns, replayed teacher-forced")
    ap.add_argument("--speech-weight", type=float, default=1.0, help="Hessian weight per speech position")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer
    from divergence_floor import REPO
    tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    model = load(torch.float16)
    t0 = time.time()
    seqs = calib_seqs(model, tok, a.gen)
    ntok = sum(len(s) for s in seqs)
    print("calibration: %d sequences, %d tokens (%.0f s)" % (len(seqs), ntok, time.time() - t0), flush=True)

    turns = torch.load(a.speech) if a.speech else []
    if turns:
        print("speech calibration: %d turns, %d positions (weight %.1f)"
              % (len(turns), sum(len(t) for t, _ in turns), a.speech_weight), flush=True)
    wcur = [1.0]
    keep = [k for k in a.keep.split(",") if k]
    over = dict((kv.split(":")[0], int(kv.split(":")[1])) for kv in a.bits_for.split(",") if kv)
    layers = model.model.language_model.layers
    result = {}
    for li, layer in enumerate(layers):
        lins = [(n, m) for n, m in layer.named_modules() if isinstance(m, torch.nn.Linear)]
        # collect H for every linear of this layer from the current (partly quantized) model
        Hs = {n: torch.zeros(m.in_features, m.in_features, device="cuda") for n, m in lins}
        cnt = {n: 0 for n, _ in lins}
        hooks = []
        for n, m in lins:
            def hk(mod, inp, out, n=n):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                Hs[n] += (2.0 * wcur[0]) * (x.t() @ x); cnt[n] += wcur[0] * x.shape[0]
            hooks.append(m.register_forward_hook(hk))
        stop = {}
        def stopper(mod, inp, out):                   # layers after li are not needed
            raise StopIteration
        if li + 1 < len(layers):
            hooks.append(layers[li + 1].register_forward_pre_hook(lambda mod, inp: stopper(mod, inp, None)))
        with torch.no_grad():
            wcur[0] = 1.0
            for s in seqs:
                try:
                    model(input_ids=torch.tensor([s], device="cuda"))
                except StopIteration:
                    pass
            wcur[0] = a.speech_weight
            for s, c in turns:
                try:
                    model(input_ids=torch.tensor([s], device="cuda"), audio_out_codes=c.cuda())
                except StopIteration:
                    pass
        for h in hooks: h.remove()
        for n, m in lins:
            full = "model.language_model.layers.%d.%s" % (li, n)
            W = m.weight.detach().float()
            if any(k in full for k in keep):
                m.weight.data.copy_(fq(W, "q8").half()); result[full] = ("q8", None); continue
            b = next((v for k, v in over.items() if k in full), a.bits)
            H = Hs[n] / cnt[n]
            Q, d, mn = gptq(W, H, b)
            deq = (Q.float().reshape(W.shape[0], -1, 32) * d.unsqueeze(-1) + mn.unsqueeze(-1)).reshape(W.shape)
            m.weight.data.copy_(deq.half())
            result[full] = (b, (Q.cpu(), d.half().cpu(), mn.half().cpu()))
        del Hs
        torch.cuda.empty_cache()
        print("layer %d done (%.0f s)" % (li, time.time() - t0), flush=True)
    torch.save(result, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    sys.exit(main())
