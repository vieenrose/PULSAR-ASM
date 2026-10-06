import json
import glob

p = glob.glob("/home/luigi/.cache/huggingface/hub/models--google--functiongemma-270m-it/snapshots/*/tokenizer.json")[0]
d = json.load(open(p, encoding="utf-8"))
vocab = d["model"]["vocab"]
merges = d["model"]["merges"]
rank = {}
for i, (a, b) in enumerate(merges):
    k = (vocab[a], vocab[b])
    if k not in rank:
        rank[k] = (i, vocab[a + b])
charm = {s: i for s, i in vocab.items() if len(s) == 1}


def enc(text):
    syms = []
    for ch in text.replace(" ", "\u2581"):
        if ch in charm:
            syms.append(charm[ch])
        else:
            for by in ch.encode("utf-8"):
                k = "<0x%02X>" % by
                syms.append(vocab.get(k, vocab.get(k.lower(), 3)))
    while True:
        best = None
        for j in range(len(syms) - 1):
            r = rank.get((syms[j], syms[j + 1]))
            if r and (best is None or r[0] < best[0]):
                best = (r[0], j, r[1])
        if best is None:
            break
        _, j, z = best
        syms[j:j + 2] = [z]
    return syms


for s in ["<bos>", "<start_of_turn>", "<end_of_turn>", "<eos>",
          "<start_of_turn>developer\n"]:
    print(repr(s)[:28], enc(s)[:8], flush=True)
