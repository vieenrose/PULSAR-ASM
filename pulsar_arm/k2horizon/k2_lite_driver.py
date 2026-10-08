#!/usr/bin/env python3
"""Drive the carved K2 meeting-agent .tflite manually (stock signatures).

prefill_128 fills KV (no logits); decode steps one token at a time.
Caveat: tokenizes with the BASE HF tokenizer (ids < 64000, valid model
inputs) because the fine-tune's +5000-merge tokenizer is unpublished;
segmentation differs from training (disclosed, smoke-grade comparison).

Usage: k2_lite_driver.py <prompt.txt> [--gen N] [--temp T] [--seed S]
Prompt file = raw text (system prompt + window, pre-formatted).
"""
import sys

import numpy as np

NLAY, NKV, HD, CTX = 28, 8, 64, 4096


def load_interpreter(path):
    from ai_edge_litert.interpreter import Interpreter
    return Interpreter(model_path=path)


def run(it, ids, gen=400, temp=0.0, seed=7):
    pre = it.get_signature_runner("prefill_128")
    dec = it.get_signature_runner("decode")
    kv = {}
    for lyr in range(NLAY):
        # K [1,8,4096,64], V TRANSPOSED [1,8,64,4096] (export flag)
        kv[f"kv_cache_k_{lyr}"] = np.zeros((1, NKV, CTX, HD), np.float32)
        kv[f"kv_cache_v_{lyr}"] = np.zeros((1, NKV, HD, CTX), np.float32)
    n = len(ids)
    nfull = (n // 128) * 128
    for s in range(0, nfull, 128):
        args = {"tokens": np.array([ids[s:s + 128]], np.int32),
                "input_pos": np.arange(s, s + 128, dtype=np.int32),
                "mask": np.zeros((1, 1, 128, CTX), np.float32)}
        m = args["mask"]
        for i in range(128):
            m[0, 0, i, s + i + 1:] = float("-inf")
        args.update(kv)
        out = pre(**args)
        kv = {k: np.asarray(v) for k, v in out.items() if "kv_cache" in k}
    # remainder (possibly empty -> re-decode last token for first logits)
    pending = list(ids[nfull:])
    pos_next = nfull
    if not pending:
        pending = [ids[-1]]
        pos_next = n - 1
    out_ids = []
    rng = np.random.default_rng(seed)
    while len(out_ids) < gen:
        tok = pending.pop(0) if pending else nxt
        args = {"tokens": np.array([[tok]], np.int32),
                "input_pos": np.array([pos_next], np.int32),
                "mask": np.zeros((1, 1, 1, CTX), np.float32)}
        args["mask"][0, 0, 0, pos_next + 1:] = float("-inf")
        args.update(kv)
        out = dec(**args)
        kv = {k: np.asarray(v) for k, v in out.items() if "kv_cache" in k}
        logits = np.asarray(out["logits"])[0, 0]
        if temp <= 0:
            nxt = int(np.argmax(logits))
        else:
            l = logits - logits.max()
            e = np.exp(l / temp)
            pr = e / e.sum()
            nxt = int(rng.choice(len(pr), p=pr))
        out_ids.append(nxt)
        pos_next += 1
        if nxt == 1:
            break
    return out_ids


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("IFM/K2-Horizon-0.9B",
                                        trust_remote_code=True)
    text = open(sys.argv[1]).read()
    gen = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "--gen" else 400
    ids = tok.encode(text)
    print(f"prompt tokens: {len(ids)}", flush=True)
    it = load_interpreter("k2_meeting_q4.tflite")
    out = run(it, ids, gen=gen)
    print("generated:", len(out), flush=True)
    print(tok.decode(out))


if __name__ == "__main__":
    main()
