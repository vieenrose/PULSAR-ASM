#!/usr/bin/env python3
"""K2-Horizon oracle: transformers forward dump for engine validation.

Loads IFM/K2-Horizon-0.9B (custom arch, trust_remote_code), tokenizes
fixture prompts, and saves per-layer hidden states + final logits +
greedy continuation ids. The engine must reproduce argmax streams
exactly and layer states to fp32-close (rope/libm paths use tolerance).

Usage: k2_oracle.py <out.npz> [--gen N] [--prompt FILE] [--device cuda]
Fixture default: short zh-TW meeting snippet (see system prompt style of
the meeting-agent fine-tune). Needs torch + transformers in ~/k2/venv.
"""
import sys

import numpy as np

PROMPT = ("S1 [9:02:11] 各位早安,今天討論資訊系統預算,總共編列 1200 萬元。\n"
          "S2 [9:03:44] 我建議改用線上報名,減少現場排隊,經費大概可以省 80 萬。\n"
          "S1 [9:05:02] 好,那這個案子就照案通過,請主辦單位兩週內提出書面報告。")


def main():
    out, gen, dev = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8, "cuda"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("torch", torch.__version__, flush=True)
    if dev == "cuda" and not torch.cuda.is_available():
        print("no cuda, falling back to cpu")
        dev = "cpu"
    tok = AutoTokenizer.from_pretrained("IFM/K2-Horizon-0.9B",
                                        trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        "IFM/K2-Horizon-0.9B", trust_remote_code=True,
        torch_dtype=torch.float32 if dev == "cpu" else torch.bfloat16)
    mdl = mdl.to(dev).eval()
    ids = tok("使用者:" + PROMPT + "\n助理:", return_tensors="pt").input_ids.to(dev)
    print("prompt ids:", ids.shape, ids[0, :8].tolist(), flush=True)
    with torch.no_grad():
        pre = mdl(ids, output_hidden_states=True, use_cache=True)
        layers = [h[0, -1].float().cpu().numpy() for h in pre.hidden_states[1:]]
        logits = pre.logits[0, -1].float().cpu().numpy()
        gen_ids = mdl.generate(ids, max_new_tokens=gen, do_sample=False,
                               eos_token_id=1, pad_token_id=64255)
    gen_list = gen_ids[0].tolist()[ids.shape[1]:]
    print("greedy continuation:", gen_list, flush=True)
    print("decoded:", tok.decode(gen_list)[:300], flush=True)
    np.savez(out, prompt=ids[0].cpu().numpy(),
             layers=np.stack(layers), logits=logits,
             gen=np.array(gen_list, dtype=np.int64))
    print(f"saved {out}: layers {np.stack(layers).shape}", flush=True)


if __name__ == "__main__":
    main()
