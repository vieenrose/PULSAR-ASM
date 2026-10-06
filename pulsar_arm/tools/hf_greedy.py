"""HF greedy reference for the asm engine (torch, fp32 + eager attention).

`model.generate()` SIGILLs on this Pi, so the loop is manual: prefill the given
ids, then argmax one token at a time with a KV cache. Prints the continuation
ids and text, plus the top-5 logits at the last prefill step.

Usage:
    hf_greedy.py <repo-dir-or-cache-name> <ids,comma,separated> <n_new>
    hf_greedy.py google/gemma-3-1b-it-qat-q4_0-unquantized 2,107,1567 32
"""
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.mkldnn.enabled = False
torch.set_num_threads(1)

# fp32 needs ~4x the checkpoint (1B fp32 does not fit the Pi's 3.8 GB), so the
# dtype is selectable: HF_DTYPE=bfloat16 for the big checkpoints.
DTYPE = {"float32": torch.float32, "bfloat16": torch.bfloat16,
         "float16": torch.float16}[os.environ.get("HF_DTYPE", "float32")]


def main():
    repo, ids, n = sys.argv[1], [int(v) for v in sys.argv[2].split(",")], int(sys.argv[3])
    tok = AutoTokenizer.from_pretrained(repo)
    m = AutoModelForCausalLM.from_pretrained(repo, dtype=DTYPE,
                                             device_map="cpu",
                                             attn_implementation="eager")
    m.eval()
    gen = []
    with torch.no_grad():
        out = m(input_ids=torch.tensor([ids]), use_cache=True)
        lg = out.logits[0, -1]
        print("TOP5_LAST_PREFILL:",
              ", ".join(f"{int(i)}:{float(v):.4f}"
                        for i, v in zip(*torch.topk(lg, 5))), flush=True)
        for _ in range(n):
            nxt = int(lg.argmax(-1).item())
            gen.append(nxt)
            out = m(input_ids=torch.tensor([[nxt]]),
                    past_key_values=out.past_key_values, use_cache=True)
            lg = out.logits[0, -1]
    print("PREFILL:", ids)
    print("GENIDS:", gen)
    print("GENTEXT:", repr(tok.decode(gen)))


if __name__ == "__main__":
    main()
