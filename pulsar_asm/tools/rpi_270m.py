"""First light, part 2: gemma-3-270m QAT-unquantized on Pi 4 (CPU).

Same deal as before: greedy en + zh prompts, text + tok/s + peak RSS. Small
enough to stay out of swap.
"""
import os
import resource
import time

import torch

MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-3-270m-qat-q4_0-unquantized")


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    print(f"load {MODEL_ID} ...", flush=True)
    tk = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="cpu")
    model.eval()
    print(f"loaded in {time.time()-t0:.0f}s, peak RSS {peak_gb():.2f} GB", flush=True)

    for tag, user in (("en", "Name one city in France."),
                      ("zh", "一個星期有幾天？請用中文回答。")):
        # this tokenizer ships no chat_template; Gemma-3 IT format by hand
        txt = ("<bos><start_of_turn>user\n" + user +
               "\n<end_of_turn>\n<start_of_turn>model\n")
        ids = tk(txt, return_tensors="pt")["input_ids"]
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=40, do_sample=False,
                                 pad_token_id=tk.eos_token_id)
        dt = time.time() - t0
        n = out.shape[1] - ids.shape[1]
        print(f"[{tag}] {n} tok in {dt:.1f}s = {n/max(dt, 1e-9):.2f} tok/s, "
              f"peak RSS {peak_gb():.2f} GB", flush=True)
        print(f"[{tag}] {tk.decode(out[0][ids.shape[1]:], skip_special_tokens=True)!r}",
              flush=True)


if __name__ == "__main__":
    main()
