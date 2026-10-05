"""First light: QAT-mobile E2B on Raspberry Pi 4 (CPU, transformers only).

Text-only probing: load the checkpoint, run one short English and one short
Traditional Chinese prompt greedily, report text + tok/s + peak RSS. This is
the baseline any ARM-kernel port in PULSAR-ASM has to beat for fidelity.
"""
import os
import resource
import time

import torch

MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-4-E2B-it-qat-mobile-transformers")


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def main():
    from transformers import AutoProcessor, AutoModelForMultimodalLM

    t0 = time.time()
    print(f"load {MODEL_ID} ...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=False)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype="auto", device_map="cpu",
    )
    model.eval()
    print(f"loaded in {time.time()-t0:.0f}s, peak RSS {peak_gb():.2f} GB, "
          f"dtype {next(model.parameters()).dtype}", flush=True)

    for tag, user in (("en", "Name one city in France."),
                      ("zh", "一個星期有幾天？請用中文回答。")):
        messages = [{"role": "user", "content": user}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True, enable_thinking=False,
        )
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        dt = time.time() - t0
        text = processor.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        n = out.shape[1] - inputs["input_ids"].shape[1]
        print(f"[{tag}] {n} tok in {dt:.1f}s = {n/max(dt, 1e-9):.2f} tok/s, "
              f"peak RSS {peak_gb():.2f} GB", flush=True)
        print(f"[{tag}] {text!r}", flush=True)


if __name__ == "__main__":
    main()
