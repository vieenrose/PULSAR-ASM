#!/usr/bin/env python
"""Speech-turn calibration set for gptq_lfm3.py: HF's own interleaved text -> speech turns
for prompts disjoint from the speak gate fixtures (speak_hf_oracle.py PROMPTS 0..4), saved
as (ids with audio_out placeholders, audio codes [n_frames, 12]) so a forward pass can
replay each turn teacher-forced: model(input_ids=ids, audio_out_codes=codes).

  python .auto/lfm3/speech_calib.py --n 80 --out /tmp/lfm3_speech_calib.pt
"""
import argparse, os, random, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from divergence_floor import load, REPO                   # noqa: E402
from gptq_lfm3 import TOPICS                              # noqa: E402

SYSTEMS = ["You are a helpful assistant.", "You are a friendly voice assistant.",
           "You are a concise narrator.", "You answer in a warm, spoken style."]
ASKS = ["Tell me something interesting about {}.", "Explain {} in two sentences.",
        "What do you think about {}?", "Give me one tip related to {}.",
        "Describe {} as if to a child.", "Why does {} matter?", "Say a short sentence about {}."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--slots", type=int, default=120)
    ap.add_argument("--out", default="/tmp/lfm3_speech_calib.pt")
    ap.add_argument("--seed", type=int, default=4321, help="prompt sampler seed (eval sets use another)")
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    model = load(torch.float16)
    rng = random.Random(a.seed)
    turns, t0 = [], time.time()
    for i in range(a.n):
        topic = rng.choice([t for t in TOPICS if t.isascii()])
        text = ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"
                % (rng.choice(SYSTEMS), rng.choice(ASKS).format(topic)))
        ids = tok(text, return_tensors="pt").input_ids.to("cuda")
        torch.manual_seed(a.seed * 7 + i)
        with torch.no_grad():
            out = model.generate(ids, generation_mode="interleaved", audio_temperature=0.0,
                                 max_new_tokens=a.slots, do_sample=False, return_audio_eos=True)
        seq = out.sequences[0].tolist()
        codes = out.audio_codes[0].cpu()                 # [12, n_frames]
        nph = sum(1 for t in seq if t == model.config.audio_out_token_id)
        if nph != codes.shape[1]:
            print("turn %d: %d placeholders vs %d frames, skipped" % (i, nph, codes.shape[1])); continue
        turns.append((seq, codes.t().contiguous().to(torch.int64)))
        if i % 10 == 0:
            print("turn %d: %d ids, %d frames (%.0f s)" % (i, len(seq), nph, time.time() - t0), flush=True)
    torch.save(turns, a.out)
    print("saved %d turns, %d positions, %d frames"
          % (len(turns), sum(len(s) for s, _ in turns), sum(c.shape[0] for _, c in turns)))


if __name__ == "__main__":
    main()
