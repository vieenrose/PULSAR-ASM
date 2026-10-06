import os
os.environ["TORCH_DISABLE_MKLDNN"] = "1"
os.environ["MKL_DISABLE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
torch.backends.mkldnn.enabled = False
torch.set_num_threads(1)
from transformers import AutoTokenizer, AutoModelForCausalLM

repo = ("/home/luigi/.cache/huggingface/hub/"
        "models--google--gemma-3-270m-it-qat-q4_0-unquantized/snapshots/"
        "8f726c6a497fd439f0d6f726e52f8e3b439a26e5")
tk = AutoTokenizer.from_pretrained(repo, trust_remote_code=False)
m = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32,
                                         device_map="cpu",
                                         attn_implementation="eager")
m.eval()

PROSE = ("The library holds thousands of books about history science mathematics "
         "poetry music art philosophy cooking travel gardening astronomy geology "
         "biology chemistry physics languages cultures inventions explorers oceans "
         "deserts forests cities villages bridges towers ships trains airplanes "
         "bicycles. Children play games in the park while parents watch from "
         "benches under old trees. ")

CASES = {
    "short": "<bos>The capital of France is",
    "mid": "<bos>" + " ".join(["word"] * 30) + "\nThe capital of France is",
    "long": "<bos>" + PROSE * 2 + "The capital of France is",
}
for name, text in CASES.items():
    ids = tk.encode(text, add_special_tokens=False)
    x = torch.tensor([ids])
    with torch.no_grad():
        lg = m(input_ids=x).logits[0, -1].float()
    top = torch.topk(lg, 3)
    toks = [tk.decode([i]) for i in top.indices.tolist()]
    print(f"{name}: n={len(ids)} TOP3={list(zip(top.indices.tolist(), toks))} "
          f"logits={[round(float(v),2) for v in top.values]}", flush=True)
