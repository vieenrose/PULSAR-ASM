import os
os.environ["TORCH_DISABLE_MKLDNN"] = "1"
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

TXT = "<bos>The capital of France is"
ids = tk.encode(TXT, add_special_tokens=False)
print("IDS:", ids, flush=True)
# also dump per-length cases as files for the asm engine
for n in (1, 2, 3, 4, 6):
    sub = ids[:n]
    open(f"/tmp/n{n}.txt", "w").write(tk.decode(sub))
    x = torch.tensor([sub])
    with torch.no_grad():
        lg = m(input_ids=x).logits[0, -1].float()
    top = torch.topk(lg, 3)
    toks = [tk.decode([i]) for i in top.indices.tolist()]
    print(f"n={n}: TOP3={list(zip(top.indices.tolist(), toks))}", flush=True)
