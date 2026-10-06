from transformers import AutoTokenizer
import os

snap = "/home/luigi/.cache/huggingface/hub/models--google--functiongemma-270m-it/snapshots/39eccb091651513a5dfb56892d3714c1b5b8276c"
tk = AutoTokenizer.from_pretrained(snap, trust_remote_code=False)
t = open("/tmp/toolo.txt", encoding="utf-8").read()
ids = tk.encode(t, add_special_tokens=False)
print("HF-N:", len(ids), flush=True)
print("HF-HEAD:", ids[:12], flush=True)
print("HF-TAIL:", ids[-8:], flush=True)
