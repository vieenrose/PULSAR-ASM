import struct
import json
import glob
import os
from huggingface_hub import snapshot_download

p = snapshot_download("google/functiongemma-270m-it",
                      allow_patterns=["model.safetensors", "tokenizer.json",
                                      "config.json", "tokenizer_config.json"])
print("SNAP:", p, flush=True)
f = glob.glob(p + "/model.safetensors")[0]
print("SIZE-MB:", os.path.getsize(f) // 1048576, flush=True)
b = open(f, "rb")
n = struct.unpack("<Q", b.read(8))[0]
hdr = json.loads(b.read(n))
names = sorted(hdr.keys() - {"__metadata__"})
print("N-TENSORS:", len(names), flush=True)
import collections
dtypes = collections.Counter(hdr[k]["dtype"] for k in names)
print("DTYPES:", dict(dtypes), flush=True)
for k in names:
    if "layer.0." in k or "embed" in k or "output" in k or "norm" in k and ".0." not in k:
        print(" ", k, hdr[k]["dtype"], hdr[k]["shape"], flush=True)
        if sum(1 for _ in names if "layer.0." in _) > 0 and k.startswith("model.layers.1."):
            break
tj = json.load(open(glob.glob(p + "/tokenizer.json")[0], encoding="utf-8"))
v = tj["model"]["vocab"]
print("VOCAB:", len(v), flush=True)
print("ADDED:", [(t.get("content"), t.get("id")) for t in tj.get("added_tokens", [])][:10], flush=True)
print("CHAT-TEMPLATE:", repr(tj.get("chat_template", ""))[:300], flush=True)
cfg = json.load(open(glob.glob(p + "/config.json")[0]))
print("ARCH:", cfg.get("num_hidden_layers"), cfg.get("hidden_size"),
      cfg.get("num_attention_heads"), cfg.get("num_key_value_heads"),
      cfg.get("vocab_size"), flush=True)
