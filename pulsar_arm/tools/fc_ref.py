import json
import glob

fc = "/home/luigi/.cache/huggingface/hub/models--google--functiongemma-270m-it/snapshots/39eccb091651513a5dfb56892d3714c1b5b8276c/tokenizer.json"
tj = json.load(open(fc, encoding="utf-8"))
fm = tj["model"]["merges"]
j = glob.glob("/home/luigi/.cache/huggingface/hub/models--google--gemma-3-270m-it-qat-q4_0-unquantized/snapshots/*/tokenizer.json")[0]
tm = json.load(open(j, encoding="utf-8"))["model"]["merges"]
print("merge lists identical:", [tuple(x) for x in fm] == [tuple(x) for x in tm], flush=True)
# FC tokenize the 18 strings via HF (oracle for asm BPE check)
from transformers import AutoTokenizer
import os
tk = AutoTokenizer.from_pretrained(os.path.dirname(fc), trust_remote_code=False)
tests = ['Hello world', 'The capital of France is', 'Hello', '\n', ' a',
         'caf\u00e9 \u65e5\u672c\u8a9e', 'Name', 'THE', '  spaces  ', 'x=1+2*3',
         '\u00e9', '\u65e5', 'A', '\U0001F600', '\x01', '\x7f', '\u200b', ' Hello']
for t in tests:
    print(repr(t)[:24], tk.encode(t, add_special_tokens=False), flush=True)
