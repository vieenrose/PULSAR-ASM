from transformers import AutoTokenizer
import glob

import os
p = glob.glob("/home/luigi/.cache/huggingface/hub/models--google--gemma-3-270m-it-qat-q4_0-unquantized/snapshots/*/tokenizer.json")[0]
tk = AutoTokenizer.from_pretrained(os.path.dirname(p), trust_remote_code=False)
msgs1 = [{"role": "user", "content": "Hello"}]
t1 = tk.apply_chat_template(msgs1, tokenize=False, add_generation_prompt=True)
print("TEMPLATE-TEXT-TURN1:", repr(t1), flush=True)
print("TEMPLATE-IDS-TURN1:", tk.encode(t1, add_special_tokens=False), flush=True)
msgs2 = msgs1 + [{"role": "assistant", "content": "Hi there!"},
                 {"role": "user", "content": "Bye"}]
t2 = tk.apply_chat_template(msgs2, tokenize=False, add_generation_prompt=True)
print("TEMPLATE-TEXT-TURN2:", repr(t2), flush=True)
print("TEMPLATE-IDS-TURN2:", tk.encode(t2, add_special_tokens=False), flush=True)
