import os
os.environ["TORCH_DISABLE_MKLDNN"] = "1"
os.environ["MKL_DISABLE"] = "1"
import torch
torch.backends.mkldnn.enabled = False
from transformers import AutoProcessor, AutoModelForCausalLM

repo = "/home/luigi/.cache/huggingface/hub/models--google--functiongemma-270m-it/snapshots/39eccb091651513a5dfb56892d3714c1b5b8276c"
p = AutoProcessor.from_pretrained(repo, trust_remote_code=False)
m = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32,
                                         device_map="cpu")
m.eval()
schema = {
    "type": "function",
    "function": {
        "name": "get_current_temperature",
        "description": "Gets the current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string",
                             "description": "The city name, e.g. San Francisco"},
            },
            "required": ["location"],
        },
    },
}
msgs = [
    {"role": "developer",
     "content": "You are a model that can do function calling with the following functions"},
    {"role": "user", "content": "What's the temperature in London?"},
]
inp = p.apply_chat_template(msgs, tools=[schema], add_generation_prompt=True,
                            return_dict=True, return_tensors="pt")
print("NIDS:", inp["input_ids"].shape[1], flush=True)
with torch.no_grad():
    out = m(input_ids=inp["input_ids"])
lg = out.logits[0, -1]
top = torch.topk(lg, 8)
print("TOP8:", list(zip(top.indices.tolist(), [round(float(v), 3) for v in top.values])), flush=True)
print("ARGMAX:", int(lg.argmax()), flush=True)
