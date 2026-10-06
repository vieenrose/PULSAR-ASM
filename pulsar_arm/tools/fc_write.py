"""Write the official FunctionGemma example prompts to /tmp for the asm engine.

Two files, same tool schema, different user turn:
  /tmp/tool_official.txt       What's the temperature in London?   (English)
  /tmp/tool_official_zhtw.txt  倫敦的溫度是多少？                      (zh-TW)

Usage (on the Pi, cache only):
  HF_HUB_OFFLINE=1 ./v/bin/python fc_write.py
  ./core <functiongemma.safetensors> fcvocab.bin 2 f fcbpe.bin 1000 950 16 \
      < /tmp/tool_official.txt        # -> call:get_current_temperature{location:London}
"""
from transformers import AutoProcessor

p = AutoProcessor.from_pretrained("google/functiongemma-270m-it",
                                  trust_remote_code=False)
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

for path, question in (("/tmp/tool_official.txt", "What's the temperature in London?"),
                       ("/tmp/tool_official_zhtw.txt", "倫敦的溫度是多少？")):
    msgs = [
        {"role": "developer",
         "content": "You are a model that can do function calling with the following functions"},
        {"role": "user", "content": question},
    ]
    t = p.apply_chat_template(msgs, tools=[schema], tokenize=False,
                              add_generation_prompt=True)
    open(path, "w", encoding="utf-8").write(t)
    print(f"{path}: {len(t)} chars ({question})", flush=True)
