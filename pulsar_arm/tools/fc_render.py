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
                             "description": "The city, e.g. Paris."},
            },
            "required": ["location"],
        },
    },
}
msgs = [
    {"role": "developer",
     "content": "You can call functions. Available: " + str([schema])},
    {"role": "user", "content": "What is the temperature in Paris?"},
]
t = p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print("RENDERED:", repr(t), flush=True)
print("IDS:", p.tokenizer.encode(t, add_special_tokens=False)[:20], flush=True)
