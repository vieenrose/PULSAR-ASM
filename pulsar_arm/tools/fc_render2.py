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
msgs = [
    {"role": "developer",
     "content": "You are a model that can do function calling with the following functions"},
    {"role": "user", "content": "What's the temperature in London?"},
]
t = p.apply_chat_template(msgs, tools=[schema], tokenize=False,
                          add_generation_prompt=True)
print("RENDERED:", repr(t), flush=True)
