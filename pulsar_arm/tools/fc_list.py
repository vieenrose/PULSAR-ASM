from huggingface_hub import list_models

ms = list(list_models(author="google", search="functiongemma", limit=20))
for m in ms:
    print(m.id, flush=True)
