from huggingface_hub import list_models

try:
    ms = list(list_models(author="google", search="gemma-3-270m", limit=50))
    for m in ms:
        print(m.id, flush=True)
except Exception as e:
    print("SEARCH-FAIL", str(e)[:120], flush=True)
