from huggingface_hub import list_models

for q in ["functiongemma", "function gemma 270m", "gemma 270m function calling"]:
    try:
        ms = list(list_models(search=q, limit=15))
        print("##", q, flush=True)
        for m in ms:
            print(" ", m.id, flush=True)
    except Exception as e:
        print("FAIL", q, str(e)[:80], flush=True)
