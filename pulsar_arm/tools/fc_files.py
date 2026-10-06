from huggingface_hub import list_repo_files

for r in ["google/functiongemma-270m-it",
          "unsloth/functiongemma-270m-it-GGUF"]:
    try:
        files = list_repo_files(r)
        print("##", r, flush=True)
        for f in files:
            if f.endswith((".safetensors", ".gguf", ".json")) and ("Q4" in f or "q4" in f or "token" in f or "config" in f or f.endswith(".safetensors")):
                print("  ", f, flush=True)
    except Exception as e:
        print(r, "MISS", str(e)[:80], flush=True)
