from huggingface_hub import list_repo_files

for r in ["unsloth/gemma-3-270m-it-GGUF", "unsloth/gemma-3-270m-GGUF",
          "bartowski/google_gemma-3-270m-it-GGUF"]:
    try:
        files = list_repo_files(r)
        print(r, "->", [f for f in files if "Q8" in f or f.endswith(".json")][:8], flush=True)
    except Exception as e:
        print(r, "MISS", str(e)[:60], flush=True)
