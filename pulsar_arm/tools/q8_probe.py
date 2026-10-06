from huggingface_hub import list_repo_files

CANDS = [
    "google/gemma-3-270m-it-qat-q8_0-gguf",
    "google/gemma-3-270m-it-qat-q8_0-unquantized",
    "google/gemma-3-270m-it-q8_0-gguf",
    "google/gemma-3-270m-qat-q8_0-unquantized",
]
for r in CANDS:
    try:
        files = list_repo_files(r)
        print(r, "->", [f for f in files if f.endswith((".gguf", ".safetensors", ".json"))][:8], flush=True)
    except Exception as e:
        print(r, "MISS", str(e)[:80], flush=True)
