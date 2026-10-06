import subprocess
import sys

REPO = ("/home/luigi/.cache/huggingface/hub/"
        "models--google--gemma-3-270m-it-qat-q4_0-unquantized/snapshots/"
        "8f726c6a497fd439f0d6f726e52f8e3b439a26e5")
PROSE = ("The library holds thousands of books about history science mathematics "
         "poetry music art philosophy cooking travel gardening astronomy geology "
         "biology chemistry physics languages cultures inventions explorers oceans "
         "deserts forests cities villages bridges towers ships trains airplanes "
         "bicycles. Children play games in the park while parents watch from "
         "benches under old trees. ")
CASES = {
    "short": "<bos>The capital of France is",
    "mid": "<bos>" + " ".join(["word"] * 30) + "\nThe capital of France is",
    "long": "<bos>" + PROSE * 2 + "The capital of France is",
}
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(REPO, trust_remote_code=False)
for name, text in CASES.items():
    p = f"/tmp/c_{name}.txt"
    open(p, "w").write(text)
    print(name, "n=", len(tk.encode(text, add_special_tokens=False)), flush=True)
