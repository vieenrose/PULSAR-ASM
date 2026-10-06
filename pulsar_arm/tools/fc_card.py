from huggingface_hub import snapshot_download
import glob

p = snapshot_download("google/functiongemma-270m-it",
                      allow_patterns=["README.md"])
f = glob.glob(p + "/README.md")[0]
txt = open(f, encoding="utf-8").read()
print("LEN:", len(txt), flush=True)
i = txt.find("unction call")
print(txt[max(0, i - 200):i + 1500] if i > 0 else txt[:1500], flush=True)
