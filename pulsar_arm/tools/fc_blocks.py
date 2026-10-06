import glob
import re

t = open(glob.glob("/home/luigi/.cache/huggingface/hub/models--google--functiongemma-270m-it/snapshots/*/README.md")[0],
         encoding="utf-8").read()
blocks = re.findall(r"```(?:python|py)?\n(.*?)```", t, re.S)
print("CODE-BLOCKS:", len(blocks), flush=True)
for i, b in enumerate(blocks):
    print(f"===== BLOCK {i} ({len(b)} chars) =====", flush=True)
    print(b[:2200], flush=True)
