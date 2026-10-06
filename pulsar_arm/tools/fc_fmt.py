import glob
import re

t = open(glob.glob("/home/luigi/.cache/huggingface/hub/models--google--functiongemma-270m-it/snapshots/*/README.md")[0],
         encoding="utf-8").read()
for pat in ["Available functions", "start_of_turn>user", "```py", "```json",
            "function call", "FUNCTION", "<parameter", " declining"]:
    ms = [m.start() for m in re.finditer(re.escape(pat), t)]
    print(pat, len(ms), [m for m in ms][:4], flush=True)
