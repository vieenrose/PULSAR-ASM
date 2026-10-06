import json
import glob
import subprocess
import sys

snap = "/home/luigi/.cache/huggingface/hub/models--google--functiongemma-270m-it/snapshots/39eccb091651513a5dfb56892d3714c1b5b8276c"
tj = json.load(open(snap + "/tokenizer.json", encoding="utf-8"))
m = tj["model"]
print("model-type:", m.get("type"), "merges:", len(m.get("merges", [])), flush=True)
r = subprocess.run(
    ["/home/luigi/pulsar-rpi/v/bin/python", "/home/luigi/PULSAR-ARM/pulsar_arm/tools/mkvocab.py",
     snap + "/tokenizer.json", "/tmp/fcvocab.bin"],
    capture_output=True, text=True)
print("MKVOCAB:", r.returncode, (r.stdout + r.stderr)[-200:], flush=True)
r = subprocess.run(
    ["/home/luigi/pulsar-rpi/v/bin/python", "/home/luigi/PULSAR-ARM/pulsar_arm/tools/mkbpe.py",
     snap + "/tokenizer.json", "/tmp/fcbpe.bin"],
    capture_output=True, text=True)
print("MKBPE:", r.returncode, (r.stdout + r.stderr)[-200:], flush=True)
