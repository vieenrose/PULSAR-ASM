import subprocess, sys
T = "/home/louis/slms/prism-llama/build-cpu/bin/llama-tokenize"; M = "/home/louis/slms/bonsai2/Ternary-Bonsai-2-4B-PTQ1_0.gguf"
art = open("/home/louis/PULSAR-ASM/.auto/prompt_zh_tw.txt").read().rstrip("\n")
cut = art.rindex("\n\n請根據以上文章")
body, q = art[:cut], art[cut + 2:]
readme = open("/home/louis/PULSAR-ASM/README.md").read()
a = readme.index("### How it works")
app = readme[a:]
def build(n):
    return ("<|im_start|>user\n" + body + "\n\n附錄（英文技術文件，與問題無關）：\n\n" + app[:n].rstrip() + "\n\n" + q +
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
def tok(s):
    open(sys.argv[1] + "/p4k.txt", "w").write(s)
    o = subprocess.run([T, "-m", M, "-f", sys.argv[1] + "/p4k.txt", "--ids", "--log-disable"], capture_output=True, text=True).stdout
    return [int(x) for x in o.strip().splitlines()[-1].strip("[]").split(",")]
lo, hi = 0, len(app)
while lo < hi:
    mid = (lo + hi + 1) // 2
    if len(tok(build(mid))) <= 3968: lo = mid
    else: hi = mid - 1
ids = tok(build(lo)); print(len(ids), lo, ids[:6], ids[-10:])
open(sys.argv[1] + "/p4k_ids.txt", "w").write(" ".join(map(str, ids)) + "\n")
open(sys.argv[1] + "/p4k_text.txt", "w").write(build(lo))
