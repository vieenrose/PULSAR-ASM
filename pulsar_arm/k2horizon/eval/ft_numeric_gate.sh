#!/bin/bash
# FT numerical gate: Q4 blob (asm + C) must agree with the fp16 reference blob
# on a short prompt's greedy stream (base was validated the same way, 8/8).
set -e
cd ~/k2
IDS=$(~/k2/venv/bin/python -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('finetune/tokenizer', trust_remote_code=True)
ids = t.encode(open('finetune/prompt_ft.txt').read())[:20]
print(' '.join(map(str, ids)))")
echo "prompt (first 20): $IDS"
G=8

taskset -c 5 ./fwd_k2q_v2 k2h_ft_fp16.blob $IDS --gen $G > ft_ref.txt 2>&1
taskset -c 5 ./fwd_k2q_v2 k2h_ft_q4.blob   $IDS --gen $G > ft_c_q4.txt 2>&1
taskset -c 5 ./k2_core     k2h_ft_q4.blob  $IDS --gen $G > ft_asm_q4.txt 2>&1

~/k2/venv/bin/python - << 'EOF'
import re
def stream(p):
    out = []
    for ln in open(p):
        m = re.match(r'step (\d+) id_in=(\d+) top=(\d+) logit=([\d.\-e+]+)', ln)
        if m: out.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)))
    return out
ref = stream('ft_ref.txt')
c4  = stream('ft_c_q4.txt')
a4  = stream('ft_asm_q4.txt')
print(f'steps: ref={len(ref)} c_q4={len(c4)} asm_q4={len(a4)}')
n = min(len(ref), len(c4), len(a4))
top_ref = [s[2] for s in ref[:n]]
ok_c = sum(1 for i in range(n) if c4[i][2] == top_ref[i])
ok_a = sum(1 for i in range(n) if a4[i][2] == top_ref[i])
print(f'argmax match vs fp16 ref: C-Q4 {ok_c}/{n}, asm-Q4 {ok_a}/{n}')
print('ref tops :', top_ref[:8])
print('asm tops :', [s[2] for s in a4[:8]])
EOF
