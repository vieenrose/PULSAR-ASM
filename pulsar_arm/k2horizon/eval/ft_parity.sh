#!/bin/bash
# K2 meeting-FT native (asm) vs LiteRT TFLite greedy parity + FT-vs-base behavior
set -e
cd ~/k2
BLOB=k2h_ft_q4.blob
[ -f "$BLOB" ] || { echo "no blob"; exit 1; }

# 680-id FT prompt (fine-tune tokenizer + IFM template + system prompt + window)
IDS=$(~/k2/venv/bin/python -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('finetune/tokenizer', trust_remote_code=True)
print(' '.join(map(str, t.encode(open('finetune/prompt_ft.txt').read()))))
")
N=$(echo $IDS | wc -w)
echo "prompt ids: $N"

# 1) native asm greedy
T0=$(date +%s%N)
taskset -c 5 ./k2_core "$BLOB" $IDS --gen 400 > ft_asm.txt 2>&1
T1=$(date +%s%N)
python3 -c "print(f'asm: {(($T1-$T0)/1e6):.0f} ms')"

# 2) TFLite greedy via driver (driver resolves k2_meeting_q4.tflite + prompt
#    relative to CWD -> run inside finetune/)
T0=$(date +%s%N)
(cd finetune && ~/k2/venv/bin/python k2_lite_driver.py prompt_ft.txt \
    --gen 400 --temp 0 --tok ./tokenizer --ids-out ../ft_tflite_ids.txt \
    > ../ft_tflite.txt 2>&1)
T1=$(date +%s%N)
python3 -c "print(f'tflite: {(($T1-$T0)/1e6):.0f} ms')"

# 3) compare generated token streams
~/k2/venv/bin/python - << 'EOF'
import re
asm_ids = []
for ln in open('ft_asm.txt'):
    m = re.match(r'step (\d+) id_in=(\d+) top=(\d+)', ln)
    if m and int(m.group(1)) >= 680:
        asm_ids.append(int(m.group(3)))   # top of step n = token n+1... 
# tokens: step s>=n0 prints id_in (the fed token) -> generated tokens are id_in of steps 680..1079
# plus top of last step. Use id_in chain for a clean compare:
asm_gen = []
tf = list(map(int, open('ft_tflite_ids.txt').read().split()))
for ln in open('ft_asm.txt'):
    m = re.match(r'step (\d+) id_in=(\d+)', ln)
    if m and int(m.group(1)) >= 680:
        asm_gen.append(int(m.group(2)))
# asm_gen = fed tokens for steps 680.. = tf[0..] (first is tf[0] chosen after step 679)
# align: asm step680 id_in = first generated token = tf[0]? driver ids-out = generated tokens only
n = min(len(asm_gen), len(tf))
match = sum(1 for i in range(n) if asm_gen[i] == tf[i])
print(f'token agreement: {match}/{n}')
first = next((i for i in range(n) if asm_gen[i] != tf[i]), None)
print('first divergence at', first)
print('asm  :', asm_gen[:12])
print('tflite:', tf[:12])
EOF

# 4) FT vs BASE behavior on the same window (protocol check)
NEURBASE=$(~/k2/venv/bin/python -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('IFM/K2-Horizon-0.9B', trust_remote_code=True)
# same content, BASE template: base chat template + same system+window text
msgs = [{'role':'system','content': open('finetune/system_prompt.txt').read()},
        {'role':'user','content': open('finetune/window.txt').read()}]
try:
    s = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
except Exception:
    s = open('finetune/system_prompt.txt').read() + '\n' + open('finetune/window.txt').read() + '\n'
print(' '.join(map(str, t.encode(s))))
")
taskset -c 5 ./k2_core k2h_09_q4.blob $NEURBASE --gen 150 > base_window.txt 2>&1 || true
echo "=== base on meeting window (first 6 lines):"
head -6 base_window.txt
echo "=== FT native (first 16 lines):"
head -16 ft_asm.txt
