#!/bin/bash
# Teacher-forced held-out ASR gate (run #210): zh fixture, engine fed HF's 224 golden ids,
# per-position engine argmax (LFM_LEGACYFEED=1 runs the head on every step) vs HF's ids.
# Shipped engine: 222/224; both misses are HF near-ties (pos 2 margin 0.082, pos 165 0.022,
# engine = HF's runner-up). Runs on picard; needs /tmp/zh_gold.txt (HF ids, space separated).
rm -rf /tmp/lfm3_asr.*
LFM_ASR_KEEP=1 LFM_ASR_PRINT_IDS=1 timeout 900 ./lfm3_asr . gate_60s_zh_16k_mono.wav 1 --tokens-only > /tmp/tf0.txt 2>&1
emb=$(ls -d /tmp/lfm3_asr.*/* 2>/dev/null | grep -E "ready|proj" | head -1); echo "emb $emb $(stat -c %s $emb)"
ids=$(grep '^IDS ' /tmp/tf0.txt | cut -d' ' -f2); n=$(echo $ids | tr , '\n' | wc -l)
gold=$(tr ' ' , < /tmp/zh_gold.txt | sed 's/,$//')
echo "n $n"; head -c 300 $(ls -t /tmp/lfm3_asr.*.engine.txt | head -1) | head -3
LFM_LEGACYFEED=1 timeout 1500 ./lfm3_nanode lfm3-audio-2b-q8.lfm3q8 --audio-emb $emb --audio-at 15 --prompt $ids,$gold --steps $((n+224)) > /tmp/tf1.txt 2>&1
echo rc $?; grep -m3 -n "^step\| step " /tmp/tf1.txt; tail -3 /tmp/tf1.txt
python3 - $n << 'P'
import re,sys
n=int(sys.argv[1]); g=open('/tmp/zh_gold.txt').read().split()
st={}
python3 /tmp/tfs.py
