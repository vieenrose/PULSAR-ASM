#!/bin/bash
# Accuracy gate for the Gemma 4 E2B Nano loop (never overridden): runs gates.sh on the
# binaries the last measure built - host asm test, GPU kernel test (all 275 Q4_0
# tensors, head, F16, GeGLU, Q8_K), and the en / zh / sampled-en goldens.
set -u
PEER=picard@picard-desktop
B=/home/picard/g4/bench/src
timeout 1500 ssh -o BatchMode=yes $PEER "cd $B && sh gates.sh /home/picard/g4/gemma-4-E2B_q4_0-it.gguf /home/picard/g4/g4_rope_2048.bin" > /tmp/g4checks.txt 2>&1
cat /tmp/g4checks.txt | tail -8
grep -q "ALL GATES OK" /tmp/g4checks.txt || { echo "CHECKS FAIL"; exit 4; }
echo "CHECKS OK"
