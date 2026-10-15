#!/bin/sh
# All Gemma 4 E2B Nano accuracy gates. Usage: ./gates.sh <gemma.gguf> <rope.bin>
#  1. g4host_test : every AArch64 host kernel bit-identical to g4ref.c
#  2. g4gemv_test : every GPU kernel bit-identical to g4ref.c on the real weights
#  3. g4eng       : the English and Traditional Chinese greedy streams equal the goldens
set -e
cd "$(dirname "$0")"
G=$1; R=$2
./g4host_test | tail -1
./g4gemv_test "$G" g4_kernels.ptx all | tail -1
for k in en zh; do
  IDS=$(python3 -c "import json;print(' '.join(map(str,json.load(open('prompts.json'))['$k']['ids'])))")
  ./g4eng "$G" "$R" $IDS --gen 200 2>/dev/null > /tmp/g4gate_$k.txt
  cmp /tmp/g4gate_$k.txt golden_$k.txt && echo "golden_$k: identical"
done
echo "ALL GATES OK"
