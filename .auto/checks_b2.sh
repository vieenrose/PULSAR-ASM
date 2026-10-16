#!/bin/bash
# Bonsai-2 (qwen35, PTQ1_0) GPU engine gates on the Jetson Nano.
# Syncs fwd35_sm53.cu, builds fwd35 in /home/pesi/b2, and checks the 48-token greedy
# streams on prompts A and B against the fork goldens .auto/golden_b2_<m>_{A,B}.txt for
# every Ternary-Bonsai-2-<M>-PTQ1_0.gguf present in the work dir. Prefill is checked
# both batched-off (E_PCHUNK=1, the sm_53 default) and chunked (E_PCHUNK=8).
# Usage: bash .auto/checks_b2.sh [models...]   (default: 2B 4B 9B, skipped if absent)
set -u
R=pesi@100.70.245.36; D=/home/pesi/b2
cd "$(dirname "$0")/.."
scp -q pulsar_arm/bonsai2/fwd35_sm53.cu pulsar_arm/bonsai2/glibc_mathf.h .auto/golden_b2_*.txt $R:$D/ 2>&1 | grep -v locale
ssh $R "cd $D && /usr/local/cuda/bin/nvcc -O3 -gencode arch=compute_53,code=sm_53 --fmad=false \
  -Xcompiler -fopenmp,-ffp-contract=off -o fwd35 fwd35_sm53.cu -lgomp 2>&1 | grep -i error" 2>&1 | grep -v locale
A="248045 846 198 826 279 2943 15127 321 799 3065 421 4203 303 1754 13 248046 198 248045 74455 198 248068 271 248069 271"
B="248045 846 198 99270 115992 104647 3709 98404 98443 96681 96594 110476 127312 1710 248046 198 248045 74455 198 248068 271 248069 271"
fail=0
for M in ${@:-2B 4B 9B}; do
  m=$(echo $M | tr A-Z a-z)
  out=$(ssh $R "cd $D; [ -f Ternary-Bonsai-2-$M-PTQ1_0.gguf ] || { echo SKIP; exit; }
    for p in A B; do for c in 1 8; do [ \$p = A ] && ids=\"$A\" || ids=\"$B\"
      E_PCHUNK=\$c ./fwd35 Ternary-Bonsai-2-$M-PTQ1_0.gguf 48 \$ids > o_\$p.txt 2> e_\$p.txt
      cmp -s o_\$p.txt golden_b2_${m}_\$p.txt && r=PASS || r=FAIL
      echo \"$M prompt \$p chunk \$c \$r \$(grep -o 'prefill.*' e_\$p.txt | tr '\n' ' ')\"; done; done" 2>&1 | grep -v locale)
  echo "$out"; echo "$out" | grep -q FAIL && fail=1
done
[ $fail = 0 ] && echo "checks_b2: ALL PASS" || { echo "checks_b2: FAIL"; exit 1; }
