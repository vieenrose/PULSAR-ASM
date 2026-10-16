#!/bin/bash
# Bonsai-2 4B long-context harness on the Jetson Nano: prefill of the 3968-token zh-TW
# prompt (.auto/prompt_b2_4k_ids.txt, built by mk_prompt_b2_4k.py) then 64 greedy tokens.
# Syncs + builds fwd35, pins clocks, checks the stream against the fork golden
# .auto/golden_b2_4b_4k.txt and prints METRIC lines: prefill_tps, decode_tps,
# score = sqrt(prefill_tps * decode_tps), peak_rss_kb (/usr/bin/time %M), golden PASS/FAIL.
# Engine env goes through B2_EXTRA="E_X=1 ...". NOBUILD=1 skips sync+build.
set -u
R=pesi@100.70.245.36; D=/home/pesi/b2
cd "$(dirname "$0")/.."
if [ -z "${NOBUILD:-}" ]; then
  scp -q pulsar_arm/bonsai2/fwd35_sm53.cu pulsar_arm/bonsai2/glibc_mathf.h .auto/prompt_b2_4k_ids.txt .auto/golden_b2_4b_4k.txt $R:$D/ 2>&1 | grep -v locale
  ssh $R "cd $D && /usr/local/cuda/bin/nvcc -O3 -gencode arch=compute_53,code=sm_53 --fmad=false \
    -Xcompiler -fopenmp,-ffp-contract=off -o fwd35 fwd35_sm53.cu -lgomp 2>&1 | grep -i error" 2>&1 | grep -v locale
fi
ssh $R "cd $D; sudo -n jetson_clocks; sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1
  env ${B2_EXTRA:-} /usr/bin/time -f 'RSSKB %M' ./fwd35 Ternary-Bonsai-2-4B-PTQ1_0.gguf 64 \$(cat prompt_b2_4k_ids.txt) > o4k.txt 2> e4k.txt
  cmp -s o4k.txt golden_b2_4b_4k.txt && g=PASS || g=FAIL
  grep -o 'prefill.*' e4k.txt; echo GOLDEN \$g; grep RSSKB e4k.txt" 2>&1 | grep -v locale | python3 -c '
import sys,re,math
t=sys.stdin.read(); print(t.strip())
p=re.search(r"prefill \d+ tok [\d.]+ s = ([\d.]+)",t); d=re.search(r"decode \d+ tok [\d.]+ s = ([\d.]+)",t)
r=re.search(r"RSSKB (\d+)",t); g="PASS" if "GOLDEN PASS" in t else "FAIL"
if p and d:
  P,Dd=float(p.group(1)),float(d.group(1))
  print(f"METRIC prefill_tps={P}\nMETRIC decode_tps={Dd}\nMETRIC score={math.sqrt(P*Dd):.4f}")
print(f"METRIC peak_rss_kb={r.group(1) if r else -1}\nMETRIC golden={g}")'
