#!/bin/bash
# .auto/ab_env.sh ENVNAME N [ids...] - interleaved A/B of the SAME engine binary
# with/without ENVNAME=1 (run #45: r128 / tm kernel selection).
# Prints per-arm decode ms/token (mean of --gen 24 steps 21-44).
set -uo pipefail
ENVNAME=${1:?usage: ab_env.sh ENVNAME N}; N=${2:-3}
NANO=${NANO:-pesi@100.70.245.36}
IDS=${3:-"151644 872 198 852 279 3040 15584 323 825 3166 429 4344 304 1817 624 151645 198 151644 77091 198"}
ARGS=${4:-"--gen 24"}
ssh -o ConnectTimeout=30 "$NANO" "cd /home/pesi/exp && for r in \$(seq 1 $N); do
  for mode in base env; do
    OMP_NUM_THREADS=4 timeout 900 env \$( [ \$mode = env ] && echo $ENVNAME=1 ) ./fwd8_q1 /home/pesi/b8q1.gguf '$IDS' $ARGS 2>&1 | \
      python3 -c \"
import re,sys
v=[float(m.group(1)) for m in (re.search(r'BQ1TIME step=(\\d+) ms=([\\d.]+)',l) for l in sys.stdin) if m and int(m.group(1))>=21]
print('%-5s %8.2f  n=%d' % ('$ENVNAME', sum(v)/len(v), len(v)) if v else 'no data')
\"
  done
done" 2>&1 | grep -v locale
