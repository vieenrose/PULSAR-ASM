#!/bin/bash
# .auto/ab_kernel_pair.sh [GITREF] [N] - cumulative kernel audit with the arm
# order BALANCED across reps (see #49: a fixed order biases the second arm by
# ~1.2-1.6%, which is the size of the effects this session measures).
#
# Builds qk_old (kernel TU from GITREF) and qk_new (working tree) ONCE against the
# same host objects, then runs N pairs, alternating which binary runs first, and
# reports R = geomean(sqrt(o1/o2)) where o1 = new/second-over-base-first and
# o2 = base/second-over-new-first (see .auto/ab_pair.sh for the derivation).
set -uo pipefail
NANO=${NANO:-pesi@100.70.245.36}
REPO=/home/louis/PULSAR-ASM
SRC=$REPO/pulsar_arm/bonsai2
R=/home/pesi/exp
MODEL=/home/pesi/b8q1.gguf
IDS="151644 872 198 852 279 3040 15584 323 825 3166 429 4344 304 1817 624 151645 198 151644 77091 198"
GITREF=${1:-902dabc}
N=${2:-2}
CU=${CU:-pq1_swar_sm53.cu}

git -C "$REPO" show "$GITREF:pulsar_arm/bonsai2/$CU" > /tmp/kp_old.cu || exit 1
cat "$REPO/.auto/ab_delegates.inc" >> /tmp/kp_old.cu   # older TUs lack r128/r256/tm entry points
scp -q "$SRC/$CU" /tmp/kp_old.cu "$NANO:$R/" || { echo SCP_FAILED; exit 1; }

ssh -o ConnectTimeout=25 "$NANO" "cd $R && export PATH=/usr/local/cuda-10.2/bin:\$PATH
nvcc -gencode arch=compute_53,code=sm_53 -O3 -maxrregcount=48 --use_fast_math -Xptxas -O3,-dlcm=ca -std=c++11 -c kp_old.cu -o kp_old.o 2>&1 | grep -E ' error' | head -3
nvcc -gencode arch=compute_53,code=sm_53 -O3 -maxrregcount=48 --use_fast_math -Xptxas -O3,-dlcm=ca -std=c++11 -c $CU -o kp_new.o 2>&1 | grep -E ' error' | head -3
g++ fwd8_q1.o pq1_host.o pq1_glue.o kp_old.o -o kp_old -L/usr/local/cuda-10.2/targets/aarch64-linux/lib -lcudart -lm -lpthread -fopenmp 2>&1 | head -2; echo OLD_LK=\$?
g++ fwd8_q1.o pq1_host.o pq1_glue.o kp_new.o -o kp_new -L/usr/local/cuda-10.2/targets/aarch64-linux/lib -lcudart -lm -lpthread -fopenmp 2>&1 | head -2; echo NEW_LK=\$?" 2>&1 | grep -v locale

LOG=$REPO/.auto/last_ab_kernel_pair.txt
: > "$LOG"
for i in $(seq 1 "$N"); do
  for order in newfirst basefirst; do
    if [ "$order" = newfirst ]; then first=kp_new; second=kp_old; label="new-first"
    else first=kp_old; second=kp_new; label="new-second"; fi
    for b in "$first" "$second"; do
      t=$(ssh -o ConnectTimeout=25 "$NANO" "cd $R && PQ1_KVTQ=0 OMP_NUM_THREADS=4 timeout 900 ./$b $MODEL '$IDS' --gen 24 2>/dev/null" \
          | python3 -c "
import re, sys
v=[float(m.group(2)) for m in (re.match(r'BQ1TIME step=(\d+) ms=([0-9.]+)', l) for l in sys.stdin) if m and int(m.group(1))>=20]
print('%.4f' % (1000.0/(sum(v)/len(v))) if v else '')
")
      pos=$([ "$b" = kp_new ] && echo new || echo base)
      echo "RESULT rep$i.order=$label arm=$pos $t" | tee -a "$LOG"
    done
  done
done

echo "--- balanced cumulative ratio ($GITREF -> HEAD) ---"
python3 - "$LOG" <<'PY'
import math, re, statistics as st, sys
by = {}
for ln in open(sys.argv[1], errors="replace"):
    m = re.match(r"RESULT (rep\d+)\.order=(\S+) arm=(\w+) ([0-9.]+)\s*$", ln)
    if m:
        by.setdefault(m.group(1), {})[f"{m.group(2)}.{m.group(3)}"] = float(m.group(4))
R, B = [], []
for rep, d in sorted(by.items()):
    # keys are "order.arm" so the two runs of an order cannot overwrite each other
    if "new-second.new" in d and "new-second.base" in d and "new-first.new" in d and "new-first.base" in d:
        o1 = d["new-second.new"] / d["new-second.base"]   # new arm runs second
        o2 = d["new-first.base"] / d["new-first.new"]     # base arm runs second
        R.append(math.sqrt(o1 / o2)); B.append(math.sqrt(o1 * o2))
        print(f"  {rep}: o1={o1:.4f} o2={o2:.4f} -> R={math.sqrt(o1/o2):.4f} bias={math.sqrt(o1*o2):.4f}")
if R:
    est = math.exp(st.fmean(math.log(x) for x in R))
    print(f"\n  cumulative decode_tps HEAD/baseline = {est:.4f}  ({100*(est-1):+.2f}%)")
    if B:
        print(f"  implied position bias b = {math.exp(st.fmean(math.log(x) for x in B)):.4f}")
    if len(R) > 1:
        lg = [math.log(x) for x in R]
        sd = st.stdev(lg); sem = sd / math.sqrt(len(lg))
        print(f"  log-ratio sd {sd:.4f}  1sigma {100*sem:.2f}%  -> "
              f"{'RESOLVED' if abs(math.log(est)) > 2*sem else 'not resolved (within 2 sigma)'}")
PY
