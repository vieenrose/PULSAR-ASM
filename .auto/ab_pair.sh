#!/bin/bash
# .auto/ab_pair.sh ENVNAME N - order-balanced A/B of the SAME engine binary with
# and without ENVNAME=1 (each of the N reps runs BOTH arm orders).
#
# WHY ORDER MATTERS (run #49, 2026-10-14): with a fixed order (new arm second in
# every pair) the second arm reads ~1.6% slower on identical binaries - 3/3 pairs,
# paired t~5.5. That is larger than the effects being measured (#45 +2.8%, #47
# +0.4%), so a fixed-order A/B cannot resolve effects below ~2 points.
#
# With one pair in each order per rep:
#   o1 = new_second/base_first = R*b     (b = position bias, b < 1)
#   o2 = base_second/new_first = b/R
#   => R = sqrt(o1/o2),  b = sqrt(o1*o2)
# so the geometric mean of sqrt(o1/o2) over reps is an unbiased estimate of the
# true decode_tps ratio, and b is reported as a sanity check (expect ~0.984).
set -uo pipefail
ENVNAME=${1:?usage: ab_pair.sh ENVNAME N [gen]}
N=${2:-2}
GEN=${3:-24}
NANO=${NANO:-pesi@100.70.245.36}
REPO=/home/louis/PULSAR-ASM
IDS="151644 872 198 852 279 3040 15584 323 825 3166 429 4344 304 1817 624 151645 198 151644 77091 198"

run_arm() { # $1 = arm (base|env), $2 = label
  if [ "$1" = env ]; then
    # ENVNAME may be a bare var name (-> =1) or a full assignment, e.g. PQ1_R128=0
    case "$ENVNAME" in *"="*) export PQ1_EXTRA="$ENVNAME";; *) export PQ1_EXTRA="$ENVNAME=1";; esac
  else
    unset PQ1_EXTRA
  fi
  local out t
  out=$(cd "$REPO" && bash .auto/measure.sh 2>&1)
  t=$(echo "$out" | grep -o 'decode_tps=[0-9.]*' | cut -d= -f2)
  echo "RESULT $2 $t"
}

LOG=$REPO/.auto/last_ab_pair.txt
: > "$LOG"
for i in $(seq 1 "$N"); do
  if [ $((i % 2)) -eq 1 ]; then
    run_arm base  "rep$i.order=base-first" | tee -a "$LOG"
    run_arm env   "rep$i.order=new-second" | tee -a "$LOG"
  else
    run_arm env   "rep$i.order=new-first" | tee -a "$LOG"
    run_arm base  "rep$i.order=base-second" | tee -a "$LOG"
  fi
done

echo "--- position-balanced summary ($ENVNAME) ---"
python3 - "$LOG" <<'PY'
import math, re, statistics as st, sys
rows = []
for ln in open(sys.argv[1], errors="replace"):
    m = re.match(r"RESULT (rep\d+)\.(\S+) ([0-9.]+)", ln)
    if m:
        rows.append((m.group(1), m.group(2).split("=", 1)[1], float(m.group(3))))
by = {}
for rep, order, tps in rows:
    by.setdefault(rep, {})[order] = tps
ratios, biases = [], []
for rep, d in sorted(by.items()):
    if "base-first" in d and "new-second" in d:
        o1 = d["new-second"] / d["base-first"]
        ratios.append(o1); biases.append(("o1", o1))
    if "new-first" in d and "base-second" in d:
        o2 = d["base-second"] / d["new-first"]
        ratios.append(o2); biases.append(("o2", o2))
# pair them: R = sqrt(o1/o2) for each (o1, o2) pair found
o1 = [v for k, v in biases if k == "o1"]
o2 = [v for k, v in biases if k == "o2"]
R, B = [], []
for i in range(min(len(o1), len(o2))):
    R.append(math.sqrt(o1[i] / o2[i]))
    B.append(math.sqrt(o1[i] * o2[i]))
print(f"pairs: {len(R)}   raw ratios new/second-over-base-first: {[round(x,4) for x in o1]}")
print(f"                          base/second-over-new-first: {[round(x,4) for x in o2]}")
if R:
    est = math.exp(st.fmean(math.log(x) for x in R))
    spread = st.stdev(R) if len(R) > 1 else 0.0
    pos = math.exp(st.fmean(math.log(x) for x in B)) if B else float('nan')
    print(f"decode_tps new/base = {est:.4f}  ({100*(est-1):+.2f}%)   "
          f"sd={spread:.4f}  position bias b = {pos:.4f}")
    if len(R) > 1:
        sem = st.stdev([math.log(x) for x in R]) / math.sqrt(len(R))
        print(f"log-ratio sd {st.stdev([math.log(x) for x in R]):.4f}  "
              f"~1sigma on the estimate {100*sem:.2f}%  -> "
              f"{'RESOLVED' if abs(math.log(est)) > 2*sem else 'not resolved (within 2 sigma)'}")
PY
