#!/usr/bin/env bash
# Order-balanced A/B for the Nano LFM3 engine: run arm1, arm2, arm2, arm1
# (one pair in each order) and report medians + R = sqrt(o1/o2), the same
# estimator .auto/ab_pair.sh uses for the Bonsai Nano objective. The board
# drifts and its 4 A57 cores are shared with three unkillable spinners from
# the previous session, so single-shot comparisons under ~2% are not
# decidable -- measure.sh's own 3-run spread on the baseline was 0.15%.
#
#   bash ab.sh "LFM_R=256"            # vs the unmodified arm
#   bash ab.sh "LFM_R=256" "LFM_R=64" # two arms
#   REPS=3 bash ab.sh ...             # 3 pairs
set -uo pipefail
cd "$(dirname "$0")"
NANO="${NANO:-picard@100.84.151.116}"
DIR="${NANO_DIR:-/home/picard/lfm3}"
BLOB="${LFM3_BLOB:-lfm3-audio-2b-q8.lfm3q8}"
BIN="${LFM3_BIN:-lfm3_nanode}"
REPS="${REPS:-2}"
A1="${1:-}"
A2="${2:-}"
PROMPT="$(python3 -c "import json;print(','.join(map(str,json.load(open('golden_greedy_A.json'))['prompt_ids'])))")"
N="${STEPS:-48}"
ARGS="${LFM3_ARGS:-}"

# One build, both arms: rebuilding between arms would measure the compiler.
scp -o ConnectTimeout=20 -q ../../pulsar_arm/asm/lfm3_nanode.cu "$NANO:$DIR/" 2>/dev/null || { echo "scp failed"; exit 2; }
BOUT=$(ssh -o ConnectTimeout=25 -o BatchMode=yes "$NANO" \
  "cd $DIR && rm -f $BIN.new && export PATH=/usr/local/cuda/bin:\$PATH && nvcc -gencode arch=compute_53,code=sm_53 -O3 -fmad=false -Xptxas -O3,-dlcm=ca -o $BIN.new lfm3_nanode.cu -lz 2>&1; test -x $BIN.new && mv -f $BIN.new $BIN && echo BUILT_OK" 2>/dev/null | grep -vi setlocale)
echo "$BOUT" | grep -E "error:" | head -12
echo "$BOUT" | grep -q BUILT_OK || { echo "BUILD FAILED (refusing to measure a stale binary)"; exit 1; }

run() {   # $1 = env string
  local out
  out=$(ssh -o ConnectTimeout=25 -o BatchMode=yes "$NANO" \
    "cd $DIR && sync; echo 31415 | sudo -S sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; timeout 1800 env $1 ./$BIN $BLOB --steps $N --prompt $PROMPT 2>&1" 2>/dev/null | grep -vi setlocale)
  echo "$out" | grep -oE 'steady-state [0-9.]+ ms/token' | tail -1 | awk '{print $2}'
}

med() {   # median of the args
  printf '%s\n' "$@" | sort -g | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'
}

o1=(); o2=()
for i in $(seq 1 "$REPS"); do
  for order in 12 21; do
    if [ "$order" = 12 ]; then e1=$(run "$A1"); e2=$(run "$A2"); else e2=$(run "$A2"); e1=$(run "$A1"); fi
    [ -z "$e1" ] && { echo "arm1 failed (rep $i order $order)"; exit 1; }
    [ -z "$e2" ] && { echo "arm2 failed (rep $i order $order)"; exit 1; }
    echo "  rep $i order $order: arm1[$A1]=$e1  arm2[$A2]=$e2"
    o1+=("$e1"); o2+=("$e2")
  done
done
m1=$(med "${o1[@]}"); m2=$(med "${o2[@]}")
echo "arm1 [$A1] median $m1 ms/token"
echo "arm2 [$A2] median $m2 ms/token"
R=$(awk -v a="$m1" -v b="$m2" 'BEGIN{printf "%.4f", sqrt(a/b)}')
echo "R = sqrt(o1/o2) = $R   (>1 means arm2 is FASTER: $A2)"
