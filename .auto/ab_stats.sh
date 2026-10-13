#!/bin/bash
# Same-job A/B: the ONLY difference between fwd8_q1_old (glue from git HEAD,
# prints "PQ2GPU") and fwd8_q1 (working tree, prints "PQ1GPU" + one deleted
# idempotent line) is in a cold atexit path. If the decode rate of the two
# binaries agrees, the -2.2% seen today vs the earlier 2.341 sample is drift,
# not a regression from the label fix.
set -uo pipefail
NANO=${NANO:-pesi@100.70.245.36}
REPO=/home/louis/PULSAR-ASM
SRC=$REPO/pulsar_arm/bonsai2
R=/home/pesi/exp
MODEL=/home/pesi/b8q1.gguf
IDS="151644 872 198 852 279 3040 15584 323 825 3166 429 4344 304 1817 624 151645 198 151644 77091 198"

git -C "$REPO" show HEAD:pulsar_arm/bonsai2/pq1_fwd8_glue.c > /tmp/pq1_glue_old.c
scp -q "$SRC/pq1_fwd8_glue.c" /tmp/pq1_glue_old.c "$NANO:$R/" || { echo SCP_FAILED; exit 1; }

ssh -o ConnectTimeout 25 "$NANO" "cd $R && export PATH=/usr/local/cuda-10.2/bin:\$PATH && \
  CUDAINC=/usr/local/cuda-10.2/targets/aarch64-linux/include && \
  gcc -O2 -fopenmp -mcpu=cortex-a57 -DSTANDALONE -DPQ1_USE_CUDA -I\$CUDAINC -c pq1_fwd8_glue.c -o pq1_glue_new.o && \
  gcc -O2 -fopenmp -mcpu=cortex-a57 -DSTANDALONE -DPQ1_USE_CUDA -I\$CUDAINC -c pq1_glue_old.c -o pq1_glue_old.o && \
  g++ fwd8_q1.o pq1_host.o pq1_glue_new.o pq1_swar_sm53.o -o q1_new -L/usr/local/cuda-10.2/targets/aarch64-linux/lib -lcudart -lm -fopenmp && \
  g++ fwd8_q1.o pq1_host.o pq1_glue_old.o pq1_swar_sm53.o -o q1_old -L/usr/local/cuda-10.2/targets/aarch64-linux/lib -lcudart -lm -fopenmp && \
  echo BUILT" 2>&1 | tail -n 3

# one ssh session, interleaved old/new so both see the same thermal state
ssh -o ConnectTimeout 25 "$NANO" "cd $R && for i in 1 2; do
  for b in old new; do
    OMP_NUM_THREADS=4 PQ1_STATS=1 ./$R/q1_$b $MODEL '$IDS' --gen 24 > ab_${b}_$i.log 2>&1
    echo \"RUN \$b \$i rc=\$?\"
  done
done" 2>&1

for b in old new; do for i in 1 2; do
  ssh -o ConnectTimeout 25 "$NANO" "cat $R/ab_${b}_$i.log" > "/tmp/ab_${b}_$i.log" 2>/dev/null
done; done

python3 - <<'PY'
import re
def dec(p):
    d=[float(m.group(2)) for m in (re.match(r"BQ1TIME step=(\d+) ms=([0-9.]+)", l)
        for l in open(p, errors="replace")) if m and int(m.group(1)) >= 20]
    return sum(d)/len(d) if d else float("nan"), len(d)
vals={}
for b in ("old","new"):
    v=[]
    for i in (1,2):
        m,n=dec(f"/tmp/ab_{b}_{i}.log"); v.append(m); print(f"  {b} run{i}: {m:.2f} ms/tok over {n} steps")
    vals[b]=sum(v)/len(v)
    print(f"  {b} mean: {vals[b]:.2f} ms/tok  ({1000.0/vals[b]:.4f} tps)")
print(f"  ratio new/old = {vals['new']/vals['old']:.4f}")
PY
