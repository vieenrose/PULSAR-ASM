#!/bin/bash
# Same-job A/B for a KERNEL-level change (working tree vs git HEAD), measuring
# decode ms/tok and peak RSS with the binaries interleaved in one ssh session.
# Usage: bash .auto/ab_kernel.sh [n_runs_per_side]
# Builds qk_old (kernel .cu from HEAD) and qk_new (working tree) against the
# same host objects, so the kernel translation unit is the only difference.
set -uo pipefail
NANO=${NANO:-pesi@100.70.245.36}
REPO=/home/louis/PULSAR-ASM
SRC=$REPO/pulsar_arm/bonsai2
R=/home/pesi/exp
MODEL=/home/pesi/b8q1.gguf
IDS="151644 872 198 852 279 3040 15584 323 825 3166 429 4344 304 1817 624 151645 198 151644 77091 198"
N=${1:-2}
CU=${CU:-pq1_swar_sm53.cu}
GITREF=${GITREF:-HEAD}

git -C "$REPO" show "$GITREF:pulsar_arm/bonsai2/$CU" > /tmp/kv_old.cu || exit 1
# Older TUs may not define entry points the current glue calls; delegate them.
cat "$REPO/.auto/ab_delegates.inc" >> /tmp/kv_old.cu
scp -q "$SRC/$CU" /tmp/kv_old.cu "$NANO:$R/" || { echo SCP_FAILED; exit 1; }

ssh -o ConnectTimeout=25 "$NANO" "cd $R && export PATH=/usr/local/cuda-10.2/bin:\$PATH
echo BUILD_OLD
nvcc -gencode arch=compute_53,code=sm_53 -O3 -maxrregcount=48 --use_fast_math -Xptxas -O3,-dlcm=ca -std=c++11 -c kv_old.cu -o kv_old.o 2>&1 | grep -E ' error' | head -3
echo BUILD_NEW
nvcc -gencode arch=compute_53,code=sm_53 -O3 -maxrregcount=48 --use_fast_math -Xptxas -O3,-dlcm=ca -std=c++11 -c $CU -o kv_new.o 2>&1 | grep -E ' error' | head -3
g++ fwd8_q1.o pq1_host.o pq1_glue.o kv_old.o -o qk_old -L/usr/local/cuda-10.2/targets/aarch64-linux/lib -lcudart -lm -fopenmp 2>/dev/null; echo OLD_LK=\$?
g++ fwd8_q1.o pq1_host.o pq1_glue.o kv_new.o -o qk_new -L/usr/local/cuda-10.2/targets/aarch64-linux/lib -lcudart -lm -fopenmp 2>/dev/null; echo NEW_LK=\$?
ls -l qk_old qk_new" 2>&1 | grep -v locale

cat > /tmp/kv_run.sh <<EOS
#!/bin/bash
cd $R || exit 1
for i in \$(seq 1 $N); do
  for b in old new; do
    OMP_NUM_THREADS=4 PQ1_STATS=1 /usr/bin/time -v ./qk_\$b "$MODEL" "$IDS" --gen 24 > kv_\${b}_\$i.log 2> kv_\${b}_\$i.time
    echo "RUN \$b \$i rc=\$?"
  done
done
EOS
scp -q /tmp/kv_run.sh "$NANO:$R/" 2>/dev/null
ssh -o ConnectTimeout=25 "$NANO" "bash $R/kv_run.sh" 2>&1 | grep -v locale

for b in old new; do for i in $(seq 1 "$N"); do
  ssh -o ConnectTimeout=25 "$NANO" "cat $R/kv_${b}_$i.log; echo '@@@'; grep -m1 'Maximum resident set size' $R/kv_${b}_$i.time; grep -m1 'PQ1GPU' $R/kv_${b}_$i.time" > "/tmp/kv_${b}_$i.log" 2>/dev/null
done; done

python3 - "$N" <<'PY'
import re, sys, statistics as st
N=int(sys.argv[1])
def parse(p):
    txt=open(p, errors="replace").read()
    body, sep, rest = txt.partition("@@@")
    d=[float(m.group(2)) for m in
       (re.match(r"BQ1TIME step=(\d+) ms=([0-9.]+)", l) for l in body.splitlines())
       if m and int(m.group(1)) >= 20]
    r=re.search(r"(\d+)", rest)
    ok = "bit" if "PARITY" in body.upper() and "FAIL" not in body.upper() else ""
    return (sum(d)/len(d) if d else float("nan"), int(r.group(1))//1024 if r else None)
res={}
for b in ("old","new"):
    ms=[];rss=[]
    for i in range(1,N+1):
        m,r=parse(f"/tmp/kv_{b}_{i}.log"); ms.append(m); rss.append(r)
        print(f"  {b} run{i}: {m:.2f} ms/tok ({1000.0/m:.4f} tps)  rss={r}MB")
    res[b]=(st.mean(ms), st.mean(rss))
    sd = st.stdev(ms) if len(ms)>1 else 0.0
    print(f"  {b} MEAN: {st.mean(ms):.2f} ms/tok = {1000/st.mean(ms):.4f} tps   rss={st.mean(rss):.1f}MB  (sd {sd:.2f}ms)")
print(f"\n  decode: new vs old {100*(res['old'][0]/res['new'][0]-1):+.2f}% tps")
print(f"  rss:    new vs old {res['new'][1]-res['old'][1]:+.1f} MB")
PY
