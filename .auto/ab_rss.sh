#!/bin/bash
# Same-job A/B for the arena right-sizing (1280MB -> 1152MB).
# q1_old = glue from git HEAD (arena default 1280), q1_new = working tree (1152).
# Measures BOTH the primary (decode ms/tok) and peak RSS, interleaved in one
# ssh session so both binaries see the same thermal/memory state.
set -uo pipefail
NANO=${NANO:-pesi@100.70.245.36}
REPO=/home/louis/PULSAR-ASM
SRC=$REPO/pulsar_arm/bonsai2
R=/home/pesi/exp
MODEL=/home/pesi/b8q1.gguf
IDS="151644 872 198 852 279 3040 15584 323 825 3166 429 4344 304 1817 624 151645 198 151644 77091 198"

git -C "$REPO" show HEAD:pulsar_arm/bonsai2/pq1_fwd8_glue.c > /tmp/pq1_glue_old.c
scp -q "$SRC/pq1_fwd8_glue.c" /tmp/pq1_glue_old.c "$NANO:$R/" || { echo SCP_FAILED; exit 1; }

ssh -o ConnectTimeout=25 "$NANO" "cd $R && CUDAINC=/usr/local/cuda-10.2/targets/aarch64-linux/include
gcc -O2 -fopenmp -mcpu=cortex-a57 -DSTANDALONE -DPQ1_USE_CUDA -I\$CUDAINC -c pq1_fwd8_glue.c -o pq1_glue_new.o 2>/dev/null; echo NEW_CC=\$?
gcc -O2 -fopenmp -mcpu=cortex-a57 -DSTANDALONE -DPQ1_USE_CUDA -I\$CUDAINC -c pq1_glue_old.c -o pq1_glue_old.o 2>/dev/null; echo OLD_CC=\$?
g++ fwd8_q1.o pq1_host.o pq1_glue_new.o pq1_swar_sm53.o -o q1_new -L/usr/local/cuda-10.2/targets/aarch64-linux/lib -lcudart -lm -fopenmp 2>/dev/null; echo NEW_LK=\$?
g++ fwd8_q1.o pq1_host.o pq1_glue_old.o pq1_swar_sm53.o -o q1_old -L/usr/local/cuda-10.2/targets/aarch64-linux/lib -lcudart -lm -fopenmp 2>/dev/null; echo OLD_LK=\$?" 2>&1 | grep -Ev "locale|^$"

cat > /tmp/ab_rss_run.sh <<'EOS'
#!/bin/bash
R=/home/pesi/exp
MODEL=/home/pesi/b8q1.gguf
IDS="151644 872 198 852 279 3040 15584 323 825 3166 429 4344 304 1817 624 151645 198 151644 77091 198"
cd "$R" || exit 1
for i in 1 2 3; do
  for b in old new; do
    OMP_NUM_THREADS=4 PQ1_STATS=1 /usr/bin/time -v "./q1_$b" "$MODEL" "$IDS" --gen 24 > "rs_${b}_$i.log" 2> "rs_${b}_$i.time"
    echo "RUN $b $i rc=$?"
  done
done
EOS
scp -q /tmp/ab_rss_run.sh "$NANO:$R/" 2>/dev/null
ssh -o ConnectTimeout=25 "$NANO" 'bash /home/pesi/exp/ab_rss_run.sh' 2>&1 | grep -v locale

for b in old new; do for i in 1 2 3; do
  ssh -o ConnectTimeout=25 "$NANO" "cat $R/rs_${b}_$i.log; echo '@@@'; grep -m1 'Maximum resident set size' $R/rs_${b}_$i.time" > "/tmp/rs_${b}_$i.log" 2>/dev/null
done; done

python3 - <<'PY'
import re, statistics as st
def parse(p):
    txt=open(p, errors="replace").read()
    body, _, rss = txt.partition("@@@")
    d=[float(m.group(2)) for m in
       (re.match(r"BQ1TIME step=(\d+) ms=([0-9.]+)", l) for l in body.splitlines())
       if m and int(m.group(1)) >= 20]
    r=re.search(r"(\d+)", rss)
    return (sum(d)/len(d) if d else float("nan"), int(r.group(1))//1024 if r else None)
res={}
for b in ("old","new"):
    ms=[];rss=[]
    for i in (1,2,3):
        m,r=parse(f"/tmp/rs_{b}_{i}.log"); ms.append(m); rss.append(r)
        print(f"  {b} run{i}: {m:.2f} ms/tok ({1000.0/m:.4f} tps)  rss={r}MB")
    res[b]=(st.mean(ms), st.mean(rss))
    print(f"  {b} MEAN: {st.mean(ms):.2f} ms/tok = {1000/st.mean(ms):.4f} tps   rss={st.mean(rss):.1f}MB")
print(f"\n  decode: new vs old {100*(res['old'][0]/res['new'][0]-1):+.2f}% tps")
print(f"  rss:    new vs old {res['new'][1]-res['old'][1]:+.1f} MB")
PY
