#!/bin/bash
# Engine A/B: decode with vs without the in-process GPU keep-alive thread
# (PQ1_KEEPWARM=1). Interleaved in one ssh session per run => drift-free.
# Optional: PQ1_KW_SPIN_MS / PQ1_KW_GAP are passed through to the engine.
set -uo pipefail
NANO=${NANO:-pesi@100.70.245.36}
R=/home/pesi/exp
MODEL=/home/pesi/b8q1.gguf
IDS="151644 872 198 852 279 3040 15584 323 825 3166 429 4344 304 1817 624 151645 198 151644 77091 198"
N=${1:-3}
G=/sys/devices/gpu.0/devfreq/57000000.gpu/cur_freq
for i in $(seq 1 "$N"); do
  for m in ctrl kw; do
    E=""
    [ "$m" = kw ] && E="PQ1_KEEPWARM=1 PQ1_KW_SPIN_MS=${PQ1_KW_SPIN_MS:-1} PQ1_KW_GAP=${PQ1_KW_GAP:-200}"
    ssh -o ConnectTimeout=25 "$NANO" "cd $R && (for j in \$(seq 1 400); do cat $G; sleep 0.05; done > clk_${m}_$i.log & CP=\$!; OMP_NUM_THREADS=4 $E timeout 900 ./fwd8_q1 $MODEL '$IDS' --gen 24 > ${m}_$i.log 2>${m}_$i.err; kill \$CP 2>/dev/null; grep -m1 'BQ1TIME step=43' ${m}_$i.log >/dev/null; echo done)" >/dev/null 2>&1
    echo "RUN $m $i done"
  done
done
for m in ctrl kw; do for i in $(seq 1 "$N"); do
  ssh -o ConnectTimeout=25 "$NANO" "cat $R/${m}_$i.log; echo '@@@'; sort -n $R/clk_${m}_$i.log 2>/dev/null | awk '{a[NR]=\$1}END{if(NR)print \"med=\"a[int(NR/2)]\" max=\"a[NR]\" n=\"NR}'" > /tmp/kwin_${m}_$i.log 2>/dev/null
done; done
python3 - "$N" <<'PY'
import re, statistics as st, sys
N=int(sys.argv[1])
def parse(p):
    txt=open(p,errors="replace").read(); body,sep,rest=txt.partition("@@@")
    d=[float(m.group(2)) for m in (re.match(r"BQ1TIME step=(\d+) ms=([0-9.]+)",l) for l in body.splitlines()) if m and int(m.group(1))>=20]
    return (sum(d)/len(d) if d else float("nan")), rest.strip()
res={}
for m in ("ctrl","kw"):
    ms=[]
    for i in range(1,N+1):
        v,clk=parse(f"/tmp/kwin_{m}_{i}.log"); ms.append(v)
        print(f"  {m:4s} run{i}: {v:.2f} ms/tok ({1000/v:.4f} tps)  clk {clk}")
    res[m]=st.mean(ms)
    print(f"  {m:4s} MEAN {st.mean(ms):.2f} ms/tok = {1000/st.mean(ms):.4f} tps  sd {st.stdev(ms):.2f}ms")
print(f"\n  keepwarm vs control: {100*(res['ctrl']/res['kw']-1):+.2f}% tps")
PY
