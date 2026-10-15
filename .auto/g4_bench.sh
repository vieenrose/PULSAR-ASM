#!/bin/bash
# Gemma 4 E2B Jetson Nano bench (autoresearch meter).
# Workload: the fixed 225-token paragraph prompt of .auto/g4_bench_prompts.json
# (held out from the accuracy gates), greedy, --gen 128: one run gives the prefill
# rate (225 prompt tokens) and the decode rate (generated steps at depth 225+).
# Each run: fresh source copy on the Nano, PTX regenerated from g4_kernels.cu and
# embedded, build.sh; one warm-up run, then 3 timed runs; medians of the engine's
# own prefill/decode timers; peak RSS = max over the 3 runs (peakrss wrapper).
set -u
PEER=picard@picard-desktop
REPO=/home/user/PULSAR-ASM
SRC=$REPO/pulsar_arm/gemma4nano
B=/home/picard/g4/bench
G=/home/picard/g4/gemma-4-E2B_q4_0-it.gguf
R=/home/picard/g4/g4_rope_2048.bin
IDS=$(python3 -I -c "import json;print(' '.join(map(str,json.load(open('$REPO/.auto/g4_bench_prompts.json'))['prefill_prompt']['ids'])))")
TMP=$(mktemp -d); trap 'rm -rf $TMP' EXIT

timeout 30 ssh -o BatchMode=yes $PEER "test -f $G && test -f $R" || { echo "ABORT: model or rope table missing on the Nano"; exit 3; }
# 1. fresh sources
tar -C "$SRC" -cf $TMP/src.tar --exclude='*.o' --exclude=probes --exclude=__pycache__ . || exit 3
timeout 120 ssh -o BatchMode=yes $PEER "rm -rf $B/src && mkdir -p $B/src && tar -C $B/src -xf -" < $TMP/src.tar || { echo "BUILD FAIL: copy"; exit 3; }
timeout 60 ssh -o BatchMode=yes $PEER 'cat > /tmp/peakrss.c && gcc -O2 -o /tmp/peakrss /tmp/peakrss.c' < $REPO/.auto/peakrss.c || { echo "BUILD FAIL: peakrss"; exit 3; }
# 2. PTX from the kernel source, embedded; build (build.sh removes binaries first)
timeout 600 ssh -o BatchMode=yes $PEER "cd $B/src && /usr/local/cuda/bin/nvcc -arch=compute_53 -ptx -fmad=false -O3 g4_kernels.cu -o g4_kernels.ptx && python3 g4_embed_ptx.py g4_kernels.ptx g4_ptx_k.h && sh build.sh" > $TMP/build.log 2>&1 \
  || { echo "BUILD FAIL"; tail -20 $TMP/build.log; exit 3; }
if grep -q "warning" $TMP/build.log; then echo "BUILD FAIL: new compiler warnings"; grep warning $TMP/build.log | head; exit 3; fi
# keep the repo's embedded PTX in sync with the kernel source it came from
scp -q $PEER:$B/src/g4_kernels.ptx $PEER:$B/src/g4_ptx_k.h $TMP/
cmp -s $TMP/g4_kernels.ptx $SRC/g4_kernels.ptx || { cp $TMP/g4_kernels.ptx $TMP/g4_ptx_k.h $SRC/; echo "note: g4_kernels.ptx / g4_ptx_k.h regenerated from g4_kernels.cu"; }
# 3. warm-up + 3 timed runs
RUN="cd $B/src && PEAKRSS_OUT=$B/rss.txt /tmp/peakrss ./g4eng $G $R $IDS --gen 128"
timeout 600 ssh -o BatchMode=yes $PEER "rm -f $B/rss.txt; $RUN > /dev/null 2>&1" || { echo "RUN FAIL: warm-up"; exit 3; }
timeout 30 ssh -o BatchMode=yes $PEER "rm -f $B/rss.txt"
for i in 1 2 3; do
  timeout 600 ssh -o BatchMode=yes $PEER "$RUN > $B/out$i.txt 2> $B/err$i.txt; echo rc=\$?" > $TMP/rc$i.txt
  grep -q "rc=0" $TMP/rc$i.txt || { echo "RUN FAIL: run $i"; ssh $PEER "tail -5 $B/err$i.txt"; exit 3; }
  ssh -o BatchMode=yes $PEER "cat $B/err$i.txt" > $TMP/err$i.txt
  ssh -o BatchMode=yes $PEER "cat $B/out$i.txt" > $TMP/out$i.txt
done
ssh -o BatchMode=yes $PEER "cat $B/rss.txt" > $TMP/rss.txt
# the 3 timed runs must agree on the stream (determinism)
cmp -s $TMP/out1.txt $TMP/out2.txt && cmp -s $TMP/out1.txt $TMP/out3.txt || { echo "RUN FAIL: timed runs differ"; exit 4; }
python3 -I - "$TMP" <<'PY'
import re, sys, statistics as st
d = sys.argv[1]
pf, dc, gpu, load, nsteps, e2e = [], [], [], [], [], []
for i in (1, 2, 3):
    e = open(f"{d}/err{i}.txt").read()
    m = re.search(r"prefill (\d+) tokens (\d+) ms \(([\d.]+) tok/s\); decode (\d+) steps (\d+) ms \(([\d.]+) tok/s\)", e)
    pf.append(float(m.group(3))); dc.append(float(m.group(6))); nsteps.append(int(m.group(4)))
    e2e.append(float(m.group(2)) + float(m.group(5)))
    ntok = int(m.group(1)) + int(m.group(4))
    g = re.search(r"([\d.]+) MB on device.*load (\d+) ms", e)
    gpu.append(float(g.group(1))); load.append(float(g.group(2)))
rss = max(int(l.split()[1]) for l in open(f"{d}/rss.txt") if l.startswith("PEAKRSS_KB"))
print(f"METRIC e2e_tok_s={ntok / (st.median(e2e) / 1000):.3f}")   # (prompt + generated tokens) / (prefill + decode time)
print(f"METRIC e2e_ms={st.median(e2e):.0f}")
print(f"METRIC decode_tok_s={st.median(dc):.3f}")
print(f"METRIC prefill_tok_s={st.median(pf):.3f}")
print(f"METRIC peak_rss_mb={rss/1024:.1f}")
print(f"METRIC gpu_mb={st.median(gpu):.1f}")
print(f"METRIC load_ms={st.median(load):.0f}")
print(f"decode runs {dc} over {nsteps[0]} steps; prefill runs {pf}")
PY
