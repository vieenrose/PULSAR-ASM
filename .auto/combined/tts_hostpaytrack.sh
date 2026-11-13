#!/bin/bash
# Which container payloads does the HOST op path touch per served request? Serve the 5 golden cases
# through one resident TTS with PT_V21_HOSTTRACK=1 (counters at the seven host-side use sites in
# pt_v21_host.c: gather, pad, layernorm, conv, convT x2, matmul-B) and print the per-record hits.
# Result 2026-11-13 (run #316): 302 of 336 records, once per request each -- so no subset of the
# 111 MB payload arena is dead after the weight upload. Diagnostic env, not in the shipped runtime.
#!/bin/bash
cd /home/picard/primetts || exit 1
pkill -x pt_v21_nano; sleep 1
IN=/tmp/hp.fifo; LOG=/tmp/hp.log; rm -f $IN $LOG; mkfifo $IN
PT_V21_HOSTTRACK=1 ./build/pt_v21_nano --container build/v21.ptk --serve-fifo $IN > $LOG 2>&1 &
i=0; while [ $i -lt 240 ]; do grep -q "PT_AWAIT 0" $LOG && break; sleep 0.5; i=$((i+1)); done
n=0
for c in T81 T20 T120 T40 sid1_T29; do
  ( sleep 0.3; echo "tests/golden_v21/$c /tmp/hp.$c.npy" ) > $IN &
  i=0; while [ $i -lt 300 ]; do grep -q "PT_DONE $n " $LOG && break; sleep 0.1; i=$((i+1)); done
  echo "== req$n $c $(grep "PT_DONE $n " $LOG | tail -1)"
  n=$((n+1))
done
echo "--- per-request host-payload reads:"
grep -n "hostpay" $LOG | tail -60
pkill -x pt_v21_nano; true
