#!/bin/bash
# Does the resident TTS need its host-side fp32 container payloads (111 MB of ANONYMOUS memory)
# while serving? Unmap them PROT_NONE before request N (PT_V21_HOSTPROTECT_AFTER, built in
# main_v21.c's serve loop by the experiment that answered this) and serve the 5 golden cases twice:
# same process, same cases, before vs after the unmap. Result 2026-11-13 (run #316): the service
# SEGFAULTS on the first post-unmap request -> the payloads are live during serving, not upload
# scratch, so they cannot be released. Pair with tts_hostpaytrack.sh for which records are read.
#!/bin/bash
# Serve the 5 golden reply cases twice through one resident TTS; unmap the host fp32 payload arena
# before request 5. Same process, same cases, before vs after: any served read of a host weight
# faults (service dies) or changes the waveform bytes.
cd /home/picard/primetts || exit 1
pkill -x pt_v21_nano; sleep 1
CASES="T81 T20 T120 T40 sid1_T29"
IN=/tmp/hr.fifo; LOG=/tmp/hr.log; rm -f $IN $LOG; mkfifo $IN
PT_V21_HOSTPROTECT_AFTER=5 ./build/pt_v21_nano --container build/v21.ptk --serve-fifo $IN > $LOG 2>&1 &
i=0; while [ $i -lt 240 ]; do grep -q "PT_AWAIT 0" $LOG && break; sleep 0.5; i=$((i+1)); done
n=0
for round in 1 2; do
  for c in $CASES; do
    d="tests/golden_v21/$c"; o=/tmp/hr.$c.$round.npy
    ( sleep 0.3; echo "$d $o" ) > $IN &
    i=0; while [ $i -lt 300 ]; do grep -q "PT_DONE $n " $LOG && break; sleep 0.1; i=$((i+1)); done
    ln=$(grep "PT_DONE $n " $LOG | tail -1)
    sz=$(stat -c %s "$o" 2>/dev/null || echo 0)
    echo "req$n $c ${ln##*rc=} size=$sz md5=$(md5sum $o 2>/dev/null | cut -c1-8) rss=$(awk '/VmRSS/{print $2}' /proc/$!/status 2>/dev/null)"
    n=$((n+1))
  done
done
grep -E "host payloads" $LOG
grep -c "PT_DONE" $LOG | sed 's/^/PT_DONE_count /'
pkill -x pt_v21_nano; true
