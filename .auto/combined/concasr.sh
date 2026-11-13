#!/bin/bash
# bash concasr.sh [TRIALS] [CLIENTS]  (on the board: /home/picard/lfm3)
#
# Concurrent ASR clients against the resident services -- the N>1 users case, which the harness
# never exercises (it runs one turn at a time) and which the service clients explicitly assume away:
# the encoder client computes its request index by COUNTING the ENC_DONE lines already in the log
# ("turns are serialised, so that count is this turn's index", lfm3_asr.c:464) and the backbone
# client waits for the first "CLIP n DONE steps=" line after the log offset it recorded before
# writing its spec. Two clients that arrive together therefore both wait for the FIRST completion:
# the loser returns before its own pass has run and reads an output file the service has not written.
#
# Each trial launches CLIENTS lfm3_asr processes at once (distinct clips, distinct --out-ids) and
# requires every one of them to finish with its own frozen golden transcript.
cd /home/picard/lfm3 || exit 1
T=${1:-8}; C=${2:-2}
CLIPV=(/tmp/clip_zh.wav /tmp/clip_en.wav /home/picard/lfm3/cv_zh9.wav /home/picard/lfm3/cv_en9.wav)
# CLIPS="a.wav b.wav ..." overrides the list (keep zh, en first: indices 0/1 carry the frozen goldens).
# Used with the 110 s clip to put a long turn -- and its CPUROWS DROP line -- among concurrent clients.
[ -n "$CLIPS" ] && read -r -a CLIPV <<< "$CLIPS"
GOV=(/tmp/gen_golden_zh.i32 /tmp/gen_golden_en.i32 "" "")
SRV_IN=/tmp/cc.srv.in; SRV_LOG=/tmp/cc.srv.log; ENC_IN=/tmp/cc.enc.in; ENC_LOG=/tmp/cc.enc.log
pkill -f "lfm3_nanode .*--clip-fifo" 2>/dev/null; pkill -f "lfm3_enc_gpu .*--serve-fifo" 2>/dev/null
i=0; while [ $i -lt 60 ]; do a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); [ "$a" -ge 1500 ] && break; sleep 0.5; i=$((i+1)); done
rm -f $SRV_IN $SRV_LOG $ENC_IN $ENC_LOG; mkfifo $SRV_IN $ENC_IN; : > $SRV_LOG; : > $ENC_LOG
./lfm3_nanode lfm3-audio-2b-q8.lfm3q8 --clip-fifo $SRV_IN --prefill-run 32 > $SRV_LOG 2>&1 & SRV=$!
i=0; while [ $i -lt 300 ]; do grep -q "CLIP_AWAIT 0" $SRV_LOG 2>/dev/null && break; sleep 0.5; i=$((i+1)); done
./lfm3_enc_gpu lfm3-audio-2b-enc.lfm3q8 - auto - --serve-fifo $ENC_IN --only proj_out > $ENC_LOG 2>&1 & ENC=$!
i=0; while [ $i -lt 240 ]; do grep -q "ENC_AWAIT 0" $ENC_LOG 2>/dev/null && break; sleep 0.5; i=$((i+1)); done
[ $i -ge 240 ] && { echo "BOOT_FAIL"; exit 1; }
# one untimed turn so both services are past their first-request paths
./lfm3_asr . /tmp/clip_zh.wav 96 --out-ids /tmp/cc.w.i32 --serve-in $SRV_IN --serve-log $SRV_LOG \
  --enc-in $ENC_IN --enc-log $ENC_LOG > /tmp/cc.w.log 2>&1
fails=0
for t in $(seq 1 $T); do
  pids=""
  for k in $(seq 0 $((C-1))); do
    rm -f /tmp/cc.$t.$k.i32
    # wall = the client's whole process time: a wall of ~2x a solo turn means this client queued
    # behind another (the flock serialises resident turns, the right model on one SM). That is the
    # multi-user latency number for the feature this tool gates.
    ( S=$(date +%s.%N); timeout 400 ./lfm3_asr . "${CLIPV[$k]}" 96 --out-ids /tmp/cc.$t.$k.i32 \
        --serve-in $SRV_IN --serve-log $SRV_LOG --enc-in $ENC_IN --enc-log $ENC_LOG \
        > /tmp/cc.$t.$k.log 2>&1; echo $? > /tmp/cc.$t.$k.rc; \
        date +%s.%N | awk -v s="$S" '{printf "%.2f\n", $1-s}' > /tmp/cc.$t.$k.t ) &
    pids="$pids $!"
  done
  # wait ONLY the clients: a bare `wait` also waits for the two long-lived services started at the
  # top, which is what made the first version of this script appear to hang for 25 minutes after
  # the clients had already finished (one of them had in fact died -- see the trial line).
  for p in $pids; do wait $p 2>/dev/null; done
  line="TRIAL $t"
  for k in $(seq 0 $((C-1))); do
    rc=$(cat /tmp/cc.$t.$k.rc 2>/dev/null || echo X)
    st=NOFILE
    if [ -s /tmp/cc.$t.$k.i32 ]; then
      if [ -n "${GOV[$k]}" ]; then cmp -s /tmp/cc.$t.$k.i32 "${GOV[$k]}" && st=GOLDEN || st=MISMATCH
      else st=IDS_$(wc -c < /tmp/cc.$t.$k.i32); fi
    fi
    w=$(cat /tmp/cc.$t.$k.t 2>/dev/null)
    line="$line c$k:rc=$rc,$st${w:+,wall=${w}s}"
    { [ "$rc" != 0 ] || [ "$st" = MISMATCH ] || [ "$st" = NOFILE ]; } && fails=$((fails+1))
  done
  echo "$line"
done
echo "CONCURRENCY $([ $fails = 0 ] && echo OK || echo "FAILS=$fails") ($T trials x $C clients)"
echo "CLIPS_DONE=$(grep -c 'DONE steps=' $SRV_LOG) ENC_DONE=$(grep -c ENC_DONE $ENC_LOG)"
for f in $SRV_IN $ENC_IN; do timeout 5 sh -c "echo Q > $f" 2>/dev/null; done
sleep 1; kill $SRV $ENC 2>/dev/null; rm -f $SRV_IN $ENC_IN
