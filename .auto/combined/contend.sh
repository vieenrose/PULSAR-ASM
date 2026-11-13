#!/usr/bin/env bash
# .auto/combined/contend.sh -- does a CPU GEMV share contend with the GPU batched prefill? (run #294)
# One pair of resident ASR services; ROUNDS x {quiet zh turn, zh turn with cpu_leaf_bench loading
# THREADS cores for the whole turn}. Reports the backbone's own prefill timer (PF4-RUN) and the
# turn wall per arm, plus the CPU kernel's achieved GMAC/s while the turn ran. ids md5 every turn.
set -u
cd /home/picard/lfm3
R=${ROUNDS:-3}; TH=${THREADS:-3}
SRV_IN=/tmp/ct.srv.in; SRV_LOG=/tmp/ct.srv.log; ENC_IN=/tmp/ct.enc.in; ENC_LOG=/tmp/ct.enc.log
pkill -f "[l]fm3_nanode .*--clip-fifo"; pkill -f "[l]fm3_enc_gpu .*--serve-fifo"; pkill -f "[p]t_v21_nano .*--serve-fifo"; sleep 1
rm -f $SRV_IN $SRV_LOG $ENC_IN $ENC_LOG; mkfifo $SRV_IN $ENC_IN; : > $SRV_LOG; : > $ENC_LOG
./lfm3_nanode lfm3-audio-2b-q8.lfm3q8 --clip-fifo $SRV_IN --prefill-run 32 > $SRV_LOG 2>&1 & SRV=$!
i=0; while [ $i -lt 300 ]; do grep -q "CLIP_AWAIT 0" $SRV_LOG 2>/dev/null && break; sleep 0.5; i=$((i+1)); done
./lfm3_enc_gpu lfm3-audio-2b-enc.lfm3q8 - auto - --serve-fifo $ENC_IN --only proj_out > $ENC_LOG 2>&1 & ENC=$!
i=0; while [ $i -lt 240 ]; do grep -q "ENC_AWAIT 0" $ENC_LOG 2>/dev/null && break; sleep 0.5; i=$((i+1)); done
turn() { ./lfm3_asr . /tmp/clip_zh.wav 96 --out-ids /tmp/ct.i32 --serve-in $SRV_IN --serve-log $SRV_LOG \
           --enc-in $ENC_IN --enc-log $ENC_LOG > /tmp/ct.log 2>&1; }
pf() { grep "PF4-RUN: prompt" $SRV_LOG | tail -1 | grep -oE "[0-9]+ ms \(" | grep -oE "[0-9]+"; }
turn; turn   # warm-up: past the first-request paths and the prefix-cache build
for r in $(seq 1 $R); do
  turn; echo "quiet  wall=$(grep -o 'client total [0-9.]*' /tmp/ct.log | awk '{print $3}') prefill=$(pf) md5=$(md5sum /tmp/ct.i32 | cut -c1-8)"
  /tmp/cpu_leaf_bench 768 8192 32 1 6.5 $TH > /tmp/ct.load 2>&1 & L=$!
  sleep 0.3; turn
  echo "loaded wall=$(grep -o 'client total [0-9.]*' /tmp/ct.log | awk '{print $3}') prefill=$(pf) md5=$(md5sum /tmp/ct.i32 | cut -c1-8)"
  wait $L; echo "       cpu: $(cat /tmp/ct.load)"
done
/tmp/cpu_leaf_bench 768 8192 32 1 3 $TH | sed 's/^/alone  cpu: /'
for f in $SRV_IN $ENC_IN; do timeout 5 sh -c "echo Q > $f" 2>/dev/null; done
sleep 1; kill $SRV $ENC 2>/dev/null; rm -f $SRV_IN $ENC_IN
