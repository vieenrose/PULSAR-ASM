#!/bin/bash
# bash ctxcap.sh: (1) client refuses a 330 s clip before the encoder; (2) a resident backbone
# refuses a 4200-id spec written straight to its fifo and still serves the next turn correctly.
cd /home/picard/lfm3 || exit 1
pkill -x lfm3_nanode; pkill -x lfm3_enc_gpu; pkill -x pt_v21_nano; sleep 2
python3 - <<'PY'
import wave
w=wave.open("/home/picard/lfm3/gate_60s_zh_16k_mono.wav"); d=w.readframes(w.getnframes()); p=w.getparams(); w.close()
o=wave.open("/tmp/cx.long.wav","wb"); o.setparams(p); o.writeframes(d*3); o.close()
PY
a=$(date +%s.%N); timeout 600 ./lfm3_asr . /tmp/cx.long.wav 96 --out-ids /tmp/cx.long.i32 > /tmp/cx.long.log 2>&1; rc=$?
echo "client 330 s clip: rc=$rc wall=$(echo "$(date +%s.%N) - $a" | bc | cut -c1-5)s encoder_ran=$(grep -c 'encoder reply\|ENCODER total' /tmp/cx.long.log)"; grep -h "exceed" /tmp/cx.long.log | cut -c1-200
S=/tmp/cx.srv.in; SL=/tmp/cx.srv.log; rm -f $S; mkfifo $S; : > $SL
./lfm3_nanode lfm3-audio-2b-q8.lfm3q8 --clip-fifo $S --prefill-run 32 > $SL 2>&1 & SP=$!
i=0; while [ $i -lt 240 ]; do grep -q "CLIP_AWAIT 0" $SL && break; sleep 0.5; i=$((i+1)); done
# spec lines: emb path, ids, audio-at, ngen, stop id. First a 4200-id prompt, then a malformed
# spec (the two first lines swapped: a non-numeric id line).
python3 -c "print('-'); print(','.join(['124894']*4200)); print(-1); print(4); print(-1)" > /tmp/cx.spec
python3 -c "print(','.join(['124894']*20)); print('-'); print(-1); print(4); print(-1)" > /tmp/cx.bad
for f in /tmp/cx.spec /tmp/cx.bad; do
  n=$(grep -c "DONE steps=" $SL); echo $f > $S
  i=0; while [ $i -lt 120 ]; do [ "$(grep -c "DONE steps=" $SL)" -gt "$n" ] && break; kill -0 $SP 2>/dev/null || break; sleep 0.5; i=$((i+1)); done
done
grep -E "REFUSED|DONE steps=0|bad_alloc" $SL | cut -c1-120
timeout 400 ./lfm3_asr . /tmp/clip_zh.wav 96 --out-ids /tmp/cx.zh.i32 --serve-in $S --serve-log $SL > /tmp/cx.zh.log 2>&1
echo "next turn rc=$? $(cmp -s /tmp/cx.zh.i32 /tmp/gen_golden_zh.i32 && echo GOLDEN_OK || echo GOLDEN_FAIL) service_alive=$(kill -0 $SP 2>/dev/null && echo yes || echo no)"
timeout 5 sh -c "echo Q > $S"; sleep 1; kill $SP 2>/dev/null; rm -f $S /tmp/cx.long.wav
