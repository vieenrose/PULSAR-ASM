#!/bin/bash
# bash aliasx.sh: proj_out md5 per (clip, arm) through the one-shot encoder child of lfm3_asr.
cd /home/picard/lfm3 || exit 1
pkill -x lfm3_enc_gpu; pkill -x lfm3_nanode; sleep 1
one() {  # $1 wav, rest env
  local w=$1; shift
  rm -rf /tmp/lfm3_asr.*.d /tmp/ax.log
  ( env "$@" timeout 300 ./lfm3_asr . $w 1 --out-ids /tmp/ax.i32 > /tmp/ax.log 2>&1 ) & local cp=$!
  local i=0; while [ $i -lt 1200 ]; do grep -q "STAGE 6" /tmp/ax.log && break; kill -0 $cp 2>/dev/null || break; sleep 0.1; i=$((i+1)); done
  local d=$(ls -d /tmp/lfm3_asr.*.d 2>/dev/null | head -1)
  echo "$(basename $w) [$*] md5=$(md5sum < $d/proj_out.bin 2>/dev/null | cut -c1-12) bytes=$(stat -c%s $d/proj_out.bin 2>/dev/null) $(grep -oE 'encoder reply  \(\+[0-9.]+ s\)' /tmp/ax.log) $(grep -ciE 'non-finite|PLAUS|rc=3' /tmp/ax.log)bad"
  pkill -P $cp; kill $cp 2>/dev/null; wait $cp 2>/dev/null; pkill -x lfm3_nanode; sleep 1
}
for w in /tmp/clip_zh.wav /tmp/clip_en.wav /home/picard/lfm3/cv_en5.wav /home/picard/lfm3/gate_60s_zh_16k_mono.wav; do
  one $w LFM_ENC_ALIAS=0; one $w LFM_ENC_ALIAS=1; one $w LFM_ENC_ALIAS=1 LFM_ENC_POISON=1
done
