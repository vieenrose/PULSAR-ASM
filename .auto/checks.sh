#!/usr/bin/env bash
# .auto/checks.sh -- correctness backpressure for the S2S loop.
#
# STRUCTURE (2026-11-11): the autoresearch tool runs this file with a 300 s timeout,
# but the full gates need ~10-15 min (text gates 7m alone). So this file is a FAST
# verifier: host gates (seconds) + a stamp check. The SLOW gates run OUTSIDE the tool:
# after every source change, run `bash .auto/checks.sh FULL=1` manually (or any slow
# subset); on success it writes .auto/lfm3/.gates_stamp, and this file then passes fast.
# In-bench gates (L/X slots+text+wav md5, WAV self, codec-on-HF SNR in measure_s2s.sh)
# still run on EVERY run_experiment, so no bench sample is ever unverified.
set -euo pipefail
HERE="/home/louis/PULSAR-ASM/.auto/lfm3"
PY="/home/louis/venvs/lfm3/bin/python"
$PY "$HERE/checks_tok.py" >/dev/null 2>&1
$PY "$HERE/checks_mel.py" >/dev/null 2>&1
$PY "$HERE/checks_wav.py" >/dev/null 2>&1
A=/home/louis/PULSAR-ASM/pulsar_arm/asm
STAMP=$(cat $A/lfm3_qlow.cuh $A/lfm3_nanode.cu $A/lfm3_depth.cu $A/lfm3_depth_neon.c $A/lfm3_codec.cu $A/lfm3_q8g.cuh $A/lfm3_speak.c $A/lfm3_asr.c $A/lfm3_enc_gpu.cu | md5sum | cut -c1-32)
if [ "${FULL:-}" = 1 ] || [ "$(cat "$HERE/.gates_stamp" 2>/dev/null)" != "$STAMP" ]; then
  # SLOW PATH. Run manually with FULL=1 (or first run after a source change); under the
  # tool's 300 s timeout this will time out -- that is the honest signal to go manual.
  # Backbone text gates (layer matrix, 6-prompt logit gate, greedy streams vs HF): every
  # GEMV kernel change reaches the text path too, so they run on every source change.
  bash "$HERE/checks.sh" > /tmp/lfm3_text_gates.log 2>&1 || { tail -20 /tmp/lfm3_text_gates.log; echo "TEXT GATES FAILED"; exit 1; }
  # Lossy default speech backbone (GPTQ Q5 mix, adopted 2026-11-11): teacher-forced turn
  # vs HF on every source change; the free-turn ASR WER gate (~10 min) whenever a source
  # on the speech path or the ASR path changed since its last pass.
  bash "$HERE/s2s_tf.sh" --gate lfm3-audio-2b-q5c.lfm3q8 > /tmp/lfm3_tf_gate.log 2>&1 || { cat /tmp/lfm3_tf_gate.log; echo "S2S TF GATE FAILED"; exit 1; }
  if [ "$(cat "$HERE/.intel_stamp" 2>/dev/null)" != "$STAMP" ]; then
    bash "$HERE/s2s_intel.sh" lfm3-audio-2b-q5c.lfm3q8 > /tmp/lfm3_intel_gate.log 2>&1 || { grep -v Warn /tmp/lfm3_intel_gate.log | tail -14; echo "S2S INTEL GATE FAILED"; exit 1; }
    echo "$STAMP" > "$HERE/.intel_stamp"
  fi
  echo "$STAMP" > "$HERE/.gates_stamp"
fi
echo "CHECKS OK"
