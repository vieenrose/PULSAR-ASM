#!/usr/bin/env bash
set -euo pipefail
HERE="/home/louis/PULSAR-ASM/.auto/lfm3"
PY="/home/louis/venvs/lfm3/bin/python"
$PY "$HERE/checks_tok.py" >/dev/null 2>&1
$PY "$HERE/checks_mel.py" >/dev/null 2>&1
$PY "$HERE/checks_wav.py" >/dev/null 2>&1
# Backbone text gates (layer matrix, 6-prompt logit gate, greedy streams vs HF): every
# GEMV kernel change reaches the text path too, so they run on every iteration.
bash "$HERE/checks.sh" > /tmp/lfm3_text_gates.log 2>&1 || { tail -20 /tmp/lfm3_text_gates.log; echo "TEXT GATES FAILED"; exit 1; }
echo "CHECKS OK"
