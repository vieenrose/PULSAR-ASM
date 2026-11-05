#!/usr/bin/env bash
set -euo pipefail
HERE="/home/louis/PULSAR-ASM/.auto/lfm3"
PY="/home/louis/venvs/lfm3/bin/python"
$PY "$HERE/checks_tok.py" >/dev/null 2>&1
$PY "$HERE/checks_mel.py" >/dev/null 2>&1
$PY "$HERE/checks_wav.py" >/dev/null 2>&1
echo "CHECKS OK"
