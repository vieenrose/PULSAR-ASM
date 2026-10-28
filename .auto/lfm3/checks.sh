#!/usr/bin/env bash
# Accuracy gate wrapper: ALWAYS rebuild from the working tree first (an
# uncommitted-checks bug in this repo's family: checks that measure a stale
# binary certify nothing), then run the three HF-derived gates.
#   bash .auto/checks.sh          # gates for the Q8 arm (6/8 certified)
#   LFM3_BLOB=lfm3-audio-2b-cert.lfm3q8 LFM3_BIN=lfm3_cert bash .auto/checks.sh A B
set -uo pipefail
cd "$(dirname "$0")/../.."
NANO="${NANO:-picard@100.84.151.116}"
DIR="${NANO_DIR:-/home/picard/lfm3}"
BIN="${LFM3_BIN:-lfm3_nanode}"
scp -o ConnectTimeout=20 -q pulsar_arm/asm/lfm3_nanode.cu "$NANO:$DIR/" 2>/dev/null || { echo "scp failed"; exit 2; }
BOUT=$(ssh -o ConnectTimeout=25 -o BatchMode=yes "$NANO" \
  "cd $DIR && rm -f $BIN.new && export PATH=/usr/local/cuda/bin:\$PATH && nvcc -gencode arch=compute_53,code=sm_53 -O3 -fmad=false -Xptxas -O3,-dlcm=ca -o $BIN.new lfm3_nanode.cu -lz 2>&1; test -x $BIN.new && mv -f $BIN.new $BIN && echo BUILT_OK" 2>/dev/null | grep -vi setlocale)
echo "$BOUT" | grep -E "error:" | head -12
echo "$BOUT" | grep -q BUILT_OK || { echo "BUILD FAILED (refusing to measure a stale binary)"; exit 1; }
exec "${PY:-/home/louis/venvs/lfm3/bin/python}" .auto/lfm3/checks_lfm3.py "$@"
