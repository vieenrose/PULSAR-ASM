#!/usr/bin/env bash
# checks_gemm.sh -- gate for the batched Q8 GEMM explorer (pulsar_arm/asm/lfm3_gemm.cu).
# Builds from the working tree on the Nano and requires the fp64 accuracy gate to pass
# on every audio shape. Performance is REPORTED, not asserted: this tool exists to
# measure the batched form, and its 2026-10-30 verdict (7.1 GFLOP/s ceiling on every
# shape, no batching win) is recorded in doc/LFM3_PORT_STAGES.md section 10.
set -u
NANO=${NANO:-picard@100.84.151.116}
DIR=${NANO_DIR:-/home/picard/lfm3}/gemm
SRC="$(cd "$(dirname "$0")/../.." && pwd)/pulsar_arm/asm/lfm3_gemm.cu"
ssh -o ConnectTimeout=25 -o BatchMode=yes "$NANO" "mkdir -p $DIR"
scp -q -o ConnectTimeout=20 "$SRC" "$NANO:$DIR/"
out=$(ssh -o ConnectTimeout=25 -o BatchMode=yes "$NANO" \
  "cd $DIR && rm -f lfm3_gemm.new && export PATH=/usr/local/cuda/bin:\$PATH && \
   nvcc -gencode arch=compute_53,code=sm_53 -O3 -Xptxas -O3,-dlcm=ca -o lfm3_gemm.new lfm3_gemm.cu 2>&1; \
   test -x lfm3_gemm.new && mv -f lfm3_gemm.new lfm3_gemm && echo BUILT_OK")
echo "$out" | grep -vi setlocale
case "$out" in *BUILT_OK*) ;; *) echo "build failed -- refusing to measure a stale binary"; exit 1;; esac
ssh -o ConnectTimeout=25 -o BatchMode=yes "$NANO" "cd $DIR && ./lfm3_gemm 2>&1" | grep -vi setlocale | tail -8
rc=${PIPESTATUS[0]}
exit $rc
