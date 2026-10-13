#!/bin/bash
# Build the Q1_0 engine (fwd8_q1) and the parity gate (pq1_parity) ON THE NANO
# from the CURRENT WORKING TREE.
#
# WHY THIS EXISTS (found in run #36, 2026-10-13): checks.sh used to run
# ./pq1_parity and ./fwd8_q1 exactly as the last build left them, so a change
# that is measured only through ab_kernel.sh - which builds its own q1_old /
# q1_new objects and never touches the canonical binaries - got a VACUOUS green
# accuracy gate. Demonstrated with a deliberately broken kernel (inner loop
# kk<4): the stale binary passed all three gates; rebuilding the same source
# made parity fail 0/256 (maxrel 1.78e+04). A gate that can be vacuously green
# invalidates every keep decision, so gates must always build first.
#
# The build commands below are byte-identical to the inline recipe in
# measure.sh (which is deliberately left untouched, being the path that
# produces the primary metric). Keep the two in sync if compiler flags change.
set -uo pipefail
NANO=${NANO:-pesi@100.70.245.36}
REPO=/home/louis/PULSAR-ASM
SRC=$REPO/pulsar_arm/bonsai2
R=/home/pesi/exp

scp -q -o ConnectTimeout=25 \
  "$SRC/fwd8.c" "$SRC/pq1_swar_sm53.cu" "$SRC/pq1_wrap.h" "$SRC/pq1_host.c" \
  "$SRC/pq1_fwd8_glue.c" "$SRC/pq1_fwd8_shim.h" "$SRC/pq1_parity.c" \
  "$NANO:$R/" || { echo "BUILD FAILED: scp"; exit 1; }

BUILD=$(ssh -o ConnectTimeout=25 "$NANO" "cd $R && export PATH=/usr/local/cuda-10.2/bin:\$PATH \
  CUDAINC=/usr/local/cuda-10.2/targets/aarch64-linux/include && set -o pipefail && \
  rm -f fwd8_q1 pq1_parity fwd8_q1.o pq1_glue.o pq1_swar_sm53.o && \
  nvcc -gencode arch=compute_53,code=sm_53 -O3 -maxrregcount=48 --use_fast_math -Xptxas -O3,-dlcm=ca -std=c++11 -c pq1_swar_sm53.cu -o pq1_swar_sm53.o 2>&1 | head -n5 && \
  gcc -O2 -fopenmp -mcpu=cortex-a57 -ffp-contract=off -DSTANDALONE -DPQ1_REAL_CUDA  -c pq1_host.c -o pq1_host.o 2>&1 | head -n3 && \
  gcc -O2 -fopenmp -mcpu=cortex-a57 -DSTANDALONE -DPQ1_USE_CUDA -DBQ1_TIME  -include $R/pq1_fwd8_shim.h -c fwd8.c -o fwd8_q1.o 2>&1 | grep -e ' error' | head -n3; \
  gcc -O2 -fopenmp -mcpu=cortex-a57 -DSTANDALONE -DPQ1_USE_CUDA  -I\$CUDAINC -c pq1_fwd8_glue.c -o pq1_glue.o 2>&1 | grep -e ' error' | head -n3 || true; \
  gcc -O2 -fopenmp -mcpu=cortex-a57 -ffp-contract=off -DSTANDALONE -DPQ1_REAL_CUDA -I\$CUDAINC -c pq1_parity.c -o pq1_parity.o 2>&1 | grep -e ' error' | head -n3 || true; \
  g++ fwd8_q1.o pq1_host.o pq1_glue.o pq1_swar_sm53.o -o fwd8_q1 -L/usr/local/cuda-10.2/targets/aarch64-linux/lib -lcudart -lm -fopenmp 2>&1 | head -n3 && \
  g++ pq1_host.o pq1_parity.o pq1_swar_sm53.o -o pq1_parity -L/usr/local/cuda-10.2/targets/aarch64-linux/lib -lcudart -lm 2>&1 | head -n3 && \
  echo BUILT" 2>&1)

echo "$BUILD" | grep -q BUILT || { echo "BUILD FAILED"; echo "$BUILD" | head -n 20; exit 1; }
echo "build_q1: fwd8_q1 + pq1_parity rebuilt from working tree"
