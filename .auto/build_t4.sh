#!/bin/bash
# Build the Ternary-Bonsai-4B engine (fwd8_t4: the pq1 engine with -DPQ1_TERN
# -DBONSAI_4B) and its kernel gate (pq1t_parity) ON THE NANO, in $R (separate
# from the Q1 build dir so the two never share objects). Same compiler flags as
# build_q1.sh plus the two defines.
set -uo pipefail
NANO=${NANO:-pesi@100.70.245.36}
REPO=/home/louis/PULSAR-ASM
SRC=$REPO/pulsar_arm/bonsai2
R=${R:-/home/pesi/t4}
D="-DPQ1_TERN -DBONSAI_4B"
ssh -o ConnectTimeout=25 "$NANO" "mkdir -p $R" || { echo "BUILD FAILED: ssh"; exit 1; }
scp -q -o ConnectTimeout=25 \
  "$SRC/fwd8.c" "$SRC/pq1_expf_neon.h" "$SRC/pq1_swar_sm53.cu" "$SRC/pq1_wrap.h" "$SRC/pq1_host.c" \
  "$SRC/pq1_fwd8_glue.c" "$SRC/pq1_fwd8_shim.h" "$SRC/pq1t_parity.c" \
  "$NANO:$R/" || { echo "BUILD FAILED: scp"; exit 1; }
BUILD=$(ssh -o ConnectTimeout=25 "$NANO" "cd $R && export PATH=/usr/local/cuda-10.2/bin:\$PATH \
  CUDAINC=/usr/local/cuda-10.2/targets/aarch64-linux/include && set -o pipefail && \
  rm -f fwd8_t4 pq1t_parity *.o && \
  nvcc -gencode arch=compute_53,code=sm_53 -O3 -maxrregcount=48 --use_fast_math -Xptxas -O3,-dlcm=ca -std=c++11 $D -c pq1_swar_sm53.cu -o pq1_swar_sm53.o 2>&1 | grep -v pq1_fa_mask | grep -e error | head -n5; \
  gcc -O2 -fopenmp -mcpu=cortex-a57 -ffp-contract=off -DSTANDALONE -DPQ1_REAL_CUDA $D -c pq1_host.c -o pq1_host.o 2>&1 | grep -e ' error' | head -n3; \
  gcc -O2 -fopenmp -mcpu=cortex-a57 -DSTANDALONE -DPQ1_USE_CUDA -DBQ1_TIME $D -include $R/pq1_fwd8_shim.h -c fwd8.c -o fwd8_t4.o 2>&1 | grep -e ' error' | head -n3; \
  gcc -O2 -fopenmp -mcpu=cortex-a57 -DSTANDALONE -DPQ1_USE_CUDA $D -I\$CUDAINC -c pq1_fwd8_glue.c -o pq1_glue.o 2>&1 | grep -e ' error' | head -n3; \
  gcc -O2 -fopenmp -mcpu=cortex-a57 -ffp-contract=off -DSTANDALONE -DPQ1_REAL_CUDA $D -I\$CUDAINC -c pq1t_parity.c -o pq1t_parity.o 2>&1 | grep -e ' error' | head -n3; \
  g++ fwd8_t4.o pq1_host.o pq1_glue.o pq1_swar_sm53.o -o fwd8_t4 -L/usr/local/cuda-10.2/targets/aarch64-linux/lib -lcudart -lm -lpthread -fopenmp 2>&1 | head -n3 && \
  g++ pq1_host.o pq1t_parity.o pq1_swar_sm53.o -o pq1t_parity -L/usr/local/cuda-10.2/targets/aarch64-linux/lib -lcudart -lm -lpthread -fopenmp 2>&1 | head -n3 && \
  echo BUILT" 2>&1)
echo "$BUILD" | grep -q BUILT || { echo "BUILD FAILED"; echo "$BUILD" | head -n 20; exit 1; }
echo "build_t4: fwd8_t4 + pq1t_parity rebuilt in $R"
