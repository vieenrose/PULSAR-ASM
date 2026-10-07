#!/bin/bash
# Runs ON the Spark (shipped by measure.sh). Args: model-path, prompt ids...
# Builds fwd_exp, times prefill (prompt --gen 0) and decode (+20 gen).
set -euo pipefail
cd ~/bonsai2/exp
MODEL=$1; shift
IDS="$*"
LB=$HOME/bonsai2/llama.cpp/build/bin
gcc -O2 -fopenmp -march=armv8.2-a+dotprod+fp16 -DTQ_XGEMV_LIB -Dmain=tq_neon_main -c tq_gemv_neon.c -o fwd_neon_mt.o
gcc -O2 -fopenmp -march=armv8.2-a+dotprod+fp16 -c fwd.c -o fwd.o
gcc -O2 -fopenmp -o fwd_exp fwd.o fwd_neon_mt.o fwht.S -L$LB -lggml-base -lggml-cpu -Wl,-rpath,$LB -lm
echo BUILD_OK
export OMP_NUM_THREADS=20
# shellcheck disable=SC2086
S=$(date +%s%N)
# shellcheck disable=SC2086
taskset -c 0-19 ./fwd_exp $MODEL $IDS --gen 0 > prefill_steps.txt 2>&1
E=$(date +%s%N)
echo "PREFILL_NS=$((E-S)) NSTEPS=$(grep -c '^step' prefill_steps.txt)"
S=$(date +%s%N)
# shellcheck disable=SC2086
taskset -c 0-19 ./fwd_exp $MODEL $IDS --gen 20 > decode_steps.txt 2>&1
E=$(date +%s%N)
echo "DECODE_NS=$((E-S)) NSTEPS=$(grep -c '^step' decode_steps.txt)"
cat /proc/loadavg
