#!/bin/bash
# Runs ON the Spark (shipped by measure.sh). Args: model-path, prompt ids...
# Builds CURRENT tree (fwd_exp) and HEAD baseline (fwd_base), runs prefill
# then decode for each (decode order reversed to cancel linear drift), prints
# timings. A/B in one invocation shares machine conditions between the two.
set -euo pipefail
cd ~/bonsai2/exp
MODEL=$1; shift
IDS="$*"
LB=$HOME/bonsai2/llama.cpp/build/bin
F="-O2 -fopenmp -march=armv8.2-a+dotprod+fp16"
gcc $F -DTQ_XGEMV_LIB -Dmain=tq_neon_main -c tq_gemv_neon.c -o fwd_neon_mt.o
gcc $F -c fwd.c -o fwd.o
gcc -O2 -fopenmp -o fwd_exp fwd.o fwd_neon_mt.o fwht.S -L$LB -lggml-base -lggml-cpu -Wl,-rpath,$LB -lm
mkdir -p base && cd base
gcc $F -DTQ_XGEMV_LIB -Dmain=tq_neon_main -c tq_gemv_neon.c -o fwd_neon_mt.o
gcc $F -c fwd.c -o fwd.o
gcc -O2 -fopenmp -o ../fwd_base fwd.o fwd_neon_mt.o fwht.S -L$LB -lggml-base -lggml-cpu -Wl,-rpath,$LB -lm
cd ..
echo BUILD_OK
export OMP_NUM_THREADS=20
run_one() { # $1=bin $2=gen $3=outfile
    S=$(date +%s%N)
    # shellcheck disable=SC2086
    taskset -c 0-19 ./$1 $MODEL $IDS --gen $2 > $3 2>&1
    E=$(date +%s%N)
    echo "$((E-S)) $(grep -c '^step' $3)"
}
echo "--- prefill A then B"
read PNS_A NPS_A <<< $(run_one fwd_exp 0 exp_pre.txt)
read PNS_B NPS_B <<< $(run_one fwd_base 0 base_pre.txt)
echo "--- decode B then A (reversed)"
read DNS_B NDS_B <<< $(run_one fwd_base 20 base_dec.txt)
read DNS_A NDS_A <<< $(run_one fwd_exp 20 exp_dec.txt)
echo "PREFILL_NS_A=$PNS_A NSTEPS_A=$NPS_A"
echo "PREFILL_NS_B=$PNS_B NSTEPS_B=$NPS_B"
echo "DECODE_NS_A=$DNS_A NSTEPS_A=$NDS_A"
echo "DECODE_NS_B=$DNS_B NSTEPS_B=$NDS_B"
cat /proc/loadavg
