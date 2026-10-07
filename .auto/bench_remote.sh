#!/bin/bash
# Runs ON the Spark (shipped by measure.sh). Args: model-path, prompt ids...
# Builds CURRENT tree with PGO (profile on prefill, optimize, then time) and
# HEAD baseline with standard flags. Runs prefill then decode for each
# (decode order reversed to cancel drift), prints timings.
set -euo pipefail
cd ~/bonsai2/exp
MODEL=$1; shift
IDS="$*"
LB=$HOME/bonsai2/llama.cpp/build/bin
F="-O2 -fopenmp -march=armv8.2-a+dotprod+fp16"
# candidate: standard build (PGO and LTO both measured neutral and reverted;
# keep this block plain so A/B compares code, not build tricks)
gcc $F -DTQ_XGEMV_LIB -Dmain=tq_neon_main -c tq_gemv_neon.c -o fwd_neon_mt.o
gcc $F -c fwd.c -o fwd.o
gcc -O2 -fopenmp -o fwd_exp fwd.o fwd_neon_mt.o fwht.S -L$LB -lggml-base -lggml-cpu -Wl,-rpath,$LB -lm
# baseline: standard flags (fixed reference)
mkdir -p base && cd base
gcc $F -DTQ_XGEMV_LIB -Dmain=tq_neon_main -c tq_gemv_neon.c -o fwd_neon_mt.o
gcc $F -c fwd.c -o fwd.o
gcc -O2 -fopenmp -o ../fwd_base fwd.o fwd_neon_mt.o fwht.S -L$LB -lggml-base -lggml-cpu -Wl,-rpath,$LB -lm
cd ..
echo BUILD_OK
: "${OMP_NT_EXP:=20}"  # candidate thread count (base is always 20)
run_one() { # $1=bin $2=gen $3=outfile $4=threads $5=launcher-prefix
    S=$(date +%s%N)
    # shellcheck disable=SC2086
    OMP_NUM_THREADS=$4 $5 ./$1 $MODEL $IDS --gen $2 > $3 2>&1
    E=$(date +%s%N)
    echo "$((E-S)) $(grep -c '^step' $3)"
}
# warm-up both binaries (clocks, caches, page faults settle before timing)
run_one fwd_exp 0 /dev/null $OMP_NT_EXP "$PRE_EXP" > /dev/null
run_one fwd_base 0 /dev/null ${OMP_NT_BASE:-16} "$PRE_BASE" > /dev/null
echo "--- prefill A then B"
read PNS_A NPS_A <<< $(run_one fwd_exp 0 exp_pre.txt $OMP_NT_EXP "$PRE_EXP")
read PNS_B NPS_B <<< $(run_one fwd_base 0 base_pre.txt ${OMP_NT_BASE:-16} "$PRE_BASE")
echo "--- decode A,B,A,B (alternated, averaged)"
read D1B N1B <<< $(run_one fwd_base 20 base_dec1.txt ${OMP_NT_BASE:-16} "$PRE_BASE")
read D1A N1A <<< $(run_one fwd_exp 20 exp_dec1.txt $OMP_NT_EXP "$PRE_EXP")
read D2A N2A <<< $(run_one fwd_exp 20 exp_dec2.txt $OMP_NT_EXP "$PRE_EXP")
read D2B N2B <<< $(run_one fwd_base 20 base_dec2.txt ${OMP_NT_BASE:-16} "$PRE_BASE")
[ "$N1B" -eq 41 ] && [ "$N1A" -eq 41 ] && [ "$N2A" -eq 41 ] && [ "$N2B" -eq 41 ] \
    || { echo "DECODE STEPS wrong: $N1B $N1A $N2A $N2B"; exit 1; }
echo "PHASES D1B=$D1B D1A=$D1A D2A=$D2A D2B=$D2B"
SPREAD=$(python3 -c "v=sorted([$D1B,$D1A,$D2A,$D2B]); print(v[-1]/v[0])")
echo "SPREAD=$SPREAD"
python3 -c "import sys; sys.exit(0 if float('$SPREAD') < 1.08 else 1)" \
    || { echo "ABORT: phase spread $SPREAD exceeds 8% (bursty contention)"; exit 3; }
DNS_B=$(( (D1B + D2B) / 2 )); NDS_B=$N1B
DNS_A=$(( (D1A + D2A) / 2 )); NDS_A=$N1A
echo "PREFILL_NS_A=$PNS_A NSTEPS_A=$NPS_A"
echo "PREFILL_NS_B=$PNS_B NSTEPS_B=$NPS_B"
echo "DECODE_NS_A=$DNS_A NSTEPS_A=$NDS_A"
echo "DECODE_NS_B=$DNS_B NSTEPS_B=$NDS_B"
cat /proc/loadavg
