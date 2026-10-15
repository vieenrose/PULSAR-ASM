#!/bin/sh
# Build the Gemma 4 E2B Nano engine and its two test programs, on the Nano
# (gcc 7.5 + the L4T driver; the CUDA toolkit is NOT needed: the PTX is embedded).
# Every binary is removed first, so a failed build can never leave an old one behind.
set -e
cd "$(dirname "$0")"
CF="-O2 -fopenmp -ffp-contract=off -Wall -Wextra"
rm -f g4eng g4gemv_test g4host_test
gcc $CF -o g4eng g4eng.c g4_host.S -ldl -lm
gcc $CF -o g4gemv_test g4gemv_test.c -ldl -lm
gcc $CF -o g4host_test g4host_test.c g4_host.S -lm
echo "built: g4eng g4gemv_test g4host_test"
