#!/usr/bin/env bash
# LFM3 ASR latency measure for autoresearch. Lower is better.
# Runs the full speech->text chain on the 60s fixture, reports wall seconds.
set -uo pipefail
cd "$(dirname "$0")/.."
NANO="${NANO:-picard@100.84.151.116}"
DIR="${NANO_DIR:-/home/picard/lfm3}"
BLOB="${LFM3_BLOB:-lfm3-audio-2b-q8.lfm3q8}"
NVCC="export PATH=/usr/local/cuda/bin:\$PATH && nvcc -std=c++11 -gencode arch=compute_53,code=sm_53 -O3 -fmad=false -Xptxas -O3,-dlcm=ca"

# Build
ssh -o ConnectTimeout=25 -o BatchMode=yes "$NANO" "cd $DIR && rm -f lfm3_nanode.new lfm3_enc_gpu.new lfm3_wav lfm3_mel mo 2>/dev/null; as -o mo pulsar_arm/asm/lfm3_mel.S && echo ASM_OK" >/dev/null 2>&1

# Simpler: just build nanode and enc on the Nano from the synced sources
for f in pulsar_arm/asm/lfm3_nanode.cu pulsar_arm/asm/lfm3_enc_gpu.cu pulsar_arm/asm/lfm3_wav.c pulsar_arm/asm/lfm3_mel.c pulsar_arm/asm/lfm3_mel.S; do
  scp -q -o ConnectTimeout=20 "$f" "$NANO:$DIR/$(basename $f)" 2>/dev/null
done

BUILD=$(ssh -o ConnectTimeout=25 -o BatchMode=yes "$NANO" \
  "cd $DIR && rm -f lfm3_nanode.new lfm3_enc_gpu.new && \
   export PATH=/usr/local/cuda/bin:\$PATH && \
   as -o mo lfm3_mel.S 2>&1 && \
   nvcc -std=c++11 -gencode arch=compute_53,code=sm_53 -O3 -fmad=false -Xptxas -O3,-dlcm=ca -o lfm3_nanode.new lfm3_nanode.cu -lz 2>&1 && \
   nvcc -std=c++11 -gencode arch=compute_53,code=sm_53 -O3 -fmad=false -Xptxas -O3,-dlcm=ca -o lfm3_enc_gpu.new lfm3_enc_gpu.cu -lz 2>&1 && \
   gcc -O2 -std=c99 -o lfm3_wav lfm3_wav.c 2>&1 && gcc -O2 -std=c99 -o lfm3_mel lfm3_mel.c mo -lm 2>&1 && \
   test -x lfm3_nanode.new && mv -f lfm3_nanode.new lfm3_nanode && \
   test -x lfm3_enc_gpu.new && mv -f lfm3_enc_gpu.new lfm3_enc_gpu && echo BUILT_OK" 2>/dev/null | grep -vi setlocale)
echo "$BUILD" | grep -i "error:" | head -5
echo "$BUILD" | grep -q BUILT_OK || { echo "BUILD FAILED"; exit 1; }

# Run ASR chain
WAV="/home/picard/lfm3/gate_60s_16k_mono.wav"
IDS=$(cat .auto/lfm3/speech/asr_prompt_ids.txt 2>/dev/null | tr -d '\n')
[ -z "$IDS" ] && IDS=$(python3 -c "import json;print(','.join(map(str,json.load(open('.auto/lfm3/speech/hf_asr_golden.json'))['prompt']['prompt_ids'] if 'prompt_ids' in json.load(open('.auto/lfm3/speech/hf_asr_golden.json'))['prompt'] else ''))" 2>/dev/null)

T0=$(date +%s%N)
OUT=$(ssh -o ConnectTimeout=25 -o BatchMode=yes "$NANO" \
  "cd $DIR && timeout 1800 ./lfm3_asr . $WAV 40 --tokens-only 2>&1" 2>/dev/null | grep -vi setlocale)
T1=$(date +%s%N)
WALL=$(python3 -c "print(($T1-$T0)/1e9)")
echo "$OUT" | tail -3
echo "METRIC s2t_wall_s=$WALL"
