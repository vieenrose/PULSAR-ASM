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
for f in pulsar_arm/asm/lfm3_nanode.cu pulsar_arm/asm/lfm3_cpurows.cpp pulsar_arm/asm/lfm3_qlow.cuh pulsar_arm/asm/lfm3_enc_gpu.cu pulsar_arm/asm/lfm3_wav.c pulsar_arm/asm/lfm3_mel.c pulsar_arm/asm/lfm3_mel.S pulsar_arm/asm/lfm3_asr.c pulsar_arm/asm/lfm3_tok.c pulsar_arm/asm/lfm3_tok.h; do
  scp -q -o ConnectTimeout=20 "$f" "$NANO:$DIR/$(basename $f)" 2>/dev/null
done

BUILD=$(ssh -o ConnectTimeout=25 -o BatchMode=yes "$NANO" \
  "cd $DIR && rm -f lfm3_nanode.new lfm3_enc_gpu.new && \
   export PATH=/usr/local/cuda/bin:\$PATH && \
   as -o mo lfm3_mel.S 2>&1 && g++ -O3 -mcpu=cortex-a57 -ffp-contract=off -c lfm3_cpurows.cpp -o lfm3_cpurows.o 2>&1 && \
   nvcc -std=c++11 -gencode arch=compute_53,code=sm_53 -O3 -fmad=false -Xptxas -O3,-dlcm=ca -o lfm3_nanode.new lfm3_nanode.cu lfm3_cpurows.o -lz -lpthread 2>&1 && \
   nvcc -std=c++11 -gencode arch=compute_53,code=sm_53 -O3 -fmad=false -Xptxas -O3,-dlcm=ca -o lfm3_enc_gpu.new lfm3_enc_gpu.cu -lz 2>&1 && \
   gcc -O2 -std=c99 -o lfm3_wav lfm3_wav.c 2>&1 && gcc -O2 -std=c99 -o lfm3_mel lfm3_mel.c mo -lm 2>&1 && \
   rm -f lfm3_asr.new && gcc -O2 -pthread -o lfm3_asr.new lfm3_asr.c lfm3_tok.c -lm 2>/dev/null && test -s lfm3_asr.new && mv -f lfm3_asr.new lfm3_asr && \
   test -x lfm3_nanode.new && mv -f lfm3_nanode.new lfm3_nanode && \
   test -x lfm3_enc_gpu.new && mv -f lfm3_enc_gpu.new lfm3_enc_gpu && echo BUILT_OK" 2>/dev/null | grep -vi setlocale)
echo "$BUILD" | grep -i "error:" | head -5
echo "$BUILD" | grep -q BUILT_OK || { echo "BUILD FAILED"; exit 1; }

# Run ASR chain
WAV="/home/picard/lfm3/gate_60s_16k_mono.wav"
IDS=$(cat .auto/lfm3/speech/asr_prompt_ids.txt 2>/dev/null | tr -d '\n')
[ -z "$IDS" ] && IDS=$(python3 -c "import json;print(','.join(map(str,json.load(open('.auto/lfm3/speech/hf_asr_golden.json'))['prompt']['prompt_ids'] if 'prompt_ids' in json.load(open('.auto/lfm3/speech/hf_asr_golden.json'))['prompt'] else ''))" 2>/dev/null)

# Cold start, every iteration: drop the model files from the page cache (no root needed for
# clean pages). Without this a previous run or a probe can leave the 1.95 GB blob cached and
# the load looks 20 s faster than a real first invocation.
ssh -o ConnectTimeout=25 -o BatchMode=yes "$NANO" "cd $DIR && python3 -c \"
import os
for p in ['lfm3-audio-2b-q8.lfm3q8','lfm3-audio-2b-enc.lfm3q8','tok.bin','mel.bin','$(basename $WAV)']:
    fd=os.open(os.path.realpath(p),os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)
print('page cache: model files evicted (cold start)')
\"" 2>/dev/null | grep -v setlocale
T0=$(date +%s%N)
OUT=$(ssh -o ConnectTimeout=25 -o BatchMode=yes "$NANO" \
  "cd $DIR && timeout 1800 ./lfm3_asr . $WAV 40 --tokens-only 2>&1" 2>/dev/null | grep -vi setlocale)
T1=$(date +%s%N)
WALL=$(python3 -c "print(($T1-$T0)/1e9)")
echo "$OUT" | tail -3
# Quality gate: the first 40 greedy ids must equal HF's golden (the engine's greedy
# prefix matches HF for 207 tokens, so any change here is a real regression).
GOT=$(echo "$OUT" | sed -n 's/^IDS_PLAIN //p' | tr -s ' ' | sed 's/ *$//')
WANT=$(tr -s ' ' < .auto/lfm3/speech/hf_first40.txt | sed 's/ *$//')
if [ "$GOT" != "$WANT" ]; then
  echo "QUALITY FAIL: first 40 ids differ from the HF golden"
  echo " got:  $GOT"; echo " want: $WANT"; exit 3
fi
echo "QUALITY OK: 40/40 ids equal the HF golden"
echo "METRIC s2t_wall_s=$WALL"
python3 -c "print('METRIC s2t_rt=%.4f' % ($WALL/129.72))"
