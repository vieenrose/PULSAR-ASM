#!/usr/bin/env bash
# measure_s2s.sh -- speech-turn (S2S output side) realtime harness on the Nano (picard).
#
# Workload: the gated interleaved turn speak_gate_q8 (25-token prompt, HF-recorded MaskGIT
# orders), 47 slots = 11 text + 36 audio frames (1 EOS), 2.80 s of 24 kHz audio.
# Builds lfm3_speak / lfm3_depth / lfm3_codec / lfm3_nanode from the working tree to /tmp,
# installs them only if every build succeeded, cold page cache, one run.
#
# Gates (any failure => exit 1, no METRIC lines):
#   slots.txt and text.i32 byte-identical to HF's reference (speak_gate_q8/)
#   out.wav byte-identical to .auto/lfm3/s2s_ref.wav.md5 (the engine's gated waveform)
# Metrics:
#   s2s_rtf     = (WAV written - first slot) / audio seconds   [primary, steady generation]
#   s2s_load_s  = start -> first slot (cold load + prefill)
#   s2s_wall_s  = whole turn
#   s2s_text_ms / s2s_audio_ms = median per-slot wall by kind
set -uo pipefail
cd "$(dirname "$0")/../.."
NANO="${NANO:-picard@100.84.151.116}"
DIR=/home/picard/lfm3
S=pulsar_arm/asm
REFMD5=$(cat .auto/lfm3/s2s_ref.wav.md5)
scp -q $S/lfm3_speak.c $S/lfm3_depth.cu $S/lfm3_depth_neon.c $S/lfm3_codec.cu $S/lfm3_q8g.cuh \
    $S/lfm3_nanode.cu pulsar_arm/bonsai2/glibc_mathf.h "$NANO:/tmp/s2s_src/" 2>/dev/null || {
  ssh "$NANO" 'mkdir -p /tmp/s2s_src' && scp -q $S/lfm3_speak.c $S/lfm3_depth.cu $S/lfm3_depth_neon.c \
    $S/lfm3_codec.cu $S/lfm3_q8g.cuh $S/lfm3_nanode.cu pulsar_arm/bonsai2/glibc_mathf.h "$NANO:/tmp/s2s_src/"; } || { echo "scp failed"; exit 2; }
OUT=$(ssh "$NANO" "bash -s" <<EOF 2>&1 | grep -v setlocale
set -u
cd /tmp/s2s_src && export PATH=/usr/local/cuda/bin:\$PATH
rm -f b_speak b_depth b_codec b_nanode
gcc -O2 -o b_speak lfm3_speak.c > b.log 2>&1
gcc -O2 -fopenmp -c lfm3_depth_neon.c -o depth_neon.o >> b.log 2>&1
nvcc -std=c++11 -gencode arch=compute_53,code=sm_53 -O2 -fmad=false -Xptxas -O3,-dlcm=ca -I. -Xcompiler -fopenmp -o b_depth lfm3_depth.cu depth_neon.o -lgomp >> b.log 2>&1
nvcc -std=c++11 -gencode arch=compute_53,code=sm_53 -O2 -fmad=false -Xptxas -O3,-dlcm=ca -I. -Xcompiler -fopenmp -o b_codec lfm3_codec.cu depth_neon.o -lgomp >> b.log 2>&1
nvcc -std=c++11 -gencode arch=compute_53,code=sm_53 -O3 -fmad=false -Xptxas -O3,-dlcm=ca -o b_nanode lfm3_nanode.cu -lz >> b.log 2>&1
for b in b_speak b_depth b_codec b_nanode; do test -s \$b || { echo "BUILD FAILED: \$b"; grep -i error b.log | head; exit 1; }; done
cd $DIR
cp /tmp/s2s_src/b_depth lfm3_depth; cp /tmp/s2s_src/b_codec lfm3_codec; cp /tmp/s2s_src/b_nanode lfm3_nanode; cp /tmp/s2s_src/b_speak lfm3_speak_t
python3 -c "
import os
for p in ['lfm3-audio-2b-q8.lfm3q8','lfm3-audio-2b-depth.lfm3q8','lfm3-audio-2b-codec.lfm3q8']:
    fd=os.open(os.path.realpath(p),os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)"
rm -f /tmp/s2s.*
timeout 900 ./lfm3_speak_t . speak_gate_q8/prompt_ids.i32 interleaved /tmp/s2s 150 speak_gate_q8 2>&1 | python3 -c "
import sys, time
t0 = time.time(); rows = []; wav = None
for l in sys.stdin:
    t = time.time() - t0
    if l.startswith('slot'): rows.append((t, l.split()[2]))
    if l.startswith('AUDIO /tmp/s2s.wav'): wav = (t, float(l.split('(')[1].split()[0]))
    if l.startswith('STIME'): print(l.strip())
end = time.time() - t0
if not rows or not wav: print('RUN FAILED'); sys.exit(1)
dt = [(rows[i][0] - rows[i-1][0], rows[i][1]) for i in range(1, len(rows))]
med = lambda v: sorted(v)[len(v)//2] if v else 0
print('TIMES load %.3f gen %.3f wall %.3f audio %.3f text_ms %.1f audio_ms %.1f' % (rows[0][0], wav[0] - rows[0][0], end, wav[1],
      1000 * med([d for d, k in dt if k == 'T']), 1000 * med([d for d, k in dt if k == 'A'])))
"
cmp -s /tmp/s2s.slots.txt speak_gate_q8/slots.txt && echo GATE_SLOTS_OK || echo GATE_SLOTS_FAIL
cmp -s /tmp/s2s.text.i32 speak_gate_q8/text.i32 && echo GATE_TEXT_OK || echo GATE_TEXT_FAIL
[ "\$(md5sum < /tmp/s2s.wav | cut -c1-32)" = "$REFMD5" ] && echo GATE_WAV_OK || echo GATE_WAV_FAIL
EOF
)
echo "$OUT" | tail -12
echo "$OUT" | grep -q "BUILD FAILED\|RUN FAILED" && exit 1
for g in SLOTS TEXT WAV; do echo "$OUT" | grep -q "GATE_${g}_OK" || { echo "GATE $g FAILED"; exit 1; }; done
echo "$OUT" | awk '/^TIMES/{printf "METRIC s2s_rtf=%.4f\nMETRIC s2s_gen_s=%.3f\nMETRIC s2s_load_s=%.3f\nMETRIC s2s_wall_s=%.3f\nMETRIC s2s_text_ms=%.1f\nMETRIC s2s_audio_ms=%.1f\n", $5/$9, $5, $3, $7, $11, $13}'
