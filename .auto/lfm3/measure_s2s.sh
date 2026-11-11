#!/usr/bin/env bash
# measure_s2s.sh -- speech-turn (S2S output side) realtime harness on the Nano (picard).
#
# Workload: the gated interleaved turn speak_gate_q8 (25-token prompt, HF-recorded MaskGIT
# orders), 47 slots = 11 text + 36 audio frames (1 EOS), 2.80 s of 24 kHz audio.
# Builds lfm3_speak / lfm3_depth / lfm3_codec / lfm3_nanode from the working tree to /tmp,
# installs them only if every build succeeded, cold page cache, one run.
#
# Two cold turns per run. L = the shipped default backbone, the lossy GPTQ Q5 mix (the
# metrics come from L); X = the exact Q8 backbone (LFM_SPK_BB=lfm3-audio-2b-q8.lfm3q8).
# Gates (any failure => exit 1, no METRIC lines):
#   X: slots.txt and text.i32 byte-identical to HF's reference (speak_gate_q8/), wav md5 ==
#      .auto/lfm3/s2s_ref.wav.md5
#   L: text.i32 == HF's (its audio frames may differ: lossy); wav md5 ==
#      .auto/lfm3/s2s_ref_q5c.wav.md5 (pinned 2026-11-11 from the first default-Q5 turn: a
#      regression detector, not a fidelity claim; re-pin only with a recorded reason)
#   L and X: wav == batch decode of the turn's own frames; codec on HF's codes >= 30 dB
#   The lossy blob's distance from HF is gated in checks.sh (s2s_tf.sh --gate, s2s_intel.sh).
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
REFMD5L=$(cat .auto/lfm3/s2s_ref_q5c.wav.md5)
scp -q $S/lfm3_speak.c $S/lfm3_depth.cu $S/lfm3_depth_neon.c $S/lfm3_codec.cu $S/lfm3_q8g.cuh $S/lfm3_qlow.cuh \
    $S/lfm3_nanode.cu pulsar_arm/bonsai2/glibc_mathf.h "$NANO:/tmp/s2s_src/" 2>/dev/null || {
  ssh "$NANO" 'mkdir -p /tmp/s2s_src' && scp -q $S/lfm3_speak.c $S/lfm3_depth.cu $S/lfm3_depth_neon.c \
    $S/lfm3_codec.cu $S/lfm3_q8g.cuh $S/lfm3_qlow.cuh $S/lfm3_nanode.cu pulsar_arm/bonsai2/glibc_mathf.h "$NANO:/tmp/s2s_src/"; } || { echo "scp failed"; exit 2; }
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
cat > /tmp/s2s_times.py <<'PYEOF'
import sys, time
pf, tag = sys.argv[1], sys.argv[2]
t0 = time.time(); rows = []; wav = None
for l in sys.stdin:
    t = time.time() - t0
    if l.startswith('slot'): rows.append((t, l.split()[2]))
    if l.startswith('AUDIO ' + pf + '.wav'): wav = (t, float(l.split('(')[1].split()[0]))
    if l.startswith('STIME'): print(tag, l.strip())
end = time.time() - t0
if not rows or not wav: print('RUN FAILED', tag); sys.exit(1)
dt = [(rows[i][0] - rows[i-1][0], rows[i][1]) for i in range(1, len(rows))]
med = lambda v: sorted(v)[len(v)//2] if v else 0
print('TIMES_%s load %.3f gen %.3f wall %.3f audio %.3f text_ms %.1f audio_ms %.1f' % (tag, rows[0][0], wav[0] - rows[0][0], end, wav[1],
      1000 * med([d for d, k in dt if k == 'T']), 1000 * med([d for d, k in dt if k == 'A'])))
PYEOF
# WAV self gate (#239): the turn's WAV must equal a batch decode of its own frame record,
# transposed to the codec's codebook-major layout here, not by lfm3_speak.
cat > /tmp/s2s_self.py <<'PYEOF'
import sys, numpy as np, subprocess
pf, tag = sys.argv[1], sys.argv[2]
fr = np.fromfile(pf + '.frames.i32', dtype=np.int32).reshape(-1, 12)
while len(fr) and (fr[-1] == 2016).all(): fr = fr[:-1]
np.ascontiguousarray(fr.T).tofile(pf + '.gate_cm.i32')
subprocess.run(['./lfm3_codec', 'lfm3-audio-2b-codec.lfm3q8', pf + '.gate_cm.i32', str(len(fr)), pf + '.gate.f32'], stdout=subprocess.DEVNULL, check=True)
subprocess.run(['./lfm3_wavout', pf + '.gate.f32', pf + '.gate.wav', '24000'], stdout=subprocess.DEVNULL, check=True)
same = open(pf + '.gate.wav', 'rb').read() == open(pf + '.wav', 'rb').read()
print('GATE_%s_WAV_SELF_%s' % (tag, 'OK' if same else 'FAIL'))
PYEOF
turn() {  # turn <tag> <prefix> [VAR=val ...]: one cold turn + its text and WAV self gates
  tag=\$1; pf=\$2; shift 2
  python3 -c "
import os
for p in ['lfm3-audio-2b-q8.lfm3q8','lfm3-audio-2b-q5c.lfm3q8','lfm3-audio-2b-depth.lfm3q8','lfm3-audio-2b-codec.lfm3q8']:
    fd=os.open(os.path.realpath(p),os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)"
  rm -f \$pf.*
  env "\$@" timeout 900 ./lfm3_speak_t . speak_gate_q8/prompt_ids.i32 interleaved \$pf 150 speak_gate_q8 2>&1 | python3 /tmp/s2s_times.py \$pf \$tag
  cmp -s \$pf.text.i32 speak_gate_q8/text.i32 && echo GATE_\${tag}_TEXT_OK || echo GATE_\${tag}_TEXT_FAIL
  python3 /tmp/s2s_self.py \$pf \$tag
}
turn L /tmp/s2s
[ "\$(md5sum < /tmp/s2s.wav | cut -c1-32)" = "$REFMD5L" ] && echo GATE_L_WAV_OK || echo GATE_L_WAV_FAIL
turn X /tmp/s2x LFM_SPK_BB=lfm3-audio-2b-q8.lfm3q8
cmp -s /tmp/s2x.slots.txt speak_gate_q8/slots.txt && echo GATE_X_SLOTS_OK || echo GATE_X_SLOTS_FAIL
[ "\$(md5sum < /tmp/s2x.wav | cut -c1-32)" = "$REFMD5" ] && echo GATE_X_WAV_OK || echo GATE_X_WAV_FAIL
# The codec on HF's own codes must reproduce HF's waveform (>= 30 dB SNR).
python3 - <<'PYEOF'
import numpy as np, subprocess
A = np.fromfile('speak_gate_q8/audio_codes.i32', dtype=np.int32); TA = len(A) // 12
A = A.reshape(12, TA)
n = TA
while n and (A[:, n - 1] == 2016).all(): n -= 1
np.ascontiguousarray(A[:, :n]).tofile('/tmp/s2s.hf_cm.i32')
subprocess.run(['./lfm3_codec', 'lfm3-audio-2b-codec.lfm3q8', '/tmp/s2s.hf_cm.i32', str(n), '/tmp/s2s.hf.f32'], stdout=subprocess.DEVNULL, check=True)
y = np.fromfile('/tmp/s2s.hf.f32', dtype=np.float32); r = np.fromfile('speak_gate_q8/audio.f32', dtype=np.float32)
m = min(len(y), len(r)); snr = 10 * np.log10((r[:m] ** 2).sum() / ((y[:m] - r[:m]) ** 2).sum())
print('CODEC_HF_SNR %.1f' % snr); print('GATE_CODEC_HF_OK' if snr >= 30 else 'GATE_CODEC_HF_FAIL')
PYEOF
EOF
)
echo "$OUT" | grep -E "^(GATE|CODEC|TIMES|RUN|BUILD)|STIME (backbone|lfm3_depth|wall)"
echo "$OUT" | grep -q "BUILD FAILED\|RUN FAILED" && exit 1
for g in L_TEXT L_WAV L_WAV_SELF X_TEXT X_SLOTS X_WAV X_WAV_SELF CODEC_HF; do echo "$OUT" | grep -q "GATE_${g}_OK" || { echo "GATE $g FAILED"; exit 1; }; done
echo "$OUT" | awk '/^TIMES_L/{printf "METRIC s2s_rtf=%.4f\nMETRIC s2s_gen_s=%.3f\nMETRIC s2s_load_s=%.3f\nMETRIC s2s_wall_s=%.3f\nMETRIC s2s_text_ms=%.1f\nMETRIC s2s_audio_ms=%.1f\n", $5/$9, $5, $3, $7, $11, $13}
/^TIMES_X/{printf "METRIC s2s_q8_rtf=%.4f\n", $5/$9}'
