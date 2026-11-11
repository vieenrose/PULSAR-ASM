#!/bin/bash
# Teacher-forced speech-turn quality gate for lossy backbone blobs (Q4 / Q5 GPTQ).
# Runs one interleaved turn on the Nano with HF's audio codes forced into the feedback
# (LFM_DEPTH_FORCE) and HF's MaskGIT reveal orders, so every audio frame is fed exactly
# what HF fed, and scores, against HF's speak_gate_q8 reference:
#   text   : the turn's text tokens == HF's (text slots run on the engine's own argmax)
#   slots  : the slot schedule == HF's
#   hidden : per-frame cosine / relative L2 of the hidden row the depth head receives
#            vs HF's hidden.<f>.f32 (the backbone's state at that frame)
#   codes  : fraction of the engine's OWN MaskGIT codes (before forcing) equal to HF's
# Usage: bash .auto/lfm3/s2s_tf.sh [--gate] [blob ...]   (default: q8 + the shipped q5c)
# --gate: exit 1 unless text == HF, slots == HF, hidden cos min >= 0.99 and rel-L2 mean
#         <= 0.05 (q5c at adoption: 0.99615 / 0.0398; exact Q8: 0.99979 / 0.0114)
set -u
NANO=${NANO:-picard@100.84.151.116}
GATE=0; [ "${1:-}" = "--gate" ] && { GATE=1; shift; }
BLOBS=${*:-lfm3-audio-2b-q8.lfm3q8 lfm3-audio-2b-q5c.lfm3q8}
TFOUT=$(
ssh "$NANO" "bash -s" <<EOF 2>&1 | grep -v setlocale
cd /home/picard/lfm3
for bb in $BLOBS; do
  [ -s \$bb ] || { echo "== \$bb: not on the board"; continue; }
  rm -f /tmp/tf.* /tmp/tf_h.f32 /tmp/tf.depth.log
  LFM_SPK_BB=\$bb LFM_DEPTH_FORCE=speak_gate_q8/audio_codes.i32 LFM_DEPTH_HDUMP=/tmp/tf_h.f32 \
    ./lfm3_speak_t . speak_gate_q8/prompt_ids.i32 interleaved /tmp/tf 150 speak_gate_q8 2>&1 | python3 -c "
import sys, time
t0 = time.time(); rows = []; out = open('/tmp/tf.out', 'w')
for l in sys.stdin:
    out.write(l)
    if l.startswith('slot'): rows.append((time.time() - t0, l.split()[2]))
dt = [(rows[i][0] - rows[i-1][0], rows[i][1]) for i in range(1, len(rows))]
med = lambda v: sorted(v)[len(v)//2] if v else 0
print('  timing: load %.2f s, text slot %.1f ms, audio slot %.1f ms (median, forced codes)' % (rows[0][0] if rows else 0,
      1000 * med([d for d, k in dt if k == 'T']), 1000 * med([d for d, k in dt if k == 'A'])))
"
  echo "== \$bb  rc=\$?"; grep -E '^STIME (backbone|wall)' /tmp/tf.out | sed 's/^/  /' 
  python3 - <<'PY'
import numpy as np, os
ref = 'speak_gate_q8'
t = np.fromfile('/tmp/tf.text.i32', np.int32) if os.path.exists('/tmp/tf.text.i32') else np.zeros(0, np.int32)
r = np.fromfile(ref + '/text.i32', np.int32)
n = min(len(t), len(r)); first = next((i for i in range(n) if t[i] != r[i]), n)
print('  text   %s (%d vs %d tokens, first diff at %s)' % ('== HF' if len(t) == len(r) and first == n else 'DIFFERS', len(t), len(r), first if first < max(len(t), len(r)) else '-'))
s = open('/tmp/tf.slots.txt').read() if os.path.exists('/tmp/tf.slots.txt') else ''
print('  slots  %s' % ('== HF' if s == open(ref + '/slots.txt').read() else 'DIFFER'))
H = np.fromfile('/tmp/tf_h.f32', np.float32).reshape(-1, 2048)
cs, rl = [], []
for f in range(len(H)):
    p = '%s/hidden.%d.f32' % (ref, f)
    if not os.path.exists(p): break
    h = np.fromfile(p, np.float32).reshape(-1)[-2048:]
    cs.append(float(H[f] @ h / np.linalg.norm(H[f]) / np.linalg.norm(h)))
    rl.append(float(np.linalg.norm(H[f] - h) / np.linalg.norm(h)))
cs, rl = np.array(cs), np.array(rl)
print('  hidden %d frames: cos mean %.5f min %.5f | rel-L2 mean %.4f max %.4f' % (len(cs), cs.mean(), cs.min(), rl.mean(), rl.max()))
own = [l for l in open('/tmp/tf.depth.log') if l.startswith('maskgit: codes')] if os.path.exists('/tmp/tf.depth.log') else []
A = np.fromfile(ref + '/audio_codes.i32', np.int32); T = len(A) // 12; A = A.reshape(12, T)
ag = [np.mean([int(x) for x in l.split('[')[1].split(']')[0].split(',')] == A[:, f]) for f, l in enumerate(own) if f < T]
print('  codes  own == HF: %.3f over %d frames (frame 0: %.2f)' % (np.mean(ag) if ag else -1, len(ag), ag[0] if ag else -1))
ok = len(t) == len(r) and first == n and s == open(ref + '/slots.txt').read() and len(cs) > 0 and cs.min() >= 0.99 and rl.mean() <= 0.05
print('  TF_GATE_%s' % ('OK' if ok else 'FAIL'))
PY
done
EOF
)
echo "$TFOUT"
[ $GATE = 0 ] || ! echo "$TFOUT" | grep -q "TF_GATE_FAIL\|not on the board"
