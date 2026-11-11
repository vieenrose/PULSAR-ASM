#!/bin/bash
# End-to-end intelligibility of free-running speech turns (lossy backbone check).
# For each blob and each speak fixture prompt (speak_hf_oracle.py PROMPTS 0..4; 1..4 are
# never used for calibration or tuning), run a free interleaved turn on the Nano (own codes,
# own MaskGIT order), transcribe its WAV with the exact Q8 speech-to-text pipeline
# (lfm3_asr, after a 24 -> 16 kHz resample on the host) and report the word error rate of
# the transcript against the turn's own text.
# Usage: bash .auto/lfm3/s2s_intel.sh [blob ...]
set -u
NANO=${NANO:-picard@100.84.151.116}
PY=${PY:-/home/louis/venvs/lfm3/bin/python}
BLOBS=${*:-lfm3-audio-2b-q8.lfm3q8}
W=/tmp/s2s_intel; rm -rf $W; mkdir -p $W
HERE=$(cd "$(dirname "$0")" && pwd)
$PY - "$W" <<'PY'
import sys, numpy as np
sys.path.insert(0, sys.argv[0] and __import__("os").path.join("/home/louis/PULSAR-ASM/.auto/lfm3/enc_ref"))
from transformers import AutoTokenizer
import importlib.util
spec = importlib.util.spec_from_file_location("o", "/home/louis/PULSAR-ASM/.auto/lfm3/enc_ref/speak_hf_oracle.py")
src = open(spec.origin).read(); i = src.index("PROMPTS = ["); j = src.index("]\n", i) + 2
ns = {}; exec(src[i:j], ns)
tok = AutoTokenizer.from_pretrained("/home/louis/models/lfm3/LFM3-Audio-2B", trust_remote_code=True)
for k, p in enumerate(ns["PROMPTS"]):
    np.array(tok(p).input_ids, np.int32).tofile("%s/p%d.i32" % (sys.argv[1], k))
PY
ssh "$NANO" "mkdir -p $W" && scp -q $W/p*.i32 "$NANO:$W/"
for bb in $BLOBS; do
  ssh "$NANO" "cd /home/picard/lfm3; for k in 0 1 2 3 4; do LFM_SPK_BB=$bb ./lfm3_speak_t . $W/p\$k.i32 interleaved $W/t\$k 150 > $W/t\$k.out 2>&1 || echo \"turn \$k failed\"; done" 2>&1 | grep -v setlocale
  scp -q "$NANO:$W/t?.wav" "$NANO:$W/t?.text.i32" $W/ 2>/dev/null
  $PY - "$W" <<'PY'
import sys, glob, numpy as np, wave
from scipy.signal import resample_poly
W = sys.argv[1]
for f in sorted(glob.glob(W + "/t?.wav")):
    w = wave.open(f); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32); w.close()
    y = np.clip(np.round(resample_poly(x, 2, 3)), -32768, 32767).astype(np.int16)
    o = wave.open(f.replace(".wav", "_16k.wav"), "wb"); o.setnchannels(1); o.setsampwidth(2); o.setframerate(16000)
    o.writeframes(y.tobytes()); o.close()
PY
  scp -q $W/t?_16k.wav "$NANO:$W/"
  ssh "$NANO" "cd /home/picard/lfm3; for k in 0 1 2 3 4; do [ -s $W/t\${k}_16k.wav ] && timeout 900 ./lfm3_asr . $W/t\${k}_16k.wav 96 --out-ids $W/a\$k.i32 > $W/a\$k.out 2>&1; done" 2>&1 | grep -v setlocale
  scp -q "$NANO:$W/a?.i32" $W/ 2>/dev/null
  echo "== $bb"
  $PY - "$W" <<'PY'
import sys, os, re, numpy as np
from transformers import AutoTokenizer
W = sys.argv[1]
tok = AutoTokenizer.from_pretrained("/home/louis/models/lfm3/LFM3-Audio-2B", trust_remote_code=True)
norm = lambda s: re.sub(r"[^a-z0-9' ]", " ", s.lower()).split()
def wer(r, h):
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))
tot_e = tot_n = 0
for k in range(5):
    t, a = "%s/t%d.text.i32" % (W, k), "%s/a%d.i32" % (W, k)
    if not (os.path.exists(t) and os.path.exists(a)): print("  p%d: missing outputs" % k); continue
    ref = tok.decode(np.fromfile(t, np.int32), skip_special_tokens=True)
    hyp = tok.decode(np.fromfile(a, np.int32), skip_special_tokens=True)
    r, h = norm(ref), norm(hyp); e = wer(r, h)
    tot_e += e * len(r); tot_n += len(r)
    print("  p%d WER %.2f | text: %s\n       heard: %s" % (k, e, " ".join(r)[:90], " ".join(h)[:90]))
print("  overall WER %.3f over %d words" % (tot_e / max(1, tot_n), tot_n))
PY
done
