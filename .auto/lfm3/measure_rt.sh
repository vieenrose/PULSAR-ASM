#!/usr/bin/env bash
# measure_rt.sh -- the realtime (ASR_RT) projection instrument, run #308.
#
# Question it answers: with the SHIPPED engine (T=1 prefill, no batched GEMM, no audio
# encoder wired), what does one ASR query cost on the Nano, and how far is that from
# ASR_RT = wall / audio_seconds <= 1.00?
#
# Workload for a SEC-second speech clip (rates measured, not assumed):
#   NPREF = 12.5 tokens/audio-s   (run #303, measured with HF get_audio_features)
#   NGEN  = 1.141 tok/word x 2.5-3.5 words/s  -> 2.85-4.0 tok/s (run #303)
# Engine wall time comes from the per-step table (prompt steps, then generation steps).
# Two adders are NOT in the engine's wall time and are added from their own measured
# sources, printed separately so each is falsifiable:
#   mel : 55 ms per audio second (S2 gate, .auto/lfm3/checks_mel.py)
#   enc : ENC_MS per audio second (pulsar_arm/asm/bench/bench_enc_gemm.cu; 0 = report
#         the realtime figure without it)
# Run with LFM_TIME=1 as a SEPARATE pass for the phase split: phEnd() synchronizes per
# phase, so it inflates the wall time and must not share a run with the timing pass.
set -uo pipefail
cd "$(dirname "$0")/../.."
NANO="${NANO:-picard@100.84.151.116}"
DIR="${NANO_DIR:-/home/picard/lfm3}"
BLOB="${LFM3_BLOB:-lfm3-audio-2b-q8.lfm3q8}"
BIN="${LFM3_BIN:-lfm3_nanode}"
SRC="pulsar_arm/asm/lfm3_nanode.cu"
SEC="${SEC:-60}"; NGEN="${NGEN:-$((SEC * 4))}"; ENC_MS="${ENC_MS:-0}"; MEL_MS="${MEL_MS:-55}"
NPREF=$(( (SEC * 25 + 1) / 2 ))                       # 12.5 tokens per audio second
STEPS=$((NPREF + NGEN))

# A real-text prompt of NPREF ids (the S1 gate's own fixtures, so the ids are genuine
# tokenizer output and inside the vocab). Content does not change GEMV cost; it keeps
# the embedding gather and any positional effects realistic.
PROMPT=$(SEC="$SEC" NPREF="$NPREF" python3 - <<'PY'
import glob, os, struct
ids = []
for f in sorted(glob.glob('.auto/lfm3/tok_fix/f*.ids')):          # packed uint32 (checks_tok.py)
    d = open(f, 'rb').read()
    ids += list(struct.unpack('<%dI' % (len(d) // 4), d))
npref = int(os.environ['NPREF'])
while len(ids) < npref: ids += ids[:1]
print(','.join(str(i % 124860) for i in ids[:npref]))
PY
)
[ -n "$PROMPT" ] || { echo "prompt build failed"; exit 2; }

scp -o ConnectTimeout=20 -q "$SRC" "$NANO:$DIR/" 2>/dev/null || { echo "scp failed (board down?)"; exit 2; }
BOUT=$(ssh -o ConnectTimeout=25 -o BatchMode=yes "$NANO" \
  "cd $DIR && rm -f $BIN.new && export PATH=/usr/local/cuda/bin:\$PATH && nvcc -gencode arch=compute_53,code=sm_53 -O3 -fmad=false -Xptxas -O3,-dlcm=ca -o $BIN.new $(basename $SRC) -lz 2>&1; test -x $BIN.new && mv -f $BIN.new $BIN && echo BUILT_OK" 2>/dev/null | grep -vi setlocale)
echo "$BOUT" | grep -E "error:" | head -12
echo "$BOUT" | grep -q BUILT_OK || { echo "BUILD FAILED (refusing to measure a stale binary)"; exit 1; }

run_pass() {                                          # $1 = extra env, $2 = tag
  ssh -o ConnectTimeout=25 -o BatchMode=yes "$NANO" \
    "cd $DIR && timeout 3600 env $1 ./$BIN $BLOB --steps $STEPS --prompt $PROMPT 2>&1" 2>/dev/null | grep -vi setlocale
}
echo "== timing pass: SEC=$SEC  NPREF=$NPREF (prompt steps)  NGEN=$NGEN (generation steps)"
OUT=$(run_pass "" timing)
echo "$OUT" | grep -E '^ *-?[0-9]+ +[0-9-]+ +[0-9.]+ +[0-9.]+' > .auto/lfm3/.auto/rt_steps.txt
test -s .auto/lfm3/.auto/rt_steps.txt || { echo "MEASURE FAILED (no step lines)"; echo "$OUT" | tail -6; exit 1; }
SEC="$SEC" NPREF="$NPREF" NGEN="$NGEN" ENC_MS="$ENC_MS" MEL_MS="$MEL_MS" \
  python3 - .auto/lfm3/.auto/rt_steps.txt <<'PY'
import os, sys
rows = []
for ln in open(sys.argv[1]):
    p = ln.split()
    if len(p) == 4: rows.append((int(p[0]), int(p[1]), float(p[2]), float(p[3])))
rows = [r for r in rows if r[0] >= 0]
npref, ngen = int(os.environ['NPREF']), int(os.environ['NGEN'])
sec = float(os.environ['SEC']); enc = float(os.environ['ENC_MS']); mel = float(os.environ['MEL_MS'])
pre = [r for r in rows if r[0] < npref]; dec = [r for r in rows if npref <= r[0] < npref + ngen]
assert len(pre) == npref and len(dec) == ngen, (len(pre), npref, len(dec), ngen, len(rows))
sp = sum(r[2] for r in pre); sd = sum(r[2] for r in dec)
def mean(v): return sum(v) / len(v) if v else 0.0
last = [r[2] for r in pre][-100:]
print("prefill  %8.0f ms  mean %6.2f ms/tok   last-100 mean %6.2f ms/tok (marginal at ctx %d)"
      % (sp, sp / npref, mean(last), pre[-1][0]))
print("decode   %8.0f ms  mean %6.2f ms/tok   (%.2f tok/s)" % (sd, sd / ngen, 1000.0 * ngen / sd))
print("first-10 prefill steps: %s" % " ".join("%.1f" % r[2] for r in pre[:10]))
gpu = sp + sd
mel_ms = mel * sec
enc_ms = enc * sec
wall = gpu + mel_ms + enc_ms
print("engine GPU     %8.0f ms" % gpu)
print("mel (S2 gate)  %8.0f ms  (%.0f ms/audio-s)" % (mel_ms, mel))
print("encoder        %8.0f ms  (%.0f ms/audio-s, ENC_MS)" % (enc_ms, enc))
print("METRIC rt_audio_s=%.0f rt_gpu_ms=%.0f rt_wall_ms=%.0f rt_prefill_ms=%.0f rt_decode_ms=%.0f"
      % (sec, gpu, wall, sp, sd))
print("METRIC ASR_RT_gpu=%.3f ASR_RT_wall=%.3f   (target <= 1.00)" % (gpu / (sec * 1000), wall / (sec * 1000)))
print("ASR_RT excluding the encoder: %.3f ; with the fp32 encoder (0.153 s/audio-s): %.3f ; with the batched half2 encoder (0.081): %.3f"
      % ((gpu + mel_ms) / (sec * 1000), (gpu + mel_ms + 153 * sec) / (sec * 1000),
         (gpu + mel_ms + 81 * sec) / (sec * 1000)))
PY
echo "== phase-split pass (LFM_TIME=1: phEnd() syncs per phase, read the split, not the totals)"
OUT2=$(run_pass "LFM_TIME=1" phases)
echo "$OUT2" | grep -E "PHASES|steady-state" | tail -3
