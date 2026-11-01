#!/usr/bin/env bash
# checks_enc.sh -- S3 audio-encoder gate: the GPU encoder must reproduce the C reference
# (pulsar_arm/asm/lfm3_enc.c, which matches the HF spec to 3.3e-6 rms, run #305) on
# every stage that is ported, on the Nano, from a fresh build.
#
#   bash .auto/lfm3/checks_enc.sh              # fp32 arms: tight gate vs the C reference
#   LFM_ENC_H2=1 bash .auto/lfm3/checks_enc.sh # batched half2 arms: tolerance gate only
#
# Two comparisons per stage, both printed:
#   GPU vs C reference  -- the correctness gate. fp32 arms must be <= 1e-4 rms rel;
#                          the half2 arms are graded on the end-to-end transcript gate
#                          (run #309: bf16 ACTIVATION rounding, not the fp16 products,
#                          sets a ~0.28 per-cell rms on this data, so bit-exactness is
#                          not the standard for them).
#   GPU vs HF           -- reported, never a gate: the blob is Q8 for some tensors
#                          (lin_out, mlp.1, mlp.3, all the layer linears) and f32 for
#                          others (lin_mid, lin_end, convs, norms), so an HF gap is the
#                          quantization gap measured in #305, amplified by cancellation.
set -uo pipefail
cd "$(dirname "$0")/../.."
NANO="${NANO:-picard@100.84.151.116}"
DIR="${NANO_DIR:-/home/picard/lfm3}"
BLOB="${LFM3_ENC_BLOB:-lfm3-audio-2b-enc.lfm3q8}"
Tm="${Tm:-400}"
TOK=$(( (Tm + 7) / 8 ))
H2="${LFM_ENC_H2:-0}"
REF=".auto/lfm3/enc_ref"

scp -o ConnectTimeout=20 -q pulsar_arm/asm/lfm3_enc_gpu.cu pulsar_arm/asm/lfm3_enc.c pulsar_arm/asm/lfm3_nanode.cu \
    .auto/lfm3/enc_ref/cmp_enc.py "$NANO:$DIR/" 2>/dev/null || { echo "scp failed (board down?)"; exit 2; }
BOUT=$(ssh -o ConnectTimeout=25 -o BatchMode=yes "$NANO" \
  "cd $DIR && rm -f lfm3_enc_gpu.new && export PATH=/usr/local/cuda/bin:\$PATH && nvcc -std=c++11 -gencode arch=compute_53,code=sm_53 -O3 -fmad=false -Xptxas -O3,-dlcm=ca -o lfm3_enc_gpu.new lfm3_enc_gpu.cu -lz 2>&1; test -x lfm3_enc_gpu.new && mv -f lfm3_enc_gpu.new lfm3_enc_gpu && gcc -O2 -std=c99 -Wall -Wextra -o lfm3_enc lfm3_enc.c -lm && echo BUILT_OK" 2>/dev/null | grep -vi setlocale)
echo "$BOUT" | grep -E "error:|warning:" | head -12
echo "$BOUT" | grep -q BUILT_OK || { echo "BUILD FAILED (refusing to measure a stale binary)"; exit 1; }

# The comparator runs on the Nano (numpy 1.13.3 is installed there), so no dump has to
# cross the wire for the gate itself; the GPU dumps are copied back for the HF report.
ssh -o ConnectTimeout=25 -o BatchMode=yes "$NANO" \
  "cd $DIR && rm -rf encgpu encgpu2 encC && mkdir -p encgpu encgpu2 encC && \
   LFM_ENC_H2=$H2 LFM_ENC_PROBE_LAYER=2 timeout 900 ./lfm3_enc_gpu $BLOB input_128xT_f32.bin $Tm encgpu 2>&1 | grep -viE '^load |RSS|host at|blob:|device free'; \
   LFM_ENC_H2=$H2 LFM_ENC_PROBE_LAYER=2 timeout 900 ./lfm3_enc_gpu $BLOB input_128xT_f32.bin $Tm encgpu2 2>&1 | grep -viE '^load |RSS|host at|blob:|device free' | tail -2; \
   LFM_ENC_PROBE_LAYER=2 timeout 3600 ./lfm3_enc $BLOB input_128xT_f32.bin $Tm encC > /dev/null 2>&1; \
   LFM_ENC_H2=$H2 python3 cmp_enc.py $TOK" 2>&1 | grep -vi setlocale | tee .auto/lfm3/.auto/enc_gate.txt
GATE=$(grep -oE 'ENC_GATE (PASS|FAIL[^ ]*)' .auto/lfm3/.auto/enc_gate.txt | tail -1)
echo "== GPU vs HF (reported on the host, not gated: the blob is part-Q8)"
mkdir -p "$REF/gpu"
for f in ds_out enc_out proj_out layer_00 layer_08 layer_16; do
  scp -o ConnectTimeout=20 -q "$NANO:$DIR/encgpu/$f.bin" "$REF/gpu/" 2>/dev/null
done
"${PY:-/home/louis/venvs/lfm3/bin/python}" - "$REF" "$TOK" <<'PY'
import numpy as np, os, sys
ref, TOK = sys.argv[1], int(sys.argv[2])
for name, cols in [("ds_out", 768), ("enc_out", 768), ("proj_out", 2048)]:
    gp, hp = os.path.join(ref, "gpu", name + ".bin"), os.path.join(ref, "ref", name + ".npy")
    if not (os.path.exists(gp) and os.path.exists(hp)): continue
    a = np.fromfile(gp, dtype=np.float32).reshape(TOK, cols)
    h = np.load(hp).astype(np.float32).reshape(TOK, cols)
    d = np.abs(a - h).astype(np.float64); m = np.abs(h) > 1e-6
    print("  %-10s rms rel %.3e vs HF  max abs %.3e  rms(GPU) %.5f rms(HF) %.5f   <- the Q8 quantization gap (#305)"
          % (name, np.sqrt(((d[m] / np.abs(h[m].astype(np.float64))) ** 2).mean()), d.max(),
             np.sqrt((a ** 2).mean()), np.sqrt((h ** 2).mean())))
PY
echo "$GATE"
case "$GATE" in *PASS) exit 0;; *) exit 1;; esac
