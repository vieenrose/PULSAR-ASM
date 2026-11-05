#!/usr/bin/env bash
# Nano-scored RTF benchmark for the RE-USE port.
# Primary metric = 8 kHz RTF_total MEASURED ON THE JETSON NANO (iPhO), using the
# deployment protocol in run_bench2.sh (CUDA_LAUNCH_BLOCKING=1, GPU iSTFT, fed
# reference STFT, best of reps 1-2 after warm-up rep 0). 16 kHz is reported as a
# held-out monitor. Accuracy gates still run on the dev host (checks.sh), and the
# Nano's 8 kHz amp output is diffed against the dev Maxwell proxy.
set -euo pipefail
ROOT=/tmp/pulsar-asm/pulsar_asm/nemotron_reuse
AUTO=/tmp/pulsar-asm/.auto
NANO=pesi@100.70.245.36
SSH="ssh -o ConnectTimeout=60 -o ServerAliveInterval=15 $NANO"
NVCC="${NVCC:-/usr/local/cuda-10.2/bin/nvcc}"
cd "$ROOT"

# 1. dev compile check (fast failure)
"$NVCC" -O3 -Xcompiler -fopenmp -arch=sm_61 -Iengine -o engine/reuse_nano_gpu \
  engine/reuse_host_gpu.c engine/reuse_gpu.cu engine/blob.c engine/wav_io.c \
  engine/stft_dft.o engine/istft_dft.o -lm 2>"$AUTO/build.err" \
  || { echo "BUILD FAILED (dev)"; grep error "$AUTO/build.err" | head; exit 1; }

# 2. dev Maxwell-proxy reference output (bit-exact vs Nano historically)
REUSE_FORCE_MAXWELL_TILES=1 REUSE_GPU_ISTFT=1 REUSE_WS_MB=1024 \
  REUSE_MAG=tests/golden/ref_out_8000/in_mag.raw REUSE_PHA=tests/golden/ref_out_8000/in_pha.raw \
  REUSE_DUMP_AMP=/tmp/proxy_amp.raw ./engine/reuse_nano_gpu tests/golden/reuse.blob \
  tests/golden/fixtures/mic_test2_8000.wav /dev/null 1 >/dev/null 2>&1

# 3. ship source to the Nano (eMMC copy + tmpfs work dir) and build in the background
rsync -a --exclude '*.o' --exclude 'reuse_nano_gpu' --exclude '__pycache__' \
  engine/ "$NANO":~/reuse_port/engine/ 2>/dev/null
rsync -a --exclude '*.o' --exclude 'reuse_nano_gpu' --exclude '__pycache__' \
  engine/ "$NANO":/dev/shm/rp/engine/ 2>/dev/null
$SSH 'cd /dev/shm/rp/engine && rm -f reuse_nano_gpu /dev/shm/build.done &&
  ( [ -f stft_dft.o ] || gcc -O2 -fopenmp -x c -c stft_dft.cu -o stft_dft.o -I. ) &&
  ( [ -f istft_dft.o ] || gcc -O2 -fopenmp -x c -c istft_dft.cu -o istft_dft.o -I. ) &&
  nohup sh -c "/usr/local/cuda-10.2/bin/nvcc -O3 -Xcompiler -fopenmp -arch=sm_53 -o reuse_nano_gpu reuse_host_gpu.c reuse_gpu.cu blob.c wav_io.c stft_dft.o istft_dft.o -lm > /dev/shm/build.log 2>&1; echo \$? > /dev/shm/build.done" >/dev/null 2>&1 &' 2>/dev/null
for i in $(seq 1 90); do
  sleep 10
  st=$($SSH 'cat /dev/shm/build.done 2>/dev/null' 2>/dev/null || true)
  [ -n "$st" ] && break
done
[ "${st:-x}" = "0" ] || { echo "BUILD FAILED (nano) status=${st:-timeout}"; $SSH 'grep error /dev/shm/build.log | head' 2>/dev/null; exit 1; }

# 4. foreign-load check, then the benchmark
load=$($SSH 'cut -d" " -f1 /proc/loadavg; grep NvMapMemFree /proc/meminfo' 2>/dev/null | tr '\n' ' ')
echo "nano pre-run: load/nvmap $load"
out=$($SSH 'cd /dev/shm/rp && ./run_bench2.sh 2>&1' 2>/dev/null)
echo "$out" | grep -E '^#####|^phases|^model best|^total'
grep -q 'forward fail\|OOM' <<<"$out" && { echo "NANO RUN FAILED (OOM?)"; exit 1; }

b8=$(sed -n '/8000 Hz/,/16000 Hz/p' <<<"$out")
b16=$(sed -n '/16000 Hz/,$p' <<<"$out")
rtf8=$(grep '^total(' <<<"$b8" | awk '{for(i=1;i<=NF;i++) if($i=="RTF_total") print $(i+1)}')
mod8=$(grep '^model best' <<<"$b8" | awk '{for(i=1;i<=NF;i++) if($i=="RTF_model") print $(i+1)}')
rtf16=$(grep '^total(' <<<"$b16" | awk '{for(i=1;i<=NF;i++) if($i=="RTF_total") print $(i+1)}')
ph=$(grep '^phases' <<<"$b8")
enc=$(grep -oE 'encoder [0-9.]+' <<<"$ph" | awk '{print $2}')
tower=$(grep -oE '30xTFMamba [0-9.]+' <<<"$ph" | awk '{print $2}')
dec=$(grep -oE 'decoders [0-9.]+' <<<"$ph" | awk '{print $2}')
[ -n "$rtf8" ] || { echo "NO RESULT"; exit 1; }

# 5. Nano output must match the dev Maxwell proxy (stdlib only: the harness
#    python may lack numpy)
scp -q "$NANO":/dev/shm/nb_8000_c_amp.raw /tmp/nano_amp.raw \
  || { echo "SCP OF NANO OUTPUT FAILED"; exit 1; }
d=$(python3 -I -c "
from array import array
a=array('f'); b=array('f')
a.frombytes(open('/tmp/proxy_amp.raw','rb').read()); b.frombytes(open('/tmp/nano_amp.raw','rb').read())
print(max(abs(x-y) for x,y in zip(a,b)) if len(a)==len(b) and len(a)>0 else 1e9)") \
  || { echo "OUTPUT DIFF SCRIPT FAILED"; exit 1; }
echo "nano-vs-proxy amp max|d| = $d"
python3 -I -c "import sys; sys.exit(0 if float('$d') < 1e-4 else 1)" \
  || { echo "NANO OUTPUT DIVERGES FROM PROXY"; exit 1; }

echo "METRIC rtf_nano8k=$rtf8"
echo "METRIC rtf_nano8k_model=$mod8"
echo "METRIC rtf_nano16k=$rtf16"
echo "METRIC nano_enc_s=$enc"
echo "METRIC nano_tower_s=$tower"
echo "METRIC nano_dec_s=$dec"
echo "METRIC nano_amp_maxd=$d"
