#!/usr/bin/env bash
# LFM3-Audio-2B on the Jetson Nano (sm_53, CUDA 10.2, 4 GB unified).
#
# Metric: decode ms/token (LOWER IS BETTER) over a fixed prompt, plus the
# tokens/s it implies, plus peak RSS. Mirrors .auto/measure.sh for the Bonsai
# Nano objective (sync -> build -> fixed run -> METRIC lines).
#
# NOTE ON THE BOARD STATE: three lfm3_conv_test processes from the previous
# session (10-06-2026, the warp-shuffle conv deadlock) are stuck in state R
# inside the CUDA driver. kill -9 does not reap them and the box cannot be
# rebooted (systemd shutdown waits on them), so 3 of the Nano's 4 A57 cores
# are permanently busy. Absolute ms/token measured today is inflated by that;
# same-session A/B ratios remain valid. See .auto/ideas.md.
set -uo pipefail
cd "$(dirname "$0")/../.."
NANO="${NANO:-picard@100.84.151.116}"
DIR="${NANO_DIR:-/home/picard/lfm3}"
BLOB="${LFM3_BLOB:-lfm3-audio-2b-q8.lfm3q8}"
BIN="${LFM3_BIN:-lfm3_nanode}"
SRC="pulsar_arm/asm/lfm3_nanode.cu"
# Fixed prompt: the first 24 ids of golden_greedy_A.json (prompt A of the
# certification set), so every run in this session is the same job.
PROMPT="$(python3 -c "import json;print(','.join(map(str,json.load(open('.auto/lfm3/golden_greedy_A.json'))['prompt_ids'])))")"
N="${STEPS:-48}"
EXTRA="${PQ1_EXTRA:-}"

scp -o ConnectTimeout=20 -q "$SRC" "$NANO:$DIR/" 2>/dev/null || { echo "scp failed (board down?)"; exit 2; }

# Build (the flags are the same set build_lfm3.sh uses -- keep them in sync).
ssh -o ConnectTimeout=25 -o BatchMode=yes "$NANO" \
  "cd $DIR && export PATH=/usr/local/cuda/bin:\$PATH && nvcc -gencode arch=compute_53,code=sm_53 -O3 -fmad=false -Xptxas -O3,-dlcm=ca -o $BIN lfm3_nanode.cu -lz 2>&1 | head -20" 2>/dev/null | grep -vi setlocale
ssh -o ConnectTimeout=25 -o BatchMode=yes "$NANO" "test -x $DIR/$BIN && echo BUILT || echo BUILD_FAILED" 2>/dev/null | grep -vi setlocale | grep -q BUILT || { echo "BUILD FAILED"; exit 1; }

# Cold page cache, so every sample sees the same first-touch cost (the plan's
# 740 MB/s eMMC finding is why this matters for --emb-host).
ssh -o ConnectTimeout=25 -o BatchMode=yes "$NANO" "sync; echo 31415 | sudo -S sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; true" 2>/dev/null

OUT=$(ssh -o ConnectTimeout=25 -o BatchMode=yes "$NANO" \
  "cd $DIR && timeout 1800 env $EXTRA ./$BIN $BLOB --steps $N --prompt $PROMPT 2>&1" 2>/dev/null | grep -vi setlocale)
echo "$OUT" | grep -E '^ *[0-9]+ +[0-9-]+ +[0-9.]+ +[0-9.]+' | tail -1 > .auto/last_step.txt
echo "$OUT" | tail -3
# The engine's own line, parsed exactly:
#   steady-state 141.13 ms/token -> 7.09 tok/s over 46 steps (...)
# It excludes the first steps (embedding H2D, graph-free warmup), the decode
# -rate convention this repo's other harnesses use. RSS comes from the
# engine's own VmHWM print: the Nano's login shell is dash, whose `time` has
# no -f, and /usr/bin/time is absent entirely.
SUM=$(echo "$OUT" | grep -oE 'steady-state [0-9.]+ ms/token -> [0-9.]+ tok/s' | tail -1)
RSS=$(echo "$OUT" | grep -oE 'RSS_KB=[0-9]+' | tail -1 | sed 's/RSS_KB=//')
MT=$(echo "$SUM" | sed -n 's|steady-state \([0-9.]*\) ms/token.*|\1|p')
TPS=$(echo "$SUM" | sed -n 's|.* -> \([0-9.]*\) tok/s|\1|p')
[ -z "$MT" ] && { echo "MEASURE FAILED (no SUMMARY)"; exit 1; }
echo "METRIC decode_ms_per_tok=$MT"
echo "METRIC decode_tps=$TPS"
[ -n "$RSS" ] && echo "METRIC peak_rss_kb=$RSS"
echo "SUMMARY ms/token=$MT tokens/s=$TPS"
