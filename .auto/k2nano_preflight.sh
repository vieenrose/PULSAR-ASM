#!/bin/bash
# Preflight for the K2 Nano GPU loop. The 2026-10-14 incident: the Nano rebooted,
# /tmp (tmpfs) was wiped, and the loop's engine core k2_cuda.S existed ONLY in
# /tmp on both machines. This script checks every asset the bench needs, in ~2s,
# and prints the exact restore command for anything missing. Run it before
# `bash .auto/measure.sh` (a wasted bench costs 9 minutes and a board cooling cycle).
set -u
PEER=picard@picard-desktop
REPO=/home/user/PULSAR-ASM
SRC=${SRC:-/tmp/k2_cuda.c}
BLOB_MD5=1b026978510f9111bb0a6a6607f3c50d   # spark:~/k2/k2h_09_q4.blob, 748976168 B
fail=0
say() { printf '%-22s %s\n' "$1" "$2"; }

# 1) local engine core asm (bench pipes /tmp/k2_cuda.S to the Nano)
if [ -f /tmp/k2_cuda.S ]; then
  m=$(md5sum /tmp/k2_cuda.S | cut -d' ' -f1)
  r=$(md5sum "$REPO/.auto/rescue/k2_cuda.S" | cut -d' ' -f1)
  if [ "$m" = "$r" ]; then say "k2_cuda.S (local)" "OK $m"
  else say "k2_cuda.S (local)" "MD5 MISMATCH $m (repo $r)"; say "" "restore: cp $REPO/.auto/rescue/k2_cuda.S /tmp/k2_cuda.S"; fail=1; fi
else
  say "k2_cuda.S (local)" "MISSING"
  say "" "restore: cp $REPO/.auto/rescue/k2_cuda.S /tmp/k2_cuda.S"; fail=1
fi

# 2) local staged engine source
if [ -f "$SRC" ]; then say "engine src ($SRC)" "OK $(wc -c <"$SRC") B"
else say "engine src ($SRC)" "MISSING"; say "" "restore: cp $REPO/pulsar_arm/k2horizon/gpu_v43/k2_gexp_engine_v43.c $SRC"; fail=1; fi

# 3) local CPU-leg asm (k2_base)
if [ -f "$REPO/pulsar_arm/k2horizon/k2_core_a57.S" ]; then
  say "k2_core_a57.S" "OK"
else say "k2_core_a57.S" "MISSING (git checkout)"; fail=1; fi

# 4) peer side: blob + binaries
R=$(timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=10 "$PEER" \
  'for f in /tmp/k2h_09_q4.blob /tmp/k2_base /tmp/k2_exp.S; do [ -f "$f" ] && printf "%s=%s " "$(basename $f)" "$(md5sum $f | cut -d" " -f1)" || printf "%s=MISSING " "$(basename $f)"; done; echo' 2>/dev/null)
if [ -z "$R" ]; then say "peer ssh" "UNREACHABLE"; fail=1; else
  say "peer assets" "$R"
  case "$R" in *"k2h_09_q4.blob=$BLOB_MD5"*) say "blob md5" "OK";;
    *) say "blob" "MISSING/WRONG (want $BLOB_MD5)"
       say "" "restore: scp -q luigi@spark:~/k2/k2h_09_q4.blob $PEER:/tmp/"; fail=1;; esac
  case "$R" in *k2_base=MISSING*) say "k2_base" "MISSING (bench builds it: as k2_base.S + gcc -static -nostartfiles)";; esac
fi

[ $fail = 0 ] && echo "PREFLIGHT OK" || echo "PREFLIGHT FAILED ($fail)"
exit $fail
