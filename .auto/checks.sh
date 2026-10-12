#!/bin/bash
# Accuracy gate for K2-0.9B Nano loop: run the LAST-BUILT /tmp/k2_exp on the
# held-out ZH prompt (26 ids, gen 100, threads 1) on the Nano; stream must be
# bit-identical to .auto/k2nano_golden_zh.txt. (Phone/27B gate preserved in
# checks_bonsai_phone.sh.)
set -u
PEER=picard@picard-desktop
ZH="64018 2985 200 2226 294 3957 29181 316 43658 10627 13 1546 294 3957 8045 13 1026 28045 15 64019 64018 612 10102 200 64029 200"
GOLDEN=/home/user/PULSAR-ASM/.auto/k2nano_golden_zh.txt
TMP=$(mktemp -d); trap 'rm -rf $TMP' EXIT
timeout 300 ssh -o BatchMode=yes $PEER \
  "cd /tmp && ./k2_exp k2h_09_q4.blob $ZH --gen 100 --threads 1 --batch 8 > /tmp/zh_check.txt 2>&1; echo rc=\$?" \
  > $TMP/rc.txt 2>&1
grep -q "rc=0" $TMP/rc.txt || { echo "CHECKS FAIL: run rc!=0"; cat $TMP/rc.txt; exit 4; }
timeout 60 ssh -o BatchMode=yes $PEER 'grep "^step " /tmp/zh_check.txt' > $TMP/steps.txt
if [ "$(wc -l < $TMP/steps.txt)" != "114" ]; then echo "CHECKS FAIL: step count"; exit 4; fi
diff $TMP/steps.txt $GOLDEN > $TMP/diff.txt 2>&1 || { echo "CHECKS FAIL: ZH stream differs"; head -5 $TMP/diff.txt; exit 4; }
echo "CHECKS OK: ZH 114/114 bit-exact"
