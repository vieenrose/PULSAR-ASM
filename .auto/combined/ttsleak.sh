#!/bin/bash
# bash ttsleak.sh [N]  (on the board: /home/picard/primetts)
#
# The partial-load leak, measured. run_case() published its feeds to the caller only after all
# seven had loaded, so a case that failed at, say, lang.npy returned with x and tone allocated and
# unowned: the next request's free() released the PREVIOUS request's pointers and the partial
# feeds leaked -- once per bad request, forever, in a process that outlives its clients. Here a
# case dir is built from T81 with lang.npy removed (loads x, tone, then fails), requested N times,
# and the service's VmRSS is read before and after.
#
# Arms: ./build/pt_v21_nano (fixed: feeds published as they load) vs /tmp/pt_old (run_case verbatim
# from git HEAD~1, with only the serve loop's `break` removed so the loop can be driven N times --
# without that removal the pre-fix binary exits on the first bad request and the leak cannot be
# looped). PT_V21_SERVE_MAXFAIL=0 disables the new give-up cap for the same reason.
cd /home/picard/primetts || exit 1
N=${1:-20}
export PT_V21_SERVE_MAXFAIL=0
rm -rf /tmp/tr.part; cp -r tests/golden_v21/T81 /tmp/tr.part; rm -f /tmp/tr.part/lang.npy
echo "case dir has $(ls /tmp/tr.part | wc -l) files (lang.npy removed: fails after 2 feeds load)"
for BIN in ./build/pt_v21_nano /tmp/pt_old; do
  [ -x "$BIN" ] || { echo "SKIP $BIN (not built)"; continue; }
  I=/tmp/tl.pt.in; L=/tmp/tl.pt.log
  pkill -x pt_v21_nano 2>/dev/null; sleep 1; rm -f $I $L; mkfifo $I; : > $L
  TAG=pt_leak_arm
  # exec -a gives the service a distinctive argv[0]. pgrep on the fifo path instead also matches the
  # `timeout sh -c "echo ... > $I"` request writers, whose pid then disappears -- what made the first
  # version of this script fail to read VmRSS at all.
  ( exec -a "$TAG" "$BIN" --container build/v21.ptk --serve-fifo $I > $L 2>&1 ) & PS=$!
  i=0; while [ $i -lt 120 ]; do grep -q "PT_AWAIT 0" $L && break; sleep 0.5; i=$((i+1)); done
  pid=$(pgrep -f "^$TAG --container" | head -1); [ -n "$pid" ] || pid=$PS
  rss() { awk '/VmRSS/{print int($2/1024)}' /proc/$pid/status 2>/dev/null || echo -1; }
  r0=$(rss)
  for k in $(seq 1 $N); do
    n=$(grep -c PT_DONE $L 2>/dev/null); [ -n "$n" ] || n=0
    ( timeout 60 tail -n +1 -f $L 2>/dev/null | grep -m1 -q "^PT_DONE $n " ) & W=$!
    timeout 20 sh -c "echo '/tmp/tr.part /tmp/tl.o.npy' > $I"
    wait $W
  done
  rmid=$(rss)
  n=$(grep -c PT_DONE $L 2>/dev/null); [ -n "$n" ] || n=0
  ( timeout 60 tail -n +1 -f $L 2>/dev/null | grep -m1 -q "^PT_DONE $n " ) & W=$!
  timeout 20 sh -c "echo 'tests/golden_v21/T81 /tmp/tl.o.npy' > $I"   # still alive, still exact?
  wait $W
  r1=$(rss)
  echo "BIN $BIN bad=$N rss ${r0} -> ${rmid} (after the bad ones) -> ${r1} MB, delta $((rmid-r0)); last=$(grep PT_DONE $L | tail -1)"
  kill $PS 2>/dev/null; pkill -f "^$TAG --container" 2>/dev/null; pkill -f "tail -n +1 -f $L" 2>/dev/null; rm -f $I
done
