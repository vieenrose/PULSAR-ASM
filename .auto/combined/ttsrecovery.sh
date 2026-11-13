#!/bin/bash
# bash ttsrecovery.sh  (on the board: /home/picard/primetts)
#
# A bad request must not kill the resident TTS. Before this change the serve loop did
# `if (sndr < 0.0) break;`, so one malformed case dir took the speech stage down for every later
# turn of the session -- the only resident stage with no recovery (the encoder answers rc=3 and
# the client falls back, #272; the backbone refuses and keeps serving, #282). Also fixed: a case
# that failed to load partway leaked the feeds already loaded (they were published to the caller
# only after all seven loaded).
#
# Checks, in order: bad path, good case (byte-identical to a one-shot reference made BEFORE the
# service boots), empty case dir, good case again, then 3 bad in a row -- which must exit rather
# than spin.
cd /home/picard/primetts || exit 1
I=/tmp/tr.pt.in; L=/tmp/tr.pt.log
pkill -x pt_v21_nano 2>/dev/null; sleep 1; rm -f $I $L; mkfifo $I; : > $L
rm -f /tmp/tr.ref.npy
./build/pt_v21_nano --container build/v21.ptk --case tests/golden_v21/T81 --out /tmp/tr.ref.npy > /tmp/tr.one.log 2>&1
[ -s /tmp/tr.ref.npy ] || { echo "REF_FAIL (one-shot)"; tail -3 /tmp/tr.one.log; exit 1; }
./build/pt_v21_nano --container build/v21.ptk --serve-fifo $I > $L 2>&1 & PS=$!
i=0; while [ $i -lt 120 ]; do grep -q "PT_AWAIT 0" $L && break; sleep 0.5; i=$((i+1)); done
mkdir -p /tmp/tr.emptydir; rm -f /tmp/tr.emptydir/*.npy
req() {  # $1 = case dir, $2 = label
  local n; n=$(grep -c PT_DONE $L 2>/dev/null); [ -n "$n" ] || n=0
  rm -f /tmp/tr.o.npy
  ( timeout 90 tail -n +1 -f $L 2>/dev/null | grep -m1 -q "^PT_DONE $n " ) & local w=$!
  timeout 30 sh -c "echo '$1 /tmp/tr.o.npy' > $I" 2>/dev/null
  wait $w
  local line; line=$(grep "^PT_DONE $n " $L | tail -1)
  local alive=yes; kill -0 $PS 2>/dev/null || alive=no
  local bytes=-
  if [ -s /tmp/tr.o.npy ]; then cmp -s /tmp/tr.o.npy /tmp/tr.ref.npy && bytes=BYTES_OK || bytes=BYTES_FAIL; fi
  echo "REQ $2 $line bytes=$bytes alive=$alive"
}
req /nonexistent/case                       badpath
req tests/golden_v21/T81                    good_after_bad
req /tmp/tr.emptydir                        emptydir
req tests/golden_v21/T81                    good_again
req /nonexistent/a                          bad1
req /nonexistent/b                          bad2
req /nonexistent/c                          bad3
sleep 1
kill -0 $PS 2>/dev/null && echo "SERVICE_STILL_UP (expected exit after 3 consecutive)" || echo "SERVICE_EXITED_AS_EXPECTED"
grep "consecutive unloadable" $L | sed 's/^/   /'
pkill -x pt_v21_nano 2>/dev/null; pkill -f "tail -n +1 -f $L" 2>/dev/null; rm -f $I
