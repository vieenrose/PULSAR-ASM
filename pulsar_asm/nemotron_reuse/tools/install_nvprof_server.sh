#!/bin/sh
# Install and start the RE-USE nvprof spool server on the Jetson Nano.
#
#   scp nvprof_server.py install_nvprof_server.sh pesi@100.70.245.36:/tmp/
#   ssh pesi@100.70.245.36 'sudo sh /tmp/install_nvprof_server.sh'
#
# READ THIS FIRST: the profiled binary runs AS ROOT (CUPTI opens the GPU
# counters from inside the profiled process, and on this L4T build there is no
# /sys/module/nvgpu/parameters entry to relax that). The server pins down WHAT
# gets run (one fixed binary, one of three nvprof modes, only REUSE_* env vars,
# no REUSE_DUMP_*), snapshots the binary as root and logs its sha256, and does
# not touch the network. But while it is running, anyone who can write
# /dev/shm/rp/engine/reuse_nano_gpu can get that file run as root - so treat
# "server running" as "pesi effectively has root on this box" and stop it
# (Ctrl-C) when you are done.
set -e

SRC_PY=${SRC_PY:-/tmp/nvprof_server.py}
DEST=/usr/local/sbin/reuse-nvprof-server
SPOOL=/dev/shm/reuse_nvprof

if [ "$(id -u)" != "0" ]; then
    echo "run me with sudo (see the header for the security model)" >&2
    exit 1
fi

if [ ! -f "$SRC_PY" ]; then
    echo "missing $SRC_PY (scp nvprof_server.py to /tmp first, or set SRC_PY=)" >&2
    exit 1
fi

# Root-owned and not writable by the requesting user, so it cannot be swapped
# out between here and execution (the server re-checks this at startup).
install -o root -g root -m 755 "$SRC_PY" "$DEST"

# Spool on tmpfs: this device's root fs is ~98% full and each binary snapshot
# is ~11 MB. The runs/ subdir stays root-only.
install -d -o root -g root -m 755 "$SPOOL"
install -d -o root -g root -m 700 "$SPOOL/runs"
install -d -o pesi -g pesi -m 700 "$SPOOL/req"
install -d -o root -g root -m 755 "$SPOOL/out"

echo
echo "installed: $DEST"
echo "request format (written by the user, e.g. by the agent):"
echo "    $SPOOL/req/<id>.json"
echo '    {"rate": 8000, "mode": "summary"}                      # fixture, 8 kHz'
echo '    {"rate": 16000, "mode": "trace", "env": {"REUSE_OP": "4"}}'
echo "results:    $SPOOL/out/<id>.txt   and  <id>.status"
echo "audit log:  $SPOOL/runs/audit.log   (sha256 of every binary run as root)"
echo
echo "starting the server in the foreground - Ctrl-C to stop it."
echo
exec "$DEST"
