#!/usr/bin/env python3
"""Spool-directory nvprof server for the RE-USE Jetson Nano port.

WHY: on L4T, nvprof needs root (ERR_NVGPUCTRPERM). This lets a non-root user
(pesi) request profiles of ONE fixed binary without a general sudo grant.

SECURITY MODEL - read before running:
  * The profiled binary runs AS ROOT (CUPTI opens the counters from inside the
    profiled process; that cannot be avoided). Whoever can write
    BINARY can therefore run code as root while this server is up. This
    server narrows and AUDITS that; it does not remove it.
  * No network: requests are JSON files in a spool directory.
  * Only BINARY is ever run, with fixed inputs; requests choose only the sample
    rate, an nvprof mode from a fixed list, and REUSE_* tuning switches.
  * Every run copies the binary into a root-owned directory first and logs its
    sha256, so you can audit exactly what ran as root.
  * REUSE_DUMP_* is rejected (root must not write files to arbitrary paths).
  * One run at a time, hard timeout, results chowned back to the requester.

INSTALL (as root, so the requesting user cannot edit the script):
  sudo install -o root -g root -m 755 nvprof_server.py /usr/local/sbin/reuse-nvprof-server
  sudo /usr/local/sbin/reuse-nvprof-server            # Ctrl-C to stop
STOP it whenever you are done; nothing runs as root while it is stopped.

REQUEST (written by the user into SPOOL/req/<id>.json):
  {"rate": 8000, "mode": "summary", "env": {"REUSE_OP": "4"}}
RESULT: SPOOL/out/<id>.txt (nvprof output) and SPOOL/out/<id>.status
"""
import hashlib
import json
import os
import pwd
import re
import shutil
import subprocess
import sys
import time

REPO = "/dev/shm/rp"
BINARY = REPO + "/engine/reuse_nano_gpu"
NVPROF = "/usr/local/cuda-10.2/bin/nvprof"
SPOOL = "/dev/shm/reuse_nvprof"
# Root-only (chmod 700) but on TMPFS, not the eMMC: this device's root fs is
# ~98% full, and each snapshot is a ~11 MB copy of the binary.
STORE = SPOOL + "/runs"                     # copies + audit log, root-only
USER = "pesi"
TIMEOUT_S = 1200
MAX_ENV = 16

RATES = {
    8000:  {"ws": "640",  "wav": "tests/golden/fixtures/mic_test2_8000.wav",
            "mag": "tests/golden/ref_out_8000/in_mag.raw",
            "pha": "tests/golden/ref_out_8000/in_pha.raw"},
    16000: {"ws": "1280", "wav": "tests/golden/fixtures/mic_test2_16000.wav",
            "mag": "tests/golden/ref_out_16000/in_mag.raw",
            "pha": "tests/golden/ref_out_16000/in_pha.raw"},
}
# nvprof argument sets a request may pick from - never free-form.
MODES = {
    "summary": ["--print-gpu-summary"],
    "trace":   ["--print-gpu-trace"],
    "api":     ["--print-api-summary"],
}
ENV_KEY = re.compile(r"^REUSE_[A-Z0-9_]{1,40}$")
ENV_VAL = re.compile(r"^[A-Za-z0-9_.\-]{0,32}$")
ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(STORE, "audit.log"), "a") as f:
        f.write(line + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def give_to_user(path, uid, gid):
    os.chown(path, uid, gid)
    os.chmod(path, 0o644)


def validate(req):
    if not isinstance(req, dict):
        raise ValueError("request must be a JSON object")
    rate = req.get("rate", 8000)
    if rate not in RATES:
        raise ValueError("rate must be one of %s" % sorted(RATES))
    mode = req.get("mode", "summary")
    if mode not in MODES:
        raise ValueError("mode must be one of %s" % sorted(MODES))
    reps = req.get("reps", 2)
    if not isinstance(reps, int) or not 1 <= reps <= 5:
        raise ValueError("reps must be an int in 1..5")
    env = req.get("env", {})
    if not isinstance(env, dict) or len(env) > MAX_ENV:
        raise ValueError("env must be an object with <= %d keys" % MAX_ENV)
    clean = {}
    for k, v in env.items():
        v = str(v)
        if not ENV_KEY.match(k) or not ENV_VAL.match(v):
            raise ValueError("rejected env %r=%r" % (k, v))
        if k.startswith("REUSE_DUMP") or k in ("REUSE_MAG", "REUSE_PHA"):
            raise ValueError("env %s is not allowed (server-controlled or writes files)" % k)
        clean[k] = v
    return rate, mode, reps, clean


def run_one(name, uid, gid):
    rid = name[:-len(".json")]
    req_path = os.path.join(SPOOL, "req", name)
    out_txt = os.path.join(SPOOL, "out", rid + ".txt")
    out_st = os.path.join(SPOOL, "out", rid + ".status")
    try:
        if not ID.match(rid):
            raise ValueError("bad request id")
        with open(req_path) as f:
            req = json.load(f)
        rate, mode, reps, env = validate(req)

        # Snapshot the binary into root-owned storage and run THAT copy, so the
        # file cannot change between the hash and the exec.
        if not os.path.isfile(BINARY):
            raise ValueError("binary not found: " + BINARY)
        digest = sha256(BINARY)
        snap = os.path.join(STORE, digest[:16] + ".bin")
        if not os.path.exists(snap):
            shutil.copy2(BINARY, snap)
            os.chown(snap, 0, 0)
            os.chmod(snap, 0o700)
        if sha256(snap) != digest:
            raise ValueError("snapshot hash mismatch")
        log("RUN id=%s sha256=%s rate=%d mode=%s reps=%d env=%s"
            % (rid, digest, rate, mode, reps, json.dumps(env, sort_keys=True)))

        r = RATES[rate]
        run_env = {"PATH": "/usr/bin:/bin", "HOME": "/root",
                   "LD_LIBRARY_PATH": "/usr/local/cuda-10.2/lib64",
                   "CUDA_LAUNCH_BLOCKING": "1", "REUSE_GPU_ISTFT": "1",
                   "REUSE_WS_MB": r["ws"], "REUSE_MAG": r["mag"], "REUSE_PHA": r["pha"]}
        run_env.update(env)
        cmd = [NVPROF, "--csv"] + MODES[mode] + [
            snap, "tests/golden/reuse.blob", r["wav"], "/dev/null", str(reps)]
        t0 = time.time()
        p = subprocess.run(cmd, cwd=REPO, env=run_env, timeout=TIMEOUT_S,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with open(out_txt, "wb") as f:
            f.write(p.stdout)
        give_to_user(out_txt, uid, gid)
        status = {"ok": p.returncode == 0, "returncode": p.returncode,
                  "sha256": digest, "seconds": round(time.time() - t0, 1)}
        log("DONE id=%s rc=%d %.1fs" % (rid, p.returncode, status["seconds"]))
    except subprocess.TimeoutExpired:
        status = {"ok": False, "error": "timeout after %ds" % TIMEOUT_S}
        log("TIMEOUT id=%s" % rid)
    except Exception as e:  # report to the requester, keep serving
        status = {"ok": False, "error": str(e)}
        log("REJECT id=%s: %s" % (rid, e))
    with open(out_st, "w") as f:
        json.dump(status, f)
    give_to_user(out_st, uid, gid)
    os.remove(req_path)


def main():
    if os.geteuid() != 0:
        sys.exit("must run as root (sudo); see the docstring for the security model")
    me = os.path.realpath(__file__)
    st = os.stat(me)
    if st.st_uid != 0 or (st.st_mode & 0o022):
        sys.exit("refusing to run: %s must be owned by root and not group/world-"
                 "writable (install it with: sudo install -o root -g root -m 755 ...)" % me)
    pw = pwd.getpwnam(USER)
    os.makedirs(SPOOL, mode=0o755, exist_ok=True)
    os.makedirs(STORE, mode=0o700, exist_ok=True)
    for d in ("req", "out"):
        os.makedirs(os.path.join(SPOOL, d), exist_ok=True)
    os.chown(STORE, 0, 0)
    os.chmod(STORE, 0o700)
    os.chown(SPOOL, 0, 0)
    os.chmod(SPOOL, 0o755)
    os.chown(os.path.join(SPOOL, "req"), pw.pw_uid, pw.pw_gid)
    os.chmod(os.path.join(SPOOL, "req"), 0o700)
    os.chown(os.path.join(SPOOL, "out"), 0, 0)
    os.chmod(os.path.join(SPOOL, "out"), 0o755)
    log("server up: binary=%s spool=%s user=%s" % (BINARY, SPOOL, USER))
    try:
        while True:
            reqs = sorted(n for n in os.listdir(os.path.join(SPOOL, "req"))
                          if n.endswith(".json"))
            for name in reqs:
                run_one(name, pw.pw_uid, pw.pw_gid)
            time.sleep(1.0)
    except KeyboardInterrupt:
        log("server stopped")


if __name__ == "__main__":
    main()
