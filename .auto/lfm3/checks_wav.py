#!/usr/bin/env python3
"""checks_wav.py -- the WAV front of the ASR path, gated two ways.

  1. bit-exactness: lfm3_wav on the Nano must reproduce numpy's decode, bit for bit,
     over fixtures that cover the parser's cases: PCM16 mono/stereo, IEEE float32
     mono/stereo, WAVE_FORMAT_EXTENSIBLE for both, chunks in hostile order (LIST and
     fact before fmt, an odd-size junk chunk with its pad byte), int16 extremes
     (-32768/32767), an empty data chunk, and a RIFF size field that lies.
  2. end-to-end: WAV -> lfm3_wav -> pcm.f32 -> lfm3_mel -> mel.f32 must equal the
     CACHED HF mel of the same fixture at the S2 tolerance. That is what makes the
     reader part of the gated path: a WAV bug shows up as mel drift, not as a green
     front end and a transcript that is subtly wrong.

Refusals are fixtures too: 44.1 kHz, 8 kHz, 24-bit PCM, 3-channel, a truncated data
chunk, a non-RIFF file, and an extensible header with a PCM24 SubFormat. Each must
exit non-zero -- a reader that guesses (assumes 16 kHz, takes channel 0, interpolates)
passes fixture 1 and silently breaks fixture 2.

Fixtures come from the mel gate's own PCM files (.auto/lfm3/mel_fix/*.f32), so no new
audio is introduced: the WAV reader is tested against signals the mel stage already
agrees with HF on.

  ~/venvs/lfm3/bin/python .auto/lfm3/checks_wav.py
"""
import json, os, subprocess, sys, struct

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
NANO = os.environ.get("NANO", "picard@100.84.151.116")
NDIR = os.environ.get("NANO_DIR", "/home/picard/lfm3")
FIXDIR = os.path.join(HERE, "mel_fix")
REFDIR = os.path.join(HERE, "mel_ref")
WORK = os.path.join(HERE, ".auto", "wav_gate")
MELBIN = os.environ.get("MEL_BIN", "/tmp/mel.bin")
MEL_TOL_LIVE = float(os.environ.get("MEL_TOL_LIVE", "1e-5"))
NFFT, HOP, NM = 400, 160, 128

import numpy as np


def wav_bytes(pcm_i16=None, pcm_f32=None, rate=16000, ch=1, bits=16, extensible=False,
              junk=None, lie_about_size=False, truncate=0, float_format=False):
    """Build a RIFF/WAVE from int16 or f32 samples. `junk` is a list of (fourcc, payload)."""
    fmt = 3 if float_format else 1
    assert pcm_i16 is None or pcm_f32 is None
    if pcm_i16 is not None:
        data = pcm_i16.astype("<i2").tobytes()
    else:
        data = pcm_f32.astype("<f4").tobytes()
    if truncate:
        data = data[:len(data) - truncate]
    bps = bits // 8
    ba = ch * bps
    fmt_body = struct.pack("<HHIIHH", fmt, ch, rate, rate * ba, ba, bits)
    if extensible:
        guid = (bytes([3, 0, 0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71])
                if float_format else
                bytes([1, 0, 0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71]))
        fmt_body += struct.pack("<HH", 22, bits) + guid
    chunks = []
    for fourcc, payload in (junk or []):
        pad = b"\x00" if len(payload) & 1 else b""
        chunks.append(fourcc + struct.pack("<I", len(payload)) + payload + pad)
    chunks.append(b"fmt " + struct.pack("<I", len(fmt_body)) + fmt_body)
    chunks.append(b"data" + struct.pack("<I", len(data)) + data)
    body = b"WAVE" + b"".join(chunks)
    size = (len(body) - 8) if not lie_about_size else 1 << 28
    return b"RIFF" + struct.pack("<I", size) + body


def ssh(cmd):
    return subprocess.run(["ssh", "-o", "ConnectTimeout=25", "-o", "BatchMode=yes", NANO, cmd],
                          capture_output=True, text=True)


def build_on_nano():
    for f in ("lfm3_wav.c", "lfm3_mel.c", "lfm3_mel.S"):
        subprocess.run(["scp", "-q", "-o", "ConnectTimeout=20",
                        os.path.join(REPO, "pulsar_arm", "asm", f), "%s:%s/" % (NANO, NDIR)],
                       check=True)
    subprocess.run(["scp", "-q", "-o", "ConnectTimeout=20", MELBIN, "%s:%s/mel.bin" % (NANO, NDIR)],
                   check=True)
    r = ssh("cd %s && as -o lfm3_mel_o lfm3_mel.S 2>&1 && "
            "gcc -O2 -std=c99 -Wall -Wextra -ffp-contract=off -o lfm3_mel lfm3_mel.c lfm3_mel_o -lm 2>&1 && "
            "gcc -O2 -std=c99 -Wall -Wextra -ffp-contract=off -o lfm3_wav lfm3_wav.c 2>&1 && echo BUILT" % NDIR)
    if "BUILT" not in (r.stdout + r.stderr):
        sys.exit("build failed on the Nano -- refusing to measure a stale binary:\n" + r.stdout + r.stderr)
    for w in r.stdout.splitlines():
        if "warning:" in w:
            print("  compiler: " + w)


def main():
    os.makedirs(WORK, exist_ok=True)
    if not os.path.isdir(FIXDIR):
        sys.exit("no mel fixtures (%s); run .auto/lfm3/checks_mel.py first -- this gate "
                 "reuses the signals the mel stage is already validated on" % FIXDIR)
    names = sorted(fn[:-4] for fn in os.listdir(FIXDIR) if fn.endswith(".f32"))
    if not names:
        sys.exit("mel_fix has no .f32 fixtures")
    build_on_nano()

    fixtures = []          # (name, wav bytes, expected f32 array, expect_ok)
    base = np.fromfile(os.path.join(FIXDIR, names[0] + ".f32"), dtype=np.float32)
    short = base[:4000]

    def i16_of(x):
        return np.clip(np.round(x * 32768.0), -32768, 32767).astype("<i2")

    # --- accepted forms -------------------------------------------------------------
    x = i16_of(short)
    fixtures.append(("pcm16_mono", wav_bytes(pcm_i16=x, ch=1, bits=16),
                     x.astype(np.float32) / np.float32(32768.0), True))
    y = i16_of(short[::-1])
    stereo = np.stack([x, y], axis=1).ravel()
    exp = ((x.astype(np.float32) + y.astype(np.float32)) * np.float32(0.5) / np.float32(32768.0))
    fixtures.append(("pcm16_stereo", wav_bytes(pcm_i16=stereo, ch=2, bits=16), exp, True))
    fixtures.append(("f32_mono", wav_bytes(pcm_f32=short, ch=1, bits=32, float_format=True),
                     short.astype(np.float32), True))
    a, b = short, short[::-1]
    st2 = np.empty(2 * len(a), dtype=np.float32)
    st2[0::2], st2[1::2] = a, b
    fixtures.append(("f32_stereo", wav_bytes(pcm_f32=st2, ch=2, bits=32, float_format=True),
                     (a.astype(np.float32) + b.astype(np.float32)) * np.float32(0.5), True))
    fixtures.append(("ext_pcm16", wav_bytes(pcm_i16=x, ch=1, bits=16, extensible=True),
                     x.astype(np.float32) / np.float32(32768.0), True))
    fixtures.append(("ext_f32", wav_bytes(pcm_f32=short, ch=1, bits=32, float_format=True, extensible=True),
                     short.astype(np.float32), True))
    fixtures.append(("chunks_junk", wav_bytes(pcm_f32=short, ch=1, bits=32, float_format=True,
                                             junk=[(b"LIST", b"INFOISFT" + b"x" * 7), (b"fact", b"\0" * 4)],
                                             lie_about_size=True),
                     short.astype(np.float32), True))
    ext = np.array([0, -32768, 32767, 1, -1, 256, -256, 16384], dtype="<i2")
    fixtures.append(("pcm16_extremes", wav_bytes(pcm_i16=ext, ch=1, bits=16),
                     ext.astype(np.float32) / np.float32(32768.0), True))
    fixtures.append(("empty", wav_bytes(pcm_f32=np.zeros(0, dtype=np.float32), ch=1, bits=32,
                                       float_format=True), np.zeros(0, dtype=np.float32), True))
    # --- refusals ------------------------------------------------------------------
    fixtures.append(("rate_44k", wav_bytes(pcm_f32=short, rate=44100, ch=1, bits=32, float_format=True), None, False))
    fixtures.append(("rate_8k", wav_bytes(pcm_i16=x, rate=8000, ch=1, bits=16), None, False))
    fixtures.append(("pcm24", wav_bytes(pcm_i16=x, ch=1, bits=24), None, False))
    fixtures.append(("three_ch", wav_bytes(pcm_i16=x, ch=3, bits=16), None, False))
    fixtures.append(("truncated", wav_bytes(pcm_i16=x, ch=1, bits=16, truncate=3), None, False))
    fixtures.append(("ext_pcm24", wav_bytes(pcm_i16=x, ch=1, bits=24, extensible=True), None, False))
    fixtures.append(("not_riff", b"RIFF" + b"\0" * 300, None, False))

    # ship the fixtures
    subprocess.run(["ssh", "-o", "ConnectTimeout=25", "-o", "BatchMode=yes", NANO,
                    "mkdir -p %s/wavfix %s/wavout" % (NDIR, NDIR)], check=True)
    for name, blob, _, _ in fixtures:
        p = os.path.join(WORK, name + ".wav")
        open(p, "wb").write(blob)
        subprocess.run(["scp", "-q", "-o", "ConnectTimeout=20", p, "%s:%s/wavfix/" % (NANO, NDIR)], check=True)

    fails = 0
    for name, blob, exp, expect_ok in fixtures:
        # one reader invocation, then a separate cp: keeping the run and the copy in
        # separate calls is what lets a REFUSAL fixture be scored (rc) without a second
        # run of the binary, and keeps the command strings free of index arithmetic.
        ssh("rm -f %s/wavout/%s.f32 /tmp/%s.f32" % (NDIR, name, name))
        one = ssh("%s/lfm3_wav %s/wavfix/%s.wav %s/wavout/%s.f32 >/tmp/w.log 2>&1; echo RC=$?; cat /tmp/w.log"
                  % (NDIR, NDIR, name, NDIR, name))
        lines = one.stdout.splitlines()
        rc = None
        for tok in lines:
            if tok.startswith("RC="):
                rc = int(tok[3:])
        msg = " | ".join(l for l in lines if not l.startswith("RC="))[-90:]
        if expect_ok:
            ssh("cp %s/wavout/%s.f32 /tmp/%s.f32" % (NDIR, name, name))
        if expect_ok:
            out = os.path.join(WORK, name + ".out.f32")
            subprocess.run(["scp", "-q", "-o", "ConnectTimeout=20",
                            "%s:/tmp/%s.f32" % (NANO, name), out], check=True)
            got = np.fromfile(out, dtype=np.float32)
            ok = got.size == exp.size and (got.size == 0 or np.array_equal(got.view(np.uint32), exp.view(np.uint32)))
            nbad = int((got != exp).sum()) if got.size == exp.size and got.size else (0 if got.size == exp.size else -1)
            print("  %-14s BIT-EXACT vs numpy: %s   (%d cells differ, %d samples)" % (name, "OK" if ok else "FAIL", nbad, got.size))
            if not ok:
                fails += 1
        else:
            ok = rc not in (0, None)
            print("  %-14s REFUSED rc=%s: %s   %s" % (name, rc, msg[:78], "OK" if ok else "FAIL (accepted!)"))
            if not ok:
                fails += 1

    # --- end-to-end: WAV -> pcm.f32 -> lfm3_mel -> mel.f32 == cached HF mel --------
    print("  -- WAV -> mel chain against the cached HF reference (S2 tolerance %g) --" % MEL_TOL_LIVE)
    nchain = 0
    for name in names[:6]:
        pcm = np.fromfile(os.path.join(FIXDIR, name + ".f32"), dtype=np.float32)
        blob = wav_bytes(pcm_f32=pcm, ch=1, bits=32, float_format=True)
        p = os.path.join(WORK, name + ".wav")
        open(p, "wb").write(blob)
        subprocess.run(["scp", "-q", "-o", "ConnectTimeout=20", p, "%s:%s/wavfix/" % (NANO, NDIR)], check=True)
        ssh(f"{NDIR}/lfm3_wav {NDIR}/wavfix/{name}.wav {NDIR}/wavout/{name}.pcm.f32 > /dev/null 2>&1; "
            f"{NDIR}/lfm3_mel {NDIR}/mel.bin {NDIR}/wavout/{name}.pcm.f32 {NDIR}/wavout/{name}.mel.f32 > /dev/null 2>&1")
        subprocess.run(["scp", "-q", "-o", "ConnectTimeout=20", f"{NANO}:{NDIR}/wavout/{name}.mel.f32",
                        os.path.join(WORK, name + ".mel.f32")], check=True)
        hp = os.path.join(REFDIR, name + ".f32")
        if not os.path.exists(hp):
            print("  %-14s no cached HF mel; skipped" % name)
            continue
        meta = json.load(open(os.path.join(REFDIR, name + ".json")))
        T = int(meta["frames"])
        got = np.fromfile(os.path.join(WORK, name + ".mel.f32"), dtype=np.float32)
        ref = np.fromfile(hp, dtype=np.float32).astype(np.float64)
        if got.size != NM * T:
            print("  %-14s FAIL shape: engine %d cells, HF %d (T=%d)" % (name, got.size, ref.size, T))
            fails += 1
            continue
        g = got.astype(np.float64).reshape(T, NM)
        r_ = ref.reshape(T, NM)
        d = np.abs(g - r_)
        live = d <= 0  # placeholder; live-cell mask below
        peak = r_.max(axis=1, keepdims=True)
        live = (peak > 0) & (r_ >= peak * 1e-6)
        lm = float(d[live].max()) if live.any() else 0.0
        lmean = float(d[live].mean()) if live.any() else 0.0
        ok = lm <= MEL_TOL_LIVE and lmean <= MEL_TOL_LIVE
        print("  %-14s T=%3d  live max %.2e  live mean %.2e  %s" % (name, T, lm, lmean, "OK" if ok else "FAIL"))
        if not ok:
            fails += 1
        nchain += 1
    print("WAV_GATE %s (%d fixtures, %d chain)" % ("PASS" if fails == 0 and nchain else "FAIL", len(fixtures), nchain))
    sys.exit(0 if fails == 0 and nchain else 1)


if __name__ == "__main__":
    main()
