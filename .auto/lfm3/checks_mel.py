#!/usr/bin/env python3
"""checks_mel.py -- S2 gate: the log-mel front end (pulsar_arm/asm/lfm3_mel.S)
against the HF reference, with an exact-arithmetic referee.

Three sources of truth, all independent of the engine:
  1. HF  -- LFM3-Audio-2B/feature_extraction_lfm3_audio.py (the model's own file,
            trust_remote_code), cached in .auto/lfm3/mel_ref/<name>.f32 so the
            gate runs without torch afterwards (same convention as the
            backbone's hf_*.npy caches).
  2. f64 -- an exact-arithmetic recomputation of the same recipe in numpy
            float64 (direct 201x200 transform, mel.bin's own window and Slaney
            filters), cached in mel_ref/<name>.x64. This is the referee: it says
            which side is right when engine and HF disagree.
  3. the C mirror in lfm3_mel_test.c, run on the board -- BIT equality with the
            kernel's documented accumulation order. That is the bit-exactness
            gate; HF parity can never be one (torch.stft = pocketfft, torch
            matmul = BLAS, so both use different summation orders).

Per fixture, PASS requires:
  * frames == HF (this is the padding rule pad_length/num_frames),
  * LIVE-CELL max |engine - HF| <= MEL_TOL_LIVE (default 1e-5). "Live" = within
    6 decades (120 dB) of that frame's peak mel -- beyond that a float32
    200-term sum is noise, and the f64 column shows HF is noise there too.
  * mean |engine - HF| over the live cells <= MEL_TOL_MEAN (default 1e-5);
    the all-cell mean is reported next to it (it is dominated by the noise cells
    on tonal fixtures, which is why it is not the bound),
  * all-cell max |engine - HF| <= MEL_TOL_ALL (default 0.06), reported with its
    cause, not hidden,
  * the engine is not LESS accurate than HF against exact arithmetic:
    mean|engine-f64| <= 1.5 * mean|HF-f64| + 1e-9.

Measured on the Nano, 19 fixtures: live max 8.3e-6, mean 2.2e-8, all-cell max
5.0e-2 (one chirp cell 13.5 decades under its frame peak, where HF is 3.9e-2
off from exact f64), mean(eng-f64)/mean(HF-f64) <= 1.28.

Usage:
  ~/venvs/lfm3/bin/python .auto/lfm3/checks_mel.py            # gate (refs cached)
  ~/venvs/lfm3/bin/python .auto/lfm3/checks_mel.py --regen    # rebuild HF refs
  MEL_TOL_LIVE=1e-6 python .auto/lfm3/checks_mel.py           # tighten probe
"""
import json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NANO = os.environ.get("NANO", "picard@100.84.151.116")
DIR = os.environ.get("NANO_DIR", "/home/picard/lfm3")
MODEL = os.path.expanduser("~/models/lfm3/LFM3-Audio-2B")
REPO = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(REPO, "pulsar_arm")
REFDIR = os.path.join(HERE, "mel_ref")
FIXDIR = os.path.join(HERE, "mel_fix")
MELBIN = os.environ.get("MEL_BIN", "/tmp/mel.bin")
TOL_LIVE = float(os.environ.get("MEL_TOL_LIVE", "1e-5"))    # measured 8.3e-6
TOL_MEAN = float(os.environ.get("MEL_TOL_MEAN", "1e-6"))   # measured 2.2e-7
TOL_ALL = float(os.environ.get("MEL_TOL_ALL", "0.06"))        # measured 0.0499 (chirp)
LIVE_DEC = float(os.environ.get("MEL_LIVE_DEC", "6"))       # decades under frame peak
SR = 16000
NM, NB, NBP, NFFT, HOP = 128, 201, 208, 400, 160


def fixtures():
    """Deterministic waveforms. Lengths hit every branch of pad_length
    (<400, ==400, 400+1, exact hops, one sample off each side, 1 s, 2 s)."""
    rng = np.random.default_rng(20261030)
    t = lambda n: np.arange(n, dtype=np.float64) / SR
    out = {}
    out["silence"] = np.zeros(1600, np.float32)
    out["near_silence"] = (rng.standard_normal(1600) * 1e-8).astype(np.float32)
    out["sine440"] = (0.5 * np.sin(2 * np.pi * 440 * t(8000))).astype(np.float32)
    ph = 2 * np.pi * (100 * t(16000) + 3750 * t(16000) ** 2 / 2)
    out["chirp"] = (0.7 * np.sin(ph)).astype(np.float32)
    out["noise"] = (0.3 * rng.standard_normal(16000)).astype(np.float32)
    n = t(32000)
    harm = sum(np.sin(2 * np.pi * (110 * k) * n) / k for k in range(1, 12))
    env = (0.5 + 0.5 * np.sin(2 * np.pi * 2.3 * n)) * (1 - np.abs(n - 16000) / 16000)
    out["speechish"] = (0.25 * harm * env + 0.02 * rng.standard_normal(32000)).astype(np.float32)
    imp = np.zeros(8000, np.float32); imp[::400] = 0.9
    out["clicks"] = imp
    out["dc"] = np.full(3000, 0.1, np.float32)
    out["square_fs"] = np.sign(np.sin(2 * np.pi * 220 * t(8000))).astype(np.float32)
    for L in (1, 159, 399, 400, 401, 559, 560, 561, 15999, 16001):
        out["len%d" % L] = (0.4 * rng.standard_normal(L)).astype(np.float32)
    return out


def tables():
    """Window + filters from mel.bin, in f64, for the exact referee."""
    b = open(MELBIN, "rb").read()
    assert b[:8] == b"LFM3MEL1", "%s is not a LFM3MEL1 blob" % MELBIN
    o = 56
    win = np.frombuffer(b[o:o + NFFT * 4], np.float32).astype(np.float64); o += NFFT * 4
    o += 2 * NB * (NFFT // 2) * 4
    filt = np.frombuffer(b[o:o + NM * NBP * 4], np.float32).reshape(NM, NBP).astype(np.float64)
    return win, filt


def exact_features(x, T):
    """The reference recipe in float64, direct transform, no BLAS, no pocketfft."""
    win, filt = tables()
    k = np.arange(NB)[:, None]; n = np.arange(NFFT // 2)[None, :]
    C = np.cos(2 * np.pi * k * n / NFFT); S = np.sin(2 * np.pi * k * n / NFFT)
    sg = 1 - 2 * (k % 2)
    xd = x.astype(np.float64)
    P = np.zeros((T, NFFT))
    for t in range(T):
        m = min(NFFT, max(0, len(xd) - t * HOP)); P[t, :m] = xd[t * HOP:t * HOP + m]
    y = P * win
    c = y[:, None, :NFFT // 2] + y[:, None, NFFT // 2:] * sg[None, :, :]
    re = (c * C).sum(2); im = -(c * S).sum(2)
    with np.errstate(divide="ignore"):
        lg = np.log10((re * re + im * im) @ filt[:, :NB].T)
    return (np.maximum(lg, -12.0) + 4.0) / 4.0, lg


def run_remote(cmd, timeout=1200):
    p = subprocess.run(["ssh", "-o", "ConnectTimeout=25", "-o", "BatchMode=yes", NANO, cmd],
                       capture_output=True, text=True, timeout=timeout)
    return "\n".join(l for l in (p.stdout + p.stderr).splitlines()
                     if "setlocale" not in l and "warning:" not in l)


def gen_fixtures():
    os.makedirs(FIXDIR, exist_ok=True); os.makedirs(REFDIR, exist_ok=True)
    fx = fixtures()
    for name, w in fx.items():
        w.astype(np.float32).tofile(os.path.join(FIXDIR, name + ".f32"))
    json.dump(sorted(fx), open(os.path.join(FIXDIR, "index.json"), "w"))
    return sorted(fx)


def gen_refs(names):
    sys.path.insert(0, MODEL)
    from feature_extraction_lfm3_audio import Lfm3AudioFeatureExtractor
    fe = Lfm3AudioFeatureExtractor.from_pretrained(MODEL, trust_remote_code=True)
    for name in names:
        w = np.fromfile(os.path.join(FIXDIR, name + ".f32"), dtype=np.float32)
        f = np.asarray(fe(raw_speech=w, sampling_rate=SR, return_tensors="np")["input_features"])[0]
        f.astype(np.float32).tofile(os.path.join(REFDIR, name + ".f32"))
        json.dump({"frames": int(f.shape[1]), "samples": int(w.shape[0])},
                  open(os.path.join(REFDIR, name + ".json"), "w"))


def gen_exact(names):
    for name in names:
        w = np.fromfile(os.path.join(FIXDIR, name + ".f32"), dtype=np.float32)
        nfr = json.load(open(os.path.join(REFDIR, name + ".json")))["frames"]
        f, _ = exact_features(w, nfr)
        f.T.astype(np.float64).tofile(os.path.join(REFDIR, name + ".x64"))


def build_on_board(names):
    d = "%s/mel" % DIR
    run_remote("mkdir -p %s" % d)
    for f in ("lfm3_mel.S", "lfm3_mel.c", "lfm3_mel_test.c"):
        subprocess.run(["scp", "-q", "-o", "ConnectTimeout=20", os.path.join(SRC, "asm", f),
                        "%s:%s/" % (NANO, d)], check=True)
    subprocess.run(["scp", "-q", "-o", "ConnectTimeout=20", MELBIN, "%s:%s/mel.bin" % (NANO, d)],
                   check=True)
    out = run_remote(
        "cd %s && as -o lfm3_mel_o lfm3_mel.S 2>&1 && "
        "gcc -O2 -ffp-contract=off -o lfm3_mel_test.new lfm3_mel_test.c lfm3_mel_o -lm 2>&1 && "
        "mv lfm3_mel_test.new lfm3_mel_test && echo BUILT_TEST && "
        "as -o lfm3_mel_o2 lfm3_mel.S 2>&1 && "
        "gcc -O2 -ffp-contract=off -o lfm3_mel.new lfm3_mel.c lfm3_mel_o2 -lm 2>&1 && "
        "mv lfm3_mel.new lfm3_mel && echo BUILT_DRV" % d)
    print(out.strip())
    if "BUILT_TEST" not in out or "BUILT_DRV" not in out:
        sys.exit("build failed (no BUILT marker) -- refusing to measure a stale binary")
    for f in os.listdir(FIXDIR):
        if f.endswith(".f32"):
            subprocess.run(["scp", "-q", "-o", "ConnectTimeout=20", os.path.join(FIXDIR, f),
                            "%s:%s/" % (NANO, d)], check=True)
    return d


def main():
    names = gen_fixtures()
    if not os.path.exists(MELBIN):
        subprocess.run([sys.executable, os.path.join(SRC, "tools", "mkmel.py"), "--out", MELBIN],
                       check=True)
    if "--regen" in sys.argv or not os.path.exists(os.path.join(REFDIR, names[0] + ".f32")):
        gen_refs(names)
    if not os.path.exists(os.path.join(REFDIR, names[0] + ".x64")):
        gen_exact(names)
    d = build_on_board(names)

    out = run_remote("cd %s && ./lfm3_mel_test 2>&1" % d)
    print(out.strip())
    parity_ok = "-> YES" in out
    if not parity_ok:
        print("!! bit-parity of the asm against the documented-order C mirror FAILED")

    print("\n%-11s %6s %6s %10s %10s %10s %9s %9s  %s"
          % ("fixture", "samples", "frames", "live max", "live mean", "all max", "eng-f64", "HF-f64", "verdict"))
    fails = []
    for name in names:
        w = np.fromfile(os.path.join(FIXDIR, name + ".f32"), dtype=np.float32)
        meta = json.load(open(os.path.join(REFDIR, name + ".json")))
        ref = np.fromfile(os.path.join(REFDIR, name + ".f32"), np.float32).reshape(NM, meta["frames"]).T
        ex = np.fromfile(os.path.join(REFDIR, name + ".x64"), np.float64).reshape(NM, meta["frames"]).T
        o = run_remote("cd %s && ./lfm3_mel mel.bin %s.f32 %s.out 2>&1" % (d, name, name))
        ln = [l for l in o.splitlines() if l.startswith("MEL ")]
        if not ln:
            fails.append((name, "run failed: " + o)); print("%-11s RUN FAILED %s" % (name, o)); continue
        frames = int(ln[0].split("frames=")[1].split()[0])
        if frames != meta["frames"]:
            fails.append((name, "frames %d != %d" % (frames, meta["frames"])))
            print("%-11s FRAME COUNT engine %d reference %d  FAIL" % (name, frames, meta["frames"])); continue
        subprocess.run(["scp", "-q", "-o", "ConnectTimeout=20", "%s:%s/%s.out" % (NANO, d, name),
                        "/tmp/mel_%s.out" % name], check=True)
        eng = np.fromfile("/tmp/mel_%s.out" % name, np.float32).reshape(NM, frames).T
        if (~np.isfinite(eng)).any():
            fails.append((name, "non-finite engine values"))
            print("%-11s non-finite engine values  FAIL" % name); continue
        _, lg = exact_features(w, frames)
        live = lg > (lg.max(1, keepdims=True) - LIVE_DEC)
        dd = np.abs(eng.astype(np.float64) - ref.astype(np.float64))
        lm = float(dd[live].max()) if live.any() else 0.0
        mn = float(dd[live].mean()) if live.any() else 0.0
        mx, amn = float(dd.max()), float(dd.mean())
        e1 = float(np.abs(ex - eng)[live].mean()) if live.any() else 0.0
        e2 = float(np.abs(ex - ref)[live].mean()) if live.any() else 0.0
        bad = []
        if lm > TOL_LIVE: bad.append("live %.3g>%g" % (lm, TOL_LIVE))
        if mn > TOL_MEAN: bad.append("mean %.3g>%g" % (mn, TOL_MEAN))
        if mx > TOL_ALL: bad.append("all %.3g>%g" % (mx, TOL_ALL))
        if e1 > 1.5 * e2 + 1e-9: bad.append("less accurate than HF (%.3g vs %.3g)" % (e1, e2))
        fails += [(name, ", ".join(bad))] if bad else []
        extra = "" if abs(amn - mn) < 10 * max(mn, 1e-12) else "  (all-cell mean %.3g)" % amn
        print("%-11s %6d %6d %10.3g %10.3g %10.3g %9.3g %9.3g  %s%s"
              % (name, w.shape[0], frames, lm, mn, mx, e1, e2,
                 "ok" if not bad else "FAIL: " + ", ".join(bad), extra))

    print("\nS2 log-mel gate: %d fixtures | bit-parity %s | tol live %g mean %g all %g live-window %g decades"
          % (len(names), "YES" if parity_ok else "NO", TOL_LIVE, TOL_MEAN, TOL_ALL, LIVE_DEC))
    for name, why in fails:
        print("  FAIL %-11s %s" % (name, why))
    ok = not fails and parity_ok
    print("S2 gate: %s" % ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
