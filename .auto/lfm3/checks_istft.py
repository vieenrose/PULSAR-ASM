#!/usr/bin/env python3
"""checks_istft.py -- S5 spike gate: the CUDA iSTFT + overlap-add against torch.

The reference is modeling_liquid_codec.py:LiquidCodecISTFT.forward +
LiquidCodecModel.decode (the model's own files, replayed in torch on the host),
NOT the engine. Cases are generated from the checkpoint's REAL decoder.lin
weights (the codec is bf16, so the spectra are bf16-rounded -- which is what the
engine's own lin GEMV will produce, so both sides see the same numbers), plus a
sine-tone case and a flat-spectrum click case.

Gate per case:
  * samples == F*320 exactly (the trim/tail arithmetic),
  * all finite,
  * SNR >= ISTFT_SNR (default 60 dB) for BOTH twiddle arms (__sincosf and the
    host-built double table),
  * max |delta| <= max(ISTFT_MAXABS, 1e-3 * rms(reference)) -- the absolute
    bound is scale-relative, because a sigma=2 hidden state gives magnitudes
    e^20 and a fixed 2e-3 there would be demanding 1e-8 relative accuracy.
  * cases whose reference is numerically zero (rms < 1e-12) are checked for a
    numerically zero engine output instead of an SNR (SNR against nothing).

WHY SNR AND NOT BITS: torch.fft.irfft is cuFFT/pocketfft, our kernel is a direct
transform in a documented order; a neural codec's waveform has no bit-exactness
to promise. The number is recorded, not fitted (doc/LFM3_PORT_STAGES.md S5).

Usage:
  ~/venvs/lfm3/bin/python .auto/lfm3/checks_istft.py            # gate
  ~/venvs/lfm3/bin/python .auto/lfm3/checks_istft.py --regen     # rebuild cases
"""
import json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NANO = os.environ.get("NANO", "picard@100.84.151.116")
DIR = os.path.join(os.environ.get("NANO_DIR", "/home/picard/lfm3"), "istft")
MODEL = os.path.expanduser("~/models/lfm3/LFM3-Audio-2B")
CODEC = os.path.join(MODEL, "audio_tokenizer")
CASEDIR = os.path.join(HERE, "istft_cases")
SRC = os.path.join(os.path.dirname(os.path.dirname(HERE)), "pulsar_arm", "asm")
NEED_SNR = float(os.environ.get("ISTFT_SNR", "60"))
MAXABS = float(os.environ.get("ISTFT_MAXABS", "2e-3"))
NFFT, NHOP, NBIN = 1280, 320, 641
CASES = [("lin_s0.1", 75, 0.1), ("lin_s0.5", 75, 0.5), ("lin_s2", 75, 2.0),
         ("lin_F6", 6, 0.5), ("lin_F12", 12, 0.5), ("lin_F24", 24, 0.5), ("lin_F300", 300, 0.5)]


def run_remote(cmd, timeout=1200):
    p = subprocess.run(["ssh", "-o", "ConnectTimeout=25", "-o", "BatchMode=yes", NANO, cmd],
                       capture_output=True, text=True, timeout=timeout)
    return "\n".join(l for l in (p.stdout + p.stderr).splitlines()
                     if "setlocale" not in l and "warning:" not in l)


def bf16_round(x):
    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint32)
    u = u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))     # RN-even
    return (u & np.uint32(0xFFFF0000)).view(np.float32)


def gen_cases():
    import torch
    import torch.nn.functional as F_
    import json as jsonl
    from safetensors.torch import load_file
    sd = load_file(os.path.join(CODEC, "model.safetensors"))
    os.makedirs(CASEDIR, exist_ok=True)
    win = sd["decoder.istft.window"].float()
    win.numpy().astype(np.float32).tofile(os.path.join(CASEDIR, "window.f32"))
    lin_w = sd["decoder.lin.weight"].bfloat16()
    lin_b = sd["decoder.lin.bias"].bfloat16()
    # the twiddle table for the accurate arm, built in double
    n = np.arange(NFFT)[:, None]
    k = np.arange(1, NFFT // 2)[None, :]
    ang = 2.0 * np.pi * (n * k % NFFT) / NFFT
    assert ang.shape == (NFFT, NFFT // 2 - 1), ang.shape      # 639 columns, k = 1..639
    tab = np.concatenate([np.cos(ang).ravel(), np.sin(ang).ravel()]).astype(np.float32)
    tab.tofile(os.path.join(CASEDIR, "table.f32"))

    rng = np.random.default_rng(5150)
    index = []
    for name, Fr, sig in CASES:
        h = torch.tensor(rng.standard_normal((512, Fr)) * sig, dtype=torch.float32).bfloat16()
        out = (lin_w @ h + lin_b.unsqueeze(1)).float().numpy()            # (1282, Fr), bf16-rounded values
        logmag, angle = bf16_round(out[:NBIN].T).T, bf16_round(out[NBIN:].T).T
        finish(name, Fr, logmag, angle, win)
        index.append((name, Fr))
    # sine tone: analyse the SAME waveform the decoder will rebuild -- take the
    # 1280-sample window at every hop and rfft it, so magnitudes/phases are the
    # ones a real frame would carry (exponential decay, 440 Hz, 24 kHz)
    Fr = 48
    t = np.arange((Fr - 1) * NHOP + NFFT) / 24000.0
    tone = np.sin(2 * np.pi * 440 * t) * np.exp(-t / 0.4)
    S = np.stack([np.fft.rfft(tone[f * NHOP: f * NHOP + NFFT], NFFT) for f in range(Fr)], 1)  # (641,F)
    mag = np.maximum(np.abs(S), 1e-12)
    finish("tone440", Fr, np.log10(mag).astype(np.float32), np.angle(S).astype(np.float32), win)
    index.append(("tone440", Fr))
    # click: a broadband flat-magnitude frame with the impulse at the WINDOW
    # CENTRE (angle = pi*k delays it to n = 640). With zero phase the irfft is a
    # Dirac at n=0, and hann(0) = 0 annihilates the whole frame -- a degenerate
    # reference (rms 1e-19) whose SNR is meaningless. That case was the first
    # version of this file and it taught nothing; a centred click does.
    finish("click", 24, np.full((NBIN, 24), -3.0, np.float32),
           (np.pi * np.arange(NBIN)[:, None] % (2 * np.pi)).repeat(24, 1).astype(np.float32), win)
    index.append(("click", 24))
    json.dump(index, open(os.path.join(CASEDIR, "index.json"), "w"))
    return index


def finish(name, Fr, logmag, angle, win):
    import torch
    import torch.nn.functional as F_
    spec = torch.polar(torch.tensor(logmag).float().exp(), torch.tensor(angle).float())  # (641,F)
    frames = torch.fft.irfft(spec, NFFT, dim=0, norm="backward") * win[:, None]
    L = (Fr - 1) * NHOP + NFFT
    fold = {"output_size": (1, L), "kernel_size": (1, NFFT), "stride": (1, NHOP)}
    audio = F_.fold(frames.unsqueeze(0), **fold)[0, 0, 0]        # (1, 1280, F) -> (1,1,L)
    env = F_.fold((win ** 2).expand(Fr, NFFT).T.unsqueeze(0), **fold)[0, 0, 0]
    trim = (NFFT - NHOP) // 2
    o = (audio[trim:-trim] / env[trim:-trim]).float().numpy()
    np.ascontiguousarray(logmag, np.float32).tofile(os.path.join(CASEDIR, name + ".lm"))
    np.ascontiguousarray(angle, np.float32).tofile(os.path.join(CASEDIR, name + ".an"))
    inp = np.concatenate([logmag.T.ravel(), angle.T.ravel()]).astype(np.float32)
    inp.tofile(os.path.join(CASEDIR, name + ".in.f32"))
    o.astype(np.float32).tofile(os.path.join(CASEDIR, name + ".ref.f32"))


def build():
    run_remote("mkdir -p %s" % DIR)
    subprocess.run(["scp", "-q", "-o", "ConnectTimeout=20", os.path.join(SRC, "lfm3_istft.cu"),
                    "%s:%s/" % (NANO, DIR)], check=True)
    # house build line (PATH, .new+mv, BUILT marker). -fmad=false on purpose: the
    # kernel's documented order is fmul-then-fadd, and contraction would change it.
    # No --use_fast_math: expf must stay IEEE (the fast arm is the explicit __sincosf).
    out = run_remote("cd %s && rm -f lfm3_istft.new && export PATH=/usr/local/cuda/bin:$PATH && "
                     "nvcc -gencode arch=compute_53,code=sm_53 -O3 -fmad=false "
                     "-Xptxas -O3,-dlcm=ca -o lfm3_istft.new lfm3_istft.cu -lm 2>&1; "
                     "test -x lfm3_istft.new && mv -f lfm3_istft.new lfm3_istft && echo BUILT" % DIR)
    print(out.strip())
    if "BUILT" not in out:
        sys.exit("build failed -- refusing to measure a stale binary")
    for f in ("window.f32", "table.f32"):
        subprocess.run(["scp", "-q", os.path.join(CASEDIR, f), "%s:%s/" % (NANO, DIR)], check=True)
    for f in os.listdir(CASEDIR):
        if f.endswith(".in.f32"):
            subprocess.run(["scp", "-q", os.path.join(CASEDIR, f), "%s:%s/" % (NANO, DIR)], check=True)


def main():
    if "--regen" in sys.argv or not os.path.exists(os.path.join(CASEDIR, "index.json")):
        index = gen_cases()
    else:
        index = json.load(open(os.path.join(CASEDIR, "index.json")))
    build()
    print("\n%-10s %5s %10s %10s %11s %11s  %s" %
          ("case", "F", "max|d|", "SNR dB", "max|d| tbl", "SNR tbl", "verdict"))
    fails = []
    for name, Fr in index:
        ref = np.fromfile(os.path.join(CASEDIR, name + ".ref.f32"), np.float32)
        got = {}
        for arm, env in (("sc", "0"), ("tb", "1")):
            o = run_remote("cd %s && LFM_ISTFT_TABLE=%s ./lfm3_istft window.f32 %s.in.f32 %s.%s.out %d table.f32 2>&1"
                           % (DIR, env, name, name, arm, Fr))
            if "ISTFT" not in o:
                fails.append((name, arm, o)); got[arm] = None; continue
            subprocess.run(["scp", "-q", "%s:%s/%s.%s.out" % (NANO, DIR, name, arm),
                            "/tmp/istft_%s_%s.out" % (name, arm)], check=True)
            got[arm] = np.fromfile("/tmp/istft_%s_%s.out" % (name, arm), np.float32)
            got[arm + "_line"] = [l for l in o.splitlines() if l.startswith("ISTFT")][0]
        ok, cells = True, []
        for arm, lbl in (("sc", "sincosf"), ("tb", "table")):
            e = got.get(arm)
            if e is None:
                cells.append("RUN FAIL"); ok = False; continue
            if e.shape[0] != ref.shape[0]:
                cells.append("len %d!=%d" % (e.shape[0], ref.shape[0])); ok = False; continue
            if not np.isfinite(e).all():
                cells.append("non-finite"); ok = False; continue
            d = np.abs(e.astype(np.float64) - ref.astype(np.float64))
            # SNR = signal power / noise power
            snr = 10 * np.log10(max(float((ref.astype(np.float64) ** 2).sum()), 1e-30) /
                                max(float((d.astype(np.float64) ** 2).sum()), 1e-30))
            rms = float(np.sqrt((ref.astype(np.float64) ** 2).mean()))
            cells.append("%.3g / %.1f dB" % (float(d.max()), snr))
            # absolute bound is scale-relative: lin_s2 has magnitudes e^20, so a
            # fixed 2e-3 there would be a 1e-8 relative demand. rms(ref) == 0
            # makes SNR meaningless (the degenerate Dirac-under-window case).
            if rms > 1e-12 and snr < NEED_SNR:
                ok = False
            if rms > 1e-12 and float(d.max()) > max(MAXABS, 1e-3 * rms):
                ok = False
            if rms <= 1e-12 and float(np.sqrt((e.astype(np.float64) ** 2).mean())) > 1e-9:
                cells.append("nonzero output for a zero reference"); ok = False
        print("%-10s %5d %14s %14s  %s" % (name, Fr, cells[0], cells[1] if len(cells) > 1 else "-",
                                           "ok" if ok else "FAIL"))
        if not ok:
            fails.append((name, "snr/maxabs", cells))
        print("    %s | %s" % (got.get("sc_line", ""), got.get("tb_line", "")))
    print("\nS5 iSTFT spike gate: %d cases, SNR >= %g dB, max|d| <= %g -> %s"
          % (len(index), NEED_SNR, MAXABS, "PASS" if not fails else "FAIL"))
    for f in fails:
        print("  FAIL", f)
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
