"""check_m2.py -- M2 (text -> speech) report: live run vs the HF-on-Q8 oracle.

Usage: check_m2.py <oracle_dir> <live_prefix> [live_audio.f32]
NOT a pass/fail gate on stream equality -- that is exactly the claim the engine cannot
make today, and this script exists to measure how far off it is instead of asserting it:

  slots      kinds compared line by line (the mode machine must agree even when a code
             differs; a kind mismatch is a controller bug, not a precision effect)
  text       exact match expected (greedy argmax, margins are 4-18)
  frames     per-codebook agreement rate and the first divergent frame
  audio      if both sides have a decoded waveform: SNR (dB) of live vs oracle, plus
             per-segment SNR so a local collapse cannot hide behind a global average

Reported as numbers, not as a verdict: the divergence root cause is the backbone's
bf16 activations (engine+oracle-hidden reproduces the oracle codes exactly; HF-on-Q8
weights pick the same codes as HF-fp32, so the weights are NOT the cause).
"""
import os
import sys

import numpy as np

G, L = sys.argv[1], sys.argv[2]
audio = sys.argv[3] if len(sys.argv) > 3 else None

ok_text = True
want = open(os.path.join(G, "slots.txt")).read().splitlines()
got = open(L + ".slots.txt").read().splitlines()
wk = [w.split()[0] for w in want]
gk = [g.split()[0] for g in got]
n = min(len(wk), len(gk))
kind_bad = [i for i in range(n) if wk[i] != gk[i]]
print("slots: oracle %d, live %d; kind agreement %d/%d%s"
      % (len(wk), len(gk), n - len(kind_bad), n,
         "" if not kind_bad else "; first kind mismatch at %d" % kind_bad[0]))
wt = np.fromfile(os.path.join(G, "text.i32"), dtype=np.int32)
gt = np.fromfile(L + ".text.i32", dtype=np.int32)
nt = min(len(wt), len(gt))
tb = [i for i in range(nt) if wt[i] != gt[i]]
print("text: oracle %d, live %d; exact %d/%d%s"
      % (len(wt), len(gt), nt - len(tb), nt,
         "" if not tb else "; first text mismatch at %d (oracle %d live %d)"
         % (tb[0], int(wt[tb[0]]), int(gt[tb[0]]))))
if len(wt) != len(gt):
    ok_text = False

wf = np.fromfile(os.path.join(G, "audio_codes.i32"), dtype=np.int32).reshape(12, -1).T
gf = np.fromfile(L + ".frames.i32", dtype=np.int32).reshape(-1, 12)
nf = min(len(wf), len(gf))
if nf:
    eq = (wf[:nf] == gf[:nf])
    print("frames: oracle %d, live %d; codebook agreement %d/%d (%.3f%%); frames fully exact %d/%d"
          % (len(wf), len(gf), eq.sum(), eq.size, 100.0 * eq.mean(),
             eq.all(1).sum(), nf))
    bad = np.where(~eq.all(1))[0]
    for i in bad[:5]:
        d = np.where(~eq[i])[0].tolist()
        print("  frame %d differs at codebooks %s: oracle %s live %s"
              % (i, d, wf[i][d].tolist(), gf[i][d].tolist()))
else:
    print("frames: no frames on one side (oracle %d, live %d)" % (len(wf), len(gf)))

if audio and os.path.exists(audio):
    ref_p = os.path.join(G, "audio.f32")
    if os.path.exists(ref_p):
        a = np.fromfile(audio, dtype=np.float32)
        b = np.fromfile(ref_p, dtype=np.float32)
        m = min(a.size, b.size)
        if m > 0:
            def snr(x, y):
                e = float(((x - y) ** 2).mean())
                s = float((y ** 2).mean())
                return 10.0 * np.log10(s / e) if e > 0 and s > 0 else float("inf")
            print("audio: live %d samples, oracle %d; overall SNR %.2f dB"
                  % (a.size, b.size, snr(a[:m], b[:m])))
            seg = m // 3
            for k in range(3):
                sl = slice(k * seg, (k + 1) * seg if k < 2 else m)
                print("  segment %d (%d..%d): SNR %.2f dB"
                      % (k, sl.start, sl.stop, snr(a[sl], b[sl])))
            print("  live rms %.5f peak %.4f; oracle rms %.5f peak %.4f"
                  % (np.sqrt((a ** 2).mean()), np.abs(a).max(),
                     np.sqrt((b[:m] ** 2).mean()), np.abs(b[:m]).max()))
        else:
            print("audio: one side is empty")
    else:
        print("audio: oracle has no audio.f32 (run decode_frames_codec.py)")
else:
    print("audio: not compared (pass the live .f32 as the third argument)")

print("M2 REPORT: text %s, mode kinds %s"
      % ("exact" if ok_text and not tb else "DIFFER",
         "exact" if not kind_bad else "DIFFER"))
