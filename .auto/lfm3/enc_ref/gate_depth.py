"""gate_depth.py -- S4 gate against the HF oracle (run on the board with python3 -I).

Usage: gate_depth.py <oracle_dir> <engine_bin> <blob>
Teacher-forces the oracle's reveal order and codes into the engine one pass at a time,
then checks (a) per-pass logit rel err, (b) top-1 agreement of the penalised argmax at
every revealed codebook, (c) full 12-codebook stream equality per frame.
"""
import json, os, subprocess, sys
from collections import deque
import numpy as np

G, ENG, BLOB = sys.argv[1], sys.argv[2], sys.argv[3]
W = "/tmp/dgate"
os.makedirs(W, exist_ok=True)
man = json.load(open(os.path.join(G, "manifest.json")))
N, sched = man["nframes"], man["schedule"]
PEN, WIN = man["repetition_penalty"], man["repetition_window"]
NCB, CBS = 12, 2017
hist = deque(maxlen=WIN)
dec_total = dec_agree = 0
rel_max = 0.0
frames_ok = 0
flips = []
for i in range(N):
    codes_final = np.fromfile(os.path.join(G, f"codes.{i}.i32"), dtype=np.int32)
    order = np.fromfile(os.path.join(G, f"order.{i}.i32"), dtype=np.int32).reshape(len(sched), -1)
    hid = os.path.join(G, f"hidden.{i}.f32")
    revealed = np.zeros(NCB, dtype=bool)
    chosen_all = np.zeros(NCB, dtype=np.int32)
    for p in range(len(sched)):
        state = np.where(revealed, codes_final, 0).astype(np.int32)
        masked = (~revealed).astype(np.uint8)
        state.tofile(os.path.join(W, "s.i32"))
        masked.tofile(os.path.join(W, "m.u8"))
        r = subprocess.run([ENG, BLOB, hid, os.path.join(W, "s.i32"), os.path.join(W, "m.u8"),
                            os.path.join(W, "o.f32")], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if r.returncode != 0:
            sys.exit(f"engine failed frame {i} pass {p}: rc={r.returncode} {r.stderr[-400:]}")
        eng = np.fromfile(os.path.join(W, "o.f32"), dtype=np.float32).reshape(NCB, CBS)
        ref = np.fromfile(os.path.join(G, f"logits.{i}.{p}.f32"), dtype=np.float32).reshape(NCB, CBS)
        rel = np.sqrt(((eng - ref) ** 2).mean()) / np.sqrt((ref ** 2).mean())
        rel_max = max(rel_max, rel)
        for idx in order[p]:
            s = eng[idx].astype(np.float32).copy()
            if hist:
                cb = np.array(hist)[:, idx].reshape(-1)
                g = s[cb]
                s[cb] = np.where(g < 0, g * PEN, g / PEN)
            c = int(s.argmax())
            chosen_all[idx] = c
            dec_total += 1
            dec_agree += int(c == int(codes_final[idx]))
            # oracle-side penalised scores, for the near-tie analysis of every flip
            ro = ref[idx].astype(np.float32).copy()
            if hist:
                ro[cb] = np.where(ro[cb] < 0, ro[cb] * PEN, ro[cb] / PEN)
            top2 = np.sort(ro)[-2:]
            gap = float(top2[1] - top2[0])
            if c != int(codes_final[idx]):
                flips.append((i, p, int(idx), gap, float(np.abs(eng[idx] - ref[idx]).max()),
                              float(np.abs(ref[idx]).max()), int(c), int(codes_final[idx])))
        revealed[order[p]] = True
        print(f"frame {i} pass {p}: logit rel {rel:.3e}", flush=True)
    ok = bool((chosen_all == codes_final).all())
    frames_ok += int(ok)
    hist.append(codes_final)
    print(f"frame {i}: stream {'EQUAL' if ok else 'DIFFERS'} codes={codes_final.tolist()}", flush=True)
print(f"GATE a: max per-pass logit rel err {rel_max:.3e}")
print(f"GATE b: top-1 agreement {dec_agree}/{dec_total}")
print(f"GATE c: frame stream equality {frames_ok}/{N}")
for (fi, p, idx, gap, maxerr, maxabs, ec, rc) in flips:
    print(f"FLIP frame {fi} pass {p} cb {idx}: oracle top1-top2 gap {gap:.4f}, "
          f"max|eng-ref| {maxerr:.4f} (max|ref| {maxabs:.2f}), engine {ec} vs oracle {rc}")
