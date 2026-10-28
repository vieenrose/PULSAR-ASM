#!/usr/bin/env python3
"""Quality gate for the LFM3-Audio-2B Nano engine (arm: linears Q8 + fp32
scales + Q8 embedding, the --emb-host configuration that fits 4 GB).

Three gates, all derived from HF transformers (NOT from the engine), cached in
.auto/lfm3/ so the gate runs without torch:

 1. layer x position residual rms vs HF, prompt A, 24 layers x 23 positions,
    every cell within 3%                       (HF: hidden_states of the bf16
                                                 model, hf_matrix_A.npy)
 2. position-0 logits vs HF: argmax equal and Pearson corr >= 0.999
    (hf_pos0_logits.npy)
 3. greedy stream vs the HF goldens: prefix length per prompt. This arm is
    CERTIFIED AT 6/8 EXACT, WORST PREFIX 1 (quant_sensitivity.py's
    q8_0_embq8 row), so the gate is "no prompt collapses below 1" and the
    exact-match count is reported, not asserted. The 8/8 gate belongs to the
    q8_0 arm (bf16 embedding) -- see LFM3_BLOB=lfm3-audio-2b-cert.lfm3q8.

Any gate failure exits non-zero. A gate failure is information, not something
to work around.
"""
import json, os, re, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))   # .auto/lfm3
NANO = os.environ.get("NANO", "picard@100.84.151.116")
DIR = os.environ.get("NANO_DIR", "/home/picard/lfm3")
BLOB = os.environ.get("LFM3_BLOB", "lfm3-audio-2b-q8.lfm3q8")
BIN = os.environ.get("LFM3_BIN", "lfm3_nanode")
FAIL = []


def run_remote(cmd, timeout=2400):
    p = subprocess.run(["ssh", "-o", "ConnectTimeout=25", "-o", "BatchMode=yes", NANO, cmd],
                       capture_output=True, text=True, timeout=timeout)
    out = "\n".join(l for l in (p.stdout + p.stderr).splitlines()
                    if "setlocale" not in l and "warning:" not in l)
    return out


def ids_of(tag):
    return json.load(open(os.path.join(HERE, "lfm3", "golden_greedy_%s.json" % tag)))


def gate_matrix():
    ids = ids_of("A")["prompt_ids"]
    out = run_remote("cd %s && timeout 1800 ./%s %s --steps %d --prompt %s --trace 2>&1"
                     % (DIR, BIN, BLOB, len(ids), ",".join(map(str, ids))))
    eng = {}
    for l in out.splitlines():
        m = re.match(r"POS\s+(\d+)\s+L(\d+)\s+(\S+)\s+rms\s+([-\d.]+)", l)
        if m:
            eng[(int(m.group(1)), int(m.group(2)))] = float(m.group(4))
    hf = np.load(os.path.join(HERE, "lfm3", "hf_matrix_A.npy"))
    NL, Lp = hf.shape
    if len(eng) < NL * Lp * 0.9:
        FAIL.append("matrix: only %d/%d cells parsed (trace did not run?)" % (len(eng), NL * Lp))
        return
    bad = [(p, L, eng[(p, L)], hf[L, p]) for p in range(Lp) for L in range(NL - 1)
           if (p, L) in eng and not (0.97 <= eng[(p, L)] / hf[L, p] <= 1.03)]
    if bad:
        FAIL.append("matrix: %d/%d cells outside 3%% of HF, first: pos %d layer %d "
                    "engine %.5f vs HF %.5f (ratio %.3f)"
                    % (len(bad), NL * Lp, bad[0][0], bad[0][1], bad[0][2], bad[0][3],
                       bad[0][2] / bad[0][3]))
        print("  matrix: WORST ratio %.3f at pos %d layer %d" % (bad[0][2] / bad[0][3], bad[0][0], bad[0][1]))
    else:
        print("  matrix: %d layers x %d positions all within 3%% of HF" % (NL - 1, Lp))


def gate_logits0():
    out = run_remote("cd %s && rm -f lg0.bin; timeout 900 ./%s %s --steps 1 --prompt %d "
                     "--dump-logits lg0.bin >/dev/null 2>&1; ls -l lg0.bin 2>&1"
                     % (DIR, BIN, BLOB, ids_of("A")["prompt_ids"][0]))
    if "lg0.bin" not in out:
        FAIL.append("logits0: no dump produced (%s)" % out.strip()[:80]); return
    run_remote("cd %s && true" % DIR)
    subprocess.run(["scp", "-o", "ConnectTimeout=20", "-q", "%s:%s/lg0.bin" % (NANO, DIR),
                    "/tmp/lg0.bin"], check=False, timeout=120)
    a = np.fromfile("/tmp/lg0.bin", dtype=np.float32)
    b = np.load(os.path.join(HERE, "lfm3", "hf_pos0_logits.npy"))
    if a.size != b.size:
        FAIL.append("logits0: size %d vs HF %d" % (a.size, b.size)); return
    top_e, top_h = int(a.argmax()), int(b.argmax())
    corr = float(np.corrcoef(a, b)[0, 1])
    if top_e != top_h:
        FAIL.append("logits0: argmax engine %d vs HF %d" % (top_e, top_h))
    if corr < 0.999:
        FAIL.append("logits0: corr %.5f < 0.999 (std %.3f vs %.3f)" % (corr, a.std(), b.std()))
    print("  logits0: top1 %d (HF %d), corr %.5f, std %.3f vs %.3f"
          % (top_e, top_h, corr, a.std(), b.std()))


def gate_streams(prompts, gen=16, floor=1):
    print("  streams (gen=%d):" % gen)
    n_exact = 0
    for p in prompts:
        g = ids_of(p)
        out = run_remote("cd %s && timeout 1800 ./%s %s --steps %d --prompt %s 2>&1 | tail -3"
                         % (DIR, BIN, BLOB, len(g["prompt_ids"]) + gen - 1,
                            ",".join(map(str, g["prompt_ids"]))))
        m = re.search(r"\[([0-9,]+)\]", out)
        got = [int(x) for x in m.group(1).split(",")] if m else []
        want = g["greedy_ids"][:gen]
        pre = 0
        for a, b in zip(got[len(g["prompt_ids"]):], want):
            if a != b:
                break
            pre += 1
        ok = pre >= floor
        n_exact += int(pre == gen)
        print("    %s: prefix %2d/%2d %s" % (p, pre, gen, "OK" if ok else "FAIL"))
        if not ok:
            FAIL.append("stream %s: prefix %d < %d (engine %s vs golden %s)"
                        % (p, pre, floor, got[len(g["prompt_ids"]):][:4], want[:4]))
    print("  streams: %d/%d prompts fully match (%d gen tokens each)" % (n_exact, len(prompts), gen))


if __name__ == "__main__":
    prompts = (sys.argv[1:] or ["A", "B"])
    print("== gate: %s / %s ==" % (BLOB, BIN))
    gate_matrix()
    gate_logits0()
    gate_streams(prompts)
    if FAIL:
        print("\nCHECKS FAILED (%d):" % len(FAIL))
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("\nCHECKS OK")
