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
    return json.load(open(os.path.join(HERE, "golden_greedy_%s.json" % tag)))


def gate_matrix():
    ids = ids_of("A")["prompt_ids"]
    out = run_remote("cd %s && timeout 1800 ./%s %s --steps %d --prompt %s --trace 2>&1"
                     % (DIR, BIN, BLOB, len(ids), ",".join(map(str, ids))))
    eng = {}
    for l in out.splitlines():
        m = re.match(r"POS\s+(\d+)\s+L(\d+)\s+(\S+)\s+rms\s+([-\d.]+)", l)
        if m:
            eng[(int(m.group(1)), int(m.group(2)))] = float(m.group(4))
    hf = np.load(os.path.join(HERE, "hf_matrix_A.npy"))
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
    b = np.load(os.path.join(HERE, "hf_pos0_logits.npy"))
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


def gate_logits_seq(n=23, k=8):
    """Per-position top-1 / top-k coverage / correlation / KL vs HF, over the
    whole teacher-forced prefix of prompt A.

    Why this gate exists: the greedy stream gate scores 1-2 matching tokens out
    of 16 on every prompt and 0/7 prompts fully match, so it is nearly blind to
    summation-order changes -- the __fmaf_rn switch that bought -2.0% left all
    16 stream tokens unchanged. Position-wise continuous metrics turn a cliff
    into a slope: a regression shows up as a shift in top-1 agreement, top-k
    coverage, correlation or KL, where a stream prefix cannot move.

    The thresholds are a REGRESSION RATCHET measured on the shipped arm, not a
    quality claim: passing means 'no worse than the arm this gate was written
    on'. Every number is printed on every run, so drift is visible even when it
    passes. HF is teacher forced on ids[:p+1], the same convention the matrix
    gate uses; the engine's --dump-logits-all appends one VOCAB-float vector per
    decode step, so row p is position p.
    """
    ref = os.path.join(HERE, "hf_logits_A_seq.npy")
    if not os.path.exists(ref):
        FAIL.append("logits-seq: %s missing (run .auto/lfm3/hf_logits_seq.py)" % ref)
        return
    hf = np.load(ref)
    n = min(n, hf.shape[0])
    ids = ids_of("A")["prompt_ids"][:n]
    run_remote("cd %s && rm -f lgall.bin" % DIR)
    out = run_remote("cd %s && timeout 1800 ./%s %s --steps %d --prompt %s "
                     "--dump-logits-all lgall.bin >/dev/null 2>&1; ls -l lgall.bin 2>&1"
                     % (DIR, BIN, BLOB, n, ",".join(map(str, ids))))
    if "lgall.bin" not in out:
        FAIL.append("logits-seq: no dump produced (%s)" % out.strip()[:80]); return
    subprocess.run(["scp", "-o", "ConnectTimeout=20", "-q", "%s:%s/lgall.bin" % (NANO, DIR),
                    "/tmp/lgall.bin"], check=False, timeout=300)
    a = np.fromfile("/tmp/lgall.bin", dtype=np.float32)
    if a.size != n * hf.shape[1]:
        FAIL.append("logits-seq: dump is %d floats, expected %d (%d positions x vocab %d)"
                    % (a.size, n * hf.shape[1], n, hf.shape[1])); return
    eng = a.reshape(n, hf.shape[1])

    def softmax(v):
        m = v.max(-1, keepdims=True)
        e = np.exp(v - m)
        return e / e.sum(-1, keepdims=True)

    same, cov, corrs, kls, worst = 0, 0, [], [], None
    for p in range(n):
        e, h = eng[p], hf[p]
        te, th = int(e.argmax()), int(h.argmax())
        same += int(te == th)
        # HF's top-1 inside the engine's top-k, and the reverse: catches a
        # near-tie that top-1 alone reports as a hard miss.
        cov += int(th in np.argpartition(e, -k)[-k:]) + int(te in np.argpartition(h, -k)[-k:])
        corrs.append(float(np.corrcoef(e, h)[0, 1]))
        pe, ph = softmax(e), softmax(h)
        kls.append(float(np.sum(ph * (np.log(ph + 1e-30) - np.log(pe + 1e-30)))))
        if worst is None or kls[-1] > worst[1]:
            worst = (p, kls[-1], te, th)
    ntop, ncov, mcorr, mkl = same / n, cov / (2 * n), sum(corrs) / n, sum(kls) / n
    # Floors sit one step below a measured clean pass, so a single lost top-1
    # position warns and two fails. See the measured line printed below.
    if same < n - 1:
        FAIL.append("logits-seq: top-1 agreement %d/%d positions (worst KL %.4f at pos %d, "
                    "engine %d vs HF %d)" % (same, n, worst[1], worst[0], worst[2], worst[3]))
    if ncov < 1.0:
        FAIL.append("logits-seq: top-%d coverage %.3f < 1.0" % (k, ncov))
    if mcorr < 0.999:
        FAIL.append("logits-seq: mean corr %.5f < 0.999 (min %.5f)" % (mcorr, min(corrs)))
    if mkl > 0.05:
        FAIL.append("logits-seq: mean KL %.4f > 0.05 (max %.4f at pos %d)"
                    % (mkl, max(kls), int(np.argmax(kls))))
    print("  logits-seq: %d/%d top-1, top-%d cov %.3f, corr %.5f (min %.5f), "
          "KL %.4f (max %.4f @pos %d)"
          % (same, n, k, ncov, mcorr, min(corrs), mkl, max(kls), int(np.argmax(kls))))


def gate_streams(prompts, gen=16, floor=1):
    """Greedy stream vs the HF goldens.

    The goldens (written by .auto/lfm3/oracle.py) store the DECODED TEXT --
    `proc.tokenizer.decode(gen)` -- not ids, and the engine has no
    detokenizer, so the comparison is: engine ids -> tokenizer.decode ->
    character prefix against the golden text. floor=1 is the arm's certified
    worst prefix (q8_0_embq8 = 6/8 exact, worst prefix 1 from
    quant_sensitivity.py); it catches a collapse, not a quantization cost.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ.get(
        "LFM3_HF", "/home/louis/models/lfm3/LFM3-Audio-2B"), trust_remote_code=True)
    print("  streams (gen=%d):" % gen)
    n_exact = 0
    for p in prompts:
        g = ids_of(p)
        if "prompt_ids" not in g:
            # Only A/B carry prompt_ids; the rest store prompt text (the
            # ids need the chat template + the <|text_start|> modality tag
            # from oracle.py's build_ids, i.e. the processor). Score what is
            # machine-checkable and say what was skipped, rather than
            # crashing mid-gate -- a gate that dies at prompt C reports
            # nothing about D..L.
            print("    %s: SKIPPED (golden has no prompt_ids; needs oracle.build_ids)" % p)
            continue
        out = run_remote("cd %s && timeout 1800 ./%s %s --steps %d --prompt %s 2>&1"
                         % (DIR, BIN, BLOB, len(g["prompt_ids"]) + gen - 1,
                            ",".join(map(str, g["prompt_ids"]))))
        # Step lines are "  <step>  <token>  <ms>  <tok/s>", and the token
        # column is the token FED at that step. The harness feeds
        # prompt[0..n-2] as teacher-forcing steps (prompt[n-1] is consumed by
        # the initial forward), so the GENERATED stream starts at index n-1.
        # Slicing at n drops gen[0] (207 = "\n" on prompt A) and makes every
        # prompt score a 0-token prefix; slicing at 0 decodes the prompt and
        # shows a stray <|startoftext|> mid-string. Both were hit here.
        toks = [int(m.group(2)) for m in
                (re.match(r"\s*(\d+)\s+(\d+)\s+[\d.]+\s+[\d.]+$", l) for l in out.splitlines())
                if m]
        got = toks[len(g["prompt_ids"]) - 1:]
        # Token-level prefix, the unit quant_sensitivity.py certifies in
        # ("q8_0_embq8: 6/8 exact, worst prefix 1"). The golden file stores
        # decoded text, so re-encode it: the tokenizer round-trips its own
        # decode exactly, and comparing characters is fragile (a leading "\n"
        # token drops out of one side's decode and every prompt then scores 0
        # matched characters, which is what the first version of this gate
        # did).
        want_ids = tok.encode(g["text"], add_special_tokens=False)
        pre = 0
        for a, b in zip(got[:gen], want_ids):
            if a != b:
                break
            pre += 1
        ok = pre >= floor
        n_exact += int(pre >= len(want_ids[:gen]))
        print("    %s: prefix %2d/%2d tokens %s  %r" % (
            p, pre, min(gen, len(want_ids)), "OK" if ok else "FAIL",
            tok.decode(got[:gen])[:40]))
        if not ok:
            FAIL.append("stream %s: token prefix %d < %d (engine %s vs golden %s)"
                        % (p, pre, floor, got[:gen][:5], want_ids[:5]))
    print("  streams: %d/%d prompts fully match (%d gen tokens each)" % (n_exact, len(prompts), gen))


if __name__ == "__main__":
    prompts = (sys.argv[1:] or ["A", "B"])
    print("== gate: %s / %s ==" % (BLOB, BIN))
    gate_matrix()
    gate_logits0()
    gate_logits_seq()
    gate_streams(prompts)
    if FAIL:
        print("\nCHECKS FAILED (%d):" % len(FAIL))
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("\nCHECKS OK")
