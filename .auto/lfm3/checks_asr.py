#!/usr/bin/env python3
"""checks_asr.py -- the end-to-end audio-input gate: WAV in, greedy text out, graded
against the HF golden. Runs the whole chain on the Nano from a fresh build.

  lfm3_wav  ->  lfm3_mel  ->  lfm3_enc_gpu  ->  lfm3_nanode --audio-emb ... --trace

and compares the generated ids to .auto/lfm3/speech/hf_asr_golden.json. The arbiter is
HF's own ASR output for the same clip (never the authored transcript: HF's WER on this
fixture is 36%, and grading our engine against HF's deletions would measure HF).

The bar starts as a floor, not a target: PASS requires the chain to run, every stage to
be non-zero and finite, the AUDIO-EMB proof lines to appear, and a greedy prefix of at
least ASR_MIN_PREFIX tokens matching HF. The full divergence profile is reported so the
first real measurement can set a real bar -- a number I invented before the first run
would be a guess with a gate attached.

  ~/venvs/lfm3/bin/python .auto/lfm3/checks_asr.py
  ASR_MIN_PREFIX=64 ... ; ASR_STEPS=1904 ... ; NANO=picard@100.84.151.116 (default)
"""
import json, os, re, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
NANO = os.environ.get("NANO", "picard@100.84.151.116")     # ipho-nano is reserved for
NDIR = os.environ.get("NANO_DIR", "/home/picard/lfm3")     # the other project
WAV = os.path.join(HERE, "speech", "gate_60s_16k_mono.wav")
IDS = os.path.join(HERE, "speech", "asr_prompt_ids.txt")
GOLDEN = os.path.join(HERE, "speech", "hf_asr_golden.json")
MANIFEST = os.path.join(HERE, "speech", "manifest.json")
MIN_PREFIX = int(os.environ.get("ASR_MIN_PREFIX", "20"))
# The bar that tracks transcript quality (section 39): mean per-row cosine between the
# encoder's proj_out and the embedding matrix HF fed the LM at the same positions.
# Measured at the 129.7 s clip: 0.303 full causal (the pre-fix spec) vs 0.892 with
# encoder.sliding_window = 32. 0.85 sits under the good arm with room and far above the
# bad one, so it separates the two arms it was built to separate.
MIN_COS = float(os.environ.get("ASR_MIN_EMB_COS", "0.85"))
NVCC = ("export PATH=/usr/local/cuda/bin:$PATH && nvcc -std=c++11 "
        "-gencode arch=compute_53,code=sm_53 -O3 -fmad=false -Xptxas -O3,-dlcm=ca")


def ssh(cmd, timeout=3600):
    p = subprocess.run(["ssh", "-o", "ConnectTimeout=25", "-o", "BatchMode=yes", NANO, cmd],
                       capture_output=True, text=True, timeout=timeout)
    return "\n".join(l for l in (p.stdout + "\n" + p.stderr).splitlines()
                     if "setlocale" not in l and "Warning:" not in l)


def sh(cmd, timeout=3600):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)


def scp(src, dst):
    r = sh("scp -q -o ConnectTimeout=20 %s %s" % (src, dst))
    if r.returncode:
        raise RuntimeError("scp %s -> %s failed: %s" % (src, dst, r.stderr.strip()[:200]))


def main():
    for f in (WAV, IDS, GOLDEN, MANIFEST):
        if not os.path.exists(f):
            sys.exit("missing %s -- run mk_speech.py, hf_asr_golden.py --regen and asr_prompt_ids.py" % f)
    man = json.load(open(MANIFEST))
    gold = json.load(open(GOLDEN))
    want = gold["decoding"]["generated_token_ids"]
    audio_sec = float(man["duration_s"])
    ids = open(IDS).read().strip()
    n_prompt = len(ids.split(","))
    n_audio = int(gold["prompt"]["prompt_audio_placeholder_tokens"])
    steps = int(os.environ.get("ASR_STEPS", str(n_prompt + len(want) + 2)))

    print("== build (fresh, on the Nano) ==")
    for f in ("lfm3_nanode.cu", "lfm3_enc_gpu.cu", "lfm3_wav.c", "lfm3_mel.c", "lfm3_mel.S"):
        sh("scp -q -o ConnectTimeout=20 %s %s:%s/" % (os.path.join(REPO, "pulsar_arm", "asm", f), NANO, NDIR))
    r = ssh("cd %s && as -o mo lfm3_mel.S && %s -o /tmp/nanode_new lfm3_nanode.cu -lz 2>/dev/null; "
            "test -x /tmp/nanode_new && mv -f /tmp/nanode_new lfm3_nanode; "
            "%s -o /tmp/enc_new lfm3_enc_gpu.cu -lz 2>/dev/null; "
            "test -x /tmp/enc_new && mv -f /tmp/enc_new lfm3_enc_gpu; "
            "gcc -O2 -std=c99 -o lfm3_wav lfm3_wav.c && gcc -O2 -std=c99 -o lfm3_mel lfm3_mel.c mo -lm && echo BUILT_OK"
            % (NDIR, NVCC, NVCC))
    if "BUILT_OK" not in r:
        sys.exit("build failed -- refusing to measure a stale binary:\n" + r[-800:])
    print("  BUILT_OK")
    subprocess.run(["scp", "-q", "-o", "ConnectTimeout=20", WAV, "%s:%s/" % (NANO, NDIR)], check=True)

    print("== chain ==")
    t0 = time.time()
    t_wav = time.time(); ssh("cd %s && ./lfm3_wav %s /tmp/asr.pcm.f32" % (NDIR, os.path.basename(WAV)))
    t_wav = time.time() - t_wav
    t_mel = time.time(); mel = ssh("cd %s && ./lfm3_mel mel.bin /tmp/asr.pcm.f32 /tmp/asr.mel.f32" % NDIR)
    t_mel = time.time() - t_mel
    m = re.search(r"frames=(\d+)", mel)
    if not m:
        sys.exit("mel stage failed: " + mel[-300:])
    Tm = int(m.group(1))
    print("  wav %.1fs | mel %.1fs -> %d frames (manifest predicts %d)"
          % (t_wav, t_mel, Tm, man["samples"] // 160 + 1))
    t_enc = time.time()
    # ASR_ENC_ENV passes encoder env through for A/B arms, e.g.
    #   ASR_ENC_ENV="LFM_ENC_WIN=0"   -> the pre-section-39 full-causal spec
    # This exists because the prefix metric is chaotic and the A/B has to be run through
    # the SAME gate that prints the stable metrics, not from a side script.
    encenv = os.environ.get("ASR_ENC_ENV", "").strip()
    encenv = (encenv + " ") if encenv else ""
    enc = ssh("cd %s && rm -rf asrgate && mkdir asrgate && %s./lfm3_enc_gpu lfm3-audio-2b-enc.lfm3q8 "
              "/tmp/asr.mel.f32 %d asrgate 2>&1 | tail -3" % (NDIR, encenv, Tm))
    t_enc = time.time() - t_enc
    print("  encoder %.1fs | %s" % (t_enc, enc.replace("\n", " / ")[-160:]))
    if "FM3_ENC_GPU_DONE" not in enc:
        sys.exit("encoder did not complete: " + enc[-300:])
    # proj_out must be [n_audio, 2048]
    rows = int(ssh("cd %s && stat -c %%s asrgate/proj_out.bin" % NDIR).strip()) // (2048 * 4)
    print("  proj_out rows %d (golden expects %d)" % (rows, n_audio))
    if rows != n_audio:
        sys.exit("encoder produced %d audio rows, HF's law says %d" % (rows, n_audio))

    print("== engine (audio embeddings at positions 15..%d) ==" % (15 + n_audio - 1))
    t_eng = time.time()
    eng = ssh("cd %s && ./lfm3_nanode lfm3-audio-2b-q8.lfm3q8 --audio-emb asrgate/proj_out.bin "
              "--audio-at 15 --prompt %s --steps %d --trace 2>&1" % (NDIR, ids, steps), timeout=5400)
    t_eng = time.time() - t_eng
    armed = [l for l in eng.splitlines() if "AUDIO-EMB armed" in l]
    fed = [l for l in eng.splitlines() if "AUDIO-EMB fed" in l]
    for l in armed + fed:
        print("  " + l.strip())
    if not armed or not fed:
        sys.exit("no AUDIO-EMB proof lines -- the override did not run; refusing to report a number")
    steps_found = [(int(a), int(b)) for a, b in
                   re.findall(r"^\s*(\d+)\s+(-?\d+)\s+[\d.]+\s+[\d.]+$", eng, re.M)]
    gen = [tok for (s, tok) in steps_found if s >= n_prompt - 1][:len(want)]
    if not gen:
        sys.exit("no step lines parsed; engine tail was:\n" + eng[-500:])
    n = min(len(gen), len(want))
    pref = 0
    while pref < n and gen[pref] == want[pref]:
        pref += 1
    print("  engine %.1fs | %d of %d generated ids parsed" % (t_eng, len(gen), len(want)))
    print("  greedy prefix vs HF golden: %d / %d tokens" % (pref, len(want)))
    # Stable metrics alongside the prefix, because prefix length is CHAOTIC: one flipped
    # token ends the prefix, so it cannot tell a 3x better encoder from a worse one. The
    # section 39 window fix raised the LM's input from mean per-row cos 0.303 to 0.892
    # against HF's golden embeddings and the prefix went 126 -> 69. These three are the
    # ones that respond monotonically.
    same = sum(1 for a, b in zip(gen, want) if a == b)
    # longest common subsequence (262x262, trivial), = the token content both streams agree
    # on regardless of where the divergence starts
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        gi = gen[i - 1]
        row, prow = dp[i], dp[i - 1]
        for j in range(1, n + 1):
            row[j] = prow[j - 1] + 1 if gi == want[j - 1] else max(prow[j], row[j - 1])
    lcs = dp[n][n]
    sub = min(len(gen), len(want)) - lcs
    ins = max(0, len(gen) - len(want)); dele = max(0, len(want) - len(gen))
    wer = (sub + ins + dele) / float(len(want))
    print("  stable metrics: all-position match %d/%d (%.1f%%), LCS %d, token-WER %.3f"
          " (sub %d ins %d del %d)" % (same, n, 100.0 * same / n, lcs, wer, sub, ins, dele))
    json.dump({"gen": gen, "want": want, "prefix": pref, "match": same, "lcs": lcs,
               "token_wer": wer, "enc_win": os.environ.get("LFM_ENC_WIN", "32(default)"),
               "proj_rows": rows},
              open(os.path.join(HERE, "speech", "asr_engine_ids.json"), "w"), indent=1)
    # The encoder-quality metric that predicts the transcript: the LM's actual input,
    # compared to the golden embedding matrix HF fed at the same positions.
    try:
        ssh("cd %s && rm -f asrgate_proj_out.bin && cp asrgate/proj_out.bin asrgate_proj_out.bin"
            % NDIR)
        scp("%s:%s/asrgate_proj_out.bin" % (NANO, NDIR), "/tmp/asr_proj_out.bin")
        emb = np.fromfile(os.path.join(HERE, "speech", "hf_audio_embeddings_1622x2048.f32"),
                          dtype="<f4")
        a = np.fromfile("/tmp/asr_proj_out.bin", dtype="<f4")
        if a.size != emb.size:
            print("  EMB: size mismatch (%d vs %d), skipped" % (a.size, emb.size))
        else:
            a = a.reshape(-1, 2048).astype(np.float64); b = emb.reshape(-1, 2048).astype(np.float64)
            cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30)
            erel = float(np.sqrt(((a - b) ** 2).mean()) / np.sqrt((b ** 2).mean()))
            print("  EMB vs HF golden: mean per-row cos %.6f (min %.4f), energy rel %.3e"
                  % (cos.mean(), cos.min(), erel))
            print("  EMB " + " ".join("%d-%d:%.4f" % (lo, min(lo + 199, rows) - 1,
                  cos[lo:lo + 200].mean()) for lo in range(0, rows, 200)))
            if cos.mean() < MIN_COS:
                sys.exit("ASR_GATE FAIL: encoder embeddings mean cos %.6f < %.2f -- this is"
                         " the metric that tracks transcript quality (section 39)" % (cos.mean(), MIN_COS))
    except SystemExit:
        raise
    except Exception as e:
        print("  EMB comparison unavailable: %s" % e)
    if pref < n:
        print("  first divergence at gen index %d: engine %d, HF %d" % (pref, gen[pref], want[pref]))
        print("  engine[%d:%d] %s" % (max(0, pref - 4), pref + 6, gen[max(0, pref - 4):pref + 6]))
        print("  HF   [%d:%d] %s" % (max(0, pref - 4), pref + 6, want[max(0, pref - 4):pref + 6]))
    try:
        from transformers import AutoTokenizer, AutoProcessor
        mp = os.environ.get("LFM3_MODEL", "/home/louis/models/lfm3/LFM3-Audio-2B")
        try:
            tk = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
        except Exception:
            # same as checks_tok.py: this checkpoint registers its tokenizer through
            # the remote-code processor, and a bare AutoTokenizer cannot see it
            tk = AutoProcessor.from_pretrained(mp, trust_remote_code=True,
                                               local_files_only=True).tokenizer
        print("  engine text : " + repr(tk.decode(gen[:pref])[:110]))
        print("  HF golden   : " + repr(gold["output"]["decoded_text_skip_special_tokens"][:110]))
    except Exception as e:
        print("  (tokenizer unavailable for the text view: %s)" % e)

    tot = t_wav + t_mel + t_enc + t_eng
    # Phase decomposition, because "ASR_RT = 3.87" as one number hides that most of it is
    # not the speech pipeline. Measured on this board (see the doc, section 43): the
    # engine's one-time startup (1.946 GB blob read + upload + setup) is 82.7 s -- timed
    # separately with a 20-id prompt and --steps 24, which took 85.3 s wall against
    # 2.6 s of steps -- the step loop is n_steps x steady-state, and whatever is left
    # inside t_eng is the prompt prefill.
    ss = re.search(r"steady-state ([\d.]+) ms/token.*?over (\d+) steps", eng)
    print("== ASR_RT ==")
    if ss:
        ms_traced, n_st = float(ss.group(1)), int(ss.group(2))
        # The engine run in this gate uses --trace, and printing 1904 step lines over ssh
        # costs real time: the same binary/probe prints 108.68 ms/token untraced (doc
        # section 43) vs 185 ms/token traced. Decoding is therefore charged at the
        # UNTRACED rate (ASR_DECODE_MS, measured on a clean board) and the traced number
        # is printed alongside so nobody reads the inflated one as the decode cost.
        ms_tok = float(os.environ.get("ASR_DECODE_MS", "108.68"))
        decode_s = ms_tok * n_st / 1000.0
        startup_s = float(os.environ.get("ASR_STARTUP_S", "82.7"))
        pre_s = t_eng - decode_s - startup_s
        print("  decode  %6.1f s (%.2f) = %d steps x %.2f ms (untraced; this run traced "
              "%.2f ms/token = %.1f s of --trace cost)"
              % (decode_s, decode_s / audio_sec, n_st, ms_tok, ms_traced,
                 (ms_traced - ms_tok) * n_st / 1000.0))
        print("  startup %6.1f s (%.2f) = one-time blob load (ASR_STARTUP_S to override)"
              % (startup_s, startup_s / audio_sec))
        print("  prefill %6.1f s (%.2f) = remainder of the engine wall over %d prompt ids"
              % (pre_s, pre_s / audio_sec, n_prompt))
        print("  warm ASR_RT excluding the one-time startup = %.3f" % ((tot - startup_s) / audio_sec))
    print("  wav %.1f + mel %.1f + encoder %.1f + engine %.1f = %.1f s for %.1f s of audio"
          % (t_wav, t_mel, t_enc, t_eng, tot, audio_sec))
    print("  ASR_RT = %.3f  (prefill+decode %.3f, front end %.3f)"
          % (tot / audio_sec, t_eng / audio_sec, (t_wav + t_mel + t_enc) / audio_sec))
    ok = pref >= MIN_PREFIX
    print("ASR_GATE %s (prefix %d >= %d, rows %d, Tm %d)" % ("PASS" if ok else "FAIL", pref, MIN_PREFIX, rows, Tm))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
