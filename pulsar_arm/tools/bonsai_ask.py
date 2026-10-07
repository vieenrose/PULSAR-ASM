#!/usr/bin/env python3
"""Ask a Bonsai blob a question and print what the engine actually rendered.

This is the demo pipeline in one place, and it is deliberately dumb: tokenise
locally (the asm encoder is gemma's SentencePiece BPE and cannot do Qwen's
regex-pretokenised byte-level BPE), pipe ids to the engine's raw-ids mode, and
print the engine's own output bytes. Nothing here rewrites or prettifies the
answer - make_chat_gif.py takes these transcripts as literals, so anything this
tool prints is what can appear on screen.

  bonsai_ask.py "List the four seasons..."                # transcript
  bonsai_ask.py --timed "..."                             # + ms/token
  bonsai_ask.py --json "..."                              # for the GIF specs

Long generations over ssh are the demo's least reliable part: the first hunt
lost 2 of 5 runs to "Connection reset by peer" when the Pi was also running a
4B oracle. The ssh calls set ServerAliveInterval so a slow token stream is not
mistaken for a dead peer.

--timed runs the engine's own bench (mode 2), which reports ms/token on the same
binary and weights; the demo status bars quote that number rather than a wall
clock measured around ssh.
"""
import argparse
import json
import re
import subprocess
import sys
import time

REPO = "prism-ml/Ternary-Bonsai-1.7B-unpacked"


def tokenise(prompt, repo=REPO):
    """Qwen chat template -> ids, on this side of the wire."""
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(repo)
    text = tk.apply_chat_template([{"role": "user", "content": prompt}],
                                  tokenize=False, add_generation_prompt=True)
    return tk.encode(text)


def run_engine(ids, host, cwd, model, vocab, temp, topp, gencap, timeout):
    cmd = (f"cd {cwd} && echo '{ids}' | timeout {timeout} ./core_ids "
           f"{model} {vocab} 2 i x {temp} {topp}")
    if gencap:
        cmd += f" {gencap}"
    out = subprocess.run(["ssh", "-o", "BatchMode=yes",
                          "-o", "ServerAliveInterval=30",
                          "-o", "ServerAliveCountMax=20",
                          host, cmd],
                         capture_output=True, text=True, timeout=timeout + 60)
    return out.stdout + out.stderr


def parse_text(log):
    """Everything between the prompt echo and the next '> ' prompt."""
    lines, grab = [], False
    for ln in log.splitlines():
        if ln.startswith("> ids:") or ln.startswith("> tpl:"):
            grab = True
            continue
        if grab and ln.startswith("> "):
            break
        if grab:
            lines.append(ln)
    return "\n".join(lines).strip("\n")


def bench_ms_per_token(host, cwd, model, vocab, timeout):
    cmd = f"cd {cwd} && timeout {timeout} ./core_ids {model} {vocab} 2"
    out = subprocess.run(["ssh", "-o", "BatchMode=yes",
                          "-o", "ServerAliveInterval=30",
                          "-o", "ServerAliveCountMax=20",
                          host, cmd],
                         capture_output=True, text=True, timeout=timeout + 60)
    m = re.search(r"ms/token:\s*([0-9.]+)", out.stdout + out.stderr)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--host", default="luigi@raspberrypi.tailf63b31.ts.net")
    ap.add_argument("--cwd", default="pw")
    ap.add_argument("--model", default="/home/luigi/bonsai_b3.bin")
    ap.add_argument("--vocab", default="/home/luigi/qwen_vocab.bin")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--temp", default="500", help="x1000, Bonsai card says 500")
    ap.add_argument("--topp", default="850", help="x1000, card says 850")
    ap.add_argument("--gencap", default="")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--timed", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    ids = tokenise(a.prompt, a.repo)
    log = run_engine(ids, a.host, a.cwd, a.model, a.vocab,
                     a.temp, a.topp, a.gencap, a.timeout)
    text = parse_text(log)
    ms = bench_ms_per_token(a.host, a.cwd, a.model, a.vocab, a.timeout) if a.timed else None

    if a.json:
        print(json.dumps({"prompt": a.prompt, "ids": ids, "text": text,
                          "ms_per_token": ms}, ensure_ascii=False, indent=1))
    else:
        print(f"PROMPT: {a.prompt}")
        print(f"IDS: {' '.join(map(str, ids))}")
        if ms:
            print(f"RATE: {ms:.0f} ms/token ({1000.0 / ms:.2f} tok/s)")
        print(f"TEXT:\n{text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
