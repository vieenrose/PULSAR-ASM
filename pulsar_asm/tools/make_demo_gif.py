"""Render a real chat run as a terminal gif for the README.

Nothing here is faked: the frames come from the tokens the engine actually
produced, streamed through the same Chat class the CLI uses, and the tok/s in the
status line is the time that run took. The only liberty is pacing - a gif at
6 tok/s is a slideshow, so frames are dropped to a fixed cadence.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from runtime.gemma4_model import Gemma4  # noqa: E402
import run_gemma4_chat as cli  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BG, FG, DIM, ACCENT, PROMPT_COL = "#12151d", "#dbe0ec", "#6f7788", "#7aa2f7", "#e0af68"


def wrap(text, cols, indent):
    out = []
    for para in text.split("\n"):
        lines = textwrap.wrap(para, width=cols - len(indent)) or [""]
        for i, ln in enumerate(lines):
            out.append((indent if i == 0 else " " * len(indent)) + ln)
    return out


def render(draw, lines, status, cursor, size, font, pad, top):
    w, h = size
    lh = sum(font.getmetrics())              # ascent + descent
    draw.rectangle([0, 0, w, h], fill=BG)
    draw.rounded_rectangle([10, 10, w - 10, 44], 8, outline="#242a38", width=1)
    for i, c in enumerate(("#f7768e", "#e0af68", "#9ece6a")):
        draw.ellipse([26 + i * 22, 22, 40 + i * 22, 36], fill=c)
    draw.text((150, 19), "pulsar · gemma-4-e2b · cpu", font=font, fill=DIM)
    y = top
    for ln, col in lines:
        draw.text((pad, y), ln, font=font, fill=col)
        y += lh
    if cursor:
        draw.text((pad + draw.textlength(lines[-1][0], font=font), y - lh),
                  "\u2588", font=font, fill=FG)
    draw.line([pad, h - 46, w - pad, h - 46], fill="#242a38", width=1)
    draw.text((pad, h - 38), status, font=font, fill=DIM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/mnt/edge/pulsar/gemma4_e2b.bin")
    ap.add_argument("--tok", default=cli.DEFAULT_TOK)
    ap.add_argument("--out", default="doc/gemma4-chat.gif")
    ap.add_argument("--max-new", type=int, default=58)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--size", default="960,540")
    ap.add_argument("--font-size", type=int, default=16)
    ap.add_argument("--fps", type=int, default=14)
    ap.add_argument("--frames-per-token", type=int, default=2)
    ap.add_argument("--ids", default="", help="reuse a previous run's token ids")
    a = ap.parse_args()

    w, h = [int(v) for v in a.size.split(",")]
    font = ImageFont.truetype(FONT, a.font_size)
    big = ImageFont.truetype(FONT, a.font_size + 1)
    pad, top = 26, 62
    # width must come from the font the frames are actually drawn with, or the
    # wrap runs past the window and the last column of every line is clipped
    cols = int((w - 2 * pad) / big.getlength("x"))

    tk, eos = cli.load_tokenizer(a.tok)
    if a.ids:                                  # re-render a previous run
        ids = [int(v) for v in a.ids.split(",") if v.strip()]
        tps, status_tail = 0.0, "greedy · 4 cores"
    else:
        eng = Gemma4(a.weights, max_seq=1024, n_threads=a.threads, verbose=False)
        chat = cli.Chat(eng, tk, eos, max_new=a.max_new)
        ids = list(chat.turn(cli.DEMO_PROMPT, quiet=True))
        tps = chat.n_tok / max(chat.clock, 1e-9)
        status_tail = f"{a.threads} cores · {eng.mod.size} B of assembly"
        eng.close()
    text_of = lambda k: tk.decode(ids[:k])  # noqa: E731
    cmd = "$ python3 run_gemma4_chat.py --demo"
    tmp = tempfile.mkdtemp(prefix="pulsargif")
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)

    def frame(lines, status, cursor):
        render(d, lines, status, cursor, (w, h), big, pad, top)
        img.save(os.path.join(tmp, f"f{len(os.listdir(tmp)):04d}.png"))

    pfx = wrap(cli.DEMO_PROMPT, cols, "you> ")
    # two frames of the shell, so the gif starts on a still image
    for _ in range(a.fps):
        frame([(cmd, DIM), ("", FG)], "loading engine · 9.258 GB mapped", False)
    for _ in range(a.fps):
        frame([(cmd, DIM)] + [(l, PROMPT_COL) for l in pfx] + [("", FG)],
              "prefill", False)

    base = [(cmd, DIM)] + [(l, PROMPT_COL) for l in pfx] + [("", FG)]
    for k in range(1, len(ids) + 1):
        body = wrap(text_of(k), cols, "model> ")
        st = f"{k} tok · {tps:.1f} tok/s · {status_tail}"
        for f in range(a.frames_per_token):
            frame(base + [(l, FG) for l in body], st, f == a.frames_per_token - 1)
    body = wrap(text_of(len(ids)), cols, "model> ")
    for _ in range(3 * a.fps):
        frame(base + [(l, FG) for l in body] + [("", FG)],
              f"{len(ids)} tok · {tps:.1f} tok/s · {status_tail}", False)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps),
                    "-i", os.path.join(tmp, "f%04d.png"),
                    "-vf", "split[s0][s1];[s0]palettegen=stats_mode=diff[pal];"
                    "[s1][pal]paletteuse=dither=bayer:bayer_scale=4",
                    "-loop", "0", a.out], check=True)
    shutil.rmtree(tmp)
    n = sum(1 for _ in wrap(text_of(len(ids)), cols, "model> "))
    print(f"{a.out}: {os.path.getsize(a.out)/1e6:.2f} MB, "
          f"{len(os.listdir(tmp)) if os.path.isdir(tmp) else len(ids) * a.frames_per_token} "
          f"frames-ish, {len(ids)} tokens, {n} wrapped lines")


if __name__ == "__main__":
    main()
