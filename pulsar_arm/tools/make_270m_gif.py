"""Render a real 270m chat run as a terminal gif (E2B fashion, 270m content).

Nothing in the frames is faked: prompt, tpl ids, and response are the exact
bytes from a real `./core ... c ...` run on Pi (see transcript below); tok/s
is the engine's measured decode rate from the same session. Only pacing is
libertied (fixed cadence reveal, as in pulsar_asm's make_demo_gif.py).

Transcript source: /tmp/gifdemo2.log on Pi ("Hello", temp 1.0, top-p 0.95,
cap 48). Response hit the gen cap (ends mid-stream, shown as-is).
"""

import os
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

BG, FG, DIM, PROMPT_COL = "#12151d", "#dbe0ec", "#6f7788", "#e0af68"
TITLE = "pulsar \u00b7 gemma-3-270m \u00b7 cpu"
CMD = "$ printf 'Hello\\n' | ./core model.safetensors vocab.bin 2 c bpe.bin 1000 950 48"
YOU = "> Hello"
TPL = "tpl: 9 105 2364 107 9259 106 107 105 4368 107"
MODEL = "model> "
RESPONSE = (
    "This is an initial message is 'best_\n"
    "\n"
    "The best _\n"
    "\n"
    "It is a.\n"
    "I\n"
    "Here is a great,\n"
    "I am\n"
    "The \n"
    "the\n"
    "I am a person of \n"
    "A strong\n"
    "The\n"
    "I"
)
STATUS = "6.9 tok/s \u00b7 temp 1.0 \u00b7 3 cores"


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        "doc", "gemma3-270m-chat-en.gif")
    w, h = 1100, 560
    font = ImageFont.truetype(MONO, 17)
    lh = sum(font.getmetrics())
    pad, top = 26, 62
    text_w = w - 2 * pad

    # wrap response to pixel width (greedy, keeps embedded newlines)
    lines = []
    for para in RESPONSE.split("\n"):
        line, lw = "", 0.0
        for word in para.split(" "):
            unit = ("" if not line else " ") + word
            uw = font.getlength(unit)
            if line and lw + uw > text_w - font.getlength(MODEL):
                lines.append(line)
                line, lw = word, font.getlength(word)
            else:
                line, lw = line + unit, lw + uw
        lines.append(line)

    tmp = tempfile.mkdtemp(prefix="pulsar270m")
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)

    def frame(rows, status, cursor):
        d.rectangle([0, 0, w, h], fill=BG)
        d.rounded_rectangle([10, 10, w - 10, 44], 8, outline="#242a38", width=1)
        for i, c in enumerate(("#f7768e", "#e0af68", "#9ece6a")):
            d.ellipse([26 + i * 22, 22, 40 + i * 22, 36], fill=c)
        d.text((150, 19), TITLE, font=font, fill=DIM)
        y = top
        for text, xoff, col in rows:
            if text:
                d.text((pad + xoff, y), text, font=font, fill=col)
            y += lh
        if cursor and rows:
            text, xoff, _ = rows[-1]
            d.text((pad + xoff + d.textlength(text, font=font), y - lh),
                   "\u2588", font=font, fill=FG)
        d.line([pad, h - 46, w - pad, h - 46], fill="#242a38", width=1)
        d.text((pad, h - 38), status, font=font, fill=DIM)
        img.save(os.path.join(tmp, f"f{len(os.listdir(tmp)):04d}.png"))

    fps = 14
    head = [(CMD, 0.0, DIM)]
    for _ in range(fps):
        frame(head + [("", 0.0, FG)], "loading engine", False)
    you_rows = head + [(YOU, 0.0, PROMPT_COL)]
    for _ in range(fps):
        frame(you_rows + [("", 0.0, FG)], "prefill", False)
    tpl_rows = you_rows + [(TPL, 0.0, DIM)]
    for _ in range(fps // 2):
        frame(tpl_rows + [("", 0.0, FG)], "prefill", False)

    mid_px = font.getlength(MODEL)
    # reveal word by word (token-ish cadence); track (line-idx, words-shown)
    shown = []
    words = RESPONSE.split(" ")
    # build reveal states: cumulative word counts mapped onto wrapped lines is
    # fiddly; simpler: reveal the raw response progressively by words and
    # re-wrap each state (same wrapper => stable layout once complete)
    for k in range(1, len(words) + 1):
        part = " ".join(words[:k])
        blines = []
        for para in part.split("\n"):
            line, lw = "", 0.0
            for word in para.split(" "):
                unit = ("" if not line else " ") + word
                uw = font.getlength(unit)
                if line and lw + uw > text_w - mid_px:
                    blines.append(line)
                    line, lw = word, font.getlength(word)
                else:
                    line, lw = line + unit, lw + uw
            blines.append(line)
        body = [(MODEL + blines[0], 0.0, FG)] + [(t, mid_px, FG) for t in blines[1:]]
        rows = head + [(YOU, 0.0, PROMPT_COL)] + [("", 0.0, FG)] + body
        # keep on screen: drop oldest body lines if overflowing (approx)
        max_rows = (h - 46 - top) // lh - 4
        if len(rows) > max_rows:
            rows = rows[:3] + rows[3 + (len(rows) - max_rows):]
        for f in range(2):
            frame(rows, STATUS, f == 1)
    for _ in range(3 * fps):
        body = [(MODEL + lines[0], 0.0, FG)] + [(t, mid_px, FG) for t in lines[1:]]
        rows = head + [(YOU, 0.0, PROMPT_COL)] + [("", 0.0, FG)] + body
        max_rows = (h - 46 - top) // lh - 4
        if len(rows) > max_rows:
            rows = rows[:3] + rows[3 + (len(rows) - max_rows):]
        frame(rows, STATUS, False)

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", os.path.join(tmp, "f%04d.png"),
                    "-vf", "split[s0][s1];[s0]palettegen=stats_mode=diff[pal];"
                    "[s1][pal]paletteuse=dither=bayer:bayer_scale=4",
                    "-loop", "0", out], check=True)
    import shutil
    shutil.rmtree(tmp)
    print(f"{out}: {os.path.getsize(out)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
