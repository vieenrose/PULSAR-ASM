"""Render a real 270m zh chat run as a terminal gif (E2B zh fashion).

Same contract as make_270m_gif.py: prompt, tpl ids, and response are the
exact bytes from a real `./core ... c ...` run on Pi (/tmp/gifdemozh.log:
Chinese prompt, temp 1.0, cap 48). CJK font (WenQuanYi, as in E2B's script)
since DejaVu has no Han glyphs. Reveal is per-character (Chinese has no
spaces to split words on); only pacing is libertied.
"""

import os
import shutil
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

MONO = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"

BG, FG, DIM, PROMPT_COL = "#12151d", "#dbe0ec", "#6f7788", "#e0af68"
TITLE = "pulsar \u00b7 gemma-3-270m \u00b7 CPU \u63a8\u8ad6"
CMD = ("$ printf '\u8acb\u7528\u4e00\u53e5\u8a71\u89e3\u4ec0\u9ebc"
       "\u662f\u91cf\u5b50\u529b\u5b78\u3002\\n' | ./core model.safetensors vocab.bin 2 c bpe.bin 1000 950 48")
YOU = "> \u8acb\u7528\u4e00\u53e5\u8a71\u89e3\u4ec0\u9ebc\u662f\u91cf\u5b50\u529b\u5b78\u3002"
TPL = ("tpl: 20 105 2364 107 239230 237105 122100 238360 237636 236959 26549 "
       "237026 199311 237473 238432 236924 106 107 105 4368 107")
MODEL = "model> "
RESPONSE = (
    "\u9019\u9996\u9019\u500b\u8acb\u60a8\u53ef\u4ee5\u9019\u6a23\u7406\u89e3\u4e86\uff0c\n"
    "\u9019\u500b\u7248\u672c\uff1a\n"
    "\n"
    "\n"
    "\u6211\u5f9e\n"
    "\u6211\uff0c\n"
    "\n"
    "\u9019\u500b\u5b57\u8a5e\u3002\n"
    "\n"
    "\u6211\n"
    "\u4f60\u9078\u64c7\u7684\u9078\u64c7\u53bb\u8b80\u7684\uff0c\u6211\u5c0d\u8a71\u4e86\u6703\u5f8c\ufffd\n"
    "\u8acb\u4f60\u9078\u64c7"
)
STATUS = "6.9 token/s \u00b7 temp 1.0 \u00b7 3 \u6838\u5fc3"


def wrap_para(para, font, limit, indent):
    # greedy fill measured in pixels; CJK breaks per character
    lines, line, w = [], "", 0.0
    i = 0
    while i < len(para):
        ch = para[i]
        uw = font.getlength(ch)
        if line and w + uw > limit:
            lines.append(line)
            line, w = "", 0.0
            continue
        line, w = line + ch, w + uw
        i += 1
    lines.append(line)
    return lines


def wrap(text, font, width, indent_px):
    out = []
    limit = width - indent_px
    for para in text.split("\n"):
        for line in wrap_para(para, font, limit, indent_px):
            out.append((line, indent_px))
    return out or [("", indent_px)]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        "doc", "gemma3-270m-chat-zh.gif")
    w, h = 1100, 560
    font = ImageFont.truetype(MONO, 17)
    lh = sum(font.getmetrics())
    pad, top = 26, 62
    text_w = w - 2 * pad

    tmp = tempfile.mkdtemp(prefix="pulsar270mzh")
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

    def trimmed(rows):
        max_rows = (h - 46 - top) // lh - 4
        if len(rows) > max_rows:
            rows = rows[:3] + rows[3 + (len(rows) - max_rows):]
        return rows

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
    chars = list(RESPONSE)
    for k in range(2, len(chars) + 1, 2):
        part = "".join(chars[:k])
        blines = []
        for para in part.split("\n"):
            blines += wrap_para(para, font, text_w - mid_px, mid_px)
        body = [(MODEL + blines[0], 0.0, FG)] + [(t, mid_px, FG) for t in blines[1:]]
        rows = trimmed(head + [(YOU, 0.0, PROMPT_COL)] + [("", 0.0, FG)] + body)
        for f in range(2):
            frame(rows, STATUS, f == 1)
    for _ in range(3 * fps):
        blines = []
        for para in RESPONSE.split("\n"):
            blines += wrap_para(para, font, text_w - mid_px, mid_px)
        body = [(MODEL + blines[0], 0.0, FG)] + [(t, mid_px, FG) for t in blines[1:]]
        rows = trimmed(head + [(YOU, 0.0, PROMPT_COL)] + [("", 0.0, FG)] + body)
        frame(rows, STATUS, False)

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", os.path.join(tmp, "f%04d.png"),
                    "-vf", "split[s0][s1];[s0]palettegen=stats_mode=diff[pal];"
                    "[s1][pal]paletteuse=dither=bayer:bayer_scale=4",
                    "-loop", "0", out], check=True)
    shutil.rmtree(tmp)
    print(f"{out}: {os.path.getsize(out)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
