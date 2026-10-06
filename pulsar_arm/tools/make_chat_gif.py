"""Render real PULSAR-ARM transcripts as terminal GIFs (E2B fashion).

Nothing in the frames is faked: every spec below is a byte-for-byte transcript
of a real `./core ...` run on the Pi with the HF-exact build (the q_norm fix),
and the status line carries that session's measured rate. Lines marked `#` are
labels, not engine output (the engine never prints them); everything else is.
Only pacing is libertied (fixed-cadence reveal), as in pulsar_asm's
make_demo_gif.py.

Run on the Pi:
    python3 tools/make_chat_gif.py                    # all four demos
    python3 tools/make_chat_gif.py en zh-tw fc-en     # any subset
"""
import os
import shutil
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
CJK = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"

BG, FG, DIM, PROMPT_COL = "#12151d", "#dbe0ec", "#6f7788", "#e0af68"
CMD_OK = "#9ece6a"

DOC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "doc")

CHAT = "6.8 tok/s \u00b7 temp 1.0 \u00b7 3 cores"

SPECS = {
    "en": dict(
        out="gemma3-270m-chat-en.gif",
        title="pulsar \u00b7 gemma-3-270m-it \u00b7 cpu",
        cmd="$ printf 'Explain gravity in two sentences for a child.\\n'"
            " | ./core model.safetensors vocab.bin 2 c bpe.bin 1000 950 48",
        you="> Explain gravity in two sentences for a child.",
        tpl="tpl: 17 105 2364 107 155122 18899 528 1156 23974 573 496 1919 "
            "236761 106 107 105 4368 107",
        response=("Gravity is a force that pulls things towards each other. "
                  "It's like a giant, invisible hug that keeps us all stuck "
                  "to the ground!"),
        status=CHAT,
        font=MONO, wrap="word"),

    "zh-tw": dict(
        out="gemma3-270m-chat-zh-tw.gif",
        title="pulsar \u00b7 gemma-3-270m-it \u00b7 cpu",
        cmd="$ printf '\u8acb\u7528\u4e00\u53e5\u8a71\u89e3\u91cb\u4ec0\u9ebc"
            "\u662f\u91cf\u5b50\u529b\u5b78\u3002\\n' | ./core model.safetensors"
            " vocab.bin 2 c bpe.bin 1000 950 48",
        you="> \u8acb\u7528\u4e00\u53e5\u8a71\u89e3\u91cb\u4ec0\u9ebc\u662f"
            "\u91cf\u5b50\u529b\u5b78\u3002",
        tpl="tpl: 19 105 2364 107 239230 237105 122100 238360 185411 26549 "
            "237026 199311 237473 238432 236924 106 107 105 4368 107",
        response="\u91cf\u5b50\u529b\u5b78\u662f\u6307\u5728\u7269\u7406\u5b78"
                 "\u4e2d\uff0c\u63cf\u8ff0\u548c\u89e3\u91cb\u5fae\u89a5\u4e16"
                 "\u754c\u7684\u7c92\u5b50\u548c\u5b83\u5011\u7684\u76f8\u4e92"
                 "\u4f5c\u7528\uff0c\u4e26\u95dc\u91cb\u91cf\u5b50\u529b\u5b78"
                 "\u7684\u539f\u7406\u3002",
        status="6.8 token/s \u00b7 temp 1.0 \u00b7 3 \u6838\u5fc3",
        font=CJK, wrap="char"),

    "fc-en": dict(
        out="functiongemma-toolcall-en.gif",
        title="pulsar \u00b7 functiongemma-270m-it \u00b7 cpu",
        cmd="$ ./core model.safetensors fcvocab.bin 2 f fcbpe.bin 1000 950 16"
            " < tool_official.txt",
        you="> What's the temperature in London?",
        tpl="tpl: 98 2 105 55060 107 3048 659 496 2028 600 740 776 1292 11687 "
            "607 506 2269 5151 46 163688 236787 828 236779 4002 236779 27495 "
            "236782 7777 236787 52 81113 506 1873 4022 573 496 2238 4563 "
            "236761 52 236764 19031 29616 15921 29616 7125 29616 7777 236787 "
            "52 818 3207 1463 236764 545 236761 236759 236761 5054 14322 52 "
            "236764 2084 236787 52 35410 52 5237 15979 24845 52 7125 52 1604 "
            "2084 236787 52 60688 52 1807 47 106 107 105 2364 107 3689 236789 "
            "236751 506 4022 528 5860 236881 106 107 105 4368 107",
        response="call:get_current_temperature{location:London}",
        status="file mode (tool_official.txt) \u00b7 greedy \u00b7 cap 16",
        font=MONO, wrap="word"),

    "fc-zh-tw": dict(
        out="functiongemma-toolcall-zh-tw.gif",
        title="pulsar \u00b7 functiongemma-270m-it \u00b7 cpu",
        cmd="$ ./core model.safetensors fcvocab.bin 2 f fcbpe.bin 1000 950 16"
            " < tool_official_zhtw.txt",
        you="> \u5011\u6566\u7684\u6eab\u5ea6\u662f\u591a\u5c11\uff1f\u3000"
            "\uff08zh-TW\uff09",
        tpl="tpl: 96 2 105 55060 107 3048 659 496 2028 600 740 776 1292 11687 "
            "607 506 2269 5151 46 163688 236787 828 236779 4002 236779 27495 "
            "236782 7777 236787 52 81113 506 1873 4022 573 496 2238 4563 "
            "236761 52 236764 19031 29616 15921 29616 7125 29616 7777 236787 "
            "52 818 3207 1463 236764 545 236761 236759 236761 5054 14322 52 "
            "236764 2084 236787 52 35410 52 5237 15979 24845 52 7125 52 1604 "
            "2084 236787 52 60688 52 1807 47 106 107 105 2364 107 241849 "
            "241281 236918 190519 187330 237536 106 107 105 4368 107",
        response="call:get_current_temperature{location:London}",
        status="file mode (tool_official_zhtw.txt) \u00b7 greedy"
               " \u00b7 cap 16",
        font=CJK, wrap="word"),
}


def wrap(text, font, width, mode):
    """Wrap `text` to `width` px. word = whitespace split, char = CJK."""
    out = []
    if mode == "char":
        line = ""
        for ch in text:
            if font.getlength(line + ch) > width and line:
                out.append(line)
                line = ch
            else:
                line += ch
        out.append(line)
        return out
    for para in text.split("\n"):
        line = ""
        for word in para.split(" "):
            unit = ("" if not line else " ") + word
            if line and font.getlength(line + unit) > width:
                out.append(line)
                line = word
            else:
                line += unit
        out.append(line)
    return out


def render(name, spec, out_dir=DOC):
    w, h = 1100, 560
    font = ImageFont.truetype(spec["font"], 17)
    lh = sum(font.getmetrics())
    pad, top = 26, 62
    text_w = w - 2 * pad
    mid_px = font.getlength("model> ")

    # rows: command, then the user turn (amber, same as the chat clips),
    # then the engine's own `tpl:` id dump. In file mode the engine does not
    # echo the prompt, so the status bar names the prompt file.
    head = [(t, 0.0, DIM)
            for t in wrap(spec["cmd"], font, text_w, "word")]
    you_rows = head if not spec["you"] else head + [(spec["you"], 0.0,
                                                     PROMPT_COL)]
    tpl_rows = you_rows + [(t, 0.0, DIM)
                           for t in wrap(spec["tpl"], font, text_w, "word")]

    full = wrap(spec["response"], font, text_w - mid_px, spec["wrap"])
    max_rows = (h - 46 - top) // lh - 3

    def compose(body):
        rows = tpl_rows + [("", 0.0, FG)] + body
        if len(rows) > max_rows:
            keep = min(len(tpl_rows), 6)
            rows = rows[:keep] + rows[len(rows) - (max_rows - keep):]
        return rows

    tmp = tempfile.mkdtemp(prefix="pulsar_demo")
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)

    def frame(rows, status, cursor):
        d.rectangle([0, 0, w, h], fill=BG)
        d.rounded_rectangle([10, 10, w - 10, 44], 8, outline="#242a38", width=1)
        for i, c in enumerate(("#f7768e", "#e0af68", CMD_OK)):
            d.ellipse([26 + i * 22, 22, 40 + i * 22, 36], fill=c)
        d.text((150, 19), spec["title"], font=font, fill=DIM)
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
    for _ in range(fps):
        frame(head + [("", 0.0, FG)], "loading engine", False)
    for _ in range(fps):
        frame(you_rows + [("", 0.0, FG)], "prefill", False)
    for _ in range(fps // 2):
        frame(tpl_rows + [("", 0.0, FG)], "prefill", False)

    units = spec["response"].split(" ") if spec["wrap"] == "word" \
        else list(spec["response"])
    for k in range(1, len(units) + 1):
        part = (" ".join(units[:k]) if spec["wrap"] == "word"
                else "".join(units[:k]))
        blines = wrap(part, font, text_w - mid_px, spec["wrap"])
        body = [("model> " + blines[0], 0.0, FG)] + \
               [(t, mid_px, FG) for t in blines[1:]]
        for f in range(2):
            frame(compose(body), spec["status"], f == 1)
    body = [("model> " + full[0], 0.0, FG)] + [(t, mid_px, FG)
                                               for t in full[1:]]
    for _ in range(3 * fps):
        frame(compose(body), spec["status"], False)

    out = os.path.join(out_dir, spec["out"])
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", os.path.join(tmp, "f%04d.png"),
                    "-vf", "split[s0][s1];[s0]palettegen=stats_mode=diff[pal];"
                    "[s1][pal]paletteuse=dither=bayer:bayer_scale=4",
                    "-loop", "0", out], check=True)
    shutil.rmtree(tmp)
    print(f"{name}: {out} ({os.path.getsize(out)/1e6:.2f} MB)")


def main():
    names = sys.argv[1:] or list(SPECS)
    for n in names:
        if n not in SPECS:
            sys.exit(f"unknown spec {n!r}; choose from {', '.join(SPECS)}")
    for n in names:
        render(n, SPECS[n])


if __name__ == "__main__":
    main()
