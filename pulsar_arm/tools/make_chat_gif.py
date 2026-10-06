"""Render real PULSAR-ARM transcripts as terminal GIFs (E2B fashion).

Nothing in the frames is faked: every spec below is a byte-for-byte transcript
of a real `./core ...` run on the Pi with the HF-exact build (the q_norm fix).
Trailing blank lines that the gen cap can leave behind are trimmed; sampling
is the model's recommended config (temp 1.0, top-k 64, top-p 0.95), which is
what the engine's sampler already implements.
and the status line carries that session's measured rate. The amber `>` line is
the user turn. Only pacing is libertied (fixed-cadence reveal), as in
pulsar_asm's make_demo_gif.py.

All four clips share one font size and one typeface: Latin is always DejaVu
Sans Mono, and WenQuanYi Zen Hei is used only for CJK glyphs (there is no
Chinese in DejaVu), at the same size and the same line height.

Run on the Pi:
    python3 tools/make_chat_gif.py                          # all six demos
    python3 tools/make_chat_gif.py 1b-en 1b-zh-tw fc-en   # any subset
"""
import os
import shutil
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
CJK = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"   # face 0: WenQuanYi Zen Hei
SIZE = 17      # one font size for every demo
LH = 23        # one line height for every demo (fits the taller CJK face too)

BG, FG, DIM, PROMPT_COL = "#12151d", "#dbe0ec", "#6f7788", "#e0af68"
CMD_OK = "#9ece6a"

DOC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "doc")

CHAT = "6.8 tok/s \u00b7 temp 1.0 \u00b7 3 cores"
CHAT_TW = "6.8 token/s \u00b7 temp 1.0 \u00b7 3 \u6838\u5fc3"

# The FunctionGemma prompt file, verbatim (both clips share it; only the user
# turn differs) - this is the developer/system turn where the tool is defined.
FC_SYS = ("<bos><start_of_turn>developer\n"
          "You are a model that can do function calling with the following "
          "functions<start_function_declaration>declaration:"
          "get_current_temperature{description:<escape>Gets the current "
          "temperature for a given location.<escape>,parameters:{properties:"
          "{location:{description:<escape>The city name, e.g. San Francisco"
          "<escape>,type:<escape>STRING<escape>}},required:[<escape>location"
          "<escape>],type:<escape>OBJECT<escape>}}"
          "<end_function_declaration><end_of_turn>\n")
FC_SYS_LABEL = ("# system prompt, verbatim from tool_official.txt"
                " (functions / tools are defined here):")

SPECS = {
    "1b-en": dict(
        out="gemma3-1b-chat-en.gif",
        title="pulsar \u00b7 gemma-3-1b-it \u00b7 cpu",
        cmd="$ printf 'List the four seasons and one thing that changes in"
            " each.\\n' | ./core gemma-3-1b-it-qat-q4_0.safetensors"
            " vocab.bin 2 c bpe.bin 1000 950 123",
        you="> List the four seasons and one thing that changes in each.",
        tpl="tpl: 20 105 2364 107 1613 506 2390 18046 532 886 3210 600 3731"
            " 528 1546 236761 106 107 105 4368 107",
        response=("Okay, here are the four seasons and one thing that changes in"
                  " each:\n\n"
                  "1.  **Spring:** Blossoms and new growth!\n"
                  "2.  **Summer:** Warm sunshine and longer days.\n"
                  "3.  **Autumn (Fall):** Falling leaves and cooler"
                  " temperatures.\n"
                  "4.  **Winter:** Snowflakes and cold weather.\n\n"
                  "Let me know if you'd like to know more about any of"
                  " these seasons!"),
        status="1.8 tok/s \u00b7 temp 1.0 \u00b7 3 cores",
        wrap="word"),

    "1b-zh-tw": dict(
        out="gemma3-1b-chat-zh-tw.gif",
        title="pulsar \u00b7 gemma-3-1b-it \u00b7 cpu",
        cmd="$ printf '\u8acb\u5217\u51fa\u4fdd\u6301\u5065\u5eb7\u7684\u4e09\u500b"
            "\u8981\u9ede\u3002\\n' | ./core gemma-3-1b-it-qat-q4_0.safetensors"
            " vocab.bin 2 c bpe.bin 1000 950 172",
        you="> \u8acb\u5217\u51fa\u4fdd\u6301\u5065\u5eb7\u7684\u4e09\u500b\u8981\u9ede\u3002",
        tpl="tpl: 18 105 2364 107 239230 238046 237191 36267 27789 102058"
            " 237629 237208 238745 236924 106 107 105 4368 107",
        response=(
            "\u597d\u7684\uff0c\u4ee5\u4e0b\u662f\u4fdd\u6301\u5065\u5eb7\u7684\u4e09\u500b\u8981\u9ede\uff1a\n\n"
            "1. **\u898f\u5f8b\u904b\u52d5\uff1a** \u904b\u52d5\u4e0d\u50c5\u80fd\u5e6b\u52a9\u7dad\u6301\u9ad4\u91cd\uff0c"
            "\u66f4\u80fd\u63d0\u5347\u5fc3\u8840\u7ba1\u5065\u5eb7\u3001\u589e\u5f37\u808c\u8089\u3001"
            "\u6539\u5584\u60c5\u7dd2\u3001\u63d0\u5347\u8a8d\u77e5\u529f\u80fd\u3002"
            "\u5efa\u8b70\u6bcf\u5929\u81f3\u5c11\u9032\u884c30\u5206\u9418\u4e2d\u7b49\u5f37\u5ea6\u904b\u52d5\uff0c"
            "\u4f8b\u5982\u5feb\u8d70\u3001\u6e38\u6cf3\u3001\u8df3\u821e\u7b49\u3002\n"
            "2. **\u5065\u5eb7\u98f2\u98df\uff1a** \u5747\u8861\u98f2\u98df\uff0c\u651d\u53d6\u8db3\u5920\u7684"
            "\u852c\u83dc\u3001\u6c34\u679c\u3001\u5168\u7a40\u985e\u3001\u86cb\u767d\u8cea\u548c\u5065\u5eb7"
            "\u8102\u80aa\u3002\u907f\u514d\u904e\u591a\u52a0\u5de5\u98df\u54c1\u3001\u7cd6\u548c\u4e0d\u5065\u5eb7"
            "\u7684\u8102\u80aa\u3002\n"
            "3. **\u5145\u8db3\u7761\u7720\uff1a** \u7761\u7720\u662f\u8eab\u9ad4\u548c\u7cbe\u795e\u7684"
            "\u5145\u96fb\uff0c\u5145\u8db3\u7684\u7761\u7720\u80fd\u5e6b\u52a9\u6062\u5fa9\u3001"
            "\u589e\u5f37\u514d\u75ab\u529b\u3001\u63d0\u9ad8\u5c08\u6ce8\u529b\uff0c\u4e26\u6709\u52a9\u65bc"
            "\u60c5\u7dd2\u7a69\u5b9a\u3002\u5efa\u8b70\u6bcf\u665a\u7372\u5f977-8\u5c0f\u6642\u7684"
            "\u7761\u7720\u3002\n\n"
            "\u5e0c\u671b\u9019\u4e9b\u8cc7\u8a0a\u5c0d\u60a8\u6709\u5e6b\u52a9\uff01 "),
        status="1.8 token/s \u00b7 temp 1.0 \u00b7 3 \u6838\u5fc3",
        wrap="char"),

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
        status=CHAT, wrap="word"),

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
        status=CHAT_TW, wrap="char"),

    "fc-en": dict(
        out="functiongemma-toolcall-en.gif",
        title="pulsar \u00b7 functiongemma-270m-it \u00b7 cpu",
        cmd="$ ./core model.safetensors fcvocab.bin 2 f fcbpe.bin 1000 950 16"
            " < tool_official.txt",
        sys_label=FC_SYS_LABEL,
        sys_prompt=FC_SYS,
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
        wrap="word"),

    "fc-zh-tw": dict(
        out="functiongemma-toolcall-zh-tw.gif",
        title="pulsar \u00b7 functiongemma-270m-it \u00b7 cpu",
        cmd="$ ./core model.safetensors fcvocab.bin 2 f fcbpe.bin 1000 950 16"
            " < tool_official_zhtw.txt",
        sys_label=FC_SYS_LABEL,
        sys_prompt=FC_SYS,
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
        status="file mode (tool_official_zhtw.txt) \u00b7 greedy \u00b7 cap 16",
        wrap="word"),
}


def _is_cjk(ch):
    o = ord(ch)
    return (0x2E80 <= o <= 0xA4CF or 0xAC00 <= o <= 0xD7FF
            or 0xF900 <= o <= 0xFAFF or 0xFE30 <= o <= 0xFE4F
            or 0xFF00 <= o <= 0xFFEF)


class Text:
    """One size, one Latin face, CJK glyphs only as a fallback.

    Both faces are loaded at SIZE, so every demo shares one font size and one
    typeface for Latin text; a character is drawn with the CJK face only when
    DejaVu has no glyph for it.
    """

    def __init__(self, size=SIZE):
        self.latin = ImageFont.truetype(MONO, size)
        self.cjk = ImageFont.truetype(CJK, size)

    def face(self, ch):
        return self.cjk if _is_cjk(ch) else self.latin

    def length(self, text):
        return sum(self.face(ch).getlength(ch) for ch in text)

    def draw(self, d, xy, text, fill):
        x, y = xy
        run, face = "", None
        for ch in text:
            f = self.face(ch)
            if face is not None and f is not face:
                d.text((x, y), run, font=face, fill=fill)
                x += sum(face.getlength(c) for c in run)
                run = ""
            face, run = f, run + ch
        if run:
            d.text((x, y), run, font=face, fill=fill)


def wrap(text, tx, width, mode):
    """Wrap `text` to `width` px. word = whitespace split, char = CJK script."""
    out = []
    if mode == "char":
        for para in text.split("\n"):
            line = ""
            for ch in para:
                if line and tx.length(line + ch) > width:
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
            if tx.length(line + unit) <= width:
                line += unit
                continue
            if line:
                out.append(line)
                line = ""
            while tx.length(word) > width:        # unbreakable token
                cut = ""
                for ch in word:
                    if cut and tx.length(cut + ch) > width:
                        break
                    cut += ch
                out.append(cut)
                word = word[len(cut):]
            line = word
        out.append(line)
    return out


def render(name, spec, out_dir=DOC):
    w, h = 1100, 560
    tx = Text()
    pad, top = 26, 62
    text_w = w - 2 * pad
    mid_px = tx.length("model> ")

    head = [(t, 0.0, DIM) for t in wrap(spec["cmd"], tx, text_w, "word")]
    stages = [head]
    if spec.get("sys_prompt"):
        stages.append(head
                      + [(t, 0.0, DIM)
                         for t in wrap(spec["sys_label"], tx, text_w, "word")]
                      + [(t, 0.0, FG)
                         for t in wrap(spec["sys_prompt"], tx, text_w, "word")])
    you_rows = stages[-1] if not spec["you"] \
        else stages[-1] + [(spec["you"], 0.0, PROMPT_COL)]
    stages.append(you_rows)
    tpl_rows = you_rows + [(t, 0.0, DIM)
                           for t in wrap(spec["tpl"], tx, text_w, "word")]

    full = wrap(spec["response"], tx, text_w - mid_px, spec["wrap"])
    # the 270m canvas, enlarged only if a transcript needs the room (the FC
    # clips show the whole tool schema)
    h = max(h, top + (len(tpl_rows) + 1 + len(full)) * LH + 46 + 8)
    available = (h - 46 - top) // LH      # rows that fit above the status bar

    def compose(body):
        rows = tpl_rows + [("", 0.0, FG)] + body
        if len(rows) > available:
            keep = min(len(tpl_rows), 6)
            rows = rows[:keep] + rows[len(rows) - (available - keep):]
        return rows

    tmp = tempfile.mkdtemp(prefix="pulsar_demo")
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)

    def frame(rows, status, cursor):
        d.rectangle([0, 0, w, h], fill=BG)
        d.rounded_rectangle([10, 10, w - 10, 44], 8, outline="#242a38", width=1)
        for i, c in enumerate(("#f7768e", "#e0af68", CMD_OK)):
            d.ellipse([26 + i * 22, 22, 40 + i * 22, 36], fill=c)
        tx.draw(d, (150, 19), spec["title"], DIM)
        y = top
        for text, xoff, col in rows:
            if text:
                tx.draw(d, (pad + xoff, y), text, col)
            y += LH
        if cursor and rows:
            text, xoff, _ = rows[-1]
            tx.draw(d, (pad + xoff + tx.length(text), y - LH), "\u2588", FG)
        d.line([pad, h - 46, w - pad, h - 46], fill="#242a38", width=1)
        tx.draw(d, (pad, h - 38), status, DIM)
        img.save(os.path.join(tmp, f"f{len(os.listdir(tmp)):04d}.png"))

    fps = 14
    for _ in range(fps):
        frame(head + [("", 0.0, FG)], "loading engine", False)
    for st in stages[1:]:
        for _ in range(fps // 2):
            frame(st + [("", 0.0, FG)], "prefill", False)

    units = spec["response"].split(" ") if spec["wrap"] == "word" \
        else list(spec["response"])
    for k in range(1, len(units) + 1):
        part = (" ".join(units[:k]) if spec["wrap"] == "word"
                else "".join(units[:k]))
        blines = wrap(part, tx, text_w - mid_px, spec["wrap"])
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
