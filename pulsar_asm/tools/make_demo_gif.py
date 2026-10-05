"""Render a real chat run as a terminal gif for the README.

Nothing in the frames is faked: the tokens come from the engine, streamed
through the same Chat class the CLI uses, and the tok/s in the status line is
what that run actually took. The only liberty is pacing - a gif at 6 tok/s is a
slideshow, so frames are dropped to a fixed cadence.

Two languages, because the README speaks both. `--lang zh` prompts the model in
Traditional Chinese (it answers in kind) and localizes the terminal chrome; it
also switches to a CJK-capable monospace font, since DejaVu Sans Mono has no
Han glyphs and would paint tofu. Wrapping is measured in pixels rather than
split on spaces, because Chinese text has none to split on.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from runtime.gemma4_model import Gemma4  # noqa: E402
import run_gemma4_chat as cli  # noqa: E402

MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
CJK = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"     # WenQuanYi Zen Hei Mono
CJK_ALT = "/usr/share/fonts/opentype/unifont/unifont.otf"

BG, FG, DIM, PROMPT_COL = "#12151d", "#dbe0ec", "#6f7788", "#e0af68"

L10N = {
    "en": {
        "title": "pulsar · gemma-4-e2b · cpu",
        "you": "you> ",
        "model": "model> ",
        "prompt": cli.DEMO_PROMPT,
        "loading": "loading engine · 9.258 GB mapped",
        "prefill": "prefill",
        "greedy": "greedy",
        "cores": "cores",
        "assembly": "B of assembly",
        "tok": "tok",
        "seed": 7,
    },
    "zh": {
        "title": "pulsar · gemma-4-e2b · CPU 推論",
        "you": "你> ",
        "model": "模型> ",
        "prompt": "用一句話說：為什麼用組合語言寫語言模型很奇怪？",
        "loading": "載入引擎 · 已映射 9.258 GB",
        "prefill": "填充 KV 快取",
        "greedy": "貪婪解碼",
        "cores": "核心",
        "assembly": "位元組組合語言",
        "tok": "token",
        "seed": 9,
    },
}

# A wrapping unit is a whitespace run, one CJK/fullwidth character, or one Latin
# word. Splitting Chinese by spaces would never break a line at all.
UNIT = re.compile(
    r"\s+|[\u2e80-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffdf]"
    r"|[^\s\u2e80-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffdf]+")


def wrap(text, font, width, indent_px):
    """Greedy fill by measured width. Returns (line, x_offset) pairs."""
    out, line, w = [], "", 0.0
    limit = width - indent_px
    for unit in UNIT.findall(text):
        uw = font.getlength(unit)
        if line and w + uw > limit:
            out.append((line, indent_px))
            line, w = unit.lstrip(" "), font.getlength(unit.lstrip(" "))
        else:
            line, w = line + unit, w + uw
    if line:
        out.append((line, indent_px))
    return out or [("", indent_px)]


def pick_font(lang, size):
    if lang != "zh":
        return ImageFont.truetype(MONO, size)
    for path in (CJK, CJK_ALT):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.truetype(MONO, size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/mnt/edge/pulsar/gemma4_e2b.bin")
    ap.add_argument("--tok", default=cli.DEFAULT_TOK)
    ap.add_argument("--lang", choices=("en", "zh"), default="en")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--size", default="1100,560")
    ap.add_argument("--font-size", type=int, default=16)
    ap.add_argument("--fps", type=int, default=14)
    ap.add_argument("--frames-per-token", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0, help="0 = the language's own default")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--prompt", default=None, help="override the language's demo prompt")
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    a = ap.parse_args()

    L = L10N[a.lang]
    # repo-root doc/, whatever directory the tool is invoked from
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = a.out or os.path.join(root, "doc",
                                f"gemma4-chat-{'zh' if a.lang == 'zh' else 'en'}.gif")
    w, h = [int(v) for v in a.size.split(",")]
    font = pick_font(a.lang, a.font_size + 1)
    lh = sum(font.getmetrics())
    pad, top = 26, 62
    text_w = w - 2 * pad
    you_px = font.getlength(L["you"])
    cmd = "$ python3 run_gemma4_chat.py --demo"

    tk, eos = cli.load_tokenizer(a.tok)
    eng = Gemma4(a.weights, max_seq=1024, n_threads=a.threads, verbose=False)
    # same sampler policy as the CLI, so the gif is the checkpoint's default
    # behaviour and not a hand-tuned argmax that happens to look tidy
    a.seed = a.seed or L["seed"]
    mode = cli.sampler_from_config(eng, a.tok, a)
    if mode == "greedy":
        mode = L["greedy"]
                          # sampler_from_config already seeded it; re-calling
                          # set_sampling with the default temp would turn
                          # sampling back on under a footer that says greedy
    chat = cli.Chat(eng, tk, eos, max_new=a.max_new)
    ids = list(chat.turn(L["prompt"], quiet=True))
    tps = chat.n_tok / max(chat.clock, 1e-9)
    nbytes = eng.mod.size
    eng.close()

    tmp = tempfile.mkdtemp(prefix="pulsargif")
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)

    def draw_frame(rows, status, cursor):
        d.rectangle([0, 0, w, h], fill=BG)
        d.rounded_rectangle([10, 10, w - 10, 44], 8, outline="#242a38", width=1)
        for i, c in enumerate(("#f7768e", "#e0af68", "#9ece6a")):
            d.ellipse([26 + i * 22, 22, 40 + i * 22, 36], fill=c)
        d.text((150, 19), L["title"], font=font, fill=DIM)
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

    head = [(cmd, 0.0, DIM)]
    prompt_rows = head + [(t, xoff, PROMPT_COL)
                          for t, xoff in wrap(L["prompt"], font, text_w, you_px)]

    for _ in range(a.fps):
        draw_frame(head + [("", 0.0, FG)], L["loading"], False)
    for _ in range(a.fps):
        draw_frame(prompt_rows + [("", 0.0, FG)], L["prefill"], False)

    mid_px = font.getlength(L["model"])
    for k in range(1, len(ids) + 1):
        body = [(t, xoff, FG) for t, xoff
                in wrap(L["model"] + tk.decode(ids[:k]), font, text_w, mid_px)]
        st = (f"{k} {L['tok']} · {tps:.1f} tok/s · {mode} · "
              f"{a.threads} {L['cores']} · {nbytes} {L['assembly']}")
        for f in range(a.frames_per_token):
            draw_frame(head + prompt_rows[1:] + [("", 0.0, FG)] + body, st,
                       f == a.frames_per_token - 1)
    for _ in range(3 * a.fps):
        body = [(t, xoff, FG) for t, xoff
                in wrap(L["model"] + tk.decode(ids), font, text_w, mid_px)]
        st = (f"{len(ids)} {L['tok']} · {tps:.1f} tok/s · {mode} · "
              f"{a.threads} {L['cores']} · {nbytes} {L['assembly']}")
        draw_frame(head + prompt_rows[1:] + [("", 0.0, FG)] + body, st, False)

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps),
                    "-i", os.path.join(tmp, "f%04d.png"),
                    "-vf", "split[s0][s1];[s0]palettegen=stats_mode=diff[pal];"
                    "[s1][pal]paletteuse=dither=bayer:bayer_scale=4",
                    "-loop", "0", out], check=True)
    shutil.rmtree(tmp)
    print(f"{out}: {os.path.getsize(out)/1e6:.2f} MB · {len(ids)} tokens · "
          f"{tps:.1f} tok/s · {len(wrap(L['model'] + tk.decode(ids), font, text_w, mid_px))} lines")


if __name__ == "__main__":
    main()
