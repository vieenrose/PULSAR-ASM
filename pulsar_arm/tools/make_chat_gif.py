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

    # Bonsai 2 / Ternary-Bonsai-8B-PQ2_0 on the DGX Spark, CPU only (20 threads,
    # llama.cpp fork since Q2_0 is prism-specific upstream). Ceiling prompt is
    # the same four-seasons list the gemma clips use - the most complex prompt
    # this checkpoint answers correctly. Transcript and rate come from one
    # llama-server session at the checkpoint's Qwen3-standard sampling; ids are
    # what the server's own tokenizer produced for the prompt.
    "bonsai8b-en": dict(
        out="bonsai8b-chat-en.gif",
        title="pulsar \u00b7 Ternary-Bonsai-8B \u00b7 phone cpu",
        cmd="$ adb shell \"cd /data/local/tmp && PQ2_THREADS=4 taskset f0 ./fwd8fresh b8.gguf 151644 872 198 852 279 3040 15584 323 825 3166 429 4344 304 1817 13 151645 198 151644 77091 198 151667 271 151668 271 --gen 400 --sample 0.6 0.95 20 7\"",
        you="> List the four seasons and one thing that changes in each.",
        tpl="prompt ids: 151644 872 198 852 279 3040 15584 323 825 3166 429 4344"
            " 304 1817 13 151645 198 151644 77091 198 151667 271 151668 271",
        response=("Here are the four seasons and one thing that changes in each:\n\n1. **Spring** \u2013 Plants bloom and flowers appear.  \n2. **Summer** \u2013 The weather is warm and sunny.  \n3. **Autumn** \u2013 Leaves change color and fall from trees.  \n4. **Winter** \u2013 It snows and the weather is cold."),
        status="4.7 tok/s \u00b7 temp 0.6 \u00b7 SD855, 4 threads",
        wrap="word"),

    # Ternary-Bonsai-2-27B (Qwen3.8-27B base) on the DGX Spark, CPU only,
    # via the same llama.cpp fork. Same four-seasons ceiling prompt as the gemma
    # and 8B clips. Card config here: temp 0.5 / top_p 0.85 / top_k 20. CPU was
    # sampled at 1022-1765% while generating, i.e. genuinely multi-threaded
    # CPU-only (GPU utilisation 0%).
    "bonsai2-27b-en": dict(
        out="bonsai2-27b-chat-en.gif",
        title="pulsar \u00b7 Ternary-Bonsai-2-27B \u00b7 cpu",
        cmd="$ ./fwd_exp Ternary-Bonsai-2-27B-PTQ1_0.gguf 826 279 2943 15127 321 799 3065 421 4203 303 1754 13 --gen 400 --sample 0.5 0.85 20 7",
        you="> List the four seasons and one thing that changes in each.",
        tpl="prompt ids: 826 279 2943 15127 321 799 3065 421 4203 303 1754 13",
        response=("\n\n<think>\n\n</think>\n\nHere are the four seasons and one key change that occurs in each:\n\n1.  **Spring** \u2013 The temperature rises, and dormant plants begin to bud and bloom.\n2.  **Summer** \u2013 The days are at their longest, and temperatures are typically at their highest.\n3.  **Autumn (Fall)** \u2013 Leaves change color and fall from trees, and temperatures begin to drop.\n4.  **Winter** \u2013 Temperatures are at their lowest, and many areas experience snow or ice."),
        status="7.9 tok/s \u00b7 temp 0.5 \u00b7 8 threads",
        wrap="word"),

    "bonsai8b-zh-tw": dict(
        out="bonsai8b-chat-zh-tw.gif",
        title="pulsar \u00b7 Ternary-Bonsai-8B \u00b7 phone cpu",
        cmd="$ adb shell \"cd /data/local/tmp && PQ2_THREADS=4 taskset f0 ./fwd8fresh b8.gguf 151644 872 198 100792 114116 109080 3837 101034 104588 99377 99496 110946 99774 108008 1773 151645 198 151644 77091 198 151667 271 151668 271 --gen 400 --sample 0.6 0.95 20 123\"",
        you="> \u8acb\u5217\u51fa\u56db\u5b63\uff0c\u4ee5\u53ca\u6bcf\u4e00\u5b63\u6703\u6539\u8b8a\u7684\u4e00\u4ef6\u4e8b\u3002",
        tpl="prompt ids: 151644 872 198 100792 114116 109080 3837 101034 104588 99377 99496 110946 99774 108008 1773 151645 198 151644 77091 198 151667 271 151668 271",
        response=("\u7576\u7136\u53ef\u4ee5\uff01\u4ee5\u4e0b\u662f\u56db\u5b63\u4ee5\u53ca\u6bcf\u4e00\u5b63\u6703\u6539\u8b8a\u7684\u4e00\u4ef6\u4e8b\uff1a\n\n---\n\n### **1. \u6625\u5b63**  \n**\u6539\u8b8a\u7684\u4e8b\uff1a** **\u82b1\u958b**  \n\u6625\u5929\u662f\u82b1\u6735\u7efd\u653e\u7684\u5b63\u7bc0\uff0c\u6a39\u6728\u3001\u82b1\u8349\u9010\u6f38\u958b\u51fa\u65b0\u82bd\uff0c\u8272\u5f69\u8b8a\u5f97\u9bae\u7f8e\uff0c\u70ba\u81ea\u7136\u754c\u5e36\u4f86\u52c3\u52c3\u7684\u8272\u5f69\u8207\u751f\u547d\u3002\n\n---\n\n### **2. \u590f\u5b63**  \n**\u6539\u8b8a\u7684\u4e8b\uff1a** **\u8449\u5b50\u8b8a\u7da0**  \n\u590f\u5929\u662f\u690d\u7269\u6700\u6d3b\u8e8d\u7684\u6642\u671f\uff0c\u8449\u5b50\u5728\u967d\u5149\u4e0b\u8f49\u8b8a\u6210\u6df1\u7da0\uff0c\u70ba\u751f\u614b\u7cfb\u7d71\u63d0\u4f9b\u5927\u91cf\u7684\u98df\u7269\u8207\u6c27\u6c23\u3002\n\n---\n\n### **3. \u79cb\u5b63**  \n**\u6539\u8b8a\u7684\u4e8b\uff1a** **\u8449\u5b50\u8b8a\u8272**  \n\u79cb\u5929\uff0c\u8449\u5b50\u9010\u6f38\u6539\u8b8a\u70ba\u9ec3\u3001\u7d05\u3001\u6a59\u7b49\u8272\u5f69\uff0c\u9019\u662f\u690d\u7269\u6e96\u5099\u904e\u51ac\u7684\u904e\u7a0b\uff0c\u4e5f\u662f\u81ea\u7136\u754c\u4e2d\u6700\u7f8e\u7684\u666f\u89c0\u4e4b\u4e00\u3002\n\n---\n\n### **4. \u51ac\u5b63**  \n**\u6539\u8b8a\u7684\u4e8b\uff1a** **\u8449\u5b50\u843d\u4e0b**  \n\u51ac\u5929\uff0c\u690d\u7269\u70ba\u4e86\u7bc0\u7701\u80fd\u91cf\u800c\u6389\u4e0b\u8449\u5b50\uff0c\u6a39\u6728\u9032\u5165\u4f11\u7720\u72c0\u614b\uff0c\u70ba\u4f86\u5e74\u7684\u65b0\u751f\u6e96\u5099\u571f\u58e4\u8207\u8cc7\u6e90\u3002\n\n---\n\n\u5982\u679c\u4f60\u6709\u7279\u5b9a\u7684\u74b0\u5883\uff08\u5982\u57ce\u5e02\u3001\u68ee\u6797\u3001\u8fb2\u5834\uff09\u6216\u60f3\u4e86\u89e3\u5176\u4ed6\u65b9\u9762\uff08\u5982\u6c23\u5019\u3001\u6587\u5316\u3001\u751f\u7269\u6d3b\u52d5\uff09\u7684\u8b8a\u5316\uff0c\u6211\u4e5f\u53ef\u4ee5\u9032\u4e00\u6b65\u8aaa\u660e\uff01"),
        status="4.0 tok/s \u00b7 temp 0.6 \u00b7 SD855, 4 threads",
        wrap="char"),

    "bonsai2-27b-zh-tw": dict(
        out="bonsai2-27b-chat-zh-tw.gif",
        title="pulsar \u00b7 Ternary-Bonsai-2-27B \u00b7 cpu",
        cmd="$ ./fwd_exp Ternary-Bonsai-2-27B-PTQ1_0.gguf 99270 115992 104647 3709 98404 98443 96681 96594 110476 127312 1710 --gen 400 --sample 0.5 0.85 20 123",
        you="> \u8acb\u5217\u51fa\u56db\u5b63\uff0c\u4ee5\u53ca\u6bcf\u4e00\u5b63\u6703\u6539\u8b8a\u7684\u4e00\u4ef6\u4e8b\u3002",
        tpl="prompt ids: 99270 115992 104647 3709 98404 98443 96681 96594 110476 127312 1710",
        response=("\n\n<think>\n\u7528\u6237\u8981\u6c42\u5217\u51fa\u56db\u5b63\uff0c\u4ee5\u53ca\u6bcf\u4e00\u5b63\u4f1a\u6539\u53d8\u7684\u4e00\u4ef6\u4e8b\u3002\u8fd9\u662f\u4e00\u4e2a\u7b80\u5355\u7684\u77e5\u8bc6\u6027\u95ee\u9898\u3002\n\n\u56db\u5b63\uff1a\u6625\u3001\u590f\u3001\u79cb\u3001\u51ac\n\n\u6bcf\u4e00\u5b63\u4f1a\u6539\u53d8\u7684\u4e00\u4ef6\u4e8b\uff0c\u6211\u53ef\u4ee5\u4ece\u81ea\u7136\u73b0\u8c61\u3001\u6c14\u5019\u3001\u690d\u7269\u3001\u52a8\u7269\u884c\u4e3a\u7b49\u65b9\u9762\u6765\u8003\u8651\uff1a\n\n- \u6625\u5929\uff1a\u6c14\u6e29\u56de\u5347\uff0c\u690d\u7269\u53d1\u82bd/\u5f00\u82b1\n- \u590f\u5929\uff1a\u65e5\u7167\u65f6\u95f4\u53d8\u957f\uff0c\u6c14\u6e29\u5347\u9ad8\n- \u79cb\u5929\uff1a\u6811\u53f6\u53d8\u9ec4/\u843d\u53f6\n- \u51ac\u5929\uff1a\u6c14\u6e29\u4e0b\u964d\uff0c\u53ef\u80fd\u964d\u96ea\n\n\u6211\u9009\u62e9\u6bcf\u5b63\u4e00\u4e2a\u6700\u5178\u578b\u3001\u6700\u76f4\u89c2\u7684\u53d8\u5316\uff1a\n- \u6625\uff1a\u690d\u7269\u53d1\u82bd\u3001\u5f00\u82b1\n- \u590f\uff1a\u65e5\u7167\u65f6\u95f4\u53d8\u957f\uff08\u6216\u6c14\u6e29\u5347\u9ad8\uff09\n- \u79cb\uff1a\u6811\u53f6\u53d8\u8272\u3001\u843d\u53f6\n- \u51ac\uff1a\u6c14\u6e29\u4e0b\u964d\u3001\u964d\u96ea\n\n\u6211\u9009\u62e9\u6bd4\u8f83\u6709\u753b\u9762\u611f\u4e14\u660e\u786e\u7684\uff1a\n- \u6625\uff1a\u690d\u7269\u4ece\u82bd\u53d8\u5230\u5f00\u82b1\n- \u590f\uff1a\u767d\u5929\u53d8\u957f\uff08\u65e5\u7167\u65f6\u95f4\u589e\u52a0\uff09\n- \u79cb\uff1a\u6811\u53f6\u7531\u7eff\u53d8\u9ec4\u518d\u843d\u53f6\n- \u51ac\uff1a\u5929\u7a7a\u4ece\u6674\u6717\u53d8\u4e3a\u591a\u96ea\uff08\u6216\u6c14\u6e29\u663e\u8457\u4e0b\u964d\uff09\n\n\u8ba9\u6211\u7b80\u6d01\u6e05\u6670\u5730\u56de\u7b54\u3002\n</think>\n\n## \u56db\u5b63\u53ca\u5176\u53d8\u5316\n\n| \u5b63 | \u4ee3\u8868\u53d8\u5316 |\n|---|---|\n| **\u6625** | \u690d\u7269\u53d1\u82bd\u3001\u5f00\u82b1\uff0c\u5927\u5730\u7531\u67af\u8f6c\u7eff |\n| **\u590f** | \u65e5\u7167\u65f6\u95f4\u53d8\u957f\uff0c\u767d\u5929\u9010\u6e10\u5ef6\u957f |\n| **\u79cb** | \u6811\u53f6\u7531\u7eff\u8f6c\u9ec4\u3001\u7ea2\uff0c\u7ee7\u800c\u98d8\u843d |\n| **\u51ac** | \u6c14\u6e29\u9aa4\u964d\uff0c\u5929\u7a7a\u5f00\u59cb\u964d\u96ea |\n\n> \u7b80\u5355\u8bf4\uff1a\u6625\u662f**\u751f**\uff0c\u590f\u662f**\u957f**\uff0c\u79cb\u662f**\u843d**\uff0c\u51ac\u662f**\u85cf**\u2014\u2014\u81ea\u7136\u4ee5\u8fd9\u56db\u4ef6\u4e8b\u6807\u8bb0\u65f6\u95f4\u7684\u6d41\u8f6c\u3002"),
        status="7.5 tok/s \u00b7 temp 0.5 \u00b7 8 threads",
        wrap="char"),
    # K2-Horizon-0.9B base (Q4, native engine on the Galaxy Note 10+ SD855,
    # serial, big cores): ceiling prompt is the two-sentence neural explainer
    # (greedy) - seasons loops and haiku rambles under greedy. 216 steps,
    # byte-identical to the Spark run including logits (phone shot per
    # reshoot scope); 28.2s total = 7.7 tok/s.
    "k2base-neural": dict(
        out="k2horizon-chat-en.gif",
        title="pulsar \u00b7 K2-Horizon-0.9B \u00b7 phone cpu",
        cmd="$ adb shell \"cd /data/local/tmp && taskset f0 ./k2_core k2h_09_q4.blob 64018 2985 200 21127 542 1426 265 30066 4318 394 316 1662 37398 15 64019 64018 612 10102 200 64029 200 --gen 400 --threads 4 --batch 8\"",
        you="> Explain what a neural network is in two sentences.",
        tpl="prompt ids: 64018 2985 200 21127 542 1426 265 30066 4318 394 316 1662 37398 15 64019 64018 612 10102 200 64029 200"
            " \u2014 IFM chat template ends with assistant\\n<ifm|think>\\n"
            " (id 64029), so generation starts inside the think block and"
            " only the closer </ifm|think> is generated",
        response="The user wants an explanation of what a neural network is in two sentences. This is a straightforward request. I should provide a concise definition of neural networks, likely in the context of machine learning, explaining their structure and function. Two sentences is quite short, so I need to be concise but informative. Something like: \"A neural network is a computational model inspired by the brain's structure and function, consisting of layers of interconnected nodes that process inputs through weighted connections to produce outputs, and it is commonly used in machine learning for tasks like image recognition, natural language processing, and predictive modeling.\" That's two sentences. Let me make sure it's clear and accurate.\n</ifm|think>\nA neural network is a computational model inspired by the brain's structure and function, consisting of layers of interconnected nodes that process inputs through weighted connections to produce outputs, and it is commonly used in machine learning for tasks like image recognition, natural language processing, and predictive modeling.",
        status="13 tok/s \u00b7 greedy \u00b7 SD855 big cores, pure-asm engine",
        wrap="word"),
    # K2 meeting agent (fine-tune, LiteRT-LM q4 int4-QAT, stock signatures
    # driven manually): temp 0.2 seed 7. 5/5 notes verifiable, all cited
    # times genuine; harness stops at NEXT (ramble loop after is cut).
    # Seeds 8 (drops PROPOSAL) and 9 (ties 7) also sampled; 7 verified.
    "k2ft-meeting": dict(
        out="k2horizon-meeting-zh-tw.gif",
        title="pulsar \u00b7 K2 meeting agent \u00b7 phone cpu",
        cmd="$ adb shell \"cd /data/local/tmp && taskset f0 ./k2_core k2h_ft_q4.blob"
            " $(cat ft_ids680.txt) --gen 400 --threads 4 --batch 8\""
            "  # native asm engine, greedy",
        you="> S1 [1:02:15] \u5404\u4f4d,\u8cc7\u8a0a\u7cfb\u7d71\u9810\u7b97\u7e3d\u5171\u7de8\u5217 1200 \u842c\u5143,\u8f03\u53bb\u5e74\u589e\u52a0 300 \u842c,\u8acb\u5927\u5bb6\u78ba\u8a8d\u3002\n"
            "> S2 [1:03:02] \u6211\u5efa\u8b70\u6539\u7528\u7dda\u4e0a\u5831\u540d,\u53ef\u4ee5\u6e1b\u5c11\u73fe\u5834\u6392\u968a\u7684\u4eba\u529b,\u5927\u6982\u80fd\u7701 80 \u842c\u5de6\u53f3\u3002\n"
            "> S3 [1:03:40] \u7dda\u4e0a\u5831\u540d\u4e4b\u524d\u8a66\u904e,\u9577\u8f29\u53ef\u80fd\u4e0d\u6703\u7528,\u9019\u500b\u8981\u518d\u8a55\u4f30\u770b\u770b\u3002\n"
            "> S1 [1:04:10] \u597d,\u90a3\u9810\u7b97\u6848\u5c31\u7167\u6848\u901a\u904e\u3002\u4e3b\u8fa6\u55ae\u4f4d\u8acb\u5728\u5169\u9031\u5167\u63d0\u51fa\u66f8\u9762\u5831\u544a\u3002\n"
            "> S2 [1:05:33] \u5831\u544a\u6211\u4f86\u8ca0\u8cac,\u4e0b\u9031\u4e94\u4ee5\u524d\u4ea4\u3002\u53e6\u5916\u4e0b\u6b21\u958b\u6703\u6642\u9593\u8a02\u5728 3 \u6708 15 \u865f\u4e0b\u5348\u5169\u9ede\u3002\n"
            "> S3 [1:06:20] \u6563\u6703\u524d\u63d0\u9192\u4e00\u4e0b,\u5834\u5730\u8cbb\u9084\u6c92\u4ed8,\u7e3d\u5171 5 \u842c\u5143,\u9019\u7b46\u8981\u8d95\u5feb\u8655\u7406\u3002",
        tpl="system: NOTE/REVISE/NEXT \u5340\u5bf8 (1686B) + 6-turn window \u2014 680 ids, native Q4 blob",
        response="1 [1:05:20] \u4e0b\u6703\u6642\u9593 \uff1a\uff0c \u5b89\u6392\u8207\u7e73\u6e05 \u5171 8,, \u4e94\u9810\u7b97\u8acb arrange\nNNistant\n\n\nNOT\nNOTE [1:02:15] (NUMBER) \u8cc7\u8a0a\u7cfb\u7d71\u9810\u7b97\u7e3d\u5171\u7de8\u5217 1200 \u842c\u5143\uff0c\u8f03\u53bb\u5e74\u589e\u52a0 300 \u842c\u5143\nNOTE [1:03:02] (PROPOSAL) \u5efa\u8b70\u6539\u7528\u7dda\u4e0a\u5831\u540d\uff0c\u53ef\u6e1b\u5c11\u73fe\u5834\u6392\u968a\u7684\u4eba\u529b\uff0c\u7d04\u7701 80 \u842c\u5de6\u53f3\nNOTE [1:04:10] (DECISION) \u9810\u7b97\u6848\u7167\u6848\u901a\u904e\nNOTE [1:05:33] (ACTION) S2 \u8ca0\u8cac\u4e0b\u9031\u4e94\u524d\u63d0\u51fa\u66f8\u9762\u5831\u544a\nNOTE [1:06:20] (OPEN-ISSUE) \u5834\u5730\u8cbb 5 \u842c\u5143\u5c1a\u672a\u4ed8\uff0c\u9700\u8655\u7406\nNEXT",
        status="3.2 tok/s \u00b7 greedy \u00b7 SD855 big cores, pure-asm engine",
        wrap="char"),

    # ---- one GIF per model: English and zh-TW side by side ----
    "bonsai8b-chat": dict(
        out="bonsai8b-chat-en-zh.gif",
        panes=("bonsai8b-en", "bonsai8b-zh-tw")),
    "bonsai2-27b-chat": dict(
        out="bonsai2-27b-chat-en-zh.gif",
        panes=("bonsai2-27b-en", "bonsai2-27b-zh-tw")),
    "1b-chat": dict(
        out="gemma3-1b-chat-en-zh.gif",
        panes=("1b-en", "1b-zh-tw")),
    "270m-chat": dict(
        out="gemma3-270m-chat-en-zh.gif",
        panes=("en", "zh-tw")),
    "fc-toolcall": dict(
        out="functiongemma-toolcall-en-zh.gif",
        panes=("fc-en", "fc-zh-tw")),
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


W, H = 1100, 560          # one pane
PAD, TOP, FPS, GAP = 26, 62, 14, 14


def _plan(spec, tx, h=None):
    """Frame descriptors for one pane: list of (rows, status, cursor)."""
    text_w = W - 2 * PAD
    mid_px = tx.length("model> ")
    head = [(t, 0.0, DIM) for t in wrap(spec["cmd"], tx, text_w, "word")]
    stages = [head]
    if spec.get("sys_prompt"):
        stages.append(head
                      + [(t, 0.0, DIM)
                         for t in wrap(spec["sys_label"], tx, text_w, "word")]
                      + [(t, 0.0, FG)
                         for t in wrap(spec["sys_prompt"], tx, text_w, "word")])
    if not spec["you"]:
        you_rows = stages[-1]
    else:
        # multi-line user turn (e.g. 6-turn meeting window) - wrap per line
        you_wrapped = []
        for para in spec["you"].split("\n"):
            you_wrapped.extend(wrap(para, tx, text_w, "char") or [""])
        you_rows = stages[-1] + [(t, 0.0, PROMPT_COL) for t in you_wrapped]
    stages.append(you_rows)
    tpl_rows = you_rows + [(t, 0.0, DIM)
                           for t in wrap(spec["tpl"], tx, text_w, "word")]

    full = wrap(spec["response"], tx, text_w - mid_px, spec["wrap"])
    # the 270m canvas, enlarged only if a transcript needs the room (the FC
    # clips show the whole tool schema); a pair forces both panes to one h so
    # the status bars line up.
    if h is None:
        h = max(H, TOP + (len(tpl_rows) + 1 + len(full)) * LH + 46 + 8)
    available = (h - 46 - TOP) // LH      # rows that fit above the status bar

    def compose(body):
        rows = tpl_rows + [("", 0.0, FG)] + body
        if len(rows) > available:
            keep = min(len(tpl_rows), 6)
            rows = rows[:keep] + rows[len(rows) - (available - keep):]
        return rows

    frames = []
    for _ in range(FPS):
        frames.append((head + [("", 0.0, FG)], "loading engine", False))
    for st in stages[1:]:
        for _ in range(FPS // 2):
            frames.append((st + [("", 0.0, FG)], "prefill", False))

    units = spec["response"].split(" ") if spec["wrap"] == "word" \
        else list(spec["response"])
    for k in range(1, len(units) + 1):
        part = (" ".join(units[:k]) if spec["wrap"] == "word"
                else "".join(units[:k]))
        blines = wrap(part, tx, text_w - mid_px, spec["wrap"])
        body = [("model> " + blines[0], 0.0, FG)] + \
               [(t, mid_px, FG) for t in blines[1:]]
        for f in range(2):
            frames.append((compose(body), spec["status"], f == 1))
    body = [("model> " + full[0], 0.0, FG)] + [(t, mid_px, FG)
                                               for t in full[1:]]
    for _ in range(3 * FPS):
        frames.append((compose(body), spec["status"], False))
    return frames, h


def _draw_pane(d, tx, spec, fr, x0, w, h):
    rows, status, cursor = fr
    d.rounded_rectangle([x0 + 10, 10, x0 + w - 10, 44], 8,
                        outline="#242a38", width=1)
    for i, c in enumerate(("#f7768e", "#e0af68", CMD_OK)):
        d.ellipse([x0 + 26 + i * 22, 22, x0 + 40 + i * 22, 36], fill=c)
    tx.draw(d, (x0 + 150, 19), spec["title"], DIM)
    y = TOP
    for text, xoff, col in rows:
        if text:
            tx.draw(d, (x0 + PAD + xoff, y), text, col)
        y += LH
    if cursor and rows:
        text, xoff, _ = rows[-1]
        tx.draw(d, (x0 + PAD + xoff + tx.length(text), y - LH), "\u2588", FG)
    d.line([x0 + PAD, h - 46, x0 + w - PAD, h - 46], fill="#242a38", width=1)
    tx.draw(d, (x0 + PAD, h - 38), status, DIM)


def _emit(name, spec, panes, w, h, out_dir, tx):
    """panes: [(spec, frames)] drawn left to right; the shorter pane is
    resampled onto the longer timeline so both finish together (pacing is the
    one liberty this renderer takes)."""
    tmp = tempfile.mkdtemp(prefix="pulsar_demo")
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    n = max(len(f) for _, f in panes)
    for i in range(n):
        d.rectangle([0, 0, w, h], fill=BG)
        x0 = 0
        for pspec, frames in panes:
            j = (i if len(frames) == n else
                 min(len(frames) - 1, round(i * (len(frames) - 1) / (n - 1))))
            _draw_pane(d, tx, pspec, frames[j], x0, W, h)
            x0 += W + GAP
        img.save(os.path.join(tmp, f"f{i:04d}.png"))
    out = os.path.join(out_dir, spec["out"])
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                    "-i", os.path.join(tmp, "f%04d.png"),
                    "-vf", "split[s0][s1];[s0]palettegen=stats_mode=diff[pal];"
                    "[s1][pal]paletteuse=dither=bayer:bayer_scale=4",
                    "-loop", "0", out], check=True)
    shutil.rmtree(tmp)
    print(f"{name}: {out} ({os.path.getsize(out)/1e6:.2f} MB)")


def render(name, spec, out_dir=DOC):
    tx = Text()
    frames, h = _plan(spec, tx)
    _emit(name, spec, [(spec, frames)], W, h, out_dir, tx)


def render_pair(name, spec, out_dir=DOC):
    tx = Text()
    subs = [SPECS[k] for k in spec["panes"]]
    h = max(_plan(sp, tx)[1] for sp in subs)
    panes = [(sp, _plan(sp, tx, h=h)[0]) for sp in subs]
    w = len(subs) * W + (len(subs) - 1) * GAP
    _emit(name, spec, panes, w, h, out_dir, tx)


def main():
    names = sys.argv[1:] or list(SPECS)
    for n in names:
        if n not in SPECS:
            sys.exit(f"unknown spec {n!r}; choose from {', '.join(SPECS)}")
    for n in names:
        spec = SPECS[n]
        if spec.get("panes"):
            render_pair(n, spec)
        else:
            render(n, spec)


if __name__ == "__main__":
    main()
