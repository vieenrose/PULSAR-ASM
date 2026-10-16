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

# Ternary-Bonsai-2 (qwen35 hybrid, PTQ1_0) on a Jetson Nano Gen1, CUDA sm_53 engine fwd35
# (pulsar_arm/bonsai2/fwd35_sm53.cu, committed HEAD build, jetson_clocks). These clips are
# TIMED: each token appears at its measured step time from the run log, and loading (repack +
# upload) and prefill last their measured durations, so the reveal runs at the engine's real
# speed. Greedy decoding: every stream equals the PrismML fork's greedy stream over its full
# length (fork_greedy, f16 K/V), so the text is exactly what the fork itself would print.
# bonsai2-2b-nano-en: Ternary-Bonsai-2-2B PTQ1_0, fwd35 greedy, per-token text pieces and
# step times (ms) verbatim from the run log (STEP lines; token 0 comes out of the prefill);
# 100 tokens, <|im_end|> excluded. Load 5071 ms, prefill 24 tok 941 ms.
PIECES_B2_2B_EN = ["Here", " are", " the", " four", " seasons", " and", " one", " thing", " that", " changes", " in", " each", ":", "\n\n", "1", ".", " ", " **", "Spring", "**", " \u2013", " The", " **", "blo", "oms", "**", " of", " flowers", " and", " the", " **", "mel", "ting", "**", " of", " ice", ".", "\n", "2", ".", " ", " **", "Summer", "**", " \u2013", " The", " **", "long", "est", " days", "**", " and", " the", " **", "int", "ense", " heat", "**.", "\n", "3", ".", " ", " **", "Aut", "umn", "**", " \u2013", " The", " **", "fall", "ing", " leaves", "**", " and", " the", " **", "cool", "ing", " temperatures", "**.", "\n", "4", ".", " ", " **", "Winter", "**", " \u2013", " The", " **", "short", "est", " days", "**", " and", " the", " **", "cold", " weather", "**."]
MS_B2_2B_EN = [0.0, 68.1, 44.3, 44.2, 44.26, 44.3, 44.31, 44.36, 44.29, 44.61, 44.68, 44.39, 44.47, 44.35, 44.44, 44.39, 44.35, 45.14, 45.03, 44.44, 44.44, 44.41, 44.39, 44.51, 44.4, 44.51, 44.51, 44.76, 44.54, 44.58, 44.69, 45.52, 45.28, 45.34, 44.75, 44.73, 44.63, 44.68, 44.74, 44.8, 44.81, 45.54, 44.86, 44.78, 44.82, 44.81, 44.75, 45.01, 45.42, 44.94, 44.91, 44.79, 44.83, 45.0, 45.13, 44.89, 44.87, 44.89, 44.91, 44.9, 44.85, 45.0, 44.99, 44.93, 44.93, 45.69, 45.06, 45.0, 44.93, 45.12, 45.13, 45.07, 45.24, 45.13, 45.12, 45.2, 45.43, 45.21, 45.34, 45.26, 45.17, 45.14, 45.35, 45.23, 45.17, 45.24, 45.2, 45.17, 45.21, 45.31, 45.95, 45.31, 45.86, 45.29, 45.34, 45.31, 45.47, 45.44, 45.65, 45.6]

# bonsai2-2b-nano-zh-tw: Ternary-Bonsai-2-2B PTQ1_0, fwd35 greedy, per-token text pieces and
# step times (ms) verbatim from the run log (STEP lines; token 0 comes out of the prefill);
# 350 tokens, <|im_end|> excluded. Load 5230 ms, prefill 23 tok 907 ms.
PIECES_B2_2B_ZH_TW = ["\u56db\u5b63", "\u662f", "\u5730\u7403", "\u4e00\u5e74", "\u4e2d", "\u7531", "\u81ea\u7136", "\u73fe\u8c61", "\u4ea4\u66ff", "\u8b8a\u5316", "\u5f62\u6210\u7684", "\u6642\u9593", "\u5468\u671f", "\uff0c", "\u6bcf", "\u5b63", "\u90fd\u6703", "\u767c\u751f", "\u8a31\u591a", "\u8b8a\u5316", "\uff0c", "\u9019\u4e9b", "\u8b8a\u5316", "\u901a\u5e38", "\u8207", "\u5b63\u7bc0", "\u7684", "\u6eab\u5ea6", "\u3001", "\u967d\u5149", "\u3001", "\u6c23\u5019", "\u4ee5\u53ca", "\u751f\u7269", "\u6d3b\u52d5", "\u6709\u95dc", "\u3002", "\u4ee5\u4e0b\u662f", "\u56db\u5b63", "\u53ca\u5176", "\u6539\u8b8a", "\u7684", "\u8aaa\u660e", "\uff1a", "\n\n", "###", " **", "\u4e00", "\u3001", "\u6625\u5b63", "\uff08", "Spring", "\uff09**", "\n", "-", " **", "\u6539\u8b8a", "\u7684\u4e00\u4ef6\u4e8b", "**", "\uff1a", "**", "\u690d\u7269", "\u9577", "\u9577", "\uff0c", "\u82b1\u6735", "\u958b", "", "\u589c", "\uff0c", "\u5b63\u7bc0", "\u8b8a", "\u71b1", "**\u3002", "\n", "-", " **", "\u8aaa\u660e", "**", "\uff1a", "\u6625\u5b63", "\u662f", "\u690d\u7269", "\u751f\u9577", "\u6700", "\u6d3b\u8e8d", "\u7684", "\u5b63\u7bc0", "\uff0c", "\u967d\u5149", "\u5145\u8db3", "\uff0c", "\u6eab\u5ea6", "\u9010\u6f38", "\u5347\u9ad8", "\uff0c", "\u5c0e\u81f4", "\u690d\u7269", "\u8449", "\u7247", "\u8b8a", "\u7da0", "\u3001", "\u8449", "\u7247", "\u8b8a", "\u9577", "\uff0c", "\u82b1\u6735", "\u958b", "", "\u589c", "\uff0c", "\u9019\u662f", "\u5b63\u7bc0", "\u8b8a", "\u5316\u7684", "\u95dc\u9375", "\u6a19\u8a8c", "\u4e4b\u4e00", "\u3002", "\n\n", "###", " **", "\u4e8c", "\u3001", "\u590f\u5b63", "\uff08", "Summer", "\uff09**", "\n", "-", " **", "\u6539\u8b8a", "\u7684\u4e00\u4ef6\u4e8b", "**", "\uff1a", "**", "\u592a\u967d", "\u9ad8\u5ea6", "\u6700\u9ad8", "\uff0c", "\u71b1", "\u6d6a", "\u7206\u767c", "\uff0c", "\u690d\u7269", "\u8449", "\u7247", "\u8b8a", "\u9ec3", "**\u3002", "\n", "-", " **", "\u8aaa\u660e", "**", "\uff1a", "\u590f\u5b63", "\u662f", "\u592a\u967d", "\u6700", "\u5f37", "\u70c8\u7684", "\u5b63\u7bc0", "\uff0c", "\u967d\u5149", "\u76f4\u5c04", "\u6642\u9593", "\u9577", "\uff0c", "\u6eab\u5ea6", "\u6700\u9ad8", "\uff0c", "\u5c0e\u81f4", "\u690d\u7269", "\u8449", "\u7247", "\u8b8a", "\u9ec3", "\u3001", "\u8449", "\u7247", "\u8b8a", "\u4e7e", "\uff0c", "\u9019\u662f", "\u5b63\u7bc0", "\u8b8a", "\u5316\u7684", "\u53e6\u4e00\u500b", "\u91cd\u8981", "\u6a19\u8a8c", "\u3002", "\n\n", "###", " **", "\u4e09", "\u3001", "\u79cb\u5b63", "\uff08", "Aut", "umn", "\uff09**", "\n", "-", " **", "\u6539\u8b8a", "\u7684\u4e00\u4ef6\u4e8b", "**", "\uff1a", "**", "\u98a8", "\u66b4", "\u4f86", "\uff0c", "\u8449", "\u7247", "\u8b8a", "\u7d05", "\uff0c", "\u5b63\u7bc0", "\u8b8a", "\u6dbc", "**\u3002", "\n", "-", " **", "\u8aaa\u660e", "**", "\uff1a", "\u79cb\u5b63", "\u662f", "\u98a8", "\u66b4", "\u3001", "\u6dbc", "\u98a8", "\u548c", "\u8449", "\u7247", "\u8b8a", "\u7d05", "\u7684", "\u5b63\u7bc0", "\uff0c", "\u6eab\u5ea6", "\u9010\u6f38", "\u964d\u4f4e", "\uff0c", "\u98a8", "\u66b4", "\u4f86", "\uff0c", "\u690d\u7269", "\u8449", "\u7247", "\u8b8a", "\u7d05", "\uff0c", "\u9019\u662f", "\u5b63\u7bc0", "\u8b8a", "\u5316\u7684", "\u53e6\u4e00\u500b", "\u91cd\u8981", "\u6a19\u8a8c", "\u3002", "\n\n", "###", " **", "\u56db", "\u3001", "\u51ac\u5b63", "\uff08", "Winter", "\uff09**", "\n", "-", " **", "\u6539\u8b8a", "\u7684\u4e00\u4ef6\u4e8b", "**", "\uff1a", "**", "\u592a\u967d", "\u9ad8\u5ea6", "\u6700\u4f4e", "\uff0c", "\u96ea", "\u843d", "\uff0c", "\u690d\u7269", "\u4f11\u7720", "**\u3002", "\n", "-", " **", "\u8aaa\u660e", "**", "\uff1a", "\u51ac\u5b63", "\u662f", "\u592a\u967d", "\u9ad8\u5ea6", "\u6700\u4f4e", "\uff0c", "\u65e5\u7167", "\u6642\u9593", "\u6700\u77ed", "\uff0c", "\u6eab\u5ea6", "\u6700\u4f4e", "\uff0c", "\u5c0e\u81f4", "\u96ea", "\u843d", "\u3001", "\u690d\u7269", "\u4f11\u7720", "\uff0c", "\u9019\u662f", "\u5b63\u7bc0", "\u8b8a", "\u5316\u7684", "\u6700\u5f8c", "\u4e00\u500b", "\u6a19\u8a8c", "\u3002", "\n\n", "\u9019\u4e9b", "\u8b8a\u5316", "\u96d6\u7136", "\u4e0d\u540c", "\uff0c", "\u4f46", "\u90fd\u662f", "\u5730\u7403", "\u81ea\u7136", "\u74b0\u5883", "\u8b8a\u5316", "\u7684\u4e00\u90e8\u5206", "\uff0c", "\u53cd\u6620\u4e86", "\u5b63\u7bc0", "\u7684", "\u898f\u5f8b", "\u8207", "\u8b8a\u5316", "\u3002"]
MS_B2_2B_ZH_TW = [0.0, 66.37, 44.41, 44.74, 44.46, 44.41, 44.34, 44.47, 44.35, 44.4, 44.45, 44.31, 44.49, 44.32, 44.55, 45.07, 44.65, 44.37, 44.43, 44.49, 44.59, 44.44, 44.51, 44.54, 44.5, 44.69, 44.54, 44.56, 44.54, 44.53, 44.55, 45.0, 44.85, 44.7, 44.57, 44.56, 45.28, 44.75, 44.85, 44.83, 44.97, 44.64, 44.79, 44.7, 44.64, 44.89, 45.05, 44.84, 44.73, 44.72, 44.77, 44.77, 44.81, 44.79, 44.97, 45.42, 44.86, 44.85, 44.82, 44.93, 45.46, 45.54, 45.6, 45.98, 45.39, 45.41, 45.41, 46.18, 45.42, 45.34, 45.52, 45.45, 45.49, 45.38, 45.47, 45.54, 45.51, 45.46, 45.54, 45.57, 45.83, 45.5, 45.75, 45.73, 45.73, 45.64, 45.68, 45.61, 45.86, 45.7, 45.75, 45.66, 45.69, 45.81, 45.69, 45.75, 45.8, 45.71, 45.75, 45.97, 45.97, 45.87, 45.8, 45.92, 46.1, 45.74, 46.13, 45.95, 45.97, 46.09, 45.9, 45.9, 45.93, 46.03, 45.98, 46.05, 45.94, 46.04, 46.03, 46.15, 46.56, 45.99, 46.04, 46.11, 46.19, 46.14, 46.29, 46.12, 46.18, 46.05, 46.12, 46.09, 46.17, 46.24, 46.27, 46.21, 46.19, 46.3, 46.38, 46.58, 46.3, 46.25, 46.4, 46.33, 46.56, 46.36, 46.42, 46.58, 46.58, 46.41, 46.53, 46.48, 46.54, 46.38, 46.56, 46.46, 46.8, 46.63, 46.76, 47.08, 46.75, 46.72, 46.95, 46.95, 46.83, 47.7, 47.29, 46.58, 46.78, 46.77, 46.78, 46.61, 46.66, 46.8, 46.79, 46.7, 46.71, 47.04, 46.79, 47.07, 46.93, 46.95, 46.92, 46.95, 46.97, 46.91, 46.8, 47.02, 46.99, 46.94, 47.15, 46.97, 47.02, 47.05, 47.01, 47.04, 47.23, 47.15, 47.15, 47.25, 47.18, 47.14, 47.24, 47.37, 47.37, 47.38, 47.44, 47.45, 47.61, 47.43, 47.91, 47.87, 48.01, 47.47, 47.61, 47.7, 47.95, 47.72, 47.81, 47.98, 48.17, 47.87, 48.01, 48.04, 48.15, 48.4, 48.21, 48.24, 48.35, 48.22, 48.33, 48.44, 48.51, 48.34, 48.47, 48.53, 48.5, 48.69, 48.59, 48.66, 48.71, 48.58, 48.99, 48.81, 48.88, 48.88, 48.95, 48.95, 49.21, 48.89, 49.11, 48.99, 49.24, 49.18, 49.11, 49.1, 49.14, 49.32, 49.11, 49.2, 49.48, 49.15, 49.36, 49.3, 49.31, 49.33, 49.36, 49.41, 49.52, 49.62, 49.43, 49.5, 49.59, 49.67, 49.54, 49.51, 49.51, 49.71, 49.53, 49.59, 49.73, 49.58, 49.54, 49.62, 49.72, 49.64, 49.62, 49.83, 49.74, 49.7, 50.02, 49.84, 50.01, 49.93, 49.82, 49.9, 49.9, 49.88, 49.9, 50.04, 49.96, 49.97, 49.94, 49.97, 50.08, 50.16, 49.99, 50.03, 50.1, 50.11, 50.13, 50.12, 50.27, 50.14, 50.52, 50.24, 50.25, 50.21, 50.29, 50.46, 50.43, 50.65, 50.26, 50.61, 50.45, 50.34, 50.38, 50.41, 51.13, 50.37, 50.46, 50.41, 50.63, 50.81, 51.04, 50.47, 50.52, 50.49, 50.64, 50.73, 50.69, 50.58, 50.69, 50.8, 50.67, 50.9, 50.78, 50.95, 51.06, 50.75]

# bonsai2-4b-nano-en: Ternary-Bonsai-2-4B PTQ1_0, fwd35 greedy, per-token text pieces and
# step times (ms) verbatim from the run log (STEP lines; token 0 comes out of the prefill);
# 82 tokens, <|im_end|> excluded. Load 11144 ms, prefill 24 tok 2224 ms.
PIECES_B2_4B_EN = ["Here", " are", " the", " four", " seasons", " and", " one", " primary", " thing", " that", " changes", " in", " each", ":", "\n\n", "*", "  ", " **", "Spring", ":**", " The", " weather", " war", "ms", " up", " and", " **", "flowers", " bloom", "**.", "\n", "*", "  ", " **", "Summer", ":**", " The", " days", " are", " longest", " and", " **", "sun", "light", " is", " strongest", "**.", "\n", "*", "  ", " **", "Aut", "umn", " (", "Fall", "):**", " The", " leaves", " turn", " colors", " and", " **", "days", " get", " shorter", "**.", "\n", "*", "  ", " **", "Winter", ":**", " The", " days", " are", " shortest", " and", " **", "snow", " may", " fall", "**."]
MS_B2_4B_EN = [0.0, 124.49, 99.12, 99.12, 99.41, 99.93, 99.96, 99.85, 99.36, 99.31, 99.49, 99.83, 99.78, 99.6, 99.67, 99.92, 99.7, 100.18, 99.68, 99.99, 99.85, 100.19, 99.94, 100.09, 100.2, 100.11, 100.42, 100.82, 99.95, 99.88, 99.85, 100.07, 100.01, 100.04, 100.11, 100.17, 100.14, 100.74, 100.5, 100.38, 100.61, 100.52, 101.15, 100.6, 100.79, 100.67, 100.96, 100.89, 100.51, 100.63, 101.1, 100.77, 100.56, 100.7, 100.67, 100.89, 100.93, 101.46, 101.18, 101.37, 101.17, 101.54, 101.19, 101.57, 101.31, 101.08, 101.19, 101.51, 101.09, 101.44, 101.18, 101.34, 101.08, 102.12, 102.23, 101.65, 101.99, 102.03, 101.77, 101.93, 101.95, 101.88]

# bonsai2-4b-nano-zh-tw: Ternary-Bonsai-2-4B PTQ1_0, fwd35 greedy, per-token text pieces and
# step times (ms) verbatim from the run log (STEP lines; token 0 comes out of the prefill);
# 404 tokens, <|im_end|> excluded. Load 11431 ms, prefill 23 tok 2133 ms.
PIECES_B2_4B_ZH_TW = ["\u56db\u5b63", "\u662f", "\u5927\u81ea\u7136", "\u8f2a", "\u8ff4", "\u7684", "\u8c61\u5fb5", "\uff0c", "\u6bcf\u4e00", "\u5b63", "\u4e0d\u50c5", "\u6c23\u5019", "\u8b8a\u5316", "\uff0c", "\u4e5f", "\u4f34", "\u96a8\u8457", "\u751f\u7269", "\u6d3b\u52d5", "\u8207", "\u4eba\u985e", "\u751f\u6d3b\u7684", "\u4e0d\u540c", "\u5074", "\u9762", "\u3002", "\u4ee5\u4e0b\u662f", "\u56db\u5b63", "\u53ca\u5176", "\u4ee3\u8868\u6027", "\u6539\u8b8a", "\u7684\u4e8b", "\u9805", "\uff1a", "\n\n", "###", " ", "1", ".", " ", "\u6625\u5b63", " (", "Spring", ")", "\n", "*", "  ", " **", "\u6838\u5fc3", "\u6539\u8b8a", "\uff1a", "\u5f9e", "\u51ac", "\u7720", "\u5230", "\u6d3b\u8e8d", "**", "\n", "   ", " *", "  ", " **", "\u81ea\u7136", "\u754c", "**", "\uff1a", "\u51b0\u96ea", "\u878d\u5316", "\uff0c", "\u690d\u7269", "\u958b\u59cb", "\u840c", "\u767c", "\u65b0", "\u8449", "\uff0c", "\u52d5\u7269", "\u5f9e", "\u51ac", "\u7720", "\u4e2d", "\u9192", "\u4f86", "\uff0c", "\u9ce5", "\u985e", "\u958b\u59cb", "\u5b75", "\u5375", "\u3002", "\n", "   ", " *", "  ", " **", "\u4eba\u985e", "\u751f\u6d3b", "**", "\uff1a", "\u5929\u6c23", "\u8f49", "\u6696", "\uff0c", "\u4eba\u5011", "\u958b\u59cb", "\u5916\u51fa", "\u6d3b\u52d5", "\u3001", "\u8e0f\u9752", "\uff0c", "\u4e26", "\u958b\u59cb", "\u6e96\u5099", "\u590f\u5b63", "\u7684", "\u8863\u7269", "\u8207", "\u6d3b\u52d5", "\u3002", "\n\n", "###", " ", "2", ".", " ", "\u590f\u5b63", " (", "Summer", ")", "\n", "*", "  ", " **", "\u6838\u5fc3", "\u6539\u8b8a", "\uff1a", "\u7e41", "\u76db", "\u8207", "\u7e41", "\u69ae", "**", "\n", "   ", " *", "  ", " **", "\u81ea\u7136", "\u754c", "**", "\uff1a", "\u690d\u7269", "\u751f\u9577", "\u6700", "\u65fa\u76db", "\uff0c", "\u82b1\u6735", "\u958b", "\u76db", "\uff0c", "\u679c", "\u5be6", "\u6210\u719f", "\uff1b", "\u52d5\u7269", "\u6d3b\u52d5", "\u6642\u9593", "\u8f03", "\u9577", "\uff0c", "\u6606", "\u87f2", "\u6d3b\u8e8d", "\u3002", "\n", "   ", " *", "  ", " **", "\u4eba\u985e", "\u751f\u6d3b", "**", "\uff1a", "\u5929\u6c23", "\u901a\u5e38", "\u8f03", "\u71b1", "\uff0c", "\u4eba\u5011", "\u6ce8\u91cd", "\u907f\u6691", "\u3001", "\u6e38\u6cf3", "\u6216", "\u6236\u5916", "\u904b\u52d5", "\uff1b", "\u4e5f\u662f", "\u8fb2", "\u4f5c\u7269", "\u6536", "\u6210\u7684", "\u5b63\u7bc0", "\u3002", "\n\n", "###", " ", "3", ".", " ", "\u79cb\u5b63", " (", "Aut", "umn", ")", "\n", "*", "  ", " **", "\u6838\u5fc3", "\u6539\u8b8a", "\uff1a", "\u6210\u719f", "\u8207", "\u8f49\u63db", "**", "\n", "   ", " *", "  ", " **", "\u81ea\u7136", "\u754c", "**", "\uff1a", "\u6a39", "\u8449", "\u8b8a", "\u8272", "\u4e26", "\u843d", "\u8449", "\uff0c", "\u6e96\u5099", "\u9032\u5165", "\u51ac", "\u7720", "\uff1b", "\u679c", "\u5be6", "\u6210\u719f", "\u4e26", "\u6210\u719f", "\uff1b", "\u5929\u6c23", "\u9010\u6f38", "\u8f49", "\u6dbc", "\u3002", "\n", "   ", " *", "  ", " **", "\u4eba\u985e", "\u751f\u6d3b", "**", "\uff1a", "\u9019\u662f", "\u6536\u83b7", "\u7bc0", "\u4efd", "\uff0c", "\u4eba\u5011", "\u6536\u85cf", "\u7ce7", "\u98df", "\u3001", "\u6e96\u5099", "\u51ac", "\u8863", "\uff1b", "\u4e5f\u662f", "\u65c5\u904a", "\u65fa\u5b63", "\uff0c", "\u8a31\u591a", "\u5730\u5340", "\u958b\u59cb", "\u8209\u8fa6", "\u7bc0", "\u6176", "\u6d3b\u52d5", "\u3002", "\n\n", "###", " ", "4", ".", " ", "\u51ac\u5b63", " (", "Winter", ")", "\n", "*", "  ", " **", "\u6838\u5fc3", "\u6539\u8b8a", "\uff1a", "", "\u86f0", "\u7720", "\u8207", "\u4fdd\u5b58", "**", "\n", "   ", " *", "  ", " **", "\u81ea\u7136", "\u754c", "**", "\uff1a", "\u690d\u7269", "\u812b", "\u8449", "\u4f11\u7720", "\uff0c", "\u52d5\u7269", "\u9032\u5165", "\u51ac", "\u7720", "\u6216", "\u5c0b\u627e", "\u98df\u7269", "\uff1b", "\u5929\u6c23", "\u5bd2\u51b7", "\uff0c", "\u96ea", "\u971c", "\u5e38\u898b", "\u3002", "\n", "   ", " *", "  ", " **", "\u4eba\u985e", "\u751f\u6d3b", "**", "\uff1a", "\u5929\u6c23", "\u5bd2\u51b7", "\uff0c", "\u4eba\u5011", "\u4f9d\u8cf4", "\u706b", "\u7210", "\u3001", "\u6696", "\u623f", "\u6216", "\u52a0\u71b1", "\u8a2d\u5099", "\uff1b", "\u4e5f\u662f", "\u50b3\u7d71", "\u7bc0", "\u6176", "\uff08", "\u5982", "\u8056\u8a95", "\u7bc0", "\u3001", "\u6625\u7bc0", "\uff09", "\u7684", "\u6642\u671f", "\uff0c", "\u5f37\u8abf", "\u5bb6\u5ead", "\u5718", "\u5713", "\u3002", "\n\n", "\u9019", "\u56db\u500b", "\u5b63\u7bc0", "\u4ea4\u66ff", "\u9032\u884c", "\uff0c", "\u69cb", "\u6210\u4e86", "\u5730\u7403", "\u751f\u614b", "\u7cfb\u7d71", "\u904b", "\u884c\u7684", "\u7bc0\u594f", "\u3002"]
MS_B2_4B_ZH_TW = [0.0, 122.83, 99.45, 99.57, 99.72, 99.51, 99.56, 99.98, 99.8, 99.48, 99.72, 99.58, 99.43, 100.21, 99.56, 99.54, 99.5, 99.77, 100.12, 100.34, 99.8, 99.83, 99.81, 99.75, 99.75, 99.86, 100.36, 100.96, 100.91, 100.95, 100.15, 100.3, 100.48, 100.36, 100.41, 100.57, 100.92, 100.67, 100.9, 100.35, 100.28, 100.42, 100.38, 100.9, 100.63, 100.6, 100.7, 100.83, 101.37, 101.06, 101.05, 101.07, 101.22, 101.12, 101.21, 101.4, 101.3, 101.09, 101.19, 101.16, 101.03, 100.97, 101.05, 101.15, 101.01, 101.4, 101.61, 102.27, 101.69, 101.95, 102.67, 102.85, 102.53, 102.63, 101.86, 101.71, 101.66, 101.9, 101.63, 101.58, 101.48, 101.87, 101.72, 101.81, 102.06, 102.21, 102.0, 102.52, 102.25, 102.13, 102.46, 102.3, 102.47, 102.27, 102.35, 102.25, 102.7, 103.6, 102.75, 102.88, 102.44, 102.76, 102.63, 102.3, 102.34, 102.36, 102.89, 102.8, 102.61, 102.48, 102.76, 102.85, 102.95, 102.92, 103.12, 102.93, 103.74, 103.08, 103.31, 103.15, 103.69, 102.83, 102.83, 103.05, 103.04, 102.98, 103.43, 103.02, 103.02, 103.33, 103.54, 103.54, 103.77, 103.52, 103.63, 103.95, 104.07, 103.73, 103.6, 103.48, 103.51, 103.58, 103.71, 103.74, 103.6, 103.98, 103.86, 103.77, 104.12, 104.23, 104.93, 104.71, 104.26, 104.25, 104.53, 105.35, 104.5, 104.25, 104.16, 104.22, 104.21, 104.29, 104.24, 104.3, 104.9, 104.79, 104.55, 104.61, 104.63, 105.3, 104.64, 104.83, 104.94, 104.78, 105.13, 105.03, 105.32, 105.14, 105.49, 105.24, 105.62, 105.47, 106.07, 105.85, 105.5, 105.41, 105.31, 105.37, 106.22, 105.36, 105.63, 105.64, 107.21, 105.88, 105.93, 105.97, 106.04, 106.13, 106.07, 106.19, 106.3, 106.08, 106.8, 106.32, 106.4, 106.29, 106.42, 106.35, 106.57, 106.81, 106.54, 107.45, 107.49, 107.37, 107.73, 107.41, 107.64, 107.52, 107.61, 107.88, 107.86, 108.03, 107.73, 107.94, 108.04, 107.98, 108.18, 109.47, 108.77, 108.18, 109.55, 108.88, 108.91, 109.0, 109.67, 109.59, 109.99, 109.98, 109.95, 110.71, 109.97, 110.06, 110.22, 110.53, 110.32, 110.53, 110.99, 110.41, 110.54, 110.37, 110.49, 110.74, 111.04, 110.65, 111.22, 110.92, 111.24, 111.59, 111.95, 112.44, 111.76, 111.61, 111.68, 111.84, 111.7, 111.93, 112.5, 112.33, 111.79, 111.91, 111.95, 111.94, 111.98, 111.92, 112.12, 112.61, 112.47, 112.71, 112.5, 112.84, 113.01, 112.89, 112.75, 112.84, 113.26, 112.95, 112.65, 112.57, 112.76, 112.74, 113.42, 112.81, 112.83, 113.18, 113.26, 113.33, 113.25, 113.33, 113.44, 113.79, 113.66, 113.68, 114.18, 113.3, 113.58, 113.39, 114.28, 113.62, 113.69, 113.79, 113.92, 113.84, 113.75, 113.65, 113.94, 113.94, 114.42, 114.06, 114.27, 114.44, 114.39, 114.24, 114.45, 114.67, 114.76, 114.63, 114.59, 114.77, 115.26, 114.87, 114.84, 114.73, 114.96, 114.93, 114.82, 115.03, 115.55, 115.33, 115.26, 115.29, 115.78, 115.79, 115.59, 115.51, 115.62, 116.02, 116.11, 115.73, 116.13, 115.43, 115.52, 115.77, 115.74, 115.6, 116.32, 115.94, 115.89, 116.11, 116.36, 116.68, 116.53, 116.49, 116.87, 116.81, 116.64, 116.62, 116.64, 116.34, 116.43, 116.5, 116.66, 116.96, 116.73, 116.8, 116.63, 116.71, 116.86, 117.05, 117.44, 120.5, 117.7, 117.66, 117.35, 117.37, 117.34, 117.58, 117.61, 117.66, 117.91, 117.67, 118.0, 117.71, 117.92, 118.09, 117.83, 117.89, 117.77, 118.11, 118.0, 117.89, 117.78, 118.02, 117.9, 118.15]


# bonsai2-9b-nano-en: Ternary-Bonsai-2-9B PTQ1_0, fwd35 greedy, per-token text pieces and
# step times (ms) verbatim from the run log (STEP lines; token 0 comes out of the prefill);
# 99 tokens, <|im_end|> excluded. Load 23459 ms, prefill 24 tok 3885 ms.
PIECES_B2_9B_EN = ["Here", " are", " the", " four", " seasons", " along", " with", " one", " distinct", " thing", " that", " changes", " in", " each", ":", "\n\n", "*", "  ", " **", "Spring", ":**", " The", " days", " become", " longer", ",", " and", " flowers", " begin", " to", " bloom", ".", "\n", "*", "  ", " **", "Summer", ":**", " The", " sun", " stays", " high", " in", " the", " sky", ",", " resulting", " in", " the", " warm", "est", " temperatures", " of", " the", " year", ".", "\n", "*", "  ", " **", "Aut", "umn", " (", "Fall", "):**", " Leaves", " change", " color", " from", " green", " to", " red", ",", " orange", ",", " and", " yellow", " before", " falling", " off", " the", " trees", ".", "\n", "*", "  ", " **", "Winter", ":**", " Snow", " begins", " to", " fall", ",", " and", " temperatures", " drop", " significantly", "."]
MS_B2_9B_EN = [0.0, 209.54, 171.12, 172.27, 170.79, 171.02, 171.1, 171.59, 171.79, 180.98, 172.24, 171.93, 171.83, 172.25, 172.37, 171.99, 171.7, 171.41, 171.56, 172.08, 180.25, 180.52, 171.48, 176.37, 172.95, 171.87, 172.04, 172.75, 181.24, 173.07, 172.94, 182.39, 172.42, 172.6, 173.0, 172.9, 173.31, 171.99, 172.16, 172.06, 177.5, 177.6, 172.93, 172.26, 172.09, 173.09, 172.9, 173.51, 174.95, 173.04, 173.17, 173.11, 172.88, 173.61, 173.63, 172.82, 172.37, 172.4, 172.65, 173.45, 172.6, 172.8, 179.33, 174.93, 184.96, 195.9, 175.5, 173.75, 173.52, 173.48, 174.4, 173.62, 173.94, 174.06, 174.12, 173.96, 174.48, 179.45, 173.94, 181.92, 174.11, 173.46, 174.13, 173.72, 173.36, 173.7, 174.53, 173.97, 173.64, 173.6, 175.36, 174.9, 174.62, 180.94, 174.77, 174.97, 174.79, 175.07, 174.86]

# bonsai2-9b-nano-zh-tw: Ternary-Bonsai-2-9B PTQ1_0, fwd35 greedy, per-token text pieces and
# step times (ms) verbatim from the run log (STEP lines; token 0 comes out of the prefill);
# 92 tokens, <|im_end|> excluded. Load 23591 ms, prefill 23 tok 3763 ms.
PIECES_B2_9B_ZH_TW = ["\u4ee5\u4e0b\u662f", "\u56db\u5b63", "\u7684", "\u5217\u8868", "\uff0c", "\u4ee5\u53ca", "\u6bcf\u4e00", "\u5b63", "\u6703", "\u6539\u8b8a", "\u7684\u4e00\u4ef6", "\u5178\u578b", "\u4e8b\u4ef6", "\uff1a", "\n\n", "*", "  ", " **", "\u6625\u5b63", "**", "\uff1a", "\u842c", "\u7269", "\u5fa9", "", "\u7526", "\uff0c", "**", "\u82b1\u6735", "\u958b\u59cb", "\u7dbb", "\u653e", "**\u3002", "\n", "*", "  ", " **", "\u590f\u5b63", "**", "\uff1a", "\u6c23", "\u6eab", "\u5347\u9ad8", "\uff0c", "**", "\u591c\u665a", "\u7684", "\u868a", "\u87f2", "\u958b\u59cb", "\u6d3b\u8e8d", "**\u3002", "\n", "*", "  ", " **", "\u79cb\u5b63", "**", "\uff1a", "\u6c23", "\u6eab", "\u8f49", "\u6dbc", "\uff0c", "**", "\u6a39", "\u8449", "\u958b\u59cb", "\u8b8a", "\u8272", "\u4e26", "\u98c4", "\u843d", "**\u3002", "\n", "*", "  ", " **", "\u51ac\u5b63", "**", "\uff1a", "\u6c23", "\u6eab", "\u5bd2\u51b7", "\uff0c", "**", "\u5730\u9762", "\u7d50", "\u51b0", "\u6216", "\u4e0b\u96ea", "**\u3002"]
MS_B2_9B_ZH_TW = [0.0, 207.6, 171.36, 171.34, 179.77, 171.01, 171.03, 170.96, 176.82, 172.71, 181.42, 180.75, 171.47, 171.13, 177.21, 171.48, 171.12, 171.23, 171.47, 171.78, 172.61, 172.15, 172.3, 171.92, 171.87, 172.43, 172.6, 172.38, 172.29, 177.67, 171.42, 172.22, 171.77, 180.39, 171.69, 171.85, 171.89, 172.47, 179.61, 172.53, 172.7, 172.57, 172.72, 173.29, 172.78, 172.58, 172.89, 172.27, 172.25, 172.45, 172.17, 181.35, 172.43, 172.3, 173.03, 172.32, 173.51, 173.04, 173.23, 173.22, 174.09, 173.36, 173.29, 173.34, 173.42, 172.49, 178.28, 172.76, 172.72, 172.73, 172.69, 175.52, 173.38, 172.85, 173.45, 173.9, 175.33, 173.95, 173.92, 173.96, 173.8, 173.9, 174.03, 174.56, 174.36, 176.86, 174.0, 174.17, 174.31, 174.64, 174.3, 181.34]


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

    "bonsai2-2b-nano-en": dict(
        out="bonsai2-2b-nano-chat-en.gif",
        title="pulsar \u00b7 Ternary-Bonsai-2-2B PTQ1_0 \u00b7 jetson nano cuda",
        cmd="$ E_STOP=1 ./fwd35 Ternary-Bonsai-2-2B-PTQ1_0.gguf 1500 248045 846 198 826 279 2943 15127 321 799 3065 421 4203 303 1754 13 248046 198 248045 74455 198 248068 271 248069 271",
        you="> List the four seasons and one thing that changes in each.",
        tpl="prompt ids: 248045 846 198 826 279 2943 15127 321 799 3065 421 4203 303 1754 13 248046 198 248045 74455 198 248068 271 248069 271",
        response="".join(PIECES_B2_2B_EN),
        pieces=PIECES_B2_2B_EN,
        ms=MS_B2_2B_EN,
        load_ms=5071.0, prefill_ms=941.0, n_prompt=24,
        status="decode 22.1 tok/s \u00b7 prefill 25.5 tok/s \u00b7 greedy \u00b7 Jetson Nano GPU (sm_53)",
        wrap="word"),
    "bonsai2-2b-nano-zh-tw": dict(
        out="bonsai2-2b-nano-chat-zh-tw.gif",
        title="pulsar \u00b7 Ternary-Bonsai-2-2B PTQ1_0 \u00b7 jetson nano cuda",
        cmd="$ E_STOP=1 ./fwd35 Ternary-Bonsai-2-2B-PTQ1_0.gguf 1500 248045 846 198 99270 115992 104647 3709 98404 98443 96681 96594 110476 127312 1710 248046 198 248045 74455 198 248068 271 248069 271",
        you="> \u8acb\u5217\u51fa\u56db\u5b63\uff0c\u4ee5\u53ca\u6bcf\u4e00\u5b63\u6703\u6539\u8b8a\u7684\u4e00\u4ef6\u4e8b\u3002",
        tpl="prompt ids: 248045 846 198 99270 115992 104647 3709 98404 98443 96681 96594 110476 127312 1710 248046 198 248045 74455 198 248068 271 248069 271",
        response="".join(PIECES_B2_2B_ZH_TW),
        pieces=PIECES_B2_2B_ZH_TW,
        ms=MS_B2_2B_ZH_TW,
        load_ms=5230.0, prefill_ms=907.0, n_prompt=23,
        status="decode 21.1 tok/s \u00b7 prefill 25.4 tok/s \u00b7 greedy \u00b7 Jetson Nano GPU (sm_53)",
        wrap="char"),
    "bonsai2-4b-nano-en": dict(
        out="bonsai2-4b-nano-chat-en.gif",
        title="pulsar \u00b7 Ternary-Bonsai-2-4B PTQ1_0 \u00b7 jetson nano cuda",
        cmd="$ E_STOP=1 ./fwd35 Ternary-Bonsai-2-4B-PTQ1_0.gguf 1500 248045 846 198 826 279 2943 15127 321 799 3065 421 4203 303 1754 13 248046 198 248045 74455 198 248068 271 248069 271",
        you="> List the four seasons and one thing that changes in each.",
        tpl="prompt ids: 248045 846 198 826 279 2943 15127 321 799 3065 421 4203 303 1754 13 248046 198 248045 74455 198 248068 271 248069 271",
        response="".join(PIECES_B2_4B_EN),
        pieces=PIECES_B2_4B_EN,
        ms=MS_B2_4B_EN,
        load_ms=11144.0, prefill_ms=2224.0, n_prompt=24,
        status="decode 9.9 tok/s \u00b7 prefill 10.8 tok/s \u00b7 greedy \u00b7 Jetson Nano GPU (sm_53)",
        wrap="word"),
    "bonsai2-4b-nano-zh-tw": dict(
        out="bonsai2-4b-nano-chat-zh-tw.gif",
        title="pulsar \u00b7 Ternary-Bonsai-2-4B PTQ1_0 \u00b7 jetson nano cuda",
        cmd="$ E_STOP=1 ./fwd35 Ternary-Bonsai-2-4B-PTQ1_0.gguf 1500 248045 846 198 99270 115992 104647 3709 98404 98443 96681 96594 110476 127312 1710 248046 198 248045 74455 198 248068 271 248069 271",
        you="> \u8acb\u5217\u51fa\u56db\u5b63\uff0c\u4ee5\u53ca\u6bcf\u4e00\u5b63\u6703\u6539\u8b8a\u7684\u4e00\u4ef6\u4e8b\u3002",
        tpl="prompt ids: 248045 846 198 99270 115992 104647 3709 98404 98443 96681 96594 110476 127312 1710 248046 198 248045 74455 198 248068 271 248069 271",
        response="".join(PIECES_B2_4B_ZH_TW),
        pieces=PIECES_B2_4B_ZH_TW,
        ms=MS_B2_4B_ZH_TW,
        load_ms=11431.0, prefill_ms=2133.0, n_prompt=23,
        status="decode 9.2 tok/s \u00b7 prefill 10.8 tok/s \u00b7 greedy \u00b7 Jetson Nano GPU (sm_53)",
        wrap="char"),
    "bonsai2-9b-nano-en": dict(
        out="bonsai2-9b-nano-chat-en.gif",
        title="pulsar \u00b7 Ternary-Bonsai-2-9B PTQ1_0 \u00b7 jetson nano cuda",
        cmd="$ E_STOP=1 ./fwd35 Ternary-Bonsai-2-9B-PTQ1_0.gguf 1500 248045 846 198 826 279 2943 15127 321 799 3065 421 4203 303 1754 13 248046 198 248045 74455 198 248068 271 248069 271",
        you="> List the four seasons and one thing that changes in each.",
        tpl="prompt ids: 248045 846 198 826 279 2943 15127 321 799 3065 421 4203 303 1754 13 248046 198 248045 74455 198 248068 271 248069 271",
        response="".join(PIECES_B2_9B_EN),
        pieces=PIECES_B2_9B_EN,
        ms=MS_B2_9B_EN,
        load_ms=23459.0, prefill_ms=3885.0, n_prompt=24,
        status="decode 5.7 tok/s \u00b7 prefill 6.2 tok/s \u00b7 greedy \u00b7 Jetson Nano GPU (sm_53)",
        wrap="word"),
    "bonsai2-9b-nano-zh-tw": dict(
        out="bonsai2-9b-nano-chat-zh-tw.gif",
        title="pulsar \u00b7 Ternary-Bonsai-2-9B PTQ1_0 \u00b7 jetson nano cuda",
        cmd="$ E_STOP=1 ./fwd35 Ternary-Bonsai-2-9B-PTQ1_0.gguf 1500 248045 846 198 99270 115992 104647 3709 98404 98443 96681 96594 110476 127312 1710 248046 198 248045 74455 198 248068 271 248069 271",
        you="> \u8acb\u5217\u51fa\u56db\u5b63\uff0c\u4ee5\u53ca\u6bcf\u4e00\u5b63\u6703\u6539\u8b8a\u7684\u4e00\u4ef6\u4e8b\u3002",
        tpl="prompt ids: 248045 846 198 99270 115992 104647 3709 98404 98443 96681 96594 110476 127312 1710 248046 198 248045 74455 198 248068 271 248069 271",
        response="".join(PIECES_B2_9B_ZH_TW),
        pieces=PIECES_B2_9B_ZH_TW,
        ms=MS_B2_9B_ZH_TW,
        load_ms=23591.0, prefill_ms=3763.0, n_prompt=23,
        status="decode 5.7 tok/s \u00b7 prefill 6.1 tok/s \u00b7 greedy \u00b7 Jetson Nano GPU (sm_53)",
        wrap="char"),
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
    if spec.get("ms"):
        _timed(spec, frame, compose, head, stages, full, tx, text_w, mid_px, fps)
        _encode(name, spec, tmp, fps, out_dir)
        return
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

    _encode(name, spec, tmp, fps, out_dir)


def _timed(spec, frame, compose, head, stages, full, tx, text_w, mid_px, fps):
    """Real-time reveal: token k is on screen from the cumulative measured
    time of tokens 0..k; loading and prefill hold for their measured times."""
    for _ in range(max(1, round(spec["load_ms"] / 1000 * fps))):
        frame(head + [("", 0.0, FG)], "loading engine: repack + upload + first forward", False)
    n_pre = max(len(stages) - 1, 1)
    per = max(1, round(spec["prefill_ms"] / 1000 * fps / n_pre))
    for st in stages[1:]:
        for _ in range(per):
            frame(st + [("", 0.0, FG)], f"prefill ({spec.get('n_prompt', 24)} prompt tokens)", False)
    cum, acc = [], 0.0
    for ms in spec["ms"]:
        acc += ms
        cum.append(acc)
    n = len(cum)
    f = 0
    while True:
        now = f * 1000.0 / fps
        k = sum(1 for c in cum if c <= now)
        part = "".join(spec["pieces"][:k])
        blines = wrap(part, tx, text_w - mid_px, spec["wrap"])
        body = [("model> " + blines[0], 0.0, FG)] + \
               [(t, mid_px, FG) for t in blines[1:]]
        frame(compose(body), f"{k}/{n} tokens \u00b7 " + spec["status"], f % 2 == 1)
        if k == n:
            break
        f += 1
    body = [("model> " + full[0], 0.0, FG)] + [(t, mid_px, FG)
                                               for t in full[1:]]
    for _ in range(3 * fps):
        frame(compose(body), f"{n}/{n} tokens \u00b7 " + spec["status"], False)


def _encode(name, spec, tmp, fps, out_dir):
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
