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
# (pulsar_arm/bonsai2/fwd35_sm53.cu at 1ab76d2, jetson_clocks, defaults incl. 8-token prefill chunks). These clips are
# TIMED: each token appears at its measured step time from the run log, and loading (repack +
# upload) and prefill last their measured durations, so the reveal runs at the engine's real
# speed. Greedy decoding: every stream equals the PrismML fork's greedy stream over its full
# length (fork_greedy, f16 K/V), so the text is exactly what the fork itself would print.
# bonsai2-2b-nano-en: Ternary-Bonsai-2-2B PTQ1_0, fwd35 greedy, per-token text pieces and
# step times (ms) verbatim from the run log (STEP lines; token 0 comes out of the prefill);
# 100 tokens, <|im_end|> excluded. Load 3368 ms, prefill 24 tok 413 ms.
PIECES_B2_2B_EN = ["Here", " are", " the", " four", " seasons", " and", " one", " thing", " that", " changes", " in", " each", ":", "\n\n", "1", ".", " ", " **", "Spring", "**", " \u2013", " The", " **", "blo", "oms", "**", " of", " flowers", " and", " the", " **", "mel", "ting", "**", " of", " ice", ".", "\n", "2", ".", " ", " **", "Summer", "**", " \u2013", " The", " **", "long", "est", " days", "**", " and", " the", " **", "int", "ense", " heat", "**.", "\n", "3", ".", " ", " **", "Aut", "umn", "**", " \u2013", " The", " **", "fall", "ing", " leaves", "**", " and", " the", " **", "cool", "ing", " temperatures", "**.", "\n", "4", ".", " ", " **", "Winter", "**", " \u2013", " The", " **", "short", "est", " days", "**", " and", " the", " **", "cold", " weather", "**."]
MS_B2_2B_EN = [0.0, 57.19, 34.85, 35.03, 34.74, 34.84, 34.86, 34.78, 34.65, 34.86, 34.88, 34.81, 34.78, 34.95, 34.68, 35.11, 34.86, 34.93, 35.1, 34.95, 34.65, 35.06, 34.93, 34.96, 34.97, 34.94, 34.95, 34.72, 34.94, 34.96, 34.74, 34.94, 34.91, 34.85, 34.94, 34.67, 34.88, 34.94, 34.85, 34.88, 34.8, 34.67, 34.78, 34.84, 34.83, 34.97, 34.93, 34.94, 34.97, 34.77, 34.63, 34.84, 34.71, 34.65, 35.1, 34.94, 34.95, 34.97, 34.9, 34.68, 34.82, 34.76, 34.69, 34.76, 34.72, 34.59, 34.71, 34.87, 34.96, 34.97, 34.9, 34.75, 34.81, 34.86, 34.7, 34.76, 34.85, 34.94, 34.97, 34.89, 34.95, 34.96, 34.86, 34.83, 34.94, 34.84, 34.75, 34.91, 34.83, 34.81, 34.85, 34.89, 34.96, 34.86, 34.86, 34.79, 34.92, 34.9, 34.84, 34.96]

# bonsai2-2b-nano-zh-tw: Ternary-Bonsai-2-2B PTQ1_0, fwd35 greedy, per-token text pieces and
# step times (ms) verbatim from the run log (STEP lines; token 0 comes out of the prefill);
# 350 tokens, <|im_end|> excluded. Load 3322 ms, prefill 23 tok 408 ms.
PIECES_B2_2B_ZH_TW = ["\u56db\u5b63", "\u662f", "\u5730\u7403", "\u4e00\u5e74", "\u4e2d", "\u7531", "\u81ea\u7136", "\u73fe\u8c61", "\u4ea4\u66ff", "\u8b8a\u5316", "\u5f62\u6210\u7684", "\u6642\u9593", "\u5468\u671f", "\uff0c", "\u6bcf", "\u5b63", "\u90fd\u6703", "\u767c\u751f", "\u8a31\u591a", "\u8b8a\u5316", "\uff0c", "\u9019\u4e9b", "\u8b8a\u5316", "\u901a\u5e38", "\u8207", "\u5b63\u7bc0", "\u7684", "\u6eab\u5ea6", "\u3001", "\u967d\u5149", "\u3001", "\u6c23\u5019", "\u4ee5\u53ca", "\u751f\u7269", "\u6d3b\u52d5", "\u6709\u95dc", "\u3002", "\u4ee5\u4e0b\u662f", "\u56db\u5b63", "\u53ca\u5176", "\u6539\u8b8a", "\u7684", "\u8aaa\u660e", "\uff1a", "\n\n", "###", " **", "\u4e00", "\u3001", "\u6625\u5b63", "\uff08", "Spring", "\uff09**", "\n", "-", " **", "\u6539\u8b8a", "\u7684\u4e00\u4ef6\u4e8b", "**", "\uff1a", "**", "\u690d\u7269", "\u9577", "\u9577", "\uff0c", "\u82b1\u6735", "\u958b", "", "\u589c", "\uff0c", "\u5b63\u7bc0", "\u8b8a", "\u71b1", "**\u3002", "\n", "-", " **", "\u8aaa\u660e", "**", "\uff1a", "\u6625\u5b63", "\u662f", "\u690d\u7269", "\u751f\u9577", "\u6700", "\u6d3b\u8e8d", "\u7684", "\u5b63\u7bc0", "\uff0c", "\u967d\u5149", "\u5145\u8db3", "\uff0c", "\u6eab\u5ea6", "\u9010\u6f38", "\u5347\u9ad8", "\uff0c", "\u5c0e\u81f4", "\u690d\u7269", "\u8449", "\u7247", "\u8b8a", "\u7da0", "\u3001", "\u8449", "\u7247", "\u8b8a", "\u9577", "\uff0c", "\u82b1\u6735", "\u958b", "", "\u589c", "\uff0c", "\u9019\u662f", "\u5b63\u7bc0", "\u8b8a", "\u5316\u7684", "\u95dc\u9375", "\u6a19\u8a8c", "\u4e4b\u4e00", "\u3002", "\n\n", "###", " **", "\u4e8c", "\u3001", "\u590f\u5b63", "\uff08", "Summer", "\uff09**", "\n", "-", " **", "\u6539\u8b8a", "\u7684\u4e00\u4ef6\u4e8b", "**", "\uff1a", "**", "\u592a\u967d", "\u9ad8\u5ea6", "\u6700\u9ad8", "\uff0c", "\u71b1", "\u6d6a", "\u7206\u767c", "\uff0c", "\u690d\u7269", "\u8449", "\u7247", "\u8b8a", "\u9ec3", "**\u3002", "\n", "-", " **", "\u8aaa\u660e", "**", "\uff1a", "\u590f\u5b63", "\u662f", "\u592a\u967d", "\u6700", "\u5f37", "\u70c8\u7684", "\u5b63\u7bc0", "\uff0c", "\u967d\u5149", "\u76f4\u5c04", "\u6642\u9593", "\u9577", "\uff0c", "\u6eab\u5ea6", "\u6700\u9ad8", "\uff0c", "\u5c0e\u81f4", "\u690d\u7269", "\u8449", "\u7247", "\u8b8a", "\u9ec3", "\u3001", "\u8449", "\u7247", "\u8b8a", "\u4e7e", "\uff0c", "\u9019\u662f", "\u5b63\u7bc0", "\u8b8a", "\u5316\u7684", "\u53e6\u4e00\u500b", "\u91cd\u8981", "\u6a19\u8a8c", "\u3002", "\n\n", "###", " **", "\u4e09", "\u3001", "\u79cb\u5b63", "\uff08", "Aut", "umn", "\uff09**", "\n", "-", " **", "\u6539\u8b8a", "\u7684\u4e00\u4ef6\u4e8b", "**", "\uff1a", "**", "\u98a8", "\u66b4", "\u4f86", "\uff0c", "\u8449", "\u7247", "\u8b8a", "\u7d05", "\uff0c", "\u5b63\u7bc0", "\u8b8a", "\u6dbc", "**\u3002", "\n", "-", " **", "\u8aaa\u660e", "**", "\uff1a", "\u79cb\u5b63", "\u662f", "\u98a8", "\u66b4", "\u3001", "\u6dbc", "\u98a8", "\u548c", "\u8449", "\u7247", "\u8b8a", "\u7d05", "\u7684", "\u5b63\u7bc0", "\uff0c", "\u6eab\u5ea6", "\u9010\u6f38", "\u964d\u4f4e", "\uff0c", "\u98a8", "\u66b4", "\u4f86", "\uff0c", "\u690d\u7269", "\u8449", "\u7247", "\u8b8a", "\u7d05", "\uff0c", "\u9019\u662f", "\u5b63\u7bc0", "\u8b8a", "\u5316\u7684", "\u53e6\u4e00\u500b", "\u91cd\u8981", "\u6a19\u8a8c", "\u3002", "\n\n", "###", " **", "\u56db", "\u3001", "\u51ac\u5b63", "\uff08", "Winter", "\uff09**", "\n", "-", " **", "\u6539\u8b8a", "\u7684\u4e00\u4ef6\u4e8b", "**", "\uff1a", "**", "\u592a\u967d", "\u9ad8\u5ea6", "\u6700\u4f4e", "\uff0c", "\u96ea", "\u843d", "\uff0c", "\u690d\u7269", "\u4f11\u7720", "**\u3002", "\n", "-", " **", "\u8aaa\u660e", "**", "\uff1a", "\u51ac\u5b63", "\u662f", "\u592a\u967d", "\u9ad8\u5ea6", "\u6700\u4f4e", "\uff0c", "\u65e5\u7167", "\u6642\u9593", "\u6700\u77ed", "\uff0c", "\u6eab\u5ea6", "\u6700\u4f4e", "\uff0c", "\u5c0e\u81f4", "\u96ea", "\u843d", "\u3001", "\u690d\u7269", "\u4f11\u7720", "\uff0c", "\u9019\u662f", "\u5b63\u7bc0", "\u8b8a", "\u5316\u7684", "\u6700\u5f8c", "\u4e00\u500b", "\u6a19\u8a8c", "\u3002", "\n\n", "\u9019\u4e9b", "\u8b8a\u5316", "\u96d6\u7136", "\u4e0d\u540c", "\uff0c", "\u4f46", "\u90fd\u662f", "\u5730\u7403", "\u81ea\u7136", "\u74b0\u5883", "\u8b8a\u5316", "\u7684\u4e00\u90e8\u5206", "\uff0c", "\u53cd\u6620\u4e86", "\u5b63\u7bc0", "\u7684", "\u898f\u5f8b", "\u8207", "\u8b8a\u5316", "\u3002"]
MS_B2_2B_ZH_TW = [0.0, 57.62, 34.91, 34.93, 34.82, 34.82, 34.88, 34.57, 34.61, 34.5, 34.58, 34.85, 34.95, 34.97, 34.93, 34.62, 34.55, 34.71, 34.46, 34.55, 34.54, 34.43, 34.8, 34.5, 34.47, 34.88, 34.93, 34.48, 34.65, 34.68, 34.63, 34.79, 34.66, 34.76, 34.98, 34.92, 34.5, 34.71, 34.65, 34.51, 34.6, 34.64, 34.5, 34.6, 34.55, 34.53, 34.72, 34.61, 34.47, 34.77, 34.58, 34.64, 34.68, 34.71, 34.55, 34.61, 34.71, 34.54, 34.57, 34.93, 34.94, 34.62, 34.68, 34.58, 34.79, 34.76, 34.51, 34.63, 34.71, 34.62, 34.67, 34.89, 34.94, 34.66, 34.59, 34.64, 34.48, 34.67, 34.66, 34.62, 34.72, 34.61, 34.87, 34.97, 34.64, 34.59, 34.67, 34.7, 34.69, 34.72, 34.64, 34.72, 34.79, 34.66, 34.64, 34.98, 34.93, 34.61, 34.74, 34.65, 34.61, 34.83, 34.68, 34.57, 34.96, 34.95, 34.59, 34.73, 34.79, 34.62, 34.73, 34.65, 34.59, 34.71, 34.69, 34.68, 34.7, 34.93, 34.53, 34.72, 34.78, 34.65, 34.66, 34.78, 34.73, 34.82, 34.77, 34.55, 34.79, 34.65, 34.6, 34.71, 34.68, 34.7, 34.76, 34.72, 34.6, 34.76, 34.69, 34.65, 34.83, 34.81, 34.87, 34.96, 34.8, 34.67, 34.86, 34.82, 34.81, 34.79, 34.92, 34.94, 34.98, 34.93, 34.96, 34.73, 34.83, 34.76, 34.79, 34.75, 34.69, 34.8, 34.69, 34.66, 34.87, 34.92, 34.72, 34.79, 34.77, 34.66, 34.9, 34.8, 34.7, 34.88, 35.35, 34.93, 34.96, 34.75, 34.73, 34.87, 34.86, 34.75, 34.84, 34.9, 34.92, 34.96, 34.93, 34.95, 34.92, 34.87, 34.74, 34.71, 34.79, 34.67, 35.0, 34.91, 34.96, 34.97, 34.93, 34.72, 34.78, 34.77, 34.71, 34.81, 34.95, 34.92, 34.98, 34.95, 34.93, 34.96, 34.85, 34.79, 34.93, 34.79, 34.81, 34.88, 35.0, 34.68, 34.84, 34.92, 34.95, 34.9, 34.87, 34.82, 34.92, 34.84, 34.75, 34.89, 34.89, 34.83, 35.0, 34.91, 34.81, 34.98, 34.92, 34.86, 34.94, 34.88, 35.1, 34.98, 34.94, 34.95, 34.98, 34.85, 34.83, 34.86, 34.92, 34.79, 34.89, 34.92, 34.99, 34.95, 35.01, 34.91, 35.04, 35.01, 34.84, 35.02, 34.97, 34.97, 35.08, 35.19, 34.96, 35.05, 34.95, 34.92, 35.06, 35.06, 34.92, 35.04, 35.01, 34.99, 35.05, 34.93, 34.91, 35.09, 34.97, 34.93, 35.11, 35.1, 35.05, 34.95, 34.94, 34.98, 35.06, 34.98, 34.89, 35.01, 34.98, 34.96, 35.16, 34.93, 35.1, 35.05, 35.34, 35.12, 35.08, 35.03, 35.06, 35.22, 35.0, 35.19, 35.1, 35.02, 35.03, 35.05, 34.84, 35.07, 34.98, 34.96, 35.09, 35.35, 34.97, 35.07, 35.08, 34.94, 35.0, 35.21, 35.0, 35.19, 35.04, 35.03, 35.12, 35.07, 35.08, 35.19, 35.17, 35.04, 35.06, 35.01, 34.97, 35.07, 35.02, 34.96, 35.05, 35.02, 34.91, 35.17, 35.0, 34.89, 34.99, 35.0, 35.08, 35.07, 34.99, 35.03, 35.13, 35.09, 34.96, 35.13]

# bonsai2-4b-nano-en: Ternary-Bonsai-2-4B PTQ1_0, fwd35 greedy, per-token text pieces and
# step times (ms) verbatim from the run log (STEP lines; token 0 comes out of the prefill);
# 82 tokens, <|im_end|> excluded. Load 9470 ms, prefill 24 tok 971 ms.
PIECES_B2_4B_EN = ["Here", " are", " the", " four", " seasons", " and", " one", " primary", " thing", " that", " changes", " in", " each", ":", "\n\n", "*", "  ", " **", "Spring", ":**", " The", " weather", " war", "ms", " up", " and", " **", "flowers", " bloom", "**.", "\n", "*", "  ", " **", "Summer", ":**", " The", " days", " are", " longest", " and", " **", "sun", "light", " is", " strongest", "**.", "\n", "*", "  ", " **", "Aut", "umn", " (", "Fall", "):**", " The", " leaves", " turn", " colors", " and", " **", "days", " get", " shorter", "**.", "\n", "*", "  ", " **", "Winter", ":**", " The", " days", " are", " shortest", " and", " **", "snow", " may", " fall", "**."]
MS_B2_4B_EN = [0.0, 103.1, 72.85, 72.81, 72.78, 72.96, 73.01, 72.86, 72.81, 72.98, 72.71, 72.87, 72.84, 73.12, 72.9, 72.88, 73.22, 73.03, 72.91, 72.94, 73.04, 73.04, 72.96, 73.1, 72.84, 72.89, 73.11, 73.02, 72.84, 72.94, 72.75, 72.85, 73.14, 73.18, 72.88, 72.76, 73.01, 72.94, 72.85, 72.91, 73.1, 73.22, 73.12, 73.1, 73.09, 73.2, 73.15, 72.84, 72.85, 72.83, 72.93, 73.02, 72.99, 73.0, 73.03, 73.03, 73.01, 72.94, 73.19, 73.05, 73.11, 73.15, 73.03, 72.84, 73.08, 73.33, 73.11, 72.87, 73.06, 73.09, 73.03, 73.02, 73.03, 72.95, 72.99, 73.26, 73.24, 73.09, 73.08, 73.34, 73.02, 73.08]

# bonsai2-4b-nano-zh-tw: Ternary-Bonsai-2-4B PTQ1_0, fwd35 greedy, per-token text pieces and
# step times (ms) verbatim from the run log (STEP lines; token 0 comes out of the prefill);
# 404 tokens, <|im_end|> excluded. Load 7369 ms, prefill 23 tok 968 ms.
PIECES_B2_4B_ZH_TW = ["\u56db\u5b63", "\u662f", "\u5927\u81ea\u7136", "\u8f2a", "\u8ff4", "\u7684", "\u8c61\u5fb5", "\uff0c", "\u6bcf\u4e00", "\u5b63", "\u4e0d\u50c5", "\u6c23\u5019", "\u8b8a\u5316", "\uff0c", "\u4e5f", "\u4f34", "\u96a8\u8457", "\u751f\u7269", "\u6d3b\u52d5", "\u8207", "\u4eba\u985e", "\u751f\u6d3b\u7684", "\u4e0d\u540c", "\u5074", "\u9762", "\u3002", "\u4ee5\u4e0b\u662f", "\u56db\u5b63", "\u53ca\u5176", "\u4ee3\u8868\u6027", "\u6539\u8b8a", "\u7684\u4e8b", "\u9805", "\uff1a", "\n\n", "###", " ", "1", ".", " ", "\u6625\u5b63", " (", "Spring", ")", "\n", "*", "  ", " **", "\u6838\u5fc3", "\u6539\u8b8a", "\uff1a", "\u5f9e", "\u51ac", "\u7720", "\u5230", "\u6d3b\u8e8d", "**", "\n", "   ", " *", "  ", " **", "\u81ea\u7136", "\u754c", "**", "\uff1a", "\u51b0\u96ea", "\u878d\u5316", "\uff0c", "\u690d\u7269", "\u958b\u59cb", "\u840c", "\u767c", "\u65b0", "\u8449", "\uff0c", "\u52d5\u7269", "\u5f9e", "\u51ac", "\u7720", "\u4e2d", "\u9192", "\u4f86", "\uff0c", "\u9ce5", "\u985e", "\u958b\u59cb", "\u5b75", "\u5375", "\u3002", "\n", "   ", " *", "  ", " **", "\u4eba\u985e", "\u751f\u6d3b", "**", "\uff1a", "\u5929\u6c23", "\u8f49", "\u6696", "\uff0c", "\u4eba\u5011", "\u958b\u59cb", "\u5916\u51fa", "\u6d3b\u52d5", "\u3001", "\u8e0f\u9752", "\uff0c", "\u4e26", "\u958b\u59cb", "\u6e96\u5099", "\u590f\u5b63", "\u7684", "\u8863\u7269", "\u8207", "\u6d3b\u52d5", "\u3002", "\n\n", "###", " ", "2", ".", " ", "\u590f\u5b63", " (", "Summer", ")", "\n", "*", "  ", " **", "\u6838\u5fc3", "\u6539\u8b8a", "\uff1a", "\u7e41", "\u76db", "\u8207", "\u7e41", "\u69ae", "**", "\n", "   ", " *", "  ", " **", "\u81ea\u7136", "\u754c", "**", "\uff1a", "\u690d\u7269", "\u751f\u9577", "\u6700", "\u65fa\u76db", "\uff0c", "\u82b1\u6735", "\u958b", "\u76db", "\uff0c", "\u679c", "\u5be6", "\u6210\u719f", "\uff1b", "\u52d5\u7269", "\u6d3b\u52d5", "\u6642\u9593", "\u8f03", "\u9577", "\uff0c", "\u6606", "\u87f2", "\u6d3b\u8e8d", "\u3002", "\n", "   ", " *", "  ", " **", "\u4eba\u985e", "\u751f\u6d3b", "**", "\uff1a", "\u5929\u6c23", "\u901a\u5e38", "\u8f03", "\u71b1", "\uff0c", "\u4eba\u5011", "\u6ce8\u91cd", "\u907f\u6691", "\u3001", "\u6e38\u6cf3", "\u6216", "\u6236\u5916", "\u904b\u52d5", "\uff1b", "\u4e5f\u662f", "\u8fb2", "\u4f5c\u7269", "\u6536", "\u6210\u7684", "\u5b63\u7bc0", "\u3002", "\n\n", "###", " ", "3", ".", " ", "\u79cb\u5b63", " (", "Aut", "umn", ")", "\n", "*", "  ", " **", "\u6838\u5fc3", "\u6539\u8b8a", "\uff1a", "\u6210\u719f", "\u8207", "\u8f49\u63db", "**", "\n", "   ", " *", "  ", " **", "\u81ea\u7136", "\u754c", "**", "\uff1a", "\u6a39", "\u8449", "\u8b8a", "\u8272", "\u4e26", "\u843d", "\u8449", "\uff0c", "\u6e96\u5099", "\u9032\u5165", "\u51ac", "\u7720", "\uff1b", "\u679c", "\u5be6", "\u6210\u719f", "\u4e26", "\u6210\u719f", "\uff1b", "\u5929\u6c23", "\u9010\u6f38", "\u8f49", "\u6dbc", "\u3002", "\n", "   ", " *", "  ", " **", "\u4eba\u985e", "\u751f\u6d3b", "**", "\uff1a", "\u9019\u662f", "\u6536\u83b7", "\u7bc0", "\u4efd", "\uff0c", "\u4eba\u5011", "\u6536\u85cf", "\u7ce7", "\u98df", "\u3001", "\u6e96\u5099", "\u51ac", "\u8863", "\uff1b", "\u4e5f\u662f", "\u65c5\u904a", "\u65fa\u5b63", "\uff0c", "\u8a31\u591a", "\u5730\u5340", "\u958b\u59cb", "\u8209\u8fa6", "\u7bc0", "\u6176", "\u6d3b\u52d5", "\u3002", "\n\n", "###", " ", "4", ".", " ", "\u51ac\u5b63", " (", "Winter", ")", "\n", "*", "  ", " **", "\u6838\u5fc3", "\u6539\u8b8a", "\uff1a", "", "\u86f0", "\u7720", "\u8207", "\u4fdd\u5b58", "**", "\n", "   ", " *", "  ", " **", "\u81ea\u7136", "\u754c", "**", "\uff1a", "\u690d\u7269", "\u812b", "\u8449", "\u4f11\u7720", "\uff0c", "\u52d5\u7269", "\u9032\u5165", "\u51ac", "\u7720", "\u6216", "\u5c0b\u627e", "\u98df\u7269", "\uff1b", "\u5929\u6c23", "\u5bd2\u51b7", "\uff0c", "\u96ea", "\u971c", "\u5e38\u898b", "\u3002", "\n", "   ", " *", "  ", " **", "\u4eba\u985e", "\u751f\u6d3b", "**", "\uff1a", "\u5929\u6c23", "\u5bd2\u51b7", "\uff0c", "\u4eba\u5011", "\u4f9d\u8cf4", "\u706b", "\u7210", "\u3001", "\u6696", "\u623f", "\u6216", "\u52a0\u71b1", "\u8a2d\u5099", "\uff1b", "\u4e5f\u662f", "\u50b3\u7d71", "\u7bc0", "\u6176", "\uff08", "\u5982", "\u8056\u8a95", "\u7bc0", "\u3001", "\u6625\u7bc0", "\uff09", "\u7684", "\u6642\u671f", "\uff0c", "\u5f37\u8abf", "\u5bb6\u5ead", "\u5718", "\u5713", "\u3002", "\n\n", "\u9019", "\u56db\u500b", "\u5b63\u7bc0", "\u4ea4\u66ff", "\u9032\u884c", "\uff0c", "\u69cb", "\u6210\u4e86", "\u5730\u7403", "\u751f\u614b", "\u7cfb\u7d71", "\u904b", "\u884c\u7684", "\u7bc0\u594f", "\u3002"]
MS_B2_4B_ZH_TW = [0.0, 104.56, 72.9, 72.95, 72.98, 72.95, 72.93, 72.87, 72.89, 72.95, 72.91, 73.12, 73.05, 72.81, 73.16, 73.39, 73.06, 72.8, 73.09, 73.06, 72.72, 72.99, 72.99, 73.08, 73.05, 73.05, 73.03, 72.88, 73.08, 73.29, 73.17, 73.2, 73.15, 72.79, 72.97, 73.05, 73.28, 72.93, 73.17, 73.35, 73.3, 72.97, 73.22, 73.2, 73.39, 73.02, 73.16, 72.99, 73.1, 73.07, 73.3, 73.1, 73.4, 74.7, 73.18, 73.18, 73.05, 72.97, 73.09, 73.19, 72.87, 72.97, 73.12, 73.11, 73.27, 72.98, 73.14, 73.06, 73.06, 73.62, 73.11, 73.11, 73.14, 72.99, 73.19, 73.26, 73.14, 72.98, 73.38, 73.19, 73.11, 73.1, 73.38, 73.38, 72.99, 73.44, 73.13, 73.16, 73.87, 73.61, 73.24, 73.33, 73.44, 73.24, 73.47, 73.5, 73.43, 73.14, 73.63, 73.53, 73.13, 73.07, 73.36, 73.21, 73.17, 73.37, 73.15, 73.27, 73.16, 73.31, 73.17, 73.18, 73.25, 73.11, 73.19, 73.23, 73.14, 73.11, 73.24, 73.46, 73.18, 72.98, 73.3, 73.31, 73.4, 73.36, 73.42, 73.59, 73.37, 73.53, 73.13, 73.09, 73.4, 73.3, 73.46, 73.34, 73.32, 73.48, 73.37, 73.51, 73.47, 73.39, 73.39, 73.51, 73.33, 73.72, 73.39, 73.97, 73.17, 73.41, 73.35, 73.4, 73.47, 73.15, 73.64, 73.52, 73.43, 73.39, 73.24, 73.46, 73.58, 73.44, 73.55, 73.36, 73.47, 73.52, 73.33, 73.35, 73.41, 73.5, 73.47, 73.33, 73.54, 73.39, 73.56, 73.71, 73.63, 73.45, 73.35, 73.67, 73.77, 73.5, 73.54, 73.79, 73.94, 73.39, 73.5, 73.39, 73.53, 73.44, 73.37, 73.69, 73.8, 73.54, 73.74, 73.5, 73.58, 73.73, 73.73, 73.72, 73.65, 73.66, 73.79, 74.1, 73.83, 73.82, 73.73, 73.85, 73.88, 73.94, 73.72, 73.63, 73.63, 73.71, 73.64, 73.79, 74.07, 73.62, 73.75, 73.91, 73.69, 73.89, 73.8, 73.91, 73.9, 73.85, 74.08, 73.73, 73.88, 73.88, 73.6, 73.82, 73.77, 73.9, 73.78, 73.7, 73.98, 73.85, 73.94, 73.91, 74.02, 73.82, 73.91, 73.79, 74.33, 74.25, 75.07, 75.26, 76.48, 75.05, 77.17, 75.14, 75.59, 74.7, 75.86, 74.37, 74.18, 74.38, 73.93, 74.82, 74.06, 74.11, 73.91, 74.07, 74.08, 73.89, 73.89, 73.92, 74.09, 74.36, 74.06, 74.05, 73.88, 73.97, 74.09, 73.9, 74.04, 73.96, 74.07, 73.95, 74.11, 74.19, 73.92, 74.13, 74.2, 74.07, 74.06, 74.05, 74.13, 74.02, 74.22, 74.23, 73.92, 74.38, 73.95, 74.27, 74.06, 74.4, 74.13, 74.1, 74.07, 73.9, 74.1, 74.02, 74.24, 74.26, 74.16, 74.4, 74.11, 74.37, 74.34, 74.26, 74.22, 74.32, 74.13, 74.28, 74.4, 73.98, 74.19, 74.11, 74.15, 74.01, 74.37, 74.22, 74.06, 74.37, 74.29, 74.16, 74.36, 74.26, 74.12, 74.44, 74.36, 74.1, 74.36, 74.96, 74.39, 74.13, 74.29, 74.32, 74.23, 74.31, 74.18, 74.31, 74.28, 74.28, 74.23, 74.31, 74.24, 74.12, 74.48, 74.22, 74.25, 74.34, 74.31, 74.36, 74.46, 74.3, 74.27, 74.25, 74.64, 74.19, 74.24, 74.41, 74.45, 74.4, 74.17, 74.62, 74.36, 74.2, 74.6, 74.69, 74.49, 74.21, 74.37, 74.37, 74.15, 74.44, 74.53, 74.36, 74.57, 74.45, 74.2, 74.33, 74.75, 74.44, 74.56, 74.77, 74.5, 74.21, 74.51, 74.37, 74.38, 74.68, 74.67, 74.49, 74.64, 74.39, 74.31, 74.5, 74.51, 74.57, 74.56, 74.36]


# bonsai2-9b-nano-en: Ternary-Bonsai-2-9B PTQ1_0, fwd35 greedy, per-token text pieces and
# step times (ms) verbatim from the run log (STEP lines; token 0 comes out of the prefill);
# 99 tokens, <|im_end|> excluded. Load 12669 ms, prefill 24 tok 1774 ms.
PIECES_B2_9B_EN = ["Here", " are", " the", " four", " seasons", " along", " with", " one", " distinct", " thing", " that", " changes", " in", " each", ":", "\n\n", "*", "  ", " **", "Spring", ":**", " The", " days", " become", " longer", ",", " and", " flowers", " begin", " to", " bloom", ".", "\n", "*", "  ", " **", "Summer", ":**", " The", " sun", " stays", " high", " in", " the", " sky", ",", " resulting", " in", " the", " warm", "est", " temperatures", " of", " the", " year", ".", "\n", "*", "  ", " **", "Aut", "umn", " (", "Fall", "):**", " Leaves", " change", " color", " from", " green", " to", " red", ",", " orange", ",", " and", " yellow", " before", " falling", " off", " the", " trees", ".", "\n", "*", "  ", " **", "Winter", ":**", " Snow", " begins", " to", " fall", ",", " and", " temperatures", " drop", " significantly", "."]
MS_B2_9B_EN = [0.0, 170.32, 126.2, 126.02, 131.54, 135.17, 126.3, 126.23, 125.56, 131.52, 125.71, 126.09, 125.98, 126.28, 125.95, 125.99, 126.02, 125.86, 126.15, 125.99, 135.19, 135.13, 125.9, 126.18, 125.96, 134.65, 125.87, 126.25, 134.84, 126.35, 126.4, 134.97, 125.62, 126.16, 125.99, 125.95, 125.5, 125.98, 126.15, 126.01, 131.82, 131.32, 129.94, 125.92, 126.07, 126.19, 126.0, 126.13, 125.82, 126.16, 125.95, 126.31, 125.95, 126.1, 125.88, 126.15, 125.93, 126.02, 126.03, 125.95, 125.97, 125.98, 132.12, 126.06, 132.19, 137.86, 135.39, 125.89, 125.82, 126.08, 126.62, 126.48, 125.77, 125.93, 126.01, 126.09, 126.28, 130.23, 126.07, 126.31, 126.2, 125.94, 126.75, 126.26, 126.08, 126.49, 126.36, 126.51, 125.94, 126.01, 126.06, 125.71, 126.17, 126.24, 126.45, 126.17, 126.0, 125.94, 126.08]

# bonsai2-9b-nano-zh-tw: Ternary-Bonsai-2-9B PTQ1_0, fwd35 greedy, per-token text pieces and
# step times (ms) verbatim from the run log (STEP lines; token 0 comes out of the prefill);
# 92 tokens, <|im_end|> excluded. Load 13080 ms, prefill 23 tok 1813 ms.
PIECES_B2_9B_ZH_TW = ["\u4ee5\u4e0b\u662f", "\u56db\u5b63", "\u7684", "\u5217\u8868", "\uff0c", "\u4ee5\u53ca", "\u6bcf\u4e00", "\u5b63", "\u6703", "\u6539\u8b8a", "\u7684\u4e00\u4ef6", "\u5178\u578b", "\u4e8b\u4ef6", "\uff1a", "\n\n", "*", "  ", " **", "\u6625\u5b63", "**", "\uff1a", "\u842c", "\u7269", "\u5fa9", "", "\u7526", "\uff0c", "**", "\u82b1\u6735", "\u958b\u59cb", "\u7dbb", "\u653e", "**\u3002", "\n", "*", "  ", " **", "\u590f\u5b63", "**", "\uff1a", "\u6c23", "\u6eab", "\u5347\u9ad8", "\uff0c", "**", "\u591c\u665a", "\u7684", "\u868a", "\u87f2", "\u958b\u59cb", "\u6d3b\u8e8d", "**\u3002", "\n", "*", "  ", " **", "\u79cb\u5b63", "**", "\uff1a", "\u6c23", "\u6eab", "\u8f49", "\u6dbc", "\uff0c", "**", "\u6a39", "\u8449", "\u958b\u59cb", "\u8b8a", "\u8272", "\u4e26", "\u98c4", "\u843d", "**\u3002", "\n", "*", "  ", " **", "\u51ac\u5b63", "**", "\uff1a", "\u6c23", "\u6eab", "\u5bd2\u51b7", "\uff0c", "**", "\u5730\u9762", "\u7d50", "\u51b0", "\u6216", "\u4e0b\u96ea", "**\u3002"]
MS_B2_9B_ZH_TW = [0.0, 159.36, 126.42, 126.48, 135.45, 127.1, 126.69, 126.62, 126.49, 126.47, 126.46, 135.59, 126.82, 126.83, 132.29, 126.67, 126.95, 126.96, 127.05, 126.11, 133.37, 126.54, 125.84, 126.64, 126.45, 126.57, 126.6, 126.4, 126.58, 134.65, 126.5, 134.31, 126.64, 135.74, 126.36, 126.54, 126.93, 126.49, 133.38, 126.44, 126.82, 126.81, 126.68, 126.94, 126.45, 126.67, 126.54, 126.45, 126.34, 126.66, 126.81, 135.7, 126.6, 126.62, 126.57, 126.32, 127.25, 127.05, 126.81, 126.79, 126.61, 126.69, 126.74, 126.88, 127.07, 126.58, 131.84, 126.71, 126.97, 127.08, 126.93, 128.74, 131.36, 130.97, 136.3, 126.95, 126.76, 126.54, 126.69, 132.49, 126.84, 126.69, 126.76, 126.88, 126.83, 126.87, 126.93, 126.94, 126.96, 126.92, 126.59, 133.3]


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
        load_ms=3368.0, prefill_ms=413.0, n_prompt=24,
        status="decode 28.5 tok/s \u00b7 prefill 58.2 tok/s \u00b7 greedy \u00b7 Jetson Nano GPU (sm_53)",
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
        load_ms=3322.0, prefill_ms=408.0, n_prompt=23,
        status="decode 28.6 tok/s \u00b7 prefill 56.4 tok/s \u00b7 greedy \u00b7 Jetson Nano GPU (sm_53)",
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
        load_ms=9470.0, prefill_ms=971.0, n_prompt=24,
        status="decode 13.6 tok/s \u00b7 prefill 24.7 tok/s \u00b7 greedy \u00b7 Jetson Nano GPU (sm_53)",
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
        load_ms=7369.0, prefill_ms=968.0, n_prompt=23,
        status="decode 13.5 tok/s \u00b7 prefill 23.8 tok/s \u00b7 greedy \u00b7 Jetson Nano GPU (sm_53)",
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
        load_ms=12669.0, prefill_ms=1774.0, n_prompt=24,
        status="decode 7.8 tok/s \u00b7 prefill 13.5 tok/s \u00b7 greedy \u00b7 Jetson Nano GPU (sm_53)",
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
        load_ms=13080.0, prefill_ms=1813.0, n_prompt=23,
        status="decode 7.8 tok/s \u00b7 prefill 12.7 tok/s \u00b7 greedy \u00b7 Jetson Nano GPU (sm_53)",
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
