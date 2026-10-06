"""Render the Chinese 270m chat demo GIF (zh-TW), compat entry point.

The transcript lives in make_chat_gif.SPECS["zh-tw"]; render everything with
`python3 tools/make_chat_gif.py`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_chat_gif import DOC, SPECS, render  # noqa: E402

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else DOC
    render("zh-tw", SPECS["zh-tw"], out)
