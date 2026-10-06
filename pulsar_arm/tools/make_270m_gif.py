"""Render the English 270m chat demo GIF (compat entry point).

The transcript lives in make_chat_gif.SPECS["en"]; render all demos with
`python3 tools/make_chat_gif.py`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_chat_gif import DOC, SPECS, render  # noqa: E402

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else DOC
    render("en", SPECS["en"], out)
