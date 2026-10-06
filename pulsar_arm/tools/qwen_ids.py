"""Text -> Qwen3 token ids for the engine's raw-ids mode.

The asm encoder is Gemma-specific (SentencePiece-style BPE with U+2581), while
Bonsai/Qwen3 uses a regex-pretokenised byte-level BPE. Rather than reimplement
that in asm, the demo pipeline tokenises here (build/serve side) and feeds the
engine ids directly:

    python3 qwen_ids.py "The capital of France is" | ./core bonsai_b3.bin ... 2 i

Prints the ids space-separated on one line, plus the decoded tokens on stderr
so the mapping is visible. --chat wraps the prompt in Qwen3's chat template.

Usage: qwen_ids.py <text> [--chat] [--repo <hf-name-or-path>]
"""
import sys

from transformers import AutoTokenizer

DEFAULT_REPO = "prism-ml/Ternary-Bonsai-1.7B-unpacked"


def main():
    args = sys.argv[1:]
    chat = "--chat" in args
    if chat:
        args.remove("--chat")
    repo = DEFAULT_REPO
    if "--repo" in args:
        i = args.index("--repo")
        repo = args[i + 1]
        del args[i:i + 2]
    text = " ".join(args)
    tk = AutoTokenizer.from_pretrained(repo)
    if chat:
        text = tk.apply_chat_template([{"role": "user", "content": text}],
                                      tokenize=False, add_generation_prompt=True)
    ids = tk.encode(text)
    print(" ".join(str(i) for i in ids))
    print(f"# {len(ids)} ids: " + "|".join(repr(tk.decode([i])) for i in ids[:12]),
          file=sys.stderr)


if __name__ == "__main__":
    main()
