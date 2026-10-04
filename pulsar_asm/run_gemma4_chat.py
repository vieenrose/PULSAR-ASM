#!/usr/bin/env python3
"""Streaming terminal chat for PULSAR-Gemma4 (Linux, CPU, pure-assembly engine).

Everything that costs anything runs in the assembled engine; the Python here is
tokenizer, chat template, and REPL only.

  python run_gemma4_chat.py --weights /mnt/edge/pulsar/gemma4_e2b.bin
  python run_gemma4_chat.py --demo            # scripted prompt, for the README gif

Decoding is greedy (argmax). The checkpoint's generation_config asks for
sampling (temp 1.0, top_k 64, top_p 0.95); greedy keeps the parity tests exact
and the tok/s comparable across modes. A sampler is a separate change.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runtime.gemma4_model import Gemma4  # noqa: E402

DEFAULT_TOK = "/mnt/edge/pulsar/tok"
# Gemma 4 replaced Gemma 3's <start_of_turn>/<end_of_turn> with <|turn>role ...
# <turn|>, and added a thinking channel <|channel>thought ... <channel|>.
TURN_OPEN, TURN_CLOSE = "<|turn>", "<turn|>"


def load_tokenizer(path):
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(path)
    eos = {1, 106, 50}                      # <eos>, <turn|>, and the config's third
    gc = os.path.join(path, "generation_config.json")
    if os.path.exists(gc):
        import json
        v = json.load(open(gc)).get("eos_token_id", [])
        eos = set(v if isinstance(v, list) else [v])
    return tk, eos


def render(messages, add_generation_prompt=True):
    """Plain-text subset of chat_template.jinja, for when jinja is unavailable.

    Matches the canonical template for messages with no tools and no reasoning
    content, which is all this CLI ever produces.
    """
    s = "<bos>"
    for m in messages:
        role = "model" if m["role"] == "assistant" else m["role"]
        s += f"{TURN_OPEN}{role}\n{m['content'].strip()}{TURN_CLOSE}\n"
    if add_generation_prompt:
        s += f"{TURN_OPEN}model\n"
    return s


def prompt_ids(tk, messages):
    try:
        return list(tk.apply_chat_template(messages, add_generation_prompt=True,
                                           tokenize=True, return_dict=False))
    except TypeError:                                  # older/newer signature
        out = tk.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        return list(out["input_ids"] if hasattr(out, "keys") else out)
    except Exception:
        return list(tk(render(messages), add_special_tokens=False)["input_ids"])


class Chat:
    def __init__(self, eng, tk, eos, max_new=512, status_every=8):
        self.eng, self.tk, self.eos = eng, tk, eos
        self.max_new, self.status_every = max_new, status_every
        self.reset_state()
        self.n_tok, self.clock = 0, 0.0

    def reset_state(self):
        self.messages, self.ids = [], []
        self.eng.reset()

    def prefill(self, text):
        """Append a user turn and return the first sampled token.

        The KV cache already holds every previously fed token, and this
        template serializes older turns identically, so only the new tail is
        pushed through the engine - that is the whole point of keeping the
        cache warm between turns.
        """
        self.messages.append({"role": "user", "content": text})
        new = prompt_ids(self.tk, self.messages)
        if new[:len(self.ids)] != self.ids:
            raise RuntimeError("re-rendered prompt is not an extension of the cached one")
        for t in new[len(self.ids):]:
            self.eng.forward(t)
        self.ids = new
        self.eng.pos = len(new)             # forward() already advanced it; belt and braces
        return int(self.eng.logits().argmax())

    def turn(self, text, quiet=False):
        """Stream one reply. Returns the generated ids.

        Layout: the reply streams on its own line and the tok/s counter lives on
        the line below, so a counter update can never splice into the text. The
        cursor always ends a write sitting on the (cleared) status line.
        """
        nxt = self.prefill(text)
        buf, shown = [], ""
        t0 = time.perf_counter()
        n = 0
        status = lambda t: sys.stdout.write("\r\x1b[2K\033[90m" + t + "\033[0m")
        emit = lambda d: sys.stdout.write("\r\x1b[2K\033[1A" + d + "\n")
        if not quiet:
            sys.stdout.write("\n")
            status("…")
        while n < self.max_new and len(self.ids) < self.eng.max_seq - 1:
            if nxt in self.eos:
                break
            buf.append(nxt)
            full = self.tk.decode(buf)
            if not quiet and full != shown:
                emit(full[len(shown):])
                status(f"… {n} tok  {n/max(time.perf_counter()-t0,1e-9):5.1f} tok/s")
            shown = full
            self.ids.append(nxt)
            nxt = int(self.eng.forward(nxt))
            n += 1
        dt = time.perf_counter() - t0
        self.n_tok += n
        self.clock += dt
        if not quiet:
            sys.stdout.write("\r\x1b[2K")
            sys.stdout.write(shown + "\n")
            status(f"{n} tok  {n/max(dt,1e-9):5.1f} tok/s  ({self.tps:5.1f} avg)")
            sys.stdout.write("\n")
        self.messages.append({"role": "assistant", "content": shown.strip()})
        return buf

    @property
    def tps(self):
        return self.n_tok / self.clock if self.clock else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/mnt/edge/pulsar/gemma4_e2b.bin")
    ap.add_argument("--tok", default=DEFAULT_TOK)
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--demo", action="store_true", help="one scripted reply, for the README gif")
    a = ap.parse_args()

    tk, eos = load_tokenizer(a.tok)
    print("loading engine ...", flush=True)
    eng = Gemma4(a.weights, max_seq=a.max_seq, n_threads=a.threads, verbose=True)
    chat = Chat(eng, tk, eos, max_new=a.max_new)

    if a.demo:
        q = ("In one short sentence: what is unusual about running a 2-billion "
             "parameter language model in pure x86 assembly?")
        print(f"\n\033[1myou\033[0m> {q}\n")
        chat.turn(q)
        print(f"\n\033[90m{chat.tps:.1f} tok/s  ·  greedy  ·  {a.threads} cores\033[0m")
        eng.close()
        return

    print("\nPULSAR-Gemma4 · E2B-it text-only · greedy · /reset clears the cache · /exit\n")
    while True:
        try:
            q = input("\033[1myou\033[0m> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in ("/exit", "/quit"):
            break
        if q == "/reset":
            chat.reset_state()
            print("\033[90m(cache cleared)\033[0m")
            continue
        print()
        chat.turn(q)
        print()
    eng.close()


if __name__ == "__main__":
    main()
