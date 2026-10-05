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
import json
import re
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runtime.gemma4_model import Gemma4  # noqa: E402

DEFAULT_TOK = "/mnt/edge/pulsar/tok"
# Gemma 4 replaced Gemma 3's <start_of_turn>/<end_of_turn> with <|turn>role ...
# <turn|>, and added a thinking channel <|channel>thought ... <channel|>.
TURN_OPEN, TURN_CLOSE = "<|turn>", "<turn|>"


# One scripted reply, used by --demo and by tools/make_demo_gif.py, so the gif
# and the CLI always show the same run.
DEMO_PROMPT = ("In one short sentence: why is a CPU running a language model "
               "limited by memory rather than by its cores?")


def gen_config(path):
    """The checkpoint's own sampling defaults, for once and for all.

    Kept separate from load_tokenizer so the model class and the tests never
    inherit a sampling policy they did not ask for - parity wants argmax.
    """
    gc = os.path.join(path, "generation_config.json")
    try:
        return json.load(open(gc))
    except OSError:
        return {}


def sampler_from_config(eng, path, args):
    """Configure the head; returns a label for the footer."""
    gc = gen_config(path)
    if getattr(args, "greedy", False) or not gc.get("do_sample", False):
        eng.set_sampling(temp=0)
        return "greedy"
    temp = args.temp if args.temp is not None else gc.get("temperature", 1.0)
    topk = args.top_k if args.top_k is not None else gc.get("top_k", 64)
    topp = args.top_p if args.top_p is not None else gc.get("top_p", 0.95)
    eng.set_sampling(temp=temp, top_k=topk, top_p=topp, seed=args.seed)
    return f"temp {temp} \u00b7 top_k {topk} \u00b7 top_p {topp}"


_BYTE_TOK = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def decode_text(tk, ids):
    """Decode generated ids the way Gemma's SentencePiece model actually means.

    gemma-4's tokenizer.json sets byte_fallback: the vocabulary carries 256
    <0xNN> tokens and any character that fell apart gets re-encoded as a byte
    run. HF's GemmaTokenizer.decode (the class this tokenizer.json declares)
    drops those tokens on the floor, so a reply that contains one silently loses
    its content: "2" disappears from "**2**-billion", and CJK, which leans on
    byte fallback hardest, turns to noise. Walking the pieces directly and
    turning <0xNN> back into bytes is what spiece_decode would have done.
    """
    specials = tk.all_special_tokens
    out = bytearray()
    for tok in tk.convert_ids_to_tokens(ids):
        if tok is None or tok in specials:
            continue
        m = _BYTE_TOK.fullmatch(tok)
        if m:
            out.append(int(m.group(1), 16))
        else:
            out.extend(tok.replace("\u2581", " ").encode("utf-8", "surrogatepass"))
    return out.decode("utf-8", "replace")


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
            # The fast path assumes decode -> strip -> re-encode round-trips,
            # which it does not (a trailing space becomes part of a token,
            # template markers shift). Fall back to replaying the whole
            # render on a fresh engine: slower on these turns, always right.
            self.eng.reset()
            self.ids = []
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
    ap.add_argument("--greedy", action="store_true", help="argmax instead of the checkpoint's sampling")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    a = ap.parse_args()

    tk, eos = load_tokenizer(a.tok)
    print("loading engine ...", flush=True)
    eng = Gemma4(a.weights, max_seq=a.max_seq, n_threads=a.threads, verbose=True)
    mode = sampler_from_config(eng, a.tok, a)
    chat = Chat(eng, tk, eos, max_new=a.max_new)

    if a.demo:
        print(f"\n\033[1myou\033[0m> {DEMO_PROMPT}\n")
        chat.turn(DEMO_PROMPT)
        print(f"\n\033[90m{chat.tps:.1f} tok/s  ·  {mode}  ·  {a.threads} cores\033[0m")
        eng.close()
        return

    print(f"\nPULSAR-Gemma4 · E2B-it text-only · {mode} · /reset clears the cache · /exit\n")
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
