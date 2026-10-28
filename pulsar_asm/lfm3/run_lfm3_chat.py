#!/usr/bin/env python3
"""LFM3 text backbone chat CLI (WIP)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runtime.lfm3_model import LFM3

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.bin")
DEFAULT_TOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tok.bin")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="?", default=DEFAULT_WEIGHTS)
    ap.add_argument("vocab", nargs="?", default=DEFAULT_TOK)
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    print("LFM3 text backbone (WIP)")
    print(f"Weights: {args.weights}")
    print(f"Vocab: {args.vocab}")
    m = LFM3(args.weights, max_seq=args.max_seq, n_threads=args.threads, verbose=True)
    print("Engine not fully implemented yet.")
    m.close()


if __name__ == "__main__":
    main()
