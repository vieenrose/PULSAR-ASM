#!/usr/bin/env python3
"""PULSAR-ARM | tools/check_adrp.py

Catch a bug class that no runtime gate can see: `adrp xN, SYM` leaves xN at the
4 KB page base of SYM, and the low half only arrives with
`add xN, xN, :lo12:SYM` (or folded into the load, `ldr s0, [xN, :lo12:SYM]`).
When the add is missing, the access lands on whatever shares that page - which
for a cell surrounded by zeros reads back as 0, so a branch on it silently takes
the default path forever. Three of the arch dispatchers had exactly this: they
read G_ARCH from the page base, got 0, and ran the gemma3 path for a qwen3 blob.
Gemma could never reveal it because 0 is the right answer for gemma3.

Usage: check_adrp.py [core.S ...]     (default: ../asm/core.S)
Exits 1 if any site uses the register before the low half is added.
"""
import re
import sys
import os

ADR = re.compile(r'\s+adrp (x\d+), ([A-Za-z_][A-Za-z_0-9]*)\s*$')


def scan(path):
    lines = open(path).read().split('\n')
    bad = []
    for i, raw in enumerate(lines):
        line = raw.split('//')[0].rstrip()
        m = ADR.match(line)
        if not m:
            continue
        reg, sym = m.group(1), m.group(2)
        done = None
        for j in range(i + 1, min(len(lines), i + 8)):
            nxt = lines[j].split('//')[0].strip()
            if not nxt:
                continue
            if nxt == 'add %s, %s, :lo12:%s' % (reg, reg, sym):
                done = True
                break
            if ':lo12:' + sym in nxt:       # folded into the load/store
                done = True
                break
            if re.search(r'\b' + reg + r'\b', nxt):
                done = False               # used before the low half exists
                break
        if done is False:
            bad.append((i + 1, line.strip(), lines[i + 1].split('//')[0].strip()))
    return bad


def main():
    args = sys.argv[1:] or [os.path.join(os.path.dirname(__file__), '..', 'asm', 'core.S')]
    rc = 0
    for path in args:
        bad = scan(path)
        for ln, stmt, nxt in bad:
            rc = 1
            print('%s:%d: %s' % (path, ln, stmt))
            print('%s:%d:   used here: %s' % (path, ln + 1, nxt))
        print('%s: %d site(s) missing :lo12' % (path, len(bad)))
    return rc


if __name__ == '__main__':
    sys.exit(main())
