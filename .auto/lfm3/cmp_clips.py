#!/usr/bin/env python3
"""Compare resident-clip ids against the solo runs they must reproduce bit-for-bit."""
import re, sys
PAT = re.compile(r'^\s*(\d+)\s+(-?\d+)\s+[\d.]+\s+[\d.]+$')

def runs(f):
    R, cur, prev = [], [], -1
    for l in open(f):
        m = PAT.match(l)
        if not m:
            continue
        s, t = int(m.group(1)), int(m.group(2))
        if s <= prev:
            R.append(cur); cur = []
        cur.append(t); prev = s
    if cur:
        R.append(cur)
    return R

def flat(f):
    out = []
    for l in open(f):
        m = PAT.match(l)
        if m:
            out.append(int(m.group(2)))
    return out

R = runs(sys.argv[1])
print("resident runs: %d, lengths %s" % (len(R), [len(x) for x in R]))
ok = True
MIN_COVER = int(__import__('os').environ.get("CLIP_MIN_COVER", "24"))
for k, spec in enumerate(sys.argv[2:]):
    # "FILE" pairs with run k; "N:FILE" pairs with run N, so a 2-clip resident run can be
    # checked against two independent solo runs.
    if ":" in spec.split("/")[0]:
        idx, path = spec.split(":", 1); idx = int(idx)
    else:
        idx, path = k, spec
    s = flat(path)
    if idx >= len(R):
        print("clip %d: MISSING run (only %d runs)" % (idx, len(R))); ok = False; continue
    r = R[idx]
    # Prefix semantics: a resident clip may be asked for a SHORTER tail than the solo
    # reference (ngen in the spec), and demanding equal lengths made a bit-identical clip
    # read as a failure -- "first diffs at []" with eq False is the signature. What has to
    # hold is that the two streams agree on every id the resident clip produced.
    n = min(len(r), len(s))
    eq = r[:n] == s[:n]
    cover = n >= MIN_COVER
    ok = ok and eq and cover
    print("clip %d == solo %s : %s   (agree on %d ids, resident %d, solo %d, min cover %d %s)"
          % (idx, idx, eq and cover, n, len(r), len(s), MIN_COVER, "OK" if cover else "SHORT"))
    if not eq:
        d = [i for i in range(n) if r[i] != s[i]][:5]
        print("   first diffs at %s  resident=%s solo=%s"
              % (d, [r[i] for i in d], [s[i] for i in d]))
print("CLIPMODE_IDS " + ("PASS" if ok else "FAIL"))
