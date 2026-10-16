#!/usr/bin/env python3
"""Minimal read-only web view of autoresearch loops (stdlib only).
Serves every autoresearch/loop-*/results.tsv as a table + score chart, the tail of the
loop's live.txt (the running measurement, if any) and the recent git log. Auto-refreshes.
usage: python3 autoresearch/webui.py [--host 100.92.209.8] [--port 8765]"""
import argparse, html, os, subprocess, glob, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
CSS = """:root{--bg:#fafafa;--fg:#1d1d1f;--mu:#6e6e73;--ln:#e3e3e6;--ac:#0a7d42;--bad:#b3261e;--card:#fff;--c1:#1f6fd1;--c2:#d1661f}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#e8e8ea;--mu:#9a9aa0;--ln:#2c2c30;--ac:#4cc38a;--bad:#ff7b72;--card:#1c1c1f;--c1:#6aa8ff;--c2:#ffa25c}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 ui-sans-serif,system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}h1{font-size:18px;margin:4px 0 12px}
.card{background:var(--card);border:1px solid var(--ln);border-radius:8px;padding:12px;margin:12px 0;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:5px 8px;border-bottom:1px solid var(--ln);text-align:left;vertical-align:top}
th{color:var(--mu);font-weight:600}.num{font-variant-numeric:tabular-nums;white-space:nowrap}
.keep,.baseline{color:var(--ac);font-weight:600}.discard,.crash,.metric-error{color:var(--bad)}
pre{margin:0;font:12px/1.4 ui-monospace,monospace;white-space:pre-wrap}a{color:inherit}
.kpi{display:flex;gap:24px;flex-wrap:wrap}.kpi div b{display:block;font-size:20px}.mu{color:var(--mu)}
select{font:inherit;background:var(--card);color:var(--fg);border:1px solid var(--ln);border-radius:6px;padding:2px 6px}"""

def loops():
    return sorted((os.path.basename(d) for d in glob.glob(os.path.join(ROOT, "loop-*")) if os.path.isdir(d)), reverse=True)

def read_tsv(p):
    rows, hdr = [], None
    try:
        for ln in open(p, encoding="utf-8"):
            ln = ln.rstrip("\n")
            if not ln or ln.startswith("#"): continue
            f = ln.split("\t")
            if hdr is None: hdr = f
            else: rows.append(dict(zip(hdr, f)))
    except OSError: pass
    return hdr or [], rows

def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return None

SPEEDS = (("prefill", "prefill tok/s", "var(--c1)"), ("decode", "generation tok/s", "var(--c2)"))

def speeds(r):
    """{'prefill': x, 'decode': y} parsed from guard-metric 'prefill=.. decode=.. rss=..'"""
    out = {}
    for kv in (r.get("guard-metric") or "").split():
        k, _, v = kv.partition("=")
        if k in ("prefill", "decode") and fnum(v) is not None: out[k] = fnum(v)
    return out

def speed_chart(rows):
    """prefill and generation tok/s per iteration, one curve each (through kept rows), shared axis from 0"""
    sp = [speeds(r) for r in rows]
    vs = [v for d in sp for v in d.values()]
    if not vs: return ""
    W, H, P, L = 1000, 260, 30, 46
    hi = max(vs) * 1.1
    n = max(len(rows) - 1, 1)
    X = lambda i: L + (W - L - P) * i / n
    Y = lambda v: H - P - (H - 2 * P) * v / hi
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="prefill and generation speed per iteration">']
    for t in range(5):
        v = hi * t / 4
        out.append(f'<line x1="{L}" x2="{W-P}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var(--ln)"/>'
                   f'<text x="{L-6}" y="{Y(v)+4:.1f}" text-anchor="end" fill="currentColor" font-size="12" opacity=".6">{v:.3g}</text>')
    for k, lab, c in SPEEDS:
        line = [f"{X(i):.1f},{Y(d[k]):.1f}" for i, (d, r) in enumerate(zip(sp, rows)) if k in d and r.get("status") in ("keep", "baseline")]
        if line: out.append(f'<polyline points="{" ".join(line)}" fill="none" stroke="{c}" stroke-width="2.5"/>')
        last = None
        for i, (d, r) in enumerate(zip(sp, rows)):
            if k not in d: continue
            kept = r.get("status") in ("keep", "baseline")
            out.append(f'<circle cx="{X(i):.1f}" cy="{Y(d[k]):.1f}" r="4" fill="{c if kept else "none"}" stroke="{c}" stroke-width="1.5">'
                       f'<title>#{r.get("iteration")} {lab} {d[k]:g} ({html.escape(r.get("status",""))})</title></circle>')
            if kept: last = (i, d[k])
        if last: out.append(f'<text x="{X(last[0])-6:.1f}" y="{Y(last[1])-10:.1f}" text-anchor="end" fill="{c}" font-size="13" font-weight="600">{last[1]:g}</text>')
    lg = "".join(f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:18px"><svg width="22" height="10"><line x1="0" x2="22" y1="5" y2="5" stroke="{c}" stroke-width="3"/></svg>{lab}</span>' for _, lab, c in SPEEDS)
    out.append("</svg>")
    return f'<div class="mu" style="margin-bottom:6px">speed per iteration (filled = kept, hollow = discarded) &nbsp; {lg}</div>' + "".join(out)

def chart(rows):
    pts = [(i, fnum(r.get("metric")), r.get("status", "")) for i, r in enumerate(rows) if fnum(r.get("metric")) is not None]
    if len(pts) < 1: return ""
    W, H, P = 1000, 220, 30
    vs = [v for _, v, _ in pts]; lo, hi = min(vs), max(vs)
    if hi == lo: hi = lo + 1
    n = max(len(rows) - 1, 1)
    X = lambda i: P + (W - 2 * P) * i / n
    Y = lambda v: H - P - (H - 2 * P) * (v - lo) / (hi - lo)
    best, path = [], []
    b = None
    for i, v, s in pts:
        if s in ("keep", "baseline"): b = v if b is None else max(b, v)
        if b is not None: best.append(f"{X(i):.1f},{Y(b):.1f}")
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="metric per iteration">',
           f'<text x="{P}" y="16" fill="currentColor" font-size="12" opacity=".6">{hi:.4g}</text>',
           f'<text x="{P}" y="{H-8}" fill="currentColor" font-size="12" opacity=".6">{lo:.4g}</text>']
    if best: out.append(f'<polyline points="{" ".join(best)}" fill="none" stroke="var(--ac)" stroke-width="2"/>')
    for i, v, s in pts:
        c = "var(--ac)" if s in ("keep", "baseline") else "var(--bad)"
        out.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="4" fill="{c}"><title>#{rows[i].get("iteration")} {v} {html.escape(s)}</title></circle>')
    out.append("</svg>")
    return "".join(out)

def fmt_sp(k, rows):
    v = [speeds(r)[k] for r in rows if r.get("status") in ("keep", "baseline") and k in speeds(r)]
    return f"{v[-1]:g} <span class=\"mu\" style=\"font-size:13px\">from {v[0]:g}</span>" if v else "-"

def page(sel):
    ls = loops()
    if sel not in ls: sel = ls[0] if ls else ""
    d = os.path.join(ROOT, sel)
    hdr, rows = read_tsv(os.path.join(d, "results.tsv"))
    ms = [(fnum(r.get("metric")), r) for r in rows if r.get("status") in ("keep", "baseline") and fnum(r.get("metric")) is not None]
    base = ms[0][0] if ms else None
    best = max(ms, key=lambda t: t[0]) if ms else None
    kept = sum(r.get("status") == "keep" for r in rows); disc = sum(r.get("status") not in ("keep", "baseline") for r in rows)
    live = ""
    lp = os.path.join(d, "live.txt")
    if os.path.exists(lp):
        try: live = "".join(open(lp, encoding="utf-8", errors="replace").readlines()[-25:])
        except OSError: pass
    try: glog = subprocess.run(["git", "-C", REPO, "log", "--oneline", "-12"], capture_output=True, text=True, timeout=5).stdout
    except Exception: glog = ""
    opts = "".join(f'<option{" selected" if l == sel else ""}>{html.escape(l)}</option>' for l in ls)
    o = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
         f'<meta http-equiv="refresh" content="20"><title>Autoresearch Results</title><style>{CSS}</style></head><body><main>',
         f'<h1>autoresearch <form style="display:inline" method="get"><select name="loop" onchange="this.form.submit()">{opts}</select></form></h1>']
    if ms:
        imp = (best[0] / base - 1) * 100 if base else 0
        o.append(f'<div class="card kpi"><div><span class="mu">baseline</span><b class="num">{base:.4g}</b></div>'
                 f'<div><span class="mu">best</span><b class="num">{best[0]:.4g}</b></div>'
                 f'<div><span class="mu">improvement</span><b class="num">{imp:+.1f}%</b></div>'
                 + "".join(f'<div><span class="mu">{lab}</span><b class="num" style="color:{c}">{fmt_sp(k, rows)}</b></div>' for k, lab, c in SPEEDS) +
                 f'<div><span class="mu">kept / other</span><b class="num">{kept} / {disc}</b></div>'
                 f'<div><span class="mu">iterations</span><b class="num">{len(rows)}</b></div></div>')
    o.append(f'<div class="card"><div class="mu" style="margin-bottom:6px">score per iteration</div>{chart(rows)}</div>')
    sc = speed_chart(rows)
    if sc: o.append(f'<div class="card">{sc}</div>')
    if rows:
        cols = [c for c in hdr]
        if any(speeds(r) for r in rows) and "guard-metric" in cols:
            gi = cols.index("guard-metric"); cols[gi:gi + 1] = ["prefill", "decode", "guard-metric"]
            rows = [dict(r, prefill=f'{speeds(r)["prefill"]:g}' if "prefill" in speeds(r) else "",
                         decode=f'{speeds(r)["decode"]:g}' if "decode" in speeds(r) else "") for r in rows]
        o.append('<div class="card"><table><tr>' + "".join(f"<th>{html.escape(c)}</th>" for c in cols) + "</tr>")
        for r in reversed(rows):
            o.append("<tr>" + "".join(
                f'<td class="{"num " if c in ("iteration","metric","delta","commit","timestamp","prefill","decode") else ""}{html.escape(r.get("status","")) if c=="status" else ""}">'
                f'{html.escape((r.get(c) or "")[:19].replace("T"," ") if c=="timestamp" else (r.get(c) or ""))}</td>' for c in cols) + "</tr>")
        o.append("</table></div>")
    if live: o.append(f'<div class="card"><div class="mu">running now (live.txt)</div><pre>{html.escape(live)}</pre></div>')
    o.append(f'<div class="card"><div class="mu">git log</div><pre>{html.escape(glog)}</pre></div>')
    o.append('<p class="mu">auto-refresh 20 s</p></main></body></html>')
    return "".join(o)

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path != "/": self.send_error(404); return
        sel = urllib.parse.parse_qs(u.query).get("loop", [""])[0]
        b = page(sel).encode()
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b))); self.send_header("Cache-Control", "no-store"); self.end_headers()
        self.wfile.write(b)
    def log_message(self, *a): pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--host", default="100.92.209.8"); ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    print(f"http://{a.host}:{a.port}/", flush=True)
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()
