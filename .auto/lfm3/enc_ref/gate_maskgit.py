"""gate_maskgit.py -- S4 completion gate: the in-binary MaskGIT sampler vs the HF oracle.

Usage: gate_maskgit.py <oracle_dir> <depth_bin> <blob>
For every frame: run  lfm3_depth <blob> hidden.<i>.f32 --maskgit order.<i>.i32 <hist> <out>
with <hist> = the oracle's earlier frames (frame-major), then require code equality.
"""
import json, os, subprocess, sys
import numpy as np

G, ENG, BLOB = sys.argv[1], sys.argv[2], sys.argv[3]
W = "/tmp/mgate"
os.makedirs(W, exist_ok=True)
man = json.load(open(os.path.join(G, "manifest.json")))
N = man["nframes"]
frames = []
ok = 0
for i in range(N):
    hist_path = os.path.join(W, "hist.i32")
    with open(hist_path, "wb") as f:
        for fr in frames[-64:]:
            f.write(np.asarray(fr, dtype=np.int32).tobytes())
    out = os.path.join(W, "codes.i32")
    r = subprocess.run([ENG, BLOB, os.path.join(G, "hidden.%d.f32" % i), "--maskgit",
                        os.path.join(G, "order.%d.i32" % i), hist_path, out],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if r.returncode != 0:
        print("FAIL frame %d: rc=%d %s" % (i, r.returncode, r.stderr[-300:]))
        sys.exit(1)
    got = np.fromfile(out, dtype=np.int32)
    ref = np.fromfile(os.path.join(G, "codes.%d.i32" % i), dtype=np.int32)
    eq = bool((got == ref).all())
    ok += int(eq)
    frames.append(ref.tolist())
    if not eq:
        print("frame %d DIFFERS: got %s ref %s" % (i, got.tolist(), ref.tolist()))
print("MASKGIT GATE: %d/%d frames exact" % (ok, N))
sys.exit(0 if ok == N else 1)
