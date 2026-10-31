#!/usr/bin/env python3
# Gate for lfm3_enc.c (S3 C reference, Q8 blob weights).
#   A) C vs numpy-on-the-SAME-dequantized-blob weights: must be ~fp32 noise (validates the C code).
#   B) C vs HF (fp32 checkpoint): the Q8 quantization error, reported and compared with the
#      error the same weights produce in numpy (so quantization, not the C code, is the gap).
# Run with -I from a directory that is not this one.
import sys, os, subprocess, json, struct, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enc_numpy as E
BLOB = sys.argv[1] if len(sys.argv) > 1 else "/home/louis/models/lfm3/lfm3-audio-2b-enc.lfm3q8"
BIN = sys.argv[2] if len(sys.argv) > 2 else "/tmp/lfm3_enc"
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref")
T = 400
OUT = "/tmp/lfm3_enc_out"
os.makedirs(OUT, exist_ok=True)
subprocess.run([BIN, BLOB, os.path.join(REF, "input_128xT_f32.bin"), str(T), OUT], check=True)

# dequantized blob weights (python mirror of the C loader) for check A
with open(BLOB, "rb") as f:
    magic = f.read(8); hl = struct.unpack("<I", f.read(4))[0]; hdr = json.loads(f.read(hl)); data = f.read()
DQ = {}
for e in hdr["table"]:
    n = int(np.prod(e["ne"]))
    if e["dtype"] == 1:
        q = np.frombuffer(data, np.int8, n, e["off"]).astype(np.float32)
        s = np.frombuffer(data, np.float32, n // 32, e["off"] + e["ncodes"])
        DQ[e["name"]] = (q.reshape(e["ne"]) * np.repeat(s, 32).reshape(-1)[:n].reshape(e["ne"])).astype(np.float32)
    else:
        DQ[e["name"]] = np.frombuffer(data, np.float32, n, e["off"]).reshape(e["ne"]).copy()

def _w(name):
    t = DQ[name]
    if name.endswith("lin_out.weight") and t.ndim == 2:    # blob stores Conv1d [768,4096,1] squeezed
        t = t[..., None]
    return t
E.W = _w                             # numpy spec now runs on the blob's weights
st = E.encoder(np.load(os.path.join(REF, "input.npy"))[0].astype(np.float32))
pr = E.projector(st["enc_out"])

def rel(a, b):
    return np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(b * b)) + 1e-30)

def cload(name, shape):
    return np.fromfile(os.path.join(OUT, name), np.float32).reshape(shape)

tok = T // 8
names = [("ds_out", (tok, 768), None)] + [("layer_%02d" % i, (tok, 768), i + 1) for i in range(16)] + \
        [("enc_out", (tok, 768), "enc"), ("proj_out", (tok, 2048), "proj")]
worst_c_np = 0.0; worst_c_hf = 0.0
print("%-10s %-14s %-16s %-16s" % ("stage", "shape", "C vs numpy-dq", "C vs HF"))
for name, shape, ref_idx in names:
    c = cload(name + ".bin" if False else name, shape)
    if name == "ds_out":
        mine_np = st["ds_out"]; hf = np.load(os.path.join(REF, "ds_out.npy"))[0]
    elif name == "enc_out":
        mine_np = st["enc_out"]; hf = np.load(os.path.join(REF, "enc_out.npy"))[0]
    elif name == "proj_out":
        mine_np = pr; hf = np.load(os.path.join(REF, "proj_out.npy"))[0]
    else:
        mine_np = st[name]; hf = np.load(os.path.join(REF, "hidden_%02d.npy" % ref_idx))[0]
    a = rel(c, mine_np); b = rel(c, hf)
    worst_c_np = max(worst_c_np, a)
    print("%-10s %-14s %-16.3e %-16.3e" % (name, str(shape), a, b))
print("worst C vs numpy-on-same-weights: %.3e  -> %s" % (worst_c_np, "C CODE OK" if worst_c_np < 1e-5 else "C CODE MISMATCH"))
# the quantization gap as the numpy spec sees it: dequantized weights vs fp32 checkpoint (HF)
print("numpy(dequant blob) vs HF, enc_out: %.3e  (Q8 quantization error, not a C bug)" %
      rel(st["enc_out"], np.load(os.path.join(REF, "enc_out.npy"))[0]))
print("proj_out:   C vs HF %.3e  numpy(dequant) vs HF %.3e" % (
      rel(cload("proj_out", (tok, 2048)), np.load(os.path.join(REF, "proj_out.npy"))[0]),
      rel(pr, np.load(os.path.join(REF, "proj_out.npy"))[0])))
sys.exit(0 if worst_c_np < 1e-5 else 1)
