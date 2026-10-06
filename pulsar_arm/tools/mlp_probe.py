"""Ground-truth probe for the layer-0 MLP path (HF order, fp64).

Prints the same stage signatures the asm engine prints in fwd_layer0 so the
two can be diffed line by line: X1 (residual), PN (pre-FF norm output),
W5 (folded pre-FF norm weight), G (gate), U (up), GG (gelu(gate)*up), D, X2.

  residual  = h + post_attention_layernorm(attn_out)
  pn        = pre_feedforward_layernorm(residual)
  gg        = gelu_pytorch_tanh(gate @ pn) * (up @ pn)
  x2        = residual + post_feedforward_layernorm(down @ gg)
"""
import struct
import json
import glob
import math
import numpy as np

F = glob.glob("/home/luigi/.cache/huggingface/hub/"
              "models--google--gemma-3-270m-it-qat-q4_0-unquantized/"
              "snapshots/*/model.safetensors")[0]
fh = open(F, "rb")
n = struct.unpack("<Q", fh.read(8))[0]
hdr = json.loads(fh.read(n))
BASE = 8 + n


def T(name, nelem):
    e = hdr[name]
    dt = e["dtype"]
    nb = 2 if dt in ("BF16", "F16") else 4
    fh.seek(BASE + e["data_offsets"][0])
    raw = fh.read(nb * nelem)
    if dt == "BF16":
        us = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
        return (us << 16).view(np.float32).astype(np.float64)
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)


def M(name, o, i):
    return T(name, o * i).reshape(o, i)


def rms(x, w, eps=1e-6):
    return x / math.sqrt(float((x * x).mean()) + eps) * (1.0 + w)


def sig(name, v):
    v = np.asarray(v, dtype=np.float64).ravel()
    print(f"{name} max={np.abs(v).max():.7f} "
          f"v=[{v[0]:.7f},{v[1]:.7f},{v[2]:.7f},{v[3]:.7f}]", flush=True)


L = "model.layers.0."
emb = T("model.embed_tokens.weight", 640 * 262144)
h = emb[2 * 640:3 * 640] * math.sqrt(640.0)          # scaled embedding
H = rms(h, T(L + "input_layernorm.weight", 640))
AV = np.tile(M(L + "self_attn.v_proj.weight", 256, 640) @ H, 4)
ao = M(L + "self_attn.o_proj.weight", 640, 1024) @ AV
POST = rms(ao, T(L + "post_attention_layernorm.weight", 640))
X1 = h + POST
w_pf = T(L + "pre_feedforward_layernorm.weight", 640)
sig("PY W5", 1.0 + w_pf)
PN = rms(X1, w_pf)
sig("PY PN", PN)
Wg = M(L + "mlp.gate_proj.weight", 2048, 640)
Wu = M(L + "mlp.up_proj.weight", 2048, 640)
Wd = M(L + "mlp.down_proj.weight", 640, 2048)
G = Wg @ PN
sig("PY G", G)
U = Wu @ PN
sig("PY U", U)
t = 0.7978845608028654 * (G + 0.044715 * G ** 3)
GG = 0.5 * G * (1.0 + np.tanh(t)) * U
sig("PY GG", GG)
D = Wd @ GG
sig("PY D", D)
X2 = X1 + rms(D, T(L + "post_feedforward_layernorm.weight", 640))
sig("PY X2", X2)
sig("PY X1", X1)
