import struct
import json
import glob
import math
import numpy as np

F = glob.glob("/home/luigi/.cache/huggingface/hub/"
             "models--google--gemma-3-270m-it-qat-q4_0-unquantized/snapshots/*/"
             "model.safetensors")[0]
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


L = "model.layers.0."
emb = T("model.embed_tokens.weight", 640 * 262144)
h = emb[2 * 640:3 * 640]
H = rms(h, T(L + "input_layernorm.weight", 640))
AV = np.tile(M(L + "self_attn.v_proj.weight", 256, 640) @ H, 4)
ao = M(L + "self_attn.o_proj.weight", 640, 1024) @ AV
resid = h + rms(ao, T(L + "post_attention_layernorm.weight", 640))
Wg = M(L + "mlp.gate_proj.weight", 2048, 640)
Wu = M(L + "mlp.up_proj.weight", 2048, 640)

ENGINE_GG = [0.0128295, -0.0473040, 0.1459861, 0.2033802]
ENGINE_MAX = 184.6179809
CANDS = {
    "pre_feedforward": L + "pre_feedforward_layernorm.weight",
    "post_feedforward": L + "post_feedforward_layernorm.weight",
    "post_attention": L + "post_attention_layernorm.weight",
    "input_layernorm": L + "input_layernorm.weight",
}
print("engine GG      v=[%s] max=%.4f" % (
    ", ".join(f"{v:.7f}" for v in ENGINE_GG), ENGINE_MAX), flush=True)
for label, name in CANDS.items():
    pn = rms(resid, T(name, 640))
    g = Wg @ pn
    u = Wu @ pn
    t = 0.7978845608 * (g + 0.044715 * g ** 3)
    gg = 0.5 * g * (1.0 + np.tanh(t)) * u
    v = np.abs(gg).max()
    print(f"{label:18s} v=[{gg[0]:.7f}, {gg[1]:.7f}, {gg[2]:.7f}, {gg[3]:.7f}]"
          f" max={v:.4f}", flush=True)
