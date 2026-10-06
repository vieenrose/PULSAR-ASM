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
    if dt == "F16":
        return np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) \
            .view(np.float32).astype(np.float64)
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)


def M(name, out_n, in_n):
    return T(name, out_n * in_n).reshape(out_n, in_n)


def rms(x, w, eps=1e-6):
    inv = 1.0 / math.sqrt(float((x * x).mean()) + eps)
    return x * inv * (1.0 + w)


def sig(name, v):
    v = np.asarray(v).ravel()
    print(f"{name} max={np.abs(v).max():.7f} "
          f"v=[{v[0]:.7f}, {v[1]:.7f}, {v[2]:.7f}, {v[3]:.7f}]", flush=True)


L = "model.layers.0."
emb = T("model.embed_tokens.weight", 640 * 262144)
h = emb[2 * 640:3 * 640] * math.sqrt(640.0)  # HF scales the embedding
# by sqrt(hidden_size) (Gemma3TextScaledWordEmbedding). Without this the
# residual after post_attention_layernorm is wrong and every MLP signature
# below diverges from HF even though the engine is correct.
H = rms(h, T(L + "input_layernorm.weight", 640))
sig("PY H", H)
AV = np.tile(M(L + "self_attn.v_proj.weight", 256, 640) @ H, 4)
sig("PY AV", AV)
ao = M(L + "self_attn.o_proj.weight", 640, 1024) @ AV
sig("PY AO", ao)
w_pa = T(L + "post_attention_layernorm.weight", 640)
w_pf = T(L + "pre_feedforward_layernorm.weight", 640)
w_ff = T(L + "post_feedforward_layernorm.weight", 640)
sig("PY POST(pa)", rms(ao, w_pa))
resid = h + rms(ao, w_pa)
pn = rms(resid, w_pf)
sig("PY PN(pf)", pn)
gate = M(L + "mlp.gate_proj.weight", 2048, 640) @ pn
up = M(L + "mlp.up_proj.weight", 2048, 640) @ pn
t = 0.7978845608 * (gate + 0.044715 * gate ** 3)
gg = 0.5 * gate * (1.0 + np.tanh(t)) * up
sig("PY GG", gg)
d = M(L + "mlp.down_proj.weight", 640, 2048) @ gg
sig("PY D", d)
sig("PY X2", resid + rms(d, w_ff))
