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
    nb = 2 if e["dtype"] in ("BF16", "F16") else 4
    fh.seek(BASE + e["data_offsets"][0])
    raw = fh.read(nb * nelem)
    if e["dtype"] == "BF16":
        us = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
        return (us << 16).view(np.float32).astype(np.float64)
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)


def M(name, o, i):
    return T(name, o * i).reshape(o, i)


def rms(x, w, eps=1e-6):
    return x / math.sqrt(float((x * x).mean()) + eps) * (1.0 + w)


L = "model.layers.0."
emb = T("model.embed_tokens.weight", 640 * 262144)
h = emb[2 * 640:3 * 640] * math.sqrt(640.0)  # HF scales the embedding
# by sqrt(hidden_size) (Gemma3TextScaledWordEmbedding). Without this the
# residual after post_attention_layernorm is wrong and every MLP signature
# below diverges from HF even though the engine is correct.
H = rms(h, T(L + "input_layernorm.weight", 640))
AV = np.tile(M(L + "self_attn.v_proj.weight", 256, 640) @ H, 4)
ao = M(L + "self_attn.o_proj.weight", 640, 1024) @ AV
resid = h + rms(ao, T(L + "post_attention_layernorm.weight", 640))
FH = rms(resid, T(L + "pre_feedforward_layernorm.weight", 640))

Wg = M(L + "mlp.gate_proj.weight", 2048, 640)
Wu = M(L + "mlp.up_proj.weight", 2048, 640)
Wd = M(L + "mlp.down_proj.weight", 640, 2048)
Wq = M(L + "self_attn.q_proj.weight", 1024, 640)
Wk = M(L + "self_attn.k_proj.weight", 256, 640)
Wo = M(L + "self_attn.o_proj.weight", 640, 1024)
pad = np.zeros((2048 - 1024, 640))
Wp = np.vstack([Wq, pad])
Wk2 = np.vstack([Wk, np.zeros((2048 - 256, 640))])
Wo2 = np.vstack([Wo, np.zeros((2048 - 640, 1024))])
Wdd = np.vstack([Wd, np.zeros((2048 - 640, 2048))])

ERF = np.vectorize(math.erf)


def gg_of(FH, A, B, tag, erf=False):
    g = A @ FH
    u = B @ FH
    if erf:
        act = 0.5 * g * (1.0 + ERF(g / math.sqrt(2.0)))
    else:
        t = 0.7978845608 * (g + 0.044715 * g ** 3)
        act = 0.5 * g * (1.0 + np.tanh(t))
    gg = act * u
    print(f"{tag:28s} v=[{gg[0]:.7f}, {gg[1]:.7f}, {gg[2]:.7f}, {gg[3]:.7f}]"
          f" max={np.abs(gg).max():.4f}", flush=True)


print("engine GG                    v=[0.0128295, -0.0473040, 0.1459861, "
      "0.2033802] max=184.6180", flush=True)
gg_of(FH, Wg, Wu, "gate,up tanh")
gg_of(FH, Wu, Wg, "up,gate swapped")
gg_of(FH, Wg, Wu, "gate,up erf", erf=True)
gg_of(FH, Wp, Wu, "qpad,up")
gg_of(FH, Wg, Wp, "gate,qpad")
gg_of(resid, Wg, Wu, "resid gate,up")
