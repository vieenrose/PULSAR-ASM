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

NAMES = sorted(k for k in hdr
               if k.startswith("model.layers.0."))


def T(name, nelem):
    e = hdr[name]
    nb = 2 if e["dtype"] in ("BF16", "F16") else 4
    fh.seek(BASE + e["data_offsets"][0])
    raw = fh.read(nb * nelem)
    if e["dtype"] == "BF16":
        us = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
        return (us << 16).view(np.float32).astype(np.float64)
    return np.frombuffer(raw, dtype=np.float32).astype(np.float64)


def rms(x, w, eps=1e-6):
    return x / math.sqrt(float((x * x).mean()) + eps) * (1.0 + w)


print("layer-0 tensors in file order (engine slot k = index):", flush=True)
for k, nm in enumerate(NAMES):
    print(f"  k={k} {nm}", flush=True)

L = "model.layers.0."
emb = T("model.embed_tokens.weight", 640 * 262144)
h = emb[2 * 640:3 * 640] * math.sqrt(640.0)  # HF scales the embedding
# by sqrt(hidden_size) (Gemma3TextScaledWordEmbedding). Without this the
# residual after post_attention_layernorm is wrong and every MLP signature
# below diverges from HF even though the engine is correct.
H = rms(h, T(L + "input_layernorm.weight", 640))
AV = np.tile(T(L + "self_attn.v_proj.weight", 256 * 640).reshape(256, 640) @ H, 4)
ao = T(L + "self_attn.o_proj.weight", 640 * 1024).reshape(640, 1024) @ AV
POST = rms(ao, T(L + "post_attention_layernorm.weight", 640))
resid = h + POST
Wg = T(L + "mlp.gate_proj.weight", 2048 * 640).reshape(2048, 640)
Wu = T(L + "mlp.up_proj.weight", 2048 * 640).reshape(2048, 640)

ENGINE = [0.0128295, -0.0473040, 0.1459861, 0.2033802]
print(f"\nengine GG v={ENGINE} max=184.6180", flush=True)
for k, nm in enumerate(NAMES):
    nel = int(np.prod(hdr[nm]["shape"]))
    if nel < 640:
        continue
    w = T(nm, nel)[:640]
    pn = rms(resid, w)
    g = Wg @ pn
    u = Wu @ pn
    t = 0.7978845608 * (g + 0.044715 * g ** 3)
    gg = 0.5 * g * (1.0 + np.tanh(t)) * u
    hit = "  <<< MATCH" if (abs(gg[0] - ENGINE[0]) < 1e-3
                            and abs(gg[1] - ENGINE[1]) < 1e-3) else ""
    print(f"k={k:2d} {nm.split('.')[-2]}.{nm.split('.')[-1]:28s} "
          f"v=[{gg[0]:.4f}, {gg[1]:.4f}, {gg[2]:.4f}, {gg[3]:.4f}] "
          f"max={np.abs(gg).max():9.4f}{hit}", flush=True)
