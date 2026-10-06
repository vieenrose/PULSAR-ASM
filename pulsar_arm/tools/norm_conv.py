import struct
import json
import glob
import math

f = glob.glob("/home/luigi/.cache/huggingface/hub/"
              "models--google--gemma-3-270m-it-qat-q4_0-unquantized/snapshots/*/"
              "model.safetensors")[0]
fh = open(f, "rb")
n = struct.unpack("<Q", fh.read(8))[0]
hdr = json.loads(fh.read(n))
base = 8 + n
print("dtype:", hdr["model.embed_tokens.weight"]["dtype"], flush=True)

SCALE = {"BF16": 2, "F16": 2, "F32": 4}


def to_f32(name, byte_off, nelem):
    dt = hdr[name]["dtype"]
    nb = nelem * SCALE[dt]
    fh.seek(base + byte_off)
    raw = fh.read(nb)
    if dt == "BF16":
        us = struct.unpack("<%dH" % nelem, raw)
        # bf16 -> f32 (shift left 16)
        return struct.unpack("<%df" % nelem,
                             struct.pack("<%dI" % nelem,
                                         *[u << 16 for u in us]))
    if dt == "F16":
        return struct.unpack("<%de" % nelem, raw) if False else \
            [float(x) for x in struct.unpack("<%dH" % nelem, raw)]
    return struct.unpack("<%df" % nelem, raw)


e = hdr["model.embed_tokens.weight"]
row2 = to_f32("model.embed_tokens.weight", e["data_offsets"][0] + 2 * 640 * SCALE[e["dtype"]], 640)
w_in = to_f32("model.layers.0.input_layernorm.weight",
              hdr["model.layers.0.input_layernorm.weight"]["data_offsets"][0], 640)
print("row2[:4] =", "[" + ", ".join(f"{row2[i]:.7f}" for i in range(4)) + "]",
      " (engine X = [-0.3350655, 0.3335214, 0.3582267, 0.2099950])", flush=True)
print("w_in[:4] =", "[" + ", ".join(f"{w_in[i]:.7f}" for i in range(4)) + "]", flush=True)
ms = sum(v * v for v in row2) / 640.0
inv = 1.0 / math.sqrt(ms + 1e-6)
A = [row2[i] * inv * w_in[i] for i in range(640)]
B = [row2[i] * inv * (1.0 + w_in[i]) for i in range(640)]
fmt = lambda v: "[" + ", ".join(f"{v[i]:.7f}" for i in range(4)) + "]"
print("engine H  = [-5.1960344, 5.4337134, 3.7394976, 8.4136829]  max 204.9158325", flush=True)
print("A raw-w   =", fmt(A), " max", f"{max(abs(v) for v in A):.7f}", flush=True)
print("B 1+w     =", fmt(B), " max", f"{max(abs(v) for v in B):.7f}", flush=True)
