"""Layer-0 stage oracle for any gemma-3 checkpoint (fp64), vs the asm L0 block."""
import glob, json, math, struct, sys, collections
import numpy as np

P = sys.argv[1] if len(sys.argv) > 1 else None
if P is None:
    P = glob.glob("/home/luigi/.cache/huggingface/hub/"
                  "models--google--gemma-3-270m-it-qat-q4_0-unquantized/snapshots/*/model.safetensors")[0]
TOK = int(sys.argv[2]) if len(sys.argv) > 2 else 2
fh = open(P, "rb"); n = struct.unpack("<Q", fh.read(8))[0]
hdr = json.loads(fh.read(n)); BASE = 8 + n

def rows(name, r0, r1=None):
    e = hdr[name]; o = BASE + e["data_offsets"][0]
    shape = e["shape"]; nb = 2 if e["dtype"] in ("BF16","F16") else 4
    r1 = r1 if r1 is not None else r0 + 1
    fh.seek(o + r0 * shape[1] * nb)
    raw = fh.read((r1 - r0) * shape[1] * nb)
    if e["dtype"] == "BF16":
        u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
        v = (u << 16).view(np.float32)
    else:
        v = np.frombuffer(raw, dtype=np.float32)
    return v.reshape(r1 - r0, shape[1]).astype(np.float64)

def w(name):
    e = hdr[name]
    return rows(name, 0, e["shape"][0])

L = "model.layers.0."
HID = hdr["model.embed_tokens.weight"]["shape"][1]
HD = hdr[L + "self_attn.q_norm.weight"]["shape"][0]
NH = hdr[L + "self_attn.q_proj.weight"]["shape"][0] // HD
rms = lambda x, ww: x / math.sqrt(float((x*x).mean()) + 1e-6) * (1.0 + ww)
def sig(t, v):
    v = np.asarray(v).ravel()
    print(f"{t} max={np.abs(v).max():.7f} v=[{v[0]:.7f},{v[1]:.7f},{v[2]:.7f},{v[3]:.7f}]")
print(f"# {P.split('/')[-3]} hid={HID} hd={HD} heads={NH} tok={TOK}")
X = rows("model.embed_tokens.weight", TOK)[0] * math.sqrt(HID)
sig("PY X", X)
H = rms(X, w(L + "input_layernorm.weight")[0]); sig("PY H", H)
Q = w(L+"self_attn.q_proj.weight") @ H; K = w(L+"self_attn.k_proj.weight") @ H
V = w(L+"self_attn.v_proj.weight") @ H
qn = w(L+"self_attn.q_norm.weight")[0]; kn = w(L+"self_attn.k_norm.weight")[0]
QN = np.concatenate([rms(Q[j*HD:(j+1)*HD], qn) for j in range(NH)])
sig("PY QN0", QN[:HD])
AV = np.tile(V, NH)
sig("PY AV", AV)
ao = w(L+"self_attn.o_proj.weight") @ AV
POST = rms(ao, w(L+"post_attention_layernorm.weight")[0]); sig("PY POST", POST)
X1 = X + POST; sig("PY X1", X1)
PN = rms(X1, w(L+"pre_feedforward_layernorm.weight")[0]); sig("PY PN", PN)
g = w(L+"mlp.gate_proj.weight") @ PN; u = w(L+"mlp.up_proj.weight") @ PN
gg = 0.5*g*(1.0+np.tanh(0.7978845608028654*(g + 0.044715*g**3)))*u
sig("PY GG", gg)
d = w(L+"mlp.down_proj.weight") @ gg
dn = rms(d, w(L+"post_feedforward_layernorm.weight")[0]); sig("PY D", dn)
sig("PY X2", X1 + dn)
