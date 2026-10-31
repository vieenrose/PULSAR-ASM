#!/usr/bin/env python3
# S3 spec check: the audio encoder re-derived from the checkpoint weights in float32
# numpy, compared stage by stage against the HF dump (dump_encoder.py). This pins the
# math (conv/attention/RoPE/mask/norm conventions) before any C or CUDA is written.
# Run with -I from a directory that is not this one.
import sys, os, numpy as np, torch
from safetensors import safe_open

MODEL = "/home/louis/models/lfm3/LFM3-Audio-2B/model.safetensors"
REF = sys.argv[1] if len(sys.argv) > 1 else "/home/louis/PULSAR-ASM/.auto/lfm3/enc_ref/ref"
EPS = 1e-5
WIN = 32   # streaming-only (unused offline; see attn_block)
THETA = 1e6
HEAD = 96

f = safe_open(MODEL, "pt")
def W(name):
    return f.get_tensor(name).to(torch.float32).numpy()

# ---------------- primitives (fp32, reference order) ----------------
def rms(x, w):
    v = np.mean(x * x, axis=-1, keepdims=True)
    return w * (x / np.sqrt(v + EPS))

def silu(x):
    return x / (1.0 + np.exp(-x))

def gelu(x):
    from math import sqrt
    import scipy.special as sp
    return 0.5 * x * (1.0 + sp.erf(x / np.sqrt(2.0)))

def conv2d(x, w, b, stride, pad, groups):
    # x: (C_in, H, W); w: (C_out, C_in/groups, kh, kw)
    cin, H, Wd = x.shape
    cout, cig, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    Ho = (H + 2 * pad - kh) // stride + 1
    Wo = (Wd + 2 * pad - kw) // stride + 1
    out = np.zeros((cout, Ho, Wo), np.float32)
    cog = cout // groups
    for g in range(groups):
        for oc in range(cog):
            o = g * cog + oc
            acc = np.zeros((Ho, Wo), np.float32)
            for ic in range(cig):
                ci = g * cig + ic
                for i in range(kh):
                    for j in range(kw):
                        acc += w[o, ic, i, j] * xp[ci, i:i + stride * Ho:stride, j:j + stride * Wo:stride]
            out[o] = acc + b[o]
    return out

def downsample(feat):
    # feat: (128, T) fp32 log-mel; returns (T/8, 768)
    p = "encoder.downsample."
    x = feat[None]                                            # (1, 128, T)
    x = silu(conv2d(x, W(p + "conv_in.weight"), W(p + "conv_in.bias"), 2, 1, 1))
    x = conv2d(x, W(p + "conv_mid.weight"), W(p + "conv_mid.bias"), 2, 1, 256)
    x = silu(conv2d(x, W(p + "lin_mid.weight"), W(p + "lin_mid.bias"), 1, 0, 1))
    x = conv2d(x, W(p + "conv_end.weight"), W(p + "conv_end.bias"), 2, 1, 256)
    x = silu(conv2d(x, W(p + "lin_end.weight"), W(p + "lin_end.bias"), 1, 0, 1))
    C, H, Wt = x.shape                                        # (256, 16, T/8)
    flat = x.reshape(C * H, Wt)                               # flatten(1,2): index c*H + h
    lw = W(p + "lin_out.weight")[:, :, 0]                     # (768, 4096)
    lb = W(p + "lin_out.bias")
    return (lw @ flat + lb[:, None]).T                        # (T/8, 768)

def rope_tables(n):
    inv = 1.0 / (THETA ** (np.arange(0, HEAD, 2, dtype=np.float64) / HEAD))
    fr = np.arange(n, dtype=np.float64)[:, None] * inv[None, :]
    emb = np.concatenate([fr, fr], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)

def rot_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)

def causal_dw_conv3(x, w):
    # x: (T, C); w: (C, 3). out[t,c] = sum_k w[c,k] * x[t-2+k, c], zeros before t=0
    T, C = x.shape
    xp = np.concatenate([np.zeros((2, C), np.float32), x], axis=0)
    out = np.zeros((T, C), np.float32)
    for k in range(3):
        out += xp[k:k + T] * w[:, k][None, :]
    return out

def conv_block(h, i):
    p = "encoder.lfm.lfm.layers.%d.conv." % i
    bc = h @ W(p + "in_proj.weight").T                        # (T, 3C), no bias
    B, Cg, X = bc[:, :768], bc[:, 768:1536], bc[:, 1536:]
    y = causal_dw_conv3(B * X, W(p + "conv.weight")[:, 0, :])
    return (Cg * y) @ W(p + "out_proj.weight").T

def attn_block(h, i, cos, sin):
    p = "encoder.lfm.lfm.layers.%d.self_attn." % i
    T = h.shape[0]
    q = (h @ W(p + "q_proj.weight").T).reshape(T, 8, HEAD)
    k = (h @ W(p + "k_proj.weight").T).reshape(T, 4, HEAD)
    v = (h @ W(p + "v_proj.weight").T).reshape(T, 4, HEAD)
    q = rms(q, W(p + "q_layernorm.weight"))
    k = rms(k, W(p + "k_layernorm.weight"))
    c, s = cos[:, None, :], sin[:, None, :]
    q = q * c + rot_half(q) * s
    k = k * c + rot_half(k) * s
    out = np.zeros((T, 8, HEAD), np.float32)
    i_idx = np.arange(T)[:, None]
    j_idx = np.arange(T)[None, :]
    # OFFLINE forward: full causal. The checkpoint's encoder layer_types are all
    # "full_attention", so HF's offline mask has no window (verified: the windowed
    # variant misses by 1e-2 rms at layer 2, full causal matches to 1.3e-7). The 32-token
    # window exists only in the streaming path (cache remaps layers to sliding_attention).
    allowed = (j_idx <= i_idx)
    for hq in range(8):
        kv = hq // 2                                          # GQA: 2 q heads per kv head
        sc = (q[:, hq, :] @ k[:, kv, :].T) * (HEAD ** -0.5)
        sc = np.where(allowed, sc, -np.inf)
        sc = sc - sc.max(axis=-1, keepdims=True)
        pr = np.exp(sc)
        pr = pr / pr.sum(axis=-1, keepdims=True)
        out[:, hq, :] = pr @ v[:, kv, :]
    return out.reshape(T, 8 * HEAD) @ W(p + "out_proj.weight").T

def mlp(x, i):
    p = "encoder.lfm.lfm.layers.%d.feed_forward." % i
    return (silu(x @ W(p + "w1.weight").T) * (x @ W(p + "w3.weight").T)) @ W(p + "w2.weight").T

def encoder(feat):
    x = downsample(feat)                                      # (T', 768)
    cos, sin = rope_tables(x.shape[0])
    stages = {"ds_out": x}
    types = ["conv", "conv", "full_attention", "conv", "conv", "full_attention", "conv", "conv",
             "full_attention", "conv", "conv", "full_attention", "conv", "conv", "full_attention",
             "full_attention", "full_attention"]
    for i, t in enumerate(types):
        p = "encoder.lfm.lfm.layers.%d." % i
        h = rms(x, W(p + "operator_norm.weight"))
        op = conv_block(h, i) if t == "conv" else attn_block(h, i, cos, sin)
        x = x + op
        x = x + mlp(rms(x, W(p + "ffn_norm.weight")), i)
        stages["layer_%02d" % i] = x
    x = rms(x, W("encoder.lfm.lfm.embedding_norm.weight"))
    stages["enc_out"] = x
    return stages

def projector(x):
    h = x
    mu = h.mean(-1, keepdims=True)
    var = ((h - mu) ** 2).mean(-1, keepdims=True)
    h = (h - mu) / np.sqrt(var + 1e-5) * W("mlp.model.0.weight") + W("mlp.model.0.bias")
    h = gelu(h @ W("mlp.model.1.weight").T + W("mlp.model.1.bias"))
    return h @ W("mlp.model.3.weight").T + W("mlp.model.3.bias")

def cmp(name, mine, ref):
    d = np.abs(mine - ref)
    rel = np.sqrt(np.mean(d * d)) / (np.sqrt(np.mean(ref * ref)) + 1e-30)
    print("%-10s shape %-14s max|d| %.3e   rms rel %.3e" % (name, str(mine.shape), d.max(), rel))
    return rel

if __name__ == "__main__":
    feat = np.load(os.path.join(REF, "input.npy"))[0].astype(np.float32)   # (128, T)
    st = encoder(feat)
    worst = 0.0
    worst = max(worst, cmp("ds_out", st["ds_out"], np.load(os.path.join(REF, "ds_out.npy"))[0]))
    for i in range(16):   # hidden_states[17] is POST embedding_norm: layer 16 is covered by enc_out
        ref = np.load(os.path.join(REF, "hidden_%02d.npy" % (i + 1)))[0]
        worst = max(worst, cmp("layer_%02d" % i, st["layer_%02d" % i], ref))
    worst = max(worst, cmp("enc_out", st["enc_out"], np.load(os.path.join(REF, "enc_out.npy"))[0]))
    pr = projector(st["enc_out"])
    worst = max(worst, cmp("proj_out", pr, np.load(os.path.join(REF, "proj_out.npy"))[0]))
    print("worst rms rel %.3e  %s" % (worst, "PASS" if worst < 1e-4 else "FAIL"))
