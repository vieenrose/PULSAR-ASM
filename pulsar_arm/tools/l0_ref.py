import os
os.environ["TORCH_DISABLE_MKLDNN"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
torch.backends.mkldnn.enabled = False
torch.set_num_threads(1)
from transformers import AutoModelForCausalLM

repo = ("/home/luigi/.cache/huggingface/hub/"
        "models--google--gemma-3-270m-it-qat-q4_0-unquantized/snapshots/"
        "8f726c6a497fd439f0d6f726e52f8e3b439a26e5")
m = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32,
                                         device_map="cpu",
                                         attn_implementation="eager")
m.eval()
L = m.model.layers[0]
ids = torch.tensor([[2]])


def sig(name, v):
    t = v.detach().float().flatten()[:4]
    print(f"L0 {name} max={t.abs().max().item():.7f} "
          f"v=[{t[0]:.7f}, {t[1]:.7f}, {t[2]:.7f}, {t[3]:.7f}]", flush=True)


with torch.no_grad():
    h = m.model.embed_tokens(ids)[0, 0]                       # [640]
    sig("X", h)
    ln = L.input_layernorm.weight.float()
    # engine folds (1+w) in float64; emulate exactly
    folded = (1.0 + ln.double()).float()
    rs = h * torch.rsqrt(h.pow(2).mean() + 1e-6)
    H = rs * folded
    sig("H", H)

    q = H @ L.self_attn.q_proj.weight.float().T              # [1024]
    k = H @ L.self_attn.k_proj.weight.float().T              # [256]
    v = H @ L.self_attn.v_proj.weight.float().T              # [256]
    HD = 256
    qh = q.view(4, HD)
    kh = k.view(1, HD)
    vh = v.view(1, HD)
    qf = (1.0 + L.self_attn.q_norm.weight.double()).float()
    kf = (1.0 + L.self_attn.k_norm.weight.double()).float()
    for j in range(4):
        x = qh[j]
        qn = x * torch.rsqrt(x.pow(2).mean() + 1e-6) * qf
        if j == 0:
            sig("QN0", qn)
    xk = kh[0]
    kn = xk * torch.rsqrt(xk.pow(2).mean() + 1e-6) * kf

    # pos 0: rope is identity
    att_w = 1.0 / (HD ** 0.5)
    scores = (torch.stack([qn * kn])).sum(-1) * att_w       # [4]
    att = torch.softmax(scores, dim=-1)
    ctx = (att[:, None] * vh).sum(0)                          # [256]
    ctx4 = ctx.repeat(4)                                      # GQA expand
    sig("AV", ctx4)                                           # engine L0 AV = pre-o_proj
    ao = ctx4 @ L.self_attn.o_proj.weight.float().T
    sig("AO", ao)

    # HF Gemma3DecoderLayer order: post_attention_layernorm(ao), THEN residual.
    pa = ao * torch.rsqrt(ao.pow(2).mean() + 1e-6) \
        * (1.0 + L.post_attention_layernorm.weight.double()).float()
    sig("POST", pa)
    resid = h + pa
    sig("X1", resid)

    pn = resid * torch.rsqrt(resid.pow(2).mean() + 1e-6) \
        * (1.0 + L.pre_feedforward_layernorm.weight.double()).float()
    sig("PN", pn)
    gate = pn @ L.mlp.gate_proj.weight.float().T
    up = pn @ L.mlp.up_proj.weight.float().T
    gg = torch.nn.functional.gelu(gate, approximate="tanh") * up
    sig("GG", gg)
    d = gg @ L.mlp.down_proj.weight.float().T
    # post_feedforward_layernorm is applied to the MLP output, THEN residual add.
    dff = d * torch.rsqrt(d.pow(2).mean() + 1e-6) \
        * (1.0 + L.post_feedforward_layernorm.weight.double()).float()
    sig("D", dff)
    x2 = resid + dff
    sig("X2", x2)
