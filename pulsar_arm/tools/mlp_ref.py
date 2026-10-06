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


def sig(name, t):
    t = t.detach().float().flatten()
    print(f"{name} max={t.abs().max().item():.7f} "
          f"v=[{t[0]:.7f}, {t[1]:.7f}, {t[2]:.7f}, {t[3]:.7f}]", flush=True)


def fold(w):
    return (1.0 + w.double()).float()


def rms(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean() + eps) * fold(w)


h = m.model.embed_tokens(torch.tensor([[2]]))[0, 0].float()
sig("REF H", rms(h, L.input_layernorm.weight))
v = rms(h, L.input_layernorm.weight) @ L.self_attn.v_proj.weight.float().T
ctx = v.repeat(4)
ao = ctx @ L.self_attn.o_proj.weight.float().T
sig("REF AO(o_proj)", ao)
resid = h + ao
sig("REF RESID", resid)
pn = rms(resid, L.post_attention_layernorm.weight)
sig("REF PN", pn)
gate = pn @ L.mlp.gate_proj.weight.float().T
up = pn @ L.mlp.up_proj.weight.float().T
gg = torch.nn.functional.gelu(gate, approximate="tanh") * up
sig("REF GG", gg)
d = gg @ L.mlp.down_proj.weight.float().T
sig("REF D", d)
pre = resid + d
sig("REF PRE", pre)
o1 = rms(pre, L.pre_feedforward_layernorm.weight)
o2 = rms(o1, L.post_feedforward_layernorm.weight)
sig("REF O1", o1)
sig("REF O2", o2)
sig("REF X2", resid + o2)
