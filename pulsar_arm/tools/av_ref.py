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
h = m.model.embed_tokens(torch.tensor([[2]]))[0, 0].float()
folded = (1.0 + L.input_layernorm.weight.double()).float()
H = h * torch.rsqrt(h.pow(2).mean() + 1e-6) * folded
v = H @ L.self_attn.v_proj.weight.float().T
k = H @ L.self_attn.k_proj.weight.float().T
print("V[0..3] (== attn out at pos0, single key):",
      [round(float(x), 7) for x in v[:4]], flush=True)
print("V max:", round(float(v.abs().max()), 7), flush=True)
kraw = k[:4]
print("K_raw[0..3]:", [round(float(x), 7) for x in kraw], flush=True)
qf = (1.0 + L.self_attn.q_norm.weight.double()).float()
kf = (1.0 + L.self_attn.k_norm.weight.double()).float()
kk = k * torch.rsqrt(k.pow(2).mean() + 1e-6) * kf
print("KN(pos0, no rope)[0..3]:", [round(float(x), 7) for x in kk[:4]], flush=True)
q = H @ L.self_attn.q_proj.weight.float().T
qq = q[:4] * torch.rsqrt((q[:4]).pow(2).mean() + 1e-6) * qf[:4]
print("QN0(pos0)[0..3]:", [round(float(x), 7) for x in qq], flush=True)
