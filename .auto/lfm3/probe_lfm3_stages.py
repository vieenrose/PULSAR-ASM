#!/usr/bin/env python
"""HF-side counterpart to `lfm3_nanode --trace`'s layer-0 stage probe.

Prints rms + the first four values of every named stage of decoder layer 0,
the same labels the engine prints, so the first stage where the two tables
stop agreeing is the stage that is wrong. Run: ~/venvs/lfm3/bin/python
.auto/lfm3/probe_lfm3_stages.py

rms alone is a weak fingerprint (permutation-invariant), which is why the
first four values travel with it.
"""
import sys, numpy as np, torch
from transformers import AutoModelForMultimodalLM
P = '/home/louis/models/lfm3/LFM3-Audio-2B'
m = AutoModelForMultimodalLM.from_pretrained(P, trust_remote_code=True, dtype=torch.bfloat16).eval()
L = m.model.language_model.layers[0]
print("ffn children:", [n for n,_ in L.feed_forward.named_children()], file=sys.stderr)
print("conv children:", [n for n,_ in L.conv.named_children()], file=sys.stderr)
rec = {}
def rms(t, name):
    v = t.detach().float().reshape(-1)
    rec[name] = (float(v.pow(2).mean().sqrt()), [float(x) for x in v[:4].tolist()])
def wrap(mod, name):
    def hook(_m, _i, out):
        rms(out, name)
    mod.register_forward_hook(hook)
wrap(L.operator_norm, 'operator_norm')
wrap(L.conv.in_proj, 'in_proj')
wrap(L.conv.conv, 'conv_raw')            # depthwise conv1d module
wrap(L.conv.out_proj, 'out_proj')
wrap(L.post_operator_norm, 'post_op_norm')
wrap(L.ffn_norm, 'ffn_norm')
ff = L.feed_forward
for nm in ('w1', 'w2', 'w3'):
    if hasattr(ff, nm): wrap(getattr(ff, nm), nm)
ids = torch.tensor([[124894]])
with torch.no_grad():
    o = m(input_ids=ids, output_hidden_states=True, use_cache=False)
# the gating multiplies are not modules, so recompute them from the recorded
# tensors: B, C, x = in_proj(h).transpose(-1,-2).chunk(3,-2); B*x -> conv -> C*y
st = {}
def cap(mod, name):
    def hook(_m, _i, out):
        st[name] = out.detach()
    mod.register_forward_hook(hook)
cap(L.conv.in_proj, 'in_proj_t')
cap(L.conv.conv, 'conv_t')
for nm in ('w1', 'w2', 'w3'):
    if hasattr(ff, nm): cap(getattr(ff, nm), nm)
with torch.no_grad():
    m(input_ids=ids, use_cache=False)
ip = st['in_proj_t']                     # (b, L, 3H)
BCx = ip.transpose(-1, -2)
B, C, x = BCx.chunk(3, dim=-2)
rms(B * x, 'B*x')
rms(st['conv_t'], 'conv')
rms(C * st['conv_t'], 'C*y')
if hasattr(ff, 'w1'):
    rms(torch.nn.functional.silu(st['w1']) * st['w3'], 'silu_mul')
print(f"{'stage':16s} {'rms':>10s}   first4")
order = ['operator_norm', 'in_proj', 'B*x', 'conv', 'C*y', 'out_proj', 'post_op_norm',
         'ffn_norm', 'w1', 'w3', 'silu_mul', 'w2', 'post_ffn_norm']
for k in order:
    if k in rec:
        r, v = rec[k]
        print(f"{k:16s} {r:10.5f}   " + " ".join(f"{y:8.4f}" for y in v))
    else:
        print(f"{k:16s} {'(n/a)':>10s}")
hs = getattr(o, "hidden_states", None) or m.model.language_model(input_ids=ids, output_hidden_states=True).hidden_states
print("embedding rms", float(hs[0][0, 0].float().pow(2).mean().sqrt()))
print("after layer0  rms", float(hs[1][0, 0].float().pow(2).mean().sqrt()))

# ---- which tap does the newest input pair with? --------------------------------
# conv output rms must equal rms(w[:,k]*x) for the k that the newest input pairs
# with. Measured above: HF conv rms 0.00273, engine conv rms 0.00493, and
# rms(x) = rms(B*x). Printing the candidate k's makes the answer arithmetic
# rather than inference.
import torch as _t
W = L.conv.conv.weight.detach().squeeze(1).float()          # (H, 7)
xvec = (B * x).detach().float().reshape(-1).contiguous()   # (H,) -- reshape, NOT
# transpose: squeeze(0).transpose(0,1) leaves shape (H,1), and W[:,k] * xvec then
# broadcasts to an (H,H) OUTER PRODUCT, so `.pow(2).mean()` was the mean of a
# rank-1 matrix rather than rms(w[:,k]*x). The numbers looked plausible.
print(f"\nrms(conv input B*x) = {float(xvec.pow(2).mean().sqrt()):.5f}")
print(f"engine conv rms 0.00493   HF conv rms 0.00273")
for k in range(7):
    print(f"  rms(w[:,{k}] * x) = {float((W[:, k] * xvec).pow(2).mean().sqrt()):.5f}")
allx = _t.stack([xvec] * 7, dim=1)                     # x in every slot
print(f"  rms(sum_k w[:,k]*x)  = {float((W * allx).sum(0).pow(2).mean().sqrt()):.5f}  (x in all 7 slots)")
print("  conv bias:", None if L.conv.conv.bias is None else float(L.conv.conv.bias.pow(2).mean().sqrt()))
print("  conv_padding:", L.conv.conv.padding, "kernel", L.conv.conv.kernel_size, "groups", L.conv.conv.groups)

# ---- attention-branch stages (for the engine's --probe-layer 2) --------------
import sys as _s
PL = int(_s.argv[1]) if len(_s.argv) > 1 else 2
LP = m.model.language_model.layers[PL]
ast = {}
for nm, mod in (('q_proj', LP.self_attn.q_proj), ('k_proj', LP.self_attn.k_proj),
                ('v_proj', LP.self_attn.v_proj), ('gate_proj', LP.self_attn.gate_proj),
                ('o_proj', LP.self_attn.o_proj), ('q_norm', LP.self_attn.q_norm),
                ('k_norm', LP.self_attn.k_norm), ('post_op_norm', LP.post_operator_norm)):
    cap(mod, f'a_{nm}')
with torch.no_grad():
    m(input_ids=ids, use_cache=False)
print(f"\n--- layer {PL} attention branch ---")
for nm in ('a_q_norm', 'a_k_norm', 'a_v_proj', 'a_gate_proj', 'a_o_proj', 'a_post_op_norm'):
    if nm in st:
        rms(st[nm], nm)
        r, v = rec[nm]
        print(f"{nm:16s} {r:10.5f}   " + " ".join(f"{y:8.4f}" for y in v[:4]))
import torch as _t
g = st['a_gate_proj'].detach().float()
print("sigmoid(gate) rms", float(_t.sigmoid(g).pow(2).mean().sqrt()),
      "first4", [round(float(x), 4) for x in _t.sigmoid(g).flatten()[:4].tolist()])
v4 = st['a_v_proj'].detach().float()
print("v_proj rms", float(v4.pow(2).mean().sqrt()), "shape", tuple(v4.shape))
