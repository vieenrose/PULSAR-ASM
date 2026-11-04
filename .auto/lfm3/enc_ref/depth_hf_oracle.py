#!/usr/bin/env python
"""depth_hf_oracle.py -- S4 oracle: the depth decoder + MaskGIT sampler, from HF itself.

WHAT IT DUMPS (all under .auto/lfm3/speech/depth_gate/):
  prompt.i32          the teacher-forced prompt ids (a real chat prompt, no audio)
  hidden.i.f32        the backbone hidden state feeding frame i (2048 f32) -- REAL
                      backbone states: prefill's last hidden state, then one decode step
                      per sampled frame with next_embeds = codebook_embedding lookup,
                      i.e. exactly the loop modeling_lfm3_audio.py::_generate runs in
                      audio mode
  order.i.p.i32       the randperm(12) reveal order, split by the schedule [4,4,4], for
                      frame i pass p. torch.randperm uses the global generator, so the
                      engine consumes THESE orders as data instead of reimplementing
                      torch's Philox RNG -- the gate stays deterministic without a
                      bitwise-RNG port.
  logits.i.p.f32      depth-decoder logits [12, 2017] for frame i pass p (all codebooks,
                      every pass)
  codes.i.i32         the sampled codes [12] at temperature 0 (greedy: penalty + argmax,
                      no multinomial -- fully deterministic)
  emb.i.f32           next_embeds [2048] fed back into the backbone
  manifest.json       shapes, seed, penalty, schedule, eos frame index

SAMPLING CHAIN (exactly _sample + sample_audio_frame, audio_temperature=0):
  scores = logits.float(); repetition penalty 1.1 over THIS codebook's frame history
  (gather -> where(score<0, *1.1, /1.1) -> scatter); no warpers at temp 0 -> argmax.
  audio_repetition_window 64 (deque maxlen 64).

Teacher-forced probe (tf/): fixed codes, masked all-true and all-false; depth_linear
out, per-layer trunk rms + first-4 values, final logits -- gate (a) of the plan.

Run: /home/louis/venvs/lfm3/bin/python depth_hf_oracle.py
"""
import json
import os
from collections import deque

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# DEPTH_Q8=<blob>: evaluate HF on the SAME dequantised Q8 depth weights the engine ships
# (HF-as-contract). The fp32 oracle measures the quantization error; the Q8 oracle is the
# gate reference, because the engine computes the Q8 function, not the fp32 one.
DEPTH_Q8 = os.environ.get("DEPTH_Q8")
_BASE = os.path.join(HERE, "speech", "depth_gate_q8" if DEPTH_Q8 else "depth_gate")
OUT = _BASE if int(os.environ.get("DEPTH_PROMPT", "0")) == 0 and not os.environ.get("DEPTH_SUB") \
    else os.path.join(_BASE, "p" + os.environ.get("DEPTH_PROMPT", "0"))
MODEL = os.environ.get("LFM3_MODEL", "/home/louis/models/lfm3/LFM3-Audio-2B")
NFRAMES = int(os.environ.get("DEPTH_FRAMES", "16"))
SEED = 1234
PENALTY = 1.1
REP_WIN = 64

# Five fixtures (the S4 gate protocol asks for 5). DEPTH_PROMPT=k selects one; the
# default 0 is the original fixture and keeps its historical output directory.
PROMPTS = [
    ("<|startoftext|><|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
     "<|im_start|>user\nSay something.<|im_end|>\n<|im_start|>assistant\n"),
    ("<|startoftext|><|im_start|>system\nPerform ASR in English.<|im_end|>\n"
     "<|im_start|>user\nTell me a short fact.<|im_end|>\n<|im_start|>assistant\n"),
    ("<|startoftext|><|im_start|>system\nYou are a concise narrator.<|im_end|>\n"
     "<|im_start|>user\nRead the weather report aloud.<|im_end|>\n<|im_start|>assistant\n"),
    ("<|startoftext|><|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
     "<|im_start|>user\nGood morning, how are you today?<|im_end|>\n<|im_start|>assistant\n"),
    ("<|startoftext|><|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
     "<|im_start|>user\nCount slowly from one to five.<|im_end|>\n<|im_start|>assistant\n"),
]
FIXTURE = int(os.environ.get("DEPTH_PROMPT", "0"))
PROMPT_TEXT = PROMPTS[FIXTURE]

def load_q8_blob(path):
    """Dequantise the depth blob exactly as lfm3_depth.cu does: float(code) * scale per
    32-wide block (fp32). Returns {tensor name: float32 array} for the dtype-1 tensors."""
    import struct
    with open(path, "rb") as f:
        assert f.read(8) == b"LFM3Q8\x01\x00", "bad magic"
        n = struct.unpack("<I", f.read(4))[0]
        hdr = json.loads(f.read(n))
        base = 12 + n
        out = {}
        for e in hdr["table"]:
            if e["dtype"] != 1:
                continue
            f.seek(base + e["off"])
            codes = np.frombuffer(f.read(e["ncodes"]), dtype=np.int8).reshape(e["ne"])
            scales = np.frombuffer(f.read(e["nscales"]), dtype=np.float32).reshape(e["ne"][0], -1)
            out[e["name"]] = (codes.astype(np.float32) * np.repeat(scales, 32, axis=1)).astype(np.float32)
    return out

def apply_q8_depth(dd, q8):
    """Replace every depth-decoder Linear with its Q8 dequant. The codebook logits use the
    STACKED Q8 head (depth_head_stacked = the 12 tied embeddings, codebook-major), not the
    bf16 embedding tensors, which stay exact for the lookups."""
    with torch.no_grad():
        dd.depth_linear.weight.copy_(torch.from_numpy(q8["depth_linear.weight"]))
        for li, layer in enumerate(dd.model.layers):
            for name, mod in [("self_attn.q_proj", layer.self_attn.q_proj), ("self_attn.k_proj", layer.self_attn.k_proj),
                              ("self_attn.v_proj", layer.self_attn.v_proj), ("self_attn.o_proj", layer.self_attn.o_proj),
                              ("self_attn.gate_proj", layer.self_attn.gate_proj),
                              ("feed_forward.w1", layer.feed_forward.w1), ("feed_forward.w2", layer.feed_forward.w2),
                              ("feed_forward.w3", layer.feed_forward.w3)]:
                mod.weight.copy_(torch.from_numpy(q8[f"depthformer.model.layers.{li}.{name}.weight"]))
    head = torch.from_numpy(q8["depth_head_stacked.weight"]).view(dd.num_codebooks, -1, dd.hidden_size)
    for c, emb in enumerate(dd.depth_embeddings):
        def logits(hidden_states, _c=c, _e=emb):
            # HF: F.linear(embedding_norm(h), embedding.weight); here the weight is the Q8 head slice
            return torch.nn.functional.linear(_e.embedding_norm(hidden_states), head[_c])
        emb.logits = logits

def w32(name, a):
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    a.tofile(os.path.join(OUT, name))

def wi32(name, a):
    a = np.ascontiguousarray(np.asarray(a, dtype=np.int32))
    a.tofile(os.path.join(OUT, name))

def main():
    os.makedirs(OUT, exist_ok=True)
    try:
        from transformers import AutoModelForMultimodalLM as Loader
    except ImportError:
        from transformers import AutoModelForCausalLM as Loader
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = Loader.from_pretrained(MODEL, dtype=torch.float32, trust_remote_code=True).eval()
    # The dev box's 5.8 GB GPU cannot hold this 2B model in fp32 (8 GB); every prior
    # HF gate ran on CPU, where the latency is seconds. Refuse the GPU explicitly:
    # a partial-cuda OOM halfway through would corrupt the fixture set.
    dev = "cpu"
    m = model.model                      # Lfm3AudioModel
    cfg = model.config
    ncb = int(getattr(cfg, "num_codebooks", 12))
    steps = int(getattr(cfg, "maskgit_steps", 3))
    sched = [ncb // steps] * steps
    sched[-1] += ncb - sum(sched)
    eos_code = int(getattr(cfg, "audio_eos_code", 2016))
    print(f"loader={Loader.__name__} num_codebooks={ncb} schedule={sched} "
          f"eos_code={eos_code} dev={dev}")
    dd = m.depth_decoder
    if DEPTH_Q8:
        apply_q8_depth(dd, load_q8_blob(DEPTH_Q8))
        print("depth decoder weights replaced by the Q8 dequant of", DEPTH_Q8, flush=True)
    print("depth_linear", tuple(dd.depth_linear.weight.shape),
          "codebook_emb", tuple(m.codebook_embedding.weight.shape))

    ids = tok(PROMPT_TEXT, add_special_tokens=False).input_ids
    wi32("prompt.i32", ids)
    with open(os.path.join(OUT, "prompt.txt"), "w") as f:
        f.write(PROMPT_TEXT)
    input_ids = torch.tensor([ids], device=dev)
    attn = torch.ones_like(input_ids)
    pos = torch.arange(input_ids.shape[1], device=dev).unsqueeze(0)

    manifest = dict(prompt_len=len(ids), seed=SEED, audio_temperature=0.0,
                    repetition_penalty=PENALTY, repetition_window=REP_WIN,
                    schedule=sched, eos_code=eos_code, nframes=0, eos_at=-1)
    with torch.no_grad():
        out = m(input_ids=input_ids, attention_mask=attn, position_ids=pos, use_cache=True)
        hidden, cache = out.last_hidden_state[:, -1], out.past_key_values
        history = deque(maxlen=REP_WIN)
        eos_at = -1
        for i in range(NFRAMES):
            h = hidden[0]
            w32(f"hidden.{i}.f32", h.float().cpu().numpy())
            hist = torch.stack(tuple(history)) if history else None
            torch.manual_seed(SEED + i)        # seeds randperm only; greedy ignores it
            codes = torch.zeros(ncb, dtype=torch.long, device=dev)
            revealed = torch.zeros_like(codes, dtype=torch.bool)
            order = torch.randperm(ncb, device=dev)
            orders = []
            for p, pick in enumerate(order.split(sched, dim=0)):
                logits = dd(h[None], codes[None], ~revealed)[0]
                w32(f"logits.{i}.{p}.f32", logits.float().cpu().numpy())
                orders.append(pick.cpu().tolist())
                for index in pick.tolist():
                    scores = logits[index].float()[None].clone()
                    if hist is not None:
                        # HF RepetitionPenaltyLogitsProcessor, 2-D path: gather the
                        # history's scores, penalty each (score<0 -> *pen, else /pen),
                        # scatter back. Duplicates write identical values, so this is
                        # order-independent and exactly what the processor computes.
                        cb_hist = hist[:, index].reshape(-1).to(scores.device)
                        sc = scores[0]
                        g = sc[cb_hist]
                        sc[cb_hist] = torch.where(g < 0, g * PENALTY, g / PENALTY)
                        scores[0] = sc
                    codes[index] = int(scores[0].argmax())
                revealed.scatter_(0, pick, True)
            wi32(f"order.{i}.i32", np.array(orders, dtype=np.int32))
            wi32(f"codes.{i}.i32", codes.cpu().numpy())
            is_eos = bool((codes == eos_code).any())
            manifest[f"eos.{i}"] = int(is_eos)
            if is_eos:
                codes = torch.full_like(codes, eos_code)
                eos_at = i
            history.append(codes)
            emb = m.get_audio_out_embeds(codes[None])
            w32(f"emb.{i}.f32", emb[0].float().cpu().numpy())
            manifest["nframes"] = i + 1
            if is_eos:
                manifest["eos_at"] = i
                break
            attn = torch.cat((attn, torch.ones((1, 1), dtype=attn.dtype, device=dev)), dim=1)
            # _generate passes ONLY the new token's position at decode (position_ids[:, -1:]
            # + active); full-length ids here made RoPE emit 24 positions for 1 token and
            # blew every downstream shape up 24x (mat1 1x49152).
            o = m.language_model(inputs_embeds=emb[:, None], attention_mask=attn,
                                 position_ids=pos[:, -1:], past_key_values=cache, use_cache=True)
            hidden, cache = o.last_hidden_state[:, -1], o.past_key_values

        # teacher-forced probe: fixed codes, masked all-true / all-false
        os.makedirs(os.path.join(OUT, "tf"), exist_ok=True)
        tf_codes = torch.tensor([[3, 100, 5, 900, 77, 1234, 42, 7, 500, 1, 2000, 999]],
                                device=dev)
        wi32("tf/codes.i32", tf_codes.cpu().numpy())
        h = hidden[0]
        w32("tf/hidden.f32", h.float().cpu().numpy())
        for name, masked in (("masked", torch.ones(1, ncb, dtype=torch.bool, device=dev)),
                             ("revealed", torch.zeros(1, ncb, dtype=torch.bool, device=dev))):
            acts = {}
            hooks = []
            for li, layer in enumerate(dd.model.layers):
                def mk(li):
                    def hook(mod, inp, o):
                        acts[f"layer{li}"] = o[0].detach().float().cpu().numpy()
                    return hook
                hooks.append(layer.register_forward_hook(mk(li)))
            dlin = torch.nn.functional.linear(h[None], dd.depth_linear.weight,
                                              dd.depth_linear.bias)
            w32(f"tf/depth_linear_{name}.f32", dlin[0].float().cpu().numpy())
            logits = dd(h[None], tf_codes, masked)[0]
            for hk in hooks:
                hk.remove()
            w32(f"tf/logits_{name}.f32", logits.float().cpu().numpy())
            meta = {k2: dict(rms=float(np.sqrt((v ** 2).mean())),
                             first4=[float(x) for x in v[0, :4]])
                    for k2, v in acts.items()}
            with open(os.path.join(OUT, f"tf/layers_{name}.json"), "w") as f:
                json.dump(meta, f, indent=1)

    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("oracle written:", manifest)

if __name__ == "__main__":
    main()
