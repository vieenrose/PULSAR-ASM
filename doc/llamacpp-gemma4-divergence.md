# llama.cpp's gemma4 path disagrees with HuggingFace on this checkpoint

Measured on one machine, one set of weights, one prompt. Everything below is
reproducible from `tools/ref_gemma4_hf.py` and the commands at the bottom.

## Setup

| | weights | prompt tokens | decode |
|---|---|---|---|
| HuggingFace `Gemma4ForCausalLM` | the converted blob, bound as zero-copy views | 24 ids from `chat_template.jinja` | greedy |
| this engine (`gemma4_engine_flat.asm`) | the same blob | the same 24 ids | greedy |
| llama.cpp (`llama_model_gemma4`) | `unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-BF16.gguf` | the same 24 ids, verified id-for-id | greedy |

The divergence was first measured against a fork build and has since been
reproduced byte-for-byte against upstream `llama.cpp` (v0.3.0-dev) with a
fresh download of the same file: first token `用` (237105) at p≈0.913, same
trajectory start. It is not a stale-file artifact.

The prompt id sequences were checked to be identical (24 = 24, `identical: True`),
including the single leading `<bos>` — llama.cpp adds its own BOS, so the
template's `<bos>` has to be removed on that side or it gets double-counted.

## First generation step

| token | HF | this engine | llama.cpp |
|---|---|---|---|
| `**` (1018) | **0.572 — rank 1** | 0.592 — rank 1 | – |
| `因為` (27502) | 0.239 — rank 2 | 0.221 — rank 2 | – |
| `用` (237105) | 0.128 — rank 3 | 0.128 — rank 3 | **0.913 — rank 1** |
| `這裡` (56674) | 0.0019 — rank 11 | – | – |

HF and this engine pick the same token. llama.cpp puts 7× more probability on a
different one. (The engine column is post-`final_logit_softcapping` fix; before
it the engine read 0.79/0.019 on rows 1/3 because it skipped the cap. The
llama.cpp column is unchanged by that fix.)

Control prompt `Name one city in France.` (15 ids): all three agree — `Paris`
(50429) at p≈0.9999 on every side, same top-6 order. So the divergence only
shows where the race is close; a 0.9999 lead survives any implementation's
rounding, a 0.59/0.22/0.13 split does not.

## Trajectory

```
HF              : 1018 27502 104492 146569 236918 64142 236900 30094 125168 125716 237070 | 22841
this engine     : 1018 27502 104492 146569 236918 64142 236900 30094 125168 125716 237070 | 44684
llama.cpp greedy: 237105 104492 ...   ('用組合語言寫語言模型很奇怪，因為它...')
```

HF and this engine are identical for 11 tokens and part ways at the 12th, where
the two candidates are close enough that bf16 rounding decides. llama.cpp takes a
different branch immediately.

## Why this engine trusts HF here

- The blob was verified against the checkpoint: 372 tensors, 356 compared
  exhaustively, values cross-checked (`layer_scalar` minimum 0.018 vs HF 0.01782).
- `tools/ref_gemma4_hf.py` does not use this repo's NumPy reference at all. It
  loads `transformers.models.gemma4.modeling_gemma4` and points it at the blob,
  so a shared misreading of the architecture cannot hide a disagreement.
- The engine is also bit-identical to that reference across an 8-token rollout
  (`tests/test_gemma4_engine.py`), and matches HF's trajectory above.

## Cause: knife-edge attention turns last-bit arithmetic into a token flip

The first theory written here — a broken shared-KV cache — is wrong and is
kept out of the record deliberately: llama.cpp's `reuse` callback maps shared
sliding layers to layer 13 and shared full layers to layer 14, which is exactly
what HF's `shared_kv_states[layer_type]` holds (full layers sit at 4, 9, 14,
…). A second theory raised during the investigation — a missing
`layer_scalar` in the GGUF — is also wrong: the tensor is present as
`blk.N.layer_output_scale.weight` (×35), and spot values are bit-identical to
the blob (as are `attn_norm`, `attn_q_norm`, `attn_k_norm`, full-layer norms,
and the `w_q` orientation). Both corrections are recorded because each looked
like a smoking gun until it was checked.

Everything else was checked the same way, against HF's `modeling_gemma4.py`
as ground truth, and cleared: attention scale 1.0 on all sides; q/k/v norms
present and same-ordered; RoPE pairing (`NEOX` stride `n/2` = HF's
`rotate_half`) and full-layer frequencies (`1e6^(-2i/512)` + zero-identity
tail, matching proportional `partial_rotary_factor=0.25`); PLE scales and
order; final `30·tanh` cap; GGUF metadata (rope bases 1e6/1e4, window 512,
pattern 4+1, 20 shared layers, sampling defaults); tokenization id-for-id
(23 + server-added BOS = the same 24); sampler penalties off; no BOS
 double-count (`tokens_evaluated: 24`).

What remains is not a wrong tensor but a property of the checkpoint: with
`self.scaling = 1.0` and head_dim 256/512, attention is nearly one-hot, and
several heads sit exactly on ties. Measured over HF itself (eager attentions
on the zh prompt, last position), the smallest top-1/top-2 gaps include
0.0000 (layers 6, 25), 0.0020, and several at 0.003–0.005. Any
implementation's last-bit arithmetic — torch bf16 vs ggml bf16 kernels differ
in blocking and FMA order well within spec — can reroute those heads to a
different source position, and a rerouted head moves the hidden state by whole
vectors, shifting logits by whole points. That is the amplifier that turns
sub-bf16-noise into a 0.59 → 0.91 flip on a close race, while leaving a
0.9999 certainty (Paris) untouched.

So llama.cpp is not running a different model, and its fluent Chinese is not
evidence of better fidelity: it is the same weights and ids, with tied
attention heads falling the other way. Which engine's rounding "wins" a
0.59/0.22/0.13 race says nothing about quality — the checkpoint's own greedy
continuation (`** 因為…預意…錯乎…`, HF-verified) is what the weights define,
awkward characters included.

Two honest caveats. First, the flip was never reproduced by perturbing this
repo's reference with synthetic noise, so the mechanism is inferred from
elimination plus the measured ties, not from a replay. Second, one real
latent bug did surface on llama.cpp's side along the way: shared sliding
layers read layer 13's *windowed* SWA cache, while HF keeps full-length
`shared_kv_states` for them — identical inside 512 tokens, divergent past it.
It does not explain step 1 here, but it will matter for long generations.

The practical lesson for anyone comparing output quality between engines: check
the first-step distribution, not the text. Fluent text can come from a
computation that does not match the checkpoint's reference semantics, and it will
look convincing.

## Reproducing

```bash
# HF over the blob, and the engine, same prompt
python3 tools/ref_gemma4_hf.py --compare

# llama.cpp, CPU only
llama-server -m gemma-4-E2B-it-BF16.gguf -ngl 0 -t 4 -c 1024 --port 8083
curl -s localhost:8083/tokenize  -d '{"content":"<rendered template>","add_special":false}'
curl -s localhost:8083/completion -d '{"prompt":"<template minus <bos>>","n_predict":24,"temp":0,\
"logprobs":true,"n_logprobs":8}'
```

The `<bos>` removal matters: with the template's `<bos>` kept, llama.cpp evaluates
25 tokens and silently runs a double BOS.
