# llama.cpp's gemma4 path disagrees with HuggingFace on this checkpoint

Measured on one machine, one set of weights, one prompt. Everything below is
reproducible from `tools/ref_gemma4_hf.py` and the commands at the bottom.

## Setup

| | weights | prompt tokens | decode |
|---|---|---|---|
| HuggingFace `Gemma4ForCausalLM` | the converted blob, bound as zero-copy views | 24 ids from `chat_template.jinja` | greedy |
| this engine (`gemma4_engine_flat.asm`) | the same blob | the same 24 ids | greedy |
| llama.cpp (fork with `llama_model_gemma4`) | `unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-BF16.gguf` | the same 24 ids, verified id-for-id | greedy |

The prompt id sequences were checked to be identical (24 = 24, `identical: True`),
including the single leading `<bos>` — llama.cpp adds its own BOS, so the
template's `<bos>` has to be removed on that side or it gets double-counted.

## First generation step

| token | HF | this engine | llama.cpp |
|---|---|---|---|
| `**` (1018) | **0.572 — rank 1** | 0.79 — rank 1 | – |
| `因為` (27502) | 0.239 — rank 2 | – | – |
| `用` (237105) | 0.128 — rank 3 | 0.019 | **0.913 — rank 1** |
| `這裡` (56674) | 0.0019 — rank 11 | – | – |

HF and this engine pick the same token. llama.cpp puts 7× more probability on a
different one.

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

## Suspected cause on the llama.cpp side

Only a guess, but a specific one. This build logs

```
resolve_fused_ops: Flash Attention not supported, set to disabled
load_model: initializing, n_slots = 4, n_ctx_slot = 1024, kv_unified = 'true'
```

Gemma 4 has an unusual cache layout: one KV head per layer, `kv_store_layers`
publishing rows from layers 13/14 that layers 15..34 read instead of allocating
their own, and a 512-token sliding window on 4 of every 5 layers. A unified cache
plus per-layer shared rows is exactly the kind of thing that silently reads the
wrong rows, and the per-layer `layer_scalar` on the residual stream would amplify
that into a different distribution rather than a slightly different one — which
is the shape of what is observed: not noise, a confident 0.913 on the wrong token.

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
