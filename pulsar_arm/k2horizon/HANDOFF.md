# K2-Horizon-0.9B engine port — arch map + plan

Branch `k2-horizon-aarch64` @ base `fecc563` (bonsai-aarch64 tip).
Goal (user): port IFM/K2-Horizon-0.9B to the native engine; quantize the
base to Q4 as the baseline; quality-test vs the fine-tuned
Luigi/K2-Horizon-0.9B-meeting-agent-zh-LiteRT (zh-TW meeting summarizer).

## Source models

- Base: `IFM/K2-Horizon-0.9B` — custom `k2_horizon` arch, safetensors only
  (single 2.16 GB file, bf16-class), Apache-2.0. NO GGUF exists for 0.9B
  (only 7B+ have community/official GGUFs). Tags: reasoning, KD, en+zh.
- Fine-tune: `Luigi/K2-Horizon-0.9B-meeting-agent-zh-LiteRT` — LiteRT-LM
  container (`-q4` + `-q8` `.litertlm`), zh-TW meeting summarization, with
  `system_prompt.txt`. NOT runnable in our engine (TFLite container, and
  extracting weights means reverse-engineering it — out of scope). It is
  the quality reference, run in its own runtime.

## Geometry (config.json, verified against modeling_k2_horizon.py)

- hidden 1536, 28 layers, 32 q-heads / 8 kv-heads (GQA group 4),
  head_dim 64, inter 5120, vocab 64256, 131072 ctx (YaRN x16 from 8192).
- Dense: num_experts 0 (mlp_only_layers lists all 28, but the MoE branch
  needs num_experts>0, so every layer is plain MLP).
- No QK norms, no attention gate, no sliding window, no biases anywhere
  (attention_bias false, MLP/lm_head bias-free), untied lm_head.
- Norms are T5-style plain RMS (`weight * x/rms`, groups=1, eps 1e-6) —
  NO 1+w fold, unlike gemma3. Layer pattern: standard pre-norm +
  residuals (input_layernorm → attn → add → post_attention_layernorm →
  SwiGLU MLP → add), final norm + head (confirm `model.norm` name from
  the weight map at convert time).
- Attention scale 1/sqrt(64) = 0.125, full causal (no window), softmax in
  fp32, per-layer KV cache.
- RoPE: NeoX-interleaved pairs, rope_head_dim == head_dim so the SIMPLE
  path (no interleave dance). YaRN-scaled inv_freq (factor 16, beta
  128/4, theta from rope_parameters) × attention_scaling multiplier,
  computed in fp32. PLAN: bake cos/sin tables in fp32 with numpy during
  conversion (replicate HF exactly once, never reimplement YaRN in C) —
  same trick as the 8B YaRN work.
- Tokenizer: custom IFM (`tokenizer.json` + manifest); bos 0, eos 1,
  pad 64255. Need a `tools/k2_ids.py` (qwen_ids.py precedent) for raw ids.
  Chat template is tool-call heavy; meeting eval uses the fine-tune's
  `system_prompt.txt` + plain turns.

## Engine plan (staged, Bonsai-style)

1. **Oracle**: transformers forward on a fixture prompt (few tokens only).
   Spark has a GB10 but no torch yet — try `pip install torch (cu12x) +
   transformers`; fall back to CPU oracle (0.9B bf16, minutes per fixture,
   acceptable one-time). llama.cpp CANNOT help (no k2_horizon support in
   the fork or upstream — checked).
2. **Convert**: safetensors → engine blob (own minimal format, bonsai_b3
   precedent: header + raw fp16/bf16 tensors + baked fp32 rope tables).
   Do NOT fight llama.cpp arch support.
3. **Forward**: new `pulsar_arm/k2horizon/fwd_k2.c` (start fp32 compute,
   fp16 weights, single-token loop exactly like fwd8.c: embed, 28×
   (rms, qkv GEMV, rope, GQA attn, o, add, rms, gate/up, silu-mul, down,
   add), final norm, head GEMV, sampler). GEMV path: reuse the Q8/neon
   pattern from pq2_gemv.c (quantize activations, integer dots) once the
   weight layout is chosen; start with fp16×fp32 reference for the
   oracle diff, then kernelize.
4. **Q4 baseline**: our own Q4_0-style quant in the converter + Q4 NEON
   kernel (pq2 is the template: 16-wide nibble dots, exactness by proof).
   Quality delta fp16-vs-Q4 on meeting prompts; speed on Spark + phone.
5. **Quality test**: meeting-summarization zh-TW prompts through our Q4
   engine vs the LiteRT fine-tune in its runtime; compare (human read,
   this is a quality gate not a metric gate).

## Per-token arithmetic (phone preview)

Weights ≈ 0.9B → fp16 blob ~1.8 GB, Q4 ~0.5 GB. Q4 decode on SD855 should
land well under 100 ms/tok (8B PQ2_0 does 192 ms at 4× the weights).

## Lessons carried in

- Bit-exact discipline: goldens + fork/oracle gates, no -ffast-math ever.
- Cross-machine claims need LOGITS (tops-only hid the stale-object split).
- Same-job benchmark discipline, thermal cooldowns on the phone.
- Static glibc binary runs on Android; pthreads pool + pure spin.
