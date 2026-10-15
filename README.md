# PULSAR-ASM: Gemma 4 E2B on a Jetson Nano

Google's **Gemma 4 E2B** (text only, the QAT Q4_0 release) running at **6.2 to 6.8 tokens/s
decode on a Jetson Nano Gen1**, the 2019 4 GB board, with its 128-core Maxwell GPU. The host
side is C with the hot loops in hand-written AArch64 assembly; the GEMVs and the MLP chain run
on the GPU through kernels driven directly by the CUDA driver API.

| | |
|---|---|
| Model | `google/gemma-4-E2B-it-qat-q4_0-gguf`, file `gemma-4-E2B_q4_0-it.gguf`, 3,349,516,256 bytes, sha256 `fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634` |
| Format | Google's own: layer projections Q4_0 (quantization-aware trained), token embedding / tied head and per-layer-input table Q6_K, per-layer projection F16 |
| Engine | C + AArch64 assembly host (`pulsar_arm/gemma4nano/g4eng.c`, `g4_host.S`), GPU kernels as embedded PTX JIT-loaded by the driver. No CUDA toolkit, cuBLAS, llama.cpp or Python at runtime; links libc, libm, libdl and libgomp |
| Accuracy | every kernel, CPU and GPU, bit-identical to the engine's own scalar reference (`g4ref.c`); greedy streams frozen as goldens and checked by `gates.sh` |

## Demo

![Gemma 4 E2B on a Jetson Nano, English over Traditional Chinese](doc/gemma4-e2b-jetson-nano-en-zh.gif)

Both runs are greedy and use **ceiling prompts**: the hardest of 16 probes the model answers
correctly and completely (`pulsar_arm/gemma4nano/CEILING_NANO.md`):

- **English:** "A shirt costs $25 after a 20% discount. What was the original price? Show your
  work." It answers $31.25 and checks it.
- **Traditional Chinese:** 「一列火車下午3點15分出發，車程2小時50分鐘，幾點抵達？」 It answers
  下午6點5分.

The probes also show where the model fails, and those failures are part of the record: the same
train problem asked in English gets 7:05 pm (it adds the carried hour twice), the same discount
problem asked in Chinese starts with 2500 and corrects itself only later, the haiku breaks the
5-7-5 form, and the French translation contains wrong forms.

How the clip was made: the committed engine on the Nano; both runs rerun byte-identical
(`pulsar_arm/gemma4nano/probes/*_steps.txt`); the text is the detokenized output up to the
end-of-turn token, unedited (the English answer's LaTeX is shown raw, as generated). The status
line shows the engine's own decode rate (generated steps over their wall time). Frames are
rendered at a fixed cadence, not in real time. Re-render with
`python3 -I pulsar_arm/tools/make_chat_gif.py g4-nano` (Pillow and ffmpeg).

## At a glance

| | |
|---|---|
| Board | Jetson Nano Gen1: Tegra X1, 4x Cortex-A57, one Maxwell SM (sm_53), 4 GB LPDDR4 shared by CPU and GPU. L4T R32.5.1, CUDA 10.2 driver |
| Decode | **6.83 tok/s** on the English gate prompt (76 steps); 6.17 and 6.56 tok/s on the demo prompts (291 and 117 steps); 5.5-6.0 tok/s at 500 steps (attention grows with context) |
| Prefill | 6.5-7.2 tok/s, token by token through the decode path |
| Load | 4.0-6.2 s (weights repacked and uploaded) |
| Memory | 1,474 MB on the GPU, peak RSS 729 MB; the 1.93 GB per-layer-input table stays memory-mapped, one 7.4 KB row read per token |
| Context | 2048 positions; KV cache only on the 15 layers that own one (the other 20 share) |
| Clocks | not pinned (no root for `jetson_clocks`): the GPU sits at 921.6 MHz under load, the CPU governor is `schedutil`, so CPU-side timings vary by a few ms between runs |

## How it works

A token goes through 35 layers. Per layer:

- **GPU** (driver API: one non-blocking stream, a timing-free event polled by the host,
  inputs and outputs in mapped pinned memory, about 5 synchronizations per layer):
  - the merged q+k+v GEMV, the o projection, the per-layer-input gate and projection;
  - the **whole MLP chain in one go**: merged gate+up GEMV, GeGLU, Q8_K quantization, down
    GEMV, with no CPU round trip in between;
  - after the last layer, the 262,144-row Q6_K tied head.
- **CPU, AArch64 NEON assembly** (`g4_host.S`): RMS norms, attention for all 8 query heads in
  one pass over the single KV head (8-row transposed score tiles, p·V with 16 accumulators),
  softmax with a two-lane double-precision `exp`, Q8_K quantization. RoPE, the per-layer-input
  mixing and argmax are C.

The GPU kernels:

| kernel | what | speed (warm GPU) |
|---|---|---|
| `q4i_gemv4` | Q4_0 GEMV, one row per lane, weights tile-interleaved so a warp reads 512 contiguous bytes, int16 activations in shared memory, 4-block load ring | 12.6-15.4 GB/s, 220 of 275 tensors |
| `q4_gemv` (from the K2 engine) | Q4_0 GEMV for the shapes where it is faster (down at 12288 columns, the per-layer-input gate) | 10.5-11.3 GB/s, 55 tensors |
| `q6i_gemv` | Q6_K head, one row per lane, four 6-bit values decoded per word | 18.0 ms per token, 19.6 GB/s |
| `geglu_dev`, `q8k_dev` | GeGLU with the same double-precision `exp`, Q8_K quantization | part of the MLP chain |
| `f16_gemv_t` | per-layer-input projection, weights transposed | 3.7 ms per token |

Where a token's ~146 ms go (English gate prompt): Q4_0 GEMVs and the MLP chain ~103 ms, head
~19 ms, F16 projection ~4 ms, CPU stages ~20 ms.

### Speed history (English gate prompt, decode)

| step | tok/s |
|---|---|
| first light: shared forward, one synchronous copy + launch + sync per GEMV, CPU stages in C | 3.23 |
| merged q+k+v and gate+up launches | 3.49 |
| K2's driver path (stream, timing-free event, mapped pinned I/O, copy kernel) | 3.97 |
| NEON assembly: exp, GeGLU, attention scores and p·V | 4.44 |
| MLP chain on the GPU | 4.98 |
| attention for all 8 heads in one pass; NEON RMS norm and Q8_K | 5.1-5.3 |
| `q4i` row-per-lane Q4_0 kernel | 5.7-5.9 |
| `q6i` row-per-lane head (41.7 -> 18.0 ms) | 6.8-6.9 |

Tried and dropped (details in `pulsar_arm/gemma4nano/PORT_PLAN.md`): a head kernel built on
16-bit multiplies (slower, the compiler kept the byte extracts), deeper load rings for `q4i`
(register pressure cost more than it gained).

## Accuracy

`g4ref.c` is the scalar reference that **defines** the engine's arithmetic: activations
quantized to Q8_K, exact integer dots with fixed-order float accumulation, float64 norms,
softmax and GeGLU through its own `exp` (only IEEE `+ - * /` and exponent assembly, within 1
ulp of libm), baked rope tables, greedy argmax on the raw logits. The engine runs that same
forward pass (shared code) with its GEMVs and hot loops swapped for GPU kernels and assembly,
each proved bit-identical:

- `g4host_test`: 16.8 M `exp` values, 4.2 M GeGLU values, attention at 12 window
  configurations and both head sizes, 3.07 M RMS-norm outputs, 51,500 Q8_K blocks, all identical.
- `g4gemv_test`: all 275 Q4_0 tensors, the full 262,144-row head and the F16 projection on three
  activation vectors each, plus the GPU GeGLU and Q8_K kernels, all identical.
- `gates.sh` runs both tests and checks the engine's English (97 steps) and Traditional Chinese
  (222 steps) greedy streams against `golden_en.txt` / `golden_zh.txt`.

**What "bit-exact" does not cover.** It is exact against `g4ref.c`, not against Hugging Face or
llama.cpp. Quantizing activations to Q8_K changes the logits slightly: against a float reference
on the same dequantized weights (`g4_np.py`, which mirrors the x86 port's HF-checked reference),
the English answer is token-for-token identical, while the Chinese one is identical up to
generated token 32, then takes a different but equally correct path. The contract also excludes
non-finite inputs (NaN payloads differ between ARM and the GPU); GEMV outputs are always finite.

Before anything was built, `g4_q4check.py` checked the source: all 275 Q4_0 tensors in Google's
GGUF are byte-identical to ggml's Q4_0 quantization of the QAT bf16 checkpoint, so the GGUF is
exactly "the QAT weights in Q4_0".

## Limits

- **Greedy only.** The engine has no sampler yet; the model card recommends temperature 1.0,
  top-k 64, top-p 0.95.
- One sequence; 2048-position context; prefill one token at a time.
- Measured on an unpinned board: absolute numbers move a few percent between runs.
- The model's own ceiling: multi-step arithmetic is unreliable (see the probes), long answers
  and code often exceed 500 tokens.

## Build and run

On the Nano (gcc 7.5 and the L4T driver; no toolkit needed):

```sh
cd pulsar_arm/gemma4nano
sh build.sh                                              # g4eng, g4gemv_test, g4host_test
python3 -I g4_rope.py g4_rope_2048.bin 2048              # rope tables (any machine with numpy)
sh gates.sh gemma-4-E2B_q4_0-it.gguf g4_rope_2048.bin    # all accuracy gates
./g4eng gemma-4-E2B_q4_0-it.gguf g4_rope_2048.bin <prompt ids...> --gen N
# prints: step <n> id_in=<id> top=<id> logit=<x.xxxx>   (logit after the 30*tanh(x/30) cap)
```

Prompt ids come from the model's `tokenizer.json` with the chat template
`<bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n` (`pulsar_arm/gemma4nano/prompts.json` has two
examples). Switches for A/B: `G4FAST=0` (plain synchronous copies), `G4Q4I=0` (K2 kernel only),
`G4GPUMLP=0` (MLP stages on the CPU), `G4ASM=0` (C host loops), `G4PROF=1` (per-hook timing).
`g4_kernels.ptx` is regenerated from `g4_kernels.cu` with
`nvcc -arch=compute_53 -ptx -fmad=false -O3` and embedded by `g4_embed_ptx.py`.

## Layout

```
pulsar_arm/gemma4nano/g4ref.c         the reference: GGUF loader, arithmetic contract, forward pass, hooks
pulsar_arm/gemma4nano/g4eng.c         the engine: GPU upload and kernels behind the hooks, driver path
pulsar_arm/gemma4nano/g4_host.S       AArch64 NEON host kernels; g4_hostglue.h is their C side
pulsar_arm/gemma4nano/g4_kernels.cu   GPU kernels (q4i, q6i, GeGLU, Q8_K, F16); g4_ptx_q4.h is K2's kernel
pulsar_arm/gemma4nano/g4*_test.c      bit-exactness tests; gates.sh, build.sh
pulsar_arm/gemma4nano/*.py            GGUF reader, Q4_0 cross-check, dequantizer, float reference, rope tables
pulsar_arm/gemma4nano/PORT_PLAN.md    the port's log: milestones, measurements, what was dropped and why
doc/K2_HORIZON_JETSON_NANO.md         the K2-Horizon-0.9B Nano engine this port builds on
doc/OTHER_ENGINES.md                  the earlier engines (Pi 4 Gemma-3, Bonsai, phone)
```
