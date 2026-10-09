# PULSAR-ASM

**Inference hardware for every demo below: a Samsung Galaxy Note 10+
(Snapdragon 855, 4 big cores, CPU-only) — except `Ternary-Bonsai-2-27B`
(5.95 GB), which streams 5.9 GB per token and runs on the DGX Spark's 20
Cortex-X925 cores, CPU-only.** The gemma-3 and FunctionGemma panes are the one
carve-out: still the Pi's bytes, because that checkpoint set is no longer on
any reachable machine — their Note10+ re-shoot is pending, not abandoned.

K2-Horizon-0.9B runs on the phone too, through the pure-assembly runtime in
`pulsar_arm/k2horizon/` (greedy streams identical to the scalar C reference,
token for token, logits included):

| checkpoint | hidden | intermediate | layers | bytes | decode (phone) |
|---|---|---|---|---|---|
| `k2h_09_q4` (original) | 1536 | 5120 | 28 | 749 MB | **~62 ms/token** (4-thread decode, short context) |
| `k2h_ft_q4` (meeting-agent FT) | 1536 | 5120 | 28 | 769 MB | same engine; long-context cost in demos |

Two Bonsai checkpoints run on the C + NEON port in `pulsar_arm/bonsai2/`
(greedy streams identical to the reference server, token for token), with
dimensions coming from the GGUF header so one binary covers each family:

| checkpoint | hidden | intermediate | layers | bytes | decode |
|---|---|---|---|---|---|
| `Ternary-Bonsai-2-27B-PTQ1_0` | 5120 | 17408 | 64 | 5.95 GB | **~0.10 s/token** (9.7 tok/s decode, 8 threads, Spark) |
| `Ternary-Bonsai-8B-PQ2_0` | 4096 | 12288 | 36 | 2.18 GB | **~35 ms/token** (28 tok/s, 6 threads Spark; 4.7 tok/s phone) |

Three gemma checkpoints run unmodified on the Pi engine, which reads its
dimensions from the safetensors header, so one binary covers all of them:

| checkpoint | hidden | intermediate | layers | bytes/token | decode |
|---|---|---|---|---|---|
| `google/gemma-3-1b-it-qat-q4_0-unquantized` | 1152 | 6912 | 26 | 2.00 GB | **~543 ms/token** (1.8 tok/s) |
| `google/gemma-3-270m-it-qat-q4_0-unquantized` | 640 | 2048 | 18 | 536 MB | ~147 ms/token (6.8 tok/s) |
| `google/functiongemma-270m-it` | 640 | 2048 | 18 | 536 MB | same as 270m |

## At a glance

| | |
|---|---|
| Target | Samsung Galaxy Note 10+, Snapdragon 855 / 4 big cores, CPU-only — every demo except the 27B; DGX Spark, 20 Cortex-X925 cores, CPU-only — for the 27B (gemma panes still Pi-shot, re-shoot pending) |
| Engine | one static binary: `as` + `ld`, direct syscalls, zero dependencies (K2-Horizon `k2_core.S`, gemma `core.S`); C + NEON + OpenMP in `pulsar_arm/bonsai2/` (Bonsai) |
| Bandwidth | ~3.6–3.9 GB/s of a measured 3.93 GB/s streaming ceiling (92–98 %) |
| Parity | greedy output identical to transformers, token for token (1B bf16, 270m fp32); greedy Bonsai streams identical to the reference server, token for token (27B 17/17, 8B 5/5), plus a full top-20 distribution match |
| Determinism | fixed seed ⇒ byte-identical transcripts across runs |

## Quick start

```sh
# 1. build the engine
cd pulsar_arm/asm
as -o core.o core.S && ld -static -o core core.o

# 2. tokenizer blobs, once (the gemma-3 checkpoints share one tokenizer)
python3 ../tools/mkvocab.py <tokenizer.json> vocab.bin   # surfaces + byte fallback
python3 ../tools/mkbpe.py   <tokenizer.json> bpe.bin     # 514,906 merge rules

# 3. greedy bench: 8-token prompt, 32 decode steps, ids + ms/token
./core gemma-3-270m-it-qat-q4_0.safetensors vocab.bin
./core gemma-3-1b-it-qat-q4_0.safetensors   vocab.bin

# 4. chat REPL (needs bpe.bin)
./core gemma-3-1b-it-qat-q4_0.safetensors vocab.bin 2 c bpe.bin 1000 950 48

# 5. file mode: raw prompt on stdin, greedy, one shot (needs bpe.bin)
./core functiongemma-270m-it.safetensors fcvocab.bin 2 f fcbpe.bin 1000 950 16 < prompt.txt
```

`functiongemma-270m-it` uses its own `fcvocab.bin` / `fcbpe.bin`, built from its
own `tokenizer.json` with the same two tools.

## Bonsai quick start (aarch64 with ARMv8.2+dotprod, e.g. the Spark)

```sh
cd pulsar_arm/bonsai2
LB=<llama.cpp>/build/bin   # fork libs, quantization helpers + oracles only
python3 sign_extract.py Ternary-Bonsai-2-27B-PTQ1_0.gguf  # -> /tmp/sign{5120,6144,17408}.bin
# 27B engine:
gcc -O2 -fopenmp -march=armv8.2-a+dotprod -DTQ_XGEMV_LIB -Dmain=tq_neon_main \
  -c tq_gemv_neon.c -o fwd_neon_mt.o
gcc -O2 -fopenmp -march=armv8.2-a+dotprod -c fwd.c -o fwd.o
gcc -O2 -fopenmp -o fwd fwd.o fwd_neon_mt.o fwht.S \
  -L$LB -lggml-base -lggml-cpu -Wl,-rpath,$LB -lm
# prompt ids from the checkpoint's own template + tokenizer:
./tokdetok tokstr Ternary-Bonsai-2-27B-PTQ1_0.gguf <prompt words...>
# greedy (deterministic, the reference behaviour) then sampled:
OMP_NUM_THREADS=20 ./fwd Ternary-Bonsai-2-27B-PTQ1_0.gguf <ids...> --gen 200
OMP_NUM_THREADS=20 ./fwd Ternary-Bonsai-2-27B-PTQ1_0.gguf <ids...> --gen 200 \
  --sample 0.5 0.85 20 <seed> 0.05
```

The sampler follows the fork chain order (top-k → top-p → min-p on raw
logits, temperature scale last); the server default min-p is 0.05. The 8B
engine builds the same way from `pq2_gemv.c` + `fwd8.c` (no sign tables —
that checkpoint carries no Hadamard transform).

## Modes and flags

```
./core <model.safetensors> <vocab.bin> [tok] [mode] [bpe.bin] [temp_milli] [topp_milli] [gencap]
```

| arg | meaning |
|---|---|
| `tok` | token id for the startup layer-0 signature self-test (the bench prompt itself is built in) |
| `mode` | `0` greedy bench (default) · `1` sampled bench · `c` chat REPL · `f` file mode |
| `bpe.bin` | required by `c` and `f` |
| `temp_milli` | sampling temperature ×1000 (default 1000, min 50) |
| `topp_milli` | top-p ×1000 (default 950) |
| `gencap` | max generated tokens (default 128, max 400) |

Sampling follows the model card: temperature 1.0, **top-k 64**, top-p 0.95
(min-p 0), which is what the sampler implements. Measured guidance beyond that:
keep temp ≥ 0.7 (0.5 degenerates into token loops), top-p is neutral here
(0.5 / 0.95 / 1.0 all stay diverse), and gencap bounds output cleanly. The chat template (SOT/EOT roles) is verified id-exact
against HF, and every run prints its prefill ids as `tpl:` for transparency.

## Demos

Every frame is a real run: prompt, `tpl:` ids and response are the engine's own
bytes, and the status bar carries that session's measured rate. The amber `>`
line is the user turn — in the FunctionGemma panes it is the user message
inside the prompt file, which file mode does not echo, and those panes also
show the system turn verbatim, i.e. where the tool is defined.

Each model gets **one GIF with both language runs stacked**: English above,
Traditional Chinese below; K2-Horizon keeps two GIFs, original then fine-tune.
The panes share a timeline, and only pacing is libertied. All clips share one
font size and one typeface — DejaVu Sans Mono, with WenQuanYi Zen Hei used
only for the CJK glyphs DejaVu lacks, at the same size and line height.

**K2-Horizon-0.9B** — 0.9B dense decoder (IFM, Llama arch), plain RMS norms,
YaRN rope, vocab 64256; shown in its **original** form and as a zh-TW
meeting-agent **fine-tune (FT)**, each bilingual (English over Traditional
Chinese). All four runs are the pure-assembly engine
(`pulsar_arm/k2horizon/k2_core.S` — `as` + `ld -static`, no libc, syscalls
only, glibc-bit-identical `expf`/`%.4f`) on the phone's big cores: greedy,
4-thread decode, 8-token prefill chunks.

*Original* — the ceiling prompt is the two-sentence neural explainer in
English (seasons loops and haiku rambles under greedy, so they don't
qualify), and the four season names in Traditional Chinese — the strongest zh
prompt the base model answers correctly (twelve greedy probes; everything
longer loops, errs, or never answers). 13 tok/s en, 19 tok/s zh; both prompts
cost about a second.

```sh
adb shell "cd /data/local/tmp && taskset f0 ./k2_core k2h_09_q4.blob 64018 2985 ... --gen 400 --threads 4 --batch 8"   # en
adb shell "cd /data/local/tmp && taskset f0 ./k2_core k2h_09_q4.blob 64018 2985 ... --gen 200 --threads 4 --batch 8"   # zh
```

![K2-Horizon original, en over zh-TW](doc/k2horizon-original-en-zh.gif)

*Fine-tune (FT)* — the same 0.9B trained for live meeting reading
(NOTE/REVISE/NEXT), converted from the published int4-QAT safetensors to our
own Q4 blob and run on the same engine, on the same 6-turn window in English
and in Traditional Chinese. Each run shows its full input window verbatim, so
every NOTE can be checked line by line: all five cite genuine timestamps and
the turn stops at NEXT. The ~700-token prefill dominates (67–71 s; decode 3.2
tok/s zh, 4.3 tok/s en over positions ~700–1100). The English window keeps the
source's 萬 figures — with converted millions the model drops to 3/5. Given
the same window the original model deliberates 150 tokens without emitting a
single NOTE — that behavioral gap is what the fine-tune buys.

```sh
adb shell "cd /data/local/tmp && taskset f0 ./k2_core k2h_ft_q4.blob $(cat ft_ids_enB.txt) --gen 400 --threads 4 --batch 8"   # en
adb shell "cd /data/local/tmp && taskset f0 ./k2_core k2h_ft_q4.blob $(cat ft_ids680.txt) --gen 400 --threads 4 --batch 8"    # zh
```

![K2-Horizon fine-tune, en over zh-TW](doc/k2horizon-meeting-en-zh.gif)

Both K2 blobs are bit-identical on Spark to the scalar C reference
(`neural2.txt`), logits included.

**Ternary-Bonsai-8B** — 1.58-bit ternary, Qwen3-8B base, Qwen3 sampling
(temp 0.6, top-p 0.95, top-k 20), 2.18 GB. Native engine
(`pulsar_arm/bonsai2/`) on the same phone — 4 big cores, no GPU: 4.7 tok/s en,
4.0 tok/s zh-TW. The same seeds rerun byte-identical on the Spark, so the clip
is both machines' bytes at once.

![Bonsai 8B chat, en over zh-TW](doc/bonsai8b-chat-en-zh.gif)

```sh
adb shell "cd /data/local/tmp && PQ2_THREADS=4 taskset f0 ./fwd8fresh b8.gguf 151644 872 ... --gen 400 --sample 0.6 0.95 20 7"    # en
adb shell "cd /data/local/tmp && PQ2_THREADS=4 taskset f0 ./fwd8fresh b8.gguf 151644 872 ... --gen 400 --sample 0.6 0.95 20 123"  # zh-TW
```

**Ternary-Bonsai-2-27B** — 1.72 bits/weight end to end, Qwen3.8-27B base, card
config (temp 0.5, top-p 0.85, top-k 20), 5.95 GB PTQ1_0. Native engine on the
DGX Spark CPU-only, 8 threads: 7.9 tok/s en, 7.5 tok/s zh-TW. This is the one
clip that stays off the phone — the 27B streams 5.9 GB per token and needs the
Spark's 20 cores.

![Bonsai 2 27B chat, en over zh-TW](doc/bonsai2-27b-chat-en-zh.gif)

```sh
./fwd_exp Ternary-Bonsai-2-27B-PTQ1_0.gguf 826 279 ... --gen 400 --sample 0.5 0.85 20 7      # en
./fwd_exp Ternary-Bonsai-2-27B-PTQ1_0.gguf 99270 115992 ... --gen 400 --sample 0.5 0.85 20 123  # zh-TW
```

Both Bonsai clips are PULSAR-ASM engine bytes end to end (the predecessor set
ran the PrismML `llama.cpp` fork and is superseded): the 27B reproduces the fork
server token for token at temp 0 (20/20 on the probe prompt), the 8B phone
binary reproduces the Spark engine bit-for-bit including logits, and both rerun
deterministically from the seeded command in the title card. The 1.7B/4B
checkpoints were evaluated and set aside over output quality. The zh-TW panes
ask the same four-seasons question: the 27B answers with a compact table
including its reasoning trace, the 8B with a longer four-section list.

**gemma-3-1b-it** — the most complex prompt the checkpoint answers *correctly*
in each language (a four-item structured list; a three-item one in zh-TW), both
complete and clean:

![1B chat, en over zh-TW](doc/gemma3-1b-chat-en-zh.gif)

```sh
printf 'List the four seasons and one thing that changes in each.\n' | ./core gemma-3-1b-it-qat-q4_0.safetensors vocab.bin 2 c bpe.bin 1000 950 123
printf '請列出保持健康的三個要點。\n'                                | ./core gemma-3-1b-it-qat-q4_0.safetensors vocab.bin 2 c bpe.bin 1000 950 172
```

The gen caps end each run on the model's closing line; the sampler is the
checkpoint's recommended configuration (temperature 1.0, top-k 64, top-p 0.95),
which is what the engine implements.

**gemma-3-270m-it** — same binary, 18 layers:

![270m chat, en over zh-TW](doc/gemma3-270m-chat-en-zh.gif)

```sh
printf 'Explain gravity in two sentences for a child.\n' | ./core gemma-3-270m-it-qat-q4_0.safetensors vocab.bin 2 c bpe.bin 1000 950 48
printf '請用一句話解釋什麼是量子力學。\n'                 | ./core gemma-3-270m-it-qat-q4_0.safetensors vocab.bin 2 c bpe.bin 1000 950 48
```

**FunctionGemma-270m-it** — an English and a Traditional-Chinese question
become the same tool call (one schema; only the user turn differs):

![FunctionGemma tool call, en over zh-TW](doc/functiongemma-toolcall-en-zh.gif)

```sh
python3 pulsar_arm/tools/fc_write.py   # -> /tmp/tool_official.txt, /tmp/tool_official_zhtw.txt
./core functiongemma.safetensors fcvocab.bin 2 f fcbpe.bin 1000 950 16 < /tmp/tool_official.txt
./core functiongemma.safetensors fcvocab.bin 2 f fcbpe.bin 1000 950 16 < /tmp/tool_official_zhtw.txt
```

The gemma and FunctionGemma panes above are still the Pi's bytes (their Note10+
re-shoot is pending the checkpoint files). Re-render any subset with
`python3 pulsar_arm/tools/make_chat_gif.py` (needs PIL + ffmpeg); the
transcripts are literals in that file, so a re-run can only reproduce these
frames, never invent them.

## Verified gates

- **Bonsai greedy parity.** 27B: 17-token stream exact vs the reference
  server at temp-0; 8B: 5/5 exact. Full top-20 distribution match at a
  sampled position (19/20 same ids, same order) — the engines' logits are
  the fork's logits, so sampling draws from the same distribution.
- **Bonsai kernels.** PTQ1_0 decoder, FWHT-1024, full 89M-weight projection
  and both NEON GEMVs bit-exact vs the fork (26.0 ms / 0.29 ns/w PTQ1_0,
  7.3 ms / 0.15 ns/w PQ2_0, single pinned core); GDN and both ropes at
  fp32-vs-op level. See `pulsar_arm/bonsai2/README.md` for the ledger.
- **HF-exact decode.** Greedy output is identical to transformers token for
  token: 32/32 bench tokens for both 270m (fp32) and 1B (bf16), and complete
  chat transcripts — 1B answers *"The capital of France is **Paris**."* exactly
  as HF does. The 270m residual also matches an fp64 oracle to ~1e-7 at every
  layer and every position.
- **One binary, header-driven dims.** `dims hid/inter/layers/vocab: 1152 6912
  26 262144` and `640 2048 18 262144` from the same build — hidden width,
  intermediate width, layer count, vocab, the folded-norm stride, every kernel
  dimension and the full-attention set (`i mod 6 == 5`) all come from the file.
- **BPE encode** 18/18 exact vs HF (char-level init, rank-order merges,
  leftmost tie-break, byte fallback).
- **Determinism.** Greedy, sampled and full chat transcripts reproduce
  bit-for-bit across runs at a fixed seed.
- **Robustness.** Empty / whitespace / CRLF input, 203-token prefill, 8-turn
  depth with automatic context reset, EOF and exit paths, gen-cap boundary.
- **asm at C parity.** GEMV reaches 3.88 GB/s using a single SHLL instruction
  for bf16 widening; the bit-identical wins the C port proved (fused
  elementwise passes, one layer body per call, precomputed tables) are present
  in the asm path too.

## Performance

- **Bonsai wall.** The 27B streams ~5.9 GB/token of ternary weights;
  single-core cost is ~7.6 s/token, OpenMP over GEMV rows brings it to
  ~0.71 s/token on 20 cores (bit-identical tops). The NEON integer kernel
  does 0.29 ns/weight (PTQ1_0) and 0.15 ns/weight (PQ2_0) — 5.3× and more
  over the scalar paths, which is what makes the port practical; the float
  path it replaced ran 153 ms per 89M-weight projection.
- **The wall.** A numpy streaming-sum ceiling of 3.93 GB/s was measured on this
  Pi. The driver moves 536 MB/token for 270m and 2.00 GB/token for 1B, i.e.
  3.6–3.9 GB/s depending on the run — the same bus saturation at both sizes,
  so the floor is ~136 ms/token for 270m and ~510 ms/token for 1B here.
- **Multi-core is not the answer.** A 3-thread static-partition GEMV is 4.2 %
  *slower* (same-job A/B, twice): the bus is already saturated.
- **Thermal drift.** Absolute numbers move 5–18 % between sessions; compare
  same-job A/B ratios, never absolutes.

## Layout

```
pulsar_arm/asm/core.S        the engine: mmap stage → kernels → layer forward
                             → sampler → BPE → chat / bench / file drivers
pulsar_arm/tools/            build-time converters (mkvocab, mkbpe, fc_write),
                             oracles (fwd_ref, l0_any, hf_greedy), the per-layer
                             debug probe patch, demo renderer
pulsar_arm/kernels, runtime  C reference path — parity oracle only, NOT shipped
pulsar_arm/bonsai2/          Bonsai ternary port (C + NEON, CPU-only): fwd/fwd8
                             engines, NEON GEMVs, FWHT asm, GGUF/tokenizer
                             tools, fork-graph oracle, own README + ledger
pulsar_arm/k2horizon/        K2-Horizon-0.9B port: pure-asm runtime (k2_core.S:
                             threaded decode + chunked prefill), Q4 converter,
                             fine-tune conversion, oracle + eval harness
pulsar_arm/tests/            parity tests (Pi-side, need torch + HF cache)
doc/                         demo GIFs and write-ups
```

## Disclosures

- Every demo clip is the in-tree engine's own bytes (no reference runtime):
  transcripts are verified programmatically against the run logs, never
  hand-typed. The K2, Bonsai 8B and Bonsai 27B clips are phone (SD855, big
  cores) and Spark shots; the gemma and FunctionGemma series is still the
  Pi's bytes, because that checkpoint set is no longer on any reachable
  machine — their re-shoot is pending, not abandoned.
- The 27B is a reasoning model: its raw stream opens inside `<think>`;
  clips show the answer content as the server renders it (reasoning kept
  out of frame, same convention).
- The 1B answers multi-item and explanatory prompts (see demos); the 270m
  manages one-line factual answers and degrades beyond that — checkpoint size,
  not the engine, whose greedy path is HF-identical for both.
- The chat path feeds `<bos>` separately on the first turn, so it is not part
  of the printed `tpl:` list; in file mode `<bos>` is literal text in the
  prompt file, so it is.
- Sampling above temp 0.7 stays coherent; greedy is the reference behaviour
  and what all parity claims use.
