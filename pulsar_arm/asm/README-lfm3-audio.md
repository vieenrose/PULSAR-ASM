# LFM3-Audio-2B on a Jetson Nano — the port

Pure-CUDA (`sm_53`, CUDA 10.2) inference path for **LFM3-Audio-2B** on a Jetson Nano Gen1
(Tegra X1: 4× Cortex-A57 + 4 SM Maxwell, 4 GB unified). One translation unit,
[`lfm3_nanode.cu`](lfm3_nanode.cu), built by `nvcc` and nothing else; weights come from a
`.lfm3q8` blob. Alongside it: AArch64 NEON kernels for the non-GPU side
([`lfm3_conv.S`](lfm3_conv.S), [`lfm3_elem.S`](lfm3_elem.S)), and a Python gate suite in
`.auto/lfm3/`.

**This engine ports the text backbone only.** It takes token ids and returns token ids:
everything left of the tokenizer (log-mel, the 17-layer audio encoder, the projector) and
everything right of the backbone (MaskGIT depth head, LiquidCodec) is **not ported yet**.
The port status, the contract each new component has to satisfy, and the measured facts a
new component inherits are what this document is about.

```
ported and gated      text backbone (LFM3, 24 mixed conv/attention layers, "lfm." 286 tensors)
                      112.8 ms/token decode, 3 gates green, at the board's bandwidth ceiling
not started           log-mel front end, audio encoder (17 layers), depthformer head,
                      LiquidCodec decoder, tokenizer, interleaved text/audio controller
```

---

## 1. Port status, component by component

Tensor counts and bytes below are **measured from the checkpoint header and the exported blob**
(`.lfm3q8` table), not estimated — except where marked *derived*.

| component | checkpoint prefix | tensors | Q8 bytes | engine | gates | status |
|---|---|---|---|---|---|---|
| text backbone (LFM3, 24 layers) | `lfm.` | 286 | **1946 MB** | `lfm3_nanode.cu` | 4 green | **ported** |
| audio encoder (LFM2, 17 layers) | `encoder.` | 170 | ~119 MB *derived* | — | — | not started |
| downsample front end (2-D convs) | `encoder.downsample.` | — | ~4 MB *derived* | — | — | not started |
| audio→backbone projector | `mlp.` | 6 | ~7 MB *derived* | — | — | not started |
| MaskGIT speech head | `depthformer.` | 37 | ~29 MB *derived* | — | — | not started |
| per-codebook logits | `depth_embeddings.` | 36 | ~56 MB *derived* | — | — | not started |
| backbone → 12 depth streams | `depth_linear.` | 2 | ~28 MB *derived* | — | — | not started |
| audio codebook table | `codebook_embedding.` | 1 | ~56 MB *derived* | — | — | not started |
| MaskGIT mask token | `mask_embedding.` | 1 | 4 KB | — | — | not started |
| LiquidCodec decoder | *(separate checkpoint)* | — | 147 MB | — | — | not started |
| log-mel front end | *(no weights)* | — | — | — | — | not started |
| `liquid7.1` BPE tokenizer | *(18 MB json)* | — | — | — | — | not started |

The blob carries the backbone and nothing else: the exporter filters on the `lfm.` prefix, and
the audio-side prefixes above are simply not in it. A ported component is therefore *always*
visible as (a) new tensors in the blob, (b) a `T(M, "<hf name>")` lookup in the engine, and
(c) a gate that scores it. Nothing is silently half-loaded — the loader prints its census
(``layer types: 11 attention, 13 conv``, arena accounting) and fails loudly with
``MISSING TENSOR <name>`` on any lookup that does not resolve.

## 2. The port contract: checkpoint → blob → engine

### 2.1 Blob format (`.lfm3q8`)

```
8 B    magic          "LFM3Q8\x01\x00"
u32    header length
json   header:  { magic, cfg, n_tensors, vocab_size, hidden_size, data_bytes,
                  table: [ {name, dtype, ne[], off, ncodes, nscales, crc}, ... ] }
bytes  payload, in table order, each entry at `off`
```

`dtype` is the only thing the engine switches on:

| code | payload | used for |
|---|---|---|
| 1 | Q8 codes (`H×K` int8) + fp32 scales, one per 32-weight block (**1.125 B/weight**) | every 2-D linear that clears the gates |
| 2 | fp32 verbatim | norms, 3-D conv taps, anything the 2-D quantizer cannot see |
| 3 | bf16 verbatim | the tied embedding in the `--emb-bf16` arm |

`cfg` is the backbone's own `config.json["lfm"]` verbatim — the engine reads `layer_types`,
`sliding_window`, `conv_L_cache`, `head_dim`, `norm_eps`, `use_sandwich_norm`,
`rope_parameters`, `tie_word_embeddings` out of it rather than hard-coding them. **This is what
makes a port mechanical: architecture facts travel in the blob, not in the C++.** A new
component exports its sub-config the same way.

`name` is the **HF tensor name verbatim** (`lfm.model.layers.7.feed_forward.w1.weight`). The
engine has no renaming table; it looks up `M.by[name]`. A new component keeps its own prefix
and lands in the same dictionary untouched.

### 2.2 What a new component must add to the exporter

`pulsar_asm/lfm3/tools/export_lfm3_q8.py` is 214 lines and the pattern is small:

1. a **prefix filter** for the component's tensors (`BACKBONE = "lfm."` today);
2. the **Q8 pattern list** — substrings that select 2-D linears for quantization;
3. the **exceptions**: 3-D tensors (`conv.conv.weight` is `[H,1,7]`) and 1-D norms go F32;
4. **assertions, not hopes** — every Q8 pattern must match ≥1 tensor (`a filter matching ZERO
   tensors silently reports the bf16 result` has cost this repo a debugging cycle twice), and
   the conv-tap count must equal the layer census from `layer_types`.

The engine side then needs: the tensor in the arena accounting (`need_q/need_s/need_f`, asserted
at load — a mismatch prints `arena accounting mismatch` and refuses to run), a dispatch site,
and shape coverage in the GEMV config picker.

### 2.3 Shape coverage is asserted, not assumed

`GemvCfg { KCHUNK, TPB, R }` is chosen per shape and **checked**: an earlier bug computed
`nchunk = 7680/2048 = 3` and read 80% of K — and the mirrored reference agreed with it, so the
bit-exactness printout said YES. Two lessons carried in the code: every shape a new component
introduces must go through the config picker with an explicit coverage assert, and a
bit-exactness gate is only as good as the reference it mirrors.

## 3. The backbone port (done) — the worked example

### 3.1 Architecture facts, as the engine sees them

Read from the blob's `cfg` and tensor list (all measured):

```
hidden 2048   intermediate 7680   vocab 128000 (tied)   24 layers
16 heads / 4 KV heads, head_dim 128      norm_eps 1e-5   use_sandwich_norm true
conv_L_cache 7 (depthwise, causal)       sliding_window 256
rope   full_attention:    partial_rotary_factor 0.5, theta 1e6
       sliding_attention: theta 1e4, no partial factor -> FULL head dim rotation
layer_types = conv, conv, full, conv, sliding, conv, full, ... (13 conv, 6 full, 5 sliding)
```

**Two layer templates**, and the difference matters for byte accounting:

| template | tensors | per layer |
|---|---|---|
| `conv` (13 layers) | `conv.{in_proj [6144,2048], out_proj [2048,2048], conv [2048,1,7]}` + FFN + 4 norms | **72.1 MB** |
| `full`/`sliding` (11 layers) | `self_attn.{q [2048,2048], k [512,2048], v [512,2048], o [2048,2048], q_norm[128], k_norm[128], gate_proj[16,2048]}` + FFN + 4 norms | **65.0 MB** |

13 × 72.1 + 11 × 65.0 = **1651 MB** of layers, + the tied embedding/lm_head **295 MB** =
**1946 MB streamed per decode step** (every linear is touched once; the tied head is a full
128 000-row pass). At the measured 112.8 ms/step that is **17.3 GB/s**, i.e. at the ceiling that
`tools/stream_ceiling.cu` measures for a pure read stream (17 GB/s; one GM20B SM, grid-stride
uint4) — the achieved rate sits at, marginally above, that probe, so the backbone is
bandwidth-saturated. Earlier revisions of this file quoted "14 GB/s against a 17 GB/s ceiling" from a
per-phase table whose byte counts assumed a 6144-wide FFN — the checkpoint's is 7680, so the
per-phase bytes understated the stream and the engine is in fact already saturated. Arithmetic levers are
therefore capped at ~2% on this port (measured: doubling GEMV FLOPs with `LFM_NOFMA=1` costs
+1.8%, i.e. the kernels are ~90% DRAM-bound). **Any new component's runtime is its bytes divided
by 11–17 GB/s — budget in bytes, not in FLOPs.**

### 3.2 The five things that bite when mapping LFM3 to kernels

1. **`conv` in `layer_types` means there is no `self_attn` at all** in those layers — 13 of 24
   layers have no attention, no RoPE, no KV cache. A port that builds one uniform "transformer
   block" wastes a third of the model's structure.
2. **Sliding layers rotate the whole head dim; full layers rotate half.** `partial_rotary_factor
   0.5` lives *inside* the `full_attention` rope params; sliding layers have no factor and must
   use `rot = head_dim`. Getting this wrong is invisible at position 0, stays under a 3% gate to
   position ~22, and then grows into a position-proportional phase error — it cost this port a
   full debugging session (#158: 243/7200 rms cells outside 3% before, 1/7200 after).
3. **`use_sandwich_norm` = four norms per layer**, applied around both the operator and the FFN
   (`operator_norm`, `post_operator_norm`, `ffn_norm`, `post_ffn_norm`) — 96 norm launches per
   token, not 48.
4. **The per-head gate** is `self_attn.gate_proj [16, 2048]` → sigmoid multiplied into the
   attention output *before* `o_proj`. It is a `[16]`-wide vector per token, not a tensor-wide
   gate.
5. **The tied lm_head is not optional and not cheap**: 295 MB, 15% of every decode step. (It is
   skipped only on prompt positions 0..n-2, where nothing consumes it.)

### 3.3 Numerics contract (carried from the HF reference)

The model runs in bf16 and the goldens come from it, so **activations are stored bf16 and
arithmetic happens in fp32 with an explicit bf16 round at every op boundary**. An fp32 engine is
*more* accurate than its reference and can never match it — it looks exactly like a math bug
(cos 0.955 on a 7680-term dot). Widening and narrowing are exact bit operations (shift by 16,
and round-to-nearest-even on the way down).

The GEMV accumulates with `__fmaf_rn` (one rounding per product) — ggml's own reference dot
product is `sum += w*x`, which compilers contract to an FMA, so this is *more* faithful to the
reference than the exact-two-rounding form. It is also 2.0% faster. Build flag `-fmad=false`
holds everywhere else.

The porting rule this implies: **reproduce the reference's rounding points first, then optimize
arithmetic only where the gate says you may.** Two levers here are bit-identical
(`LFM_NOFUSE`, `LFM_R=64`) and double as the gate's own determinism control.

## 4. Audio encoder — the next port

### 4.1 Facts (from the checkpoint's `encoder` sub-config and safetensors headers)

```
17 LFM2 layers, hidden 768, intermediate 1792 (2560 auto-adjusted down by block_multiple_of 256)
8 heads / 4 KV heads, head_dim 96   sliding_window 32   conv_L_cache 3
10 conv layers + 7 attention layers        ~106.2 M params
downsample front end: 3x [256,1,3,3] + 2x [256,256,1,1] convs, lin_out [768, 4096, 1]
   (4096 = 4 x 256 frame concat -> the encoder emits 25 tokens per second of audio)
projector: 3 x 768->768, then into the 2048-wide backbone
mel front end: 16 kHz, n_mels 128, n_fft 400, win 400, hop 160, hann periodic, center false,
   power 2.0, slaney mel + slaney norm, f0/f8k, log10 with floor -12, affine (x-4)/4
```

### 4.2 Reuse matrix

| need | backbone has | verdict |
|---|---|---|
| Q8 GEMV, any `N`, `K∈{2048, 7680, 2816, ...}` | `q8_partial` family + config picker | **reuse** (K=768/1792/1024 are new shapes → new `GemvCfg` entries + coverage assert) |
| RMSNorm / per-head norm | `k_rmsnorm`, `k_headnorm` | **reuse** with `H=768`, `hd=96` as parameters |
| depthwise causal conv + ring state | `k_conv7`, `k_shift_in` (width 7) | **generalize** to width 3 (`conv_L_cache` 3) |
| GQA attention with sliding window | `k_attn_head` (window parameter) | **reuse**; window 32, `hd=96`, 8 heads |
| RoPE | `k_rope` (rot is a parameter) | **reuse** (LFM2 encoder config decides full vs partial) |
| silu·gate, sigmoid gate, residual add | `k_silu_mul_bf`, `k_sigmoid_gate`, `k_add_bf` | **reuse** |
| **strided / 2-D convs, 4-frame concat** | — | **new kernel** (`downsample`) |
| **log-mel (FFT-400, slaney filterbank)** | — | **new** (DSP; no weights, no reuse) |

### 4.3 Cost, derived from bytes

119 MB of weights per encoder frame × 100 frames/s = **~11.9 GB per second of audio**, i.e.
**0.7–1.0 s of GPU per second of audio** at 11–17 GB/s — the encoder is roughly realtime on this
board and is *not* a rounding error in an audio deployment. For comparison the backbone costs
25 tokens per second of audio × ~98 ms/token of sequential prefill ≈ **2.4 s per second of
audio**. Both numbers are byte/rate derivations, flagged as such: the encoder is unported, and
the backbone figure is the measured per-step cost applied to the derived token rate (25/s from
the 4-frame concat above).

That ordering is the port's decision input: **the backbone prefill dominates audio latency, the
encoder is second, and the mel front end (a few ms per 5 s of audio) is free.** Anything that
batches the backbone prefill is worth ~70× more for audio than for a 23-token text prompt — and
batching is currently blocked by `ffn.w2`'s x-layout (Appendix A).

### 4.4 Gate plan for the encoder

Gate the component **at its own boundary**, not through the backbone: dump the encoder's last
hidden state per frame from HF for a fixed audio fixture, then teacher-force the engine with the
same encoder inputs and compare with the matrix gate (per-layer × per-frame rms within 3%) plus a
logit-level gate after the projector. This is the same protocol the backbone used (§6), applied
one level down — and it needs the mel fixture to be bit-reproducible, which means the front end
has to be ported (or frozen to a file) before the encoder can be scored.

## 5. Depthformer head and codec — a different pattern

```
depthformer  3 layers, hidden 1024, intermediate 2816, 32 heads / 8 KV heads, head_dim 32
             max_position_embeddings 12, is_causal FALSE (bidirectional), maskgit_steps 3
depth_linear [12288, 2048]   = 12 codebooks x 1024
            12 x depth_embeddings: embed [2017,1024], norm, to_logits [2017,1024]
codebooks 12, audio_vocab 2017, codebook_embedding [24204, 2048] (= 12 x 2017, exact)
interleave: 6 text tokens / 12 audio tokens per step
```

Why this is not "more of the same":

* **Bidirectional over 12 positions.** There is no KV cache, no sliding window, no ring: the
  whole sequence is 12 tokens, so one block per frame can hold q/k/v in shared memory. The
  causal `k_attn_head` does not apply — a 12×12 full-mask variant is a genuinely new kernel, and
  it is small enough to be trivially correct (the cheapest place in the whole port to start).
* **12 logits heads in one GEMV**: `depth_linear` gives `N = 12288`, then 12 independent
  `[2017,1024]` to_logits — a single `N = 24204` GEMV if the 12 weights are stacked at repack
  time (the same merge trick the backbone uses for gate+up).
* **MaskGIT is a sampling loop, not a forward pass**: 3 steps of mask/predict, with the
  repetition penalty and top-k from the model card. That is controller logic, and it belongs in
  the same place as the interleaved text/audio controller.
* **LiquidCodec is a separate 147 MB model** (tokens → 24 kHz waveform, streaming, 12 tokens per
  80 ms). It is the largest untouched surface in the whole port and has no kernel overlap with
  anything above except elementwise and conv.

## 6. How a port is gated (the protocol, reusable as-is)

### 6.1 The four reference gates

| gate | what it catches | current backbone result |
|---|---|---|
| per-layer × per-position rms vs HF (`--trace` matrix) | a wrong wiring in one layer/op | 23 layers × 23 positions all within 3% |
| pos-0 logits vs HF | wrong embedding/head wiring | top-1 = HF's (11), corr 0.99998 |
| 131 positions × 128 000 logits, 6 prompts | accumulation-order drift | top-1 = HF on 130/131, KL 0.0007–0.0019 |
| greedy streams vs HF goldens, 7 prompts × 16 | end-to-end | 6/7 exact (floor is a ratchet) |

Long-context additions (all in `checks_lfm3.py --long`): the rms matrix to 1000 positions
against an **fp16** reference (0/24 000 cells outside 3%), the short-prompt tie landscape
(top-1 vs fp16 and vs bf16 references separately), and a 300-position tie gate.

### 6.2 Making references for a new component

The suite's whole value is that the reference is **the same model run by HF on the same inputs**,
dumped at the exact boundary being scored:

```bash
~/venvs/lfm3/bin/python .auto/lfm3/hf_logits_seq.py --prompt A,C,E,F,K,L --dtype fp16|bf16
~/venvs/lfm3/bin/python .auto/lfm3/matrix_long.py --nt 1000 --dtype fp16
```

Two reference-precision lessons, both measured, both load-bearing:

* **bf16 and fp16 references disagree with each other** at ~4.5% of teacher-forced positions
  (6/134 short-prompt, 3/20 at 300 positions). Those are argmax ties where precision flips the
  answer. A "miss" inside the tie zone is not an engine regression — score both references.
* **The engine tracks the fp16 reference on magnitudes and the bf16 reference on argmax** (rms
  0 cells outside 3% vs fp16 to 1000 positions; 133/134 top-1 vs bf16). Both are true, and a gate
  must say which question it asks.

### 6.3 Floors are ratchets, not targets

`gate_logits_seq` thresholds were measured on the shipped arm; passing means "no worse than the
arm this gate was written on", and every number prints each run so drift is visible on a pass.
When a floor fails, the question is *which* position and *whether the references agree there* —
never "raise the floor".

### 6.4 Measurement discipline (all of these cost a session to learn)

* **Engine knobs reach the board only through `LFM3_ENV`** (`LFM3_ENV="LFM_H2=1" bash
  .auto/lfm3/checks.sh A`). Setting the variable in the shell sets it locally only and every gate
  number comes back byte-identical — that identity is the tell.
* **Prove the code path ran** before believing a timing: `R` exactly 1.0000, or gate output
  identical to the digit, means the probe never executed.
* **`test -x $BIN` is not a build check**: nvcc leaves the previous binary on failure. Build to
  `.new`, `mv` only on a clean compile.
* **`pgrep -x lfm3_conv_test` must be 0** before measuring. Three orphaned (killable, `STAT=R`,
  *not* zombies) processes held 3 of 4 A57 cores for days; at steady state they were invisible,
  under sustained load they produced +15…47% degraded board states. Kill recipe in place; they
  are not respawning.
* Single samples swing ±1% (unpinned clocks: `schedutil` / `nvhost_podgov`, 76.8 → 921.6 MHz).
  Every claim here is a same-session paired A/B in both orders, `R = sqrt(o1/o2)`. Runs ≥60 s
  need ≥3 samples per arm. Never compare against a number from another session.

## 7. Memory and bandwidth budget for the whole port

Measured on the board (engine startup lines):

```
device: 4.16 GB total, 0.82 GB free before the arena is created
weights+arena with the backbone blob: 2.15 GB   ->  0.79 GB free while running
host: MemAvailable 1252 MB, RSS 307 MB
```

*Derived* size of everything still to port, at Q8 (1.125 B/weight):

```
audio encoder + downsample + projector   ~119 MB
depthformer + depth_embeddings + depth_linear + codebook_embedding   ~169 MB
LiquidCodec decoder (shipped weights)     147 MB
                                    total ~435 MB
```

So the full LFM3-Audio port at Q8 is ~2.4 GB of weights on a 4 GB unified board with ~0.8 GB
free today — feasible, but only just, and the codec (needed last) should be loaded worst-case as
an evict-and-load rather than resident. **Do not plan on raising precision anywhere**: the plan
doc's measured verdict is that 4-bit does not survive the gates (0/48 prompts at Q2), and the
certified arm is Q8 linears + fp32 scales + bf16 embeddings.

Time budget follows bytes at 11–17 GB/s. A worked example, 5 s of audio:

```
encoder   119 MB x 100 frames/s x 5 s = 59.5 GB  / ~14 GB/s   ~= 4.3 s
backbone  125 audio tokens x 98 ms/step (sequential prefill)  ~= 12.3 s
depth head + codec 12 tokens/80 ms, small bytes, per-frame    ~= to be measured
```

Both rows are derivations, not measurements — but they are why the port order in §1 is what it
is, and why the backbone prefill (not the encoder, not the head) is the first latency problem the
audio path will hit.

## 8. Build and run

```bash
bash .auto/lfm3/measure.sh                  # sync -> build -> fixed job -> METRIC lines
bash .auto/lfm3/checks.sh A B C E F K L     # the four gates
LFM3_ENV="LFM_H2=1" bash .auto/lfm3/checks.sh A   # same gates, other engine arm
bash .auto/lfm3/ab.sh "" "LFM_NOFMA=1"      # paired A/B of an env knob, both orders
python .auto/lfm3/checks_lfm3.py --long A B C E F K L   # + long-context and tie gates
```

```
nvcc -gencode arch=compute_53,code=sm_53 -O3 -fmad=false -Xptxas -O3,-dlcm=ca lfm3_nanode.cu -lz
```

Engine usage: `./lfm3_nanode <blob.lfm3q8> --steps N --prompt id,id,...`. Flags: `--trace`
(per-layer x per-position rms matrix, what the matrix gate parses), `--probe-layer L
--probe-step P`, `--dump-logits FILE`, `--dump-logits-all FILE` (appends every step — the
logit gates), `--emb-host`, `--merge`, `--lmrows N`, `--prefill-minimal`, `--lm-selftest`.
Context is compile-time (`MAXTOK`, 4096).

Env knobs, all defaulting to the shipping configuration: `LFM_TIME` (per-phase),
`LFM_KTIME` (per-kernel), `LFM_H2` (half2 inner loop), `LFM_WARMMS` (DVFS-ramp warm-up,
default 0), `LFM_NOWARM` (control for the load-time warm-up), `LFM_LEGACYFEED` (feeding fix
control), `LFM_NOFMA`, `LFM_NOFUSE`, `LFM_EMB`, `LFM_R`, `LFM_TPB`, `LFM_KCHUNK`, `LFM_ROWS`,
`LFM_GRAPHS`, `LFM_STREAM`, `LFM_XRANGE`, `LFM_DBG_*`. The measured-out probes (`LFM_MERGE`,
`LFM_UNROLL2`, `LFM_ROWLANE`, `LFM_LMDBUF`, `LFM_RADAPT`) carry their numbers in Appendix A —
do not re-walk them.

## 9. Known issues

* **Prompt D cannot be scored**: goldens are not reproducible even by HF across devices (a bf16
  tie breaks differently on CPU vs GPU and diverges at token 6), so `make_prompt_ids.py` refuses
  to certify ids for it. The stream gate covers 7 of 8 prompts.
* **`ffn.w2` reads `x + row*K`** beyond the 7680-element silu buffer, so its output depends on
  step-recursive arena scratch. This pins the prefill to one position at a time (Appendix A). Any
  change to that layout is a **goldens change** — a deliberate decision, not a cleanup.
* `--trace` cannot run inside a graph capture, so `LFM_GRAPHS` forces it off.
* Clocks are unpinned and unstabilisable without root (`jetson_clocks` absent, sudo
  `requiretty`), so absolutes here would not reproduce on a clock-locked board; ratios do.

---

## Appendix A — the backbone latency record (evidence, not the objective)

Decode, 141.04 → **112.79 ms/token** (−20.0%), each step paired A/B in both orders:

```
141.04 ┃█ baseline (real CUDA path)
139.12 ┃█−0.39% fuse k_bf2f into the GEMV's x staging (~110 launches/token gone)
136.41 ┃ −2.0%  __fmaf_rn accumulation in the Q8 GEMV
112.79 ┃█−17.3% __launch_bounds__(256,6) on the Q8 GEMV (resource hint, exact)
```

TTFT on the 23-token prompt: 2755.4 → **2256.2 ms** (−18.1%), from first-true-token accounting
(−107 ms), the load-time warm-up (−58.7 ms), and the feeding fix + lm_head skip on prompt steps
(−318.5 ms). Prefill is sequential (one position per full step), so TTFT ≈ 22 × 97.6 ms + one
112.4 ms step.

**Measured-out, with numbers** (`faster ◄─────────────────────────► slower`):

```
 +0.33  merged gate+up GEMV          +0.33% for 679 MB — rejected
 -0.55  CUDA graphs of the layer loop    ~250 launches ≈ 1 ms of CPU vs a 112 ms step
 -2.7   R=128 under launch bounds       R=256 stands
 -3.6   TPB=512                         512 threads halve the block count on 4 SMs
 -5.3   KCHUNK=1024                     KCHUNK=K under all three codegens
 -6.9   exact integer int8→float        extra integer op outweighs the saved I2F
-10.5   inter-row ILP (unroll 2)        register spill at R=256/TPB=256
-12.5   -maxrregcount=32                forced spills
-157    one-row-per-lane wide-K GEMV    32 scattered sectors per warp instruction
-162    double-buffered lm_head         H2D + GEMV share the one bus: two full passes
```

Two configuration findings worth carrying: **never stream the lm_head from pinned host on Tegra**
(the copy and the GEMV share the 21.9 GB/s bus — 10.4% worse), and the GEMV config is a
characterised optimum (R=256, TPB=256, KCHUNK=K, FMA on, serial row loop,
`__launch_bounds__(256,6)`), confirmed across three codegen regimes.

**`ffn.w2` (N=2048, K=7680) is the one remaining shape anomaly** — same bytes as `w1`
(N=7680, K=2048), ~17% slower after launch bounds (1.47× before them). Five hypotheses are measured out (block fill, ILP,
cache hints, chunk interleaving, per-lane rows). **Prefill batching is therefore closed**: a
bit-exact CT=2 weight-reuse kernel existed and was 1.15–1.45× standalone (8/8 shapes bit-exact),
but a layer-first integration diverged because `w2`'s x-indexing reads arena scratch (§9), and no
per-position tiling reproduces those bytes. Batching the other nine GEMVs does not pay.

**The H2 arm** (raw `half2`/HFMA2 inner loop, `LFM_H2=1`, not bit-exact) is measured and gated
but **not shipped**: clean board −1.8% TTFT (23 pos), −2.5% (300), −3.9% (360), −5.0% (1000),
decode −1.7%; with one CPU co-tenant −4.6…−5.9% TTFT and −2.6…−5.0% decode. Quality is at the
shipped arm's level on every axis (formal gates gate-for-gate identical, KL at the floor short
and long, argmax identical on both references, rms 0 cells vs fp16 to 1000 positions). The
`H2=2` rescaled variant is **dominated**: ~0% speed and it fails the logit gate (127/131).

Board state is a first-class measurement condition: unpinned clocks (76.8–921.6 MHz), a 2–6%
hour-to-hour drift, ~15% state shifts during ≥60 s runs, and three orphaned CPU spinners that
were killable all along — killing them restored a clean board whose shipped band is
**2280 ± 2 ms** with a 0.13% spread. Every speed number above is from that clean board; pre-kill
speed figures are retired.

## Appendix B — gate results on the current tree

```
bash .auto/lfm3/checks.sh A B C E F K L        # short
python .auto/lfm3/checks_lfm3.py --long A B C E F K L   # + long-context + ties
```

```
matrix (23 layers x 23 positions vs HF)      all within 3%
matrix_long, 1000 positions vs fp16 ref      0 / 24000 cells outside 3%
pos-0 logits                                 top-1 = HF's (11), corr 0.99998
131 positions x 128k logits, 6 prompts       top-1 130/131, top-8 coverage 1.000,
                                             corr 0.99946-0.99973, KL 0.0007-0.0019
greedy streams, 7 prompts x 16 tokens        6/7 exact (F 11/16, its known near-tie)
tie landscape, 134 positions                 top-1 133/134 vs bf16, 127/134 vs fp16
tie landscape, 300-position trajectory       19-20/20 vs fp16 (15% ref-disagreement density)
tie landscape, 600-position trajectory       12/12 both arms, both references
```

```
prompt   top-1   top-8 cov   corr (min)          KL mean   KL max
A        23/23     1.000     0.99964 (0.99900)    0.0014    0.0084
C        21/21     1.000     0.99946 (0.99770)    0.0015    0.0161
E        23/23     1.000     0.99968 (0.99897)    0.0017    0.0121
F        20/21     1.000     0.99963 (0.99884)    0.0015    0.0079
K        23/23     1.000     0.99973 (0.99932)    0.0007    0.0029
L        23/23     1.000     0.99966 (0.99921)    0.0019    0.0126
```

Arm-neutrality of the gates themselves: `LFM_NOFUSE=1` and `LFM_R=64` are bit-identical to the
shipped arm and print the same numbers to the digit — that is the determinism control, and it is
why a probe whose output is byte-identical to the default arm means "the code never ran".
