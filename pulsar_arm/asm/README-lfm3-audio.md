# LFM3-Audio-2B on a Jetson Nano — decode latency

Pure-CUDA (`sm_53`, CUDA 10.2) decode engine for **LFM3-Audio-2B** at Q8 on a Jetson Nano
Gen1 (Tegra X1: 4× Cortex-A57 + 4 SM Maxwell, 4 GB unified, 21.9 GB/s bus). One translation
unit — [`lfm3_nanode.cu`](lfm3_nanode.cu) — built with `nvcc` and nothing else; weights come
from a `.lfm3q8` blob (Q8 linears + fp32 scales + Q8 tied embedding, 286 tensors, 1466 MB).

Primary metric: **decode ms/token, lower is better**, single sequence, greedy.

```
current  136.30 ms/token   =   7.34 tok/s        (RSS 306 MB, 4 GB board)
baseline 141.04 ms/token   =   7.09 tok/s        ->  -3.4%
```

## How the number is measured

`.auto/lfm3/measure.sh` scps the `.cu`, builds it on the Nano (compile to `$BIN.new`, `mv`
only on a clean compile, refuse to run without `BUILT_OK`), drops the page cache, then runs
one fixed job: blob `lfm3-audio-2b-q8.lfm3q8`, the 23 prompt ids of prompt A, `--steps 48`.

Timing is **on the device**, bracketing one decode step with `cudaEventRecord`:

```
 evA ─► embedding-row dequant
      ─► 24 layers: operator_norm ─► GEMVs ─► ( conv branch | q,k,v ─► headnorm
      │                                        ─► RoPE ─► KV write ─► attention ─► gate )
      │                                        ─► FFN w1/w3 ─► silu ─► w2
      │                                        ─► post-op norm ─► add1 ─► ffn_norm
      │                                        ─► post-ffn norm ─► add2
      ─► final embedding_norm
      ─► lm_head GEMV over all 128,000 rows
      ─► k_argmax                                   (greedy token, on-device)
 evB ─► cudaEventElapsedTime
```

Steps 0–22 feed the prompt ids (one position each), 23–47 are free-running greedy; the first
two steps are dropped and the average is over the remaining 46 (`steady-state … over 46
steps`, which is what the harness parses). It excludes blob load, tokenisation and the 4-byte
readback of the sampled token. Repeat spread on one binary is **0.03–0.14%**; the board drifts
**2–6% between hours**, so every claim below is a same-session paired A/B (`ab.sh`, one build,
both arms, each pair in both orders, `R = sqrt(o1/o2)`), never a comparison across sessions.

## Where the 136 ms goes

`LFM_TIME=1` splits the step by layer type. Its timers carry their own sync overhead, so the
parts sum to ~145.8 ms against a 136.3 ms whole — read the split, not the sum.

```
per-phase ms/token (LFM_TIME)          bytes/token  GB/s   [ 1 block = 2 ms ]
conv layers (13)  ███████████████████████████████████░  70.6     ~715 MB  9.4→10.1
attention (11)    ████████████████████████████░░░░░░░░  56.6     ~545 MB  9.0→ 9.6
lm_head           █████████░░░░░░░░░░░░░░░░░░░░░░░░░░░  18.5      278 MB      14.0
final norm        ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.1        4 KB         —
embedding         ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.0        4 KB         —

Same scale for every row: conv 35 blocks, attention 28, lm_head 9. Parts sum to ~145.8
ms against a 136.3 ms whole because each phase timer carries its own sync.

The plan's budget assumed the lm_head was ~half the step. It is **14%**, and it already runs at
14 GB/s against a **measured 17 GB/s streaming ceiling** on this board — which is why tuning the
lm_head slab size measured exactly `R = 1.0000`. All the headroom is in the 24 layer GEMVs.

## The diagnosis that produced the win

The layer GEMVs sat at 9.0–9.4 GB/s, so they looked bandwidth-bound. Counting FP instructions
said otherwise: the inner loop did `dot = __fadd_rn(dot, __fmul_rn(c[j], x[j]))` — **two FP
instructions per product** (the plan's exact-per-product rounding, reinforced by the build's
`-fmad=false`) — plus a loop-carried `acc` chain. That is 2432 FP warp-instructions per warp
where the arithmetic minimum is 512 FMAs, and it reproduces the measured 1.33 ms/layer almost
exactly. **FP-pipe-bound; 9.2 GB/s was a symptom.**

`__fmaf_rn` accumulation: **−2.0%** (paired, R=1.0100, faster 3/3), every GEMV phase −6 to
−8.5%, gates green, greedy streams byte-identical. ggml's own reference dot product is
`sum += w*x` in C, which compilers contract to an FMA — so this is at least as faithful to the
reference, with one rounding per product instead of two.

```
kept chain, ms/token (each step paired A/B in both orders)
141.04 ┃█ baseline (measured on the real CUDA path)
140.32 ┃█−0.5%  R=256 rows/block for all shapes (bit-identical row→block remap)
139.62 ┃█       bake R=256 into pick()
139.56 ┃█       bf16 payload support for the certified arm
139.12 ┃█−0.39% fuse k_bf2f into the GEMV's x staging (~110 launches/token gone)
136.41 ┃ −2.0%  __fmaf_rn accumulation in the Q8 GEMV
136.30 ┃█       shipped (RSS 306 MB, gates green)
       ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  135      141 ms
```

## Bandwidth by shape — the one anomaly left

Per-kernel timing (`LFM_KTIME=1`) gives a clean correlation between achieved bandwidth and
**row length**, at fixed bytes:

```
GB/s (measured streaming ceiling on this board = 17)
lm_head        N=128000 K=2048   ██████████████░░░░░░░░░░░░░░  14.0
conv.in_proj   N=  6144 K=2048   █████████████░░░░░░░░░░░░░░░  13.3
ffn.w1         N=  6144 K=2048   ███████████░░░░░░░░░░░░░░░░░  10.9
ffn.w3         N=  6144 K=2048   ███████████░░░░░░░░░░░░░░░░░  10.9
conv.out_proj  N=  2048 K=2048   ██████████░░░░░░░░░░░░░░░░░░  10.3
ffn.w2         N=  2048 K=6144   ███████░░░░░░░░░░░░░░░░░░░░░   7.1   ◄─ same BYTES as w1/w3
                 0     5    10    15   17
```

`ffn.w2` (N=2048, K=6144) costs 1.773 ms where `ffn.w1` (N=6144, K=2048) costs 1.161 ms for an
identical 12.6 MB. Five hypotheses were implemented and measured against it — block fill
(R=64, shape-adaptive R), inter-row ILP (`#pragma unroll 2`), cache hints (`__ldcs`), chunk
interleaving (KCHUNK=3072), and per-lane row ownership (a one-row-per-lane wide-K GEMV) — and
**all five are measured out**. The strided-lane form, per-lane 32-block stride + warp tree, is
the right structure for this layout on this board: giving each lane a whole row makes every
warp instruction read 32 bytes from 32 different rows, consuming 32 sectors per 1 KB of useful
data instead of covering 1024 contiguous bytes.

## Structural levers measured out (so nobody re-walks them)

All are paired A/B, one build, both orders. Kept behind env flags, **off by default**; the
shipping path never touches them.

```
faster ◄──────────────────────────────────────────────────────────────► slower
 +0.33 │ merged gate+up GEMV (24→48 blocks)        +0.33% for 679 MB — rejected
 +0.06 │ -fmad=true build-wide                      nothing to contract (GEMVs already FMA)
 +0.01 │ __ldcs streaming loads                     L2 pollution costs nothing
 +0.01 │ KCHUNK=3072 for K=6144                     verified null at the kernel level
   0   ┼───────────────────────────────────────────────────────────────────
 -0.55 │ CUDA graphs of the layer loop              ~250 launches ≈ 1 ms of CPU vs a 136 ms
                                                     step — launches were never the bottleneck
 -3.0  │ shape-adaptive R (64/32 for few-row shapes)
 -3.6  │ TPB=512
 -5.3  │ KCHUNK=1024
 -6.5  │ R=64 (global)                              x-stage amortization dominates
-10.5  │ inter-row ILP (unroll 2)                   register spill at R=256/TPB=256
-157   │ one-row-per-lane wide-K GEMV               32 scattered sectors per warp instruction
-162   │ double-buffered lm_head streaming          19.8 → 52.5 ms; Tegra copy engine + events
```

Two configuration findings worth carrying forward:

* **Do not stream the lm_head from pinned host on Tegra.** The resident tied embedding beats
  the plan's prescribed `--emb-host` row streaming by **10.4%** (139.19 vs 169.65 ms/token,
  paired 3/3). The H2D copy and the GEMV share the same 21.9 GB/s bus, so streaming costs
  *two* full passes over 278 MB. The plan's `--emb-host` requirement was sized for the
  1944 MB whole-backbone-Q8_0 config; this Q8-embedding blob is 1466 MB and fits resident with
  ~2.8 GB device free, so streaming is pure loss. `LFM_EMB=resident|host` A/Bs it.
* **GEMV config is a characterised optimum**: R=256, TPB=256, KCHUNK=K, FMA on, serial row
  loop, plain loads — confirmed on both accumulation forms.

## Quality gates

```
bash .auto/lfm3/checks.sh A B C E F K L
```

| gate | current |
|---|---|
| per-layer × per-position rms vs HF, 23 layers × 23 positions | all within 3% |
| pos-0 logits | top-1 = HF's (11), corr 0.99998, std 4.564 vs 4.547 |
| **131 positions × 128,000 logits, 6 prompts** | **top-1 = HF's on 130/131**, top-8 coverage 1.000 everywhere, corr 0.99946–0.99973, KL 0.0007–0.0019 |
| greedy stream prefix vs HF goldens, 7 prompts | every prompt ≥ floor (A=2, B/C/E/F/K/L=1) |

```
prompt    top-1    top-8 cov    corr (min)          KL mean   KL max
A         23/23      1.000     0.99964 (0.99900)    0.0014    0.0084
C         21/21      1.000     0.99946 (0.99770)    0.0015    0.0161
E         23/23      1.000     0.99968 (0.99897)    0.0017    0.0121
F         20/21      1.000     0.99963 (0.99884)    0.0015    0.0079
K         23/23      1.000     0.99973 (0.99932)    0.0007    0.0029
L         23/23      1.000     0.99966 (0.99921)    0.0019    0.0126
```

**Read this honestly.** Prefixes are 1–2 tokens of 16 and **0/7 prompts fully match**: the
engine diverges from HF within a couple of tokens on every prompt. `CHECKS OK` certifies the
arm's floor (1 token, the plan's certified worst prefix for this arm), not stream equality.

The logit gate is what makes that disclosure mean what it should: over the full distribution
the engine's top-1 equals HF's at 130 of 131 positions, HF's top-1 is always inside the
engine's top-8 (and the converse), and mean KL is under 2e-3. So the greedy divergence is
argmax/tie amplification on a distribution that agrees closely — not a distributional gap.
The one miss is prompt F at a single position, where HF's top-1 is still in the engine's top-8:
a near-tie, which is what the "one lost position warns, two fails" floor exists to surface.
Prompt A alone would never have shown it.

**The logit gate is also the reason to distrust the stream gate as an instrument.** The
`__fmaf_rn` switch that bought −2.0% left all 16 stream tokens on both prompts unchanged, so
stream equality could not have detected a regression there. Running every kept change through
the 6-prompt gate, against the same HF reference, with no threshold moved:

```
arm                                top-1 (of 131)   corr range       KL mean range   verdict
shipped (__fmaf_rn, fused staging)      130         .99946-.99973    .0014-.0019     reference
LFM_NOFMA=1 (exact mul-then-add)        130         .99942-.99970    .0015-.0019     indistinguishable
LFM_NOFUSE=1 (separate k_bf2f)          130         .99946-.99973    .0014-.0019     bit-identical
LFM_R=64 (row->block remap)             130         .99946-.99973    .0014-.0019     bit-identical
```

Two things follow. The fused-staging and R knobs are **bit-identical**, so their lines above
are also a positive control: a bit-exact arm must print identical numbers, and it does — that
is the determinism check for the gate itself. And the FMA switch is **quality-neutral**:
FMA wins corr on 4 prompts (A, C, F, K) and loses on 2 (E, L) by ~2e-5, while the spread
*between prompts* (KL 0.0007–0.0019) is about 10× the spread *between arms*. An early
two-prompt read said FMA "agrees with HF better"; that was prompt-A-specific and did not
survive the other five, so it is not claimed here. What is claimed is the safe half: the
−2.0% cost nothing measurable.

Thresholds in `gate_logits_seq` are a **regression ratchet** measured on the shipped arm —
passing means "no worse than the arm this gate was written on", and every number is printed
each run so drift is visible even on a pass. The measured KL sits 25×+ below the 0.05 floor,
so the ratchet has room and is not fitted to these prompts.

**Env forwarding gotcha (cost an hour, now fixed):** engine knobs reach the Nano through
`LFM3_ENV`, not by setting them in the shell — `LFM_NOFMA=1 bash checks.sh A` sets the
variable *locally only*, the engine runs the default arm, and every gate number comes back
byte-identical. Use `LFM3_ENV="LFM_NOFMA=1" bash checks.sh A`; the header echoes
`engine env: …` so an arm is self-documenting. Identical-to-the-digit gate output is the tell.

Two structural facts about the goldens, both measured:

* **The plan's 8/8 greedy-goldens gate is not reachable by a from-scratch GEMV.** The goldens
  are HF/cuBLAS outputs; this engine matches the *ggml* summation order. Each Linear differs by
  ~1 bf16 ulp (the per-layer ratio table sits at 0.995–1.006 = bf16 resolution) and greedy
  argmax flips once the top-2 gap falls under the accumulated perturbation.
* **They are not reproducible even by HF across devices.** Regenerating golden D on GPU from its
  own stored prompt diverges at token 6 (22 vs 20) — oracle.py generates on CPU and bf16 logit
  ties break per device/kernel. C/E/F/K/L reproduce exactly (24/24, 24/24, 24/24, 17/17, 24/24)
  and carry verified `prompt_ids`; D is refused by `.auto/lfm3/make_prompt_ids.py`, which writes
  ids only when (a) rebuilding A/B from their stored text equals their stored ids and (b) HF
  greedy reproduces the golden's own `gen`.

The natural next gate is **top-k logit agreement vs HF**, which is quantitative where stream
prefix is cliff-like — stream equality is a weak discriminator for accumulation-order changes
(16 tokens were unchanged by the FMA switch).

## Build, run, knobs

```bash
bash .auto/lfm3/measure.sh                      # sync -> build -> fixed job -> METRIC lines
bash .auto/lfm3/checks.sh A B C E F K L         # the gates
~/venvs/lfm3/bin/python .auto/lfm3/hf_logits_seq.py --prompt A,C,E,F,K,L   # regen logit refs
LFM3_ENV="LFM_NOFMA=1" bash .auto/lfm3/checks.sh   # same gates, other engine arm (see gotcha)
bash .auto/lfm3/ab.sh "" "LFM_NOFMA=1"          # paired A/B of an env knob, both orders
LFM_TIME=1  ./lfm3_nanode lfm3-audio-2b-q8.lfm3q8 --steps 24 --prompt …   # per-phase
LFM_KTIME=1 ./lfm3_nanode …                     # per-kernel table at exit
```

```
nvcc -gencode arch=compute_53,code=sm_53 -O3 -fmad=false -Xptxas -O3,-dlcm=ca lfm3_nanode.cu -lz
```

Knobs (all default to the shipping configuration): `LFM_NOFMA`, `LFM_NOFUSE`, `LFM_EMB`,
`LFM_R`, `LFM_TPB`, `LFM_KCHUNK`, `LFM_TIME`, `LFM_KTIME`, and the measured-out probes
`LFM_MERGE`, `LFM_LMDBUF`, `LFM_GRAPHS`, `LFM_STREAM`, `LFM_RADAPT`, `LFM_UNROLL2`,
`LFM_ROWLANE`.

Two harness traps cost real time and are now defended against: `test -x $BIN` is **not** a
build check (nvcc leaves the previous binary on failure, so three broken builds silently
measured stale numbers), and engine env does **not** reach the Nano by setting it in the
shell — `LFM_NOFMA=1 bash checks.sh A` sets it locally only, the engine runs the default arm,
and every gate number comes back byte-identical. Use `LFM3_ENV="LFM_NOFMA=1" bash checks.sh A`
(the header echoes `engine env: …`), and pass program args through `LFM3_ARGS`. The tell for a
probe that never executed: `R` exactly 1.0000, or gate output identical to the digit.

## Known issues

* **Three stuck processes on the Nano, measured harmless to decode.** Three `lfm3_conv_test`
  processes from 2026-10-28 are stuck in state R inside the CUDA driver (`kill -9` does not
  reap them, `systemctl reboot` is swallowed), spinning cores 0, 2 and 3. Pinning the engine
  to the one idle core (`taskset -c 1`) is a null result in a paired, order-balanced A/B
  (137.66 vs 137.58 ms/token, R = 1.0006, 2/4 pairs each way): the launch thread is not on
  the critical path, consistent with CUDA graphs measuring -0.55%. So the absolutes are
  **not** inflated by them, and the GEMV config sweep does not need repeating after a reboot.
  The ~1% gap between this session's 137.5 and the 136.3 best is ordinary board drift.
* Prompt D cannot be scored (see above), and the stream gate therefore covers 7 of 8 prompts.
* `--trace` (the per-layer/per-position matrix) cannot run inside a graph capture, so
  `LFM_GRAPHS` forces it off.
