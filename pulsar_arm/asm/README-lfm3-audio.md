# LFM3-Audio-2B on a Jetson Nano — decode latency and time to first token

Pure-CUDA (`sm_53`, CUDA 10.2) decode engine for **LFM3-Audio-2B** at Q8 on a Jetson Nano
Gen1 (Tegra X1: 4× Cortex-A57 + 4 SM Maxwell, 4 GB unified, 21.9 GB/s bus). One translation
unit — [`lfm3_nanode.cu`](lfm3_nanode.cu) — built with `nvcc` and nothing else; weights come
from a `.lfm3q8` blob (Q8 linears + fp32 scales + Q8 tied embedding, 286 tensors, 1466 MB).

Primary metric: **decode ms/token, lower is better**, single sequence, greedy.

```
current  112.79 ms/token   =   8.87 tok/s        (RSS 306 MB, 4 GB board)
baseline 141.04 ms/token   =   7.09 tok/s        ->  -20.0%
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

Board state (audited, not changed): `nvpmodel` MAXN, but clocks are **not pinned** -- CPU
governor `schedutil`, GPU governor `nvhost_podgov`, ranging 76.8 MHz idle to 921.6 MHz under
load. That is a mechanistic source of the drift (clock ramp on first kernels, which is part
of what the load-time warm-up pays for) and a reason the absolutes here would not reproduce
on a clock-locked board. At run lengths >=60 s the state can shift ~15% between consecutive
runs (#227: shipped 1000-token samples {105.1, 120.8} s), so ratios at that length need >=3
samples per arm and best-of comparisons -- single pairs are not informative there.
Three unreapable `lfm3_conv_test` zombies keep 3 of 4 A57 cores
spinning permanently (thermal pressure plus the load average ~3.2); #118 showed they do not
affect decode, but they are part of the thermal backdrop. Clocks were deliberately left
unlocked so every number stays comparable with the session baseline.

Two harness facts from #177: `jetson_clocks` is not installed on this board and governor
writes need sudo, and sudo is `requiretty` -- so the `echo PW | sudo -S` pattern used by
measure.sh/ab.sh for `drop_caches` has always silently no-opped (hidden by 2>/dev/null;
true). The metric is unaffected -- the engine fadvise(DONTNEED)s the blob at load and the
warm-up absorbs first-touch, so cache state is engine-managed run to run -- but do not
claim "cold page cache" in anything citing these scripts, and clock-pinning is not
available to quantify the drift with current permissions.

Metric-definition caveat (#176): measure.sh's `decode_ms_per_tok` averages steps 2..47,
which INCLUDES the 21 prompt steps. Since #139 skipped the lm_head on prompt steps, the
same definition moved from ~112.8 to ~106.2 ms/token with no arithmetic change. Decode
cross-era comparisons must use the step lines' decode-phase average (112.4-112.8 all
session), not the mixed steady-state mean.

## Where the 112.8 ms goes

`LFM_TIME=1` splits the step by layer type. Its timers carry their own sync overhead, so the
parts sum to ~127.0 ms against a ~114.9 ms whole on a short run — read the split, not the
sum. (Pre-launch-bounds: conv 70.6 / attn 56.6 / lm_head 18.5 ms; the win spread evenly
because every layer GEMV shares the one `q8_partial` kernel.)

```
per-phase ms/token (LFM_TIME)          bytes/token  GB/s   [ 1 block = 2 ms ]
conv layers (13)  ███████████████████████████████░░░░░  61.8     ~715 MB  10.6→11.6
attention (11)    ████████████████████████░░░░░░░░░░░░  48.9     ~545 MB  10.4→11.1
lm_head           ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  16.3      278 MB      14.0
final norm        ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.1        4 KB         —
embedding         ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.0        4 KB         —

Same scale for every row: conv 31 blocks, attention 24, lm_head 8. Parts sum to ~127.0
ms against a ~114.9 ms whole because each phase timer carries its own sync.

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

`__launch_bounds__(256, 6)` on the Q8 GEMV: **−17.3%** (paired, R=1.1043, faster 4/4).
The kernel used 40 regs/thread at TPB=256, so six blocks/SM already fit — but the compiler
did not know that. Declaring the target moved it to 39 regs with no spills and much better
scheduling; gates green with numbers identical to the unannotated build. Register caps in
the other direction (`-maxrregcount=32`) cost +12.5% in forced spills, and relaxing to
five blocks/SM is a measured null (compiler keeps 39 regs), so six is the characterized
target. A resource hint only: it cannot change a value or the reduction order.

```
kept chain, ms/token (each step paired A/B in both orders)
141.04 ┃█ baseline (measured on the real CUDA path)
140.32 ┃█−0.5%  R=256 rows/block for all shapes (bit-identical row→block remap)
139.62 ┃█       bake R=256 into pick()
139.12 ┃█−0.39% fuse k_bf2f into the GEMV's x staging (~110 launches/token gone)
136.41 ┃ −2.0%  __fmaf_rn accumulation in the Q8 GEMV
112.79 ┃█−17.3% __launch_bounds__(256,6) on the Q8 GEMV (resource hint, exact)
       ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  110      141 ms
```

## Time to first token (the e2e objective)

Steady-state decode is the wrong number for interactive use: what matters is last input to
first output token. On prompt A (23 ids) the engine prefills sequentially -- one prompt id per
full decode step -- so TTFT was 2648 ms, 96% of it prefill. Two changes moved it:

```
TTFT, prompt A (each step paired A/B in both orders where noted)
2755.4 ┃  baseline (sequential prefill, cold start, shifted feeding)
2648.2 ┃  -107 ms  first-true-token accounting (the step lines print sampled tokens;
       ┃           the first true output completes at step n-1, not n)
2589.6 ┃█ -58.7 ms  load-time warm-up (one BOS forward at startup, state reset;
       ┃           LFM_NOWARM=1 is the control; +166.7 ms at load)
       ┃ LFM_WARMMS=N (default 0) extends that warm-up to N ms of wall-clock GPU
       ┃ load, absorbing the devfreq DVFS ramp (idle board runs 76.8-153.6 MHz
       ┃ for the first ~5-24 s, #210/#209): neutral in the normal state, recovers
       ┃ ~1% from an 8-min idle start, and is the fix for +7% cold-board days.
       ┃ The GPU clock decays 921.6 -> 76.8 MHz within ~2-4 s of process exit
       ┃ (#223), so ANY query gap >~5 s starts cold: for interactive/audio use
       ┃ WARMMS=6000 is the per-startup recommendation; the -20 ms cold penalty
       ┃ is front-loaded, hence -0.1% at 300-token prefills (#220)
2271.1 ┃█-318.5 ms  feeding fix + lm_head skip on prompt steps 0..n-2
       ┃           (the loop fed every prompt step's prediction back as input;
       ┃           LFM_LEGACYFEED=1 is the control)
2270.6 ┃█          reproduced on a fresh build (step 0 now the fastest step: no lm_head)
2260.4 ┃█          band re-verified post #162-#165 (gates green on a fresh build; step 0 =
       ┃           steady prompt step 97.6 ms -- #138's warm-up is complete, and TTFT is
       ┃           fully attributed: 22 x 97.6 + 112.4. Prompt B runs 98.2 ms/step: the
       ┃           per-step cost is prompt-independent, not prompt-A overfit.)
```

Prefill is still ~95% of TTFT, and batching positions with identical arithmetic tops out near
1.0x (four tile-size/staging variants measured, all bit-exact, best 1.02x) because the GEMVs
are FP-pipe-bound: batching removes weight bytes but not a single FMA -- and an unpadded
staging variant proved the padding load-bearing by running 0.09x (16-way bank conflicts).
A fifth variant (CT=2 weight-reuse tiles, bit-exact at 8/8 shapes and 1.15-1.45x standalone,
#162-#164) was integrated as a layer-first batched prefill in #165 and CLOSED there: the
engine's ffn.w2 GEMV indexes x at row*K, so its rows 1..2047 read step-recursive arena
scratch beyond the 7680-element silu buffer, and no per-position tiling reproduces those
bytes. Layer-first w2 moved positions 10-17 by +37..+158% and made step-22 logits garbage
(top-1 survived on tie luck). w2 is pinned sequential by the goldens; batching the other
nine GEMVs does not pay. Batching prefill would now require changing the engine's w2
x-layout -- a goldens change, the user's call.
The remaining prefill lever is fewer FP operations per weight. Measured, all behind flags,
all gates green; the call is the user's since every option below breaks bit-exactness:

```
arm                              TTFT      decode    streams  KL A/C/K          range review
shipped (fp32 path)              2257.7    112.6     6/7      0.0013/0.0005/0.0006  n/a
LFM_H2=1 (raw half2 + HFMA2)     2132.3    106.6     6/7      0.0016/0.0006/0.0005  28% of one layer's
 -5.6% TTFT / -5.3% decode                                        inputs in fp16's low-        precision range;             theoretical accumulator        overflow on ffn.w2
LFM_H2=2 (exact per-block 2^k     ~2270    ~112.8    6/7      0.0012/0.0007/0.0005  safe by construction
 rescale; the scan costs what                              (post-RoPE re-score,     (bound 65024, no
 the faster loop saves)                                    #159: the 7/7 was tie     subnormals)
                                                           luck; F drops to 19/21,
                                                           a floor the shipped passes)

speed rows: same-session pairs 2026-02-10 (#180/#189). The raw arm's win is REGIME-dependent
(same-session pairs at each length): 23 pos -5.7% (#205), 300 pos -7.8% (#206), 360 pos -8.0%
(order-balanced, #203); at 1000 pos not resolvable on this board -- the four samples span a
~15% board-state drift (shipped {105.1, 120.8} s vs H2 {105.3, 106.1} s: the H2 arm is
state-insensitive, the shipped arm degrades in bad states, #225/#227). HFMA2 pays while the
GEMVs are the GPU-side bottleneck; audio sweet spot is the 5-30 s prefill band (300-360
tokens); chat-class -5.7%; 80 s+ single-shot prefills at best neutral. Quality: formal
suite gate-for-gate identical to shipped (#182 short, #199 full --long), tie-landscape argmax
identical on both references (#195/#196), rms vs fp16 reference 0/24000 cells to 1000
positions in both arms (#192/#201).
```

The formal gate suite itself passes on the raw arm (#182, current tree): matrix, logits0, streams,
and the same 130/131 logit top-1 the shipped arm gets. The formal long-context gate
(matrix_long, layer x position rms cells) shows 5/7200 over 3% at 300 positions vs the shipped
arm's 1/7200 -- all engine-low by at most 3.2% (#184) -- but against an fp16 HF reference both
arms are 0 cells, at 300 AND at 1000 positions (#192): those cells are bf16 reference
quantization at the largest residuals, not engine arithmetic. The engine tracks fp16 truth
within 3% across the full 1000-position ladder, in both arms. The rescaled arm, by contrast, formally FAILS the logit gate (127/131, F 19/21,
one top-1 short on four prompts, #186) at ~0% speed: it is dominated on every axis. Top-1 agreement is 47/48 in every arm on
every divergence prompt; nothing grows. At audio-length
context the raw arm is also clean (#171): on a 300-id prompt scored on HF fp16's own states, the
20 long-context states give shipped KL 0.0008 top-1 19/20 vs H2=1 KL 0.0005 top-1 20/20 -- the
fp16 range exposure flagged on one conv layer shows no measurable cost at 300+ positions. int8
activations were considered and rejected without building: sm_53 has no int8 dot-product
instruction, so there is no FMA-count win, only smaller x traffic that the L2 already
serves -- the mechanism does not exist on this board.

Scope: this engine is a text-backbone harness -- it takes token ids, so audio→tokens runs
upstream and all 8 gate prompts are text-only. TTFT here is in-engine e2e.

Context is 4096 positions (raised from a benchmark-sized 64 that silently truncated anything
longer). A 300-token prompt runs to completion with per-step cost rising 98 → 112 ms;
the KV cache at 4k costs 192 MB against 1.1 GB free. Per-step cost is essentially flat to
300 positions, so a 1500-token audio prefill is ~170 s sequential -- which is why batching
the prefill matters ~70x more for audio than for the 23-token benchmark.

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

`ffn.w2` (N=2048, K=6144) cost 1.773 ms where `ffn.w1` (N=6144, K=2048) cost 1.161 ms for an
identical 12.6 MB. Five hypotheses were implemented and measured against it — block fill
(R=64, shape-adaptive R), inter-row ILP (`#pragma unroll 2`), cache hints (`__ldcs`), chunk
interleaving (KCHUNK=3072), and per-lane row ownership (a one-row-per-lane wide-K GEMV) — and
**all five are measured out**. The launch-bounds codegen narrowed the same-byte w2/w1 ratio
from 1.47× to 1.17× (per-kernel timing under the new allocation), so the new codegen helped
the long-K shape most — but w2 remains the single costliest GEMV, and the narrowed ratio does
not reopen any of the five: they address the old 1.47× signature, and R/KCHUNK re-tunes under
the new codegen both failed (R=128 −2.7%, KCHUNK=1024 −5.5%). The strided-lane form, per-lane 32-block stride + warp tree, is
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
 -2.7  │ R=128 under launch-bounds codegen          re-tune after the codegen change; R=256 stands
 -3.0  │ shape-adaptive R (64/32 for few-row shapes)
 -3.6  │ TPB=512
 -5.3  │ KCHUNK=1024 (also −5.5% under launch bounds; KCHUNK=K stands under all three codegens)
 -6.5  │ R=64 (global)                              x-stage amortization dominates
 -6.9  │ exact integer int8→float (skip I2F)        extra integer op + FADD outweigh the saved I2F
-10.5  │ inter-row ILP (unroll 2)                   register spill at R=256/TPB=256
-12.5  │ -maxrregcount=32 (force 8 blocks/SM)       forced spills outweigh the occupancy gain
-157   │ one-row-per-lane wide-K GEMV               32 scattered sectors per warp instruction
-162   │ double-buffered lm_head streaming          19.8 → 52.5 ms; Tegra copy engine + events
   0   │ launch bounds (256,5) on the Q8 GEMV       compiler keeps 39 regs; identical codegen, null
   0   │ launch bounds (256,8) on the lm_head GEMV  30 regs both builds; bandwidth-bound, null
```

Two configuration findings worth carrying forward:

* **Do not stream the lm_head from pinned host on Tegra.** The resident tied embedding beats
  the plan's prescribed `--emb-host` row streaming by **10.4%** (139.19 vs 169.65 ms/token,
  paired 3/3). The H2D copy and the GEMV share the same 21.9 GB/s bus, so streaming costs
  *two* full passes over 278 MB. The plan's `--emb-host` requirement was sized for the
  1944 MB whole-backbone-Q8_0 config; this Q8-embedding blob is 1466 MB and fits resident with
  ~2.8 GB device free, so streaming is pure loss. `LFM_EMB=resident|host` A/Bs it.
* **GEMV config is a characterised optimum**: R=256, TPB=256, KCHUNK=K, FMA on, serial row
  loop, plain loads, `__launch_bounds__(256,6)` — confirmed on all three codegen regimes
  (pre-FMA, FMA, launch-bounds). TPB=512 cannot be re-tested without touching the
  annotation (max 256 threads is baked in).

## Quality gates

```
bash .auto/lfm3/checks.sh A B C E F K L
```

| gate | current |
|---|---|
| per-layer × per-position rms vs HF, 23 layers × 23 positions | all within 3% |
| pos-0 logits | top-1 = HF's (11), corr 0.99998, std 4.564 vs 4.547 |
| **131 positions × 128,000 logits, 6 prompts** | **top-1 = HF's on 130/131**, top-8 coverage 1.000 everywhere, corr 0.99946–0.99973, KL 0.0007–0.0019 |
| greedy stream vs HF goldens, 7 prompts, 16 tokens | **6/7 exact** (F 11/16, at its known near-tie); floor stays 1 |

```
prompt    top-1    top-8 cov    corr (min)          KL mean   KL max
A         23/23      1.000     0.99964 (0.99900)    0.0014    0.0084
C         21/21      1.000     0.99946 (0.99770)    0.0015    0.0161
E         23/23      1.000     0.99968 (0.99897)    0.0017    0.0121
F         20/21      1.000     0.99963 (0.99884)    0.0015    0.0079
K         23/23      1.000     0.99973 (0.99932)    0.0007    0.0029
L         23/23      1.000     0.99966 (0.99921)    0.0019    0.0126
```

**Read this honestly.** Streams are **6/7 exact over 16 tokens** (F stops at 11/16, at its
known near-tie). `CHECKS OK` still certifies the arm's floor (1 token, the plan's certified
worst prefix for this arm) — the floor was not raised to chase 6/7.

The 6/7 is recent: until the feeding fix the score was 0/7 with prefixes of 1–2, and two
session conclusions were built on that number. Both were wrong, and both are retracted here:
the streams did not diverge by argmax/tie amplification on an agreeing distribution — the
step loop fed every prompt step's prediction back as generation input, so only the first
output token was ever right. And the plan's greedy-goldens gate is not unreachable by a
from-scratch GEMV after all: with correct feeding this Q8 arm reaches its certified 6/8
figure (D remains unscored, refused for device-dependent ties -- see below). The logit gate
was right all along (130/131 top-1, KL under 2e-3): the distributions agreed; the harness
was feeding the wrong tokens.

The one logit miss is prompt F at a single position, where HF's top-1 is still in the engine's
top-8: a near-tie, which is what the "one lost position warns, two fails" floor exists to
surface. Prompt A alone would never have shown it.

**The logit gate is still the reason to distrust the stream gate as a fine instrument.** The
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

One structural fact about the goldens, measured: **they are not reproducible even by HF across
devices.** Regenerating golden D on GPU from its own stored prompt diverges at token 6
(22 vs 20) — oracle.py generates on CPU and bf16 logit ties break per device/kernel.
C/E/F/K/L reproduce exactly (24/24, 24/24, 24/24, 17/17, 24/24) and carry verified
`prompt_ids`; D is refused by `.auto/lfm3/make_prompt_ids.py`, which writes ids only when
(a) rebuilding A/B from their stored text equals their stored ids and (b) HF greedy
reproduces the golden's own `gen`. (An earlier version of this section claimed the goldens
were unreachable by a from-scratch GEMV; the feeding fix retracted that — 6/7 is the arm's
certified figure.)

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
  (Superseded in the best sense: the launch-bounds win moved the best from 136.30 to 112.79,
  and the post-win numbers reproduce to 0.05%.)
* Prompt D cannot be scored (see above), and the stream gate therefore covers 7 of 8 prompts.
* `--trace` (the per-layer/per-position matrix) cannot run inside a graph capture, so
  `LFM_GRAPHS` forces it off.
