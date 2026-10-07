# HANDOFF — Bonsai (Qwen3 + ternary) runtime on the Pi 4

Branch **`bonsai-rpi4`** @ `946dd19` (tracks `fork/bonsai-rpi4`; the `auto:` handoff
commits land on top of it), tree clean.
`gemma-3-rpi4` is the Gemma-only branch (tip `84e3fe0`); last Gemma-only commit
before the Bonsai prep is `c91433a` if strict separation is ever wanted.

## Objective

Run prism-ml **Ternary-Bonsai 1.7B and 4B** (Qwen3 arch, ternary weights) in the
pure-asm engine on the Pi 4, exactly as Gemma-3 270m/1B run today — then find
each checkpoint's **ceiling prompt** and present it in the README with a demo GIF.

## Verified state (do not re-derive)

| item | evidence |
|---|---|
| GGUF `Q2_0` → lossless base-3 blob | all 140 oracle block signatures diff **0.0** vs Q2_0; GEMM level too |
| blob size | 1.7B **376.8 MB**, 4B **880.6 MB** (82.4 % of Q2_0; 5 trits/byte, 26 B codes + 2 B fp16 scale per 128-weight group) |
| base-3 layout | byte = Σ c_i·3^i, LSB-first, 5 trits/byte, last byte holds 3 trits |
| Python oracle | `tools/q2_0_ref.py`, both dtypes; argmax `12095` = `' Paris'` on 1.7B **and** 4B |
| ternary GEMV C reference | `kernels/ternary_gemv.c` — 4.4e-06…5.2e-05 vs oracle on 8 fixtures |
| ternary GEMV asm | `asm/ternary_gemv.S` — **bit-identical to C** on 8 fixtures (`tests/test_ternary_asm.c`, `tests/mk_syn_ternary.py`, PTGV fixtures) |
| ternary row gather asm | `embed_row_b3` in the same file — 0 bit-diff vs the C mirror and vs an independent fp64 dequantise (`tests/test_ternary_row.c`) |
| arch dispatch in the engine | `gemv_arch` / `embed_arch`: gemma3 reaches the same kernels with the same arguments (L0 + bench ids diff-empty at every step) |
| SwiGLU C reference | `kernels/neon_ops.c` `silu_mul_f32` — 5.8e-08 vs double reference, tails exact |
| tokenizer front-end | `tools/qwen_ids.py` — `"The capital of France is"` → `785 6722 315 9625 374`; `--chat` emits the Qwen3 template |

Engine substeps landed, each with the Gemma gate green (`L0 IDENTICAL`,
`IDS IDENTICAL`) and `bonsai_b3.bin` refusing cleanly (exit 2):

```
946dd19 engine: ternary embedding row gather (embed_row_b3) + its test
68dd461 engine: ternary GEMV at the tied head
4c35949 engine: ternary GEMV at the MLP projections (gate/up/down)
d75cae3 engine: ternary GEMV linked into the build, dispatched at q/k/v/o
05f7469 auto: METRIC drift resolved by same-job A/B (drift, not regression)
dafd0d4 engine: SwiGLU for qwen3 (silu_mul_f32 + arch branch at the 2 call sites)
84e3fe0 engine: arch-dependent norm fold (plain w vs 1+w, in fold_vec)
69449d4 engine: loader reads pulsar.* geometry via pget -> G_* cells
09def5e engine: loader reads pulsar.arch_id, refuses non-Gemma blobs cleanly
```

## Gates (unchanged, apply to every commit)

1. Gemma 270m regression: `L0` signature block **and** bench greedy ids diff
   empty vs the previously committed build, exit 0.
2. Non-Gemma blob: clean refusal, exit 2 — never a silent mis-run.
3. Never leave `core.S` half-edited: additive substep → build → gate → commit,
   else `git checkout --`.

## Remaining work (the demo path)

1. **~~dtype-aware loading~~ — not needed, measured.** The 1.7B blob holds 310
   tensors: **197 `B3_128`, all 2-D weight matrices, and 113 `F32`, all shaped
   `[n,1]` — and those 113 are exactly the norms** (4 per layer x 28 + the final
   `model.norm.weight`: 28x4+1 = 113). So the discriminator is free: inside an
   `arch_id==1` blob, `shape[1] == 1` means norm (F32), everything else is
   ternary. `find_tensor` already parses shape, so **no asm dtype parser is
   required**; reading the `dtype` string is optional belt-and-braces. Note the
   embedding is ternary too (`embed_tokens.weight` `[151669, 2048]` B3_128) — it
   is a row gather, so a ternary gather, not a GEMV. Metadata also carries
   `pulsar.ternary_group: 128` and `pulsar.pack: 1`.
2. **~~wire `asm/ternary_gemv.S`~~ — done (commits `d75cae3..946dd19`).** Every
   weight read in both forward bodies dispatches on `G_ARCH` now: `gemv_arch`
   (same args as `gemv_bf16`) tail-branches to `gemv_bf16` for gemma3 and calls
   `ternary_gemv_b3` for a blob; `embed_arch` does the same for the embedding,
   where the ternary path is `embed_row_b3` (a row gather — a GEMV over the tied
   table would sweep 151669 rows for one token). Row counts moved from literals
   to cells so both arches are honest: `G_QDIM`/`G_KVDIM` default to the gemma3
   1024/256 and the blob branch recomputes them as `n_head*head_dim` /
   `n_kv*head_dim`; `pulsar.ternary_group != 128` now refuses (exit 2) instead
   of mis-reading every group. **`core.S` ends with `.include "ternary_gemv.S"`**, so
   a build needs both files in the build directory (see the recipe below) and the
   engine assembles the same bytes the kernel test does. Still unmeasured: speed
   — this is the reference scalar decode, and the mask-based/243-entry-table
   path from the plan is the follow-up that decides whether ~99 ms/token holds.
3. **attention generalisation** — DONE, on `feat/qwen3-attention`, four substeps:
   `6c4cab1` geometry cells + `geom_cells`, `6b983d8` rope tables from the cells
   (+ `ln_f64`, `lay_full`), `a814ffb` `kcache_row`, `aba5295` head loop + GQA
   (`head_off`, per-kv-head K norm/rope, row-stride argument in both attn
   kernels), `c815ef0` `rmsadd_arch` (gemma3 keeps the normed residual, qwen3 gets
   a plain add). Gemma byte-identical at every step, timing unchanged
   (169-171 vs 171-173 ms/token same-job). Everything the forward path needs is a
   cell now: `G_NHEAD G_NKV G_HD G_HHALF G_HDB G_QDIM G_KVDIM G_GROUP G_ASCALE
   G_WIN G_FULLMOD G_KVCAP G_TROW G_KVROW G_KSTRIDE`.
4. **FL two-region layout + loader slot addressing** — DONE, `feat/qwen3-attention`:
   `86f2e5b` (G_TSLOT/WK/NKIND + `wslot`), `68c7f26` (`FLTAB` built by
   fold_norms; `nptr`/`wptr` deleted), `c30a5c6` (**correction**: the tensor table
   is filled by NAME into the canonical slots `2 + 13*layer + kind`, embedding at
   0 and final norm at 1, **for both formats** — a qwen3 blob is that same table
   with kinds 8/12 (pre/post_feedforward_layernorm) absent, and q_norm/k_norm at
   kinds 1/2 where the forward already looks. The per-arch maps of `86f2e5b` were
   wrong and made the first blob run NaN).
   Arch-1 FL layout: `FL + (1+2i)*HIDB` = input_layernorm, `FL + (2+2i)*HIDB` =
   post_attention_layernorm (also the pre-MLP norm, which is what qwen3 means),
   `G_SFL + 2i*HDB` / `+(HD*4)` = q_norm/k_norm with `G_SFL = FL + (1+2*NLAY)*HIDB`.
   1.7B needs 123,904 of FL's 200,000 floats, 4B needs 196,096 — 2% headroom.
   Gemma stayed byte-identical at every one of the three commits.

   **Step 5 status, first scratch run (refusal lifted with `b f_stage_ret`, not
   committed):** the engine gets to the forward and prints the L0 block, but `X`
   (the embedding) is garbage from element 0 — `max=9.2e18`, values mixing 3.9e15
   with 0.0087 — then H/FL-derived values are 0 and everything after is NaN, and
   the run dies with SIGBUS later. Checked and cleared already: the data base is
   right (`file 376763327 - (8+38647) = 376724672 = max data_offset`, so no
   padding and `base+8+N` is correct), the embedding row stride is right
   (448 B = 16 groups × 28), the tables print the right geometry, and the slot
   map is the canonical one. So the next probe is `embed_row_b3` on a *real* blob
   row against `tools/q2_0_ref.py`'s embedding for the same token id — the kernel
   has only ever been validated on synthetic fixtures, never on `bonsai_b3.bin`.
   Worth checking in the same pass: whether `embed_arch` really dispatches to the
   b3 row gather for arch 1 (a bf16 read of a 448-byte packed row would look
   exactly like this), and the `G_N`/`G_base` the blob branch stores.
5. **first blob run** — the arch-1 refusal is still in place (gemma gates green,
   `bonsai_b3.bin` exits 2), but lifting it in a scratch build now runs the whole
   model, and **layer 0 is validated stage by stage against python** at pos 0 with
   prompt `2,107,1567,85096,107,304,2505,9694` (note `def_prompt`'s 236765 is out
   of range for qwen's 151669 vocab - use 85096 for blob runs):
   X/H/AV/POST/X1/GG/D/X2 all match `q2_0_ref.py`'s arithmetic to fp32 accumulation
   order (`H 0.9861995 vs 0.9861994`, `AV 0.9563199 vs 0.9563196`,
   `X2 10.6656312 vs 10.6656303`). At pos 0 attention is exactly the kv-head value
   vector tiled over its query heads, so a pos-0 match validates the norms, both
   GEMV paths, silu and the GQA tiling, but **not** rope or the softmax.
   Two bugs found by that first run, both fixed in `bb20079`, both unreachable
   from a gemma run and therefore invisible to every gate that ever ran:
   - the arch dispatchers did `adrp x8, G_ARCH; ldr w8, [x8]` with **no
     `add x8, x8, :lo12:G_ARCH`** - the page base reads 0, which is gemma3's
     answer, so every arch branch silently took the gemma path. `tools/check_adrp.py`
     now checks the class (0 sites on the fixed file, 3 on the old one).
   - `fold_vec` chose its addend by arch but always loaded bf16 (`ldrh`+`lsl #16`).
     Gemma3's norms are bf16; a blob stores them as **F32**, so arch 1 decoded
     every norm at twice the density - `[w0,w1,w2,...]` became `[0,w0,0,w1,...]`.
     Still to fix (cosmetic): the `in_norm0[0..3]` debug print has the same
     bf16-blindness and runs before the arch is parsed, so it prints `0 w0 0 w1`
     for a blob; a comment says so, the fold itself is right.
   **Speed, unsolved: 7070-7730 ms/token** on the blob (target ~99). The ternary
   decode is the scalar reference; nothing has been attempted on it yet.
   Next: oracle8 (`/tmp/oracle8.log`, ids above) for layers 1+ and pos > 0, then
   the generated-id comparison; rope and softmax are the untested parts.
6. **raw-ids mode + `G_TOPK`** — `i` suffix = ids already tokenised (pipe
   `tools/qwen_ids.py`); cells `G_TEMP`/`G_TOPP` exist (defaults 1.0 / 0.95,
   G_GENCAP 128) but top-k is hardcoded 64 — the Bonsai card needs 20 with
   temp 0.5 / top_p 0.85. Until this exists, blob comparisons have to go through
   `def_prompt` patched in a scratch build, which is what step 5 did.
7. **oracle diff → demo** — per (layer, position) vs `/tmp/oracle_test.log`
   (1.7B, 140 DX blocks) and `/tmp/oracle4b_b3.log` (4B, 180 blocks), then the
   ceiling-prompt hunt, GIFs (DejaVu 17 / line height 23, frames = real engine
   bytes, transcripts byte-diffed), README entry.

## Pi inventory (`luigi@raspberrypi.tailf63b31.ts.net`)

- repo clone `~/PULSAR-ARM` (its working tree is deliberately left on the old
  commit; push/pull only what a test needs), venv `~/pulsar-rpi/v/bin/python`
- blobs `~/bonsai_b3.bin`, `~/bonsai4b_b3.bin` (+ Q2_0 originals)
- Gemma model `~/.cache/huggingface/hub/models--google--gemma-3-270m-it-qat-q4_0-unquantized/snapshots/8f726c6a497fd439f0d6f726e52f8e3b439a26e5/model.safetensors`
- vocab `~/PULSAR-ARM/pulsar_arm/{vocab,bpe,fcvocab,fcbpe}.bin`
- tools copies `~/q2_0_ref.py`, `~/q2_0_gguf.py`, `~/qwen_ids.py`
- oracle logs `/tmp/oracle_test.log`, `/tmp/oracle4b_b3.log`; fixtures `/tmp/tv_*.tv`, `/tmp/syn_*.tv`
- binaries `core_p5` (pre-arch reference), `core_arch{2..6}`, `test_tgv`, `test_silu`
- build/gate work dir `~/pw` (used by the recipe below) with `bin_gemma.sh` /
  `bin_bonsai.sh` wrappers and the kernel tests `test_ternary_row`, `test_tgv3`

Build (the x86 box cannot assemble aarch64 — always build on the Pi). `core.S`
now `.include`s `ternary_gemv.S`, so **copy both**; a missing include fails
loudly at `as`, never silently.

```sh
PI=luigi@raspberrypi.tailf63b31.ts.net
M=/home/luigi/.cache/huggingface/hub/models--google--gemma-3-270m-it-qat-q4_0-unquantized/snapshots/8f726c6a497fd439f0d6f726e52f8e3b439a26e5/model.safetensors
V=/home/luigi/PULSAR-ARM/pulsar_arm
mkdir -p pw && scp pulsar_arm/asm/core.S $PI:pw/core_X.S && scp pulsar_arm/asm/ternary_gemv.S $PI:pw/ternary_gemv.S
ssh $PI 'cd pw && as -o core_X.o core_X.S && ld -static -o core_X core_X.o && echo BUILD_OK'
ssh $PI "cd pw && ./core_X $M $V/vocab.bin 2 > /tmp/gX.log 2>&1; echo exit=\$?"
# gate: diff L0 block + bench ids vs the previous /tmp/g*.log (ignore the timing
# lines), then the blob refusal: ./core_X ~/bonsai_b3.bin <vocab> 2 -> exit 2
# kernel tests: gcc -O2 -o t test_ternary_row.c ternary_gemv.c ternary_gemv.S -lm
#               ./t  and  ./t /tmp/tv_*.tv /tmp/syn_*.tv
```

## Facts worth knowing

- 1.7B: hidden 2048, inter 6144, 28 layers, 16 q / 8 kv × 128, θ 1e6, vocab 151669.
  4B: 2560 / 9728 / 36, 32 q / 8 kv × 128, θ **5e6**.
- **Speed floor, measured 2026-10-06 (supersedes the bandwidth-only estimate).**
  Peak 4-lane fp32 `fmla` on this Pi is **7.0 G fma-lanes/s** (3.9/cycle at
  1.8 GHz - that is A72's single vector-FMA pipe, so it is the machine peak, not
  a weak benchmark). A ternary weight still needs one add, so touching every
  weight once costs **243 ms/token for 1.7B and 571 ms for 4B** no matter how
  good the kernel gets. The storage numbers (376 MB → 96 ms, 880 MB → 226 ms at
  3.93 GB/s) are *below* that, so **ternary is compute-bound, not
  bandwidth-bound** - unlike gemma, where the opposite is true. The old line
  here ("~99 ms/token 1.7B, ~238 ms 4B") was pure storage arithmetic and is
  unreachable; ~250 ms and ~570 ms are the walls. Gemma reference: 270m 147,
  1B 543 (cool box).
- The demo path works end to end (2026-10-07): `2 i` takes qwen ids on stdin
  (raw-ids mode, `3f48752`), the sampler now scopes to G_VOC instead of the
  262144-slot buffer (`5952f1f` - it was returning token 0 forever), top-k is
  per-arch (64 gemma3 / 20 Bonsai), end-of-turn is per-arch (1 / 151643 /
  151645), and `tools/mkv_qwen.py` + the existing put_surf table path render
  qwen text with no new asm decoder (`011a1ab`). Sampling config was checked
  against the checkpoint's generation_config.json, not the card's prose:
  temperature 0.5, top_k 20, top_p 0.85, eos 151645, repetition_penalty 1.0.
  `tools/bonsai_ask.py` prints a transcript for the GIF specs.
- **Sampler: one open corner.** Verified with instrumentation (printing raw
  logits, post-softmax values, top-k slots and the chosen index from inside the
  sampler): logits are clean, softmax is correct at temp 0.5 and 1.0
  (post-softmax max 0.54 / 0.45, peaked as expected), top-k sorts descending,
  the top-p walk picks j=0 with v0=1.0 when the distribution is peaked, and
  `expf_asm` only ever touches s0-s4/x9/x10 so it cannot corrupt the max or the
  temperature held in s6/s7. Greedy (sampler bypassed entirely) produces a
  perfect four-item answer. What does NOT add up: at temp 0.1/0.05 the run
  degenerates into token repetition ("are are are"), and the sampled top-1 at
  the first position does not match the greedy argmax, which it must when
  v0 ~ 1.0. At the card's temp 0.5 the output is genuinely good - a clean haiku
  ("Tides roll in, / waves whisper secrets, / the sea sings.") and a well-formed
  zh-TW answer - so the demos use 0.5 and this is a corner, not a blocker.
  Beware two red herrings the instrumentation produced: a `kept` that looked
  like garbage was my stack-offset mislabel, and a "post = nan" reading came
  from a different call than the one being examined.
- **4B: FIXED (`723fe62`). Root cause was one missing header key.** The 4B blob
  carries `pulsar.rope_theta` (5e6) but not `pulsar.rope_theta_int`, and the
  engine only looked for the _int spelling, so G_ROPE_T stayed 0. `ln_f64(0)`
  returns exactly -1023*ln2 = -709.0895 (exponent field zero), which made
  inv[1] = exp(709/64) = 64830 instead of 0.78583, so every RoPE frequency was
  garbage: first sample valid, all later ones NaN, model answering "!" forever.
  The tell was the engine's own log line `blob ln(theta): -709.0895385`. Fixed
  by falling back to the float key (pget stops at '.', so 5000000.0 parses
  exactly) plus a loud warning, guarded on arch != 0 because gemma3 has no rope
  key and never reads the blob tables. 4B now answers the four-seasons prompt
  correctly; small stray-token artifacts ("1!") remain to look at.
  **Lesson worth keeping: a silently-zero geometry cell produced a
  plausible-looking ln, a plausible-looking inv table, and a model that just
  repeated "!". Warn on missing keys instead of defaulting to 0.**
- **4B: NaN is born in LAYER 0 at position 2.** (superseded by the fix above) A 4-id run shows p0/p1
  finite and p2/p3 already NaN at DX L0 - the first layer, not the deep stack.
  So it is L0's attention at n=3 (norms/rope are position-independent, the
  embedding is a table read, so the candidates are the rope row for pos 2, the
  score/softmax over 3 keys, and the KV row written for pos 2). Next: print the
  L0 attention stage maxima inside fwd_token (q after rope -> scores -> softmax
  -> AV) for tokens 0..3 and see which stage first goes NaN. This supersedes the
  earlier "attention/KV at n>=3" note only in that it is one layer earlier and
  therefore a much smaller search: no layer-1..35 involvement at all.
- **4B: the forward goes bad from position 2.** Instrumenting the sampler on 4B
  showed the FIRST sample is valid (tok=220, j=2, kept=12, val0=0.132,
  idx0=279) and every later one reads val0 = -1e30, i.e. the softmax is looking
  at NaN. p0 and p1 are fine (36 sane DX layers each, logits max 6.0-8.8), so
  this is an attention/KV bug at n >= 3 that the earlier DX validation could not
  see, because it only compared p0 and p1. Chase it by running the oracle on a
  3-4 id prompt (see below) and diffing p2 onward.
- **4B is NOT validated.** /tmp/oracle4b.log is garbage: every layer prints the
  same value (3.0e21) with numpy overflow warnings, i.e. the run fed an
  out-of-range prompt id and never reached the forward properly. Before any 4B
  clip, re-run q2_0_ref.py on a 2-id in-vocab prompt and diff DX signatures the
  way 1.7B was done.
- Ternary GEMV, done 2026-10-07 (`8deeb5e`, 3.7x end to end, 1888 ms/token).
  Scalar was 3.98 ns/weight (pw2/tgb predicted the 7079 ms token within 3%);
  `ternary_gemv_b3_v` (asm/ternary_gemv.S, prototype V6 in tools/) does 0.91 and
  is wired into gemv_arch's arch-1 branch. All 8 validation argmaxes re-earned,
  gemma byte-identical, bonsai ids identical. The fp64 oracle agrees with fp32
  on all 8 argmaxes including p3 (margin 0.0171), which is what green-lit the
  Q-fold's 4.7e-6. Still open: asm row-blocking (lost in C to spilling, needs a
  hand-rolled register plan to judge), and the bit-plane repack fork (~350 ms
  needs +425 MB resident and a README story change - user's call, deferred).
- KV cap is now a **disclosed design constant**: 1024 positions → 151 MB per
  cache at 36 layers. 32k positions would need 268 MB per cache — do not "fix"
  this silently.
- `Q2_0` group = 34 B (fp16 scale + 32 B of 2-bit codes, `w = (q−1)·scale`);
  base-3 group = 28 B. The 243-entry decode table is **not** the route: the
  partial-sum table depends on the byte's position in x, so it is 416 positions
  × 256 entries = 425 KB per token - all L2 misses at 26 lookups per group per
  row. The route that fits is digit extraction by magic multiply (`sum_i d_i·x_i
  − S_g`, with x deinterleaved per token into digit-plane vectors, ~10 KB and L1
  resident), which is ~0.4 ops/weight instead of today's ~10.

## Lessons (this session, the expensive kind)

- Save **x30** before calling helpers; fp16→fp32 needs mantissa `lsl #13`.
- Do **not** global-rename labels in `core.S` — a blanket `k_rope` rename hit a
  pre-existing self-test symbol and broke the build. New labels: `pk_`/`pq_`/
  `tp_`/`ep_`, identified by payload not name.
- Absolute ms/token are meaningless across hours: interleave both binaries in one
  job. The 170 vs 147 scare resolved as drift (`core_p5` itself read 169-170).
- Keep commit messages honest — say when something does not pass yet.
- **`d8`–`d15` (s8–s15) are callee-saved** in the AArch64 ABI. `embed_row_b3`
  parked its `escale` argument in `s8`; gcc had hoisted the loop-invariant
  `(float)escale` into `d8` and the C reference silently received the kernel's
  leftover group scale — wrong by exactly a factor of it. Use s0–s7 in kernels,
  or spill to a stack slot. Caught by asm-vs-C *bit*-diff, not by value tolerance.
- A dispatcher that `bl`s instead of tail-`b`s also needs its own x30 saved
  (`gemv_arch`/`embed_arch` do). Same class of bug as the kernel's first draft.
- **`nptr`/`wptr` work in x0/x1.** Building a call's `out` pointer before one of
  those calls loses it. In the GQA commit this made `fwd_layer0` write the
  normalised K over `FL[3]` — layer 0's k_norm weights — which the *bench run
  that happens afterwards* then used. Every L0 signature line still matched,
  because softmax over one key makes the scores irrelevant at pos 0. The bench
  ids line caught it; the signature block could not.
- Related: the L0 debug block is value-blind by design (it prints the same buffer
  for `norm(x)*(1+w)` and `rms(x)*w`), so it cannot see a wrong norm kind, a wrong
  n, or a wrong fold. Gate on the ids.
- When adding `add xN, xN, xM` to an `adrp`-loaded symbol address, the
  `add xN, xN, :lo12:SYM` must still be there — `adrp` alone is page-aligned.
  Six sites did that in one commit and every one of them was silent. Grep after
  writing: `adrp xN, F[A-Z]+` with no `:lo12:` before the next `bl`.
