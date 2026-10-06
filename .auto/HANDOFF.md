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
5. **raw-ids mode + `G_TOPK`** — `i` suffix = ids already tokenised (pipe
   `tools/qwen_ids.py`); cells `G_TEMP`/`G_TOPP` exist (defaults 1.0 / 0.95,
   G_GENCAP 128) but top-k is hardcoded 64 — the Bonsai card needs 20 with
   temp 0.5 / top_p 0.85.
6. **oracle diff → demo** — per (layer, position) vs `/tmp/oracle_test.log`
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
- Expected on the Pi at the measured ~3.9 GB/s ceiling: **~99 ms/token 1.7B**,
  **~238 ms/token 4B**. Gemma reference: 270m 147, 1B 543 (cool box).
- KV cap is now a **disclosed design constant**: 1024 positions → 151 MB per
  cache at 36 layers. 32k positions would need 268 MB per cache — do not "fix"
  this silently.
- `Q2_0` group = 34 B (fp16 scale + 32 B of 2-bit codes, `w = (q−1)·scale`);
  base-3 group = 28 B. A 243-entry decode table is the cheap asm route.

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
