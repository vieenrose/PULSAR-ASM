# HANDOFF — Bonsai (Qwen3 + ternary) runtime on the Pi 4

Branch **`bonsai-rpi4`** @ `05f7469` (tracks `fork/bonsai-rpi4`), tree clean.
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
| SwiGLU C reference | `kernels/neon_ops.c` `silu_mul_f32` — 5.8e-08 vs double reference, tails exact |
| tokenizer front-end | `tools/qwen_ids.py` — `"The capital of France is"` → `785 6722 315 9625 374`; `--chat` emits the Qwen3 template |

Engine substeps landed, each with the Gemma gate green (`L0 IDENTICAL`,
`IDS IDENTICAL`) and `bonsai_b3.bin` refusing cleanly (exit 2):

```
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

1. **dtype-aware loading** — `find_tensor` (core.S:909) must also read each
   tensor's `dtype` string from the blob header (`B3_128` vs `F32`) so ternary
   tensors are distinguishable from bf16. Keep the 56-byte tab entry stride if
   possible; a parallel type array is cheaper than touching every offset.
2. **wire `asm/ternary_gemv.S`** (already at C parity) into q/k/v/o, gate/up/down,
   the embedding row gather and the tied head. `gemv_bf16` (core.S:328) is the
   call shape to mirror. Prefer the mask-based add path (ternary is memory-bound).
3. **attention generalisation** — `rope_half` (core.S:1680) and
   `attn_scores_f32` (core.S:1806) plus the two forward bodies (G_SQRT uses at
   1189/1349, G_HIDB at 1163/1345): kv head = `j/(n_head/n_kv)`, rope pairs
   `hd/2`, scores over `hd`, **no** sliding window (`lo=0`, `n=pos+1`), θ from
   `G_ROPE_T`, scale `1/sqrt(hd)`.
4. **FL two-region layout** — HID-wide (input/post-attn/final) + HD-wide (128)
   q/k norms. Gemma's rule is `(1+6·NLAY) × HID` with full-attn at `i mod 6 == 5`;
   Qwen3 needs 4 norms/layer of which two are 128-wide. Buffer already sized
   (`FL` 200k floats, core.S:2595).
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

Build (the x86 box cannot assemble aarch64 — always build on the Pi):

```sh
scp pulsar_arm/asm/core.S pi:core_X.S
ssh pi 'as -o core_X.o core_X.S && ld -static -o core_X core_X.o && echo BUILD_OK'
ssh pi './core_X $MODEL $V/vocab.bin 2 > /tmp/gX.log 2>&1; echo exit=$?'
# gate: diff L0 block + bench ids vs the previous /tmp/g*.log, then grep METRIC
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
