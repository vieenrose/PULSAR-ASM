# Fork f16-KV flash-attention: exact arithmetic the Nano engine must reproduce

Reference = PrismML fork `6bfcd79` on Spark (`~/bonsai2/llama.cpp`), CPU build `-mcpu=native`
under GCC 13.3, which resolves to plain ARMv8 + FMA: **no SVE, no fp16 vector arithmetic**
(objdump of vec.cpp.o / ops.cpp.o: 0 `fmla .8h`, 0 SVE regs). So every f16 op goes through
`GGML_F32Cx4` (widen f16->f32, fp32 FMA, round back RNE). `llama-server -c 4096 -ctk f16 -ctv f16`;
`-fa auto` resolves to **FA on** on CPU. FA on vs off diverge at zh token 36, so FA on is the reference.

## Path selection (ggml_compute_forward_flash_attn_ext_f16)
- query rows in a ubatch `N >= 64` (GGML_FA_TILE_Q) -> **tiled** path (prefill ubatches; server n_ubatch 512).
- `N == 1` and padded n_kv `nek1 >= 512` -> **split-KV**: KV range cut into `nth` chunks, fp32 partials
  merged by `reduce_partials`. **Thread-count dependent.** With `-t 1` it is one chunk and the merge is an
  exact identity (expf(-inf)=0, x*1), i.e. identical to one_chunk. Goldens are therefore made with `-t 1`.
- otherwise -> **one_chunk** (decode with n_kv < 512, short prefills like prompt A's 20 rows).

## one_chunk (decode) per query head
- Q (fp32) -> f16 RNE.  s = vec_dot_f16(K_f16[ic], Q_f16) for n=128: 16 fp32 FMA chains,
  chain c = elements {c, c+16, ..., c+112}, acc = fma(x, y, acc) in order; chains grouped as
  sum[j][l] (j = c/4, l = c%4); reduce sum0+=sum2, sum1+=sum3, sum0+=sum1, then vaddvq_f32
  = (l0+l1)+(l2+l3); widened to double then stored to float (exact).
- s *= scale; s += mask (0 / -inf; -inf entries skipped).
- online softmax over ic ascending with libm expf:
  if s > M: M=s, ms=expf(Mold-M), VKQ16[d] = f16(f32(VKQ16[d]) * ms)    (vs=1)
  else:     vs = expf(s-M)
  VKQ16[d] = f16(fma(f32(V_f16[ic][d]), vs, f32(VKQ16[d])))           (elementwise)
  S = S*ms + vs   (fp32)
- out[d] = f32(VKQ16[d]) * (1/S)    (S_inv = 1.0f/S, then multiply)

## tiled (prefill ubatch, N >= 64) per query row
- Q stays fp32; K, V widened f16->f32.
- KV tiles of 64 from ic = 0 to nek1 (padded); tile skipped if fully masked for the whole Q tile
  (a row with all -inf in a computed tile is a no-op: P zeroed, fma(V,0,acc)=acc).
- KQ[kv] = sequential fp32 FMA chain over d=0..127 of Q[d]*K[kv][d], from 0; then *scale; then +mask.
- tile_max = sequential max; Mnew = fmaxf(M, tile_max); if Mnew > M: ms = expf(M-Mnew) (libm),
  VKQ32 *= ms, S *= ms.  M = Mnew.
- P = ggml_v_expf(KQ - Mnew) (NEON polynomial, 4 lanes; code in vec.h:1133), S += double sum where each
  4-group contributes (double)vaddvq_f32 = (p0+p1)+(p2+p3) and S(float) = float(double(S) + sum).
- VKQ32[d] = fma chain continued across tiles: for tk in 0..63: acc = fma(V[tk][d], P[tk], acc).
- out = VKQ32 * (1.0f/S).

## Goldens
- prompt zh: `.auto/prompt_zh_tw.txt` (3189 text tokens, 3197 with chat template), greedy top_k=1,
  `-t 1 -fa on -c 4096 -ctk f16 -ctv f16`. Spark scripts: `~/bonsai2/zh_srv3.sh`, `zh_gen.py`.
- prompt A with f16 KV (fork, -t 20 and -t 1) reproduces `.auto/golden_q1_A.txt` generated tail (25 tokens).
