/* g4_host.S - AArch64 NEON host kernels of the Gemma 4 E2B Nano engine.
 *
 * Each function reproduces a g4ref.c loop bit for bit: the same IEEE operations
 * on each element, in the same order, only several elements at a time. No fused
 * operation is introduced where the reference has none (g4ref is built with
 * -ffp-contract=off), and every fmla here stands for an explicit fmaf there.
 *
 *   g4_exp_vec(x, y, n, tab)      y[i] = g4_exp(x[i])                n even
 *   g4_geglu(g, u, n, tab)        g[i] = gelu(g[i]) * u[i]           n % 4 == 0
 *   g4_attn_scores(q, K, n, hd, s) s[t] = fmaf chain over d of K[t][d]*q[d]
 *   g4_attn_pv(p, V, n, hd, o)    o[d] = fmaf chain over t of p[t]*V[t][d], o starts at 0
 *
 * tab: the exp/GELU constants, each duplicated into a 16-byte pair, built by
 * g4eng.c from the very definitions g4_exp() and gelu() use:
 *   [0] 1/13!  [1..13] EXP_C[0..12]  [14] LOG2E  [15] SH  [16] LN2HI  [17] LN2LO
 *   [18] 709.0  [19] -745.0  [20] +inf  [21] GELU_C1  [22] GELU_C3  [23] 1.0
 */
    .text
    .arch armv8-a

/* g4_exp on two doubles. In: \t. Out: \p. Temps: \k \r \mhi \mlo \mnn \tmp \c.
 * Constants: v16 LOG2E, v17 SH, v18 LN2HI, v19 LN2LO, v20 709, v21 -745, v22 +inf,
 * v26 = 1023 (int64 lanes); x9 = tab. Same sequence as g4ref.c g4_exp:
 *   kd = (x*LOG2E + SH) - SH; r = (x - kd*LN2HI) - kd*LN2LO;
 *   p = 1/13!; p = p*r + C[i] (13 times, separate mul and add);
 *   k = (int)kd; k1 = k/2 (toward zero); k2 = k - k1; result (p*2^k1)*2^k2;
 *   x > 709 -> +inf, x < -745 -> +0, NaN -> x (lanes selected by masks). */
.macro EXP2 t, p, k, r, mhi, mlo, mnn, tmp, c
    fcmgt   v\mhi\().2d, v\t\().2d, v20.2d
    fcmgt   v\mlo\().2d, v21.2d, v\t\().2d
    fcmeq   v\mnn\().2d, v\t\().2d, v\t\().2d
    fmul    v\k\().2d, v\t\().2d, v16.2d
    fadd    v\k\().2d, v\k\().2d, v17.2d
    fsub    v\k\().2d, v\k\().2d, v17.2d
    fmul    v\tmp\().2d, v\k\().2d, v18.2d
    fsub    v\r\().2d, v\t\().2d, v\tmp\().2d
    fmul    v\tmp\().2d, v\k\().2d, v19.2d
    fsub    v\r\().2d, v\r\().2d, v\tmp\().2d
    ldr     q\p, [x9]
    .irp i, 1,2,3,4,5,6,7,8,9,10,11,12,13
    fmul    v\p\().2d, v\p\().2d, v\r\().2d
    ldr     q\c, [x9, #(16*\i)]
    fadd    v\p\().2d, v\p\().2d, v\c\().2d
    .endr
    fcvtzs  v\k\().2d, v\k\().2d
    ushr    v\tmp\().2d, v\k\().2d, #63
    add     v\tmp\().2d, v\k\().2d, v\tmp\().2d
    sshr    v\tmp\().2d, v\tmp\().2d, #1            /* k1 = k/2, toward zero */
    sub     v\k\().2d, v\k\().2d, v\tmp\().2d        /* k2 = k - k1 */
    add     v\tmp\().2d, v\tmp\().2d, v26.2d
    shl     v\tmp\().2d, v\tmp\().2d, #52
    fmul    v\p\().2d, v\p\().2d, v\tmp\().2d
    add     v\k\().2d, v\k\().2d, v26.2d
    shl     v\k\().2d, v\k\().2d, #52
    fmul    v\p\().2d, v\p\().2d, v\k\().2d
    bit     v\p\().16b, v22.16b, v\mhi\().16b       /* x > 709  -> +inf */
    bic     v\p\().16b, v\p\().16b, v\mlo\().16b     /* x < -745 -> +0   */
    bif     v\p\().16b, v\t\().16b, v\mnn\().16b     /* NaN      -> x    */
.endm

.macro LOAD_EXP_CONSTS
    ldr     q16, [x9, #16*14]
    ldr     q17, [x9, #16*15]
    ldr     q18, [x9, #16*16]
    ldr     q19, [x9, #16*17]
    ldr     q20, [x9, #16*18]
    ldr     q21, [x9, #16*19]
    ldr     q22, [x9, #16*20]
    mov     x10, #1023
    dup     v26.2d, x10
.endm

.macro SAVE_D8_D15
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]
.endm
.macro RESTORE_D8_D15
    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    ldp     d8, d9, [sp], #64
.endm

/* void g4_exp_vec(const double *x, double *y, long n, const double *tab), n even > 0 */
    .global g4_exp_vec
    .type g4_exp_vec, %function
    .align 4
g4_exp_vec:
    SAVE_D8_D15
    mov     x9, x3
    LOAD_EXP_CONSTS
1:  ldr     q4, [x0], #16
    EXP2    4, 8, 9, 10, 11, 12, 13, 27, 14
    str     q8, [x1], #16
    subs    x2, x2, #2
    b.gt    1b
    RESTORE_D8_D15
    ret
    .size g4_exp_vec, .-g4_exp_vec

/* void g4_geglu(float *g, const float *u, long n, const double *tab), n % 4 == 0 > 0
 * per element, as gelu(): x = (double)g; t = -(C1*x + C3*((x*x)*x));
 * y = x / (1.0 + exp(t)); g = (float)y * u. Four elements per iteration, as two
 * independent two-lane streams (A: lanes 0-1, B: lanes 2-3). */
    .global g4_geglu
    .type g4_geglu, %function
    .align 4
g4_geglu:
    SAVE_D8_D15
    mov     x9, x3
    LOAD_EXP_CONSTS
    ldr     q23, [x9, #16*21]                 /* C1 */
    ldr     q24, [x9, #16*22]                 /* C3 */
    ldr     q25, [x9, #16*23]                 /* 1.0 */
1:  ldr     q0, [x0]
    fcvtl   v2.2d, v0.2s                      /* xA */
    fcvtl2  v3.2d, v0.4s                      /* xB */
    fmul    v4.2d, v2.2d, v2.2d
    fmul    v4.2d, v4.2d, v2.2d               /* (x*x)*x */
    fmul    v4.2d, v24.2d, v4.2d              /* C3 * x^3 */
    fmul    v5.2d, v23.2d, v2.2d              /* C1 * x */
    fadd    v4.2d, v5.2d, v4.2d
    fneg    v4.2d, v4.2d                      /* tA */
    fmul    v6.2d, v3.2d, v3.2d
    fmul    v6.2d, v6.2d, v3.2d
    fmul    v6.2d, v24.2d, v6.2d
    fmul    v7.2d, v23.2d, v3.2d
    fadd    v6.2d, v7.2d, v6.2d
    fneg    v6.2d, v6.2d                      /* tB */
    EXP2    4, 8, 9, 10, 11, 12, 13, 27, 14
    EXP2    6, 28, 29, 30, 31, 15, 5, 7, 1
    fadd    v8.2d, v8.2d, v25.2d              /* 1.0 + e */
    fdiv    v2.2d, v2.2d, v8.2d
    fadd    v28.2d, v28.2d, v25.2d
    fdiv    v3.2d, v3.2d, v28.2d
    fcvtn   v0.2s, v2.2d
    fcvtn2  v0.4s, v3.2d                      /* (float) gelu */
    ldr     q1, [x1], #16
    fmul    v0.4s, v0.4s, v1.4s               /* * u */
    str     q0, [x0], #16
    subs    x2, x2, #4
    b.gt    1b
    RESTORE_D8_D15
    ret
    .size g4_geglu, .-g4_geglu

/* void g4_attn_scores(const float *q, const float *K, long n, long hd, float *s)
 * Eight rows (cached positions) at a time, as two groups of four: 4x4 tiles are
 * transposed so each vector lane is one row's running dot, then advanced with
 * fmla by element (one fmaf per row per d, d ascending). Rows left over (< 8)
 * run the scalar fmadd chain. hd % 4 == 0. */
    .global g4_attn_scores
    .type g4_attn_scores, %function
    .align 4
g4_attn_scores:
    SAVE_D8_D15
    lsl     x5, x3, #2                        /* row bytes */
8:  cmp     x2, #8
    b.lt    5f
    mov     x6, x1                            /* row 0 */
    add     x7, x6, x5
    add     x8, x7, x5
    add     x10, x8, x5                       /* row 3 */
    add     x11, x10, x5
    add     x12, x11, x5
    add     x13, x12, x5
    add     x14, x13, x5                      /* row 7 */
    mov     x15, x0                           /* q */
    mov     x16, x3                           /* d counter */
    movi    v16.4s, #0
    movi    v22.4s, #0
1:  ldr     q4, [x15], #16                    /* q[d..d+3] */
    ldr     q0, [x6], #16
    ldr     q1, [x7], #16
    ldr     q2, [x8], #16
    ldr     q3, [x10], #16
    ldr     q24, [x11], #16
    ldr     q25, [x12], #16
    ldr     q26, [x13], #16
    ldr     q27, [x14], #16
    trn1    v5.4s, v0.4s, v1.4s
    trn2    v6.4s, v0.4s, v1.4s
    trn1    v7.4s, v2.4s, v3.4s
    trn2    v17.4s, v2.4s, v3.4s
    trn1    v18.2d, v5.2d, v7.2d              /* column d   of rows 0-3 */
    trn1    v19.2d, v6.2d, v17.2d             /* column d+1 */
    trn2    v20.2d, v5.2d, v7.2d              /* column d+2 */
    trn2    v21.2d, v6.2d, v17.2d             /* column d+3 */
    trn1    v8.4s, v24.4s, v25.4s
    trn2    v9.4s, v24.4s, v25.4s
    trn1    v10.4s, v26.4s, v27.4s
    trn2    v11.4s, v26.4s, v27.4s
    trn1    v12.2d, v8.2d, v10.2d             /* column d   of rows 4-7 */
    trn1    v13.2d, v9.2d, v11.2d
    trn2    v14.2d, v8.2d, v10.2d
    trn2    v15.2d, v9.2d, v11.2d
    fmla    v16.4s, v18.4s, v4.s[0]
    fmla    v22.4s, v12.4s, v4.s[0]
    fmla    v16.4s, v19.4s, v4.s[1]
    fmla    v22.4s, v13.4s, v4.s[1]
    fmla    v16.4s, v20.4s, v4.s[2]
    fmla    v22.4s, v14.4s, v4.s[2]
    fmla    v16.4s, v21.4s, v4.s[3]
    fmla    v22.4s, v15.4s, v4.s[3]
    subs    x16, x16, #4
    b.gt    1b
    stp     q16, q22, [x4], #32
    add     x1, x1, x5, lsl #3                /* K += 8 rows */
    sub     x2, x2, #8
    b       8b
5:  cbz     x2, 9f                            /* scalar rows */
    mov     x6, x1
    mov     x15, x0
    mov     x16, x3
    fmov    s0, wzr
6:  ldr     s1, [x6], #4
    ldr     s2, [x15], #4
    fmadd   s0, s1, s2, s0                    /* fmaf(k[d], q[d], s) */
    subs    x16, x16, #1
    b.gt    6b
    str     s0, [x4], #4
    add     x1, x1, x5
    subs    x2, x2, #1
    b.gt    5b
9:  RESTORE_D8_D15
    ret
    .size g4_attn_scores, .-g4_attn_scores

/* void g4_attn_pv(const float *p, const float *V, long n, long hd, float *o)
 * o[d] = 0, then o[d] = fmaf(p[t], V[t][d], o[d]) for t = 0..n-1 ascending.
 * 32 output lanes (8 accumulators) per pass over t; hd % 32 == 0, n > 0. */
    .global g4_attn_pv
    .type g4_attn_pv, %function
    .align 4
g4_attn_pv:
    lsl     x5, x3, #2                        /* row bytes */
    mov     x9, #0                            /* byte offset of this 32-lane chunk */
1:  movi    v16.4s, #0
    movi    v17.4s, #0
    movi    v18.4s, #0
    movi    v19.4s, #0
    movi    v20.4s, #0
    movi    v21.4s, #0
    movi    v22.4s, #0
    movi    v23.4s, #0
    add     x6, x1, x9                        /* V[0] + chunk */
    mov     x7, x0                            /* p */
    mov     x8, x2                            /* t counter */
2:  ld1r    {v0.4s}, [x7], #4
    ldp     q1, q2, [x6]
    ldp     q3, q4, [x6, #32]
    ldp     q5, q6, [x6, #64]
    ldp     q7, q24, [x6, #96]
    fmla    v16.4s, v1.4s, v0.4s
    fmla    v17.4s, v2.4s, v0.4s
    fmla    v18.4s, v3.4s, v0.4s
    fmla    v19.4s, v4.4s, v0.4s
    fmla    v20.4s, v5.4s, v0.4s
    fmla    v21.4s, v6.4s, v0.4s
    fmla    v22.4s, v7.4s, v0.4s
    fmla    v23.4s, v24.4s, v0.4s
    add     x6, x6, x5
    subs    x8, x8, #1
    b.gt    2b
    add     x10, x4, x9
    stp     q16, q17, [x10]
    stp     q18, q19, [x10, #32]
    stp     q20, q21, [x10, #64]
    stp     q22, q23, [x10, #96]
    add     x9, x9, #128
    cmp     x9, x5
    b.lt    1b
    ret
    .size g4_attn_pv, .-g4_attn_pv

/* void g4_attn_scores8(const float *qT, const float *K, long n, long hd, float *S)
 * All 8 query heads at once (Gemma 4 has one kv head). qT is the queries
 * transposed, [hd][8]; S[t*8 + h] = fmaf chain over d ascending of K[t][d]*q_h[d].
 * Vector lanes are heads: for each d, acc(h) = fma(acc(h), qT[d][h], K[t][d]).
 * Four rows t per pass (8 accumulators), so each K row is read once for all heads.
 * hd % 4 == 0. */
    .global g4_attn_scores8
    .type g4_attn_scores8, %function
    .align 4
g4_attn_scores8:
    SAVE_D8_D15
    lsl     x5, x3, #2                        /* K row bytes */
4:  cmp     x2, #4
    b.lt    5f
    mov     x6, x1
    add     x7, x6, x5
    add     x8, x7, x5
    add     x10, x8, x5
    mov     x11, x0                           /* qT */
    mov     x12, x3
    movi    v16.4s, #0
    movi    v17.4s, #0
    movi    v18.4s, #0
    movi    v19.4s, #0
    movi    v20.4s, #0
    movi    v21.4s, #0
    movi    v22.4s, #0
    movi    v23.4s, #0
1:  ldr     q0, [x6], #16                     /* K[t0][d..d+3] */
    ldr     q1, [x7], #16
    ldr     q2, [x8], #16
    ldr     q3, [x10], #16
    ld1     {v4.4s-v7.4s}, [x11], #64         /* qT[d], qT[d+1] (lo, hi) */
    ld1     {v8.4s-v11.4s}, [x11], #64        /* qT[d+2], qT[d+3] */
    .irp j, 0,1,2,3
    .if \j == 0
    mov     v24.16b, v4.16b
    mov     v25.16b, v5.16b
    .elseif \j == 1
    mov     v24.16b, v6.16b
    mov     v25.16b, v7.16b
    .elseif \j == 2
    mov     v24.16b, v8.16b
    mov     v25.16b, v9.16b
    .else
    mov     v24.16b, v10.16b
    mov     v25.16b, v11.16b
    .endif
    fmla    v16.4s, v24.4s, v0.s[\j]
    fmla    v17.4s, v25.4s, v0.s[\j]
    fmla    v18.4s, v24.4s, v1.s[\j]
    fmla    v19.4s, v25.4s, v1.s[\j]
    fmla    v20.4s, v24.4s, v2.s[\j]
    fmla    v21.4s, v25.4s, v2.s[\j]
    fmla    v22.4s, v24.4s, v3.s[\j]
    fmla    v23.4s, v25.4s, v3.s[\j]
    .endr
    subs    x12, x12, #4
    b.gt    1b
    st1     {v16.4s-v19.4s}, [x4], #64        /* S[t0][0..7], S[t1][0..7] */
    st1     {v20.4s-v23.4s}, [x4], #64
    add     x1, x1, x5, lsl #2
    sub     x2, x2, #4
    b       4b
5:  cbz     x2, 9f                            /* one row at a time */
    mov     x6, x1
    mov     x11, x0
    mov     x12, x3
    movi    v16.4s, #0
    movi    v17.4s, #0
6:  ld1r    {v0.4s}, [x6], #4                 /* K[t][d] */
    ld1     {v4.4s, v5.4s}, [x11], #32        /* qT[d] */
    fmla    v16.4s, v4.4s, v0.4s
    fmla    v17.4s, v5.4s, v0.4s
    subs    x12, x12, #1
    b.gt    6b
    st1     {v16.4s, v17.4s}, [x4], #32
    add     x1, x1, x5
    subs    x2, x2, #1
    b.gt    5b
9:  RESTORE_D8_D15
    ret
    .size g4_attn_scores8, .-g4_attn_scores8

/* void g4_attn_pv8(const float *P, const float *V, long n, long hd, float *O)
 * All 8 heads at once: P is [t][8]; O[h*hd + d] = 0, then
 * O[h][d] = fmaf(P[t][h], V[t][d], O[h][d]) for t ascending. Eight dims per pass
 * (16 accumulators: 8 heads x 2 vectors), so each V row is read once for all
 * heads. hd % 8 == 0. */
    .global g4_attn_pv8
    .type g4_attn_pv8, %function
    .align 4
g4_attn_pv8:
    SAVE_D8_D15
    lsl     x5, x3, #2                        /* V row bytes = O head stride */
    mov     x9, #0                            /* byte offset of this 8-dim chunk */
    mov     x11, x5                           /* end offset: all hd dims */
    b       .Lpv8_chunk
    .size g4_attn_pv8, .-g4_attn_pv8

/* void g4_attn_pv8r(const float *P, const float *V, long n, long hd, float *O, long d0, long d1)
 * g4_attn_pv8 restricted to dims [d0, d1) (multiples of 8): the same per-(h, d)
 * chains, so any split of the dims gives the full call's O exactly. */
    .global g4_attn_pv8r
    .type g4_attn_pv8r, %function
    .align 4
g4_attn_pv8r:
    SAVE_D8_D15
    lsl     x9, x5, #2                        /* start offset d0 */
    lsl     x11, x6, #2                       /* end offset d1 */
    lsl     x5, x3, #2                        /* V row bytes = O head stride */
.Lpv8_chunk:
1:  movi    v16.4s, #0
    movi    v17.4s, #0
    movi    v18.4s, #0
    movi    v19.4s, #0
    movi    v20.4s, #0
    movi    v21.4s, #0
    movi    v22.4s, #0
    movi    v23.4s, #0
    movi    v24.4s, #0
    movi    v25.4s, #0
    movi    v26.4s, #0
    movi    v27.4s, #0
    movi    v28.4s, #0
    movi    v29.4s, #0
    movi    v30.4s, #0
    movi    v31.4s, #0
    add     x6, x1, x9                        /* V[0] + chunk */
    mov     x7, x0                            /* P */
    mov     x8, x2
2:  ld1     {v0.4s, v1.4s}, [x6]              /* V[t][d..d+7] */
    ld1     {v2.4s, v3.4s}, [x7], #32         /* P[t][0..7] */
    fmla    v16.4s, v0.4s, v2.s[0]
    fmla    v17.4s, v1.4s, v2.s[0]
    fmla    v18.4s, v0.4s, v2.s[1]
    fmla    v19.4s, v1.4s, v2.s[1]
    fmla    v20.4s, v0.4s, v2.s[2]
    fmla    v21.4s, v1.4s, v2.s[2]
    fmla    v22.4s, v0.4s, v2.s[3]
    fmla    v23.4s, v1.4s, v2.s[3]
    fmla    v24.4s, v0.4s, v3.s[0]
    fmla    v25.4s, v1.4s, v3.s[0]
    fmla    v26.4s, v0.4s, v3.s[1]
    fmla    v27.4s, v1.4s, v3.s[1]
    fmla    v28.4s, v0.4s, v3.s[2]
    fmla    v29.4s, v1.4s, v3.s[2]
    fmla    v30.4s, v0.4s, v3.s[3]
    fmla    v31.4s, v1.4s, v3.s[3]
    add     x6, x6, x5
    subs    x8, x8, #1
    b.gt    2b
    add     x10, x4, x9                       /* O[0][chunk] */
    st1     {v16.4s, v17.4s}, [x10]
    add     x10, x10, x5
    st1     {v18.4s, v19.4s}, [x10]
    add     x10, x10, x5
    st1     {v20.4s, v21.4s}, [x10]
    add     x10, x10, x5
    st1     {v22.4s, v23.4s}, [x10]
    add     x10, x10, x5
    st1     {v24.4s, v25.4s}, [x10]
    add     x10, x10, x5
    st1     {v26.4s, v27.4s}, [x10]
    add     x10, x10, x5
    st1     {v28.4s, v29.4s}, [x10]
    add     x10, x10, x5
    st1     {v30.4s, v31.4s}, [x10]
    add     x9, x9, #32
    cmp     x9, x11
    b.lt    1b
    RESTORE_D8_D15
    ret
    .size g4_attn_pv8r, .-g4_attn_pv8r

/* RMS norm, g4ref.c rms_r + rmsnorm / rmsnorm_noscale:
 *   ss = sum of (double)x[i]*(double)x[i], strictly in order (one fadd chain);
 *   r = sqrt(ss / n + 1e-6);  y[i] = (float)(((double)x[i] / r) * (double)w[i])
 * The squares, divisions and products are done two doubles at a time; the sum
 * stays a sequential scalar chain. n % 4 == 0.
 * The divisions x/r use Markstein's correction instead of FDIV (the A57's double
 * divider costs ~30 cycles per element): y = RN(1/r) once, then per element
 *   q0 = x*y;  e = fma(-q0, r, x) (FMLS, exact residual);  q = fma(e, y, q0) (FMLA)
 * which is the correctly rounded quotient RN(x/r) (Markstein's theorem: y within
 * half an ulp of 1/r, q0 within an ulp of x/r; x, r doubles, so no exact midpoint).
 * Checked bit for bit against x/r on 2.5e11 (float x, double r) pairs, including
 * r with all-ones mantissas.
 * double g4_rms_r(const float *x, long n) -> d0
 * void g4_rmsnorm(const float *x, const float *w, float *y, long n)
 * void g4_rmsnorm_ns(float *x, long n)   (in place, no weight)                 */
    .global g4_rms_r
    .type g4_rms_r, %function
    .align 4
g4_rms_r:
    fmov    d0, xzr                           /* ss = 0 (+0.0, as the C init) */
    mov     x2, x0
    mov     x3, x1
1:  ldr     q1, [x2], #16
    fcvtl   v2.2d, v1.2s
    fcvtl2  v3.2d, v1.4s
    fmul    v2.2d, v2.2d, v2.2d
    fmul    v3.2d, v3.2d, v3.2d
    fadd    d0, d0, d2                        /* x0^2 */
    mov     d4, v2.d[1]
    fadd    d0, d0, d4                        /* x1^2 */
    fadd    d0, d0, d3                        /* x2^2 */
    mov     d4, v3.d[1]
    fadd    d0, d0, d4                        /* x3^2 */
    subs    x3, x3, #4
    b.gt    1b
    scvtf   d1, x1                            /* (double)n */
    fdiv    d0, d0, d1
    adr     x4, .Leps
    ldr     d1, [x4]
    fadd    d0, d0, d1
    fsqrt   d0, d0
    ret
    .size g4_rms_r, .-g4_rms_r
    .align 3
.Leps: .double 1e-6

    .global g4_rmsnorm
    .type g4_rmsnorm, %function
    .align 4
g4_rmsnorm:
    stp     x29, x30, [sp, #-48]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    mov     x19, x0
    mov     x20, x1
    mov     x21, x2
    mov     x22, x3
    mov     x1, x3
    bl      g4_rms_r
    dup     v5.2d, v0.d[0]                    /* r */
    fmov    d17, #1.0
    fdiv    d17, d17, d0
    dup     v17.2d, v17.d[0]                  /* y = RN(1/r) */
1:  ldr     q1, [x19], #16
    ldr     q6, [x20], #16
    fcvtl   v2.2d, v1.2s
    fcvtl2  v3.2d, v1.4s
    fcvtl   v7.2d, v6.2s
    fcvtl2  v16.2d, v6.4s
    fmul    v18.2d, v2.2d, v17.2d             /* q0 = x*y */
    fmul    v19.2d, v3.2d, v17.2d
    fmls    v2.2d, v18.2d, v5.2d              /* e = x - q0*r (fused) */
    fmls    v3.2d, v19.2d, v5.2d
    fmla    v18.2d, v2.2d, v17.2d             /* x/r = q0 + e*y (fused) */
    fmla    v19.2d, v3.2d, v17.2d
    fmul    v2.2d, v18.2d, v7.2d
    fmul    v3.2d, v19.2d, v16.2d
    fcvtn   v1.2s, v2.2d
    fcvtn2  v1.4s, v3.2d
    str     q1, [x21], #16
    subs    x22, x22, #4
    b.gt    1b
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #48
    ret
    .size g4_rmsnorm, .-g4_rmsnorm

    .global g4_rmsnorm_ns
    .type g4_rmsnorm_ns, %function
    .align 4
g4_rmsnorm_ns:
    stp     x29, x30, [sp, #-32]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    mov     x19, x0
    mov     x20, x1
    bl      g4_rms_r
    dup     v5.2d, v0.d[0]
    fmov    d17, #1.0
    fdiv    d17, d17, d0
    dup     v17.2d, v17.d[0]                  /* y = RN(1/r) */
1:  ldr     q1, [x19]
    fcvtl   v2.2d, v1.2s
    fcvtl2  v3.2d, v1.4s
    fmul    v18.2d, v2.2d, v17.2d             /* Markstein, as g4_rmsnorm */
    fmul    v19.2d, v3.2d, v17.2d
    fmls    v2.2d, v18.2d, v5.2d
    fmls    v3.2d, v19.2d, v5.2d
    fmla    v18.2d, v2.2d, v17.2d
    fmla    v19.2d, v3.2d, v17.2d
    fcvtn   v1.2s, v18.2d
    fcvtn2  v1.4s, v19.2d
    str     q1, [x19], #16
    subs    x20, x20, #4
    b.gt    1b
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #32
    ret
    .size g4_rmsnorm_ns, .-g4_rmsnorm_ns

/* void g4_q8k(const float *x, uint8_t *y, long n) - g4ref.c q8k, bit for bit.
 * The K2 engine's NEON quantizer (Exp80, proven bit-exact there), relabelled;
 * same algorithm: max |x| (fmaxnm), the FIRST element holding it gives the signed
 * max, iscale = -127/max, n = (fma(x, iscale, 0x1.8p23) - 0x1.8p23) clamped at 127
 * (equal to the C bit trick: the fma result is an exact integer in that range),
 * d = 1/iscale, 292-byte blocks. n % 256 == 0. Clobbers x0-x8, x10, x11, v0-v5. */
    .global g4_q8k
    .type g4_q8k, %function
    .align 4
g4_q8k:
    cbz x2, .Lg8_ret
    mov x3, #0                      /* block index */
    lsr x8, x2, #8                  /* nb = k/256 */
    mov x7, #292
.Lg8_blk:
    /* pass 1 (Exp80, NEON): amax = max |x|. fmaxnm is exact and ignores NaN,
       like the scalar `fcmp/b.ls` it replaces (amax starts at 0). */
    mov x2, x0
    mov x4, #16
    movi v5.4s, #0
.Lg8_max:
    ldp q0, q1, [x2], #32
    ldp q2, q3, [x2], #32
    fabs v0.4s, v0.4s
    fabs v1.4s, v1.4s
    fabs v2.4s, v2.4s
    fabs v3.4s, v3.4s
    fmaxnm v0.4s, v0.4s, v1.4s
    fmaxnm v2.4s, v2.4s, v3.4s
    fmaxnm v0.4s, v0.4s, v2.4s
    fmaxnm v5.4s, v5.4s, v0.4s
    subs x4, x4, #1
    b.ne .Lg8_max
    fmaxnmv s2, v5.4s               /* amax */
    madd x5, x3, x7, x1             /* yb */
    fcmp s2, #0.0
    b.ne .Lg8_nz
    /* zero block: d=0, qs=0 */
    str wzr, [x5]
    mov x4, #0
    mov x6, #0
.Lg8_z:
    str x6, [x5, x4]
    add x4, x4, #8
    cmp x4, #264                    /* covers d(4)+qs(256) */
    b.ne .Lg8_z
    b .Lg8_next
.Lg8_nz:
    /* s3 = signed value of the FIRST element with |x| == amax (strict-> max) */
    dup v2.4s, v2.s[0]
    mov x2, x0
.Lg8_find:
    ldr q0, [x2], #16
    fabs v1.4s, v0.4s
    fcmeq v1.4s, v1.4s, v2.4s
    umaxv s3, v1.4s
    fmov w10, s3
    cbz w10, .Lg8_find
    umov w10, v1.s[0]
    cbz w10, 1f
    mov s3, v0.s[0]
    b .Lg8_have
1:  umov w10, v1.s[1]
    cbz w10, 2f
    mov s3, v0.s[1]
    b .Lg8_have
2:  umov w10, v1.s[2]
    cbz w10, 3f
    mov s3, v0.s[2]
    b .Lg8_have
3:  mov s3, v0.s[3]
.Lg8_have:
    /* iscale = -127/max */
    mov w10, #0xc2fe0000
    fmov s4, w10                    /* -127.0f */
    fdiv s4, s4, s3
    dup v4.4s, v4.s[0]              /* lane 0 keeps iscale for the d = 1/iscale below */
    mov w10, #0x4b40                /* 12582912.f = 0x4B400000 */
    lsl w10, w10, #16
    dup v1.4s, w10
    mov w10, #0x42fe                /* 127.0f = 0x42FE0000 */
    lsl w10, w10, #16
    dup v2.4s, w10
    add x6, x5, #4                  /* qs */
    mov x2, x0
    mov x4, #32
.Lg8_q:
    /* n = fcvtzs(min((x*iscale + magic) - magic, 127)). The fmla is fused like
       the scalar fmadd; magic+n is exact, so the subtraction recovers n exactly. */
    ldp q0, q3, [x2], #32
    mov v5.16b, v1.16b
    fmla v5.4s, v0.4s, v4.4s
    fsub v5.4s, v5.4s, v1.4s
    fmin v5.4s, v5.4s, v2.4s
    fcvtzs v5.4s, v5.4s
    mov v0.16b, v1.16b
    fmla v0.4s, v3.4s, v4.4s
    fsub v0.4s, v0.4s, v1.4s
    fmin v0.4s, v0.4s, v2.4s
    fcvtzs v0.4s, v0.4s
    xtn v5.4h, v5.4s
    xtn2 v5.8h, v0.4s
    xtn v5.8b, v5.8h
    str d5, [x6], #8
    subs x4, x4, #1
    b.ne .Lg8_q
    /* d = 1.0/iscale */
    fmov s0, #1.0
    fdiv s0, s0, s4
    str s0, [x5]
.Lg8_next:
    add x11, x5, #260                /* bytes 260..291 unused: zero, as g4ref q8k */
    stp xzr, xzr, [x11]
    stp xzr, xzr, [x11, #16]
    add x3, x3, #1
    add x0, x0, #1024
    cmp x3, x8
    b.ne .Lg8_blk
.Lg8_ret:
    ret
    .size g4_q8k, .-g4_q8k

    .section .note.GNU-stack,"",%progbits
