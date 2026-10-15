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

    .section .note.GNU-stack,"",%progbits
