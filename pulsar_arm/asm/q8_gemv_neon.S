/* q8_gemv_neon.S -- Q8_0 GEMV dot product, AArch64 NEON, for Cortex-A57.
 *
 * Target: Jetson Nano Gen1. Assembled with `as` (fasm 1.73.02 has no arm64
 * backend -- it emits pe64/elf64 only), per the pulsar_arm/asm house style.
 *
 * Why: decode is bandwidth-bound on the Q8_0 GEMVs (LFM3-Audio-2B streams
 * 1.558 GB of weights per token) and they run on the Nano's ONE SM at
 * 6.8 GB/s (kg) / 9.05 GB/s (staged) while all 4 Cortex-A57 cores idle. A
 * hand-written NEON GEMV is the prerequisite for splitting layers across CPU
 * and GPU; naive C measured 0.43-0.53 GB/s on 4 threads, so -O3 is not the
 * path.
 *
 *   float q8_dot_row(const int8_t *q, const float *s, const float *x, long K)
 *     x0=q  x1=s  x2=x  x3=K  -> s0.   K is a multiple of 32.
 *
 * REGISTER CONTRACT (AAPCS64) -- the low 64 bits of v8-v15 (d8-d15) are
 * CALLEE-SAVED. An earlier version used v10-v16 as the widening temporaries
 * and clobbered d10-d15. Symptom: ZERO effect on the direct-call parity gate,
 * then libgomp died at the first parallel region with "Out of memory
 * allocating 766920482040 bytes" -- the saved FP state it restored was
 * garbage. A kernel can be bit-exact under every gate in this file and still
 * destroy its caller. Only v0-v7 and v16-v31 are free.
 *
 *   v0-v3                x[0..15] of the current Q8_0 block
 *   v16                  16 raw int8 codes
 *   v17,v18              int8 -> int16
 *   v19,v23,v28,v29      int16 -> int32 -> float (4 lanes each)
 *   v24,v25,v26,v27      accumulator groups a0..a3
 *   v4,v5,v6             reduction fold
 *   v7 (s7)              fmul temp
 *   v30 (s30)            block scale
 *   v31 (s31)            running row accumulator
 *
 * REDUCTION ORDER -- the contract, pinned deliberately. A 32-term sequential
 * sum cannot be vectorized (4 lane accumulators are a tree, not a sequence),
 * so the order must be chosen, not inherited; the repo's convention
 * (pq1_swar_sm53.cu:19-22) is that the reference defines it and every engine
 * matches. This order is what NEON produces naturally, so the fast path costs
 * no shuffles:
 *
 *   per Q8_0 block b, 32 products in 4 accumulator groups, 4 lanes each:
 *     a_g lane i = q[4g+i]*x[4g+i]  THEN  += q[16+4g+i]*x[16+4g+i]
 *   (two terms, ascending, second term fused -- it is an fmla)
 *   fold 1 (vector): A = a0+a2,  B = a1+a3,  C = A+B        elementwise
 *   fold 2 (lanes):  D0 = C0+C2, D1 = C1+C3, blocksum = D0+D1
 *   blocks accumulate sequentially, scale applied to the block sum, UNFUSED
 *   (no scalar fmla exists on binutils 2.30, and the reference spells it the
 *   same way): acc = acc + s[b] * blocksum(b)
 *
 * The gate reference in q8_gemv_neon_test.c mirrors this operation for
 * operation, using fmaf() at exactly the two fused sites. A CUDA kernel
 * written to the same contract is bit-identical to this -- which is the
 * point: a hybrid engine must not let the CPU and GPU paths drift.
 */
    .arch armv8-a+fp
    .text
    .p2align 4
    .globl q8_dot_row
    .type  q8_dot_row, %function
q8_dot_row:
    cbz     x3, .Lzero
    lsr     x4, x3, #5                    /* nb = K/32 */
    mov     x5, xzr                       /* b */
    fmov    s31, wzr                      /* acc */
.Lblk:
    movi    v24.16b, #0
    movi    v25.16b, #0
    movi    v26.16b, #0
    movi    v27.16b, #0

    /* ---- codes 0..15: one 16-byte load feeds the whole widening chain.
     * q = 16 BYTES, d = 8: using d here leaves the upper half stale and
     * sxtl2 then converts garbage (measured: 1e+87 results). ---- */
    ldr     q16, [x0]
    sxtl    v17.8h, v16.8b                /* codes 0..7  -> int16 */
    sxtl2   v18.8h, v16.16b               /* codes 8..15 -> int16 */
    sxtl    v19.4s, v17.4h                /* codes 0..3  -> int32 */
    sxtl2   v23.4s, v17.8h                /* codes 4..7  -> int32 */
    sxtl    v28.4s, v18.4h                /* codes 8..11 -> int32 */
    sxtl2   v29.4s, v18.8h                /* codes 12..15 -> int32 */
    scvtf   v19.4s, v19.4s                /* int8 -> float is exact */
    scvtf   v23.4s, v23.4s
    scvtf   v28.4s, v28.4s
    scvtf   v29.4s, v29.4s
    ldr     q0, [x2]                      /* x[0..3] */
    ldr     q1, [x2, #16]
    ldr     q2, [x2, #32]
    ldr     q3, [x2, #48]
    fmla    v24.4s, v19.4s, v0.4s         /* a0 lane i = q[i]*x[i] */
    fmla    v25.4s, v23.4s, v1.4s         /* a1 lane i = q[4+i]*x[4+i] */
    fmla    v26.4s, v28.4s, v2.4s         /* a2 lane i = q[8+i]*x[8+i] */
    fmla    v27.4s, v29.4s, v3.4s         /* a3 lane i = q[12+i]*x[12+i] */

    /* ---- codes 16..31 ---- */
    ldr     q16, [x0, #16]
    sxtl    v17.8h, v16.8b
    sxtl2   v18.8h, v16.16b
    sxtl    v19.4s, v17.4h
    sxtl2   v23.4s, v17.8h
    sxtl    v28.4s, v18.4h
    sxtl2   v29.4s, v18.8h
    scvtf   v19.4s, v19.4s
    scvtf   v23.4s, v23.4s
    scvtf   v28.4s, v28.4s
    scvtf   v29.4s, v29.4s
    ldr     q0, [x2, #64]
    ldr     q1, [x2, #80]
    ldr     q2, [x2, #96]
    ldr     q3, [x2, #112]
    fmla    v24.4s, v19.4s, v0.4s         /* a0 lane i += q[16+i]*x[16+i] */
    fmla    v25.4s, v23.4s, v1.4s
    fmla    v26.4s, v28.4s, v2.4s
    fmla    v27.4s, v29.4s, v3.4s

    /* fold 1: (a0+a2) + (a1+a3), elementwise */
    fadd    v4.4s, v24.4s, v26.4s
    fadd    v5.4s, v25.4s, v27.4s
    fadd    v4.4s, v4.4s, v5.4s           /* C = [C0,C1,C2,C3] */
    /* fold 2: (C0+C2) + (C1+C3). The mnemonic is ext, not vext; a shift of 2
     * floats is 8 bytes, of 1 float is 4. */
    ext     v5.16b, v4.16b, v4.16b, #8
    fadd    v4.4s, v4.4s, v5.4s           /* lane0 = C0+C2, lane1 = C1+C3 */
    ext     v5.16b, v4.16b, v4.16b, #4
    fadd    v4.4s, v4.4s, v5.4s           /* lane0 = (C0+C2)+(C1+C3) */

    /* acc += s[b] * blocksum. x1 is NOT advanced per block: the scale is
     * indexed by b, so bumping the base too made block b read s[2b] (every
     * odd block got a scale belonging elsewhere -- 0 in the one-block-alive
     * probe, garbage past the array end at random data). */
    ldr     s30, [x1, x5, lsl #2]
    fmul    s7, s4, s30
    fadd    s31, s31, s7
    add     x0, x0, #32
    add     x2, x2, #128
    add     x5, x5, #1
    cmp     x5, x4
    blt     .Lblk
    fmov    s0, s31
    ret
.Lzero:
    fmov    s0, wzr
    ret
    .size q8_dot_row, .-q8_dot_row

    .section .note.GNU-stack,"",@progbits
