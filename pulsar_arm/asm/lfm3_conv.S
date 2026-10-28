/* lfm3_conv.S -- LFM2/LFM3 gated short conv, decode step, AArch64 NEON.
 *
 * Target: Jetson Nano Gen1 (4x Cortex-A57, armv8-a + fp). Assembled with
 *   as -o lfm3_conv.o lfm3_conv.S
 * per the pulsar_arm/asm house style (as + ld -static, no libc in the kernel).
 *
 * Why this file: 13 of LFM3's 24 layers are this conv, not attention, and
 * nothing like it exists anywhere in this repo (Gemma and Bonsai have no conv
 * path), so it is written from scratch against pulsar_asm/lfm3/tools/
 * ref_lfm3_np.py -- which is itself verified stage-by-stage against HF's own
 * conv sub-modules (probe_lfm3_conv.py: in_proj, B/C/xg split, B*xg, conv,
 * out_proj all match to bf16 rounding).
 *
 * Math (modeling_lfm3.py Lfm3ShortConv; width 7, no bias in this checkpoint):
 *   BCx = in_proj(h)                       split into B, C, xg in that order
 *   y   = C * causal_conv1d(B * xg)        C = OUTPUT gate, xg = conv INPUT gate
 *
 * This kernel is the conv + output gate for ONE decode position (T=1) -- the
 * hot path of a token runtime. The gated in_proj split and out_proj GEMM are
 * the surrounding engine's job.
 *
 *   void lfm3_conv_step(float *out, const float *state,
 *                       const float *wT, const float *Cgate, long H)
 *       x0=out  x1=state  x2=wT  x3=Cgate  x4=H
 *   void lfm3_conv_shift(float *state, long H)
 *       x0=state  x1=H
 *
 * TAP-MAJOR layout: state is [7][H] and wT is [7][H], both channel-contiguous.
 * Deliberate -- with the checkpoint's native [H][7] weight the per-channel
 * weight is a stride-7 scalar, so a lane-over-channels vector load is
 * impossible and every tap needs a broadcast. Tap-major makes all 14 streams
 * contiguous in the channel index: the inner loop is 2 vector loads + 1 FMA
 * per tap per 4 channels, no broadcasts, no strided accesses. The load-time
 * repack converts conv_w [H,7] -> wT [7][H] once; state is maintained here.
 *
 * FMA ORDER: taps are accumulated k = 0..6 ascending, matching the scalar
 * reference exactly, so this is bit-identical to it (no reassociation).
 */
    .arch armv8-a+fp
    .text

/* ------------------------------------------------------------------ shift
 * void lfm3_conv_shift(float *state, long H)
 * state[0..5] = state[1..6]; slot 6 is left for the caller to fill with the
 * new B*xg. Shift and update are separate contiguous passes, so neither does
 * per-element strided work.
 */
    .p2align 4
    .globl lfm3_conv_shift
    .type  lfm3_conv_shift, %function
lfm3_conv_shift:
    cbz     x1, .Lsh_done
    lsl     x2, x1, 2                 /* bytes per tap = H*4 */
    mul     x3, x2, xzr               /* keep x2; x3 scratch */
    mov     x4, #6
    mul     x3, x2, x4                /* 6*H bytes to move */
    add     x6, x0, x2                /* src = state + 1 tap */
    mov     x5, x0                    /* dst = state        (dst < src, so a
                                       * forward copy is the correct overlap
                                       * direction; dst was mistakenly src+1
                                       * tap first, which copies the block
                                       * onto itself = a silent no-op shift) */
.Lsh_64:
    subs    x3, x3, #64
    blt     .Lsh_16
    ldp     q0, q1, [x6]
    ldp     q2, q3, [x6, #32]
    stp     q0, q1, [x5]
    stp     q2, q3, [x5, #32]
    add     x5, x5, #64
    add     x6, x6, #64
    b       .Lsh_64
.Lsh_16:
    adds    x3, x3, #64
    ble     .Lsh_done
    /* Remainder is 0..63 bytes and always a multiple of 4 (the block is 6*H
     * floats). A 16-byte q copy here over-reads/over-writes by up to 15 bytes
     * -- it corrupted the heap on the H=1 case. Copy 4 bytes at a time. */
.Lsh_4:
    ldr     s0, [x6]
    str     s0, [x5]
    add     x5, x5, #4
    add     x6, x6, #4
    subs    x3, x3, #4
    bgt     .Lsh_4
.Lsh_done:
    ret

/* ------------------------------------------------------------- conv + gate
 * void lfm3_conv_step(float *out, const float *state,
 *                     const float *wT, const float *Cgate, long H)
 * Precondition: the caller ran lfm3_conv_shift and stored B*xg into state[6].
 *   out[c] = Cgate[c] * sum_{k=0..6} wT[k][c] * state[k][c]
 *
 * 4 channels per iteration. Live at any time: 2 tap vectors + 2 weight vectors
 * + 1 accumulator + 1 gate, so no register pressure and no spills on A57.
 */
    .p2align 4
    .globl lfm3_conv_step
    .type  lfm3_conv_step, %function
lfm3_conv_step:
    cbz     x4, .Lcs_done
    lsl     x5, x4, 2                 /* x5 = tap stride bytes = H*4 */
    /* x10 = byte limit for the 4-wide loop = 4*floor(H/4) channels. Bounding
     * by H*4 instead lets a partial block run when H%4!=0: at H=7 it writes
     * channels 4..7, one float past `out` -- that was the heap corruption. */
    lsr     x10, x4, #2
    lsl     x10, x10, #2
    lsl     x10, x10, #2
    mov     x6, xzr                   /* x6 = c*4, byte offset of channel c */
    mov     x9, xzr                   /* x9 = one-shot flag for the H%4 tail */
.Lcs_blk:
    cmp     x6, x10
    bge     .Lcs_tail
.Lcs_body:

    /* v0 = 0.0f x 4 (accumulator). movi v0.16b, #0 is the canonical zeroing
     * form; fmov d,fpmov are longer and were the source of an earlier bug. */
    movi    v0.16b, #0

    /* k = 0 */
    ldr     q1, [x1, x6]
    ldr     q2, [x2, x6]
    fmla    v0.4s, v1.4s, v2.4s
    /* k = 1..6 */
    add     x7, x1, x5
    add     x8, x2, x5
    ldr     q1, [x7, x6]
    ldr     q2, [x8, x6]
    fmla    v0.4s, v1.4s, v2.4s

    add     x7, x1, x5, lsl #1
    add     x8, x2, x5, lsl #1
    ldr     q1, [x7, x6]
    ldr     q2, [x8, x6]
    fmla    v0.4s, v1.4s, v2.4s

    add     x7, x1, x5
    add     x7, x7, x5, lsl #1
    add     x8, x2, x5
    add     x8, x8, x5, lsl #1
    ldr     q1, [x7, x6]
    ldr     q2, [x8, x6]
    fmla    v0.4s, v1.4s, v2.4s

    add     x7, x1, x5, lsl #2
    add     x8, x2, x5, lsl #2
    ldr     q1, [x7, x6]
    ldr     q2, [x8, x6]
    fmla    v0.4s, v1.4s, v2.4s

    add     x7, x1, x5, lsl #2
    add     x7, x7, x5
    add     x8, x2, x5, lsl #2
    add     x8, x8, x5
    ldr     q1, [x7, x6]
    ldr     q2, [x8, x6]
    fmla    v0.4s, v1.4s, v2.4s

    add     x7, x1, x5, lsl #2
    add     x7, x7, x5, lsl #1
    add     x8, x2, x5, lsl #2
    add     x8, x8, x5, lsl #1
    ldr     q1, [x7, x6]
    ldr     q2, [x8, x6]
    fmla    v0.4s, v1.4s, v2.4s

    /* output gate: out[c] = Cgate[c] * acc[c] */
    ldr     q1, [x3, x6]
    fmul    v0.4s, v0.4s, v1.4s
    str     q0, [x0, x6]

    add     x6, x6, #16
    cbnz    x9, .Lcs_done               /* overlap recompute runs exactly once
                                         * (without this the tail re-entered the
                                         * body and the two looped forever) */
    b       .Lcs_blk

.Lcs_tail:
    /* H % 4 != 0. The scalar tail used fmul+fadd (two roundings) while the
     * vector path uses fmla (one), so it can never be bit-identical to it,
     * and binutils 2.30 on the Nano rejects the `fmaf' mnemonic. So re-run
     * the VECTOR body once at channel H-4: an overlapping recompute of the
     * last 4 channels. Same instructions, same fused semantics, every access
     * in bounds, and the recomputed values are identical -- no second
     * numeric path to keep in sync. */
    lsr     x7, x4, #2
    lsl     x7, x7, #2
    cmp     x7, x4
    bge     .Lcs_done
    cmp     x4, #4
    blt     .Lcs_scalar
    lsl     x6, x4, #2
    sub     x6, x6, #16                 /* (H-4)*4 */
    mov     x9, #1
    b       .Lcs_body

    /* H < 4: no full vector block exists, so this tiny path stays scalar.
     * It is UNFUSED (fmul+fadd) and therefore last-ulp from the fused
     * reference -- documented, and unreachable for LFM3 (H=2048) and for any
     * real head_dim. The gate reports it as such. */
.Lcs_scalar:
    lsr     x7, x4, #2
    lsl     x7, x7, #2
    lsl     x6, x7, 2                  /* first unvectorized channel, bytes */
    cmp     x7, x4
    bge     .Lcs_done
.Lcs_tch:
    fmov    d0, xzr                    /* 0.0f */
    mov     x8, xzr                    /* k */
.Lcs_tap:
    madd    x9, x8, x5, x6
    ldr     s1, [x1, x9]
    ldr     s2, [x2, x9]
    fmul    s3, s1, s2
    fadd    s0, s0, s3
    add     x8, x8, #1
    cmp     x8, #7
    blt     .Lcs_tap
    ldr     s1, [x3, x6]
    fmul    s0, s0, s1
    str     s0, [x0, x6]
    add     x6, x6, #4
    add     x7, x7, #1
    cmp     x7, x4
    blt     .Lcs_tch
.Lcs_done:
    ret


/* ------------------------------------------------- conv + gate, ring variant
 * void lfm3_conv_step_r(float *out, const float *const *srow,
 *                       const float *wT, const float *Cgate, long H)
 *     x0=out  x1=srow  x2=wT  x3=Cgate  x4=H
 *
 * Same arithmetic as lfm3_conv_step, same ascending-tap order, same fused
 * fmla -- so it is bit-identical to it, and the gate checks exactly that. The
 * only change is addressing: srow[k] is the caller's pointer to tap k's row,
 * so the [7][H] state is a RING. The caller keeps a write index and writes
 * the newest B*xg into the slot that just fell out; nothing is copied.
 *
 * Why this exists: lfm3_conv_step's shift moves 6*H floats read + 6*H written
 * = 196 KB per layer per decoded token at H=2048, measured at ~75% of the
 * step's traffic (61.9 us/step, ~4.5 GB/s effective on one A57 core; the
 * repo's measured Nano ceiling is 21.9 GB/s over 4 cores, so one core
 * streams ~5 GB/s). The 14336 FMAs are NOT the cost -- the memcpy is.
 * Deleting a copy that is 3/4 of the traffic is the point.
 */
    .p2align 4
    .globl lfm3_conv_step_r
    .type  lfm3_conv_step_r, %function
lfm3_conv_step_r:
    cbz     x4, .Lcr_done
    stp     x19, x20, [sp, #-64]!
    stp     x21, x22, [sp, #16]
    stp     x23, x24, [sp, #32]
    stp     x25, x26, [sp, #48]
    ldp     x19, x20, [x1]
    ldp     x21, x22, [x1, #16]
    ldp     x23, x24, [x1, #32]
    ldr     x25, [x1, #48]
    lsl     x5, x4, 2                 /* weight tap stride bytes = H*4 */
    lsr     x10, x4, #2
    lsl     x10, x10, #2
    lsl     x10, x10, #2              /* 4-wide byte limit */
    mov     x6, xzr
    mov     x9, xzr
.Lcr_blk:
    cmp     x6, x10
    bge     .Lcr_tail
.Lcr_body:
    movi    v0.16b, #0
    /* tap k: state row from the ring table, weight row = wT + k*stride, and
     * BOTH indexed by the channel byte offset x6. Dropping x6 from the weight
     * side is the classic bug here: every block then reads tap k at channel 0
     * -- bit-wrong in a way no tolerance hides. */
    ldr     q1, [x19, x6]
    ldr     q2, [x2, x6]
    fmla    v0.4s, v1.4s, v2.4s
    ldr     q1, [x20, x6]
    add     x7, x2, x5
    ldr     q2, [x7, x6]
    fmla    v0.4s, v1.4s, v2.4s
    ldr     q1, [x21, x6]
    add     x7, x2, x5, lsl #1
    ldr     q2, [x7, x6]
    fmla    v0.4s, v1.4s, v2.4s
    ldr     q1, [x22, x6]
    add     x7, x2, x5
    add     x7, x7, x5, lsl #1
    ldr     q2, [x7, x6]
    fmla    v0.4s, v1.4s, v2.4s
    ldr     q1, [x23, x6]
    add     x7, x2, x5, lsl #2
    ldr     q2, [x7, x6]
    fmla    v0.4s, v1.4s, v2.4s
    ldr     q1, [x24, x6]
    add     x7, x2, x5, lsl #2
    add     x7, x7, x5
    ldr     q2, [x7, x6]
    fmla    v0.4s, v1.4s, v2.4s
    ldr     q1, [x25, x6]
    add     x7, x2, x5, lsl #2
    add     x7, x7, x5, lsl #1
    ldr     q2, [x7, x6]
    fmla    v0.4s, v1.4s, v2.4s
    ldr     q1, [x3, x6]
    fmul    v0.4s, v0.4s, v1.4s
    str     q0, [x0, x6]
    add     x6, x6, #16
    cbnz    x9, .Lcr_fin
    b       .Lcr_blk
.Lcr_tail:
    /* H % 4 != 0: re-run the fused body once at channel H-4 (overlapping,
     * in bounds, identical values). H<4 has no full vector block; callers
 * with that shape use lfm3_conv_step, which has the scalar path. */
    lsr     x7, x4, #2
    lsl     x7, x7, #2
    cmp     x7, x4
    bge     .Lcr_fin
    cmp     x4, #4
    blt     .Lcr_fin
    lsl     x6, x4, #2
    sub     x6, x6, #16
    mov     x9, #1
    b       .Lcr_body
.Lcr_fin:
    ldp     x25, x26, [sp, #48]
    ldp     x23, x24, [sp, #32]
    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #64
.Lcr_done:
    ret

    .section .note.GNU-stack,"",@progbits
