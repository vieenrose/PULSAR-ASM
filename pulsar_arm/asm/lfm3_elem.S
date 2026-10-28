/* lfm3_elem.S -- AArch64/NEON elementwise kernels for the LFM3-Audio decode
   step, Jetson Nano (Cortex-A57, ARMv8.0: no FEAT_BF16, no SDOT).

   WHY THESE THREE. The GEMV side is measured and near the bandwidth ceiling
   (14-16.5 GB/s, plan 13.1). The non-GEMV per-token work is the last unpriced
   term, and in the Bonsai engine that side is ~28% of decode. Per token the
   backbone runs 96 rmsnorms (4 x 24 layers), 26 bf16 multiplies (the conv
   op's B*xg and C*y gates, plus the head-gate path) and 48 residual adds.
   silu and the sigmoid gate need exp(); they stay on the GPU, because this
   path is libc-free (svc syscalls only, house style) and a polynomial exp()
   would not be bit-comparable to CUDA's expf. Disclosure, not omission.

   ACTIVATIONS ARE bf16 AND THE ROUNDING POINTS ARE THE CONTRACT. The HF
   model runs in bf16 and the goldens came from it, so an fp32 engine is MORE
   accurate than what it must reproduce and can never match it
   (ref_lfm3_np.py: cos 0.955 / maxrel 0.17 on a 7680-term dot, which looks
   exactly like a math bug). Storage is 16-bit; arithmetic widens:
     widen  bf16 -> f32:  shll / shll2 by 16 (exact -- it is a bit placement)
     narrow f32 -> bf16:  round-to-nearest-even,
            u = bits(f); u += 0x7FFF + ((u>>16)&1); out = (u16)(u>>16)
            -- the same bits torch's .to(bfloat16) produces. AArch64's
            fcvtn-to-bf16 is FEAT_BF16 (Armv8.4), NOT on A57, so this integer
            sequence is the only bit-exact option on this board.
   lfm3_rmsnorm_bf16 rounds to bf16 BEFORE multiplying by the weight, because
   modeling_lfm3.py says `self.weight * hidden_states.to(input_dtype)`.
   Rounding after the weight multiply is the most common way an rmsnorm
   parity gate fails.

   THE REDUCTION ORDER IS DEFINED HERE AND THE TEST MIRRORS IT EXACTLY.
   lfm3_elem_test.c does not compute "a" sum and compare within a tolerance.
   Variance order: 8 f32 accumulators a0..a7 (a0..a3 in vector A lanes 0-3,
   a4..a7 in vector B lanes 4-7), chunks ascending over H/8, a_j += x^2;
   then tA = ((a0+a1)+(a2+a3)), tB = ((a4+a5)+(a6+a7)), total = tA+tB.
   From 5fd7e79: a reference that mirrors the kernel's structure mirrors its
   bugs -- so the ORDER is pinned in this comment and the gate asserts
   equality, not closeness.

   rsqrt IS IEEE, NOT AN APPROXIMATION. This kernel uses fsqrt then fdiv
   (both IEEE round-to-nearest on A57) and the C reference uses 1.0f/sqrtf(v),
   also IEEE. CUDA's rsqrtf is a ~1 ulp APPROXIMATION and is not bit-equal to
   either, so the CUDA engine kernel must use __frsqrt_rn (correctly rounded)
   for the NEON and CUDA paths to agree; lfm3_nanode.cu currently uses
   rsqrtf and will differ from this kernel by ~1 ulp in the normalization.
   Flagged, not hidden.

   AAPCS64: d8-d15 (low halves of v8-v15) are CALLEE-SAVED. These kernels use
   v0-v7 as scratch, v16-v29 for data, v30/v31 for constants, and never touch
   v8-v15. The test build carries the register-contract checker from
   q8_gemv_neon_test.c, because a clobbered d8 once caused a 714 GB OpenMP
   allocation with zero symptom under a direct-call gate.

   Build on the Nano:
     as -o lfm3_elem.o lfm3_elem.S
     gcc -O2 -ffp-contract=off -o lfm3_elem_test lfm3_elem_test.c lfm3_elem.o -lm
   -ffp-contract=off is required for the reference: a fused a*b+c is a
   different real number than this kernel's mul-then-add.

   n / H must be a multiple of 8. Every shape here (H=2048, INTER=7680) is.
*/
        .section .rodata
        .align 4
.Lc_7fff:
        .word   0x00007fff, 0x00007fff, 0x00007fff, 0x00007fff
.Lc_one:
        .word   1, 1, 1, 1

        .text
        .align 4

/* narrow 4 f32 in \s to 4 bf16 in \s (its low 4 half-lanes), RN-even.
   \t, \u are scratch vectors; v30 = 0x7FFF per lane.
   binutils 2.30 (the Nano's as) rejects the bare vector `lsr` and `and`
   spellings used in the first draft, so this uses the canonical NEON names
   and gets the round bit with a second shift instead of an AND:
     (u>>16)>>31 == bit 16 of u, no mask needed.
   Same class of gap as the repo notes for smovl/vcvt/vext on this binutils. */
.macro  NARROW s, t, u
        ushr    \t\().4s, \s\().4s, #16
        /* Isolate bit 0 of t (= bit 16 of u, the round bit). It has to be
           shift-left-then-right: the obvious `ushr #31` reads bit 31 of t,
           i.e. bit 47 of u, which is always 0. That version made every
           exact tie round DOWN instead of to even -- the gate caught it on
           add: a+b = 2.1484375 exactly, asm 0x4009 (odd), IEEE RN-even
           0x400a. It is also what made rmsnorm disagree by 1 ulp at the
           tie rate (~0.2% of elements), so one macro, three symptoms. */
        shl     \u\().4s, \t\().4s, #31
        ushr    \u\().4s, \u\().4s, #31            /* bit0 of t, isolated */
        add     \s\().4s, \s\().4s, \u\().4s        /* + RN tie bit */
        add     \s\().4s, \s\().4s, v30.4s           /* + 0x7FFF */
        shrn    \s\().4h, \s\().4s, #16              /* take the top half */
.endm

/* load the two loop constants; must be re-run after any clobber of v30/v31 */
.macro  LOADCONST
        adrp    x4, .Lc_7fff
        adrp    x5, .Lc_one
        ldr     q30, [x4, :lo12:.Lc_7fff]
        ldr     q31, [x5, :lo12:.Lc_one]
.endm

/* ====================================================================
   void lfm3_rmsnorm_bf16(const unsigned short* x, const float* w,
                          unsigned short* y, int H, float eps);
   y = bf16( w * bf16( f32(x) * r ) ),  r = 1/sqrtf( mean(x^2) + eps )
   x=x0  w=x1  y=x2  H=w3  eps=s0
   (H is the FOURTH argument, so it arrives in w3 -- the first draft read w0,
   which is a pointer, as the loop count, and the kernel walked off the end
   of the buffer. The symptom was a segfault inside the kernel and the
   registers told the story: x7 = 178958706 and x6 holding ASCII text.)
   ==================================================================== */
        .global lfm3_rmsnorm_bf16
        .type   lfm3_rmsnorm_bf16, %function
        .align 4
lfm3_rmsnorm_bf16:
        LOADCONST
        mov     x8, x0                            /* x base */
        mov     x9, x1                            /* w base */
        sxtw    x7, w3
        lsr     x7, x7, #3                        /* x7 = H/8 chunks */

/* ---- pass 1: sum of squares, accumulators A=v20 (a0..a3), B=v21 (a4..a7) */
        movi    v20.4s, #0                          /* A = a0..a3 */
        movi    v21.4s, #0                          /* B = a4..a7 */
        mov     x5, x8
.Lrn_sum:
        ld1     {v16.8h}, [x5], #16
        shll    v22.4s, v16.4h, #16
        shll2   v23.4s, v16.8h, #16
        fmul    v22.4s, v22.4s, v22.4s
        fmul    v23.4s, v23.4s, v23.4s
        fadd    v20.4s, v20.4s, v22.4s
        fadd    v21.4s, v21.4s, v23.4s
        subs    x7, x7, #1
        b.ne    .Lrn_sum

/* reduce in the documented order: ((a0+a1)+(a2+a3)) + ((a4+a5)+(a6+a7)).
   Done with scalar `mov Sd, Vn.s[n]` extracts (binutils 2.30 wants `mov`,
   not `fmov`, for the vector-element form, and rejects the indexed
   `fadd Vd.2s, Vn.2s, Vn.2s[1]` spelling). The reduction is O(1) work, so
   the scalar form costs nothing. It also makes the order legible in the
   listing, which is the point -- the test mirrors THIS order. */
        mov     s24, v20.s[0]
        mov     s25, v20.s[1]
        mov     s26, v20.s[2]
        mov     s27, v20.s[3]
        fadd    s24, s24, s25                     /* a0+a1 */
        fadd    s26, s26, s27                     /* a2+a3 */
        fadd    s24, s24, s26                     /* tA */
        mov     s28, v21.s[0]
        mov     s29, v21.s[1]
        mov     s22, v21.s[2]
        mov     s23, v21.s[3]
        fadd    s28, s28, s29                     /* a4+a5 */
        fadd    s22, s22, s23                     /* a6+a7 */
        fadd    s28, s28, s22                     /* tB */
        fadd    s20, s24, s28                     /* total */

/* var = total/H ; r = 1/sqrtf(var+eps) -- fsqrt and fdiv are IEEE RN */
        scvtf   s21, w3
        fdiv    s20, s20, s21
        fadd    s20, s20, s0
        fsqrt   s21, s20
        fmov    s23, #1.0
        fdiv    s21, s23, s21
        /* Scratch GPR must be caller-saved: the first draft used w26, and
           x19-x28 are CALLEE-SAVED, so main's loop pointer came back from the
           call holding a float's bit pattern (0x3f086752) and the -O2 test
           binary died on `ldr w20, [x26]`. The d8-d15 checker cannot see
           this class -- it is an INTEGER register clobber, and the -O0 build
           runs clean because gcc happens not to keep anything live there.
           Use x10 (caller-saved) and broadcast with dup from a GPR, which is
           the spelling binutils 2.30 accepts. */
        fmov    w10, s21
        dup     v26.4s, w10

/* ---- pass 2: y = bf16( w * bf16( x*r ) ) ---- */
        sxtw    x7, w3
        lsr     x7, x7, #3
        mov     x5, x8                            /* x cursor  */
        mov     x6, x9                            /* w cursor  */
.Lrn_scale:
        ld1     {v16.8h}, [x5], #16
        /* post-increment is #16 (one q register), NOT #32: the first draft
           advanced the weight cursor by 32 on the FIRST load, so lanes 4-7
           multiplied by w[8..11] instead of w[4..7], and the cursor ran 32
           bytes per chunk past the end of w -- which is also what corrupted
           the heap and made the -O2 test binary fault in main after the
           kernel returned. One wrong immediate, two very different symptoms. */
        ldr     q28, [x6], #16                    /* w lanes 0-3 */
        ldr     q29, [x6], #16                    /* w lanes 4-7 */
        shll    v22.4s, v16.4h, #16
        shll2   v23.4s, v16.8h, #16
        fmul    v22.4s, v22.4s, v26.4s            /* x*r */
        fmul    v23.4s, v23.4s, v26.4s
        NARROW  v22, v2, v4                         /* -> bf16, in v22.4h */
        NARROW  v23, v3, v5
        /* Widen each narrowed half from ITS OWN register. The first draft
           used `shll2 v25.4s, v22.8h`, i.e. the upper half-lanes of v22 --
           but NARROW's shrn only writes the LOW 4 half-lanes, so bits 64-127
           of v22 are stale f32 bits. Symptom: rmsnorm failed at exactly
           i>=4 (lanes 4-7), with the asm emitting 0x8000/0x0000. */
        shll    v24.4s, v22.4h, #16               /* lanes 0-3, rounded */
        shll    v25.4s, v23.4h, #16               /* lanes 4-7, rounded */
        fmul    v22.4s, v24.4s, v28.4s            /* w * bf16(x*r) */
        fmul    v23.4s, v25.4s, v29.4s
        NARROW  v22, v2, v4
        NARROW  v23, v3, v5
        st1     {v22.4h, v23.4h}, [x2], #16
        subs    x7, x7, #1
        b.ne    .Lrn_scale
        ret
        .size   lfm3_rmsnorm_bf16, . - lfm3_rmsnorm_bf16

/* ====================================================================
   void lfm3_mul_bf16(const unsigned short* a, const unsigned short* b,
                      unsigned short* y, int n);      y = bf16(f32(a)*f32(b))
   a=x0 b=x1 y=x2 n=w3
   ==================================================================== */
        .global lfm3_mul_bf16
        .type   lfm3_mul_bf16, %function
        .align 4
lfm3_mul_bf16:
        LOADCONST
        sxtw    x7, w3
        lsr     x7, x7, #3
.Lmul_loop:
        ld1     {v16.8h}, [x0], #16
        ld1     {v17.8h}, [x1], #16
        shll    v0.4s, v16.4h, #16
        shll2   v1.4s, v16.8h, #16
        shll    v2.4s, v17.4h, #16
        shll2   v3.4s, v17.8h, #16
        fmul    v0.4s, v0.4s, v2.4s
        fmul    v1.4s, v1.4s, v3.4s
        NARROW  v0, v4, v6
        NARROW  v1, v5, v7
        st1     {v0.4h, v1.4h}, [x2], #16
        subs    x7, x7, #1
        b.ne    .Lmul_loop
        ret
        .size   lfm3_mul_bf16, . - lfm3_mul_bf16

/* ====================================================================
   void lfm3_add_bf16(const unsigned short* a, const unsigned short* b,
                      unsigned short* y, int n);      y = bf16(f32(a)+f32(b))
   The residual stream update. a=x0 b=x1 y=x2 n=w3
   ==================================================================== */
        .global lfm3_add_bf16
        .type   lfm3_add_bf16, %function
        .align 4
lfm3_add_bf16:
        LOADCONST
        sxtw    x7, w3
        lsr     x7, x7, #3
.Ladd_loop:
        ld1     {v16.8h}, [x0], #16
        ld1     {v17.8h}, [x1], #16
        shll    v0.4s, v16.4h, #16
        shll2   v1.4s, v16.8h, #16
        shll    v2.4s, v17.4h, #16
        shll2   v3.4s, v17.8h, #16
        fadd    v0.4s, v0.4s, v2.4s
        fadd    v1.4s, v1.4s, v3.4s
        NARROW  v0, v4, v6
        NARROW  v1, v5, v7
        st1     {v0.4h, v1.4h}, [x2], #16
        subs    x7, x7, #1
        b.ne    .Ladd_loop
        ret
        .size   lfm3_add_bf16, . - lfm3_add_bf16
