/* PULSAR-ASM | pulsar_arm/asm/core.S
 *
 * Pure AArch64 assembly core. No C, no Python, no libc: assembled with
 * `as`, linked `ld -static`, entry _start, Linux SVC syscalls only.
 *
 * Stage A: print helpers, libm-free expf/tanhf/sinf/cosf, bf16 GEMV that
 * mirrors kernels/neon_gemv.c lane-for-lane (bit-identical), plus a GEMV
 * microbench driven by CNTVCT_EL0.
 */
    .arch armv8-a
    .text
    .align 2
    .global _start

/* ---------------- syscalls ---------------- */
    .equ SYS_WRITE, 64
    .equ SYS_EXIT, 93

/* ---------------- tiny print helpers (x9-x18 scratch) ---------------- */

/* print_cstr: x0 = asciz address. Clobbers x0-x2,x8,x9. */
print_cstr:
    mov x9, x0
0:  ldrb w1, [x9], #1
    cbnz w1, 0b
    sub x2, x9, x0
    sub x2, x2, #1              // length, drop NUL
    mov x1, x0
    mov x0, #1                  // stdout
    mov x8, #SYS_WRITE
    svc #0
    ret

/* print_u64: x0 = value. Clobbers x0-x4,x8,x9. */
print_u64:
    adrp x9, numbuf
    add x9, x9, :lo12:numbuf
    add x9, x9, #64             // end of buffer
    mov x1, #10
    cbnz x0, 1f
    sub x9, x9, #1
    mov w2, #'0'
    strb w2, [x9]
    b 2f
1:  udiv x2, x0, x1
    msub x3, x2, x1, x0         // rem
    add w3, w3, #'0'
    sub x9, x9, #1
    strb w3, [x9]
    mov x0, x2
    cbnz x0, 1b
2:  adrp x1, numbuf
    add x1, x1, :lo12:numbuf
    add x1, x1, #64
    sub x2, x1, x9              // length
    mov x1, x9
    mov x0, #1
    mov x8, #SYS_WRITE
    svc #0
    ret

/* print_ch: w0 = char. */
print_ch:
    adrp x9, numbuf
    add x9, x9, :lo12:numbuf
    strb w0, [x9]
    mov x1, x9
    mov x2, #1
    mov x0, #1
    mov x8, #SYS_WRITE
    svc #0
    ret

/* print_f32: s0 = value, 7 decimals fixed. Clobbers s0-s3 + print_u64 set. */
print_f32:
    str x30, [sp, #-16]!
    fcmp s0, s0
    b.vs 9f                     // nan
    mov w9, #0x7f80
    lsl w9, w9, #8              // 0x7f800000 = +inf bits
    fmov s1, w9
    fcmp s0, s1
    b.eq 8f
    fneg s1, s1
    fcmp s0, s1
    b.eq 8f
    fcmp s0, #0.0
    b.ge 0f
    mov w0, #'-'
    bl print_ch
    fneg s0, s0
0:  frintm s1, s0                // floor
    fcvtzs x0, s1
    bl print_u64                 // int part (destroys s0? no: print uses x/GPR only)
    // NOTE: s0-s3 survive print_u64 (GPR-only). Recompute frac:
    frintm s1, s0
    fsub s2, s0, s1             // frac
    mov w0, #'.'
    bl print_ch
    fmov s3, #10.0
    mov w12, #7
3:  fmul s2, s2, s3
    frintm s1, s2
    fcvtzs x0, s1
    add x0, x0, #'0'
    bl print_ch_direct           // leaf: preserves lr and s-regs
    fsub s2, s2, s1
    subs w12, w12, #1
    b.ne 3b
    ldr x30, [sp], #16
    ret
8:  adrp x0, s_inf
    add x0, x0, :lo12:s_inf
    ldr x30, [sp], #16
    b print_cstr                // tail-call (lr restored)
9:  adrp x0, s_nan
    add x0, x0, :lo12:s_nan
    ldr x30, [sp], #16
    b print_cstr
print_ch_direct:                // print_ch without lr games (lr already saved)
    adrp x9, numbuf
    add x9, x9, :lo12:numbuf
    strb w0, [x9]
    mov x1, x9
    mov x2, #1
    mov x0, #1
    mov x8, #SYS_WRITE
    svc #0
    ret

/* ---------------- expf: s0 = exp(s0). ~1e-7 rel. ---------------- */
    .align 2
expf_asm:
    fcmp s0, s0
    b.vs expf_nan
    adrp x9, f_exp_hi
    ldr s1, [x9, :lo12:f_exp_hi]
    fcmp s0, s1
    b.gt expf_inf
    adrp x9, f_exp_lo
    ldr s1, [x9, :lo12:f_exp_lo]
    fcmp s0, s1
    b.lt expf_zero
    adrp x9, f_inv_ln2
    ldr s1, [x9, :lo12:f_inv_ln2]
    fmul s1, s0, s1
    frinti s1, s1               // k = nearbyint(x/ln2)
    fcvtzs x9, s1               // k int
    adrp x10, f_ln2
    ldr s2, [x10, :lo12:f_ln2]
    scvtf s3, x9
    fmul s3, s3, s2
    fsub s2, s0, s3             // r in [-ln2/2, ln2/2]
    // degree-6 Taylor: ((((r/720+1/120)r+1/24)r+1/6)r+1/2)r+1)r+1
    adrp x10, f_e6
    ldr s3, [x10, :lo12:f_e6]
    fmul s3, s3, s2
    adrp x10, f_e5
    ldr s4, [x10, :lo12:f_e5]
    fadd s3, s3, s4
    fmul s3, s3, s2
    adrp x10, f_e4
    ldr s4, [x10, :lo12:f_e4]
    fadd s3, s3, s4
    fmul s3, s3, s2
    adrp x10, f_e3
    ldr s4, [x10, :lo12:f_e3]
    fadd s3, s3, s4
    fmul s3, s3, s2
    fmov s4, #0.5
    fadd s3, s3, s4
    fmul s3, s3, s2
    fmov s4, #1.0
    fadd s3, s3, s4
    fmul s3, s3, s2
    fadd s0, s3, s4             // p(r)
    // ldexp: result bits += k<<23 (adds k to the biased exponent; no
    // mantissa carry since low 23 bits add zero)
    fmov w10, s0
    lsl x9, x9, #23
    add w10, w10, w9
    fmov s0, w10
    ret
expf_nan:
    ret                          // s0 already NaN
expf_inf:
    mov w9, #0x7f80
    lsl w9, w9, #8
    fmov s0, w9
    ret
expf_zero:
    fmov s0, wzr
    ret

/* ---------------- tanhf: s0 = tanh(s0) via expf ---------------- */
tanhf_asm:
    str x30, [sp, #-16]!
    fcmp s0, s0
    b.vs tanhf_done             // nan passthrough
    fabs s1, s0
    adrp x9, f_tanh_hi
    ldr s2, [x9, :lo12:f_tanh_hi]
    fcmp s1, s2
    b.gt tanhf_sat
    fadd s0, s0, s0             // 2x
    bl expf_asm                  // e2 = e^2x
    fmov s1, #1.0
    fsub s2, s0, s1
    fadd s0, s0, s1
    fdiv s0, s2, s0
tanhf_done:
    ldr x30, [sp], #16
    ret
tanhf_sat:
    fmov s1, #1.0
    fcmp s0, #0.0
    b.ge 0f
    fneg s1, s1
0:  fmov s0, s1
    ldr x30, [sp], #16
    ret

/* ---------------- sincos: reduce in f64, poly in f32 ---------------- */
/* sincos_asm: s0 = x -> s0 = sin(x), s1 = cos(x). |x| < 600. */
sincos_asm:
    stp x19, x20, [sp, #-16]!     // flags must survive adrp (clobbers x9)
    fcvt d2, s0                  // x in f64
    adrp x9, d_inv_2pi
    ldr d3, [x9, :lo12:d_inv_2pi]
    fmul d3, d2, d3
    frinti d3, d3                // q
    adrp x9, d_2pi
    ldr d4, [x9, :lo12:d_2pi]
    fmul d4, d3, d4
    fsub d2, d2, d4              // r in [-pi, pi]
    // quadrant fold to [-pi/2, pi/2]: r>pi/2 -> (pi-r, cos negates);
    // r<-pi/2 -> (r+pi, both negate). Polynomial error then < 1e-8.
    adrp x9, d_pi
    ldr d5, [x9, :lo12:d_pi]
    fmov d6, #0.5
    fmul d7, d5, d6              // pi/2
    fcmp d2, d7
    b.le 4f
    fsub d2, d5, d2
    mov w19, #0
    mov w20, #1
    b 5f
4:  fneg d7, d7
    fcmp d2, d7
    b.ge 6f
    fadd d2, d2, d5
    mov w19, #1
    mov w20, #1
    b 5f
6:  mov w19, #0
    mov w20, #0
5:  fcvt s2, d2                   // r'
    fmul s3, s2, s2              // x2
    // sin degree 13
    adrp x9, f_s13
    ldr s4, [x9, :lo12:f_s13]
    fmul s4, s4, s3
    adrp x9, f_s11
    ldr s5, [x9, :lo12:f_s11]
    fadd s4, s4, s5
    fmul s4, s4, s3
    adrp x9, f_s9
    ldr s5, [x9, :lo12:f_s9]
    fadd s4, s4, s5
    fmul s4, s4, s3
    adrp x9, f_s7
    ldr s5, [x9, :lo12:f_s7]
    fadd s4, s4, s5
    fmul s4, s4, s3
    adrp x9, f_s5
    ldr s5, [x9, :lo12:f_s5]
    fadd s4, s4, s5
    fmul s4, s4, s3
    adrp x9, f_s3
    ldr s5, [x9, :lo12:f_s3]
    fadd s4, s4, s5
    fmul s4, s4, s3
    fmov s5, #1.0
    fadd s4, s4, s5
    fmul s0, s4, s2
    cbnz w19, 7f
    b 8f
7:  fneg s0, s0
    // cos degree 12
8:  adrp x9, f_c12
    ldr s4, [x9, :lo12:f_c12]
    fmul s4, s4, s3
    adrp x9, f_c10
    ldr s5, [x9, :lo12:f_c10]
    fadd s4, s4, s5
    fmul s4, s4, s3
    adrp x9, f_c8
    ldr s5, [x9, :lo12:f_c8]
    fadd s4, s4, s5
    fmul s4, s4, s3
    adrp x9, f_c6
    ldr s5, [x9, :lo12:f_c6]
    fadd s4, s4, s5
    fmul s4, s4, s3
    adrp x9, f_c4
    ldr s5, [x9, :lo12:f_c4]
    fadd s4, s4, s5
    fmul s4, s4, s3
    adrp x9, f_c2
    ldr s5, [x9, :lo12:f_c2]
    fadd s4, s4, s5
    fmul s4, s4, s3
    fmov s5, #1.0
    fadd s1, s4, s5
    cbnz w20, 9f
    ldp x19, x20, [sp], #16
    ret
9:  fneg s1, s1
    ldp x19, x20, [sp], #16
    ret

/* ---------------- gemv_bf16: mirrors neon_gemv.c lane-for-lane ----------------
 * void gemv_bf16(int M, int K, const uint16_t *W, const float *x, float *y)
 * w0=M w1=K x2=W x3=x x4=y. 4 NEON accs, k+=16, same hsum order, same
 * scalar tail (separate mul+add): bit-identical to the C kernel. */
    .align 2
    .global gemv_bf16
gemv_bf16:
    mov w9, #0                   // m
    lsl x10, x1, #1              // row byte stride = K*2 (x1=K zero-extended)
    mov x5, x2                   // row ptr
    bic w12, w1, #15             // klim = K & ~15
0:  cmp w9, w0
    b.hs 9f
    movi v20.4s, #0
    movi v21.4s, #0
    movi v22.4s, #0
    movi v23.4s, #0
    mov w11, #0                  // k
1:  cmp w11, w12
    b.hs 2f
    add x13, x5, x11, lsl #1     // waddr = row + 2k
    ld1 {v0.8h}, [x13], #16
    ld1 {v1.8h}, [x13]
    shll v2.4s, v0.4h, #16       // single-insn bf16->fp32 (gcc-proven)
    shll2 v3.4s, v0.8h, #16
    shll v4.4s, v1.4h, #16
    shll2 v5.4s, v1.8h, #16
    add x13, x3, x11, lsl #2     // xaddr = x + 4k
    ld1 {v6.4s}, [x13], #16
    ld1 {v7.4s}, [x13], #16
    ld1 {v8.4s}, [x13], #16
    ld1 {v9.4s}, [x13]
    fmla v20.4s, v2.4s, v6.4s
    fmla v21.4s, v3.4s, v7.4s
    fmla v22.4s, v4.4s, v8.4s
    fmla v23.4s, v5.4s, v9.4s
    add w11, w11, #16
    b 1b
2:  faddp v20.4s, v20.4s, v20.4s
    faddp v20.4s, v20.4s, v20.4s
    faddp v21.4s, v21.4s, v21.4s
    faddp v21.4s, v21.4s, v21.4s
    faddp v22.4s, v22.4s, v22.4s
    faddp v22.4s, v22.4s, v22.4s
    faddp v23.4s, v23.4s, v23.4s
    faddp v23.4s, v23.4s, v23.4s
    fmov s24, s20
    fmov s25, s21
    fmov s26, s22
    fmov s27, s23
    fadd s24, s24, s25
    fadd s24, s24, s26
    fadd s24, s24, s27
3:  cmp w11, w1
    b.hs 4f
    ldrh w13, [x5, x11, lsl #1]
    lsl w13, w13, #16
    fmov s12, w13
    ldr s13, [x3, x11, lsl #2]
    fmul s12, s12, s13
    fadd s24, s24, s12
    add w11, w11, #1
    b 3b
4:  str s24, [x4, x9, lsl #2]
    add x5, x5, x10
    add x9, x9, #1
    b 0b
9:  ret

/* ---------------- self-tests ---------------- */
run_tests:
    str x30, [sp, #-16]!
    adrp x0, t_exp1
    add x0, x0, :lo12:t_exp1
    bl print_cstr
    adrp x9, f_1
    ldr s0, [x9, :lo12:f_1]
    bl expf_asm
    bl print_f32
    bl print_nl
    adrp x0, t_expm3
    add x0, x0, :lo12:t_expm3
    bl print_cstr
    adrp x9, f_m3
    ldr s0, [x9, :lo12:f_m3]
    bl expf_asm
    bl print_f32
    bl print_nl
    adrp x0, t_sin1
    add x0, x0, :lo12:t_sin1
    bl print_cstr
    adrp x9, f_1
    ldr s0, [x9, :lo12:f_1]
    bl sincos_asm                 // s0=sin s1=cos
    fmov s4, s1                   // stash cos (print uses s1-s3)
    bl print_f32
    bl print_sp
    fmov s0, s4
    bl print_f32
    bl print_nl
    adrp x0, t_cos2
    add x0, x0, :lo12:t_cos2
    bl print_cstr
    adrp x9, f_2
    ldr s0, [x9, :lo12:f_2]
    bl sincos_asm
    fmov s4, s1                   // stash cos
    fmov s0, s4                  // cos(2)
    bl print_f32
    bl print_nl
    adrp x0, t_tanh
    add x0, x0, :lo12:t_tanh
    bl print_cstr
    adrp x9, f_half
    ldr s0, [x9, :lo12:f_half]
    bl tanhf_asm
    bl print_f32
    bl print_nl
    adrp x0, t_sqrt
    add x0, x0, :lo12:t_sqrt
    bl print_cstr
    adrp x9, f_2
    ldr s0, [x9, :lo12:f_2]
    fsqrt s0, s0
    bl print_f32
    bl print_nl
    ldr x30, [sp], #16
    ret
print_nl:
    mov w0, #10
    b print_ch
print_sp:
    mov w0, #32
    b print_ch

/* ---------------- microbench: 20x gemv(2048,640), CNTVCT ---------------- */
microbench:
    str x30, [sp, #-16]!
    // fill W with LCG low16, x[i] = i*0.01
    adrp x9, Wmat
    add x9, x9, :lo12:Wmat
    mov x10, #0x1234
    movz x11, #0x14, lsl #16      // 2048*640
0:  movz x12, #0x66ad
    movk x12, #0x19, lsl #16     // 1664525
    mul x10, x10, x12
    movz x12, #0xf35f
    movk x12, #0x3c6e, lsl #16   // 1013904223
    add x10, x10, x12
    strh w10, [x9], #2
    subs x11, x11, #1
    b.ne 0b
    adrp x9, Xvec
    add x9, x9, :lo12:Xvec
    mov w10, #0
    adrp x11, f_001
    ldr s1, [x11, :lo12:f_001]
1:  scvtf s0, w10
    fmul s0, s0, s1
    str s0, [x9], #4
    add w10, w10, #1
    cmp w10, #640
    b.ne 1b
    mrs x20, cntfrq_el0
    mrs x21, cntvct_el0
    mov w22, #20
2:  mov w0, #2048
    mov w1, #640
    adrp x2, Wmat
    add x2, x2, :lo12:Wmat
    adrp x3, Xvec
    add x3, x3, :lo12:Xvec
    adrp x4, Yvec
    add x4, x4, :lo12:Yvec
    bl gemv_bf16                  // preserves x20-x22 (uses x0-x13, v0-v27)
    subs w22, w22, #1
    b.ne 2b
    mrs x9, cntvct_el0
    sub x9, x9, x21              // ticks
    mov x19, x9                  // stash: ms print clobbers x9-x11
    // ms = ticks*1000/freq ; print
    mov x10, #1000
    mul x11, x19, x10
    udiv x11, x11, x20
    adrp x0, t_bench
    add x0, x0, :lo12:t_bench
    bl print_cstr
    mov x0, x11
    bl print_u64
    adrp x0, t_ms
    add x0, x0, :lo12:t_ms
    bl print_cstr
    // GB/s (weight bytes) = 20*2048*640*2 * freq / ticks / 1e9
    movz x11, #0x320, lsl #16     // 20*2048*640*2 weight bytes
    mul x11, x11, x20
    udiv x11, x11, x19            // /ticks
    movz x10, #0xca00
    movk x10, #0x3b9a, lsl #16   // 1e9
    udiv x12, x11, x10           // int part
    msub x13, x12, x10, x11      // rem
    mov x10, #100
    mul x13, x13, x10
    movz x10, #0xca00
    movk x10, #0x3b9a, lsl #16   // 1e9
    udiv x13, x13, x10           // 2 decimals
    mov x0, x12
    bl print_u64
    mov w0, #'.'
    bl print_ch
    mov x0, x13
    // print 2 digits zero-padded
    mov x1, #10
    udiv x2, x0, x1
    msub x3, x2, x1, x0
    add x2, x2, #'0'
    add x3, x3, #'0'
    adrp x9, numbuf
    add x9, x9, :lo12:numbuf
    strb w2, [x9]
    strb w3, [x9, #1]
    mov x1, x9
    mov x2, #2
    mov x0, #1
    mov x8, #SYS_WRITE
    svc #0
    adrp x0, t_gbs
    add x0, x0, :lo12:t_gbs
    bl print_cstr
    ldr x30, [sp], #16
    ret

/* ---------------- math microbench: expf/tanhf ns per call ---------------- */
math_bench:
    str x30, [sp, #-16]!
    stp x19, x20, [sp, #-16]!
    adrp x19, T0
    add x19, x19, :lo12:T0       // base (survives expf/tanhf)
    adrp x0, f_00625
    add x0, x0, :lo12:f_00625
    ldr s1, [x0]                 // 0.0625 (reloaded per iter? NO: s1 dies at bl. RELOAD inside fill only (no calls in fill) ✓ fill first, then bench.)
    adrp x0, f_3
    add x0, x0, :lo12:f_3
    ldr s2, [x0]                 // 3.0
    mov w10, #0
0:  cmp w10, #2048
    b.hs 1f
    and w11, w10, #63
    scvtf s0, w11
    fmul s0, s0, s1
    fsub s0, s0, s2              // -3..0.9375
    str s0, [x19, x10, lsl #2]
    add w10, w10, #1
    b 0b
1:  // expf: 2048 iters x 2048 = 4M calls
    mrs x20, cntfrq_el0
    mrs x13, cntvct_el0          // start
    mov w11, #0
2:  cmp w11, #2048
    b.hs 3f
    mov w12, #0
4:  cmp w12, #2048
    b.hs 5f
    ldr s0, [x19, x12, lsl #2]
    bl expf_asm
    add w12, w12, #1
    b 4b
5:  add w11, w11, #1
    b 2b
3:  mrs x9, cntvct_el0           // end
    sub x9, x9, x13              // dt ticks (4M expf)
    mov x10, #100
    mul x10, x9, x10             // dt*100
    movz x11, #0x40, lsl #16     // 4194304 (x11: expf-safe? no calls below until prints; prints clobber x0-x9! x11 SAFE across prints? print funcs use x0-x4,x8,x9. x11 SAFE ✓)
    udiv x11, x10, x11           // tpc_x100
    str x11, [sp, #-16]!         // save
    adrp x0, t_expf
    add x0, x0, :lo12:t_expf
    bl print_cstr
    ldr x11, [sp], #16           // pop tpc
    mov x3, #100
    udiv x0, x11, x3             // int
    msub x5, x0, x3, x11         // frac (x5 safe across prints ✓)
    bl print_u64
    mov x2, #10
    udiv x3, x5, x2
    msub x4, x3, x2, x5
    add x3, x3, #'0'
    add x4, x4, #'0'
    mov w0, #'.'
    bl print_ch
    mov w0, w3
    bl print_ch
    mov w0, w4
    bl print_ch
    bl print_nl
    // tanhf: same shape (4M calls)
    mrs x13, cntvct_el0
    mov w11, #0
9:  cmp w11, #2048
    b.hs 10f
    mov w12, #0
11: cmp w12, #2048
    b.hs 12f
    ldr s0, [x19, x12, lsl #2]
    bl tanhf_asm
    add w12, w12, #1
    b 11b
12: add w11, w11, #1
    b 9b
10: mrs x9, cntvct_el0
    sub x9, x9, x13
    mov x10, #100
    mul x10, x9, x10
    movz x11, #0x40, lsl #16
    udiv x11, x10, x11
    str x11, [sp, #-16]!
    adrp x0, t_tanhf
    add x0, x0, :lo12:t_tanhf
    bl print_cstr
    ldr x11, [sp], #16
    mov x3, #100
    udiv x0, x11, x3
    msub x5, x0, x3, x11
    bl print_u64
    mov x2, #10
    udiv x3, x5, x2
    msub x4, x3, x2, x5
    add x3, x3, #'0'
    add x4, x4, #'0'
    mov w0, #'.'
    bl print_ch
    mov w0, w3
    bl print_ch
    mov w0, w4
    bl print_ch
    bl print_nl
    ldp x19, x20, [sp], #16
    ldr x30, [sp], #16
    ret
/* ---------------- entry ---------------- */
_start:
    adrp x0, banner
    add x0, x0, :lo12:banner
    bl print_cstr
    bl run_tests
    bl microbench
    bl math_bench
    bl kern_tests
    ldr x0, [sp]                  // argc
    add x1, sp, #8                // argv
    bl file_stage_after_bench
    bl tables_test
    bl fwd_layer0
    adrp x0, J_path
    add x0, x0, :lo12:J_path
    ldr x0, [x0]
    bl vocab_load
    adrp x0, rng_seed
    add x0, x0, :lo12:rng_seed
    ldr x0, [x0]                 // seed const
    adrp x1, G_RNG
    add x1, x1, :lo12:G_RNG
    str x0, [x1]                 // deterministic RNG start
    adrp x0, t_temp
    add x0, x0, :lo12:t_temp
    bl print_cstr
    adrp x0, G_TEMP
    add x0, x0, :lo12:G_TEMP
    ldr s0, [x0]
    bl print_f32
    adrp x0, t_topp
    add x0, x0, :lo12:t_topp
    bl print_cstr
    adrp x0, G_TOPP
    add x0, x0, :lo12:G_TOPP
    ldr s0, [x0]
    bl print_f32
    adrp x0, t_gencap
    add x0, x0, :lo12:t_gencap
    bl print_cstr
    adrp x0, G_GENCAP
    add x0, x0, :lo12:G_GENCAP
    ldr x0, [x0]
    bl print_u64
    bl print_nl
    adrp x0, J_bpe
    add x0, x0, :lo12:J_bpe
    ldr x0, [x0]
    // chat mode? (J_chat -> require bpe, chat INSTEAD of decode)
    adrp x1, J_chat
    add x1, x1, :lo12:J_chat
    ldr x1, [x1]
    cbnz x1, 1f
    // file mode? (J_f -> bpe + file_run instead of decode)
    adrp x1, J_f
    add x1, x1, :lo12:J_f
    ldr x1, [x1]
    cbnz x1, 3f
    cbz x0, 0f                   // no bpe path: skip BPE stage
    bl bpe_load
    bl encode_test
0:  bl decode_test
    mov x0, #0
    mov x8, #93
    svc #0
1:  // chat: bpe required
    cbnz x0, 2f
    adrp x0, t_nobpe
    add x0, x0, :lo12:t_nobpe
    bl print_cstr
    mov x0, #1
    mov x8, #93
    svc #0
2:  bl bpe_load
    bl chat                      // noreturn (exits on EOF)
3:  // file: bpe required (x0 = bpe path or 0? x0 CLOBBERED by chat path? This label reached ONLY from J_f check above (x0 = bpe path intact? Between J_f load and here: cbnz (no clobber) ✓ x0 intact ✓).
    cbnz x0, 4f
    adrp x0, t_nobpe
    add x0, x0, :lo12:t_nobpe
    bl print_cstr
    mov x0, #1
    mov x8, #93
    svc #0
4:  bl bpe_load
    bl file_run                  // noreturn (exits after one shot)
    mov x0, #0
    mov x8, #SYS_EXIT
    svc #0

/* ---------------- rodata ---------------- */
    .section .rodata
    .align 3
banner:      .asciz "pulsar-asm core stages A+B (pure asm, no libc)\n"
t_exp1:      .asciz "exp(1)   = "
t_expm3:     .asciz "exp(-3)  = "
t_sin1:      .asciz "sin(1) cos(1) = "
t_cos2:      .asciz "cos(2)   = "
t_tanh:      .asciz "tanh(.5) = "
t_sqrt:      .asciz "sqrt(2)  = "
t_bench:     .asciz "gemv(2048x640)x20: "
t_ms:        .asciz " ms  "
t_gbs:       .asciz " GB/s\n"
s_inf:       .asciz "inf"
s_nan:       .asciz "nan"
    .align 2
f_1:         .float 1.0
f_2:         .float 2.0
f_m3:        .float -3.0
f_half:      .float 0.5
f_001:       .float 0.01
f_00625:     .float 0.0625
f_3:         .float 3.0
t_expf:      .asciz "expf ticks/call: "
t_tanhf:     .asciz "tanhf ticks/call: "
    .align 2
f_exp_hi:    .float 88.72283
f_exp_lo:    .float -103.97208
f_inv_ln2:   .float 1.4426950408889634
f_ln2:       .float 0.6931471805599453
f_e3:        .float 0.16666667
f_e4:        .float 0.041666668
f_e5:        .float 0.008333334
f_e6:        .float 0.0013888889
f_tanh_hi:   .float 9.0
f_s3:        .float -0.16666667
f_s5:        .float 0.008333334
f_s7:        .float -0.0001984127
f_s9:        .float 0.0000027557319
f_c2:        .float -0.5
f_c4:        .float 0.041666668
f_c6:        .float -0.0013888889
f_c8:        .float 0.000024801587
f_c10:       .float -0.00000027557319
f_c12:       .float 0.0000000020876752
f_s11:       .float -0.0000000250521
f_s13:       .float 0.00000000016059
    .align 3
d_pi:        .double 3.141592653589793
d_inv_2pi:   .double 0.15915494309189535
d_2pi:       .double 6.283185307179586

/* ---------------- bss ---------------- */
    .section .bss
    .align 4
    .global numbuf
numbuf:      .skip 64
    .align 4
Wmat:        .skip 2048*640*2
Xvec:        .skip 640*4
Yvec:        .skip 2048*4

    .section .text
    .align 2
/* ---------------- stage B-1: file I/O + safetensors header scan ----------------
 * x0 = argv[1] (model path). Opens, sizes via lseek, mmaps whole file,
 * locates all 236 tensors by quoted-name search, parses data_offsets +
 * shape into tab[]. Prints anchors incl. real weight values. Noreturns
 * via exit on hard error (w1 = code path id in x0... exit code in x0). */

/* memmem: x0=hay, x1=hlen, x2=needle, x3=nlen -> x0=hit or 0. */
memmem:
    cbz x3, 9f                   // empty needle: return hay
    cmp x1, x3
    b.lo 8f
    sub x4, x1, x3              // last start index
    ldrb w5, [x2]               // first needle byte
0:  ldrb w6, [x0]
    cmp w6, w5
    b.ne 3f
    mov x7, #1
2:  cmp x7, x3
    b.hs 9f                     // full match, x0 = start
    ldrb w6, [x0, x7]
    ldrb w8, [x2, x7]
    cmp w6, w8
    b.ne 3f
    add x7, x7, #1
    b 2b
3:  add x0, x0, #1
    subs x4, x4, #1
    b.ge 0b
8:  mov x0, #0
    ret
9:  ret

/* parse_u64: x0 = cursor -> x0 = value, x1 = cursor after digits. */
parse_u64:
    mov x2, #0
0:  ldrb w3, [x0]
    sub w3, w3, #'0'
    cmp w3, #9
    b.hi 1f
    mov x4, #10
    mul x2, x2, x4
    add x2, x2, x3
    add x0, x0, #1
    b 0b
1:  mov x1, x0
    mov x0, x2
    ret

/* build_name: w0 = layer 0..17, x1 = template asciz with '@' slot,
 * x2 = out buffer -> x0 = out buffer (quoted "name"). */
build_name:
    str x30, [sp, #-16]!
    mov x3, x2
    mov w4, #'"'
    strb w4, [x3], #1
0:  ldrb w4, [x1], #1
    cbz w4, 2f
    cmp w4, #'@'
    b.ne 1f
    // layer number 0..17, 1-2 digits
    cmp w0, #10
    b.lo 3f
    mov x5, #10
    udiv x6, x0, x5
    msub x7, x6, x5, x0
    add x6, x6, #'0'
    add x7, x7, #'0'
    strb w6, [x3], #1
    mov w4, w7
    b 1f
3:  add w4, w0, #'0'
1:  strb w4, [x3], #1
    b 0b
2:  mov w4, #'"'
    strb w4, [x3], #1
    mov w4, #0
    strb w4, [x3]
    mov x0, x2
    ldr x30, [sp], #16
    ret

/* strlen: x0 = asciz -> x0 = len. */
strlen_asm:
    mov x1, x0
0:  ldrb w2, [x1], #1
    cbnz w2, 0b
    sub x0, x1, x0
    sub x0, x0, #1
    ret


/* find_tensor: x0 = quoted name, x4 = hbase, x5 = hlen, x6 = slot ptr
 * Slot: +0 off, +8 end, +16 d0, +24 d1, +32 d2, +40 d3, +48 nd.
 * Returns x0 = 1 ok / 0 missing. Keys searched explicitly (order-proof). */
find_tensor:
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    str x30, [sp, #-16]!
    mov x19, x0                  // name
    mov x20, x4                  // hbase
    mov x21, x5                  // hlen
    mov x22, x6                  // slot
    mov x0, x19
    bl strlen_asm
    mov x3, x0                   // nlen
    mov x0, x20
    mov x1, x21
    mov x2, x19
    bl memmem
    cbz x0, ft_miss
    mov x23, x0                  // name hit
    // shape key from name hit
    adrp x0, k_shape
    add x0, x0, :lo12:k_shape
    bl strlen_asm
    mov x3, x0
    adrp x2, k_shape
    add x2, x2, :lo12:k_shape
    mov x0, x23
    sub x1, x21, x23
    add x1, x1, x20
    bl memmem
    cbz x0, ft_miss
    mov x23, x0                  // shape-key pos
0:  ldrb w2, [x23], #1
    cmp w2, #'['
    b.ne 0b
    mov x24, #0                  // nd (x24: parse_u64 can't clobber)
1:  ldrb w2, [x23]
    cmp w2, #']'
    b.eq 2f
    cmp w2, #'0'
    b.lo 3f
    cmp w2, #'9'
    b.hi 3f
    mov x0, x23
    bl parse_u64                  // x0=val, x1=after
    mov x23, x1
    cmp x24, #4
    b.hs 4f
    lsl x4, x24, #3
    add x4, x4, #16
    str x0, [x22, x4]
4:  add x24, x24, #1
    b 1b
3:  add x23, x23, #1
    b 1b
2:  str x24, [x22, #48]          // nd
    // offsets key from name hit (x19 was overwritten? no: x19=name ptr;
    // name hit is in x23's history... RECOVER: x19 still = name string ptr,
    // not hit. Hit was overwritten by shape pos. RE-SEARCH name? No:
    // offsets necessarily follow the name; search from shape-key pos is
    // wrong if shape came after offsets. RELOAD hit: re-search name first.
    mov x0, x19
    bl strlen_asm
    mov x3, x0
    mov x0, x20
    mov x1, x21
    mov x2, x19
    bl memmem
    cbz x0, ft_miss
    mov x23, x0                  // name hit again
    adrp x0, k_offs
    add x0, x0, :lo12:k_offs
    bl strlen_asm
    mov x3, x0
    adrp x2, k_offs
    add x2, x2, :lo12:k_offs
    mov x0, x23
    sub x1, x21, x23
    add x1, x1, x20
    bl memmem
    cbz x0, ft_miss
    mov x23, x0
5:  ldrb w2, [x23], #1
    cmp w2, #'['
    b.ne 5b
    // parse a
6:  ldrb w2, [x23]
    cmp w2, #'0'
    b.lo 7f
    cmp w2, #'9'
    b.hi 7f
    mov x0, x23
    bl parse_u64
    mov x23, x1
    str x0, [x22]                // off
    b 8f
7:  add x23, x23, #1
    b 6b
    // parse b
8:  ldrb w2, [x23]
    cmp w2, #'0'
    b.lo 9f
    cmp w2, #'9'
    b.hi 9f
    mov x0, x23
    bl parse_u64
    mov x23, x1
    str x0, [x22, #8]            // end
    b 10f
9:  add x23, x23, #1
    b 8b
10: mov x0, #1
    b ft_ret
ft_miss:
    mov x0, #0
ft_ret:
    ldr x30, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ret

/* file_stage: x0 = model path. Prints anchors incl. real weight values. */
file_stage:
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp x25, x26, [sp, #-16]!
    str x30, [sp, #-16]!
    mov x19, x0                  // path
    mov x8, #56                  // openat
    mov x0, #-100                // AT_FDCWD
    mov x1, x19
    mov x2, #0
    mov x3, #0
    svc #0
    cmp x0, #0
    b.lt f_err_open
    mov x19, x0                  // fd
    mov x8, #62                  // lseek
    mov x0, x19
    mov x1, #0
    mov x2, #2                   // SEEK_END
    svc #0
    cmp x0, #0
    b.le f_err_seek
    mov x20, x0                  // size
    mov x8, #222                 // mmap
    mov x0, #0
    mov x1, x20
    mov x2, #1                   // PROT_READ
    mov x3, #2                   // MAP_PRIVATE
    mov x4, x19
    mov x5, #0
    svc #0
    cmp x0, #0
    b.lt f_err_map
    mov x21, x0                  // base
    mov x8, #57                  // close
    mov x0, x19
    svc #0
    ldr x22, [x21]               // N header bytes
    add x23, x21, #8             // hbase
    adrp x0, G_base
    add x0, x0, :lo12:G_base
    str x21, [x0]
    adrp x0, G_N
    add x0, x0, :lo12:G_N
    str x22, [x0]
    adrp x0, t_hdr
    add x0, x0, :lo12:t_hdr
    bl print_cstr
    mov x0, x22
    bl print_u64
    bl print_nl
    adrp x0, t_fsz
    add x0, x0, :lo12:t_fsz
    bl print_cstr
    mov x0, x20
    bl print_u64
    bl print_nl
    // scan: slot0=embed, slot1=norm, then 13/layer
    mov x24, #0                  // found
    adrp x0, n_embed
    add x0, x0, :lo12:n_embed
    mov x4, x23
    mov x5, x22
    adrp x6, tab
    add x6, x6, :lo12:tab
    bl find_tensor
    add x24, x24, x0
    adrp x0, n_norm
    add x0, x0, :lo12:n_norm
    mov x4, x23
    mov x5, x22
    adrp x6, tab
    add x6, x6, :lo12:tab
    add x6, x6, #56
    bl find_tensor
    add x24, x24, x0
    mov x25, #0                  // layer i
11: cmp x25, #18
    b.hs 12f
    mov x26, #0                  // kind k
13: cmp x26, #13
    b.hs 14f
    // template addr via pointer table (templates vary in length)
    adrp x1, tmpl_ptrs
    add x1, x1, :lo12:tmpl_ptrs
    ldr x1, [x1, x26, lsl #3]
    mov x0, x25                  // layer
    adrp x2, namebuf
    add x2, x2, :lo12:namebuf
    bl build_name                // x0 = quoted name
    mov x4, x23
    mov x5, x22
    // slot = 2 + 13*i + k  (x25=i, x26=k)
    mov x6, #13
    mul x6, x25, x6
    add x6, x6, x26
    add x6, x6, #2
    mov x0, #56
    mul x6, x6, x0
    adrp x1, tab
    add x1, x1, :lo12:tab
    add x6, x6, x1
    // find_tensor needs name in x0! RELOAD: namebuf addr
    adrp x0, namebuf
    add x0, x0, :lo12:namebuf
    bl find_tensor
    add x24, x24, x0
    add x26, x26, #1
    b 13b
14: add x25, x25, #1
    b 11b
12: adrp x0, t_found
    add x0, x0, :lo12:t_found
    bl print_cstr
    mov x0, x24
    bl print_u64
    bl print_nl
    // anchors: embed shape (slot0 d0/d1)
    adrp x0, t_embed
    add x0, x0, :lo12:t_embed
    bl print_cstr
    adrp x1, tab
    add x1, x1, :lo12:tab
    ldr x19, [x1, #16]              // (x1 dies in prints; x19+ survive)
    ldr x20, [x1, #24]
    mov x0, x19
    bl print_u64
    bl print_sp
    mov x0, x20
    bl print_u64
    bl print_nl
    // in_norm0 values (slot2): expect 15.0625 15.875 9.8125 40.5
    adrp x0, t_in0
    add x0, x0, :lo12:t_in0
    bl print_cstr
f_anchors_fixed:
    adrp x1, tab
    add x1, x1, :lo12:tab
    add x1, x1, #112
    ldr x3, [x1]                 // off
    add x2, x21, #8
    add x2, x2, x22
    add x2, x2, x3
    mov x4, #4
15: ldrh w0, [x2], #2
    lsl w0, w0, #16
    fmov s0, w0
    // save x2/x4 across print (uses x0-x4,x8,x9!)
    str x2, [sp, #-16]!
    str x4, [sp, #-16]!
    bl print_f32
    bl print_sp
    ldr x4, [sp], #16
    ldr x2, [sp], #16
    subs x4, x4, #1
    b.ne 15b
    bl print_nl
    // l17 down shape (slot 234): expect 640 x 2048
    adrp x0, t_l17d
    add x0, x0, :lo12:t_l17d
    bl print_cstr
    adrp x1, tab
    add x1, x1, :lo12:tab
    mov x0, #234
    mov x2, #56
    mul x0, x0, x2
    add x1, x1, x0
    ldr x19, [x1, #16]
    ldr x20, [x1, #24]
    mov x0, x19
    bl print_u64
    bl print_sp
    mov x0, x20
    bl print_u64
    bl print_nl
    b f_stage_ret
f_err_open:
    adrp x0, e_open
    add x0, x0, :lo12:e_open
    bl print_cstr
    b f_stage_fail
f_err_seek:
    adrp x0, e_seek
    add x0, x0, :lo12:e_seek
    bl print_cstr
    b f_stage_fail
f_err_map:
    adrp x0, e_map
    add x0, x0, :lo12:e_map
    bl print_cstr
f_stage_fail:
    mov x0, #1
    mov x8, #SYS_EXIT
    svc #0
f_stage_ret:
    ldr x30, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ret

/* ---------------- stage B-1 rodata/bss + entry wiring ---------------- */
    .section .rodata
    .align 3
k_shape:     .asciz "\"shape\""
k_offs:      .asciz "\"data_offsets\""
n_embed:     .asciz "\"model.embed_tokens.weight\""
n_norm:      .asciz "\"model.norm.weight\""
t_in:        .asciz "model.layers.@.input_layernorm.weight"
t_qn:        .asciz "model.layers.@.self_attn.q_norm.weight"
t_kn:        .asciz "model.layers.@.self_attn.k_norm.weight"
t_q:         .asciz "model.layers.@.self_attn.q_proj.weight"
t_k:         .asciz "model.layers.@.self_attn.k_proj.weight"
t_v:         .asciz "model.layers.@.self_attn.v_proj.weight"
t_o:         .asciz "model.layers.@.self_attn.o_proj.weight"
t_pa:        .asciz "model.layers.@.post_attention_layernorm.weight"
t_pf:        .asciz "model.layers.@.pre_feedforward_layernorm.weight"
t_g:         .asciz "model.layers.@.mlp.gate_proj.weight"
t_u:         .asciz "model.layers.@.mlp.up_proj.weight"
t_d:         .asciz "model.layers.@.mlp.down_proj.weight"
t_ff:        .asciz "model.layers.@.post_feedforward_layernorm.weight"
    .align 3
tmpl_ptrs:
    .quad t_in, t_qn, t_kn, t_q, t_k, t_v, t_o
    .quad t_pa, t_pf, t_g, t_u, t_d, t_ff
t_hdr:       .asciz "header bytes: "
t_fsz:       .asciz "file bytes: "
t_found:     .asciz "tensors found: "
t_embed:     .asciz "embed rows cols: "
t_in0:       .asciz "in_norm0[0..3] = "
t_l17d:      .asciz "l17_down rows cols: "
e_open:      .asciz "ERR open\n"
e_seek:      .asciz "ERR seek\n"
e_map:       .asciz "ERR mmap\n"
t_usage:     .asciz "usage: core <model.safetensors> <vocab.bin> [tok] [mode] [bpe.bin] [temp_milli] [topp_milli] [gencap]\n  mode: 0 greedy (default), 1 sample, c chat REPL (needs bpe.bin)\n  temp default 1000 (min 50), topp default 950, gencap default 128 (max 400)\n"

    .section .bss
    .align 3
tab:         .skip 236*56
namebuf:     .skip 128

/* _start appends the file stage: microbench -> argv -> file_stage -> exit */
    .section .text
    .align 2
    .global file_stage_after_bench
file_stage_after_bench:
    // x0 = argc, x1 = argv. argv[1]=snap, argv[2]=tokjson, argv[3]=tok.
    cmp x0, #3
    b.lo 0f
    str x30, [sp, #-16]!
    str x1, [sp, #-16]!          // argv
    stp x19, x20, [sp, #-16]!    // (temp/topp hold argc/argv across parses)
    adrp x2, J_path
    add x2, x2, :lo12:J_path
    ldr x3, [x1, #16]            // argv[2] tokjson
    str x3, [x2]
    mov x2, #2
    adrp x3, G_tok
    add x3, x3, :lo12:G_tok
    str x2, [x3]                 // default tok=2
    // G_MODE from argv[4] (default 0; '1' -> sample mode)
    adrp x2, G_MODE
    add x2, x2, :lo12:G_MODE
    mov x3, #0
    str x3, [x2]
    cmp x0, #5
    b.lo 3f
    ldr x3, [x1, #32]            // argv[4]
    ldrb w3, [x3]
    cmp w3, #'1'
    b.ne 3f
    mov x3, #1
    str x3, [x2]                 // x2 = G_MODE addr (intact? str doesn't modify ✓)
3:  // J_bpe from argv[5] (default 0 = skip BPE stage). x0=argc, x1=argv ✓
    adrp x2, J_bpe
    add x2, x2, :lo12:J_bpe
    mov x3, #0
    str x3, [x2]
    cmp x0, #6
    b.lo 4f
    ldr x3, [x1, #40]            // argv[5] bpe.bin path
    str x3, [x2]
4:  cmp x0, #4                   // x0=argc, x1=argv intact (mode+bpe used x2/x3 only ✓)
    // defaults (unconditional): G_TEMP=1.0, G_TOPP=0.95, G_GENCAP=128
    adrp x2, G_TEMP
    add x2, x2, :lo12:G_TEMP
    movz w3, #0x3F80, lsl #16
    str w3, [x2]
    adrp x2, G_TOPP
    add x2, x2, :lo12:G_TOPP
    movz w3, #0x3333
    movk w3, #0x3F73, lsl #16
    str w3, [x2]
    adrp x2, G_GENCAP
    add x2, x2, :lo12:G_GENCAP
    mov x3, #128
    str x3, [x2]
    mov x19, x0                  // argc (parse-safe: parse uses x0-x4 ✓)
    mov x20, x1                  // argv
    cmp x19, #7
    b.lo 8f                      // no temp arg: keep all defaults
    ldr x0, [x20, #48]           // argv[6]
    bl parse_u64
    cmp x0, #50
    b.hs 6f
    mov x0, #50
6:  mov x3, #5000
    cmp x0, x3
    b.lo 7f
    mov x0, #5000
7:  scvtf s0, x0
    adrp x1, f_1000
    add x1, x1, :lo12:f_1000
    ldr s1, [x1]
    fdiv s0, s0, s1
    adrp x1, G_TEMP
    add x1, x1, :lo12:G_TEMP
    str s0, [x1]
    cmp x19, #8
    b.lo 8f                      // no topp arg (also guards argv[7] NULL deref!)
    ldr x0, [x20, #56]           // argv[7]
    bl parse_u64
    cmp x0, #100
    b.hs 9f
    mov x0, #100
9:  cmp x0, #1000
    b.lo 10f
    mov x0, #1000
10: scvtf s0, x0
    adrp x1, f_1000
    add x1, x1, :lo12:f_1000
    ldr s1, [x1]
    fdiv s0, s0, s1
    adrp x1, G_TOPP
    add x1, x1, :lo12:G_TOPP
    str s0, [x1]
    cmp x19, #9
    b.lo 8f                      // no gencap arg
    ldr x0, [x20, #64]           // argv[8]
    bl parse_u64
    cmp x0, #1
    b.hs 12f
    mov x0, #1
12: cmp x0, #400
    b.lo 13f
    mov x0, #400
13: adrp x2, G_GENCAP
    add x2, x2, :lo12:G_GENCAP
    str x0, [x2]
8:  mov x0, x19                  // restore argc
    mov x1, x20                  // restore argv

    // file mode: argv[4][0]=='f' -> J_f=1, G_MODE=0 (greedy). x0=argc,x1=argv.
    adrp x2, J_f
    add x2, x2, :lo12:J_f
    mov x3, #0
    str x3, [x2]
    cmp x0, #5
    b.lo file_join
    ldr x3, [x1, #32]            // argv[4]
    ldrb w3, [x3]
    cmp w3, #'f'
    b.ne file_join
    mov x3, #1
    str x3, [x2]                 // J_f = 1
    adrp x2, G_MODE
    add x2, x2, :lo12:G_MODE
    mov x3, #1
    mov x3, #0
    str x3, [x2]                 // G_MODE = 0 (file = GREEDY, matching official example)
file_join:
    // (fall through to chat block; x0/x1 intact ✓)

    // chat mode: argv[4][0]=='c' -> J_chat=1, G_MODE=1 (sample)
    adrp x2, J_chat
    add x2, x2, :lo12:J_chat
    mov x3, #0
    str x3, [x2]
    cmp x0, #5
    b.lo 5f
    ldr x3, [x1, #32]            // argv[4]
    ldrb w3, [x3]
    cmp w3, #'c'
    b.ne 5f
    mov x3, #1
    str x3, [x2]                 // J_chat = 1
    adrp x2, G_MODE
    add x2, x2, :lo12:G_MODE
    str x3, [x2]                 // G_MODE = 1 (chat samples)
    b 11f
5:  cmp x0, #4
    b.lo 1f
    b.lo 1f
    ldr x0, [x1, #24]            // argv[3]
    bl parse_u64                 // x0 = tok (clobbers x0-x3; x1 dead? x1=argv NEEDED below! RELOAD from stack after (saved at entry ✓))
    adrp x1, G_tok
    add x1, x1, :lo12:G_tok
    str x0, [x1]
1:  ldp x19, x20, [sp], #16        // restore (reverse of push order)
    ldr x1, [sp], #16            // pop argv
    ldr x0, [x1, #8]             // argv[1] path -- WAIT x1 just popped (=argv) ✓, x0 = path ✓. But x0 was tok (dead ✓ fine).
    ldr x30, [sp], #16
    b file_stage
0:  adrp x0, t_usage
    add x0, x0, :lo12:t_usage
    bl print_cstr
    mov x0, #0
    mov x8, #SYS_EXIT
    svc #0


/* ---------------- stage B-2: compute kernels ---------------- */
    .section .text
    .align 2

/* rmsnorm_f32(out,x0 w,x1 x,x2 n,w3 eps,s0). Mirrors neon_ops.c incl. the
 * (x*w)*inv vector vs (x*inv)*w tail quirk. */
    .global rmsnorm_f32
rmsnorm_f32:
    str s0, [sp, #-16]!          // spill eps
    movi v0.4s, #0
    mov x5, x2                   // x cursor
    mov w9, #0
0:  add w10, w9, #4
    cmp w10, w3
    b.hi 1f
    ld1 {v1.4s}, [x5], #16
    fmla v0.4s, v1.4s, v1.4s
    add w9, w9, #4
    b 0b
1:  faddp v0.4s, v0.4s, v0.4s    // pairwise hsum (gcc-proven); total in s0
    faddp v0.4s, v0.4s, v0.4s
    mov w10, w9
2:  cmp w10, w3
    b.hs 3f
    ldr s1, [x2, x10, lsl #2]
    fmul s1, s1, s1
    fadd s0, s0, s1
    add w10, w10, #1
    b 2b
3:  scvtf s1, w3
    fdiv s1, s0, s1              // e/n
    ldr s2, [sp], #16            // eps
    fadd s1, s1, s2
    fsqrt s1, s1
    fmov s2, #1.0
    fdiv s1, s2, s1              // inv
    dup v4.4s, v1.s[0]           // broadcast inv (s1)
    mov x5, x2
    mov x6, x1
    mov x7, x0
    mov w9, #0
4:  add w10, w9, #4
    cmp w10, w3
    b.hi 5f
    ld1 {v2.4s}, [x5], #16
    ld1 {v3.4s}, [x6], #16
    fmul v2.4s, v2.4s, v3.4s
    fmul v2.4s, v2.4s, v4.4s
    st1 {v2.4s}, [x7], #16
    add w9, w9, #4
    b 4b
5:  cmp w9, w3
    b.hs 6f
    ldr s2, [x2, x9, lsl #2]
    fmul s2, s2, s1
    ldr s3, [x1, x9, lsl #2]
    fmul s2, s2, s3              // (x*inv)*w
    str s2, [x0, x9, lsl #2]
    add w9, w9, #1
    b 5b
6:  ret

/* rope_half(v,x0 cos,x1 sin,x2 h,w3). Scalar exact mirror. */
    .global rope_half
rope_half:
    lsl x4, x3, #2               // h*4
    mov w9, #0
0:  cmp w9, w3
    b.hs 1f
    lsl x5, x9, #2
    ldr s0, [x0, x5]
    add x6, x5, x4
    ldr s1, [x0, x6]
    ldr s2, [x1, x5]
    ldr s3, [x2, x5]
    fmul s4, s0, s2              // a*c (separate roundings, like C)
    fmul s5, s1, s3              // b*s
    fsub s4, s4, s5
    str s4, [x0, x5]
    fmul s4, s1, s2              // b*c
    fmul s5, s0, s3              // a*s
    fadd s4, s4, s5
    str s4, [x0, x6]
    add w9, w9, #1
    b 0b
1:  ret

/* softmax_f32(buf,x0 n,w1). Scalar mirror; m + counter survive expf via stack. */
    .global softmax_f32
softmax_f32:
    str x30, [sp, #-16]!          // NON-LEAF: preserve lr across expf calls
    ldr s0, [x0]
    mov w9, #1
0:  cmp w9, w1
    b.hs 1f
    ldr s1, [x0, x9, lsl #2]
    fcmp s1, s0
    b.le 2f
    fmov s0, s1
2:  add w9, w9, #1
    b 0b
1:  fmov s16, s0                 // m (no stack: expf keeps s16+, x11+)
    movi v6.4s, #0                // sum
    mov w11, #0
3:  cmp w11, w1
    b.hs 4f
    ldr s0, [x0, x11, lsl #2]
    fsub s0, s0, s16
    bl expf_asm
    str s0, [x0, x11, lsl #2]
    fadd s6, s6, s0
    add w11, w11, #1
    b 3b
4:  fmov s0, #1.0
    fdiv s0, s0, s6
    mov w9, #0
5:  cmp w9, w1
    b.hs 6f
    ldr s1, [x0, x9, lsl #2]
    fmul s1, s1, s0
    str s1, [x0, x9, lsl #2]
    add w9, w9, #1
    b 5b
6:  ldr x30, [sp], #16
    ret

/* gelu_mul_f32(out,x0 x,x1 mult,x2 n,w3). Scalar; w9 survives tanhf via stack. */
    .global gelu_mul_f32
gelu_mul_f32:
    str x30, [sp, #-16]!          // NON-LEAF: preserve lr across tanhf calls
    mov w9, #0
0:  cmp w9, w3
    b.hs 1f
    ldr s0, [x1, x9, lsl #2]     // v
    str x9, [sp, #-16]!
    adrp x9, f_c1
    add x9, x9, :lo12:f_c1
    ldr s1, [x9]                 // c1
    adrp x9, f_c3
    add x9, x9, :lo12:f_c3
    ldr s2, [x9]                 // c3
    fmul s3, s1, s0              // c1*v
    fmul s4, s2, s0              // c3*v
    fmul s4, s4, s0              // c3*v^2
    fmul s4, s4, s0              // c3*v^3
    fadd s0, s3, s4              // u = c1*v + c3*v^3 (C order)
    bl tanhf_asm                 // s0 = t(u)
    ldr x9, [sp], #16
    ldr s1, [x1, x9, lsl #2]     // v
    fmov s2, #0.5
    fmul s1, s2, s1              // 0.5*v first (C order)
    fmov s2, #1.0
    fadd s2, s2, s0              // 1+t
    fmul s1, s1, s2
    ldr s2, [x2, x9, lsl #2]
    fmul s1, s1, s2
    str s1, [x0, x9, lsl #2]
    add w9, w9, #1
    b 0b
1:  ldr x30, [sp], #16
    ret

/* attn_scores_f32(s,x0 q,x1 K,x2 n,w3 hd,w4 scale,s0). NEON dots, C order. */
    .global attn_scores_f32
attn_scores_f32:
    str s0, [sp, #-16]!          // spill scale
    mov w9, #0                   // p
    uxtw x10, w4
    lsl x10, x10, #2              // hd*4 row stride
0:  cmp w9, w3
    b.hs 9f
    mul x11, x9, x10
    add x13, x2, x11             // row cursor
    mov x14, x1                  // q cursor
    movi v20.4s, #0              // single acc (C order)
    bic w15, w4, #15             // klim
    mov w12, #0                  // d
1:  cmp w12, w15
    b.hs 2f
    ld1 {v0.4s}, [x13], #16
    ld1 {v1.4s}, [x14], #16
    fmla v20.4s, v0.4s, v1.4s
    add w12, w12, #4
    b 1b
2:  faddp v20.4s, v20.4s, v20.4s
    faddp v20.4s, v20.4s, v20.4s
    fmov s0, s20                 // total -> s0 (v20 lanes != s0!)
3:  cmp w12, w4
    b.hs 4f
    ldr s1, [x13], #4
    ldr s2, [x14], #4
    fmul s1, s1, s2              // separate mul+add (C order)
    fadd s0, s0, s1
    add w12, w12, #1
    b 3b
4:  ldr s1, [sp]                 // scale
    fmul s0, s0, s1
    str s0, [x0, x9, lsl #2]
    add w9, w9, #1
    b 0b
9:  add sp, sp, #16
    ret

/* attn_values_f32(out,x0 V,x1 w,x2 n,w3 hd,w4). C order: d-outer, p-inner. */
    .global attn_values_f32
attn_values_f32:
    mov w10, w4                  // hd (write w-reg: x10 zero-extended)
    mov w9, #0                   // d
0:  cmp w9, w4
    b.hs 9f
    movi v0.4s, #0
    mov w11, #0                  // p
1:  cmp w11, w3
    b.hs 2f
    madd x12, x11, x10, x9       // p*hd + d (elements)
    lsl x12, x12, #2
    add x12, x1, x12
    ld1 {v1.4s}, [x12]
    lsl x13, x11, #2
    add x13, x2, x13
    ld1r {v2.4s}, [x13]  // broadcast w[p]
    fmla v0.4s, v1.4s, v2.4s
    add w11, w11, #1
    b 1b
2:  lsl x13, x9, #2
    add x13, x0, x13
    st1 {v0.4s}, [x13]
    add w9, w9, #4
    b 0b
9:  ret

/* rmsnorm_add_f32(out,x0 w,x1 x,x2 acc,x3 n,w4 eps,s0). Norm + residual. */
    .global rmsnorm_add_f32
rmsnorm_add_f32:
    str s0, [sp, #-16]!
    movi v0.4s, #0
    mov x5, x2                   // x cursor
    mov w9, #0
0:  add w10, w9, #4
    cmp w10, w4
    b.hi 1f
    ld1 {v1.4s}, [x5], #16
    fmla v0.4s, v1.4s, v1.4s
    add w9, w9, #4
    b 0b
1:  faddp v0.4s, v0.4s, v0.4s
    faddp v0.4s, v0.4s, v0.4s
    mov w10, w9
2:  cmp w10, w4
    b.hs 3f
    ldr s1, [x2, x10, lsl #2]
    fmul s1, s1, s1
    fadd s0, s0, s1
    add w10, w10, #1
    b 2b
3:  scvtf s1, w4
    fdiv s1, s0, s1
    ldr s2, [sp], #16            // eps
    fadd s1, s1, s2
    fsqrt s1, s1
    fmov s2, #1.0
    fdiv s1, s2, s1              // inv
    dup v4.4s, v1.s[0]
    mov x5, x2
    mov x6, x1
    mov x7, x0
    mov x8, x3
    mov w9, #0
4:  add w10, w9, #4
    cmp w10, w4
    b.hi 5f
    ld1 {v2.4s}, [x5], #16
    ld1 {v3.4s}, [x6], #16
    fmul v2.4s, v2.4s, v3.4s
    fmul v2.4s, v2.4s, v4.4s
    st1 {v2.4s}, [x7], #16
    ld1 {v3.4s}, [x8]
    fadd v3.4s, v3.4s, v2.4s
    st1 {v3.4s}, [x8], #16
    add w9, w9, #4
    b 4b
5:  cmp w9, w4
    b.hs 6f
    ldr s2, [x2, x9, lsl #2]
    fmul s2, s2, s1
    ldr s3, [x1, x9, lsl #2]
    fmul s2, s2, s3
    str s2, [x0, x9, lsl #2]
    ldr s3, [x3, x9, lsl #2]
    fadd s3, s3, s2
    str s3, [x3, x9, lsl #2]
    add w9, w9, #1
    b 5b
6:  ret

/* embed_row_f32(out,x0 emb,x1 token,w2 dim,w3 scale,s0). bf16 gather+scale. */
    .global embed_row_f32
embed_row_f32:
    dup v6.4s, v0.s[0]           // broadcast scale (s0)
    uxtw x5, w2
    uxtw x6, w3
    mul x5, x5, x6               // token*dim (elements)
    lsl x5, x5, #1
    add x5, x1, x5               // row (u16 ptr)
    mov w9, #0                   // i
    bic w10, w3, #7              // ilim = dim & ~7
0:  cmp w9, w10
    b.hs 1f
    ld1 {v0.8h}, [x5], #16
    shll v1.4s, v0.4h, #16
    shll2 v2.4s, v0.8h, #16
    fmul v1.4s, v1.4s, v6.4s
    fmul v2.4s, v2.4s, v6.4s
    st1 {v1.4s}, [x0], #16
    st1 {v2.4s}, [x0], #16
    add w9, w9, #8
    b 0b
1:  cmp w9, w3
    b.hs 2f
    ldrh w11, [x5], #2
    lsl w11, w11, #16
    fmov s1, w11
    fmul s1, s1, s0              // w*scale (s0 intact: no calls)
    str s1, [x0], #4
    add w9, w9, #1
    b 1b
2:  ret

/* argmax_f32(buf,x0 n,w1) -> w0. First max wins. */
    .global argmax_f32
argmax_f32:
    ldr s0, [x0]
    mov w2, #0                   // best idx (x0 is the buffer; never clobber!)
    mov w9, #1
0:  cmp w9, w1
    b.hs 1f
    ldr s1, [x0, x9, lsl #2]
    fcmp s1, s0
    b.le 2f
    fmov s0, s1
    mov w2, w9
2:  add w9, w9, #1
    b 0b
1:  mov w0, w2
    ret

/* sum_sq(ptr,x0 n,w1) -> s0 = sum of squares (NEON 4-acc, C hsum order). */
sum_sq:
    movi v0.4s, #0
    mov x2, x0                   // cursor
    mov w9, #0
0:  add w10, w9, #4
    cmp w10, w1
    b.hi 1f
    ld1 {v1.4s}, [x2], #16
    fmla v0.4s, v1.4s, v1.4s
    add w9, w9, #4
    b 0b
1:  faddp v0.4s, v0.4s, v0.4s
    faddp v0.4s, v0.4s, v0.4s
2:  cmp w9, w1
    b.hs 3f
    ldr s1, [x0, x9, lsl #2]
    fmul s1, s1, s1
    fadd s0, s0, s1
    add w9, w9, #1
    b 2b
3:  ret

/* kern_tests: self-consistent kernel checks (no checkpoint needed). */
kern_tests:
    str x30, [sp, #-16]!
    stp x19, x20, [sp, #-16]!
    // -- gemv value: W=[[1,2,3,4],[5,6,7,8]] bf16, x=[1,1,1,1] -> [10,26]
    adrp x0, k_gemv
    add x0, x0, :lo12:k_gemv
    bl print_cstr
    adrp x19, T0
    add x19, x19, :lo12:T0      // x19 survives prints; W base
    adrp x20, T1
    add x20, x20, :lo12:T1      // x base
    movz w0, #0x3f80
    strh w0, [x19]
    movz w0, #0x4000
    strh w0, [x19, #2]
    movz w0, #0x4040
    strh w0, [x19, #4]
    movz w0, #0x4080
    strh w0, [x19, #6]
    movz w0, #0x40a0
    strh w0, [x19, #8]
    movz w0, #0x40c0
    strh w0, [x19, #10]
    movz w0, #0x40e0
    strh w0, [x19, #12]
    movz w0, #0x4100
    strh w0, [x19, #14]
    adrp x0, f_1
    add x0, x0, :lo12:f_1
    ldr s0, [x0]
    str s0, [x20]
    str s0, [x20, #4]
    str s0, [x20, #8]
    str s0, [x20, #12]
    mov w0, #2
    mov w1, #4
    mov x2, x19
    mov x3, x20
    adrp x4, T2
    add x4, x4, :lo12:T2
    bl gemv_bf16
    adrp x9, T2
    add x9, x9, :lo12:T2
    ldr s0, [x9]                 // expect 10.0
    bl print_f32
    bl print_sp
    adrp x9, T2
    add x9, x9, :lo12:T2
    ldr s0, [x9, #4]             // expect 26.0
    bl print_f32
    bl print_nl
    // n=4 vector-path test: rmsnorm([1,2,3,4], w=1) -> out[0] ~= 0.3651
    adrp x9, T0
    add x9, x9, :lo12:T0
    adrp x10, f_1
    ldr s0, [x10, :lo12:f_1]
    fmov s1, #2.0
    fmov s2, #3.0
    fmov s3, #4.0
    str s0, [x9]
    str s1, [x9, #4]
    str s2, [x9, #8]
    str s3, [x9, #12]
    adrp x10, T1
    add x10, x10, :lo12:T1
    str s0, [x10]
    str s0, [x10, #4]
    str s0, [x10, #8]
    str s0, [x10, #12]
    adrp x9, f_eps
    add x9, x9, :lo12:f_eps
    ldr s0, [x9]
    bl print_f32                 // expect 0.000001
    bl print_nl
    adrp x0, T2
    add x0, x0, :lo12:T2
    adrp x1, T1
    add x1, x1, :lo12:T1
    adrp x2, T0
    add x2, x2, :lo12:T0
    mov w3, #4
    adrp x4, f_eps
    ldr s0, [x4, :lo12:f_eps]
    bl rmsnorm_f32
    adrp x9, T2
    add x9, x9, :lo12:T2
    ldr s0, [x9]                 // expect 0.3651
    bl print_f32
    bl print_sp
    // tail-only: x=[2], w=[1], n=1 -> out=1.0
    adrp x9, T0
    add x9, x9, :lo12:T0
    adrp x10, f_2
    ldr s0, [x10, :lo12:f_2]
    str s0, [x9]
    adrp x0, T2
    add x0, x0, :lo12:T2
    adrp x1, T1
    add x1, x1, :lo12:T1
    adrp x2, T0
    add x2, x2, :lo12:T0
    mov w3, #1
    adrp x4, f_eps
    ldr s0, [x4, :lo12:f_eps]
    bl rmsnorm_f32
    adrp x9, T2
    add x9, x9, :lo12:T2
    ldr s0, [x9]                 // expect 1.0
    bl print_f32
    bl print_nl
    // -- rmsnorm: patterned x, w=0 -> sum(out^2) ~= 640
    adrp x0, k_rms
    add x0, x0, :lo12:k_rms
    bl print_cstr
    adrp x9, T0
    add x9, x9, :lo12:T0
    mov w10, #0
0:  cmp w10, #640
    b.hs 1f
    mov x11, #7
    mul x11, x10, x11
    mov x12, #13
    udiv x13, x11, x12
    msub x13, x13, x12, x11      // (7i)%13
    sub x13, x13, #6
    scvtf s0, x13
    adrp x11, f_001
    ldr s1, [x11, :lo12:f_001]
    fmul s0, s0, s1
    str s0, [x9, x10, lsl #2]
    add w10, w10, #1
    b 0b
1:  adrp x9, T1                  // w = 1 (zeros would null the signal)
    add x9, x9, :lo12:T1
    adrp x10, f_1
    add x10, x10, :lo12:f_1
    ldr s0, [x10]
    mov w10, #0
2:  cmp w10, #640
    b.hs 3f
    str s0, [x9, x10, lsl #2]
    add w10, w10, #1
    b 2b
3:  adrp x0, T2
    add x0, x0, :lo12:T2
    adrp x1, T1
    add x1, x1, :lo12:T1
    adrp x2, T0
    add x2, x2, :lo12:T0
    mov w3, #640
    adrp x4, f_eps
    ldr s0, [x4, :lo12:f_eps]
    bl rmsnorm_f32
    adrp x0, T2
    add x0, x0, :lo12:T2
    mov w1, #640
    bl sum_sq
    bl print_f32                 // expect ~639.9994
    bl print_nl
    // -- rope: norm preserved
    adrp x0, k_rope
    add x0, x0, :lo12:k_rope
    bl print_cstr
    adrp x9, T0
    add x9, x9, :lo12:T0
    mov w10, #0
4:  cmp w10, #256
    b.hs 5f
    sub w11, w10, #128
    scvtf s0, w11
    adrp x11, f_002
    ldr s1, [x11, :lo12:f_002]
    fmul s0, s0, s1
    str s0, [x9, x10, lsl #2]
    add w10, w10, #1
    b 4b
5:  adrp x0, T0
    add x0, x0, :lo12:T0
    mov w1, #256
    bl sum_sq
    fmov s6, s0                  // before (s6 survives: callees use s0-s5)
    bl print_f32
    bl print_sp
    adrp x9, f_07
    ldr s0, [x9, :lo12:f_07]
    bl sincos_asm                // s0=sin s1=cos
    adrp x9, T1
    add x9, x9, :lo12:T1
    adrp x10, T2
    add x10, x10, :lo12:T2
    mov w11, #0
6:  cmp w11, #128
    b.hs 7f
    str s0, [x9, x11, lsl #2]
    str s1, [x10, x11, lsl #2]
    add w11, w11, #1
    b 6b
7:  adrp x0, T0
    add x0, x0, :lo12:T0
    adrp x1, T1
    add x1, x1, :lo12:T1
    adrp x2, T2
    add x2, x2, :lo12:T2
    mov w3, #128
    bl rope_half
    adrp x0, T0
    add x0, x0, :lo12:T0
    mov w1, #256
    bl sum_sq
    bl print_f32                 // expect == before
    bl print_nl
    // -- softmax: sum == 1, argmax == 39
    adrp x0, k_soft
    add x0, x0, :lo12:k_soft
    bl print_cstr
    adrp x9, T0
    add x9, x9, :lo12:T0
    mov w10, #0
8:  cmp w10, #40
    b.hs 9f
    scvtf s0, w10
    fmov s1, #0.5
    fmul s0, s0, s1
    adrp x11, f_10
    ldr s1, [x11, :lo12:f_10]
    fsub s0, s0, s1
    str s0, [x9, x10, lsl #2]
    add w10, w10, #1
    b 8b
9:  adrp x0, T0
    add x0, x0, :lo12:T0
    mov w1, #40
    bl softmax_f32
    // scalar sum
    fmov s6, wzr
    mov w10, #0
10: cmp w10, #40
    b.hs 11f
    adrp x9, T0
    add x9, x9, :lo12:T0
    ldr s0, [x9, x10, lsl #2]
    fadd s6, s6, s0
    add w10, w10, #1
    b 10b
11: fmov s0, s6
    bl print_f32                 // expect 1.0000000
    bl print_sp
    adrp x0, T0
    add x0, x0, :lo12:T0
    mov w1, #40
    bl argmax_f32
    bl print_u64                 // expect 39
    bl print_nl
    // -- gelu: gelu_mul([1],[1],1)[0] ~= 0.8412
    adrp x0, k_gelu
    add x0, x0, :lo12:k_gelu
    bl print_cstr
    adrp x9, T0
    add x9, x9, :lo12:T0
    adrp x10, f_1
    ldr s0, [x10, :lo12:f_1]
    str s0, [x9]
    adrp x9, T1
    add x9, x9, :lo12:T1
    str s0, [x9]
    adrp x0, T2
    add x0, x0, :lo12:T2
    adrp x1, T0
    add x1, x1, :lo12:T0
    adrp x2, T1
    add x2, x2, :lo12:T1
    mov w3, #1
    bl gelu_mul_f32
    adrp x9, T2
    add x9, x9, :lo12:T2
    ldr s0, [x9]
    bl print_f32
    bl print_nl
    // -- scores: q=e0, K=[e0,e1,-e0] -> 0.0625 0 -0.0625
    adrp x0, k_scores
    add x0, x0, :lo12:k_scores
    bl print_cstr
    adrp x9, T0                  // q = e0 (256)
    add x9, x9, :lo12:T0
    mov w10, #0
12: cmp w10, #256
    b.hs 13f
    str wzr, [x9, x10, lsl #2]
    add w10, w10, #1
    b 12b
13: adrp x10, f_1
    ldr s0, [x10, :lo12:f_1]
    str s0, [x9]                 // q[0]=1
    adrp x9, T1                  // K 3x256 zeroed
    add x9, x9, :lo12:T1
    mov w10, #0
14: cmp w10, #768
    b.hs 15f
    str wzr, [x9, x10, lsl #2]
    add w10, w10, #1
    b 14b
15: str s0, [x9]                 // K[0][0]=1
    adrp x10, f_1
    ldr s0, [x10, :lo12:f_1]
    str s0, [x9, #1028]          // K[1][1]=1 (row1 off 1024 + 4)
    fneg s0, s0
    str s0, [x9, #2048]          // K[2][0]=-1 (row2 off 2048)
    adrp x0, T2
    add x0, x0, :lo12:T2
    adrp x1, T0
    add x1, x1, :lo12:T0
    adrp x2, T1
    add x2, x2, :lo12:T1
    mov w3, #3
    mov w4, #256
    adrp x5, f_scale
    ldr s0, [x5, :lo12:f_scale]
    bl attn_scores_f32
    adrp x9, T2
    add x9, x9, :lo12:T2
    ldr s0, [x9]
    bl print_f32
    bl print_sp
    adrp x9, T2
    add x9, x9, :lo12:T2
    ldr s0, [x9, #4]
    bl print_f32
    bl print_sp
    adrp x9, T2
    add x9, x9, :lo12:T2
    ldr s0, [x9, #8]
    bl print_f32
    bl print_nl
    // -- values: rows 1/2/3, w=one-hot[1] -> out=2.0
    adrp x0, k_values
    add x0, x0, :lo12:k_values
    bl print_cstr
    adrp x9, T0
    add x9, x9, :lo12:T0
    mov w10, #0
16: cmp w10, #768
    b.hs 17f
    lsr x11, x10, #8             // row = i/256 (0,1,2)
    add x11, x11, #1
    scvtf s0, x11
    str s0, [x9, x10, lsl #2]
    add w10, w10, #1
    b 16b
17: adrp x9, T1                  // w = [0,1,0]
    add x9, x9, :lo12:T1
    str wzr, [x9]
    adrp x10, f_1
    ldr s0, [x10, :lo12:f_1]
    str s0, [x9, #4]
    str wzr, [x9, #8]
    adrp x0, T2
    add x0, x0, :lo12:T2
    adrp x1, T0
    add x1, x1, :lo12:T0
    adrp x2, T1
    add x2, x2, :lo12:T1
    mov w3, #3
    mov w4, #256
    bl attn_values_f32
    adrp x9, T2
    add x9, x9, :lo12:T2
    ldr s0, [x9]
    bl print_f32
    bl print_sp
    adrp x9, T2
    add x9, x9, :lo12:T2
    ldr s0, [x9, #1020]          // out[255]
    bl print_f32
    bl print_nl
    // -- rmsnorm_add: acc=1 + norm
    adrp x0, k_rmsadd
    add x0, x0, :lo12:k_rmsadd
    bl print_cstr
    adrp x9, T3                  // acc = 1.0 (640)
    add x9, x9, :lo12:T3
    adrp x10, f_1
    ldr s0, [x10, :lo12:f_1]
    mov w11, #0
18: cmp w11, #640
    b.hs 19f
    str s0, [x9, x11, lsl #2]
    add w11, w11, #1
    b 18b
19: // rebuild x pattern in T0, ones w in T1[0..639]
    adrp x9, T0
    add x9, x9, :lo12:T0
    adrp x10, T1
    add x10, x10, :lo12:T1
    mov w11, #0
21: cmp w11, #640
    b.hs 22f
    mov x12, #7
    mul x12, x11, x12
    mov x13, #13
    udiv x14, x12, x13
    msub x14, x14, x13, x12
    sub x14, x14, #6
    scvtf s0, x14
    adrp x12, f_001
    ldr s1, [x12, :lo12:f_001]
    fmul s0, s0, s1
    str s0, [x9, x11, lsl #2]
    adrp x12, f_1
    add x12, x12, :lo12:f_1
    ldr s1, [x12]
    str s1, [x10, x11, lsl #2]   // w = 1.0 (unfolded)
    add w11, w11, #1
    b 21b
22: adrp x0, T2                  // out
    add x0, x0, :lo12:T2
    adrp x1, T1                  // w (ones)
    add x1, x1, :lo12:T1
    adrp x2, T0                  // x
    add x2, x2, :lo12:T0
    adrp x3, T3                  // acc (ones)
    add x3, x3, :lo12:T3
    mov w4, #640
    adrp x5, f_eps
    ldr s0, [x5, :lo12:f_eps]
    bl rmsnorm_add_f32
    adrp x9, T3
    add x9, x9, :lo12:T3
    ldr s0, [x9]                 // acc[0] = 1 + out[0]
    bl print_f32
    bl print_sp
    adrp x9, T2
    add x9, x9, :lo12:T2
    ldr s0, [x9]                 // out[0]
    bl print_f32
    bl print_nl
20: ldp x19, x20, [sp], #16
    ldr x30, [sp], #16
    ret

    .section .rodata
    .align 3
k_diag:      .asciz "diag: T0 f_001 = "
k_gemv:      .asciz "kern: gemv [[1..4],[5..8]] x 1 = "
k_rms:       .asciz "kern: rmsnorm sum(out^2)[640] = "
k_rope:      .asciz "kern: rope |v| before after = "
k_soft:      .asciz "kern: softmax sum argmax = "
k_gelu:      .asciz "kern: gelu(1) = "
k_scores:    .asciz "kern: scores = "
k_values:    .asciz "kern: values[0] values[255] = "
k_rmsadd:    .asciz "kern: acc[0] out[0] = "
    .align 2
f_eps:       .float 0.000001
f_002:       .float 0.02
f_07:        .float 0.7
f_10:        .float 10.0
f_scale:     .float 0.0625
f_c1:        .float 0.79788456
f_c3:        .float 0.03567741

    .section .bss
    .align 4
T0:          .skip 2048*4
T1:          .skip 2048*4
T2:          .skip 2048*4
T3:          .skip 2048*4

    .section .text
    .align 2
    .global kern_tests

/* ---------------- stage B-3a: RoPE tables + norm folding + embed ----------------
 * Tables: cos/sin for sliding (theta 1e4) and full (1e6), 512 positions.
 * inv_i = exp((-i/128) * ln theta). Angles pos*inv via sincos_asm.
 * Norms: float64 (1+w) fold of every 1D tensor (matches f32vec bitwise).
 * Embed: token row gather+scale. Anchors vs session-recorded values. */
    .section .text
    .align 2

/* build_tables: fills INV_S/F + TCOS/TSIN_S/F. Clobbers x0-x14, s0-s7. */
build_tables:
    str x30, [sp, #-16]!
    // inv arrays first (theta s=1e4, f=1e6)
    adrp x0, f_ln4
    add x0, x0, :lo12:f_ln4      // &ln (callee reloads per iter)
    adrp x1, INV_S
    add x1, x1, :lo12:INV_S
    bl build_inv
    adrp x0, f_ln6
    add x0, x0, :lo12:f_ln6      // &ln
    adrp x1, INV_F
    add x1, x1, :lo12:INV_F
    bl build_inv
    // tables from inv arrays
    adrp x0, INV_S
    add x0, x0, :lo12:INV_S
    adrp x1, TCOS_S
    add x1, x1, :lo12:TCOS_S
    adrp x2, TSIN_S
    add x2, x2, :lo12:TSIN_S
    bl build_one
    adrp x0, INV_F
    add x0, x0, :lo12:INV_F
    adrp x1, TCOS_F
    add x1, x1, :lo12:TCOS_F
    adrp x2, TSIN_F
    add x2, x2, :lo12:TSIN_F
    bl build_one
    ldr x30, [sp], #16
    ret
/* build_inv: x0 = &ln_theta, x1 = out[128]. inv_i = exp(-(i/128)*ln).
 * x0/x1/x11-x13 survive expf (uses x9,x10,s0-s4). One push (lr). */
build_inv:
    str x30, [sp, #-16]!
    mov w11, #0                  // i
0:  cmp w11, #128
    b.hs 1f
    ldr s1, [x0]                 // ln
    adrp x13, f_00078125
    add x13, x13, :lo12:f_00078125
    ldr s2, [x13]                // 1/128
    scvtf s3, w11
    fmul s3, s3, s2              // i/128
    fmul s3, s3, s1              // i*ln/128
    fneg s0, s3                  // e
    bl expf_asm                  // s0 = inv
    str s0, [x1, x11, lsl #2]
    add w11, w11, #1
    b 0b
1:  ldr x30, [sp], #16
    ret
/* build_one: x0 = inv[128], x1 = cos[512][128], x2 = sin.
 * x0-x2/x23/x24/s6 survive sincos (uses x9,s0-s5,d2-d5). */
build_one:
    str x30, [sp, #-16]!
    mov w24, #0                  // p (x24: sincos-safe, tables_test-safe)
0:  cmp w24, #512
    b.hs 9f
    mov w11, #0                  // i (x11: sincos-safe)
1:  cmp w11, #128
    b.hs 2f
    ldr s1, [x0, x11, lsl #2]    // inv_i
    scvtf s5, w24                // (float)p fresh (s5 dies in sincos, read first)
    fmul s0, s5, s1              // ang
    bl sincos_asm                // s0=sin s1=cos
    lsl x3, x24, #7              // p*128
    add x3, x3, x11
    lsl x3, x3, #2               // byte offset
    add x4, x1, x3
    str s1, [x4]                 // cos
    add x4, x2, x3
    str s0, [x4]                 // sin
    add w11, w11, #1
    b 1b
2:  add w24, w24, #1
    b 0b
9:  ldr x30, [sp], #16
    ret

    .section .rodata
    .align 3
f_ln4:       .float 9.21034037
f_ln6:       .float 13.81551056
f_00078125:  .float 0.0078125
f_sqrt640:   .float 25.29822128
t_inv:       .asciz "inv_s[0..1] = "
t_rcos:      .asciz "rcos_s[2][0..3] = "
t_rsin:      .asciz "rsin_s[2][0..3] = "
t_fnorm:     .asciz "fnorm_in0[0..3] = "
t_emb:       .asciz "embed[0..3] = "
t_vocab:     .asciz "vocab ids: "
e_vload:     .asciz "ERR vocab load\n"
    .align 3
v_magic:     .quad 0x564F4341424C4242

    .section .bss
    .align 4
INV_S:       .skip 128*4
INV_F:       .skip 128*4
TCOS_S:      .skip 512*128*4
TSIN_S:      .skip 512*128*4
TCOS_F:      .skip 512*128*4
TSIN_F:      .skip 512*128*4
FL:          .skip 110*640*4
G_base:      .skip 8
G_N:         .skip 8
G_tok:       .skip 8

    .section .text
    .align 2

/* fold_vec: x0 = src u16, x1 = dst f32, w2 = n. float64 (1+w). No calls. */
fold_vec:
    adrp x3, d_one
    add x3, x3, :lo12:d_one
    ldr d1, [x3]                 // 1.0 f64
0:  cbz w2, 1f
    ldrh w3, [x0], #2
    lsl w3, w3, #16
    fmov s0, w3
    fcvt d0, s0
    fadd d0, d0, d1
    fcvt s0, d0
    str s0, [x1], #4
    sub w2, w2, #1
    b 0b
1:  ret
/* fold_norms: folds norms via tab[] into FL (FL[0]=final, FL[1+6i+j]). */
fold_norms:
    str x30, [sp, #-16]!
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp x25, x26, [sp, #-16]!
    adrp x0, G_base
    add x0, x0, :lo12:G_base
    ldr x19, [x0]                // file base
    adrp x0, G_N
    add x0, x0, :lo12:G_N
    ldr x20, [x0]                // N
    add x19, x19, #8
    add x19, x19, x20            // x19 = data base
    adrp x20, tab
    add x20, x20, :lo12:tab      // x20 = tab
    adrp x21, FL
    add x21, x21, :lo12:FL       // x21 = folded base
    adrp x22, kkind
    add x22, x22, :lo12:kkind    // kind table
    // final norm: slot1 -> FL[0]
    mov x0, #56
    add x24, x20, x0             // slot1 ptr
    ldr x0, [x24]                // off
    add x0, x19, x0              // src
    ldr w2, [x24, #16]           // n = d0
    mov x1, x21                  // dst = FL[0]
    bl fold_vec
    mov w25, #0                  // i = layer
0:  cmp w25, #18
    b.hs 9f
    mov w26, #0                  // j = norm kind 0..5
1:  cmp w26, #6
    b.hs 2f
    // dst = FL + (1+6i+j)*2560 (x0/x4 scratch)
    mov x0, #6
    mul x0, x25, x0
    add x0, x0, x26
    add x0, x0, #1
    lsl x0, x0, #8               // *256
    mov x4, #10
    mul x0, x0, x4               // *2560
    add x1, x21, x0              // dst
    // src + n from slot 2+13i+k
    ldrb w3, [x22, x26]          // kind k
    mov x0, #13
    mul x0, x25, x0
    add x0, x0, x3
    add x0, x0, #2
    mov x4, #56
    mul x0, x0, x4
    add x24, x20, x0             // slot ptr
    ldr x0, [x24]                // off
    add x0, x19, x0              // src
    ldr w2, [x24, #16]           // n
    bl fold_vec
    add w26, w26, #1
    b 1b
2:  add w25, w25, #1
    b 0b
9:  ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ldr x30, [sp], #16
    ret

    .section .rodata
    .align 3
d_one:       .double 1.0
kkind:       .byte 0,1,2,7,8,12

    .section .bss
    .align 4
XB:          .skip 640*4

    .section .text
    .align 2
/* tables_test: build tables, print anchors, fold norms, embed. */
tables_test:
    str x30, [sp, #-16]!
    bl build_tables
    // inv anchors (expect 1.0, 0.9305720)
    adrp x0, t_inv
    add x0, x0, :lo12:t_inv
    bl print_cstr
    adrp x9, INV_S
    add x9, x9, :lo12:INV_S
    ldr s0, [x9]
    bl print_f32
    bl print_sp
    adrp x9, INV_S
    add x9, x9, :lo12:INV_S
    ldr s0, [x9, #4]
    bl print_f32
    bl print_nl
    // table anchors pos2 (row byte offset 2*128*4 = 1024)
    adrp x0, t_rcos
    add x0, x0, :lo12:t_rcos
    bl print_cstr
    adrp x19, TCOS_S
    add x19, x19, :lo12:TCOS_S
    add x19, x19, #1024
    ldr s0, [x19]
    bl print_f32
    bl print_sp
    ldr s0, [x19, #4]
    bl print_f32
    bl print_sp
    ldr s0, [x19, #8]
    bl print_f32
    bl print_sp
    ldr s0, [x19, #12]
    bl print_f32
    bl print_nl
    adrp x0, t_rsin
    add x0, x0, :lo12:t_rsin
    bl print_cstr
    adrp x19, TSIN_S
    add x19, x19, :lo12:TSIN_S
    add x19, x19, #1024
    ldr s0, [x19]
    bl print_f32
    bl print_sp
    ldr s0, [x19, #4]
    bl print_f32
    bl print_sp
    ldr s0, [x19, #8]
    bl print_f32
    bl print_sp
    ldr s0, [x19, #12]
    bl print_f32
    bl print_nl
    bl fold_norms
    // fnorm anchor: FL[1] (layer0 in) = FL + 2560
    adrp x0, t_fnorm
    add x0, x0, :lo12:t_fnorm
    bl print_cstr
    adrp x19, FL
    add x19, x19, :lo12:FL
    add x19, x19, #2560
    ldr s0, [x19]
    bl print_f32
    bl print_sp
    ldr s0, [x19, #4]
    bl print_f32
    bl print_sp
    ldr s0, [x19, #8]
    bl print_f32
    bl print_sp
    ldr s0, [x19, #12]
    bl print_f32
    bl print_nl
    // embed: token=G_tok into XB
    adrp x0, G_tok
    add x0, x0, :lo12:G_tok
    ldr w2, [x0]                 // token
    adrp x1, G_base
    add x1, x1, :lo12:G_base
    ldr x1, [x1]                 // file base
    adrp x3, G_N
    add x3, x3, :lo12:G_N
    ldr x3, [x3]                 // N
    add x1, x1, #8
    add x1, x1, x3               // data base
    adrp x3, tab
    add x3, x3, :lo12:tab
    ldr x3, [x3]                 // slot0 off
    add x1, x1, x3               // emb base
    adrp x0, XB
    add x0, x0, :lo12:XB         // out
    mov w3, #640
    adrp x4, f_sqrt640
    add x4, x4, :lo12:f_sqrt640
    ldr s0, [x4]                 // scale
    bl embed_row_f32
    adrp x0, t_emb
    add x0, x0, :lo12:t_emb
    bl print_cstr
    adrp x19, XB
    add x19, x19, :lo12:XB       // x19 survives prints
    ldr s0, [x19]
    bl print_f32
    bl print_sp
    ldr s0, [x19, #4]
    bl print_f32
    bl print_sp
    ldr s0, [x19, #8]
    bl print_f32
    bl print_sp
    ldr s0, [x19, #12]
    bl print_f32
    bl print_nl
    ldr x30, [sp], #16
    ret

    .section .rodata
    .align 3
s_X:         .asciz "L0 X "
s_H:         .asciz "L0 H "
s_QN0:       .asciz "L0 QN0 "
s_AV:        .asciz "L0 AV "
s_POST:      .asciz "L0 POST "
s_X1:        .asciz "L0 X1 "
s_GG:        .asciz "L0 GG "
s_D:         .asciz "L0 D "
s_X2:        .asciz "L0 X2 "
s_max:       .asciz "max="
s_vec:       .asciz " v=["
s_comma:     .asciz ","
s_rb:        .asciz "]"

    .section .bss
    .align 4
FH:          .skip 640*4
FQ:          .skip 1024*4
FQN:         .skip 1024*4
FKV:         .skip 256*4
FKN:         .skip 256*4
FV:          .skip 256*4
FS:          .skip 512*4
FAV:         .skip 1024*4
FG:          .skip 2048*4
FU:          .skip 2048*4

    .section .text
    .align 2
/* print_sig: x0=label, x1=vec, w2=n. Prints "label max=.. v=[..]".
 * Stack-peek discipline: vec/n live on stack, reloaded before EVERY use
 * (prints clobber x0-x9 and s0-s3). Nothing held across bl in regs. */
print_sig:
    str x30, [sp, #-16]!
    stp x1, x2, [sp, #-16]!      // [sp] = vec, n
    bl print_cstr                // label
    ldr x3, [sp]                 // vec
    ldr w4, [sp, #8]             // n
    ldr s0, [x3]
    fabs s0, s0
    mov w5, #1
0:  cmp w5, w4
    b.hs 1f
    ldr s1, [x3, x5, lsl #2]
    fabs s1, s1
    fcmp s1, s0
    b.le 2f
    fmov s0, s1
2:  add w5, w5, #1
    b 0b
1:  str s0, [sp, #-16]!          // save max across print
    adrp x0, s_max
    add x0, x0, :lo12:s_max
    bl print_cstr
    ldr s0, [sp], #16
    bl print_f32
    adrp x0, s_vec
    add x0, x0, :lo12:s_vec
    bl print_cstr
    ldr x3, [sp]
    ldr s0, [x3]
    bl print_f32
    adrp x0, s_comma
    add x0, x0, :lo12:s_comma
    bl print_cstr
    ldr x3, [sp]
    ldr s0, [x3, #4]
    bl print_f32
    adrp x0, s_comma
    add x0, x0, :lo12:s_comma
    bl print_cstr
    ldr x3, [sp]
    ldr s0, [x3, #8]
    bl print_f32
    adrp x0, s_comma
    add x0, x0, :lo12:s_comma
    bl print_cstr
    ldr x3, [sp]
    ldr s0, [x3, #12]
    bl print_f32
    adrp x0, s_rb
    add x0, x0, :lo12:s_rb
    bl print_cstr
    bl print_nl
    ldp x1, x2, [sp], #16        // pop vec,n (balance)
    ldr x30, [sp], #16
    ret
/* wptr: w0 = slot -> x0 = data+off. Uses x0/x1. x19=data (preserved). */
wptr:
    mov x1, #56
    mul x0, x0, x1
    adrp x1, tab
    add x1, x1, :lo12:tab
    add x0, x1, x0               // slot addr
    ldr x1, [x0]                 // off
    add x0, x19, x1              // data+off
    ret
/* nptr: w0 = fldidx -> x0 = FL+idx*2560. Uses x0/x1. */
nptr:
    lsl x0, x0, #8               // *256
    mov x1, #10
    mul x0, x0, x1               // *2560
    adrp x1, FL
    add x1, x1, :lo12:FL
    add x0, x1, x0
    ret
/* fwd_layer0: single token (G_tok), pos 0, layer 0 sliding. Prints L0 sig.
 * x19=data, x20=scratch-pad base? NO: adrp fresh per call except x19.
 * Holds x19=data across all calls (kernels keep x19+). Saves x19/x30. */
fwd_layer0:
    str x30, [sp, #-16]!
    stp x19, x20, [sp, #-16]!
    adrp x0, G_base
    add x0, x0, :lo12:G_base
    ldr x19, [x0]
    adrp x0, G_N
    add x0, x0, :lo12:G_N
    ldr x0, [x0]
    add x19, x19, #8
    add x19, x19, x0             // x19 = data base
    // X = embed(tok)
    adrp x0, G_tok
    add x0, x0, :lo12:G_tok
    ldr w2, [x0]                 // token
    mov w0, #0
    bl wptr                      // x0 = emb (slot0)
    mov x1, x0                   // emb
    adrp x0, XB
    add x0, x0, :lo12:XB
    mov w3, #640
    adrp x4, f_sqrt640
    add x4, x4, :lo12:f_sqrt640
    ldr s0, [x4]
    bl embed_row_f32             // (XB, emb, tok, 640, esc)
    adrp x0, s_X
    add x0, x0, :lo12:s_X
    adrp x1, XB
    add x1, x1, :lo12:XB
    mov w2, #640
    bl print_sig
    // H = norm(XB) with FL[1]
    mov w0, #1
    bl nptr                      // x0 = FL[1]
    mov x1, x0                   // w
    adrp x0, FH
    add x0, x0, :lo12:FH         // out
    adrp x2, XB
    add x2, x2, :lo12:XB         // x
    mov w3, #640
    adrp x4, f_eps
    add x4, x4, :lo12:f_eps
    ldr s0, [x4]
    bl rmsnorm_f32
    adrp x0, s_H
    add x0, x0, :lo12:s_H
    adrp x1, FH
    add x1, x1, :lo12:FH
    mov w2, #640
    bl print_sig
    // Q = q-proj(slot5,1024x640) . KV = k-proj(slot6,256x640)
    mov w0, #5
    bl wptr
    mov x2, x0                   // Wq
    adrp x3, FH
    add x3, x3, :lo12:FH
    adrp x4, FQ
    add x4, x4, :lo12:FQ
    mov w0, #1024
    mov w1, #640
    bl gemv_bf16
    mov w0, #6
    bl wptr
    mov x2, x0                   // Wk
    adrp x3, FH
    add x3, x3, :lo12:FH
    adrp x4, FKV
    add x4, x4, :lo12:FKV
    mov w0, #256
    mov w1, #640
    bl gemv_bf16
    // KN = norm(KV) with FL[3]; rope (sliding pos0)
    mov w0, #3
    bl nptr
    mov x1, x0
    adrp x0, FKN
    add x0, x0, :lo12:FKN
    adrp x2, FKV
    add x2, x2, :lo12:FKV
    mov w3, #256
    adrp x4, f_eps
    add x4, x4, :lo12:f_eps
    ldr s0, [x4]
    bl rmsnorm_f32
    adrp x0, FKN
    add x0, x0, :lo12:FKN
    adrp x1, TCOS_S
    add x1, x1, :lo12:TCOS_S
    adrp x2, TSIN_S
    add x2, x2, :lo12:TSIN_S
    mov w3, #128
    bl rope_half
    // V = v-proj(slot7,256x640) into FV
    mov w0, #7
    bl wptr
    mov x2, x0                   // Wv
    adrp x3, FH
    add x3, x3, :lo12:FH
    adrp x4, FV
    add x4, x4, :lo12:FV
    mov w0, #256
    mov w1, #640
    bl gemv_bf16
    // heads j=0..3 (w20; kernels keep x19+, print_sig keeps x20)
    mov w20, #0                  // j
20: cmp w20, #4
    b.hs 29f
    lsl x3, x20, #10             // j*1024
    mov w0, #2
    bl nptr                      // x0 = FL[2] (qn)
    mov x1, x0                   // w
    adrp x0, FQN
    add x0, x0, :lo12:FQN
    add x0, x0, x3               // out
    adrp x2, FQ
    add x2, x2, :lo12:FQ
    add x2, x2, x3               // x
    mov w3, #256
    adrp x4, f_eps
    add x4, x4, :lo12:f_eps
    ldr s0, [x4]
    bl rmsnorm_f32
    lsl x3, x20, #10
    adrp x0, FQN
    add x0, x0, :lo12:FQN
    add x0, x0, x3
    adrp x1, TCOS_S
    add x1, x1, :lo12:TCOS_S
    adrp x2, TSIN_S
    add x2, x2, :lo12:TSIN_S
    mov w3, #128
    bl rope_half
    cbnz w20, 21f
    adrp x0, s_QN0
    add x0, x0, :lo12:s_QN0
    adrp x1, FQN
    add x1, x1, :lo12:FQN
    mov w2, #256
    bl print_sig
21: lsl x3, x20, #10
    adrp x0, FS
    add x0, x0, :lo12:FS
    adrp x1, FQN
    add x1, x1, :lo12:FQN
    add x1, x1, x3
    adrp x2, FKN
    add x2, x2, :lo12:FKN
    mov w3, #1
    mov w4, #256
    adrp x5, f_scale
    add x5, x5, :lo12:f_scale
    ldr s0, [x5]
    bl attn_scores_f32
    adrp x0, FS
    add x0, x0, :lo12:FS
    mov w1, #1
    bl softmax_f32
    lsl x3, x20, #10
    adrp x0, FAV
    add x0, x0, :lo12:FAV
    add x0, x0, x3
    adrp x1, FV
    add x1, x1, :lo12:FV
    adrp x2, FS
    add x2, x2, :lo12:FS
    mov w3, #1
    mov w4, #256
    bl attn_values_f32
    add w20, w20, #1
    b 20b
29: adrp x0, s_AV
    add x0, x0, :lo12:s_AV
    adrp x1, FAV
    add x1, x1, :lo12:FAV
    mov w2, #1024
    bl print_sig
    // H = o-proj(slot8, 640x1024)
    mov w0, #8
    bl wptr
    mov x2, x0                   // Wo
    adrp x3, FAV
    add x3, x3, :lo12:FAV
    adrp x4, FH
    add x4, x4, :lo12:FH
    mov w0, #640
    mov w1, #1024
    bl gemv_bf16
    // H = rmsnorm_add(H, FL[4], H, XB); print POST, X1
    mov w0, #4
    bl nptr
    mov x1, x0                   // pa
    adrp x0, FH
    add x0, x0, :lo12:FH
    adrp x2, FH
    add x2, x2, :lo12:FH
    adrp x3, XB
    add x3, x3, :lo12:XB
    mov w4, #640
    adrp x5, f_eps
    add x5, x5, :lo12:f_eps
    ldr s0, [x5]
    bl rmsnorm_add_f32
    adrp x0, s_POST
    add x0, x0, :lo12:s_POST
    adrp x1, FH
    add x1, x1, :lo12:FH
    mov w2, #640
    bl print_sig
    adrp x0, s_X1
    add x0, x0, :lo12:s_X1
    adrp x1, XB
    add x1, x1, :lo12:XB
    mov w2, #640
    bl print_sig
    // H = rmsnorm(FH, FL[5], XB); G = gate(slot11); U = up(slot12)
    mov w0, #5
    bl nptr
    mov x1, x0                   // pf
    adrp x0, FH
    add x0, x0, :lo12:FH
    adrp x2, XB
    add x2, x2, :lo12:XB
    mov w3, #640
    adrp x4, f_eps
    add x4, x4, :lo12:f_eps
    ldr s0, [x4]
    bl rmsnorm_f32
    mov w0, #11
    bl wptr
    mov x2, x0                   // Wg
    adrp x3, FH
    add x3, x3, :lo12:FH
    adrp x4, FG
    add x4, x4, :lo12:FG
    mov w0, #2048
    mov w1, #640
    bl gemv_bf16
    mov w0, #12
    bl wptr
    mov x2, x0                   // Wu
    adrp x3, FH
    add x3, x3, :lo12:FH
    adrp x4, FU
    add x4, x4, :lo12:FU
    mov w0, #2048
    mov w1, #640
    bl gemv_bf16
    adrp x0, FG
    add x0, x0, :lo12:FG
    adrp x1, FG
    add x1, x1, :lo12:FG
    adrp x2, FU
    add x2, x2, :lo12:FU
    mov w3, #2048
    bl gelu_mul_f32
    adrp x0, s_GG
    add x0, x0, :lo12:s_GG
    adrp x1, FG
    add x1, x1, :lo12:FG
    mov w2, #2048
    bl print_sig
    // H = down(slot13, 640x2048); H = rmsnorm_add(FL[6]); print D, X2
    mov w0, #13
    bl wptr
    mov x2, x0                   // Wd
    adrp x3, FG
    add x3, x3, :lo12:FG
    adrp x4, FH
    add x4, x4, :lo12:FH
    mov w0, #640
    mov w1, #2048
    bl gemv_bf16
    mov w0, #6
    bl nptr
    mov x1, x0                   // ff
    adrp x0, FH
    add x0, x0, :lo12:FH
    adrp x2, FH
    add x2, x2, :lo12:FH
    adrp x3, XB
    add x3, x3, :lo12:XB
    mov w4, #640
    adrp x5, f_eps
    add x5, x5, :lo12:f_eps
    ldr s0, [x5]
    bl rmsnorm_add_f32
    adrp x0, s_D
    add x0, x0, :lo12:s_D
    adrp x1, FH
    add x1, x1, :lo12:FH
    mov w2, #640
    bl print_sig
    adrp x0, s_X2
    add x0, x0, :lo12:s_X2
    adrp x1, XB
    add x1, x1, :lo12:XB
    mov w2, #640
    bl print_sig
    ldp x19, x20, [sp], #16
    ldr x30, [sp], #16
    ret

    .section .bss
    .align 4
SAVXB:       .skip 640*4

    .section .text
    .align 2
    .section .bss
    .align 4
KC:          .skip 18*512*256*4
VC:          .skip 18*512*256*4
LG:          .skip 262144*4
G_pos:       .skip 8
G_t0:        .skip 8
G_t1:        .skip 8
PROMPT:      .skip 8*4

    .section .rodata
    .align 3
t_tok:       .asciz "tok: "
t_tps:       .asciz "tok/s: "
t_mpt:       .asciz "ms/token: "
t_metric:    .asciz "METRIC ms_per_token="
t_dot:       .asciz "."
t_nl2:       .asciz "\n"
def_prompt:  .word 2,107,1567,236765,107,304,2505,9694

    .section .text
    .align 2
/* fwd_token: w0 = token in, G_pos set. Full 18 layers + head. Returns w0.
 * x19=data x20=tab x21=FL x22=pos x23=li (saved). Scratch F*. */
fwd_token:
    str x30, [sp, #-16]!
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp x25, x26, [sp, #-16]!
    str x27, [sp, #-16]!
    mov w9, w0                   // token (w9 dies at first bl; move on)
    adrp x0, G_base
    add x0, x0, :lo12:G_base
    ldr x19, [x0]
    adrp x0, G_N
    add x0, x0, :lo12:G_N
    ldr x0, [x0]
    add x19, x19, #8
    add x19, x19, x0             // x19 = data
    adrp x20, tab
    add x20, x20, :lo12:tab
    adrp x21, FL
    add x21, x21, :lo12:FL
    adrp x0, G_pos
    add x0, x0, :lo12:G_pos
    ldr w22, [x0]                // pos
    // X = embed(tok): emb = data + tab[0].off
    ldr x0, [x20]                // slot0 off
    add x1, x19, x0              // emb
    adrp x0, XB
    add x0, x0, :lo12:XB
    mov w2, w9                   // token
    mov w3, #640
    adrp x4, f_sqrt640
    add x4, x4, :lo12:f_sqrt640
    ldr s0, [x4]
    bl embed_row_f32
    mov w23, #0                  // li
0:  cmp w23, #18
    b.hs 90f
    // full? li==5||11||17 -> F tables else S
    cmp w23, #5
    b.eq 31f
    cmp w23, #11
    b.eq 31f
    cmp w23, #17
    b.eq 31f
    adrp x0, TCOS_S
    add x0, x0, :lo12:TCOS_S
    adrp x1, TSIN_S
    add x1, x1, :lo12:TSIN_S
    b 32f
31: adrp x0, TCOS_F
    add x0, x0, :lo12:TCOS_F
    adrp x1, TSIN_F
    add x1, x1, :lo12:TSIN_F
32: lsl x2, x22, #9              // pos*512
    add x16, x0, x2              // cos row (x16 survives kernels)
    add x17, x1, x2              // sin row
    // H = norm(XB) with FL[1+6i]
    mov x0, #6
    mul x0, x23, x0
    add x0, x0, #1
    lsl x0, x0, #8
    mov x1, #10
    mul x0, x0, x1               // (1+6i)*2560
    add x1, x21, x0              // norm w
    adrp x0, FH
    add x0, x0, :lo12:FH
    adrp x2, XB
    add x2, x2, :lo12:XB
    mov w3, #640
    adrp x4, f_eps
    add x4, x4, :lo12:f_eps
    ldr s0, [x4]
    bl rmsnorm_f32
    // Q = q(slot 2+13i+3), KV = k(slot+4)
    mov x0, #13
    mul x0, x23, x0
    add x0, x0, #2               // 2+13i (layer base slot)
    mov x24, x0                  // base slot (x24: kernels keep x19+? x24 SAFE: kernels use x0-x18 max... VERIFY: gemv x13, scores x15, rest <x14. x24+ SAFE. But fwd must preserve x24 for caller? caller=driver holds tok in w19 only. x24 dead across call. NO SAVE needed... EXCEPT ABI for future. Driver recomputes. FINE.)
    add x0, x24, #3              // q slot
    mov x4, #56
    mul x0, x0, x4
    add x0, x20, x0
    ldr x0, [x0]                 // Wq off
    add x2, x19, x0              // Wq
    adrp x3, FH
    add x3, x3, :lo12:FH
    adrp x4, FQ
    add x4, x4, :lo12:FQ
    mov w0, #1024
    mov w1, #640
    bl gemv_bf16
    add x0, x24, #4              // k slot
    mov x4, #56
    mul x0, x0, x4
    add x0, x20, x0
    ldr x0, [x0]
    add x2, x19, x0              // Wk
    adrp x3, FH
    add x3, x3, :lo12:FH
    adrp x4, FKV
    add x4, x4, :lo12:FKV
    mov w0, #256
    mov w1, #640
    bl gemv_bf16
    // KN = norm(FKV) with FL[3+6i]; rope
    mov x0, #6
    mul x0, x23, x0
    add x0, x0, #3
    lsl x0, x0, #8
    mov x4, #10
    mul x0, x0, x4               // (3+6i)*2560
    add x1, x21, x0              // kn w
    adrp x0, FKN
    add x0, x0, :lo12:FKN
    adrp x2, FKV
    add x2, x2, :lo12:FKV
    mov w3, #256
    adrp x4, f_eps
    add x4, x4, :lo12:f_eps
    ldr s0, [x4]
    bl rmsnorm_f32
    adrp x0, FKN
    add x0, x0, :lo12:FKN
    mov x1, x16                  // cos
    mov x2, x17                  // sin
    mov w3, #128
    bl rope_half
    // V = v-proj(slot base+5, 256x640) -> FV
    add x0, x24, #5
    mov x4, #56
    mul x0, x0, x4
    add x0, x20, x0
    ldr x0, [x0]
    add x2, x19, x0              // Wv
    adrp x3, FH
    add x3, x3, :lo12:FH
    adrp x4, FV
    add x4, x4, :lo12:FV
    mov w0, #256
    mov w1, #640
    bl gemv_bf16
    // Kcache[i][pos] = KN; Vcache[i][pos] = FV(256 floats via FKN? NO: V is in FV!)
    // cache bases: KC + i*524288 (x18 scratch: kernels keep x19+? x18 SAFE (max x15))
    lsl x0, x23, #19             // i*524288
    adrp x1, KC
    add x1, x1, :lo12:KC
    add x1, x1, x0               // Kbase[i]
    adrp x2, VC
    add x2, x2, :lo12:VC
    add x2, x2, x0               // Vbase[i]
    lsl x0, x22, #10             // pos*1024
    add x0, x1, x0               // Krow dst
    adrp x1, FKN
    add x1, x1, :lo12:FKN        // KN src
    add x2, x2, x0               // WAIT: x0 reused! recompute Vrow separately below
    b 34f
34: // lo/n by layer type (full: 5/11/17)
    cmp w23, #5
    b.eq 35f
    cmp w23, #11
    b.eq 35f
    cmp w23, #17
    b.eq 35f
    add w0, w22, #1              // pos+1
    sub w0, w0, #512
    cmp w0, #0
    b.ge 36f
    mov w0, #0                   // lo
36: add w1, w22, #1              // pos+1
    cmp w1, #512
    b.lo 37f
    mov w1, #512
37: mov w18, w1                  // n (held through heads)
    b 38f
35: mov w0, #0                   // lo = 0
    add w18, w22, #1             // n = pos+1
38: // krow -> x26 ; vrow -> x27 (x0 = lo)
    lsl x1, x23, #19             // i*524288
    adrp x2, KC
    add x2, x2, :lo12:KC
    add x2, x2, x1
    lsl x1, x0, #10              // lo*1024
    add x26, x2, x1
    lsl x1, x23, #19
    adrp x2, VC
    add x2, x2, :lo12:VC
    add x2, x2, x1
    lsl x1, x0, #10
    add x27, x2, x1
    // K[i][pos] = KN (64 quads)
    lsl x0, x22, #10             // pos*1024
    lsl x1, x23, #19             // i*524288
    adrp x2, KC
    add x2, x2, :lo12:KC
    add x2, x2, x1
    add x2, x2, x0               // K dst
    adrp x0, FKN
    add x0, x0, :lo12:FKN
    mov w1, #64
39: cbz w1, 40f
    ld1 {v0.4s}, [x0], #16
    st1 {v0.4s}, [x2], #16
    sub w1, w1, #1
    b 39b
40: // V[i][pos] = FV
    lsl x0, x22, #10
    lsl x1, x23, #19
    adrp x2, VC
    add x2, x2, :lo12:VC
    add x2, x2, x1
    add x2, x2, x0               // V dst
    adrp x0, FV
    add x0, x0, :lo12:FV
    mov w1, #64
41: cbz w1, 42f
    ld1 {v0.4s}, [x0], #16
    st1 {v0.4s}, [x2], #16
    sub w1, w1, #1
    b 41b
42: // heads j in w25
    mov w25, #0
43: cmp w25, #4
    b.hs 44f
    lsl x3, x25, #10             // j*1024
    mov w0, #2
    bl nptr                      // x0 = FL_qn
    mov x5, x0                   // w (x5 dead across call? recomputed? NO: x5 set here, used below AFTER?? order: set x5, then set x0 (out, overwrites FL_qn addr? ALREADY SAVED in x5 ✓), then x1=x5, x2... then call. x5 live INTO the call setup (not across a call) ✓ SAFE.
    adrp x0, FQN
    add x0, x0, :lo12:FQN
    add x0, x0, x3               // out
    mov x1, x5                   // w
    adrp x2, FQ
    add x2, x2, :lo12:FQ
    add x2, x2, x3               // x
    mov w3, #256
    adrp x4, f_eps
    add x4, x4, :lo12:f_eps
    ldr s0, [x4]
    bl rmsnorm_f32
    lsl x3, x25, #10
    adrp x0, FQN
    add x0, x0, :lo12:FQN
    add x0, x0, x3
    mov x1, x16                  // cos
    mov x2, x17                  // sin
    mov w3, #128
    bl rope_half
    lsl x3, x25, #10
    adrp x0, FS
    add x0, x0, :lo12:FS
    adrp x1, FQN
    add x1, x1, :lo12:FQN
    add x1, x1, x3               // q
    mov x2, x26                  // krow
    mov w3, w18                  // n
    mov w4, #256
    adrp x5, f_scale
    add x5, x5, :lo12:f_scale
    ldr s0, [x5]
    bl attn_scores_f32
    adrp x0, FS
    add x0, x0, :lo12:FS
    mov w1, w18
    bl softmax_f32
    lsl x3, x25, #10
    adrp x0, FAV
    add x0, x0, :lo12:FAV
    add x0, x0, x3
    mov x1, x27                  // vrow
    adrp x2, FS
    add x2, x2, :lo12:FS
    mov w3, w18
    mov w4, #256
    bl attn_values_f32
    add w25, w25, #1
    b 43b
44: // o = gemv(FH, Wo[slot base+6], FAV)
    add x0, x24, #6
    mov x4, #56
    mul x0, x0, x4
    add x0, x20, x0
    ldr x0, [x0]
    add x2, x19, x0              // Wo
    adrp x3, FAV
    add x3, x3, :lo12:FAV
    adrp x4, FH
    add x4, x4, :lo12:FH
    mov w0, #640
    mov w1, #1024
    bl gemv_bf16
    // H = rmsnorm_add(FH, FL[4+6i], FH, XB)
    mov x0, #6
    mul x0, x23, x0
    add x0, x0, #4
    lsl x0, x0, #8
    mov x4, #10
    mul x0, x0, x4               // (4+6i)*2560
    add x1, x21, x0              // pa
    adrp x0, FH
    add x0, x0, :lo12:FH
    adrp x2, FH
    add x2, x2, :lo12:FH
    adrp x3, XB
    add x3, x3, :lo12:XB
    mov w4, #640
    adrp x5, f_eps
    add x5, x5, :lo12:f_eps
    ldr s0, [x5]
    bl rmsnorm_add_f32
    // H = rmsnorm(FH, FL[5+6i], XB)
    mov x0, #6
    mul x0, x23, x0
    add x0, x0, #5
    lsl x0, x0, #8
    mov x4, #10
    mul x0, x0, x4
    add x1, x21, x0              // pf
    adrp x0, FH
    add x0, x0, :lo12:FH
    adrp x2, XB
    add x2, x2, :lo12:XB
    mov w3, #640
    adrp x4, f_eps
    add x4, x4, :lo12:f_eps
    ldr s0, [x4]
    bl rmsnorm_f32
    // G = gate(base+9), U = up(base+10)
    add x0, x24, #9
    mov x4, #56
    mul x0, x0, x4
    add x0, x20, x0
    ldr x0, [x0]
    add x2, x19, x0              // Wg
    adrp x3, FH
    add x3, x3, :lo12:FH
    adrp x4, FG
    add x4, x4, :lo12:FG
    mov w0, #2048
    mov w1, #640
    bl gemv_bf16
    add x0, x24, #10
    mov x4, #56
    mul x0, x0, x4
    add x0, x20, x0
    ldr x0, [x0]
    add x2, x19, x0              // Wu
    adrp x3, FH
    add x3, x3, :lo12:FH
    adrp x4, FU
    add x4, x4, :lo12:FU
    mov w0, #2048
    mov w1, #640
    bl gemv_bf16
    adrp x0, FG
    add x0, x0, :lo12:FG
    adrp x1, FG
    add x1, x1, :lo12:FG
    adrp x2, FU
    add x2, x2, :lo12:FU
    mov w3, #2048
    bl gelu_mul_f32
    // H = down(base+11); H = rmsnorm_add(FL[6+6i])
    add x0, x24, #11
    mov x4, #56
    mul x0, x0, x4
    add x0, x20, x0
    ldr x0, [x0]
    add x2, x19, x0              // Wd
    adrp x3, FG
    add x3, x3, :lo12:FG
    adrp x4, FH
    add x4, x4, :lo12:FH
    mov w0, #640
    mov w1, #2048
    bl gemv_bf16
    mov x0, #6
    mul x0, x23, x0
    add x0, x0, #6
    lsl x0, x0, #8
    mov x4, #10
    mul x0, x0, x4
    add x1, x21, x0              // ff
    adrp x0, FH
    add x0, x0, :lo12:FH
    adrp x2, FH
    add x2, x2, :lo12:FH
    adrp x3, XB
    add x3, x3, :lo12:XB
    mov w4, #640
    adrp x5, f_eps
    add x5, x5, :lo12:f_eps
    ldr s0, [x5]
    bl rmsnorm_add_f32
    add w23, w23, #1
    b 0b
90: // final norm (FL[0]) + head + argmax
    adrp x0, FH
    add x0, x0, :lo12:FH
    adrp x1, FL
    add x1, x1, :lo12:FL
    adrp x2, XB
    add x2, x2, :lo12:XB
    mov w3, #640
    adrp x4, f_eps
    add x4, x4, :lo12:f_eps
    ldr s0, [x4]
    bl rmsnorm_f32
    mov w0, #262144
    mov w1, #640
    adrp x2, tab
    add x2, x2, :lo12:tab
    ldr x2, [x2]                 // slot0 off
    add x2, x19, x2              // head table
    adrp x3, FH
    add x3, x3, :lo12:FH
    adrp x4, LG
    add x4, x4, :lo12:LG
    bl gemv_bf16
    adrp x0, LG
    add x0, x0, :lo12:LG
    mov w1, #262144
    bl argmax_f32
    b 91f
91: ldr x27, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ldr x30, [sp], #16
    ret

    .section .rodata
    .align 3
t_ids:       .asciz "ids:"
t_text:      .asciz "text: "
t_mpt2:      .asciz "ms/token: "
t_tps2:      .asciz "tok/s: "
t_metric2:   .asciz "METRIC ms_per_token="

    .section .text
    .align 2
/* decode_test: prefill 8 fixed prompts, decode 32, time, print IDs+stats. */
decode_test:
    str x30, [sp, #-16]!
    stp x19, x20, [sp, #-16]!
    str x21, [sp, #-16]!
    adrp x21, def_prompt
    add x21, x21, :lo12:def_prompt  // prompt base (x21 survives fwd: saved)
    mov w20, #0
0:  cmp w20, #8
    b.hs 1f
    adrp x0, G_pos
    add x0, x0, :lo12:G_pos
    str w20, [x0]
    ldr w0, [x21, x20, lsl #2]   // prompt[i]
    bl fwd_token                 // -> w0 (discard except last)
    add w20, w20, #1
    b 0b
1:  // timed decode: tok = prompt[7] at pos 8..
    ldr w19, [x21, #28]          // prompt[7] (x19 survives fwd: saved)
    mrs x0, cntvct_el0
    adrp x1, G_t0
    add x1, x1, :lo12:G_t0
    str x0, [x1]
    mov w20, #8                  // pos
2:  cmp w20, #40
    b.hs 3f
    adrp x0, G_pos
    add x0, x0, :lo12:G_pos
    str w20, [x0]
    mov w0, w19
    bl fwd_token
    // sample mode? (G_MODE 1 -> sampler over LG; else greedy w0)
    adrp x1, G_MODE
    add x1, x1, :lo12:G_MODE
    ldr x1, [x1]
    cbz x1, 4f
    bl sampler                   // w0 = sampled (LG ready)
4:  mov w19, w0                  // tok
    // save IDBUF[pos-8] = tok (pos in w20)
    sub x1, x20, #8
    lsl x1, x1, #3
    adrp x2, IDBUF
    add x2, x2, :lo12:IDBUF
    add x2, x2, x1
    str w0, [x2]
    mov x0, x19                  // print ID -- x0 clobbered? print_u64 takes x0 ✓ set fresh
    str x19, [sp, #-16]!         // save tok across prints (x0-x9 die!)
    str x20, [sp, #-16]!         // save pos
    bl print_u64
    bl print_sp
    ldr x20, [sp], #16
    ldr x19, [sp], #16
    add w20, w20, #1
    b 2b
3:  mrs x0, cntvct_el0           // end ticks (G_t1 was never stored!)
    adrp x1, G_t1
    add x1, x1, :lo12:G_t1
    str x0, [x1]
    bl print_nl
    mrs x0, cntfrq_el0           // freq
    adrp x1, G_t0
    add x1, x1, :lo12:G_t0
    ldr x1, [x1]
    adrp x2, G_t1
    add x2, x2, :lo12:G_t1
    ldr x2, [x2]
    sub x2, x2, x1               // dt ticks
    str x2, [sp, #-16]!          // save dt
    // ms_x100/token = dt*1e5/(freq*32)
    movz x1, #0x86a0
    movk x1, #0x1, lsl #16        // 100000
    mul x1, x2, x1
    mov x3, #32
    mul x3, x0, x3
    udiv x4, x1, x3
    str x4, [sp, #-16]!          // save ms_x100
    adrp x0, t_mpt2
    add x0, x0, :lo12:t_mpt2
    bl print_cstr
    ldr x4, [sp]                 // peek ms_x100
    mov x3, #100
    udiv x0, x4, x3              // int
    msub x5, x0, x3, x4          // frac -> x5 (safe across print_u64)
    bl print_u64                 // int
    mov x2, #10
    udiv x3, x5, x2              // tens
    msub x4, x3, x2, x5          // ones (x3/x4 safe across print_ch)
    add x3, x3, #'0'
    add x4, x4, #'0'
    mov w0, #'.'
    bl print_ch
    mov w0, w3
    bl print_ch                  // tens
    mov w0, w4
    bl print_ch                  // ones
    bl print_nl
    // tok/s from PROVEN ms_x100: tps_x100 = 1e7/ms_x100
    // stack: [ms_x100][dt][x21][x19,x20][x30].
    adrp x0, t_tps2
    add x0, x0, :lo12:t_tps2
    bl print_cstr
    ldr x4, [sp]                 // ms_x100 (METRIC-proven)
    movz x0, #0x9680
    movk x0, #0x98, lsl #16      // 10000000
    udiv x0, x0, x4              // tps_x100
    mov x3, #100
    udiv x4, x0, x3              // int
    msub x5, x4, x3, x0          // frac
    mov x0, x4
    bl print_u64
    mov x2, #10
    udiv x3, x5, x2
    msub x4, x3, x2, x5
    add x3, x3, #'0'
    add x4, x4, #'0'
    mov w0, #'.'
    bl print_ch
    mov w0, w3
    bl print_ch
    mov w0, w4
    bl print_ch
    bl print_nl
    // METRIC (ms_x100 still on stack top)
    adrp x0, t_metric2
    add x0, x0, :lo12:t_metric2
    bl print_cstr
    ldr x4, [sp]                 // ms_x100
    mov x3, #100
    udiv x0, x4, x3
    msub x5, x0, x3, x4
    bl print_u64
    mov x2, #10
    udiv x3, x5, x2
    msub x4, x3, x2, x5
    add x3, x3, #'0'
    add x4, x4, #'0'
    mov w0, #'.'
    bl print_ch
    mov w0, w3
    bl print_ch
    mov w0, w4
    bl print_ch
    bl print_nl                  // (end METRIC line)
    // text: surfaces of the 32 decoded IDs
    adrp x0, t_text
    add x0, x0, :lo12:t_text
    bl print_cstr
    mov w19, #0                  // i (x19: put_surf keeps x19+)
0:  cmp w19, #32
    b.hs 1f
    lsl x0, x19, #3
    adrp x1, IDBUF
    add x1, x1, :lo12:IDBUF
    add x1, x1, x0
    ldr x0, [x1]                 // id
    bl put_surf
    add w19, w19, #1
    b 0b
1:  bl out_flush
    bl print_nl
    add sp, sp, #16              // drop ms_x100
    add sp, sp, #16              // drop dt
    ldr x21, [sp], #16
    ldp x19, x20, [sp], #16
    ldr x30, [sp], #16
    ret

    .section .bss
    .align 4
VTAB:        .skip 262144*16
OUTBUF:      .skip 1048576
IDBUF:       .skip 32*8
J_base:      .skip 8
J_len:       .skip 8

    .section .text
    .align 2
/* hexval: w0 = hex char -> x0 = value (0-15). No validation (trusted). */
hexval:
    cmp w0, #'9'
    b.hi 0f
    sub w0, w0, #'0'
    ret
0:  cmp w0, #'F'
    b.hi 1f
    sub w0, w0, #'A'
    add w0, w0, #10
    ret
1:  sub w0, w0, #'a'
    add w0, w0, #10
    ret
/* out_flush: write(1, OUTBUF, cursor-OUTBUF); reset cursor. */
out_flush:
    adrp x0, OUTCUR
    add x0, x0, :lo12:OUTCUR
    ldr x1, [x0]                 // cursor
    adrp x2, OUTBUF
    add x2, x2, :lo12:OUTBUF
    sub x1, x1, x2               // len
    cbz x1, 0f
    mov x3, x1                   // len
    mov x1, x2                   // buf
    mov x2, x3                   // len
    mov x0, #1
    mov x8, #64
    svc #0
    adrp x0, OUTCUR
    add x0, x0, :lo12:OUTCUR
    adrp x1, OUTBUF
    add x1, x1, :lo12:OUTBUF
    str x1, [x0]                 // reset cursor
0:  ret
/* put_surf: x0 = token id. Appends surface to OUTBUF (skips controls,
 * byte-tokens emit raw byte, U+2581 -> space). Missing/empty -> skip. */
put_surf:
    str x30, [sp, #-16]!
    lsl x1, x0, #4               // id*16
    adrp x2, VTAB
    add x2, x2, :lo12:VTAB
    add x2, x2, x1
    ldr x3, [x2]                 // off (abs)
    ldr x4, [x2, #8]             // len
    cbz x4, 9f                   // missing/empty: skip
    adrp x5, OUTCUR
    add x5, x5, :lo12:OUTCUR
    ldr x5, [x5]                 // cursor (x5: no calls below? hexval only if byte-tok. bl hexval clobbers x0 only? hexval uses w0! x0 = id (dead after table load? id needed? NO (done). x1-x4: x1 dead, x2 dead, x3=off (NEEDED for loop!), x4=len (NEEDED!). hexval CLOBBERS x0 only? hexval: cmp/sub/add on w0, ret. SAFE for x3/x4/x5 ✓.)
    ldrb w6, [x3]                // first byte
    cmp w6, #'<'
    b.ne 0f
    // '<...': byte token "<0xHH>"? (len 6)
    cmp x4, #6
    b.ne 8f                      // other <...>: skip
    ldrb w6, [x3, #1]
    cmp w6, #'0'
    b.ne 8f
    ldrb w6, [x3, #2]
    cmp w6, #'x'
    b.ne 8f
    ldrb w6, [x3, #5]
    cmp w6, #'>'
    b.ne 8f
    ldrb w0, [x3, #3]
    bl hexval                    // w0 = hi nibble (clobbers w0 only)
    lsl w0, w0, #4
    str w0, [sp, #-16]!          // save hi (need w0 for lo call!)
    ldrb w0, [x3, #4]
    bl hexval                    // w0 = lo
    ldr w6, [sp], #16            // hi
    lsl w6, w6, #4
    orr w0, w6, w0               // byte
    strb w0, [x5], #1
    b 7f                         // store cursor, ret
8:  // control token ('<...>' non-byte, or '[...'): skip. Also check '[':
    // (already know s[0]=='<': skip unconditionally)
    b 9f
0:  cmp w6, #'['
    b.ne 1f
    b 9f                         // '[...': skip
1:  // normal piece: emit with U+2581 -> space (bounds-checked)
    mov x6, x3                   // ptr = surface (x6 was first BYTE!)
    add x7, x3, x4               // end = surface + len
2:  cmp x6, x7
    b.hs 7f
    ldrb w0, [x6]                // peek B0
    cmp w0, #0xE2
    b.ne 4f
    add x0, x6, #2               // need 3 bytes
    cmp x0, x7
    b.hs 4f
    ldrb w0, [x6, #1]
    cmp w0, #0x96
    b.ne 4f
    ldrb w0, [x6, #2]
    cmp w0, #0x81
    b.ne 4f
    add x6, x6, #3               // consume U+2581
    mov w0, #32
    strb w0, [x5], #1
    b 2b
4:  ldrb w0, [x6], #1            // consume+emit single byte
    strb w0, [x5], #1
    b 2b
7:  adrp x0, OUTCUR
    add x0, x0, :lo12:OUTCUR
    str x5, [x0]
9:  ldr x30, [sp], #16
    ret

    .section .bss
    .align 3
OUTCUR:      .skip 8
J_path:      .skip 8
J_bpe:       .skip 8

    .section .text
    .align 2
/* vocab_load: x0 = tokenizer.json path. mmaps, vparses, spot-checks.
 * Prints count (expect 262144) + surfaces of 2,107,1567,108. */
vocab_load:  // x0 = vocab.bin path. mmaps read-only, verifies, memcpys table.
    str x30, [sp, #-16]!
    stp x19, x20, [sp, #-16]!
    mov x1, x0                   // path
    mov x0, #-100                // AT_FDCWD
    mov x2, #0
    mov x3, #0
    mov x8, #56
    svc #0                       // openat
    cmp x0, #0
    b.lt 9f
    str x0, [sp, #-16]!          // save fd
    mov x8, #62                  // lseek END
    ldr x0, [sp]                 // fd (peek)
    mov x1, #0
    mov x2, #2
    svc #0
    cmp x0, #0
    b.le 9f
    mov x20, x0                  // size
    mov x8, #222                 // mmap (read-only: blob needs no writes)
    mov x0, #0
    mov x1, x20
    mov x2, #1                   // PROT_READ
    mov x3, #2                   // MAP_PRIVATE
    ldr x4, [sp]                 // fd (peek)
    mov x5, #0
    svc #0
    cmp x0, #0
    b.lt 9f
    mov x19, x0                  // base
    ldr x0, [sp], #16            // fd (pop)
    mov x8, #57
    svc #0                       // close
    // verify magic + count
    ldr x0, [x19]                // magic
    adrp x1, v_magic
    add x1, x1, :lo12:v_magic
    ldr x1, [x1]
    cmp x0, x1
    b.ne 9f
    ldr x0, [x19, #8]            // count
    movz x1, #4, lsl #16         // 262144
    cmp x0, x1
    b.ne 9f
    str x0, [sp, #-16]!          // save count
    // memcpy table (4MB) base+16 -> VTAB
    add x0, x19, #16             // src
    adrp x1, VTAB
    add x1, x1, :lo12:VTAB       // dst
    movz x2, #8, lsl #16         // 524288 words
12: cbz x2, 11f
    ldr x3, [x0], #8
    str x3, [x1], #8
    sub x2, x2, #1
    b 12b
11: adrp x0, t_vocab
    add x0, x0, :lo12:t_vocab
    bl print_cstr
    ldr x0, [sp], #16            // pop count
    bl print_u64                 // expect 262144
    bl print_nl
    // fixup: VTAB[i].off += base (file offsets -> abs addresses)
    adrp x0, VTAB
    add x0, x0, :lo12:VTAB
    movz x1, #4, lsl #16         // 262144
0:  cbz x1, 1f
    ldr x2, [x0]
    add x2, x2, x19              // + base (x19 = blob base, intact? memcpy used x0-x3; x19 intact ✓)
    str x2, [x0], #16
    sub x1, x1, #1
    b 0b
1:  // OUTCUR = OUTBUF
    adrp x0, OUTCUR
    add x0, x0, :lo12:OUTCUR
    adrp x1, OUTBUF
    add x1, x1, :lo12:OUTBUF
    str x1, [x0]
    // spot checks: raw surfaces of 2,107,1567,108
    mov w0, #2
    bl vspot
    mov w0, #107
    bl vspot
    mov w0, #1567
    bl vspot
    mov w0, #108
    bl vspot
    ldp x19, x20, [sp], #16
    ldr x30, [sp], #16
    ret
9:  adrp x0, e_vload
    add x0, x0, :lo12:e_vload
    bl print_cstr
    mov x0, #1
    mov x8, #SYS_EXIT
    svc #0
vspot:
    str x30, [sp, #-16]!
    lsl x1, x0, #4
    adrp x2, VTAB
    add x2, x2, :lo12:VTAB
    add x2, x2, x1
    ldr x1, [x2]                 // off
    ldr x2, [x2, #8]             // len
    cbz x2, 0f                   // missing: print nothing (still nl below? just ret)
    mov x0, #1
    // write(1, off, len): x0=1, x1=off, x2=len
    mov x8, #64
    svc #0
0:  mov w0, #10
    bl print_ch
    ldr x30, [sp], #16
    ret

    .section .bss
    .align 3
G_MODE:      .skip 8
G_TEMP:      .skip 8
G_TOPP:      .skip 8
G_GENCAP:    .skip 8
G_RNG:       .skip 8
    .align 4
TKIDX:       .skip 64*4
TKVAL:       .skip 64*4

    .section .rodata
    .align 3
rng_seed:    .quad 0x9E3779B97F4A7C15
f_095:       .float 0.95
f_1000:      .float 1000.0
t_temp:      .asciz "temp: "
t_topp:      .asciz " topp: "
t_gencap:    .asciz " gencap: "
f_inv24:     .float 0.000000059604644775390625

    .section .text
    .align 2
/* rng_next: xorshift64* on G_rng. Returns x0. Leaf (no calls/stack). */
rng_next:
    adrp x1, G_RNG
    add x1, x1, :lo12:G_RNG
    ldr x0, [x1]
    lsr x2, x0, #12
    eor x0, x0, x2
    lsl x2, x0, #25
    eor x0, x0, x2
    lsr x2, x0, #27
    eor x0, x0, x2
    movz x2, #0xcdd1
    movk x2, #0x4f6c, lsl #16
    movk x2, #0xf491, lsl #32
    movk x2, #0x2545, lsl #48     // C = 0x2545F4914F6CDD1D
    mul x0, x0, x2
    adrp x1, G_RNG
    add x1, x1, :lo12:G_RNG
    str x0, [x1]                 // store back
    ret

/* softmax_temp(buf,x0 n,w1 temp,s0): softmax with temperature.
 * temp in s7 across expf (uses s0-s4). Counter x11. No stack. */
softmax_temp:
    str x30, [sp, #-16]!
    fmov s7, s0                  // temp (expf-safe: uses s0-s4)
    ldr s0, [x0]
    mov w11, #0
    add w11, w11, #1             // start max-scan at 1 (buf[0] loaded)
0:  cmp w11, w1
    b.hs 1f
    ldr s1, [x0, x11, lsl #2]
    fcmp s1, s0
    b.le 2f
    fmov s0, s1
2:  add w11, w11, #1
    b 0b
1:  fmov s6, s0                  // max (s6 survives expf calls)
    mov w11, #0
3:  cmp w11, w1
    b.hs 4f
    ldr s0, [x0, x11, lsl #2]
    fsub s0, s0, s6              // x - max
    fdiv s0, s0, s7              // /temp
    bl expf_asm
    str s0, [x0, x11, lsl #2]
    add w11, w11, #1
    b 3b
4:  // sum
    movi v6.4s, #0               // CAREFUL: s6 holds max, needed below? max NOT needed after exp loop (all (x-max) computed). REUSE s6/v6 freely ✓.
    mov w11, #0
5:  cmp w11, w1
    b.hs 6f
    ldr s0, [x0, x11, lsl #2]
    fadd s6, s6, s0
    add w11, w11, #1
    b 5b
6:  fmov s0, #1.0
    fdiv s0, s0, s6
    mov w11, #0
7:  cmp w11, w1
    b.hs 8f
    ldr s1, [x0, x11, lsl #2]
    fmul s1, s1, s0
    str s1, [x0, x11, lsl #2]
    add w11, w11, #1
    b 7b
8:  ldr x30, [sp], #16
    ret
/* topk_select(probs,x0 n,w1 k,w2 idx,x3 val,x4).
 * Descending top-k via insertion. No calls (leaf). Uses x0-x9,s0-s1. */
topk_select:
    // init val[*] = -1e30
    adrp x5, f_neg30
    add x5, x5, :lo12:f_neg30
    ldr s0, [x5]
    mov w5, #0
0:  cmp w5, w2
    b.hs 1f
    str s0, [x4, x5, lsl #2]
    add w5, w5, #1
    b 0b
1:  mov w5, #0                   // i
2:  cmp w5, w1
    b.hs 9f
    ldr s0, [x0, x5, lsl #2]     // v
    mov w6, #0                   // p scan
3:  cmp w6, w2
    b.hs 4f                      // no slot
    ldr s1, [x4, x6, lsl #2]
    fcmp s0, s1
    b.le 5f
    b 6f                         // found p
5:  add w6, w6, #1
    b 3b
6:  sub w7, w2, #1               // t = k-1; shift [p..k-2] right
7:  cmp w7, w6
    b.le 8f                      // t <= p: place
    sub w8, w7, #1
    ldr s1, [x4, x8, lsl #2]     // val[t-1]
    str s1, [x4, x7, lsl #2]     // val[t]
    ldr w8, [x3, x8, lsl #2]     // idx[t-1]
    str w8, [x3, x7, lsl #2]     // idx[t]
    sub w7, w7, #1
    b 7b
8:  str s0, [x4, x6, lsl #2]
    str w5, [x3, x6, lsl #2]
4:  add w5, w5, #1
    b 2b
9:  ret

    .section .rodata
    .align 2
f_neg30:     .word 0xF0D1328A            // -1e30 (sentinel)

    .section .text
    .align 2
/* sampler: LG ready. G_MODE 0 -> argmax; else temp/topk/topp sample.
 * Returns w0 = token. Uses x0-x12,s0-s7 (no x19+). Saves lr. */
sampler:
    str x30, [sp, #-16]!
    adrp x0, G_MODE
    add x0, x0, :lo12:G_MODE
    ldr x0, [x0]
    cbz x0, 0f                   // mode 0: greedy argmax (bit-identical path)
    // softmax_temp(LG, 262144, G_TEMP)
    adrp x0, LG
    add x0, x0, :lo12:LG
    movz w1, #4, lsl #16         // 262144
    adrp x2, G_TEMP
    add x2, x2, :lo12:G_TEMP
    ldr s0, [x2]                 // temp (x2 free here ✓)
    bl softmax_temp
    // topk(LG, 262144, 64, TKIDX, TKVAL)
    adrp x0, LG
    add x0, x0, :lo12:LG
    movz w1, #4, lsl #16
    mov w2, #64
    adrp x3, TKIDX
    add x3, x3, :lo12:TKIDX
    adrp x4, TKVAL
    add x4, x4, :lo12:TKVAL
    bl topk_select
    // total = sum TKVAL[0..63]; cutoff = total*0.95
    adrp x0, TKVAL
    add x0, x0, :lo12:TKVAL
    fmov s0, wzr                 // 0.0 (fmov from WZR is legal)
    adrp x0, TKVAL
    add x0, x0, :lo12:TKVAL
    mov w1, #0
1:  cmp w1, #64
    b.hs 2f
    ldr s1, [x0, x1, lsl #2]
    fadd s0, s0, s1              // total
    add w1, w1, #1
    b 1b
2:  adrp x1, G_TOPP
    add x1, x1, :lo12:G_TOPP
    ldr s1, [x1]                 // top-p (was fixed 0.95)
    fmul s1, s0, s1              // cutoff = total*0.95 (s1; s0=total dead? total NEEDED? No (cutoff suffices). s0 dead ✓)
    // walk cumulative to kept set
    adrp x0, TKVAL
    add x0, x0, :lo12:TKVAL
    fmov s0, wzr                 // acc
    mov w1, #0                   // j
3:  cmp w1, #64
    b.hs 4f                      // failsafe (rounding): kept=64
    ldr s2, [x0, x1, lsl #2]
    fadd s0, s0, s2
    fcmp s0, s1
    b.ge 4f                      // acc >= cutoff: kept=j+1, mass=s0
    add w1, w1, #1
    b 3b
4:  cmp w1, #64
    b.eq sel_full                // fell through (w1=64): kept=64
    add w4, w1, #1               // hit path: kept = j+1
    b sel_go
sel_full:
    mov w4, #64
sel_go:
    bl rng_next                  // x0 = u64 (uses x0-x2; w4/s0 safe)
    lsr x0, x0, #40              // top 24 bits
    scvtf s1, x0
    adrp x0, f_inv24
    add x0, x0, :lo12:f_inv24
    ldr s2, [x0]                 // 2^-24
    fmul s1, s1, s2              // frac in [0,1)
    fmul s1, s1, s0              // r = frac*mass
    fmov s2, wzr                 // acc
    mov w5, #0                   // j
sel_walk:
    cmp w5, w4
    b.hs sel_clamp               // failsafe (b.ge below always hits first)
    adrp x0, TKVAL
    add x0, x0, :lo12:TKVAL
    ldr s3, [x0, x5, lsl #2]
    fadd s2, s2, s3
    fcmp s2, s1
    b.ge sel_hit
    add w5, w5, #1
    b sel_walk
sel_clamp:
    sub w5, w4, #1
sel_hit:
    adrp x0, TKIDX
    add x0, x0, :lo12:TKIDX
    ldr w0, [x0, x5, lsl #2]     // token
    b 9f                         // epilogue
0:  // greedy
    adrp x0, LG
    add x0, x0, :lo12:LG
    movz w1, #4, lsl #16
    bl argmax_f32                // w0 = argmax
9:  ldr x30, [sp], #16
    ret

    .section .bss
    .align 3
BBASE:       .skip 8
B_RULES:     .skip 8
B_CHARS:     .skip 8
B_BYTEMAP:   .skip 8
B_NRULES:    .skip 8
B_NCHARS:    .skip 8
    .align 4
SYM:         .skip 2048*4

    .section .rodata
    .align 3
bpe_magic:   .quad 0x4250455F424C4F42
t_bpe:       .asciz "bpe rules: "

    .section .text
    .align 2
/* bpe_load: x0 = bpe.bin path. mmaps read-only, verifies, sets cells. */
bpe_load:
    str x30, [sp, #-16]!
    stp x19, x20, [sp, #-16]!
    mov x1, x0                   // path
    mov x0, #-100
    mov x2, #0
    mov x3, #0
    mov x8, #56
    svc #0                       // openat
    cmp x0, #0
    b.lt 9f
    str x0, [sp, #-16]!          // fd
    mov x8, #62
    ldr x0, [sp]                 // fd
    mov x1, #0
    mov x2, #2
    svc #0                       // lseek END -> size (unused; mmap needs len! USE size)
    cmp x0, #0
    b.le 9f
    mov x20, x0                  // size
    mov x8, #222
    mov x0, #0
    mov x1, x20
    mov x2, #1                   // PROT_READ
    mov x3, #2
    ldr x4, [sp]                 // fd
    mov x5, #0
    svc #0
    cmp x0, #0
    b.lt 9f
    mov x19, x0                  // base
    ldr x0, [sp], #16            // fd
    mov x8, #57
    svc #0                       // close
    // verify magic
    ldr x0, [x19]
    adrp x1, bpe_magic
    add x1, x1, :lo12:bpe_magic
    ldr x1, [x1]
    cmp x0, x1
    b.ne 9f
    // cells: BBASE=base; NRULES=[base+8]; NCHARS=[base+16]
    adrp x0, BBASE
    add x0, x0, :lo12:BBASE
    str x19, [x0]
    ldr x0, [x19, #8]            // nrules
    adrp x1, B_NRULES
    add x1, x1, :lo12:B_NRULES
    str x0, [x1]
    // RULES = base+24
    add x0, x19, #24
    adrp x1, B_RULES
    add x1, x1, :lo12:B_RULES
    str x0, [x1]
    // CHARS = base+24+nrules*16 ; NCHARS = [base+16]
    ldr x0, [x19, #16]           // nchars
    adrp x1, B_NCHARS
    add x1, x1, :lo12:B_NCHARS
    str x0, [x1]
    ldr x0, [x19, #8]            // nrules
    lsl x0, x0, #4               // *16
    add x0, x19, x0
    add x2, x0, #24              // x2 = charmap base (x2 free? YES (recomputed). SAVE to cell + keep in x2? need x2 later? no. STORE then RELOAD when needed below.)
    adrp x1, B_CHARS
    add x1, x1, :lo12:B_CHARS
    str x2, [x1]
    // BYTEMAP = charmap + nchars*8
    ldr x0, [x19, #16]           // nchars
    lsl x0, x0, #3               // *8
    add x0, x2, x0               // + charmap base (x2 intact? YES (no calls/overwrites between) ✓)
    adrp x1, B_BYTEMAP
    add x1, x1, :lo12:B_BYTEMAP
    str x0, [x1]
    // print n_rules (verify 514906)
    adrp x0, t_bpe
    add x0, x0, :lo12:t_bpe
    bl print_cstr
    ldr x0, [x19, #8]
    bl print_u64
    bl print_nl
    ldp x19, x20, [sp], #16
    ldr x30, [sp], #16
    ret
    b 9f
9:  adrp x0, e_vload
    add x0, x0, :lo12:e_vload
    bl print_cstr
    mov x0, #1
    mov x8, #SYS_EXIT
    svc #0

/* utf8_decode: x0 = ptr, x1 = end. Returns w0 = cp (or U+FFFD), x1 = next.
 * Bounds-checked (never reads past end). Uses w2-w4 (+x4 saved ptr). No calls. */
utf8_decode:
    mov x4, x0                   // save ptr
    cmp x0, x1
    b.hs 9f                      // empty
    ldrb w2, [x0]
    cmp w2, #0x80
    b.lo 8f                      // ASCII
    cmp w2, #0xE0
    b.lo 0f                      // 2-byte
    cmp w2, #0xF0
    b.lo 1f                      // 3-byte
    cmp w2, #0xF8
    b.hs 9f                      // 5+ byte lead: invalid
    add x3, x0, #4               // need 4
    cmp x3, x1
    b.hi 9f
    and w2, w2, #0x07
    lsl w2, w2, #18
    ldrb w3, [x0, #1]
    and w3, w3, #0x3F
    lsl w3, w3, #12
    orr w2, w2, w3
    ldrb w3, [x0, #2]
    and w3, w3, #0x3F
    lsl w3, w3, #6
    orr w2, w2, w3
    ldrb w3, [x0, #3]
    and w3, w3, #0x3F
    orr w0, w2, w3               // cp
    add x1, x4, #4               // next
    ret
8:  mov w0, w2                   // ASCII cp
    add x1, x4, #1               // next
    ret
0:  add x3, x0, #2               // need 2
    cmp x3, x1
    b.hi 9f
    and w2, w2, #0x1F
    lsl w2, w2, #6
    ldrb w3, [x0, #1]
    and w3, w3, #0x3F
    orr w0, w2, w3               // cp
    add x1, x4, #2               // next
    ret
1:  add x3, x0, #3               // need 3
    cmp x3, x1
    b.hi 9f
    and w2, w2, #0x0F
    lsl w2, w2, #12
    ldrb w3, [x0, #1]
    and w3, w3, #0x3F
    lsl w3, w3, #6
    orr w2, w2, w3
    ldrb w3, [x0, #2]
    and w3, w3, #0x3F
    orr w0, w2, w3               // cp
    add x1, x4, #3               // next
    ret
9:  mov w0, #0xFFFD
    add x1, x4, #1               // skip bad byte (guarantees progress)
    ret
/* char_lookup: w0 = codepoint -> w0 = id, or -1. Binary search charmap.
 * Uses x0-x4. No calls. */
char_lookup:
    adrp x1, B_CHARS
    add x1, x1, :lo12:B_CHARS
    ldr x1, [x1]                 // base (each entry cp:u32,id:u32)
    adrp x2, B_NCHARS
    add x2, x2, :lo12:B_NCHARS
    ldr x2, [x2]                 // n
    mov x3, #0                   // lo
0:  cmp x3, x2
    b.hs 1f                      // miss -> -1
    add x4, x3, x2
    lsr x4, x4, #1               // mid
    lsl x5, x4, #3               // mid*8 -- x5 free? YES (uses x0-x5, no calls; caller recomputes). SAFE (leaf).
    add x5, x1, x5
    ldr w6, [x5]                 // entry cp -- w6? uses x0-x6 now. SAFE (leaf, caller recomputes).
    cmp w0, w6
    b.eq 2f
    b.lo 3f                      // cp < entry: hi = mid... WAIT hi/lo update needs care: standard: if cp < entry.cp: hi = mid else lo = mid+1. x2 = hi, x3 = lo.
    add x3, x4, #1               // lo = mid+1
    b 0b
3:  mov x2, x4                   // hi = mid
    b 0b
2:  ldr w0, [x5, #4]             // id
    ret
1:  mov w0, #-1
    ret

    .section .rodata
    .align 3
t_enc:       .asciz "enc: "
t_ids2:      .asciz " -> "
t_enc_over:  .asciz "encode overflow (prompt too long)\n"

    .section .text
    .align 2
encode:
    str x30, [sp, #-16]!
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    str x25, [sp, #-16]!
    mov x19, x0                  // text
    add x20, x0, x1              // end
    mov x21, x2                  // out
    mov x22, x3                  // max (cap; informational: SYM fixed 2048)
    adrp x23, SYM
    add x23, x23, :lo12:SYM
    mov x24, x19                 // p = text
    mov w25, #0                  // n = 0
0:  cmp x24, x20
    b.hs 1f                      // init done
    mov x26, x24                 // byte-start (fallback needs it)
    ldrb w6, [x24]
    cmp w6, #0x3C                // '<'? (w6, NOT w0: x0 is ct_tab base!)
    b.ne ctrl_no
    // table walk: ct_tab entries (ptr,len,id); x0-x5 scratch (all dead here ✓)
    adrp x0, ct_tab
    add x0, x0, :lo12:ct_tab
    mov w1, #0                   // entry idx
ctrl_entry:
    cmp w1, #7
    b.hs ctrl_no                 // no match: normal path
    lsl x2, x1, #4               // idx*16
    add x2, x0, x2               // &entry
    ldr x3, [x2]                 // lit ptr
    ldr w4, [x2, #8]             // len
    ldr w5, [x2, #12]            // id
    add x2, x24, x4              // p+len
    cmp x2, x20
    b.hi ctrl_next               // not enough bytes left
    mov x2, #0                   // k
ctrl_cmp:
    cmp x2, x4
    b.hs ctrl_hit                // all matched
    ldrb w6, [x24, x2]           // text[k] (w6 preserves x0=ct_tab, x3=lit!)
    ldrb w7, [x3, x2]            // lit[k] (x7 free: no calls in check)
    cmp w6, w7
    b.ne ctrl_next
    add x2, x2, #1
    b ctrl_cmp
ctrl_next:
    // (x3 dead here; recomputed per entry)
    add w1, w1, #1
    b ctrl_entry
ctrl_hit:
    // store id (w5), advance p, count
    cmp w25, #2048
    b.hs 9f                      // encode overflow exit
    str w5, [x23, x25, lsl #2]   // SYM[n] = id
    add x25, x25, #1
    add x24, x24, x4             // p += len
    b 0b                         // phase-1 top
ctrl_no:
    mov x0, x24
    mov x1, x20
    bl utf8_decode               // w0 = cp, x1 = next (uses x0-x4; x26 safe)
    mov x24, x1                  // p = next
    cmp w0, #0x20
    b.ne 2f
    mov w0, #0x2581              // space -> U+2581
2:  bl char_lookup               // w0 = id or -1 (uses x0-x6; x26 safe)
    cmp w0, #0
    b.ge 3f                      // hit
    adrp x0, B_BYTEMAP
    add x0, x0, :lo12:B_BYTEMAP
    ldr x0, [x0]                 // bytemap
4:  cmp x26, x24
    b.hs 0b                      // fallback bytes done -> next char
    ldrb w1, [x26]
    ldr w1, [x0, x1, lsl #2]     // bytemap[b]
    cmp w25, #2048
    b.hs 9f
    str w1, [x23, x25, lsl #2]
    add x25, x25, #1
    add x26, x26, #1
    b 4b
3:  cmp w25, #2048
    b.hs 9f
    str w0, [x23, x25, lsl #2]
    add x25, x25, #1
    b 0b
1:  // Phase 2: merge loop (no calls; x26+ free as temps, x25 = n live)
    adrp x8, B_RULES
    add x8, x8, :lo12:B_RULES
    ldr x8, [x8]                 // rules base
    adrp x9, B_NRULES
    add x9, x9, :lo12:B_NRULES
    ldr x9, [x9]                 // nrules
5:  movn w26, #0                   // best_rank = INF (-1)
    mov w2, #0                   // j = 0
6:  sub w3, w25, #1              // n-1
    cmp w2, w3
    b.hs 7f                      // scan done (j >= n-1)
    ldr w4, [x23, x2, lsl #2]    // left = sym[j]
    add x6, x2, #1                 // j+1 (x6: bsearch-lo, dead here ✓)
    ldr w5, [x23, x6, lsl #2]    // right = sym[j+1]
    lsl x10, x5, #32             // target key = left | right<<32
    orr x10, x10, x4             // (x4/x5 zero-extended? w-reg writes zero-extend to x ✓)
    mov x6, #0                   // lo
    mov x7, x9                   // hi = nrules
bs_loop:
    cmp x6, x7
    b.hs pair_next               // miss -> j++
    add x11, x6, x7
    lsr x11, x11, #1             // mid
    lsl x12, x11, #4
    add x12, x8, x12             // &rule[mid]
    ldr x13, [x12]               // stored key (left | right<<32)
    cmp x10, x13
    b.eq bs_hit
    b.lo bs_lo                   // target < stored: hi = mid
    add x6, x11, #1              // lo = mid+1
    b bs_loop
bs_lo:
    mov x7, x11                  // hi = mid
    b bs_loop
bs_hit:
    ldr w13, [x12, #12]          // rank
    cmp w13, w26
    b.hs pair_next               // not better (strict < keeps leftmost ✓)
    mov w26, w13                 // best_rank
    mov w0, w2                   // best_j
    ldr w1, [x12, #8]            // best_z = result
pair_next:
    add w2, w2, #1               // j++
    b 6b                         // scan top
7:  // scan done: any merge?
    cmn w26, #1                   // best_rank == -1 (none)?
    b.eq 8f                      // none -> copy out
    str w1, [x23, x0, lsl #2]    // sym[best_j] = best_z
    // memmove [best_j+2, n-1] down by one (result at best_j must survive!)
    add w2, w0, #2               // k = best_j+2
mm_loop:
    cmp w2, w25
    b.hs mm_done
    ldr w3, [x23, x2, lsl #2]    // sym[k]
    sub w4, w2, #1
    str w3, [x23, x4, lsl #2]    // sym[k-1] (x4: k-1 zero-extended ✓)
    add w2, w2, #1
    b mm_loop
mm_done:
    sub w25, w25, #1             // n--
    b 5b                         // next pass
8:  // Phase 3: copy SYM[0..n-1] to out (x21), return n
    cmp w25, w22                 // n <= caller max?
    b.hi 9f
    mov w2, #0                   // k
cp_loop:
    cmp w2, w25
    b.hs cp_done
    ldr w3, [x23, x2, lsl #2]
    str w3, [x21, x2, lsl #2]
    add w2, w2, #1
    b cp_loop
cp_done:
    mov w0, w25                  // return n
    ldr x25, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ldr x30, [sp], #16
    ret
9:  adrp x0, t_enc_over
    add x0, x0, :lo12:t_enc_over
    bl print_cstr
    mov x0, #1
    mov x8, #SYS_EXIT
    svc #0
    .section .bss
    .align 4
ENC_OUT:     .skip 512*4

    .section .rodata
et0:  .asciz "Hello world"
et1:  .asciz "The capital of France is"
et2:  .asciz "Hello"
et3:  .asciz "\n"
et4:  .asciz " a"
et5:  .asciz "caf\303\251 \346\227\245\346\234\254\350\252\236"
et6:  .asciz "Name"
et7:  .asciz "THE"
et8:  .asciz "  spaces  "
et9:  .asciz "x=1+2*3"
et10: .asciz "\303\251"
et11: .asciz "\346\227\245"
et12: .asciz "A"
et13: .asciz "\360\237\230\200"
et14: .asciz "\001"
et15: .asciz "\177"
et16: .asciz "\342\200\213"
et17: .asciz " Hello"
et18: .asciz "<bos>"
et19: .asciz "<start_of_turn>Hi<end_of_turn>"
et20: .asciz "a<eos>b"
et21: .asciz "<escape>"
et22: .asciz "<start_function_declaration>x<end_function_declaration>"

    .section .text
    .align 2
/* enc_one: x0 = asciz text. Prints "enc: <text> -> <ids...>". */
enc_one:
    str x30, [sp, #-16]!
    stp x19, x20, [sp, #-16]!
    mov x19, x0                  // text (print_* and encode preserve x19+ ✓)
    adrp x0, t_enc
    add x0, x0, :lo12:t_enc
    bl print_cstr
    mov x0, x19
    bl print_cstr                // echo text
    adrp x0, t_ids2
    add x0, x0, :lo12:t_ids2
    bl print_cstr
    mov x0, x19
    bl strlen_asm                // x0 = len
    mov x1, x0                   // len
    mov x0, x19                  // text
    adrp x2, ENC_OUT
    add x2, x2, :lo12:ENC_OUT
    mov w3, #512
    bl encode                    // w0 = n (uses SYM; preserves x19+ ✓)
    mov w20, w0                  // n (w20: encode preserves x20 ✓ SAFE)
    adrp x19, ENC_OUT
    add x19, x19, :lo12:ENC_OUT  // REUSE x19 (text dead ✓; encode preserved x19 but we overwrite deliberately ✓)
    mov w21, #0                  // k -- w21? x21 dead? x21 was out (dead ✓). BUT encode preserves x21; we overwrite w21 deliberately ✓. w21 vs... print_u64 clobbers x0-x2? print clobbers x0-x2,x8,x9 (per print_cstr comment; print_u64 similar?). w21 SAFE if print_* use x0-x9 max. VERIFY print_u64/print_sp regs! (check below before trusting).
    b 0f
0:  cmp w21, w20                 // k < n?
    b.hs 1f
    ldr w0, [x19, x21, lsl #2]   // id
    bl print_u64                 // clobbers x0-x3,x8,x9; x19-x21 safe
    bl print_sp
    add w21, w21, #1
    b 0b
1:  bl print_nl
    ldp x19, x20, [sp], #16
    ldr x30, [sp], #16
    ret

/* encode_test: runs enc_one over et0..et17. */
encode_test:
    str x30, [sp, #-16]!
    bl rule_probe
    adrp x0, et0
    add x0, x0, :lo12:et0
    bl enc_one
    adrp x0, et1
    add x0, x0, :lo12:et1
    bl enc_one
    adrp x0, et2
    add x0, x0, :lo12:et2
    bl enc_one
    adrp x0, et3
    add x0, x0, :lo12:et3
    bl enc_one
    adrp x0, et4
    add x0, x0, :lo12:et4
    bl enc_one
    adrp x0, et5
    add x0, x0, :lo12:et5
    bl enc_one
    adrp x0, et6
    add x0, x0, :lo12:et6
    bl enc_one
    adrp x0, et7
    add x0, x0, :lo12:et7
    bl enc_one
    adrp x0, et8
    add x0, x0, :lo12:et8
    bl enc_one
    adrp x0, et9
    add x0, x0, :lo12:et9
    bl enc_one
    adrp x0, et10
    add x0, x0, :lo12:et10
    bl enc_one
    adrp x0, et11
    add x0, x0, :lo12:et11
    bl enc_one
    adrp x0, et12
    add x0, x0, :lo12:et12
    bl enc_one
    adrp x0, et13
    add x0, x0, :lo12:et13
    bl enc_one
    adrp x0, et14
    add x0, x0, :lo12:et14
    bl enc_one
    adrp x0, et15
    add x0, x0, :lo12:et15
    bl enc_one
    adrp x0, et16
    add x0, x0, :lo12:et16
    bl enc_one
    adrp x0, et17
    add x0, x0, :lo12:et17
    bl enc_one
    adrp x0, et18
    add x0, x0, :lo12:et18
    bl enc_one
    adrp x0, et19
    add x0, x0, :lo12:et19
    bl enc_one
    adrp x0, et20
    add x0, x0, :lo12:et20
    bl enc_one
    adrp x0, et21
    add x0, x0, :lo12:et21
    bl enc_one
    adrp x0, et22
    add x0, x0, :lo12:et22
    bl enc_one
    ldr x30, [sp], #16
    ret

    .section .rodata
t_probe:     .asciz "probe(He): "

    .section .text
    .align 2
/* rule_probe: bsearches (236814,236744), prints result + rank. Expect 2209 5240. */
rule_probe:
    str x30, [sp, #-16]!
    adrp x8, B_RULES
    add x8, x8, :lo12:B_RULES
    ldr x8, [x8]
    adrp x9, B_NRULES
    add x9, x9, :lo12:B_NRULES
    ldr x9, [x9]
    movz w4, #0x9D0E              // left H = 236814 (0x39D0E)
    movk w4, #0x3, lsl #16
    movz w5, #0x9CC8              // right e = 236744 (0x39CC8)
    movk w5, #0x3, lsl #16
    lsl x10, x5, #32
    orr x10, x10, x4
    mov x6, #0
    mov x7, x9
rp_bs:
    cmp x6, x7
    b.hs rp_miss
    add x11, x6, x7
    lsr x11, x11, #1
    lsl x12, x11, #4
    add x12, x8, x12
    ldr x13, [x12]
    cmp x10, x13
    b.eq rp_hit
    b.lo rp_lo
    add x6, x11, #1
    b rp_bs
rp_lo:
    mov x7, x11
    b rp_bs
rp_hit:
    ldr w0, [x12, #8]            // result
    ldr w1, [x12, #12]           // rank
    b rp_out
rp_miss:
    mov w0, #-1
    mov w1, #-1
rp_out:
    stp x0, x1, [sp, #-16]!      // save (print clobbers x0-x9? print_u64 uses x0-x3,x9! SAVE both. x0/x1 on stack ✓)
    adrp x0, t_probe
    add x0, x0, :lo12:t_probe
    bl print_cstr
    ldp x0, x1, [sp], #16        // restore (x0=result, x1=rank)
    stp x0, x1, [sp, #-16]!      // re-save rank (print_u64 clobbers x1? uses x0-x3! YES clobbers x1. SAVE.)
    bl print_u64                 // result
    bl print_sp
    ldp x0, x1, [sp], #16        // x1 = rank (x0 = stale result, dead)
    mov x0, x1
    bl print_u64                 // rank
    bl print_nl
    ldr x30, [sp], #16
    ret

    .section .bss
    .align 4
INBUF:       .skip 4096
J_chat:      .skip 8

    .section .rodata
t_chat:      .asciz "chat ready (sample mode; empty line skips, EOF quits)\n"
t_gt:        .asciz "> "
t_reset:     .asciz "[context full, resetting]\n"
t_tpl:       .asciz "tpl: "
t_nobpe:     .asciz "chat mode needs bpe.bin (argv[5])\n"
t_toolong:   .asciz "prompt too long for context\n"
t_lgchk:     .asciz "lg nan inf: "

    .section .text
    .align 2
/* readline: -> w0 = len (0 = empty line) or -1 (EOF, no bytes).
 * Byte-at-a-time (no leftover logic; prompts are short).
 * x1=cursor, x2=read-count(1), w4=len, w3=cap. Leaf (x19+ safe). */
readline:
    adrp x1, INBUF
    add x1, x1, :lo12:INBUF      // cursor
    mov w4, #0                   // len (x4: SVC preserves x1-x7 ✓)
    mov w3, #4095                // cap
    mov x2, #1                   // count (SVC preserves x2 ✓ set once)
0:  cmp w4, w3
    b.hs 2f                      // full: truncate
    mov x0, #0                   // fd stdin
    mov x8, #63                  // read
    svc #0                       // n (x1=cursor, x2=1 preserved ✓)
    cmp x0, #0
    b.le 1f                      // EOF/error
    ldrb w0, [x1]
    cmp w0, #10                  // \n?
    b.eq 2f                      // done (strip)
    add x1, x1, #1
    add w4, w4, #1
    b 0b
1:  cbz w4, 3f                   // EOF with no bytes -> -1
    mov w0, w4                   // partial line -> len
    ret
3:  mov w0, #-1
    ret
2:  mov w0, w4                   // len (0 = empty line)
    ret

/* chat: interactive REPL. x19=pos, x20=turn, x21=n, x22=i/count, x23=tok. */
chat:
    str x30, [sp, #-16]!
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    adrp x0, t_chat
    add x0, x0, :lo12:t_chat
    bl print_cstr
    mov x19, #0                  // pos = 0
    mov x20, #0                  // turn = 0
chat_next:
    adrp x0, t_gt
    add x0, x0, :lo12:t_gt
    bl print_cstr                // "> " (x19-x23 safe across prints ✓)
    bl readline                  // w0 = len or -1 (uses x0-x3; x19+ safe ✓)
    cmp w0, #-1
    b.eq chat_bye
    cbz w0, chat_next            // empty line -> reprompt
    // strip trailing \r
    adrp x1, INBUF
    add x1, x1, :lo12:INBUF
    sub x2, x0, #1               // x0 = len (w0 -> x0 zero-extended ✓)
    ldrb w2, [x1, x2]
    cmp w2, #13
    b.ne 0f
    sub w0, w0, #1               // len--
    cbz w0, chat_next            // line was just \r
0:  mov x24, x0                  // hold len (both paths ✓)
chat_enc:
    // x24 = len (held). tpl_build assembles template into ENC_OUT.
    bl tpl_build                 // w0 = n (preserves x19+; restores x24 ✓)
    mov x21, x0                  // n
    // transparency: print "tpl: n ids..."
    adrp x0, t_tpl
    add x0, x0, :lo12:t_tpl
    bl print_cstr
    mov x0, x21
    bl print_u64                 // n
    bl print_sp
    adrp x22, ENC_OUT
    add x22, x22, :lo12:ENC_OUT  // x22 = base (x22 free here? prefill not started ✓; fwd not yet called ✓ SAFE)
    mov w23, #0                  // k -- w23 free? tok not set yet ✓ SAFE
tpl_pl:
    cmp w23, w21
    b.hs tpl_pd
    ldr w0, [x22, x23, lsl #2]   // ids[k] (x23 offset ✓)
    bl print_u64                 // (x22/w23/w21 safe? prints use x0-x9 ✓)
    bl print_sp
    add w23, w23, #1
    b tpl_pl
tpl_pd:
    bl print_nl
    // capacity: pos + n + (turn==0) + 128 > 480?
    add x0, x19, x21             // pos + n
    cbnz x20, chat_nobos
    add x0, x0, #1               // +bos
chat_nobos:
    adrp x1, G_GENCAP
    add x1, x1, :lo12:G_GENCAP
    ldr x1, [x1]
    add x0, x0, x1               // +gen cap (x1 free here ✓)
    cmp x0, #480
    b.lo chat_prefill
    adrp x0, t_reset
    add x0, x0, :lo12:t_reset
    bl print_cstr
    mov x19, #0
    mov x20, #0
    b chat_enc                   // re-encode same line fresh (INBUF+x24 intact ✓)
chat_prefill:
    // bos on first turn
    cbnz x20, chat_feed
    adrp x0, G_pos
    add x0, x0, :lo12:G_pos
    str w19, [x0]                // G_pos = pos
    mov w0, #2                   // bos
    bl fwd_token                 // (preserves x19+ ✓)
    add x19, x19, #1             // pos++
chat_feed:
    mov x22, #0                  // i = 0
chat_floop:
    cmp x22, x21                 // i < n?
    b.hs chat_gen
    adrp x0, G_pos
    add x0, x0, :lo12:G_pos
    str w19, [x0]
    adrp x0, ENC_OUT
    add x0, x0, :lo12:ENC_OUT
    ldr w0, [x0, x22, lsl #2]    // ids[i] (x22 offset ✓)
    bl fwd_token
    add x19, x19, #1             // pos++
    add x22, x22, #1             // i++
    b chat_floop
chat_gen:
    // tok = ENC_OUT[n-1] (n>=1: encode of len>0 yields >=1 symbol ✓)
    adrp x0, ENC_OUT
    add x0, x0, :lo12:ENC_OUT
    sub x1, x21, #1
    ldr w23, [x0, x1, lsl #2]    // tok = last
    adrp x0, G_GENCAP
    add x0, x0, :lo12:G_GENCAP
    ldr x24, [x0]                // gencap (x24=len dead here ✓; x19+ safe ✓)
    mov x22, #0                  // count = 0
chat_gloop:
    cmp x22, x24
    b.hs chat_end                // gen cap
    adrp x0, G_pos
    add x0, x0, :lo12:G_pos
    str w19, [x0]
    mov w0, w23
    bl fwd_token                 // LG set
    bl sampler                   // w0 = token (G_MODE=1 in chat → samples)
    mov w23, w0                  // tok
    cmp w23, #1
    b.eq chat_end                // eos
    mov x0, x23
    bl put_surf                  // buffer (preserves x19+ ✓)
    add x19, x19, #1             // pos++
    add x22, x22, #1
    b chat_gloop
chat_end:
    // close model turn: feed EOT (106)
    adrp x0, G_pos
    add x0, x0, :lo12:G_pos
    str w19, [x0]
    mov w0, #106
    bl fwd_token
    add x19, x19, #1             // pos++
    bl out_flush
    bl print_nl
    add x20, x20, #1             // turn++
    b chat_next
chat_bye:
    mov x0, #0
    mov x8, #93
    svc #0

    .section .bss
    .align 4
TBUF:        .skip 4096

    .section .rodata
t_model:     .asciz "model\n"

    .section .text
    .align 2
/* tpl_build: INBUF+x24(len) -> ENC_OUT assembled turn ids. Returns w0 = n.
 * Layout: [105] + enc("user\n"+line) + [106,107,105] + enc("model\n").
 * Bit-exact vs HF apply_chat_template (verified: turn1 [2]+tpl == oracle
 * [2,105,2364,107,9259,106,107,105,4368,107]; multi-turn structure matches
 * official re-render incl EOT-close). With turn-0 bos, this IS the official
 * gemma-3 template, not raw input.
 * Saves/restores x24 (encode clobbers it). Uses TBUF/ENC_OUT. No x19+ touched. */
tpl_build:
    str x30, [sp, #-16]!
    str x24, [sp, #-16]!          // len (encode clobbers x24!)
    // TBUF = "user\n" + line
    adrp x0, TBUF
    add x0, x0, :lo12:TBUF
    mov w1, #0x7375              // "us" ([0]='u',[1]='s'; LE strh!)
    strh w1, [x0]
    mov w1, #0x7265              // "er"
    strh w1, [x0, #2]
    mov w1, #10                  // "\n"
    strb w1, [x0, #4]            // TBUF[0..4] = "user\n" (exact, no spill ✓)
    // copy line INBUF[0..x24-1] -> TBUF[5..]
    adrp x1, INBUF
    add x1, x1, :lo12:INBUF
    add x0, x0, #5               // dst = TBUF+5 (x0 recomputed below ✓)
    mov x2, #0                   // i
0:  cmp x2, x24
    b.hs 1f
    ldrb w3, [x1, x2]
    strb w3, [x0, x2]
    add x2, x2, #1
    b 0b
1:  // encode(TBUF, len+5 -> ENC_OUT+1, max 500)
    adrp x0, TBUF
    add x0, x0, :lo12:TBUF
    add x1, x24, #5              // len+5
    adrp x2, ENC_OUT
    add x2, x2, :lo12:ENC_OUT
    add x2, x2, #4               // +1 slot (ENC_OUT[0] = 105 below)
    mov w3, #500
    bl encode                    // w0 = nu (x24 clobbered; restored at exit ✓)
    str x0, [sp, #-16]!          // save nu (encode#2 clobbers x0-x8!)
    mov x1, x0                   // nu
    adrp x0, ENC_OUT
    add x0, x0, :lo12:ENC_OUT
    mov w2, #105
    str w2, [x0]                 // [0] = SOT
    lsl x3, x1, #2               // nu*4
    add x2, x0, x3
    add x2, x2, #4               // x2 = &ENC_OUT[1+nu]
    mov w3, #106
    str w3, [x2]
    mov w3, #107
    str w3, [x2, #4]
    mov w3, #105
    str w3, [x2, #8]             // [106,107,105]
    // encode("model\n" -> ENC_OUT[4+nu], max 8)
    adrp x0, t_model
    add x0, x0, :lo12:t_model
    mov x1, #6
    add x2, x2, #12              // &ENC_OUT[4+nu]
    mov w3, #8
    bl encode                    // w0 = nm
    mov x1, x0                   // nm
    ldr x0, [sp], #16            // nu
    add x0, x0, x1               // nu + nm
    add x0, x0, #4               // n = 1 + nu + 3 + nm
    ldr x24, [sp], #16           // restore len
    ldr x30, [sp], #16
    ret


    .section .rodata
    .align 3
ct_bos:      .ascii "<bos>"
ct_eos:      .ascii "<eos>"
ct_eot:      .ascii "<end_of_turn>"
ct_sot:      .ascii "<start_of_turn>"
ct_sfd:      .ascii "<start_function_declaration>"
ct_efd:      .ascii "<end_function_declaration>"
ct_esc:      .ascii "<escape>"
    .align 3
ct_tab:       // (ptr, len, id) x7
    .quad ct_bos
    .word 5, 2
    .quad ct_eos
    .word 5, 1
    .quad ct_eot
    .word 13, 106
    .quad ct_sot
    .word 15, 105
    .quad ct_esc
    .word 8, 52
    .quad ct_efd
    .word 26, 47
    .quad ct_sfd
    .word 28, 46

    .section .text
    .align 2

    .section .bss
    .align 3
J_f:         .skip 8

    .section .text
    .align 2
/* read_all: read stdin to INBUF until EOF/full. Returns w0 = len (0 = empty).
 * x1=cursor, x2=COUNT (kernel count reg!), w4=len, w3=cap (reloaded per iter).
 * Uses x0-x4. Leaf. */
read_all:
    adrp x1, INBUF
    add x1, x1, :lo12:INBUF      // cursor
    mov w4, #0                   // len
0:  mov w3, #4095                // cap (fresh; no preservation assumptions)
    cmp w4, w3
    b.hs 1f                      // full -> done
    mov x0, #0                   // fd
    sub x2, x3, x4               // count = cap-len (x2! kernel reads x2)
    mov x8, #63                  // read
    svc #0                       // n (x1/x2/x4 survive? x1 proven by readline; x2(count)/x4(len) recomputed/advanced below from RETURNED n only... x4 MUST survive (len accumulates)! readline proves w-len survives SVC ✓ same pattern. x2 recomputed per iter ✓ immune.)
    cmp x0, #0
    b.le 1f                      // EOF/error -> done
    add x1, x1, x0               // cursor += n
    add w4, w4, w0               // len += n
    b 0b
1:  mov w0, w4
    ret
/* file_run: raw-prefill one-shot. INBUF=len via read_all; encode; prefill
 * (no bos/template; prompt carries its own controls); gen; print; exit.
 * Holds x19=pos, x21=n, x22=count, x23=tok (all fwd-safe, x19+ ✓). */
file_run:
    str x30, [sp, #-16]!
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    str x23, [sp, #-16]!
    bl read_all                  // w0 = len
    mov x19, x0                  // hold len (x19 free until pos set ✓)
    cbz x19, file_bye            // empty -> exit 0
    mov x0, x19                  // len -- WAIT x0=len (just restored ✓). But cbz AFTER prints (prints ran even for len 0 (shows 0, fine, then bye ✓).
    mov x1, x0                   // len
    adrp x0, INBUF
    add x0, x0, :lo12:INBUF
    adrp x2, ENC_OUT
    add x2, x2, :lo12:ENC_OUT
    mov w3, #512
    bl encode                    // w0 = n (preserves x19+ ✓)
    mov x21, x0                  // n
    // transparency: print prefill ids (verify vs HF tokenizer)
    adrp x0, t_tpl
    add x0, x0, :lo12:t_tpl
    bl print_cstr
    mov x0, x21
    bl print_u64
    bl print_sp
    adrp x22, ENC_OUT
    add x22, x22, :lo12:ENC_OUT  // x22 free? prefill not started ✓ (x22=i later, set in file_prefill ✓ SAFE)
    mov w23, #0                  // k (w23 free? tok set later ✓ SAFE)
file_pl:
    cmp w23, w21
    b.hs file_pd
    ldr w0, [x22, x23, lsl #2]
    bl print_u64                 // (x21/x22/w23 safe? prints x0-x9 ✓)
    bl print_sp
    add w23, w23, #1
    b file_pl
file_pd:
    bl print_nl
    mov x19, #0                  // pos = 0
    // capacity: n + gencap <= 480?
    adrp x0, G_GENCAP
    add x0, x0, :lo12:G_GENCAP
    ldr x0, [x0]
    add x0, x21, x0              // n + gencap
    cmp x0, #480
    b.lo file_prefill
    adrp x0, t_toolong
    add x0, x0, :lo12:t_toolong
    bl print_cstr
    mov x0, #1
    mov x8, #93
    svc #0
file_prefill:
    mov x22, #0                  // i = 0
file_floop:
    cmp x22, x21
    b.hs file_gen
    adrp x0, G_pos
    add x0, x0, :lo12:G_pos
    str w19, [x0]
    adrp x0, ENC_OUT
    add x0, x0, :lo12:ENC_OUT
    ldr w0, [x0, x22, lsl #2]    // ids[i]
    bl fwd_token
    add x19, x19, #1
    add x22, x22, #1
    b file_floop
file_gen:
    adrp x0, ENC_OUT
    add x0, x0, :lo12:ENC_OUT
    sub x1, x21, #1
    ldr w23, [x0, x1, lsl #2]    // tok = last (n>=1? encode of len>0 gives >=1 ✓ (each byte >=1 symbol; controls give 1). TRUST (like chat ✓).)
    adrp x0, G_GENCAP
    add x0, x0, :lo12:G_GENCAP
    ldr x24, [x0]                // gencap (x24 free? file_run holds x19,x21,x22,x23. x24 NOT saved! fwd preserves x19+ (incl x24 ✓) but x24 UNSAVED across... file_run never sets x24 except here; no one else needs it. SAFE without save (like chat's x24? chat SAVES x24 (probing... chat saves x23,x24 (I added). For file_run I save x23 only. x24 used as gencap-holder (fwd preserves ✓, no calls clobber x19+ (put_surf? preserves x19+ per decode comment ✓). SAFE.)
    mov x22, #0                  // count
file_gloop:
    cmp x22, x24
    b.hs file_end
    adrp x0, G_pos
    add x0, x0, :lo12:G_pos
    str w19, [x0]
    mov w0, w23
    bl fwd_token
    cbnz x22, lg_skip            // first iter only (w0 dead: sampler overwrites)
    bl lgcheck                   // (x0-x6 scratch; x19+ holds safe ✓)
lg_skip:
    bl sampler
    mov w23, w0                  // tok                  // tok
    cmp w23, #1
    b.eq file_end                // eos
    mov x0, x23
    bl put_surf
    add x19, x19, #1
    add x22, x22, #1
    b file_gloop
file_end:
    bl out_flush
    bl print_nl
file_bye:
    mov x0, #0
    mov x8, #93
    svc #0

    .section .text
    .align 2
/* lgcheck TEMP: scans LG[0..262143] (f32), counts NaN/inf via bits. Prints counts. */
lgcheck:
    str x30, [sp, #-16]!
    adrp x0, LG
    add x0, x0, :lo12:LG
    mov w1, #0                   // i
    mov w2, #0                   // nan count
    mov w3, #0                   // inf count (exp==255, mantissa==0)
    // need n=262144 = 0x40000: cmp w1, #0x40000? 0x40000 > 4095 (movz? use lsr? Load via movz+movk? 0x40000 = 4<<16: movz w4, #4, lsl #16.
    movz w4, #4, lsl #16         // 262144
0:  cmp w1, w4
    b.hs 1f
    ldr w5, [x0, x1, lsl #2]     // bits (x1 offset ✓)
    lsr w6, w5, #23              // exp+sign? exp = (bits>>23)&0xFF. Use ubfx? ubfx w6, w5, #23, #8 (untested? ubfx basic? Use lsr+and (proven mov/and?). and w6, w6? Need mask 0xFF (movz ✓). DO lsr+and.
    and w6, w6, #0xFF            // BUG: and-imm 0xFF ✓ encodable. But w6 currently = bits>>23 (includes sign at bit8!). and 0xFF keeps exp+sign? bits>>23 = [sign, exp(8)] (9 bits). &0xFF keeps exp (low 8)? sign is bit 8 (0x100), &0xFF clears it ✓ exp only ✓.
    cmp w6, #0xFF
    b.ne 2f                      // exp != 255: finite
    lsl w6, w5, #9               // mantissa <<9 (drop sign+exp)? mantissa = bits & 0x7FFFFF. Use: lsl to drop top 9, then... check nonzero via: lsl w6, w5, #9; cbnz (nonzero = NaN, else inf).
    cbnz w6, 3f                  // NaN
    add w3, w3, #1               // inf
    b 2f
3:  add w2, w2, #1               // NaN
2:  add w1, w1, #1
    b 0b
1:  // stack: [sp]=x3(inf), [sp+8]=x2(nan), [sp+16]=x30
    str x2, [sp, #-16]!
    str x3, [sp, #-16]!
    adrp x0, t_lgchk
    add x0, x0, :lo12:t_lgchk
    bl print_cstr
    ldr x0, [sp, #8]             // nan
    bl print_u64
    bl print_sp
    ldr x0, [sp]                 // inf
    bl print_u64
    bl print_nl
    add sp, sp, #32              // drop x3+x2 saves
    ldr x30, [sp], #16
    ret
