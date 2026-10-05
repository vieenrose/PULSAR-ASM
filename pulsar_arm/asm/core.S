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

/* ---------------- entry ---------------- */
_start:
    adrp x0, banner
    add x0, x0, :lo12:banner
    bl print_cstr
    bl run_tests
    bl microbench
    bl kern_tests
    ldr x0, [sp]                  // argc
    add x1, sp, #8                // argv
    bl file_stage_after_bench
    bl tables_test
    bl fwd_layer0
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
t_usage:     .asciz "usage: core <model.safetensors>\n"

    .section .bss
    .align 3
tab:         .skip 236*56
namebuf:     .skip 128

/* _start appends the file stage: microbench -> argv -> file_stage -> exit */
    .section .text
    .align 2
    .global file_stage_after_bench
file_stage_after_bench:
    // x0 = argc, x1 = argv
    cmp x0, #2
    b.lo 0f
    str x30, [sp, #-16]!
    str x1, [sp, #-16]!          // argv
    mov x2, #2
    adrp x3, G_tok
    add x3, x3, :lo12:G_tok
    str x2, [x3]                 // default tok=2
    cmp x0, #3
    b.lo 1f
    ldr x0, [x1, #16]            // argv[2]
    bl parse_u64                 // x0 = tok
    adrp x1, G_tok
    add x1, x1, :lo12:G_tok
    str x0, [x1]
1:  ldr x1, [sp], #16            // argv
    ldr x0, [x1, #8]             // argv[1] path
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
    mov w19, w0                  // tok
    mov x0, x19                  // print ID -- x0 clobbered? print_u64 takes x0 ✓ set fresh
    str x19, [sp, #-16]!         // save tok across prints (x0-x9 die!)
    str x20, [sp, #-16]!         // save pos
    bl print_u64
    bl print_sp
    ldr x20, [sp], #16
    ldr x19, [sp], #16
    add w20, w20, #1
    b 2b
3:  bl print_nl
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
    mov x1, #100000
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
    // tok/s: tps_x100 = 3200000/ms_total; ms_total = dt*1000/freq
    // stack: [ms_x100][dt][x21][x19,x20][x30]. dt at [sp,#8].
    adrp x0, t_tps2
    add x0, x0, :lo12:t_tps2
    bl print_cstr
    ldr x2, [sp, #8]             // dt
    mrs x0, cntfrq_el0
    mov x1, #1000
    mul x1, x2, x1               // dt*1000
    udiv x1, x1, x0              // ms_total
    movz x0, #0xd400
    movk x0, #0x30, lsl #16      // 3200000
    udiv x0, x0, x1              // tps_x100
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
    bl print_nl
    add sp, sp, #16              // drop ms_x100
    add sp, sp, #16              // drop dt
    ldr x21, [sp], #16
    ldp x19, x20, [sp], #16
    ldr x30, [sp], #16
    ret
