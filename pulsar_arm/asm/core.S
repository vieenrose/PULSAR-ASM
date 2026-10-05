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
2:  addv s24, v20.4s
    addv s25, v21.4s
    addv s26, v22.4s
    addv s27, v23.4s
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
    ldr x0, [sp]                  // argc
    add x1, sp, #8                // argv
    bl file_stage_after_bench
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
    ldr x0, [x1, #16]
    bl print_u64
    bl print_sp
    ldr x0, [x1, #24]
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
    ldr x0, [x1, #16]
    bl print_u64
    bl print_sp
    ldr x0, [x1, #24]
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
    // x0 = argc, x1 = argv[1] or 0
    cmp x0, #2
    b.lo 0f
    ldr x0, [x1, #8]             // argv[1]
    b file_stage
0:  adrp x0, t_usage
    add x0, x0, :lo12:t_usage
    bl print_cstr
    mov x0, #0
    mov x8, #SYS_EXIT
    svc #0
