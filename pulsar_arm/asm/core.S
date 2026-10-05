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
    bl kern_tests
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
1:  addv s1, v0.4s              // NEVER addv into its own vector (overlap hazard)
    fmov s0, s1
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
1:  str s0, [sp, #-16]!          // m
    movi v6.4s, #0                // sum (expf keeps s6+)
    mov w9, #0
3:  cmp w9, w1
    b.hs 4f
    ldr s0, [x0, x9, lsl #2]
    ldr s1, [sp]                 // m
    fsub s0, s0, s1
    str x9, [sp, #-16]!
    bl expf_asm
    ldr x9, [sp], #16
    str s0, [x0, x9, lsl #2]     // x0 intact (expf uses x9,x10)
    fadd s6, s6, s0
    add w9, w9, #1
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
6:  add sp, sp, #16              // drop m
    ret

/* gelu_mul_f32(out,x0 x,x1 mult,x2 n,w3). Scalar; w9 survives tanhf via stack. */
    .global gelu_mul_f32
gelu_mul_f32:
    mov w9, #0
0:  cmp w9, w3
    b.hs 1f
    ldr s0, [x1, x9, lsl #2]
    str x9, [sp, #-16]!
    bl tanhf_asm                 // s0 = t (clobbers x9!)
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
1:  ret

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
2:  addv s0, v20.4s
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
1:  addv s1, v0.4s              // NEVER addv into its own vector (overlap hazard)
    fmov s0, s1
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
    ld1 {v3.4s}, [x8], #16
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
    mov w0, #0
    mov w9, #1
0:  cmp w9, w1
    b.hs 1f
    ldr s1, [x0, x9, lsl #2]
    fcmp s1, s0
    b.le 2f
    fmov s0, s1
    mov w0, w9
2:  add w9, w9, #1
    b 0b
1:  ret

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
1:  addv s1, v0.4s              // NEVER addv into its own vector (overlap hazard)
    fmov s0, s1
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
    // DIAG2 (temporary): rmsnorm([1,2,3,4], w=1) -> out[0] ~= 0.3651
    adrp x9, T0
    add x9, x9, :lo12:T0
    adrp x10, f_1
    ldr s0, [x10, :lo12:f_1]
    fmov s1, #2.0
    fmov s0, s1
    bl print_f32                 // expect 2.0 (fmov sanity)
    bl print_nl
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
    adrp x0, T0
    add x0, x0, :lo12:T0
    bl print_u64                 // expect 6913632 (T0 addr)
    bl print_sp
    adrp x9, T0
    add x9, x9, :lo12:T0
    mov x0, x9
    bl print_u64                 // expect same
    bl print_nl
    adrp x9, T0
    add x9, x9, :lo12:T0
    ldr s0, [x9]
    bl print_f32                 // expect 1.0
    bl print_sp
    adrp x9, T0
    add x9, x9, :lo12:T0
    ldr s0, [x9, #4]             // expect 2.0
    bl print_f32
    bl print_sp
    adrp x9, T0
    add x9, x9, :lo12:T0
    ldr s0, [x9, #8]             // expect 3.0
    bl print_f32
    bl print_sp
    adrp x9, T0
    add x9, x9, :lo12:T0
    ldr s0, [x9, #12]            // expect 4.0
    bl print_f32
    bl print_nl
    adrp x9, T1
    add x9, x9, :lo12:T1
    ldr s0, [x9]                 // w[0], expect 1.0
    bl print_f32
    bl print_nl
    ld1 {v1.4s}, [x9]            // inline: expect [1,2,3,4]
    movi v0.4s, #0
    fmla v0.4s, v1.4s, v1.4s
    addv s1, v0.4s
    fmov s0, s1
    bl print_f32                 // expect 30.0 (energy direct)
    bl print_sp
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
1:  adrp x9, T1                  // w = 0
    add x9, x9, :lo12:T1
    mov w10, #0
2:  cmp w10, #640
    b.hs 3f
    str wzr, [x9, x10, lsl #2]
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
19: // rebuild x pattern in T0, zero w in T1[0..639]
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
    str wzr, [x10, x11, lsl #2]
    add w11, w11, #1
    b 21b
22: adrp x0, T2                  // out
    add x0, x0, :lo12:T2
    adrp x1, T1                  // w (zeros)
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

    .section .bss
    .align 4
T0:          .skip 2048*4
T1:          .skip 2048*4
T2:          .skip 2048*4
T3:          .skip 2048*4

    .section .text
    .align 2
    .global kern_tests
