/* PULSAR-ARM | asm/ternary_gemv.S
 *
 * Scalar ternary GEMV for the Bonsai blobs (B3_128 groups), mirroring
 * kernels/ternary_gemv.c exactly: per 128-weight group read the fp16 scale,
 * decode the base-3 codes to trits, reduce sum(trit * x), then acc += scale*sum.
 *
 *   void ternary_gemv_b3(int rows, int cols, const uint8_t *w,
 *                        const float *x, float *y);
 *
 * Block: 28 bytes = 2-byte fp16 scale + 26 bytes of base-3 codes, 5 trits per
 * byte, little-endian base 3 (byte = sum(c_i * 3^i), c in {0,1,2}).
 *
 * Register plan: w19 rows, w20 groups/row, x21 w, x22 x, x23 y, w24 row,
 * w25 group; s2 accumulator, s3 scale, s4 group sum.
 */
    .text
    .global ternary_gemv_b3
    .type ternary_gemv_b3, %function
ternary_gemv_b3:
    str  x30, [sp, #-16]!        // bl half_to_float clobbers lr
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp x25, x26, [sp, #-16]!
    mov  w19, w0                 // rows
    lsr  w20, w1, #7             // cols / 128
    mov  x21, x2                 // w
    mov  x22, x3                 // x
    mov  x23, x4                 // y
    mov  w24, #0                 // r
1:  cmp  w24, w19
    b.hs 9f
    fmov s2, wzr                 // acc = 0
    mov  w25, #0                 // g
2:  cmp  w25, w20
    b.hs 8f
    madd w5, w24, w20, w25       // r*ng + g
    mov  w6, #28
    mul  w5, w5, w6
    add  x6, x21, x5             // blk
    ldrh w0, [x6]                // fp16 scale bits
    bl   half_to_float           // s0 = scale
    fmov s3, s0
    add  x7, x6, #2              // codes
    lsl  x9, x25, #9             // g*128*4
    add  x9, x22, x9             // xg
    fmov s4, wzr                 // group sum
    mov  w10, #0                 // k
3:  cmp  w10, #26
    b.hs 6f
    ldrb w11, [x7, x10]          // v
    mov  w12, #0                 // i
4:  cmp  w12, #5
    b.hs 5f
    mov  w13, #5
    madd w14, w10, w13, w12      // idx = k*5 + i
    cmp  w14, #128
    b.hs 5f
    mov  w15, #3
    udiv w16, w11, w15
    msub w17, w16, w15, w11      // v mod 3
    sub  w17, w17, #1            // trit in {-1,0,1}
    scvtf s5, w17
    ldr  s6, [x9, x14, lsl #2]
    fmadd s4, s5, s6, s4         // sum += trit * xg[idx]
    mov  w11, w16                // v /= 3
    add  w12, w12, #1
    b    4b
5:  add  w10, w10, #1
    b    3b
6:  fmadd s2, s3, s4, s2         // acc += scale * sum
    add  w25, w25, #1
    b    2b
8:  str  s2, [x23, w24, uxtw #2]
    add  w24, w24, #1
    b    1b
9:  ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ldr  x30, [sp], #16
    ret
    .size ternary_gemv_b3, .-ternary_gemv_b3

/* half_to_float: w0 = fp16 bits -> s0 = fp32. Clobbers w1,w2,w3. */
    .type half_to_float, %function
half_to_float:
    and  w1, w0, #0x8000         // sign
    lsl  w1, w1, #16             // -> bit 31
    ubfx w2, w0, #10, #5         // exponent
    and  w3, w0, #0x3ff          // mantissa
    cbz  w2, 0f                  // zero / subnormal
    cmp  w2, #31
    b.eq 1f                      // inf / nan
    add  w2, w2, #112            // bias adjust (127 - 15)
    lsl  w2, w2, #23
    orr  w3, w3, w2
    orr  w3, w3, w1
    fmov s0, w3
    ret
1:  movz w2, #0x7f80
    lsl  w2, w2, #16
    lsl  w3, w3, #13
    orr  w3, w3, w2
    orr  w3, w3, w1
    fmov s0, w3
    ret
0:  cbz  w3, 2f                  // zero
    scvtf s0, w3                 // subnormal: mant * 2^-24
    adrp x4, f_2m24
    add  x4, x4, :lo12:f_2m24
    ldr  s1, [x4]
    fmul s0, s0, s1
    fmov w3, s0
    orr  w3, w3, w1
    fmov s0, w3
    ret
2:  fmov s0, w1                  // +/- 0.0
    ret
    .size half_to_float, .-half_to_float

    .section .rodata
    .align 2
f_2m24:  .float 0.00000005960464478
