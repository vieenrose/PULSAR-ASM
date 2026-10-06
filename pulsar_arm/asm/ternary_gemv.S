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

/* embed_row_b3: dequantise ONE row of a B3_128 matrix - the token embedding
 * gather. The table is tied to the head, so the same blocks are a GEMV there
 * but a row read here; a GEMV would sweep all 151669 rows for one token.
 *
 *   void embed_row_b3(float *out, const uint8_t *w, int row, int cols,
 *                     float escale);
 *
 * Trits are decoded in exactly the order ternary_gemv_b3 uses (byte k, trit i,
 * idx = k*5+i, stop at 128), so a gathered row is the row the python oracle
 * dequantises. escale is folded into the per-group scale instead of applied per
 * element: for Qwen3 it is 1.0, where the two are bit-for-bit identical, and
 * otherwise they differ by at most one rounding.
 *
 * Register plan: x19 out cursor, x20 block ptr, x21 groups/row, x22 group,
 * x10-x17 decode temps, s7 the folded scale. Only s0-s6 are used, because
 * s8-s15 (d8-d15) are callee-saved in the AArch64 ABI: an earlier version kept
 * escale in s8 and silently corrupted a caller that had parked a value there.
 * escale lives in a stack slot instead, since half_to_float clobbers s0/s1.
 * Saves x19-x22 and x30; leaves x18 and x23+ alone. */
    .global embed_row_b3
    .type embed_row_b3, %function
embed_row_b3:
    str  x30, [sp, #-16]!
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    sub  sp, sp, #16
    str  s0, [sp]                // escale, across the half_to_float call
    lsr  w21, w3, #7             // groups per row (w-form: x3 upper bits unspecified)
    cbz  x21, er_ret             // cols < 128: no group to read, nothing to write
    uxtw x5, w2                  // row
    mov  x6, #28
    mul  x5, x5, x21             // row * ng
    mul  x5, x5, x6              // * group bytes
    add  x20, x1, x5             // first block of the row
    mov  x19, x0                 // out cursor
    mov  x22, #0                 // g
er_group:
    cmp  x22, x21
    b.hs er_ret
    ldrh w0, [x20]               // fp16 scale bits
    bl   half_to_float           // s0 = scale (clobbers s1, x4)
    ldr  s1, [sp]
    fmul s7, s0, s1              // sg = group scale * embed scale
    add  x11, x20, #2            // codes
    mov  w10, #0                 // k
er_byte:
    cmp  w10, #26
    b.hs er_next
    ldrb w12, [x11, x10]         // v
    mov  w13, #0                 // i
er_trit:
    cmp  w13, #5
    b.hs er_nxtbyte
    mov  w14, #5
    madd w15, w10, w14, w13      // idx = k*5 + i
    cmp  w15, #128
    b.hs er_nxtbyte              // the 26th byte carries only 3 trits
    mov  w14, #3
    udiv w16, w12, w14
    msub w17, w16, w14, w12      // v mod 3
    sub  w17, w17, #1            // trit in {-1,0,1}
    scvtf s5, w17
    fmul s6, s5, s7              // trit * sg
    str  s6, [x19], #4
    mov  w12, w16                // v /= 3
    add  w13, w13, #1
    b    er_trit
er_nxtbyte:
    add  w10, w10, #1
    b    er_byte
er_next:
    add  x20, x20, #28
    add  x22, x22, #1
    b    er_group
er_ret:
    add  sp, sp, #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ldr  x30, [sp], #16
    ret
    .size embed_row_b3, .-embed_row_b3

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
    lsl  w3, w3, #13             // mantissa -> fp32 bit position
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
