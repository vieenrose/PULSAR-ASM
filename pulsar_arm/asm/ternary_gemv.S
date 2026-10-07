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
    .section .text
    .align 2
/* ternary_gemv_b3_v: the vectorised decode. Prototype variant V6 of
 * tools/proto_ternary_neon.c: 0.91 ns/weight vs the scalar kernel's 3.90 (4.3x),
 * which projects to ~1540 ms/token on the 1.7B blob.
 *
 *   void ternary_gemv_b3_v(int rows, int cols, const uint8_t *w,
 *                          const float *x, float *y);
 *
 * x is first deinterleaved by digit position (build_planes). With t_j =
 * floor(v / 3^j) and
 *      sum_i (d_i - 1)*x_i  =  sum_j t_j*Q_j - S,   Q_j = P_j - 3*P_{j-1}
 * where P_j[k] = x[g*128 + 5k + j] and S is the group's sum of x, the inner
 * loop is one FMA and one floor per weight per digit and nothing else. Two
 * decisions were measurement, not taste:
 *   - no integer->float convert anywhere. A72's vector convert is ~9 cycles of
 *     throughput and the digit form needs 40 per group - 1.6 of the scalar
 *     kernel's 3.9 ns/weight all by itself. float(byte) comes from adding
 *     0x4B000000 and subtracting it back: integer pipes only, and exact for
 *     every value under 2^23.
 *   - the t chain is float (fmul + frintm). The SQDMULH version measured 1.61 ns
 *     because it spent 120 integer instructions per row-group to save 64 FP ones.
 * THIRD is biased up by 1e-6 so exact multiples of 3 still floor to the right
 * quotient; every t is <= 255 so the bias can never reach the next integer.
 *
 * NOT wired into the forward. The Q-fold moves agreement with the scalar
 * reference from 4e-7 to 4.7e-6 relative, and that has to be shown harmless at
 * logit level first: position 3 of the validation prompt carries a top1-top2
 * margin of only 0.0171, so a numerical regression there is a wrong demo.
 *
 * Registers: v0-v7 = t values as (q0lo,q0hi,q1lo,q1hi,q2lo,q2hi,q3lo,q3hi);
 * v8-v15 = the eight per-group accumulators (callee-saved, saved); v16-v23 =
 * planes; v28 = 0x4B000000 bits, v29 = 8388608.0f, v30 = THIRD. x19 rows,
 * x20 groups/row, x21 w, x22 x, x23 y, x24 row, x25 group, x26 group Q base,
 * x27 block pointer, x28 CODEBUF, x9 digit, x10/x11/x12/x13 scratch.
 *
 * Code bytes are staged through CODEBUF by reading 16 + 8 + 2 bytes, all inside
 * the 28-byte block, so nothing is ever read past the tensor. Lanes 26..31 then
 * hold stale buffer bytes times zero planes, which is 0 however you slice it. */
    .global ternary_gemv_b3_v
    .type ternary_gemv_b3_v, %function
ternary_gemv_b3_v:
    stp x29, x30, [sp, #-16]!
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp x25, x26, [sp, #-16]!
    stp x27, x28, [sp, #-16]!
    stp q8,  q9,  [sp, #-32]!             // 32: a Q pair is 2x16 bytes;
    stp q10, q11, [sp, #-32]!             // #-16 stored q9 over the x27/x28
    stp q12, q13, [sp, #-32]!             // slot and corrupted the caller's x27,
    stp q14, q15, [sp, #-32]!             // which surfaced as a bad fprintf format
    adrp x8, f_4b000000
    add  x8, x8, :lo12:f_4b000000
    ldr  w9, [x8]
    dup  v28.4s, w9
    fmov s29, w9                          // same bits as float: 8388608.0
    dup  v29.4s, v29.s[0]
    adrp x8, f_third
    add  x8, x8, :lo12:f_third
    ldr  s30, [x8]
    dup  v30.4s, v30.s[0]
    mov  w19, w0                          // rows
    lsr  w20, w1, #7                      // cols / 128
    mov  x21, x2                          // w
    mov  x22, x3                          // x
    mov  x23, x4                          // y
    adrp x28, CODEBUF
    add  x28, x28, :lo12:CODEBUF
    bl   build_planes
    mov  w24, #0
tv_row:
    cmp  w24, w19
    b.ge tv_done
    madd x10, x24, x20, xzr
    mov  x11, #28
    mul  x10, x10, x11
    add  x27, x21, x10                    // first block of the row
    fmov s25, wzr                         // row accumulator (s25: v24-v27 are
                                          // free; s14/s15 would alias v14/v15)
    mov  w25, #0
tv_grp:
    cmp  w25, w20
    b.ge tv_row_end
    ldur q0, [x27, #2]                    // codes 0..15 (ldur: offset 2)
    str  q0, [x28]
    ldur d1, [x27, #18]                   // codes 16..23
    str  d1, [x28, #16]
    ldrh w0, [x27, #26]                   // codes 24..25; +26..27 go to zero
    str  w0, [x28, #24]
    ldrh w0, [x27]                        // fp16 scale
    bl   half_to_float                    // s0 (clobbers w1-w3, x4: dead here)
    fmov s24, s0                          // scale (see above: not s14)
    lsl  x10, x25, #9                     // group offset: 5 planes x 32 x 4
    lsl  x11, x25, #7                     //   = 640 bytes
    add  x10, x10, x11
    adrp x11, QPLANE
    add  x11, x11, :lo12:QPLANE
    add  x26, x11, x10
    movi v8.16b,  #0
    movi v9.16b,  #0
    movi v10.16b, #0
    movi v11.16b, #0
    movi v12.16b, #0
    movi v13.16b, #0
    movi v14.16b, #0
    movi v15.16b, #0
    // t_0 for four quarters, as floats through the magic-add
    ldr  d0, [x28]
    ldr  d1, [x28, #8]
    ldr  d2, [x28, #16]
    ldr  d3, [x28, #24]
    uxtl v16.8h, v0.8b
    uxtl v17.8h, v1.8b
    uxtl v18.8h, v2.8b
    uxtl v19.8h, v3.8b
    uxtl v0.4s,  v16.4h
    ext  v1.16b, v16.16b, v16.16b, #8
    uxtl v1.4s,  v1.4h
    uxtl v2.4s,  v17.4h
    ext  v3.16b, v17.16b, v17.16b, #8
    uxtl v3.4s,  v3.4h
    uxtl v4.4s,  v18.4h
    ext  v5.16b, v18.16b, v18.16b, #8
    uxtl v5.4s,  v5.4h
    uxtl v6.4s,  v19.4h
    ext  v7.16b, v19.16b, v19.16b, #8
    uxtl v7.4s,  v7.4h
    add  v0.4s, v0.4s, v28.4s
    add  v1.4s, v1.4s, v28.4s
    add  v2.4s, v2.4s, v28.4s
    add  v3.4s, v3.4s, v28.4s
    add  v4.4s, v4.4s, v28.4s
    add  v5.4s, v5.4s, v28.4s
    add  v6.4s, v6.4s, v28.4s
    add  v7.4s, v7.4s, v28.4s
    fsub v0.4s, v0.4s, v29.4s
    fsub v1.4s, v1.4s, v29.4s
    fsub v2.4s, v2.4s, v29.4s
    fsub v3.4s, v3.4s, v29.4s
    fsub v4.4s, v4.4s, v29.4s
    fsub v5.4s, v5.4s, v29.4s
    fsub v6.4s, v6.4s, v29.4s
    fsub v7.4s, v7.4s, v29.4s
    mov  x9, xzr                          // digit j
tv_j:
    cmp  x9, #5
    b.ge tv_j_done
    lsl  x10, x9, #7                      // 32 floats per plane
    add  x13, x26, x10
    ldp  q16, q17, [x13]
    ldp  q18, q19, [x13, #32]
    fmla v8.4s,  v0.4s, v16.4s
    fmla v9.4s,  v1.4s, v17.4s
    fmla v10.4s, v2.4s, v18.4s
    fmla v11.4s, v3.4s, v19.4s
    ldp  q20, q21, [x13, #64]
    ldp  q22, q23, [x13, #96]
    fmla v12.4s, v4.4s, v20.4s
    fmla v13.4s, v5.4s, v21.4s
    fmla v14.4s, v6.4s, v22.4s
    fmla v15.4s, v7.4s, v23.4s
    cmp  x9, #4
    b.ge tv_j_next
    fmul v0.4s, v0.4s, v30.4s
    fmul v1.4s, v1.4s, v30.4s
    fmul v2.4s, v2.4s, v30.4s
    fmul v3.4s, v3.4s, v30.4s
    fmul v4.4s, v4.4s, v30.4s
    fmul v5.4s, v5.4s, v30.4s
    fmul v6.4s, v6.4s, v30.4s
    fmul v7.4s, v7.4s, v30.4s
    frintm v0.4s, v0.4s
    frintm v1.4s, v1.4s
    frintm v2.4s, v2.4s
    frintm v3.4s, v3.4s
    frintm v4.4s, v4.4s
    frintm v5.4s, v5.4s
    frintm v6.4s, v6.4s
    frintm v7.4s, v7.4s
tv_j_next:
    add  x9, x9, #1
    b    tv_j
tv_j_done:
    fadd v0.4s, v8.4s, v9.4s
    fadd v1.4s, v10.4s, v11.4s
    fadd v2.4s, v12.4s, v13.4s
    fadd v3.4s, v14.4s, v15.4s
    fadd v0.4s, v0.4s, v1.4s
    fadd v2.4s, v2.4s, v3.4s
    fadd v0.4s, v0.4s, v2.4s
    faddp v1.4s, v0.4s, v0.4s
    faddp s0, v1.2s
    adrp x8, GSUM
    add  x8, x8, :lo12:GSUM
    ldr  s1, [x8, x25, lsl #2]
    fsub s0, s0, s1
    fmul s0, s0, s24
    fadd s25, s25, s0
    add  x27, x27, #28
    add  x25, x25, #1
    b    tv_grp
tv_row_end:
    str  s25, [x23, x24, lsl #2]
    add  x24, x24, #1
    b    tv_row
tv_done:
    ldp q14, q15, [sp], #32
    ldp q12, q13, [sp], #32
    ldp q10, q11, [sp], #32
    ldp q8,  q9,  [sp], #32
    ldp x27, x28, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ldp x29, x30, [sp], #16
    ret
    .size ternary_gemv_b3_v, .-ternary_gemv_b3_v

/* build_planes: deinterleave x into the digit planes, fold them, sum them.
 *   PLANE[g][j][k]  = (5k+j < 128) ? x[g*128 + 5k + j] : 0
 *   QPLANE[g][0][k] = PLANE[g][0][k]
 *   QPLANE[g][j][k] = PLANE[g][j][k] - 3 * PLANE[g][j-1][k]
 *   GSUM[g]         = sum of all PLANE entries (the zero pads make that
 *                     exactly the group's 128 x values)
 * Entry: x22 = x, x20 = groups/row. Preserves x19-x28, x30. Clobbers x8-x18,
 * v0-v3, v6-v8. Costs ~160 scalar pairs per group - a few microseconds next
 * to a megawight sweep, and it happens once per GEMV call, not per row. */
build_planes:
    adrp x16, PLANE
    add  x16, x16, :lo12:PLANE            // PLANE base (x16 is fold's GSUM
                                          // base later; nothing live yet)
    mov  x9, xzr                          // group
bp_grp:
    cmp  x9, x20
    b.ge bp_fold
    add  x8, x22, x9, lsl #9              // x + g*128 floats, as bytes
    lsl  x12, x9, #9                      // g*512
    lsl  x14, x9, #7                      // g*128
    add  x12, x12, x14                    // g*640
    add  x12, x16, x12                    // P base for this group
    mov  x10, xzr                         // k
    // Plane layout is [g][j][k] at (j*32 + k)*4, which the kernel reads;
    // writing sequentially would transpose the 5x32 block, so the destination
    // is formed explicitly (this is the bug the first validation run found).
bp_k:
    cmp  x10, #32
    b.ge bp_k_done
    mov  x11, xzr                         // j
bp_j:
    cmp  x11, #5
    b.ge bp_j_done
    lsl  x13, x10, #2
    add  x13, x13, x10                    // x13 = 5k
    add  x13, x13, x11                    // x13 = 5k + j
    lsl  x15, x11, #7                     // j*128
    add  x15, x15, x12                    // + P base for this group
    cmp  x13, #128
    b.ge bp_j_pad
    ldr  s2, [x8, x13, lsl #2]
    str  s2, [x15, x10, lsl #2]           // + k*4
    b    bp_j_next
bp_j_pad:
    fmov s2, wzr
    str  s2, [x15, x10, lsl #2]
bp_j_next:
    add  x11, x11, #1
    b    bp_j
bp_j_done:
    add  x10, x10, #1
    b    bp_k
bp_k_done:
    add  x9, x9, #1
    b    bp_grp
bp_fold:
    adrp x14, PLANE
    add  x14, x14, :lo12:PLANE
    adrp x15, QPLANE
    add  x15, x15, :lo12:QPLANE
    adrp x16, GSUM
    add  x16, x16, :lo12:GSUM
    adrp x17, tv_m3
    add  x17, x17, :lo12:tv_m3
    ldr  s7, [x17]
    dup  v7.4s, v7.s[0]                   // -3.0
    mov  x9, xzr                          // group
bf_grp:
    cmp  x9, x20
    b.ge bf_done
    lsl  x10, x9, #9                      // g*512
    lsl  x11, x9, #7                      // g*128
    add  x10, x10, x11                    // g*640
    add  x12, x14, x10                    // P base
    add  x13, x15, x10                    // Q base
    fmov s6, wzr
    dup  v6.4s, v6.s[0]                   // group sum
    mov  x17, xzr                         // vector index 0..7
bf_v:
    cmp  x17, #8
    b.ge bf_v_done
    lsl  x18, x17, #4                     // vector offset within a plane
    ldr  q0, [x12, x18]                   // P_0
    str  q0, [x13, x18]                   // Q_0 = P_0
    fadd v6.4s, v6.4s, v0.4s
    mov  v3.16b, v0.16b                   // P_{j-1}, starting at P_0
    mov  x11, #1                          // j
bf_j:
    cmp  x11, #5
    b.ge bf_j_done
    lsl  x10, x11, #7                     // j*128
    add  x10, x10, x18
    ldr  q2, [x12, x10]                   // P_j
    fadd v6.4s, v6.4s, v2.4s
    fmul v1.4s, v3.4s, v7.4s              // -3 * P_{j-1}
    fadd v1.4s, v2.4s, v1.4s              // Q_j
    str  q1, [x13, x10]
    mov  v3.16b, v2.16b                   // P_{j-1} <- P_j for the next round
    add  x11, x11, #1
    b    bf_j
bf_j_done:
    add  x17, x17, #1
    b    bf_v
bf_v_done:
    faddp v1.4s, v6.4s, v6.4s
    faddp s0, v1.2s
    str  s0, [x16], #4
    add  x9, x9, #1
    b    bf_grp
bf_done:
    ret

/* Scratch for the vectorised kernel. 80 groups covers cols up to 10240, which
 * the 4B down projection (cols 9728, 76 groups) fits with room. QPLANE/PLANE
 * are 5 planes of 32 floats per group; the lanes past 26 stay zero. */
    .section .bss
    .align 4
    .global PLANE, QPLANE, GSUM, CODEBUF
PLANE:   .skip 80*5*32*4
QPLANE:  .skip 80*5*32*4
GSUM:    .skip 80*4
CODEBUF: .skip 32
    .section .rodata
    .align 2
f_4b000000: .word 0x4B000000
f_third:    .float 0.33333369
tv_m3:       .float -3.0
