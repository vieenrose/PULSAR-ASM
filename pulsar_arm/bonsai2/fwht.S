    .text
    .align 2
/* fwht1024_f32: x0 = ptr to 1024 f32, in-place normalized FWHT.
 * Matches the fork's ggml_compute_forward_fwht: prescale by 1/sqrt(1024)=1/32,
 * then butterfly stages len=1,2,4,...,512 with (u+v, u-v).
 * Uses x0-x5, v0-v4. Preserves x19+. */
    .global fwht1024_f32
    .type fwht1024_f32, %function
fwht1024_f32:
    adrp x1, f_1_32
    add  x1, x1, :lo12:f_1_32
    ldr  s0, [x1]
    dup  v0.4s, v0.s[0]
    mov  x1, #0
fw_prescale:
    cmp  x1, #1024
    b.hs fw_len1
    add  x2, x0, x1, lsl #2
    ldr  q1, [x2]
    fmul v1.4s, v1.4s, v0.4s
    str  q1, [x2]
    add  x1, x1, #4
    b    fw_prescale
fw_len1:
    mov  x1, #0
fw_l1:
    cmp  x1, #1024
    b.hs fw_len2
    ldr  s1, [x0, x1, lsl #2]
    add  x2, x1, #1
    ldr  s2, [x0, x2, lsl #2]
    fadd s3, s1, s2
    fsub s4, s1, s2
    str  s3, [x0, x1, lsl #2]
    str  s4, [x0, x2, lsl #2]
    add  x1, x1, #2
    b    fw_l1
fw_len2:
    mov  x1, #0
fw_l2:
    cmp  x1, #1024
    b.hs fw_vec
    add  x2, x1, #2
    ldr  s1, [x0, x1, lsl #2]
    ldr  s2, [x0, x2, lsl #2]
    fadd s3, s1, s2
    fsub s4, s1, s2
    str  s3, [x0, x1, lsl #2]
    str  s4, [x0, x2, lsl #2]
    add  x1, x1, #1
    add  x2, x2, #1
    ldr  s1, [x0, x1, lsl #2]
    ldr  s2, [x0, x2, lsl #2]
    fadd s3, s1, s2
    fsub s4, s1, s2
    str  s3, [x0, x1, lsl #2]
    str  s4, [x0, x2, lsl #2]
    add  x1, x1, #3
    b    fw_l2
fw_vec:
    mov  x1, #4
fw_lenloop:
    cmp  x1, #1024
    b.hs fw_done
    mov  x2, #0
fw_iloop:
    cmp  x2, #1024
    b.hs fw_nextlen
    mov  x3, #0
fw_jloop:
    cmp  x3, x1
    b.hs fw_nexti
    add  x4, x2, x3
    lsl  x4, x4, #2
    add  x4, x4, x0
    add  x5, x2, x1
    add  x5, x5, x3
    lsl  x5, x5, #2
    add  x5, x5, x0
    ldr  q1, [x4]
    ldr  q2, [x5]
    fadd v3.4s, v1.4s, v2.4s
    fsub v4.4s, v1.4s, v2.4s
    str  q3, [x4]
    str  q4, [x5]
    add  x3, x3, #4
    b    fw_jloop
fw_nexti:
    add  x2, x2, x1, lsl #1
    b    fw_iloop
fw_nextlen:
    lsl  x1, x1, #1
    b    fw_lenloop
fw_done:
    ret
    .size fwht1024_f32, .-fwht1024_f32
    .section .rodata
    .align 2
f_1_32: .float 0.03125
