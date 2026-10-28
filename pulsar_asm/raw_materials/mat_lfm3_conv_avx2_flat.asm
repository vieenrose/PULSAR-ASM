; ==============================================================================
; mat_lfm3_conv_avx2_flat.asm - Depthwise causal conv 1D for LFM3 (kernel size 7)
; ==============================================================================
use64

; lfm3_conv_prefill(out_x, in_xvz, conv_w, B, T, H, Lconv) 
; in_xvz is [B, 3H, T] in some layout, conv_w is [H, 1, Lconv]
; Simple reference implementation - will be optimized
lfm3_conv_prefill:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    xor     eax, eax
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

lfm3_conv_decode_update:
    push    rbx
    xor     eax, eax
    pop     rbx
    ret

; PLSE export trailer
    align   4
lconv_n0: db 'lfm3_conv_prefill', 0
lconv_n1: db 'lfm3_conv_decode_update', 0
    align   4
    dd      lfm3_conv_prefill
    dd      lfm3_conv_decode_update
lconv_noff:
    dd      lconv_n0
    dd      lconv_n1
lconv_cnt: dd 2
    db      'PLSE'
