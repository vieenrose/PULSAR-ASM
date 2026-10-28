; ==============================================================================
; sub_lfm3_layer_flat.asm - LFM3 decoder layer with sandwich norm
; Supports conv and attention (full/sliding) layers
; ==============================================================================
use64

; lfm3_layer(ctx, desc, layer_idx)
lfm3_layer:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    mov     rbx, rcx
    mov     r15, rdx
    ; TODO: implement full layer
    xor     eax, eax
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret
