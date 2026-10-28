; ==============================================================================
; lfm3_engine_flat.asm - LFM3 text backbone engine for x86-64 AVX2
; ==============================================================================
format binary as ''

include '../raw_materials/mat_bf16_gemb_avx2_flat.asm'
include '../raw_materials/mat_gemma4_kernels_avx2_flat.asm'
include '../raw_materials/mat_lfm3_conv_avx2_flat.asm'
include '../raw_materials/mat_lfm3_kernels_avx2_flat.asm'
include '../sub_assemblies/sub_lfm3_layer_flat.asm'
include '../sub_assemblies/sub_sampler_nucleus_flat.asm'

CTR_WBASE            equ 0
CTR_DESC             equ 8
CTR_NLAYER           equ 16
CTR_POS              equ 24
CTR_TOKEN            equ 32
CTR_VOCAB            equ 40
CTR_HIDDEN           equ 48
CTR_SMP              equ 56
CTR_NTHR             equ 64
CTR_EMB              equ 72
CTR_FINAL_NORM       equ 80
CTR_RMS_EPS          equ 88
CTR_B                equ 96
CTR_TOKENS           equ 104
CTR_OUTTOK           equ 112
CTR_XSTRIDE          equ 120
CTR_BUF_X            equ 128
CTR_BUF_T1           equ 136
CTR_BUF_T2           equ 144
CTR_BUF_T3           equ 152
CTR_BUF_Q            equ 160
CTR_BUF_K            equ 168
CTR_BUF_V            equ 176
CTR_BUF_A            equ 184
CTR_BUF_SCORE        equ 192
CTR_BUF_G            equ 200
CTR_BUF_U            equ 208
CTR_BUF_LOGITS       equ 216
CTR_BUF_CONV_STATE   equ 224
CTR_COS_SLIDE        equ 232
CTR_SIN_SLIDE        equ 240
CTR_COS_FULL         equ 248
CTR_SIN_FULL         equ 256
CTR_ROPE_STR_SLIDE   equ 264
CTR_ROPE_STR_FULL    equ 272
CTR_NHEAD            equ 280
CTR_NKVHEAD          equ 288
CTR_HEADDIM          equ 296
CTR_SLIDING_WINDOW   equ 304
CTR_INTER            equ 312
CTR_MAXB             equ 320
CTR_SIZE             equ 328

D_QW                 equ 0
D_KW                 equ 8
D_VW                 equ 16
D_OW                 equ 24
D_GATE               equ 32
D_UP                 equ 40
D_DOWN               equ 48
D_QNORM              equ 56
D_KNORM              equ 64
D_NORM_OP            equ 72
D_NORM_FFN           equ 80
D_NORM_POST_OP       equ 88
D_NORM_POST_FFN      equ 96
D_CONV_IN_PROJ       equ 104
D_CONV_W             equ 112
D_CONV_OUT_PROJ      equ 120
D_INTER              equ 128
D_HEADDIM            equ 136
D_LAYER_TYPE         equ 144
D_FLAGS              equ 152
DESC_SIZE            equ 160

FLAG_FULL            equ 1
FLAG_SLIDING         equ 2

macro GEMMB dst, wp, xp, kk, mm, bb {
    GEMMBX dst, wp, xp, kk, mm, bb, 0, 0
}

macro GEMMBX dst, wp, xp, kk, mm, bb, xs, os {
    mov     rcx, dst
    mov     rdx, wp
    mov     r8, xp
    mov     r9, kk
    mov     rax, mm
    mov     [rsp + 32], rax
    mov     rax, bb
    mov     [rsp + 40], rax
    mov     rax, [rbx + CTR_SMP]
    mov     [rsp + 48], rax
    mov     rax, xs
    mov     [rsp + 56], rax
    mov     rax, os
    mov     [rsp + 64], rax
    call    smp_bf16_gemb_avx2
}

macro GEMB dst, wp, xp, kk, mm {
    mov     rcx, dst
    mov     rdx, wp
    mov     r8, xp
    mov     r9, kk
    mov     rax, mm
    mov     [rsp + 32], rax
    mov     qword [rsp + 40], 1
    mov     rax, [rbx + CTR_SMP]
    mov     [rsp + 48], rax
    mov     qword [rsp + 56], 0
    mov     qword [rsp + 64], 0
    call    smp_bf16_gemb_avx2
}

macro GEMB_RESID {
    mov     rcx, [rbx + CTR_BUF_X]
    mov     rdx, [rbx + CTR_BUF_X]
    mov     r8, [rbx + CTR_BUF_T2]
    mov     r9, [rbx + CTR_HIDDEN]
    imul    r9, [rbx + CTR_B]
    mov     eax, C_ONE
    mov     [rsp + 32], rax
    mov     [rsp + 40], rax
    call    add_scaled_avx2
}

lfm3_step:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 136
    mov     rbx, rcx
    mov     rax, 0
    add     rsp, 136
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; PLSE export trailer
    align   4
plse_n0: db 'lfm3_step', 0
plse_n1: db 'smp_worker_proc4', 0
plse_n2: db 'smp_bf16_gemb_avx2', 0
plse_n3: db 'bf16_gemb_avx2', 0
    align   4
    dd      lfm3_step
    dd      smp_worker_proc4
    dd      smp_bf16_gemb_avx2
    dd      bf16_gemb_avx2
plse_noff:
    dd      plse_n0
    dd      plse_n1
    dd      plse_n2
    dd      plse_n3
plse_cnt: dd 4
    db      'PLSE'
