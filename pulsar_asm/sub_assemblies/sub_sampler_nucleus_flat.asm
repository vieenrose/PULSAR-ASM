; ==============================================================================
; Project PULSAR-ASM | Sub-Assembly: sub_sampler_nucleus_flat.asm
; ------------------------------------------------------------------------------
; Temperature -> top-k -> top-p (nucleus) sampling, AVX2, zero CRT.
;
; Internal calling convention (same as the other gemma4 kernels):
;   RCX  = logits           float[n]
;   RDX  = n                candidate count (vocab)
;   R8   = k                top-k width, clamped to [1,64]
;   R9   = p                nucleus mass, float BITS by value
;   [RSP+32] = temperature  float bits by value; the caller skips us if 0
;   [RSP+40] = rng          uint64* xorshift state, updated in place
;   [RSP+48] = out          int64* receiving the chosen token id
;   RAX  = chosen token id
;
; k passes of the argmax kernel yield the candidates already in descending
; order, which is what the nucleus cut wants; each pick is masked to -inf so the
; next pass finds the runner-up. k is 64 for this checkpoint, so k full scans
; cost a few ms against a ~180 ms step. The masks are undone before returning:
; the logits row belongs to the caller, and in a batched pass the next row must
; read what the head actually produced.
;
; Weights come from softmax_avx2 over the k values - the engine's one exp path -
; after dividing by temperature. Randomness is xorshift64*: one 64-bit state,
; three shifts, and the top 24 bits become a float in [0,1).
;
; The cut matches what transformers does with these values: keep the shortest
; prefix whose cumulative mass reaches p, renormalise over it, draw from it.
; ==============================================================================

use64

SK_CAND equ 0                                 ; 64 int64 candidate ids
SK_VAL  equ 512                               ; 64 float weights, then values
SK_LOGITS equ 800
SK_N    equ 808
SK_K    equ 816
SK_OUT  equ 824
SK_RNG  equ 832
SK_M    equ 840                               ; candidates found
SK_TEMP equ 848
SK_P    equ 856
SK_ACC  equ 864
SK_NORM equ 872
SK_KEEP equ 880
NEG_INF equ 0FF800000h
SK_FRAME equ 1032
SK_PUSH  equ 48                             ; six pushes in the prologue
SK_ARG   equ SK_FRAME + SK_PUSH + 8         ; +8: CALL pushed the return address
SK_A5 equ SK_ARG + 32
SK_A6 equ SK_ARG + 40
SK_A7 equ SK_ARG + 48
SK_RNGMUL equ 0x2545F4914F6CDD1D

sampler_nucleus_avx2:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, SK_FRAME                     ; scratch plus realignment
    mov     [rsp + SK_LOGITS], rcx
    mov     [rsp + SK_N], rdx
    mov     [rsp + SK_P], r9                  ; float bits
    ; Where the stack args are, exactly. The caller put arg5..arg7 in its own
    ; [rsp+32,40,48]; CALL pushed the return address (+8); then the six pushes
    ; and the frame below moved rsp by another 48 + SK_FRAME. Both shifts count,
    ; and forgetting the pushes is what made this kernel fault: it read the rng
    ; and out pointers out of the return-address slot, so the state load walked
    ; into whatever address a return pointer happens to be.
    mov     rax, [rsp + SK_A5]                ; temperature
    mov     [rsp + SK_TEMP], rax
    mov     rax, [rsp + SK_A6]                ; rng state
    mov     [rsp + SK_RNG], rax
    mov     rax, [rsp + SK_A7]                ; out
    mov     [rsp + SK_OUT], rax
    mov     qword [rsp + SK_M], 0
    test    r8, r8                            ; k, still in place from entry
    jnz     .sk_kclamp
    mov     r8, 1
.sk_kclamp:
    cmp     r8, 64
    jbe     .sk_kset
    mov     r8, 64
.sk_kset:
    mov     [rsp + SK_K], r8

    ; ---- top-k by repeated argmax, masking each pick -------------------------
.sk_pick:
    mov     rcx, [rsp + SK_LOGITS]
    mov     rdx, [rsp + SK_N]
    call    sampler_argmax_avx2               ; Win64 ABI, keeps rbp/r12-r15
    mov     rcx, [rsp + SK_LOGITS]
    mov     edx, dword [rcx + rax * 4]
    cmp     edx, NEG_INF
    je      .sk_pick_done                     ; ran out of live candidates
    mov     rbx, [rsp + SK_M]
    mov     [rsp + rbx * 8], rax
    mov     [rsp + SK_VAL + rbx * 4], edx
    mov     dword [rcx + rax * 4], NEG_INF
    inc     rbx
    mov     [rsp + SK_M], rbx
    cmp     rbx, [rsp + SK_K]
    jb      .sk_pick

.sk_pick_done:
    ; ---- hand the logits row back exactly as it came ------------------------
    xor     rbx, rbx
.sk_restore:
    cmp     rbx, [rsp + SK_M]
    jae     .sk_restored
    mov     rax, [rsp + rbx * 8]
    mov     rcx, [rsp + SK_LOGITS]
    mov     edx, [rsp + SK_VAL + rbx * 4]
    mov     [rcx + rax * 4], edx
    inc     rbx
    jmp     .sk_restore

.sk_restored:
    ; ---- weights = softmax(v / temp) over the candidates ---------------------
    mov     eax, dword [rsp + SK_TEMP]
    test    eax, eax
    jnz     .sk_tprep
    mov     eax, 0x3F800000                   ; a zero temperature means don't
    mov     [rsp + SK_TEMP], rax              ; rescale, just don't divide by it
.sk_tprep:
    vmovss  xmm0, dword [rsp + SK_TEMP]
    mov     eax, 0x3F800000
    vmovd   xmm1, eax
    vdivss  xmm0, xmm1, xmm0                  ; 1/temp
    xor     rbx, rbx
.sk_scale:
    cmp     rbx, [rsp + SK_M]
    jae     .sk_scaled
    vmulss  xmm2, xmm0, dword [rsp + SK_VAL + rbx * 4]
    vmovss  dword [rsp + SK_VAL + rbx * 4], xmm2
    inc     rbx
    jmp     .sk_scale
.sk_scaled:
    lea     rcx, [rsp + SK_VAL]               ; softmax in place over the
    mov     rdx, [rsp + SK_M]                 ; candidate list
    call    softmax_avx2

    ; ---- nucleus: shortest prefix reaching p --------------------------------
    vmovss  xmm0, dword [rsp + SK_P]          ; p
    vxorps  xmm1, xmm1, xmm1
    vmovss  dword [rsp + SK_ACC], xmm1
    vmovss  dword [rsp + SK_NORM], xmm1
    mov     qword [rsp + SK_KEEP], 0
.sk_nuc:
    mov     rax, [rsp + SK_KEEP]
    cmp     rax, [rsp + SK_M]
    jae     .sk_nuc_done
    vmovss  xmm2, dword [rsp + SK_ACC]
    vcmpps  xmm3, xmm2, xmm0, 13              ; _CMP_GE_OQ: stop at acc >= p
    vmovmskps r11d, xmm3
    test    r11d, 1
    jnz     .sk_nuc_done
    vmovss  xmm2, dword [rsp + SK_VAL + rax * 4]
    vaddss  xmm4, xmm2, dword [rsp + SK_NORM]
    vmovss  dword [rsp + SK_NORM], xmm4
    vaddss  xmm2, xmm2, dword [rsp + SK_ACC]
    vmovss  dword [rsp + SK_ACC], xmm2
    inc     rax
    mov     [rsp + SK_KEEP], rax
    jmp     .sk_nuc
.sk_nuc_done:
    cmp     qword [rsp + SK_KEEP], 0
    jne     .sk_draw
    mov     qword [rsp + SK_KEEP], 1          ; p smaller than the first weight
    mov     eax, dword [rsp + SK_NORM]        ; still has to be non-zero
    test    eax, eax
    jnz     .sk_draw
    vmovss  xmm2, dword [rsp + SK_VAL]
    vmovss  dword [rsp + SK_NORM], xmm2

    ; ---- draw: xorshift64*, top 24 bits as a float in [0,1) ------------------
.sk_draw:
    mov     rbx, [rsp + SK_RNG]
    mov     r12, [rbx]
    test    r12, r12
    jnz     .sk_mix
    mov     r12, 0x9E3779B97F4A7C15           ; a zero state would stay zero
    mov     [rbx], r12
.sk_mix:
    mov     r13, r12
    shr     r13, 12
    xor     r12, r13
    mov     r13, r12
    shl     r13, 25
    xor     r12, r13
    mov     r13, r12
    shr     r13, 27
    xor     r12, r13
    mov     [rbx], r12
    mov     r13, SK_RNGMUL                    ; imul has no 64-bit immediate
    imul    r12, r13
    shr     r12, 40                           ; 24 bits
    vcvtsi2ss xmm3, xmm3, r12d
    mov     eax, 0x33800000                   ; 2^-24
    vmovd   xmm4, eax
    vmulss  xmm3, xmm3, xmm4

    ; target = u * norm, then walk the kept weights subtracting
    vmulss  xmm3, xmm3, dword [rsp + SK_NORM]
    xor     rbx, rbx
.sk_walk:
    mov     rax, [rsp + SK_KEEP]
    lea     r13, [rax - 1]
    cmp     rbx, r13
    jae     .sk_picked
    vmovss  xmm5, dword [rsp + SK_VAL + rbx * 4]
    vcomiss xmm3, xmm5
    jbe     .sk_picked                        ; u*norm lands in this bucket
    vsubss  xmm3, xmm3, xmm5
    inc     rbx
    jmp     .sk_walk

.sk_picked:
    mov     rax, [rsp + rbx * 8]
    mov     rbx, [rsp + SK_OUT]
    mov     [rbx], rax
    vzeroupper
    add     rsp, SK_FRAME
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret
