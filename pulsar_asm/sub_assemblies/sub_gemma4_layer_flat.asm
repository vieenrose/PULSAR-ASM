; ==============================================================================
; Project PULSAR-ASM | Gemma 4 E2B  (sub_assemblies/sub_gemma4_layer_flat.asm)
; ------------------------------------------------------------------------------
; One decoder layer, transcribed from Gemma4TextDecoderLayer.forward().
;
; ENTRY:  rbx = ctx (layout in engine/gemma4_engine_flat.asm)
;         r15 = &layer descriptor
;         rax = layer index (selects this layer's PLE input slice)
; EXIT:   rax = 0, rbx/rbp/r12..r15 preserved
;
; Register use: rbx ctx | r15 descriptor | r13 cos | r14 sin | r12 head counter
; | rbp scratch. Legal because PULSAR kernels preserve callee-saved registers -
; the two attention kernels did not and had to be fixed: a kernel returning with
; r12 changed corrupts whatever the interpreter kept there, and fails far from
; its cause.
;
; Frame:  [rsp+0..32)   home space (kernels' shadow area, never touched)
;         [rsp+32..64)  stack args for kernel calls (5th arg at +32, etc.)
;         [rsp+64..]    locals: +64 &K[pos], +72 &V[pos], +80 kv_start,
;                       +88 kv_len, +96 layer index
; ==============================================================================

gemma4_layer:
    push    rbp
    push    r12
    push    r13
    push    r14
    sub     rsp, 128
    mov     [rsp + 96], rax
    mov     rax, 200
    add     rax, [rsp + 96]
    mov     [rbx + CTR_MARK], rax             ; 100 + layer index, for post-mortem

    ; ---- rope tables: full-attention layers use theta 1e6, sliding 1e4 ------
    mov     rbp, [rbx + CTR_POS]
    test    qword [r15 + D_FLAGS], FLAG_FULL
    jz      .ly_rope_slide
    mov     r13, [rbx + CTR_COS_FULL]
    mov     r14, [rbx + CTR_SIN_FULL]
    imul    rbp, [rbx + CTR_ROPE_STR_FULL]
    jmp     .ly_rope_done
.ly_rope_slide:
    mov     r13, [rbx + CTR_COS_SLIDE]
    mov     r14, [rbx + CTR_SIN_SLIDE]
    imul    rbp, [rbx + CTR_ROPE_STR_SLIDE]
.ly_rope_done:
    add     r13, rbp
    add     r14, rbp

    ; ---- attention ---------------------------------------------------------
    mov     rcx, [rbx + CTR_BUF_T1]
    mov     rdx, [r15 + D_NIN]
    mov     r8,  [rbx + CTR_BUF_X]
    mov     r9,  [rbx + CTR_HIDDEN]
    mov     eax, [rbx + CTR_RMS_EPS]
    mov     [rsp + 32], rax
    call    rmsnorm_avx2                        ; (out, w, x, N, eps)

    mov     rbp, [r15 + D_HEADDIM]
    imul    rbp, [rbx + CTR_NHEAD]              ; M = n_head * head_dim
    GEMB    <[rbx + CTR_BUF_Q]>, <[r15 + D_QW]>, <[rbx + CTR_BUF_T1]>, <[rbx + CTR_HIDDEN]>, <rbp>
    add     qword [rbx + CTR_MARK], 1

    ; per head: q_norm then rope. The norm is RMSNorm(head_dim) shared by all
    ; heads, and it runs BEFORE rope - rotating first is a different function.
    xor     r12d, r12d
.ly_qnr:
    mov     rbp, [r15 + D_HEADDIM]
    mov     rcx, [rbx + CTR_BUF_Q]
    mov     rax, r12
    imul    rax, rbp
    shl     rax, 2
    add     rcx, rax
    mov     rdx, [r15 + D_QNORM]
    mov     r8, rcx                             ; in place
    mov     r9, rbp
    mov     eax, [rbx + CTR_RMS_EPS]
    mov     [rsp + 32], rax
    call    rmsnorm_avx2
    mov     rcx, [rbx + CTR_BUF_Q]
    mov     rax, r12
    imul    rax, [r15 + D_HEADDIM]
    shl     rax, 2
    add     rcx, rax
    mov     rdx, r13
    mov     r8, r14
    mov     r9, [r15 + D_HEADDIM]
    mov     rax, r9
    shr     rax, 1                              ; rp = head_dim/2 pairs
    mov     [rsp + 32], rax
    call    rope_apply_avx2                     ; (buf, cos, sin, H, rp)
    inc     r12
    cmp     r12, [rbx + CTR_NHEAD]
    jb      .ly_qnr
    add     qword [rbx + CTR_MARK], 2

    ; ---- K / V --------------------------------------------------------------
    ; KV-shared layers (15..34) have no k_proj/v_proj tensors at all: their
    ; descriptor points at the cache published by the storing layer (13 for
    ; sliding, 14 for full), whose contents are already normed and rotated, so
    ; this entire block is skipped for them.
    test    qword [r15 + D_FLAGS], FLAG_KVSHARED
    jnz     .ly_attn
    mov     rbp, [rbx + CTR_POS]
    mov     rax, rbp
    imul    rax, [r15 + D_KVSTRIDE]
    add     rax, [r15 + D_KV]
    mov     [rsp + 64], rax                     ; &K[pos]
    mov     rax, rbp
    imul    rax, [r15 + D_KVSTRIDE]
    add     rax, [r15 + D_VV]
    mov     [rsp + 72], rax                     ; &V[pos]

    ; k goes straight into its cache row, then k_norm + rope in place
    mov     rbp, [r15 + D_HEADDIM]
    GEMB    <[rsp + 64]>, <[r15 + D_KW]>, <[rbx + CTR_BUF_T1]>, <[rbx + CTR_HIDDEN]>, <rbp>
    mov     rcx, [rsp + 64]
    mov     rdx, [r15 + D_KNORM]
    mov     r8, rcx
    mov     r9, [r15 + D_HEADDIM]
    mov     eax, [rbx + CTR_RMS_EPS]
    mov     [rsp + 32], rax
    call    rmsnorm_avx2
    mov     rcx, [rsp + 64]
    mov     rdx, r13
    mov     r8, r14
    mov     r9, [r15 + D_HEADDIM]
    mov     rax, r9
    shr     rax, 1
    mov     [rsp + 32], rax
    call    rope_apply_avx2

    ; v likewise; its norm is scale-free (no weight tensor in the checkpoint),
    ; which is exactly why rmsnorm_scale exists.
    mov     rbp, [r15 + D_HEADDIM]
    GEMB    <[rsp + 72]>, <[r15 + D_VW]>, <[rbx + CTR_BUF_T1]>, <[rbx + CTR_HIDDEN]>, <rbp>
    mov     rcx, [rsp + 72]
    mov     rdx, rcx
    mov     r8, [r15 + D_HEADDIM]
    mov     r9, C_ONE
    mov     eax, [rbx + CTR_RMS_EPS]
    mov     [rsp + 32], rax
    call    rmsnorm_scale_avx2                  ; (out, x, N, scale, eps)

    ; ---- visible window ----------------------------------------------------
.ly_attn:
    add     qword [rbx + CTR_MARK], 4
    mov     rbp, [rbx + CTR_POS]
    inc     rbp
    mov     [rsp + 88], rbp                     ; kv_len = pos + 1
    mov     qword [rsp + 80], 0                 ; kv_start = 0
    test    qword [r15 + D_FLAGS], FLAG_FULL
    jnz     .ly_head
    mov     rax, rbp
    sub     rax, [rbx + CTR_WINDOW]             ; pos + 1 - window
    jle     .ly_head
    mov     [rsp + 80], rax                     ; first visible row
    mov     rax, [rbx + CTR_WINDOW]
    mov     [rsp + 88], rax

    ; ---- scores -> softmax -> values, per head ------------------------------
    ; MQA (one KV head) means the 8 heads share these K/V rows; the attention
    ; scaling is 1.0 in Gemma 4 (q_norm/k_norm already bound the magnitudes),
    ; so no 1/sqrt(head_dim) appears anywhere.
.ly_head:
    xor     r12d, r12d
.ly_att:
    mov     rbp, [r15 + D_HEADDIM]
    mov     rcx, [rbx + CTR_BUF_SCORE]
    mov     rdx, [rbx + CTR_BUF_Q]
    mov     rax, r12
    imul    rax, rbp
    shl     rax, 2
    add     rdx, rax
    mov     r8, [r15 + D_KV]
    mov     r9, [rsp + 88]
    mov     [rsp + 32], rbp                     ; head_dim
    mov     rax, [rsp + 80]
    mov     [rsp + 40], rax                     ; kv_start
    call    attn_scores_avx2                    ; (scores, q, K, kv_len, hd, start)

    mov     rcx, [rbx + CTR_BUF_SCORE]
    mov     rdx, [rsp + 88]
    call    softmax_avx2                        ; (buf, n)

    mov     rcx, [rbx + CTR_BUF_A]
    mov     rax, r12
    imul    rax, [r15 + D_HEADDIM]
    shl     rax, 2
    add     rcx, rax
    mov     rdx, [r15 + D_VV]
    mov     r8, [rbx + CTR_BUF_SCORE]
    mov     r9, [rsp + 88]
    mov     rax, [r15 + D_HEADDIM]
    mov     [rsp + 32], rax
    mov     rax, [rsp + 80]
    mov     [rsp + 40], rax
    call    attn_values_avx2                    ; (out, V, w, n, hd, start)

    inc     r12
    cmp     r12, [rbx + CTR_NHEAD]
    jb      .ly_att
    add     qword [rbx + CTR_MARK], 8

    ; x += post_attention_layernorm(o_proj(a))
    mov     rax, [r15 + D_HEADDIM]
    imul    rax, [rbx + CTR_NHEAD]
    mov     rdx, [rbx + CTR_BUF_A]
    mov     rcx, [rbx + CTR_BUF_T2]
    mov     r8, rdx
    mov     rdx, [r15 + D_OW]
    mov     r9, rax                             ; K = n_head * head_dim
    mov     rax, [rbx + CTR_HIDDEN]
    mov     [rsp + 32], rax                     ; M = hidden
    mov     qword [rsp + 40], 1                 ; B = 1
    mov     rax, [rbx + CTR_SMP]
    mov     [rsp + 48], rax
    call    smp_bf16_gemb_avx2
    mov     rcx, [rbx + CTR_BUF_T2]
    mov     rdx, [r15 + D_NPOSTATT]
    mov     r8, [rbx + CTR_BUF_T2]
    mov     r9, [rbx + CTR_HIDDEN]
    mov     eax, [rbx + CTR_RMS_EPS]
    mov     [rsp + 32], rax
    call    rmsnorm_avx2
    GEMB_RESID                                  ; x = x + t2
    add     qword [rbx + CTR_MARK], 16

    ; ---- MLP (double-wide on KV-shared layers) ------------------------------
    mov     rcx, [rbx + CTR_BUF_T3]
    mov     rdx, [r15 + D_NPREFF]
    mov     r8, [rbx + CTR_BUF_X]
    mov     r9, [rbx + CTR_HIDDEN]
    mov     eax, [rbx + CTR_RMS_EPS]
    mov     [rsp + 32], rax
    call    rmsnorm_avx2

    mov     rbp, [r15 + D_INTER]
    GEMB    <[rbx + CTR_BUF_G]>, <[r15 + D_GATE]>, <[rbx + CTR_BUF_T3]>, <[rbx + CTR_HIDDEN]>, <rbp>
    mov     rbp, [r15 + D_INTER]
    GEMB    <[rbx + CTR_BUF_U]>, <[r15 + D_UP]>, <[rbx + CTR_BUF_T3]>, <[rbx + CTR_HIDDEN]>, <rbp>
    mov     rcx, [rbx + CTR_BUF_G]
    mov     rdx, [rbx + CTR_BUF_G]
    mov     r8, [rbx + CTR_BUF_U]
    mov     r9, [r15 + D_INTER]
    call    geglu_avx2                          ; (out, gate, up, N)
    mov     rbp, [rbx + CTR_HIDDEN]
    GEMB    <[rbx + CTR_BUF_T2]>, <[r15 + D_DOWN]>, <[rbx + CTR_BUF_G]>, <[r15 + D_INTER]>, <rbp>

    mov     rcx, [rbx + CTR_BUF_T2]
    mov     rdx, [r15 + D_NPOSTFF]
    mov     r8, [rbx + CTR_BUF_T2]
    mov     r9, [rbx + CTR_HIDDEN]
    mov     eax, [rbx + CTR_RMS_EPS]
    mov     [rsp + 32], rax
    call    rmsnorm_avx2
    GEMB_RESID                                  ; x = x + t2
    add     qword [rbx + CTR_MARK], 32

    ; ---- per-layer embeddings ----------------------------------------------
    ; gate = gelu_tanh(per_layer_input_gate(x)); mix = gate * ple_in[layer];
    ; x += post_per_layer_input_norm(per_layer_projection(mix))
    mov     rbp, [rbx + CTR_PLE_DIM]
    GEMB    <[rbx + CTR_BUF_TMP256]>, <[r15 + D_PLEGATE]>, <[rbx + CTR_BUF_X]>, <[rbx + CTR_HIDDEN]>, <rbp>

    mov     rcx, [rbx + CTR_BUF_TMP256]
    mov     rdx, [rbx + CTR_BUF_TMP256]
    mov     r8, [rbx + CTR_PLE_DIM]
    call    gelu_tanh_avx2                      ; (out, x, N)

    mov     rcx, [rbx + CTR_BUF_TMP256]
    mov     rax, [rsp + 96]
    imul    rax, [rbx + CTR_PLE_DIM]
    shl     rax, 2
    add     rax, [rbx + CTR_BUF_PLE_IN]
    mov     rdx, rax
    mov     r9, [rbx + CTR_PLE_DIM]
    call    mul_avx2                            ; gate *= this layer's PLE input

    mov     rbp, [rbx + CTR_HIDDEN]
    GEMB    <[rbx + CTR_BUF_T2]>, <[r15 + D_PLEPROJ]>, <[rbx + CTR_BUF_TMP256]>, <[rbx + CTR_PLE_DIM]>, <rbp>

    mov     rcx, [rbx + CTR_BUF_T2]
    mov     rdx, [r15 + D_PLENORM]
    mov     r8, [rbx + CTR_BUF_T2]
    mov     r9, [rbx + CTR_HIDDEN]
    mov     eax, [rbx + CTR_RMS_EPS]
    mov     [rsp + 32], rax
    call    rmsnorm_avx2

    mov     rcx, [rbx + CTR_BUF_X]
    mov     rdx, [rbx + CTR_BUF_X]
    mov     r8, [rbx + CTR_BUF_T2]
    mov     r9, [rbx + CTR_HIDDEN]
    mov     eax, C_ONE
    mov     [rsp + 32], rax
    mov     [rsp + 40], rax                     ; x = x*1 + t2*1
    call    add_scaled_avx2

    ; x *= layer_scalar: a per-layer buffer (1.0 unless fine-tuned) applied to
    ; the whole stream AFTER the PLE add, exactly as the reference orders it.
    mov     rcx, [rbx + CTR_BUF_X]
    mov     rdx, [rbx + CTR_HIDDEN]
    ; Scalars travel BY VALUE in a register (that is the internal convention:
    ; scale_avx2 does `vmovd xmm0, r8d`). Dereference the descriptor's pointer
    ; first - passing the address itself reinterprets 0x7f..4c40 as a float and
    ; multiplies the residual stream by ~1e27, which survives RMSNorm (it is
    ; scale-free) and only detonates at the logits.
    mov     r8, [r15 + D_SCALAR]
    mov     eax, [r8]
    mov     r8d, eax
    call    scale_avx2                          ; (buf, N, float value)

    add     rsp, 128
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    xor     eax, eax
    ret
