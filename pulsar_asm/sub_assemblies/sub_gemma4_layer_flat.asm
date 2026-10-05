; ==============================================================================
; Project PULSAR-ASM | Gemma 4 E2B  (sub_assemblies/sub_gemma4_layer_flat.asm)
; ------------------------------------------------------------------------------
; One decoder layer, transcribed from Gemma4TextDecoderLayer.forward(), run for
; CTR_B tokens at once (CTR_B = 1 is ordinary decode, B > 1 is speculative
; verification).
;
; ENTRY:  rbx = ctx (layout in engine/gemma4_engine_flat.asm)
;         r15 = &layer descriptor
;         rax = layer index (selects this layer's PLE input slice)
; EXIT:   rbx/rbp/r12..r15 preserved
;
; BATCHING
; Activation buffers are [B][dim] row-major, so row b starts at base + b*stride
; with stride = dim*4. Elementwise kernels therefore need no loop at all - they
; just get N = B*dim - and only the row-normalizing work (RMSNorm, RoPE,
; attention, anything position-dependent) loops over b. The GEMMs are the reason
; for the whole exercise: one pass over the weights serves all B rows, and decode
; is DRAM-bound, so that IS the speedup.
;
; Register use: rbx ctx | r15 descriptor | r13 cos | r14 sin | r12 head counter
; | rbp scratch/first GEMM arg. Legal because PULSAR kernels preserve
; callee-saved registers - the two attention kernels did not and had to be fixed:
; a kernel returning with r12 changed corrupts whatever the interpreter kept
; there, and fails far from its cause.
;
; Frame:  [rsp+0..32)    home space (kernels' shadow area, never touched)
;         [rsp+32..64)   stack args for kernel calls
;         [rsp+64..]     locals, see EQU-ish list below (byte offsets)
;              +64 scratch ptr   +72 scratch ptr   +80 kv_start  +88 kv_len
;              +96 layer index   +104 b            +112 B
;              +120 position of row b              +128 x stride
;              +136 q stride     +144 ple row stride
;              +152 inter stride +160 ple-scratch stride
; ==============================================================================

L_B       equ 104                             ; current row
L_N       equ 112                             ; B
L_POS     equ 120                             ; position of the current row
L_XS      equ 128                             ; row stride of the [*,hidden] buffers
L_QS      equ 136                             ; row stride of Q/A
L_PS      equ 144                             ; row stride of the PLE buffers
L_IS      equ 152                             ; row stride of G/U
L_TS      equ 160                             ; row stride of the PLE gate scratch

gemma4_layer:
    push    rbp
    push    r12
    push    r13
    push    r14
    sub     rsp, 192
    mov     [rsp + 96], rax
    mov     rax, [rbx + CTR_B]
    test    rax, rax
    jnz     .ly_b_ok
    mov     rax, 1
.ly_b_ok:
    mov     [rsp + L_N], rax
    mov     qword [rsp + L_B], 0
    mov     rax, 200
    add     rax, [rsp + 96]
    mov     [rbx + CTR_MARK], rax             ; 100 + layer index, for post-mortem

    ; row strides, in bytes
    mov     rax, [rbx + CTR_HIDDEN]
    shl     rax, 2
    mov     [rsp + L_XS], rax
    mov     rax, [rbx + CTR_QSTRIDE]
    mov     [rsp + L_QS], rax          ; n_head*MAX_head_dim*4, not this layer's:
                                       ; the buffer is sized by the widest layer, so
                                       ; deriving it from D_HEADDIM makes batch b>=1
                                       ; land inside batch 0 on sliding layers
    mov     rax, [rbx + CTR_PLE_DIM]
    imul    rax, [rbx + CTR_NLAYER]
    shl     rax, 2
    mov     [rsp + L_PS], rax
    mov     rax, [rbx + CTR_ISTRIDE]
    mov     [rsp + L_IS], rax          ; MAX_inter*4, same reason
    mov     rax, [rbx + CTR_PLE_DIM]
    shl     rax, 2
    mov     [rsp + L_TS], rax

    ; ---- input_layernorm, per row ------------------------------------------
    mov     qword [rsp + L_B], 0
.ly_nin:
    mov     rax, [rsp + L_B]
    imul    rax, [rsp + L_XS]
    mov     rcx, [rbx + CTR_BUF_T1]
    add     rcx, rax
    mov     rdx, [r15 + D_NIN]
    mov     r8, [rbx + CTR_BUF_X]
    add     r8, rax
    mov     r9, [rbx + CTR_HIDDEN]
    mov     eax, [rbx + CTR_RMS_EPS]
    mov     [rsp + 32], rax
    call    rmsnorm_avx2
    mov     rax, [rsp + L_B]
    inc     rax
    mov     [rsp + L_B], rax
    cmp     rax, [rsp + L_N]
    jb      .ly_nin

    ; ---- q = q_proj(t1): M = n_head*head_dim rows, B tokens per row --------
    mov     rbp, [r15 + D_HEADDIM]
    imul    rbp, [rbx + CTR_NHEAD]            ; M
    GEMMBX  <[rbx + CTR_BUF_Q]>, <[r15 + D_QW]>, <[rbx + CTR_BUF_T1]>, <[rbx + CTR_HIDDEN]>, <rbp>, <[rsp + L_N]>, 0, <[rsp + L_QS]>   ; out rows are max_hd-wide

    ; ---- per row, per head: q_norm then rope -------------------------------
    ; The norm is RMSNorm(head_dim), shared by all heads, and it runs BEFORE
    ; rope - rotating first is a different function. RoPE uses position POS+b,
    ; so the cos/sin row moves with b (a batched pass has B different
    ; positions, which is exactly what a single-token engine never exercises).
    mov     qword [rsp + L_B], 0
.ly_qrow:
    mov     rax, [rbx + CTR_POS]
    add     rax, [rsp + L_B]
    mov     [rsp + L_POS], rax
    mov     r13, [rbx + CTR_COS_SLIDE]
    mov     r14, [rbx + CTR_SIN_SLIDE]
    mov     rcx, [rbx + CTR_ROPE_STR_SLIDE]
    test    qword [r15 + D_FLAGS], FLAG_FULL
    jz      .ly_qrope_row
    mov     r13, [rbx + CTR_COS_FULL]
    mov     r14, [rbx + CTR_SIN_FULL]
    mov     rcx, [rbx + CTR_ROPE_STR_FULL]
.ly_qrope_row:
    imul    rax, rcx
    add     r13, rax
    add     r14, rax
    mov     rax, [rsp + L_B]
    imul    rax, [rsp + L_QS]
    add     rax, [rbx + CTR_BUF_Q]
    mov     [rsp + 64], rax                   ; &q[b][0]
    xor     r12d, r12d
.ly_qhead:
    mov     rcx, [rsp + 64]
    mov     rax, r12
    imul    rax, [r15 + D_HEADDIM]
    shl     rax, 2
    add     rcx, rax
    mov     rdx, [r15 + D_QNORM]
    mov     r8, rcx                           ; in place
    mov     r9, [r15 + D_HEADDIM]
    mov     eax, [rbx + CTR_RMS_EPS]
    mov     [rsp + 32], rax
    call    rmsnorm_avx2
    mov     rcx, [rsp + 64]
    mov     rax, r12
    imul    rax, [r15 + D_HEADDIM]
    shl     rax, 2
    add     rcx, rax
    mov     rdx, r13
    mov     r8, r14
    mov     r9, [r15 + D_HEADDIM]
    mov     rax, r9
    shr     rax, 1                            ; rp = head_dim/2 pairs
    mov     [rsp + 32], rax
    call    rope_apply_avx2                   ; (buf, cos, sin, H, rp)
    inc     r12
    cmp     r12, [rbx + CTR_NHEAD]
    jb      .ly_qhead
    mov     rax, [rsp + L_B]
    inc     rax
    mov     [rsp + L_B], rax
    cmp     rax, [rsp + L_N]
    jb      .ly_qrow

    ; ---- K / V --------------------------------------------------------------
    ; KV-shared layers (15..34) have no k_proj/v_proj tensors at all: their
    ; descriptor points at the cache published by the storing layer (13 for
    ; sliding, 14 for full), whose contents are already normed and rotated, so
    ; this entire block is skipped for them.
    test    qword [r15 + D_FLAGS], FLAG_KVSHARED
    jnz     .ly_window

    ; B consecutive cache rows, written by one batched GEMM: row b lands at
    ; (POS+b)*stride because the rows are contiguous and out_stride is M*4.
    mov     rbp, [r15 + D_HEADDIM]
    mov     rax, [rbx + CTR_POS]
    imul    rax, [r15 + D_KVSTRIDE]
    add     rax, [r15 + D_KV]
    GEMMB   <rax>, <[r15 + D_KW]>, <[rbx + CTR_BUF_T1]>, <[rbx + CTR_HIDDEN]>, <rbp>, <[rsp + L_N]>
    mov     rax, [rbx + CTR_POS]
    imul    rax, [r15 + D_KVSTRIDE]
    add     rax, [r15 + D_VV]
    GEMMB   <rax>, <[r15 + D_VW]>, <[rbx + CTR_BUF_T1]>, <[rbx + CTR_HIDDEN]>, <rbp>, <[rsp + L_N]>

    ; k_norm + rope, and v's scale-free norm, per row (v has no weight tensor in
    ; the checkpoint, which is exactly why rmsnorm_scale exists)
    mov     qword [rsp + L_B], 0
.ly_kvrow:
    mov     rax, [rbx + CTR_POS]
    add     rax, [rsp + L_B]
    mov     [rsp + L_POS], rax
    mov     r13, [rbx + CTR_COS_SLIDE]
    mov     r14, [rbx + CTR_SIN_SLIDE]
    mov     rcx, [rbx + CTR_ROPE_STR_SLIDE]
    test    qword [r15 + D_FLAGS], FLAG_FULL
    jz      .ly_kvrope_row
    mov     r13, [rbx + CTR_COS_FULL]
    mov     r14, [rbx + CTR_SIN_FULL]
    mov     rcx, [rbx + CTR_ROPE_STR_FULL]
.ly_kvrope_row:
    imul    rax, rcx
    add     r13, rax
    add     r14, rax
    mov     rax, [rsp + L_POS]
    imul    rax, [r15 + D_KVSTRIDE]
    mov     rcx, rax
    add     rcx, [r15 + D_KV]
    mov     [rsp + 64], rcx
    add     rax, [r15 + D_VV]
    mov     [rsp + 72], rax

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

    mov     rcx, [rsp + 72]
    mov     rdx, rcx
    mov     r8, [r15 + D_HEADDIM]
    mov     r9, C_ONE
    mov     eax, [rbx + CTR_RMS_EPS]
    mov     [rsp + 32], rax
    call    rmsnorm_scale_avx2                ; (out, x, N, scale, eps)

    mov     rax, [rsp + L_B]
    inc     rax
    mov     [rsp + L_B], rax
    cmp     rax, [rsp + L_N]
    jb      .ly_kvrow

    ; ---- attention, per row then per head ----------------------------------
    ; MQA (one KV head) means the heads share these K/V rows, and the attention
    ; scaling is 1.0 in Gemma 4 (q_norm/k_norm already bound the magnitudes), so
    ; no 1/sqrt(head_dim) appears anywhere. Row b attends rows
    ; [max(0, POS+b+1-window), POS+b+1) for sliding layers and [0, POS+b+1) for
    ; full ones - a per-row window, which a single-token pass cannot express.
.ly_window:
    mov     qword [rsp + L_B], 0
.ly_wrow:
    mov     rax, [rbx + CTR_POS]
    add     rax, [rsp + L_B]
    mov     [rsp + L_POS], rax
    test    qword [r15 + D_FLAGS], FLAG_KVSHARED
    jz      .ly_wrange
    mov     rax, [rbx + CTR_POS]              ; shared layers read the stored
    add     rax, [rsp + L_B]                  ; cache, whose rows were written
    mov     [rsp + L_POS], rax                ; for the same positions
.ly_wrange:
    mov     rbp, rax
    inc     rbp
    mov     [rsp + 88], rbp                   ; kv_len = position + 1
    mov     qword [rsp + 80], 0               ; kv_start = 0
    test    qword [r15 + D_FLAGS], FLAG_FULL
    jnz     .ly_att_head
    sub     rbp, [rbx + CTR_WINDOW]
    jle     .ly_att_head
    mov     [rsp + 80], rbp
    mov     rbp, [rbx + CTR_WINDOW]
    mov     [rsp + 88], rbp

.ly_att_head:
    xor     r12d, r12d
.ly_att:
    mov     rbp, [r15 + D_HEADDIM]
    mov     rax, [rsp + L_B]
    imul    rax, [rsp + L_QS]
    add     rax, [rbx + CTR_BUF_Q]
    mov     rdx, rax
    mov     rax, r12
    imul    rax, rbp
    shl     rax, 2
    add     rdx, rax                          ; &q[b][head]
    mov     rcx, [rbx + CTR_BUF_SCORE]
    mov     r8, [r15 + D_KV]
    mov     r9, [rsp + 88]
    mov     [rsp + 32], rbp                   ; head_dim
    mov     rax, [rsp + 80]
    mov     [rsp + 40], rax                   ; kv_start
    call    attn_scores_avx2                  ; (scores, q, K, kv_len, hd, start)

    mov     rcx, [rbx + CTR_BUF_SCORE]
    mov     rdx, [rsp + 88]
    call    softmax_avx2

    mov     rax, [rsp + L_B]
    imul    rax, [rsp + L_QS]
    add     rax, [rbx + CTR_BUF_A]
    mov     rbp, rax
    mov     rax, r12
    imul    rax, [r15 + D_HEADDIM]
    shl     rax, 2
    add     rbp, rax                          ; &a[b][head]
    mov     rcx, rbp
    mov     rdx, [r15 + D_VV]
    mov     r8, [rbx + CTR_BUF_SCORE]
    mov     r9, [rsp + 88]
    mov     rax, [r15 + D_HEADDIM]
    mov     [rsp + 32], rax
    mov     rax, [rsp + 80]
    mov     [rsp + 40], rax
    call    attn_values_avx2                  ; (out, V, w, n, hd, start)

    inc     r12
    cmp     r12, [rbx + CTR_NHEAD]
    jb      .ly_att
    mov     rax, [rsp + L_B]
    inc     rax
    mov     [rsp + L_B], rax
    cmp     rax, [rsp + L_N]
    jb      .ly_wrow

    ; ---- o_proj, then the attention residual -------------------------------
    mov     rbp, [r15 + D_HEADDIM]
    imul    rbp, [rbx + CTR_NHEAD]
    GEMMBX  <[rbx + CTR_BUF_T1]>, <[r15 + D_OW]>, <[rbx + CTR_BUF_A]>, <rbp>, <[rbx + CTR_HIDDEN]>, <[rsp + L_N]>, <[rsp + L_QS]>, 0

    mov     qword [rsp + L_B], 0
.ly_postr:
    mov     rax, [rsp + L_B]
    imul    rax, [rsp + L_XS]
    mov     rcx, [rbx + CTR_BUF_T2]
    add     rcx, rax
    mov     rdx, [r15 + D_NPOSTATT]
    mov     r8, [rbx + CTR_BUF_T1]
    add     r8, rax
    mov     r9, [rbx + CTR_HIDDEN]
    mov     eax, [rbx + CTR_RMS_EPS]
    mov     [rsp + 32], rax
    call    rmsnorm_avx2
    mov     rax, [rsp + L_B]
    inc     rax
    mov     [rsp + L_B], rax
    cmp     rax, [rsp + L_N]
    jb      .ly_postr
    GEMB_RESID                         ; x = x + t2, all rows at once

    ; ---- MLP (double-wide on KV-shared layers) ------------------------------
    mov     qword [rsp + L_B], 0
    mov     qword [rbx + CTR_MARK], 32
.ly_preffr:
    mov     rax, [rsp + L_B]
    imul    rax, [rsp + L_XS]
    mov     rcx, [rbx + CTR_BUF_T3]
    add     rcx, rax
    mov     rdx, [r15 + D_NPREFF]
    mov     r8, [rbx + CTR_BUF_X]
    add     r8, rax
    mov     r9, [rbx + CTR_HIDDEN]
    mov     eax, [rbx + CTR_RMS_EPS]
    mov     [rsp + 32], rax
    call    rmsnorm_avx2
    mov     rax, [rsp + L_B]
    inc     rax
    mov     [rsp + L_B], rax
    cmp     rax, [rsp + L_N]
    jb      .ly_preffr

    mov     rbp, [r15 + D_INTER]
    GEMMBX  <[rbx + CTR_BUF_G]>, <[r15 + D_GATE]>, <[rbx + CTR_BUF_T3]>, <[rbx + CTR_HIDDEN]>, <rbp>, <[rsp + L_N]>, 0, <[rsp + L_IS]>
    GEMMBX  <[rbx + CTR_BUF_U]>, <[r15 + D_UP]>, <[rbx + CTR_BUF_T3]>, <[rbx + CTR_HIDDEN]>, <rbp>, <[rsp + L_N]>, 0, <[rsp + L_IS]>
    ; One call PER ROW. geglu_avx2 walks its three operands linearly, so a single
    ; flat pass over B*inter is only correct when inter is also the row stride.
    ; G/U live in a buffer sized by the widest layer, so on a sliding layer the
    ; rows are max_inter apart and a flat pass would read row 0's unused back half
    ; as if it were batch 1 - which is exactly the batched-only corruption here.
    mov     qword [rsp + L_B], 0
.ly_geglu:
    mov     rax, [rsp + L_B]
    imul    rax, [rsp + L_IS]
    mov     rcx, [rbx + CTR_BUF_G]
    add     rcx, rax
    mov     rdx, rcx                          ; gate (out aliases gate in place)
    mov     r8, [rbx + CTR_BUF_U]
    add     r8, rax                           ; up
    mov     r9, [r15 + D_INTER]
    call    geglu_avx2                        ; (out, gate, up, N)
    mov     rax, [rsp + L_B]
    inc     rax
    mov     [rsp + L_B], rax
    cmp     rax, [rsp + L_N]
    jb      .ly_geglu

    mov     rbp, [rbx + CTR_HIDDEN]
    GEMMBX  <[rbx + CTR_BUF_T2]>, <[r15 + D_DOWN]>, <[rbx + CTR_BUF_G]>, <[r15 + D_INTER]>, <rbp>, <[rsp + L_N]>, <[rsp + L_IS]>, 0

    mov     qword [rsp + L_B], 0
.ly_postrff:
    mov     rax, [rsp + L_B]
    imul    rax, [rsp + L_XS]
    mov     rcx, [rbx + CTR_BUF_T2]
    add     rcx, rax
    mov     rdx, [r15 + D_NPOSTFF]
    mov     r8, [rbx + CTR_BUF_T2]
    add     r8, rax
    mov     r9, [rbx + CTR_HIDDEN]
    mov     eax, [rbx + CTR_RMS_EPS]
    mov     [rsp + 32], rax
    call    rmsnorm_avx2
    mov     rax, [rsp + L_B]
    inc     rax
    mov     [rsp + L_B], rax
    cmp     rax, [rsp + L_N]
    jb      .ly_postrff
    GEMB_RESID                         ; x = x + t2

    ; ---- per-layer embeddings ----------------------------------------------
    ; gate = gelu_tanh(per_layer_input_gate(x)); mix = gate * ple_in[layer];
    ; x += post_per_layer_input_norm(per_layer_projection(mix))
    ; The gate GEMM, the gelu and the projection are batched; only the multiply
    ; by this layer's PLE slice needs a loop, because that slice is ple_dim wide
    ; inside a nl*ple_dim row.
    mov     rbp, [rbx + CTR_PLE_DIM]
    GEMMB   <[rbx + CTR_BUF_TMP256]>, <[r15 + D_PLEGATE]>, <[rbx + CTR_BUF_X]>, <[rbx + CTR_HIDDEN]>, <rbp>, <[rsp + L_N]>
    mov     r8, [rbx + CTR_PLE_DIM]
    imul    r8, [rsp + L_N]
    mov     rcx, [rbx + CTR_BUF_TMP256]
    mov     rdx, [rbx + CTR_BUF_TMP256]
    call    gelu_tanh_avx2                    ; (out, x, N) over all rows

    mov     rax, [rsp + 96]
    imul    rax, [rbx + CTR_PLE_DIM]
    shl     rax, 2
    mov     [rsp + 72], rax                   ; slice offset inside a PLE row
    mov     qword [rsp + L_B], 0
.ly_plemul:
    mov     rax, [rsp + L_B]
    imul    rax, [rsp + L_TS]
    mov     rcx, [rbx + CTR_BUF_TMP256]
    add     rcx, rax
    mov     rdx, rcx
    mov     rax, [rsp + L_B]
    imul    rax, [rsp + L_PS]
    add     rax, [rsp + 72]
    add     rax, [rbx + CTR_BUF_PLE_IN]
    mov     rdx, rax                          ; (dst, src, N): N is the third
    mov     r8, [rbx + CTR_PLE_DIM]           ; argument, so the pointer goes in
    call    mul_avx2                          ; rdx first - r8 is already N
    mov     rax, [rsp + L_B]
    inc     rax
    mov     [rsp + L_B], rax
    cmp     rax, [rsp + L_N]
    jb      .ly_plemul

    mov     rbp, [rbx + CTR_HIDDEN]
    GEMMB   <[rbx + CTR_BUF_T2]>, <[r15 + D_PLEPROJ]>, <[rbx + CTR_BUF_TMP256]>, <[rbx + CTR_PLE_DIM]>, <rbp>, <[rsp + L_N]>

    mov     qword [rsp + L_B], 0
.ly_plenorm:
    mov     rax, [rsp + L_B]
    imul    rax, [rsp + L_XS]
    mov     rcx, [rbx + CTR_BUF_T2]
    add     rcx, rax
    mov     rdx, [r15 + D_PLENORM]
    mov     r8, [rbx + CTR_BUF_T2]
    add     r8, rax
    mov     r9, [rbx + CTR_HIDDEN]
    mov     eax, [rbx + CTR_RMS_EPS]
    mov     [rsp + 32], rax
    call    rmsnorm_avx2
    mov     rax, [rsp + L_B]
    inc     rax
    mov     [rsp + L_B], rax
    cmp     rax, [rsp + L_N]
    jb      .ly_plenorm
    GEMB_RESID                         ; x = x + ple projection

    ; ---- layer_scalar, by value --------------------------------------------
    ; Scalars travel BY VALUE in a register (the internal convention: scale_avx2
    ; does `vmovd xmm0, r8d`). Dereference the descriptor's pointer first -
    ; passing the address itself reinterprets 0x7f..4c40 as a float, multiplies
    ; the residual stream by ~1e27, survives RMSNorm because RMSNorm is
    ; scale-free, and only detonates at the logits.
    mov     rcx, [rbx + CTR_BUF_X]
    mov     rdx, [rbx + CTR_HIDDEN]
    imul    rdx, [rsp + L_N]
    mov     r8, [r15 + D_SCALAR]
    mov     eax, [r8]
    mov     r8d, eax
    call    scale_avx2                        ; (buf, N, float value)

    add     rsp, 192
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    xor     eax, eax
    ret
