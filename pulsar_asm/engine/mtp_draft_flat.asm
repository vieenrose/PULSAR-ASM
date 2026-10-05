; ==============================================================================
; Project PULSAR-ASM | engine/mtp_draft_flat.asm
; ------------------------------------------------------------------------------
; Gemma 4 MTP assistant: one drafting step in pure AVX2, no PyTorch, no NumPy.
;
; The assistant is deliberately BORING next to the target engine, and that is
; the point. Drafting is sequential (one token per step, constant position for
; the whole round), so there is no batching, no worker pool, no PLE, no MoE:
;
;   emb  = target_embed[prev_tok] * sqrt(backbone_hidden)      (embed_bf16)
;   x    = pre_proj[256,3072] @ cat(emb, h_prev)               (direct GEMM)
;   x    = layer(x) x4   // rmsnorm, 4-head MHA reading the TARGET's KV
;                        // caches (never written), SwiGLU MLP, * scalar
;   h    = rmsnorm(x, final_norm)
;   best = argmax over the masked head: 2048 centroid dots, top 32 clusters,
;          32*128 = 4096 gathered embed rows, argmax over those 4096
;   h_next = post_proj[1536,256] @ h                           (feeds next draft)
;
; Returns best in rax and writes h_next (1536 fp32) to [ctx H_NEXT].
;
; What this file does NOT do: KV writes (there are none - the assistant reads
; the target's published rows through K_S/V_S/K_F/V_F), PLE (the assistant has
; none), sampling (the draft is always the argmax; verification decides).
;
; New code here is only the driver, the top-32 select and the row gather. Every
; flop goes through the shared kernels from raw_materials, called with the
; same ABI the target engine uses: rcx, rdx, r8, r9, then caller [rsp+32],
; [rsp+40], ... - every stack slot written explicitly before each call, and
; the frame (sub rsp,184, %16==8 for aligned calls) never moves mid-step.
; ==============================================================================

use64

; ---- assistant context: indexes, dims, then pointers (loader parses these) --
MTP_W_PRE        equ 0
MTP_W_POST       equ 8
MTP_W_CENT       equ 16
MTP_TOK_ORDER    equ 24
MTP_W_EMB_A      equ 32
MTP_W_FIN        equ 40
MTP_W_Q0         equ 48
MTP_W_Q1         equ 56
MTP_W_Q2         equ 64
MTP_W_Q3         equ 72
MTP_W_O0         equ 80
MTP_W_O1         equ 88
MTP_W_O2         equ 96
MTP_W_O3         equ 104
MTP_N_IN0        equ 112
MTP_N_IN1        equ 120
MTP_N_IN2        equ 128
MTP_N_IN3        equ 136
MTP_N_PA0        equ 144
MTP_N_PA1        equ 152
MTP_N_PA2        equ 160
MTP_N_PA3        equ 168
MTP_N_PF0        equ 176
MTP_N_PF1        equ 184
MTP_N_PF2        equ 192
MTP_N_PF3        equ 200
MTP_N_PFF0       equ 208
MTP_N_PFF1       equ 216
MTP_N_PFF2       equ 224
MTP_N_PFF3       equ 232
MTP_N_Q0         equ 240
MTP_N_Q1         equ 248
MTP_N_Q2         equ 256
MTP_N_Q3         equ 264
MTP_MG0          equ 272
MTP_MG1          equ 280
MTP_MG2          equ 288
MTP_MG3          equ 296
MTP_MU0          equ 304
MTP_MU1          equ 312
MTP_MU2          equ 320
MTP_MU3          equ 328
MTP_MD0          equ 336
MTP_MD1          equ 344
MTP_MD2          equ 352
MTP_MD3          equ 360
MTP_SC0          equ 368
MTP_SC1          equ 376
MTP_SC2          equ 384
MTP_SC3          equ 392
MTP_TGT_EMB      equ 400
MTP_K_S          equ 408
MTP_V_S          equ 416
MTP_K_F          equ 424
MTP_V_F          equ 432
MTP_COS_S        equ 440
MTP_SIN_S        equ 448
MTP_COS_F        equ 456
MTP_SIN_F        equ 464
MTP_STRIDE_S     equ 472
MTP_STRIDE_F     equ 480
MTP_EMB_SCALE    equ 488
MTP_EPS          equ 496
MTP_POS          equ 504
MTP_PREV_TOK     equ 512
MTP_H_PREV       equ 520
MTP_H_NEXT       equ 528
MTP_WINDOW       equ 536
MTP_SIZE         equ 544

; ---- manifest-locked geometry (asserted by the loader, not probed here) -----
MC_HID      equ 256
MC_NHEAD    equ 4
MC_HD_S     equ 256
MC_HD_F     equ 512
MC_INTER    equ 2048
MC_NCENT    equ 2048
MC_TOPK     equ 32
MC_VPC      equ 128
MC_NCAND    equ 4096
MC_BHID     equ 1536
MC_WIN      equ 512

include '../raw_materials/mat_bf16_gemb_avx2_flat.asm'
include '../raw_materials/mat_gemma4_kernels_avx2_flat.asm'

; ==============================================================================
; mtp_draft_step(ctx) -> rax = best token; h_next written to [ctx H_NEXT].
;   rcx = ctx. Preserves rbx, r12-r15. Frame: sub rsp,184 once; saves at
;   +128..+160, scratch at +64..+120, kernel stack args at +32..+56.
; ==============================================================================

F_BEST   equ 64
F_HD     equ 72
F_LO     equ 80
F_N      equ 88
F_KB     equ 96
F_VB     equ 104
F_COS    equ 112
F_SIN    equ 120
F_OFF    equ 168
F_RBX    equ 128
F_R12    equ 136
F_R13    equ 144
F_R14    equ 152
F_R15    equ 160

mtp_draft_step:
    sub     rsp, 184
    mov     [rsp + F_RBX], rbx
    mov     [rsp + F_R12], r12
    mov     [rsp + F_R13], r13
    mov     [rsp + F_R14], r14
    mov     [rsp + F_R15], r15
    mov     rbx, rcx                          ; ctx lives in rbx from here on

    ; ---- 1) target embedding row for prev_tok, scaled ---------------------
    lea     rcx, [MTP_BUF_EMB]
    mov     rdx, [rbx + MTP_TGT_EMB]
    mov     r8, [rbx + MTP_PREV_TOK]
    mov     r9, MC_BHID
    mov     eax, [rbx + MTP_EMB_SCALE]
    mov     [rsp + 32], rax
    call    embed_bf16_avx2                   ; EMB = tgt_emb[tok] * sqrt(1536)

    ; ---- 2) concat + pre_proj ---------------------------------------------
    lea     rcx, [MTP_BUF_CAT]
    lea     rdx, [MTP_BUF_EMB]
    mov     r8, MC_BHID
    call    copy_avx2                         ; CAT[0:1536] = emb
    lea     rcx, [MTP_BUF_CAT + MC_BHID * 4]
    mov     rdx, [rbx + MTP_H_PREV]
    mov     r8, MC_BHID
    call    copy_avx2                         ; CAT[1536:3072] = h_prev
    lea     rcx, [MTP_BUF_X]
    mov     rdx, [rbx + MTP_W_PRE]
    lea     r8, [MTP_BUF_CAT]
    mov     r9, MC_BHID * 2
    mov     qword [rsp + 32], MC_HID
    mov     qword [rsp + 40], 1
    mov     qword [rsp + 48], 0
    mov     qword [rsp + 56], 0
    call    bf16_gemb_avx2                    ; X = pre_proj @ cat

    ; ---- 3) four assistant layers ------------------------------------------
    mov     r12, 0                            ; layer index
.step_layer:
    cmp     r12, 4
    jae     .step_norm

    ; geometry + tables for this layer (layer 3 is the full one)
    mov     r14, MC_HD_S
    mov     rax, [rbx + MTP_K_S]
    mov     [rsp + F_KB], rax
    mov     rax, [rbx + MTP_V_S]
    mov     [rsp + F_VB], rax
    mov     rax, [rbx + MTP_COS_S]
    mov     [rsp + F_COS], rax
    mov     rax, [rbx + MTP_SIN_S]
    mov     [rsp + F_SIN], rax
    cmp     r12, 3
    jne     .step_geom_ok
    mov     r14, MC_HD_F
    mov     rax, [rbx + MTP_K_F]
    mov     [rsp + F_KB], rax
    mov     rax, [rbx + MTP_V_F]
    mov     [rsp + F_VB], rax
    mov     rax, [rbx + MTP_COS_F]
    mov     [rsp + F_COS], rax
    mov     rax, [rbx + MTP_SIN_F]
    mov     [rsp + F_SIN], rax
.step_geom_ok:
    mov     [rsp + F_HD], r14

    ; window rows: full -> [0, pos+1); sliding -> last 512 ending at pos
    mov     rax, [rbx + MTP_POS]
    inc     rax                               ; n = pos + 1
    xor     r15, r15                          ; lo = 0
    cmp     r12, 3
    jne     .step_win
    mov     [rsp + F_LO], r15
    mov     [rsp + F_N], rax
    jmp     .step_attn_prep
.step_win:
    cmp     rax, MC_WIN
    jbe     .step_win_ok
    mov     r15, rax
    sub     r15, MC_WIN                      ; lo = n - 512
    mov     rax, MC_WIN                      ; n = 512
.step_win_ok:
    mov     [rsp + F_LO], r15
    mov     [rsp + F_N], rax

.step_attn_prep:
    ; H1 = rmsnorm(X)
    lea     rcx, [MTP_BUF_H1]
    mov     rax, r12
    shl     rax, 3
    mov     rdx, [rbx + MTP_N_IN0 + rax]
    lea     r8, [MTP_BUF_X]
    mov     r9, MC_HID
    mov     eax, [rbx + MTP_EPS]
    mov     [rsp + 32], rax
    call    rmsnorm_avx2

    ; Q = w_q @ H1   (M = 4*hd, K = 256)
    lea     rcx, [MTP_BUF_Q]
    mov     rax, r12
    shl     rax, 3
    mov     rdx, [rbx + MTP_W_Q0 + rax]
    lea     r8, [MTP_BUF_H1]
    mov     r9, MC_HID
    mov     rax, [rsp + F_HD]
    shl     rax, 2                            ; M = 4 * hd
    mov     [rsp + 32], rax
    mov     qword [rsp + 40], 1
    mov     qword [rsp + 48], 0
    mov     qword [rsp + 56], 0
    call    bf16_gemb_avx2

    ; per-head: norm_q, rope, scores, softmax, values.
    ; The counter lives in R15: attn_values_avx2 overwrites R13/R14 with
    ; head_dim/kv_start BEFORE pushing them, so it restores garbage into
    ; those two - only R12/R15/RBX come back intact. The byte offset is
    ; recomputed from the counter per use (living registers do not survive
    ; kernel calls except r12/r15/rbx, and r12 is the layer index).
    mov     r15, 0                            ; head index
.step_head:
    cmp     r15, MC_NHEAD
    jae     .step_o
    mov     rax, r15
    imul    rax, [rsp + F_HD]
    shl     rax, 2                            ; head byte offset = j * hd * 4
    mov     [rsp + F_OFF], rax
    ; QN_seg = rmsnorm(Q_seg) into the matching QN segment
    mov     rax, [rsp + F_OFF]
    lea     rcx, [MTP_BUF_QN]
    add     rcx, rax
    mov     rax, r12
    shl     rax, 3
    mov     rdx, [rbx + MTP_N_Q0 + rax]
    mov     rax, [rsp + F_OFF]
    lea     r8, [MTP_BUF_Q]
    add     r8, rax
    mov     r9, [rsp + F_HD]
    mov     eax, [rbx + MTP_EPS]
    mov     [rsp + 32], rax
    call    rmsnorm_avx2
    ; rope in place over the head segment; table row = base + pos * stride
    mov     rax, [rsp + F_OFF]
    lea     rcx, [MTP_BUF_QN]
    add     rcx, rax
    mov     rax, [rbx + MTP_POS]
    cmp     r12, 3
    jne     .step_rope_s
    imul    rax, [rbx + MTP_STRIDE_F]
    add     rax, [rsp + F_COS]
    mov     rdx, rax
    mov     rax, [rbx + MTP_POS]
    imul    rax, [rbx + MTP_STRIDE_F]
    add     rax, [rsp + F_SIN]
    mov     r8, rax
    jmp     .step_rope_do
.step_rope_s:
    imul    rax, [rbx + MTP_STRIDE_S]
    add     rax, [rsp + F_COS]
    mov     rdx, rax
    mov     rax, [rbx + MTP_POS]
    imul    rax, [rbx + MTP_STRIDE_S]
    add     rax, [rsp + F_SIN]
    mov     r8, rax
.step_rope_do:
    ; the kernel takes rp (hd/2) on the STACK and ignores r9; an unwritten
    ; stack slot here once sent it roaming off the buffers (segfault).
    mov     r9, [rsp + F_HD]                  ; full head dim, as the target does
    mov     rax, r9
    shr     rax, 1                           ; rp = hd / 2 pairs
    mov     [rsp + 32], rax
    call    rope_apply_avx2
    ; scores = K[lo:lo+n] . qj ; softmax ; head_out = weights . V
    lea     rcx, [MTP_BUF_S]
    mov     rax, [rsp + F_OFF]
    lea     rdx, [MTP_BUF_QN]
    add     rdx, rax
    mov     r8, [rsp + F_KB]
    mov     r9, [rsp + F_N]
    mov     rax, [rsp + F_HD]
    mov     [rsp + 32], rax
    mov     rax, [rsp + F_LO]
    mov     [rsp + 40], rax
    call    attn_scores_avx2
    lea     rcx, [MTP_BUF_S]
    mov     rdx, [rsp + F_N]
    call    softmax_avx2
    mov     rax, [rsp + F_OFF]
    lea     rcx, [MTP_BUF_AV]
    add     rcx, rax
    mov     rdx, [rsp + F_VB]
    lea     r8, [MTP_BUF_S]
    mov     r9, [rsp + F_N]
    mov     rax, [rsp + F_HD]
    mov     [rsp + 32], rax
    mov     rax, [rsp + F_LO]
    mov     [rsp + 40], rax
    call    attn_values_avx2
    inc     r15
    jmp     .step_head

.step_o:
    ; O = w_o @ AVec (K = 4*hd) ; X += rmsnorm(O)
    lea     rcx, [MTP_BUF_O]
    mov     rax, r12
    shl     rax, 3
    mov     rdx, [rbx + MTP_W_O0 + rax]
    lea     r8, [MTP_BUF_AV]
    mov     rax, [rsp + F_HD]
    shl     rax, 2                            ; K = 4 * hd
    mov     r9, rax
    mov     qword [rsp + 32], MC_HID
    mov     qword [rsp + 40], 1
    mov     qword [rsp + 48], 0
    mov     qword [rsp + 56], 0
    call    bf16_gemb_avx2
    lea     rcx, [MTP_BUF_ON]
    mov     rax, r12
    shl     rax, 3
    mov     rdx, [rbx + MTP_N_PA0 + rax]
    lea     r8, [MTP_BUF_O]
    mov     r9, MC_HID
    mov     eax, [rbx + MTP_EPS]
    mov     [rsp + 32], rax
    call    rmsnorm_avx2
    lea     rcx, [MTP_BUF_X]
    lea     rdx, [MTP_BUF_X]
    lea     r8, [MTP_BUF_ON]
    mov     r9, MC_HID
    mov     eax, 0x3F800000                   ; 1.0f
    mov     [rsp + 32], rax
    mov     [rsp + 40], rax
    call    add_scaled_avx2                   ; X += ON

    ; MLP: H2 = rmsnorm(X); G = gate@H2; U = up@H2; G = gelu(G)*U;
    ;      D = down@G; X += rmsnorm(D); X *= scalar
    lea     rcx, [MTP_BUF_H2]
    mov     rax, r12
    shl     rax, 3
    mov     rdx, [rbx + MTP_N_PF0 + rax]
    lea     r8, [MTP_BUF_X]
    mov     r9, MC_HID
    mov     eax, [rbx + MTP_EPS]
    mov     [rsp + 32], rax
    call    rmsnorm_avx2
    lea     rcx, [MTP_BUF_G]
    mov     rax, r12
    shl     rax, 3
    mov     rdx, [rbx + MTP_MG0 + rax]
    lea     r8, [MTP_BUF_H2]
    mov     r9, MC_HID
    mov     qword [rsp + 32], MC_INTER
    mov     qword [rsp + 40], 1
    mov     qword [rsp + 48], 0
    mov     qword [rsp + 56], 0
    call    bf16_gemb_avx2
    lea     rcx, [MTP_BUF_U]
    mov     rax, r12
    shl     rax, 3
    mov     rdx, [rbx + MTP_MU0 + rax]
    lea     r8, [MTP_BUF_H2]
    mov     r9, MC_HID
    mov     qword [rsp + 32], MC_INTER
    mov     qword [rsp + 40], 1
    mov     qword [rsp + 48], 0
    mov     qword [rsp + 56], 0
    call    bf16_gemb_avx2
    lea     rcx, [MTP_BUF_G]
    lea     rdx, [MTP_BUF_G]
    lea     r8, [MTP_BUF_U]
    mov     r9, MC_INTER
    call    geglu_avx2                        ; G = gelu(G) * U, in place
    lea     rcx, [MTP_BUF_D]
    mov     rax, r12
    shl     rax, 3
    mov     rdx, [rbx + MTP_MD0 + rax]
    lea     r8, [MTP_BUF_G]
    mov     r9, MC_INTER
    mov     qword [rsp + 32], MC_HID
    mov     qword [rsp + 40], 1
    mov     qword [rsp + 48], 0
    mov     qword [rsp + 56], 0
    call    bf16_gemb_avx2
    lea     rcx, [MTP_BUF_DN]
    mov     rax, r12
    shl     rax, 3
    mov     rdx, [rbx + MTP_N_PFF0 + rax]
    lea     r8, [MTP_BUF_D]
    mov     r9, MC_HID
    mov     eax, [rbx + MTP_EPS]
    mov     [rsp + 32], rax
    call    rmsnorm_avx2
    lea     rcx, [MTP_BUF_X]
    lea     rdx, [MTP_BUF_X]
    lea     r8, [MTP_BUF_DN]
    mov     r9, MC_HID
    mov     eax, 0x3F800000
    mov     [rsp + 32], rax
    mov     [rsp + 40], rax
    call    add_scaled_avx2                   ; X += DN
    lea     rcx, [MTP_BUF_X]
    mov     rdx, MC_HID
    mov     rax, r12
    shl     rax, 3
    mov     rax, [rbx + MTP_SC0 + rax]
    mov     r8d, [rax]                        ; scalar value, float bits in R8
    call    scale_avx2                        ; X *= scalar

    inc     r12
    jmp     .step_layer

.step_norm:
    ; HF = rmsnorm(X, final_norm); CDOT = centroids @ HF
    lea     rcx, [MTP_BUF_HF]
    mov     rdx, [rbx + MTP_W_FIN]
    lea     r8, [MTP_BUF_X]
    mov     r9, MC_HID
    mov     eax, [rbx + MTP_EPS]
    mov     [rsp + 32], rax
    call    rmsnorm_avx2
    lea     rcx, [MTP_BUF_CDOT]
    mov     rdx, [rbx + MTP_W_CENT]
    lea     r8, [MTP_BUF_HF]
    mov     r9, MC_HID
    mov     qword [rsp + 32], MC_NCENT
    mov     qword [rsp + 40], 1
    mov     qword [rsp + 48], 0
    mov     qword [rsp + 56], 0
    call    bf16_gemb_avx2

    ; top 32 clusters, gather their 4096 rows, argmax over candidates
    lea     rcx, [MTP_BUF_CDOT]
    lea     rdx, [MTP_BUF_TIDX]
    call    mtp_top32_avx2
    lea     rcx, [MTP_BUF_CBUF]
    mov     rdx, [rbx + MTP_W_EMB_A]
    mov     r8, [rbx + MTP_TOK_ORDER]
    lea     r9, [MTP_BUF_TIDX]
    call    mtp_gather_rows_avx2
    lea     rcx, [MTP_BUF_CL]
    lea     rdx, [MTP_BUF_CBUF]
    lea     r8, [MTP_BUF_HF]
    mov     r9, MC_HID
    mov     qword [rsp + 32], MC_NCAND
    mov     qword [rsp + 40], 1
    mov     qword [rsp + 48], 0
    mov     qword [rsp + 56], 0
    call    bf16_gemb_avx2                    ; CL = CBUF @ HF
    lea     rcx, [MTP_BUF_CL]
    mov     rdx, MC_NCAND
    call    sampler_argmax_avx2               ; rax = winning candidate slot
    mov     [rsp + F_BEST], rax

    ; h_next = post_proj @ HF  (feeds the next draft's h_prev)
    mov     rcx, [rbx + MTP_H_NEXT]
    mov     rdx, [rbx + MTP_W_POST]
    lea     r8, [MTP_BUF_HF]
    mov     r9, MC_HID
    mov     qword [rsp + 32], MC_BHID
    mov     qword [rsp + 40], 1
    mov     qword [rsp + 48], 0
    mov     qword [rsp + 56], 0
    call    bf16_gemb_avx2

    ; winning token id: slot/128 = which of the 32, slot%128 = which of 128
    mov     rax, [rsp + F_BEST]
    mov     rcx, rax
    shr     rcx, 7
    lea     r10, [MTP_BUF_TIDX]
    mov     edx, [r10 + rcx * 4]              ; centroid id
    and     rax, 127
    shl     rdx, 7
    add     rdx, rax                          ; order index = cent*128 + k
    mov     rax, [rbx + MTP_TOK_ORDER]
    mov     eax, [rax + rdx * 4]              ; token id
    mov     [rsp + F_BEST], rax

    mov     rbx, [rsp + F_RBX]
    mov     r12, [rsp + F_R12]
    mov     r13, [rsp + F_R13]
    mov     r14, [rsp + F_R14]
    mov     r15, [rsp + F_R15]
    mov     rax, [rsp + F_BEST]
    add     rsp, 184
    vzeroupper
    ret

; ==============================================================================
; mtp_top32_avx2(vals, idx) - indices of the 32 largest of 2048 fp32, any order.
;   rcx = vals (MUTILATED: picked entries are set to -inf), rdx = idx [32] u32.
;   32 linear scans; strictly-greater keeps the first max on ties, matching
;   the argmax convention downstream (and the reference's argmax over the set).
; ==============================================================================

mtp_top32_avx2:
    mov     r10, 0                            ; round
.top_round:
    cmp     r10, MC_TOPK
    jae     .top_done
    mov     eax, 0xFF800000
    vmovd   xmm0, eax                         ; best = -inf
    mov     r11, -1                           ; best idx
    xor     rax, rax                          ; i
.top_scan:
    cmp     rax, MC_NCENT
    jae     .top_pick
    vmovss  xmm1, dword [rcx + rax * 4]
    vucomiss xmm1, xmm0
    jbe     .top_next                         ; not strictly greater: keep first
    vmovaps xmm0, xmm1
    mov     r11, rax
.top_next:
    inc     rax
    jmp     .top_scan
.top_pick:
    mov     [rdx + r10 * 4], r11d
    mov     eax, 0xFF800000
    mov     [rcx + r11 * 4], eax              ; mask the pick
    inc     r10
    jmp     .top_round
.top_done:
    vzeroupper
    ret

; ==============================================================================
; mtp_gather_rows_avx2(dst, table, order, idx) - gather the 4096 candidate rows.
;   rcx = dst (4096*256 bf16), rdx = assist embed base, r8 = tok_order (i32),
;   r9 = idx (32 centroid ids). Geometry (32 clusters x 128 vocab x 256 dim)
;   is manifest-locked; the loader asserts it.
; ==============================================================================

mtp_gather_rows_avx2:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15
    mov     rbx, rdx                          ; table
    mov     r12, r8                           ; order
    mov     r13, r9                           ; idx
    mov     r14, rcx                          ; dst
    xor     r15, r15                          ; c = cluster slot 0..31
.g_c:
    cmp     r15, MC_TOPK
    jae     .g_done
    mov     eax, [r13 + r15 * 4]              ; centroid id
    shl     rax, 7                            ; cent * 128
    lea     r10, [r12 + rax * 4]              ; &order[cent*128]
    mov     rax, r15
    shl     rax, 16                           ; c * 128 * 512 = c * 65536
    lea     r11, [r14 + rax]                  ; &dst[(c*128)*256] bf16 bytes
    xor     r9, r9                            ; k = 0..127
.g_k:
    cmp     r9, MC_VPC
    jae     .g_next
    mov     eax, [r10 + r9 * 4]               ; token id
    shl     rax, 9                            ; tok * 512 bytes
    lea     rsi, [rbx + rax]                  ; src row
    mov     rdi, r11                          ; dst row
    mov     rcx, 16                           ; 16 x 32B = 512B
.g_row:
    vmovups ymm0, [rsi]
    vmovups [rdi], ymm0
    add     rsi, 32
    add     rdi, 32
    dec     rcx
    jnz     .g_row
    add     r11, 512
    inc     r9
    jmp     .g_k
.g_next:
    inc     r15
    jmp     .g_c
.g_done:
    vzeroupper
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret

; ---- module scratch (.bss in the mapped image; single-threaded draft) -------

MTP_BUF_EMB:  rb MC_BHID * 4
MTP_BUF_CAT:  rb MC_BHID * 2 * 4
MTP_BUF_X:    rb MC_HID * 4
MTP_BUF_H1:   rb MC_HID * 4
MTP_BUF_Q:    rb MC_NHEAD * MC_HD_F * 4
MTP_BUF_QN:   rb MC_NHEAD * MC_HD_F * 4
MTP_BUF_S:    rb 512 * 4
MTP_BUF_AV:   rb MC_NHEAD * MC_HD_F * 4
MTP_BUF_O:    rb MC_HID * 4
MTP_BUF_ON:   rb MC_HID * 4
MTP_BUF_H2:   rb MC_HID * 4
MTP_BUF_G:    rb MC_INTER * 4
MTP_BUF_U:    rb MC_INTER * 4
MTP_BUF_D:    rb MC_HID * 4
MTP_BUF_DN:   rb MC_HID * 4
MTP_BUF_HF:   rb MC_HID * 4
MTP_BUF_CDOT: rb MC_NCENT * 4
MTP_BUF_TIDX: rb MC_TOPK * 4
MTP_BUF_CBUF: rb MC_NCAND * MC_HID * 2
MTP_BUF_CBUF_END:
MTP_BUF_CL:   rb MC_NCAND * 4

; ------------------------------------------------------------------------------
; PLSE export trailer (same shape as the target engine's).
; ------------------------------------------------------------------------------

mtp_n0: db 'mtp_draft_step', 0
mtp_n1: db 'mtp_top32_avx2', 0
mtp_n2: db 'mtp_gather_rows_avx2', 0
mtp_n3: db 'bf16_gemb_avx2', 0
    align 4
    dd mtp_draft_step, mtp_top32_avx2, mtp_gather_rows_avx2, bf16_gemb_avx2
    dd mtp_n0, mtp_n1, mtp_n2, mtp_n3
    dd 4
    db 'PLSE'
