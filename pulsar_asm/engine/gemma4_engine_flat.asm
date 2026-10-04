; ==============================================================================
; Project PULSAR-ASM | Gemma 4 E2B text engine  (engine/gemma4_engine_flat.asm)
; ------------------------------------------------------------------------------
; Pure x86-64 forward pass for google/gemma-4-E2B-it, text-only, CPU, fp32
; activations over bf16 weights, AVX2 + FMA3, 4 cores. No OS calls, no libc, no
; libraries: the loader (runtime/gemma4_model.py) hands us ONE context block of
; absolute addresses and we never look anything up again.
;
; Everything computed here is a transcription of
; transformers/models/gemma4/modeling_gemma4.py (read, not guessed):
;
;   x      = embed[token] * sqrt(hidden)
;   ple_in = ( RMSNorm_256(proj(x) * 1/sqrt(hidden)) + ple_emb[token][l] * sqrt(ple_dim) ) / sqrt(2)
;   for layer in 0..34:                                     (Gemma4TextDecoderLayer)
;       r = x;  x = input_layernorm(x)
;       q = q_norm(q_proj(x)); q = rope(q)
;       own-KV layers:  k = k_norm(k_proj(x)); k = rope(k); v = v_norm(v_proj(x))
;       shared layers:  (k, v) = shared_kv_states[layer_type]   (no k/v weights exist)
;       a = softmax(q . k over the window) . v      (attention scaling == 1.0)
;       x = post_attention_layernorm(o_proj(a)); x = r + x
;       r = x; x = post_feedforward_layernorm(down(geglu(pre_feedforward_layernorm(x))))
;       x = r + x
;       r = x; g = gelu_tanh(per_layer_input_gate(x)); g = g * ple_in[l]
;              x = r + post_per_layer_input_norm(per_layer_projection(g))
;       x = x * layer_scalar
;   x = norm(x); logits = x . embed^T (tied); next = argmax(logits)
;
; A tanh logit softcap would be applied to the logits but is monotone, so argmax
; is unchanged and it is skipped (verified in the kernel tests).
;
; CONTEXT LAYOUT - the only ABI between Python and assembly. The loader PARSES
; these equates out of this file, so there is exactly one source of truth.
; ==============================================================================

use64

CTR_WBASE            equ 0     ; bf16 weight blob (mmap'd read-only)
CTR_DESC             equ 8     ; per-layer descriptor table (DESC_SIZE/layer)
CTR_NLAYER           equ 16    ; 35
CTR_POS              equ 24    ; position of the token being processed
CTR_TOKEN            equ 32    ; input token id
CTR_VOCAB            equ 40    ; 262144
CTR_HIDDEN           equ 48    ; 1536
CTR_SMP              equ 56    ; spin-worker state block
CTR_NTHR             equ 64    ; participants incl. master (4)
CTR_EMB              equ 72    ; embed_tokens            [vocab, hidden] bf16
CTR_PLE_EMB          equ 80    ; embed_tokens_per_layer  [vocab, nlayer*ple_dim] bf16
CTR_PLE_PROJ         equ 88    ; per_layer_model_projection [nlayer*ple_dim, hidden] bf16
CTR_PLE_PROJ_NORM    equ 96    ; per_layer_projection_norm [ple_dim] fp32
CTR_FINAL_NORM       equ 104   ; language_model.norm.weight [hidden] fp32
CTR_ONES             equ 112   ; fp32[512] of 1.0 - scale-free RMSNorm (v_norm)
CTR_PLE_DIM          equ 120   ; 256
CTR_SOFTCAP          equ 128   ; float final_logit_softcapping (30.0)
CTR_PROJ_SCALE       equ 132   ; float 1/sqrt(hidden)
CTR_INV_SQRT2        equ 136   ; float 1/sqrt(2)
CTR_EMB_SCALE        equ 140   ; float sqrt(hidden)
CTR_PLE_TOK_SCALE    equ 144   ; float sqrt(ple_dim) = 16
CTR_RMS_EPS          equ 148   ; float 1e-6
CTR_WINDOW           equ 152   ; int64 sliding_window (512)
CTR_NSHARE           equ 160   ; int64 first layer that shares KV (15)
CTR_NHEAD            equ 168   ; 8
CTR_BUF_X            equ 176   ; [hidden]        residual stream
CTR_BUF_Q            equ 184   ; [nhead*max_headdim]
CTR_BUF_K            equ 192   ; [max_headdim]
CTR_BUF_V            equ 200   ; [max_headdim]
CTR_BUF_A            equ 208   ; [nhead*max_headdim] attention output
CTR_BUF_SCORE        equ 216   ; [max_seq] attention scores
CTR_BUF_G            equ 224   ; [max_inter] MLP gate
CTR_BUF_U            equ 232   ; [max_inter] MLP up
CTR_BUF_M            equ 240   ; [max_inter] MLP act
CTR_BUF_PLE_CUR      equ 248   ; [nlayer*ple_dim] running PLE state
CTR_BUF_PLE_NEXT     equ 256   ; [nlayer*ple_dim]
CTR_BUF_PLE_IN       equ 264   ; [nlayer*ple_dim] per-layer PLE input, this token
CTR_BUF_LOGITS       equ 272   ; [vocab]
CTR_BUF_T1           equ 280   ; [hidden]
CTR_BUF_T2           equ 288   ; [hidden]
CTR_BUF_T3           equ 296   ; [hidden]
CTR_BUF_TMP256       equ 304   ; [ple_dim]
CTR_COS_SLIDE        equ 312   ; fp32[max_seq, slide_headdim/2]
CTR_SIN_SLIDE        equ 320
CTR_COS_FULL         equ 328   ; fp32[max_seq, full_headdim/2]
CTR_SIN_FULL         equ 336
CTR_ROPE_STR_SLIDE   equ 344   ; bytes per position for the sliding tables
CTR_ROPE_STR_FULL    equ 352   ; bytes per position for the full tables
CTR_LOGITS_DONE      equ 360   ; int64: 1 while logits are valid (debug aid)
CTR_SIZE             equ 384

; Per-layer descriptor. All pointers absolute; filled by the loader from the
; converted manifest, so the assembly contains no weight-layout knowledge.
D_QW                 equ 0
D_KW                 equ 8     ; 0 on KV-shared layers (no k_proj in the checkpoint)
D_VW                 equ 16    ; 0 on KV-shared layers
D_OW                 equ 24
D_QNORM              equ 32
D_KNORM              equ 40    ; 0 on KV-shared layers
D_NIN                equ 48    ; input_layernorm
D_NPOSTATT           equ 56    ; post_attention_layernorm
D_NPREFF             equ 64    ; pre_feedforward_layernorm
D_NPOSTFF            equ 72    ; post_feedforward_layernorm
D_PLEGATE            equ 80    ; per_layer_input_gate   [ple_dim, hidden]
D_PLEPROJ            equ 88    ; per_layer_projection   [hidden, ple_dim]
D_PLENORM            equ 96    ; post_per_layer_input_norm [hidden]
D_SCALAR             equ 104   ; layer_scalar (fp32, 1.0 unless trained otherwise)
D_KV                 equ 112   ; K rows used by this layer (own cache, or shared)
D_VV                 equ 120   ; V rows used by this layer
D_INTER              equ 128   ; intermediate_size (6144, or 12288 when double-wide)
D_HEADDIM            equ 136   ; 256 sliding / 512 full
D_KVSTRIDE           equ 144   ; bytes between KV rows = head_dim*4
D_FLAGS              equ 152   ; FLAG_*
DESC_SIZE            equ 192

FLAG_FULL            equ 1     ; full attention (512-wide heads, proportional RoPE)
FLAG_KVSHARED        equ 2     ; reads another layer's K/V
FLAG_KVSTORE         equ 4     ; publishes its K/V as the shared state for its type

include '../raw_materials/mat_bf16_gemb_avx2_flat.asm'
include '../raw_materials/mat_gemma4_kernels_avx2_flat.asm'
include '../sub_assemblies/sub_gemma4_layer_flat.asm'

; ------------------------------------------------------------------------------
; macro: fp32 out[M] = bf16 W(M x K) * fp32 x[K], spread over the worker pool.
; Expanded inline at each call site; arguments are operands, so register
; pressure is resolved by the assembler, not by hand.
; ------------------------------------------------------------------------------
macro GEMB out, w, x, k, m end
macro GEMB out, w, x, k, m
{
    mov     rcx, out
    mov     rdx, w
    mov     r8, x
    mov     r9, k
    mov     rax, m
    mov     [rsp + 32], rax                  ; M
    mov     qword [rsp + 40], 1              ; B = 1 (single-token decode)
    mov     rax, [rbx + CTR_SMP]
    mov     [rsp + 48], rax                  ; worker pool state
    call    smp_bf16_gemb_avx2
}

; ------------------------------ forward step ----------------------------------
; gemma4_step(ctx) -> rax = argmax token id for the next position.
; Reads CTR_TOKEN/CTR_POS, writes the KV rows for CTR_POS and CTR_BUF_LOGITS.
; The loader bumps CTR_POS/CTR_TOKEN; the engine owns no sequence state.
gemma4_step:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 136                         ; 16-byte aligned, room for stack args
    mov     rbx, rcx

    ; ---- 1) token embedding: x = embed[token] * sqrt(hidden) ----------------
    mov     rcx, [rbx + CTR_BUF_X]
    mov     rdx, [rbx + CTR_EMB]
    mov     r8, [rbx + CTR_TOKEN]
    mov     r9, [rbx + CTR_HIDDEN]
    mov     eax, [rbx + CTR_EMB_SCALE]       ; the float itself, not a pointer
    mov     [rsp + 32], rax
    call    embed_bf16_avx2

    ; ---- 2) PLE inputs for this token (HF's `per_layer_inputs`) -------------
    ; proj = per_layer_model_projection(x) * 1/sqrt(hidden)  -> [nlayer*ple_dim]
    mov     r13, [rbx + CTR_NLAYER]
    mov     r14, [rbx + CTR_PLE_DIM]
    mov     r15, r13
    imul    r15, r14                         ; M = 35 * 256 = 8960
    GEMB    <[rbx + CTR_BUF_PLE_NEXT]>, <[rbx + CTR_PLE_PROJ]>, <[rbx + CTR_BUF_X]>, <[rbx + CTR_HIDDEN]>, <r15>

    ; per layer: ( RMSNorm(proj_slice) + ple_emb[token][slice] * sqrt(ple_dim) ) / sqrt(2)
    mov     r12, 0                           ; layer index
.step_ple:
    cmp     r12, r13
    jae     .step_layers
    mov     rcx, r15                         ; r15 = M (elements) - reuse as byte math below
    mov     rax, r12
    imul    rax, r14                         ; slice offset in elements
    mov     rb... <PLACEHOLDER>              ; (see sub_gemma4_layer_flat for the real code)
    inc     r12
    jmp     .step_ple

.step_layers:
    mov     r12, 0
.step_layer_loop:
    cmp     r12, r13
    jae     .step_head
    mov     rbp, r12
    mov     r15, r12
    imul    r15, DESC_SIZE
    add     r15, [rbx + CTR_DESC]            ; descriptor for this layer
    call    gemma4_layer
    mov     r12, rbp
    inc     r12
    jmp     .step_layer_loop

.step_head:
    ; x -> t1 = norm(x); logits = embed * t1; next = argmax(logits)
    mov     rcx, [rbx + CTR_BUF_T1]
    mov     rdx, [rbx + CTR_BUF_X]
    mov     r8, [rbx + CTR_FINAL_NORM]
    mov     r9, [rbx + CTR_HIDDEN]
    mov     eax, [rbx + CTR_RMS_EPS]
    mov     [rsp + 32], rax
    call    rmsnorm_avx2

    GEMB    <[rbx + CTR_BUF_LOGITS]>, <[rbx + CTR_EMB]>, <[rbx + CTR_BUF_T1]>, <[rbx + CTR_HIDDEN]>, <[rbx + CTR_VOCAB]>

    mov     rcx, [rbx + CTR_BUF_LOGITS]
    mov     rdx, [rbx + CTR_VOCAB]
    call    sampler_argmax_avx2
    add     rsp, 136
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret
