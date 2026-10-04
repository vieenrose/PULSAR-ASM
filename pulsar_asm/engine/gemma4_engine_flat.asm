; ==============================================================================
; Project PULSAR-ASM | Gemma 4 E2B text engine  (engine/gemma4_engine_flat.asm)
; ------------------------------------------------------------------------------
; Pure x86-64 forward pass for google/gemma-4-E2B-it: text-only, CPU, fp32
; activations over bf16 weights, AVX2 + FMA3, 4 cores. No OS calls, no libc, no
; libraries: the loader (runtime/gemma4_model.py) hands us ONE context block of
; absolute addresses and we never look anything up again.
;
; The math is a transcription of transformers/models/gemma4/modeling_gemma4.py
; (read from the installed reference, not guessed):
;
;   x      = embed[token] * sqrt(hidden)
;   ple_in = ( RMSNorm_256(proj(x) * 1/sqrt(hidden)) + ple_emb[token][l] * sqrt(ple_dim) ) / sqrt(2)
;   for layer in 0..34:                            (Gemma4TextDecoderLayer)
;       r = x;  x = input_layernorm(x)
;       q = q_norm(q_proj(x)); q = rope(q)
;       own-KV layers:  k = k_norm(k_proj(x)); k = rope(k); v = v_norm(v_proj(x))
;       shared layers:  (k, v) = shared_kv_states[layer_type]   (no k/v weights exist)
;       a = softmax(q . k over the window) . v     (attention scaling == 1.0)
;       x = post_attention_layernorm(o_proj(a)); x = r + x
;       r = x; x = post_feedforward_layernorm(down(geglu(pre_feedforward_layernorm(x))))
;       x = r + x
;       r = x; g = gelu_tanh(per_layer_input_gate(x)); g *= ple_in[l]
;              x = r + post_per_layer_input_norm(per_layer_projection(g))
;       x = x * layer_scalar
;   x = norm(x); logits = x . embed^T (tied weights); next = argmax(logits)
;
; The tanh logit softcap is applied by the reference but is monotone, so argmax
; is identical without it (asserted in tests/test_gemma4_kernels.py).
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
CTR_ONES             equ 112   ; fp32[512] of 1.0 - unused: v_norm uses rmsnorm_scale
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
CTR_BUF_K            equ 192   ; [max_headdim] (unused: K is GEMV'd into its cache row)
CTR_BUF_V            equ 200   ; [max_headdim] (unused, same reason)
CTR_BUF_A            equ 208   ; [nhead*max_headdim] attention output
CTR_BUF_SCORE        equ 216   ; [max_seq] attention scores
CTR_BUF_G            equ 224   ; [max_inter] MLP gate
CTR_BUF_U            equ 232   ; [max_inter] MLP up
CTR_BUF_M            equ 240   ; [max_inter] MLP act (unused: geglu writes in place)
CTR_BUF_PLE_CUR      equ 248   ; [nlayer*ple_dim] running PLE state (unused)
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
CTR_ROPE_STR_SLIDE   equ 344   ; bytes per position, sliding tables
CTR_ROPE_STR_FULL    equ 352   ; bytes per position, full tables
CTR_MARK             equ 360   ; int64 progress breadcrumb, debug only
CTR_B                equ 368   ; int64 tokens per pass: 1 = decode, >1 = verify
CTR_TOKENS           equ 376   ; int64* -> token id per row; TOKENS[0] == CTR_TOKEN
CTR_OUTTOK           equ 384   ; int64* -> argmax per row, written by the head
CTR_XSTRIDE          equ 392   ; bytes between activation rows = hidden*4
CTR_PLEROW           equ 400   ; bytes between PLE rows = nlayer*ple_dim*4
                                ; FASM lets two equates share a value silently:
                                ; MAXLAYER briefly aliased B here, so a debug
                                ; write to one changed the other. parse_abi now
                                ; rejects duplicate offsets in these tables.
CTR_MAXLAYER         equ 408   ; int64: run layers [0,MAXLAYER); = NLAYER normally.
                                ; Separate from NLAYER because that also sizes
                                ; the PLE row stride, so truncating it corrupts
                                ; the per-layer embedding inputs.
CTR_SIZE             equ 416

; Per-layer descriptor: absolute pointers, built by the loader from the converted
; manifest, so the assembly contains no weight-layout knowledge.
D_QW                 equ 0
D_KW                 equ 8     ; 0 on KV-shared layers (no k_proj in the checkpoint)
D_VW                 equ 16    ; 0 on KV-shared layers
D_OW                 equ 24
D_GATE               equ 32    ; mlp.gate_proj
D_UP                 equ 40    ; mlp.up_proj
D_DOWN               equ 48    ; mlp.down_proj
D_QNORM              equ 56
D_KNORM              equ 64    ; 0 on KV-shared layers
D_NIN                equ 72    ; input_layernorm
D_NPOSTATT           equ 80    ; post_attention_layernorm
D_NPREFF             equ 88    ; pre_feedforward_layernorm
D_NPOSTFF            equ 96    ; post_feedforward_layernorm
D_PLEGATE            equ 104   ; per_layer_input_gate   [ple_dim, hidden]
D_PLEPROJ            equ 112   ; per_layer_projection   [hidden, ple_dim]
D_PLENORM            equ 120   ; post_per_layer_input_norm [hidden]
D_SCALAR             equ 128   ; layer_scalar (fp32)
D_KV                 equ 136   ; K rows used by this layer (own cache, or shared)
D_VV                 equ 144   ; V rows used by this layer
D_INTER              equ 152   ; intermediate_size (6144, or 12288 when double-wide)
D_HEADDIM            equ 160   ; 256 sliding / 512 full
D_KVSTRIDE           equ 168   ; bytes between KV rows = head_dim*4
D_FLAGS              equ 176   ; FLAG_*
DESC_SIZE            equ 192

FLAG_FULL            equ 1     ; full attention (512-wide heads, proportional RoPE)
FLAG_KVSHARED        equ 2     ; reads another layer's K/V
FLAG_KVSTORE         equ 4     ; publishes its K/V as the shared state of its type

; ------------------------------------------------------------------------------
; fp32 out[M] = bf16 W(M x K) * fp32 x[K] across the worker pool. Written as a
; macro so operands resolve where they are used; operands may use rbx/r12-r15
; but must not depend on rcx/rdx/r8/r9/rax (the macro fills those in order).
; ------------------------------------------------------------------------------
; NOTE: the parameters must not be named out/w/x - FASM parses those as the OUT
; instruction or as expressions, and reports it as "illegal instruction".
; Batched GEMM. Rows are the output dimension (M) and the batch is the second
; dimension, so outputs land as [B][M] with the default stride M*4 - which is
; the entire point: the weights are streamed once for all B tokens, and decode
; is bound by streaming them.
macro GEMMB dst, wp, xp, kk, mm, bb {
    mov     rcx, dst
    mov     rdx, wp
    mov     r8, xp
    mov     r9, kk
    mov     rax, mm
    mov     [rsp + 32], rax                   ; M
    mov     rax, bb
    mov     [rsp + 40], rax                   ; B
    mov     rax, [rbx + CTR_SMP]
    mov     [rsp + 48], rax                   ; worker pool state
    call    smp_bf16_gemb_avx2
}

macro GEMB dst, wp, xp, kk, mm {
    mov     rcx, dst
    mov     rdx, wp
    mov     r8, xp
    mov     r9, kk
    mov     rax, mm
    mov     [rsp + 32], rax                   ; M
    mov     qword [rsp + 40], 1               ; B = 1 (single-token decode)
    mov     rax, [rbx + CTR_SMP]
    mov     [rsp + 48], rax                   ; worker pool state
    call    smp_bf16_gemb_avx2
}

; x += t2 over the residual stream (both residual adds in a layer look like this)
macro GEMB_RESID {
    mov     rcx, [rbx + CTR_BUF_X]
    mov     rdx, [rbx + CTR_BUF_X]
    mov     r8, [rbx + CTR_BUF_T2]
    mov     r9, [rbx + CTR_HIDDEN]
    imul    r9, [rbx + CTR_B]                 ; elementwise: one call covers B rows
    mov     eax, C_ONE
    mov     [rsp + 32], rax
    mov     [rsp + 40], rax                   ; out = x*1 + t2*1
    call    add_scaled_avx2
}

include '../raw_materials/mat_bf16_gemb_avx2_flat.asm'
include '../raw_materials/mat_gemma4_kernels_avx2_flat.asm'
include '../sub_assemblies/sub_gemma4_layer_flat.asm'

; ------------------------------------------------------------------------------
; gemma4_step(ctx) -> rax = argmax token id for the next position.
; Consumes CTR_TOKEN/CTR_POS, writes the KV rows for CTR_POS and the logits.
; Sequence state (token, position) belongs to the loader, not the engine.
; ------------------------------------------------------------------------------
gemma4_step:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 136                          ; 16-byte aligned, room for stack args
    mov     rbx, rcx
    mov     qword [rbx + CTR_MARK], 1

    ; ---- 1) token embedding: x[b] = embed[token[b]] * sqrt(hidden) ----------
    mov     qword [rsp + 128], 0              ; row counter (frame local)
.step_emb:
    mov     rcx, [rsp + 128]
    imul    rcx, [rbx + CTR_XSTRIDE]
    add     rcx, [rbx + CTR_BUF_X]
    mov     rdx, [rbx + CTR_EMB]
    mov     rax, [rsp + 128]
    shl     rax, 3                            ; &tokens[b]
    add     rax, [rbx + CTR_TOKENS]
    mov     r8, [rax]
    mov     r9, [rbx + CTR_HIDDEN]
    mov     eax, [rbx + CTR_EMB_SCALE]        ; the float itself, not a pointer
    mov     [rsp + 32], rax
    call    embed_bf16_avx2
    mov     rax, [rsp + 128]
    inc     rax
    mov     [rsp + 128], rax
    cmp     rax, [rbx + CTR_B]
    jb      .step_emb
    mov     qword [rbx + CTR_MARK], 2

    ; ---- 2) per-layer PLE inputs for this token (HF's per_layer_inputs) -----
    ; context branch: proj = per_layer_model_projection(x) * 1/sqrt(hidden)
    mov     r13, [rbx + CTR_NLAYER]
    mov     r14, [rbx + CTR_PLE_DIM]
    mov     r15, r13
    imul    r15, r14                          ; M = 35 * 256 = 8960
    GEMMB   <[rbx + CTR_BUF_PLE_NEXT]>, <[rbx + CTR_PLE_PROJ]>, <[rbx + CTR_BUF_X]>, <[rbx + CTR_HIDDEN]>, <r15>, <[rbx + CTR_B]>

    ; per layer: ( RMSNorm_256(proj_slice) + ple_emb[token][slice] * sqrt(ple_dim) ) / sqrt(2)
    ; The scale is applied INSIDE the norm, before the RMSNorm weight: eps is
    ; not negligible against (pscale*|proj|)^2/D, so the order is observable.
    mov     qword [rbx + CTR_MARK], 3
    ; NOTE: r13 (NLAYER) also sets the per-layer-embedding ROW STRIDE below, so
    ; it must be the model's real layer count - overriding it to debug "run the
    ; first k layers" silently reads the wrong PLE rows. Truncation belongs in a
    ; separate field if we ever want it.
    mov     qword [rsp + 128], 0              ; outer loop: batch row
.step_plerow:
    mov     rbp, [rsp + 128]
    mov     r12, 0
.step_ple:
    cmp     r12, r13
    jae     .step_ple_done
    mov     rax, r12
    imul    rax, r14                          ; this layer's slice, in ELEMENTS
    mov     r11, rbp
    imul    r11, [rbx + CTR_PLEROW]           ; plus this row's PLE row, in bytes
    mov     rcx, rax
    shl     rcx, 2
    add     rcx, r11
    add     rcx, [rbx + CTR_BUF_PLE_IN]       ; out   = ple_in[b][l]
    mov     rdx, rax
    shl     rdx, 2
    add     rdx, r11
    add     rdx, [rbx + CTR_BUF_PLE_NEXT]     ; proj  = proj[b][l]
    ; token branch = ple_emb + (token*nlayer + layer)*ple_dim, shifted only
    ; AFTER the base is added - shifting the pointer itself doubles the address
    ; and faults where it cannot be explained.
    mov     r10, r13
    imul    r10, r14                          ; nlayer * ple_dim
    mov     r8, rbp
    shl     r8, 3
    add     r8, [rbx + CTR_TOKENS]
    mov     r8, [r8]                          ; this row's token id
    imul    r8, r10
    add     r8, rax                           ; + layer*ple_dim
    shl     r8, 1                             ; bf16 -> bytes
    add     r8, [rbx + CTR_PLE_EMB]
    mov     eax, [rbx + CTR_PROJ_SCALE]
    mov     [rsp + 32], rax
    mov     eax, [rbx + CTR_PLE_TOK_SCALE]
    mov     [rsp + 40], rax
    mov     eax, [rbx + CTR_INV_SQRT2]
    mov     [rsp + 48], rax
    mov     rax, [rbx + CTR_PLE_PROJ_NORM]
    mov     [rsp + 56], rax                   ; per_layer_projection_norm weight
    ; arg4 (D = ple_dim) is set last: everything above needed r9/r10/r11 as
    ; scratch, and leaving D as nlayer*ple_dim tells the kernel a 256-wide slice
    ; is 8960 long, which runs off the end of the buffer.
    mov     r9, r14
    call    ple_combine_avx2
    inc     r12
    jmp     .step_ple
.step_ple_done:
    mov     rax, [rsp + 128]
    inc     rax
    mov     [rsp + 128], rax
    cmp     rax, [rbx + CTR_B]
    jb      .step_plerow

    ; ---- 3) the 35 decoder layers -------------------------------------------
.step_layers:
    mov     qword [rbx + CTR_MARK], 4
    mov     rbp, [rbx + CTR_MAXLAYER]         ; debug bound; NLAYER also sizes the
                                              ; PLE stride so it can't be reused
    mov     r12, 0
.step_layer_loop:
    cmp     r12, rbp
    jae     .step_head
    mov     rax, r12
    mov     r15, r12
    imul    r15, DESC_SIZE
    add     r15, [rbx + CTR_DESC]             ; descriptor for this layer
    call    gemma4_layer                      ; (rbx = ctx, r15 = desc, rax = index)
    mov     rax, 100
    add     rax, r12
    mov     [rbx + CTR_MARK], rax
    inc     r12                               ; r12 survives: the layer preserves it
    jmp     .step_layer_loop

    ; ---- 4) final norm, tied logits head, greedy sample ---------------------
.step_head:
    mov     qword [rbx + CTR_MARK], 5
    mov     qword [rsp + 128], 0
.step_headrow:
    mov     rax, [rsp + 128]
    imul    rax, [rbx + CTR_XSTRIDE]
    mov     rcx, [rbx + CTR_BUF_T1]
    add     rcx, rax
    mov     rdx, [rbx + CTR_FINAL_NORM]
    mov     r8, [rbx + CTR_BUF_X]
    add     r8, rax
    mov     r9, [rbx + CTR_HIDDEN]
    mov     eax, [rbx + CTR_RMS_EPS]
    mov     [rsp + 32], rax
    call    rmsnorm_avx2                      ; t1[b] = norm(x[b])
    mov     rax, [rsp + 128]
    inc     rax
    mov     [rsp + 128], rax
    cmp     rax, [rbx + CTR_B]
    jb      .step_headrow

    ; Tied lm head, batched. This GEMM streams the whole 805 MB embedding table,
    ; so doing it once for B rows instead of B times is most of the bandwidth
    ; that speculative decoding saves; logits land as [B][vocab].
    mov     qword [rbx + CTR_MARK], 6
    GEMMB   <[rbx + CTR_BUF_LOGITS]>, <[rbx + CTR_EMB]>, <[rbx + CTR_BUF_T1]>, <[rbx + CTR_HIDDEN]>, <[rbx + CTR_VOCAB]>, <[rbx + CTR_B]>

    ; One greedy sample per row: rax = argmax per row -> OUTTOK[b]
    mov     rax, [rbx + CTR_VOCAB]
    shl     rax, 2
    mov     [rsp + 120], rax                  ; logits row stride
    mov     qword [rsp + 128], 0
.step_sample:
    mov     rax, [rsp + 128]
    imul    rax, [rsp + 120]
    mov     rcx, [rbx + CTR_BUF_LOGITS]
    add     rcx, rax
    mov     rdx, [rbx + CTR_VOCAB]
    call    sampler_argmax_avx2               ; softcap is monotone, so the head
    mov     rdx, rax                          ; skips it and the answer holds
    mov     rax, [rsp + 128]
    shl     rax, 3
    add     rax, [rbx + CTR_OUTTOK]
    mov     [rax], rdx                        ; outtok[b] = argmax(row b)
    mov     rax, [rsp + 128]
    inc     rax
    mov     [rsp + 128], rax
    cmp     rax, [rbx + CTR_B]
    jb      .step_sample

    mov     rax, [rbx + CTR_OUTTOK]
    mov     rax, [rax]                        ; row 0's token is the return value
    add     rsp, 136
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ==============================================================================
; PLSE export trailer: [offsets * N][name offsets * N][N]['PLSE']. The loader
; adds its own load base, so the engine is position-independent and needs no
; hardcoded entry offsets (v1.0's loader had them baked in).
; ==============================================================================
g4_n0: db 'gemma4_step', 0
g4_n1: db 'smp_worker_proc4', 0
g4_n2: db 'smp_bf16_gemb_avx2', 0
g4_n3: db 'bf16_gemb_avx2', 0
    align 4
    dd gemma4_step, smp_worker_proc4, smp_bf16_gemb_avx2, bf16_gemb_avx2
    dd g4_n0, g4_n1, g4_n2, g4_n3
    dd 4
    db 'PLSE'
