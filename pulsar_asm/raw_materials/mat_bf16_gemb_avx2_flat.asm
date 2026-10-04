; ==============================================================================
; Project PULSAR-ASM | Raw Material 6-BF16: mat_bf16_gemb_avx2_flat.asm
; ------------------------------------------------------------------------------
; Multi-Right-Hand-Side BF16 GEMM Kernel (AVX2 + FMA3):  out = W(bf16) x X
;
; Why bf16 instead of f16 (Gemma 2 -> Gemma 4):
;   The Gemma 4 checkpoint ships bfloat16. bf16 -> fp32 is not an F16C convert,
;   it is a 16-bit left shift, because the bf16 pattern IS an fp32 with the low
;   mantissa bits absent:
;       vpmovzxwd ymm, [w]     ; 8 x bf16 -> 8 x uint32 (zero-extend)
;       vpslld    ymm, 16      ;          -> 8 x fp32   (exact, 1 cycle)
;   Same 2 bytes/weight of DRAM traffic as f16, LESS decode ALU than F16C, and
;   the full 8-bit exponent range (no fp16 underflow on small activations).
;   All arithmetic stays FP32.
;
; The "B" (right-hand sides) is the heart of the MTP design: a speculative verify
;   pass pushes B = 1 + draft_len positions through ONE weight stream. DRAM
;   traffic per step is unchanged while FMA utilisation rises B-fold - which is
;   how you spend the ~90 % of the ALU that plain decode leaves idle.
;   B == 1 dispatches to a dedicated branch-free GEMV path (autoregressive decode).
;
; Internal ABI (identical on Win64 and SysV; only the Python boundary is bridged
; by runtime/pulsar_abi.py):
;   RCX = out  (float* [B * M],  out[b*M + m])
;   RDX = W    (uint16_t* [M * K], row-major bf16)
;   R8  = x    (float* [B * K],  row-major, x[b*K + k])
;   R9  = K    (uint64_t)
;   [RSP + 32] = M (uint64_t)   ; caller stores arg5 here before CALL
;   [RSP + 40] = B (uint64_t)   ; caller stores arg6 here before CALL  (>= 1)
;   [RSP + 48] = out_stride     ; bytes between out[b] and out[b+1]; 0 => M*4.
;                A row-partitioned worker MUST pass the GLOBAL M*4, otherwise
;                batch b >= 1 lands on the wrong output rows.
;
; Zero CRT, Zero intrinsics, 100% native machine code.
; ==============================================================================

use64

; ==============================================================================
; Exported: bf16_gemb_avx2   (general B >= 1)
; ==============================================================================
bf16_gemb_avx2:
    push    rbx
    push    rsi
    push    rdi
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 64                    ; 8 pushes + 64 = 128 bytes of frame shift
    ; arg5 (M) is therefore at [rsp + 128 + 40] = [rsp + 168]
    ; arg6 (B) is therefore at [rsp + 128 + 48] = [rsp + 176]

    mov     r10, [rsp + 176]           ; B
    mov     r11, [rsp + 168]           ; M
    test    r10, r10
    jnz     .l_b_ok
    mov     r10, 1                     ; B == 0 is treated as 1
.l_b_ok:
    test    r11, r11
    jz      .l_all_done
    test    r9, r9
    jz      .l_all_done

    cmp     r10, 1
    je      .l_single_rhs

    ; --------------------------------------------------------------------------
    ; Multi-RHS path.  Strides:
    ;   W row  = K*2 bytes   (streamed once per row, reused by all B passes)
    ;   x  rhs = K*4 bytes
    ;   out rhs= M*4 bytes
    ; A row is at most 24 KB (K = 12288 in the double-wide MLP), so passes 1..B-1
    ; hit L1/L2: the weight stream still crosses DRAM exactly once per step.
    ;
    ; Registers here: rcx=out base  r8=x base  r9=K  r12=W row  r13=m  r14=b
    ;                 rsi=&out[b][m]  rbp=&x[b][0]  rbx=k  r15=temp
    ; --------------------------------------------------------------------------
    mov     [rsp + 0],  r11            ; M
    mov     [rsp + 8],  r10            ; B
    mov     rax, r9
    shl     rax, 2
    mov     [rsp + 16], rax            ; x rhs stride = K*4
    mov     rax, [rsp + 184]           ; arg7: explicit out stride (0 => M*4)
    test    rax, rax
    jnz     .l_have_ostep
    mov     rax, r11
    shl     rax, 2
.l_have_ostep:
    mov     [rsp + 24], rax            ; out rhs stride
    mov     r12, rdx                   ; W row base
    xor     r13, r13                   ; m = 0

.l_row_loop:
    prefetchnta [r12 + r9 * 2]         ; start of the NEXT row
    mov     rsi, rcx
    mov     rax, r13
    shl     rax, 2
    add     rsi, rax                   ; &out[0*M + m]
    mov     rbp, r8                    ; &x[0*K]
    xor     r14, r14                   ; b = 0

.l_rhs_loop:
    vxorps  ymm0, ymm0, ymm0
    vxorps  ymm1, ymm1, ymm1
    vxorps  ymm2, ymm2, ymm2
    vxorps  ymm3, ymm3, ymm3
    xor     rbx, rbx                   ; k = 0

.l_b_col32:
    lea     r15, [rbx + 32]
    cmp     r15, r9
    ja      .l_b_col8

    vpmovzxwd   ymm5, [r12 + rbx * 2]
    vpmovzxwd   ymm6, [r12 + rbx * 2 + 16]
    vpmovzxwd   ymm7, [r12 + rbx * 2 + 32]
    vpmovzxwd   ymm8, [r12 + rbx * 2 + 48]
    vpslld      ymm5, ymm5, 16
    vpslld      ymm6, ymm6, 16
    vpslld      ymm7, ymm7, 16
    vpslld      ymm8, ymm8, 16
    vfmadd231ps ymm0, ymm5, [rbp + rbx * 4]
    vfmadd231ps ymm1, ymm6, [rbp + rbx * 4 + 32]
    vfmadd231ps ymm2, ymm7, [rbp + rbx * 4 + 64]
    vfmadd231ps ymm3, ymm8, [rbp + rbx * 4 + 96]

    add     rbx, 32
    jmp     .l_b_col32

.l_b_col8:
    lea     r15, [rbx + 8]
    cmp     r15, r9
    ja      .l_b_reduce
    vpmovzxwd   ymm5, [r12 + rbx * 2]
    vpslld      ymm5, ymm5, 16
    vfmadd231ps ymm0, ymm5, [rbp + rbx * 4]
    add     rbx, 8
    jmp     .l_b_col8

.l_b_reduce:
    vaddps  ymm0, ymm0, ymm1
    vaddps  ymm2, ymm2, ymm3
    vaddps  ymm0, ymm0, ymm2
    vextractf128 xmm1, ymm0, 1
    vaddps  xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovss  dword [rsi], xmm0

    mov     rax, [rsp + 24]
    add     rsi, rax                   ; next output slot
    mov     rax, [rsp + 16]
    add     rbp, rax                   ; next input row
    inc     r14
    cmp     r14, [rsp + 8]
    jb      .l_rhs_loop

    mov     rax, r9
    shl     rax, 1
    add     r12, rax                   ; next weight row
    inc     r13
    cmp     r13, [rsp + 0]
    jb      .l_row_loop
    jmp     .l_all_done

; ==============================================================================
; Fast path B == 1 : tight single-RHS GEMV, 4 independent accumulator chains
; ==============================================================================
.l_single_rhs:
    mov     [rsp + 40], r11            ; M
    xor     r10d, r10d                 ; m = 0
.l_g_row:
    prefetchnta [rdx + r9 * 2]
    vxorps  ymm0, ymm0, ymm0
    vxorps  ymm1, ymm1, ymm1
    vxorps  ymm2, ymm2, ymm2
    vxorps  ymm3, ymm3, ymm3
    xor     r11, r11                   ; k = 0
.l_g_col32:
    lea     rax, [r11 + 32]
    cmp     rax, r9
    ja      .l_g_col8
    prefetchnta [rdx + r11 * 2 + 512]

    vpmovzxwd   ymm5, [rdx + r11 * 2]
    vpmovzxwd   ymm6, [rdx + r11 * 2 + 16]
    vpmovzxwd   ymm7, [rdx + r11 * 2 + 32]
    vpmovzxwd   ymm8, [rdx + r11 * 2 + 48]
    vpslld      ymm5, ymm5, 16
    vpslld      ymm6, ymm6, 16
    vpslld      ymm7, ymm7, 16
    vpslld      ymm8, ymm8, 16
    vfmadd231ps ymm0, ymm5, [r8 + r11 * 4]
    vfmadd231ps ymm1, ymm6, [r8 + r11 * 4 + 32]
    vfmadd231ps ymm2, ymm7, [r8 + r11 * 4 + 64]
    vfmadd231ps ymm3, ymm8, [r8 + r11 * 4 + 96]

    add     r11, 32
    jmp     .l_g_col32

.l_g_col8:
    lea     rax, [r11 + 8]
    cmp     rax, r9
    ja      .l_g_reduce
    vpmovzxwd   ymm5, [rdx + r11 * 2]
    vpslld      ymm5, ymm5, 16
    vfmadd231ps ymm0, ymm5, [r8 + r11 * 4]
    add     r11, 8
    jmp     .l_g_col8

.l_g_reduce:
    vaddps  ymm0, ymm0, ymm1
    vaddps  ymm2, ymm2, ymm3
    vaddps  ymm0, ymm0, ymm2
    vextractf128 xmm1, ymm0, 1
    vaddps  xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vmovss  dword [rcx + r10 * 4], xmm0

    lea     rdx, [rdx + r9 * 2]
    inc     r10
    cmp     r10, [rsp + 40]
    jb      .l_g_row

.l_all_done:
    add     rsp, 64
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rdi
    pop     rsi
    pop     rbx
    vzeroupper
    ret

; ==============================================================================
; Exported: smp_bf16_gemb_avx2 -- rows partitioned over the MESI spin-worker pool
;
;   RCX = out, RDX = W, R8 = x, R9 = K,
;   [RSP+32] = M, [RSP+40] = B, [RSP+48] = smp_state (0 => single core)
;
; PULSAR_SmpState:
;   +0   uint32 done_counter      +4  uint32 job_seq       +8  uint32 stop_flag
;   +12  pad                      +16 uint64 n_participants
;   +64  worker ctx #1  (128 B)   +192 worker ctx #2       +320 worker ctx #3
; Worker ctx:
;   +0  uint32 job_id     +4 uint32 last_job_id  +8 uint32 stop_flag  +12 thread_id
;   +16 float* out        +24 uint16* W          +32 float* x
;   +40 uint64 K          +48 uint64 M           +56 uint32* done_counter
;   +64 uint64 B          +72 void*  kernel      +80 uint64 out_stride
; ==============================================================================
smp_bf16_gemb_avx2:
    push    rbx
    push    rsi
    push    rdi
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 64
    ; arg5 M -> [rsp+168], arg6 B -> [rsp+176], arg7 smp_state -> [rsp+184]
    mov     r10, [rsp + 184]           ; smp_state
    mov     r11, [rsp + 168]           ; M
    mov     rbp, [rsp + 176]           ; B
    test    rbp, rbp
    jnz     .l_smp_b_ok
    mov     rbp, 1
.l_smp_b_ok:
    test    r10, r10
    jz      .l_smp_single
    cmp     r11, 256
    jb      .l_smp_single              ; too few rows to pay for the handshake

    mov     rbx, r10                   ; smp
    mov     r12, r11                   ; M
    mov     r13, rcx                   ; out
    mov     r14, rdx                   ; W
    mov     [rsp + 8], r8              ; x base
    mov     [rsp + 56], r9             ; K

    mov     dword [rbx + 0], 0         ; done_counter = 0
    inc     dword [rbx + 4]            ; job_seq++
    mov     eax, [rbx + 4]
    mov     [rsp + 40], rax            ; park job_seq in a SLOT: rax/rax's aliases
                                       ; get reused for row math in the loop below

    mov     rax, r12
    shr     rax, 2
    mov     [rsp + 16], rax            ; chunk = M / 4
    mov     rax, r9
    shl     rax, 1
    mov     [rsp + 24], rax            ; W row stride (bytes)

    ; Runtime address of the kernel. In a flat, non-relocated binary `mov r8, label`
    ; would embed the LINK-TIME address (the module offset) and send workers to
    ; NULL, so we take our own instruction pointer with call/pop and subtract the
    ; assembler-known delta - the canonical position-independent anchor.
    call    .l_anchor
.l_anchor:
    pop     rax
    sub     rax, (.l_anchor - bf16_gemb_avx2)
    mov     [rsp + 0], rax             ; kernel runtime address

    mov     r15, 1                     ; worker index w = 1..3
.l_smp_setup:
    mov     rcx, r15
    shl     rcx, 7                     ; w * 128
    add     rcx, rbx
    add     rcx, 64                    ; worker ctx base

    ; rows [w*chunk .. min((w+1)*chunk, M)) -- worker 3 absorbs the ragged tail
    mov     rax, [rsp + 16]
    imul    rax, r15
    mov     rdx, rax                   ; RDX = row0
    mov     rsi, rax
    add     rsi, [rsp + 16]            ; row0 + chunk
    cmp     rsi, r12
    jbe     .l_smp_clamp
    mov     rsi, r12
.l_smp_clamp:
    mov     rdi, rsi
    sub     rdi, rdx                   ; rows_here

    mov     r8, rdx
    shl     r8, 2
    add     r8, r13
    mov     [rcx + 16], r8             ; out  = out + row0*4

    mov     r8, [rsp + 24]
    imul    r8, rdx
    add     r8, r14
    mov     [rcx + 24], r8             ; W    = W + row0*row_stride

    mov     r8, [rsp + 8]
    mov     [rcx + 32], r8             ; x
    mov     r8, [rsp + 56]
    mov     [rcx + 40], r8             ; K
    mov     [rcx + 48], rdi            ; M = rows_here
    lea     r8, [rbx + 0]
    mov     [rcx + 56], r8             ; &done_counter
    mov     [rcx + 64], rbp            ; B
    mov     r11, r12                   ; GLOBAL M: batch stride must be global.
    shl     r11, 2                     ; (r11, NOT rax - rax still holds job_seq!)
    mov     [rcx + 80], r11            ; out_stride
    mov     r8, [rsp + 0]
    mov     [rcx + 72], r8             ; kernel (runtime address, see anchor above)
    ; Publish LAST, and from the SLOT, not from a register this loop recomputes.
    ; This used to be 'mov [rcx+0], eax' where eax still held row0 (= w*M/4, so
    ; 64/128/192 for M=256): the first dispatch woke the workers because 64 != 0,
    ; every LATER dispatch republished the same 64, matched last_job_id, and the
    ; master spun forever in .l_smp_wait. Tests that varied M per call hid it -
    ; a different M made row0 differ, so the workers still woke up.
    mov     eax, [rsp + 40]
    mov     [rcx + 0], eax             ; job_id = job_seq  -> MESI invalidation

    inc     r15
    cmp     r15, 3
    jbe     .l_smp_setup

    ; ---- master core executes rows [0 .. chunk) -------------------------------
    mov     rcx, r13
    mov     rdx, r14
    mov     r8,  [rsp + 8]
    mov     r9,  [rsp + 56]
    mov     rax, [rsp + 16]
    mov     [rsp + 32], rax            ; M = chunk
    mov     [rsp + 40], rbp            ; B
    mov     rax, r12
    shl     rax, 2
    mov     [rsp + 48], rax            ; out_stride = M_global*4
    call    bf16_gemb_avx2

.l_smp_wait:
    mov     eax, [rbx + 0]
    cmp     eax, 3
    jae     .l_smp_done
    pause
    jmp     .l_smp_wait

.l_smp_done:
    add     rsp, 64
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rdi
    pop     rsi
    pop     rbx
    ret

.l_smp_single:
    mov     [rsp + 32], r11
    mov     [rsp + 40], rbp
    mov     qword [rsp + 48], 0        ; out_stride => default M*4
    call    bf16_gemb_avx2
    add     rsp, 64
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rdi
    pop     rsi
    pop     rbx
    ret

; ==============================================================================
; Exported: smp_worker_proc4 -- generic MESI spin worker for the Gemma 4 engine
; ==============================================================================
smp_worker_proc4:
    push    rbx
    push    rsi
    push    rdi
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 64

    mov     r15, rcx                   ; WorkerContext*

.w4_spin:
    mov     eax, [r15 + 0]
    cmp     eax, [r15 + 4]
    jne     .w4_work

    mov     edx, [r15 + 8]
    test    edx, edx
    jnz     .w4_exit

    pause
    jmp     .w4_spin

.w4_work:
    mov     [r15 + 4], eax
    mov     rcx, [r15 + 16]            ; out
    mov     rdx, [r15 + 24]            ; W
    mov     r8,  [r15 + 32]            ; x
    mov     r9,  [r15 + 40]            ; K
    mov     rax, [r15 + 48]            ; M (rows for this worker)
    mov     [rsp + 32], rax
    mov     rax, [r15 + 64]            ; B
    mov     [rsp + 40], rax
    mov     rax, [r15 + 80]            ; out_stride (global M*4)
    mov     [rsp + 48], rax
    mov     rax, [r15 + 72]            ; kernel
    call    rax
    mov     r11, [r15 + 56]
    lock inc dword [r11]
    jmp     .w4_spin

.w4_exit:
    add     rsp, 64
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rdi
    pop     rsi
    pop     rbx
    xor     eax, eax
    ret

; ------------------------------------------------------------------------------
; Export directory (trailer). v1.0 located entry points by scanning the loaded
; image for magic byte sequences (fragile: a `xor eax,eax; ret` can appear in the
; middle of unrelated code). Every module now publishes its own offset table:
;
;   [ ...code... ][ name strings ][ dd off*N ][ dd nameoff*N ][ dd N ][ 'PLSE' ]
;
; runtime/pulsar_abi.load_module() reads the last 8 bytes and resolves names,
; so entry points are explicit, checked, and identical on Linux and Windows.
; ------------------------------------------------------------------------------
    align   4
plsar_n0: db 'bf16_gemb_avx2', 0
plsar_n1: db 'smp_bf16_gemb_avx2', 0
plsar_n2: db 'smp_worker_proc4', 0
    align   4
    dd      bf16_gemb_avx2
    dd      smp_bf16_gemb_avx2
    dd      smp_worker_proc4
plsar_names:
    dd      plsar_n0
    dd      plsar_n1
    dd      plsar_n2
    dd      3
    db      'PLSE'
