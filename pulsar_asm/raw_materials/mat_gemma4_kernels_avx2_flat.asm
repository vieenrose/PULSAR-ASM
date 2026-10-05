; ==============================================================================
; Project PULSAR-ASM | Raw Material 7: mat_gemma4_kernels_avx2_flat.asm
; ------------------------------------------------------------------------------
; Pointwise + attention vector kernels for the Gemma 4 E2B text engine.
;
; INCLUDED by engine/gemma4_engine_flat.asm (flat composition, like v1.0's
; raw_materials). No OS, no libc, no libraries: AVX2 + FMA3 on caller buffers.
;
; Internal ABI (OS-independent; bridged at the Python boundary by
; runtime/pulsar_abi.make_reg_entry):
;   RCX, RDX, R8, R9 = args 1..4 ; stack args 5..7
;   CALLER side:   store arg5/6/7 to [RSP+32], [RSP+40], [RSP+48] before CALL
;   CALLEE side:   they arrive at [RSP+40], [RSP+48], [RSP+56] (CALL pushed a
;                  return address), which is what every kernel below reads.
;   Verified by probe in tests/test_gemma4_kernels.py (see ABI note in README).
;
; FMA form used throughout (verified empirically, see tests/test_fma_sem.py):
;   vfmadd231ps  dst, src1, src2   ->  dst = src1*src2 + dst     ("acc += a*b")
;   vfnmadd231ps dst, src1, src2   ->  dst = -(src1*src2) + dst
;   vfmadd213ps  dst, src1, src2   ->  dst = src1*dst + src2     (Horner steps)
;   VEX encodes the memory operand as src2, so a [mem] operand must be LAST.
;
; VECTOR REGISTER CONVENTION (all kernels in this file):
;   ymm0..ymm8   scratch / accumulators
;   ymm9..ymm13  EXP constants: log2(e), ln2, +44, -44, 127.0 (bias)
;   ymm14        horizontal-reduce scratch (HADDY / MAXREDUCE)
;   ymm15        1.0f
; Any kernel that runs EXP must leave ymm9..ymm13 alone - softcap once used ymm13
; as a result register and every vector after the first computed exp with a
; corrupted bias (correct lane group 0, saturated everywhere else).
;
; Exports:
;   rmsnorm_avx2        (out, w, x, N, eps)              Gemma 4: weight direct, NO 1+w
;   rmsnorm_scale_avx2  (out, x, N, scale, eps)          scale-free RMSNorm (v_norm)
;   gelu_tanh_avx2      (out, x, N)                     gelu_pytorch_tanh
;   geglu_avx2          (out, gate, up, N)              gelu_tanh(gate) * up
;   add_scaled_avx2     (out, a, b, N, sa, sb)          out = a*sa + b*sb
;   scale_avx2          (buf, N, s)                     in-place
;   zero_avx2           (buf, N)
;   embed_bf16_avx2     (out, emb, token, dim, scale)   row gather bf16 -> fp32*scale
;   rope_apply_avx2     (vec, cos, sin, H)              rotate_half, H = head_dim/2
;   softmax_avx2        (buf, n)
;   attn_scores_avx2    (scores, q, K, kv_len, head_dim, kv_start)
;   attn_values_avx2    (out, V, w, kv_len, head_dim, kv_start)
;   softcap_tanh_avx2   (logits, N, cap)                cap*tanh(x/cap)
;   ple_combine_avx2    (out, proj, tok, D, pscale, tscale, inv_sqrt2)
; ==============================================================================

use64

C_ONE    equ 0x3F800000        ; 1.0f
C_HALF   equ 0x3F000000        ; 0.5f
C_THREE  equ 0x40400000        ; 3.0f
C_TWO    equ 0x40000000        ; 2.0f
C_LOG2E  equ 0x3FB8AA3B        ; log2(e)
C_LN2    equ 0x3F317218        ; ln(2)
C_C1     equ 0x3FCC422A        ; 2*sqrt(2/pi)  = 1.5957691216f
C_C3     equ 0x3D922279        ; 2*sqrt(2/pi)*0.044715 = 0.0713548162f
C_CLAMP  equ 0x42300000        ; 44.0f  (exp argument clamp: exp(+-44) stays finite)
C_127    equ 0x42FE0000        ; 127.0f (IEEE-754 exponent bias; 0x42FC0000 is 126.0f!)

; ------------------------------------------------------------------------------
; Shared prologue: load the exp() helper broadcasts into ymm9..ymm12, ymm15 = 1.
;   ymm9 = log2(e)   ymm10 = ln2   ymm11 = +44   ymm12 = -44   ymm15 = 1.0
; ------------------------------------------------------------------------------
macro EXP_CONSTS
{
    vbroadcastss ymm9, dword [K_LOG2E]
    vbroadcastss ymm10, dword [K_LN2]
    vbroadcastss ymm11, dword [K_CLAMP]          ; +44.0f (exp arg clamp)
    vbroadcastss ymm12, dword [K_NCLAMP]         ; -44.0f
    vbroadcastss ymm15, dword [K_ONE]            ; 1.0f
    vbroadcastss ymm13, dword [K_127]            ; 127.0f: IEEE-754 exponent bias
}

; ------------------------------------------------------------------------------
; exp(z) for z in [-44, 44] into dst, using k = round(z*log2e), r = z - k*ln2,
;   result = P5(r) * 2^k  with P5 the degree-5 Taylor polynomial (|r|<=ln2/2 =>
;   relative error < 1e-7).  Scratch: t1 t2 t3 (vectors), eax.
;   K_P* are 32-byte BROADCAST vectors: vmovaps requires 32-byte alignment, so a
;   plain `dd` here would fault (#GP) on the first lane it tried to load.
; -----------------------------------------------------------------------------
; void copy_avx2(dst, src, N)   fp32 copy        (N a multiple of 8)
;   N is arg3 = R8. Nothing calls this today; had it counted from r9 like its
;   old neighbour mul_avx2, it would have run past the count the same way.
copy_avx2:
    xor     r10, r10
.cp_l:
    lea     rax, [r10 + 8]
    cmp     rax, r8
    ja      .cp_done
    vmovups ymm0, [rdx + r10 * 4]
    vmovups [rcx + r10 * 4], ymm0
    add     r10, 8
    jmp     .cp_l
.cp_done:
    ret

; -----------------------------------------------------------------------------
; void mul_avx2(dst, src, N)   dst[i] *= src[i]        (N a multiple of 8)
; Used by the PLE gate: gate *= per-layer input.
; N is arg3, so it arrives in R8. Reading it from R9 (as this did) is not a
; harmless typo: the caller's preceding GEMMB leaves hidden=1536 in r9, so the
; loop ran 1536 elements instead of ple_dim=256 and wrote 1280 floats past the
; end of TMP256. Row 0's first 256 elements were still right, so B=1 parity
; stayed green and the overrun only surfaced at B>1, where row 1 is what gets
; stomped. Count-register/arg-position mismatches hide behind B=1 parity.
mul_avx2:
    xor     r10, r10
.mul_l:
    lea     rax, [r10 + 8]
    cmp     rax, r8
    ja      .mul_done
    vmovups ymm0, [rcx + r10 * 4]           ; VEX has no 2-operand memory form:
    vmulps  ymm0, ymm0, [rdx + r10 * 4]     ; the memory operand must be src3
    vmovups [rcx + r10 * 4], ymm0
    add     r10, 8
    jmp     .mul_l
.mul_done:
    ret

; ------------------------------------------------------------------------------
macro EXP dst, t1, t2, t3
{
    vmulps  t1, dst, ymm9                       ; t1 = z * log2(e)
    vaddps  t1, t1, ymm13                       ; + 127: PRE-BIAS the exponent
    vroundps t2, t1, 00001000b                  ; t2 = round(k + 127)
    vcvttps2dq t2, t2
    vpslld  t3, t2, 23                          ; (k+127)<<23 == 2^k as a float
    vcvtdq2ps t1, t2
    vsubps  t1, t1, ymm13                       ; back to k
    vmulps  t1, t1, ymm10                       ; k * ln2
    vsubps  dst, dst, t1                        ; r = z - k*ln2   (small)
    vmovaps t1, [K_P5]                           ; c5 (32-byte broadcast vector)
    vfmadd213ps t1, dst, [K_P4]
    vfmadd213ps t1, dst, [K_P3]
    vfmadd213ps t1, dst, [K_P2]
    vfmadd213ps t1, dst, [K_P1]
    vfmadd213ps t1, dst, [K_P0]
    vmulps  dst, t1, t3
}

macro HADDY v                                    ; sum 8 floats of v -> low element
{
    vextractf128 xmm14, v, 1
    vaddps  v, v, ymm14
    vpermilps ymm14, v, 0x4E
    vaddps  v, v, ymm14
    vpermilps ymm14, v, 0xB1
    vaddps  v, v, ymm14
}

macro MAXREDUCE v                                ; max of 8 floats of v -> low element
{
    vextractf128 xmm14, v, 1
    vmaxps  v, v, ymm14
    vpermilps ymm14, v, 0x4E
    vmaxps  v, v, ymm14
    vpermilps ymm14, v, 0xB1
    vmaxps  v, v, ymm14
}

; ==============================================================================
; rmsnorm_avx2(out, w, x, N, eps)
;   out_i = x_i * rsqrt(mean(x^2) + eps) * w_i
;   Gemma 4 removed Gemma 2's (1.0 + w) offset: the weight multiplies directly.
; ==============================================================================
rmsnorm_avx2:
    test    r9, r9
    jz      .rn_done
    vmovss  xmm10, dword [rsp + 40]              ; eps  (arg5: caller stores +32, RET pushes 8)
    mov     r11, r9                              ; element count
    shl     r9, 2                                ; byte count

    vxorps  ymm0, ymm0, ymm0
    vxorps  ymm1, ymm1, ymm1
    vxorps  ymm2, ymm2, ymm2
    vxorps  ymm3, ymm3, ymm3
    xor     r10, r10
.rn_ss:
    lea     rax, [r10 + 32]
    cmp     rax, r9
    ja      .rn_reduce
    vmovups ymm4, [r8 + r10]
    vfmadd231ps ymm0, ymm4, ymm4
    vmovups ymm5, [r8 + r10 + 32]
    vfmadd231ps ymm1, ymm5, ymm5
    add     r10, 64
    jmp     .rn_ss
.rn_reduce:
    vaddps  ymm0, ymm0, ymm1
    HADDY   ymm0
    vcvtsi2ss xmm2, xmm2, r11d
    vdivss  xmm0, xmm0, xmm2                     ; mean(x^2)
    vaddss  xmm0, xmm0, xmm10                    ; + eps

    ; rsqrt + one Newton step: 1/sqrt(y) ~= r*(1.5 - 0.5*y*r*r) applied to r
    vrsqrtss xmm1, xmm0, xmm0
    vmulss  xmm2, xmm1, xmm1
    vmulss  xmm2, xmm2, xmm0
    mov     eax, C_HALF
    vmovd   xmm4, eax
    mov     eax, C_THREE
    vmovd   xmm5, eax
    vsubss  xmm2, xmm5, xmm2
    vmulss  xmm2, xmm2, xmm4
    vmulss  xmm1, xmm1, xmm2
    vinsertf128 ymm0, ymm1, xmm1, 1        ; AVX2-legal broadcast of xmm1[31:0]
    vshufps   ymm0, ymm0, ymm0, 0x00

    xor     r10, r10
.rn_norm:
    lea     rax, [r10 + 32]
    cmp     rax, r9
    ja      .rn_done
    vmovups ymm4, [r8 + r10]
    vmulps  ymm4, ymm4, ymm0
    vmulps  ymm4, ymm4, [rdx + r10]
    vmovups [rcx + r10], ymm4
    add     r10, 32
    jmp     .rn_norm
.rn_done:
    vzeroupper
    ret

; ==============================================================================
; rmsnorm_scale_avx2(out, x, N, scale, eps)  -- scale-free norm * post scale
;   Used for per-layer-attention `v_norm`, which has no weight tensor:
;   out = x * rsqrt(mean(x^2) + eps) * scale
; ==============================================================================
rmsnorm_scale_avx2:
    test    r8, r8
    jz      .rs_done
    vmovss  xmm10, dword [rsp + 40]              ; eps   (arg5 -> +40 at entry)
    vmovd   xmm11, r9d                           ; scale (arg4 arrives in R9)
    mov     r11, r8
    shl     r8, 2
    vxorps  ymm0, ymm0, ymm0
    xor     r10, r10
.rs_ss:
    lea     rax, [r10 + 32]
    cmp     rax, r8
    ja      .rs_reduce
    vmovups ymm4, [rdx + r10]
    vfmadd231ps ymm0, ymm4, ymm4
    vmovups ymm5, [rdx + r10 + 32]
    vfmadd231ps ymm0, ymm5, ymm5
    add     r10, 64
    jmp     .rs_ss
.rs_reduce:
    HADDY   ymm0
    vcvtsi2ss xmm2, xmm2, r11d
    vdivss  xmm0, xmm0, xmm2
    vaddss  xmm0, xmm0, xmm10
    vrsqrtss xmm1, xmm0, xmm0
    vmulss  xmm2, xmm1, xmm1
    vmulss  xmm2, xmm2, xmm0
    mov     eax, C_HALF
    vmovd   xmm4, eax
    mov     eax, C_THREE
    vmovd   xmm5, eax
    vsubss  xmm2, xmm5, xmm2
    vmulss  xmm2, xmm2, xmm4
    vmulss  xmm1, xmm1, xmm2
    vmulss  xmm1, xmm1, xmm11
    vinsertf128 ymm0, ymm1, xmm1, 1        ; AVX2-legal broadcast of xmm1[31:0]
    vshufps   ymm0, ymm0, ymm0, 0x00
    xor     r10, r10
.rs_out:
    lea     rax, [r10 + 32]
    cmp     rax, r8
    ja      .rs_done
    vmovups ymm4, [rdx + r10]
    vmulps  ymm4, ymm4, ymm0
    vmovups [rcx + r10], ymm4
    add     r10, 32
    jmp     .rs_out
.rs_done:
    vzeroupper
    ret

; ==============================================================================
; gelu_tanh_avx2(out, x, N)
;   x / (1 + exp(-(C1*x + C3*x^3)))     (== x * sigmoid(2*sqrt(2/pi)*(x+0.044715x^3)))
; ==============================================================================
gelu_tanh_avx2:
    test    r8, r8
    jz      .gt_done
    vbroadcastss ymm6, dword [K_C1]
    vbroadcastss ymm7, dword [K_C3]
    EXP_CONSTS
    xor     r10, r10
.gt_loop:
    lea     rax, [r10 + 8]
    cmp     rax, r8
    ja      .gt_done
    vmovups ymm0, [rdx + r10 * 4]                ; x
    vmulps  ymm2, ymm0, ymm0                     ; x^2
    vmovaps ymm3, ymm7
    vfmadd213ps ymm3, ymm2, ymm6                 ; C3*x^2 + C1
    vmulps  ymm3, ymm3, ymm0                     ; w = x*(C1 + C3*x^2)
    vxorps  ymm2, ymm2, ymm2
    vsubps  ymm2, ymm2, ymm3                     ; -w
    vmaxps  ymm2, ymm2, ymm12                    ; clamp >= -44
    vminps  ymm2, ymm2, ymm11                    ; clamp <= +44
    EXP     ymm2, ymm3, ymm4, ymm5               ; exp(-w)
    vaddps  ymm2, ymm2, ymm15                    ; 1 + exp(-w)
    vmovups ymm0, [rdx + r10 * 4]
    vdivps  ymm0, ymm0, ymm2
    vmovups [rcx + r10 * 4], ymm0
    add     r10, 8
    jmp     .gt_loop
.gt_done:
    vzeroupper
    ret

; ==============================================================================
; geglu_avx2(out, gate, up, N) = gelu_tanh(gate) * up
; ==============================================================================
geglu_avx2:
    test    r9, r9
    jz      .gg_done
    vbroadcastss ymm6, dword [K_C1]
    vbroadcastss ymm7, dword [K_C3]
    EXP_CONSTS
    xor     r10, r10
.gg_loop:
    lea     rax, [r10 + 8]
    cmp     rax, r9
    ja      .gg_done
    vmovups ymm0, [rdx + r10 * 4]                ; gate
    vmovups ymm1, [r8  + r10 * 4]                ; up
    vmulps  ymm2, ymm0, ymm0
    vmovaps ymm3, ymm7
    vfmadd213ps ymm3, ymm2, ymm6
    vmulps  ymm3, ymm3, ymm0                     ; w
    vxorps  ymm2, ymm2, ymm2
    vsubps  ymm2, ymm2, ymm3
    vmaxps  ymm2, ymm2, ymm12
    vminps  ymm2, ymm2, ymm11
    EXP     ymm2, ymm3, ymm4, ymm5
    vaddps  ymm2, ymm2, ymm15                    ; 1 + exp(-w)
    vmulps  ymm0, ymm0, ymm1                     ; gate * up
    vdivps  ymm0, ymm0, ymm2
    vmovups [rcx + r10 * 4], ymm0
    add     r10, 8
    jmp     .gg_loop
.gg_done:
    vzeroupper
    ret

; ==============================================================================
; add_scaled_avx2(out, a, b, N, sa, sb)  ->  out = a*sa + b*sb
;   residual stream + `cur = (x + delta) * layer_scalar` in one pass
; ==============================================================================
add_scaled_avx2:
    test    r9, r9
    jz      .as_done
    vbroadcastss ymm0, dword [rsp + 40]
    vbroadcastss ymm1, dword [rsp + 48]
    xor     r10, r10
.as_loop:
    lea     rax, [r10 + 8]
    cmp     rax, r9
    ja      .as_done
    vmulps  ymm2, ymm0, [rdx + r10 * 4]
    vmulps  ymm3, ymm1, [r8 + r10 * 4]
    vaddps  ymm2, ymm2, ymm3
    vmovups [rcx + r10 * 4], ymm2
    add     r10, 8
    jmp     .as_loop
.as_done:
    vzeroupper
    ret

; ==============================================================================
; scale_avx2(buf, N, s)  /  zero_avx2(buf, N)
; ==============================================================================
scale_avx2:
    test    rdx, rdx
    jz      .sc_done
    vmovd   xmm0, r8d                            ; s (arg3 arrives in R8)
    vinsertf128 ymm0, ymm0, xmm0, 1
    vshufps   ymm0, ymm0, ymm0, 0x00
    xor     r10, r10
.sc_loop:
    lea     rax, [r10 + 8]
    cmp     rax, rdx
    ja      .sc_done
    vmulps  ymm1, ymm0, [rcx + r10 * 4]
    vmovups [rcx + r10 * 4], ymm1
    add     r10, 8
    jmp     .sc_loop
.sc_done:
    vzeroupper
    ret

zero_avx2:
    vxorps  ymm0, ymm0, ymm0
    shl     rdx, 2                               ; element count -> bytes
    xor     r10, r10
.z_loop:
    lea     rax, [r10 + 32]
    cmp     rax, rdx
    ja      .z_done
    vmovups [rcx + r10], ymm0
    add     r10, 32
    jmp     .z_loop
.z_done:
    vzeroupper
    ret

; ==============================================================================
; embed_bf16_avx2(out, emb, token, dim, scale)
;   out[i] = fp32(emb[token*dim + i]) * scale      (hidden emb: scale = sqrt(1536))
; ==============================================================================
embed_bf16_avx2:
    mov     rax, r9
    shl     rax, 1                               ; dim * 2 bytes
    imul    rax, r8                              ; token * dim * 2
    add     rdx, rax
    vbroadcastss ymm7, dword [rsp + 40]
    xor     r10, r10
.e_loop:
    lea     rax, [r10 + 8]
    cmp     rax, r9
    ja      .e_done
    vpmovzxwd ymm5, [rdx + r10 * 2]
    vpslld  ymm5, ymm5, 16                       ; bf16 -> f32 (zero-extend mantissa)
    vmulps  ymm5, ymm5, ymm7
    vmovups [rcx + r10 * 4], ymm5
    add     r10, 8
    jmp     .e_loop
.e_done:
    vzeroupper
    ret

; ==============================================================================
; rope_apply_avx2(vec, cos, sin, H)      H = head_dim/2, pairs (i, i+H)
;   vec[i]    = x[i]*cos[i]   - x[i+H]*sin[i]
;   vec[i+H]  = x[i+H]*cos[i] + x[i]*sin[i]
;   Partial rotary (full-attention layers, partial_rotary_factor=0.25) is expressed
;   by cos=1/sin=0 in the unused tail of the table, so this one kernel serves both.
; ==============================================================================
rope_apply_avx2:
    ; buf, cos, sin, H, rp        rotate pairs (i, i+rp), rp = head_dim/2
    ;
    ; HF's proportional_rope pads inv_freq with zeros out to head_dim/2, so the
    ; rotary window is always the whole head and the pairing stride is always
    ; head_dim/2 - partial rotation shows up as cos=1/sin=0 in the tail of the
    ; table, NOT as a narrower window. (Assuming rotate_half over a narrow
    ; window pairs (i, i+rotary_dim/2) instead; it does not, here.)
    mov     r11, [rsp + 40]
    mov     rsi, rcx
    mov     rax, r11
    shl     rax, 2
    add     rsi, rax                          ; hi base = buf + rp*4 (single index reg)
    xor     r10, r10
.rp_loop:
    lea     rax, [r10 + 8]
    cmp     rax, r11
    ja      .rp_done
    vmovups ymm0, [rcx + r10 * 4]             ; x[i]
    vmovups ymm1, [rsi + r10 * 4]             ; x[i+rp]
    vmovups ymm4, [rdx + r10 * 4]             ; cos[i]
    vmovups ymm5, [r8 + r10 * 4]              ; sin[i]
    vmulps  ymm2, ymm0, ymm4
    vmulps  ymm3, ymm1, ymm5
    vsubps  ymm2, ymm2, ymm3
    vmulps  ymm3, ymm1, ymm4
    vmulps  ymm0, ymm0, ymm5
    vaddps  ymm3, ymm3, ymm0
    vmovups [rcx + r10 * 4], ymm2
    vmovups [rsi + r10 * 4], ymm3
    add     r10, 8
    jmp     .rp_loop
.rp_done:
    vzeroupper
    ret

; ==============================================================================
softmax_avx2:
    test    rdx, rdx
    jz      .sm_done
    mov     eax, 0xFF800000                      ; -inf seed
    vmovd   xmm8, eax
    vinsertf128 ymm0, ymm8, xmm8, 1        ; AVX2-legal broadcast of xmm8[31:0]
    vshufps   ymm0, ymm0, ymm0, 0x00
    xor     r10, r10
    mov     r11, rdx
    and     r11, -8                              ; elements covered by full vectors
.sm_max:
    cmp     r10, r11
    jae     .sm_max_tail
    vmaxps  ymm0, ymm0, [rcx + r10 * 4]
    add     r10, 8
    jmp     .sm_max
.sm_max_tail:
    cmp     r11, rdx
    jae     .sm_apply
    cmp     rdx, 8
    jb      .sm_max_scalar
    mov     r10, rdx
    sub     r10, 8                               ; overlapping window, all in-bounds
    vmaxps  ymm0, ymm0, [rcx + r10 * 4]
    jmp     .sm_apply
.sm_max_scalar:
    xor     r10, r10
.sm_max_sloop:
    cmp     r10, rdx
    jae     .sm_apply
    vmovss  xmm1, dword [rcx + r10 * 4]
    vmaxps  xmm0, xmm0, xmm1
    inc     r10
    jmp     .sm_max_sloop
.sm_apply:
    MAXREDUCE ymm0
    vinsertf128 ymm4, ymm0, xmm0, 1        ; AVX2-legal broadcast of xmm0[31:0]
    vshufps   ymm4, ymm4, ymm4, 0x00                      ; m
    EXP_CONSTS
    vxorps  ymm7, ymm7, ymm7                     ; running sum
    xor     r10, r10
.sm_exp:
    cmp     r10, r11
    jae     .sm_exp_tail
    vmovups ymm2, [rcx + r10 * 4]
    vsubps  ymm2, ymm2, ymm4                     ; z = x - m <= 0
    vmaxps  ymm2, ymm2, ymm12                    ; floor at -44
    EXP     ymm2, ymm5, ymm6, ymm8
    vmovups [rcx + r10 * 4], ymm2
    vaddps  ymm7, ymm7, ymm2
    add     r10, 8
    jmp     .sm_exp
.sm_exp_tail:
    cmp     r11, rdx
    jae     .sm_norm
    mov     r10, r11
    mov     rax, rdx
    sub     rax, r11                             ; tail count 1..7
    vmovd   xmm3, eax
    vpbroadcastd ymm3, xmm3
    vpcmpgtd ymm3, ymm3, [K_LANES]               ; lane i active iff i < tail
    vmaskmovps ymm2, ymm3, [rcx + r10 * 4]       ; masked load: no read past the end
    vsubps  ymm2, ymm2, ymm4
    vmaxps  ymm2, ymm2, ymm12
    EXP     ymm2, ymm5, ymm6, ymm0
    vandps  ymm2, ymm2, ymm3                     ; inactive lanes contribute nothing
    vmaskmovps [rcx + r10 * 4], ymm3, ymm2
    vaddps  ymm7, ymm7, ymm2
.sm_norm:
    HADDY   ymm7
    vdivss  xmm7, xmm15, xmm7                    ; 1/sum
    vinsertf128 ymm7, ymm7, xmm7, 1
    vshufps   ymm7, ymm7, ymm7, 0x00
    xor     r10, r10
.sm_scale:
    cmp     r10, r11
    jae     .sm_scale_tail
    vmulps  ymm1, ymm7, [rcx + r10 * 4]
    vmovups [rcx + r10 * 4], ymm1
    add     r10, 8
    jmp     .sm_scale
.sm_scale_tail:
    cmp     r11, rdx
    jae     .sm_done
    mov     r10, r11
    mov     rax, rdx
    sub     rax, r11
    vmovd   xmm3, eax
    vpbroadcastd ymm3, xmm3
    vpcmpgtd ymm3, ymm3, [K_LANES]
    vmaskmovps ymm1, ymm3, [rcx + r10 * 4]
    vmulps  ymm1, ymm1, ymm7
    vmaskmovps [rcx + r10 * 4], ymm3, ymm1
.sm_done:
    vzeroupper
    ret

; ==============================================================================
; attn_scores_avx2(scores, q_head, Kcache, kv_len, head_dim, kv_start)
;   scores[p] = dot(q_head, Kcache[kv_start + p])     (MQA: one K row, 8 heads)
; ==============================================================================
attn_scores_avx2:
    test    r9, r9
    jz      .at_ret
    mov     r14, [rsp + 40]                      ; head_dim (elements). Stack args are
    mov     r11, [rsp + 48]                      ; kv_start  read BEFORE any frame shift
    push    r12                                  ; R12/R13/R14 are callee-saved: a kernel
    push    r13                                  ; reached through a ctypes thunk must not
    push    r14                                  ; come back with them changed (it did, and
    sub     rsp, 8                               ; CPython noticed only intermittently)
    shl     r14, 2                               ; head_dim bytes = K row stride
    mov     rax, r14
    imul    rax, r11
    add     r8, rax                              ; K row for kv_start
    xor     r12, r12                             ; p
.at_p:
    cmp     r12, r9
    jae     .at_done
    vxorps  ymm0, ymm0, ymm0
    vxorps  ymm1, ymm1, ymm1
    vxorps  ymm2, ymm2, ymm2
    vxorps  ymm3, ymm3, ymm3
    xor     r13, r13
.at_d:
    cmp     r13, r14
    jae     .at_red
    vmovups ymm4, [rdx + r13]
    vfmadd231ps ymm0, ymm4, [r8 + r13]
    vmovups ymm5, [rdx + r13 + 32]
    vfmadd231ps ymm1, ymm5, [r8 + r13 + 32]
    vmovups ymm6, [rdx + r13 + 64]
    vfmadd231ps ymm2, ymm6, [r8 + r13 + 64]
    vmovups ymm7, [rdx + r13 + 96]
    vfmadd231ps ymm3, ymm7, [r8 + r13 + 96]
    add     r13, 128
    jmp     .at_d
.at_red:
    vaddps  ymm0, ymm0, ymm1
    vaddps  ymm2, ymm2, ymm3
    vaddps  ymm0, ymm0, ymm2
    HADDY   ymm0
    vmovss  dword [rcx + r12 * 4], xmm0
    add     r8, r14
    inc     r12
    jmp     .at_p
.at_done:
    add     rsp, 8
    pop     r14
    pop     r13
    pop     r12
.at_ret:
    vzeroupper
    ret

; ==============================================================================
; attn_values_avx2(out, V, w, kv_len, head_dim, kv_start)
;   out[d] = sum_p w[p] * V[kv_start + p][d]
;   64 floats live in 8 register accumulators at a time, so each V row is one
;   contiguous 256-byte streaming run (this loop is pure memory bandwidth).
; ==============================================================================
attn_values_avx2:
    test    r9, r9
    jz      .av_ret
    mov     r13, [rsp + 40]                      ; head_dim (read before any frame shift)
    mov     r14, [rsp + 48]                      ; kv_start
    push    r12                                  ; callee-saved, see attn_scores
    push    r13
    push    r14
    sub     rsp, 8
    shl     r13, 2                               ; V row stride (bytes)
    mov     rax, r13
    imul    rax, r14
    add     rdx, rax                             ; V base row
    xor     r10, r10                             ; chunk byte offset
.av_chunk:
    cmp     r10, r13
    jae     .av_done
    vxorps  ymm0, ymm0, ymm0
    vxorps  ymm1, ymm1, ymm1
    vxorps  ymm2, ymm2, ymm2
    vxorps  ymm3, ymm3, ymm3
    vxorps  ymm4, ymm4, ymm4
    vxorps  ymm5, ymm5, ymm5
    vxorps  ymm6, ymm6, ymm6
    vxorps  ymm7, ymm7, ymm7
    mov     r12, rdx
    add     r12, r10                             ; V row + chunk
    xor     r11, r11                             ; p
.av_p:
    cmp     r11, r9
    jae     .av_store
    vbroadcastss ymm8, dword [r8 + r11 * 4]      ; w[p]
    vfmadd231ps ymm0, ymm8, [r12]
    vfmadd231ps ymm1, ymm8, [r12 + 32]
    vfmadd231ps ymm2, ymm8, [r12 + 64]
    vfmadd231ps ymm3, ymm8, [r12 + 96]
    vfmadd231ps ymm4, ymm8, [r12 + 128]
    vfmadd231ps ymm5, ymm8, [r12 + 160]
    vfmadd231ps ymm6, ymm8, [r12 + 192]
    vfmadd231ps ymm7, ymm8, [r12 + 224]
    add     r12, r13
    inc     r11
    jmp     .av_p
.av_store:
    lea     rax, [rcx + r10]
    vmovups [rax], ymm0
    vmovups [rax + 32], ymm1
    vmovups [rax + 64], ymm2
    vmovups [rax + 96], ymm3
    vmovups [rax + 128], ymm4
    vmovups [rax + 160], ymm5
    vmovups [rax + 192], ymm6
    vmovups [rax + 224], ymm7
    add     r10, 256
    jmp     .av_chunk
.av_done:
    add     rsp, 8
    pop     r14
    pop     r13
    pop     r12
.av_ret:
    vzeroupper
    ret

; ==============================================================================
; softcap_tanh_avx2(logits, N, cap)  ->  cap * tanh(logits / cap)
;   tanh(z) = 2/(1 + exp(-2z)) - 1     (NOT 1 - 2/(1+exp(-2z)), which is -tanh)
;   Only needed when sampling with a temperature: under argmax the transform is
;   monotone and the chosen token cannot change. The engine calls it from the head
;   so that both paths read capped logits; nothing guards cap == 0, so the caller
;   must (a zero cap means the model has no softcap, and 1/cap is infinite).
; ==============================================================================
softcap_tanh_avx2:
    test    rdx, rdx
    jz      .st_done
    vmovd   xmm7, r8d                            ; cap (arg3 arrives in R8)
    vinsertf128 ymm7, ymm7, xmm7, 1
    vshufps   ymm7, ymm7, ymm7, 0x00
    vmovss  xmm0, dword [K_ONE]
    vdivss  xmm0, xmm0, xmm7                     ; 1/cap
    vinsertf128 ymm6, ymm0, xmm0, 1        ; AVX2-legal broadcast of xmm0[31:0]
    vshufps   ymm6, ymm6, ymm6, 0x00
    vbroadcastss ymm14, dword [K_TWO]
    EXP_CONSTS
    xor     r10, r10
.st_loop:
    lea     rax, [r10 + 8]
    cmp     rax, rdx
    ja      .st_done
    vmulps  ymm0, ymm6, [rcx + r10 * 4]
    vaddps  ymm0, ymm0, ymm0                     ; -2z argument (negated below)
    vxorps  ymm1, ymm1, ymm1
    vsubps  ymm1, ymm1, ymm0
    vmaxps  ymm1, ymm1, ymm12
    vminps  ymm1, ymm1, ymm11
    EXP     ymm1, ymm2, ymm3, ymm5
    vaddps  ymm1, ymm1, ymm15                    ; 1 + exp(-2z)
    vdivps  ymm1, ymm14, ymm1                    ; 2/(1+exp(-2z)) = 1 + tanh(z)
    vsubps  ymm0, ymm1, ymm15                    ; tanh(z)
    vmulps  ymm0, ymm0, ymm7
    vmovups [rcx + r10 * 4], ymm0
    add     r10, 8
    jmp     .st_loop
.st_done:
    vzeroupper
    ret

; ==============================================================================
; ple_combine_avx2(out, proj, tok, D, pscale, tscale, inv_sqrt2)
;   out = ( RMSNorm_D(proj * pscale) + tok * tscale ) * inv_sqrt2
;   project_per_layer_inputs: the model-projection branch and the token-identity
;   branch of Per-Layer Embeddings mixed at 1/sqrt(2). The scale is applied to
;   `proj` BEFORE norming, exactly as the reference does (eps is not negligible
;   against (pscale*|proj|)^2/D, so the order matters).
; ==============================================================================
ple_combine_avx2:
    test    r9, r9
    jz      .pc_done
    vbroadcastss ymm10, dword [rsp + 40]                    ; pscale
    vbroadcastss ymm11, dword [rsp + 48]                    ; tscale = sqrt(ple_dim) = 16
    mov     r11, [rsp + 64]              ; RMSNorm weight (per_layer_projection_norm)
    vbroadcastss ymm12, dword [rsp + 56]                    ; 1/sqrt(2)
    vmovss  xmm13, dword [K_EPS]
    vxorps  ymm0, ymm0, ymm0
    xor     r10, r10
.pc_ss:
    lea     rax, [r10 + 8]
    cmp     rax, r9
    ja      .pc_red
    vmulps  ymm3, ymm10, [rdx + r10 * 4]
    vfmadd231ps ymm0, ymm3, ymm3
    add     r10, 8
    jmp     .pc_ss
.pc_red:
    HADDY   ymm0
    vcvtsi2ss xmm2, xmm2, r9d
    vdivss  xmm0, xmm0, xmm2
    vaddss  xmm0, xmm0, xmm13
    vrsqrtss xmm1, xmm0, xmm0
    vmulss  xmm2, xmm1, xmm1
    vmulss  xmm2, xmm2, xmm0
    mov     eax, C_HALF
    vmovd   xmm4, eax
    mov     eax, C_THREE
    vmovd   xmm5, eax
    vsubss  xmm2, xmm5, xmm2
    vmulss  xmm2, xmm2, xmm4
    vmulss  xmm1, xmm1, xmm2
    vinsertf128 ymm0, ymm1, xmm1, 1        ; AVX2-legal broadcast of xmm1[31:0]
    vshufps   ymm0, ymm0, ymm0, 0x00                      ; rsqrt(mean+eps)
    xor     r10, r10
.pc_out:
    lea     rax, [r10 + 8]
    cmp     rax, r9
    ja      .pc_done
    vmulps  ymm3, ymm10, [rdx + r10 * 4]         ; proj * pscale
    vmulps  ymm3, ymm3, ymm0                     ; * rsqrt
    vmulps  ymm3, ymm3, [r11 + r10 * 4]          ; * RMSNorm weight
    vpmovzxwd ymm4, [r8 + r10 * 2]               ; per_layer_embeddings is bf16
    vpslld  ymm4, ymm4, 16
    vfmadd231ps ymm3, ymm4, ymm11                 ; + tok * tscale
    vmulps  ymm3, ymm3, ymm12
    vmovups [rcx + r10 * 4], ymm3
    add     r10, 8
    jmp     .pc_out
.pc_done:
    vzeroupper
    ret

; ==============================================================================
; sampler_argmax_avx2(logits, N) -> rax = argmax index   (greedy: raw logits,
; so the monotone tanh softcap is intentionally not applied)
; ==============================================================================
sampler_argmax_avx2:
    mov     eax, 0xFF800000                      ; -inf: correct for all-negative rows
    vmovd   xmm2, eax
    vinsertf128 ymm2, ymm2, xmm2, 1
    vshufps   ymm2, ymm2, ymm2, 0x00
    mov     r10, 0
    mov     r11, -1
.ax_loop:
    lea     rax, [r10 + 8]
    cmp     rax, rdx
    ja      .ax_scalar
    vmovups ymm0, [rcx + r10 * 4]
    vcmpps  ymm1, ymm0, ymm2, 0x1E               ; any lane beats the running max?
    vmovmskps eax, ymm1
    test    eax, eax
    jz      .ax_next                             ; (common case: nothing to do)
    ; A new maximum exists in this block. Compare each lane against the NEW GLOBAL
    ; MAX, not against the per-lane max(old, x): with per-lane comparison, a lane
    ; that merely EQUALS the old max also matches, and bsf then reports lane 0 of
    ; the block - which is exactly the bug that made argmax return k & ~7.
    vmaxps  ymm3, ymm0, ymm2
    MAXREDUCE ymm3                               ; rare path: reduce here only
    vinsertf128 ymm3, ymm3, xmm3, 1
    vshufps   ymm3, ymm3, ymm3, 0x00             ; uniform new max
    vcmpps  ymm1, ymm0, ymm3, 0x00               ; lanes achieving it
    vmovmskps eax, ymm1
    bsf     eax, eax                             ; first (lowest index) wins
    lea     r11, [r10 + rax]
    vmovaps ymm2, ymm3
.ax_next:
    add     r10, 8
    jmp     .ax_loop
.ax_scalar:
    MAXREDUCE ymm2                               ; vector max now in the low element
.ax_sloop:
    cmp     r10, rdx
    jae     .ax_done
    vmovss  xmm0, dword [rcx + r10 * 4]
    vucomiss xmm0, xmm2
    jbe     .ax_snext
    vmovaps xmm2, xmm0
    mov     r11, r10
.ax_snext:
    inc     r10
    jmp     .ax_sloop
.ax_done:
    mov     rax, r11
    vzeroupper
    ret

; ------------------------------------------------------------------------------
align 4
K_ONE:  dd C_ONE
K_LOG2E: dd C_LOG2E
K_LN2:  dd C_LN2
K_CLAMP: dd C_CLAMP
K_NCLAMP: dd 0xC2300000                          ; -44.0f
K_127:  dd C_127                                 ; 127.0f exactly
K_C1:   dd C_C1
K_C3:   dd C_C3
K_TWO:  dd C_TWO
K_NINF: dd 0xFF800000                            ; -inf (max/argmax seeds)
K_EPS:  dd 0x358637BD                            ; 1e-6f EXACTLY (Gemma 4 rms_norm_eps).
                                        ; 0x3727C5AC is 1e-5, a 10x slip that costs
                                        ; 0.7% on the PLE mix - caught by the test.
K_LANES: dd 0, 1, 2, 3, 4, 5, 6, 7               ; lane indices for tail masks
; Degree-5 Taylor coefficients of exp(r) for |r| <= ln2/2 (rel. err < 1e-7).
; IMPORTANT: these are consumed directly as VEX memory operands, so each one must
; occupy a FULL 32-byte ymm. A lone `dd` here would silently feed the *adjacent
; constant* to lanes 1..7 (observed: correct lane 0, nan in lanes 5-7), and
; `dq lo, hi` is only 16 bytes, which additionally breaks VMOVAPS's alignment.
align 32          ; ONE align for the whole block: each constant is exactly 32
                  ; bytes, so every label stays 32-byte aligned *and* points at
                  ; its own data. (A label written before an in-macro `align`
                  ; points at the zero padding instead - same silent corruption.)
K_P0:   dd 0x3F800000, 0x3F800000, 0x3F800000, 0x3F800000, 0x3F800000, 0x3F800000, 0x3F800000, 0x3F800000
K_P1:   dd 0x3F800000, 0x3F800000, 0x3F800000, 0x3F800000, 0x3F800000, 0x3F800000, 0x3F800000, 0x3F800000
K_P2:   dd 0x3F000000, 0x3F000000, 0x3F000000, 0x3F000000, 0x3F000000, 0x3F000000, 0x3F000000, 0x3F000000
K_P3:   dd 0x3E2AAAAB, 0x3E2AAAAB, 0x3E2AAAAB, 0x3E2AAAAB, 0x3E2AAAAB, 0x3E2AAAAB, 0x3E2AAAAB, 0x3E2AAAAB
K_P4:   dd 0x3D2AAAAB, 0x3D2AAAAB, 0x3D2AAAAB, 0x3D2AAAAB, 0x3D2AAAAB, 0x3D2AAAAB, 0x3D2AAAAB, 0x3D2AAAAB
K_P5:   dd 0x3C088889, 0x3C088889, 0x3C088889, 0x3C088889, 0x3C088889, 0x3C088889, 0x3C088889, 0x3C088889
