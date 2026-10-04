; Test-only composition: the Gemma 4 pointwise/attention kernels plus the
; standard PLSE export trailer (production modules get the same trailer from
; engine/gemma4_engine_flat.asm).
use64

include '../raw_materials/mat_gemma4_kernels_avx2_flat.asm'

; ------------------------------------------------------------------------------
; Export directory: [ code ][ names ][ dd offs * N ][ dd nameoffs * N ][ dd N ][ 'PLSE' ]
; ------------------------------------------------------------------------------
    align 4
plsar_n0:  db 'rmsnorm_avx2', 0
plsar_n1:  db 'rmsnorm_scale_avx2', 0
plsar_n2:  db 'gelu_tanh_avx2', 0
plsar_n3:  db 'geglu_avx2', 0
plsar_n4:  db 'add_scaled_avx2', 0
plsar_n5:  db 'scale_avx2', 0
plsar_n6:  db 'zero_avx2', 0
plsar_n7:  db 'embed_bf16_avx2', 0
plsar_n8:  db 'rope_apply_avx2', 0
plsar_n9:  db 'softmax_avx2', 0
plsar_n10: db 'attn_scores_avx2', 0
plsar_n11: db 'attn_values_avx2', 0
plsar_n12: db 'softcap_tanh_avx2', 0
plsar_n13: db 'ple_combine_avx2', 0
plsar_n14: db 'sampler_argmax_avx2', 0
plsar_n15: db 'mul_avx2', 0
    align 4
    dd rmsnorm_avx2, rmsnorm_scale_avx2, gelu_tanh_avx2, geglu_avx2
    dd add_scaled_avx2, scale_avx2, zero_avx2, embed_bf16_avx2
    dd rope_apply_avx2, softmax_avx2, attn_scores_avx2, attn_values_avx2
    dd softcap_tanh_avx2, ple_combine_avx2, sampler_argmax_avx2
    dd mul_avx2
plsar_names:
    dd plsar_n0, plsar_n1, plsar_n2, plsar_n3, plsar_n4
    dd plsar_n5, plsar_n6, plsar_n7, plsar_n8, plsar_n9
    dd plsar_n10, plsar_n11, plsar_n12, plsar_n13, plsar_n14
    dd plsar_n15
    dd 16
    db 'PLSE'
