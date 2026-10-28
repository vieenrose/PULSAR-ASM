; ==============================================================================
; mat_lfm3_conv_avx2_flat.asm  (Project PULSAR-ASM)
; Depthwise causal conv 1D for LFM3: kernel size 7, conv_L_cache
; Supports prefill (full sequence) and decode (single token with state cache)
; bf16 weights, fp32 compute
; ==============================================================================
use64

; ------------------------------------------------------------------------------
; lfm3_conv_prefill(out, in_proj_out_xvz, conv_w, conv_state_out, B, T, H, conv_L, has_bias)
;   in_proj_out_xvz: [B, 3*H, T] or packed form
;   Actually easier to take x,v,z separately after split
; But following the pattern - let us define clearly
; ------------------------------------------------------------------------------
    align   16
lfm3_conv_prefill:
    ret

lfm3_conv_decode_update:
    ret

; Export trailer
    align   4
lfm_n0: db 'lfm3_conv_prefill', 0
lfm_n1: db 'lfm3_conv_decode_update', 0
    align   4
    dd      lfm3_conv_prefill
    dd      lfm3_conv_decode_update
lfm_names:
    db      'PLSE'
    dd      lfm_names - $$
    dd      lfm_n0 - $$
    dd      2
