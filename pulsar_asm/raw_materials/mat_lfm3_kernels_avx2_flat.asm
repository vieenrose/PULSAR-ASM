; ==============================================================================
; mat_lfm3_kernels_avx2_flat.asm  (Project PULSAR-ASM)
; LFM3-specific kernels: partial rotary, sandwich norm helpers if needed
; Reuses most from mat_gemma4_kernels_avx2_flat.asm
; ==============================================================================
use64

; Export trailer
    align   4
lfmk_n0: db 'lfm3_dummy', 0
    align   4
    dd      0
lfmk_names:
    db      'PLSE'
    dd      lfmk_names - $$
    dd      lfmk_n0 - $$
    dd      1
