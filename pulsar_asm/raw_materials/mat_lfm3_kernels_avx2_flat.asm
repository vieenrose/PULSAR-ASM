; ==============================================================================
; mat_lfm3_kernels_avx2_flat.asm
; ==============================================================================
use64

lfm3_dummy:
    ret

; PLSE export trailer
    align   4
lfmk_n0: db 'lfm3_dummy', 0
    align   4
    dd      lfm3_dummy
lfmk_noff:
    dd      lfmk_n0
lfmk_cnt: dd 1
    db      'PLSE'
