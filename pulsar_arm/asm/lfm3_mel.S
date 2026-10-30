/* lfm3_mel.S -- AArch64/NEON log-mel front end for LFM3-Audio input,
   Jetson Nano (Cortex-A57, ARMv8.0: no FEAT_BF16, no SDOT, 128-bit NEON).

   WHAT THIS IS. Stage S2 of doc/LFM3_PORT_STAGES.md: the 16 kHz waveform -> 128
   log-mel frames the audio encoder consumes. Reference is
   LFM3-Audio-2B/feature_extraction_lfm3_audio.py:_log_mel, which is
     window = torch.hann_window(400)                       (periodic, f32)
     spec   = torch.stft(x, 400, 160, 400, window, center=False)   (onesided)
     power  = spec.abs().square()
     mel    = (filters.T @ power).log10().clamp(min=-12)
     out    = (mel + 4) / 4
   Every constant (window, mel filterbank, twiddles) comes from mel.bin built by
   pulsar_arm/tools/mkmel.py, which calls the reference's own
   mel_filter_bank()/torch.hann_window -- the runtime cannot drift from HF on a
   constant by construction. The log10/clamp/affine is arithmetic and lives in
   the C driver (libm log10f): this file is the O(N^2) math.

   WHY A DIRECT DFT AND NOT AN FFT. 400 = 16*25, so a Cooley-Tukey FFT in asm
   needs radix-25 butterflies for ~zero benefit: one frame costs 201 bins x 200
   taps = 40k FMA, i.e. ~25k cycles (~12 us at 1.5 GHz), and a 5 s clip is 500
   frames = 6 ms. The FFT would be a correctness risk bought for microseconds.
   The direct form is also the one whose accumulation order a C mirror can
   reproduce bit-for-bit, which is what the gate in lfm3_mel_test.c needs.

   THE HALF-LENGTH PAIR FORM. For a real frame y of length N = 400,
     X[k] = sum_{n<200} (y[n] + (-1)^k y[n+200]) * exp(-2i*pi*k*n/400)
   so the kernel forms cp = y[n]+y[n+200] and co = y[n]-y[n+200] once per frame
   (400 adds) and does a 200-tap dot per bin, selecting cp for even k and co for
   odd k. cos/sin(2*pi*k*n/400) are precomputed per bin in mel.bin, so the inner
   loop is three sequential vector loads and four FP ops per 8 taps -- no lane
   gathers (NEON has none), no per-tap modulo.

   THE ACCUMULATION ORDER IS THE CONTRACT, AND THE TEST MIRRORS IT.
   Per bin: taps ascend in n; 8 lanes at a time; 4 accumulator vectors
   (rA = a0..a3 for cos, rB = a4..a7, iA/iB likewise for sin); each step is a
   separate fmul then fadd -- no FMA, because fma(a,b,c) is a different real
   number than a*b+c rounded twice, and lfm3_mel_test.c compiles the reference
   with -ffp-contract=off. Final reduction is the house tree
     total = ((a0+a1)+(a2+a3)) + ((a4+a5)+(a6+a7))
   (same convention as lfm3_elem.S). power[k] = re*re + im*im, again mul-then-add.
   The mel matmul is the same shape: k ascends, 8 lanes, 4 accumulator vectors,
   house tree. Filter rows are zero-padded to 208 columns in mel.bin so no tail
   code is needed and the 7 extra terms are exact zeros.

   BIT-EXACTNESS IS ASKED OF THE C MIRROR, NOT OF TORCH. torch.stft uses
   pocketfft (a different order entirely) and torch's matmul uses BLAS, so
   parity with HF is a tolerance gate -- max |delta| per frame, run by
   .auto/lfm3/checks_mel.py. Claiming bit-parity against a BLAS matmul would be
   the kind of unverifiable claim this repo does not make.
*/
        .text

/* ------------------------------------------------------------------ */
/* void lfm3_mel_power(const float *x, const float *win,
                       const float *twc, const float *tws, float *pw,
                       int n_bins)

   x    : 400 samples of the frame, unwindowed
   win  : 400 window samples
   twc  : n_bins rows of 200 floats, twc[k*200+n] = cos(2*pi*k*n/400)
   tws  : the same for sin
   pw   : n_bins floats of power. Rows n_bins..207 of the caller's buffer are
          LEFT ALONE (the driver zeroes them once) so pw can be a 208-float
          row that feeds lfm3_mel_matmul directly.
*/
        .global lfm3_mel_power
        .type   lfm3_mel_power, %function
        .p2align 5
lfm3_mel_power:
        stp     x29, x30, [sp, #-96]!
        mov     x29, sp
        str     x19, [sp, #80]
        sub     sp, sp, #3216                 /* y[400] @sp, cp[200] @+1600, co[200] @+2400 */
        mov     x19, sp
        sxtw    x15, w5                       /* n_bins: capture BEFORE x5 becomes the y cursor --
                                                 reading w5 later sign-extended the y-pointer's low
                                                 word and the bin loop ran exactly once. */

/* ---- y[n] = x[n] * win[n], n = 0..399 (50 blocks of 8) ---- */
        mov     x5, x0
        mov     x6, x1
        mov     x7, x19
        mov     w8, #50
.Lmel_y:
        ld1     {v0.4s, v1.4s}, [x5], #32
        ld1     {v2.4s, v3.4s}, [x6], #32
        fmul    v4.4s, v0.4s, v2.4s
        fmul    v5.4s, v1.4s, v3.4s
        st1     {v4.4s, v5.4s}, [x7], #32
        subs    w8, w8, #1
        b.ne    .Lmel_y

/* ---- cp[n] = y[n]+y[n+200], co[n] = y[n]-y[n+200], n = 0..199 ---- */
        mov     x5, x19
        add     x6, x19, #800                 /* y + 200 floats */
        add     x7, x19, #1600                /* cp */
        add     x9, x19, #2400                /* co */
        mov     w8, #25
.Lmel_pair:
        ld1     {v0.4s, v1.4s}, [x5], #32
        ld1     {v2.4s, v3.4s}, [x6], #32
        fadd    v4.4s, v0.4s, v2.4s
        fsub    v5.4s, v0.4s, v2.4s
        fadd    v6.4s, v1.4s, v3.4s
        fsub    v7.4s, v1.4s, v3.4s
        st1     {v4.4s}, [x7], #16                /* binutils 2.30 wants consecutive */
        st1     {v6.4s}, [x7], #16                /* registers in an st1 list          */
        st1     {v5.4s}, [x9], #16
        st1     {v7.4s}, [x9], #16
        subs    w8, w8, #1
        b.ne    .Lmel_pair

/* ---- per bin k: re = sum c[n]*cos, im = sum c[n]*sin, power = re^2+im^2 ---- */
        mov     x10, x19
        add     x10, x10, #1600               /* cp base */
        add     x11, x19, #2400               /* co base */
        mov     x12, x2                       /* twc cursor */
        mov     x13, x3                       /* tws cursor */
        mov     x14, x4                       /* pw cursor  */
        mov     w16, #0                       /* k */
.Lmel_bin:
        tbz     w16, #0, .Lmel_even
        mov     x17, x11
        b       .Lmel_havec
.Lmel_even:
        mov     x17, x10
.Lmel_havec:
        mov     x5, x12
        mov     x6, x13
        mov     x7, x17
        movi    v16.4s, #0                    /* rA = a0..a3 (cos) */
        movi    v17.4s, #0                    /* rB = a4..a7       */
        movi    v18.4s, #0                    /* iA = a0..a3 (sin) */
        movi    v19.4s, #0                    /* iB = a4..a7       */
        mov     w8, #25                       /* 200 taps / 8 */
.Lmel_tap:
        ld1     {v0.4s, v1.4s}, [x5], #32     /* cos */
        ld1     {v2.4s, v3.4s}, [x6], #32     /* sin */
        ld1     {v4.4s, v5.4s}, [x7], #32     /* c   */
        fmul    v20.4s, v0.4s, v4.4s
        fmul    v21.4s, v1.4s, v5.4s
        fmul    v22.4s, v2.4s, v4.4s
        fmul    v23.4s, v3.4s, v5.4s
        fadd    v16.4s, v16.4s, v20.4s
        fadd    v17.4s, v17.4s, v21.4s
        fadd    v18.4s, v18.4s, v22.4s
        fadd    v19.4s, v19.4s, v23.4s
        subs    w8, w8, #1
        b.ne    .Lmel_tap

/* house reduction: ((a0+a1)+(a2+a3)) + ((a4+a5)+(a6+a7)), scalar extracts
   (binutils 2.30 wants `mov`, not `fmov`, for the vector-element form) */
        mov     s24, v16.s[0]
        mov     s25, v16.s[1]
        mov     s26, v16.s[2]
        mov     s27, v16.s[3]
        fadd    s24, s24, s25
        fadd    s26, s26, s27
        fadd    s24, s24, s26                 /* tR */
        mov     s28, v17.s[0]
        mov     s29, v17.s[1]
        mov     s20, v17.s[2]
        mov     s21, v17.s[3]
        fadd    s28, s28, s29
        fadd    s20, s20, s21
        fadd    s28, s28, s20                 /* tI (second half of re) */
        fadd    s24, s24, s28                 /* re */
        mov     s25, v18.s[0]
        mov     s26, v18.s[1]
        mov     s27, v18.s[2]
        mov     s20, v18.s[3]
        fadd    s25, s25, s26
        fadd    s27, s27, s20
        fadd    s25, s25, s27                 /* tR */
        mov     s26, v19.s[0]
        mov     s27, v19.s[1]
        mov     s20, v19.s[2]
        mov     s21, v19.s[3]
        fadd    s26, s26, s27
        fadd    s20, s20, s21
        fadd    s26, s26, s20                 /* tI */
        fadd    s25, s25, s26                 /* im */

        fmul    s27, s24, s24
        fmul    s28, s25, s25
        fadd    s27, s27, s28                 /* power = re*re + im*im */
        str     s27, [x14], #4
        add     x12, x12, #800                /* one row = 200 floats */
        add     x13, x13, #800
        add     w16, w16, #1
        cmp     x16, x15
        b.lt    .Lmel_bin

        ldr     x19, [sp, #3296]                /* stored at [sp,#80] before the sub */
        add     sp, sp, #3216
        ldp     x29, x30, [sp], #96
        ret
        .size   lfm3_mel_power, .-lfm3_mel_power

/* ------------------------------------------------------------------ */
/* void lfm3_mel_matmul(const float *pw, const float *filt, float *mel,
                       int n_mels)

   pw   : 208 floats (201 bins + 7 zeros, zero-filled by the driver)
   filt : n_mels rows of 208 floats (row m = mel filter m)
   Order: k ascends, 8 lanes (two 4-lane accumulator vectors), house tree.
*/
        .global lfm3_mel_matmul
        .type   lfm3_mel_matmul, %function
        .p2align 5
lfm3_mel_matmul:
        stp     x29, x30, [sp, #-32]!
        mov     x29, sp
        str     x19, [sp, #16]
        mov     x19, x0                       /* pw base (constant across rows) */
        mov     x12, x1                       /* filt cursor */
        mov     x14, x2                       /* mel cursor */
        sxtw    x15, w3
        mov     w16, #0
.Lmel_row:
        mov     x5, x19
        mov     x6, x12
        movi    v16.4s, #0                    /* a0..a3 */
        movi    v17.4s, #0                    /* a4..a7 */
        mov     w8, #26                       /* 208 / 8 */
.Lmel_mm:
        ld1     {v0.4s}, [x6], #16
        ld1     {v1.4s}, [x5], #16
        ld1     {v2.4s}, [x6], #16
        ld1     {v3.4s}, [x5], #16
        fmul    v4.4s, v0.4s, v1.4s
        fmul    v5.4s, v2.4s, v3.4s
        fadd    v16.4s, v16.4s, v4.4s
        fadd    v17.4s, v17.4s, v5.4s
        subs    w8, w8, #1
        b.ne    .Lmel_mm
        mov     s18, v16.s[0]
        mov     s19, v16.s[1]
        mov     s20, v16.s[2]
        mov     s21, v16.s[3]
        fadd    s18, s18, s19
        fadd    s20, s20, s21
        fadd    s18, s18, s20                 /* tA */
        mov     s19, v17.s[0]
        mov     s20, v17.s[1]
        mov     s21, v17.s[2]
        mov     s22, v17.s[3]
        fadd    s19, s19, s20
        fadd    s21, s21, s22
        fadd    s19, s19, s21                 /* tB */
        fadd    s18, s18, s19
        str     s18, [x14], #4
        add     x12, x12, #832                /* row = 208 floats */
        add     w16, w16, #1
        cmp     x16, x15
        b.lt    .Lmel_row
        ldr     x19, [sp, #16]
        ldp     x29, x30, [sp], #32
        ret
        .size   lfm3_mel_matmul, .-lfm3_mel_matmul

/* ------------------------------------------------------------------ */
/* void lfm3_mel_norm(const float *mel, float *out, int n,
                      float floor_v, float offset, float divisor)

   out[i] = (max(mel[i], floor_v) + offset) / divisor, the affine tail of
   _log_mel. Kept in asm so the front end has one home; log10() is the caller's
   job (libm log10f in the C driver -- the reference does .log10() on the mel
   values, and this kernel's clamp is applied to the ALREADY-LOGGED input, so
   the caller passes log10(mel) and the clamp is on the log side, matching
   torch's .log10().clamp(min=-12)).
   log10(0) = -inf, and max(-inf, -12) = -12, which is what torch's clamp gives.
*/
        .global lfm3_mel_norm
        .type   lfm3_mel_norm, %function
        .p2align 4
lfm3_mel_norm:
        dup     v16.4s, v0.s[0]                 /* floor  (s0) */
        dup     v17.4s, v1.s[0]                 /* offset (s1) */
        dup     v18.4s, v2.s[0]                 /* divisor(s2) */
        sxtw    x3, w2                          /* n is the THIRD arg (w2); FP args are s0-s2 */
        mov     x4, #0
.Lmel_nrm:
        cmp     x4, x3
        b.ge    .Lmel_nrm_end
        sub     x9, x3, x4
        cmp     x9, #4
        b.lt    .Lmel_nrm_tail
        ld1     {v19.4s}, [x0], #16
        fmax    v19.4s, v19.4s, v16.4s          /* fmax(NaN-free) mirrors clamp(min=) */
        fadd    v19.4s, v19.4s, v17.4s
        fdiv    v19.4s, v19.4s, v18.4s
        st1     {v19.4s}, [x1], #16
        add     x4, x4, #4
        b       .Lmel_nrm
.Lmel_nrm_tail:
        ldr     s19, [x0], #4
        fmax    s19, s19, s16
        fadd    s19, s19, s17
        fdiv    s19, s19, s18
        str     s19, [x1], #4
        add     x4, x4, #1
        b       .Lmel_nrm
.Lmel_nrm_end:
        ret
        .size   lfm3_mel_norm, .-lfm3_mel_norm
