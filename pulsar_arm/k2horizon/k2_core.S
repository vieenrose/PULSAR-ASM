/* PULSAR-ASM | pulsar_arm/k2horizon/k2_core.S
 *
 * Pure AArch64 assembly runtime for IFM K2-Horizon-0.9B (Q4 blobs).
 * No C, no libc: `as` + `ld -static`, entry _start, svc syscalls only.
 * Replaces fwd_k2.c (base) and the Python+TFLite driver (meeting FT).
 *
 * Bit-exactness contract vs fwd_k2q (gcc -O2 -march=armv8.2-a+dotprod+fp16):
 *  - expf: bit-identical mirror of glibc 2.39 __expf_finite (fast poly +
 *    table @ libm 0x7a5f0, threshold specials from probe).
 *  - rms: f64 fmadd accumulate, fsqrt, y=(1/sqrt)*x*w (two fmul).
 *  - Q4 GEMV: fmul(ws*db) + fmadd(acc, ...) per block, SDOT lanes.
 *  - rope: fmul(b,S) / fnmsub(C,a,t) and fmul(b,C) / fmadd(S,a,t).
 *  - attention: scalar fmul+fadd dot in element order, f64 expf sum,
 *    f32 p = expf/(f32)es, fmla V-accum, strict-> max.
 *  - SwiGLU: fneg -> expf -> fadd 1 -> fdiv g/.. -> fmul LU.
 *  - printf: "step %d id_in=%d top=%d logit=%.4f\n" with glibc-correct
 *    %.4f (f64 x10000 exact + frintn ties-even + signbit).
 * Run: ./k2_core <blob> <ids...> --gen G [--sample temp topp topk seed [minp]]
 */
    .arch armv8.2-a
    .arch_extension dotprod
    .text
    .align 2
    .global _start

/* ---------------- syscalls ---------------- */
    .equ SYS_WRITE, 64
    .equ SYS_EXIT, 94            /* exit_group: kill spinning workers too */
    .equ SYS_OPENAT, 56
    .equ SYS_LSEEK, 62
    .equ SYS_MMAP, 222
    .equ SYS_CLOSE, 57

/* ---------------- tiny print helpers (x9-x18 scratch) ---------------- */

/* print_cstr: x0 = asciz address. Clobbers x0-x2,x8,x9. */
print_cstr:
    mov x9, x0
0:  ldrb w1, [x9], #1
    cbnz w1, 0b
    sub x2, x9, x0
    sub x2, x2, #1
    mov x1, x0
    mov x0, #1
    mov x8, #SYS_WRITE
    svc #0
    ret

/* print_u64: x0 = value. Clobbers x0-x4,x8,x9. */
print_u64:
    adrp x9, numbuf
    add x9, x9, :lo12:numbuf
    add x9, x9, #64
    cbnz x0, 1f
    sub x9, x9, #1
    mov w2, #'0'
    strb w2, [x9]
    b 2f
1:  mov x1, #10
    udiv x2, x0, x1
    msub x3, x2, x1, x0
    add w3, w3, #'0'
    sub x9, x9, #1
    strb w3, [x9]
    mov x0, x2
    cbnz x0, 1b
2:  adrp x1, numbuf
    add x1, x1, :lo12:numbuf
    add x1, x1, #64
    sub x2, x1, x9
    mov x1, x9
    mov x0, #1
    mov x8, #SYS_WRITE
    svc #0
    ret

/* print_ch: w0 = char. */
print_ch:
    adrp x9, numbuf
    add x9, x9, :lo12:numbuf
    strb w0, [x9]
    mov x1, x9
    mov x2, #1
    mov x0, #1
    mov x8, #SYS_WRITE
    svc #0
    ret

/* print_f32_4: s0 = value -> glibc-style "%.4f" text.
 * Exact path: (f64)v * 10000 is exact for f32 inputs; frintn = ties-to-even
 * (glibc %.4f confirmed ties-to-even on the dyadic k/32 grid by probe).
 * Non-finite -> "nan"/"inf"/"-inf" like glibc. Clobbers x0-x4,x9,d0-d3. */
print_f32_4:
    str x30, [sp, #-16]!
    fcmp s0, s0
    b.vs .Lpf_nan
    adrp x9, f_inf32
    ldr s1, [x9, :lo12:f_inf32]
    fcmp s0, s1
    b.eq .Lpf_inf
    fneg s1, s1
    fcmp s0, s1
    b.eq .Lpf_ninf
    fmov w2, s0
    tst w2, #0x80000000
    b.eq 0f
    mov w0, #'-'
    bl print_ch
    and w2, w2, #0x7fffffff
    fmov s0, w2
0:  fcvt d1, s0                 /* exact */
    adrp x9, f_tenthousand
    ldr d2, [x9, :lo12:f_tenthousand]
    fmul d1, d1, d2             /* exact (f32 24-bit mantissa) */
    frintn d1, d1               /* ties-to-even = glibc correct rounding */
    fcvtzs x3, d1               /* n: KEEP in x3 across print calls */
    mov x1, #10000
    udiv x0, x3, x1
    str x3, [sp, #8]
    bl print_u64
    ldr x3, [sp, #8]
    mov w0, #'.'
    bl print_ch
    ldr x3, [sp, #8]
    mov x1, #10000
    udiv x2, x3, x1
    msub x4, x2, x1, x3         /* frac */
    mov x1, #1000
    udiv x0, x4, x1             /* d3 */
    add w0, w0, #'0'
    str x4, [sp, #8]
    bl print_ch
    ldr x4, [sp, #8]
    mov x1, #100
    udiv x2, x4, x1             /* frac/100 */
    mov x1, #10
    udiv x0, x2, x1
    msub x0, x0, x1, x2         /* (frac/100)%10 */
    add w0, w0, #'0'
    bl print_ch
    ldr x4, [sp, #8]
    mov x1, #10
    udiv x2, x4, x1             /* frac/10 */
    mov x1, #10
    udiv x0, x2, x1
    msub x0, x0, x1, x2         /* (frac/10)%10 */
    add w0, w0, #'0'
    bl print_ch
    ldr x4, [sp, #8]
    mov x1, #10
    udiv x2, x4, x1
    msub x0, x2, x1, x4         /* frac%10 */
    add w0, w0, #'0'
    bl print_ch
    ldr x30, [sp], #16
    ret
.Lpf_nan:
    adrp x0, s_nan
    add x0, x0, :lo12:s_nan
    ldr x30, [sp], #16
    b print_cstr
.Lpf_inf:
    adrp x0, s_inf
    add x0, x0, :lo12:s_inf
    ldr x30, [sp], #16
    b print_cstr
.Lpf_ninf:
    adrp x0, s_ninf
    add x0, x0, :lo12:s_ninf
    ldr x30, [sp], #16
    b print_cstr

/* print_nl: */
print_nl:
    mov w0, #'\n'
    b print_ch

/* ---------------- expf: glibc 2.39 __expf_finite mirror (s0 -> s0) ----------
 * Fast path: d = xd*INV32LN2; n=frinta/fcvtas(d); r=d-n;
 *   p = 1 + c1*r + (c2 + c3*r)*r*r  (c1..c3 from libm @+320,+312,+304)
 *   scale bits = table[n&31] + (n<<47); s0 = (f32)(p * bits)
 * Thresholds (probe-verified): x > 88.72289f -> +inf;
 *   x < -103.97208f -> +0; [-103.97208, -103.49402) -> min subnormal;
 *   -inf -> +0; NaN/inf -> s0+s0.
 * Clobbers x0-x4, d0-d5, s0-s5. */
    .align 2
expf:
    fmov w0, s0                     /* bits */
    fcvt d1, s0
    ubfx x1, x0, #20, #11
    cmp w1, #0x42a
    b.hi .Lexp_slow
.Lexp_fast:
    adrp x0, k_exp_table
    add x0, x0, :lo12:k_exp_table
    fmov d0, #1.0
    ldr d3, [x0, #296]              /* 46.166241308446828 = 32/ln2 */
    ldr d4, [x0, #304]              /* c3 */
    ldr d2, [x0, #312]              /* c2 */
    fmul d1, d1, d3
    ldr d3, [x0, #320]              /* c1 */
    frinta d5, d1
    fcvtas x1, d1
    fsub d1, d1, d5                 /* r */
    and x2, x1, #0x1f
    fmadd d2, d4, d1, d2            /* c3*r + c2 */
    fmadd d0, d3, d1, d0            /* c1*r + 1 */
    fmul d1, d1, d1                 /* r*r */
    ldr x0, [x0, x2, lsl #3]
    add x1, x0, x1, lsl #47
    fmadd d0, d2, d1, d0            /* poly */
    fmov d1, x1                     /* scale bits */
    fmul d0, d0, d1
    fcvt s0, d0
    ret
.Lexp_slow:
    mov w2, #0xff80
    lsl w2, w2, #16                 /* -inf bits */
    cmp w0, w2
    b.eq .Lexp_ninf
    cmp w1, #0x7f7
    b.hi .Lexp_naninf               /* NaN/+inf -> s0+s0 */
    mov w2, #0x7217
    movk w2, #0x42b1, lsl #16       /* 88.72289f */
    fmov s2, w2
    fcmp s0, s2
    b.gt .Lexp_inf
    mov w2, #0xf1b4
    movk w2, #0xc2cf, lsl #16       /* -103.97208f */
    fmov s2, w2
    fcmp s0, s2
    b.lt .Lexp_zero
    mov w2, #0x8ecf
    movk w2, #0xc2ce, lsl #16       /* -103.49402f */
    fmov s2, w2
    fcmp s0, s2
    b.lt .Lexp_minsub
    b .Lexp_fast
.Lexp_ninf:
    fmov s0, wzr
    ret
.Lexp_naninf:
    fadd s0, s0, s0
    ret
.Lexp_inf:
    mov w0, #0x7f80
    lsl w0, w0, #16
    fmov s0, w0
    ret
.Lexp_zero:
    fmov s0, wzr
    ret
.Lexp_minsub:
    mov w0, #1
    fmov s0, w0
    ret

/* ---------------- h2f: fp16 bits (w0) -> f32 (s0). Mirrors fwd_k2.c h2f.
 * e==0,m==0 -> +0.0 (sign dropped, as C does); e==0,m -> m*2^-24 | s;
 * e==31 -> inf/nan bits; else biased-exponent rebuild. Clobbers w1-w4,s1. */
    .align 2
h2f:
    and w1, w0, #0x3ff             /* m */
    ubfx w2, w0, #10, #5           /* e */
    and w3, w0, #0x8000            /* isolate sign FIRST (C: (h & 0x8000) << 16) */
    lsl w3, w3, #16                /* sign -> bit31 */
    cbnz w2, 1f
    cbz w1, 0f                     /* e==0 m==0 -> +0.0 */
    ucvtf s0, w1
    mov w4, #0x33800000            /* 2^-24 */
    fmov s1, w4
    fmul s0, s0, s1
    fmov w4, s0
    orr w4, w4, w3
    fmov s0, w4
    ret
0:  fmov s0, wzr
    ret
1:  cmp w2, #31
    b.ne 2f
    mov w4, #0x7f80
    lsl w4, w4, #16                /* 0x7f800000 */
    orr w4, w4, w3
    lsl w1, w1, #13
    orr w4, w4, w1
    fmov s0, w4
    ret
2:  add w2, w2, #112
    lsl w2, w2, #23
    orr w4, w3, w2
    lsl w1, w1, #13
    orr w4, w4, w1
    fmov s0, w4
    ret

/* ---------------- k2_q8_quant: x0=float*, x1=dst, x2=k (elems, /256).
 * Mirrors fwd_k2.c: per 256-block max-scan, iscale=-127/max, magic
 * nearest_int (+12582912.f bit trick), clamp <127, d=1/iscale at yb[0..4),
 * qs at yb+4. bsums SKIPPED (never read by k2q4_gemv_range; 292-stride kept).
 * Clobbers x0-x7, w0-w7, s0-s5. */
    .align 2
k2_q8_quant:
    cbz x2, 9f
    mov x3, #0                      /* block index */
    lsr x8, x2, #8                  /* nb = k/256 (x2 gets clobbered below!) */
    mov x7, #292
0:  /* max scan over 256 */
    mov x4, #0                      /* j */
    fmov s2, wzr                    /* amax */
    fmov s3, wzr                    /* max */
1:  ldr s0, [x0, x4, lsl #2]
    fabs s1, s0
    fcmp s1, s2
    b.ls 2f
    fmov s2, s1
    fmov s3, s0
2:  add x4, x4, #1
    cmp x4, #256
    b.ne 1b
    /* yb = x1 + block*292 */
    madd x5, x3, x7, x1             /* yb */
    fcmp s2, #0.0
    b.ne 3f
    /* zero block: d=0, qs=0 */
    str wzr, [x5]
    mov x4, #0
    mov x6, #0
4:  str x6, [x5, x4]
    add x4, x4, #8
    cmp x4, #264                    /* covers d(4)+qs(256) */
    b.ne 4b
    b 8f
3:  /* iscale = -127/max */
    mov w4, #0xc2fe0000
    fmov s4, w4          /* -127.0f */
    fdiv s4, s4, s3
    /* qs[j] = clamp(nearest_int(iscale*x[j])) */
    mov x4, #0
    add x6, x5, #4                  /* qs */
5:  ldr s0, [x0, x4, lsl #2]
    mov w2, #0x4b40                 /* 12582912.f = 0x4B400000 */
    lsl w2, w2, #16
    fmov s1, w2
    fmadd s0, s0, s4, s1             /* FUSED x*iscale + magic (gcc contracts
                                        `iscale*x + 12582912.f` to one fmadd;
                                        separate fmul+fadd double-rounds and
                                        can flip near-.5 values by 1) */
    fmov w2, s0
    and w2, w2, #0x007fffff
    sub w2, w2, #0x00400000         /* nearest_int */
    mov w10, #127                   /* NOT w4: x4 = loop index j! */
    cmp w2, #127
    csel w2, w2, w10, lt            /* v < 127 ? v : 127 */
    strb w2, [x6, x4]
    add x4, x4, #1
    cmp x4, #256
    b.ne 5b
    /* d = 1.0/iscale */
    fmov s0, #1.0
    fdiv s0, s0, s4
    str s0, [x5]
8:  add x3, x3, #1
    add x0, x0, #1024
    cmp x3, x8                      /* block < nb */
    b.ne 0b
9:  ret

/* ---------------- k2q4_gemv_range (C signature):
 *   w0=r0, w1=r1, w2=cols, x3=w, x4=x(unused), x5=y, x6=xq
 * Per row: acc = sum_b fma(round(ws*db), int32 nibble-dot, acc) — mirrors
 * the gcc contract of `acc += (ws*db)*(float)sum` into fmul+fmadd.
 * Live across blocks: x8=row, x9=nb32, x11=rowptr, x14=b, x15=r1, x5=y,
 * x6=xq, s4=acc, s5=ws. Clobbers x0-x4,x7,x12,x13, w0-w4, v0-v5. */
    .align 2
k2q4_gemv_range:
    cmp w0, w1
    b.ge 9f
    asr x9, x2, #5                  /* nb32 = cols/32 */
    mov x7, #18
    mul x10, x9, x7                 /* row stride = nb32*18 */
    madd x11, x0, x10, x3           /* row ptr */
    mov x7, #292                    /* xq block stride (hoisted) */
    mov x8, x0                      /* r */
    mov x15, x1                     /* r1 */
    movi v7.16b, #15
    movi v6.16b, #8
0:  /* ---- row ---- */
    mov x14, #0                     /* b */
    fmov s8, wzr                    /* row acc: reg8, immune to all integer and v0-v7 writes */
1:  /* ---- 32-weight block ---- */
    ldrh w0, [x11]
    add x11, x11, #18
    /* inline h2f (raw fp16 scale -> s0); scratch w1-w4, s0-s1 */
    and w1, w0, #0x3ff
    ubfx w2, w0, #10, #5
    and w3, w0, #0x8000            /* MASK before shift: raw lsl drags exp bits up! */
    lsl w3, w3, #16
    cbnz w2, 2f
    cbz w1, 3f
    ucvtf s0, w1
    mov w4, #0x33800000             /* 2^-24 */
    fmov s1, w4
    fmul s0, s0, s1
    fmov w4, s0
    orr w4, w4, w3
    fmov s0, w4
    b 4f
3:  fmov s0, wzr
    b 4f
2:  cmp w2, #31
    b.ne 5f
    mov w4, #0x7f80
    lsl w4, w4, #16
    orr w4, w4, w3
    lsl w1, w1, #13
    orr w4, w4, w1
    fmov s0, w4
    b 4f
5:  add w2, w2, #112
    lsl w2, w2, #23
    orr w4, w3, w2
    lsl w1, w1, #13
    orr w4, w4, w1
    fmov s0, w4
4:  fmov s5, s0                    /* ws (keep out of dot regs) */
    /* xq: wb = b>>3, woff = (b&7)<<5 */
    lsr w1, w14, #3
    and w2, w14, #7
    madd x3, x1, x7, x6             /* xq + wb*292 */
    ldr s15, [x3]                   /* db (reg15 written only at entry) */
    add x3, x3, #4
    add x3, x3, x2, lsl #5          /* + woff */
    sub x4, x11, #16                /* codes (rowptr-18+2) */
    ld1 {v0.16b}, [x4]
    and v1.16b, v0.16b, v7.16b
    ushr v0.16b, v0.16b, #4
    sub v1.16b, v1.16b, v6.16b      /* lo-8 */
    sub v0.16b, v0.16b, v6.16b      /* hi-8 */
    ld1 {v2.16b}, [x3]
    ldr q3, [x3, #16]
    movi v4.4s, #0
    sdot v4.4s, v1.16b, v2.16b
    sdot v4.4s, v0.16b, v3.16b
    addv s2, v4.4s
    scvtf s2, s2
    fmul s3, s5, s15                /* round(ws*db) */
    fmadd s8, s3, s2, s8            /* fused acc (reg8: no integer/v0-v7 writer) */
    add x14, x14, #1                /* b++ (x14 = reg14 ≠ s8) */
    cmp x14, x9
    b.ne 1b
    str s8, [x5, x8, lsl #2]
    add x8, x8, #1
    cmp x8, x15
    b.ne 0b
9:  ret

/* ---------------- rms: x0=x, x1=w(norm), x2=n, x3=y.
 * f64 fmadd accumulate, divide by n, f32 +1e-6f, fsqrt (always >0),
 * sc=1.0f/sqrt (fdiv), y = (sc*x)*w — mirrors fwd_k2.c exactly.
 * Clobbers x4, s0-s4, d0-d2. */
    .align 2
rms:
    fmov d1, xzr
    mov x4, #0
0:  ldr s0, [x0, x4, lsl #2]
    fcvt d0, s0
    fmadd d1, d0, d0, d1
    add x4, x4, #1
    cmp x4, x2
    b.ne 0b
    scvtf d0, w2
    fdiv d1, d1, d0                 /* s/n */
    fcvt s0, d1
    adrp x4, f_1em6
    ldr s1, [x4, :lo12:f_1em6]
    fadd s0, s0, s1                 /* + 1e-6f */
    fsqrt s0, s0                    /* always > 0 */
    fmov s1, #1.0
    fdiv s1, s1, s0                 /* sc = 1/sqrt */
    mov x4, #0
1:  ldr s0, [x0, x4, lsl #2]
    ldr s2, [x1, x4, lsl #2]
    fmul s0, s1, s0
    fmul s0, s0, s2
    str s0, [x3, x4, lsl #2]
    add x4, x4, #1
    cmp x4, x2
    b.ne 1b
    ret

/* ---------------- blob_load: x0 = path. mmap K2H1 blob, verify geometry,
 * compute tensor offsets into globals. Error -> message + exit(1). */
    .align 2
blob_load:
    stp x29, x30, [sp, #-32]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    mov x19, x0                     /* path */
    mov x8, #56                     /* openat */
    mov x0, #-100                   /* AT_FDCWD */
    mov x1, x19
    mov x2, #0                      /* O_RDONLY */
    svc #0
    cmp x0, #0
    b.lt .Lerr_open
    mov x20, x0                     /* fd */
    mov x8, #62                     /* lseek end */
    mov x1, #0
    mov x2, #2
    svc #0
    cmp x0, #0
    b.le .Lerr_seek
    mov x19, x0                     /* size */
    mov x8, #62                     /* lseek start */
    mov x0, x20
    mov x1, #0
    mov x2, #0
    svc #0
    mov x8, #222                    /* mmap */
    mov x0, #0
    mov x1, x19
    mov x2, #1                      /* PROT_READ */
    mov x3, #2                      /* MAP_PRIVATE */
    mov x4, x20
    mov x5, #0
    svc #0
    cmp x0, #0
    b.lt .Lerr_mmap
    mov x19, x0                     /* blob base */
    mov x8, #57                     /* close */
    mov x0, x20
    svc #0
    /* magic "K2H1" = 0x3148324B; header: +4 nl, +8 hid, +12 inter,
       +16 nhead, +20 nkv, +24 hdim, +28 vocab, +32 wtype, +36 npos */
    ldr w0, [x19]
    mov w1, #0x324b
    movk w1, #0x3148, lsl #16
    cmp w0, w1
    b.ne .Lerr_magic
    ldp w0, w2, [x19, #4]           /* nl, hid */
    cmp w0, #28
    b.ne .Lerr_geom
    cmp w2, #1536
    b.ne .Lerr_geom
    ldp w0, w2, [x19, #16]          /* nhead, nkv */
    cmp w0, #32
    b.ne .Lerr_geom
    cmp w2, #8
    b.ne .Lerr_geom
    ldr w0, [x19, #24]              /* hdim */
    cmp w0, #64
    b.ne .Lerr_geom
    ldr w0, [x19, #32]              /* wtype */
    cmp w0, #2
    b.ne .Lerr_wtype                /* asm runtime: Q4 blobs only */
    ldr w0, [x19, #28]              /* vocab */
    cmp w0, #131072
    b.hi .Lerr_geom
    adrp x1, g_vocab
    str w0, [x1, :lo12:g_vocab]
    ldr w0, [x19, #36]              /* npos */
    adrp x1, g_npos
    str w0, [x1, :lo12:g_npos]
    adrp x1, g_blob
    str x19, [x1, :lo12:g_blob]
    /* offsets (Q4 layout of k2_convert.py): 28 layers x 17707008 B */
    movz x0, #0x010e, lsl #16       /* 17707008 = 0x010E3000 */
    movk x0, #0x3000
    mov x1, #28
    mul x2, x0, x1                  /* layer bytes total */
    add x3, x19, #40                /* first layer */
    adrp x1, g_LBASE
    str x3, [x1, :lo12:g_LBASE]     /* layer 0 base */
    add x4, x3, x2                  /* layer-end = O_EMB */
    adrp x1, g_O_EMB
    str x4, [x1, :lo12:g_O_EMB]
    ldr w5, [x19, #28]              /* vocab */
    mov x0, #3072                   /* HID*2 (fp16 embed row) */
    madd x6, x5, x0, x4             /* O_NORM = O_EMB + vocab*3072 */
    adrp x1, g_O_NORM
    str x6, [x1, :lo12:g_O_NORM]
    mov x7, #6144
    add x6, x6, x7                  /* + HID*4 */
    adrp x1, g_O_HEAD
    str x6, [x1, :lo12:g_O_HEAD]
    /* head bytes = (vocab*1536/32)*18 = vocab*864 */
    mov x0, #864
    madd x7, x5, x0, x6             /* O_COS */
    adrp x1, g_O_COS
    str x7, [x1, :lo12:g_O_COS]
    ldr w0, [x19, #36]              /* npos: COS bytes = npos*32*4 */
    lsl x0, x0, #7                  /* npos*128 */
    add x0, x7, x0
    adrp x1, g_O_SIN
    str x0, [x1, :lo12:g_O_SIN]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #32
    ret
.Lerr_open:
    adrp x0, m_open
    add x0, x0, :lo12:m_open
    b .Lfatal
.Lerr_seek:
    adrp x0, m_seek
    add x0, x0, :lo12:m_seek
    b .Lfatal
.Lerr_mmap:
    adrp x0, m_mmap
    add x0, x0, :lo12:m_mmap
    b .Lfatal
.Lerr_magic:
    adrp x0, m_magic
    add x0, x0, :lo12:m_magic
    b .Lfatal
.Lerr_geom:
    adrp x0, m_geom
    add x0, x0, :lo12:m_geom
    b .Lfatal
.Lerr_wtype:
    adrp x0, m_wtype
    add x0, x0, :lo12:m_wtype
    b .Lfatal

/* .Lfatal: x0 = asciz message -> print to stderr-ish (stdout) + exit(1) */
.Lfatal:
    bl print_cstr
    mov w0, #'\n'
    bl print_ch
    mov x0, #1
    mov x8, #SYS_EXIT
    svc #0


/* ---------------- embed: w0 = token id -> X[1536] (fp16 rows via h2f).
 * h2f clobbers w0-w4/s0-s1 -> ptrs live in x19-x21 (saved here). */
    .align 2
embed:
    stp x29, x30, [sp, #-48]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    str x21, [sp, #32]
    adrp x1, g_O_EMB
    ldr x1, [x1, :lo12:g_O_EMB]
    mov x4, #3072                    /* HID*2 (NOT a shift: 3072 != 2^11) */
    madd x19, x0, x4, x1            /* src = O_EMB + id*3072 */
    adrp x20, b_X
    add x20, x20, :lo12:b_X
    mov x21, #1536
0:  ldrh w0, [x19], #2
    bl h2f
    str s0, [x20], #4
    subs x21, x21, #1
    b.ne 0b
    ldr x21, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #48
    ret


/* ---------------- gemvW: w0=il, w1=k, x2=x, x3=y.
 * w = g_LBASE + il*17707008 + koff[k]; q8_quant(x, XQQ, cols) then
 * q4_gemv(0, rows, cols, w, x, y, XQQ). Saves x19-x24. */
    .align 2
gemvW:
    stp x29, x30, [sp, #-64]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    mov x19, x1                     /* k */
    mov x20, x2                     /* x act */
    mov x21, x3                     /* y */
    mov x22, x0                     /* il */
    /* w ptr = LBASE + il*17707008 + koff[k] */
    adrp x1, g_LBASE
    ldr x2, [x1, :lo12:g_LBASE]
    movz x0, #0x010e, lsl #16       /* 17707008 = 0x010E3000 */
    movk x0, #0x3000
    madd x2, x22, x0, x2
    adrp x1, tbl_koff
    add x1, x1, :lo12:tbl_koff
    ldr x0, [x1, x19, lsl #3]
    add x24, x2, x0                 /* w */
    adrp x1, tbl_krows
    add x1, x1, :lo12:tbl_krows
    ldr w23, [x1, x19, lsl #2]      /* rows */
    adrp x1, tbl_kcols
    add x1, x1, :lo12:tbl_kcols
    ldr w22, [x1, x19, lsl #2]      /* cols */
    /* q8_quant(x, XQQ, cols) */
    mov x0, x20
    adrp x1, b_XQQ
    add x1, x1, :lo12:b_XQQ
    mov w2, w22
    bl k2_q8_quant
    /* q4_gemv(0, rows, cols, w, x, y, XQQ) */
    mov w0, wzr
    mov w1, w23
    mov w2, w22
    mov x3, x24
    mov x4, x20
    mov x5, x21
    adrp x6, b_XQQ
    add x6, x6, :lo12:b_XQQ
    bl q4_gemv_dispatch
    ldp x19, x20, [sp, #16]
    ldp x21, x22, [sp, #32]
    ldp x23, x24, [sp, #48]
    ldp x29, x30, [sp], #64
    ret


/* ---------------- rope: x0 = buf (QF or KF), w1 = nheads.
 * C/S = O_COS/O_SIN + pos*128 (per-position 32 f32 tables).
 * Exact pair contraction (j, j+32), mirror of gcc codegen:
 *   t  = b*Sj (fmul);   xh[j]   = fnmsub(Cj, a, t)  = Cj*a - t
 *   t2 = b*Cj (fmul);   xh[j+32] = fmadd(Sj, a, t2) = Sj*a + t2
 * Saves x19-x26. */
    .align 2
rope:
    stp x29, x30, [sp, #-80]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    mov x19, x0                      /* buf */
    mov w20, w1                      /* nheads */
    adrp x1, g_O_COS
    ldr x2, [x1, :lo12:g_O_COS]
    adrp x1, g_pos
    ldr x3, [x1, :lo12:g_pos]
    add x21, x2, x3, lsl #7          /* C = O_COS + pos*128 */
    adrp x1, g_O_SIN
    ldr x2, [x1, :lo12:g_O_SIN]
    add x22, x2, x3, lsl #7          /* S */
    mov w23, wzr                      /* h */
.Lrope_head:
    mov x0, #256
    madd x24, x23, x0, x19           /* xh = buf + h*256 */
    add x25, x24, #128               /* xh + 32 floats */
    mov w26, wzr                      /* j */
.Lrope_j:
    ldr s0, [x24, x26, lsl #2]       /* a */
    ldr s1, [x25, x26, lsl #2]       /* b */
    ldr s2, [x21, x26, lsl #2]       /* Cj */
    ldr s3, [x22, x26, lsl #2]       /* Sj */
    fmul s4, s1, s3                  /* t = b*Sj */
    fnmsub s5, s2, s0, s4            /* Cj*a - t */
    str s5, [x24, x26, lsl #2]       /* xh[j] */
    fmul s4, s1, s2                  /* t2 = b*Cj */
    fmadd s5, s3, s0, s4             /* Sj*a + t2 */
    str s5, [x25, x26, lsl #2]       /* xh[j+32] */
    add w26, w26, #1
    cmp w26, #32
    b.ne .Lrope_j
    add w23, w23, #1
    cmp w23, w20
    b.ne .Lrope_head
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #80
    ret

/* ---------------- attention: w0 = il. 32 heads, GQA kh=h>>2.
 * scores: scalar fmul+fadd in element order (== gcc's 4-wide lane order),
 * sc = s*0.125, strict-> mx; es f64 sum of expf; p = expf/(f32)es;
 * o[64] zeroed then fmla (fused, as C contracts) from V rows.
 * K/V row for (il,t) = base + (il*CTX + t)*2048; + kh*256 bytes.
 * Saves x19-x27. */
    .align 2
attention:
    stp x29, x30, [sp, #-96]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    str x27, [sp, #80]
    mov w19, w0                      /* il */
    adrp x1, g_pos
    ldr x20, [x1, :lo12:g_pos]       /* pos (loop bound, inclusive) */
    /* K/V per-layer base = g_KCA + il*3145728 (CTX=1536*512 floats*4) */
    mov x0, #3145728
    madd x22, x19, x0, xzr
    adrp x1, g_KCA
    ldr x2, [x1, :lo12:g_KCA]
    add x22, x2, x22                 /* KCA layer base */
    adrp x1, g_VCA
    ldr x2, [x1, :lo12:g_VCA]
    madd x23, x19, x0, xzr           /* il*3145728 */
    add x23, x2, x23                 /* VCA layer base */
    adrp x25, b_attsc
    add x25, x25, :lo12:b_attsc      /* sc[CTX] */
    mov w21, wzr                     /* h */
.Lattn_head:
    /* Q ptr = b_QF + h*256; ATTO = b_ATTO + h*256; khoff = (h>>2)<<8 */
    adrp x1, b_QF
    add x1, x1, :lo12:b_QF
    mov x0, #256
    madd x24, x21, x0, x1           /* Q head */
    adrp x1, b_ATTO
    add x1, x1, :lo12:b_ATTO
    madd x26, x21, x0, x1           /* ATTO head */
    lsr w2, w21, #2
    lsl x2, x2, #8                   /* khoff bytes */
    /* zero ATTO head: 256 B */
    movi v0.2d, #0
    stp q0, q0, [x26]
    stp q0, q0, [x26, #32]
    stp q0, q0, [x26, #64]
    stp q0, q0, [x26, #96]
    stp q0, q0, [x26, #128]
    stp q0, q0, [x26, #160]
    stp q0, q0, [x26, #192]
    stp q0, q0, [x26, #224]
    adrp x1, f_n1e30
    ldr s11, [x1, :lo12:f_n1e30]     /* mx = -1e30f */
    /* ---- scores: t = 0..pos ---- */
    mov x27, xzr
.Lattn_score_t:
    add x9, x22, x27, lsl #11        /* KCA row */
    add x9, x9, x2                   /* + khoff */
    mov x10, x24                     /* Q walk */
    fmov s4, wzr                     /* dot acc (element order) */
    mov x11, #64
.Lattn_score_i:
    ldr s0, [x10], #4
    ldr s1, [x9], #4
    fmul s2, s0, s1
    fadd s4, s4, s2
    subs x11, x11, #1
    b.ne .Lattn_score_i
    fmov s7, #0.125
    fmul s0, s4, s7                  /* sc = s*0.125 */
    str s0, [x25, x27, lsl #2]
    fcmp s0, s11
    b.le .Lattn_score_next           /* update iff sc > mx (C: if (sc > mx)) */
    fmov s11, s0                     /* mx = sc  (b.le also catches NaN -> keep) */
.Lattn_score_next:
    add x27, x27, #1
    cmp x27, x20
    b.ls .Lattn_score_t              /* t <= pos */
    /* ---- es = sum expf(sc-mx) in f64 ---- */
    fmov d10, xzr
    mov x27, xzr
.Lattn_es_t:
    ldr s0, [x25, x27, lsl #2]
    fsub s0, s0, s11
    bl expf
    fcvt d0, s0
    fadd d10, d10, d0
    add x27, x27, #1
    cmp x27, x20
    b.ls .Lattn_es_t
    fcvt s10, d10                    /* (f32)es */
    /* ---- p-loop: o += p * V ---- */
    mov x27, xzr
.Lattn_p_t:
    ldr s0, [x25, x27, lsl #2]
    fsub s0, s0, s11
    bl expf
    fdiv s2, s0, s10                 /* p */
    add x9, x23, x27, lsl #11        /* VCA row */
    lsr w3, w21, #2
    lsl x3, x3, #8                   /* khoff (x2 dead across expf) */
    add x9, x9, x3
    mov x11, x26                     /* o walk */
    dup v2.4s, v2.s[0]
    mov x12, #16                     /* 16 x 16B = 64 floats */
.Lattn_p_i:
    ld1 {v0.4s}, [x11]
    ld1 {v1.4s}, [x9]
    fmla v0.4s, v2.4s, v1.4s
    st1 {v0.4s}, [x11], #16
    add x9, x9, #16
    subs x12, x12, #1
    b.ne .Lattn_p_i
    add x27, x27, #1
    cmp x27, x20
    b.ls .Lattn_p_t
    add w21, w21, #1
    cmp w21, #32
    b.ne .Lattn_head
    ldr x27, [sp, #80]
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #96
    ret

/* ---------------- layer_forward: w0 = il. Full decoder layer.
 * rms->qkv->rope->KV store->attn->o->resid->rms->gate/up->silu->down->resid
 * Saves x19-x22. */
    .align 2
layer_forward:
    stp x29, x30, [sp, #-48]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    mov w19, w0                      /* il */
    adrp x1, g_LBASE
    ldr x20, [x1, :lo12:g_LBASE]
    movz x0, #0x010e, lsl #16       /* 17707008 = 0x010E3000 */
    movk x0, #0x3000
    madd x20, x19, x0, x20           /* layer base */
    /* rms(X, in_norm, 1536, XN) */
    adrp x0, b_X
    add x0, x0, :lo12:b_X
    mov x1, x20                      /* in_norm at +0 */
    mov w2, #1536
    adrp x3, b_XN
    add x3, x3, :lo12:b_XN
    bl rms
    /* q, k, v */
    mov w0, w19
    mov w1, #1
    adrp x2, b_XN
    add x2, x2, :lo12:b_XN
    adrp x3, b_QF
    add x3, x3, :lo12:b_QF
    bl gemvW
    mov w0, w19
    mov w1, #2
    adrp x2, b_XN
    add x2, x2, :lo12:b_XN
    adrp x3, b_KF
    add x3, x3, :lo12:b_KF
    bl gemvW
    mov w0, w19
    mov w1, #3
    adrp x2, b_XN
    add x2, x2, :lo12:b_XN
    adrp x3, b_VF
    add x3, x3, :lo12:b_VF
    bl gemvW
    /* rope Q (32 heads), K (8 heads) */
    adrp x0, b_QF
    add x0, x0, :lo12:b_QF
    mov w1, #32
    bl rope
    adrp x0, b_KF
    add x0, x0, :lo12:b_KF
    mov w1, #8
    bl rope
    /* KV store: KCA[il][pos] = KF; VCA[il][pos] = VF (2048 B each) */
    mov x0, #3145728
    madd x21, x19, x0, xzr           /* il*3145728 */
    adrp x1, g_KCA
    ldr x1, [x1, :lo12:g_KCA]
    add x1, x1, x21
    adrp x2, g_pos
    ldr x2, [x2, :lo12:g_pos]
    add x1, x1, x2, lsl #11          /* dst = KCA + (il layer) + pos*2048 */
    adrp x0, b_KF
    add x0, x0, :lo12:b_KF
    mov x2, #64
1:  ldp q0, q1, [x0], #32
    stp q0, q1, [x1], #32
    subs x2, x2, #1
    b.ne 1b
    adrp x1, g_VCA
    ldr x1, [x1, :lo12:g_VCA]
    add x1, x1, x21
    adrp x2, g_pos
    ldr x2, [x2, :lo12:g_pos]
    add x1, x1, x2, lsl #11
    adrp x0, b_VF
    add x0, x0, :lo12:b_VF
    mov x2, #64
1:  ldp q0, q1, [x0], #32
    stp q0, q1, [x1], #32
    subs x2, x2, #1
    b.ne 1b
    /* attention */
    mov w0, w19
    bl attention
    /* o proj + residual */
    mov w0, w19
    mov w1, #4
    adrp x2, b_ATTO
    add x2, x2, :lo12:b_ATTO
    adrp x3, b_AO
    add x3, x3, :lo12:b_AO
    bl gemvW
    adrp x0, b_X
    add x0, x0, :lo12:b_X
    adrp x1, b_AO
    add x1, x1, :lo12:b_AO
    bl vec_add1536
    /* rms(X, post_norm) */
    adrp x0, b_X
    add x0, x0, :lo12:b_X
    mov x1, x20
    movz x7, #0x43, lsl #16         /* 4429824 = 0x439800 */
    movk x7, #0x9800
    add x1, x1, x7                  /* post_attention_layernorm */
    mov w2, #1536
    adrp x3, b_XN
    add x3, x3, :lo12:b_XN
    bl rms
    /* gate, up */
    mov w0, w19
    mov w1, #6
    adrp x2, b_XN
    add x2, x2, :lo12:b_XN
    adrp x3, b_LG
    add x3, x3, :lo12:b_LG
    bl gemvW
    mov w0, w19
    mov w1, #7
    adrp x2, b_XN
    add x2, x2, :lo12:b_XN
    adrp x3, b_LU
    add x3, x3, :lo12:b_LU
    bl gemvW
    /* silu: LU[i] = (LG[i] / (1.0f + expf(-LG[i]))) * LU[i] */
    adrp x20, b_LG
    add x20, x20, :lo12:b_LG
    adrp x21, b_LU
    add x21, x21, :lo12:b_LU
    mov x22, #5120
2:  ldr s10, [x20], #4
    fneg s0, s10
    bl expf
    fmov s1, #1.0
    fadd s0, s0, s1
    fdiv s0, s10, s0
    ldr s1, [x21]
    fmul s0, s0, s1
    str s0, [x21], #4
    subs x22, x22, #1
    b.ne 2b
    /* down + residual */
    mov w0, w19
    mov w1, #8
    adrp x2, b_LU
    add x2, x2, :lo12:b_LU
    adrp x3, b_FO
    add x3, x3, :lo12:b_FO
    bl gemvW
    adrp x0, b_X
    add x0, x0, :lo12:b_X
    adrp x1, b_FO
    add x1, x1, :lo12:b_FO
    bl vec_add1536
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #48
    ret

/* ---------------- vec_add1536: x0 = dst, x1 = src. dst += src (f32, 4-wide,
 * elementwise -> order-independent, bit-exact vs scalar fadd). */
    .align 2
vec_add1536:
    mov x2, #0
0:  ldr q0, [x0, x2]
    ldr q1, [x1, x2]
    fadd v0.4s, v0.4s, v1.4s
    str q0, [x0, x2]
    add x2, x2, #16
    mov x4, #6144
    cmp x2, x4
    b.ne 0b
    ret

/* ---------------- argmax: x0 = buf, w1 = n -> w0 = index (strict >,
 * first-max wins — mirrors C argmax). */
    .align 2
argmax:
    mov w2, wzr
    ldr s1, [x0]
    mov x3, #1
0:  ldr s0, [x0, x3, lsl #2]
    fcmp s0, s1
    b.ls 1f                          /* ordered <= -> keep old (first wins) */
    fmov s1, s0
    mov w2, w3
1:  add x3, x3, #1
    cmp x3, x1
    b.ne 0b
    mov w0, w2
    ret

/* ---------------- frand01 -> s0. LCG mirror of fwd_k2.c (exact u64). */
    .align 2
frand01:
    adrp x1, g_RS
    ldr x2, [x1, :lo12:g_RS]
    movz x3, #0x7f2d
    movk x3, #0x4c95, lsl #16
    movk x3, #0xf42d, lsl #32
    movk x3, #0x5851, lsl #48        /* 6364136223846793005 */
    mul x2, x2, x3
    movz x3, #0x814f
    movk x3, #0xf767, lsl #16
    movk x3, #0x7b7e, lsl #32
    movk x3, #0x1405, lsl #48        /* 1442695040888963407 */
    add x2, x2, x3
    str x2, [x1, :lo12:g_RS]
    lsr x2, x2, #11
    ucvtf d0, x2                     /* exact: <=2^53 */
    adrp x1, f_2pm53
    ldr d1, [x1, :lo12:f_2pm53]
    fmul d0, d0, d1
    fcvt s0, d0
    ret

/* ---------------- sample_topkpp: x0=lg, w1=n, s0=temp, s1=topp, w2=topk,
 * s3=minp -> w0 index. Mirrors fwd_k2.c: descending top-k insertion
 * (v <= sc[k] advances -> equal keeps earlier index), then expf renorm in
 * f64 with topp cut, then LCG draw over keep-prefix exp2 weights.
 * minp > 0 unsupported (logf) -> fatal at parse time instead.
 * Saves x19-x26; keeps temps in s12-s14 (safe across my expf: s0-s5 only). */
    .align 2
sample_topkpp:
    stp x29, x30, [sp, #-80]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    mov x19, x0                      /* lg */
    mov w20, w1                      /* n */
    fmov s12, s0                     /* temp */
    fmov s13, s1                     /* topp */
    mov w21, w2                      /* topk */
    fmov s14, s3                     /* minp */
    fcmp s12, #0.0
    b.gt 0f
    mov x0, x19
    mov w1, w20
    bl argmax
    b .Lsmp_done
0:  /* topk clamp: <=0 || >1024 -> 1024; > n -> n */
    cmp w21, #0
    b.le 1f
    cmp w21, #1024
    b.le 2f
1:  mov w21, #1024
2:  cmp w21, w20
    ble 3f
    mov w21, w20
3:  /* descending insertion into b_smpsc/b_smsi (keep <= topk) */
    mov w22, wzr                     /* m */
    mov x23, xzr                     /* i */
.Lsmp_ins_i:
    cmp x23, x20
    b.hs .Lsmp_pick
    ldr s0, [x19, x23, lsl #2]      /* v */
    mov w24, wzr                     /* k */
.Lsmp_ins_k:
    cmp w24, w22
    b.hs .Lsmp_ins_kdone
    adrp x1, b_smpsc
    add x1, x1, :lo12:b_smpsc
    ldr s1, [x1, x24, lsl #2]
    fcmp s0, s1
    b.ls .Lsmp_ins_knext             /* v <= sc[k] -> advance (strict earlier-wins) */
    b .Lsmp_ins_kdone
.Lsmp_ins_knext:
    add w24, w24, #1
    b .Lsmp_ins_k
.Lsmp_ins_kdone:
    cmp w24, w21
    b.hs .Lsmp_ins_inext             /* k >= topk -> skip element */
    /* shift [min(m, topk-1) .. k+1] down */
    cmp w22, w21
    b.lt 4f
    sub w25, w21, #1                /* start = topk-1 when m >= topk */
    b 5f
4:  mov w25, w22                    /* start = m */
5:
.Lsmp_shift:
    cmp w25, w24
    b.le .Lsmp_shiftdone
    sub w26, w25, #1                /* j-1 */
    adrp x1, b_smpsc
    add x1, x1, :lo12:b_smpsc
    ldr s1, [x1, x26, lsl #2]
    str s1, [x1, x25, lsl #2]       /* sc[j] = sc[j-1] */
    adrp x1, b_smsi
    add x1, x1, :lo12:b_smsi
    ldr w2, [x1, x26, lsl #2]
    str w2, [x1, x25, lsl #2]       /* si[j] = si[j-1] */
    sub w25, w25, #1
    b .Lsmp_shift
.Lsmp_shiftdone:
    /* sc[k] = v; si[k] = i; if (m < topk) m++ */
    adrp x1, b_smpsc
    add x1, x1, :lo12:b_smpsc
    str s0, [x1, x24, lsl #2]
    adrp x1, b_smsi
    add x1, x1, :lo12:b_smsi
    str w23, [x1, x24, lsl #2]
    cmp w22, w21
    b.hs 6f
    add w22, w22, #1
6:
.Lsmp_ins_inext:
    add x23, x23, #1
    b .Lsmp_ins_i
    /* ---- pick ---- */
.Lsmp_pick:
    adrp x1, b_smpsc
    add x1, x1, :lo12:b_smpsc
    ldr s11, [x1]                   /* mx = sc[0] */
    /* es = sum expf(sc[k]-mx), f64 (d8) */
    fmov d8, xzr
    mov w24, wzr
.Lsmp_es:
    cmp w24, w22
    b.hs .Lsmp_esdone
    adrp x1, b_smpsc
    add x1, x1, :lo12:b_smpsc
    ldr s0, [x1, x24, lsl #2]
    fsub s0, s0, s11
    bl expf
    fcvt d0, s0
    fadd d8, d8, d0
    add w24, w24, #1
    b .Lsmp_es
.Lsmp_esdone:
    fcvt d14, s13                    /* topp f64; d14 NOT d1: expf clobbers d0-d5! */
    /* acc walk -> keep */
    fmov d9, xzr
    mov w25, w22                     /* keep = m */
    mov w24, wzr
.Lsmp_acc:
    cmp w24, w22
    b.hs .Lsmp_accdone
    adrp x1, b_smpsc
    add x1, x1, :lo12:b_smpsc
    ldr s0, [x1, x24, lsl #2]
    fsub s0, s0, s11
    bl expf
    fcvt d0, s0
    fdiv d0, d0, d8
    fadd d9, d9, d0
    add w24, w24, #1                 /* k+1 */
    fcmp d9, d14
    b.ge .Lsmp_acck                  /* acc >= topp -> keep = k+1 */
    b .Lsmp_acc
.Lsmp_acck:
    mov w25, w24                     /* keep = k+1 */
    b .Lsmp_accdone
.Lsmp_accdone:
    /* es2 = sum_{k<keep} expf((sc[k]-mx)/temp), f64 (d10) */
    fmov d10, xzr
    mov w24, wzr
.Lsmp_es2:
    cmp w24, w25
    b.hs .Lsmp_es2done
    adrp x1, b_smpsc
    add x1, x1, :lo12:b_smpsc
    ldr s0, [x1, x24, lsl #2]
    fsub s0, s0, s11
    fdiv s0, s0, s12                 /* /temp (f32) */
    bl expf
    fcvt d0, s0
    fadd d10, d10, d0
    add w24, w24, #1
    b .Lsmp_es2
.Lsmp_es2done:
    /* r = frand01() * es2 */
    bl frand01
    fcvt d0, s0
    fmul d13, d0, d10                 /* r; d13 NOT d11: d11 would wipe mx(s11)! */
    /* a2 walk: first k with r < a2, else k == keep-1 */
    fmov d15, xzr          /* NOT d12: fmov d12 would wipe s12 = temp! */
    mov w24, wzr
.Lsmp_a2:
    cmp w24, w25
    b.hs .Lsmp_a2miss
    adrp x1, b_smpsc
    add x1, x1, :lo12:b_smpsc
    ldr s0, [x1, x24, lsl #2]
    fsub s0, s0, s11
    fdiv s0, s0, s12
    bl expf
    fcvt d0, s0
    fadd d15, d15, d0
    fcmp d13, d15                    /* r < a2 ? */
    b.lo .Lsmp_retk
    sub w2, w25, #1
    cmp w24, w2
    b.eq .Lsmp_retk                  /* k == keep-1 */
    add w24, w24, #1
    b .Lsmp_a2
.Lsmp_a2miss:
    sub w24, w25, #1                 /* si[keep-1] fallback */
.Lsmp_retk:
    adrp x1, b_smsi
    add x1, x1, :lo12:b_smsi
    ldr w0, [x1, x24, lsl #2]
.Lsmp_done:
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #80
    ret

/* ---------------- pure-spin thread pool (row-parallel Q4 GEMV) ------------
 * pq2 lesson: never sleep. Workers spin on an acquire-loaded generation
 * counter; row slices are disjoint, so results are bit-exact at any thread
 * count. clone(CLONE_VM|FS|FILES|SIGHAND|THREAD|SYSVSEM) with .bss stacks.
 * w0 = nthreads (1..8); call once after arg parsing. */
    .align 2
pool_init:
    stp x29, x30, [sp, #-32]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    adrp x1, g_nthreads
    str w0, [x1, :lo12:g_nthreads]
    mov w19, #1
.Lpool_spawn:
    adrp x1, g_nthreads
    ldr w20, [x1, :lo12:g_nthreads]
    cmp w19, w20
    b.ge .Lpool_done
    adrp x1, b_stacks
    add x1, x1, :lo12:b_stacks
    mov x2, #65536
    madd x1, x19, x2, x1
    add x1, x1, x2
    sub x1, x1, #16
    mov w0, #0x0f00                 /* CLONE_VM|FS|FILES|SIGHAND|THREAD|SYSVSEM */
    movk w0, #0x5, lsl #16          /* = 0x00050f00 */
    mov x2, #0
    mov x3, #0
    mov x4, #0
    mov x8, #220
    svc #0
    cbz x0, worker_loop             /* child: x0 == 0 */
    add w19, w19, #1
    b .Lpool_spawn
.Lpool_done:
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #32
    ret

worker_loop:
    /* worker id derived from own stack address: (sp+16)/65536 - 1 - base */
    adrp x1, b_stacks
    add x1, x1, :lo12:b_stacks
    sub x2, sp, x1
    add x2, x2, #16
    lsr x2, x2, #16
    sub w19, w2, #1                 /* id 1..N-1 */
    mov w20, #0                     /* last seen gen */
.Lw_spin:
    adrp x2, g_gen
    add x2, x2, :lo12:g_gen
    ldar w2, [x2]
    cmp w2, w20
    b.eq .Lw_idle
    mov w20, w2
    adrp x1, g_nthreads
    ldr w1, [x1, :lo12:g_nthreads]
    adrp x2, g_task_r0
    ldr w2, [x2, :lo12:g_task_r0]
    adrp x3, g_task_r1
    ldr w3, [x3, :lo12:g_task_r1]
    sub w4, w3, w2
    add w4, w4, w1
    sub w4, w4, #1
    udiv w4, w4, w1                  /* chunk = ceil(rows/N) */
    madd w12, w19, w4, w2            /* my r0 (w12/w13: NOT arg regs x5/x6!) */
    cmp w12, w3
    b.ge .Lw_done                    /* empty slice */
    add w13, w12, w4
    cmp w13, w3
    csel w13, w13, w3, lt            /* my r1 */
    adrp x7, g_task_cols
    ldr w2, [x7, :lo12:g_task_cols]
    adrp x7, g_task_w
    ldr x3, [x7, :lo12:g_task_w]
    adrp x7, g_task_x
    ldr x4, [x7, :lo12:g_task_x]
    adrp x7, g_task_y
    ldr x5, [x7, :lo12:g_task_y]
    adrp x7, g_task_xq
    ldr x6, [x7, :lo12:g_task_xq]
    mov w0, w12
    mov w1, w13
    adrp x7, g_task_B
    ldr w7, [x7, :lo12:g_task_B]
    cmp w7, #1
    b.gt .Lw_batch
    bl k2q4_gemv_range
    b .Lw_done
.Lw_batch:
    adrp x7, g_task_rows
    ldr w7, [x7, :lo12:g_task_rows]
    mov x5, x4                      /* y_base (g_task_x) */
    mov x4, x6                      /* xq_base (g_task_xq) */
    adrp x8, g_task_B
    ldr w6, [x8, :lo12:g_task_B]
    bl k2q4_gemv_batch
.Lw_done:
    adrp x1, g_done
    add x1, x1, :lo12:g_done
1:  ldxr w2, [x1]
    add w2, w2, #1
    stlxr w3, w2, [x1]              /* RELEASE: y[] stores visible before done */
    cbnz w3, 1b
    b .Lw_spin
.Lw_idle:
    yield
    b .Lw_spin

/* q4_gemv_dispatch: same args as k2q4_gemv_range. Serial when nthreads==1
 * (bit-identical); otherwise publish + gen++ (release) + own slice + wait. */
    .align 2
q4_gemv_dispatch:
    adrp x9, g_nthreads
    ldr w9, [x9, :lo12:g_nthreads]
    cmp w9, #1
    b.gt .Lq4d_par
    b k2q4_gemv_range               /* serial tail-call */
.Lq4d_par:
    str x30, [sp, #-16]!            /* bl below clobbers lr; ret must return to caller */
    adrp x10, g_task_r0
    str w0, [x10, :lo12:g_task_r0]
    adrp x10, g_task_r1
    str w1, [x10, :lo12:g_task_r1]
    adrp x10, g_task_cols
    str w2, [x10, :lo12:g_task_cols]
    adrp x10, g_task_w
    str x3, [x10, :lo12:g_task_w]
    adrp x10, g_task_x
    str x4, [x10, :lo12:g_task_x]
    adrp x10, g_task_y
    str x5, [x10, :lo12:g_task_y]
    adrp x10, g_task_xq
    str x6, [x10, :lo12:g_task_xq]
    adrp x10, g_task_B
    mov w11, #1                     /* scalar task: workers take the B=1 path */
    str w11, [x10, :lo12:g_task_B]
    adrp x10, g_done
    str wzr, [x10, :lo12:g_done]
    adrp x10, g_gen
    add x10, x10, :lo12:g_gen
    ldr w11, [x10]
    add w11, w11, #1
    stlr w11, [x10]                 /* release: task fields visible */
    /* main takes slice 0 (x12/x13 scratch: x6 IS the xq argument!) */
    sub w12, w1, w0
    add w12, w12, w9
    sub w12, w12, #1
    udiv w12, w12, w9
    add w13, w0, w12
    cmp w13, w1
    csel w13, w13, w1, lt
    mov w1, w13
    bl k2q4_gemv_range
    /* wait for N-1 workers */
    adrp x10, g_nthreads
    ldr w9, [x10, :lo12:g_nthreads]
    sub w9, w9, #1
    adrp x10, g_done
    add x10, x10, :lo12:g_done
2:  ldar w11, [x10]
    cmp w11, w9
    b.lt 2b
    ldr x30, [sp], #16
    ret


/* ---------------- memcpy_q: x0=dst, x1=src, w2=bytes (multiple of 16).
 * Clobbers x2,x3,v0. */
    .align 2
memcpy_q:
    cbz w2, 9f
    mov x3, xzr
0:  ldr q0, [x1, x3]
    str q0, [x0, x3]
    add x3, x3, #16
    cmp x3, x2
    b.lo 0b
9:  ret

/* ---------------- k2q4_gemv_batch:
 *   w0=r0, w1=r1, w2=cols, x3=w, x4=xq_base, x5=y_base, w6=B, w7=rows
 * Row-outer, token-inner: each row's Q4 bytes (<= 2880 B) are re-scanned
 * once per token but stay in L1, so DRAM weight traffic drops ~B-fold.
 * Every (row, token) is delegated to k2q4_gemv_range -> bit-exact by
 * construction. Token b: xq at xq_base+b*xqs, y at y_base+b*rows*4.
 * Saves x19-x28. */
    .align 2
k2q4_gemv_batch:
    stp x29, x30, [sp, #-128]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    /* [sp,#96] y token ptr, [sp,#104] xq token ptr, [sp,#112] b counter:
     * k2q4_gemv_range clobbers x8-x15, so the b-loop state lives in memory */
    mov w19, w0                     /* r0 */
    mov w20, w1                     /* r1 */
    mov w21, w2                     /* cols */
    mov x22, x3                     /* w */
    mov x23, x4                     /* xq_base */
    mov x24, x5                     /* y_base */
    mov w25, w6                     /* B */
    mov w26, w7                     /* rows */
    lsr x27, x21, #8                /* nb256 = cols/256 */
    mov x28, #292
    mul x27, x27, x28               /* xqs */
    lsl x28, x26, #2                /* ystride = rows*4 */
.Lbg_row:
    cmp w19, w20
    b.ge .Lbg_end
    str x24, [sp, #96]              /* y token base: callee adds r*4 */
    str x23, [sp, #104]
    str wzr, [sp, #112]
.Lbg_b:
    ldr x5, [sp, #96]               /* y = y_b + r*4 */
    ldr x6, [sp, #104]
    mov w0, w19
    add w1, w19, #1
    mov w2, w21
    mov x3, x22
    mov x4, xzr
    bl k2q4_gemv_range
    ldr x9, [sp, #96]
    add x9, x9, x28
    str x9, [sp, #96]
    ldr x9, [sp, #104]
    add x9, x9, x27
    str x9, [sp, #104]
    ldr w9, [sp, #112]
    add w9, w9, #1
    str w9, [sp, #112]
    cmp w9, w25
    b.ne .Lbg_b
    add w19, w19, #1
    b .Lbg_row
.Lbg_end:
    ldp x27, x28, [sp, #80]
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #128
    ret

/* ---------------- q4_gemv_dispatch_b: batched row-parallel dispatch.
 *   w0=r0, w1=r1, w2=cols, x3=w, x4=xq_base, x5=y_base, w6=B, w7=rows
 * Same pool protocol as q4_gemv_dispatch (release publish, acquire wait).
 * Saves x30. Clobbers x9-x13. */
    .align 2
q4_gemv_dispatch_b:
    adrp x9, g_nthreads
    ldr w9, [x9, :lo12:g_nthreads]
    cmp w9, #1
    b.gt .Lq4bd_par
    b k2q4_gemv_batch
.Lq4bd_par:
    str x30, [sp, #-16]!
    adrp x10, g_task_r0
    str w0, [x10, :lo12:g_task_r0]
    adrp x10, g_task_r1
    str w1, [x10, :lo12:g_task_r1]
    adrp x10, g_task_cols
    str w2, [x10, :lo12:g_task_cols]
    adrp x10, g_task_w
    str x3, [x10, :lo12:g_task_w]
    adrp x10, g_task_x
    str x5, [x10, :lo12:g_task_x]   /* y_base */
    adrp x10, g_task_xq
    str x4, [x10, :lo12:g_task_xq]  /* xq_base */
    adrp x10, g_task_B
    str w6, [x10, :lo12:g_task_B]
    adrp x10, g_task_rows
    str w7, [x10, :lo12:g_task_rows]
    adrp x10, g_done
    str wzr, [x10, :lo12:g_done]
    adrp x10, g_gen
    add x10, x10, :lo12:g_gen
    ldar w11, [x10]
    add w11, w11, #1
    stlr w11, [x10]
    sub w12, w1, w0
    add w12, w12, w9
    sub w12, w12, #1
    udiv w12, w12, w9
    add w13, w0, w12
    cmp w13, w1
    csel w13, w13, w1, lt
    mov w1, w13
    bl k2q4_gemv_batch
    adrp x10, g_nthreads
    ldr w9, [x10, :lo12:g_nthreads]
    sub w9, w9, #1
    adrp x10, g_done
    add x10, x10, :lo12:g_done
2:  ldar w11, [x10]
    cmp w11, w9
    b.lt 2b
    ldr x30, [sp], #16
    ret

/* ---------------- gemvW_batch: w0=il, w1=k, x2=XB, x3=YB, w4=B.
 * Quantizes B token activations into b_XQQB then dispatches the batched
 * GEMV over the same rows/cols tables as gemvW. Saves x19-x28. */
    .align 2
gemvW_batch:
    stp x29, x30, [sp, #-96]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    mov w19, w1                     /* k */
    mov x20, x2                     /* XB */
    mov x21, x3                     /* YB */
    mov w22, w0                     /* il */
    mov w23, w4                     /* B */
    adrp x1, g_LBASE
    ldr x2, [x1, :lo12:g_LBASE]
    movz x0, #0x010e, lsl #16
    movk x0, #0x3000
    madd x2, x22, x0, x2
    adrp x1, tbl_koff
    add x1, x1, :lo12:tbl_koff
    ldr x0, [x1, x19, lsl #3]
    add x24, x2, x0                 /* w */
    adrp x1, tbl_krows
    add x1, x1, :lo12:tbl_krows
    ldr w25, [x1, x19, lsl #2]      /* rows */
    adrp x1, tbl_kcols
    add x1, x1, :lo12:tbl_kcols
    ldr w26, [x1, x19, lsl #2]      /* cols */
    lsr x28, x26, #8
    mov x9, #292
    mul x28, x28, x9                /* xqs */
    lsl x11, x26, #2                /* cols*4 (x10 is clobbered by q8_quant!) */
    adrp x27, b_XQQB
    add x27, x27, :lo12:b_XQQB
    mov w9, wzr                     /* b */
1:  mov x0, x20
    mov x1, x27
    mov w2, w26
    bl k2_q8_quant
    add x20, x20, x11
    add x27, x27, x28
    add w9, w9, #1
    cmp w9, w23
    b.ne 1b
    mov w0, wzr
    mov w1, w25
    mov w2, w26
    mov x3, x24
    adrp x4, b_XQQB
    add x4, x4, :lo12:b_XQQB
    mov x5, x21
    mov w6, w23
    mov w7, w25
    bl q4_gemv_dispatch_b
    ldp x27, x28, [sp, #80]
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #96
    ret

/* ---------------- head_batch: w0=B. final rms + q8_quant + batched head
 * GEMV (rows=vocab) -> b_HEADB[token*vocab]. Saves x19-x24. */
    .align 2
head_batch:
    stp x29, x30, [sp, #-64]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    mov w19, w0                     /* B */
    mov x20, xzr                    /* b */
    mov x21, xzr                    /* byte offset */
1:  adrp x0, b_XB
    add x0, x0, :lo12:b_XB
    add x0, x0, x21
    adrp x1, g_O_NORM
    ldr x1, [x1, :lo12:g_O_NORM]
    mov w2, #1536
    adrp x3, b_XNB
    add x3, x3, :lo12:b_XNB
    add x3, x3, x21
    bl rms
    mov x9, #6144
    add x21, x21, x9
    add x20, x20, #1
    cmp w20, w19
    b.ne 1b
    mov x20, xzr                    /* b */
    mov x21, xzr                    /* XN offset */
    mov x22, xzr                    /* xq offset (6*292 = 1752) */
2:  adrp x0, b_XNB
    add x0, x0, :lo12:b_XNB
    add x0, x0, x21
    adrp x1, b_XQQB
    add x1, x1, :lo12:b_XQQB
    add x1, x1, x22
    mov w2, #1536
    bl k2_q8_quant
    mov x9, #6144
    add x21, x21, x9
    add x22, x22, #1752
    add x20, x20, #1
    cmp w20, w19
    b.ne 2b
    mov w0, wzr
    adrp x1, g_vocab
    ldr w1, [x1, :lo12:g_vocab]
    mov w2, #1536
    adrp x3, g_O_HEAD
    ldr x3, [x3, :lo12:g_O_HEAD]
    adrp x4, b_XQQB
    add x4, x4, :lo12:b_XQQB
    adrp x5, b_HEADB
    add x5, x5, :lo12:b_HEADB
    mov w6, w19
    adrp x7, g_vocab
    ldr w7, [x7, :lo12:g_vocab]
    bl q4_gemv_dispatch_b
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #64
    ret

/* ---------------- layer_forward_batch: w0=il, w1=B. Batched decoder layer.
 * Norms/rope/attention/silu stay per-token (cheap, sequential semantics);
 * the 7 projections run as batched GEMVs. g_pos must be pos0 on entry;
 * it is set to pos0+b for each token's attention and left at pos0+B.
 * Saves x19-x28. */
    .align 2
layer_forward_batch:
    stp x29, x30, [sp, #-96]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    mov w19, w0                     /* il */
    mov w20, w1                     /* B */
    adrp x1, g_LBASE
    ldr x21, [x1, :lo12:g_LBASE]
    movz x0, #0x010e, lsl #16
    movk x0, #0x3000
    madd x21, x19, x0, x21          /* layer base */
    adrp x1, g_pos
    ldr x28, [x1, :lo12:g_pos]      /* pos0 */
    /* 1. rms per token */
    mov x27, xzr                    /* b */
    mov x22, xzr                    /* X offset */
1:  adrp x0, b_XB
    add x0, x0, :lo12:b_XB
    add x0, x0, x22
    mov x1, x21
    mov w2, #1536
    adrp x3, b_XNB
    add x3, x3, :lo12:b_XNB
    add x3, x3, x22
    bl rms
    mov x9, #6144
    add x22, x22, x9
    add x27, x27, #1
    cmp w27, w20
    b.ne 1b
    /* 2. q,k,v batched */
    mov w0, w19
    mov w1, #1
    adrp x2, b_XNB
    add x2, x2, :lo12:b_XNB
    adrp x3, b_QFB
    add x3, x3, :lo12:b_QFB
    mov w4, w20
    bl gemvW_batch
    mov w0, w19
    mov w1, #2
    adrp x2, b_XNB
    add x2, x2, :lo12:b_XNB
    adrp x3, b_KFB
    add x3, x3, :lo12:b_KFB
    mov w4, w20
    bl gemvW_batch
    mov w0, w19
    mov w1, #3
    adrp x2, b_XNB
    add x2, x2, :lo12:b_XNB
    adrp x3, b_VFB
    add x3, x3, :lo12:b_VFB
    mov w4, w20
    bl gemvW_batch
    /* 3. rope + KV store per token */
    mov x27, xzr                    /* b */
    mov x23, xzr                    /* QF offset (8192) */
    mov x24, xzr                    /* KF offset (2048) */
2:  add x9, x28, x27               /* g_pos = pos0 + b (rope reads it) */
    adrp x1, g_pos
    str x9, [x1, :lo12:g_pos]
    adrp x0, b_QFB
    add x0, x0, :lo12:b_QFB
    add x0, x0, x23
    mov w1, #32
    bl rope
    adrp x0, b_KFB
    add x0, x0, :lo12:b_KFB
    add x0, x0, x24
    mov w1, #8
    bl rope
    mov x9, #3145728
    madd x9, x19, x9, xzr
    add x10, x28, x27               /* pos0 + b */
    adrp x2, g_KCA
    ldr x2, [x2, :lo12:g_KCA]
    add x2, x2, x9
    add x2, x2, x10, lsl #11        /* dst = KCA[layer][pos] */
    adrp x1, b_KFB
    add x1, x1, :lo12:b_KFB
    add x1, x1, x24                 /* src = KFB token */
    mov x0, x2
    mov w2, #2048
    bl memcpy_q
    adrp x2, g_VCA
    ldr x2, [x2, :lo12:g_VCA]
    add x2, x2, x9
    add x2, x2, x10, lsl #11
    adrp x1, b_VFB
    add x1, x1, :lo12:b_VFB
    add x1, x1, x24
    mov x0, x2
    mov w2, #2048
    bl memcpy_q
    add x23, x23, #8192
    add x24, x24, #2048
    add x27, x27, #1
    cmp w27, w20
    b.ne 2b
    /* 4. attention per token (copy Q into b_QF, result back from b_ATTO) */
    mov x27, xzr                    /* b */
    mov x23, xzr                    /* offset */
3:  adrp x0, b_QF
    add x0, x0, :lo12:b_QF
    adrp x1, b_QFB
    add x1, x1, :lo12:b_QFB
    add x1, x1, x23
    mov w2, #8192
    bl memcpy_q
    add x9, x28, x27
    adrp x1, g_pos
    str x9, [x1, :lo12:g_pos]
    mov w0, w19
    bl attention
    adrp x0, b_ATTOB
    add x0, x0, :lo12:b_ATTOB
    add x0, x0, x23
    adrp x1, b_ATTO
    add x1, x1, :lo12:b_ATTO
    mov w2, #8192
    bl memcpy_q
    add x23, x23, #8192
    add x27, x27, #1
    cmp w27, w20
    b.ne 3b
    /* 5. o proj + residual */
    mov w0, w19
    mov w1, #4
    adrp x2, b_ATTOB
    add x2, x2, :lo12:b_ATTOB
    adrp x3, b_AOB
    add x3, x3, :lo12:b_AOB
    mov w4, w20
    bl gemvW_batch
    mov x27, xzr
    mov x22, xzr
4:  adrp x0, b_XB
    add x0, x0, :lo12:b_XB
    add x0, x0, x22
    adrp x1, b_AOB
    add x1, x1, :lo12:b_AOB
    add x1, x1, x22
    bl vec_add1536
    mov x9, #6144
    add x22, x22, x9
    add x27, x27, #1
    cmp w27, w20
    b.ne 4b
    /* 6. post rms */
    mov x27, xzr
    mov x22, xzr
5:  adrp x0, b_XB
    add x0, x0, :lo12:b_XB
    add x0, x0, x22
    mov x1, x21
    movz x7, #0x43, lsl #16
    movk x7, #0x9800
    add x1, x1, x7
    mov w2, #1536
    adrp x3, b_XNB
    add x3, x3, :lo12:b_XNB
    add x3, x3, x22
    bl rms
    mov x9, #6144
    add x22, x22, x9
    add x27, x27, #1
    cmp w27, w20
    b.ne 5b
    /* 7. gate, up */
    mov w0, w19
    mov w1, #6
    adrp x2, b_XNB
    add x2, x2, :lo12:b_XNB
    adrp x3, b_LGB
    add x3, x3, :lo12:b_LGB
    mov w4, w20
    bl gemvW_batch
    mov w0, w19
    mov w1, #7
    adrp x2, b_XNB
    add x2, x2, :lo12:b_XNB
    adrp x3, b_LUB
    add x3, x3, :lo12:b_LUB
    mov w4, w20
    bl gemvW_batch
    /* 8. silu per token */
    mov x27, xzr                    /* b */
6:  mov x9, #20480
    adrp x23, b_LGB
    add x23, x23, :lo12:b_LGB
    madd x23, x27, x9, x23
    adrp x24, b_LUB
    add x24, x24, :lo12:b_LUB
    madd x24, x27, x9, x24
    mov x22, #5120
7:  ldr s10, [x23], #4
    fneg s0, s10
    bl expf
    fmov s1, #1.0
    fadd s0, s0, s1
    fdiv s0, s10, s0
    ldr s1, [x24]
    fmul s0, s0, s1
    str s0, [x24], #4
    subs x22, x22, #1
    b.ne 7b
    add x27, x27, #1
    cmp w27, w20
    b.ne 6b
    /* 9. down + residual */
    mov w0, w19
    mov w1, #8
    adrp x2, b_LUB
    add x2, x2, :lo12:b_LUB
    adrp x3, b_FOB
    add x3, x3, :lo12:b_FOB
    mov w4, w20
    bl gemvW_batch
    mov x27, xzr
    mov x22, xzr
8:  adrp x0, b_XB
    add x0, x0, :lo12:b_XB
    add x0, x0, x22
    adrp x1, b_FOB
    add x1, x1, :lo12:b_FOB
    add x1, x1, x22
    bl vec_add1536
    mov x9, #6144
    add x22, x22, x9
    add x27, x27, #1
    cmp w27, w20
    b.ne 8b
    /* restore g_pos = pos0: the next layer reads it as its own base */
    adrp x1, g_pos
    str x28, [x1, :lo12:g_pos]
    ldp x27, x28, [sp, #80]
    ldp x25, x26, [sp, #64]
    ldp x23, x24, [sp, #48]
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #96
    ret

/* ---------------- .data ---------------- */
    .data
    .align 3
f_tenthousand:  .double 10000.0
f_2pm53:        .double 1.1102230246251565e-16      /* 2^-53 (frand01) */
    .align 2
f_inf32:        .word 0x7f800000
f_1em6:         .word 0x358637bd                    /* 1e-6f (rms eps) */
f_n1e30:        .word 0xf149f2ca                    /* -1e30f (mx init) */
    .align 3
/* glibc 2.39 expf fast path: 32-entry u64 table @ libm 0x7a5f0, then a
 * 40-byte gap, then 4 f64 constants at +296..+320 (offsets as expf loads). */
k_exp_table:
    .quad 0x3FF0000000000000
    .quad 0x3FEFD9B0D3158574
    .quad 0x3FEFB5586CF9890F
    .quad 0x3FEF9301D0125B51
    .quad 0x3FEF72B83C7D517B
    .quad 0x3FEF54873168B9AA
    .quad 0x3FEF387A6E756238
    .quad 0x3FEF1E9DF51FDEE1
    .quad 0x3FEF06FE0A31B715
    .quad 0x3FEEF1A7373AA9CB
    .quad 0x3FEEDEA64C123422
    .quad 0x3FEECE086061892D
    .quad 0x3FEEBFDAD5362A27
    .quad 0x3FEEB42B569D4F82
    .quad 0x3FEEAB07DD485429
    .quad 0x3FEEA47EB03A5585
    .quad 0x3FEEA09E667F3BCD
    .quad 0x3FEE9F75E8EC5F74
    .quad 0x3FEEA11473EB0187
    .quad 0x3FEEA589994CCE13
    .quad 0x3FEEACE5422AA0DB
    .quad 0x3FEEB737B0CDC5E5
    .quad 0x3FEEC49182A3F090
    .quad 0x3FEED503B23E255D
    .quad 0x3FEEE89F995AD3AD
    .quad 0x3FEEFF76F2FB5E47
    .quad 0x3FEF199BDD85529C
    .quad 0x3FEF3720DCEF9069
    .quad 0x3FEF5818DCFBA487
    .quad 0x3FEF7C97337B9B5F
    .quad 0x3FEFA4AFA2A490DA
    .quad 0x3FEFD0765B6E4540
    .space 40                       /* libm gap: +256..+295 */
    /* +296 */ .quad 0x40471547652B82FE            /* 46.166241308446828 = 32/ln2 */
    /* +304 */ .quad 0x3EBC6AF84B912394            /* 1.6938359250920212e-06 */
    /* +312 */ .quad 0x3F2EBFCE50FAC4F3            /* 2.3459809789509004e-04 = (ln2/32)^2/2 */
    /* +320 */ .quad 0x3F962E42FF0C52D6            /* 0.021660849396613134 = ln2/32 */
    .align 3
g_KCA:          .quad b_KCAbuf
g_VCA:          .quad b_VCAbuf
    .align 2
s_step:         .asciz "step "
s_idin:         .asciz " id_in="
s_top:          .asciz " top="
s_logit:        .asciz " logit="
s_gen:          .asciz "--gen"
s_sample:       .asciz "--sample"
s_threads:      .asciz "--threads"
s_batch:        .asciz "--batch"
s_nan:          .asciz "nan"
s_inf:          .asciz "inf"
s_ninf:         .asciz "-inf"
m_open:         .asciz "k2_core: open failed"
m_seek:         .asciz "k2_core: lseek failed"
m_mmap:         .asciz "k2_core: mmap failed"
m_magic:        .asciz "k2_core: not a K2H1 blob"
m_geom:         .asciz "k2_core: geometry mismatch"
m_wtype:        .asciz "k2_core: only Q4 blobs (wtype=2) supported"
m_minp:         .asciz "k2_core: minp>0 unsupported (logf not ported)"
m_usage:        .asciz "usage: k2_core <k2h_q4.blob> <id...> [--gen N] [--sample temp topp topk seed [minp]]"
    .align 3
tbl_koff:       .quad 0, 6144, 1775616, 2217984, 2660352, 4429824, 4435968, 8859648, 13283328
    .align 2
tbl_krows:      .word 1536, 2048, 512, 512, 1536, 1536, 5120, 5120, 1536
tbl_kcols:      .word 0, 1536, 1536, 1536, 2048, 0, 1536, 1536, 5120

/* ---------------- .bss ---------------- */
    .bss
    .align 6
b_X:            .skip 6144
b_XN:           .skip 6144
b_AO:           .skip 6144
b_FO:           .skip 6144
b_QF:           .skip 8192
b_KF:           .skip 2048
b_VF:           .skip 2048
b_ATTO:         .skip 8192
b_LG:           .skip 20480
b_LU:           .skip 20480
b_HEAD:         .skip 524288               /* VOCAB_MAX 131072 f32 */
b_XQQ:          .skip 5840                 /* (5120/256)*292 */
b_XB:           .skip 49152                /* 8 x 1536 f32 */
b_XNB:          .skip 49152
b_AOB:          .skip 49152
b_FOB:          .skip 49152
b_QFB:          .skip 65536                /* 8 x 2048 f32 */
b_KFB:          .skip 16384                /* 8 x 512 f32 */
b_VFB:          .skip 16384
b_ATTOB:        .skip 65536
b_LGB:          .skip 163840               /* 8 x 5120 f32 */
b_LUB:          .skip 163840
b_XQQB:         .skip 46720                /* 8 x 5840 */
b_HEADB:        .skip 4194304              /* 8 x 131072 f32 */
b_attsc:        .skip 6144                 /* attention sc[1536] */
b_smpsc:        .skip 4096                 /* sampler sc[1024] */
b_smsi:         .skip 4096                 /* sampler si[1024] */
b_ids:          .skip 16384                /* 4096 ids */
numbuf:         .skip 128
g_blob:         .skip 8
g_LBASE:        .skip 8
g_O_EMB:        .skip 8
g_O_NORM:       .skip 8
g_O_HEAD:       .skip 8
g_O_COS:        .skip 8
g_O_SIN:        .skip 8
g_pos:          .skip 8
g_batch:        .skip 4                    /* prefill chunk size (1 = off) */
g_task_B:       .skip 4
g_task_rows:    .skip 4
    .align 3                        /* g_RS is loaded with 64-bit ldr */
g_RS:           .skip 8
g_vocab:        .skip 4
g_npos:         .skip 4
g_stopp:        .skip 4
g_stemp:        .skip 4
g_sminp:        .skip 4
g_stopk:        .skip 4
g_nthreads:     .skip 4
g_gen:          .skip 4
g_done:         .skip 4
g_task_r0:      .skip 4
g_task_r1:      .skip 4
g_task_cols:    .skip 4
    .align 3
g_task_w:       .skip 8
g_task_x:       .skip 8
g_task_y:       .skip 8
g_task_xq:      .skip 8
    .align 16                       /* 64 KiB: worker id = (sp-base)>>16 */
b_stacks:       .skip 524288            /* 8 worker stacks x 64 KiB */
    .align 12
b_KCAbuf:       .skip 88080384            /* 28*1536*512 f32 */
b_VCAbuf:       .skip 88080384

    .text
    .align 2
/* ---------------- string/number parsers ---------------- */

/* str_is: x0 = str, x1 = expect (both asciz) -> w0 = 1/0 */
    .align 2
str_is:
0:  ldrb w2, [x0], #1
    ldrb w3, [x1], #1
    cmp w2, w3
    b.ne 1f
    cbnz w2, 0b
    mov w0, #1
    ret
1:  mov w0, wzr
    ret

/* parse_i64: x0 = str -> x0 = value (s64, optional sign), x1 = after */
    .align 2
parse_i64:
    mov x2, #0                       /* acc */
    mov w3, #0                       /* neg flag */
    ldrb w4, [x0]
    cmp w4, #'-'
    b.ne 0f
    mov w3, #1
    add x0, x0, #1
0:  ldrb w4, [x0]
    sub w4, w4, #'0'
    cmp w4, #9
    b.hi 1f
    mov x5, #10
    madd x2, x2, x5, x4              /* acc = acc*10 + d (w4 zero-extended? w4 sub -> upper garbage!) */
    add x0, x0, #1
    b 0b
1:  mov x1, x0
    cmp w3, #0
    csneg x0, x2, x2, eq             /* if neg: -x2 */
    ret

/* parse_u64: x0 = str -> x0 = value, x1 = after (unsigned, no sign) */
    .align 2
parse_u64:
    mov x2, #0
0:  ldrb w4, [x0]
    sub w4, w4, #'0'
    cmp w4, #9
    b.hi 1f
    mov x5, #10
    madd x2, x2, x5, x4
    add x0, x0, #1
    b 0b
1:  mov x1, x0
    mov x0, x2
    ret

/* parse_f32: x0 = str -> s0 = value, x1 = after.
 * decimal N[.F] -> f64 (single correctly-rounded division total/scale ==
 * strtod for these forms) -> f32 (== atof-then-float assignment). */
    .align 2
parse_f32:
    mov x2, #0                       /* int part N */
    mov w6, #0                       /* neg */
    ldrb w4, [x0]
    cmp w4, #'-'
    b.ne 0f
    mov w6, #1
    add x0, x0, #1
0:  ldrb w4, [x0]
    sub w4, w4, #'0'
    cmp w4, #9
    b.hi 1f
    mov x5, #10
    uxtw x4, w4
    madd x2, x2, x5, x4
    add x0, x0, #1
    b 0b
1:  mov x7, #1                       /* scale = 10^d */
    mov x8, #0                       /* frac F */
    ldrb w4, [x0]
    cmp w4, #'.'
    b.ne 3f
    add x0, x0, #1
2:  ldrb w4, [x0]
    sub w4, w4, #'0'
    cmp w4, #9
    b.hi 3f
    mov x5, #10
    uxtw x4, w4
    madd x8, x8, x5, x4              /* F = F*10 + d */
    mov x5, #10
    mul x7, x7, x5                  /* scale *= 10 */
    add x0, x0, #1
    b 2b
3:  /* total = N*scale + F ; v = total/scale in f64 */
    mul x9, x2, x7                  /* N*scale */
    add x9, x9, x8                  /* + F */
    ucvtf d0, x9                    /* exact for our arg forms (<2^53) */
    ucvtf d1, x7
    fdiv d0, d0, d1                 /* correctly rounded f64 == strtod */
    cbz w6, 4f
    fneg d0, d0
4:  fcvt s0, d0                     /* == atof->float double rounding */
    mov x1, x0
    ret

/* ---------------- main ----------------
 * State in callee-saved regs (all my functions save/restore x19-x28):
 *   x19 argc/il, x20 argv, x21 parse idx, x22 nids, x23 gen,
 *   x24 n0, x25 step, x26 id_in, x27 top, x28 = n0+gen bound
 */
    .align 2
main:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
    ldr x19, [x1]                    /* argc — x1 = sp passed from _start */
    add x20, x1, #8                  /* argv */
    cmp x19, #2
    b.lt .Lusage
    ldr x0, [x20, #8]                /* argv[1] = blob path */
    bl blob_load
    /* defaults: STOPP = 1.0f (rest are 0 in BSS) */
    adrp x1, g_stopp
    fmov s0, #1.0
    str s0, [x1, :lo12:g_stopp]
    mov w22, wzr
    mov x23, xzr
    mov x21, #2
.Lparse:
    cmp x21, x19
    b.hs .Lparsedone
    ldr x0, [x20, x21, lsl #3]
    adrp x1, s_gen
    add x1, x1, :lo12:s_gen
    bl str_is
    cbnz w0, .Larg_gen
    ldr x0, [x20, x21, lsl #3]
    adrp x1, s_sample
    add x1, x1, :lo12:s_sample
    bl str_is
    cbnz w0, .Larg_sample
    ldr x0, [x20, x21, lsl #3]
    adrp x1, s_threads
    add x1, x1, :lo12:s_threads
    bl str_is
    cbnz w0, .Larg_threads
    ldr x0, [x20, x21, lsl #3]
    adrp x1, s_batch
    add x1, x1, :lo12:s_batch
    bl str_is
    cbnz w0, .Larg_batch
    /* plain token -> id */
    ldr x0, [x20, x21, lsl #3]
    bl parse_i64
    adrp x2, b_ids
    add x2, x2, :lo12:b_ids
    str w0, [x2, w22, sxtw #2]
    add w22, w22, #1
    cmp w22, #4096
    b.hs .Lusage
    add x21, x21, #1
    b .Lparse
.Larg_gen:
    add x21, x21, #1
    cmp x21, x19
    b.hs .Lusage
    ldr x0, [x20, x21, lsl #3]
    bl parse_i64
    mov x23, x0
    add x21, x21, #1
    b .Lparse
.Larg_sample:
    add x21, x21, #1
    add x2, x21, #4
    cmp x2, x19
    b.gt .Lusage                     /* need temp,topp,topk,seed */
    ldr x0, [x20, x21, lsl #3]
    bl parse_f32
    adrp x2, g_stemp
    str s0, [x2, :lo12:g_stemp]
    add x21, x21, #1
    ldr x0, [x20, x21, lsl #3]
    bl parse_f32
    adrp x2, g_stopp
    str s0, [x2, :lo12:g_stopp]
    add x21, x21, #1
    ldr x0, [x20, x21, lsl #3]
    bl parse_i64
    adrp x2, g_stopk
    str w0, [x2, :lo12:g_stopk]
    add x21, x21, #1
    ldr x0, [x20, x21, lsl #3]
    bl parse_u64
    adrp x2, g_RS
    str x0, [x2, :lo12:g_RS]
    add x21, x21, #1
    /* optional minp (only if present and not a flag) */
    cmp x21, x19
    b.hs .Lsample_done
    ldr x0, [x20, x21, lsl #3]
    ldrb w1, [x0]
    cmp w1, #'-'
    b.eq .Lsample_done
    bl parse_f32
    fcmp s0, #0.0
    b.ls 7f
    adrp x0, m_minp
    add x0, x0, :lo12:m_minp
    b .Lfatal                        /* logf not ported; demos use minp=0 */
7:  add x21, x21, #1
.Lsample_done:
    b .Lparse
.Larg_threads:
    add x21, x21, #1
    cmp x21, x19
    b.hs .Lusage
    ldr x0, [x20, x21, lsl #3]
    bl parse_i64
    cmp w0, #1
    b.lt .Lusage
    cmp w0, #8
    b.le 1f
    mov w0, #8
1:  adrp x2, g_nthreads
    str w0, [x2, :lo12:g_nthreads]
    add x21, x21, #1
    b .Lparse
.Larg_batch:
    add x21, x21, #1
    cmp x21, x19
    b.hs .Lusage
    ldr x0, [x20, x21, lsl #3]
    bl parse_i64
    cmp w0, #1
    b.lt .Lusage
    cmp w0, #8
    b.le .Lbatch_clamp
    mov w0, #8
.Lbatch_clamp:
    adrp x2, g_batch
    str w0, [x2, :lo12:g_batch]
    add x21, x21, #1
    b .Lparse
.Lparsedone:
    cmp w22, #0
    b.eq .Lusage                     /* need at least one prompt id */
    mov x24, x22                     /* n0 */
    add x28, x24, x23                /* n0 + gen bound */
    adrp x1, g_pos
    str xzr, [x1, :lo12:g_pos]
    /* thread pool (default 1 -> serial, bit-identical) */
    adrp x1, g_nthreads
    ldr w0, [x1, :lo12:g_nthreads]
    cbnz w0, 1f
    mov w0, #1
1:  bl pool_init
    mov x25, xzr                     /* step */
    adrp x1, g_batch
    ldr w9, [x1, :lo12:g_batch]
    cmp w9, #1
    b.gt .Lchunk                     /* batched prefill enabled */
.Lloop:
    /* --- choose id --- */
    cmp x25, x24
    b.hs .Lgen_id
    adrp x1, b_ids
    add x1, x1, :lo12:b_ids
    ldr w26, [x1, x25, lsl #2]
    b .Lfwd_id
.Lgen_id:
    adrp x1, g_stemp
    ldr s0, [x1, :lo12:g_stemp]
    fcmp s0, #0.0
    b.gt .Lsample
    adrp x0, b_HEAD
    add x0, x0, :lo12:b_HEAD
    adrp x1, g_vocab
    ldr w1, [x1, :lo12:g_vocab]
    bl argmax
    mov w26, w0
    b .Lappend
.Lsample:
    adrp x0, b_HEAD
    add x0, x0, :lo12:b_HEAD
    adrp x1, g_vocab
    ldr w1, [x1, :lo12:g_vocab]
    adrp x2, g_stemp
    ldr s0, [x2, :lo12:g_stemp]
    adrp x2, g_stopp
    ldr s1, [x2, :lo12:g_stopp]
    adrp x2, g_stopk
    ldr w2, [x2, :lo12:g_stopk]
    adrp x3, g_sminp
    ldr s3, [x3, :lo12:g_sminp]
    bl sample_topkpp
    mov w26, w0
.Lappend:
    adrp x1, b_ids
    add x1, x1, :lo12:b_ids
    str w26, [x1, w22, sxtw #2]
    add w22, w22, #1
    cmp w22, #4096
    b.hs .Lusage
    b .Lfwd_id
/* ---- batched prefill: process min(g_batch, n0-step) prompt tokens ---- */
.Lchunk:
    cmp x25, x24
    b.hs .Lloop                      /* generation phase -> single-token path */
    sub x9, x24, x25
    adrp x1, g_batch
    ldr w10, [x1, :lo12:g_batch]
    cmp w10, #1
    b.lt 7f
    cmp x10, x9
    csel x10, x10, x9, lt
    b 8f
7:  mov x10, x9
8:  mov x20, x10                     /* B */
    /* embed each token into b_XB (x22 = n0: never clobber it here) */
    mov x21, xzr
9:  adrp x1, b_ids
    add x1, x1, :lo12:b_ids
    add x2, x25, x21
    ldr w0, [x1, x2, lsl #2]
    bl embed
    mov x9, #6144
    mul x9, x21, x9
    adrp x0, b_XB
    add x0, x0, :lo12:b_XB
    add x0, x0, x9
    adrp x1, b_X
    add x1, x1, :lo12:b_X
    mov w2, #6144
    bl memcpy_q
    add x21, x21, #1
    cmp w21, w20
    b.ne 9b
    adrp x1, g_pos
    str x25, [x1, :lo12:g_pos]
    mov x19, xzr                     /* il */
10: mov w0, w19
    mov w1, w20
    bl layer_forward_batch
    add x19, x19, #1
    cmp x19, #28
    b.ne 10b
    mov w0, w20
    bl head_batch
    /* per-token top + print (prompt steps: EOS cannot trigger) */
    mov x21, xzr                     /* b */
11: adrp x1, g_vocab
    ldr w9, [x1, :lo12:g_vocab]
    lsl x9, x9, #2
    mul x9, x21, x9                  /* logits offset = b*vocab*4 */
    adrp x0, b_HEADB
    add x0, x0, :lo12:b_HEADB
    add x0, x0, x9
    adrp x1, g_vocab
    ldr w1, [x1, :lo12:g_vocab]
    bl argmax
    mov w27, w0
    adrp x1, b_ids
    add x1, x1, :lo12:b_ids
    add x2, x25, x21
    ldr w26, [x1, x2, lsl #2]
    adrp x0, s_step
    add x0, x0, :lo12:s_step
    bl print_cstr
    add x0, x25, x21
    bl print_u64
    adrp x0, s_idin
    add x0, x0, :lo12:s_idin
    bl print_cstr
    mov x0, x26
    bl print_u64
    adrp x0, s_top
    add x0, x0, :lo12:s_top
    bl print_cstr
    mov x0, x27
    bl print_u64
    adrp x0, s_logit
    add x0, x0, :lo12:s_logit
    bl print_cstr
    adrp x1, g_vocab
    ldr w9, [x1, :lo12:g_vocab]
    lsl x9, x9, #2
    mul x9, x21, x9
    adrp x0, b_HEADB
    add x0, x0, :lo12:b_HEADB
    add x0, x0, x9
    ldr s0, [x0, w27, uxtw #2]
    bl print_f32_4
    bl print_nl
    add x21, x21, #1
    cmp w21, w20
    b.ne 11b
    /* hand the last chunk token's logits to the single-token generation
     * path (it argmaxes b_HEAD) */
    adrp x1, g_vocab
    ldr w9, [x1, :lo12:g_vocab]     /* w: g_vocab is 4 bytes */
    lsl x9, x9, #2
    sub x10, x20, #1
    mul x10, x10, x9
    adrp x0, b_HEAD
    add x0, x0, :lo12:b_HEAD
    adrp x1, b_HEADB
    add x1, x1, :lo12:b_HEADB
    add x1, x1, x10
    mov x2, x9
    bl memcpy_q
    add x25, x25, x20
    adrp x1, g_pos
    str x25, [x1, :lo12:g_pos]
    b .Lchunk

.Lfwd_id:
    /* --- embed + 28 layers --- */
    mov w0, w26
    bl embed
    mov x19, xzr                     /* il (argc dead) */
.Lil:
    mov w0, w19
    bl layer_forward
    add x19, x19, #1
    cmp x19, #28
    b.ne .Lil
    /* --- final rms + head --- */
    adrp x0, b_X
    add x0, x0, :lo12:b_X
    adrp x1, g_O_NORM
    ldr x1, [x1, :lo12:g_O_NORM]
    mov w2, #1536
    adrp x3, b_XN
    add x3, x3, :lo12:b_XN
    bl rms
    adrp x0, b_XN
    add x0, x0, :lo12:b_XN
    adrp x1, b_XQQ
    add x1, x1, :lo12:b_XQQ
    mov w2, #1536
    bl k2_q8_quant
    mov w0, wzr
    adrp x1, g_vocab
    ldr w1, [x1, :lo12:g_vocab]
    mov w2, #1536
    adrp x3, g_O_HEAD
    ldr x3, [x3, :lo12:g_O_HEAD]
    adrp x4, b_XN
    add x4, x4, :lo12:b_XN
    adrp x5, b_HEAD
    add x5, x5, :lo12:b_HEAD
    adrp x6, b_XQQ
    add x6, x6, :lo12:b_XQQ
    bl q4_gemv_dispatch
    adrp x0, b_HEAD
    add x0, x0, :lo12:b_HEAD
    adrp x1, g_vocab
    ldr w1, [x1, :lo12:g_vocab]
    bl argmax
    mov w27, w0                      /* top */
    /* --- print: step %d id_in=%d top=%d logit=%.4f\n --- */
    adrp x0, s_step
    add x0, x0, :lo12:s_step
    bl print_cstr
    mov x0, x25
    bl print_u64
    adrp x0, s_idin
    add x0, x0, :lo12:s_idin
    bl print_cstr
    mov x0, x26
    bl print_u64
    adrp x0, s_top
    add x0, x0, :lo12:s_top
    bl print_cstr
    mov x0, x27
    bl print_u64
    adrp x0, s_logit
    add x0, x0, :lo12:s_logit
    bl print_cstr
    adrp x0, b_HEAD
    add x0, x0, :lo12:b_HEAD
    ldr s0, [x0, w27, uxtw #2]
    bl print_f32_4
    bl print_nl
    /* --- EOS: step >= n0 && id_in == 1 --- */
    cmp x25, x24
    b.lo .Lnoeos
    cmp w26, #1
    b.eq .Ldone
.Lnoeos:
    add x25, x25, #1
    adrp x1, g_pos
    ldr x2, [x1, :lo12:g_pos]
    add x2, x2, #1
    str x2, [x1, :lo12:g_pos]
    cmp x25, x28
    b.lo .Lloop
.Ldone:
    ldp x29, x30, [sp], #16
    ret

.Lusage:
    adrp x0, m_usage
    add x0, x0, :lo12:m_usage
    b .Lfatal

/* ---------------- _start ---------------- */
    .global _start
_start:
    mov x1, sp                       /* argc/argv block */
    bl main
    mov x0, #0
    mov x8, #SYS_EXIT                /* exit_group */
    svc #0
