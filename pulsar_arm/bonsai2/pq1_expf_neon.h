/* pq1_expf_neon.h - glibc 2.27 expf (sysdeps/ieee754/flt-32/e_expf.c, the
 * Szabolcs Nagy table+polynomial algorithm, N=32, degree 3, evaluated in
 * double) reproduced lane-wise in float64x2 NEON so that silu can be
 * vectorised while staying BIT-EXACT with the board's libm. The table is
 * asuint64(2^(j/32)) - (j<<47), correctly rounded (regenerated with mpmath,
 * mantissas identical to glibc's e_exp2f_data.c). Verified exhaustively on
 * the Nano against glibc 2.27 expf for every float with |x| < 88 (see
 * pq1_expf_test.c); |x| >= 88 and NaN take the scalar libm call.
 * MUST be compiled with -ffp-contract=off: under gcc's default contraction
 * 2 of 4.28e9 inputs come out 1 ulp off (pq1_host.c is built that way). */
#ifndef PQ1_EXPF_NEON_H
#define PQ1_EXPF_NEON_H
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>

static const uint64_t PQ1_EXP2F_TAB[32] = {
    0x3ff0000000000000ULL, 0x3fefd9b0d3158574ULL, 0x3fefb5586cf9890fULL, 0x3fef9301d0125b51ULL,
    0x3fef72b83c7d517bULL, 0x3fef54873168b9aaULL, 0x3fef387a6e756238ULL, 0x3fef1e9df51fdee1ULL,
    0x3fef06fe0a31b715ULL, 0x3feef1a7373aa9cbULL, 0x3feedea64c123422ULL, 0x3feece086061892dULL,
    0x3feebfdad5362a27ULL, 0x3feeb42b569d4f82ULL, 0x3feeab07dd485429ULL, 0x3feea47eb03a5585ULL,
    0x3feea09e667f3bcdULL, 0x3fee9f75e8ec5f74ULL, 0x3feea11473eb0187ULL, 0x3feea589994cce13ULL,
    0x3feeace5422aa0dbULL, 0x3feeb737b0cdc5e5ULL, 0x3feec49182a3f090ULL, 0x3feed503b23e255dULL,
    0x3feee89f995ad3adULL, 0x3feeff76f2fb5e47ULL, 0x3fef199bdd85529cULL, 0x3fef3720dcef9069ULL,
    0x3fef5818dcfba487ULL, 0x3fef7c97337b9b5fULL, 0x3fefa4afa2a490daULL, 0x3fefd0765b6e4540ULL,
};
#define PQ1_EXPF_N 32.0
#define PQ1_EXPF_INVLN2N (0x1.71547652b82fep+0 * PQ1_EXPF_N)
#define PQ1_EXPF_SHIFT 0x1.8p+52   /* expf uses the UNSCALED shift (z = x*N/ln2 is rounded to an integer) */
#define PQ1_EXPF_C0 (0x1.c6af84b912394p-5 / PQ1_EXPF_N / PQ1_EXPF_N / PQ1_EXPF_N)
#define PQ1_EXPF_C1 (0x1.ebfce50fac4f3p-3 / PQ1_EXPF_N / PQ1_EXPF_N)
#define PQ1_EXPF_C2 (0x1.62e42ff0c52d6p-1 / PQ1_EXPF_N)

/* two lanes, |x| < 88 assumed */
static inline float64x2_t pq1_expf_core2(float64x2_t xd) {
    float64x2_t z = vmulq_f64(vdupq_n_f64(PQ1_EXPF_INVLN2N), xd);
    float64x2_t kd = vaddq_f64(z, vdupq_n_f64(PQ1_EXPF_SHIFT));
    uint64x2_t ki = vreinterpretq_u64_f64(kd);
    kd = vsubq_f64(kd, vdupq_n_f64(PQ1_EXPF_SHIFT));
    float64x2_t r = vsubq_f64(z, kd);
    uint64x2_t t = vcombine_u64(vcreate_u64(PQ1_EXP2F_TAB[vgetq_lane_u64(ki, 0) & 31]),
                                vcreate_u64(PQ1_EXP2F_TAB[vgetq_lane_u64(ki, 1) & 31]));
    t = vaddq_u64(t, vshlq_n_u64(ki, 47));
    float64x2_t s = vreinterpretq_f64_u64(t);
    float64x2_t zz = vfmaq_f64(vdupq_n_f64(PQ1_EXPF_C1), vdupq_n_f64(PQ1_EXPF_C0), r);
    float64x2_t r2 = vmulq_f64(r, r);
    float64x2_t y = vfmaq_f64(vdupq_n_f64(1.0), vdupq_n_f64(PQ1_EXPF_C2), r);
    y = vfmaq_f64(y, zz, r2);
    return vmulq_f64(y, s);
}

/* four lanes; falls back to libm expf per lane when any |x| >= 88 or NaN */
static inline float32x4_t pq1_expf4(float32x4_t x) {
    uint32x4_t big = vcgeq_f32(vabsq_f32(x), vdupq_n_f32(88.0f));
    uint32x4_t nan = vmvnq_u32(vceqq_f32(x, x));
    if (__builtin_expect(vmaxvq_u32(vorrq_u32(big, nan)) != 0, 0)) {
        float v[4]; vst1q_f32(v, x);
        for (int i = 0; i < 4; i++) v[i] = expf(v[i]);
        return vld1q_f32(v);
    }
    float64x2_t lo = pq1_expf_core2(vcvt_f64_f32(vget_low_f32(x)));
    float64x2_t hi = pq1_expf_core2(vcvt_high_f64_f32(x));
    return vcombine_f32(vcvt_f32_f64(lo), vcvt_f32_f64(hi));
}
#endif
