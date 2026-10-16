// glibc_mathf.h: glibc 2.27 expf / logf / sinf / cosf, transcribed so that they return the
// same bits on any IEEE platform (x86 host, AArch64 A57, CUDA sm_53 in double precision).
//
// The PrismML fork's goldens are made on x86-64, where libm dispatches to the *_fma ifunc
// variants: the same C sources compiled with -mfma, so GCC contracted every a*b+c whose
// product has only additive uses. Those contractions are written out here as fma(), and
// the file must be compiled with -ffp-contract=off (nvcc: --fmad=false) so that nothing
// else is contracted. Verified exhaustively over all 2^32 inputs against the host libm by
// glibc_mathf_test.c. Only the round-to-nearest, no-errno paths matter for the engine;
// special values return what libm returns (NaN payloads aside).
#ifndef GLIBC_MATHF_H
#define GLIBC_MATHF_H
#include <stdint.h>
#include <string.h>
#include <math.h>

#ifdef __CUDACC__
#define GM_FN static __host__ __device__ __forceinline__
#define GM_TAB static __device__ __constant__
#else
#define GM_FN static inline
#define GM_TAB static const
#endif

#ifndef GM_EXPF_ZFMA
#define GM_EXPF_ZFMA 1   // z = InvLn2N*x has two additive uses, both fused (GCC widening_mul)
#endif

GM_FN uint32_t gm_asu(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
GM_FN float gm_asf(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
GM_FN double gm_asd(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }
GM_FN uint64_t gm_asu64(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }

GM_TAB uint64_t gm_exp2f_tab[32] = {
0x3ff0000000000000ull, 0x3fefd9b0d3158574ull, 0x3fefb5586cf9890full, 0x3fef9301d0125b51ull,
0x3fef72b83c7d517bull, 0x3fef54873168b9aaull, 0x3fef387a6e756238ull, 0x3fef1e9df51fdee1ull,
0x3fef06fe0a31b715ull, 0x3feef1a7373aa9cbull, 0x3feedea64c123422ull, 0x3feece086061892dull,
0x3feebfdad5362a27ull, 0x3feeb42b569d4f82ull, 0x3feeab07dd485429ull, 0x3feea47eb03a5585ull,
0x3feea09e667f3bcdull, 0x3fee9f75e8ec5f74ull, 0x3feea11473eb0187ull, 0x3feea589994cce13ull,
0x3feeace5422aa0dbull, 0x3feeb737b0cdc5e5ull, 0x3feec49182a3f090ull, 0x3feed503b23e255dull,
0x3feee89f995ad3adull, 0x3feeff76f2fb5e47ull, 0x3fef199bdd85529cull, 0x3fef3720dcef9069ull,
0x3fef5818dcfba487ull, 0x3fef7c97337b9b5full, 0x3fefa4afa2a490daull, 0x3fefd0765b6e4540ull,
};

GM_FN float gm_expf(float x) {
    const double InvLn2N = 0x1.71547652b82fep+0 * 32, SHIFT = 0x1.8p+52;
    const double C0 = 0x1.c6af84b912394p-5 / 32 / 32 / 32, C1 = 0x1.ebfce50fac4f3p-3 / 32 / 32,
                 C2 = 0x1.62e42ff0c52d6p-1 / 32;
    const double xd = (double)x;
    const uint32_t abstop = (gm_asu(x) >> 20) & 0x7ff;
    if (abstop >= (gm_asu(88.0f) >> 20)) {
        if (gm_asu(x) == gm_asu(-INFINITY)) return 0.0f;
        if (abstop >= (gm_asu(INFINITY) >> 20)) return x + x;
        if (x > 0x1.62e42ep6f) return gm_asf(0x7f800000u);   // overflow
        if (x < -0x1.9fe368p6f) return 0.0f;                  // underflow
    }
#if GM_EXPF_ZFMA
    double kd = fma(InvLn2N, xd, SHIFT);
    const uint64_t ki = gm_asu64(kd);
    kd -= SHIFT;
    const double r = fma(InvLn2N, xd, -kd);
#else
    const double z = InvLn2N * xd;
    double kd = z + SHIFT;
    const uint64_t ki = gm_asu64(kd);
    kd -= SHIFT;
    const double r = z - kd;
#endif
    uint64_t t = gm_exp2f_tab[ki % 32];
    t += ki << (52 - 5);
    const double s = gm_asd(t);
    const double zz = fma(C0, r, C1);
    const double r2 = r * r;
    double y = fma(C2, r, 1.0);
    y = fma(zz, r2, y);
    y = y * s;
    return (float)y;
}

GM_TAB double gm_logf_tab[16][2] = {
  { 0x1.661ec79f8f3bep+0, -0x1.57bf7808caadep-2 }, { 0x1.571ed4aaf883dp+0, -0x1.2bef0a7c06ddbp-2 },
  { 0x1.49539f0f010bp+0, -0x1.01eae7f513a67p-2 },  { 0x1.3c995b0b80385p+0, -0x1.b31d8a68224e9p-3 },
  { 0x1.30d190c8864a5p+0, -0x1.6574f0ac07758p-3 }, { 0x1.25e227b0b8eap+0, -0x1.1aa2bc79c81p-3 },
  { 0x1.1bb4a4a1a343fp+0, -0x1.a4e76ce8c0e5ep-4 }, { 0x1.12358f08ae5bap+0, -0x1.1973c5a611cccp-4 },
  { 0x1.0953f419900a7p+0, -0x1.252f438e10c1ep-5 }, { 0x1p+0, 0x0p+0 },
  { 0x1.e608cfd9a47acp-1, 0x1.aa5aa5df25984p-5 },  { 0x1.ca4b31f026aap-1, 0x1.c5e53aa362eb4p-4 },
  { 0x1.b2036576afce6p-1, 0x1.526e57720db08p-3 },  { 0x1.9c2d163a1aa2dp-1, 0x1.bc2860d22477p-3 },
  { 0x1.886e6037841edp-1, 0x1.1058bc8a07ee1p-2 },  { 0x1.767dcf5534862p-1, 0x1.4043057b6ee09p-2 },
};

GM_FN float gm_logf(float x) {
    const double Ln2 = 0x1.62e42fefa39efp-1;
    const double A0 = -0x1.00ea348b88334p-2, A1 = 0x1.5575b0be00b6ap-2, A2 = -0x1.ffffef20a4123p-2;
    uint32_t ix = gm_asu(x);
    if (ix == 0x3f800000u) return 0.0f;
    if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u) {
        if (ix * 2 == 0) return gm_asf(0xff800000u);          // -inf
        if (ix == 0x7f800000u) return x;
        if ((ix & 0x80000000u) || ix * 2 >= 0xff000000u) return (x - x) / (x - x);
        ix = gm_asu(x * 0x1p23f);
        ix -= 23u << 23;
    }
    const uint32_t tmp = ix - 0x3f330000u;
    const int i = (tmp >> (23 - 4)) % 16;
    const int k = (int32_t)tmp >> 23;
    const uint32_t iz = ix - (tmp & 0x1ffu << 23);
    const double invc = gm_logf_tab[i][0], logc = gm_logf_tab[i][1];
    const double z = (double)gm_asf(iz);
    const double r = fma(z, invc, -1.0);
    const double y0 = fma((double)k, Ln2, logc);
    const double r2 = r * r;
    double y = fma(A1, r, A2);
    y = fma(A0, r2, y);
    y = fma(y, r2, y0 + r);
    return (float)y;
}

/* sinf / cosf (glibc 2.27 flt-32 s_sinf.c, s_cosf.c, s_sincosf.h) */
#define GM_C0 (-0x1.ffffffffe98aep-2)
#define GM_C1 ( 0x1.55555545c50c7p-5)
#define GM_C2 (-0x1.6c16b348b6874p-10)
#define GM_C3 ( 0x1.a00eb9ac43ccp-16)
#define GM_C4 (-0x1.23c97dd8844d7p-22)
#define GM_S0 (-0x1.5555555551cd9p-3)
#define GM_S1 ( 0x1.1111110c2688bp-7)
#define GM_S2 (-0x1.a019f8b4bd1f9p-13)
#define GM_S3 ( 0x1.71d7264e6b5b4p-19)
#define GM_S4 (-0x1.a947e1674b58ap-26)
#define GM_PI_2_hi 0x1.921fb544p+0
#define GM_PI_2_lo 0x1.0b4611a626332p-34
#define GM_PI_2 1.57079632679489661923
#define GM_PI_4 0.78539816339744830962
#define GM_INV_PI_4 0x1.45f306dc9c883p+0

GM_TAB double gm_invpio4[8] = {
  0x0p+0, 0x1.45f306cp+0, 0x1.c9c882ap-28, 0x1.4fe13a8p-58,
  0x1.f47d4dp-85, 0x1.bb81b6cp-112, 0x1.4acc9ep-142, 0x1.0e4107cp-169
};

GM_FN double gm_sinpoly(double theta, double theta2) {
    double sx = fma(theta2, GM_S4, GM_S3);
    sx = fma(theta2, sx, GM_S2);
    sx = fma(theta2, sx, GM_S1);
    sx = fma(theta2, sx, GM_S0);
    return fma(theta * theta2, sx, theta);
}
GM_FN double gm_cospoly(double theta2) {
    double cx = fma(theta2, GM_C4, GM_C3);
    cx = fma(theta2, cx, GM_C2);
    cx = fma(theta2, cx, GM_C1);
    cx = fma(theta2, cx, GM_C0);
    return fma(theta2, cx, 1.0);
}
GM_FN float gm_reduced_sin(double theta, unsigned n, unsigned signbit) {
    const double theta2 = theta * theta;
    const double sign = (((n >> 2) & 1) ^ signbit) ? -1.0 : 1.0;
    const double sx = ((n & 2) == 0) ? gm_sinpoly(theta, theta2) : gm_cospoly(theta2);
    return (float)(sign * sx);
}
GM_FN float gm_reduced_cos(double theta, unsigned n) {
    const double theta2 = theta * theta;
    n += 2;
    const double sign = ((n >> 2) & 1) ? -1.0 : 1.0;
    const double cx = ((n & 2) == 0) ? gm_sinpoly(theta, theta2) : gm_cospoly(theta2);
    return (float)(sign * cx);
}
// |x| >= 2^23: Payne-Hanek style reduction; returns reduced e and the octant count l+1
GM_FN double gm_bigreduce(float ax, unsigned *nout) {
    int exponent = (int)(gm_asu(ax) >> 23) - 127;
    exponent += 3;
    exponent /= 28;
    double a = gm_invpio4[exponent] * ax;
    const double b = gm_invpio4[exponent + 1] * ax;
    const double c = gm_invpio4[exponent + 2] * ax;
    const double d = gm_invpio4[exponent + 3] * ax;
    uint64_t l = (uint64_t)a;
    l &= ~(uint64_t)7;
    a -= (double)l;
    double e = a + b;
    l = (uint64_t)e;
    e = a - (double)l;
    if (l & 1) {
        e -= 1.0; e += b; e += c; e += d; e *= GM_PI_4;
        *nout = (unsigned)(l + 1); return e;
    }
    e += b; e += c; e += d;
    if (e <= 1.0) { e *= GM_PI_4; *nout = (unsigned)(l + 1); return e; }
    l++; e -= 2.0; e *= GM_PI_4;
    *nout = (unsigned)(l + 1); return e;
}

GM_FN float gm_sinf(float x) {
    double theta = x;
    const double abstheta = fabs(theta);
    if (abstheta < GM_PI_4) {
        if (abstheta >= 0x1p-5) return (float)gm_sinpoly(theta, theta * theta);
        if (abstheta >= 0x1p-27) {
            const double theta2 = theta * theta;
            const double cx = fma(theta2, 0x1.110f475cec8c5p-7, -0x1.555555543d49dp-3);
            return (float)fma(theta * theta2, cx, theta);
        }
        if (theta != 0.0) return (float)fma(-theta, 0x1p-50, theta);
        return (float)theta;
    }
    const unsigned signbit = x < 0;
    if (abstheta < 9 * GM_PI_4) {
        const unsigned n = (unsigned)(abstheta * GM_INV_PI_4) + 1;
        theta = abstheta - (double)(n / 2) * GM_PI_2;
        return gm_reduced_sin(theta, n, signbit);
    }
    if (abstheta < INFINITY) {
        if (abstheta < 0x1p+23) {
            const unsigned n = ((unsigned)(abstheta * GM_INV_PI_4)) + 1;
            const double xx = n / 2;
            theta = fma(-xx, GM_PI_2_lo, fma(-xx, GM_PI_2_hi, abstheta));
            return gm_reduced_sin(theta, n, signbit);
        }
        unsigned n; const double e = gm_bigreduce(fabsf(x), &n);
        return gm_reduced_sin(e, n, signbit);
    }
    return x - x;
}

// sinf(x) as x86-64 glibc 2.27's __sincosf_fma returns it. GCC fuses a cosf/sinf pair of one
// argument (the fork's rope_yarn) into sincosf, and that build differs from sinf only for
// 2^-27 <= |x| < 2^-5: its vectorized code evaluates the inner term from x^3, not x^2
// (fma(x^3, SS1, SS0)), then x + (x^2 cx) x. Exhaustively equal to libm sincosf over all
// floats; its cos half is gm_cosf.
GM_FN float gm_sincos_sinf(float x) {
    const double theta = x, abstheta = fabs(theta);
    if (abstheta < GM_PI_4 && abstheta >= 0x1p-27 && abstheta < 0x1p-5) {
        const double theta2 = theta * theta, theta3 = theta * theta2;
        const double cx = fma(theta3, 0x1.110f475cec8c5p-7, -0x1.555555543d49dp-3);
        return (float)fma(theta2 * cx, theta, theta);
    }
    return gm_sinf(x);
}

GM_FN float gm_cosf(float x) {
    double theta = x;
    const double abstheta = fabs(theta);
    if (abstheta < GM_PI_4) {
        if (abstheta >= 0x1p-5) return (float)gm_cospoly(theta * theta);
        if (abstheta >= 0x1p-27) {
            const double theta2 = theta * theta;
            const double cx = fma(theta * theta2, 0x1.55514b178dac5p-5, -0x1.fffffff5cc6fdp-2);
            return (float)fma(theta2, cx, 1.0);
        }
        return (float)(1.0 - abstheta);
    }
    if (abstheta < 9 * GM_PI_4) {
        const unsigned n = (unsigned)(abstheta * GM_INV_PI_4) + 1;
        theta = abstheta - (double)(n / 2) * GM_PI_2;
        return gm_reduced_cos(theta, n);
    }
    if (abstheta < INFINITY) {
        if (abstheta < 0x1p+23) {
            const unsigned n = ((unsigned)(abstheta * GM_INV_PI_4)) + 1;
            const double xx = n / 2;
            theta = fma(-xx, GM_PI_2_lo, fma(-xx, GM_PI_2_hi, abstheta));
            return gm_reduced_cos(theta, n);
        }
        unsigned n; const double e = gm_bigreduce(fabsf(x), &n);
        return gm_reduced_cos(e, n);
    }
    return x - x;
}
#endif
