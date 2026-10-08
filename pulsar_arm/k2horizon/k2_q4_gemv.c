/* k2_q4_gemv: NEON Q4_0 x Q8_K GEMV for the K2-Horizon port.
 * Block (18B): fp16 scale d + 16B codes, 4 bits/weight, 32 weights.
 * Value = ((byte & 0xF) or (byte >> 4)) - 8, low nibble = first 16,
 * high nibble = second 16 (ggml Q4_0 order). Two PQ... Q4 blocks share
 * one Q8_K block only when cols align; here cols (1536/5120) give ng of
 * 48/160 groups of 32... the activation side is plain Q8_K halves per
 * 256-wide super-block: handle by processing 8 Q4 blocks (256 weights)
 * against one Q8_K block, halves (128B each) per 4-block half.
 * Simpler formulation used: per 32-weight block, dot the two nibble
 * int8x16 vectors against the matching Q8_K 16B quarters (loaded from
 * the shared XQ). Integer-exact; validation vs fp64 dequant reference
 * (maxrel) + argmax-stream equality vs the fp16 engine path.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <arm_neon.h>

static float h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu, o;
    if (e == 0) { if (!m) return 0.0f; float f = (float)m * 5.9604645e-8f; memcpy(&o, &f, 4); o |= s; }
    else if (e == 31) o = s | 0x7f800000u | (m << 13);
    else o = s | ((e + 112u) << 23) | (m << 13);
    float r; memcpy(&r, &o, 4); return r;
}

/* Q8_K activation quantizer (proven copy of the fwd8 self-contained one) */
static inline int nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}
void k2_q8_quant(const float *x, uint8_t *yv, int64_t k) {
    const int nb = (int)(k / 256);
    for (int i = 0; i < nb; i++) {
        float max = 0, amax = 0;
        for (int j = 0; j < 256; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) { amax = ax; max = x[j]; }
        }
        uint8_t *yb = yv + (size_t)i * 292;
        int8_t *qs = (int8_t *)(yb + 4);
        int16_t *bsums = (int16_t *)(yb + 4 + 256);
        if (!amax) {
            memset(yb, 0, 4);
            memset(yb + 4, 0, 256);
            x += 256;
            continue;
        }
        const float iscale = -127.f / max;
        for (int j = 0; j < 256; ++j) {
            int v = nearest_int(iscale * x[j]);
            qs[j] = (int8_t)(v < 127 ? v : 127);
        }
        for (int j = 0; j < 16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) sum += qs[j * 16 + ii];
            bsums[j] = (int16_t)sum;
        }
        float d = 1.0f / iscale;
        memcpy(yb, &d, 4);
        x += 256;
    }
}

/* one row range, Q4 path. w layout per row: ng34? NO - Q4 rows are
   ng4*18 bytes (32 weights per 18B block). xq = Q8_K super-blocks. */
void k2q4_gemv_range(int r0, int r1, int cols, const uint8_t *w,
                     const float *x, float *y, const uint8_t *xq) {
    (void)x;
    int nb = cols / 32;
    uint8x16_t m15 = vdupq_n_u8(15);
    int8x16_t eight = vdupq_n_s8(8);
    /* Row-parallel: rows are independent and each row's block order is
       unchanged, so the integer path stays bit-exact at any thread count
       (the bonsai pq2 lesson). */
    #pragma omp parallel for schedule(static) shared(m15, eight)
    for (int r = r0; r < r1; r++) {
        const uint8_t *row = w + (size_t)r * nb * 18;
        float acc = 0.0f;
        for (int b = 0; b < nb; b++) {
            const uint8_t *blk = row + b * 18;
            float ws = h2f((uint16_t)(blk[0] | (blk[1] << 8)));
            const uint8_t *bqs = blk + 2;
            /* Q8_K 256-block covering weights [b*32, b*32+256): block qb,
               first or second half of its 256 quants */
            int wb = (b * 32) / 256, woff = (b * 32) % 256;
            const uint8_t *q8b = xq + (size_t)wb * 292;
            float db;
            memcpy(&db, q8b, 4);
            const int8_t *qs = (const int8_t *)(q8b + 4 + woff);
            uint8x16_t bb = vld1q_u8(bqs);
            int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(bb, m15));
            int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(bb, 4));
            lo = vsubq_s8(lo, eight);
            hi = vsubq_s8(hi, eight);
            int32x4_t a = vdupq_n_s32(0);
            a = vdotq_s32(a, lo, vld1q_s8(qs));
            a = vdotq_s32(a, hi, vld1q_s8(qs + 16));
            acc += (ws * db) * (float)vaddvq_s32(a);
        }
        y[r] = acc;
    }
}
