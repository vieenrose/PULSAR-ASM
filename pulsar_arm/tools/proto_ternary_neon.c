/* PULSAR-ARM | tools/proto_ternary_neon.c
 *
 * Prototype of the vectorised B3_128 GEMV, measured before committing to an asm
 * port. The scalar kernel costs ~10 instructions per weight (3.98 ns/weight)
 * against an fp32-FMA floor of ~0.14 ns/weight, and the floor is real: 7.0 G
 * fma-lanes/s measured, one add per weight, 1.7e9 weights.
 *
 * The trick is to stop decoding trit-by-trit. A group is 28 bytes = fp16 scale +
 * 26 code bytes, 5 base-3 digits per byte, weight idx = 5k+i, w = (d-1)*scale.
 * So per group
 *
 *     sum_idx (d-1)*x[idx]  =  sum_i sum_k d_i(v_k)*PL[i][k]  -  S
 *
 * where PL[i][k] = x[g*128 + 5k + i] and S = sum of the group's x. PL is x
 * DEINTERLEAVED BY DIGIT POSITION, built once per token (2048 floats -> 16
 * groups x 5 planes, ~10 KB, L1 resident), which is what removes the transpose:
 * after that a code byte's lane k lines up with a plane lane k. The -1 offset
 * became the constant -S per group, so no per-trit subtract.
 *
 * t_j = floor(v / 3^j) comes from one widening multiply and a shift (A72 has no
 * vector integer divide), and the digit is d_i = t_i - 3*t_{i+1}, so five
 * multiplies give all five digits of a lane - no repeated division like the
 * scalar loop does.
 *
 * Costs about 1.1 ops/weight instead of ~10. Build on the Pi:
 *   gcc -O2 -o proto_ternary_neon proto_ternary_neon.c ../asm/ternary_gemv.S
 *   taskset -c 3 ./proto_ternary_neon ~/bonsai_b3.bin
 */
#include <arm_neon.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

void ternary_gemv_b3(int rows, int cols, const uint8_t *w, const float *x, float *y);

#define COLS 2048
#define NG   (COLS / 128)
#define ROWS 2048

/* floor(v / 3^j) = (v * ceil(2^16/3^j)) >> 16. A72 NEON has no vector integer
 * divide and no unsigned multiply-high, but SQDMULH (signed doubling multiply
 * high) does exist, and every magic here is even - so halving the constant turns
 * (v*M)>>16 into exactly vqdmulh(v, M/2), one instruction per 8 bytes. No
 * saturation risk: the result is at most 255. */
static const int16_t MAGIC[6] = {0 /*unused*/, 10923, 3641, 1214, 405, 135};

/* x deinterleaved by digit position, plus the per-group sum of x.
 * Lanes 26..31 are padding and stay zero; so do PL[i][26..31] for the two
 * trits of byte 25 that would index x[128], x[129]. */
static float PL[NG][5][32] __attribute__((aligned(16)));
static float GS[NG];

static void deinterleave(const float *x) {
    for (int g = 0; g < NG; g++) {
        const float *xg = x + g * 128;
        float s = 0;
        for (int k = 0; k < 32; k++)
            for (int i = 0; i < 5; i++) {
                int idx = 5 * k + i;
                PL[g][i][k] = (k < 26 && idx < 128) ? xg[idx] : 0.0f;
            }
        for (int idx = 0; idx < 128; idx++) s += xg[idx];
        GS[g] = s;
    }
}

/* half -> float in integer ops. A72 has no FCVT between half and single, so
 * `(float)__fp16` is a libgcc CALL, once per group, 32768 times a token. */
static inline float h2f(uint16_t h) {
    uint32_t sgn = (uint32_t)(h & 0x8000u) << 16, ex = (h >> 10) & 0x1fu,
             mant = h & 0x3ffu, out;
    if (ex == 0) {                              /* zero / subnormal */
        if (!mant) return (float)(int32_t)sgn * 0.0f;
        float f = (float)mant * 5.960464478e-8f;
        memcpy(&out, &f, 4); out |= sgn;
    } else if (ex == 31) {
        out = sgn | 0x7f800000u | (mant << 13);
    } else {
        out = sgn | ((ex + 112u) << 23) | (mant << 13);
    }
    float r; memcpy(&r, &out, 4); return r;
}

/* Cost breakdown switches. VARIANT 0 is the real kernel; the others drop one
 * stage at a time so the remaining ns/weight says which stage actually costs.
 * (#if cannot sit inside a macro body, hence a macro per stage.)
 * Measured on the Pi, 2026-10-06, real layer-5 q_proj bytes:
 *   V5 loads only                          0.04 ns/weight
 *   V3 fma + plane loads, no digit math    0.54   <- load:fma ratio
 *   V4 convert + fma, no digit math        2.17   <- the converts are the wall
 *   V1 digit math + load, no convert       1.38
 *   V0 full, digit form                    1.98   (2.0x)
 *   V6 full, no convert                    0.91   (4.3x, ~1540 ms/token)  WINNER
 *   V7 integer digit chain                 1.61   dead end: 120 int instrs per
 *                                                     row-group cost more than
 *                                                     the 64 FP ones they saved
 *   V8 16-lane blocked quarters            1.65   dead end, and wrong: a 16-byte
 *                                                     load against an 8-lane
 *                                                     quarter maps half the
 *                                                     lanes to the wrong column
 *   V9 V6 with rows hoisted inside         1.67   dead end in C: t0[4]/a0[4] as
 *                                                     arrays spill to the stack.
 *                                                     Whether sharing plane loads
 *                                                     across 4 rows actually wins
 *                                                     can only be answered in asm,
 *                                                     where the registers are
 *                                                     chosen by hand.
 * Two traps in these numbers, both recorded so nobody quotes a wrong one: the
 * argv dispatch once compared argv[2][1] to 6 instead of '6', so every "variant"
 * silently ran the slow kernel and all five timings came out identical; and a
 * kernel that never stores its result gets its body deleted, which made V7 look
 * like 0.33 ns/weight (11.8x) until the missing store was fixed.
 * A72's vector int->float convert is ~9 cycles of throughput; the digit form
 * needs 40 per group, so it costs 1.6 ns/weight all by itself.
 */
#if VARIANT == 1              /* digit math + plane load, no convert, no fma */
#define QOPS(i, Q, a, b, d)                                                  \
    {                                                                        \
        float32x4_t fa = vld1q_f32(pg + i * 32 + (Q) * 8);                   \
        float32x4_t fb = vld1q_f32(pg + i * 32 + (Q) * 8 + 4);               \
        a = vaddq_f32(a, vaddq_f32(fa, vreinterpretq_f32_s16(d)));           \
        b = vaddq_f32(b, fb);                                                \
    }
#elif VARIANT == 2            /* digit math + convert, no plane, no fma */
#define QOPS(i, Q, a, b, d)                                                  \
    {                                                                        \
        a = vaddq_f32(a, vcvtq_f32_s32(vmovl_s16(vget_low_s16(d))));         \
        b = vaddq_f32(b, vcvtq_f32_s32(vmovl_s16(vget_high_s16(d))));        \
    }
#elif VARIANT == 3            /* plane load + fma, no digit math */
#define QOPS(i, Q, a, b, d)                                                  \
    {                                                                        \
        const float32x4_t kc = vdupq_n_f32(0.5f);                            \
        a = vfmaq_f32(a, kc, vld1q_f32(pg + i * 32 + (Q) * 8));              \
        b = vfmaq_f32(b, kc, vld1q_f32(pg + i * 32 + (Q) * 8 + 4));          \
    }
#elif VARIANT == 4            /* convert + fma, no digit math, no plane load */
#define QOPS(i, Q, a, b, d)                                                  \
    {                                                                        \
        const float32x4_t kc = vdupq_n_f32(0.5f);                            \
        a = vfmaq_f32(a, vcvtq_f32_s32(vmovl_s16(vget_low_s16(d))), kc);     \
        b = vfmaq_f32(b, vcvtq_f32_s32(vmovl_s16(vget_high_s16(d))), kc);    \
    }
#elif VARIANT == 5            /* nothing but the loads of the code bytes */
#define QOPS(i, Q, a, b, d) (void)(i); (void)(d);
#else                         /* the real kernel */
#define QOPS(i, Q, a, b, d)                                                  \
    {                                                                        \
        a = vfmaq_f32(a, vcvtq_f32_s32(vmovl_s16(vget_low_s16(d))),          \
                      vld1q_f32(pg + i * 32 + (Q) * 8));                     \
        b = vfmaq_f32(b, vcvtq_f32_s32(vmovl_s16(vget_high_s16(d))),         \
                      vld1q_f32(pg + i * 32 + (Q) * 8 + 4));                 \
    }
#endif

/* Quarter body, written as a macro so the accumulator and the plane offsets are
 * literals. Keeping the eight per-group accumulators in a C array made GCC push
 * and pop them every iteration; two named registers per quarter stay in regs. */
#define QUARTER(Q)                                                            \
    {                                                                         \
        int16x8_t v16 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(c + (Q) * 8)));\
        int16x8_t tp = v16;                                                   \
        float32x4_t a = vdupq_n_f32(0.0f), b = vdupq_n_f32(0.0f);              \
        for (int i = 0; i < 5; i++) {                                         \
            int16x8_t tc = vqdmulhq_s16(v16, vdupq_n_s16(MAGIC[i + 1]));       \
            int16x8_t d = vsubq_s16(tp, vmulq_n_s16(tc, 3));                   \
            QOPS(i, Q, a, b, d);  \
            tp = tc;                                                          \
        }                                                                     \
        S##Q##0 = vaddq_f32(S##Q##0, a);                                      \
        S##Q##1 = vaddq_f32(S##Q##1, b);                                      \
    }

static void gemv_neon(int rows, const uint8_t *w, float *y) {
    for (int r = 0; r < rows; r++) {
        const uint8_t *blk = w + (size_t)r * NG * 28;
        float acc = 0.0f;
        for (int g = 0; g < NG; g++) {
            const uint8_t *b = blk + g * 28;
            const float *pg = &PL[g][0][0];
            float scale = h2f(*(const uint16_t *)b);
            const uint8_t *c = b + 2;
            float32x4_t S00 = vdupq_n_f32(0.0f), S01 = vdupq_n_f32(0.0f);
            float32x4_t S10 = vdupq_n_f32(0.0f), S11 = vdupq_n_f32(0.0f);
            float32x4_t S20 = vdupq_n_f32(0.0f), S21 = vdupq_n_f32(0.0f);
            float32x4_t S30 = vdupq_n_f32(0.0f), S31 = vdupq_n_f32(0.0f);
            QUARTER(0) QUARTER(1) QUARTER(2) QUARTER(3)
            float32x4_t tot = vaddq_f32(vaddq_f32(S00, S01), vaddq_f32(S10, S11));
            tot = vaddq_f32(tot, vaddq_f32(vaddq_f32(S20, S21),
                                           vaddq_f32(S30, S31)));
            float32x2_t p2 = vpadd_f32(vget_low_f32(tot), vget_high_f32(tot));
            float tot_f = vget_lane_f32(vpadd_f32(p2, p2), 0);
            acc += scale * (tot_f - GS[g]);
        }
        y[r] = acc;
    }
}


/* ---- VARIANT 6: no integer->float conversion anywhere -------------------
 * A72's vector SCVTF/FCVT is ~9 cycles throughput, and the digit form needs 40
 * of them per group, which is the whole 1.6 ns/weight. Two changes kill them:
 *   1. float(v) comes from the magic-add trick (add 0x4B000000, subtract it
 *      back) - integer-pipe ops, no FP convert;
 *   2. the t-chain runs in float (t_j = floor(t_{j-1} * third)), and because
 *      sum_i d_i*P_i = sum_j t_j*(P_j - 3*P_{j-1}), the per-token planes become
 *      Q_j = P_j - 3*P_{j-1}. So the weight side is 5 FMAs per digit position
 *      and nothing else. third is biased by 1e-6 so exact multiples of 3 floor
 *      to the right quotient; every t is <= 255, so the bias can never cross an
 *      integer.
 * Cost: Q folds two plane values into one, which is algebraically the same sum
 * but not bit-identical - checked against the scalar reference here and against
 * the oracle argmaxes in the engine. */
static float QP[NG][5][32] __attribute__((aligned(16)));
static const float THIRD = 0.33333336f * (1.0f + 1e-6f);

static void build_q(void) {
    for (int g = 0; g < NG; g++)
        for (int i = 0; i < 5; i++)
            for (int k = 0; k < 32; k++)
                QP[g][i][k] = (i == 0) ? PL[g][0][k]
                                       : PL[g][i][k] - 3.0f * PL[g][i - 1][k];
}

static inline float32x4_t bytef32(uint8x8_t v, int half) {
    uint16x8_t w = vmovl_u8(v);
    uint32x4_t w32 = half ? vmovl_u16(vget_high_u16(w)) : vmovl_u16(vget_low_u16(w));
    const int32x4_t magic = vdupq_n_s32(0x4B000000);   /* as float: 8388608.0 */
    float32x4_t f = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_u32(w32),
                                                   magic));
    return vsubq_f32(f, vreinterpretq_f32_s32(magic));  /* exact for v < 2^23 */
}

#define QTR(VB, OFF, S0, S1)                                                 \
    {                                                                        \
        float32x4_t t0 = bytef32((VB), 0), t1 = bytef32((VB), 1);            \
        float32x4_t a = vdupq_n_f32(0.0f), bb = vdupq_n_f32(0.0f);           \
        a = vfmaq_f32(a, t0, vld1q_f32(qg + (OFF)));                         \
        bb = vfmaq_f32(bb, t1, vld1q_f32(qg + (OFF) + 4));                   \
        for (int j = 1; j < 5; j++) {                                        \
            t0 = vrndmq_f32(vmulq_f32(t0, third));                           \
            t1 = vrndmq_f32(vmulq_f32(t1, third));                           \
            a = vfmaq_f32(a, t0, vld1q_f32(qg + j * 32 + (OFF)));            \
            bb = vfmaq_f32(bb, t1, vld1q_f32(qg + j * 32 + (OFF) + 4));      \
        }                                                                    \
        S0 = vaddq_f32(S0, a); S1 = vaddq_f32(S1, bb);                       \
    }

static void gemv_neon6(int rows, const uint8_t *w, float *y) {
    const float32x4_t third = vdupq_n_f32(THIRD);
    for (int r = 0; r < rows; r++) {
        const uint8_t *blk = w + (size_t)r * NG * 28;
        float acc = 0.0f;
        for (int g = 0; g < NG; g++) {
            const uint8_t *b = blk + g * 28;
            const float *qg = &QP[g][0][0];
            float scale = h2f(*(const uint16_t *)b);
            const uint8_t *c = b + 2;
            uint8x16_t a0 = vld1q_u8(c), a1 = vld1q_u8(c + 16);
            float32x4_t S00 = vdupq_n_f32(0), S01 = vdupq_n_f32(0);
            float32x4_t S10 = vdupq_n_f32(0), S11 = vdupq_n_f32(0);
            float32x4_t S20 = vdupq_n_f32(0), S21 = vdupq_n_f32(0);
            float32x4_t S30 = vdupq_n_f32(0), S31 = vdupq_n_f32(0);
            QTR(vget_low_u8(a0), 0, S00, S01)
            QTR(vget_high_u8(a0), 8, S10, S11)
            QTR(vget_low_u8(a1), 16, S20, S21)
            QTR(vget_high_u8(a1), 24, S30, S31)
            float32x4_t tot = vaddq_f32(vaddq_f32(S00, S01), vaddq_f32(S10, S11));
            tot = vaddq_f32(tot, vaddq_f32(vaddq_f32(S20, S21),
                                           vaddq_f32(S30, S31)));
            float32x2_t p2 = vpadd_f32(vget_low_f32(tot), vget_high_f32(tot));
            acc += scale * (vget_lane_f32(vpadd_f32(p2, p2), 0) - GS[g]);
        }
        y[r] = acc;
    }
}


/* ---- VARIANT 7: 4-row blocking, integer digit chain, no converts ----------
 * Two more things to fix after V6:
 *   - every FMA carried two plane loads, so the load ports, not the FMA, set the
 *     rate (that is why V3 with no digit math at all still cost 0.54). Four rows
 *     share each plane load.
 *   - the float t-chain put a mul and a rint on the same FP pipe as the FMAs,
 *     and it was serial: t_{j+1} = rint(t_j * third). Computing each t_j
 *     directly from v with SQDMULH (integer pipe, ~0.25 instr per weight) makes
 *     all five chains independent and moves 2/3 of the work off FP1.
 * float(t_j) comes from the magic-add, which is integer-pipe too.
 */
static inline float32x4_t i16lo_f(int16x8_t v) {
    const int32x4_t magic = vdupq_n_s32(0x4B000000);
    float32x4_t f = vreinterpretq_f32_s32(vaddq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vreinterpretq_u16_s16(v)))), magic));
    return vsubq_f32(f, vreinterpretq_f32_s32(magic));
}
static inline float32x4_t i16hi_f(int16x8_t v) {
    const int32x4_t magic = vdupq_n_s32(0x4B000000);
    float32x4_t f = vreinterpretq_f32_s32(vaddq_s32(
        vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vreinterpretq_u16_s16(v)))), magic));
    return vsubq_f32(f, vreinterpretq_f32_s32(magic));
}

static void gemv_neon7(int rows, const uint8_t *w, float *y) {
    for (int r0 = 0; r0 < rows; r0 += 4) {
        const uint8_t *b[4];
        float sc[4];
        for (int k = 0; k < 4; k++) {
            b[k] = w + (size_t)(r0 + k) * NG * 28;
            sc[k] = 0.0f;
        }
        float tot[4] = {0, 0, 0, 0};
        for (int g = 0; g < NG; g++) {
            const float *qg = &QP[g][0][0];
            float gs = GS[g];
            float t4[4];
            for (int k = 0; k < 4; k++) {
                float acc = 0.0f;
                for (int q = 0; q < 4; q++) {
                    int16x8_t v16 = vreinterpretq_s16_u16(
                        vmovl_u8(vld1_u8(b[k] + g * 28 + 2 + q * 8)));
                    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
                    for (int j = 0; j < 5; j++) {
                        int16x8_t t = j == 0 ? v16
                            : vqdmulhq_s16(v16, vdupq_n_s16(MAGIC[j]));
                        a0 = vfmaq_f32(a0, i16lo_f(t), vld1q_f32(qg + j * 32 + q * 8));
                        a1 = vfmaq_f32(a1, i16hi_f(t), vld1q_f32(qg + j * 32 + q * 8 + 4));
                    }
                    float32x4_t s = vaddq_f32(a0, a1);
                    float32x2_t p = vadd_f32(vget_low_f32(s), vget_high_f32(s));
                    acc += vget_lane_f32(vpadd_f32(p, p), 0);
                }
                t4[k] = (acc - gs) * h2f(*(const uint16_t *)(b[k] + g * 28));
            }
            /* the four rows read the same 10 plane vectors per digit position;
             * the loop above still reloads them, which is what the next step
             * (reordering q/j outside k) removes */
            for (int k = 0; k < 4; k++) tot[k] += t4[k];
        }
        for (int k = 0; k < 4; k++) y[r0 + k] = tot[k];
    }
}


/* ---- VARIANT 8: V6's math, rows hoisted inside ---------------------------
 * V7 moved the digit chain to the integer pipe and lost (120 int instructions
 * per row-group beat the 64 FP ones it saved). What it never did was the actual
 * blocking: the row loop sat inside q, so every row reloaded the plane vectors.
 * Here the row loop is inside the digit loop, so one plane load feeds four rows.
 * Register budget: t0/t1 x4 rows (8) + acc x4 rows (8) + 2 planes = 18.
 */
static void gemv_neon8(int rows, const uint8_t *w, float *y) {
    const float32x4_t third = vdupq_n_f32(THIRD);
    for (int r0 = 0; r0 < rows; r0 += 4) {
        const uint8_t *b[4];
        float tot[4] = {0, 0, 0, 0};
        for (int k = 0; k < 4; k++) b[k] = w + (size_t)(r0 + k) * NG * 28;
        for (int g = 0; g < NG; g++) {
            const float *qg = &QP[g][0][0];
            float part[4];
            for (int k = 0; k < 4; k++) part[k] = 0.0f;
            for (int q = 0; q < 4; q++) {
                float32x4_t t0[4], t1[4], a0[4], a1[4];
                for (int k = 0; k < 4; k++) {
                    uint8x8_t vb = vget_low_u8(vld1q_u8(b[k] + g * 28 + 2 + q * 8));
                    uint8x8_t wl = vb, wh = vget_high_u8(vld1q_u8(b[k] + g * 28 + 2 + q * 8));
                    t0[k] = bytef32(wl, 0); t1[k] = bytef32(wh, 0);
                    (void)vb;
                    a0[k] = vdupq_n_f32(0); a1[k] = vdupq_n_f32(0);
                }
                for (int j = 0; j < 5; j++) {
                    float32x4_t p0 = vld1q_f32(qg + j * 32 + q * 8);
                    float32x4_t p1 = vld1q_f32(qg + j * 32 + q * 8 + 4);
                    for (int k = 0; k < 4; k++) {
                        a0[k] = vfmaq_f32(a0[k], t0[k], p0);
                        a1[k] = vfmaq_f32(a1[k], t1[k], p1);
                        if (j < 4) {
                            t0[k] = vrndmq_f32(vmulq_f32(t0[k], third));
                            t1[k] = vrndmq_f32(vmulq_f32(t1[k], third));
                        }
                    }
                }
                for (int k = 0; k < 4; k++) {
                    float32x4_t s = vaddq_f32(a0[k], a1[k]);
                    float32x2_t pp = vadd_f32(vget_low_f32(s), vget_high_f32(s));
                    part[k] += vget_lane_f32(vpadd_f32(pp, pp), 0);
                }
            }
            for (int k = 0; k < 4; k++)
                tot[k] += (part[k] - GS[g]) * h2f(*(const uint16_t *)(b[k] + g * 28));
        }
        for (int k = 0; k < 4; k++) y[r0 + k] = tot[k];
    }
}


/* ---- VARIANT 9: V6 with the row loop hoisted inside ----------------------
 * V8 had the right idea and the wrong lane mapping (it loaded 16 code bytes for
 * an 8-lane quarter, so half the lanes indexed the wrong plane column, and the
 * 16-lane version needs 32 registers). This keeps V6's 8-byte quarters and just
 * moves the row loop inside the digit loop, so each plane vector is loaded once
 * and used by four rows: 10 plane loads per row-group instead of 40.
 */
static void gemv_neon9(int rows, const uint8_t *w, float *y) {
    const float32x4_t third = vdupq_n_f32(THIRD);
    for (int r0 = 0; r0 < rows; r0 += 4) {
        const uint8_t *b[4];
        float tot[4] = {0, 0, 0, 0};
        for (int k = 0; k < 4; k++) b[k] = w + (size_t)(r0 + k) * NG * 28;
        for (int g = 0; g < NG; g++) {
            const float *qg = &QP[g][0][0];
            float part[4] = {0, 0, 0, 0};
            for (int q = 0; q < 4; q++) {
                float32x4_t t0[4], t1[4], a0[4], a1[4];
                for (int k = 0; k < 4; k++) {
                    uint8x8_t vb = vld1_u8(b[k] + g * 28 + 2 + q * 8);
                    t0[k] = bytef32(vb, 0); t1[k] = bytef32(vb, 1);
                    a0[k] = vdupq_n_f32(0.0f); a1[k] = vdupq_n_f32(0.0f);
                }
                for (int j = 0; j < 5; j++) {
                    float32x4_t p0 = vld1q_f32(qg + j * 32 + q * 8);
                    float32x4_t p1 = vld1q_f32(qg + j * 32 + q * 8 + 4);
                    for (int k = 0; k < 4; k++) {
                        a0[k] = vfmaq_f32(a0[k], t0[k], p0);
                        a1[k] = vfmaq_f32(a1[k], t1[k], p1);
                        if (j < 4) {
                            t0[k] = vrndmq_f32(vmulq_f32(t0[k], third));
                            t1[k] = vrndmq_f32(vmulq_f32(t1[k], third));
                        }
                    }
                }
                for (int k = 0; k < 4; k++) {
                    float32x4_t sm = vaddq_f32(a0[k], a1[k]);
                    float32x2_t pp = vadd_f32(vget_low_f32(sm), vget_high_f32(sm));
                    part[k] += vget_lane_f32(vpadd_f32(pp, pp), 0);
                }
            }
            for (int k = 0; k < 4; k++)
                tot[k] += (part[k] - GS[g]) * h2f(*(const uint16_t *)(b[k] + g * 28));
        }
        for (int k = 0; k < 4; k++) y[r0 + k] = tot[k];
    }
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/home/luigi/bonsai_b3.bin";
    int fd = open(path, O_RDONLY); struct stat st; fstat(fd, &st);
    uint8_t *m = mmap(0, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    uint64_t hl; memcpy(&hl, m, 8);
    char *json = strndup((char *)m + 8, hl);
    char *p = strstr(json, "\"model.layers.5.self_attn.q_proj.weight\"");
    char *q = strchr(strstr(p, "\"data_offsets\""), '[');
    long o0, o1; sscanf(q + 1, " %ld , %ld", &o0, &o1);
    const uint8_t *W = m + 8 + hl + o0;

    /* the magic constants must be exact floor division for every code byte */
    int bad = 0;
    for (int j = 1; j < 6; j++)
        for (int v = 0; v < 256; v++) {
            uint32_t t = (((uint32_t)v * (uint32_t)MAGIC[j]) << 1) >> 16;
            uint32_t want = (uint32_t)v; for (int k = 0; k < j; k++) want /= 3;
            if (t != want) { if (bad < 3) printf("MAGIC[%d] wrong at v=%d: %u != %u\n", j, v, t, want); bad++; }
        }
    printf("magic divide check: %s (%d bad)\n", bad ? "FAIL" : "exact for all 256 bytes", bad);
    if (bad) return 1;

    static float X[COLS];
    static float YA[ROWS], YB[ROWS];
    for (int i = 0; i < COLS; i++) X[i] = ((i * 37) % 251) / 125.0f - 1.0f;
    deinterleave(X);
    ternary_gemv_b3(ROWS, COLS, W, X, YA);
    if (argc > 2 && argv[2][1] == '6') { build_q(); gemv_neon6(ROWS, W, YB); } else if (argc > 2 && argv[2][1] == '7') { build_q(); gemv_neon7(ROWS, W, YB); } else if (argc > 2 && argv[2][1] == '8') { build_q(); gemv_neon8(ROWS, W, YB); } else if (argc > 2 && argv[2][1] == '9') { build_q(); gemv_neon9(ROWS, W, YB); } else gemv_neon(ROWS, W, YB);
    double mx = 0, md = 0;
    for (int r = 0; r < ROWS; r++) {
        double d = fabs(YA[r] - YB[r]);
        if (fabs(YA[r]) > mx) mx = fabs(YA[r]);
        if (d > md) md = d;
    }
    printf("agreement vs scalar: max|ref| %.6f  max abs diff %.3e (%.2e rel)\n",
           mx, md, md / (mx + 1e-30));

    double bn = 1e30, bb = 1e30; struct timespec a, b;
    volatile float sink = 0;   /* keeps timed kernels alive */
    for (int t = 0; t < 4; t++) {
        clock_gettime(CLOCK_MONOTONIC, &a); ternary_gemv_b3(ROWS, COLS, W, X, YA);
        clock_gettime(CLOCK_MONOTONIC, &b);
        sink += YA[0];
        sink += YA[0];
        double s = (b.tv_sec - a.tv_sec) + 1e-9 * (b.tv_nsec - a.tv_nsec);
        if (s < bn) bn = s;
    }
    for (int t = 0; t < 4; t++) {
        clock_gettime(CLOCK_MONOTONIC, &a);
        if (argc > 2 && argv[2][1] == '6') gemv_neon6(ROWS, W, YB);
        else if (argc > 2 && argv[2][1] == '7') gemv_neon7(ROWS, W, YB);
        else if (argc > 2 && argv[2][1] == '8') gemv_neon8(ROWS, W, YB);
        else if (argc > 2 && argv[2][1] == '9') gemv_neon9(ROWS, W, YB);
        else gemv_neon(ROWS, W, YB);
        clock_gettime(CLOCK_MONOTONIC, &b);
        sink += YB[0];
        sink += YB[0];
        double s = (b.tv_sec - a.tv_sec) + 1e-9 * (b.tv_nsec - a.tv_nsec);
        if (s < bb) bb = s;
    }
    double nw = (double)ROWS * COLS;
    printf("scalar %.2f ns/weight   neon %.2f ns/weight   speedup %.1fx\n",
           bn * 1e9 / nw, bb * 1e9 / nw, bn / bb);
    printf("projected token: scalar %.0f ms   neon %.0f ms   (FMA floor 243 ms)\n",
           bn * 1e9 / nw * 1.7e9 / 1e6, bb * 1e9 / nw * 1.7e9 / 1e6);
    return 0;
}
