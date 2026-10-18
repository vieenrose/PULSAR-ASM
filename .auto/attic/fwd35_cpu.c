/* fwd35 E_CPUF: the A57 share of a prefill (nt > 1) batched GEMV, linked into fwd35_sm53.cu
 * (nvcc's front end cannot parse arm_neon.h, so this is plain C for gcc).
 *
 * Rows are the last ones of a tensor, copied from the GPU planes at load with each 32-weight
 * word's bits permuted into k_hg's half order (element e of a word is fp16 half e of its 16
 * q/2 words), as nz = nonzero and ng = negative masks. Per (row, token) the arithmetic is
 * k_hg's: the integer word sum s = sum w * q (q = 2 * (q/2), exact), si = fma(d8, s, si) over
 * the group's 4 words from 0, sumf = fma(scale, si, sumf) over the groups. Lanes are tokens,
 * and a NEON fma lane is fmaf, so the bits are equal. */
#include <arm_neon.h>
#include <stdint.h>
#include <stdlib.h>

static inline int8x16_t cpu_ex(uint32_t b16, uint8x16_t sel) {
    const uint8x16_t b = vcombine_u8(vdup_n_u8(b16 & 0xff), vdup_n_u8((b16 >> 8) & 0xff));
    return vreinterpretq_s8_u8(vtstq_u8(b, sel));
}
/* w = nz ? (ng ? -1 : 1) : 0 from 0 / -1 bytes: -nz + 2 ng */
static inline int8x16_t cpu_w(int8x16_t z, int8x16_t n) { return vaddq_s8(vnegq_s8(z), vshlq_n_s8(n, 1)); }

static int8_t *cpu_q; static float *cpu_d; static size_t cpu_qcap, cpu_dcap;

/* hq: fp16 q/2, token t at hq + t * hqs halves; dw: d8 words, token t group g word k at
 * dw[t * dws + g * 40 + k]; out[t * rows + r] */
void cpu_hg(const uint32_t *nz, const uint32_t *ng, const float *sc, int rows, int G,
            const uint16_t *hq, int hqs, const float *dw, int dws, int nt, float *out) {
    const int K = G * 128, W = G * 4;
    if (cpu_qcap < (size_t)8 * K) { free(cpu_q); cpu_qcap = (size_t)8 * K; cpu_q = (int8_t *)malloc(cpu_qcap); }
    if (cpu_dcap < (size_t)8 * W) { free(cpu_d); cpu_dcap = (size_t)8 * W; cpu_d = (float *)malloc(cpu_dcap * 4); }
    int8_t *q = cpu_q; float *dT = cpu_d;
    for (int t = 0; t < 8; t++) {
        const __fp16 *h = (const __fp16 *)hq + (size_t)t * hqs;
        for (int e = 0; e < K; e++) q[(size_t)t * K + e] = t < nt ? (int8_t)(int)(2.0f * (float)h[e]) : 0;
        for (int g = 0; g < G; g++)
            for (int k = 0; k < 4; k++) dT[(g * 4 + k) * 8 + t] = t < nt ? dw[(size_t)t * dws + g * 40 + k] : 0.0f;
    }
    const uint8x16_t sel = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
#pragma omp parallel for schedule(static)
    for (int r0 = 0; r0 < rows; r0 += 2) {
        float32x4_t F[2][2];
        for (int i = 0; i < 2; i++) F[i][0] = F[i][1] = vdupq_n_f32(0.0f);
        for (int g = 0; g < G; g++) {
            float32x4_t S[2][2];
            for (int i = 0; i < 2; i++) S[i][0] = S[i][1] = vdupq_n_f32(0.0f);
            for (int k = 0; k < 4; k++) {
                const int w = g * 4 + k;
                int8x16_t w0[2], w1[2];
                for (int i = 0; i < 2; i++) {
                    const uint32_t a = nz[(size_t)(r0 + i) * W + w], b = ng[(size_t)(r0 + i) * W + w];
                    w0[i] = cpu_w(cpu_ex(a, sel), cpu_ex(b, sel));
                    w1[i] = cpu_w(cpu_ex(a >> 16, sel), cpu_ex(b >> 16, sel));
                }
                int16x8_t s[2][8];
                for (int t = 0; t < 8; t++) {
                    const int8x16_t q0 = vld1q_s8(q + (size_t)t * K + w * 32), q1 = vld1q_s8(q + (size_t)t * K + w * 32 + 16);
                    for (int i = 0; i < 2; i++) s[i][t] = vpadalq_s8(vpaddlq_s8(vmulq_s8(w0[i], q0)), vmulq_s8(w1[i], q1));
                }
                const float32x4_t d0 = vld1q_f32(dT + w * 8), d1 = vld1q_f32(dT + w * 8 + 4);
                for (int i = 0; i < 2; i++) {
                    int16x8_t p0 = vpaddq_s16(s[i][0], s[i][1]), p1 = vpaddq_s16(s[i][2], s[i][3]);
                    int16x8_t p2 = vpaddq_s16(s[i][4], s[i][5]), p3 = vpaddq_s16(s[i][6], s[i][7]);
                    p0 = vpaddq_s16(vpaddq_s16(p0, p1), vpaddq_s16(p2, p3)); /* lane t: token t's word sum */
                    S[i][0] = vfmaq_f32(S[i][0], d0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(p0))));
                    S[i][1] = vfmaq_f32(S[i][1], d1, vcvtq_f32_s32(vmovl_high_s16(p0)));
                }
            }
            for (int i = 0; i < 2; i++) {
                const float32x4_t c = vdupq_n_f32(sc[(size_t)(r0 + i) * G + g]);
                F[i][0] = vfmaq_f32(F[i][0], c, S[i][0]); F[i][1] = vfmaq_f32(F[i][1], c, S[i][1]);
            }
        }
        for (int i = 0; i < 2; i++) {
            float o[8]; vst1q_f32(o, F[i][0]); vst1q_f32(o + 4, F[i][1]);
            for (int t = 0; t < nt; t++) out[(size_t)t * rows + r0 + i] = o[t];
        }
    }
}
