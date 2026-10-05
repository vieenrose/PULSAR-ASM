/* PULSAR-ARM | kernels/neon_gemv.c
 *
 * BF16 GEMV for Gemma-3 270m on Cortex-A72 (NEON only: no bfdot, no sdot,
 * no SVE on this chip, so bf16 is widened to fp32 explicitly).
 *
 *   y[m] = sum_k fp32(W[m*K + k]) * x[k]      fp32 accumulation
 *
 * Widening is exact (bf16 -> fp32 is a left shift by 16); the only rounding
 * is the fp32 accumulation order, same as every other implementation. The
 * parity test pins rel < 1e-3 against fp32 numpy, dominated by the bf16
 * input rounding both sides share.
 */
#include <arm_neon.h>
#include <stdint.h>
#include <string.h>

void gemv_bf16(int M, int K, const uint16_t *W, const float *x, float *y) {
    for (int m = 0; m < M; m++) {
        const uint16_t *row = W + (int64_t)m * K;
        float32x4_t a0 = vdupq_n_f32(0.0f);
        float32x4_t a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f);
        float32x4_t a3 = vdupq_n_f32(0.0f);
        int k = 0;
        for (; k + 16 <= K; k += 16) {
            uint16x8_t w8a = vld1q_u16(row + k);
            uint16x8_t w8b = vld1q_u16(row + k + 8);
            float32x4_t w0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(w8a), 16));
            float32x4_t w1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(w8a), 16));
            float32x4_t w2 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(w8b), 16));
            float32x4_t w3 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(w8b), 16));
            a0 = vfmaq_f32(a0, w0, vld1q_f32(x + k));
            a1 = vfmaq_f32(a1, w1, vld1q_f32(x + k + 4));
            a2 = vfmaq_f32(a2, w2, vld1q_f32(x + k + 8));
            a3 = vfmaq_f32(a3, w3, vld1q_f32(x + k + 12));
        }
        float s = vaddvq_f32(a0) + vaddvq_f32(a1) + vaddvq_f32(a2) + vaddvq_f32(a3);
        for (; k < K; k++) {
            uint32_t u = (uint32_t)row[k] << 16;
            float w;
            memcpy(&w, &u, 4);
            s += w * x[k];
        }
        y[m] = s;
    }
}
