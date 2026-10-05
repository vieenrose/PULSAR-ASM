/* PULSAR-ARM | kernels/neon_ops.c
 *
 * Pointwise/attention kernels for Gemma-3 270m. Plain C first, NEON where it
 * is one instruction (rms energy, exp-free paths stay scalar until profiled).
 * All fp32; bf16 weights are widened at the GEMV boundary only.
 */
#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

/* out[i] = x[i] * rsqrt(mean(x^2) + eps) * w[i] */
void rmsnorm_f32(float *out, const float *w, const float *x, int n, float eps) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4)
        acc = vfmaq_f32(acc, vld1q_f32(x + i), vld1q_f32(x + i));
    float e = vaddvq_f32(acc);
    for (; i < n; i++)
        e += x[i] * x[i];
    float inv = 1.0f / sqrtf(e / n + eps);
    for (i = 0; i + 4 <= n; i += 4)
        vst1q_f32(out + i, vmulq_n_f32(vmulq_f32(vld1q_f32(x + i), vld1q_f32(w + i)), inv));
    for (; i < n; i++)
        out[i] = x[i] * inv * w[i];
}

/* in-place rotate_half pairs (i, i+h) over 2*h floats */
void rope_half(float *v, const float *cos, const float *sin, int h) {
    for (int i = 0; i < h; i++) {
        float a = v[i], b = v[i + h];
        v[i] = a * cos[i] - b * sin[i];
        v[i + h] = b * cos[i] + a * sin[i];
    }
}

/* in-place softmax over n (max-subtracted, expf) */
void softmax_f32(float *buf, int n) {
    float m = buf[0];
    for (int i = 1; i < n; i++)
        if (buf[i] > m)
            m = buf[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        buf[i] = expf(buf[i] - m);
        s += buf[i];
    }
    float inv = 1.0f / s;
    for (int i = 0; i < n; i++)
        buf[i] *= inv;
}

/* out[i] = gelu_tanh(x[i]) (matches hidden_activation gelu_pytorch_tanh) */
void gelu_tanh_f32(float *out, const float *x, int n) {
    const float c1 = 0.7978845608028654f;   /* sqrt(2/pi) */
    const float c3 = 0.044715f * c1;
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float t = tanhf(c1 * v + c3 * v * v * v);
        out[i] = 0.5f * v * (1.0f + t);
    }
}

/* scores[p] = dot(q, K[p]) * scale over n rows of dim hd (K row-major).
 * The 1/sqrt(d) scale is fused on the store: identical flops to scaling
 * the buffer afterwards, one less memory pass. */
void attn_scores_f32(float *s, const float *q, const float *K, int n, int hd, float scale) {
    for (int p = 0; p < n; p++) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        int d = 0;
        for (; d + 4 <= hd; d += 4)
            acc = vfmaq_f32(acc, vld1q_f32(K + (int64_t)p * hd + d), vld1q_f32(q + d));
        float t = vaddvq_f32(acc);
        for (; d < hd; d++)
            t += K[(int64_t)p * hd + d] * q[d];
        s[p] = t * scale;
    }
}

/* out[i] = gelu_tanh(x[i]) * mult[i] (gate*up fuse; in-place safe) */
void gelu_mul_f32(float *out, const float *x, const float *mult, int n) {
    const float c1 = 0.7978845608028654f;
    const float c3 = 0.044715f * c1;
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float t = tanhf(c1 * v + c3 * v * v * v);
        out[i] = 0.5f * v * (1.0f + t) * mult[i];
    }
}

/* out[i] = x[i]*inv*w[i]; acc[i] += out[i]  (norm+residual fuse) */
void rmsnorm_add_f32(float *out, const float *w, const float *x, float *res, int n, float eps) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4)
        acc = vfmaq_f32(acc, vld1q_f32(x + i), vld1q_f32(x + i));
    float e = vaddvq_f32(acc);
    for (; i < n; i++)
        e += x[i] * x[i];
    float inv = 1.0f / sqrtf(e / n + eps);
    for (i = 0; i + 4 <= n; i += 4) {
        float32x4_t v = vmulq_n_f32(vmulq_f32(vld1q_f32(x + i), vld1q_f32(w + i)), inv);
        vst1q_f32(out + i, v);
        vst1q_f32(res + i, vaddq_f32(vld1q_f32(res + i), v));
    }
    for (; i < n; i++) {
        out[i] = x[i] * inv * w[i];
        res[i] += out[i];
    }
}

/* out[d] = sum_p w[p] * V[p][d] over n rows of dim hd */
void attn_values_f32(float *out, const float *V, const float *w, int n, int hd) {
    for (int d = 0; d < hd; d += 4) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (int p = 0; p < n; p++)
            acc = vfmaq_f32(acc, vld1q_f32(V + (int64_t)p * hd + d),
                             vld1q_dup_f32(w + p));
        vst1q_f32(out + d, acc);
    }
}

/* argmax over n fp32 (first max wins ties) */
int argmax_f32(const float *buf, int n) {
    int bi = 0;
    float best = buf[0];
    for (int i = 1; i < n; i++) {
        if (buf[i] > best) {
            best = buf[i];
            bi = i;
        }
    }
    return bi;
}

/* out[i] = fp32(emb[token*dim + i]) * scale  (bf16 row gather + widen) */
void embed_row_f32(float *out, const uint16_t *emb, int token, int dim, float scale) {
    const uint16_t *row = emb + (int64_t)token * dim;
    int i = 0;
    float32x4_t sc = vdupq_n_f32(scale);
    for (; i + 8 <= dim; i += 8) {
        uint16x8_t w8 = vld1q_u16(row + i);
        float32x4_t w0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(w8), 16));
        float32x4_t w1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(w8), 16));
        vst1q_f32(out + i, vmulq_f32(w0, sc));
        vst1q_f32(out + i + 4, vmulq_f32(w1, sc));
    }
    for (; i < dim; i++) {
        uint32_t u = (uint32_t)row[i] << 16;
        float w;
        memcpy(&w, &u, 4);
        out[i] = w * scale;
    }
}
