/* lfm3_depth_neon.c -- the depth engine's Q8 GEMM kernels (host, NEON + OpenMP).
 *
 * Split out of lfm3_depth.cu because nvcc 10.2's front end cannot parse GCC's arm_neon.h
 * ("identifier __Int8x8_t is undefined"), even in the host pass. Built by gcc and linked:
 *   gcc -O2 -fopenmp -c lfm3_depth_neon.c -o lfm3_depth_neon.o
 *   nvcc -std=c++11 -gencode arch=compute_53,code=sm_53 -O2 -Xcompiler -fopenmp \
 *        -o lfm3_depth lfm3_depth.cu lfm3_depth_neon.o -lgomp
 *
 * Contract: each output is acc = 0; for k ascending: acc = fma(w_k, x_k, acc) with
 * w_k = (float)code_k * scale_{k/32} -- exactly the per-position loop the engine was gated
 * with (it compiled to scalar fmadd). vfmaq_f32 is the same IEEE fused multiply-add per lane,
 * so results are bit-identical to the original engine; the change is gated by comparing
 * logits byte for byte against the pre-change binary, then by the S4 gates.
 *
 * Why the layout: each weight row is streamed once for all 12 positions (the per-position
 * GEMV re-read it 12 times); X is k-major so the 12 activations of one k are contiguous;
 * the 12 accumulators are three NEON registers (at -O2 an acc[12] array stayed on the stack,
 * one load/fmadd/store per FMA, ~38 cycles each); rows are split across the four cores.
 */
#include <stddef.h>
#include <stdint.h>
#include <arm_neon.h>

/* 4 rows per step: the 12 activations of one k are loaded once and feed 4 rows x 3
   registers = 12 independent fmla chains. One row per step re-streamed the whole k-major
   X block (48 KB at K=1024, 135 KB at K=2816 -- past L1) for every row and left only 3
   chains to hide the fmla latency: measured ~90 cycles per k at one thread. Each row still
   accumulates its own outputs in k order with the same fused op. */
static void rows4_12(const int8_t *q, const float *s, const float *XT, float *Y, int ldy,
                     int n, int K, int nb) {
  float32x4_t a[4][3];
  for (int r = 0; r < 4; r++) for (int i = 0; i < 3; i++) a[r][i] = vdupq_n_f32(0.f);
  const int8_t *q0 = q + (size_t)n * K, *q1 = q0 + K, *q2 = q1 + K, *q3 = q2 + K;
  const float *s0 = s + (size_t)n * nb, *s1 = s0 + nb, *s2 = s1 + nb, *s3 = s2 + nb;
  for (int b = 0; b < nb; b++) {
    const float c0 = s0[b], c1 = s1[b], c2 = s2[b], c3 = s3[b];
    for (int j = 0; j < 32; j++) {
      const int k = b * 32 + j;
      const float *xk = XT + (size_t)k * 12;
      const float32x4_t x0 = vld1q_f32(xk), x1 = vld1q_f32(xk + 4), x2 = vld1q_f32(xk + 8);
      const float32x4_t w0 = vdupq_n_f32((float)q0[k] * c0);
      const float32x4_t w1 = vdupq_n_f32((float)q1[k] * c1);
      const float32x4_t w2 = vdupq_n_f32((float)q2[k] * c2);
      const float32x4_t w3 = vdupq_n_f32((float)q3[k] * c3);
      a[0][0] = vfmaq_f32(a[0][0], w0, x0); a[0][1] = vfmaq_f32(a[0][1], w0, x1); a[0][2] = vfmaq_f32(a[0][2], w0, x2);
      a[1][0] = vfmaq_f32(a[1][0], w1, x0); a[1][1] = vfmaq_f32(a[1][1], w1, x1); a[1][2] = vfmaq_f32(a[1][2], w1, x2);
      a[2][0] = vfmaq_f32(a[2][0], w2, x0); a[2][1] = vfmaq_f32(a[2][1], w2, x1); a[2][2] = vfmaq_f32(a[2][2], w2, x2);
      a[3][0] = vfmaq_f32(a[3][0], w3, x0); a[3][1] = vfmaq_f32(a[3][1], w3, x1); a[3][2] = vfmaq_f32(a[3][2], w3, x2);
    }
  }
  for (int r = 0; r < 4; r++) {
    float acc[12];
    vst1q_f32(acc, a[r][0]); vst1q_f32(acc + 4, a[r][1]); vst1q_f32(acc + 8, a[r][2]);
    for (int t = 0; t < 12; t++) Y[(size_t)t * ldy + n + r] = acc[t];
  }
}
static void row1_12(const int8_t *q, const float *s, const float *XT, float *Y, int ldy,
                    int n, int K, int nb) {
  float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f), a2 = vdupq_n_f32(0.f);
  const int8_t *qr = q + (size_t)n * K;
  const float *sr = s + (size_t)n * nb;
  for (int b = 0; b < nb; b++) {
    const float sc = sr[b];
    for (int j = 0; j < 32; j++) {
      const int k = b * 32 + j;
      const float32x4_t vw = vdupq_n_f32((float)qr[k] * sc);
      const float *xk = XT + (size_t)k * 12;
      a0 = vfmaq_f32(a0, vw, vld1q_f32(xk));
      a1 = vfmaq_f32(a1, vw, vld1q_f32(xk + 4));
      a2 = vfmaq_f32(a2, vw, vld1q_f32(xk + 8));
    }
  }
  float acc[12];
  vst1q_f32(acc, a0); vst1q_f32(acc + 4, a1); vst1q_f32(acc + 8, a2);
  for (int t = 0; t < 12; t++) Y[(size_t)t * ldy + n] = acc[t];
}
void lfm3_q8_gemm12_kmajor(const int8_t *q, const float *s, const float *XT,
                           float *Y, int ldy, int N, int K) {
  const int nb = K / 32;
  const int n4 = N / 4;
  #pragma omp parallel for schedule(static)
  for (int g = 0; g < n4; g++) rows4_12(q, s, XT, Y, ldy, g * 4, K, nb);
  for (int n = n4 * 4; n < N; n++) row1_12(q, s, XT, Y, ldy, n, K, nb);
}

/* one position, 4 rows per step: 4 independent scalar chains share each x[k] load */
void lfm3_q8_gemv1(const int8_t *q, const float *s, const float *x, float *y, int N, int K) {
  const int nb = K / 32;
  const int n4 = N / 4;
  #pragma omp parallel for schedule(static)
  for (int g = 0; g < n4; g++) {
    const int n = g * 4;
    const int8_t *q0 = q + (size_t)n * K, *q1 = q0 + K, *q2 = q1 + K, *q3 = q2 + K;
    const float *s0 = s + (size_t)n * nb, *s1 = s0 + nb, *s2 = s1 + nb, *s3 = s2 + nb;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (int b = 0; b < nb; b++) {
      const float c0 = s0[b], c1 = s1[b], c2 = s2[b], c3 = s3[b];
      for (int j = 0; j < 32; j++) {
        const int k = b * 32 + j;
        const float xv = x[k];
        const float w0 = (float)q0[k] * c0, w1 = (float)q1[k] * c1;
        const float w2 = (float)q2[k] * c2, w3 = (float)q3[k] * c3;
        a0 += w0 * xv; a1 += w1 * xv; a2 += w2 * xv; a3 += w3 * xv;
      }
    }
    y[n] = a0; y[n + 1] = a1; y[n + 2] = a2; y[n + 3] = a3;
  }
  for (int n = n4 * 4; n < N; n++) {
    const int8_t *qr = q + (size_t)n * K;
    const float *sr = s + (size_t)n * nb;
    float acc = 0.f;
    for (int b = 0; b < nb; b++) {
      const float sc = sr[b];
      for (int j = 0; j < 32; j++) {
        const int k = b * 32 + j;
        const float w = (float)qr[k] * sc;
        acc += w * x[k];
      }
    }
    y[n] = acc;
  }
}
