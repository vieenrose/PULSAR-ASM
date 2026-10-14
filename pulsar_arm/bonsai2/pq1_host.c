/* pq1_host.c — Q1_0 split repack + fork-shaped host reference (no CUDA). */
#include "pq1_wrap.h"
#include "pq1_expf_neon.h"
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <math.h>

float pq1_h2f(uint16_t h) {
  uint32_t s = (uint32_t)(h & 0x8000u) << 16;
  uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu, o;
  if (e == 0) {
    if (!m) return 0.0f;
    float f = (float)m * 5.9604645e-8f;
    memcpy(&o, &f, 4); o |= s;
  } else if (e == 31) o = s | 0x7f800000u | (m << 13);
  else o = s | ((e + 112u) << 23) | (m << 13);
  float r; memcpy(&r, &o, 4); return r;
}

void pq1_split_repack(const uint8_t *q1, int rows, int K, uint32_t *w_bits,
                      uint16_t *w_scales_h) {
  int nb = K / 128;                 /* 18B blocks per row */
#ifdef _OPENMP
  /* Pure data movement: each row writes a disjoint slice, so the loop is a
   * partition of the output and the result is bit-identical to the serial loop,
   * at any thread count. The repack is the dominant part of the engine's startup
   * (it walks all 7.57G weights), so parallelising it is a pure startup win. */
#pragma omp parallel for schedule(static)
#endif
  for (int r = 0; r < rows; r++) {
    const uint8_t *row = q1 + (size_t)r * nb * 18;
    uint32_t *wb = w_bits + (size_t)r * (K / 32);
    uint16_t *sc = w_scales_h + (size_t)r * nb;
    for (int b = 0; b < nb; b++) {
      memcpy(wb + b * 4, row + b * 18 + 2, 16); /* 128 bits, LSB-first */
      memcpy(sc + b, row + b * 18, 2);          /* fp16 scale, verbatim */
    }
  }
}

void pq1_gemv_cpu(int rows, int K, const uint32_t *w_bits,
                  const uint16_t *w_scales_h, const uint8_t *xq, float *y) {
  int nb = K / 128;
  for (int r = 0; r < rows; r++) {
    const uint32_t *wb = w_bits + (size_t)r * (K / 32);
    float acc = 0.0f;
    for (int b = 0; b < nb; b++) {
      const uint8_t *qb = xq + (size_t)(b >> 1) * 292;
      float db; memcpy(&db, qb, 4);
      const int8_t *qs = (const int8_t *)(qb + 4 + (b & 1) * 128);
      float ws = pq1_h2f(w_scales_h[(size_t)r * nb + b]);
      const uint32_t *w4 = wb + b * 4;
      int isum = 0;
      for (int i = 0; i < 128; i++) {
        int bit = (w4[i >> 5] >> (i & 31)) & 1;
        isum += (bit ? 1 : -1) * (int)qs[i];
      }
      acc += (ws * db) * (float)isum;
    }
    y[r] = acc;
  }
}

/* T-MAC bit-plane repack of the Q8_K activations (run #41 experiment).
 *
 * For every 32 activations (one 32-bit code word = 32 weights) build the 8
 * bit-plane words: plane word p has bit a = bit p of the unsigned biased byte
 * x' = (uint8)(x + 128) of activation (32c + a). Then for a code word W
 *   Pu = sum_p 2^p * popc(plane[c][p] & W)
 * is EXACTLY the sum of x' over the selected weights: each activation
 * contributes its own byte value once (bit a of plane p carries bit p of that
 * byte, weighted by 2^p). Integer, so order-independent and bit-exact.
 *
 * Size: 8 words per 32 activations = 1 byte per activation - byte-identical
 * traffic to the int8 array it replaces, and the block stride (73 words =
 * 292B) mirrors Q8_K so the kernel's staging code and addresses are unchanged
 * (word 0 = the fp32 scale d, words 1..64 = 64 plane words). */
void pq1_build_planes(const uint8_t *xq, int K, uint32_t *planes) {
  int nb = K >> 8;
  for (int b = 0; b < nb; b++) {
    const uint8_t *src = xq + (size_t)b * 292;
    uint32_t *dst = planes + (size_t)b * 73;
    memcpy(dst, src, 4);            /* fp32 scale d, verbatim (word 0) */
    memset(dst + 1, 0, 64 * 4);
    const uint8_t *act = src + 4;   /* 256 int8 activations */
    for (int i = 0; i < 256; i++) {
      unsigned v = (unsigned)act[i] ^ 0x80u;   /* x' = x + 128, unsigned */
      uint32_t *pw = dst + 1 + (i >> 5) * 8;
      uint32_t m = 1u << (i & 31);
      if (v & 1u) pw[0] |= m;
      if (v & 2u) pw[1] |= m;
      if (v & 4u) pw[2] |= m;
      if (v & 8u) pw[3] |= m;
      if (v & 16u) pw[4] |= m;
      if (v & 32u) pw[5] |= m;
      if (v & 64u) pw[6] |= m;
      if (v & 128u) pw[7] |= m;
    }
  }
}

/* Au[g] = sum of the unsigned biased bytes x' over the group's 128 activations
 * (= sum(int8) + 16384). Row-independent, so the host computes it once per
 * token; the kernel's isum = 2*Pu - Au + 16384 - 256*pc is unchanged. */
void pq1_build_au(const uint8_t *xq, int K, int *au) {
  int nb = K >> 7;
  for (int b = 0; b < nb; b++) {
    const int8_t *qs = (const int8_t *)(xq + (size_t)(b >> 1) * 292 + 4 +
                                         (b & 1) * 128);
    int s = 0;
    for (int i = 0; i < 128; i++) s += (int)qs[i] + 128;
    au[b] = s;
  }
}

#ifndef PQ1_REAL_CUDA
int pq1_gemv_sm53_launch(const unsigned char *a, const unsigned short *b,
                         const unsigned char *c, float *d, int e, int f,
                         void *g) {
  (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g;
  return -99;
}
int pq1_gemv_tm_launch(const unsigned char *a, const unsigned short *b,
                       const unsigned int *c, const int *d, float *e, int f,
                       int g, void *h) {
  (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; (void)h;
  return -99;
}
int pq1_tm_prep_launch(const unsigned char *a, unsigned int *b, int *c, int d,
                       void *e) {
  (void)a; (void)b; (void)c; (void)d; (void)e;
  return -99;
}
#endif

/* silu(g)*u in place: LU[i] = (g * (1/(1+expf(-g)))) * LU[i], four lanes at
 * a time. Lives here because this file is built with -ffp-contract=off: in
 * fwd8.c's default-contract build gcc fused one of the header's double ops
 * and 2 of 4.28e9 inputs came out 1 ulp off (exhaustive test). */
void pq1_silu_mul(float *lu, const float *lg, int n) {
    int i = 0;
    /* two independent 4-lane chains per iteration (#74): the expf core is a
     * dependent chain of ~20 double ops plus a divide, so one chain at a time
     * left the A57's second FP pipe idle. Per-lane arithmetic unchanged. */
    const int n8 = n & ~7;
    /* Serial on purpose: OpenMP over 4 cores read 9.4 (passive) / 10.5 ms/tok
     * (active) against 7.5 serial; 36 fork/joins per token cost more than the
     * work they split (#74). */
    for (int j = 0; j < n8; j += 8) {
        const float32x4_t g0 = vld1q_f32(lg + j), u0 = vld1q_f32(lu + j);
        const float32x4_t g1 = vld1q_f32(lg + j + 4), u1 = vld1q_f32(lu + j + 4);
        const float32x4_t e0 = pq1_expf4(vnegq_f32(g0));
        const float32x4_t e1 = pq1_expf4(vnegq_f32(g1));
        const float32x4_t sg0 = vdivq_f32(vdupq_n_f32(1.0f), vaddq_f32(vdupq_n_f32(1.0f), e0));
        const float32x4_t sg1 = vdivq_f32(vdupq_n_f32(1.0f), vaddq_f32(vdupq_n_f32(1.0f), e1));
        vst1q_f32(lu + j, vmulq_f32(vmulq_f32(g0, sg0), u0));
        vst1q_f32(lu + j + 4, vmulq_f32(vmulq_f32(g1, sg1), u1));
    }
    i = n8;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t g = vld1q_f32(lg + i), u = vld1q_f32(lu + i);
        const float32x4_t e = pq1_expf4(vnegq_f32(g));
        const float32x4_t sg = vdivq_f32(vdupq_n_f32(1.0f), vaddq_f32(vdupq_n_f32(1.0f), e));
        vst1q_f32(lu + i, vmulq_f32(vmulq_f32(g, sg), u));
    }
    for (; i < n; i++) lu[i] = (lg[i] * (1.0f / (1.0f + expf(-lg[i])))) * lu[i];
}
