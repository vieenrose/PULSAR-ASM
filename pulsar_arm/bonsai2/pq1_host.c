/* pq1_host.c — Q1_0 split repack + fork-shaped host reference (no CUDA). */
#include "pq1_wrap.h"
#include <string.h>
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

#ifndef PQ1_REAL_CUDA
int pq1_gemv_sm53_launch(const unsigned char *a, const unsigned short *b,
                         const unsigned char *c, float *d, int e, int f,
                         void *g) {
  (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g;
  return -99;
}
#endif
