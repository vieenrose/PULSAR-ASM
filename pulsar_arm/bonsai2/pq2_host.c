/* pq2_host.c — exact PQ2_0 split repack + fork-faithful host reference.
 * No CUDA, no libggml, STANDALONE-safe. Mirrors pq2_gemv.c's arithmetic
 * (which the repo verified bit-exact vs ggml_vec_dot_pq2_0_q8_K) so the
 * device kernel can be gated against it.
 */
#include "pq2_wrap.h"
#include <string.h>
#include <math.h>

float pq2_h2f(uint16_t h) {
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

#ifdef PQ2_HIST
/* Code histogram over the 2-bit codes, to decide whether a ternary
 * (1.58-bit) repack of this checkpoint would be lossless: counts code
 * value c = 0..3 (weight value c-1, i.e. -1,0,+1,+2). */
unsigned long pq2_hist[256];
void pq2_hist_report(unsigned long *code) {
  for (int c = 0; c < 4; c++) code[c] = 0;
  for (int v = 0; v < 256; v++) {
    unsigned long h = pq2_hist[v];
    if (!h) continue;
    code[(v) & 3] += h;
    code[(v >> 2) & 3] += h;
    code[(v >> 4) & 3] += h;
    code[(v >> 6) & 3] += h;
  }
}
#endif

void pq2_split_repack(const uint8_t *pq2, int rows, int K,
                      uint8_t *w_codes, uint16_t *w_scales_h) {
  int ng = K / 128;
  for (int r = 0; r < rows; r++) {
    const uint8_t *row = pq2 + (size_t)r * ng * 34;
    uint8_t *cd = w_codes + (size_t)r * ng * 32;
    uint16_t *sc = w_scales_h + (size_t)r * ng;
    for (int g = 0; g < ng; g++) {
#ifdef PQ2_HIST
      for (int b = 0; b < 32; b++) pq2_hist[row[g * 34 + 2 + b]]++;
#else
      memcpy(cd + g * 32, row + g * 34 + 2, 32);
#endif
      memcpy(sc + g, row + g * 34, 2);
    }
  }
}

/* Per-(row,group) bias correction for the unsigned-activation kernel:
 * with x' = x XOR 0x80 (unsigned), Pu = sum_{b0} x', Mu = sum_{b1} x',
 * Au = sum_all x' (row-independent):
 *   isum = Pu + 2*Mu - Au + corr,
 *   corr = 16384 - 128*popcount(b0) - 256*popcount(b1),
 * which fits in int16 (range [16384-16384-32768, 16384] = [-32768, 16384]).
 * Counts depend only on the weight codes, so this is repack-time work. */
void pq2_corr_build(const uint8_t *w_codes, int rows, int K, int16_t *w_corr) {
  int ng = K / 128;
  for (int r = 0; r < rows; r++) {
    const uint8_t *cd = w_codes + (size_t)r * ng * 32;
    int16_t *co = w_corr + (size_t)r * ng;
    for (int g = 0; g < ng; g++) {
      const uint8_t *c = cd + g * 32;
      int pc0 = 0, pc1 = 0;
      for (int b = 0; b < 32; b++) {
        pc0 += __builtin_popcount((unsigned)(c[b] & 0x55u));
        pc1 += __builtin_popcount((unsigned)((c[b] >> 1) & 0x55u));
      }
      co[g] = (int16_t)(16384 - 128 * pc0 - 256 * pc1);
    }
  }
}

void pq2_gemv_cpu(int rows, int K, const uint8_t *w_codes,
                  const uint16_t *w_scales_h, const uint8_t *xq, float *y) {
  int ng = K / 128;
  for (int r = 0; r < rows; r++) {
    const uint8_t *cd = w_codes + (size_t)r * ng * 32;
    float acc = 0.0f;
    for (int g = 0; g < ng; g++) {
      const uint8_t *qb = xq + (size_t)(g >> 1) * 292;
      float db;
      memcpy(&db, qb, 4);
      const int8_t *qs = (const int8_t *)(qb + 4 + (g & 1) * 128);
      float ws = pq2_h2f(w_scales_h[(size_t)r * ng + g]);
      const uint8_t *c = cd + g * 32;
      int isum = 0;
      for (int i = 0; i < 128; i++) {
        int code = (c[i >> 2] >> ((i & 3) * 2)) & 3;
        isum += (code - 1) * (int)qs[i];
      }
      acc += (ws * db) * (float)isum;
    }
    y[r] = acc;
  }
}

#ifndef PQ2_REAL_CUDA /* define when linking pq2_swar_sm53.cu */
int pq2_gemv_sm53_launch(const unsigned char *a, const unsigned short *b,
                         const unsigned char *c, float *d, int e, int f,
                         void *g) {
  (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g;
  return -99; /* host build: no CUDA */
}
#endif
