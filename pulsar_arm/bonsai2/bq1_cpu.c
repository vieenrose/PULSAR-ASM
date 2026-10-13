/* bq1_cpu.c — host reference + quant + repack for the BQ1 binary path.
 * No CUDA, compiles with gcc (also under -DSTANDALONE, no libggml).
 * Bit-exact vs bq1_swar_sm53.cu by construction: same S_pos/S_all split,
 * same int32 partials, same tile->superblock fp order.
 */
#include "bq1_wrap.h"
#include <math.h>
#include <string.h>
#include <stdint.h>

float bq1_h2f(uint16_t h) {
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

uint16_t bq1_f2h(float f) {
  uint32_t u; memcpy(&u, &f, 4);
  uint32_t s = (u >> 16) & 0x8000u;
  int e = (int)((u >> 23) & 0xffu) - 112;
  uint32_t m = u & 0x7fffffu;
  if (e <= 0) return (uint16_t)s; /* flush subnormals, fine for scales */
  if (e >= 31) return (uint16_t)(s | 0x7bffu);
  return (uint16_t)(s | ((uint32_t)e << 10) | (m >> 13));
}

static inline int bq1_nearest_int(float fval) {
  float v = fval + 12582912.f;
  int i; memcpy(&i, &v, sizeof(int));
  return (i & 0x007fffff) - 0x00400000;
}

float bq1_quant_row_i8(const float *x, int8_t *xq, int K) {
  float amax = 0.0f, mx = 0.0f;
  for (int i = 0; i < K; i++) {
    float a = fabsf(x[i]);
    if (a > amax) { amax = a; mx = x[i]; }
  }
  if (amax == 0.0f) { memset(xq, 0, (size_t)K); return 0.0f; }
  float iscale = -127.f / mx;
  for (int i = 0; i < K; i++) {
    int v = bq1_nearest_int(iscale * x[i]);
    xq[i] = (int8_t)(v < 127 ? v : 127);
  }
  return 1.0f / iscale;
}

/* 4 weights + 4 i8 -> S_pos accumulate (scalar mirror of swar_dot4). */
static inline int bq1_dot4_cpu(uint32_t apack, uint32_t w4, int acc) {
  uint32_t m0 = (0u - (w4 & 1u)) & 0xFFu;
  uint32_t m1 = (0u - ((w4 >> 1) & 1u)) & 0xFFu;
  uint32_t m2 = (0u - ((w4 >> 2) & 1u)) & 0xFFu;
  uint32_t m3 = (0u - ((w4 >> 3) & 1u)) & 0xFFu;
  uint32_t keep = m0 | (m1 << 8) | (m2 << 16) | (m3 << 24);
  uint32_t mk = apack & keep;
  int s0 = (int)(mk << 24) >> 24;
  int s1 = (int)(mk << 16) >> 24;
  int s2 = (int)(mk << 8) >> 24;
  int s3 = (int)((int32_t)mk) >> 24;
  return acc + s0 + s1 + s2 + s3;
}

static inline int bq1_hsum4_cpu(uint32_t apack) {
  int s0 = (int)(apack << 24) >> 24;
  int s1 = (int)(apack << 16) >> 24;
  int s2 = (int)(apack << 8) >> 24;
  int s3 = (int)((int32_t)apack) >> 24;
  return s0 + s1 + s2 + s3;
}

void bq1_gemv_cpu(int rows, int cols,
                  const uint32_t *w_bits, const uint16_t *w_scales_h,
                  const int8_t *x_q, float x_scale, float *y) {
#define TILE_K 1024
  int K32 = cols >> 5, K256 = cols >> 8;
  const uint32_t *xq32 = (const uint32_t *)x_q;
  for (int r = 0; r < rows; r++) {
    float acc_f = 0.0f;
    for (int tile = 0; tile < cols; tile += TILE_K) {
      int pos_sb[4] = {0, 0, 0, 0};
      int all_sb[4] = {0, 0, 0, 0};
      int packs = TILE_K >> 2;
      const uint32_t *tp = xq32 + (tile >> 2);
      for (int p = 0; p < packs; p++) {
        uint32_t ap = tp[p];
        int w_word = (tile >> 5) + (p >> 3);
        uint32_t w32 = w_bits[(size_t)r * K32 + w_word];
        uint32_t w4 = (w32 >> ((p & 7) * 4)) & 0xFu;
        int sb = p >> 6;
        pos_sb[sb] = bq1_dot4_cpu(ap, w4, pos_sb[sb]);
        all_sb[sb] += bq1_hsum4_cpu(ap);
      }
      int sb_base = tile >> 8;
      for (int sb = 0; sb < 4; sb++) {
        float ws = bq1_h2f(w_scales_h[(size_t)r * K256 + sb_base + sb]);
        acc_f += (float)((pos_sb[sb] << 1) - all_sb[sb]) * ws;
      }
    }
    y[r] = acc_f * x_scale;
  }
#undef TILE_K
}

/* Packed Q1_0 block: half d + u32 qs, 6B, 32 weights, LSB-first.
 * Superblock: 8 consecutive blocks share... here each block keeps its
 * own scale replicated per-256 as the average (keeps the kernel's single
 * scale/block exact when the source is already per-256). */
void bq1_repack_from_q1_0(const void *q1_blocks, int rows, int K,
                          uint32_t *w_bits_out, uint16_t *w_scales_h_out) {
  const uint8_t *src = (const uint8_t *)q1_blocks;
  int K32 = K >> 5, K256 = K >> 8;
  for (int r = 0; r < rows; r++) {
    for (int b = 0; b < K32; b++) {
      const uint8_t *blk = src + ((size_t)r * K32 + b) * 6;
      uint16_t d; memcpy(&d, blk, 2);
      uint32_t q; memcpy(&q, blk + 2, 4);
      w_bits_out[(size_t)r * K32 + b] = q;
      if ((b & 7) == 0) {
        /* average 8 halves -> 1 superblock scale */
        double s = 0;
        for (int k = 0; k < 8; k++) {
          uint16_t dk; memcpy(&dk, src + ((size_t)r * K32 + b + k) * 6, 2);
          s += bq1_h2f(dk);
        }
        w_scales_h_out[(size_t)r * K256 + (b >> 3)] = bq1_f2h((float)(s / 8.0));
      }
      (void)d;
    }
  }
}

void bq1_repack_from_pq2_0(const uint8_t *pq2, int rows, int K,
                           uint32_t *w_bits_out, uint16_t *w_scales_h_out) {
  /* PQ2_0 block: half scale + 32B qs, 128 weights, val=((q>>2j)&3)-1. */
  int ng = K / 128, K32 = K >> 5, K256 = K >> 8;
  for (int r = 0; r < rows; r++) {
    const uint8_t *row = pq2 + (size_t)r * ng * 34;
    for (int g = 0; g < ng; g++) {
      const uint8_t *blk = row + g * 34;
      uint16_t sc; memcpy(&sc, blk, 2);
      const uint8_t *q = blk + 2;
      /* 128 weights -> 4 words */
      for (int w = 0; w < 4; w++) {
        uint32_t bits = 0;
        for (int j = 0; j < 32; j++) {
          int idx = w * 32 + j;
          int code = (q[idx / 4] >> ((idx % 4) * 2)) & 3;
          int bit = (code - 1) >= 0 ? 1 : 0; /* sign projection for 1-bit path */
          bits |= (uint32_t)bit << j;
        }
        w_bits_out[(size_t)r * K32 + g * 4 + w] = bits;
      }
      if ((g & 1) == 0) {
        uint16_t sc2; memcpy(&sc2, blk + 34, 2);
        float a = bq1_h2f(sc), b = bq1_h2f(sc2);
        w_scales_h_out[(size_t)r * K256 + (g >> 1)] = bq1_f2h((a + b) * 0.5f);
      }
    }
  }
}

int bq1_cuda_available(void) { return 0; /* host build: no CUDA */ }
