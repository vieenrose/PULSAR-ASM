/* bq1_gemv_neon.c — A57-portable BQ1 binary GEMV (no SDOT, ARMv8.0).
 *
 * Playbook: k2 f6b8cfe (k2_core_a57.S scalar int8-dot via SMULL/SADALP+UZP,
 * bit-identical on Spark + Nano 81/81). A57 has no SDOT (ARMv8.2+dotprod),
 * so the bonsai-aarch64 tq/pq2 NEON kernels (vdotq_s32) cannot run here.
 *
 * BQ1 masked-dot per 8 weights, v2 (all ARMv8.0 NEON, no stack tmp):
 *   ld1 actsstituto (16xi8) + scalar-expanded keep mask (16B, 0x00/0xFF)
 *   and   v2.16b, v0.16b, v1.16b   // masked: 0 where w==-1
 *   saddlv s_all, v0.16b            // S_all_16, s8x16 -> s32
 *   saddlv s_pos, v2.16b            // S_pos_16
 *   contrib_16 = 2*S_pos - S_all, int-exact, flushed per 256 with its
 *   fp16 scale exactly like bq1_cpu.c / bq1_swar_sm53.cu (same fp order).
 *
 * Scalar mask expansion (16 ops / 16 weights) dual-issues with NEON on
 * A57; the vector path is ~5 instr / 16 weights (~0.3 instr/weight).
 * Threading mirrors tq_gemv_neon.c: OpenMP parallel-for over rows
 * (bit-exact: one thread owns whole rows, identical serial code).
 *
 * Layout: same split as bq1_wrap.h (w_bits[N*K/32], w_scales[N*K/256],
 * xq[K] i8, xs float). K%256==0.
 *
 * Build (Nano): gcc -O2 -fopenmp -mcpu=cortex-a57 -DSTANDALONE ...
 * Without NEON (host smoke): -DBQ1_NEON_SCALAR falls back to bq1_cpu math.
 */
#include <stdint.h>
#include <string.h>
#include "bq1_wrap.h"

#if defined(__aarch64__) && defined(__ARM_NEON) && !defined(BQ1_NEON_SCALAR)
#include <arm_neon.h>
#define BQ1_HAVE_NEON 1
#else
#define BQ1_HAVE_NEON 0
#endif

void bq1_gemv_neon(int rows, int cols,
                   const uint32_t *w_bits, const uint16_t *w_scales_h,
                   const int8_t *x_q, float x_scale, float *y) {
  int K32 = cols >> 5, K256 = cols >> 8;
#if BQ1_HAVE_NEON
  static const int8_t shbits[8] = {0, -1, -2, -3, -4, -5, -6, -7};
  int8x8_t vsh = vld1_s8(shbits);
  uint8x8_t vone = vdup_n_u8(1);
  uint8x8_t vzero = vdup_n_u8(0);
  #pragma omp parallel for schedule(static)
  for (int r = 0; r < rows; r++) {
    float acc_f = 0.0f;
    /* 256-weight superblocks; 16 weights per inner step */
    for (int sb = 0; sb < K256; sb++) {
      int pos = 0, all = 0;
      const uint32_t *wb = w_bits + (size_t)r * K32 + sb * 8;
      const int8_t *xq = x_q + sb * 256;
      for (int s = 0; s < 32; s++) {
        /* 8 weights = 1 byte of the superblock's 8 words (32 bytes);
         * word s>>2, byte s&3. Mask expanded in NEON (vshl {0..-7} +
         * AND 1 + NEG -> 0x00/0xFF), no stack tmp. */
        uint8_t b8 = (uint8_t)(wb[s >> 2] >> ((s & 3) * 8));
        uint8x8_t vb = vdup_n_u8(b8);
        uint8x8_t t = vshl_u8(vb, vsh);
        uint8x8_t k = vsub_u8(vzero, vand_u8(t, vone));
        int8x8_t va = vld1_s8(xq + s * 8);
        int8x8_t vm = vreinterpret_s8_u8(
            vand_u8(vreinterpret_u8_s8(va), k));
        pos += vaddlv_s8(vm);
        all += vaddlv_s8(va);
      }
      float ws = bq1_h2f(w_scales_h[(size_t)r * K256 + sb]);
      acc_f += (float)((pos << 1) - all) * ws;
    }
    y[r] = acc_f * x_scale;
  }
#else
  bq1_gemv_cpu(rows, cols, w_bits, w_scales_h, x_q, x_scale, y);
#endif
}

/* Row-parallel OpenMP needs <omp.h> only under fopenmp; serial fallback
 * without it still compiles (pragma ignored). */
