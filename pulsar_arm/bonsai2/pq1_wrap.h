/* pq1_wrap.h — real Q1_0 (1-bit) path for the Nano (Bonsai-8B-Q1_0.gguf). */
#ifndef PQ1_WRAP_H
#define PQ1_WRAP_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Q1_0 block (18B): fp16 scale + 16B of 128 packed bits, LSB-first.
 * Split into w_bits [rows*K/32] u32 (bit=1 -> +1) and w_scales [rows*K/128]
 * fp16 verbatim. */
void pq1_split_repack(const uint8_t *q1, int rows, int K, uint32_t *w_bits,
                      uint16_t *w_scales_h);

/* Fork-shaped host reference: per 128-group,
 *   isum = sum_i (bit_i ? +1 : -1) * qs[i]   (qs from the Q8_K half)
 *   acc += (ws * db) * isum                  (fp32, group order)
 * Bit-exact target for the device kernel. */
void pq1_gemv_cpu(int rows, int K, const uint32_t *w_bits,
                  const uint16_t *w_scales_h, const uint8_t *xq, float *y);

int pq1_gemv_sm53_launch(const unsigned char *w_bits,
                         const unsigned short *w_scales_h,
                         const unsigned char *xq, float *y, int K, int N,
                         void *stream);

float pq1_h2f(uint16_t h);
#ifdef __cplusplus
}
#endif
#endif
