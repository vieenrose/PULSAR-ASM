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

/* In-process GPU keep-alive (run #43): starts a side-stream spin thread when
 * PQ1_KEEPWARM=1, so the devfreq governor sees a busy GPU during the engine's
 * short GEMV bursts. Returns 1 if started, 0 if not requested, <0 on error. */
int pq1_keepwarm_start(void);

/* T-MAC bit-sliced variant (run #41). pq1_build_planes mirrors the Q8_K block
 * layout (73 words per 256-activation block; word 0 = the fp32 scale d, words
 * 1..64 = the 64 bit-plane words) so the kernel's addresses are unchanged.
 * planes needs (K/256)*73 words, au needs K/128 ints. */
void pq1_build_planes(const uint8_t *xq, int K, uint32_t *planes);
void pq1_build_au(const uint8_t *xq, int K, int *au);

int pq1_gemv_tm_launch(const unsigned char *w_bits,
                       const unsigned short *w_scales_h, const unsigned int *xpl,
                       const int *xau, float *y, int K, int N, void *stream);

/* 128-rows-per-block variant (run #45): same math, 2 lanes/row. */
int pq1_gemv_r128_launch(const unsigned char *w_bits,
                         const unsigned short *w_scales_h, const unsigned char *xq,
                         float *y, int K, int N, void *stream);

float pq1_h2f(uint16_t h);
#ifdef __cplusplus
}
#endif
#endif
