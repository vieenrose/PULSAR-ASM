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

/* T-MAC bit-sliced variant (run #41). pq1_build_planes mirrors the Q8_K block
 * layout (73 words per 256-activation block; word 0 = the fp32 scale d, words
 * 1..64 = the 64 bit-plane words) so the kernel's addresses are unchanged.
 * planes needs (K/256)*73 words, au needs K/128 ints. */
void pq1_build_planes(const uint8_t *xq, int K, uint32_t *planes);
void pq1_build_au(const uint8_t *xq, int K, int *au);

int pq1_gemv_tm_launch(const unsigned char *w_bits,
                       const unsigned short *w_scales_h, const unsigned int *xpl,
                       const int *xau, float *y, int K, int N, void *stream);
/* Device-side pq1_build_planes + pq1_build_au (run #72): xq_d holds the raw
 * Q8_K bytes (as uploaded for the SWAR kernels), pl_d/au_d receive the planes
 * (same byte size as xq) and the K/128 group sums. Graph-capturable. */
int pq1_tm_prep_launch(const unsigned char *xq_d, unsigned int *pl_d, int *au_d,
                       int K, void *stream);
/* Loop iter 2: v = silu(g)*u (gu_d = [g | u], n each) quantized to Q8_K in
 * xq_d, bit-exact with fwd8's pq1_silu_mul + q8_K_quant (bsums left
 * untouched: the T-MAC prep never reads them). vout_d may be NULL. */
int pq1_silu_q8k_launch(const float *gu_d, int n, unsigned char *xq_d, float *vout_d,
                        void *stream);

/* 128-rows-per-block variant (run #45): same math, 2 lanes/row. */
int pq1_gemv_r128_launch(const unsigned char *w_bits,
                         const unsigned short *w_scales_h, const unsigned char *xq,
                         float *y, int K, int N, void *stream);

/* 256-rows-per-block variant (run #46): 1 lane/row, no shared epilogue. */
int pq1_gemv_r256_launch(const unsigned char *w_bits,
                         const unsigned short *w_scales_h, const unsigned char *xq,
                         float *y, int K, int N, void *stream);

float pq1_h2f(uint16_t h);

/* Hybrid CPU/GPU row split (run #75): the tail rows of a tensor are computed
 * on the A57s while the GPU streams the head rows. pq1_hyb_rows is the NEON
 * port of pq1_gemv_cpu for rows [0, rows) of a (codes, scales) slice: the
 * 128-group isum is an exact integer (2 * sum of the selected int8 minus the
 * group sum) and the fp32 chain acc += (ws * db) * (float)isum runs in the
 * reference's group order in this -ffp-contract=off unit, so y is bit-exact.
 * scales_f are the fp16 scales already converted by pq1_h2f. pq1_hyb_run
 * splits the rows over the calling thread plus PQ1_HYB_THREADS-1 spinning
 * workers (created on first use). */
void pq1_hyb_rows(const uint32_t *codes, const float *scales_f, int rows, int K,
                  const uint8_t *xq, const int *gsum, float *y);
void pq1_hyb_gsum(const uint8_t *xq, int K, int *gsum);
void pq1_hyb_run(const uint32_t *codes, const float *scales_f, int rows, int K,
                 const uint8_t *xq, float *y);
void pq1_hyb_threads(int n);
/* split form: start the workers before the GPU launch, wait after its sync */
void pq1_hyb_start(const uint32_t *codes, const float *scales_f, int rows, int K,
                   const uint8_t *xq, float *y);
void pq1_hyb_wait(void);
#ifdef __cplusplus
}
#endif
#endif
