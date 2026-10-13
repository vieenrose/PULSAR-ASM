/* bq1_wrap.h — shared CPU/CUDA API for the BQ1 binary path (Nano Gen1 port).
 *
 * Layout (see bq1_swar_sm53.cu):
 *   w_bits   [N*K/32] u32, bit=1 -> +1, bit=0 -> -1, LSB-first
 *   w_scales [N*K/256] fp16 bits (uint16_t, IEEE), 1 per 256 weights
 *   x_q      [K] int8, x_scale float (single scale per token)
 *   y        [N] float
 * Companion: bq1_cpu.c (host ref + quant + repack, no CUDA),
 *            bq1_swar_sm53.cu (sm_53 kernel + launcher).
 */
#ifndef BQ1_WRAP_H
#define BQ1_WRAP_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* f32 row -> int8 row. Returns x_scale (amax/127, 0 if all-zero).
 * nearest_int logic is textually identical to fwd8.c:q8_K_quant so the
 * bytes match the fork quantizer on the same inputs. */
float bq1_quant_row_i8(const float *x, int8_t *xq, int K);

/* Scalar SWAR GEMV, bit-exact vs the GPU kernel (int-exact partials,
 * fp accumulation order tile-outer / superblock-inner). */
void bq1_gemv_cpu(int rows, int cols,
                  const uint32_t *w_bits, const uint16_t *w_scales_h,
                  const int8_t *x_q, float x_scale, float *y);

/* Repack helpers. */
void bq1_repack_from_q1_0(const void *q1_blocks, int rows, int K,
                          uint32_t *w_bits_out, uint16_t *w_scales_h_out);
/* Binary projection of PQ2_0 (Bonsai-8B checkpoint as shipped):
 * code ((q>>2j)&3)-1, sign = code>=0 ? 1 : 0, scale preserved per 128
 * replicated to the 256 superblock (average of the 2 halves). Lets the
 * Nano experiment run the real 8B weights through the 1-bit path for
 * roofline/speed validation before a true 1-bit checkpoint exists. */
void bq1_repack_from_pq2_0(const uint8_t *pq2, int rows, int K,
                           uint32_t *w_bits_out, uint16_t *w_scales_h_out);

/* Half<->float (IEEE, no lib dependency, same as fwd8.c:h2f). */
float bq1_h2f(uint16_t h);
uint16_t bq1_f2h(float f);

/* CUDA launcher (defined in bq1_swar_sm53.cu). Returns 0 ok, <0 err.
 * stream may be NULL (default stream). reports -99 if built without CUDA. */
int bq1_gemv_sm53_launch(const uint32_t *w_bits,
                         const unsigned short *w_scales_h,
                         const signed char *x_q, float x_scale,
                         float *y, int K, int N, void *stream);
int bq1_cuda_available(void);

#ifdef __cplusplus
}
#endif
#endif
