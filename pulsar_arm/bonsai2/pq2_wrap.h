/* pq2_wrap.h — exact PQ2_0 path (Nano Gen1 port), host + device API. */
#ifndef PQ2_WRAP_H
#define PQ2_WRAP_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Split PQ2_0 blocks (34B: fp16 scale + 32B 2-bit codes per 128 weights)
 * into two contiguous arrays. w_codes = [rows*K/4] bytes of codes verbatim
 * (byte b holds weights 4b..4b+3, weight j = (q>>2j)&3, value = code-1),
 * w_scales = [rows*K/128] fp16 verbatim (one per 128). */
void pq2_split_repack(const uint8_t *pq2, int rows, int K,
                      uint8_t *w_codes, uint16_t *w_scales_h);

/* Fork-faithful host reference: per 128-group
 *   acc += (ws * db) * (float)isum,  isum = sum (code-1)*qs[i]
 * with db/qs from a Q8_K block (292B per 256 acts). Bit-exact target for
 * the device kernel (int partials exact, fp order = group order). */
void pq2_gemv_cpu(int rows, int K, const uint8_t *w_codes,
                  const uint16_t *w_scales_h, const uint8_t *xq, float *y);

/* CUDA launcher (pq2_swar_sm53.cu). <0 on error, -99 if not built. */
int pq2_gemv_sm53_launch(const unsigned char *w_codes,
                         const unsigned short *w_scales_h,
                         const unsigned char *xq, float *y, int K, int N,
                         void *stream);

float pq2_h2f(uint16_t h);

#ifdef PQ2_HIST
void pq2_hist_report(unsigned long *code);
#endif

#ifdef __cplusplus
}
#endif
#endif
