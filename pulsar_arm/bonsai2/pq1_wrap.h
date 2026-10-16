/* pq1_wrap.h — real Q1_0 (1-bit) path for the Nano (Bonsai-8B-Q1_0.gguf). */
#ifndef PQ1_WRAP_H
#define PQ1_WRAP_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* PQ1_TERN (compile-time, every TU): the same engine on a PQ2_0 checkpoint
 * (Ternary-Bonsai-4B; 34B blocks = fp16 scale + 128 2-bit codes, value =
 * code - 1). pq1_split_repack then writes each row as two bit planes of
 * K/32 words each, [b0 plane | b1 plane] (group g's 128 bits at words
 * 4g..4g+3 of each plane, bit a of word j = weight 32j + a), so a row is
 * K/16 words; pq1_gemv_cpu is the fork's PQ2_0 x Q8_K dot. PQ1_GGUF_BB is the
 * GGUF block size, PQ1_RW(K) the repacked u32 words per row. */
#ifdef PQ1_TERN
#define PQ1_GGUF_BB 34
#define PQ1_RW(K) ((size_t)(K) / 16)
#else
#define PQ1_GGUF_BB 18
#define PQ1_RW(K) ((size_t)(K) / 32)
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
/* Group-stationary Harley-Seal kernel (same inputs and result as tm; K = 4096
 * or 12288, other K fall back to tm). pq1_tmx_launch is the engine entry:
 * hs is the default; PQ1_HS=0 selects tm. */
int pq1_gemv_hs_launch(const unsigned char *w_bits,
                       const unsigned short *w_scales_h, const unsigned int *xpl,
                       const int *xau, float *y, int K, int N, void *stream);
int pq1_tmx_launch(const unsigned char *w_bits,
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
/* loop iter 4: plain q8_K_quant of a device vector (same quant as above) */
int pq1_q8k_launch(const float *x_d, int n, unsigned char *xq_d, void *stream);
int pq1_tmx_uses_au(int K, int N);
int pq1_silu_q8k_pl_launch(int silu, const float *x_d, int n, unsigned char *xq_d,
                           unsigned int *pl_d, int *au_d, void *stream);
/* loop iter 4: attention block (head rms + rope + K/V row store, then q.k, softmax,
 * p.v) for one layer. par_d holds, per token: cos[64] | sin[64] at PQ1_APAR_ROPE,
 * int npos at PQ1_APAR_NPOS, the Q8_K activation of attn_norm(x) at PQ1_APAR_XQ. */
#define PQ1_APAR_ROPE 0
#define PQ1_APAR_NPOS 512
#define PQ1_APAR_XQ 576
#define PQ1_APAR_BYTES (576 + 8192)
/* 4k context: K/V caches are f16 rows ([kv head][ctx][128] halves per layer);
 * sc_d ([nh][ctx] f32 scores) and mv_d ([nh][ctx] (ms, vs) float pairs) are
 * scratch. The grids depend on ctx only, never on npos (graph-capturable). */
#define PQ1_FA_MAXH 64
int pq1_attn_launch(const float *qkv_d, const float *nw_d, const unsigned char *par_d,
                    float *qo_d, unsigned short *kc_d, unsigned short *vc_d, float *sc_d,
                    float *mv_d, float *atto_d, int nh, int nkv, int ctx, void *stream);
/* the attention part alone (q.k, softmax parameters, V accumulation) on post-rope
 * q (qo_d) and the f16 caches; npos is read from par_d + PQ1_APAR_NPOS */
int pq1_fa_launch(const float *qo_d, const unsigned short *kc_d, const unsigned short *vc_d,
                  float *sc_d, float *mv_d, const unsigned char *par_d, float *atto_d, int nh,
                  int nkv, int ctx, void *stream);
extern int pq1_fa_mask;   /* bit 0 qk, 1 softmax parameters, 2 V accumulation */
/* TurboQuant K/V cache (PQ1_KVTQ=3|4|43, lossy, opt-in; pq1_swar_sm53.cu): bits
 * 3 or 4 for K and V, 43 = 4-bit K + 3-bit V. Per layer K codes [nkv][ctx][CW]
 * u32 (CW = 12 at 3 bits, 16 at 4), V codes, K scales [nkv][ctx] f32, V scales. stage_d = 2 nkv f16 rows (K rows, then V rows). */
int pq1_tq_store_launch(int bits, const unsigned short *stage_d, int npos, unsigned int *kcode,
                        float *ksc, unsigned int *vcode, float *vsc, int nkv, int ctx, void *stream);
int pq1_tq_fa_launch(int bits, const float *qo_d, const unsigned char *par_d, unsigned int *kcode,
                     float *ksc, unsigned int *vcode, float *vsc, float *sc_d, float *part_d,
                     float *inv_d, float *atto_d, int nh, int nkv, int ctx, void *stream);
void pq1_tq_variant(int v);   /* long-chain kernel variants (bit mask, PQ1_TQV) */
void pq1_tq_short_next(int on);   /* next captures: short TQ pair at ctx > 256 */
int pq1_tq_short_ok(void);         /* PQ1_TQSHORT != 0 */
int pq1_tq_short_launch(int bits, const float *qo_d, const unsigned short *stage_d, const unsigned char *par_d,
                        unsigned int *kcode, float *ksc, unsigned int *vcode, float *vsc, float *sc_d,
                        float *atto_d, int nh, int nkv, int ctx, void *stream);
int pq1_attn_tq_launch(int bits, const float *qkv_d, const float *nw_d, const unsigned char *par_d,
                       float *qo_d, unsigned short *stage_d, unsigned int *kcode, float *ksc,
                       unsigned int *vcode, float *vsc, float *sc_d, float *part_d, float *inv_d,
                       float *atto_d, int nh, int nkv, int ctx, void *stream);
/* host reference (pq1_host.c) */
void pq1_f2h_row(const float *x, uint16_t *h, int n);
void pq1_fa16_head(const float *q, const uint16_t *K, const uint16_t *V, int T, float scale,
                   float *out);
/* whole-token graph: X (+= R when R_d is not NULL, stored back), then fwd8's
 * rms(X, w) and q8_K_quant into xq_d, bit-exact; one block, n % 256 == 0,
 * n <= 4096. serial=1 forces the serial double sum (parity), nfb_d (may be
 * NULL) counts the calls that needed it. */
int pq1_rms_q8k_launch(float *X_d, const float *R_d, const float *w_d, unsigned char *xq_d,
                       int n, int serial, int *nfb_d, void *stream);
int pq1_rms_q8k_pl_launch(float *X_d, const float *R_d, const float *w_d, unsigned char *xq_d,
                          unsigned int *pl_d, int *au_d, int n, void *stream);
/* fwd8's argmax() on the device: out_d[0] = index, out_d[1] = bits of v[index] */
int pq1_argmax_launch(const float *v_d, int n, int *out_d, void *stream);
int pq1_attn_pre_launch(int ver, int serial, int *nfb_d, const float *qkv_d, const float *nw_d,
                        const unsigned char *par_d, float *qo_d, unsigned short *kc_d,
                        unsigned short *vc_d, int nh, int nkv, int ctx, void *stream);
/* exhaustive pq1_r16 check; bad_d = 2 zeroed u64 on the device */
int pq1_r16_test_launch(unsigned long long *bad_d);

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
