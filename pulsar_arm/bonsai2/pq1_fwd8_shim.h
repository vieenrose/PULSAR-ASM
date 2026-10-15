/* pq1_fwd8_shim.h — run fwd8.c on the EXACT PQ1_0 device path, zero edits.
 * Compile fwd8.c with: -include pq1_fwd8_shim.h and link pq1_fwd8_glue.c
 * (instead of pq1_gemv.c, which needs ARMv8.2+dotprod / vdotq_s32 and does
 * not exist on A57). The shim renames fwd8's pq1_gemv() call to the exact
 * compat entry, which uses fwd8's own Q8_K activations (no requant) and the
 * sm_53 SWAR kernel (bit-exact vs the fork-faithful reference).
 */
#ifndef PQ1_FWD8_SHIM_H
#define PQ1_FWD8_SHIM_H
#include <stdint.h>
#define PQ1_BUILD 1 /* fwd8.c: decode the Q1_0 embedding layout */
void pq1_pq1_compat_gemv(int rows, int cols, const unsigned char *w,
                         const float *x, float *y, const unsigned char *xq);
/* Batched variant (run #54): n tensors that share one Q8_K activation, launched
 * once. Returns 0 if the batch ran, non-zero if the caller must run the scalar
 * path per tensor (unrepacked tensors, no CUDA, launch failure). */
__attribute__((weak)) int pq1_pq1_compat_gemv_batch(const unsigned char *const *w,
                                const int *rows, int n, int cols,
                                float *const *ys, const unsigned char *xq);
#define pq2_gemv pq1_pq1_compat_gemv
/* silu(g)*u, NEON, bit-exact with glibc 2.27 expf (pq1_host.c) */
void pq1_silu_mul(float *lu, const float *lg, int n);
#define pq2_gemv_batch pq1_pq1_compat_gemv_batch
/* loop iter 2: the FFN tail of one layer as one device graph (glue); 0 = ran,
 * FO written. Non-zero = caller runs the host-driven path. */
__attribute__((weak)) int pq1_ffn_fused(const unsigned char *wg, const unsigned char *wu,
                                        const unsigned char *wd, int inter, int hid,
                                        const unsigned char *xq, float *fo);
__attribute__((weak)) void pq1_ffn_dbg_xq(unsigned char *dst, unsigned long n);
/* loop iter 4: the attention block of one layer as one device graph (glue);
 * 0 = ran, AO written. kv_put mirrors host-path K/V rows into the device cache. */
__attribute__((weak)) int pq1_attn_fused(int il, const unsigned char *wq, const unsigned char *wk,
                                         const unsigned char *wv, const unsigned char *wo,
                                         const float *qnw, const float *knw, int nh, int nkv,
                                         int hid, int ctx, int npos, const float *rcc,
                                         const float *rcs, const unsigned char *xq, float *ao);
__attribute__((weak)) int pq1_attn_kv_put(int il, int nl, int nkv, int ctx, int npos,
                                          const uint16_t *k16, const uint16_t *v16);
__attribute__((weak)) void pq1_attn_dbg(int il, int npos, uint16_t *k16, uint16_t *v16, float *q,
                                        float *atto, int nh);
/* 4k context: the run's K/V cache shape, before the first device call (the
 * arena is sized from it) */
__attribute__((weak)) void pq1_kv_shape(int nl, int nkv, int ctx);

#ifdef PQ1_USE_CUDA
/* GPU-RAM report printed once at process exit (PQ1_STATS=1). Declared here
 * with no CUDA headers: this shim is -include'd into fwd8.c, which is built
 * by plain gcc, so pulling cuda_runtime.h in would break that TU. The
 * definition (and cudaMemGetInfo) lives in pq1_fwd8_glue.c. */
#include <stdio.h>
#include <stdlib.h>
const char *pq1_gpu_stats_mb(char *buf, int n);
static void pq1_gpu_stats_dump(void) {
  if (!getenv("PQ1_STATS")) return;
  char b[256];
  fprintf(stderr, "%s\n", pq1_gpu_stats_mb(b, sizeof b));
}
__attribute__((constructor)) static void pq1_gpu_stats_init(void) {
  atexit(pq1_gpu_stats_dump);
}
#endif
#endif
