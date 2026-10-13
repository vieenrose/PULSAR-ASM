/* bq1_fwd8_glue.c — pq2_gemv-compatible entry running the BQ1 binary path.
 * Linked with fwd8.c (built with -include bq1_fwd8_shim.h) instead of
 * pq2_gemv.c. A57-safe (no SDOT), STANDALONE-safe (no libggml).
 *
 * Per GEMV call: lazy-repack W (PQ2_0 34B blocks -> BQ1 split, cached by
 * pointer; sign projection, scales averaged per 256), quantize x from the
 * float activations via bq1_quant_row_i8 into static scratch, dispatch to
 * CUDA (sm_53, synchronous) if built with -DBQ1_USE_CUDA, else A57 NEON.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "bq1_wrap.h"
#include <unistd.h>

void bq1_gemv_neon(int rows, int cols, const uint32_t *w_bits,
                   const uint16_t *w_scales_h, const int8_t *x_q,
                   float x_scale, float *y);
#ifdef BQ1_USE_CUDA
#include <cuda_runtime.h>
int bq1_gemv_sm53_launch(const uint32_t *w_bits,
                         const unsigned short *w_scales_h,
                         const signed char *x_q, float x_scale,
                         float *y, int K, int N, void *stream);
#endif

#define BQ1_MAXT 300
static struct {
  const uint8_t *w;
  uint32_t *bits;
  uint16_t *scales;
#ifdef BQ1_USE_CUDA
  uint32_t *d_bits;
  unsigned short *d_scales;
  signed char *d_xq;
  float *d_y;
#endif
  int rows, cols;
} BQ1C[BQ1_MAXT];
static int BQ1N = 0;
static int8_t *BQ1XQ = NULL; /* 12288 flat */
/* Single device arena (see note): per-tensor cudaMallocs exhaust Tegra's
 * fixed GPU-VA mapping table (~386 mappings) and libcuda segfaults instead
 * of erroring. One arena = one mapping. Total sized for Bonsai-8B:
 * 36 layers x 23MB + head 78MB + scales ~14MB -> 940MB. */
#define BQ1_ARENA_MB 940
static uint8_t *BQ1ARENA = NULL;
static size_t BQ1ACUR = 0;
#ifdef BQ1_USE_CUDA
static signed char *BQ1DXQ = NULL; /* shared device xq scratch (12288) */
static float *BQ1DY = NULL;        /* shared device y scratch (151669*4) */
static size_t BQ1DEV_USED = 0, BQ1DEV_N = 0;
#endif

void bq1_pq2_compat_gemv(int rows, int cols, const unsigned char *w,
                         const float *x, float *y, const unsigned char *xq) {
  /* Lookup is by tensor identity: a hit must NEVER re-repack, even when
   * host splits were released after device upload (bits/scales NULL). */
  (void)xq; /* quantize from float x directly (reference numerics) */
  if (!BQ1XQ) BQ1XQ = (int8_t *)malloc(12288);
  { static int e = 0, tr = -1;
    if (tr < 0) tr = getenv("BQ1_TRACE") ? 1 : 0;
    e++;
    if (tr) fprintf(stderr, "bq1: enter #%d pid=%d %dx%d cache=%d W=%p\n",
                    e, (int)getpid(), rows, cols, BQ1N, (const void *)w); }
  uint32_t *bits = NULL;
  uint16_t *scales = NULL;
  int ci = -1;
  for (int i = 0; i < BQ1N; i++)
    if (BQ1C[i].w == w && BQ1C[i].rows == rows && BQ1C[i].cols == cols) {
      ci = i; break;
    }
  if (ci < 0) {
    int K32 = cols >> 5, K256 = cols >> 8;
    bits = (uint32_t *)malloc((size_t)rows * K32 * 4);
    scales = (uint16_t *)malloc((size_t)rows * K256 * 2);
    if (!bits || !scales) { fprintf(stderr, "bq1: OOM repack\n"); exit(1); }
    bq1_repack_from_pq2_0(w, rows, cols, bits, scales);
    if (BQ1N < BQ1_MAXT) {
      ci = BQ1N;
      BQ1C[BQ1N].w = w; BQ1C[BQ1N].bits = bits;
      BQ1C[BQ1N].scales = scales;
      BQ1C[BQ1N].rows = rows; BQ1C[BQ1N].cols = cols;
#ifdef BQ1_USE_CUDA
      BQ1C[BQ1N].d_bits = NULL; BQ1C[BQ1N].d_scales = NULL;
#endif
      BQ1N++;
    } else {
      fprintf(stderr, "bq1: cache full (%d), %dx%d uncached\n", BQ1N, rows, cols);
    }
    fprintf(stderr, "bq1: repacked %dx%d (%d cached)\n", rows, cols, BQ1N);
  } else {
    bits = BQ1C[ci].bits; scales = BQ1C[ci].scales; /* may be NULL (freed) */
  }
  float xs = bq1_quant_row_i8(x, BQ1XQ, cols);
#ifdef BQ1_BIGCPU
  if (rows == 12288) goto neon;
#endif
#ifdef BQ1_USE_CUDA
#define BQ1_CK(call, what) do { cudaError_t e_ = (call); \
  if (e_ != cudaSuccess) { \
    fprintf(stderr, "bq1: cuda %s fail: %s -> NEON\n", what, \
            cudaGetErrorString(e_)); goto neon; } } while (0)
  if (ci >= 0 && BQ1C[ci].d_bits) {
    /* hot path: already on device */
  } else if (bits) {
    int K32 = cols >> 5, K256 = cols >> 8;
    size_t nbb = (size_t)rows * K32 * 4, nbs = (size_t)rows * K256 * 2;
    if (!BQ1ARENA) {
      void *ar = NULL;
      BQ1_CK(cudaMalloc(&ar, (size_t)BQ1_ARENA_MB << 20), "malloc arena");
      BQ1ARENA = (uint8_t *)ar; BQ1ACUR = 0;
      fprintf(stderr, "bq1: arena %dMB ok\n", BQ1_ARENA_MB);
    }
    size_t need = (nbb + 63) / 64 * 64 + (nbs + 63) / 64 * 64;
    if (BQ1ACUR + need > ((size_t)BQ1_ARENA_MB << 20)) {
      fprintf(stderr, "bq1: arena full at tensor %d -> NEON\n", BQ1N);
      goto neon;
    }
    uint8_t *pb = BQ1ARENA + BQ1ACUR;
    uint8_t *ps = pb + (nbb + 63) / 64 * 64;
    BQ1_CK(cudaMemcpy(pb, bits, nbb, cudaMemcpyHostToDevice), "arena bits");
    BQ1_CK(cudaMemcpy(ps, scales, nbs, cudaMemcpyHostToDevice), "arena scales");
    BQ1ACUR += need;
    if (ci >= 0) {
      BQ1C[ci].d_bits = (uint32_t *)pb;
      BQ1C[ci].d_scales = (unsigned short *)ps;
      /* Release host splits after upload. Entry keeps (NULL) splits;
       * CUDA path uses d_*; NEON fallback re-repacks (logged). */
      free(bits); free(scales);
      BQ1C[ci].bits = NULL; BQ1C[ci].scales = NULL;
      bits = NULL; scales = NULL;
    }
  } else {
    goto neon; /* uncached (cache full) and no splits: NEON re-repacks */
  }
  if (!BQ1DXQ) {
    void *px = NULL, *py = NULL;
    BQ1_CK(cudaMalloc(&px, 12288), "malloc xq scratch");
    BQ1_CK(cudaMalloc(&py, (size_t)151669 * 4), "malloc y scratch");
    BQ1DXQ = (signed char *)px; BQ1DY = (float *)py;
  }
  {
    uint32_t *db = ci >= 0 ? BQ1C[ci].d_bits : NULL;
    unsigned short *ds = ci >= 0 ? BQ1C[ci].d_scales : NULL;
    /* ci<0 uncached tensors uploaded above only if bits existed; if the
     * entry is missing we uploaded to nowhere -- handle: uncached path
     * always has local bits (fresh repack), so upload them transiently. */
    uint8_t *tb = NULL, *ts = NULL;
    if (!db && bits) {
      int K32 = cols >> 5, K256 = cols >> 8;
      size_t nbb = (size_t)rows * K32 * 4, nbs = (size_t)rows * K256 * 2;
      size_t need = (nbb + 63) / 64 * 64 + (nbs + 63) / 64 * 64;
      if (!BQ1ARENA || BQ1ACUR + need > ((size_t)BQ1_ARENA_MB << 20)) goto neon;
      tb = BQ1ARENA + BQ1ACUR; ts = tb + (nbb + 63) / 64 * 64;
      BQ1_CK(cudaMemcpy(tb, bits, nbb, cudaMemcpyHostToDevice), "trans bits");
      BQ1_CK(cudaMemcpy(ts, scales, nbs, cudaMemcpyHostToDevice), "trans scales");
      BQ1ACUR += need;
      db = (uint32_t *)tb; ds = (unsigned short *)ts;
      free(bits); free(scales); bits = NULL; scales = NULL;
    }
    if (!db) goto neon;
    BQ1_CK(cudaMemcpy(BQ1DXQ, BQ1XQ, (size_t)cols,
                      cudaMemcpyHostToDevice), "copy xq");
    if (bq1_gemv_sm53_launch(db, ds, BQ1DXQ, xs, BQ1DY,
                             cols, rows, NULL) != 0) {
      fprintf(stderr, "bq1: cuda launch fail -> NEON\n");
      goto neon;
    }
    BQ1_CK(cudaDeviceSynchronize(), "sync");
    BQ1_CK(cudaMemcpy(y, BQ1DY, (size_t)rows * 4,
                      cudaMemcpyDeviceToHost), "copy y");
    return;
  }
neon:
  if (!bits) {
    fprintf(stderr, "bq1: NEON fallback re-repack %dx%d\n", rows, cols);
    int K32n = cols >> 5, K256n = cols >> 8;
    bits = malloc((size_t)rows * K32n * 4);
    scales = malloc((size_t)rows * K256n * 2);
    bq1_repack_from_pq2_0(w, rows, cols, bits, scales);
  }
  bq1_gemv_neon(rows, cols, bits, scales, BQ1XQ, xs, y);
#endif
#undef BQ1_CK
}
