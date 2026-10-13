/* pq2_fwd8_glue.c — pq2_gemv-compatible entry for the EXACT PQ2_0 path.
 * Linked with fwd8.c (built -include pq2_fwd8_shim.h) instead of pq2_gemv.c.
 *
 * Uses fwd8's own Q8_K activations verbatim (no re-quantization), so the
 * numerics are the fork's: the device kernel is bit-exact vs pq2_gemv_cpu
 * for both K=4096 and K=12288, and pq2_gemv_cpu mirrors the fork's
 * ggml_vec_dot_pq2_0_q8_K accumulation order.
 *
 * Repack = pointer-keyed split (codes/scales), uploaded once per tensor into
 * a single device arena (Tegra's fixed GPU-VA mapping table dies at ~190
 * separate cudaMallocs). Host splits are dropped after a successful upload.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include "pq2_wrap.h"

#ifdef PQ2_USE_CUDA
#include <cuda_runtime.h>
#endif

/* Bonsai-8B needs ~2.2GB of device weights (2 bits/weight + scales). The
 * GGUF stays mmap'd but its pages are clean/reclaimable, so they evict as
 * the arena fills. Override with PQ2_ARENA_MB. */
#define PQ2_MAXT 300

typedef struct {
  const uint8_t *w;
  uint8_t *codes;   /* host splits (may be NULL after upload) */
  uint16_t *scales;
#ifdef PQ2_USE_CUDA
  uint8_t *d_codes;
  unsigned short *d_scales;
#endif
  int rows, cols;
} pq2_ent;

static pq2_ent P2C[PQ2_MAXT];
static int P2N = 0;

const char *pq2_compat_name(void) { return "exact-pq2"; }

#ifdef PQ2_USE_CUDA
static uint8_t *P2ARENA = NULL;
static size_t P2ACUR = 0, P2ALIM = 0;
static uint8_t *d_xq = NULL;
static float *d_y = NULL;
static size_t P2XQMAX = 0, P2YMAX = 0;
static int P2ARENA_DEAD = 0;

static int pq2_arena_init(size_t need) {
  if (P2ARENA_DEAD) return -1;
  if (P2ARENA) {
    if (P2ACUR + need <= P2ALIM) return 0;
    fprintf(stderr, "pq2: arena full (%.0fMB used) -> CPU fallback\n",
            P2ACUR / 1048576.0);
    P2ARENA_DEAD = 1;
    return -1;
  }
  const char *e = getenv("PQ2_ARENA_MB");
  size_t mb = e ? (size_t)atoi(e) : 2300;
  void *ar = NULL;
  cudaError_t rc = cudaMalloc(&ar, mb << 20);
  if (rc != cudaSuccess) {
    fprintf(stderr, "pq2: arena %zuMB cudaMalloc: %s -> CPU fallback\n", mb,
            cudaGetErrorString(rc));
    P2ARENA_DEAD = 1;
    return -1;
  }
  P2ARENA = (uint8_t *)ar;
  P2ALIM = mb << 20;
  P2ACUR = 0;
  fprintf(stderr, "pq2: arena %zuMB ok\n", mb);
  return 0;
}
#endif

void pq2_pq2_compat_gemv(int rows, int cols, const unsigned char *w,
                         const float *x, float *y, const unsigned char *xq) {
  (void)x; /* activations come from Q8_K xq (fork numerics, no requant) */
  int ng = cols / 128;
  int ci = -1;
  for (int i = 0; i < P2N; i++)
    if (P2C[i].w == w && P2C[i].rows == rows && P2C[i].cols == cols) {
      ci = i;
      break;
    }

  uint8_t *codes = NULL;
  uint16_t *scales = NULL;
  if (ci < 0) {
    codes = (uint8_t *)malloc((size_t)rows * ng * 32);
    scales = (uint16_t *)malloc((size_t)rows * ng * 2);
    if (!codes || !scales) { fprintf(stderr, "pq2: OOM repack\n"); exit(1); }
    pq2_split_repack(w, rows, cols, codes, scales);
    if (P2N < PQ2_MAXT) {
      ci = P2N++;
      P2C[ci].w = w;
      P2C[ci].codes = codes;
      P2C[ci].scales = scales;
      P2C[ci].rows = rows;
      P2C[ci].cols = cols;
#ifdef PQ2_USE_CUDA
      P2C[ci].d_codes = NULL;
      P2C[ci].d_scales = NULL;
#endif
    } else {
      fprintf(stderr, "pq2: cache full, %dx%d kept local\n", rows, cols);
    }
  } else {
    codes = P2C[ci].codes;
    scales = P2C[ci].scales;
  }

#ifdef PQ2_USE_CUDA
  if (ci >= 0) {
    if (!P2C[ci].d_codes && codes) {
      size_t ncb = (size_t)rows * ng * 32, nsb = (size_t)rows * ng * 2;
      size_t a = (ncb + 63) / 64 * 64, b = (nsb + 63) / 64 * 64;
      if (pq2_arena_init(a + b) == 0) {
        uint8_t *pb = P2ARENA + P2ACUR;
        uint8_t *ps = pb + a;
        if (cudaMemcpy(pb, codes, ncb, cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(ps, scales, nsb, cudaMemcpyHostToDevice) == cudaSuccess) {
          P2ACUR += a + b;
          P2C[ci].d_codes = pb;
          P2C[ci].d_scales = (unsigned short *)ps;
          free(codes);
          free(scales);
          P2C[ci].codes = NULL;
          P2C[ci].scales = NULL;
          codes = NULL;
          scales = NULL;
        } else {
          fprintf(stderr, "pq2: arena memcpy failed -> CPU\n");
          cudaGetLastError();
        }
      }
    }
    uint8_t *dc = P2C[ci].d_codes;
    unsigned short *dsc = P2C[ci].d_scales;
    if (dc && dsc) {
      size_t xqb = (size_t)(cols / 256) * 292, yb = (size_t)rows * 4;
      if (!d_xq || xqb > P2XQMAX) {
        if (d_xq) cudaFree(d_xq);
        d_xq = NULL;
        if (cudaMalloc(&d_xq, xqb) == cudaSuccess) P2XQMAX = xqb;
      }
      if (!d_y || yb > P2YMAX) {
        if (d_y) cudaFree(d_y);
        d_y = NULL;
        if (cudaMalloc(&d_y, yb) == cudaSuccess) P2YMAX = yb;
      }
      if (d_xq && d_y &&
          cudaMemcpy(d_xq, xq, xqb, cudaMemcpyHostToDevice) == cudaSuccess &&
          pq2_gemv_sm53_launch(dc, dsc, d_xq, d_y, cols, rows, NULL) == 0 &&
          cudaDeviceSynchronize() == cudaSuccess &&
          cudaMemcpy(y, d_y, yb, cudaMemcpyDeviceToHost) == cudaSuccess)
        return;
      fprintf(stderr, "pq2: device path failed: %s -> CPU fallback\n",
              cudaGetErrorString(cudaGetLastError()));
    }
  }
#endif

  /* CPU fallback (bit-identical result, ~50x slower) */
  if (ci >= 0) {
    if (!P2C[ci].codes) {
      fprintf(stderr, "pq2: CPU fallback re-repack %dx%d\n", rows, cols);
      P2C[ci].codes = (uint8_t *)malloc((size_t)rows * ng * 32);
      P2C[ci].scales = (uint16_t *)malloc((size_t)rows * ng * 2);
      if (!P2C[ci].codes || !P2C[ci].scales) { fprintf(stderr, "pq2: OOM\n"); exit(1); }
      pq2_split_repack(w, rows, cols, P2C[ci].codes, P2C[ci].scales);
    }
    pq2_gemv_cpu(rows, cols, P2C[ci].codes, P2C[ci].scales, xq, y);
  } else {
    /* cache full and no entry: use the local splits */
    pq2_gemv_cpu(rows, cols, codes, scales, xq, y);
  }
}
