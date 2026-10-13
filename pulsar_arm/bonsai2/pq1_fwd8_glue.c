/* pq1_fwd8_glue.c — pq1_gemv-compatible entry for the EXACT PQ1_0 path.
 * Linked with fwd8.c (built -include pq1_fwd8_shim.h) instead of pq1_gemv.c.
 *
 * Uses fwd8's own Q8_K activations verbatim (no re-quantization), so the
 * numerics are the fork's: the device kernel is bit-exact vs pq1_gemv_cpu
 * for both K=4096 and K=12288, and pq1_gemv_cpu mirrors the fork's
 * ggml_vec_dot_pq1_0_q8_K accumulation order.
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
#include <sys/mman.h>
#include "pq1_wrap.h"

#ifdef PQ1_USE_CUDA
#include <cuda_runtime.h>
#endif

/* Bonsai-8B needs ~2.2GB of device weights (2 bits/weight + scales). The
 * GGUF stays mmap'd but its pages are clean/reclaimable, so they evict as
 * the arena fills. Override with PQ1_ARENA_MB. */
#define PQ1_MAXT 300

typedef struct {
  const uint8_t *w;
  uint8_t *codes;   /* host splits (may be NULL after upload) */
  uint16_t *scales;
#ifdef PQ1_USE_CUDA
  uint8_t *d_codes;
  unsigned short *d_scales;
#endif
  int rows, cols;
} pq1_ent;

static pq1_ent PQ1C[PQ1_MAXT];
static int PQ1N = 0;

const char *pq1_compat_name(void) { return "exact-pq2"; }

#ifdef PQ1_HIST
void pq1_hist_report(unsigned long *code);
#endif


#ifdef PQ1_USE_CUDA
static uint8_t *PQ1ARENA = NULL;
static size_t PQ1ACUR = 0, PQ1ALIM = 0;
static uint8_t *d_xq = NULL;
static float *d_y = NULL;
static size_t P2XQMAX = 0, P2YMAX = 0;
static int PQ1ARENA_DEAD = 0;

/* GPU-RAM accounting for the harness (PQ1_STATS=1 prints one line at exit).
 * gpu_used_mb  = bytes of the device arena actually holding weights
 * gpu_arena_mb = cudaMalloc size of that arena (one mapping; Tegra's fixed
 *                GPU-VA table cannot take a malloc per tensor)
 * gpu_peak_mb  = high-water mark of arena use, i.e. the real requirement
 * gpu_free_mb / gpu_total_mb = cudaMemGetInfo at exit */
#ifdef PQ1_USE_CUDA
static size_t P2PEAK = 0;
const char *pq1_gpu_stats_mb(char *buf, int n) {
  size_t freeb = 0, totb = 0;
  cudaMemGetInfo(&freeb, &totb);
#ifdef PQ1_HIST
  {
    unsigned long code[4];
    pq1_hist_report(code);
    unsigned long tot = code[0] + code[1] + code[2] + code[3];
    fprintf(stderr,
            "PQ2HIST w=-1:%.2f%% 0:%.2f%% +1:%.2f%% +2:%.2f%% (n=%.3fG) "
            "ternary_lossless=%.2f%%\n",
            100.0 * code[0] / tot, 100.0 * code[1] / tot,
            100.0 * code[2] / tot, 100.0 * code[3] / tot, tot / 1e9,
            100.0 * (code[0] + code[1] + code[2]) / tot);
  }
#endif
  snprintf(buf, n,
           "PQ1GPU used_mb=%.1f arena_mb=%.1f peak_mb=%.1f free_mb=%.1f "
           "total_mb=%.1f tensors=%d",
           PQ1ACUR / 1048576.0, PQ1ALIM / 1048576.0, P2PEAK / 1048576.0,
           freeb / 1048576.0, totb / 1048576.0, PQ1N);
  return buf;
}
#endif

static int pq1_arena_init(size_t need) {
  if (PQ1ARENA_DEAD) return -1;
  if (PQ1ARENA) {
    if (PQ1ACUR + need <= PQ1ALIM) return 0;
    fprintf(stderr, "pq2: arena full (%.0fMB used) -> CPU fallback\n",
            PQ1ACUR / 1048576.0);
    PQ1ARENA_DEAD = 1;
    return -1;
  }
  /* Real Q1_0: 1.125 bits/weight -> the 8B needs ~1.19GB (1063MB bits +
   * 132MB scales). 1280MB leaves room for a few larger models; override
   * with PQ1_ARENA_MB. */
  const char *e = getenv("PQ1_ARENA_MB");
  size_t mb = e ? (size_t)atoi(e) : 1280;
  void *ar = NULL;
  cudaError_t rc = cudaMalloc(&ar, mb << 20);
  if (rc != cudaSuccess) {
    fprintf(stderr, "pq2: arena %zuMB cudaMalloc: %s -> CPU fallback\n", mb,
            cudaGetErrorString(rc));
    PQ1ARENA_DEAD = 1;
    return -1;
  }
  PQ1ARENA = (uint8_t *)ar;
  PQ1ALIM = mb << 20;
  PQ1ACUR = 0;
  fprintf(stderr, "pq2: arena %zuMB ok\n", mb);
  return 0;
}
#endif

void pq1_pq1_compat_gemv(int rows, int cols, const unsigned char *w,
                         const float *x, float *y, const unsigned char *xq) {
  (void)x; /* activations come from Q8_K xq (fork numerics, no requant) */
  int ng = cols / 128;
  int ci = -1;
  for (int i = 0; i < PQ1N; i++)
    if (PQ1C[i].w == w && PQ1C[i].rows == rows && PQ1C[i].cols == cols) {
      ci = i;
      break;
    }

  uint8_t *codes = NULL;
  uint16_t *scales = NULL;
  if (ci < 0) {
    codes = (uint8_t *)malloc((size_t)rows * ng * 32);
    scales = (uint16_t *)malloc((size_t)rows * ng * 2);
    if (!codes || !scales) { fprintf(stderr, "pq2: OOM repack\n"); exit(1); }
    pq1_split_repack(w, rows, cols, codes, scales);
    if (PQ1N < PQ1_MAXT) {
      ci = PQ1N++;
      PQ1C[ci].w = w;
      PQ1C[ci].codes = codes;
      PQ1C[ci].scales = scales;
      PQ1C[ci].rows = rows;
      PQ1C[ci].cols = cols;
#ifdef PQ1_USE_CUDA
      PQ1C[ci].d_codes = NULL;
      PQ1C[ci].d_scales = NULL;
#endif
    } else {
      fprintf(stderr, "pq2: cache full, %dx%d kept local\n", rows, cols);
    }
  } else {
    codes = PQ1C[ci].codes;
    scales = PQ1C[ci].scales;
  }

#ifdef PQ1_USE_CUDA
  if (ci >= 0) {
    if (!PQ1C[ci].d_codes && codes) {
      size_t ncb = (size_t)rows * (cols / 32) * 4, nsb = (size_t)rows * ng * 2;
      size_t a = (ncb + 63) / 64 * 64, b = (nsb + 63) / 64 * 64;
      if (pq1_arena_init(a + b) == 0) {
        uint8_t *pb = PQ1ARENA + PQ1ACUR;
        uint8_t *ps = pb + a;
        if (getenv("PQ1_TRACE"))
          fprintf(stderr, "PQ1T bits pb=%p codes=%p n=%zu | scales ps=%p n=%zu\n",
                  (void *)pb, (void *)codes, ncb, (void *)ps, nsb);
        if (cudaMemcpy(pb, codes, ncb, cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(ps, scales, nsb, cudaMemcpyHostToDevice) == cudaSuccess) {
          PQ1ACUR += a + b;
          if (PQ1ACUR > P2PEAK) P2PEAK = PQ1ACUR;
          PQ1C[ci].d_codes = pb;
          PQ1C[ci].d_scales = (unsigned short *)ps;
          free(codes);
          free(scales);
          PQ1C[ci].codes = NULL;
          PQ1C[ci].scales = NULL;
          codes = NULL;
          scales = NULL;
          /* The GGUF pages are single-use for the device path: drop them
           * once the tensor lives in the arena. They are clean file-backed
           * pages (MAP_PRIVATE read-only), so madvise(DONTNEED) only frees
           * memory -- a fallback re-repack would re-read them from the
           * file. This is the resident mass on a 4GB board. */
          {
            size_t tbytes = (size_t)rows * ng * 18;
            uintptr_t p0 = (uintptr_t)w & ~(uintptr_t)4095;
            uintptr_t p1 = ((uintptr_t)w + tbytes + 4095) & ~(uintptr_t)4095;
            if (p1 > p0) madvise((void *)p0, (size_t)(p1 - p0), MADV_DONTNEED);
          }
        } else {
          fprintf(stderr, "pq2: arena memcpy failed -> CPU\n");
          cudaGetLastError();
        }
      }
    }
    uint8_t *dc = PQ1C[ci].d_codes;
    unsigned short *dsc = PQ1C[ci].d_scales;
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
      if (getenv("PQ1_TRACE"))
        fprintf(stderr, "PQ1T act d_xq=%p xq=%p n=%zu d_y=%p yb=%zu K=%d N=%d\n",
                (void *)d_xq, (void *)xq, xqb, (void *)d_y, yb, cols, rows);
      if (d_xq && d_y &&
          cudaMemcpy(d_xq, xq, xqb, cudaMemcpyHostToDevice) == cudaSuccess &&
          pq1_gemv_sm53_launch(dc, dsc, d_xq, d_y, cols, rows, NULL) == 0 &&
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
    if (!PQ1C[ci].codes) {
      fprintf(stderr, "pq2: CPU fallback re-repack %dx%d\n", rows, cols);
      PQ1C[ci].codes = (uint8_t *)malloc((size_t)rows * ng * 32);
      PQ1C[ci].scales = (uint16_t *)malloc((size_t)rows * ng * 2);
      if (!PQ1C[ci].codes || !PQ1C[ci].scales) { fprintf(stderr, "pq2: OOM\n"); exit(1); }
      pq1_split_repack(w, rows, cols, PQ1C[ci].codes, PQ1C[ci].scales);
    }
    pq1_gemv_cpu(rows, cols, PQ1C[ci].codes, PQ1C[ci].scales, xq, y);
  } else {
    /* cache full and no entry: use the local splits */
    pq1_gemv_cpu(rows, cols, codes, scales, xq, y);
  }
}
