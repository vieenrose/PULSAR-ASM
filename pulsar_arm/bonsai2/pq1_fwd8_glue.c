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
#include <time.h>
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
/* T-MAC path (run #41): the bit-plane mirror of xq is the SAME size (73 words
 * per 256-activation block, word 0 = the fp32 scale) so d_xq is reused as the
 * device plane buffer; only the host scratch is new. PQ1_NO_TM=1 falls back to
 * the SWAR kernel, which lets one binary be A/B'd against itself. */
static uint8_t *d_au = NULL;
static size_t P2AUMAX = 0;
static uint32_t *PL_H = NULL;
static int *AU_H = NULL;
static size_t PL_HMAX = 0, AU_HMAX = 0;
static int PQ1_TM = -1;
static int PQ1_R128 = -1;   /* run #45: 128 rows/block, 2 lanes/row (default) */
static int PQ1_R256 = -1;   /* run #46: 256 rows/block; -1 = auto (rows>=8192) */
static long PQ1_TM_CALLS = 0, PQ1_R128_CALLS = 0, PQ1_R256_CALLS = 0, PQ1_SWAR_CALLS = 0; /* diagnostic: which kernel ran */
static long PQ1_BATCH_CALLS = 0, PQ1_BATCH_ROWS = 0; /* diagnostic: batched launches */

/* PQ1_KTIME=1: per-shape device-path timing (upload + launch + sync + readback),
 * printed by pq1_stats at exit. Needed because a kernel-level speedup measured
 * on a standalone launch does not automatically appear in the engine. */
typedef struct { int rows, cols; double ms; long n; double h2d, launch, sync, d2h; } pq1_kt;
static pq1_kt KT[64];
static int NKT = 0, KTON = -1;
static double pq1_now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}
static void pq1_kt_add(int rows, int cols, double ms) {
  for (int i = 0; i < NKT; i++)
    if (KT[i].rows == rows && KT[i].cols == cols) {
      KT[i].ms += ms;
      KT[i].n++;
      return;
    }
  if (NKT < 64) {
    KT[NKT].rows = rows;
    KT[NKT].cols = cols;
    KT[NKT].ms = ms;
    KT[NKT].n = 1;
    NKT++;
  }
}
static int PQ1ARENA_DEAD = 0;

/* Zero-copy output buffers (PQ1_ZC=1 to enable; OFF by default, measured -390%).
 *
 * Run #53's phase timing (PQ1_KTIME=3) measured the engine's per-call API cost
 * at 190us for a 4096x4096 GEMV: h2d 40us + launch 49us + d2h 101us, i.e.
 * 46.6ms per token of the 365ms decode step (12.8%), of which the D2H readback
 * of the output vector is 26ms/token (7%). Those copies are pure overhead on
 * this board: Tegra shares one physical address space between the CPU and the
 * GPU, so the kernel can write the engine's output vectors (QF/KF/VF/AO/LG/LU/
 * FO/HEAD) in place. Each distinct buffer is DMA-mapped once at its first use
 * (cudaHostRegister + cudaHostGetDevicePointer); the kernel then writes the
 * device pointer of that buffer instead of the arena staging buffer, and the
 * cudaMemcpy D2H disappears. The values written are identical, so the greedy
 * stream and the parity gate are unaffected. Any registration failure disables
 * the path for the rest of the run and the copy path is used. */
#define PQ1_ZC_MAX 24
static struct { const void *h; void *d; size_t n; } ZC[PQ1_ZC_MAX];
static int NZC = 0, ZCON = -1, ZCFAIL = 0;

static void *pq1_out_dev(const void *h, size_t n) {
  if (ZCON < 0) {
    const char *z = getenv("PQ1_ZC");
    /* DEFAULT OFF. Run #53 measured the zero-copy output path (kernel writes the
     * engine's output buffers in place, removing the 24ms/token of D2H memcpy):
     * the copies do disappear (d2h = 0.00ms with 8 buffers mapped), but decode
     * goes 365 -> 1862 ms/token, i.e. -390%. cudaHostAlloc memory on Tegra is
     * CPU-uncached, so the engine's own CPU passes over those vectors (attention
     * over QF/KF/VF, silu over LG/LU, residual adds over AO/FO, argmax over
     * HEAD) cost several times what the DMA copies cost. The API overhead it
     * removes is real (46.6ms/token = 12.8% of decode by the same phase timing);
     * the way to reclaim it is batching the calls, not mapping the buffers. */
    ZCON = (z && atoi(z) != 0) ? 1 : 0;
  }
  if (!ZCON || ZCFAIL >= 4 || !h) return NULL;
  for (int i = 0; i < NZC; i++)
    if (ZC[i].h == h && ZC[i].n == n) return ZC[i].d;
  if (NZC >= PQ1_ZC_MAX) return NULL;
  void *d = NULL;
  /* Fast path: the buffer came from pq1_hostmap (cudaHostAlloc Mapped), which
   * cudaHostGetDevicePointer translates directly. Probing it first is required
   * on Tegra, where cudaHostRegister is not supported at all (run #53 measured
   * "operation not supported" for malloc'd memory). */
  if (cudaHostGetDevicePointer(&d, (void *)h, 0) == cudaSuccess && d) {
    ZC[NZC].h = h;
    ZC[NZC].n = n;
    ZC[NZC].d = d;
    NZC++;
    return ZC[NZC - 1].d;
  }
  cudaGetLastError(); /* clear the probe's error so it cannot poison later checks */
  /* Fallback for a plain malloc'd buffer: register the whole set of pages that
   * contains it (cudaHostRegister is page-granular; malloc gives 16-byte
   * alignment), then offset the device pointer by the in-page offset. */
  uintptr_t base = (uintptr_t)h & ~(uintptr_t)4095u;
  size_t off = (uintptr_t)h - base;
  size_t len = ((off + n + 4095u) & ~(size_t)4095u);
  d = NULL;
  cudaError_t e = cudaHostRegister((void *)base, len, cudaHostRegisterDefault);
  if (e != cudaSuccess) {
    cudaGetLastError();
    if (getenv("PQ1_TRACE"))
      fprintf(stderr, "PQ1ZC register %p n=%zu -> %s\n", h, n, cudaGetErrorString(e));
    ZCFAIL++;
    return NULL;
  }
  if (cudaHostGetDevicePointer(&d, (void *)base, 0) != cudaSuccess || !d) {
    cudaHostUnregister((void *)base);
    ZCFAIL++;
    return NULL;
  }
  ZC[NZC].h = h;
  ZC[NZC].n = n;
  ZC[NZC].d = (void *)((uintptr_t)d + off);
  NZC++;
  return ZC[NZC - 1].d;
}

/* Host allocation the GPU can address directly. On Tegra the CPU and GPU share
 * one physical address space, so a cudaHostAlloc(Mapped) buffer is ordinary
 * RAM that both sides can touch: fwd8.c allocates its GEMV output vectors with
 * this, the kernel writes them in place, and the per-call D2H memcpy goes away
 * (run #53 measured that copy at 26ms per token = 7% of the decode step).
 * Returns NULL when mapping is unavailable; the caller then uses malloc and the
 * glue keeps the (bit-identical) copy path. */
void *pq1_hostmap(size_t bytes) {
  if (ZCON < 0) {
    const char *z = getenv("PQ1_ZC");
    ZCON = (z && atoi(z) != 0) ? 1 : 0; /* off by default, see pq1_out_dev (#53) */
  }
  if (!ZCON) return NULL;
  void *p = NULL;
  if (cudaHostAlloc(&p, bytes, cudaHostAllocMapped) != cudaSuccess) {
    if (getenv("PQ1_TRACE")) fprintf(stderr, "PQ1ZC hostalloc %zu -> failed\n", bytes);
    ZCFAIL++;
    return NULL;
  }
  return p;
}

/* PQ1_KTIME=3 additionally splits the per-call cost into its API phases
 * (H2D / launch / device-sync / D2H), because the engine spends ~44ms/token
 * outside the GEMV kernels and the batch-launch idea is only worth building if
 * a meaningful part of that is per-call API overhead rather than CPU math. */
static void pq1_kt_phase(int rows, int cols, double h2d, double launch,
                         double sync, double d2h) {
  for (int i = 0; i < NKT; i++)
    if (KT[i].rows == rows && KT[i].cols == cols) {
      KT[i].h2d += h2d; KT[i].launch += launch;
      KT[i].sync += sync; KT[i].d2h += d2h;
      return;
    }
}

/* GPU-RAM accounting for the harness (PQ1_STATS=1 prints one line at exit).
 * gpu_used_mb  = bytes of the device arena actually holding weights
 * gpu_arena_mb = cudaMalloc size of that arena (one mapping; Tegra's fixed
 *                GPU-VA table cannot take a malloc per tensor)
 * gpu_peak_mb  = high-water mark of arena use, i.e. the real requirement
 * gpu_free_mb / gpu_total_mb = cudaMemGetInfo, sampled at arena allocation.
 *                NOT at exit: CUDA's own atexit handler has torn the context
 *                down by then, the query fails, and both outputs keep their 0
 *                initialisers - which is why the dashboard printed 0.0 on every
 *                run. A real query cannot report total=0, so a 0 in these fields
 *                meant an unmeasured value read as a measurement; unmeasured now
 *                prints NA. */
#ifdef PQ1_USE_CUDA
static size_t P2PEAK = 0;
/* Device memory is sampled once, at arena allocation, while the CUDA context is
 * alive (the atexit query fails - see the note above). PQ1MEMQ records that the
 * sample is valid. */
static size_t PQ1FREE = 0, PQ1TOTAL = 0;
static int PQ1MEMQ = 0;
const char *pq1_gpu_stats_mb(char *buf, int n) {
  char fb[16], tb[16];
  const char *fs = "NA", *ts = "NA"; /* NA = not measured, never a fake 0 */
  if (PQ1MEMQ) {
    snprintf(fb, sizeof fb, "%.1f", PQ1FREE / 1048576.0);
    snprintf(tb, sizeof tb, "%.1f", PQ1TOTAL / 1048576.0);
    fs = fb;
    ts = tb;
  }
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
  if (KTON) {
    for (int i = 0; i < NKT; i++)
      fprintf(stderr,
              "PQ1KT rows=%-7d cols=%-6d n=%-4ld total=%8.2f ms  per=%7.4f ms\n",
              KT[i].rows, KT[i].cols, KT[i].n, KT[i].ms, KT[i].ms / KT[i].n);
    if (KTON >= 3)
      for (int i = 0; i < NKT; i++)
        fprintf(stderr,
                "PQ1PH rows=%-7d cols=%-6d n=%-4ld h2d=%7.2f launch=%7.2f "
                "sync=%8.2f d2h=%7.2f ms  (us/call %.1f / %.1f / %.1f / %.1f)\n",
                KT[i].rows, KT[i].cols, KT[i].n, KT[i].h2d, KT[i].launch,
                KT[i].sync, KT[i].d2h,
                KT[i].n ? 1e3 * KT[i].h2d / KT[i].n : 0.0,
                KT[i].n ? 1e3 * KT[i].launch / KT[i].n : 0.0,
                KT[i].n ? 1e3 * KT[i].sync / KT[i].n : 0.0,
                KT[i].n ? 1e3 * KT[i].d2h / KT[i].n : 0.0);
  }
  snprintf(buf, n,
           "PQ1GPU used_mb=%.1f arena_mb=%.1f peak_mb=%.1f free_mb=%s "
           "total_mb=%s tensors=%d tm=%ld r128=%ld r256=%ld swar=%ld zc=%d zcfail=%d batch=%ld brows=%ld",
           PQ1ACUR / 1048576.0, PQ1ALIM / 1048576.0, P2PEAK / 1048576.0,
           fs, ts, PQ1N, PQ1_TM_CALLS, PQ1_R128_CALLS, PQ1_R256_CALLS,
           PQ1_SWAR_CALLS, NZC, ZCFAIL, PQ1_BATCH_CALLS, PQ1_BATCH_ROWS);
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
  { /* sample device memory here: the context is alive now, not at atexit */
    size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) == cudaSuccess) {
      PQ1FREE = f;
      PQ1TOTAL = t;
      PQ1MEMQ = 1;
    }
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
      if (PQ1_TM < 0) {
        /* Run #41: T-MAC is 1.36-1.53x faster per kernel launch in isolation
         * but -1.8% END TO END, because the engine's 253 short launches per
         * token keep the devfreq governor at ~460MHz (measured: median 461MHz,
         * max 845MHz) where the kernel is not ALU-bound. Opt-in only, so the
         * kernel stays available for kernel-level study. */
        PQ1_TM = getenv("PQ1_TM") ? 1 : 0;
      }
      if (PQ1_R128 < 0) {
        /* Run #45: 128 rows/block (2 lanes/row) is the DEFAULT. Interleaved
         * A/B through the canonical harness, 3 runs/side: 2.6952 vs 2.6225
         * decode_tps = +2.8%, every pair separated (t~8). Bit-exact vs the host
         * reference at every shape the gate uses (256x4096, 512x12288).
         * This supersedes run #20's "amortisation saturates at 64 rows": that
         * was a single cross-session harness sample of the same shape, i.e. a
         * null inside a +-2-6% drift band, not a measured zero. Set PQ1_SWAR=1
         * to A/B the 64-row kernel against it. */
        PQ1_R128 = getenv("PQ1_R128") ? atoi(getenv("PQ1_R128"))
                                       : (getenv("PQ1_SWAR") ? 0 : 1);
        /* PQ1_R128 unset -> default rule (rows>=8192 use r256, the rest r128);
         * PQ1_R128=0 -> the 64-row kernel for the shapes the rule sends to r128
         * (the small-N arm of run #48's dispatch experiment, which the old
         * PQ1_SWAR-only initialisation made unreachable - #49 proved the env
         * var was silently ignored); PQ1_R128=1 -> r128 for every shape. */
      }
      if (PQ1_R256 < 0) {
        /* Run #46: 256 rows/block (1 lane/row) is used for LARGE matrices only.
         * Kernel-level (PQ1_TIME, 200 iters, same process as r128):
         *   151669x4096  18.55 vs 19.53 ms  = +5.3% vs r128
         *    12288x4096   1.639 vs 1.744 ms = +6.4% vs r128
         *    4096x12288   2.473 vs 1.756 ms = -41%  (16 blocks underfill one SM)
         * so the dispatch is by row count: >= 8192 rows -> r256, else r128.
         * PQ1_R256=0 forces it off (the A/B arm), PQ1_R256=1 forces it on for
         * every shape (the kernel-level measurement). All three kernels are
         * bit-exact against the host reference, so the rule cannot change the
         * numerics - only the speed. */
        const char *e256 = getenv("PQ1_R256");
        PQ1_R256 = e256 ? atoi(e256) : -1;
      }
      if (KTON < 0) {
        const char *ke = getenv("PQ1_KTIME");
        KTON = ke ? atoi(ke) : 0;
      }
      double kt0 = KTON ? pq1_now() : 0.0;
      double t_b = 0, t_c = 0, t_e = 0;
      size_t aub = PQ1_TM ? (size_t)(cols >> 7) * 4 : 0;
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
      uint32_t *pl = NULL;
      int *auh = NULL;
      int tm_ok = 0;
      if (PQ1_TM) {
        if (!PL_H || xqb > PL_HMAX) {
          free(PL_H);
          PL_H = (uint32_t *)malloc(xqb ? xqb : 1);
          PL_HMAX = PL_H ? xqb : 0;
        }
        if (!AU_H || aub > AU_HMAX) {
          free(AU_H);
          AU_H = (int *)malloc(aub ? aub : 1);
          AU_HMAX = AU_H ? aub : 0;
        }
        if (!d_au || aub > P2AUMAX) {
          if (d_au) cudaFree(d_au);
          d_au = NULL;
          if (cudaMalloc(&d_au, aub) == cudaSuccess) P2AUMAX = aub;
        }
        tm_ok = PL_H && AU_H && d_au;
        if (tm_ok) {
          t_b = KTON > 1 ? pq1_now() : 0.0;
          pq1_build_planes(xq, cols, PL_H);
          pq1_build_au(xq, cols, AU_H);
          t_c = KTON > 1 ? pq1_now() : 0.0;
          pl = (uint32_t *)d_xq;
          auh = (int *)d_au;
        }
      }
      if (getenv("PQ1_TRACE"))
        fprintf(stderr, "PQ1T act d_xq=%p xq=%p n=%zu d_y=%p yb=%zu K=%d N=%d\n",
                (void *)d_xq, (void *)xq, xqb, (void *)d_y, yb, cols, rows);
      if (tm_ok) {
        if (pl && d_y && auh &&
            cudaMemcpy(pl, PL_H, xqb, cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(auh, AU_H, aub, cudaMemcpyHostToDevice) == cudaSuccess &&
            pq1_gemv_tm_launch(dc, dsc, pl, auh, d_y, cols, rows, NULL) == 0 &&
            cudaDeviceSynchronize() == cudaSuccess &&
            cudaMemcpy(y, d_y, yb, cudaMemcpyDeviceToHost) == cudaSuccess) {
          PQ1_TM_CALLS++;
          if (KTON) {
            t_e = pq1_now();
            pq1_kt_add(rows, cols, t_e - kt0);
            if (KTON > 1)
              fprintf(stderr,
                      "PQ1PH rows=%d cols=%d build=%.1f upload+launch+sync+readback"
                      "=%.1f total=%.1f us\n", rows, cols, t_c - t_b, t_e - t_c,
                      t_e - kt0);
          }
          return;
        }
        fprintf(stderr, "pq2: tm device path failed: %s -> SWAR\n",
                cudaGetErrorString(cudaGetLastError()));
        cudaGetLastError();
        PQ1_TM = 0;
      }
      int use256 = (PQ1_R256 < 0) ? (rows >= 8192) : (PQ1_R256 != 0);
      if (d_xq && (d_y || pq1_out_dev(y, yb))) {
        int ok = 1;
        double ta = 0, tb = 0, tc = 0, td = 0, te = 0;
        void *ydev = pq1_out_dev(y, yb);
        if (!ydev && !d_y) ok = 0;
        if (KTON >= 3) ta = pq1_now();
        if (cudaMemcpy(d_xq, xq, xqb, cudaMemcpyHostToDevice) != cudaSuccess) ok = 0;
        if (KTON >= 3) tb = pq1_now();
        if (ok) {
          void *yo = ydev ? ydev : d_y;
          int rc = use256 ? pq1_gemv_r256_launch(dc, dsc, d_xq, yo, cols, rows, NULL)
                   : PQ1_R128 ? pq1_gemv_r128_launch(dc, dsc, d_xq, yo, cols, rows, NULL)
                              : pq1_gemv_sm53_launch(dc, dsc, d_xq, yo, cols, rows, NULL);
          if (rc != 0) ok = 0;
        }
        if (KTON >= 3) tc = pq1_now();
        if (ok && cudaDeviceSynchronize() != cudaSuccess) ok = 0;
        if (KTON >= 3) td = pq1_now();
        if (ok && !ydev && cudaMemcpy(y, d_y, yb, cudaMemcpyDeviceToHost) != cudaSuccess) ok = 0;
        if (KTON >= 3) {
          te = pq1_now();
          pq1_kt_phase(rows, cols, tb - ta, tc - tb, td - tc,
                       ydev ? 0.0 : te - td);
        }
        if (ok) {
          if (use256) PQ1_R256_CALLS++;
          else if (PQ1_R128) PQ1_R128_CALLS++;
          else PQ1_SWAR_CALLS++;
          if (KTON) pq1_kt_add(rows, cols, pq1_now() - kt0);
          return;
        }
      }
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

/* ---- Run #54: batched launches (q+k+v, gate+up) ---------------------------
 * Run #53 (PQ1_KTIME=3) measured the engine's per-call API cost at ~40us H2D +
 * ~45us launch + ~100us D2H, i.e. 46.6ms per token = 12.8% of the 365ms decode
 * step. The engine issues one launch per projection (253/token) and each pays
 * that fixed cost. The projections that share an input vector can share a launch:
 * q, k, v all read the same Q8_K XN, and gate and up both read the post-attention
 * norm. This entry takes up to PQ1_BATCH_MAX tensors, uploads the activation
 * ONCE, launches ONCE, syncs ONCE, then reads back each output (the D2H cannot
 * be merged: the CPU consumes QF/KF/VF as separate vectors).
 *
 * Returns 0 when the batch ran, non-zero when the caller must run the scalar path
 * (no CUDA, cache miss that cannot be uploaded, launch failure...). The math per
 * output row is the same kernel body as the single-tensor path, so the results are
 * bit-identical; pq1_parity verifies the batch kernels against the host reference.
 * PQ1_BATCH=0 disables it (the A/B arm). */
#define PQ1_BATCH_MAX 3
static int PQ1_BATCH = -1;
static uint8_t *d_yb = NULL;
static size_t P2YBMAX = 0;

int pq1_pq1_compat_gemv_batch(const unsigned char *const *w, const int *rows, int n,
                               int cols, float *const *ys, const unsigned char *xq) {
#ifdef PQ1_USE_CUDA
  if (PQ1_BATCH < 0) {
    const char *e = getenv("PQ1_BATCH");
    PQ1_BATCH = e ? atoi(e) : 1;
  }
  if (!PQ1_BATCH || n < 1 || n > PQ1_BATCH_MAX) return -1;
  if ((cols & 127) != 0 || cols <= 0) return -1;

  int ng = cols / 128;
  const uint8_t *dc[PQ1_BATCH_MAX];
  unsigned short *dsc[PQ1_BATCH_MAX];
  int nrow[PQ1_BATCH_MAX];
  int total = 0;
  for (int i = 0; i < n; i++) {
    dc[i] = NULL;
    dsc[i] = NULL;
    nrow[i] = rows[i];
    int ci = -1;
    for (int j = 0; j < PQ1N; j++)
      if (PQ1C[j].w == w[i] && PQ1C[j].rows == rows[i] && PQ1C[j].cols == cols) {
        ci = j;
        break;
      }
    if (ci < 0) return -1; /* not repacked+cached yet: let the caller use the scalar path */
    dc[i] = PQ1C[ci].d_codes;
    dsc[i] = PQ1C[ci].d_scales;
    if (!dc[i] || !dsc[i]) return -1;
    total += rows[i];
  }

  size_t xqb = (size_t)(cols / 256) * 292, yb = (size_t)total * 4;
  if (!d_xq || xqb > P2XQMAX) {
    if (d_xq) cudaFree(d_xq);
    d_xq = NULL;
    if (cudaMalloc(&d_xq, xqb) == cudaSuccess) P2XQMAX = xqb;
  }
  if (!d_yb || yb > P2YBMAX) {
    if (d_yb) cudaFree(d_yb);
    d_yb = NULL;
    if (cudaMalloc(&d_yb, yb) == cudaSuccess) P2YBMAX = yb;
  }
  if (!d_xq || !d_yb) { cudaGetLastError(); return -1; }

  float *dy[PQ1_BATCH_MAX];
  int off = 0;
  for (int i = 0; i < n; i++) {
    dy[i] = (float *)d_yb + off;
    off += rows[i];
  }
  int use256 = 1;
  for (int i = 0; i < n; i++)
    if (rows[i] < 8192) use256 = 0;
  /* Run #54 measured the two shapes in one job (PQ1_KTIME=3, per launch):
   *   gate+up (24576 rows, 256 rows/block): batch sync 3.152ms vs 2 x 1.731ms
   *     separate = 3.462ms -> the batch kernel is 9% FASTER, and it saves one
   *     H2D + one launch + one D2H of API time (0.168ms) => +0.48ms per layer.
   *   q+k+v (6144 rows, 128 rows/block): batch sync 1.645ms vs 1.101ms for the
   *     three separate launches (+49%), which costs more than the 0.237ms of API
   *     time it saves => -0.31ms per layer. The extra parameters (15 vs 6) put
   *     the small-shape kernel on the critical path even at identical register
   *     counts (40, 0 spills, same shared memory).
   * So the default batches only groups whose tensors all have >= 8192 rows. Set
   * PQ1_BATCH=2 to batch every group (the arm this run measured and rejected). */
  if (!use256 && PQ1_BATCH < 2) return -1;

  double kt0 = KTON ? pq1_now() : 0.0;
  double ta = 0, tb = 0, tc = 0, td = 0, te = 0;
  if (KTON >= 3) ta = pq1_now();
  if (cudaMemcpy(d_xq, xq, xqb, cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaGetLastError();
    return -1;
  }
  if (KTON >= 3) tb = pq1_now();
  if (pq1_gemv_batch_launch(use256, dc, dsc, dy, nrow, n, cols, total, d_xq,
                            NULL) != 0) {
    cudaGetLastError();
    return -1;
  }
  if (KTON >= 3) tc = pq1_now();
  if (cudaDeviceSynchronize() != cudaSuccess) {
    cudaGetLastError();
    return -1;
  }
  if (KTON >= 3) td = pq1_now();
  off = 0;
  for (int i = 0; i < n; i++) {
    if (cudaMemcpy(ys[i], d_yb + (size_t)off * 4, (size_t)rows[i] * 4,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      cudaGetLastError();
      return -1;
    }
    off += rows[i];
  }
  if (KTON >= 3) {
    te = pq1_now();
    for (int i = 0; i < n; i++) pq1_kt_phase(rows[i], cols, 0.0, 0.0, 0.0, 0.0);
    for (int i = 0; i < n; i++)
      pq1_kt_add(rows[i], cols, (te - kt0) / n); /* one launch, charged evenly */
    fprintf(stderr,
            "PQ1BT n=%d cols=%-6d h2d=%6.1f launch=%6.1f sync=%7.1f d2h=%6.1f ms\n",
            n, cols, tb - ta, tc - tb, td - tc, te - td);
  }
  PQ1_BATCH_CALLS++;
  PQ1_BATCH_ROWS += total;
  if (use256) PQ1_R256_CALLS += n;
  else PQ1_R128_CALLS += n;
  if (KTON) pq1_kt_add(-n, cols, te - kt0);
  (void)ng;
  return 0;
#else
  (void)w; (void)rows; (void)n; (void)cols; (void)ys; (void)xq;
  return -1;
#endif
}
