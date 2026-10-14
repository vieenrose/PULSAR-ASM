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

/* Fixed sizes for the staging buffers a captured graph bakes in (run #57). A
 * graph records the pointers it sees, so these are allocated once at their final
 * size and NEVER reallocated; a request that exceeds them declines the graph path
 * (and the merged path) and falls back to the explicit API sequence. The engine's
 * largest activation is K=12288 (14KB) and its largest output is the 151669-row
 * head (607KB), so 256KB / 1MB leave 10x headroom. */
#define PQ1_GRAPH_XQ (256 * 1024)
#define PQ1_GRAPH_YB (1024 * 1024)

#ifdef PQ1_USE_CUDA
typedef struct {
  int seen;              /* calls with this key (graph captured at 2) */
  int ok;                /* graph is instantiated and usable */
  int use256;            /* kernel/shape selector, part of the key */
  const void *dc, *dsc;  /* device codes/scales, must be stable */
  const void *xq, *y;    /* host activation, device/dest output, must be stable */
  const void *ydev;      /* device buffer the kernel writes (stable!) */
  int K, N;
  size_t xqb, yb;
  cudaGraphExec_t ex;
} pq1_graph;

static int PQ1_GRAPH = -1;  /* run #57: CUDA graphs for the GEMV API sequence; -1 = auto (on), 0 = off (the A/B arm) */
static long PQ1_GRAPH_OK = 0, PQ1_GRAPH_CALLS = 0; /* graphs instantiated / launched */
static cudaStream_t PQ1_CS = NULL; /* capture stream, created on first use */
static uint32_t *d_pl; static int *d_au; /* T-MAC device prep buffers (defined below) */

#ifdef PQ1_USE_CUDA
static void pq1_graph_init(void) {
  if (PQ1_GRAPH < 0) {
    const char *e = getenv("PQ1_GRAPH");
    PQ1_GRAPH = e ? atoi(e) : 1;
  }
}

static int pq1_graph_run(pq1_graph *G, int use256 /* 0 r128, 1 r256, 2 tm */, const void *dc, const void *dsc,
                         const void *xq, void *xq_d, void *y_dev, void *y_host,
                         int K, int N, size_t xqb, size_t yb) {
  if (PQ1_GRAPH == 0) return -1;
  int same = G->seen && G->use256 == use256 && G->dc == dc && G->dsc == dsc &&
             G->xq == xq && G->y == y_host && G->ydev == y_dev && G->K == K &&
             G->N == N && G->xqb == xqb && G->yb == yb;
  if (!same) {
    G->seen = 1; G->ok = 0;
    G->use256 = use256; G->dc = dc; G->dsc = dsc; G->xq = xq; G->y = y_host;
    G->ydev = y_dev; G->K = K; G->N = N; G->xqb = xqb; G->yb = yb;
    return -1;
  }
  if (!G->ok) {
    if (G->seen < 2) { G->seen = 2; return -1; }   /* capture on the 2nd call */
    if (!PQ1_CS && cudaStreamCreateWithFlags(&PQ1_CS, cudaStreamNonBlocking) != cudaSuccess) {
      cudaGetLastError();
      PQ1_GRAPH = 0;
      return -1;
    }
    cudaGraph_t g = NULL;
    int cap = 0, rc = 0;
    if (cudaStreamBeginCapture(PQ1_CS, cudaStreamCaptureModeThreadLocal) == cudaSuccess) {
      cap = 1;
      cudaMemcpyAsync(xq_d, (void *)xq, xqb, cudaMemcpyHostToDevice, PQ1_CS);
      if (use256 == 2) {
        /* run #72: raw bytes up, planes + group sums built on the device, then
         * the bit-sliced kernel - all inside the one graph */
        rc = pq1_tm_prep_launch((const uint8_t *)xq_d, d_pl, d_au, K, PQ1_CS);
        if (rc == 0)
          rc = pq1_gemv_tm_launch((const uint8_t *)dc, (const unsigned short *)dsc,
                                  d_pl, d_au, (float *)y_dev, K, N, PQ1_CS);
      } else
      rc = use256
        ? pq1_gemv_r256_launch((const uint8_t *)dc, (const unsigned short *)dsc,
                               (const uint8_t *)xq_d, (float *)y_dev, K, N, PQ1_CS)
        : pq1_gemv_r128_launch((const uint8_t *)dc, (const unsigned short *)dsc,
                               (const uint8_t *)xq_d, (float *)y_dev, K, N, PQ1_CS);
      cudaMemcpyAsync(y_host, y_dev, yb, cudaMemcpyDeviceToHost, PQ1_CS);
      if (cudaStreamEndCapture(PQ1_CS, &g) != cudaSuccess) { cap = 0; g = NULL; }
      if (rc != 0) cap = 0;
    }
    if (cap && g) {
      if (cudaGraphInstantiate(&G->ex, g, NULL, NULL, 0) == cudaSuccess && G->ex) {
        G->ok = 1;
        PQ1_GRAPH_OK++;
      } else {
        cudaGetLastError();
        PQ1_GRAPH = 0;   /* this platform will not instantiate graphs */
      }
    } else if (g) {
      cudaGraphDestroy(g);
    }
    cudaGetLastError();
    if (!G->ok) { G->seen = 1; return -1; }
  }
  if (cudaGraphLaunch(G->ex, 0) != cudaSuccess) {
    cudaGetLastError();
    G->ok = 0;
    return -1;
  }
  return 0;
}
#endif

typedef struct {
  const uint8_t *w;
  uint8_t *codes;   /* host splits (may be NULL after upload) */
  uint16_t *scales;
#ifdef PQ1_USE_CUDA
  uint8_t *d_codes;
  unsigned short *d_scales;
  pq1_graph g;   /* run #57: captured {H2D, kernel, D2H} for this tensor */
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
/* T-MAC path (run #41, device-side prep since #72): d_xq receives the raw
 * Q8_K bytes exactly as for the SWAR kernels, pq1_tm_prep builds the bit-plane
 * mirror (same byte size, 73 words per 256-activation block) into d_pl and the
 * K/128 group sums into d_au on the GPU, and the whole {H2D, prep, tm, D2H}
 * sequence is one captured graph. Both buffers are allocated ONCE at a fixed
 * size (graphs bake their pointers in) and never reallocated. */
static uint32_t *d_pl = NULL;
static int *d_au = NULL;
static size_t P2PLMAX = 0, P2AUMAX = 0;
#define PQ1_GRAPH_AU (PQ1_GRAPH_XQ / 64)   /* 4 KB: K/128 ints up to K=131072 */
static int pq1_tm_bufs(int K, size_t xqb, size_t yb, size_t ymax) {
  if (!d_pl && cudaMalloc((void **)&d_pl, PQ1_GRAPH_XQ) == cudaSuccess) P2PLMAX = PQ1_GRAPH_XQ;
  if (!d_au && cudaMalloc((void **)&d_au, PQ1_GRAPH_AU) == cudaSuccess) P2AUMAX = PQ1_GRAPH_AU;
  size_t aub = (size_t)(K >> 7) * 4;
  return d_pl && d_au && xqb <= P2PLMAX && aub <= P2AUMAX && yb <= ymax;
}
static int PQ1_TM = -1;
static int PQ1_R128 = -1;   /* run #45: 128 rows/block, 2 lanes/row (default) */
static int PQ1_R256 = -1;   /* run #46: 256 rows/block; -1 = auto (rows>=8192) */
static long PQ1_TM_CALLS = 0, PQ1_R128_CALLS = 0, PQ1_R256_CALLS = 0, PQ1_SWAR_CALLS = 0; /* diagnostic: which kernel ran */
static int PQ1_BATCH = -1;  /* run #55: merged-tensor launches; -1 = auto, 0 = off, 2 = every group */
static long PQ1_BATCH_CALLS = 0, PQ1_BATCH_ROWS = 0; /* diagnostic: merged launches */

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
           "total_mb=%s tensors=%d tm=%ld r128=%ld r256=%ld swar=%ld zc=%d zcfail=%d batch=%ld brows=%ld graphs=%ld/%ld",
           PQ1ACUR / 1048576.0, PQ1ALIM / 1048576.0, P2PEAK / 1048576.0,
           fs, ts, PQ1N, PQ1_TM_CALLS, PQ1_R128_CALLS, PQ1_R256_CALLS,
           PQ1_SWAR_CALLS, NZC, ZCFAIL, PQ1_BATCH_CALLS, PQ1_BATCH_ROWS,
           PQ1_GRAPH_OK, PQ1_GRAPH_CALLS);
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
        /* Run #46/#50: T-MAC (bit-sliced inner loop) is 1.36-1.53x faster per
         * kernel launch in isolation and bit-exact, but measures -3% to -3.7%
         * END TO END when it replaces the SWAR kernel on EVERY shape. #42's
         * per-shape table that appeared to show the long shapes winning was
         * VOID (#45: that build omitted -DPQ1_REAL_CUDA, so T-MAC never ran).
         * Modes: unset/0 = off (default), 1 = all shapes, 2 = only the long
         * shapes (rows >= 100000, i.e. the head GEMV) - the shape-conditional
         * arm, which has never had valid data. Kernel stays available for study. */
        const char *e = getenv("PQ1_TM");
        /* DEFAULT = 4 (head + wide). Run #60 gated T-MAC to the head (+0.92%,
         * rows >= 100000); run #61 extended it to the WIDE merged blocks
         * (gate+up, 24576 rows = 38% of decode) for +6.9% more, while the
         * long-K arm (down, mode 3) measured -3.05% and stays OFF. Those are
         * the first VALID shape-conditional T-MAC measurements (#42's per-shape
         * table was void per #45: T-MAC never ran in that build).
         * PQ1_TM=0 = pure SWAR, 1 = every shape (measured -3% at #50), 5 = also
         * long K (different algorithm, kept for study). */
        /* Run #72: DEFAULT = 1 (every shape). With the planes built on the
         * device inside the captured graph (pq1_tm_prep) the per-call cost
         * that made T-MAC lose on the narrow shapes is gone, and the 2-rows/
         * thread kernel (#71) is the fastest kernel at EVERY decode shape
         * (hot loop vs r128: 6144x4096 0.43 vs 0.69 ms, 4096x4096 0.32 vs
         * 0.48, 4096x12288 0.81 vs 1.35, 24576x4096 1.40 vs 2.55). Cold
         * harness 4.61 -> 5.92 tps; PQ1_TM=4 is the old gate (A/B arm). */
        PQ1_TM = e ? atoi(e) : 1;
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
      /* Shape gate for the bit-sliced kernel. T-MAC's advantage grows with the
       * length of the per-lane loop (it amortises the plane work over the K
       * dimension), and #46/#50 showed it LOSES when applied to every shape.
       * Modes: 0 = off everywhere, 1 = all shapes, 2 = the long-ROW shapes
       * (head, 151669 rows), 3 = long K (down, K=12288), 4 = wide (gate+up,
       * 24576 rows), 5 = 3 or 4 combined. */
      int tm_here = 0;
      if (PQ1_TM == 1) tm_here = 1;
      else if (PQ1_TM >= 2) {
        int long_rows = rows >= 100000;                      /* always: proven +13% */
        int long_k    = (PQ1_TM == 3 || PQ1_TM == 5) && cols >= 12288;
        int wide      = (PQ1_TM == 4 || PQ1_TM == 5) && rows >= 20000;
        tm_here = long_rows || long_k || wide;
      }
      if (!d_xq) {
        if (cudaMalloc(&d_xq, PQ1_GRAPH_XQ) == cudaSuccess) P2XQMAX = PQ1_GRAPH_XQ;
      }
      if (!d_y) {
        if (cudaMalloc(&d_y, PQ1_GRAPH_YB) == cudaSuccess) P2YMAX = PQ1_GRAPH_YB;
      }
      int tm_ok = tm_here && d_xq && d_y && xqb <= P2XQMAX &&
                  pq1_tm_bufs(cols, xqb, yb, P2YMAX);
      if (getenv("PQ1_TRACE"))
        fprintf(stderr, "PQ1T act d_xq=%p xq=%p n=%zu d_y=%p yb=%zu K=%d N=%d tm=%d\n",
                (void *)d_xq, (void *)xq, xqb, (void *)d_y, yb, cols, rows, tm_ok);
      if (tm_ok) {
        /* run #72: {H2D raw xq, device prep, tm kernel, D2H} as ONE graph launch,
         * same as the SWAR path. Until the graph is captured (2nd call) or if
         * graphs are off, the same four steps run as plain API calls. */
        if (PQ1_GRAPH < 0) pq1_graph_init();
        int ok = 1;
        int gr = pq1_graph_run(&PQ1C[ci].g, 2, dc, dsc, xq, d_xq, d_y, (void *)y,
                               cols, rows, xqb, yb);
        t_b = KTON > 1 ? pq1_now() : 0.0;
        if (gr == 0) {
          if (cudaDeviceSynchronize() != cudaSuccess) ok = 0;
          else PQ1_GRAPH_CALLS++;
        } else {
          ok = cudaMemcpy(d_xq, xq, xqb, cudaMemcpyHostToDevice) == cudaSuccess &&
               pq1_tm_prep_launch(d_xq, d_pl, d_au, cols, NULL) == 0 &&
               pq1_gemv_tm_launch(dc, dsc, d_pl, d_au, d_y, cols, rows, NULL) == 0 &&
               cudaDeviceSynchronize() == cudaSuccess &&
               cudaMemcpy(y, d_y, yb, cudaMemcpyDeviceToHost) == cudaSuccess;
        }
        if (ok) {
          PQ1_TM_CALLS++;
          if (KTON) {
            t_e = pq1_now();
            pq1_kt_add(rows, cols, t_e - kt0);
            if (KTON > 1)
              fprintf(stderr, "PQ1PH rows=%d cols=%d graph=%d api=%.1f total=%.1f us\n",
                      rows, cols, gr == 0, (t_e - t_b) * 1e3, (t_e - kt0) * 1e3);
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
        if (xqb > P2XQMAX || yb > P2YMAX) ok = 0; /* too big for the fixed buffers */
        int gr = pq1_graph_run(&PQ1C[ci].g, use256, dc, dsc, xq, d_xq, d_y, (void *)y,
                               cols, rows, xqb, yb);
        if (KTON >= 3) ta = pq1_now();
        if (gr == 0) {
          /* the graph contains the H2D, the kernel and the D2H: one API call */
          if (cudaDeviceSynchronize() != cudaSuccess) ok = 0;
          if (KTON >= 3) { tb = tc = td = pq1_now(); te = pq1_now(); }
          if (ok) PQ1_GRAPH_CALLS++;
        } else if (ok) {
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

/* ---- Run #55: merged (concatenated) tensors, one scalar launch -------------
 * Run #53 (PQ1_KTIME=3) measured the engine's per-call API cost at ~40us H2D +
 * ~45us launch + ~100us D2H, 46.6ms per token = 12.8% of the decode step, paid
 * once per projection (253 launches/token). Run #54 batched the projections that
 * share an activation vector into one launch of a MULTI-TENSOR kernel and found
 * the win is real for the large group (gate+up: +1.37% end to end) but NEGATIVE
 * for small shapes, because that kernel needs 15 parameters (3 code + 3 scale +
 * 3 y pointers + 3 row counts + K + N) and its start-up cost is not amortised on
 * a 6144-row launch (+49% kernel time).
 *
 * This version removes the multi-pointer kernel entirely: the tensors of a group
 * are CONCATENATED AT REPACK TIME into one contiguous device tensor of
 * rows_total rows, and the EXISTING scalar kernel runs the whole matrix. The
 * kernel is byte-for-byte the one the parity gate certifies, so kernel time is
 * the scalar kernel's; the API saving is 1 H2D + 1 launch + 1 D2H per group.
 * The single output buffer is read back in one D2H into a host staging buffer and
 * scattered to QF/KF/VF (or LG/LU) with plain CPU memcpy - a few KB, microseconds,
 * and bit-identical because the copy moves the kernel's floats verbatim.
 *
 * Returns 0 when the merged launch ran, non-zero when the caller must run the
 * scalar path per tensor (no CUDA, OOM, launch failure). PQ1_BATCH=0 disables it
 * (the A/B arm), PQ1_BATCH=1 batches only groups whose tensors all have >= 8192
 * rows (measured: that is where merging wins), PQ1_BATCH=2 batches every group.
 */
#define PQ1_BATCH_MAX 3
static uint8_t *MDY = NULL;        /* device output buffer for merged launches */
static size_t P2MDYMAX = 0;


typedef struct {
  const unsigned char *w[PQ1_BATCH_MAX];
  int rows[PQ1_BATCH_MAX];
  int n, cols;
  uint8_t *d_codes;
  unsigned short *d_scales;
#ifdef PQ1_USE_CUDA
  pq1_graph g;
#endif
} pq1_merged;
/* Capacity of the merged-tensor cache. The engine needs 72 entries (36 layers x
 * 2 groups: q/k/v and gate/up), but the default has been 64, so the last 4
 * layers silently never take the merged path - they run as separate scalar
 * launches (visible in the counters: 64 merged calls per step, not 72). Each
 * merged entry is a second copy of that group's weights in the arena, so a
 * larger cache trades GPU RAM for fewer launches; PQ1_MERGED_CAP makes the size
 * runtime-tunable so the trade can be A/B'd. Default = 64 (the historical
 * behaviour); unset means 64. */
#define PQ1_MERGED_MAX 256
static int PQ1_MERGED_CAP = 0;
static pq1_merged PQ1M[PQ1_MERGED_MAX];
static int PQ1NM = 0;

/* ---- Run #57: CUDA graphs for the per-GEMV API sequence -------------------
 * tools/graph_probe.cu measured, on this board and in one process: the engine's
 * per-GEMV API pattern (H2D of the activation, one kernel, D2H of the output,
 * then cudaDeviceSynchronize) costs 285us with a tiny kernel, and 204us as a
 * captured graph - 81us per call of pure API overhead, 145 calls/token = 11.7ms
 * = 3.3% of the 352ms decode step. The launch+sync floor alone is 133us.
 *
 * A graph bakes in the pointers and the launch configuration, so a graph may only
 * be captured once every buffer it references is PERMANENT. That is why MDY and
 * d_xq below are allocated at their maximum size once and never reallocated (a
 * reallocation would leave a captured graph pointing at freed memory - a silent
 * corruption, not a crash). The capture is lazy: the first call records the key
 * and runs the explicit path; the second call with the SAME key captures; every
 * later call with that key launches the graph. Any failure at any step falls back
 * to the explicit path, which is bit-identical, so a platform without usable
 * graphs loses nothing.
 *
 * Numerics: the graph contains the SAME kernel launch with the SAME parameters and
 * the SAME D2H, so the values are identical; the greedy goldens are the proof. */
#endif



int pq1_pq1_compat_gemv_batch(const unsigned char *const *w, const int *rows, int n,
                               int cols, float *const *ys, const unsigned char *xq) {
#ifdef PQ1_USE_CUDA
  if (PQ1_BATCH < 0) {
    const char *e = getenv("PQ1_BATCH");
    PQ1_BATCH = e ? atoi(e) : 2;
  }
  if (!PQ1_BATCH || n < 1 || n > PQ1_BATCH_MAX) return -1;
  if ((cols & 127) != 0 || cols <= 0) return -1;
  int total = 0;
  for (int i = 0; i < n; i++) {
    if (rows[i] <= 0) return -1;
    total += rows[i];
  }
  int use256 = 1;
  for (int i = 0; i < n; i++)
    if (rows[i] < 8192) use256 = 0;
  /* Run #54 measured batching the small group as a LOSS (+49% kernel time) - but
   * that was its multi-pointer batch KERNEL. The merged path runs the unmodified
   * scalar kernel, so that cost is gone, and run #55 measured q+k+v merging at
   * +3.3% in a position-balanced A/B. Default (PQ1_BATCH unset -> 2) therefore
   * merges every group; PQ1_BATCH=1 restricts merging to groups whose tensors all
   * have >= 8192 rows (the #55 default), PQ1_BATCH=0 disables it (the A/B arm).
   * The dispatch below is the scalar kernels' own rule applied to the merged row
   * count: total >= 8192 -> r256 (1 lane/row), else r128 (2 lanes/row). */
  if (!use256 && PQ1_BATCH < 2) return -1;

  /* find or build the merged tensor */
  int mi = -1;
  for (int i = 0; i < PQ1NM; i++) {
    if (PQ1M[i].n == n && PQ1M[i].cols == cols) {
      int same = 1;
      for (int k = 0; k < n; k++)
        if (PQ1M[i].w[k] != w[k] || PQ1M[i].rows[k] != rows[k]) { same = 0; break; }
      if (same) { mi = i; break; }
    }
  }
  size_t rowcb = (size_t)(cols / 32) * 4, rowsc = (size_t)(cols / 128) * 2;
  size_t ncb = (size_t)total * rowcb, nsb = (size_t)total * rowsc;
  size_t a = (ncb + 63) / 64 * 64, b = (nsb + 63) / 64 * 64;
  if (mi < 0) {
    if (!PQ1_MERGED_CAP) {
      const char *e = getenv("PQ1_MERGED_CAP");
      /* Default 128: the engine needs 72 entries, so 64 silently left the last
       * 4 layers unmerged (see the header comment above). Measured +0.82%
       * decode (order-balanced A/B, pairs +0.87%/+0.78%). */
      PQ1_MERGED_CAP = e ? atoi(e) : 128;
      if (PQ1_MERGED_CAP < 1) PQ1_MERGED_CAP = 1;
      if (PQ1_MERGED_CAP > PQ1_MERGED_MAX) PQ1_MERGED_CAP = PQ1_MERGED_MAX;
    }
    if (PQ1NM >= PQ1_MERGED_CAP) return -1;
    if (pq1_arena_init(a + b) != 0) return -1;
    uint8_t *pb = PQ1ARENA + PQ1ACUR;
    uint8_t *ps = pb + a;
    uint8_t *tmpc = NULL;
    uint16_t *tmps = NULL;
    int ok = 1;
    size_t oc = 0, os = 0;
    for (int i = 0; i < n; i++) {
      size_t cb = (size_t)rows[i] * rowcb, sb = (size_t)rows[i] * rowsc; /* both bytes */
      void *tc = realloc(tmpc, cb);
      void *ts = realloc(tmps, sb ? sb : 2);
      if (!tc || !ts) { ok = 0; break; }
      tmpc = (uint8_t *)tc;
      tmps = (uint16_t *)ts;
      pq1_split_repack(w[i], rows[i], cols, (uint32_t *)tmpc, tmps);
      if (cudaMemcpy(pb + oc, tmpc, cb, cudaMemcpyHostToDevice) != cudaSuccess ||
          cudaMemcpy(ps + os, tmps, sb, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaGetLastError();
        ok = 0;
        break;
      }
      oc += cb;
      os += sb;
      /* the GGUF pages of a tensor consumed by a merged tensor are single-use */
      size_t tbytes = (size_t)rows[i] * (cols / 128) * 18;
      uintptr_t p0 = (uintptr_t)w[i] & ~(uintptr_t)4095;
      uintptr_t p1 = ((uintptr_t)w[i] + tbytes + 4095) & ~(uintptr_t)4095;
      if (p1 > p0) madvise((void *)p0, (size_t)(p1 - p0), MADV_DONTNEED);
    }
    free(tmpc);
    free(tmps);
    if (!ok) return -1;
    PQ1ACUR += a + b;
    if (PQ1ACUR > P2PEAK) P2PEAK = PQ1ACUR;
    mi = PQ1NM++;
    for (int i = 0; i < n; i++) { PQ1M[mi].w[i] = w[i]; PQ1M[mi].rows[i] = rows[i]; }
    PQ1M[mi].n = n;
    PQ1M[mi].cols = cols;
    PQ1M[mi].d_codes = pb;
    PQ1M[mi].d_scales = (unsigned short *)ps;
    if (getenv("PQ1_TRACE"))
      fprintf(stderr, "PQ1M merged n=%d cols=%d rows=%d codes=%p+%zu scales=%p+%zu\n",
              n, cols, total, (void *)pb, ncb, (void *)ps, nsb);
  }
  const uint8_t *dc = PQ1M[mi].d_codes;
  const unsigned short *dsc = PQ1M[mi].d_scales;
  if (!dc || !dsc) return -1;

  size_t xqb = (size_t)(cols / 256) * 292, yb = (size_t)total * 4;
  /* A captured graph bakes in every pointer it sees, so these three buffers must
   * be allocated at their final size the FIRST time and never reallocated. The
   * engine's largest activation is K=12288 (14KB) and its largest merged output
   * is gate+up (24576 rows = 98KB); 1MB each is 10x headroom for any larger model
   * and costs 2MB of the arena. If a request exceeds it, the merged path declines
   * (returns -1) and the caller uses the scalar path - never a dangling graph. */
  if (!d_xq) {
    if (cudaMalloc(&d_xq, PQ1_GRAPH_XQ) == cudaSuccess) P2XQMAX = PQ1_GRAPH_XQ;
  }
  if (!MDY) {
    if (cudaMalloc(&MDY, PQ1_GRAPH_YB) == cudaSuccess) P2MDYMAX = PQ1_GRAPH_YB;
  }
  static uint8_t *stage = NULL;
  if (!stage) stage = (uint8_t *)malloc(PQ1_GRAPH_YB);
  if (xqb > P2XQMAX || yb > P2MDYMAX || !d_xq || !MDY || !stage) {
    cudaGetLastError();
    return -1;
  }
  if (!d_xq || !MDY) { cudaGetLastError(); return -1; }

  double kt0 = KTON ? pq1_now() : 0.0;
  double ta = 0, tb = 0, tc = 0, td = 0, te = 0;
  int tm_done = 0;
  if (PQ1_TM < 0) { const char *e = getenv("PQ1_TM"); PQ1_TM = e ? atoi(e) : 1; }
  /* Shape-gated bit-sliced kernel for the WIDE merged blocks (gate+up, 24576
   * rows = 38% of decode). Same mode table as the scalar path: 1 = all shapes,
   * 4/5 = wide (rows >= 20000), 3/5 = long K. The scalar path cannot see these
   * matrices - they are concatenated and launched here - so the gate lives here
   * too. Bit-exactness is the same TM kernel the parity gate certifies. */
  {
    int tm_here = (PQ1_TM == 1) ||
                  ((PQ1_TM == 4 || PQ1_TM == 5) && total >= 20000) ||
                  ((PQ1_TM == 3 || PQ1_TM == 5) && cols >= 12288);
    if (tm_here && pq1_tm_bufs(cols, xqb, yb, P2MDYMAX)) {
      /* run #72: one graph {H2D raw xq, device prep, tm kernel, D2H} exactly as
       * the SWAR merged path below; plain API calls until it is captured. */
      if (PQ1_GRAPH < 0) pq1_graph_init();
      int gr = pq1_graph_run(&PQ1M[mi].g, 2, dc, dsc, xq, d_xq, (void *)MDY,
                             (void *)stage, cols, total, xqb, yb);
      int ok;
      if (gr == 0) {
        ok = cudaDeviceSynchronize() == cudaSuccess;
        if (ok) PQ1_GRAPH_CALLS++;
      } else {
        ok = cudaMemcpy(d_xq, xq, xqb, cudaMemcpyHostToDevice) == cudaSuccess &&
             pq1_tm_prep_launch(d_xq, d_pl, d_au, cols, NULL) == 0 &&
             pq1_gemv_tm_launch(dc, dsc, d_pl, d_au, MDY, cols, total, NULL) == 0 &&
             cudaDeviceSynchronize() == cudaSuccess &&
             cudaMemcpy(stage, MDY, yb, cudaMemcpyDeviceToHost) == cudaSuccess;
      }
      if (ok) {
        PQ1_TM_CALLS++;
        tm_done = 1;
        if (KTON) pq1_kt_add(total, cols, pq1_now() - kt0);
        /* The caller receives this group's outputs by scattering from the
         * staging buffer below (the scalar kernels write directly into the
         * engine's own vectors; merged ones are concatenated, so the launch
         * writes MDY and the results are copied out here). Returning early
         * skipped the scatter and silently starved the engine of gate/up
         * outputs - which the GOLDEN STREAMS caught (run #61) while the
         * parity gate stayed green, because the bug was in this glue, not in
         * any kernel. Fall through to the scatter instead. */
        goto pq1_merged_scatter;
      }
      cudaGetLastError();
    }
  }
  if (PQ1_GRAPH < 0) pq1_graph_init();
  int gr = pq1_graph_run(&PQ1M[mi].g, use256, dc, dsc, xq, d_xq, (void *)MDY,
                         (void *)stage, cols, total, xqb, yb);
  if (KTON >= 3) ta = pq1_now();
  if (gr == 0) {
    /* the graph contains the H2D, the kernel and the D2H: one API call */
    if (cudaDeviceSynchronize() != cudaSuccess) { cudaGetLastError(); return -1; }
    if (KTON >= 3) { tb = tc = td = pq1_now(); te = pq1_now(); }
    PQ1_GRAPH_CALLS++;
  } else {
    if (cudaMemcpy(d_xq, xq, xqb, cudaMemcpyHostToDevice) != cudaSuccess) {
      cudaGetLastError();
      return -1;
    }
    if (KTON >= 3) tb = pq1_now();
    int rc = use256 ? pq1_gemv_r256_launch(dc, dsc, d_xq, (float *)MDY, cols, total, NULL)
                    : pq1_gemv_r128_launch(dc, dsc, d_xq, (float *)MDY, cols, total, NULL);
    if (rc != 0) { cudaGetLastError(); return -1; }
    if (KTON >= 3) tc = pq1_now();
    if (cudaDeviceSynchronize() != cudaSuccess) { cudaGetLastError(); return -1; }
    if (KTON >= 3) td = pq1_now();
    if (cudaMemcpy(stage, MDY, yb, cudaMemcpyDeviceToHost) != cudaSuccess) {
      cudaGetLastError();
      return -1;
    }
    if (KTON >= 3) te = pq1_now();
  }
pq1_merged_scatter:
  {
  size_t off = 0;
  for (int i = 0; i < n; i++) {
    memcpy(ys[i], stage + (off * 4), (size_t)rows[i] * 4);
    off += rows[i];
  }
  }
  if (KTON >= 3) {
    fprintf(stderr,
            "PQ1BT n=%d cols=%-6d h2d=%6.1f launch=%6.1f sync=%7.1f d2h=%6.1f ms\n",
            n, cols, tb - ta, tc - tb, td - tc, te - td);
    for (int i = 0; i < n; i++) {
      pq1_kt_add(rows[i], cols, (te - kt0) / n);
      pq1_kt_phase(rows[i], cols, (tb - ta) / n, (tc - tb) / n, (td - tc) / n,
                   (te - td) / n);
    }
  }
  PQ1_BATCH_CALLS++;
  PQ1_BATCH_ROWS += total;
  if (tm_done) ;                       /* counted as tm above (#72: the goto used to land here and double-count) */
  else if (use256) PQ1_R256_CALLS += n;
  else PQ1_R128_CALLS += n;
  return 0;
#else
  (void)w; (void)rows; (void)n; (void)cols; (void)ys; (void)xq;
  return -1;
#endif
}
