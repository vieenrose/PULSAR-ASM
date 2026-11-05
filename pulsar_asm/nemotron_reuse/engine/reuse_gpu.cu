// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
// CUDA implementation of the RE-USE Nano port (sm_53 target).
// fp32 dataflow; long reductions use Kahan compensation because sm_53 fp64
// runs at 1/64 rate and InstanceNorm sums reach 1.7e5 elements per channel.
#include "reuse_gpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <time.h>

static double g_t_enc=0,g_t_blk=0,g_t_dec=0,g_t_head=0;
extern "C" void reuse_gpu_get_times(double *enc,double *blk,double *dec,double *head){*enc=g_t_enc;*blk=g_t_blk;*dec=g_t_dec;*head=g_t_head;}
static double gt_now(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+1e-9*ts.tv_nsec; }


#define HID 64
#define GT 32
#define MT32 32
#define NT64 64
#define DEPTH 4
#define DINNER 256
#define DSTATE 16
#define DCONV 4
#define DTRANK 4
#define XP (DTRANK + 2*DSTATE)

struct entry { const char *name; const float *dptr; };

struct reuse_gpu_s {
    float *base;
    int n;
    entry *ents;
    int bad;
    float *ws;
    size_t ws_cap, ws_off;
};

static int cmp_str(const void *a, const void *b)
{
    const entry *ea = (const entry*)a, *eb = (const entry*)b;
    return strcmp(ea->name, eb->name);
}

extern "C" int reuse_gpu_ok(reuse_gpu_t *g, const char *where)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(e));
        if (g) g->bad = 1;
        return 1;
    }
    return 0;
}

extern "C" void reuse_gpu_destroy(reuse_gpu_t *g)
{
    if (!g) return;
    if (g->ents) { for (int i2 = 0; i2 < g->n; i2++) free((void*)g->ents[i2].name); }
    if (g->base) cudaFree(g->base);
    if (g->ws) cudaFree(g->ws);
    free(g->ents);
    free(g);
}

static int g_kprof = -1;
static double g_kp[512]; static const char *g_kpn[512]; static int g_kpc;
static void kp_mark(const char *name, double *last)
{
    if (g_kprof < 0) g_kprof = getenv("REUSE_KPROF") ? 1 : 0;
    if (!g_kprof) return;
    cudaDeviceSynchronize();
    double t = gt_now();
    if (g_kpc < 512) { g_kpn[g_kpc] = name; g_kp[g_kpc] = t - *last; g_kpc++; }
    *last = t;
}
extern "C" void reuse_gpu_print_kprof(void)
{
    if (!g_kprof || !g_kpc) return;
    double tot = 0;
    for (int i = 0; i < g_kpc; i++) tot += g_kp[i];
    fprintf(stderr, "--- per-kernel (accumulated over the run, %d marks, total %.3f s) ---\n", g_kpc, tot);
    for (int i = 0; i < g_kpc; i++) {
        if (!g_kpn[i]) continue;
        double sum = g_kp[i]; int cnt = 1;
        for (int j = i+1; j < g_kpc; j++) if (g_kpn[j] && strcmp(g_kpn[j], g_kpn[i]) == 0) { sum += g_kp[j]; cnt++; g_kpn[j] = 0; }
        fprintf(stderr, "  %-18s %8.4f s  %5.1f%%   n=%d  avg %7.5f s\n", g_kpn[i], sum, 100.0*sum/tot, cnt, sum/cnt);
    }
}

__global__ void k_touch_pages(float *ws, long cap);

extern "C" reuse_gpu_t *reuse_gpu_create(const reuse_blob_t *blob)
{
    size_t total = 0;
    for (int i = 0; i < blob->ntensors; i++) total += blob->tensors[i].nelem;
    reuse_gpu_t *g = (reuse_gpu_t*)calloc(1, sizeof(*g));
    g->n = blob->ntensors;
    g->ents = (entry*)calloc((size_t)g->n, sizeof(entry));
    if (cudaMalloc(&g->base, total * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "gpu weight OOM %zu floats\n", total); free(g->ents); free(g); return NULL;
    }
    size_t off = 0;
    for (int i = 0; i < blob->ntensors; i++) {
        if (cudaMemcpy(g->base + off, blob->tensors[i].data, blob->tensors[i].nelem * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            fprintf(stderr, "upload failed for %s\n", blob->tensors[i].name);
            reuse_gpu_destroy(g); return NULL;
        }
        // own the name strings: the host blob is freed once weights are uploaded
        g->ents[i].name = strdup(blob->tensors[i].name);
        g->ents[i].dptr = g->base + off;
        off += blob->tensors[i].nelem;
    }
    qsort(g->ents, (size_t)g->n, sizeof(entry), cmp_str);
    // REUSE_WS_MB is megabytes of BYTES (this used to be 512 *Mi* floats, i.e.
    // 2 GiB, which the Nano's 3 GiB shared-memory GPU only over-commits: the
    // pages fault when a kernel touches them, and Tegra reports that as
    // "the launch timed out and was terminated").
    size_t want_mb = (size_t)(getenv("REUSE_WS_MB") ? atoi(getenv("REUSE_WS_MB")) : 1024);
    g->ws = NULL;
    for (size_t mb = want_mb; mb >= 32 && !g->ws; mb >>= 1) {
        size_t cap = mb * 1024 * 1024 / sizeof(float);
        if (cudaMalloc(&g->ws, cap * sizeof(float)) != cudaSuccess) {
            cudaGetLastError(); g->ws = NULL; continue;
        }
        // Tegra backs pages lazily: a 640 MiB cudaMalloc "succeeds" and then a
        // kernel touching an unbacked page produces a poisoned-page MMU fault
        // reported as a launch timeout. Touch the arena now so the failure, if
        // any, is a clean allocation failure at startup.
        if (getenv("REUSE_WS_TOUCH") && atoi(getenv("REUSE_WS_TOUCH")) == 0) {
            if (cudaMemset(g->ws, 0, cap * sizeof(float)) != cudaSuccess) {
                cudaGetLastError(); cudaFree(g->ws); g->ws = NULL; continue;
            }
        } else {
            k_touch_pages<<<(unsigned)((cap/1024 + 255)/256), 256>>>(g->ws, (long)cap);
            if (cudaDeviceSynchronize() != cudaSuccess) {
                cudaGetLastError(); cudaFree(g->ws); g->ws = NULL; continue;
            }
        }
        g->ws_cap = cap;
        if (mb != want_mb) fprintf(stderr, "workspace: %zu MiB (requested %zu MiB)\n", mb, want_mb);
        break;
    }
    if (!g->ws) { fprintf(stderr, "gpu workspace OOM (requested %zu MiB)\n", want_mb); reuse_gpu_destroy(g); return NULL; }
    return g;
}

// Touch one float per 4 KiB page to force the driver to back the arena.
// Tegra backs pages lazily, so the arena has to be touched at startup (see the
// call site); it does NOT have to be zeroed - every consumer writes its whole
// wallc() slice before reading it, and the arena is reused across reps, so any
// kernel that needs zeros already memsets its own target. Zeroing all 1 GiB
// costs 6.7 ms of GPU time, and because the STFT's first pageable H2D copy is
// synchronous with the device, the whole STFT phase waited for it: 4.5 ms of
// the 7.1 ms rtf_stft. Touching one float per page is the same page-fault
// coverage in ~0.05 ms. REUSE_WS_TOUCH=0 restores the cudaMemset.
__global__ void k_touch_pages(float *ws, long cap)
{
    long j = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 1024;   // 4 KiB
    if (j < cap) ws[j] = 0.0f;
}

extern "C" const float *reuse_gpu_w(reuse_gpu_t *g, const char *name)
{
    entry key; key.name = name;
    entry *r = (entry*)bsearch(&key, g->ents, (size_t)g->n, sizeof(entry), cmp_str);
    if (!r) { fprintf(stderr, "MISSING GPU TENSOR: %s\n", name); g->bad = 1; return NULL; }
    return r->dptr;
}

// Maxwell (sm_53 / Jetson Nano) launch-resource guard. Some register-tiled
// kernels (64 accumulators per thread) COMPILE for sm_53 but fail at LAUNCH
// time with "too many resources requested for launch", because ptxas hands back
// a kernel needing more than Maxwell's 255 registers/thread. The sm_53 compile
// gate in checks.sh cannot see a launch failure, so the fallback has to be
// chosen at runtime. REUSE_FORCE_MAXWELL_TILES=1 exercises the branch on a
// dev host.
static int gpu_is_maxwell(void)
{
    static int mw = -1;
    if (mw < 0) {
        cudaDeviceProp dp;
        mw = (cudaGetDeviceProperties(&dp, 0) == cudaSuccess && dp.major < 6) ? 1 : 0;
        if (getenv("REUSE_FORCE_MAXWELL_TILES")) mw = 1;
    }
    return mw;
}

static float *wallc(reuse_gpu_t *g, size_t n)
{
    if (g->ws_off + n > g->ws_cap) {
        fprintf(stderr, "workspace exhausted (%zu + %zu > %zu)\n", g->ws_off, n, g->ws_cap);
        g->bad = 1; return NULL;
    }
    float *p = g->ws + g->ws_off;
    g->ws_off += (n + 31) & ~(size_t)31;
    return p;
}

void gdump(const char *tag, const float *d, size_t n)
{
    const char *base = getenv("GPU_DEBUG");
    if (!base || !d) return;
    float *h = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h, d, n * sizeof(float), cudaMemcpyDeviceToHost);
    char path[512]; snprintf(path, sizeof(path), "%s_gpu_%s.raw", base, tag);
    FILE *f = fopen(path, "wb"); fwrite(h, sizeof(float), n, f); fclose(f);
    free(h);
}

// Grow-only device allocation cache for the per-forward scratch buffers.
// The model's forward and the iSTFT allocated ~11 buffers per call with
// cudaMalloc/cudaFree; nsys shows the driver calls alone costing ~2.5 ms per
// forward (a cudaFree of the iSTFT's 11 MB arena measured 1.1-2.2 ms, and it
// sat in a gap between kernels). The sizes are fixed per shape, so cache them
// for the process lifetime: first call allocates, later calls hit the cache and
// dfree() is a no-op. Peak device memory grows by the ~150 MB of scratch, which
// fits the Nano's 3 GiB alongside the 1 GiB workspace and the weights.
struct dcache_ent { size_t n; void *p; int in_use; };
static dcache_ent dcache[24];
static int dcache_n = 0;

static void *dalloc(size_t n)
{
    // An entry is only reusable once its dfree() has released it: the model
    // allocates several identical-sized buffers live at once (xa/xb ping-pong,
    // dm/dp, fa/fr/fi), so the size alone does not identify a free slot.
    for (int i = 0; i < dcache_n; i++)
        if (dcache[i].n == n && !dcache[i].in_use) { dcache[i].in_use = 1; return dcache[i].p; }
    void *p = NULL;
    if (cudaMalloc(&p, n * sizeof(float)) != cudaSuccess) { fprintf(stderr, "gpu OOM %zu\n", n); return NULL; }
    if (dcache_n < (int)(sizeof dcache / sizeof dcache[0])) {
        dcache[dcache_n].n = n; dcache[dcache_n].p = p; dcache[dcache_n].in_use = 1; dcache_n++;
    } else cudaFree(p);          // cache full: fall back to the old behaviour
    return p;
}

static void dfree(void *p)
{
    for (int i = 0; i < dcache_n; i++)
        if (dcache[i].p == p) { dcache[i].in_use = 0; return; }
    cudaFree(p);                 // not from the cache (cache was full)
}

// ================================================================= kernels

__device__ __forceinline__ void kahan_add(float *s, float *c, float v)
{
    float t = *s + v;
    if (fabsf(*s) >= fabsf(v)) *c += (*s - t) + v; else *c += (v - t) + *s;
    *s = t;
}

__device__ __forceinline__ float dsilu(float v) { return v / (1.0f + __expf(-v)); }
__device__ __forceinline__ float dsoftplus(float v) { return v > 20.0f ? v : log1pf(__expf(v)); }

// y[co, i] = b[co] + sum_ci w[co,ci] * x[ci, i]
__global__ void k_conv1x1(const float *x, const float *w, const float *b,
                          float *y, int C_in, int C_out, long HW, int co0)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int co = co0 + blockIdx.y;
    if (i >= HW) return;
    float acc = b ? b[co] : 0.0f;
    for (int ci = 0; ci < C_in; ci++)
        acc = fmaf(x[(long)ci * HW + i], w[(long)co * C_in + ci], acc);
    y[(long)co * HW + i] = acc;
}

// Direct 2D conv; covers 3x3 dilated and (1,3) strided.
// Tiled kernel for the encoder's strided patchify conv (kH=1, kW=3, dilW=1,
// strideW=2, padW=0, strideH>=1): C_in=64, 1023x163 -> 256x81, C_out=64. The
// naive k_conv2d gives each thread one (row, col, channel) and one output
// channel per z-block, so per thread it issues 192 dependent scalar loads
// against 192 FMAs (1 FMA/load) and all 64 z-blocks re-read the input.
// Here a block owns TR output rows x TC output cols x PCO channels, stages each
// ci's input strip and the PCOx3 weights in shared memory, and each thread
// computes RW consecutive output columns for all PCO channels: per ci that is
// ceil((RW*strideW+kW-1)/4) float4 window loads plus PCO*3 warp-uniform weight
// loads against RW*PCO*3 FMAs (3.8 FMA/load at PCO=16, RW=4), and the input is
// re-read C_out/PCO times instead of C_out.
// The per-output accumulation walks ci ascending and kw ascending, exactly like
// the naive loop, so results are bit-identical.
template<int TR, int TC, int PCO, int RW, int SW>
__global__ void k_conv2d_st(const float *__restrict__ x, const float *__restrict__ w,
                            const float *b, int C_in, int H, int W,
                            int padW, int strideH,
                            float *y, int co0, int H_out, int W_out)
{
    const int TW = ((TC*SW + 2) + 3) / 4 * 4;      /* staged input width, float4 padded */
    const int TXN = TC/RW, NT = TXN*TR;
    __shared__ __align__(16) float xs[TR][TW + 4];
    __shared__ __align__(16) float ws[PCO*4];
    const int ho0 = blockIdx.y*TR;
    const int wo0 = blockIdx.x*TC;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty*TXN + tx;
    const int co = co0 + blockIdx.z*PCO;
    const int wl = tx*RW;                                /* local output column group */
    const int wa = wo0*SW - padW;                   /* first input column of the tile */
    float acc[PCO][RW];
#pragma unroll
    for (int j = 0; j < PCO; j++)
#pragma unroll
        for (int i = 0; i < RW; i++) acc[j][i] = b ? b[co + j] : 0.0f;
    const int NG = TR*TW/4;                              /* float4 groups to stage */
    for (int ci = 0; ci < C_in; ci++) {
        const float *xv = x + (long)ci*H*W;
        for (int i = tid; i < NG; i += NT) {
            const int r = i / (TW/4), q = 4*(i - r*(TW/4));
            float4 v = make_float4(0.f,0.f,0.f,0.f);
            const int hi = (ho0 + r)*strideH;
            if (hi < H) {
                const float *xr = xv + (long)hi*W;
                const int sw = wa + q;
                if (sw+0 >= 0 && sw+0 < W) v.x = xr[sw+0];
                if (sw+1 >= 0 && sw+1 < W) v.y = xr[sw+1];
                if (sw+2 >= 0 && sw+2 < W) v.z = xr[sw+2];
                if (sw+3 >= 0 && sw+3 < W) v.w = xr[sw+3];
            }
            *(float4*)&xs[r][q] = v;
        }
        {
            const float *wv = w + (long)co*C_in*3 + (long)ci*3;   /* w[co][ci][0..2] */
            for (int i = tid; i < PCO; i += NT) {
                const float4 t = make_float4(wv[(long)i*C_in*3 + 0], wv[(long)i*C_in*3 + 1],
                                             wv[(long)i*C_in*3 + 2], 0.f);
                *(float4*)&ws[4*i] = t;
            }
        }
        __syncthreads();
        const int lb = wl*SW;                        /* local window base in xs row */
        const int NW = (RW*SW + 2)/4 + 1;
        float xv_[RW*SW + 4];
        {   /* the thread owns output row ty (blockDim.y = TR), not all TR rows */
            const int r = ty;
#pragma unroll
            for (int q = 0; q < NW; q++) {
                const float4 t = *(const float4*)&xs[r][lb + 4*q];
                xv_[4*q+0] = t.x; xv_[4*q+1] = t.y; xv_[4*q+2] = t.z; xv_[4*q+3] = t.w;
            }
#pragma unroll
            for (int j = 0; j < PCO; j++) {
                const float c0 = ws[4*j+0], c1 = ws[4*j+1], c2 = ws[4*j+2];
#pragma unroll
                for (int i = 0; i < RW; i++) {
                    const int o = i*SW;
                    acc[j][i] = fmaf(xv_[o+0], c0, acc[j][i]);
                    acc[j][i] = fmaf(xv_[o+1], c1, acc[j][i]);
                    acc[j][i] = fmaf(xv_[o+2], c2, acc[j][i]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < PCO; j++) {
        float *yp = y + (long)(co + j)*H_out*W_out;
        {
            const int ho = ho0 + ty;
            if (ho >= H_out) continue;
            float *yr = yp + (long)ho*W_out + (wo0 + wl);   /* output is dense in W_out */
#pragma unroll
            for (int i = 0; i < RW; i++)
                if (wo0 + wl + i < W_out) yr[i] = acc[j][i];
        }
    }
}

__global__ void k_conv2d(const float *x, const float *w, const float *b,
                         int C_in, int H, int W, int C_out, int kH, int kW,
                         int padH, int padW, int dilH, int dilW, int strideH, int strideW,
                         float *y, int co0)
{
    // threadIdx.x maps to WIDTH (the contiguous output dimension): with ho
    // first, a warp read x with stride W (26x over-fetch).
    int wo = blockIdx.x * blockDim.x + threadIdx.x;
    int ho = blockIdx.y * blockDim.y + threadIdx.y;
    int co = co0 + blockIdx.z;
    int H_out = (H + 2*padH - dilH*(kH-1) - 1)/strideH + 1;
    int W_out = (W + 2*padW - dilW*(kW-1) - 1)/strideW + 1;
    if (ho >= H_out || wo >= W_out) return;
    float acc = b ? b[co] : 0.0f;
    for (int ci = 0; ci < C_in; ci++) {
        const float *xv = x + (long)ci * H * W;
        const float *wv = w + ((long)co * C_in + ci) * kH * kW;
        for (int kh = 0; kh < kH; kh++) {
            int hi = ho*strideH - padH + kh*dilH;
            if (hi < 0 || hi >= H) continue;
            for (int kw = 0; kw < kW; kw++) {
                int wi = wo*strideW - padW + kw*dilW;
                if (wi < 0 || wi >= W) continue;
                acc = fmaf(xv[(long)hi*W + wi], wv[kh*kW + kw], acc);
            }
        }
    }
    y[((long)co * H_out + ho) * W_out + wo] = acc;
}

#define INORM_THREADS 256
// InstanceNorm2d(affine)+PReLU, one block per (n,c); two-pass Kahan reduction.
__global__ void k_inorm_prelu(const float *x, float *y, int C, int HW,
                              const float *w, const float *b, const float *alpha, float eps)
{
    int nc = blockIdx.x;
    const float *xs = x + (long)nc * HW;
    float *ys = y + (long)nc * HW;
    __shared__ float red[INORM_THREADS * 2];
    __shared__ float s_mean, s_inv;

    float s = 0.0f, c = 0.0f;
    for (int i = threadIdx.x; i < HW; i += INORM_THREADS) kahan_add(&s, &c, xs[i]);
    red[threadIdx.x*2] = s; red[threadIdx.x*2+1] = c;
    __syncthreads();
    for (int off = INORM_THREADS/2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            red[threadIdx.x*2]   += red[(threadIdx.x+off)*2];
            red[threadIdx.x*2+1] += red[(threadIdx.x+off)*2+1];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) s_mean = (red[0] + red[1]) / HW;
    __syncthreads();
    float mean = s_mean;

    float q = 0.0f, cq = 0.0f;
    for (int i = threadIdx.x; i < HW; i += INORM_THREADS) {
        float d = xs[i] - mean;
        kahan_add(&q, &cq, d*d);
    }
    red[threadIdx.x*2] = q; red[threadIdx.x*2+1] = cq;
    __syncthreads();
    for (int off = INORM_THREADS/2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            red[threadIdx.x*2]   += red[(threadIdx.x+off)*2];
            red[threadIdx.x*2+1] += red[(threadIdx.x+off)*2+1];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) s_inv = rsqrtf(fmaxf((red[0] + red[1]) / HW, 0.0f) + eps);
    __syncthreads();
    float inv = s_inv;
    float ww = w ? w[nc % C] : 1.0f;
    float bb = b ? b[nc % C] : 0.0f;
    float aa = alpha ? alpha[nc % C] : 0.25f;
    for (int i = threadIdx.x; i < HW; i += INORM_THREADS) {
        float v = (xs[i] - mean) * inv * ww + bb;
        ys[i] = v < 0.0f ? aa * v : v;
    }
}

// xz = x @ W_in^T, split into x0 / z; rev=1 reads x reversed along L.
__global__ void k_in_proj(const float *x, const float *W, float *x0, float *z,
                          int S, int L, int rev)
{
    long st = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int o = blockIdx.y;
    if (st >= (long)S * L) return;
    int s = (int)(st / L), t = (int)(st % L);
    const float *xi = x + ((long)s * L + (rev ? L - 1 - t : t)) * HID;
    const float *wr = W + (long)o * HID;
    float acc = 0.0f;
    for (int c = 0; c < HID; c++) acc = fmaf(xi[c], wr[c], acc);
    if (o < DINNER) x0[st * DINNER + o] = acc;
    else            z [st * DINNER + (o - DINNER)] = acc;
}

// causal depthwise conv1d (k=4) + SiLU along L
// in_proj with a 64x64 tile and 16 accumulators per thread.
//
// k_in_proj_t tiles 32x64 with 8 outputs/thread: 6 shared loads per 8 FMAs,
// i.e. 1.33 FMA per shared load, which is the shared-memory ceiling (~32
// floats/clk/SM) and pins the kernel at 424 GFLOPS - 7% of peak for a 1.36
// GFlop GEMM launched 60 times. A 64x64 tile with a 4x4 accumulator gives 16
// outputs from 8 shared loads per k (2 FMA/load), and chunking K=64 into two
// 32-wide tiles keeps shared at 17.4 KB so 3 blocks stay resident per SM.
__global__ void k_in_proj_r(const float *x, const float *W, float *x0, float *z,
                            int S, int L, int rev)
{
    const int KC = 32, MT = 64, NT = 64;
    __shared__ float Xs[MT][KC+4], Ws[NT][KC+4];
    const long SL = (long)S * L;
    const long m0 = (long)blockIdx.x * MT;
    const int n0 = blockIdx.y * NT;
    const int tx = threadIdx.x, ty = threadIdx.y;                 // (16,16)
    const int tid = ty*16 + tx;
    float acc[4][4];
#pragma unroll
    for (int a = 0; a < 4; a++)
#pragma unroll
        for (int b = 0; b < 4; b++) acc[a][b] = 0.f;
    for (int k0 = 0; k0 < HID; k0 += KC) {
        for (int i = tid; i < MT*KC; i += 256) {
            int r = i / KC, c = i % KC;
            long m = m0 + r;
            int sm = (int)(m / L), tm = (int)(m % L);
            if (m >= SL) { Xs[r][c] = 0.f; continue; }
            Xs[r][c] = x[((long)sm*L + (rev ? L-1-tm : tm))*HID + k0 + c];
        }
        {
            const int RS = (KC+4)/4;     // padded row stride in float4 units
            for (int i = tid; i < NT*RS; i += 256) {
                const int r = i / RS, q = 4*(i - r*RS);
                if (q >= KC) continue;   // the 4 pad columns stay untouched
                // HID (256) and n0 (x64) make this 16B aligned by construction
                ((float4*)&Ws[0][0])[i] = *(const float4*)(W + (long)(n0+r)*HID + k0 + q);
            }
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < KC; k++) {
            float a[4], w[4];
#pragma unroll
            for (int a4 = 0; a4 < 4; a4++) a[a4] = Xs[ty*4 + a4][k];
#pragma unroll
            for (int b4 = 0; b4 < 4; b4++) w[b4] = Ws[tx*4 + b4][k];
#pragma unroll
            for (int a4 = 0; a4 < 4; a4++)
#pragma unroll
                for (int b4 = 0; b4 < 4; b4++) acc[a4][b4] = fmaf(a[a4], w[b4], acc[a4][b4]);
        }
        __syncthreads();
    }
    float *dst = (n0 >= DINNER) ? z : x0;
    const int ob = (n0 >= DINNER) ? n0 - DINNER : n0;
#pragma unroll
    for (int a4 = 0; a4 < 4; a4++) {
        long m = m0 + ty*4 + a4;
        if (m >= SL) continue;
#pragma unroll
        for (int b4 = 0; b4 < 4; b4++) dst[m*DINNER + ob + tx*4 + b4] = acc[a4][b4];
    }
}


// in_proj fused with the causal depthwise conv, for one 80x64 tile.
//
// in_proj writes x0 (SL*DINNER floats) and the very next kernel reads it back,
// 42 MB per launch x 60 launches per forward, and x0 is used by nothing else.
// A thread here owns 4 consecutive rows of the tile, so the conv's 3-tap
// history (DCONV=4) is in its own registers for taps inside the tile, comes
// from the ty-1 lane group with a shuffle, and for the tile's first row comes
// from a 3-row halo computed redundantly by the same block. x0 never reaches
// global memory at all, and one launch replaces two.
//
// The GEMM keeps k_in_proj_r's shape (64 columns per block, 4x4 accumulators,
// K=64 in two 32-wide chunks, 2 FMA per shared load). The z half has no conv,
// so it just writes z as before.
#define FUSE_MT 80
#ifndef LBV
#define LBV 3
#endif
template<int KC, int LBMIN = 2>
__global__ void __launch_bounds__(256, LBMIN) k_in_conv_r(const float *x, const float *W, float *z, float *xc,
                            const float *Wc, const float *Bc,
                            int S, int L, int rev, int zhalo = 1)
{
    // Tile: 80 rows (L index) x 64 channels, 256 threads as (16,16).
    // tx owns 5 consecutive rows, ty owns 4 channels -> 20 accumulators and
    // 9 shared loads per k (2.2 FMA/load). Putting the row axis on tx is what
    // makes the conv's 3-tap history a warp-local __shfl_up(1): with the row
    // axis on ty, "the group above" for even ty sits in the previous warp and
    // shfl_up(16) silently returns the caller's own value (rel-rms 0.57).
    const int NT = 64, R = FUSE_MT + 3, RA = 5;
    __shared__ __align__(16) float Xs[R][KC+4], Ws[NT][KC+4], Hl[3][NT+1];
    const long SL = (long)S * L;
    const long m0 = (long)blockIdx.x * FUSE_MT;
    const int n0 = blockIdx.y * NT;
    const int tx = threadIdx.x, ty = threadIdx.y;                 // (16,16)
    const int tid = ty*16 + tx;
    const bool halo_thr = (tid < 3*NT) && (zhalo || n0 < DINNER);
    // Rows 0..2 of the A-tile are the conv's 3-row halo, consumed only through
    // Hl, i.e. only by the x0 half. Loading them costs 27 float4 global loads +
    // 27 shared stores per block (each with the m/L divide), so the z half
    // skips them. Only blockIdx.x == 0 can have m < 0, hence mlo_ok.
    const int rlo = (zhalo || n0 < DINNER) ? 0 : 3;
    const bool mlo_ok = (m0 >= 3);
    const int hr = tid / NT, hc = tid % NT;
    // NOTE: hoisting this row->(sequence,position) divide out of the tile-load
    // loop (rows are consecutive, so one divide + a walk suffices) measured 3%
    // on the model, and moved the model's complex rel-rms vs PyTorch from
    // <1e-4 to exactly 1.8e-4, i.e. it FAILS the accuracy gate, purely through
    // register allocation / FMA contraction. The model runs with almost no
    // fp32 margin against the 1e-4 gate, so do not "optimise" this divide back
    // in without re-running the gates.

    float acc[RA][4];
#pragma unroll
    for (int a = 0; a < RA; a++)
#pragma unroll
        for (int b = 0; b < 4; b++) acc[a][b] = 0.f;
    float hacc = 0.f;
    for (int k0 = 0; k0 < HID; k0 += KC) {
        // One index computation (incl. the 64-bit m/L divide) per 4 elements;
        // the divide itself is untouched, only computed 4x less often (see the
        // note above on not touching its scheduling). RS is in float4 units of
        // the PADDED row stride (KC+4), not KC: indexing by KC/4 writes rows
        // compactly while the GEMM reads them at the padded stride, i.e. reads
        // uninitialised shared memory (measured: all-NaN model output).
        {
            const int RS = (KC+4)/4;
            for (int i = tid; i < R*RS; i += 256) {
                const int r = i / RS, q = 4*(i - r*RS);
                if (r < rlo) continue;
                const long m = m0 - 3 + r;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (q < KC && m < SL && (mlo_ok || m >= 0)) {
                    const int sm = (int)(m / L), tm = (int)(m % L);
                    v = *(const float4*)(x + ((long)sm*L + (rev ? L-1-tm : tm))*HID + k0 + q);
                }
                ((float4*)&Xs[0][0])[i] = v;
            }
        }
        {
            const int RS = (KC+4)/4;     // padded row stride in float4 units
            for (int i = tid; i < NT*RS; i += 256) {
                const int r = i / RS, q = 4*(i - r*RS);
                if (q >= KC) continue;   // the 4 pad columns stay untouched
                // HID (256) and n0 (x64) make this 16B aligned by construction
                ((float4*)&Ws[0][0])[i] = *(const float4*)(W + (long)(n0+r)*HID + k0 + q);
            }
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < KC; k++) {
            float a[RA], w[4];
#pragma unroll
            for (int a4 = 0; a4 < RA; a4++) a[a4] = Xs[tx*RA + 3 + a4][k];
#pragma unroll
            for (int b4 = 0; b4 < 4; b4++) w[b4] = Ws[ty*4 + b4][k];
#pragma unroll
            for (int a4 = 0; a4 < RA; a4++)
#pragma unroll
                for (int b4 = 0; b4 < 4; b4++) acc[a4][b4] = fmaf(a[a4], w[b4], acc[a4][b4]);
            if (halo_thr) hacc = fmaf(Xs[hr][k], Ws[hc][k], hacc);
        }
        __syncthreads();
    }
    if (halo_thr) Hl[hr][hc] = hacc;
    if (n0 < DINNER) __syncthreads();   // Hl readers only
    if (n0 >= DINNER) {                                             // z half: plain GEMM
#pragma unroll
        for (int a4 = 0; a4 < RA; a4++) {
            long m = m0 + tx*RA + a4;
            if (m >= SL) continue;
            *(float4*)(z + m*DINNER + (n0 - DINNER) + ty*4) =
                make_float4(acc[a4][0], acc[a4][1], acc[a4][2], acc[a4][3]);
        }
        return;
    }
    // Taps for output row m = m0+tx*RA+a4 are rows m-3..m = tile rows
    // (tx*RA-3+a4) .. ; j = a4+k indexes them from m-3. j<3 lives in tx-1's
    // rows 2,3,4 (one lane away, same warp); j>=3 is this thread's acc[j-3].
    const unsigned FULL = 0xffffffffu;
    float v[3][4];
#pragma unroll
    for (int jj = 0; jj < 3; jj++)
#pragma unroll
        for (int b4 = 0; b4 < 4; b4++) v[jj][b4] = __shfl_up_sync(FULL, acc[jj+2][b4], 1);
    if (tx == 0) {
#pragma unroll
        for (int h = 0; h < 3; h++)
#pragma unroll
            for (int b4 = 0; b4 < 4; b4++) v[h][b4] = Hl[h][ty*4 + b4];
    }
    const int c0 = n0 + ty*4;
#pragma unroll
    for (int a4 = 0; a4 < RA; a4++) {
        long m = m0 + tx*RA + a4;
        if (m >= SL) continue;
        const int t = (int)(m % L);
        const int ntap = (t >= DCONV-1) ? DCONV : t + 1;
        float fv[4];
#pragma unroll
        for (int b4 = 0; b4 < 4; b4++) fv[b4] = Bc[c0 + b4];
        // Fully unrolled with a predicate, never a runtime trip count: with
        // "k = DCONV - ntap" as the loop start, j = a4+k is a runtime index and
        // v[]/acc[] spill to local memory (128-176 byte stack frames).
#pragma unroll
        for (int k = 0; k < DCONV; k++) {
            if (k < DCONV - ntap) continue;
            const int j = a4 + k;
#pragma unroll
            for (int b4 = 0; b4 < 4; b4++)
                fv[b4] = fmaf((j < 3 ? v[j][b4] : acc[j-3][b4]),
                              Wc[(long)(c0 + b4)*DCONV + k], fv[b4]);
        }
        *(float4*)(xc + m*DINNER + c0) = make_float4(dsilu(fv[0]), dsilu(fv[1]), dsilu(fv[2]), dsilu(fv[3]));
    }
}

__global__ void k_in_proj_t(const float *x, const float *W, float *x0, float *z,
                            int S, int L, int rev)
{
    // x0/z[m][o] = sum_c x[perm(m)][c] * W[o][c];  K = HID = 64, N = 2*DINNER.
    // The naive version assigned one (row, column) per thread, so every output
    // column re-read the whole input row: 2.7 GB of reads for 42 MB of output
    // (37 GFLOPS). Tiling 32 rows x 64 columns into shared memory and giving
    // each thread 8 outputs (2x4) reads the tile once per 8 outputs.
    const int MT = 32, NT = 64;
    __shared__ float Xs[MT][HID+4];
    __shared__ float Ws[NT][HID+4];
    const int m0 = blockIdx.x * MT, n0 = blockIdx.y * NT;
    const int tx = threadIdx.x, ty = threadIdx.y;              // (16,16) = 256
    const int tid = ty * 16 + tx;
    const long SL = (long)S * L;
    for (int i = tid; i < MT*HID; i += 256) {
        int r = i / HID, c = i % HID;
        int m = m0 + r;
        if (m >= SL) { Xs[r][c] = 0.f; continue; }
        int sm = m / L, tm = m % L;
        Xs[r][c] = x[((long)sm*L + (rev ? L-1-tm : tm))*HID + c];
    }
    for (int i = tid; i < NT*HID; i += 256) {
        int r = i / HID, c = i % HID;
        Ws[r][c] = W[(long)(n0+r)*HID + c];
    }
    __syncthreads();
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.f;
#pragma unroll
    for (int k = 0; k < HID; k++) {
        float a0 = Xs[ty*2][k], a1 = Xs[ty*2+1][k];
        float b0 = Ws[tx*4][k], b1 = Ws[tx*4+1][k], b2 = Ws[tx*4+2][k], b3 = Ws[tx*4+3][k];
        acc[0] += a0*b0; acc[1] += a1*b0; acc[2] += a0*b1; acc[3] += a1*b1;
        acc[4] += a0*b2; acc[5] += a1*b2; acc[6] += a0*b3; acc[7] += a1*b3;
    }
    float *dst = (n0 >= DINNER) ? z : x0;
    const int ob = (n0 >= DINNER) ? n0 - DINNER : n0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        int m = m0 + ty*2 + (i & 1), c = tx*4 + (i >> 1);
        if (m < SL) dst[(long)m*DINNER + ob + c] = acc[i];
    }
}

__global__ void k_conv_causal(const float *x0, const float *Wc, const float *Bc,
                              float *xc, int S, int L)
{
    long st = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y;
    if (st >= (long)S * L) return;
    int t = (int)(st % L);
    const float *wc = Wc + (long)c * DCONV;
    float acc = Bc[c];
    for (int k = 0; k < DCONV; k++) {
        int tt = t - (DCONV - 1) + k;
        if (tt < 0) continue;
        acc = fmaf(x0[(st - (DCONV - 1) + k) * DINNER + c], wc[k], acc);
    }
    xc[st * DINNER + c] = dsilu(acc);
}

__global__ void k_x_proj(const float *xc, const float *W, float *xdbl, long SL)
{
    long st = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int o = blockIdx.y;
    if (st >= SL) return;
    const float *xi = xc + st * DINNER;
    const float *wr = W + (long)o * DINNER;
    float acc = 0.0f;
    for (int c = 0; c < DINNER; c++) acc = fmaf(xi[c], wr[c], acc);
    xdbl[st * XP + o] = acc;
}

// x_proj as a 128-row x 32-col register-tiled GEMM.
//
// k_x_proj_t (MT=64, 4 rows x 2 cols = 8 accumulators) issued 6 shared loads
// per 8 FMAs and its tile loaders did per-element index math (2048 + 1024
// scalar loads per block per K chunk). This version: MT=128 with 8x2 = 16
// accumulators (10 loads per 16 FMAs) and float4 tile loads, one index/mask
// computation per 4 elements. RS is in float4 units of the PADDED row stride.
__global__ void k_x_proj_r(const float *xc, const float *W, float *xdbl, long SL)
{
    const int MT = 128, NT = 32, KT = 32, RS = (KT+4)/4;
    __shared__ __align__(16) float As[MT][KT+4];
    __shared__ __align__(16) float Bs[NT][KT+4];
    const int m0 = blockIdx.y * MT, n0 = blockIdx.x * NT;
    const int tx = threadIdx.x, ty = threadIdx.y;                 // (16,16)
    const int tid = ty * 16 + tx;
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; i++) acc[i] = 0.f;
    for (int k0 = 0; k0 < DINNER; k0 += KT) {
        for (int i = tid; i < MT*RS; i += 256) {
            const int r = i / RS, q = 4*(i - r*RS);
            const int m = m0 + r;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (q < KT && m < SL) v = *(const float4*)(xc + (long)m*DINNER + k0 + q);
            ((float4*)&As[0][0])[i] = v;
        }
        for (int i = tid; i < NT*RS; i += 256) {
            const int r = i / RS, q = 4*(i - r*RS);
            const int n = n0 + r;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (q < KT && n < XP) v = *(const float4*)(W + (long)n*DINNER + k0 + q);
            ((float4*)&Bs[0][0])[i] = v;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < KT; k++) {
            const float b0 = Bs[tx*2][k], b1 = Bs[tx*2+1][k];
#pragma unroll
            for (int r = 0; r < 8; r++) {
                const float a = As[ty*8 + r][k];
                acc[r*2] = fmaf(a, b0, acc[r*2]);
                acc[r*2+1] = fmaf(a, b1, acc[r*2+1]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 16; i++) {
        int m = m0 + ty*8 + (i >> 1), n = n0 + tx*2 + (i & 1);
        if (m < SL && n < XP) xdbl[(long)m*XP + n] = acc[i];
    }
}

/* xdbl[m][o] = sum_k xc[m][k]*W[o][k], K = DINNER = 256, N = XP = 36, M = SL rows.
 * Templated tile: block (TX, 256/TX), each thread owns RA = MT/(256/TX) rows and
 * CB = NT/TX columns, so FMA per shared load = RA*CB/(RA+CB). The shipped shape
 * (128,32,(16,16)) = 8 rows x 2 cols = 16 accumulators = 1.6 FMA/load and 2 blocks
 * in x for XP=36 (the second tile has 4 useful columns of 32). NT=40 covers XP in
 * one tile with 10% padding; more accumulators is the lever that moved every other
 * GEMM here. K order is unchanged (k0 chunks of 32, k ascending), so results are
 * bit-identical. REUSE_XPROJ=MT,NT,TX selects the shape; 2 = the shipped one. */
template<int MT, int NT, int TX>
__global__ void k_x_proj_g(const float *xc, const float *W, float *xdbl, long SL)
{
    const int KT = 32, RS = (KT+4)/4, TY = 256/TX, RA = MT/TY, CB = NT/TX;
    static_assert(NT % TX == 0 && MT % (256/TX) == 0 && 256 % TX == 0, "tile must divide the block");
    __shared__ __align__(16) float As[MT][KT+4];
    __shared__ __align__(16) float Bs[NT][KT+4];
    const int m0 = blockIdx.y * MT, n0 = blockIdx.x * NT;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty*TX + tx;
    float acc[RA*CB];
#pragma unroll
    for (int i = 0; i < RA*CB; i++) acc[i] = 0.f;
    for (int k0 = 0; k0 < DINNER; k0 += KT) {
        for (int i = tid; i < MT*RS; i += 256) {
            const int r = i / RS, q = 4*(i - r*RS);
            const int m = m0 + r;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (q < KT && m < SL) v = *(const float4*)(xc + (long)m*DINNER + k0 + q);
            ((float4*)&As[0][0])[i] = v;
        }
        for (int i = tid; i < NT*RS; i += 256) {
            const int r = i / RS, q = 4*(i - r*RS);
            const int n = n0 + r;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (q < KT && n < XP) v = *(const float4*)(W + (long)n*DINNER + k0 + q);
            ((float4*)&Bs[0][0])[i] = v;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < KT; k++) {
            float bv[CB];
#pragma unroll
            for (int c = 0; c < CB; c++) bv[c] = Bs[tx*CB + c][k];
#pragma unroll
            for (int r = 0; r < RA; r++) {
                const float a = As[ty*RA + r][k];
#pragma unroll
                for (int c = 0; c < CB; c++) acc[r*CB + c] = fmaf(a, bv[c], acc[r*CB + c]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < RA*CB; i++) {
        const int r = i / CB, c = i % CB;
        int m = m0 + ty*RA + r, n = n0 + tx*CB + c;
        if (m < SL && n < XP) xdbl[(long)m*XP + n] = acc[i];
    }
}

__global__ void k_x_proj_t(const float *xc, const float *W, float *xdbl, long SL)
{
    // xdbl[m][o] = sum_k xc[m][k] * W[o][k];  K = DINNER = 256, N = XP = 36.
    // Same naive-GEMV problem as in_proj: 756 MB of reads for 3 MB of output.
    const int MT = 64, NT = 32, KT = 32;
    __shared__ float As[MT][KT+4];
    __shared__ float Bs[NT][KT+4];
    const int m0 = blockIdx.y * MT, n0 = blockIdx.x * NT;
    const int tx = threadIdx.x, ty = threadIdx.y;                 // (16,16)
    const int tid = ty * 16 + tx;
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.f;
    for (int k0 = 0; k0 < DINNER; k0 += KT) {
        for (int i = tid; i < MT*KT; i += 256) {
            int r = i / KT, c = i % KT, m = m0 + r;
            As[r][c] = (m < SL) ? xc[(long)m*DINNER + k0 + c] : 0.f;
        }
        for (int i = tid; i < NT*KT; i += 256) {
            int r = i / KT, c = i % KT, n = n0 + r;
            Bs[r][c] = (n < XP) ? W[(long)n*DINNER + k0 + c] : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < KT; k++) {
            float b0 = Bs[tx*2][k], b1 = Bs[tx*2+1][k];
#pragma unroll
            for (int r = 0; r < 4; r++) {
                float a = As[ty*4 + r][k];
                acc[r*2] += a*b0; acc[r*2+1] += a*b1;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        int m = m0 + ty*4 + (i >> 1), n = n0 + tx*2 + (i & 1);
        if (m < SL && n < XP) xdbl[(long)m*XP + n] = acc[i];
    }
}

__global__ void k_dt_proj(const float *xdbl, const float *W, const float *B,
                          float *dtt, long SL)
{
    long st = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y;
    if (st >= SL) return;
    const float *di = xdbl + st * XP;
    const float *wr = W + (long)c * DTRANK;
    float acc = B[c];
    for (int r = 0; r < DTRANK; r++) acc = fmaf(di[r], wr[r], acc);
    dtt[st * DINNER + c] = dsoftplus(acc);
}

// selective scan: one thread per (s, channel), 16 states in registers.
__global__ void k_conv_causal_t(const float *x0, const float *Wc, const float *Bc,
                                float *xc, int S, int L)
{
    // Coalesced sibling of k_conv_causal. The naive kernel assigns each thread a
    // different ROW for a single channel, so a warp's reads are 32 floats spaced
    // DINNER apart (26x over-fetch). Here a thread owns 8 CONSECUTIVE channels of
    // one row, and all 32 lanes of a warp share that row: every access is 128 B
    // contiguous.
    long SL = (long)S * L;
    int m = (int)((long)blockIdx.x * 8 + threadIdx.x / 32);
    int c0 = (threadIdx.x % 32) * 8;
    if (m >= SL) return;
    int t = m % L;
    int ntap = (t >= DCONV - 1) ? DCONV : t + 1;          // causal zero-pad
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; j++) acc[j] = Bc[c0 + j];
    for (int k = DCONV - ntap; k < DCONV; k++) {
        const float *p = x0 + ((long)m - (DCONV - 1) + k) * DINNER + c0;
        const float *w = Wc + (long)c0 * DCONV + k;
#pragma unroll
        for (int j = 0; j < 8; j++) acc[j] = fmaf(p[j], w[j*DCONV], acc[j]);
    }
    float *o = xc + (long)m * DINNER + c0;
#pragma unroll
    for (int j = 0; j < 8; j++) o[j] = dsilu(acc[j]);
}

__global__ void k_dt_proj_t(const float *xdbl, const float *W, const float *B,
                            float *dtt, long SL)
{
    // Same fix as k_conv_causal_t: 8 consecutive channels per thread, warp-coherent.
    int m = (int)((long)blockIdx.x * 8 + threadIdx.x / 32);
    int c0 = (threadIdx.x % 32) * 8;
    if (m >= SL) return;
    const float *di = xdbl + (long)m * XP;
    float d0 = di[0], d1 = di[1], d2 = di[2], d3 = di[3];
    const float *wr = W + (long)c0 * DTRANK;
    const float *bb = B + c0;
    float *o = dtt + (long)m * DINNER + c0;
#pragma unroll
    for (int j = 0; j < 8; j++)
        o[j] = dsoftplus(bb[j] + d0*wr[j*DTRANK] + d1*wr[j*DTRANK+1]
                                    + d2*wr[j*DTRANK+2] + d3*wr[j*DTRANK+3]);
}

// dt_proj with the 4 KB weight matrix staged in shared memory.
//
// dt_proj is M=SL, N=DINNER, K=DTRANK=4: a near-elementwise kernel whose only
// real cost is writing SL*DINNER floats. k_dt_proj_t reads W[c][p] with 8
// channels per thread, so for a fixed tap the warp's 32 lanes are 16 floats
// apart: 32 sectors per warp load, 8x L1 sector amplification. Measured in
// isolation: 0.0236 s write-only, 0.0541 s with the FMA loop (and 0.0541 s
// with softplus removed, so the transcedentals cost nothing). Staging W in
// shared (loaded coalesced once per block) and reading it as float4 per lane
// makes each warp read 512 B contiguous, conflict free.
__global__ void k_dt_proj_r(const float *xdbl, const float *W, const float *B,
                            float *dtt, long SL)
{
    __shared__ float4 Ws[DINNER];
    const int m = (int)((long)blockIdx.x * 8 + threadIdx.x / 32);
    const int c0 = (threadIdx.x % 32) * 8;
    // every thread stages its part of W before any row work
    for (int i = threadIdx.x; i < DINNER; i += 256)
        Ws[i] = ((const float4*)W)[i];
    __syncthreads();
    if (m >= SL) return;
    const float *di = xdbl + (long)m * XP;
    const float d0 = di[0], d1 = di[1], d2 = di[2], d3 = di[3];
    const float *bb = B + c0;
    float *o = dtt + (long)m * DINNER + c0;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const float4 w = Ws[c0 + j];
        o[j] = dsoftplus(bb[j] + d0*w.x + d1*w.y + d2*w.z + d3*w.w);
    }
}

__global__ void k_scan_b(const float *xc, const float *dtt, const float *A_log,
                         const float *Dv, const float *xdbl, float *y, int S, int L, int rev)
{
    // Same serial L-scan as k_scan, but 256-thread blocks that cover 8 scan
    // rows: the 32-thread version gives the whole tensor only 648 warps and
    // runs 3x off this shape. One channel per thread (more channels per thread
    // costs more in registers than it saves: measured CPZ=2 0.00055 s,
    // CPZ=4 0.00073 s vs 0.00044 s here).
    int s = (int)((long)blockIdx.x * 8 + threadIdx.x / 32);
    int c = (threadIdx.x % 32) + blockIdx.y * 32;
    if (s >= S || c >= DINNER) return;
    float h[DSTATE];
#pragma unroll
    for (int n = 0; n < DSTATE; n++) h[n] = 0.0f;
    float aA[DSTATE];
#pragma unroll
    for (int n = 0; n < DSTATE; n++) aA[n] = -__expf(A_log[c * DSTATE + n]);
    const float Dc = Dv[c];
    for (int ii = 0; ii < L; ii++) {
        long sl = (long)s * L + ii;
        float d = dtt[sl * DINNER + c];
        float u = xc[sl * DINNER + c];
        const float *Bt = xdbl + sl * XP + DTRANK;
        const float *Ct = Bt + DSTATE;
        float acc = 0.0f;
#pragma unroll
        for (int n = 0; n < DSTATE; n++) {
            h[n] = __expf(d * aA[n]) * h[n] + d * Bt[n] * u;
            acc = fmaf(h[n], Ct[n], acc);
        }
        y[sl * DINNER + c] = acc + u * Dc;
    }
}

__global__ void k_scan(const float *xc, const float *dtt, const float *A_log,
                       const float *Dv, const float *xdbl, float *y, int S, int L, int rev)
{
    int s = blockIdx.x;
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= DINNER) return;
    float h[DSTATE];
#pragma unroll
    for (int n = 0; n < DSTATE; n++) h[n] = 0.0f;
    float aA[DSTATE];
#pragma unroll
    for (int n = 0; n < DSTATE; n++) aA[n] = -__expf(A_log[c * DSTATE + n]);
    const float Dc = Dv[c];
    for (int ii = 0; ii < L; ii++) {
        long sl = (long)s * L + ii;
        float d = dtt[sl * DINNER + c];
        float u = xc[sl * DINNER + c];
        const float *Bt = xdbl + sl * XP + DTRANK;
        const float *Ct = Bt + DSTATE;
        float acc = 0.0f;
#pragma unroll
        for (int n = 0; n < DSTATE; n++) {
            h[n] = __expf(d * aA[n]) * h[n] + d * Bt[n] * u;
            acc = fmaf(h[n], Ct[n], acc);
        }
        // Write on the scan axis (reversed for rev=1); k_concat un-flips the
        // backward direction once, so do not un-flip here.
        y[sl * DINNER + c] = acc + u * Dc;
    }
}

__global__ void k_gate_outproj(const float *y, const float *z, const float *W,
                               float *out, long SL)
{
    long st = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int o = blockIdx.y;
    if (st >= SL) return;
    const float *yi = y + st * DINNER;
    const float *zi = z + st * DINNER;
    const float *wr = W + (long)o * DINNER;
    float acc = 0.0f;
    for (int c = 0; c < DINNER; c++) acc = fmaf(yi[c] * dsilu(zi[c]), wr[c], acc);
    out[st * HID + o] = acc;
}


// The gated output projection as a 32x64 register-tiled GEMM.
// k_out_proj_tiled uses GT=32 tiles with 4 accumulators per thread (6 shared
// loads per 8 FMAs, i.e. at the shared-memory issue ceiling) and scalar tile
// loads. This variant owns 64 rows x all HID=64 outputs per block with a 4x4
// accumulator per thread (8 loads per 16 FMAs), stages W in shared with
// float4 loads, and writes the result back as float4.
// 8x8 register tile: 64 accumulators per thread, so the k-loop issues 16 shared
// loads per 64 FMAs (4 FMA/load) against k_out_proj_r's 8 loads per 16 FMAs.
// The weight tile is stored K-major (WsT[k][co]) so a warp's 8 column groups read
// 32 contiguous floats; with the co-major layout the 8 lanes sit 8 rows apart and
// 8*row_stride is always even, so no padding could have made that read
// conflict-free. Tile is 256 rows x 64 cols, blockDim (8,32), K chunked by 16.
__global__ void k_out_proj_r8(const float *y, const float *z, const float *W,
                              float *out, int SL)
{
    const int KC = 16, MT = 256;
    __shared__ __align__(16) float As[MT][KC+4];
    __shared__ __align__(16) float WsT[KC][HID+4];
    const long m0 = (long)blockIdx.x * MT;
    const int tx = threadIdx.x, ty = threadIdx.y;                 // (8,32)
    const int tid = ty*8 + tx;
    float acc[8][8];
#pragma unroll
    for (int a = 0; a < 8; a++)
#pragma unroll
        for (int b = 0; b < 8; b++) acc[a][b] = 0.f;
    for (int k0 = 0; k0 < DINNER; k0 += KC) {
        for (int i = tid; i < MT*(KC/4); i += 256) {              /* 4 float4/thread */
            const int r = i / (KC/4), q = 4*(i - r*(KC/4));
            const long o_ = (m0 + r)*DINNER + k0 + q;
            float4 v = make_float4(0.f,0.f,0.f,0.f);
            if (m0 + r < SL) {
                const float4 yy = *(const float4*)(y + o_);
                const float4 zz = *(const float4*)(z + o_);
                v.x = yy.x * dsilu(zz.x); v.y = yy.y * dsilu(zz.y);
                v.z = yy.z * dsilu(zz.z); v.w = yy.w * dsilu(zz.w);
            }
            *(float4*)&As[r][q] = v;
        }
        for (int i = tid; i < KC*(HID/4); i += 256) {             /* transpose W tile */
            const int co = i / (KC/4), q = 4*(i - co*(KC/4));
            const float4 wv = *(const float4*)(W + (long)co*DINNER + k0 + q);
            WsT[q+0][co] = wv.x; WsT[q+1][co] = wv.y;
            WsT[q+2][co] = wv.z; WsT[q+3][co] = wv.w;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < KC; k++) {
            float a[8], w[8];
#pragma unroll
            for (int a4 = 0; a4 < 8; a4++) a[a4] = As[ty*8 + a4][k];
#pragma unroll
            for (int b4 = 0; b4 < 8; b4++) w[b4] = WsT[k][tx*8 + b4];
#pragma unroll
            for (int a4 = 0; a4 < 8; a4++)
#pragma unroll
                for (int b4 = 0; b4 < 8; b4++) acc[a4][b4] = fmaf(a[a4], w[b4], acc[a4][b4]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int a4 = 0; a4 < 8; a4++) {
        const long m = m0 + ty*8 + a4;
        if (m >= SL) continue;
        *(float4*)(out + m*HID + tx*8)     = make_float4(acc[a4][0], acc[a4][1], acc[a4][2], acc[a4][3]);
        *(float4*)(out + m*HID + tx*8 + 4) = make_float4(acc[a4][4], acc[a4][5], acc[a4][6], acc[a4][7]);
    }
}
// k_out_proj_r8 with k_concat_f fused into the epilogue: the gated projection
// writes its half of cat directly, with the block input x added (the residual the
// concat used to apply). Saves writing fw/bw and reading them back: 10.6 MB per
// TFMamba block and 30 launches per forward. Same adds, so bit-identical.
__global__ void k_out_proj_r8c(const float *y, const float *z, const float *W,
                              const float *xres, float *cat, int coff, int L, int rev, int SL)
{
    const int KC = 16, MT = 256;
    __shared__ __align__(16) float As[MT][KC+4];
    __shared__ __align__(16) float WsT[KC][HID+4];
    const long m0 = (long)blockIdx.x * MT;
    const int tx = threadIdx.x, ty = threadIdx.y;                 // (8,32)
    const int tid = ty*8 + tx;
    float acc[8][8];
#pragma unroll
    for (int a = 0; a < 8; a++)
#pragma unroll
        for (int b = 0; b < 8; b++) acc[a][b] = 0.f;
    for (int k0 = 0; k0 < DINNER; k0 += KC) {
        for (int i = tid; i < MT*(KC/4); i += 256) {              /* 4 float4/thread */
            const int r = i / (KC/4), q = 4*(i - r*(KC/4));
            const long o_ = (m0 + r)*DINNER + k0 + q;
            float4 v = make_float4(0.f,0.f,0.f,0.f);
            if (m0 + r < SL) {
                const float4 yy = *(const float4*)(y + o_);
                const float4 zz = *(const float4*)(z + o_);
                v.x = yy.x * dsilu(zz.x); v.y = yy.y * dsilu(zz.y);
                v.z = yy.z * dsilu(zz.z); v.w = yy.w * dsilu(zz.w);
            }
            *(float4*)&As[r][q] = v;
        }
        for (int i = tid; i < KC*(HID/4); i += 256) {             /* transpose W tile */
            const int co = i / (KC/4), q = 4*(i - co*(KC/4));
            const float4 wv = *(const float4*)(W + (long)co*DINNER + k0 + q);
            WsT[q+0][co] = wv.x; WsT[q+1][co] = wv.y;
            WsT[q+2][co] = wv.z; WsT[q+3][co] = wv.w;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < KC; k++) {
            float a[8], w[8];
#pragma unroll
            for (int a4 = 0; a4 < 8; a4++) a[a4] = As[ty*8 + a4][k];
#pragma unroll
            for (int b4 = 0; b4 < 8; b4++) w[b4] = WsT[k][tx*8 + b4];
#pragma unroll
            for (int a4 = 0; a4 < 8; a4++)
#pragma unroll
                for (int b4 = 0; b4 < 8; b4++) acc[a4][b4] = fmaf(a[a4], w[b4], acc[a4][b4]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int a4 = 0; a4 < 8; a4++) {
        const long m = m0 + ty*8 + a4;
        if (m >= SL) continue;
        /* k_concat_f's job, folded in: cat[s*L+t][0:HID] = fw + x,
           cat[s*L+t][HID:2HID] = bw[s*L+L-1-t] + x. The backward direction
           stores at the time-flipped row so cat needs no reversal pass. */
        int mr = (int)m;
        if (rev) { const int sh = mr / L, tt = mr - sh*L; mr = sh*L + (L-1-tt); }
        const float4 xa = *(const float4*)(xres + (long)mr*HID + tx*8);
        const float4 xb = *(const float4*)(xres + (long)mr*HID + tx*8 + 4);
        float *d = cat + (long)mr*(2*HID) + coff + tx*8;
        *(float4*)d = make_float4(acc[a4][0] + xa.x, acc[a4][1] + xa.y,
                                  acc[a4][2] + xa.z, acc[a4][3] + xa.w);
        *(float4*)(d+4) = make_float4(acc[a4][4] + xb.x, acc[a4][5] + xb.y,
                                      acc[a4][6] + xb.z, acc[a4][7] + xb.w);
    }
}

/* A-tile transposed through shared memory.
 *
 * k_out_proj_r8c keeps As as [m][k], so the thread's 8-value a[] group costs 8
 * scalar LDS at an 80 B row stride, while its w[] group is already vectorised
 * out of WsT. Staging shared as AsT[k][m] instead puts those 8 values
 * contiguous and 16 B aligned (MT+4 = 260 floats = 1040 B keeps every k-row
 * 16 B aligned), so the a[] group becomes 2 LDS.128: 6 fewer LSU issue slots
 * per k, ~96 per chunk of ~1300 instructions. The staging side pays 4 scalar
 * STS instead of 1 STS.128 per float4, which the coalesced global load along k
 * keeps cheap. Accumulation order is unchanged, so the output is bit-identical
 * to k_out_proj_r8c. Selected by REUSE_APT=1 (0 = shipped path). */
__global__ void k_out_proj_r8c_t(const float *y, const float *z, const float *W,
                              const float *xres, float *cat, int coff, int L, int rev, int SL)
{
    const int KC = 16, MT = 256;
    __shared__ __align__(16) float AsT[KC][MT+4];
    __shared__ __align__(16) float WsT[KC][HID+4];
    const long m0 = (long)blockIdx.x * MT;
    const int tx = threadIdx.x, ty = threadIdx.y;                 // (8,32)
    const int tid = ty*8 + tx;
    float acc[8][8];
#pragma unroll
    for (int a = 0; a < 8; a++)
#pragma unroll
        for (int b = 0; b < 8; b++) acc[a][b] = 0.f;
    for (int k0 = 0; k0 < DINNER; k0 += KC) {
        for (int i = tid; i < MT*(KC/4); i += 256) {              /* 4 float4/thread */
            const int r = i / (KC/4), q = 4*(i - r*(KC/4));
            const long o_ = (m0 + r)*DINNER + k0 + q;
            float4 v = make_float4(0.f,0.f,0.f,0.f);
            if (m0 + r < SL) {
                const float4 yy = *(const float4*)(y + o_);
                const float4 zz = *(const float4*)(z + o_);
                v.x = yy.x * dsilu(zz.x); v.y = yy.y * dsilu(zz.y);
                v.z = yy.z * dsilu(zz.z); v.w = yy.w * dsilu(zz.w);
            }
            AsT[q+0][r] = v.x; AsT[q+1][r] = v.y;
            AsT[q+2][r] = v.z; AsT[q+3][r] = v.w;
        }
        for (int i = tid; i < KC*(HID/4); i += 256) {             /* transpose W tile */
            const int co = i / (KC/4), q = 4*(i - co*(KC/4));
            const float4 wv = *(const float4*)(W + (long)co*DINNER + k0 + q);
            WsT[q+0][co] = wv.x; WsT[q+1][co] = wv.y;
            WsT[q+2][co] = wv.z; WsT[q+3][co] = wv.w;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < KC; k++) {
            float a[8], w[8];
            const float4 a_lo = *(const float4*)&AsT[k][ty*8];
            const float4 a_hi = *(const float4*)&AsT[k][ty*8 + 4];
            a[0] = a_lo.x; a[1] = a_lo.y; a[2] = a_lo.z; a[3] = a_lo.w;
            a[4] = a_hi.x; a[5] = a_hi.y; a[6] = a_hi.z; a[7] = a_hi.w;
#pragma unroll
            for (int b4 = 0; b4 < 8; b4++) w[b4] = WsT[k][tx*8 + b4];
#pragma unroll
            for (int a4 = 0; a4 < 8; a4++)
#pragma unroll
                for (int b4 = 0; b4 < 8; b4++) acc[a4][b4] = fmaf(a[a4], w[b4], acc[a4][b4]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int a4 = 0; a4 < 8; a4++) {
        const long m = m0 + ty*8 + a4;
        if (m >= SL) continue;
        int mr = (int)m;
        if (rev) { const int sh = mr / L, tt = mr - sh*L; mr = sh*L + (L-1-tt); }
        const float4 xa = *(const float4*)(xres + (long)mr*HID + tx*8);
        const float4 xb = *(const float4*)(xres + (long)mr*HID + tx*8 + 4);
        float *d = cat + (long)mr*(2*HID) + coff + tx*8;
        *(float4*)d = make_float4(acc[a4][0] + xa.x, acc[a4][1] + xa.y,
                                  acc[a4][2] + xa.z, acc[a4][3] + xa.w);
        *(float4*)(d+4) = make_float4(acc[a4][4] + xb.x, acc[a4][5] + xb.y,
                                      acc[a4][6] + xb.z, acc[a4][7] + xb.w);
    }
}

__global__ void k_out_proj_r(const float *y, const float *z, const float *W,
                             float *out, int SL)
{
    const int KC = 32, MT = 64;
    __shared__ float As[MT][KC+4], Ws[HID][KC+4];
    const long m0 = (long)blockIdx.x * MT;
    const int tx = threadIdx.x, ty = threadIdx.y;                 // (16,16)
    const int tid = ty*16 + tx;
    float acc[4][4];
#pragma unroll
    for (int a = 0; a < 4; a++)
#pragma unroll
        for (int b = 0; b < 4; b++) acc[a][b] = 0.f;
    for (int k0 = 0; k0 < DINNER; k0 += KC) {
        for (int i = tid; i < MT*KC; i += 256) {
            const int r = i / KC, c = i % KC;
            const long o_ = (m0 + r)*DINNER + k0 + c;
            As[r][c] = (m0 + r < SL) ? y[o_] * dsilu(z[o_]) : 0.f;
        }
        {
            const int RS = (KC+4)/4;
            for (int i = tid; i < HID*RS; i += 256) {
                const int r = i / RS, q = 4*(i - r*RS);
                if (q >= KC) continue;
                ((float4*)&Ws[0][0])[i] = *(const float4*)(W + (long)r*DINNER + k0 + q);
            }
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < KC; k++) {
            float a[4], w[4];
#pragma unroll
            for (int a4 = 0; a4 < 4; a4++) a[a4] = As[ty*4 + a4][k];
#pragma unroll
            for (int b4 = 0; b4 < 4; b4++) w[b4] = Ws[tx*4 + b4][k];
#pragma unroll
            for (int a4 = 0; a4 < 4; a4++)
#pragma unroll
                for (int b4 = 0; b4 < 4; b4++) acc[a4][b4] = fmaf(a[a4], w[b4], acc[a4][b4]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int a4 = 0; a4 < 4; a4++) {
        const long m = m0 + ty*4 + a4;
        if (m >= SL) continue;
        *(float4*)(out + m*HID + tx*4) = make_float4(acc[a4][0], acc[a4][1], acc[a4][2], acc[a4][3]);
    }
}

__global__ void k_out_proj_tiled(const float *y, const float *z, const float *W, float *out,
                                 int SL)
{
    // out[m][o] = sum_k ( y[m][k] * silu(z[m][k]) ) * W[o][k]      (K = DINNER, N = HID)
    __shared__ float As[GT][GT+4];      // [m][k] = gated y
    __shared__ float Bs[GT][GT+4];      // [o][k]
    const int n0 = blockIdx.x * GT, m0 = blockIdx.y * GT;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * GT/2 + tx;                    // block is (16,16) = 256 threads
    float acc[4];
#pragma unroll
    for (int i = 0; i < 4; i++) acc[i] = 0.f;
    for (int k0 = 0; k0 < DINNER; k0 += GT) {
#pragma unroll
        for (int t = tid; t < GT*GT; t += 256) {
            int r = t / GT, c = t % GT;
            int am = m0 + r, ak = k0 + c;
            if (am < SL) {
                long o_ = (long)am*DINNER + ak;
                As[r][c] = y[o_] * dsilu(z[o_]);
            } else As[r][c] = 0.f;
            int bn = n0 + r;
            Bs[r][c] = (bn < HID) ? W[(long)bn*DINNER + ak] : 0.f;
        }
        __syncthreads();
#pragma unroll 4
        for (int k = 0; k < GT; k++) {
            float a0 = As[ty*2][k], a1 = As[ty*2+1][k];
            float b0 = Bs[tx*2][k], b1 = Bs[tx*2+1][k];
            // acc[i]: m = ty*2 + (i&1), o = tx*2 + (i>>1)   (matches the epilogue)
            acc[0] += a0*b0; acc[1] += a1*b0; acc[2] += a0*b1; acc[3] += a1*b1;
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int o = n0 + tx*2 + (i >> 1), m = m0 + ty*2 + (i & 1);
        if (o < HID && m < SL) out[(long)m*HID + o] = acc[i];
    }
}

__global__ void k_concat(const float *fw, const float *bw, const float *x,
                         float *cat, int S, int L)
{
    long st = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (st >= (long)S * L) return;
    int s = (int)(st / L), t = (int)(st % L);
    long ir = (long)s * L + (L - 1 - t);
    for (int c = 0; c < HID; c++) {
        cat[st * 2*HID + c]       = fw[st * HID + c] + x[st * HID + c];
        cat[st * 2*HID + HID + c] = bw[ir * HID + c] + x[st * HID + c];
    }
}

// output_proj (512 -> 64) + LayerNorm; one block of 64 threads per row.
__global__ void k_outproj_ln(const float *cat, const float *W, const float *B,
                             const float *LW, const float *LB, float *out, long SL)
{
    long st = blockIdx.x;
    int o = threadIdx.x;
    const float *ci = cat + st * 2*HID;
    __shared__ float red[HID];
    const float *wr = W + (long)o * 2*HID;
    float v = B[o];
    for (int c = 0; c < 2*HID; c++) v = fmaf(ci[c], wr[c], v);
    red[o] = v;
    __syncthreads();
    float s1 = 0.0f, c1 = 0.0f, s2 = 0.0f, c2 = 0.0f;
    for (int k = 0; k < HID; k++) { kahan_add(&s1, &c1, red[k]); kahan_add(&s2, &c2, red[k]*red[k]); }
    float mean = (s1 + c1) / HID;
    float inv  = rsqrtf(fmaxf((s2 + c2) / HID - mean*mean, 0.0f) + 1e-5f);
    out[st * HID + o] = (v - mean) * inv * LW[o] + LB[o];
}

// concat, coalesced: a thread owns one (row, channel) pair instead of walking
// a whole row, so every stream (fw, bw, x reads and the cat write) is
// contiguous. The old one-thread-per-row version read/wrote with a 64-float
// stride, i.e. 4.4 GB/s on a ~250 GB/s device.
__global__ void k_concat_f(const float *fw, const float *bw, const float *x,
                           float *cat, int S, int L)
{
    const int row = blockIdx.x*4 + threadIdx.y;
    const int c = threadIdx.x;
    if (row >= S*L) return;
    const int s = row / L, t = row % L;
    const long ir = (long)s*L + (L-1-t);
    const long base = (long)row*HID;
    float xv = x[base + c];
    float *dst = cat + (long)row*2*HID;
    dst[c]       = fw[base + c] + xv;
    dst[HID + c] = bw[(long)ir*HID + c] + xv;
}

// output_proj (512->64) + LayerNorm.
//
// The old kernel spent ~1150 flops per thread on Kahan accumulations: 128
// FMAs for the dot product, then two 64-term Kahan sums per thread for the
// mean/variance (8 flops each). Those Kahan loops were the real cost, not
// memory - the kernel ran at fp32 peak for 16 MB of traffic. The reference
// (PyTorch LayerNorm) accumulates mean/variance in fp32 over 64 terms, where
// a shuffled tree reduction is ~1e-7, so plain fp32 partial sums + warp
// shuffles reproduce it. Naive path kept behind REUSE_NAIVE_OUTPROJ for A/B.
__global__ void k_outproj_ln_f(const float *cat, const float *W, const float *B,
                               const float *LW, const float *LB, float *out, long SL)
{
    const int ty = threadIdx.y;                    // 0..3: row within block
    const int row = blockIdx.x*4 + ty;
    const int o = threadIdx.x;                     // 0..63: output unit, 2 warps
    if (row >= SL) return;
    __shared__ float red[4][2][2];                 // [row][warp][sum/sumsq]
    const float *ci = cat + (long)row*2*HID;
    const float *wr = W + (long)o*2*HID;
    float v = B[o];
#pragma unroll
    for (int c = 0; c < 2*HID; c++) v = fmaf(ci[c], wr[c], v);
    float s = v, q = v*v;
#pragma unroll
    for (int d = 16; d; d >>= 1) {
        s += __shfl_down_sync(0xffffffffu, s, d);
        q += __shfl_down_sync(0xffffffffu, q, d);
    }
    if ((o & 31) == 0) { red[ty][o>>5][0] = s; red[ty][o>>5][1] = q; }
    __syncthreads();
    s = red[ty][0][0] + red[ty][1][0];
    q = red[ty][0][1] + red[ty][1][1];
    const float mean = s * (1.0f/HID);
    const float inv  = rsqrtf(fmaxf(q * (1.0f/HID) - mean*mean, 0.0f) + 1e-5f);
    out[(long)row*HID + o] = (v - mean) * inv * LW[o] + LB[o];
}

// output_proj (128 -> 64) + LayerNorm as a register-tiled GEMM.
//
// The one-block-per-row version had every one of the SL blocks read the whole
// 32 KB output_proj.weight, and thread o read W[o*128 + c] with a 128-float
// stride: 32 sectors per warp load (8x sector amplification), ~663 MB of L2
// traffic per launch. Ablation put it at 0.295 s per forward - 27% of the
// tower - and it was NOT compute bound (replacing the Kahan reductions with
// warp shuffles bought nothing). Here a block owns 32 rows x all 64 outputs,
// stages A and W tiles in shared per 32-wide K chunk, and each thread keeps a
// 4x4 accumulator, so every weight element is read once per block (21 MB)
// instead of once per row. The LayerNorm rides the same registers: a thread's
// 4 rows reduce across the 16-lane tx group with shuffles.
__global__ void k_outproj_ln_g(const float *cat, const float *W, const float *B,
                               const float *LW, const float *LB, float *out, long SL)
{
    const int KC = 32;
    __shared__ float As[32][KC+1], Ws[64][KC+1];
    const int tx = threadIdx.x, ty = threadIdx.y;          // 16 x 8
    const int tid = ty*16 + tx;
    const long rb = (long)blockIdx.x * 32;          // block's first row
    const long r0 = rb + ty*4;                      // this thread's first row
    const int c0 = tx*4;
    // stage the first K chunk
    for (int i = tid; i < 32*KC; i += 128) {
        int r = i >> 5, k = i & (KC-1);
        As[r][k] = (rb + r < SL) ? cat[(rb + r)*2*HID + k] : 0.f;
    }
    for (int i = tid; i < 64*KC; i += 128) {
        int o = i >> 5, k = i & (KC-1);
        Ws[o][k] = W[(long)o*2*HID + k];
    }
    float acc[4][4];
#pragma unroll
    for (int a = 0; a < 4; a++)
#pragma unroll
        for (int b = 0; b < 4; b++) acc[a][b] = B[c0 + b];
    __syncthreads();
    // Single-buffered tiles, so the pipeline is compute -> barrier -> prefetch
    // -> barrier. Prefetching into the tile the compute loop is still reading
    // is a race: it produced a rel-rms of 2.6 that only showed up on some
    // schedules, which is exactly what the accuracy gate is for.
    for (int k0 = 0; k0 < 2*HID; k0 += KC) {
#pragma unroll
        for (int k = 0; k < KC; k++) {
            float a[4], w[4];
#pragma unroll
            for (int a4 = 0; a4 < 4; a4++) a[a4] = As[ty*4 + a4][k];
#pragma unroll
            for (int b4 = 0; b4 < 4; b4++) w[b4] = Ws[c0 + b4][k];
#pragma unroll
            for (int a4 = 0; a4 < 4; a4++)
#pragma unroll
                for (int b4 = 0; b4 < 4; b4++) acc[a4][b4] = fmaf(a[a4], w[b4], acc[a4][b4]);
        }
        if (k0 + KC < 2*HID) {
            __syncthreads();
            for (int i = tid; i < 32*KC; i += 128) {
                int r = i >> 5, k = i & (KC-1);
                As[r][k] = (rb + r < SL) ? cat[(rb + r)*2*HID + k0 + KC + k] : 0.f;
            }
            for (int i = tid; i < 64*KC; i += 128) {
                int o = i >> 5, k = i & (KC-1);
                Ws[o][k] = W[(long)o*2*HID + k0 + KC + k];
            }
            __syncthreads();
        }
    }
    // LayerNorm over the 64 outputs of each row: the row is spread over the 16
    // threads with the same ty, which never straddle a warp boundary.
    float mean[4], inv[4];
#pragma unroll
    for (int a4 = 0; a4 < 4; a4++) {
        float s = acc[a4][0] + acc[a4][1] + acc[a4][2] + acc[a4][3];
        float q = acc[a4][0]*acc[a4][0] + acc[a4][1]*acc[a4][1]
                + acc[a4][2]*acc[a4][2] + acc[a4][3]*acc[a4][3];
#pragma unroll
        for (int d = 1; d < 16; d <<= 1) {
            s += __shfl_xor_sync(0xffffffffu, s, d);
            q += __shfl_xor_sync(0xffffffffu, q, d);
        }
        mean[a4] = s * (1.0f/HID);
        inv[a4]  = rsqrtf(fmaxf(q * (1.0f/HID) - mean[a4]*mean[a4], 0.0f) + 1e-5f);
    }
#pragma unroll
    for (int a4 = 0; a4 < 4; a4++) {
        long r = r0 + a4;
        if (r >= SL) continue;
        float *dst = out + r*HID;
#pragma unroll
        for (int b4 = 0; b4 < 4; b4++)
            dst[c0 + b4] = (acc[a4][b4] - mean[a4]) * inv[a4] * LW[c0 + b4] + LB[c0 + b4];
    }
}

// [HID,T,F] -> [F,T,HID]
// [C,T,F] -> [F,T,C] gather (the tower's per-block input view). k_gather_time
// walks the output linearly, so for 32 consecutive outputs a warp reads 32
// floats spaced T*F apart: 32 sectors per warp load, measured 63 GB/s of the
// 158 GB/s ceiling. T is the shared middle dimension in both layouts, so this
// is a plain 2D transpose of (C,F) per t, tiled 32x32 through padded shared.
__global__ void k_gather_time_t(const float *x, float *xt, int T, int F)
{
    __shared__ float sh[32][33];
    const int t = blockIdx.z, f0 = blockIdx.x*32, c0 = blockIdx.y*32;
    const int tx = threadIdx.x, ty = threadIdx.y;
#pragma unroll
    for (int j = ty; j < 32; j += 8) {
        const int c = c0 + j, f = f0 + tx;
        sh[j][tx] = (c < HID && f < F) ? x[((long)c*T + t)*F + f] : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int j = ty; j < 32; j += 8) {
        const int f = f0 + j, c = c0 + tx;
        if (c < HID && f < F) xt[((long)f*T + t)*HID + c] = sh[tx][j];
    }
}

__global__ void k_gather_time(const float *x, float *xt, int T, int F)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)F*T*HID) return;
    int c = i % HID, t = (int)((i / HID) % T), f = (int)(i / ((long)HID*T));
    xt[i] = x[(long)c*T*F + (long)t*F + f];
}
// residual on the time block + resample into [T,F,HID] for the freq block

// Diagnostic prewarm: burn GPU cycles with a compute-heavy, large-grid kernel
// at the start of a forward, to test whether the ~10x penalty on the first
// big conv2d launch of the rep is a clock/DVFS ramp (a 256-block spin kernel
// was "fully additive" earlier, so it never ramped anything) or a
// large-grid-launch effect (the conv launches 5632 blocks; every earlier probe
// used 256-block kernels). REUSE_PREWARM=1 (FMA iters) and REUSE_PREWARM_GRID.
__global__ void k_burn(float *sink, long iters)
{
    float a = threadIdx.x * 1e-8f + blockIdx.x * 1e-9f;
    const float b = 1.00000012f;
    for (long i = 0; i < iters; i++) a = fmaf(a, b, 1e-9f);
    if (a == 12345.678f) sink[threadIdx.x + blockIdx.x*blockDim.x] = a;
}

static void prewarm(reuse_gpu_t *g)
{
    const char *e = getenv("REUSE_PREWARM");
    if (!e) return;
    long iters = atol(e);
    int grid = getenv("REUSE_PREWARM_GRID") ? atoi(getenv("REUSE_PREWARM_GRID")) : 5632;
    k_burn<<<grid, 256>>>(g->ws, iters);
    cudaDeviceSynchronize();
}


// 128x64-tile version of the fused in_proj+conv.
//
// Tile traffic for a GEMM of M rows x N columns tiled R x C is
// M*N*K*(1/R + 1/C): the 80x64 tile re-reads each input row once per output
// column tile (8 times) and each weight once per row tile. Doubling the tile
// height to 128 rows (8 rows x 4 cols = 32 accumulators per thread) cuts that
// 44%, and with 8 shared row loads per 32 FMAs the shared-load ratio goes
// 2.2 -> 4.
#define FUSE_MT2 128
template<int KC>
__global__ void k_in_conv_r2(const float *x, const float *W, float *z, float *xc,
                             const float *Wc, const float *Bc,
                             int S, int L, int rev, int zhalo = 1)
{
    const int NT = 64, R = FUSE_MT2 + 3, RA = 8;
    __shared__ __align__(16) float Xs[R][KC+4], Ws[NT][KC+4], Hl[3][NT+1];
    const long SL = (long)S * L;
    const long m0 = (long)blockIdx.x * FUSE_MT2;
    const int n0 = blockIdx.y * NT;
    const int tx = threadIdx.x, ty = threadIdx.y;                 // (16,16)
    const int tid = ty*16 + tx;
    const bool halo_thr = (tid < 3*NT) && (zhalo || n0 < DINNER);
    // Rows 0..2 of the A-tile are the conv's 3-row halo, consumed only through
    // Hl, i.e. only by the x0 half. Loading them costs 27 float4 global loads +
    // 27 shared stores per block (each with the m/L divide), so the z half
    // skips them. Only blockIdx.x == 0 can have m < 0, hence mlo_ok.
    const int rlo = (zhalo || n0 < DINNER) ? 0 : 3;
    const bool mlo_ok = (m0 >= 3);
    const int hr = tid / NT, hc = tid % NT;
    float acc[RA][4];
#pragma unroll
    for (int a = 0; a < RA; a++)
#pragma unroll
        for (int b = 0; b < 4; b++) acc[a][b] = 0.f;
    float hacc = 0.f;
    for (int k0 = 0; k0 < HID; k0 += KC) {
        // One index computation (incl. the 64-bit m/L divide) per 4 elements;
        // the divide itself is untouched, only computed 4x less often (see the
        // note above on not touching its scheduling). RS is in float4 units of
        // the PADDED row stride (KC+4), not KC: indexing by KC/4 writes rows
        // compactly while the GEMM reads them at the padded stride, i.e. reads
        // uninitialised shared memory (measured: all-NaN model output).
        {
            const int RS = (KC+4)/4;
            for (int i = tid; i < R*RS; i += 256) {
                const int r = i / RS, q = 4*(i - r*RS);
                if (r < rlo) continue;
                const long m = m0 - 3 + r;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (q < KC && m < SL && (mlo_ok || m >= 0)) {
                    const int sm = (int)(m / L), tm = (int)(m % L);
                    v = *(const float4*)(x + ((long)sm*L + (rev ? L-1-tm : tm))*HID + k0 + q);
                }
                ((float4*)&Xs[0][0])[i] = v;
            }
        }
        {
            const int RS = (KC+4)/4;     // padded row stride in float4 units
            for (int i = tid; i < NT*RS; i += 256) {
                const int r = i / RS, q = 4*(i - r*RS);
                if (q >= KC) continue;   // the 4 pad columns stay untouched
                // HID (256) and n0 (x64) make this 16B aligned by construction
                ((float4*)&Ws[0][0])[i] = *(const float4*)(W + (long)(n0+r)*HID + k0 + q);
            }
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < KC; k++) {
            float a[RA], w[4];
#pragma unroll
            for (int a4 = 0; a4 < RA; a4++) a[a4] = Xs[tx*RA + 3 + a4][k];
#pragma unroll
            for (int b4 = 0; b4 < 4; b4++) w[b4] = Ws[ty*4 + b4][k];
#pragma unroll
            for (int a4 = 0; a4 < RA; a4++)
#pragma unroll
                for (int b4 = 0; b4 < 4; b4++) acc[a4][b4] = fmaf(a[a4], w[b4], acc[a4][b4]);
            if (halo_thr) hacc = fmaf(Xs[hr][k], Ws[hc][k], hacc);
        }
        __syncthreads();
    }
    if (halo_thr) Hl[hr][hc] = hacc;
    if (n0 < DINNER) __syncthreads();   // Hl readers only
    if (n0 >= DINNER) {                                             // z half: plain GEMM
#pragma unroll
        for (int a4 = 0; a4 < RA; a4++) {
            long m = m0 + tx*RA + a4;
            if (m >= SL) continue;
#pragma unroll
            for (int b4 = 0; b4 < 4; b4++) z[m*DINNER + (n0 - DINNER) + ty*4 + b4] = acc[a4][b4];
        }
        return;
    }
    const unsigned FULL = 0xffffffffu;
    float v[3][4];
#pragma unroll
    for (int jj = 0; jj < 3; jj++)
#pragma unroll
        for (int b4 = 0; b4 < 4; b4++) v[jj][b4] = __shfl_up_sync(FULL, acc[jj+5][b4], 1);
    if (tx == 0) {
#pragma unroll
        for (int h = 0; h < 3; h++)
#pragma unroll
            for (int b4 = 0; b4 < 4; b4++) v[h][b4] = Hl[h][ty*4 + b4];
    }
    const int c0 = n0 + ty*4;
#pragma unroll
    for (int a4 = 0; a4 < RA; a4++) {
        long m = m0 + tx*RA + a4;
        if (m >= SL) continue;
        const int t = (int)(m % L);
        const int ntap = (t >= DCONV-1) ? DCONV : t + 1;
        float fv[4];
#pragma unroll
        for (int b4 = 0; b4 < 4; b4++) fv[b4] = Bc[c0 + b4];
        // Fully unrolled with a predicate, never a runtime trip count: with
        // "k = DCONV - ntap" as the loop start, j = a4+k is a runtime index and
        // v[]/acc[] spill to local memory (128-176 byte stack frames).
#pragma unroll
        for (int k = 0; k < DCONV; k++) {
            if (k < DCONV - ntap) continue;
            const int j = a4 + k;
#pragma unroll
            for (int b4 = 0; b4 < 4; b4++)
                fv[b4] = fmaf((j < 3 ? v[j][b4] : acc[j-3][b4]),
                              Wc[(long)(c0 + b4)*DCONV + k], fv[b4]);
        }
#pragma unroll
        for (int b4 = 0; b4 < 4; b4++) xc[m*DINNER + c0 + b4] = dsilu(fv[b4]);
    }
}

static int inproj_mode(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("REUSE_INPROJ");
        v = e ? atoi(e) : (getenv("REUSE_NAIVE_INPROJ") ? 0 : 2);
    }
    return v;
}

// in_proj and the causal depthwise conv are fused by default: x0 is consumed
// only by the conv, so fusing keeps 42 MB per launch out of DRAM and drops a
// launch. REUSE_NO_FUSE=1 falls back to the two-kernel path for A/B.
static int fuse_tile(void)
{
    static int v = -1;
    if (v < 0) { const char *e = getenv("REUSE_FUSE_TILE"); v = e ? atoi(e) : 80; if (v != 128) v = 80; }
    return v;
}

static int fuse_kc(void)
{
    static int v = -1;
    if (v < 0) { const char *e = getenv("REUSE_FUSE_KC"); v = e ? atoi(e) : 32; if (v != 8 && v != 16) v = 32; }
    return v;
}

static int fuse_conv(void)
{
    static int v = -1;
    if (v < 0)
        v = (inproj_mode() == 2 && !getenv("REUSE_NAIVE_CONV") && !getenv("REUSE_NO_FUSE")) ? 1 : 0;
    return v;
}

static int naive_resid(void)
{
    static int v = -1;
    if (v < 0) v = getenv("REUSE_NAIVE_RESID") ? 1 : 0;
    return v;
}

__global__ void k_resid_scatter_freq(const float *y, const float *x, float *out,
                                     float *xf, int T, int F)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)F*T*HID) return;
    int c = i % HID, t = (int)((i / HID) % T), f = (int)(i / ((long)HID*T));
    long dst = (long)c*T*F + (long)t*F + f;
    float v = y[i] + x[dst];
    out[dst] = v;
    xf[((long)t*F + f)*HID + c] = v;
}
// Tiled versions of the two residual scatters. A-layout is [F,T,C] (c
// contiguous), B-layout is [C,T,F] (f contiguous), so any elementwise op
// between them has one side strided by T*F floats: a warp touched 32 separate
// 128 B sectors, 32x traffic amplification, which is exactly why the naive
// kernels measured 12.7 GB/s on a ~250 GB/s device. A 32x32 (f x c) tile per
// t makes every load and store contiguous; the two decompositions of the tile
// index differ per layout so each side is coalesced in its own order.
__global__ void k_resid_scatter_freq_t(const float *y, const float *x, float *out,
                                       float *xf, int T, int F)
{
    __shared__ float sy[32][33], sx[32][33];
    const int t = blockIdx.z, f0 = blockIdx.x*32, c0 = blockIdx.y*32;
    const int tid = threadIdx.y*32 + threadIdx.x;
    // A-layout load: c contiguous -> i = f*32 + c
#pragma unroll
    for (int i = tid; i < 32*32; i += 1024) {
        int fi = i >> 5, ci = i & 31;
        int f = f0 + fi, c = c0 + ci;
        sy[fi][ci] = (f < F) ? y[((long)f*T + t)*HID + c] : 0.f;
    }
    // B-layout load: f contiguous -> i = c*32 + f
#pragma unroll
    for (int i = tid; i < 32*32; i += 1024) {
        int ci = i >> 5, fi = i & 31;
        int f = f0 + fi, c = c0 + ci;
        sx[fi][ci] = (f < F) ? x[(long)c*T*F + (long)t*F + f] : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int i = tid; i < 32*32; i += 1024) {
        int fi = i >> 5, ci = i & 31;
        int f = f0 + fi, c = c0 + ci;
        if (f >= F) continue;
        float v = sy[fi][ci] + sx[fi][ci];
        xf[((long)t*F + f)*HID + c] = v;                      // A layout, c contiguous
    }
#pragma unroll
    for (int i = tid; i < 32*32; i += 1024) {
        int ci = i >> 5, fi = i & 31;
        int f = f0 + fi, c = c0 + ci;
        if (f < F) out[(long)c*T*F + (long)t*F + f] = sy[fi][ci] + sx[fi][ci];   // B, f contiguous
    }
}

__global__ void k_resid_scatter_out_t(const float *y, const float *xf, float *out, int T, int F)
{
    __shared__ float sa[32][33];
    const int t = blockIdx.z, f0 = blockIdx.x*32, c0 = blockIdx.y*32;
    const int tid = threadIdx.y*32 + threadIdx.x;
#pragma unroll
    for (int i = tid; i < 32*32; i += 1024) {
        int fi = i >> 5, ci = i & 31;
        int f = f0 + fi, c = c0 + ci;
        float v = 0.f;
        if (f < F) { long src = ((long)t*F + f)*HID + c; v = y[src] + xf[src]; }
        sa[fi][ci] = v;
    }
    __syncthreads();
#pragma unroll
    for (int i = tid; i < 32*32; i += 1024) {
        int ci = i >> 5, fi = i & 31;
        int f = f0 + fi, c = c0 + ci;
        if (f < F) out[(long)c*T*F + (long)t*F + f] = sa[fi][ci];
    }
}

__global__ void k_resid_scatter_out(const float *y, const float *xf, float *out, int T, int F)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)F*T*HID) return;
    int c = i % HID, t = (int)((i / HID) % T), f = (int)(i / ((long)HID*T));
    long src = ((long)t*F + f)*HID + c;
    out[(long)c*T*F + (long)t*F + f] = y[src] + xf[src];
}

// SPConvTranspose2d: zero-pad W by 1, conv (1,3) to C_out=64*r, sub-pixel scatter.
// y[c, h, w*r + a] = conv[a*64 + c, h, w]
// Sub-pixel conv with 8 output channels per thread.
//
// k_sp_up is a GEMM (M = H*W, N = 64*r, K = HID*3) launched with one output
// channel per block, so every one of the 256 co-blocks re-reads the whole
// 11.5 MB input: ~3 GB of DRAM traffic per launch, which is why an 11 ms
// launch moves its 46 MB of output at 4 GB/s. Owning 8 co per thread reuses
// the 3 loaded taps 8x; the 24 weight floats per (ci) are the same addresses
// across the warp (lanes differ only in w), so they broadcast.
// Sub-pixel conv with the r sub-pixel phases written as one vector store.
// k_sp_up8 writes its 8 outputs as 8 scalar stores at stride r (the phases of
// one output channel are r consecutive elements), which costs r sectors per
// warp store - 46 MB of output at 14 GB/s. Grouping the 8 channels as
// (8/r) channels x r phases lets each thread write the r phases of one
// channel as a single float2/float4 at a w*r*4-byte aligned address, so a
// warp's writes are contiguous.
// Sub-pixel conv with the r sub-pixel phases written as one vector store.
// k_sp_up8's 8 channels per thread are 8 channels of ONE phase (the channel
// index is co % HID and the phase is co / HID), so its 8 stores each go to a
// stride-r address: 46 MB of output at 14 GB/s. Here the 8 slots per thread
// are (8/r) channels x r consecutive phases, so a thread writes the r phases
// of one channel as a single float2/float4 at a w*r*4-byte aligned address
// and a warp's writes become contiguous.
// Sub-pixel conv with the weights pre-transposed to [ci][co][tap].
// k_sp_up8 reads its 24 weights per (ci, 8 co) as 8 strided float3 loads
// (one per co, HID*3 floats apart): 11 LSU per 24 FMAs = 2.2 FMA/LSU, which
// is what pins it at 14 GB/s of writes. With [ci][co][tap] the same 24
// weights are 24 contiguous floats = 6 float4 loads, 9 LSU per 24 FMAs.
__global__ void k_wtrans_sp(const float *w, float *wT, int CO)
{
    // w: [CO][HID][3] -> wT: [HID][CO][3]
    int i = blockIdx.x*256 + threadIdx.x;
    if (i >= CO*HID*3) return;
    const int t = i % 3, co = (i / 3) / HID, ci = (i / 3) % HID;
    wT[((long)ci*CO + co)*3 + t] = w[i];
}

template<int RW, int RC, int TW, int NC>
__global__ void k_sp_up8t(const float *x, const float *wT, const float *b, float *y,
                          int H, int W, int r, int CO)
{
    const int TXN = TW/RW, TBY = NC/RC, NT = TXN*TBY;
    const int w0 = blockIdx.x*TW;
    const int h = blockIdx.y;
    const int co0 = blockIdx.z*NC;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty*TXN + tx;
    __shared__ __align__(16) float xs[TW+8];
    __shared__ __align__(16) float ws[NC*3 + 4];
    const int wl = tx*RW, cl = ty*RC;
    float acc[RC][RW];
#pragma unroll
    for (int j = 0; j < RC; j++)
#pragma unroll
        for (int i = 0; i < RW; i++) acc[j][i] = b[co0 + cl + j];
    const long hw = (long)h*W;
    for (int ci = 0; ci < HID; ci++) {
        const float *xv = x + (long)ci*H*W + hw;
        for (int i = tid; i < (TW+3)/4 + 1; i += NT) {
            const int q = 4*i;
            float4 v = make_float4(0.f,0.f,0.f,0.f);
            /* xs[i] = x[w0-1+i] (zero outside the row), so lane t sources w0-1+q+t */
            const int s0 = w0 - 1 + q;
            if (q+0 < TW+3) v.x = (s0+0 >= 0 && s0+0 < W) ? xv[s0+0] : 0.f;
            if (q+1 < TW+3) v.y = (s0+1 >= 0 && s0+1 < W) ? xv[s0+1] : 0.f;
            if (q+2 < TW+3) v.z = (s0+2 >= 0 && s0+2 < W) ? xv[s0+2] : 0.f;
            if (q+3 < TW+3) v.w = (s0+3 >= 0 && s0+3 < W) ? xv[s0+3] : 0.f;
            *(float4*)&xs[q] = v;
        }
        { const float4 *wp = (const float4*)(wT + ((long)ci*CO + co0)*3);
          for (int i = tid; i < NC*3/4; i += NT) *(float4*)&ws[4*i] = wp[i]; }
        __syncthreads();
        const float4 a0 = *(const float4*)&xs[wl];
        const float4 a1 = *(const float4*)&xs[wl+4];
        const float4 a2 = *(const float4*)&xs[wl+8];
        const float xv2[16] = { a0.x,a0.y,a0.z,a0.w, a1.x,a1.y,a1.z,a1.w, a2.x,a2.y,a2.z,a2.w, 0,0,0,0 };
#pragma unroll
        for (int j = 0; j < RC; j++) {
            const float c0 = ws[3*(cl+j)], c1 = ws[3*(cl+j)+1], c2 = ws[3*(cl+j)+2];
#pragma unroll
            for (int i = 0; i < RW; i++) {
                acc[j][i] = fmaf(xv2[i],   c0, acc[j][i]);
                acc[j][i] = fmaf(xv2[i+1], c1, acc[j][i]);
                acc[j][i] = fmaf(xv2[i+2], c2, acc[j][i]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < RC; j++) {
        const int cj = (co0 + cl + j) % HID, aj = (co0 + cl + j) / HID;
        float *yp = y + ((long)cj*H + h)*(long)(W*r) + (long)(w0+wl)*r + aj;
#pragma unroll
        for (int i = 0; i < RW; i++) if (w0 + wl + i < W) yp[i*r] = acc[j][i];
    }
}

template<int NC>
__global__ void k_sp_up8w(const float *x, const float *wT, const float *b, float *y,
                          int H, int W, int r, int CO)
{
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int co = blockIdx.z * NC;
    if (h >= H || w >= W) return;
    float acc[NC];
#pragma unroll
    for (int j = 0; j < NC; j++) acc[j] = b[co + j];
    const long hw = (long)h*W;
    for (int ci = 0; ci < HID; ci++) {
        const float *xv = x + (long)ci*H*W + hw;
        const float v1 = xv[w];
        const float v0 = (w >= 1)   ? xv[w-1] : 0.f;
        const float v2 = (w+1 < W)  ? xv[w+1] : 0.f;
        const float4 *wp = (const float4*)(wT + ((long)ci*CO + co)*3);
        float wt[3*NC];
#pragma unroll
        for (int q = 0; q < 3*NC/4; q++) {
            const float4 f = wp[q];
            wt[4*q] = f.x; wt[4*q+1] = f.y; wt[4*q+2] = f.z; wt[4*q+3] = f.w;
        }
#pragma unroll
        for (int j = 0; j < NC; j++) {
            acc[j] = fmaf(v0, wt[3*j],   acc[j]);
            acc[j] = fmaf(v1, wt[3*j+1], acc[j]);
            acc[j] = fmaf(v2, wt[3*j+2], acc[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < NC; j++) {
        const int cj = (co + j) % HID, aj = (co + j) / HID;
        y[((long)cj*H + h)*(long)(W*r) + (long)w*r + aj] = acc[j];
    }
}

template<int RR>
__global__ void k_sp_up8v(const float *x, const float *Ww, const float *b, float *y,
                          int H, int W, int r)
{
    const int CH = 8 / RR;                 // channels per thread
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int c0 = (int)blockIdx.z * CH;   // channel base; phases are all r
    if (h >= H || w >= W) return;
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int co = (j % RR)*HID + c0 + (j / RR);
        acc[j] = b[co];
    }
    const long hw = (long)h*W;
    const int Wc = W*RR;
    for (int ci = 0; ci < HID; ci++) {
        const float *xv = x + (long)ci*H*W + hw;
        const float v1 = xv[w];
        const float v0 = (w >= 1)   ? xv[w-1] : 0.f;
        const float v2 = (w+1 < W)  ? xv[w+1] : 0.f;
#pragma unroll
        for (int q = 0; q < CH; q++) {
            const float *wp = Ww + ((long)(c0 + q)*HID + ci)*3;
#pragma unroll
            for (int a = 0; a < RR; a++) {
                const float3 wt = *(const float3*)(wp + (long)a*HID*HID*3);
                const int j = q*RR + a;
                acc[j] = fmaf(v0, wt.x, acc[j]);
                acc[j] = fmaf(v1, wt.y, acc[j]);
                acc[j] = fmaf(v2, wt.z, acc[j]);
            }
        }
    }
#pragma unroll
    for (int q = 0; q < CH; q++) {
        float *p = y + ((long)(c0 + q)*H + h)*Wc + (long)w*RR;
        if (RR == 4) *(float4*)p = make_float4(acc[4*q], acc[4*q+1], acc[4*q+2], acc[4*q+3]);
        else         *(float2*)p = make_float2(acc[2*q], acc[2*q+1]);
    }
    (void)r;
}

__global__ void k_sp_up8(const float *x, const float *Ww, const float *b, float *y,
                         int H, int W, int r)
{
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int co = blockIdx.z * 8;
    if (h >= H || w >= W) return;
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; j++) acc[j] = b[co + j];
    const long hw = (long)h*W;
    for (int ci = 0; ci < HID; ci++) {
        const float *xv = x + (long)ci*H*W + hw;
        const float v1 = xv[w];
        const float v0 = (w >= 1)   ? xv[w-1] : 0.f;
        const float v2 = (w+1 < W)  ? xv[w+1] : 0.f;
        const float *wp = Ww + ((long)co*HID + ci)*3;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const float3 wt = *(const float3*)(wp + j*HID*3);
            acc[j] = fmaf(v0, wt.x, acc[j]);
            acc[j] = fmaf(v1, wt.y, acc[j]);
            acc[j] = fmaf(v2, wt.z, acc[j]);
        }
    }
    const int c = co % HID, a = co / HID;
    float *yo = y + ((long)c*H + h)*(long)(W*r) + (long)w*r + a;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int cj = (co + j) % HID, aj = (co + j) / HID;
        y[((long)cj*H + h)*(long)(W*r) + (long)w*r + aj] = acc[j];
    }
    (void)yo; (void)c; (void)a;
}

__global__ void k_sp_up(const float *x, const float *Ww, const float *b, float *y,
                        int H, int W, int r)
{
    int w = blockIdx.x * blockDim.x + threadIdx.x;     // contiguous dim first
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int co = blockIdx.z;                       // 0 .. 64*r-1
    if (h >= H || w >= W) return;
    const int c = co % HID, a = co / HID;
    float acc = b[co];
    for (int ci = 0; ci < HID; ci++) {
        const float *xv = x + (long)ci * H * W;
        const float *wv = Ww + ((long)co * HID + ci) * 3;
        for (int kw = 0; kw < 3; kw++) {
            int wi = w - 1 + kw;                       // pad1 = 1 on W
            if (wi < 0 || wi >= W) continue;
            acc = fmaf(xv[(long)h*W + wi], wv[kw], acc);
        }
    }
    y[((long)c*H + h)*(long)(W*r) + (long)w*r + a] = acc;
}

/* x[C][H][W] -> y[C][W][H]. The naive version reads coalesced in w and writes
   with a stride of H floats (32 sectors per warp store), which is why 12 of
   these per forward cost 4 ms for ~100 MB of traffic. Tiled through a padded
   32x32 shared tile: both phases become coalesced and conflict-free (the lane
   index advances by 1 in the load, and the store reads tile[x][k] at banks
   x+33k mod 32 = x+k). */
__global__ void k_transpose_t(const float *__restrict__ x, float *__restrict__ y, int C, int H, int W)
{
    __shared__ float tile[32][33];
    const int c = blockIdx.z;
    const float *xc = x + (long)c*H*W;
    float *yc = y + (long)c*H*W;                     /* y is [C][W][H] */
    const int h0 = blockIdx.y*32, w0 = blockIdx.x*32;
    for (int k = threadIdx.y; k < 32; k += 8) {
        const int h = h0 + k, w = w0 + threadIdx.x;
        tile[k][threadIdx.x] = (h < H && w < W) ? xc[(long)h*W + w] : 0.0f;
    }
    __syncthreads();
    for (int k = threadIdx.y; k < 32; k += 8) {
        const int w = w0 + k, h = h0 + threadIdx.x;
        if (w < W && h < H) yc[(long)w*H + h] = tile[threadIdx.x][k];
    }
}


__global__ void k_transpose(const float *x, float *y, int C, int H, int W)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)C*H*W) return;
    int w = (int)(i % W), h = (int)((i / W) % H), c = (int)(i / ((long)H*W));
    y[((long)c*W + w)*(long)H + h] = x[i];
}

static void launch_transpose(int C, int H, int W, const float *x, float *y)
{
    static int t = -1;
    if (t < 0) t = getenv("REUSE_TRANSPOSE") ? atoi(getenv("REUSE_TRANSPOSE")) : 1;
    if (t) k_transpose_t<<<dim3((unsigned)((W+31)/32), (unsigned)((H+31)/32), (unsigned)C), dim3(32,8)>>>(x, y, C, H, W);
    else   k_transpose<<<(unsigned)(((long)C*H*W+255)/256), 256>>>(x, y, C, H, W);
}

// out_mag[f*T+t] = src[t*Fx+f]   (decoder output [Tx,Fx] -> [F,T] crop)
__global__ void k_crop_ft(const float *src, float *out, int T, int F, int Fx)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)T*F) return;
    int t = (int)(i / F), f = (int)(i % F);
    out[(long)f*T + t] = src[(long)t*Fx + f];
}

// mag/pha are [F,T]; xin is [2,Tp,Fp] with 2 zero freq cols and 2 zero time rows.
__global__ void k_stft_prep(const float *x, int T, double *xw, int n_fft, int hop,
                            int pad, const float *w)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)T * n_fft;
    if (i >= n) return;
    int t = (int)(i / n_fft), k = (int)(i % n_fft);
    int idx = t*hop - pad + k;
    float v;
    if (idx < 0)        v = x[-idx - 1];
    else if (idx >= T)  v = x[2*T - 2 - idx];
    else                v = x[idx];
    xw[i] = (double)v * (double)w[k];            // the reference forms this product in double
}

// ---- two-float (DD) STFT -------------------------------------------------
// k_stft_dft is a fp64 GEMM: 89 MFMA per call against a GP107 roof of 2 fp64
// FMA/clk/SM = 1.98 ms, and the measured 2.02 ms says it is AT that roof. On
// tegra210 fp64 is 1/64 rate, which is why the Nano runs the STFT on the CPU.
// Double-float (dd: a float pair whose value is hi+lo, ~48 mantissa bits) runs
// on the fp32 pipes at full rate. The inputs are conversion-exact: xw is the
// product of two floats (Dekker: p = a*b and e = fmaf(a,b,-p) reproduce it
// exactly), and a double twiddle split as hi=(float)c, lo=(float)(c-hi) carries
// 2^-48 - 1e5 times tighter than the float mag/pha the STFT finally emits.
typedef float2 dd;
__device__ __forceinline__ void dd_mul(dd a, dd b, dd *p)
{
    p->x = a.x * b.x;
    p->y = fmaf(a.x, b.x, -p->x);
    p->y = fmaf(a.x, b.y, p->y);
    p->y = fmaf(a.y, b.x, p->y);
}
__device__ __forceinline__ void dd_add(dd *s, dd a)
{
    // Knuth two-sum; the low halves are exact by construction (dd_add is the
    // only place a sum rounds, so the running error stays at the dd floor).
    float th = s->x + a.x;
    float z  = s->x - th;
    float t  = ((s->x - (th - z)) + (a.x - z)) + (s->y + a.y);
    s->x = th; s->y = t;
}
__device__ __forceinline__ dd dd_split(double v)
{
    dd r; r.x = (float)v; r.y = (float)(v - (double)r.x); return r;
}
__device__ __forceinline__ double dd_to_double(dd v) { return (double)v.x + (double)v.y; }

__global__ void k_stft_twiddles_dd(dd *w, dd *Cc, dd *Cs, int F, int n_fft, int win)
{
    const double pi = 3.14159265358979323846;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_fft) {
        float v = (i < win) ? 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)win)) : 0.0f;
        w[i].x = v; w[i].y = 0.0f;
        return;
    }
    i -= n_fft;
    long total = (long)F * n_fft;
    if (i >= total) return;
    int k = (int)(i / n_fft);
    int n = (int)(i % n_fft);
    double ang = -2.0 * pi * ((double)n * (double)k) / (double)n_fft;
    Cc[i] = dd_split(cos(ang));
    Cs[i] = dd_split(sin(ang));
}

__global__ void k_stft_prep_dd(const float *x, int T, dd *xw, int n_fft, int hop,
                               int pad, const dd *w)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long n = (long)T * n_fft;
    if (i >= n) return;
    int t = (int)(i / n_fft), k = (int)(i % n_fft);
    int idx = t*hop - pad + k;
    float v;
    if (idx < 0)        v = x[-idx - 1];
    else if (idx >= T)  v = x[2*T - 2 - idx];
    else                v = x[idx];
    // the reference forms this product in double; Dekker's split is exact for
    // a product of two floats, so (p, e) IS the double value.
    float p = v * w[k].x;
    xw[i].x = p;
    xw[i].y = fmaf(v, w[k].x, -p);
}

__global__ void k_stft_dft_dd(const dd *xw, const dd *Cc, const dd *Cs,
                              double *re_out, double *im_out, int F, int T, int n_fft)
{
    const int KT = 32;
    __shared__ dd Ct[16][KT+1], St[16][KT+1], Xt[KT][16+1];
    const int t0 = blockIdx.x * 16, k0 = blockIdx.y * 16;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty*16 + tx;
    dd re, im;
    re.x = re.y = im.x = im.y = 0.0f;
    for (int n0 = 0; n0 < n_fft; n0 += KT) {
        __syncthreads();
        for (int i = tid; i < 16*KT; i += 256) {
            int kk = i / KT, nn = i % KT;
            int n = n0 + nn;
            dd z; z.x = 0.0f; z.y = 0.0f;
            if (k0 + kk < F && n < n_fft) {
                Ct[kk][nn] = Cc[(long)(k0+kk)*n_fft + n];
                St[kk][nn] = Cs[(long)(k0+kk)*n_fft + n];
            } else { Ct[kk][nn] = z; St[kk][nn] = z; }
            Xt[nn][kk] = (t0 + kk < T && n < n_fft) ? xw[(long)(t0+kk)*n_fft + n] : z;
        }
        __syncthreads();
        if (k0 + ty < F && t0 + tx < T) {
#pragma unroll
            for (int nn = 0; nn < KT; nn++) {
                dd xv = Xt[nn][tx], p;
                dd_mul(xv, Ct[ty][nn], &p); dd_add(&re, p);
                dd_mul(xv, St[ty][nn], &p); dd_add(&im, p);
            }
        }
    }
    if (k0 + ty < F && t0 + tx < T) {
        long off = (long)(k0+ty)*T + (t0+tx);
        re_out[off] = dd_to_double(re);
        im_out[off] = dd_to_double(im);
    }
}

__global__ void k_stft_dft(const double *xw, const double *Cc, const double *Cs,
                           double *re_out, double *im_out, int F, int T, int n_fft)
{
    const int KT = 32;
    __shared__ double Ct[16][KT+1], St[16][KT+1], Xt[KT][16+1];
    const int t0 = blockIdx.x * 16, k0 = blockIdx.y * 16;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty*16 + tx;
    double re = 0.0, im = 0.0;
    for (int n0 = 0; n0 < n_fft; n0 += KT) {
        __syncthreads();
        for (int i = tid; i < 16*KT; i += 256) {
            int kk = i / KT, nn = i % KT;
            int n = n0 + nn;
            if (k0 + kk < F && n < n_fft) {
                Ct[kk][nn] = Cc[(long)(k0+kk)*n_fft + n];
                St[kk][nn] = Cs[(long)(k0+kk)*n_fft + n];
            } else { Ct[kk][nn] = 0.0; St[kk][nn] = 0.0; }
            Xt[nn][kk] = (t0 + kk < T && n < n_fft) ? xw[(long)(t0+kk)*n_fft + n] : 0.0;
        }
        __syncthreads();
        if (k0 + ty < F && t0 + tx < T) {
#pragma unroll
            for (int nn = 0; nn < KT; nn++) {
                double xv = Xt[nn][tx];
                re += Ct[ty][nn]*xv;
                im += St[ty][nn]*xv;
            }
        }
    }
    if (k0 + ty < F && t0 + tx < T) {
        long off = (long)(k0+ty)*T + (t0+tx);
        re_out[off] = re;
        im_out[off] = im;
    }
}

// The DC row, computed directly. k_stft_dft is bit-identical to the CPU
// reference for every bin k>=1 (max|d| 1e-7 on the real fixture), but in the
// app the k=0 tile ends up with the Nyquist row's twiddles: its output
// alternates in sign like cos(pi*n) and equals the k=160 sums, on 999 of 1021
// frames (rel-rms 2.08 against a direct DFT), while the same kernel run
// standalone matches the reference exactly. The DC is just the sum of the
// windowed frame, so computing it here (same serial n order as the CPU loop,
// with the multiply-by-one folded out) removes the dependence on that tile.
__global__ void k_stft_dc(const double *xw, double *re_out, double *im_out,
                          int T, int n_fft)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    const double *src = xw + (long)t * n_fft;
    double re = 0.0;
    for (int n = 0; n < n_fft; n++) re += src[n];
    re_out[(long)t] = re;
    im_out[(long)t] = 0.0;
}

__global__ void k_stft_dc_dd(const dd *xw, double *re_out, double *im_out,
                             int T, int n_fft)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    const dd *src = xw + (long)t * n_fft;
    dd re; re.x = re.y = 0.0f;
    for (int n = 0; n < n_fft; n++) dd_add(&re, src[n]);
    re_out[(long)t] = dd_to_double(re);
    im_out[(long)t] = 0.0;
}

__global__ void k_stft_twiddles(float *w, double *Cc, double *Cs, int F, int n_fft, int win)
{
    const double pi = 3.14159265358979323846;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_fft) {
        w[i] = (i < win) ? 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)win)) : 0.0f;
        return;
    }
    i -= n_fft;
    long total = (long)F * n_fft;
    if (i >= total) return;
    int k = (int)(i / n_fft);
    int n = (int)(i % n_fft);
    double ang = -2.0 * pi * ((double)n * (double)k) / (double)n_fft;
    Cc[i] = cos(ang);
    Cs[i] = sin(ang);
}

__global__ void k_stft_head(const double *re, const double *im, float *mag, float *pha, long n)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double r = re[i], m = im[i];
    mag[i] = (float)log1p(sqrt(r*r + m*m));
    pha[i] = (float)atan2(m, r);
}

extern "C" int reuse_gpu_stft(reuse_gpu_t *g, const float *d_wav, int T, int n_fft,
                              int hop, int win, int F, int num_frames,
                              float *d_mag, float *d_pha)
{
    // Mirrors reuse_stft_mag_pha() bit-for-bit in spirit: reflect-pad, raise-
    // cosine window, twiddles precomputed in double, double accumulation.
    // The reference accumulates in double on purpose - a single-precision sum
    // flips the sign of Im near the negative-real axis and moves whole bins by
    // 2*pi, and float twiddles cost ~1.8e-4 rad of phase, which the 30-block
    // tower amplifies down to 26 dB of output SNR.
    g->ws_off = 0;
    static int stft_dd = -1;
    if (stft_dd < 0) { const char *e = getenv("REUSE_STFT_DD"); stft_dd = (e && atoi(e) == 0) ? 0 : 1; }
    double *xw  = (double*)wallc(g, (size_t)num_frames*n_fft*2);
    double *Cc  = (double*)wallc(g, (size_t)F*n_fft*2);
    double *Cs  = (double*)wallc(g, (size_t)F*n_fft*2);
    double *dre = (double*)wallc(g, (size_t)F*num_frames*2);
    double *dim = (double*)wallc(g, (size_t)F*num_frames*2);
    float  *w   = (float*) wallc(g, (size_t)n_fft*2);   // 2 slots: the dd path reads it as pairs
    if (!xw || !Cc || !Cs || !dre || !dim || !w) return 1;
    /* Twiddles + window are generated ON DEVICE: the host loop (double
       cos/sin over F*n_fft = 824 KB of tables) cost 2.6 ms per call on the
       dev host - 25% of the whole STFT - and pushed 1.6 MB over PCIe.
       k_stft_twiddles computes the identical values (same double angle,
       cos/sin per element), so the DFT input is unchanged. */
    long pn = (long)num_frames*n_fft;
    long on = (long)F*num_frames;
    if (stft_dd) {
        // Same 8 bytes/element layout, read as (hi, lo) float pairs; see
        // k_stft_dft_dd for why this is accurate enough.
        dd *dxw = (dd*)xw, *dCc = (dd*)Cc, *dCs = (dd*)Cs, *dw = (dd*)w;
        k_stft_twiddles_dd<<<(unsigned)(((long)(F+2)*n_fft+255)/256), 256>>>(dw, dCc, dCs, F, n_fft, win);
        { cudaError_t e = cudaGetLastError(); if (e) { fprintf(stderr, "[stfterr] twiddles_dd %s\n", cudaGetErrorString(e)); return reuse_gpu_ok(g, "stft twiddles"); } }
        k_stft_prep_dd<<<(unsigned)((pn+255)/256), 256>>>(d_wav, T, dxw, n_fft, hop, n_fft/2, dw);
        k_stft_dft_dd<<<dim3((unsigned)((num_frames+15)/16), (unsigned)((F+15)/16)), dim3(16,16)>>>(
            dxw, dCc, dCs, dre, dim, F, num_frames, n_fft);
        k_stft_dc_dd<<<(unsigned)((num_frames+255)/256), 256>>>(dxw, dre, dim, num_frames, n_fft);
    } else {
    k_stft_twiddles<<<(unsigned)(((long)(F+2)*n_fft+255)/256), 256>>>(w, Cc, Cs, F, n_fft, win);
    { cudaError_t e = cudaGetLastError(); if (e) { fprintf(stderr, "[stfterr] twiddles %s\n", cudaGetErrorString(e)); return reuse_gpu_ok(g, "stft twiddles"); } }
    k_stft_prep<<<(unsigned)((pn+255)/256), 256>>>(d_wav, T, xw, n_fft, hop, n_fft/2, w);
    k_stft_dft<<<dim3((unsigned)((num_frames+15)/16), (unsigned)((F+15)/16)), dim3(16,16)>>>(
        xw, Cc, Cs, dre, dim, F, num_frames, n_fft);
    k_stft_dc<<<(unsigned)((num_frames+255)/256), 256>>>(xw, dre, dim, num_frames, n_fft);
    }
    k_stft_head<<<(unsigned)((on+255)/256), 256>>>(dre, dim, d_mag, d_pha, on);
    if (getenv("REUSE_DUMP_STFT_RAW")) {
        double *hr = (double*)malloc((size_t)on*sizeof(double)), *hi = (double*)malloc((size_t)on*sizeof(double));
        cudaMemcpy(hr, dre, (size_t)on*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(hi, dim, (size_t)on*sizeof(double), cudaMemcpyDeviceToHost);
        FILE *f = fopen(getenv("REUSE_DUMP_STFT_RAW"), "wb");
        fwrite(hr, sizeof(double), (size_t)on, f);
        fwrite(hi, sizeof(double), (size_t)on, f);
        fclose(f); free(hr); free(hi);
    }
    return reuse_gpu_ok(g, "stft_gpu");
}

__global__ void k_mag_pha_split(const float *mag, const float *pha, float *Cr, float *Ci, long n)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double m = (double)mag[i];
    if (m < 0.0) m = 0.0;
    double mv = exp(m) - 1.0;
    double p = (double)pha[i];
    Cr[i] = (float)(mv * cos(p));
    Ci[i] = (float)(mv * sin(p));
}

__global__ void k_build_input(const float *mag, const float *pha, float *xin, int T, int F)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long Tp = T + 2, Fp = F + 2;
    if (i >= (long)T*F) return;
    int f = (int)(i / T), t = (int)(i % T);
    xin[(long)t*Fp + f] = mag[i];
    xin[Tp*Fp + (long)t*Fp + f] = pha[i];
}

// phase = atan2(x_i, x_r) for the [F,T] crop
__global__ void k_atan2_head(const float *xr, const float *xi, float *out,
                             int Tx, int Fx, int T, int F)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)T*F) return;
    int f = (int)(i / T), t = (int)(i % T);
    out[(long)f*T + t] = atan2f(xi[(long)t*Fx + f], xr[(long)t*Fx + f]);
}

// sweep-artifact mask: zero frames whose >50% bins have expm1(relu(amp)) == 0
__global__ void k_sweep_mask(float *amp, int T, int F)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    int nz = 0;
    for (int f = 0; f < F; f++) {
        float a = amp[(long)f*T + t];
        if (a <= 0.0f) nz++;
    }
    if ((float)nz / (float)F > 0.5f)
        for (int f = 0; f < F; f++) amp[(long)f*T + t] = 0.0f;
}

// ================================================================= iSTFT (GPU)

// y[n][t] = (m_k / n_fft) * sum_k ( Cr[k][t]*cos(2*pi*n*k/N) - Ci[k][t]*sin(...) )
// with m_k = 1 for DC/Nyquist and 2 for interior bins (conjugate pair).
// Built as two real GEMMs against a float64-rounded DFT matrix so the result
// matches the double-accumulating CPU reference to fp32 rounding.
// Kahan (compensated) accumulation of one float term into (s, c).
// The sum keeps its own rounding error in c, so the running sum of N terms has
// an O(u*|sum|) error bound instead of O(N*u*|sum|) for a naive float sum.
#define KAHAN_ADD(s, c, v) do { float y_ = (v) - (c); float t_ = (s) + y_; \
                                (c) = ((t_) - (s)) - (y_); (s) = t_; } while (0)

// Fast path: the same GEMM accumulated in fp32 with Kahan compensation, and the
// exact product residual folded in. Each term v = a*b is split into (p, e) with
// p = a*b (rounded) and e = fmaf(a, b, -p) (exact, by Sterbenz/FMA), and BOTH
// parts are pushed through the compensated sum, so the products contribute
// exactly and only the additions round - 514 terms land at ~1e-7 relative
// (2x the float output rounding; the fp32 iSTFT atomics already sit at 9e-8),
// where a plain fp32 sum would be ~1e-5 and the fp64 form costs 64x the issue
// slots on Maxwell. REUSE_IDFT=0 restores the fp64 form for A/B.
__global__ void k_idft_k(const float *Cr, const float *Ci, const float *Cc, const float *Cs,
                         float *Y, int F, int T, int n_fft)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int t0 = blockIdx.y * 16;
    if (n >= n_fft) return;
    float acc[16], comp[16];
#pragma unroll
    for (int i2 = 0; i2 < 16; i2++) { acc[i2] = 0.0f; comp[i2] = 0.0f; }
    const int rem = T - t0;
    for (int k = 0; k < F; k++) {
        // tables are stored [k][n]: lanes run along n -> 128 B per warp
        const float cv = Cc[(long)k*n_fft + n], sv = Cs[(long)k*n_fft + n];
        const float *crp = Cr + (long)k*T + t0;
        const float *cip = Ci + (long)k*T + t0;
#pragma unroll 4
        for (int j2 = 0; j2 < 16; j2++) {
            if (j2 < rem) {
                float xr = crp[j2], xi = cip[j2];
                float p = cv * xr;
                KAHAN_ADD(acc[j2], comp[j2], p);
                KAHAN_ADD(acc[j2], comp[j2], fmaf(cv, xr, -p));
                float q = sv * xi;
                KAHAN_ADD(acc[j2], comp[j2], q);
                KAHAN_ADD(acc[j2], comp[j2], fmaf(sv, xi, -q));
            }
        }
    }
#pragma unroll
    for (int j2 = 0; j2 < 16; j2++)
        if (j2 < rem) Y[(long)n*T + t0 + j2] = acc[j2];
}

__global__ void k_idft(const float *Cr, const float *Ci, const float *Cc, const float *Cs,
                       float *Y, int F, int T, int n_fft)
{
    // Y[n][t] = sum_k Cr[k][t]*Cc[n][k] + Ci[k][t]*Cs[n][k], accumulated in double
    // (the CPU reference sums in double; float accumulation would not match).
    // Tables arrive in [k][n] order (coalesced for the lane-along-n pattern).
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int t0 = blockIdx.y * 16;
    if (n >= n_fft) return;
    double acc[16];
#pragma unroll
    for (int i2 = 0; i2 < 16; i2++) acc[i2] = 0.0;
    for (int k = 0; k < F; k++) {
        double cv = (double)Cc[(long)k*n_fft + n], sv = (double)Cs[(long)k*n_fft + n];
        const float *crp = Cr + (long)k*T + t0;
        const float *cip = Ci + (long)k*T + t0;
        int rem = T - t0;
#pragma unroll 4
        for (int j2 = 0; j2 < 16; j2++) {
            if (j2 < rem) {
                acc[j2] = fma(cv, (double)crp[j2], acc[j2]);
                acc[j2] = fma(sv, (double)cip[j2], acc[j2]);
            }
        }
    }
    int rem = T - t0;
#pragma unroll
    for (int j2 = 0; j2 < 16; j2++)
        if (j2 < rem) Y[(long)n*T + t0 + j2] = (float)acc[j2];
}

// overlap-add with the analysis window, then normalise by sum(w^2).
__global__ void k_ola(const float *Y, const float *w, float *out, float *wsum,
                      int F, int T, int n_fft, int hop, int pad, int out_len)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (n >= n_fft || t >= T) return;
    int idx = t*hop - pad + n;
    float wv = w[n];
    if (idx >= 0 && idx < out_len) {
        atomicAdd(&out[idx],  Y[(long)n*T + t] * wv);
        atomicAdd(&wsum[idx], wv * wv);
    }
}

/* Device-side iSTFT twiddle generation (same lever as k_stft_twiddles, #81):
   the host loop built Cc[n][k]/Cs[n][k] in double with a nested n/k loop
   (n_fft*F = 51610 elements per table, 2 tables = 413 KB) on every call, then
   copied them + the window over PCIe. Same values, same float storage. */
__global__ void k_istft_tables(float *Cc, float *Cs, float *w, int F, int n_fft, int win)
{
    const double pi = 3.14159265358979323846;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)n_fft * F;
    if (i < total) {
        // [k][n] layout: k_idft's lanes run along n, so the old [n][k] layout
        // put one useful float in every 32 B sector (168 MB of sector traffic
        // per call at 8 kHz). Same values, transposed index.
        int k = (int)(i / n_fft), n = (int)(i % n_fft);
        double m = (k == 0 || 2*k == n_fft) ? 1.0 : 2.0;
        double ang = 2.0 * pi * (double)n * (double)k / (double)n_fft;
        Cc[i] = (float)( m * cos(ang) / n_fft);
        Cs[i] = (float)(-m * sin(ang) / n_fft);
        return;
    }
    i -= total;
    if (i < (long)n_fft) {
        int n = (int)i;
        /* the shipped host loop filled w with 1.0f, then overwrote [0, win)
           with the hann window; the ternary reproduces that exactly */
        w[n] = (n < win) ? 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)n / (float)win)) : 1.0f;
        return;
    }
}

__global__ void k_norm_ola(float *out, const float *wsum, int out_len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_len && wsum[i] > 1e-12f) out[i] /= wsum[i];
}

extern "C" int reuse_gpu_istft(reuse_gpu_t *g, const float *d_mag, const float *d_pha,
                               int F, int T, int n_fft, int hop, int win,
                               float *d_out, int out_len)
{
    if (g->bad) return 1;
    /* tables are generated ON DEVICE (k_istft_tables): removes the host
       double-cos/sin loop and 826 KB of PCIe traffic per call (#81 lever). */
    /* One arena allocation instead of 7 cudaMalloc + 7 cudaFree per call:
       each driver call is a synchronized operation, measurable under the
       Nano's CUDA_LAUNCH_BLOCKING and ~15 us each on the dev host. */
    float *d_Cc, *d_Cs, *d_w, *d_Cr, *d_Ci, *d_Y, *d_wsum, *istft_base = NULL;
    size_t nf = (size_t)n_fft*F, ft = (size_t)F*T, nt = (size_t)n_fft*T;
    size_t istft_total = 2*nf + n_fft + 2*ft + nt + (size_t)out_len;
    istft_base = (float*)dalloc(istft_total);
    if (!istft_base) {
        fprintf(stderr, "istft: gpu OOM\n"); cudaGetLastError();
        return 1;
    }
    d_Cc = istft_base; d_Cs = d_Cc + nf; d_w = d_Cs + nf; d_Cr = d_w + n_fft;
    d_Ci = d_Cr + ft;  d_Y = d_Ci + ft;  d_wsum = d_Y + nt;
    k_istft_tables<<<(unsigned)(((long)nf+n_fft+255)/256), 256>>>(d_Cc, d_Cs, d_w, F, n_fft, win);
    { cudaError_t e = cudaGetLastError(); if (e) fprintf(stderr, "[istfterr] tables %s\n", cudaGetErrorString(e)); }
    static int idft_k = -1;
    if (idft_k < 0) { const char *e = getenv("REUSE_IDFT"); idft_k = (e && atoi(e) == 0) ? 0 : 1; }
    // mag -> linear (expm1(relu)) then split phase
    long nsp = ft;
    k_mag_pha_split<<<(unsigned)((nsp+255)/256), 256>>>(d_mag, d_pha, d_Cr, d_Ci, nsp);
    if (idft_k)
        k_idft_k<<<dim3((unsigned)((n_fft+127)/128), (unsigned)((T+15)/16)), 128>>>(
            d_Cr, d_Ci, d_Cc, d_Cs, d_Y, F, T, n_fft);
    else
        k_idft<<<dim3((unsigned)((n_fft+127)/128), (unsigned)((T+15)/16)), 128>>>(
            d_Cr, d_Ci, d_Cc, d_Cs, d_Y, F, T, n_fft);
    cudaMemset(d_out, 0, (size_t)out_len*sizeof(float));
    cudaMemset(d_wsum, 0, (size_t)out_len*sizeof(float));
    k_ola<<<dim3((unsigned)((n_fft+127)/128), (unsigned)T), dim3(128, 1)>>>(
        d_Y, d_w, d_out, d_wsum, F, T, n_fft, hop, n_fft/2, out_len);
    k_norm_ola<<<(unsigned)((out_len+255)/256), 256>>>(d_out, d_wsum, out_len);
    int rc = reuse_gpu_ok(g, "istft_gpu");
    dfree(istft_base);
    return rc;
}

// ================================================================= helpers


// Jetson's GPU watchdog kills a CUDA channel that stays continuously busy
// beyond ~10 s. REUSE_SYNC_EVERY=N issues a device sync every N launches so a
// long stream of kernels can be punctuated on Nano; 0 = never (dev hosts).
static int launch_kick(void)
{
    static int n = -1, every = 0;
    if (n < 0) { const char *e = getenv("REUSE_SYNC_EVERY"); every = e ? atoi(e) : 0; n = 0; }
    if (every <= 0) return 0;
    if (++n >= every) { n = 0; return cudaDeviceSynchronize(); }
    return 0;
}

// Register-tiled 3x3 dilated conv for the DenseBlock shapes.
//
// The naive kernel computes one output per thread and reads 9*C_in inputs for
// 9*C_in FMAs: one FMA per load, which is what pins it at ~160 GFLOPS (2.8% of
// peak) on the encoder's dense block (DEPTH=4, C_in = 64/128/192/256, 1023x163
// = 121 GFlop, 0.754 s per forward = 45% of the model). Here a block owns a
// 16x32 output tile, stages the (16+2*dilH) x 34 input window for one input
// channel in shared, and each thread keeps 2 row accumulators, so the block
// issues 1088 global loads for 4608 FMAs: 4.2 FMA per load.
//
// R is the tile height plus the two dilated halos, so the shared row for tap
// kh of output row oh is exactly oh + kh*dilH. Warp lanes run along x with a
// fixed row, so both the shared reads (S[sh][tx+kw]) and the y writes are
// contiguous; the 34-float row stride keeps shared banks conflict-free.
// Restricted to the shapes the dense block uses; everything else takes the
// naive kernel, and REUSE_NAIVE_CONV2D=1 forces that for A/B.
template<int R>
__global__ void k_conv2d_r(const float *x, const float *w, const float *b,
                           int C_in, int H, int W, int padH, int padW, int dilH,
                           float *y, int co0)
{
    const int TW = 32, LS = TW+4;   // multiple of 4: aligned float4 tile stores
    __shared__ __align__(16) float S[R][LS];
    const int co = co0 + blockIdx.z;
    const int w0 = blockIdx.x*TW, h0 = blockIdx.y*16;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty*32 + tx;
    float acc[2] = { b[co], b[co] };
    for (int ci = 0; ci < C_in; ci++) {
        const float *xin = x + (long)ci*H*W;
        __syncthreads();
        // One index/mask computation per 4 elements: the scalar form spends
        // ~12 instructions (divide by LS, two bounds pairs, select, address)
        // on EVERY element, i.e. 20.6 x 12 ~ 250 issue slots per thread per
        // input channel against 144 FMAs. Measured ceilings for this kernel:
        // FMA 4%, LSU 15%, DRAM 15% of peak, so loader issue was the limit.
        {
            const int LSF = LS/4, wbase = w0 - padW, hbase = h0 - padH;
            float4 *dv = (float4*)&S[0][0];
            for (int i = tid; i < R*LSF; i += 256) {
                const int r = i / LSF, q = i - r*LSF;
                const int hh = hbase + r, ww = wbase + q*4;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (hh >= 0 && hh < H) {
                    const float *sp = xin + (long)hh*W + ww;
                    // the vector load needs the global element index
                    // 16B-aligned; with odd W (163 here) that is row
                    // dependent, so the scalar path covers the rest.
                    if (ww >= 0 && ww + 4 <= W &&
                        ((((long)ci*H*W + (long)hh*W + ww) & 3) == 0))
                        v = *(const float4*)sp;
                    else {
                        if (ww     >= 0 && ww     < W) v.x = sp[0];
                        if (ww + 1 >= 0 && ww + 1 < W) v.y = sp[1];
                        if (ww + 2 >= 0 && ww + 2 < W) v.z = sp[2];
                        if (ww + 3 >= 0 && ww + 3 < W) v.w = sp[3];
                    }
                }
                dv[i] = v;
            }
        }
        __syncthreads();
        const float *wv = w + ((long)co*C_in + ci)*9;
        float a0 = 0.f, a1 = 0.f;
#pragma unroll
        for (int kh = 0; kh < 3; kh++) {
            const int sh0 = ty*2 + kh*dilH, sh1 = ty*2 + 1 + kh*dilH;
            const float w0f = wv[kh*3], w1f = wv[kh*3+1], w2f = wv[kh*3+2];
            a0 = fmaf(S[sh0][tx],   w0f, a0); a1 = fmaf(S[sh1][tx],     w0f, a1);
            a0 = fmaf(S[sh0][tx+1], w1f, a0); a1 = fmaf(S[sh1][tx+1],   w1f, a1);
            a0 = fmaf(S[sh0][tx+2], w2f, a0); a1 = fmaf(S[sh1][tx+2],   w2f, a1);
        }
        acc[0] += a0; acc[1] += a1;
    }
    const int ww = w0 + tx;
    if (ww < W) {
        for (int rr = 0; rr < 2; rr++) {
            int hh = h0 + ty*2 + rr;
            if (hh < H) y[(long)co*H*W + (long)hh*W + ww] = acc[rr];
        }
    }
}

// 64x64-tile version of the register-tiled conv: 16 outputs per thread
// (4 rows x 4 cols). The 16x32 tile used 2 row accumulators, so each thread
// issued one shared load per FMA (9 taps x 2 rows = 18 shared loads for 18
// FMAs), which is the shared-memory ceiling and is what kept the dense block
// at ~370 GFLOPS. With 4 rows a thread reads 4 shared values (+1 block-uniform
// weight) per 16 FMAs, and the bigger tile raises the global-load ratio from
// 4.2 to 7.0 FMA/load.
/* Dense-block concat mode: the conv input is the concatenation of ng channel
   groups living in slot planes plus the caller's x_in tail. The 16 per-group
   base pointers ride as kernel PARAMS (constant bank 0, zero loads, uniform).
   A __device__ table version worked on sm_61 but corrupted output on sm_53
   (#72/#73); params are the codegen-robust delivery. */
#define CONV_SRCS_PARAMS const float *s0, const float *s1, const float *s2, const float *s3, \
    const float *s4, const float *s5, const float *s6, const float *s7, \
    const float *s8, const float *s9, const float *s10, const float *s11, \
    const float *s12, const float *s13, const float *s14, const float *s15
#define CONV_SRCS_ARGS s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15
__device__ __forceinline__ const float *conv_src_pick(int g,
    const float *s0, const float *s1, const float *s2, const float *s3,
    const float *s4, const float *s5, const float *s6, const float *s7,
    const float *s8, const float *s9, const float *s10, const float *s11,
    const float *s12, const float *s13, const float *s14, const float *s15)
{
    switch (g) {
    case 0: return s0;  case 1: return s1;  case 2: return s2;  case 3: return s3;
    case 4: return s4;  case 5: return s5;  case 6: return s6;  case 7: return s7;
    case 8: return s8;  case 9: return s9;  case 10: return s10; case 11: return s11;
    case 12: return s12; case 13: return s13; case 14: return s14; default: return s15;
    }
}

template<int R, bool TABLE = false>
__global__ void k_conv2d_r2(const float *x, const float *w, const float *b,
                            int C_in, int H, int W, int padH, int padW, int dilH,
                            float *y, int co0, const float *x2, int ng, CONV_SRCS_PARAMS)
{
    const int TW = 64, LS = TW+4;   // multiple of 4: aligned float4 tile stores
    __shared__ __align__(16) float S[R][LS];
    const int co = co0 + blockIdx.z;
    const int w0 = blockIdx.x*TW, h0 = blockIdx.y*64;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty*16 + tx;
    float acc[4][4];
#pragma unroll
    for (int a = 0; a < 4; a++)
#pragma unroll
        for (int bq = 0; bq < 4; bq++) acc[a][bq] = b[co];
    for (int ci = 0; ci < C_in; ci++) {
        const float *xin; long pl;
        if (!TABLE) { xin = x + (long)ci*H*W; pl = ci; }
        else {
            const int g = ci/HID;
            xin = conv_src_pick(g, CONV_SRCS_ARGS) + (long)(ci%HID)*H*W;
            pl = (g < ng) ? (long)(ng-1-g)*HID + ci%HID : (long)(g-ng)*HID + ci%HID;
        }
        __syncthreads();
        // One index/mask computation per 4 elements: the scalar form spends
        // ~12 instructions (divide by LS, two bounds pairs, select, address)
        // on EVERY element, i.e. 20.6 x 12 ~ 250 issue slots per thread per
        // input channel against 144 FMAs. Measured ceilings for this kernel:
        // FMA 4%, LSU 15%, DRAM 15% of peak, so loader issue was the limit.
        {
            const int LSF = LS/4, wbase = w0 - padW, hbase = h0 - padH;
            float4 *dv = (float4*)&S[0][0];
            for (int i = tid; i < R*LSF; i += 256) {
                const int r = i / LSF, q = i - r*LSF;
                const int hh = hbase + r, ww = wbase + q*4;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (hh >= 0 && hh < H) {
                    const float *sp = xin + (long)hh*W + ww;
                    // the vector load needs the global element index
                    // 16B-aligned; with odd W (163 here) that is row
                    // dependent, so the scalar path covers the rest.
                    if (ww >= 0 && ww + 4 <= W &&
                        ((((long)pl*H*W + (long)hh*W + ww) & 3) == 0))
                        v = *(const float4*)sp;
                    else {
                        if (ww     >= 0 && ww     < W) v.x = sp[0];
                        if (ww + 1 >= 0 && ww + 1 < W) v.y = sp[1];
                        if (ww + 2 >= 0 && ww + 2 < W) v.z = sp[2];
                        if (ww + 3 >= 0 && ww + 3 < W) v.w = sp[3];
                    }
                }
                dv[i] = v;
            }
        }
        __syncthreads();
        const float *wv = w + ((long)co*C_in + ci)*9;
        // output (row h0+ty*4+a, col w0+tx*4+bq) reads tile row
        // ty*4+a+kh*dilH and tile col tx*4+bq+kw (the +1 pad cancels padW=1)
#pragma unroll
        for (int kh = 0; kh < 3; kh++) {
            const float w0f = wv[kh*3], w1f = wv[kh*3+1], w2f = wv[kh*3+2];
            const int rb = ty*4 + kh*dilH, cb = tx*4;
#pragma unroll
            for (int a = 0; a < 4; a++) {
                const float *sr = S[rb + a];
#pragma unroll
                for (int bq = 0; bq < 4; bq++) {
                    acc[a][bq] = fmaf(sr[cb + bq],     w0f, acc[a][bq]);
                    acc[a][bq] = fmaf(sr[cb + bq + 1], w1f, acc[a][bq]);
                    acc[a][bq] = fmaf(sr[cb + bq + 2], w2f, acc[a][bq]);
                }
            }
        }
    }
    const int wc = w0 + tx*4;
    float *yo = y + (long)co*H*W;
#pragma unroll
    for (int a = 0; a < 4; a++) {
        int hh = h0 + ty*4 + a;
        if (hh >= H) continue;
        float *dst = yo + (long)hh*W;
#pragma unroll
        for (int bq = 0; bq < 4; bq++) if (wc + bq < W) dst[wc + bq] = acc[a][bq];
    }
}

// Double-buffered register-tiled conv, 32 rows x 64 cols, 8 outputs/thread.
//
// The 64x64 kernel runs, per input channel, load-phase -> barrier ->
// compute-phase -> barrier, 256 times, with nothing saturated: FMA issue ~4%
// of peak, LSU ~15%, DRAM ~15% of the measured 158 GB/s. That is the
// signature of exposed memory latency across the barrier, so this variant
// issues the ci+1 tile loads before computing ci, from a second buffer.
// The tile is 32x64 (not 64x64) so the two buffers fit in 25 KB and keep two
// blocks resident per SM.
template<int R>
__global__ void k_conv2d_r3(const float *x, const float *w, const float *b,
                            int C_in, int H, int W, int padH, int padW, int dilH,
                            float *y, int co0)
{
    const int TW = 64, LS = TW+4;   // multiple of 4: aligned float4 tile stores
    __shared__ float S[2][R][LS];
    const int co = co0 + blockIdx.z;
    const int w0 = blockIdx.x*TW, h0 = blockIdx.y*32;
    const int tx = threadIdx.x, ty = threadIdx.y;                 // (16,16)
    const int tid = ty*16 + tx;
    float acc[2][4];
#pragma unroll
    for (int a = 0; a < 2; a++)
#pragma unroll
        for (int bq = 0; bq < 4; bq++) acc[a][bq] = b[co];
    int par = 0;
    for (int ci = 0; ci <= C_in; ci++) {
        if (ci < C_in) {
            const float *xin = x + (long)ci*H*W;
            float *dst = (float*)S[par];
            for (int i = tid; i < R*LS; i += 256) {
                int r = i / LS, c = i % LS;
                int hh = h0 - padH + r, ww = w0 - padW + c;
                dst[r*LS + c] = (hh >= 0 && hh < H && ww >= 0 && ww < W)
                              ? xin[(long)hh*W + ww] : 0.f;
            }
        }
        __syncthreads();
        if (ci >= C_in) break;
        const float *src = (const float*)S[par];
        const float *wv = w + ((long)co*C_in + ci)*9;
#pragma unroll
        for (int kh = 0; kh < 3; kh++) {
            const float w0f = wv[kh*3], w1f = wv[kh*3+1], w2f = wv[kh*3+2];
            const int rb = ty*2 + kh*dilH, cb = tx*4;
#pragma unroll
            for (int a = 0; a < 2; a++) {
                const float *sr = src + (rb + a)*LS + cb;
                acc[a][0] = fmaf(sr[0], w0f, acc[a][0]); acc[a][0] = fmaf(sr[1], w1f, acc[a][0]); acc[a][0] = fmaf(sr[2], w2f, acc[a][0]);
                acc[a][1] = fmaf(sr[1], w0f, acc[a][1]); acc[a][1] = fmaf(sr[2], w1f, acc[a][1]); acc[a][1] = fmaf(sr[3], w2f, acc[a][1]);
                acc[a][2] = fmaf(sr[2], w0f, acc[a][2]); acc[a][2] = fmaf(sr[3], w1f, acc[a][2]); acc[a][2] = fmaf(sr[4], w2f, acc[a][2]);
                acc[a][3] = fmaf(sr[3], w0f, acc[a][3]); acc[a][3] = fmaf(sr[4], w1f, acc[a][3]); acc[a][3] = fmaf(sr[5], w2f, acc[a][3]);
            }
        }
        par ^= 1;
    }
    const int wc = w0 + tx*4;
    float *yo = y + (long)co*H*W;
#pragma unroll
    for (int a = 0; a < 2; a++) {
        int hh = h0 + ty*2 + a;
        if (hh >= H) continue;
        float *d2 = yo + (long)hh*W;
#pragma unroll
        for (int bq = 0; bq < 4; bq++) if (wc + bq < W) d2[wc + bq] = acc[a][bq];
    }
}

template<int R>
__global__ void k_conv2d_r5(const float *x, const float *w, const float *b,
                            int C_in, int H, int W, int padH, int padW, int dilH,
                            float *y, int co0)
{
    const int TW = 64, LS = TW+4;   // multiple of 4: aligned float4 tile stores
    __shared__ __align__(16) float S[R][LS];
    const int co = co0 + blockIdx.z;
    const int w0 = blockIdx.x*TW, h0 = blockIdx.y*32;
    const int tx = threadIdx.x, ty = threadIdx.y;                 // (16,16)
    const int tid = ty*16 + tx;
    float acc[2][4];
#pragma unroll
    for (int a = 0; a < 2; a++)
#pragma unroll
        for (int bq = 0; bq < 4; bq++) acc[a][bq] = b[co];
    for (int ci = 0; ci < C_in; ci++) {
        if (ci) __syncthreads();
        {
            const float *xin = x + (long)ci*H*W;
            const int RS = LS/4, wbase = w0 - padW, hbase = h0 - padH;
            float4 *dv = (float4*)&S[0][0];
            for (int i = tid; i < R*RS; i += 256) {
                const int r = i / RS, q = 4*(i - r*RS);
                const int hh = hbase + r, ww = wbase + q;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (hh >= 0 && hh < H) {
                    const float *sp = xin + (long)hh*W + ww;
                    if (ww >= 0 && ww + 4 <= W && ((((long)ci*H*W + (long)hh*W + ww) & 3) == 0))
                        v = *(const float4*)sp;
                    else {
                        if (ww     >= 0 && ww     < W) v.x = sp[0];
                        if (ww + 1 >= 0 && ww + 1 < W) v.y = sp[1];
                        if (ww + 2 >= 0 && ww + 2 < W) v.z = sp[2];
                        if (ww + 3 >= 0 && ww + 3 < W) v.w = sp[3];
                    }
                }
                dv[i] = v;
            }
        }
        __syncthreads();
        const float *src = &S[0][0];
        const float *wv = w + ((long)co*C_in + ci)*9;
#pragma unroll
        for (int kh = 0; kh < 3; kh++) {
            const float w0f = wv[kh*3], w1f = wv[kh*3+1], w2f = wv[kh*3+2];
            const int rb = ty*2 + kh*dilH, cb = tx*4;
#pragma unroll
            for (int a = 0; a < 2; a++) {
                const float *sr = src + (rb + a)*LS + cb;
                acc[a][0] = fmaf(sr[0], w0f, acc[a][0]); acc[a][0] = fmaf(sr[1], w1f, acc[a][0]); acc[a][0] = fmaf(sr[2], w2f, acc[a][0]);
                acc[a][1] = fmaf(sr[1], w0f, acc[a][1]); acc[a][1] = fmaf(sr[2], w1f, acc[a][1]); acc[a][1] = fmaf(sr[3], w2f, acc[a][1]);
                acc[a][2] = fmaf(sr[2], w0f, acc[a][2]); acc[a][2] = fmaf(sr[3], w1f, acc[a][2]); acc[a][2] = fmaf(sr[4], w2f, acc[a][2]);
                acc[a][3] = fmaf(sr[3], w0f, acc[a][3]); acc[a][3] = fmaf(sr[4], w1f, acc[a][3]); acc[a][3] = fmaf(sr[5], w2f, acc[a][3]);
            }
        }
    }
    const int wc = w0 + tx*4;
    float *yo = y + (long)co*H*W;
#pragma unroll
    for (int a = 0; a < 2; a++) {
        int hh = h0 + ty*2 + a;
        if (hh >= H) continue;
        float *d2 = yo + (long)hh*W;
#pragma unroll
        for (int bq = 0; bq < 4; bq++) if (wc + bq < W) d2[wc + bq] = acc[a][bq];
    }
}


// 32 rows x 64 cols x 4 output channels per block.
//
// k_conv2d_r2 launches ONE output channel per block (its gridZ is the co
// chunk), so the shared input window is re-read by every output channel: for
// the encoder's C_in=256 conv that is ~8.7 GB of DRAM reads per launch, and
// it is why a conv runs at 0.55 TFLOPS (11% of peak) while the project's
// plain GEMMs sit at 2-4.4 TFLOPS. Sharing one staged window across 4 co cuts
// global traffic 1.7x (the tile is half as tall), raises FMA per global load
// 7 -> 22 and FMA per shared load 2 -> 6, and keeps registers at 32
// accumulators + ~12.
template<int R, int PCO, int NCB, int NRA = 32/(PCO*NCB), bool TABLE = false, bool WS = false>
__global__ void k_conv2d_r8(const float *x, const float *w, const float *b,
                            int C_in, int H, int W, int padH, int padW, int dilH,
                            float *y, int co0, const float *x2, int ng, CONV_SRCS_PARAMS)
{
    /* Weight-shared staging (WS=true, REUSE_WST=1).
     *
     * The (j,kh) weight wv[j*C_in*9 + kh*3 + t] depends only on the output
     * channel group and tap - it is IDENTICAL for all 256 threads of the
     * block - but the shipped loop has every thread load its PCO*9 weights
     * privately as scalar LDGs (256x redundancy). With WS, one cooperative
     * pass per ci stages the PCO*3 taps into Ws[j][kh] (padded to float4 so
     * every slot is 16 B aligned) and the loop reads them back as LDS.128:
     * PCO*3 shared loads replace PCO*9 global loads per ci per thread. The
     * staging rides the existing per-ci __syncthreads pair, so no new syncs,
     * and the FMAs run in exactly the shipped order (same floats), making
     * output bit-identical to WS=false. WS=false instantiations fold the
     * branch and ternaries away and are codegen-identical to shipped. */
    const int TW = 16*NCB, LS = TW+4;
    __shared__ __align__(16) float S[R][LS];
    __shared__ __align__(16) float Ws[16][3][4];
    const int w0 = blockIdx.x*TW, h0 = blockIdx.y*(16*NRA);
    const int tx = threadIdx.x, ty = threadIdx.y;                 // (16,16)
    const int tid = ty*16 + tx;
    float acc[PCO][NRA][NCB];
#pragma unroll
    for (int j = 0; j < PCO; j++)
#pragma unroll
        for (int a = 0; a < NRA; a++)
#pragma unroll
            for (int bq = 0; bq < NCB; bq++) acc[j][a][bq] = b[co0 + blockIdx.z*PCO + j];
    for (int ci = 0; ci < C_in; ci++) {
        const float *xin; long pl;
        if (!TABLE) { xin = x + (long)ci*H*W; pl = ci; }
        else {
            const int g = ci/HID;
            xin = conv_src_pick(g, CONV_SRCS_ARGS) + (long)(ci%HID)*H*W;
            pl = (g < ng) ? (long)(ng-1-g)*HID + ci%HID : (long)(g-ng)*HID + ci%HID;
        }
        if (ci) __syncthreads();
        {
            const int RS = LS/4, wbase = w0 - padW, hbase = h0 - padH;
            float4 *dv = (float4*)&S[0][0];
            for (int i = tid; i < R*RS; i += 256) {
                const int r = i / RS, q = 4*(i - r*RS);
                const int hh = hbase + r, ww = wbase + q;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (hh >= 0 && hh < H) {
                    const float *sp = xin + (long)hh*W + ww;
                    if (ww >= 0 && ww + 4 <= W && ((((long)pl*H*W + (long)hh*W + ww) & 3) == 0))
                        v = *(const float4*)sp;
                    else {
                        if (ww     >= 0 && ww     < W) v.x = sp[0];
                        if (ww + 1 >= 0 && ww + 1 < W) v.y = sp[1];
                        if (ww + 2 >= 0 && ww + 2 < W) v.z = sp[2];
                        if (ww + 3 >= 0 && ww + 3 < W) v.w = sp[3];
                    }
                }
                dv[i] = v;
            }
        }
        if (WS) {
            const float *wbs = w + ((long)(co0 + blockIdx.z*PCO)*C_in + ci)*9;
            for (int i = tid; i < PCO*3; i += 256) {
                const int j = i / 3, kh = i - j*3;
                Ws[j][kh][0] = wbs[(long)j*C_in*9 + kh*3];
                Ws[j][kh][1] = wbs[(long)j*C_in*9 + kh*3 + 1];
                Ws[j][kh][2] = wbs[(long)j*C_in*9 + kh*3 + 2];
            }
        }
        __syncthreads();
        const float *wv = w + ((long)(co0 + blockIdx.z*PCO)*C_in + ci)*9;
#pragma unroll
        for (int j = 0; j < PCO; j++) {
#pragma unroll
            for (int kh = 0; kh < 3; kh++) {
                float4 ww4 = make_float4(0.f, 0.f, 0.f, 0.f);
                if (WS) ww4 = *(const float4*)&Ws[j][kh];
                const float w0f = WS ? ww4.x : wv[j*C_in*9 + kh*3], w1f = WS ? ww4.y : wv[j*C_in*9 + kh*3+1], w2f = WS ? ww4.z : wv[j*C_in*9 + kh*3+2];
                const int rb = ty*NRA + kh*dilH, cb = tx*NCB;
#pragma unroll
                for (int a = 0; a < NRA; a++) {
                    const float *sr = &S[rb + a][cb];
#pragma unroll
                    for (int bq = 0; bq < NCB; bq++) {
                        acc[j][a][bq] = fmaf(sr[bq], w0f, acc[j][a][bq]);
                        acc[j][a][bq] = fmaf(sr[bq+1], w1f, acc[j][a][bq]);
                        acc[j][a][bq] = fmaf(sr[bq+2], w2f, acc[j][a][bq]);
                    }
                }
            }
        }
    }
    const int wc = w0 + tx*NCB;
#pragma unroll
    for (int j = 0; j < PCO; j++) {
        float *yo = y + (long)(co0 + blockIdx.z*PCO + j)*H*W;
#pragma unroll
        for (int a = 0; a < NRA; a++) {
            int hh = h0 + ty*NRA + a;
            if (hh >= H) continue;
            float *d2 = yo + (long)hh*W;
#pragma unroll
            for (int bq = 0; bq < NCB; bq++) if (wc + bq < W) d2[wc + bq] = acc[j][a][bq];
        }
    }
}

// ============================================================================
// Paired mask/phase decoder (REUSE_DEC_PAIR).
// The mask and phase decoders run the SAME chain shape on the SAME tower
// output `enc`, so every launch in the tail runs both heads with one grid:
// head h is blockIdx.z / znb and reads the head-pair buffer at +h*hoff
// (all paired tensors are head-contiguous: single wallc of 2*size).
// One uniform head SEL in the prologue; per-head weights arrive pre-offset
// (blob neighbours, verified by the caller), so the kernel body is the
// shipped one and the plain (PAIR=false) instantiations are untouched.

template<int RW, int RC, int TW, int NC>
__global__ void k_sp_up8t2(const float *x, const float *wT, const float *b0, const float *b1,
                           float *y, int H, int W, int r, int CO,
                           long hoff_in, long hoff_out, int zheads, long hoff_wt)
{
    const int TXN = TW/RW, TBY = NC/RC, NT = TXN*TBY;
    const int hd = blockIdx.z / zheads;
    const int w0 = blockIdx.x*TW;
    const int h = blockIdx.y;
    const int co0 = (blockIdx.z % zheads)*NC;
    const float *xb = x + (long)hd * hoff_in;
    const float *wtb = wT + (long)hd * hoff_wt;
    const float *bn = hd ? b1 : b0;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty*TXN + tx;
    __shared__ __align__(16) float xs[TW+8];
    __shared__ __align__(16) float ws[NC*3 + 4];
    const int wl = tx*RW, cl = ty*RC;
    float acc[RC][RW];
#pragma unroll
    for (int j = 0; j < RC; j++)
#pragma unroll
        for (int i = 0; i < RW; i++) acc[j][i] = bn[co0 + cl + j];
    const long hw = (long)h*W;
    for (int ci = 0; ci < HID; ci++) {
        const float *xv = xb + (long)ci*H*W + hw;
        for (int i = tid; i < (TW+3)/4 + 1; i += NT) {
            const int q = 4*i;
            float4 v = make_float4(0.f,0.f,0.f,0.f);
            const int s0 = w0 - 1 + q;
            if (q+0 < TW+3) v.x = (s0+0 >= 0 && s0+0 < W) ? xv[s0+0] : 0.f;
            if (q+1 < TW+3) v.y = (s0+1 >= 0 && s0+1 < W) ? xv[s0+1] : 0.f;
            if (q+2 < TW+3) v.z = (s0+2 >= 0 && s0+2 < W) ? xv[s0+2] : 0.f;
            if (q+3 < TW+3) v.w = (s0+3 >= 0 && s0+3 < W) ? xv[s0+3] : 0.f;
            *(float4*)&xs[q] = v;
        }
        { const float4 *wp = (const float4*)(wtb + ((long)ci*CO + co0)*3);
          for (int i = tid; i < NC*3/4; i += NT) *(float4*)&ws[4*i] = wp[i]; }
        __syncthreads();
        const float4 a0 = *(const float4*)&xs[wl];
        const float4 a1 = *(const float4*)&xs[wl+4];
        const float4 a2 = *(const float4*)&xs[wl+8];
        const float xv2[16] = { a0.x,a0.y,a0.z,a0.w, a1.x,a1.y,a1.z,a1.w, a2.x,a2.y,a2.z,a2.w, 0,0,0,0 };
#pragma unroll
        for (int j = 0; j < RC; j++) {
            const float c0 = ws[3*(cl+j)], c1 = ws[3*(cl+j)+1], c2 = ws[3*(cl+j)+2];
#pragma unroll
            for (int i = 0; i < RW; i++) {
                acc[j][i] = fmaf(xv2[i],   c0, acc[j][i]);
                acc[j][i] = fmaf(xv2[i+1], c1, acc[j][i]);
                acc[j][i] = fmaf(xv2[i+2], c2, acc[j][i]);
            }
        }
        __syncthreads();
    }
    float *yb = y + (long)hd * hoff_out;
#pragma unroll
    for (int j = 0; j < RC; j++) {
        const int cj = (co0 + cl + j) % HID, aj = (co0 + cl + j) / HID;
        float *yp = yb + ((long)cj*H + h)*(long)(W*r) + (long)(w0+wl)*r + aj;
#pragma unroll
        for (int i = 0; i < RW; i++) if (w0 + wl + i < W) yp[i*r] = acc[j][i];
    }
}

template<int R, int PCO, int NCB, int NRA, bool TABLE, bool PAIR = false>
__global__ void k_conv2d_r8p(const float *x, const float *w, const float *b,
                             const float *w2, const float *b2,
                             int C_in, int H, int W, int padH, int padW, int dilH,
                             float *y, int co0, const float *x2, int ng, CONV_SRCS_PARAMS,
                             long hoff_out, int znb, long hoff_x)
{
    const int TW = 16*NCB, LS = TW+4;
    __shared__ __align__(16) float S[R][LS];
    const int hd = PAIR ? blockIdx.z / znb : 0;
    const int w0 = blockIdx.x*TW, h0 = blockIdx.y*(16*NRA);
    const int tx = threadIdx.x, ty = threadIdx.y;                 // (16,16)
    const int tid = ty*16 + tx;
    const float *xh = x + (long)hd * (PAIR ? hoff_x : 0L);
    float *yh = y + (long)hd * (PAIR ? hoff_out : 0L);
    const float *wb = PAIR ? (hd ? w2 : w) : w;
    const float *bb = PAIR ? (hd ? b2 : b) : b;
    float acc[PCO][NRA][NCB];
#pragma unroll
    for (int j = 0; j < PCO; j++)
#pragma unroll
        for (int a = 0; a < NRA; a++)
#pragma unroll
            for (int bq = 0; bq < NCB; bq++) acc[j][a][bq] = bb[co0 + (blockIdx.z % znb)*PCO + j];
    for (int ci = 0; ci < C_in; ci++) {
        const float *xin; long pl;
        if (!TABLE) { xin = xh + (long)ci*H*W; pl = ci; }
        else {
            const int g = ci/HID;
            // head1 shifts the SLOT groups only; x2 (the tower tensor) is shared
            const float *pg = conv_src_pick(g, CONV_SRCS_ARGS) + (PAIR && hd && g < ng ? hd*hoff_x : 0L);
            xin = pg + (long)(ci%HID)*H*W;
            pl = (g < ng) ? (long)(ng-1-g)*HID + ci%HID : (long)(g-ng)*HID + ci%HID;
        }
        if (ci) __syncthreads();
        {
            const int RS = LS/4, wbase = w0 - padW, hbase = h0 - padH;
            float4 *dv = (float4*)&S[0][0];
            for (int i = tid; i < R*RS; i += 256) {
                const int r = i / RS, q = 4*(i - r*RS);
                const int hh = hbase + r, ww = wbase + q;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (hh >= 0 && hh < H) {
                    const float *sp = xin + (long)hh*W + ww;
                    if (ww >= 0 && ww + 4 <= W && ((((long)pl*H*W + (long)hh*W + ww) & 3) == 0))
                        v = *(const float4*)sp;
                    else {
                        if (ww     >= 0 && ww     < W) v.x = sp[0];
                        if (ww + 1 >= 0 && ww + 1 < W) v.y = sp[1];
                        if (ww + 2 >= 0 && ww + 2 < W) v.z = sp[2];
                        if (ww + 3 >= 0 && ww + 3 < W) v.w = sp[3];
                    }
                }
                dv[i] = v;
            }
        }
        __syncthreads();
        const float *wv = wb + ((long)(co0 + (blockIdx.z % znb)*PCO)*C_in + ci)*9;
#pragma unroll
        for (int j = 0; j < PCO; j++) {
#pragma unroll
            for (int kh = 0; kh < 3; kh++) {
                const float w0f = wv[j*C_in*9 + kh*3], w1f = wv[j*C_in*9 + kh*3+1], w2f = wv[j*C_in*9 + kh*3+2];
                const int rb = ty*NRA + kh*dilH, cb = tx*NCB;
#pragma unroll
                for (int a = 0; a < NRA; a++) {
                    const float *sr = &S[rb + a][cb];
#pragma unroll
                    for (int bq = 0; bq < NCB; bq++) {
                        acc[j][a][bq] = fmaf(sr[bq], w0f, acc[j][a][bq]);
                        acc[j][a][bq] = fmaf(sr[bq+1], w1f, acc[j][a][bq]);
                        acc[j][a][bq] = fmaf(sr[bq+2], w2f, acc[j][a][bq]);
                    }
                }
            }
        }
    }
    const int wc = w0 + tx*NCB;
#pragma unroll
    for (int j = 0; j < PCO; j++) {
        float *yo = yh + (long)(co0 + (blockIdx.z % znb)*PCO + j)*H*W;
#pragma unroll
        for (int a = 0; a < NRA; a++) {
            int hh = h0 + ty*NRA + a;
            if (hh >= H) continue;
            float *d2 = yo + (long)hh*W;
#pragma unroll
            for (int bq = 0; bq < NCB; bq++) if (wc + bq < W) d2[wc + bq] = acc[j][a][bq];
        }
    }
}

__global__ void k_inorm_prelu2(const float *x, float *y, int C, int HW, long hoff,
                               const float *w0, const float *b0, const float *a0,
                               const float *w1, const float *b1, const float *a1, float eps)
{
    const int hd = blockIdx.x / C;
    const int nc = blockIdx.x % C;
    const float *ws = hd ? w1 : w0;
    const float *bs = hd ? b1 : b0;
    const float *as = hd ? a1 : a0;
    const float *xs = x + (long)hd * hoff + (long)nc * HW;
    float *ys = y + (long)hd * hoff + (long)nc * HW;
    __shared__ float red[INORM_THREADS * 2];
    __shared__ float s_mean, s_inv;

    float s = 0.0f, c = 0.0f;
    for (int i = threadIdx.x; i < HW; i += INORM_THREADS) kahan_add(&s, &c, xs[i]);
    red[threadIdx.x*2] = s; red[threadIdx.x*2+1] = c;
    __syncthreads();
    for (int off = INORM_THREADS/2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            red[threadIdx.x*2]   += red[(threadIdx.x+off)*2];
            red[threadIdx.x*2+1] += red[(threadIdx.x+off)*2+1];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) s_mean = (red[0] + red[1]) / HW;
    __syncthreads();
    float mean = s_mean;

    float q = 0.0f, cq = 0.0f;
    for (int i = threadIdx.x; i < HW; i += INORM_THREADS) {
        float d = xs[i] - mean;
        kahan_add(&q, &cq, d*d);
    }
    red[threadIdx.x*2] = q; red[threadIdx.x*2+1] = cq;
    __syncthreads();
    for (int off = INORM_THREADS/2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            red[threadIdx.x*2]   += red[(threadIdx.x+off)*2];
            red[threadIdx.x*2+1] += red[(threadIdx.x+off)*2+1];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) s_inv = rsqrtf(fmaxf((red[0] + red[1]) / HW, 0.0f) + eps);
    __syncthreads();
    float inv = s_inv;
    float ww = ws ? ws[nc % C] : 1.0f;
    float bb = bs ? bs[nc % C] : 0.0f;
    float aa = as ? as[nc % C] : 0.25f;
    for (int i = threadIdx.x; i < HW; i += INORM_THREADS) {
        float v = (xs[i] - mean) * inv * ww + bb;
        ys[i] = v < 0.0f ? aa * v : v;
    }
}

__global__ void k_transpose2(const float *__restrict__ x, float *__restrict__ y, int C, int H, int W, long hoff)
{
    __shared__ float tile[32][33];
    const int hd = blockIdx.z / C;
    const int c = blockIdx.z % C;
    const long choff = (long)hd * hoff + (long)c*H*W;
    const float *xc = x + choff;
    float *yc = y + choff;                               /* y is [head][C][W][H] */
    const int h0 = blockIdx.y*32, w0 = blockIdx.x*32;
    for (int k = threadIdx.y; k < 32; k += 8) {
        const int h = h0 + k, w = w0 + threadIdx.x;
        tile[k][threadIdx.x] = (h < H && w < W) ? xc[(long)h*W + w] : 0.0f;
    }
    __syncthreads();
    for (int k = threadIdx.y; k < 32; k += 8) {
        const int w = w0 + k, h = h0 + threadIdx.x;
        if (w < W && h < H) yc[(long)w*H + h] = tile[threadIdx.x][k];
    }
}

template<int R, int PCO, int NCB>
__global__ void k_conv2d_r9(const float *x, const float *w, const float *b,
                            int C_in, int H, int W, int padH, int padW, int dilH,
                            float *y, int co0)
{
    const int TW = 16*NCB, LS = TW+4;
    const int NRA = 32/(PCO*NCB);
    __shared__ __align__(16) float S[R][LS];
    const int w0 = blockIdx.x*TW, h0 = blockIdx.y*32;
    const int tx = threadIdx.x, ty = threadIdx.y;                 // (16,16)
    const int tid = ty*16 + tx;
    float acc[PCO][NRA][NCB];
#pragma unroll
    for (int j = 0; j < PCO; j++)
#pragma unroll
        for (int a = 0; a < NRA; a++)
#pragma unroll
            for (int bq = 0; bq < NCB; bq++) acc[j][a][bq] = b[co0 + blockIdx.z*PCO + j];
    for (int ci = 0; ci < C_in; ci++) {
        const float *xin = x + (long)ci*H*W;
        if (ci) __syncthreads();
        {
            const int RS = LS/4, wbase = w0 - padW, hbase = h0 - padH;
            float4 *dv = (float4*)&S[0][0];
            for (int i = tid; i < R*RS; i += 256) {
                const int r = i / RS, q = 4*(i - r*RS);
                const int hh = hbase + r, ww = wbase + q;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (hh >= 0 && hh < H) {
                    const float *sp = xin + (long)hh*W + ww;
                    if (ww >= 0 && ww + 4 <= W && ((((long)ci*H*W + (long)hh*W + ww) & 3) == 0))
                        v = *(const float4*)sp;
                    else {
                        if (ww     >= 0 && ww     < W) v.x = sp[0];
                        if (ww + 1 >= 0 && ww + 1 < W) v.y = sp[1];
                        if (ww + 2 >= 0 && ww + 2 < W) v.z = sp[2];
                        if (ww + 3 >= 0 && ww + 3 < W) v.w = sp[3];
                    }
                }
                dv[i] = v;
            }
        }
        __syncthreads();
        const float *wv = w + ((long)(co0 + blockIdx.z*PCO)*C_in + ci)*9;
#pragma unroll
        for (int kh = 0; kh < 3; kh++) {
            const int rb = ty*NRA + kh*dilH, cb = tx*NCB;
#pragma unroll
            for (int a = 0; a < NRA; a++) {
                const float *sr = &S[rb + a][cb];
                const float s0 = sr[0], s1 = sr[1], s2 = sr[2], s3 = sr[3], s4 = sr[4];
#pragma unroll
                for (int j = 0; j < PCO; j++) {
                    const float w0f = wv[j*C_in*9 + kh*3], w1f = wv[j*C_in*9 + kh*3+1], w2f = wv[j*C_in*9 + kh*3+2];
#pragma unroll
                    for (int bq = 0; bq < NCB; bq++) {
                        const float p0 = (bq == 0) ? s0 : ((bq == 1) ? s1 : ((bq == 2) ? s2 : s3));
                        const float p1 = (bq == 0) ? s1 : ((bq == 1) ? s2 : ((bq == 2) ? s3 : s4));
                        const float p2 = (bq == 0) ? s2 : ((bq == 1) ? s3 : ((bq == 2) ? s4 : s4));
                        acc[j][a][bq] = fmaf(p0, w0f, acc[j][a][bq]);
                        acc[j][a][bq] = fmaf(p1, w1f, acc[j][a][bq]);
                        acc[j][a][bq] = fmaf(p2, w2f, acc[j][a][bq]);
                    }
                }
            }
        }
    }
    const int wc = w0 + tx*NCB;
#pragma unroll
    for (int j = 0; j < PCO; j++) {
        float *yo = y + (long)(co0 + blockIdx.z*PCO + j)*H*W;
#pragma unroll
        for (int a = 0; a < NRA; a++) {
            int hh = h0 + ty*NRA + a;
            if (hh >= H) continue;
            float *d2 = yo + (long)hh*W;
#pragma unroll
            for (int bq = 0; bq < NCB; bq++) if (wc + bq < W) d2[wc + bq] = acc[j][a][bq];
        }
    }
}

// dt_proj fused into the scan. dtt[s,c] = softplus(B_dt[c] + sum_p D_p[s]*W_dt[c][p])
// is a 4-term dot product consumed only by the scan's recurrence, so keeping it
// in registers removes the 21 MB dtt write and its re-read: of the pair's 126 MB
// per launch, 42 MB (33%) was that round trip. Same operand order as
// k_dt_proj_r, so the result is bit-identical to the two-kernel path.
__global__ void k_scan_dt(const float *xc, const float *Wdt, const float *Bdt,
                          const float *A_log, const float *Dv, const float *xdbl,
                          float *y, int S, int L, int rev)
{
    __shared__ float4 Ws[DINNER];
    for (int i = threadIdx.x; i < DINNER; i += 256) Ws[i] = ((const float4*)Wdt)[i];
    __syncthreads();
    int s = (int)((long)blockIdx.x * 8 + threadIdx.x / 32);
    int c = (threadIdx.x % 32) + blockIdx.y * 32;
    if (s >= S || c >= DINNER) return;
    float h[DSTATE];
#pragma unroll
    for (int n = 0; n < DSTATE; n++) h[n] = 0.0f;
    float aA[DSTATE];
#pragma unroll
    for (int n = 0; n < DSTATE; n++) aA[n] = -__expf(A_log[c * DSTATE + n]);
    const float Dc = Dv[c];
    const float4 w = Ws[c];
    const float bc = Bdt[c];
    for (int ii = 0; ii < L; ii++) {
        long sl = (long)s * L + ii;
        const float *dr = xdbl + sl * XP;
        float d = dsoftplus(bc + dr[0]*w.x + dr[1]*w.y + dr[2]*w.z + dr[3]*w.w);
        float u = xc[sl * DINNER + c];
        const float *Bt = xdbl + sl * XP + DTRANK;
        const float *Ct = Bt + DSTATE;
        float acc = 0.0f;
#pragma unroll
        for (int n = 0; n < DSTATE; n++) {
            h[n] = __expf(d * aA[n]) * h[n] + d * Bt[n] * u;
            acc = fmaf(h[n], Ct[n], acc);
        }
        y[sl * DINNER + c] = acc + u * Dc;
    }
}
// k_scan_dt with aA[] in padded shared memory instead of registers. ptxas
// spends 56 registers on the register version, which caps occupancy at 4
// blocks (32 of 64 warps) per SM; the kernel is latency-bound (72 GB/s of the
// 158 GB/s ceiling, 3x off its DRAM floor), so trading 16 registers for 16
// warp-broadcast shared loads per L step buys in-flight capacity.
__global__ void __launch_bounds__(256, 4) k_scan_dts(const float *xc, const float *Wdt, const float *Bdt,
                          const float *A_log, const float *Dv, const float *xdbl,
                          float *y, int S, int L, int rev)
{
    __shared__ float4 Ws[DINNER];
    __shared__ float aAs[32][DSTATE+1];   // stride 17: coprime with 32 banks
    for (int i = threadIdx.x; i < DINNER; i += 256) Ws[i] = ((const float4*)Wdt)[i];
    __syncthreads();
    int s = (int)((long)blockIdx.x * 8 + threadIdx.x / 32);
    int c = (threadIdx.x % 32) + blockIdx.y * 32;
    if (s >= S || c >= DINNER) return;
    float h[DSTATE];
#pragma unroll
    for (int n = 0; n < DSTATE; n++) h[n] = 0.0f;
    const int lane = (int)threadIdx.x % 32;
#pragma unroll
    for (int n = 0; n < DSTATE; n++) aAs[lane][n] = -__expf(A_log[c * DSTATE + n]);
    __syncthreads();
    const float Dc = Dv[c];
    const float4 w = Ws[c];
    const float bc = Bdt[c];
    for (int ii = 0; ii < L; ii++) {
        long sl = (long)s * L + ii;
        const float *dr = xdbl + sl * XP;
        float4 drv = *(const float4*)dr;
        float d = dsoftplus(bc + drv.x*w.x + drv.y*w.y + drv.z*w.z + drv.w*w.w);
        float u = xc[sl * DINNER + c];
        const float *Bt = xdbl + sl * XP + DTRANK;
        const float *Ct = Bt + DSTATE;
        float4 Bv0 = *(const float4*)(Bt + 0), Bv1 = *(const float4*)(Bt + 4);
        float4 Bv2 = *(const float4*)(Bt + 8), Bv3 = *(const float4*)(Bt + 12);
        float4 Cv0 = *(const float4*)(Ct + 0), Cv1 = *(const float4*)(Ct + 4);
        float4 Cv2 = *(const float4*)(Ct + 8), Cv3 = *(const float4*)(Ct + 12);
        float Bt_[16] = {Bv0.x,Bv0.y,Bv0.z,Bv0.w,Bv1.x,Bv1.y,Bv1.z,Bv1.w,Bv2.x,Bv2.y,Bv2.z,Bv2.w,Bv3.x,Bv3.y,Bv3.z,Bv3.w};
        float Ct_[16] = {Cv0.x,Cv0.y,Cv0.z,Cv0.w,Cv1.x,Cv1.y,Cv1.z,Cv1.w,Cv2.x,Cv2.y,Cv2.z,Cv2.w,Cv3.x,Cv3.y,Cv3.z,Cv3.w};
        float acc = 0.0f;
#pragma unroll
        for (int n = 0; n < DSTATE; n++) {
            h[n] = __expf(d * aAs[lane][n]) * h[n] + d * Bt_[n] * u;
            acc = fmaf(h[n], Ct_[n], acc);
        }
        y[sl * DINNER + c] = acc + u * Dc;
    }
}

static int conv_chunk(void)
{
    static int c = -1;
    if (c < 0) { const char *e = getenv("REUSE_CONV_CHUNK"); c = e ? atoi(e) : 64; if (c <= 0) c = 1 << 30; }
    return c;
}

static int launch_conv1x1(reuse_gpu_t *g, const float *x, const float *w, const float *b,
                          float *y, int C_in, int C_out, long HW)
{
    // Chunk over output channels: Jetson's GPU watchdog kills any single
    // launch that runs too long, so bound the work per launch.
    const int CH = conv_chunk();
    if (getenv("REUSE_TRACE_CONV"))
        fprintf(stderr, "[conv1x1] C_in=%d C_out=%d HW=%ld chunk=%d\n", C_in, C_out, HW, CH);
    for (int co0 = 0; co0 < C_out; co0 += CH) {
        int n = (C_out - co0 < CH) ? C_out - co0 : CH;
        dim3 blk(128), grd((unsigned)((HW + 127) / 128), (unsigned)n);
        // the kernel adds co0 itself, so pass the un-offset pointers
        k_conv1x1<<<grd, blk>>>(x, w, b, y, C_in, C_out, HW, co0);
        launch_kick();
    }
    return reuse_gpu_ok(g, "conv1x1");
}

#define R8ONE(RN,PC,NC) do { \
            if (regtile && dilH == 1)      do { if (ng >= 0) k_conv2d_r8<RN,PC,NC,32/(PC*NC),true><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); else k_conv2d_r8<RN,PC,NC,32/(PC*NC),false><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); } while (0); \
            else if (regtile && dilH == 2) do { if (ng >= 0) k_conv2d_r8<RN+2,PC,NC,32/(PC*NC),true><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); else k_conv2d_r8<RN+2,PC,NC,32/(PC*NC),false><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); } while (0); \
            else if (regtile && dilH == 4) do { if (ng >= 0) k_conv2d_r8<RN+8,PC,NC,32/(PC*NC),true><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); else k_conv2d_r8<RN+8,PC,NC,32/(PC*NC),false><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); } while (0); \
            else if (regtile && dilH == 8) do { if (ng >= 0) k_conv2d_r8<RN+24,PC,NC,32/(PC*NC),true><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); else k_conv2d_r8<RN+24,PC,NC,32/(PC*NC),false><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); } while (0); \
        } while (0)
#define R9LAUNCH(R1,R2,R4,R8) \
            if      (regtile && dilH == 1) { k_conv2d_r9<R1,8,2><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0); } \
            else if (regtile && dilH == 2) { k_conv2d_r9<R2,8,2><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0); } \
            else if (regtile && dilH == 4) { k_conv2d_r9<R4,8,2><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0); } \
            else if (regtile && dilH == 8) { k_conv2d_r9<R8,8,2><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0); }
/* Weight-staged twin launcher (REUSE_WST=1, cws): picks the WS=true
 * instantiation of k_conv2d_r8, which stages the block-uniform filter taps
 * in shared once per ci instead of loading them redundantly per thread. */
#define R8WS(R_,PC_,NC_,NR_,T_) do { \
    if (cws) k_conv2d_r8<R_,PC_,NC_,NR_,T_,true><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); \
    else     k_conv2d_r8<R_,PC_,NC_,NR_,T_><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); } while (0)
#define R64LAUNCH(PC_,NC_,NR_,R1,R2,R4,R8) \
            if      (regtile && dilH == 1) { do { if (ng >= 0) R8WS(R1,PC_,NC_,NR_,true); else k_conv2d_r8<R1,PC_,NC_,NR_,false><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); } while (0); } \
            else if (regtile && dilH == 2) { do { if (ng >= 0) R8WS(R2,PC_,NC_,NR_,true); else k_conv2d_r8<R2,PC_,NC_,NR_,false><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); } while (0); } \
            else if (regtile && dilH == 4) { do { if (ng >= 0) R8WS(R4,PC_,NC_,NR_,true); else k_conv2d_r8<R4,PC_,NC_,NR_,false><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); } while (0); } \
            else if (regtile && dilH == 8) { do { if (ng >= 0) R8WS(R8,PC_,NC_,NR_,true); else k_conv2d_r8<R8,PC_,NC_,NR_,false><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); } while (0); }
#define R8LAUNCH(R1,R2,R4,R8) \
            if      (regtile && dilH == 1) { if (pco==4) do { if (ng >= 0) k_conv2d_r8<R1,4,4,32/(4*4),true><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); else k_conv2d_r8<R1,4,4,32/(4*4),false><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); } while (0); else if (pco==8) do { if (ng >= 0) k_conv2d_r8<R1,8,2,32/(8*2),true><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); else k_conv2d_r8<R1,8,2,32/(8*2),false><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); } while (0); else do { if (ng >= 0) k_conv2d_r8<R1,16,2,32/(16*2),true><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); else k_conv2d_r8<R1,16,2,32/(16*2),false><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); } while (0); } \
            else if (regtile && dilH == 2) { if (pco==4) do { if (ng >= 0) k_conv2d_r8<R2,4,4,32/(4*4),true><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); else k_conv2d_r8<R2,4,4,32/(4*4),false><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); } while (0); else if (pco==8) do { if (ng >= 0) k_conv2d_r8<R2,8,2,32/(8*2),true><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); else k_conv2d_r8<R2,8,2,32/(8*2),false><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); } while (0); else do { if (ng >= 0) k_conv2d_r8<R2,16,2,32/(16*2),true><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); else k_conv2d_r8<R2,16,2,32/(16*2),false><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); } while (0); } \
            else if (regtile && dilH == 4) { if (pco==4) do { if (ng >= 0) k_conv2d_r8<R4,4,4,32/(4*4),true><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); else k_conv2d_r8<R4,4,4,32/(4*4),false><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); } while (0); else if (pco==8) do { if (ng >= 0) k_conv2d_r8<R4,8,2,32/(8*2),true><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); else k_conv2d_r8<R4,8,2,32/(8*2),false><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); } while (0); else do { if (ng >= 0) k_conv2d_r8<R4,16,2,32/(16*2),true><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); else k_conv2d_r8<R4,16,2,32/(16*2),false><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); } while (0); } \
            else if (regtile && dilH == 8) { if (pco==4) do { if (ng >= 0) k_conv2d_r8<R8,4,4,32/(4*4),true><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); else k_conv2d_r8<R8,4,4,32/(4*4),false><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); } while (0); else if (pco==8) do { if (ng >= 0) k_conv2d_r8<R8,8,2,32/(8*2),true><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); else k_conv2d_r8<R8,8,2,32/(8*2),false><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); } while (0); else do { if (ng >= 0) k_conv2d_r8<R8,16,2,32/(16*2),true><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); else k_conv2d_r8<R8,16,2,32/(16*2),false><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); } while (0); }

static int dectile(void)
{
    static int t = -1;
    if (t < 0) { const char *e = getenv("REUSE_DEC_TILE"); t = e ? atoi(e) : 644; }
    return t;
}

static int conv_chunk(void);
static int launch_conv2d_pair(reuse_gpu_t *g, const float *x, const float *w, const float *b,
                              const float *w2, const float *b2,
                              int C_in, int H, int W, int C_out, int padH, int dilH,
                              float *y, int tilehint, const float *x2, int ng,
                              long hoff_out, long hoff_x);
static int sp_up(reuse_gpu_t *g, const char *prefix, const float *x, int H, int W, int r, float *y);
static int dense_block_gpu(reuse_gpu_t *g, const char *prefix, const float *x_in,
                           int T, int F, float *out, int tilehint);
static int inorm(reuse_gpu_t *g, const float *x, float *y, int N, int C, int H, int W,
                 const char *nkey, const char *akey);

// Paired dense block: both heads' DEPTH layers run each launch jointly.
// slots/tmp are single 2-head buffers (head h at +h*headstride); the conv
// gets head0's w/b (head1 = +hoff_w, caller-verified blob neighbours) and
// the shared x_in (enc) as x2. Returns 0 for shapes the paired kernels do
// not cover (caller falls back to dense_block_gpu twice).
static int dense_block_pair(reuse_gpu_t *g, const char *pfx0, const char *pfx1,
                            const float *x_in, int T, int F,
                            float *out0, float *out1, int tilehint,
                            long hoff_out_pair)
{
    if (DEPTH > 4) return 0;                       // dil 16/32 have no paired tile
    if (gpu_is_maxwell()) return 0;                // TABLE convs corrupt on sm_53 (#73/#74)
    float *slots = wallc(g, (size_t)2*HID*(DEPTH-1)*T*F);
    float *tmp   = wallc(g, (size_t)2*HID*T*F);
    if (!slots || !tmp) return 1;
    const long hslot = (long)(DEPTH-1)*HID*T*F;    // head-to-head stride of slots
    const long htf = (long)HID*T*F;
    char key0[256], key1[256], nk0[256], nk1[256], ak0[256], ak1[256], b0[256], b1[256];
    for (int i = 0; i < DEPTH; i++) {
        snprintf(key0, sizeof(key0), "%s.%d.0.weight", pfx0, i);
        snprintf(key1, sizeof(key1), "%s.%d.0.weight", pfx1, i);
        const float *wm = reuse_gpu_w(g, key0), *wp = reuse_gpu_w(g, key1);
        snprintf(key0, sizeof(key0), "%s.%d.0.bias", pfx0, i);
        snprintf(key1, sizeof(key1), "%s.%d.0.bias", pfx1, i);
        const float *bm = reuse_gpu_w(g, key0), *bp = reuse_gpu_w(g, key1);
        if (g->bad) return 1;
        const int dil = 1 << i;
        if (!launch_conv2d_pair(g, slots, wm, bm, wp, bp, (i+1)*HID, T, F, HID, dil, dil, tmp,
                                tilehint, x_in, i, htf, hslot)) return 0;
        snprintf(nk0, sizeof(nk0), "%s.%d.1.weight", pfx0, i);
        snprintf(nk1, sizeof(nk1), "%s.%d.1.weight", pfx1, i);
        snprintf(b0, sizeof(b0), "%s.%d.1.bias", pfx0, i);
        snprintf(b1, sizeof(b1), "%s.%d.1.bias", pfx1, i);
        snprintf(ak0, sizeof(ak0), "%s.%d.2.weight", pfx0, i);
        snprintf(ak1, sizeof(ak1), "%s.%d.2.weight", pfx1, i);
        const float *nw0 = reuse_gpu_w(g, nk0), *nw1 = reuse_gpu_w(g, nk1);
        const float *nb0 = reuse_gpu_w(g, b0),  *nb1 = reuse_gpu_w(g, b1);
        const float *pa0 = reuse_gpu_w(g, ak0), *pa1 = reuse_gpu_w(g, ak1);
        if (g->bad) return 1;
        // layer i's inorm output lands directly in slot i of each head (the
        // #74 zero-copy convention): head1 = ybase + hslot, per-head stride
        // hslot, ybase advanced by i planes for the slot index.
        if (i == DEPTH-1)
            k_inorm_prelu2<<<2*HID, INORM_THREADS>>>(tmp, out0, HID, T*F, hoff_out_pair,
                                                     nw0, nb0, pa0, nw1, nb1, pa1, 1e-5f);
        else
            k_inorm_prelu2<<<2*HID, INORM_THREADS>>>(tmp, slots + (long)i*htf, HID, T*F, hslot,
                                                     nw0, nb0, pa0, nw1, nb1, pa1, 1e-5f);
        if (reuse_gpu_ok(g, "inorm_pair")) return 1;
    }
    return !g->bad;
}

static int sp_up_pair(reuse_gpu_t *g, const char *pfx0, const char *pfx1,
                      const float *x, int H, int W, int r, float *y,
                      long hoff_in, long hoff_out)
{
    static int RW_ = 0, RC_ = 0, TW_ = 0, NC_ = 0;
    static int t8 = -1;
    if (t8 < 0) t8 = getenv("REUSE_SPUP") ? atoi(getenv("REUSE_SPUP")) : 4;
    if (t8 != 4) return 0;                          // paired form is the tiled kernel only
    if (RW_ == 0) {
        RW_ = 4; RC_ = 8; TW_ = 32; NC_ = 64;
        const char *e = getenv("REUSE_SPUP_T"); int a=0,b=0,c=0,d=0;
        if (e && sscanf(e, "%d,%d,%d,%d", &a,&b,&c,&d) == 4 && a) { RW_ = a; RC_ = b; TW_ = c; NC_ = d; }
    }
    char key[256];
    const int CO = 64*r;
    snprintf(key, sizeof(key), "%s.0.conv.weight", pfx0); const float *w0 = reuse_gpu_w(g, key);
    snprintf(key, sizeof(key), "%s.0.conv.bias",   pfx0); const float *b0 = reuse_gpu_w(g, key);
    snprintf(key, sizeof(key), "%s.0.conv.weight", pfx1); const float *w1 = reuse_gpu_w(g, key);
    snprintf(key, sizeof(key), "%s.0.conv.bias",   pfx1); const float *b1 = reuse_gpu_w(g, key);
    if (g->bad) return 1;
    if (NC_ != 64) return 0;
    float *wt = wallc(g, (size_t)2*CO*HID*3);
    if (!wt) return 1;
    k_wtrans_sp<<<(unsigned)((CO*HID*3+255)/256), 256>>>(w0, wt, CO);
    k_wtrans_sp<<<(unsigned)((CO*HID*3+255)/256), 256>>>(w1, wt + (long)CO*HID*3, CO);
    const int zheads = CO/NC_;
    #define SPUP2(RWv,RCv,TWv,NCv) do { if (RW_==RWv && RC_==RCv && TW_==TWv && NC_==NCv) \
        k_sp_up8t2<RWv,RCv,TWv,NCv><<<dim3((unsigned)((W+TWv-1)/TWv),(unsigned)H,(unsigned)(2*(CO/NCv))), dim3(TWv/RWv,NCv/RCv)>>>(x,wt,b0,b1,y,H,W,r,CO,hoff_in,hoff_out,zheads,(long)CO*HID*3); } while(0)
    SPUP2(4,8,32,64); SPUP2(8,8,32,64); SPUP2(4,8,64,64); SPUP2(4,16,32,64);
    SPUP2(8,16,32,64); SPUP2(8,8,16,64); SPUP2(8,8,32,128); SPUP2(4,8,32,128);
    SPUP2(4,8,64,128); SPUP2(8,8,64,128);
    #undef SPUP2
    launch_kick();
    return !g->bad;
}

// 0 = paired path ran, 1 = error, 2 = unsupported shape (caller falls back).
extern "C" int reuse_gpu_decoders_pair(reuse_gpu_t *g, const float *enc,
                                       float *dm, float *dp, int T, int F)
{
    static int on = -1;
    if (on < 0) { const char *e = getenv("REUSE_DEC_PAIR"); on = e ? atoi(e) : (gpu_is_maxwell() ? 0 : 1); }
    if (!on) return 2;
    if (dp <= dm) return 2;                          // heads must be contiguous
    g->ws_off = 0;
    const long htf = (long)HID*T*F;
    const long hoff_dp = dp - dm;
    // weight-neighbourhood probes (mask vs phase decoder, same layout):
    // verify the two decoders' parameter tensors are equal-length pairs so a
    // single hoff works for every conv/norm weight. Checked at every use in
    // dense_block_pair; a cheap precheck here is the up_conv biases.
    char k0[256], k1[256];
    snprintf(k0, sizeof(k0), "mask_decoder.up_conv1.0.conv.bias");
    snprintf(k1, sizeof(k1), "phase_decoder.up_conv1.0.conv.bias");
    const float *mub = reuse_gpu_w(g, k0), *pub = reuse_gpu_w(g, k1);
    if (g->bad) return 1;
    if (!mub || !pub) return 2;

    float *d_pair = wallc(g, (size_t)2*HID*T*F);
    if (!d_pair) return 1;
    snprintf(k0, sizeof(k0), "%s.dense_block.dense_block", "mask_decoder");
    snprintf(k1, sizeof(k1), "%s.dense_block.dense_block", "phase_decoder");
    int rc = dense_block_pair(g, k0, k1, enc, T, F, d_pair, d_pair + htf, dectile(), htf);
    if (rc < 0) return 1;
    if (!rc) {
        // TABLE convs unsupported on this arch (sm_53 corrupts the 32-acc
        // table shapes, #73/#74): run the two dense blocks sequentially with
        // the shipped copy path, then pair the (non-conv) tail.
        if (dense_block_gpu(g, k0, enc, T, F, d_pair, dectile())) return 1;
        if (dense_block_gpu(g, k1, enc, T, F, d_pair + htf, dectile())) return 1;
    }
    gdump("d_db", d_pair, (size_t)2*HID*T*F);

    const long hu1 = (long)HID*T*(F*2);
    float *u1  = wallc(g, (size_t)2*hu1);
    float *u1n = wallc(g, (size_t)2*hu1);
    if (!u1 || !u1n) return 1;
    if (!sp_up_pair(g, "mask_decoder.up_conv1", "phase_decoder.up_conv1", d_pair, T, F, 2, u1, htf, hu1)) return 2;
    gdump("d_u1", u1, (size_t)2*hu1);
    {
        snprintf(k0, sizeof(k0), "%s", "mask_decoder.up_conv1.1.weight");
        snprintf(k1, sizeof(k1), "%s", "phase_decoder.up_conv1.1.weight");
        const float *nw0 = reuse_gpu_w(g, k0), *nw1 = reuse_gpu_w(g, k1);
        char b0[256], b1[256];
        snprintf(b0, sizeof(b0), "%s", "mask_decoder.up_conv1.1.bias");
        snprintf(b1, sizeof(b1), "%s", "phase_decoder.up_conv1.1.bias");
        const float *nb0 = reuse_gpu_w(g, b0), *nb1 = reuse_gpu_w(g, b1);
        snprintf(k0, sizeof(k0), "%s", "mask_decoder.up_conv1.2.weight");
        snprintf(k1, sizeof(k1), "%s", "phase_decoder.up_conv1.2.weight");
        const float *pa0 = reuse_gpu_w(g, k0), *pa1 = reuse_gpu_w(g, k1);
        if (g->bad) return 1;
        k_inorm_prelu2<<<2*HID, INORM_THREADS>>>(u1, u1n, HID, T*(F*2), hu1, nw0, nb0, pa0, nw1, nb1, pa1, 1e-5f);
        if (reuse_gpu_ok(g, "inorm_pair")) return 1;
    }
    const int Tt = F*2, Ft = T;
    const long hsw = (long)HID*Tt*Ft;
    float *sw = wallc(g, (size_t)2*hsw);
    if (!sw) return 1;
    k_transpose2<<<dim3((unsigned)((F*2+31)/32), (unsigned)((T+31)/32), (unsigned)(2*HID)), dim3(32,8)>>>(u1n, sw, HID, T, F*2, hsw);
    gdump("d_sw", sw, (size_t)2*hsw);
    // up_conv2 (r=4) + inorm + final transpose run per head: its u2 buffer is
    // 8x the block tensor, and doubling it would raise the arena peak above
    // the Nano's budget for a marginal launch-count gain.
    float *u2  = wallc(g, (size_t)HID*Tt*(size_t)(Ft*4));
    float *u2n = wallc(g, (size_t)HID*Tt*(size_t)(Ft*4));
    if (!u2 || !u2n) return 1;
    for (int hnd = 0; hnd < 2; hnd++) {
        const char *pref = hnd ? "phase_decoder" : "mask_decoder";
        snprintf(k0, sizeof(k0), "%s.up_conv2", pref);
        if (sp_up(g, k0, sw + (long)hnd*hsw, Tt, Ft, 4, u2)) return 1;
        char nk[256], ak[256];
        snprintf(nk, sizeof(nk), "%s.up_conv2.1.weight", pref);
        snprintf(ak, sizeof(ak), "%s.up_conv2.2.weight", pref);
        if (inorm(g, u2, u2n, 1, HID, Tt, Ft*4, nk, ak)) return 1;
        launch_transpose(HID, Tt, Ft*4, u2n, hnd ? dp : dm);
    }
    gdump("d_pre", dm, (size_t)HID*Tt*(Ft*4));
    (void)hoff_dp;
    return g->bad ? 1 : 0;
}

// Paired-dense conv launcher: only the r8/r6 register-tile paths (the dense
// block's 3x3 dilated shapes), grid.z doubled, head = z / znb. Returns 0 if
// the shape is outside what the paired kernels cover (caller falls back to
// the sequential path). All paired tensors must be head-contiguous: x = the
// 2-head slots buffer, y = the 2-head tmp, x2 = shared enc, w/b = head0's
// (blob neighbours verified by the caller, head1 = +hoff_w).
static int launch_conv2d_pair(reuse_gpu_t *g, const float *x, const float *w, const float *b,
                              const float *w2, const float *b2,
                              int C_in, int H, int W, int C_out, int padH, int dilH,
                              float *y, int tilehint, const float *x2, int ng,
                              long hoff_out, long hoff_x)
{
    const long shw = (long)H*W;
    const bool regtile = dilH == 1 || dilH == 2 || dilH == 4 || dilH == 8;
    if (!regtile || C_out % conv_chunk() != 0) return 0;
    const int H_out = H, W_out = W;
    int big = tilehint ? tilehint : 645;
    static int bigenv = -1;
    if (bigenv < 0) { const char *e = getenv("REUSE_CONV_TILE"); bigenv = e ? atoi(e) : 644; }
    if (!tilehint) big = bigenv;
    if (big >= 644 && big <= 646 && gpu_is_maxwell()) big = 82;
    if (big == 92) big = 82;                       // r9 has no paired form
    int pco, ncb, nra;
    if (big == 644) { pco = 8; ncb = 2; nra = 4; }
    else if (big == 645) { pco = 8; ncb = 4; nra = 2; }
    else if (big == 646) { pco = 16; ncb = 2; nra = 2; }
    else if (big == 81) { pco = 4; ncb = 4; nra = 2; }
    else if (big == 82) { pco = 8; ncb = 2; nra = 2; }
    else return 0;
    const int trh = 16*nra;
    const int CH = conv_chunk();
    const float *S[16];
    for (int t = 0; t < 16; t++)
        S[t] = (t < ng) ? (x + (long)(ng-1-t)*HID*shw)
                        : (x2 ? x2 + (long)(t-ng)*HID*shw : (const float*)0);
    const float *s0=S[0],*s1=S[1],*s2=S[2],*s3=S[3],*s4=S[4],*s5=S[5],*s6=S[6],*s7=S[7];
    const float *s8=S[8],*s9=S[9],*s10=S[10],*s11=S[11],*s12=S[12],*s13=S[13],*s14=S[14],*s15=S[15];
    for (int co0 = 0; co0 < C_out; co0 += CH) {
        int n = (C_out - co0 < CH) ? C_out - co0 : CH;
        if (n % pco) return 0;
        const int znb = n/pco;
        const dim3 grdb((unsigned)((W_out + 16*ncb - 1) / (16*ncb)), (unsigned)((H_out + trh - 1) / trh), (unsigned)(2*(n/pco)));
        const dim3 blkb(16, 16);
        #define LAUNCHP1(RV,PC_,NC_,NR_) k_conv2d_r8p<RV,PC_,NC_,NR_,true,true><<<grdb,blkb>>>(x,w,b,w2,b2,C_in,H,W,padH,1,dilH,y,co0,x2,ng,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,hoff_out,znb,hoff_x)
        #define LAUNCHP(RV) do { \
            if      (pco == 4) LAUNCHP1(RV,4,4,32/(4*4)); \
            else if (big == 644) LAUNCHP1(RV,8,2,4); \
            else if (big == 645) LAUNCHP1(RV,8,4,2); \
            else if (big == 646) LAUNCHP1(RV,16,2,2); \
            else LAUNCHP1(RV,8,2,2); } while (0)
        switch (dilH) {
        case 1: if (big == 644) LAUNCHP(66); else LAUNCHP(34); break;
        case 2: if (big == 644) LAUNCHP(68); else LAUNCHP(36); break;
        case 4: if (big == 644) LAUNCHP(72); else LAUNCHP(40); break;
        default: if (big == 644) LAUNCHP(80); else LAUNCHP(48); break;
        }
        #undef LAUNCHP
        launch_kick();
        (void)hoff_x;
    }
    return !g->bad;
}

static int launch_conv2d(reuse_gpu_t *g, const float *x, const float *w, const float *b,
                         int C_in, int H, int W, int C_out, int kH, int kW,
                         int padH, int padW, int dilH, int dilW, int strideH, int strideW,
                         float *y, int tilehint, const float *x2, int ng)
{
    const long shw = (long)H*W;
    static int cws = -2;
    if (cws == -2) cws = getenv("REUSE_WST") ? atoi(getenv("REUSE_WST")) : 1;
    const float *S[16];
    for (int t = 0; t < 16; t++)
        S[t] = (t < ng) ? (x + (long)(ng-1-t)*HID*shw)
                        : (x2 ? x2 + (long)(t-ng)*HID*shw : (const float*)0);
    const float *s0=S[0],*s1=S[1],*s2=S[2],*s3=S[3],*s4=S[4],*s5=S[5],*s6=S[6],*s7=S[7];
    const float *s8=S[8],*s9=S[9],*s10=S[10],*s11=S[11],*s12=S[12],*s13=S[13],*s14=S[14],*s15=S[15];
    // tilehint: 0 = use REUSE_CONV_TILE, else a tile code for this call site.
    // The encoder's dense block prefers 644 (8co x 2col x 4row) and the
    // decoders' prefer 644 (8x2x4, a 64-row tile); re-swept after the WS
    // weight staging landed (#194), which moved the encoder optimum from 645.
    const int tileuse = tilehint;
    int H_out = (H + 2*padH - dilH*(kH-1) - 1)/strideH + 1;
    int W_out = (W + 2*padW - dilW*(kW-1) - 1)/strideW + 1;
    dim3 blk(16, 16);
    const int CH = conv_chunk();
    if (getenv("REUSE_TRACE_CONV"))
        fprintf(stderr, "[conv2d] C_in=%d H=%d W=%d C_out=%d kH=%d kW=%d padH=%d padW=%d dilH=%d dilW=%d sH=%d sW=%d -> H_out=%d W_out=%d chunk=%d\n",
                C_in, H, W, C_out, kH, kW, padH, padW, dilH, dilW, strideH, strideW, H_out, W_out, CH);
    for (int co0 = 0; co0 < C_out; co0 += CH) {
        int n = (C_out - co0 < CH) ? C_out - co0 : CH;
        dim3 grd((unsigned)((W_out + 15) / 16), (unsigned)((H_out + 15) / 16), (unsigned)n);
        // the kernel adds co0 itself, so pass the un-offset pointers
        // the register-tiled kernel owns a 16x32 output tile, so it needs its
        // own grid/block (blk here is the naive kernel's 16x16)
        const bool regtile = !getenv("REUSE_NAIVE_CONV2D") && kH == 3 && kW == 3 &&
                             dilW == 1 && strideH == 1 && strideW == 1 && padW == 1 && padH == dilH;
        static int bigenv = -1;
        if (bigenv < 0) { const char *e = getenv("REUSE_CONV_TILE"); bigenv = e ? atoi(e) : 644; }
        int big = tileuse ? tileuse : bigenv;
        // Maxwell (sm_53, the Jetson Nano) cannot launch the 64-accumulator
        // tiles: ptxas hands back a kernel whose per-thread register demand
        // exceeds the 255/thread limit, so the launch fails with "too many
        // resources requested" - it compiles for sm_53 fine, so the compile
        // gate cannot see it. The 32-accumulator code 82 is the Maxwell
        // fallback (it is what the Nano ran before the 64-acc tiles existed).
        if (big >= 644 && big <= 646 && gpu_is_maxwell()) big = 82;
        // 64-accumulator shapes: codes 644 (8co x 2col x 4row), 645 (8x4x2),
        // 646 (16x2x2). Each output amortises its 9 weight loads over
        // rows*cols positions and the warp-uniform weight loads are ~50% of
        // this kernel's issue slots, so doubling rows*cols (4 -> 8) is the
        // lever that moved the conv off 0.9 TFLOPS. trh MUST stay 16*NRA to
        // match the kernel's h0, and each branch needs the naive k_conv2d
        // fallback for dilH outside {1,2,4,8} (the encoder's dil 16/32
        // layers) - omitting it leaves those convs unlaunched.
        {
            static int stmode = -1;
            if (stmode < 0) stmode = getenv("REUSE_CONV_ST_OFF") ? 0 : 1;
            static int TR_ = 0, TC_, PCO_, RW_;
            if (TR_ == 0) {
                const char *e = getenv("REUSE_CONV_ST"); int a=0,b=0,c=0,d=0;
                TR_ = 16; TC_ = 16; PCO_ = 16; RW_ = 4;   /* sweep winner: (4,16) threads,
                                                             16 rows x 16 cols x 16 channels,
                                                             4 output cols per thread = 64 accs */
                if (e && sscanf(e, "%d,%d,%d,%d", &a,&b,&c,&d) == 4) { TR_ = a; TC_ = b; PCO_ = c; RW_ = d; }
            }
            const bool ok = stmode && kH == 1 && kW == 3 && dilH == 1 && dilW == 1 &&
                            strideH >= 1 && (strideW == 1 || strideW == 2) &&
                            padW >= 0 && padW <= 1 && padH == 0 &&
                            n % PCO_ == 0 && (TC_ % RW_) == 0 && PCO_ <= n;
            if (ok) {
                #define CST(TRv,TCv,PCOv,RWv) do { if (TR_==TRv && TC_==TCv && PCO_==PCOv && RW_==RWv) { \
                    if ((TCv/RWv)*TRv > 1024 || (TCv/RWv)*TRv < 16) break; \
                    if (strideW == 2) k_conv2d_st<TRv,TCv,PCOv,RWv,2><<<dim3((unsigned)((W_out+TCv-1)/TCv),(unsigned)((H_out+TRv-1)/TRv),(unsigned)(n/PCOv)), dim3(TCv/RWv,TRv)>>>( \
                            x, w, b, C_in, H, W, padW, strideH, y, co0, H_out, W_out); \
                    else k_conv2d_st<TRv,TCv,PCOv,RWv,1><<<dim3((unsigned)((W_out+TCv-1)/TCv),(unsigned)((H_out+TRv-1)/TRv),(unsigned)(n/PCOv)), dim3(TCv/RWv,TRv)>>>( \
                            x, w, b, C_in, H, W, padW, strideH, y, co0, H_out, W_out); } } while(0)
                CST(8,32,16,4); CST(4,32,16,4); CST(8,32,8,4); CST(16,32,16,4);
                CST(8,16,16,4); CST(8,32,32,2); CST(8,64,16,8); CST(4,16,16,4);
                CST(8,32,16,8); CST(16,16,16,4); CST(8,32,64,1); CST(32,16,16,4);
                #undef CST
                launch_kick();
                continue;
            }
        }
        int pco = 0, ncb = 2, nra = 2;
        if (big == 644) { pco = 8; ncb = 2; nra = 4; }
        else if (big == 645) { pco = 8; ncb = 4; nra = 2; }
        else if (big == 646) { pco = 16; ncb = 2; nra = 2; }
        const bool r6 = (pco != 0);
        if (!r6) { pco = (big == 81) ? 4 : (big == 82 || big == 92 ? 8 : 16); ncb = (big == 81) ? 4 : 2; nra = 32/(pco*ncb); }
        const int trh = r6 ? 16*nra : ((big == 83) ? 16 : 32);
        if (((big >= 81 && big <= 92) || r6) && n % pco == 0) {
            const bool r9 = (big == 92);
            (void)nra;
            const dim3 grdb((unsigned)((W_out + 16*ncb - 1) / (16*ncb)), (unsigned)((H_out + trh - 1) / trh), (unsigned)(n/pco));
            const dim3 blkb(16, 16);
            if (big == 644) { R64LAUNCH(8,2,4,66,68,72,80) else k_conv2d<<<grd, blk>>>(x, w, b, C_in, H, W, C_out, kH, kW, padH, padW, dilH, dilW, strideH, strideW, y, co0); }
            else if (big == 645) { R64LAUNCH(8,4,2,34,36,40,48) else k_conv2d<<<grd, blk>>>(x, w, b, C_in, H, W, C_out, kH, kW, padH, padW, dilH, dilW, strideH, strideW, y, co0); }
            else if (big == 646) { R64LAUNCH(16,2,2,34,36,40,48) else k_conv2d<<<grd, blk>>>(x, w, b, C_in, H, W, C_out, kH, kW, padH, padW, dilH, dilW, strideH, strideW, y, co0); }
            else if (r9) { R9LAUNCH(34,36,40,48)
                      else k_conv2d<<<grd, blk>>>(x, w, b, C_in, H, W, C_out, kH, kW, padH, padW, dilH, dilW, strideH, strideW, y, co0); }
            else    { R8LAUNCH(34,36,40,48)
                      else k_conv2d<<<grd, blk>>>(x, w, b, C_in, H, W, C_out, kH, kW, padH, padW, dilH, dilW, strideH, strideW, y, co0); }
            launch_kick();
            continue;
        }
        if (big == 31) {
            const dim3 grdb((unsigned)((W_out + 63) / 64), (unsigned)((H_out + 31) / 32), (unsigned)n);
            const dim3 blkb(16, 16);
            if (regtile && dilH == 1)      k_conv2d_r5<34><<<grdb, blkb>>>(x, w, b, C_in, H, W, padH, padW, dilH, y, co0);
            else if (regtile && dilH == 2) k_conv2d_r5<36><<<grdb, blkb>>>(x, w, b, C_in, H, W, padH, padW, dilH, y, co0);
            else if (regtile && dilH == 4) k_conv2d_r5<40><<<grdb, blkb>>>(x, w, b, C_in, H, W, padH, padW, dilH, y, co0);
            else if (regtile && dilH == 8) k_conv2d_r5<48><<<grdb, blkb>>>(x, w, b, C_in, H, W, padH, padW, dilH, y, co0);
            else k_conv2d<<<grd, blk>>>(x, w, b, C_in, H, W, C_out, kH, kW, padH, padW, dilH, dilW, strideH, strideW, y, co0);
            launch_kick();
            continue;
        }
        if (big == 32) {
            const dim3 grdb((unsigned)((W_out + 63) / 64), (unsigned)((H_out + 31) / 32), (unsigned)n);
            const dim3 blkb(16, 16);
            if (regtile && dilH == 1)      k_conv2d_r3<34><<<grdb, blkb>>>(x, w, b, C_in, H, W, padH, padW, dilH, y, co0);
            else if (regtile && dilH == 2) k_conv2d_r3<36><<<grdb, blkb>>>(x, w, b, C_in, H, W, padH, padW, dilH, y, co0);
            else if (regtile && dilH == 4) k_conv2d_r3<40><<<grdb, blkb>>>(x, w, b, C_in, H, W, padH, padW, dilH, y, co0);
            else if (regtile && dilH == 8) k_conv2d_r3<48><<<grdb, blkb>>>(x, w, b, C_in, H, W, padH, padW, dilH, y, co0);
            else k_conv2d<<<grd, blk>>>(x, w, b, C_in, H, W, C_out, kH, kW, padH, padW, dilH, dilW, strideH, strideW, y, co0);
            launch_kick();
            continue;
        }
        if (big == 16) big = 0;
        if (big == 64) {
            const dim3 grdb((unsigned)((W_out + 63) / 64), (unsigned)((H_out + 63) / 64), (unsigned)n);
            const dim3 blkb(16, 16);
            if (regtile && dilH == 1)      do { if (ng >= 0) k_conv2d_r2<66,true><<<grdb, blkb>>>(x, w, b, C_in, H, W, padH, padW, dilH, y, co0, x2, ng, s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); else k_conv2d_r2<66,false><<<grdb, blkb>>>(x, w, b, C_in, H, W, padH, padW, dilH, y, co0, x2, ng, s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); } while (0);
            else if (regtile && dilH == 2) do { if (ng >= 0) k_conv2d_r2<68,true><<<grdb, blkb>>>(x, w, b, C_in, H, W, padH, padW, dilH, y, co0, x2, ng, s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); else k_conv2d_r2<68,false><<<grdb, blkb>>>(x, w, b, C_in, H, W, padH, padW, dilH, y, co0, x2, ng, s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); } while (0);
            else if (regtile && dilH == 4) do { if (ng >= 0) k_conv2d_r2<72,true><<<grdb, blkb>>>(x, w, b, C_in, H, W, padH, padW, dilH, y, co0, x2, ng, s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); else k_conv2d_r2<72,false><<<grdb, blkb>>>(x, w, b, C_in, H, W, padH, padW, dilH, y, co0, x2, ng, s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); } while (0);
            else if (regtile && dilH == 8) do { if (ng >= 0) k_conv2d_r2<80,true><<<grdb, blkb>>>(x, w, b, C_in, H, W, padH, padW, dilH, y, co0, x2, ng, s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); else k_conv2d_r2<80,false><<<grdb, blkb>>>(x, w, b, C_in, H, W, padH, padW, dilH, y, co0, x2, ng, s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15); } while (0);
            else k_conv2d<<<grd, blk>>>(x, w, b, C_in, H, W, C_out, kH, kW, padH, padW, dilH, dilW, strideH, strideW, y, co0);
            launch_kick();
            continue;
        }
        const dim3 grdt((unsigned)((W_out + 31) / 32), (unsigned)((H_out + 15) / 16), (unsigned)n);
        const dim3 blkt(32, 8);
        if (regtile && dilH == 1)
            k_conv2d_r<18><<<grdt, blkt>>>(x, w, b, C_in, H, W, padH, padW, dilH, y, co0);
        else if (regtile && dilH == 2)
            k_conv2d_r<20><<<grdt, blkt>>>(x, w, b, C_in, H, W, padH, padW, dilH, y, co0);
        else if (regtile && dilH == 4)
            k_conv2d_r<24><<<grdt, blkt>>>(x, w, b, C_in, H, W, padH, padW, dilH, y, co0);
        else if (regtile && dilH == 8)
            k_conv2d_r<32><<<grdt, blkt>>>(x, w, b, C_in, H, W, padH, padW, dilH, y, co0);
        else
            k_conv2d<<<grd, blk>>>(x, w, b,
                                   C_in, H, W, C_out, kH, kW, padH, padW, dilH, dilW, strideH, strideW,
                                   y, co0);
        launch_kick();
    }
    return reuse_gpu_ok(g, "conv2d");
}

static int inorm(reuse_gpu_t *g, const float *x, float *y, int N, int C, int H, int W,
                 const char *nkey, const char *akey)
{
    char bkey[256];
    snprintf(bkey, sizeof(bkey), "%s", nkey);
    size_t bl = strlen(bkey);
    if (bl > 7 && strcmp(bkey + bl - 7, ".weight") == 0) strcpy(bkey + bl - 7, ".bias");
    else snprintf(bkey, sizeof(bkey), "%s.bias", nkey);
    const float *nw = reuse_gpu_w(g, nkey); if (!nw) return 1;
    const float *nb = reuse_gpu_w(g, bkey); if (!nb) return 1;
    const float *pa = akey ? reuse_gpu_w(g, akey) : NULL;
    if (g->bad) return 1;
    k_inorm_prelu<<<N*C, INORM_THREADS>>>(x, y, C, H*W, nw, nb, pa, 1e-5f);
    return reuse_gpu_ok(g, "inorm");
}

// PyTorch DenseBlock over a scratch region of the arena.
static int dense_block_gpu(reuse_gpu_t *g, const char *prefix, const float *x_in,
                           int T, int F, float *out, int tilehint)
{
    const int cap = HID * (DEPTH + 1) * T * F;
    /* Two concat buffers are enough: layer 0 reads the caller's tensor directly
       (the old code copied it into a third buffer), and each layer's [cur, skip]
       pair is written into the buffer that is not being read. The last layer's
       inorm writes `out` directly, which removes the trailing cur->out copy. */
    float *tmp   = wallc(g, (size_t)HID*T*F);
    if (!tmp) return 1;
    static int slots_mode = -1;
    if (slots_mode < 0) {
        const char *e = getenv("REUSE_DENSE_SLOTS");
        /* Param-delivery table mode: removes the 11 [cur,skip] memcpys per forward
           (444 MB at 8 kHz). Verified bit-compatible on the 64-accumulator shapes
           the dev host uses (max|d| 8.9e-08 = the iSTFT-atomic floor) and 0.9%
           faster there. The 32-accumulator shapes (Maxwell/Tile-82 family, the
           Nano's deployment path) corrupt output under sm_53 codegen (#72/#73/#74:
           max|d| 0.234 with both __device__ and param delivery), so Maxwell is
           pinned to the copy path. REUSE_DENSE_SLOTS=0/1 overrides. */
        slots_mode = e ? atoi(e) : (gpu_is_maxwell() ? 0 : 1);
    }
    if (slots_mode) {
        float *slots = wallc(g, (size_t)HID*(DEPTH-1)*T*F);
        if (!slots) return 1;
        char key[256], nkey[256], akey[256];
        for (int i = 0; i < DEPTH; i++) {
            snprintf(key, sizeof(key), "%s.%d.0.weight", prefix, i); const float *w = reuse_gpu_w(g, key);
            snprintf(key, sizeof(key), "%s.%d.0.bias",   prefix, i); const float *b = reuse_gpu_w(g, key);
            if (g->bad) return 1;
            int dil = 1 << i;
            launch_conv2d(g, slots, w, b, (i+1)*HID, T, F, HID, 3, 3, dil, 1, dil, 1, 1, 1,
                          tmp, tilehint, x_in, i);
            snprintf(nkey, sizeof(nkey), "%s.%d.1.weight", prefix, i);
            snprintf(akey, sizeof(akey), "%s.%d.2.weight", prefix, i);
            inorm(g, tmp, (i == DEPTH-1) ? out : slots + (size_t)i*HID*T*F, 1, HID, T, F, nkey, akey);
        }
        static int dbsync2 = -1;
        if (dbsync2 < 0) dbsync2 = getenv("REUSE_DBSYNC") ? atoi(getenv("REUSE_DBSYNC")) : 0;
        if (dbsync2) cudaDeviceSynchronize();
        return g->bad;
    }
    float *bufA  = wallc(g, (size_t)cap);
    float *bufB  = wallc(g, (size_t)cap);
    float *cur   = wallc(g, (size_t)HID*T*F);
    if (!bufA || !bufB || !cur) return 1;
    const float *srcs[3] = { x_in, bufA, bufB };
    int cs = 0, ds = 1;                              /* source/destination slot ids */
    int c_skip = HID;
    char key[256], nkey[256], akey[256];
    for (int i = 0; i < DEPTH; i++) {
        snprintf(key, sizeof(key), "%s.%d.0.weight", prefix, i); const float *w = reuse_gpu_w(g, key);
        snprintf(key, sizeof(key), "%s.%d.0.bias",   prefix, i); const float *b = reuse_gpu_w(g, key);
        if (g->bad) return 1;
        int dil = 1 << i;
        launch_conv2d(g, srcs[cs], w, b, c_skip, T, F, HID, 3, 3, dil, 1, dil, 1, 1, 1, tmp, tilehint, NULL, -1);
        snprintf(nkey, sizeof(nkey), "%s.%d.1.weight", prefix, i);
        snprintf(akey, sizeof(akey), "%s.%d.2.weight", prefix, i);
        float *ocur = (i == DEPTH - 1) ? out : cur;
        inorm(g, tmp, ocur, 1, HID, T, F, nkey, akey);
        if (i != DEPTH - 1) {
            float *d2 = (ds == 1) ? bufA : bufB;
            cudaMemcpyAsync(d2, ocur, (size_t)HID*T*F*sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(d2 + (size_t)HID*T*F, srcs[cs], (size_t)c_skip*T*F*sizeof(float), cudaMemcpyDeviceToDevice);
            cs = ds; ds = (cs == 1) ? 2 : 1;      /* the next layer reads this slot,
                                                     writes the other arena buffer */
        }
        c_skip += HID;
    }
    static int dbsync = -1;
    if (dbsync < 0) dbsync = getenv("REUSE_DBSYNC") ? atoi(getenv("REUSE_DBSYNC")) : 0;
    if (dbsync) cudaDeviceSynchronize();            /* A/B: drains the pipeline per dense block */
    return g->bad;
}

static int sp_up(reuse_gpu_t *g, const char *prefix, const float *x, int H, int W, int r, float *y)
{
    char key[256];
    snprintf(key, sizeof(key), "%s.0.conv.weight", prefix); const float *w = reuse_gpu_w(g, key);
    snprintf(key, sizeof(key), "%s.0.conv.bias",   prefix); const float *b = reuse_gpu_w(g, key);
    if (g->bad) return 1;
    dim3 blk(16, 16);
    dim3 grd((unsigned)((W + 15)/16), (unsigned)((H + 15)/16), (unsigned)(HID*r));
    {
        static int t8 = -1;
        if (t8 < 0) t8 = getenv("REUSE_SPUP") ? atoi(getenv("REUSE_SPUP")) : 4;   /* 4 = block-tiled (default), 3 = k_sp_up8w, 1 = k_sp_up8, 0 = naive */
        if (t8 == 4) {
            const int CO = 64*r;
            static int RW_ = 0, RC_ = 0, TW_ = 0, NC_ = 0;
            if (RW_ == 0) {
                const char *e = getenv("REUSE_SPUP_T"); int a=0,b=0,c=0,d=0;
                RW_ = 4; RC_ = 8; TW_ = 32; NC_ = 64;      /* sweep winner: (8,8) threads,
                                                              32 w x 64 co per block, i.e. the
                                                              input strip is read once per 64
                                                              output channels instead of per 8 */
                if (e) sscanf(e, "%d,%d,%d,%d", &a,&b,&c,&d);
                if (a) { RW_ = a; RC_ = b; TW_ = c; NC_ = d; }
            }
            float *wT = wallc(g, (size_t)CO*HID*3);
            if (!wT) return 1;
            k_wtrans_sp<<<(unsigned)((CO*HID*3+255)/256), 256>>>(w, wT, CO);
            #define SPT(RWv,RCv,TWv,NCv) do { if (RW_==RWv && RC_==RCv && TW_==TWv && NC_==NCv) \
                k_sp_up8t<RWv,RCv,TWv,NCv><<<dim3((unsigned)((W+TWv-1)/TWv),(unsigned)H,(unsigned)(CO/NCv)), dim3(TWv/RWv,NCv/RCv)>>>(x,wT,b,y,H,W,r,CO); } while(0)
            SPT(8,8,32,64); SPT(8,8,64,64); SPT(4,8,32,32); SPT(8,4,32,32);
            SPT(4,4,32,32); SPT(8,8,32,32); SPT(4,8,16,32); SPT(8,16,64,64);
            SPT(4,8,32,64); SPT(8,16,32,64); SPT(8,8,16,64); SPT(8,8,32,128);
            SPT(4,8,64,64); SPT(4,8,32,128); SPT(4,8,64,128); SPT(4,16,32,64);
            SPT(4,4,32,64); SPT(2,8,32,64); SPT(4,8,16,64); SPT(8,8,64,128);
            #undef SPT
            return g->bad;
        }
        if (t8 == 3) {
            const int CO = 64*r;
            float *wT = wallc(g, (size_t)CO*HID*3);
            if (!wT) return 1;
            k_wtrans_sp<<<(unsigned)((CO*HID*3+255)/256), 256>>>(w, wT, CO);
            const int nc = getenv("REUSE_SPUP_NC") ? atoi(getenv("REUSE_SPUP_NC")) : 8;
            dim3 g8(grd.x, grd.y, (unsigned)(CO/nc));
            if (nc == 32)      k_sp_up8w<32><<<g8, blk>>>(x, wT, b, y, H, W, r, CO);
            else if (nc == 16) k_sp_up8w<16><<<g8, blk>>>(x, wT, b, y, H, W, r, CO);
            else               k_sp_up8w<8><<<g8, blk>>>(x, wT, b, y, H, W, r, CO);
        } else if (t8 == 2 && (64*r) % 8 == 0 && (r == 2 || r == 4)) {
            dim3 g8(grd.x, grd.y, (unsigned)((64*r)/8));
            if (r == 4) k_sp_up8v<4><<<g8, blk>>>(x, w, b, y, H, W, r);
            else        k_sp_up8v<2><<<g8, blk>>>(x, w, b, y, H, W, r);
        } else if (t8 && (64*r) % 8 == 0) {
            dim3 g8(grd.x, grd.y, (unsigned)((64*r)/8));
            k_sp_up8<<<g8, blk>>>(x, w, b, y, H, W, r);
        } else {
            k_sp_up<<<grd, blk>>>(x, w, b, y, H, W, r);
        }
    }
    return reuse_gpu_ok(g, "sp_up");
}

// ================================================================= encoder

extern "C" int reuse_gpu_encoder(reuse_gpu_t *g, const float *xin, float *enc, int Tp, int Fp)
{
    const int T2 = (Tp - 1)/4 + 1, F2 = (Fp - 3)/2 + 1;
    g->ws_off = 0;
    float *x1  = wallc(g, (size_t)HID*Tp*Fp);
    float *x1n = wallc(g, (size_t)HID*Tp*Fp);
    float *xdb = wallc(g, (size_t)HID*Tp*Fp);
    if (g->bad) return 1;
    float *x2  = wallc(g, (size_t)HID*T2*F2);
    if (!x1 || !x1n || !xdb || !x2) return 1;

    const float *w1 = reuse_gpu_w(g, "dense_encoder.dense_conv_1.0.weight");
    const float *b1 = reuse_gpu_w(g, "dense_encoder.dense_conv_1.0.bias");
    const float *w2 = reuse_gpu_w(g, "dense_encoder.dense_conv_2.0.weight");
    const float *b2 = reuse_gpu_w(g, "dense_encoder.dense_conv_2.0.bias");
    if (g->bad) return 1;
    if (!getenv("REUSE_SKIP_ENC1X1"))
        launch_conv1x1(g, xin, w1, b1, x1, 2, HID, (long)Tp*Fp);
    gdump("c1", x1, (size_t)HID*Tp*Fp);
    if (!getenv("REUSE_SKIP_IN1"))
        inorm(g, x1, x1n, 1, HID, Tp, Fp, "dense_encoder.dense_conv_1.1.weight", "dense_encoder.dense_conv_1.2.weight");
    gdump("c1n", x1n, (size_t)HID*Tp*Fp);
    if (!getenv("REUSE_SKIP_DB"))
        dense_block_gpu(g, "dense_encoder.dense_block.dense_block", x1n, Tp, Fp, xdb, 0);
    gdump("db", xdb, (size_t)HID*Tp*Fp);
    if (!getenv("REUSE_SKIP_ENC2D"))
        launch_conv2d(g, xdb, w2, b2, HID, Tp, Fp, HID, 1, 3, 0, 0, 1, 1, 4, 2, x2, 0, NULL, -1);
    gdump("c2", x2, (size_t)HID*T2*F2);
    if (!getenv("REUSE_SKIP_IN2"))
        inorm(g, x2, enc, 1, HID, T2, F2, "dense_encoder.dense_conv_2.1.weight", "dense_encoder.dense_conv_2.2.weight");
    gdump("enc", enc, (size_t)HID*T2*F2);
    return g->bad;
}

// ================================================================= Mamba

static int outproj_tiled(void)
{
    static int t = -1;
    if (t < 0) {
        t = getenv("REUSE_OP") ? atoi(getenv("REUSE_OP")) : (getenv("REUSE_NAIVE_OP") ? 0 : 3);
        /* k_out_proj_r8's 64 accumulators exceed Maxwell's per-thread
           register limit: it compiles for sm_53 and fails to launch. */
        if (t == 3 && gpu_is_maxwell()) t = 2;
    }
    return t;
}
static int mamba_dir_gpu(reuse_gpu_t *g, const char *pfx, const float *x, float *out,
                         int S, int L, int rev, cudaStream_t st,
                         float *cat = NULL, int coff = 0)
{
    char key[256];
    snprintf(key, sizeof(key), "%s.in_proj.weight", pfx);  const float *W_in  = reuse_gpu_w(g, key);
    snprintf(key, sizeof(key), "%s.conv1d.weight", pfx);   const float *W_c   = reuse_gpu_w(g, key);
    snprintf(key, sizeof(key), "%s.conv1d.bias", pfx);     const float *B_c   = reuse_gpu_w(g, key);
    snprintf(key, sizeof(key), "%s.x_proj.weight", pfx);   const float *W_xp  = reuse_gpu_w(g, key);
    snprintf(key, sizeof(key), "%s.dt_proj.weight", pfx);  const float *W_dt  = reuse_gpu_w(g, key);
    snprintf(key, sizeof(key), "%s.dt_proj.bias", pfx);    const float *B_dt  = reuse_gpu_w(g, key);
    snprintf(key, sizeof(key), "%s.A_log", pfx);           const float *A_log = reuse_gpu_w(g, key);
    snprintf(key, sizeof(key), "%s.D", pfx);               const float *Dv    = reuse_gpu_w(g, key);
    snprintf(key, sizeof(key), "%s.out_proj.weight", pfx); const float *W_op  = reuse_gpu_w(g, key);
    if (g->bad) return 1;
    long SL = (long)S * L;
    float *x0   = wallc(g, (size_t)SL*DINNER);
    float *z    = wallc(g, (size_t)SL*DINNER);
    float *xc   = wallc(g, (size_t)SL*DINNER);
    float *xdbl = wallc(g, (size_t)SL*XP);
    float *dtt  = wallc(g, (size_t)SL*DINNER);
    float *y    = wallc(g, (size_t)SL*DINNER);
    if (!x0 || !z || !xc || !xdbl || !dtt || !y) return 1;
    dim3 b1(256);
    static double klast = 0;
    if (g_kprof < 0) g_kprof = getenv("REUSE_KPROF") ? 1 : 0;
    if (g_kprof) { cudaDeviceSynchronize(); klast = gt_now(); }
    int dbg = getenv("GPU_DEBUG") && strncmp(pfx, "TSMamba.0.time_mamba.", 19) == 0;
    char dt[8]; snprintf(dt, sizeof(dt), "%s_", rev ? "bw" : "fw");
        #define GDBG(tag, ptr, n) do { if (dbg) { char t_[32]; snprintf(t_, sizeof(t_), "%s%s", dt, tag); gdump(t_, ptr, n); } } while (0)
    {
        const int tiled = inproj_mode();
        // REUSE_ZHALO=1 restores the shipped form (every block computes the
        // 3-row conv halo, including the z-half blocks that discard it).
        static int zhalo = -1;
        if (zhalo < 0) zhalo = getenv("REUSE_ZHALO") ? atoi(getenv("REUSE_ZHALO")) : 0;
        if (getenv("REUSE_SKIP_INPROJ")) { } else
        if (fuse_conv()) {
            const int kc = fuse_kc();
            if (fuse_tile() == 128) {          // 128 measured slower (0.638 vs 0.555 s):
                                               // the traffic model lost to 110 registers/block
                k_in_conv_r2<32><<<dim3((unsigned)((SL+FUSE_MT2-1)/FUSE_MT2), (2*DINNER)/64), dim3(16,16), 0, st>>>(
                    x, W_in, z, xc, W_c, B_c, S, L, rev, zhalo);
            }
            else if (kc == 16)
                k_in_conv_r<16, LBV><<<dim3((unsigned)((SL+FUSE_MT-1)/FUSE_MT), (2*DINNER)/64), dim3(16,16), 0, st>>>(
                    x, W_in, z, xc, W_c, B_c, S, L, rev, zhalo);
            else if (kc == 8)
                k_in_conv_r<8, LBV><<<dim3((unsigned)((SL+FUSE_MT-1)/FUSE_MT), (2*DINNER)/64), dim3(16,16), 0, st>>>(
                    x, W_in, z, xc, W_c, B_c, S, L, rev, zhalo);
            else
                k_in_conv_r<32, LBV><<<dim3((unsigned)((SL+FUSE_MT-1)/FUSE_MT), (2*DINNER)/64), dim3(16,16), 0, st>>>(
                    x, W_in, z, xc, W_c, B_c, S, L, rev, zhalo);
        }
        else if (tiled == 2) k_in_proj_r<<<dim3((unsigned)((SL+63)/64), (2*DINNER)/64), dim3(16,16), 0, st>>>(x, W_in, x0, z, S, L, rev);
        else if (tiled) k_in_proj_t<<<dim3((unsigned)((SL+MT32-1)/MT32), (2*DINNER)/NT64), dim3(16,16), 0, st>>>(x, W_in, x0, z, S, L, rev);
        else       k_in_proj<<<dim3((unsigned)((SL+255)/256), 2*DINNER), b1, 0, st>>>(x, W_in, x0, z, S, L, rev);
    }
    kp_mark("in_proj", &klast);
    GDBG("x0", x0, (size_t)SL*DINNER); GDBG("z", z, (size_t)SL*DINNER);
    {
        static int t = -1; if (t < 0) t = getenv("REUSE_NAIVE_CONV") ? 0 : 1;
        if (getenv("REUSE_SKIP_CONV") || fuse_conv()) { } else
        if (t) k_conv_causal_t<<<(unsigned)((SL+7)/8), 256, 0, st>>>(x0, W_c, B_c, xc, S, L);
        else   k_conv_causal<<<dim3((unsigned)((SL+255)/256), DINNER), b1, 0, st>>>(x0, W_c, B_c, xc, S, L);
    }
    kp_mark("conv", &klast);
    GDBG("gxact", xc, (size_t)SL*DINNER);
    {
        static int tiled = -1;
        if (tiled < 0) tiled = getenv("REUSE_XPROJ") ? atoi(getenv("REUSE_XPROJ"))
                                                      : (getenv("REUSE_NAIVE_XPROJ") ? 0 : 1284008);
        /* default 1284008 = k_x_proj_g<128,40,8>: 4 rows x 5 cols = 20 accumulators
           (2.2 FMA per shared load vs 1.6) and ONE tile column covering all XP=36
           outputs, where the shipped (128,32) shape needed two and the second had 4
           useful columns of 32. x_proj 36 ms -> ~27 ms, model -2.2%, bit-compatible.
           =2 restores k_x_proj_r, =0 the naive GEMV. */
        if (getenv("REUSE_SKIP_XPROJ")) { } else
        if (tiled == 2) k_x_proj_r<<<dim3((XP+31)/32, (unsigned)((SL+127)/128)), dim3(16,16), 0, st>>>(xc, W_xp, xdbl, SL);
        else if (tiled >= 100) {
            /* REUSE_XPROJ = MT*10000 + NT*100 + TX, e.g. 12804016 = MT=128 NT=40 TX=16 */
            const int gmt = tiled/10000, gnt = (tiled/100) % 100, gtx = tiled % 100;
            const dim3 grd((unsigned)((XP+gnt-1)/gnt), (unsigned)((SL+gmt-1)/gmt));
            #define XPG(MTv,NTv,TXv) do { if (gmt==MTv && gnt==NTv && gtx==TXv)                 k_x_proj_g<MTv,NTv,TXv><<<grd, dim3(TXv,256/TXv), 0, st>>>(xc, W_xp, xdbl, SL); } while (0)
            XPG(128,32,16); XPG(128,40,8);  XPG(256,40,8);  XPG(128,64,16);
            XPG(256,32,16); XPG(64,40,8);   XPG(128,48,16);
            XPG(64,32,16);  XPG(256,64,16); XPG(128,80,16); XPG(32,40,8);
            XPG(192,40,8);  XPG(128,40,4);  XPG(64,72,8);   XPG(96,40,8);
            XPG(128,72,8);  XPG(256,40,8);
            #undef XPG
        }
        else if (tiled) k_x_proj_t<<<dim3((XP+31)/32, (unsigned)((SL+63)/64)), dim3(16,16), 0, st>>>(xc, W_xp, xdbl, SL);
        else       k_x_proj<<<dim3((unsigned)((SL+255)/256), XP), b1, 0, st>>>(xc, W_xp, xdbl, SL);
    }
    kp_mark("x_proj", &klast);
    GDBG("gxdbl", xdbl, (size_t)SL*XP);
    {
        static int t = -1;
        if (t < 0) { const char *e = getenv("REUSE_DTPROJ"); t = e ? atoi(e) : (getenv("REUSE_NAIVE_DTPROJ") ? 0 : 2); }
        // dt_proj fused into the scan (default on; REUSE_SCANDT=0 restores the pair).
        static int f = -1;
        if (f < 0) f = getenv("REUSE_SCANDT") ? atoi(getenv("REUSE_SCANDT")) : 1;
        const bool fused = (f != 0) && t == 2 && !getenv("REUSE_SKIP_DTPROJ") && !getenv("REUSE_SKIP_SCAN");
        if (fused) {
            static int ts = -1;
            if (ts < 0) ts = getenv("REUSE_SCAN") ? atoi(getenv("REUSE_SCAN")) : 2;
            if (ts == 2) k_scan_dts<<<dim3((unsigned)((S+7)/8), DINNER/32), 256, 0, st>>>(xc, W_dt, B_dt, A_log, Dv, xdbl, y, S, L, rev);
            else         k_scan_dt<<<dim3((unsigned)((S+7)/8), DINNER/32), 256, 0, st>>>(xc, W_dt, B_dt, A_log, Dv, xdbl, y, S, L, rev);
        } else if (getenv("REUSE_SKIP_DTPROJ")) { } else
        if (t == 2) k_dt_proj_r<<<(unsigned)((SL+7)/8), 256, 0, st>>>(xdbl, W_dt, B_dt, dtt, SL);
        else if (t) k_dt_proj_t<<<(unsigned)((SL+7)/8), 256, 0, st>>>(xdbl, W_dt, B_dt, dtt, SL);
        else   k_dt_proj<<<dim3((unsigned)((SL+255)/256), DINNER), b1, 0, st>>>(xdbl, W_dt, B_dt, dtt, SL);
    }
    kp_mark("dt_proj", &klast);
    GDBG("gdt", dtt, (size_t)SL*DINNER);   // dtt is not written on the fused path
    {
        static int t = -1; if (t < 0) t = getenv("REUSE_NAIVE_SCAN") ? 0 : 1;
        static int f = -1; if (f < 0) f = getenv("REUSE_SCANDT") ? atoi(getenv("REUSE_SCANDT")) : 1;
        static int tp = -1; if (tp < 0) { const char *e = getenv("REUSE_DTPROJ"); tp = e ? atoi(e) : (getenv("REUSE_NAIVE_DTPROJ") ? 0 : 2); }
        const bool fused = (f != 0) && tp == 2 && !getenv("REUSE_SKIP_DTPROJ") && !getenv("REUSE_SKIP_SCAN");
        if (fused || getenv("REUSE_SKIP_SCAN")) { } else
        if (t) k_scan_b<<<dim3((unsigned)((S+7)/8), DINNER/32), 256, 0, st>>>(xc, dtt, A_log, Dv, xdbl, y, S, L, rev);
        else   k_scan<<<dim3((unsigned)S, DINNER/32), 32, 0, st>>>(xc, dtt, A_log, Dv, xdbl, y, S, L, rev);
    }
    kp_mark("scan", &klast);
    GDBG("gscan", y, (size_t)SL*DINNER);
/* Resolved out_proj kernel selector, shared by mamba_dir_gpu (which kernel
 * to launch) and mamba_block_gpu (whether the tower may skip the separate
 * concat). The fused concat epilogue exists ONLY in the tiled==3 kernels
 * (k_out_proj_r8c[_t]); k_out_proj_r/r8/tiled and the naive gate kernel all
 * write `out`, so the tower must run the separate concat for them. OP=2 is
 * the Maxwell production path, and with REUSE_CONCAT_FUSE defaulting to 1
 * the tower skipped the concat while k_out_proj_r wrote `out`, leaving
 * `cat` unfilled (model rel-rms 2.2 on every Nano run since the fuse was
 * introduced - invisible to the dev gates, which run OP=3). */
    {
        int tiled = outproj_tiled();
        if (tiled) {
            dim3 g((HID+GT-1)/GT, (unsigned)((SL+GT-1)/GT));
            static int cf = -1;
            if (cf < 0) cf = getenv("REUSE_CONCAT_FUSE") ? atoi(getenv("REUSE_CONCAT_FUSE")) : 1;
            static int apt = -1;
            if (apt < 0) apt = getenv("REUSE_APT") ? atoi(getenv("REUSE_APT")) : 1;
            if      (tiled == 3 && cf && cat && apt) k_out_proj_r8c_t<<<(unsigned)((SL+255)/256), dim3(8,32), 0, st>>>(y, z, W_op, x, cat, coff, L, rev, (int)SL);
            else if (tiled == 3 && cf && cat) k_out_proj_r8c<<<(unsigned)((SL+255)/256), dim3(8,32), 0, st>>>(y, z, W_op, x, cat, coff, L, rev, (int)SL);
            else if (tiled == 3) k_out_proj_r8<<<(unsigned)((SL+255)/256), dim3(8,32), 0, st>>>(y, z, W_op, out, (int)SL);
            else if (tiled == 2) k_out_proj_r<<<(unsigned)((SL+63)/64), dim3(16,16), 0, st>>>(y, z, W_op, out, (int)SL);
            else            k_out_proj_tiled<<<g, dim3(GT/2, GT/2), 0, st>>>(y, z, W_op, out, (int)SL);
        } else {
            if (!getenv("REUSE_SKIP_GATEOP"))
            k_gate_outproj<<<dim3((unsigned)((SL+255)/256), HID), b1, 0, st>>>(y, z, W_op, out, SL);
        }
    }
    kp_mark("gate_outproj", &klast);
    GDBG("gout", out, (size_t)SL*HID);
    return reuse_gpu_ok(g, "mamba_dir");
}

static int mamba_block_gpu(reuse_gpu_t *g, const char *pfx, const float *x, float *out,
                           int S, int L)
{
    long SL = (long)S * L;
    static int cfuse = -1;
    if (cfuse < 0) {
        cfuse = getenv("REUSE_CONCAT_FUSE") ? atoi(getenv("REUSE_CONCAT_FUSE")) : 1;
        /* The fused epilogue lives only in the tiled==3 kernels: with any
           other out_proj kernel the tower must run the separate concat,
           even if fusion was requested (see outproj_tiled). */
        if (cfuse && outproj_tiled() != 3) cfuse = 0;
    }
    static double klast = 0;
    if (g_kprof < 0) g_kprof = getenv("REUSE_KPROF") ? 1 : 0;
    if (g_kprof) { cudaDeviceSynchronize(); klast = gt_now(); }
    float *fw  = wallc(g, (size_t)SL*HID);
    float *bw  = wallc(g, (size_t)SL*HID);
    float *cat = wallc(g, (size_t)SL*2*HID);
    if (!fw || !bw || !cat) return 1;
    char p[256];
    snprintf(p, sizeof(p), "%s.forward_blocks", pfx);
    // The two scan directions are independent, so the backward one runs on a
    // side stream: each direction is a chain of 5 dependent launches, and the
    // ramp-down of one overlaps the ramp-up of the other.
    static cudaStream_t st_side = 0;
    static cudaEvent_t ev1, ev2;
    static int two = -1;
    if (two < 0) {
        two = getenv("REUSE_STREAMS") ? atoi(getenv("REUSE_STREAMS")) : 1;
        if (two && !st_side) {
            if (cudaStreamCreateWithFlags(&st_side, cudaStreamNonBlocking) ||
                cudaEventCreateWithFlags(&ev1, cudaEventDisableTiming) ||
                cudaEventCreateWithFlags(&ev2, cudaEventDisableTiming)) { g->bad = 1; return 1; }
        }
    }
    if (two) {
        if (cudaEventRecord(ev1, 0) || cudaStreamWaitEvent(st_side, ev1, 0)) { g->bad = 1; return 1; }
        if (mamba_dir_gpu(g, p, x, fw, S, L, 0, 0, cfuse ? cat : NULL, 0)) return 1;
        snprintf(p, sizeof(p), "%s.backward_blocks", pfx);
        if (mamba_dir_gpu(g, p, x, bw, S, L, 1, st_side, cfuse ? cat : NULL, HID)) return 1;
        if (cudaEventRecord(ev2, st_side) || cudaStreamWaitEvent(0, ev2, 0)) { g->bad = 1; return 1; }
    } else {
        if (mamba_dir_gpu(g, p, x, fw, S, L, 0, 0, cfuse ? cat : NULL, 0)) return 1;
        snprintf(p, sizeof(p), "%s.backward_blocks", pfx);
        if (mamba_dir_gpu(g, p, x, bw, S, L, 1, 0, cfuse ? cat : NULL, HID)) return 1;
    }
    if (cfuse && !getenv("REUSE_NAIVE_OUTPROJ")) {
        /* cat was filled by the fused out_proj epilogues of both directions */
    } else if (getenv("REUSE_SKIP_CONCAT")) {
    } else if (getenv("REUSE_NAIVE_CONCAT")) {
        k_concat<<<(unsigned)((SL+255)/256), 256>>>(fw, bw, x, cat, S, L);
    } else {
        k_concat_f<<<(unsigned)((SL+3)/4), dim3(64,4)>>>(fw, bw, x, cat, S, L);
    }
        kp_mark("concat", &klast);
    char key[256];
    snprintf(key, sizeof(key), "%s.output_proj.weight", pfx); const float *W = reuse_gpu_w(g, key);
    snprintf(key, sizeof(key), "%s.output_proj.bias",   pfx); const float *B = reuse_gpu_w(g, key);
    snprintf(key, sizeof(key), "%s.norm.weight",        pfx); const float *LW = reuse_gpu_w(g, key);
    snprintf(key, sizeof(key), "%s.norm.bias",          pfx); const float *LB = reuse_gpu_w(g, key);
    if (g->bad) return 1;
    if (getenv("REUSE_SKIP_OUTPROJ")) {
    } else if (getenv("REUSE_NAIVE_OUTPROJ")) {
        k_outproj_ln<<<(unsigned)SL, HID>>>(cat, W, B, LW, LB, out, SL);
    } else {
        k_outproj_ln_g<<<(unsigned)((SL+31)/32), dim3(16,8)>>>(cat, W, B, LW, LB, out, SL);
    }
    return reuse_gpu_ok(g, "mamba_block");
}

/* Tower resid chain, all in the "A" ([f*T+t]*HID+c) layout, so the [C,T,F]
   tensor that k_resid_scatter_out_t wrote and the next block's k_gather_time_t
   read straight back never exists: the two transposes compose to the identity
   and the block residual becomes a flat add. freq input:
       xf[t*F+f][c] = yt[f*T+t][c] + xt[f*T+t][c]
   (identical arithmetic to k_resid_scatter_freq_t, with its [C,T,F] operand
   supplied in the gather form it was produced in). */
__global__ void k_resid_freq_a(const float *__restrict__ yt, const float *__restrict__ xt,
                              float *__restrict__ xf, int T, int F)
{
    const long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (long)F*T*HID) return;
    const int f = (int)(i / ((long)T*HID));
    const int r = (int)(i - (long)f*T*HID);
    xf[((long)(r/HID)*F + f)*HID + (r - (r/HID)*HID)] = yt[i] + xt[i];
}

/* Next block's time input: xt_next[f*T+t][c] = yf[t*F+f][c] + xf[t*F+f][c]. */
__global__ void k_resid_out_a(const float *__restrict__ yf, const float *__restrict__ xf,
                              float *__restrict__ xa, int T, int F)
{
    const long i = (long)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= (long)F*T*HID) return;
    /* reads are linear in the freq sub-block's [t*F+f][c] layout; the time
       sub-block's input is [f*T+t][c], so the store carries the (f,t) swap */
    const int t = (int)(i / ((long)F*HID));
    const int r = (int)(i - (long)t*F*HID);
    xa[(long)(r/HID)*T*HID + (long)t*HID + (r - (r/HID)*HID)] = yf[i] + xf[i];
}

static int reuse_gpu_tfmamba_a(reuse_gpu_t *g, int idx, const float *xt_in,
                               float *out_B, float *xt_next, int T, int F, int last)
{
    static double klast = 0;
    if (g_kprof < 0) g_kprof = getenv("REUSE_KPROF") ? 1 : 0;
    if (g_kprof) { cudaDeviceSynchronize(); klast = gt_now(); }
    g->ws_off = 0;
    const int S_t = F, S_f = T;
    const long n = (long)F*T*HID;
    float *yt = wallc(g, (size_t)n);
    float *xf = wallc(g, (size_t)n);
    float *yf = wallc(g, (size_t)n);
    if (!yt || !xf || !yf) return 1;
    char p[256];
    snprintf(p, sizeof(p), "TSMamba.%d.time_mamba", idx);
    if (mamba_block_gpu(g, p, xt_in, yt, S_t, T)) return 1;

    k_resid_freq_a<<<(unsigned)((n+255)/256), 256>>>(yt, xt_in, xf, T, F);
    kp_mark("resid_freq", &klast);
    snprintf(p, sizeof(p), "TSMamba.%d.freq_mamba", idx);
    if (mamba_block_gpu(g, p, xf, yf, S_f, F)) return 1;

    if (last) k_resid_scatter_out_t<<<dim3((unsigned)((F+31)/32), HID/32, (unsigned)T), dim3(32,32)>>>(yf, xf, out_B, T, F);
    else      k_resid_out_a<<<(unsigned)((n+255)/256), 256>>>(yf, xf, xt_next, T, F);
    kp_mark("resid_out", &klast);
    return reuse_gpu_ok(g, "tfmamba_a");
}

extern "C" int reuse_gpu_tfmamba(reuse_gpu_t *g, int idx, const float *x, float *y, int T, int F)
{
    static double klast = 0;
    if (g_kprof < 0) g_kprof = getenv("REUSE_KPROF") ? 1 : 0;
    if (g_kprof) { cudaDeviceSynchronize(); klast = gt_now(); }

    g->ws_off = 0;
    const int S_t = F, S_f = T;
    float *xt = wallc(g, (size_t)F*T*HID);
    float *yt = wallc(g, (size_t)F*T*HID);
    float *xf = wallc(g, (size_t)F*T*HID);
    float *yf = wallc(g, (size_t)F*T*HID);
    if (!xt || !yt || !xf || !yf) return 1;
    {
        static int t = -1;
        if (t < 0) t = getenv("REUSE_GATHER") ? atoi(getenv("REUSE_GATHER")) : 1;
        if (t) k_gather_time_t<<<dim3((unsigned)((F+31)/32), (unsigned)((HID+31)/32), (unsigned)T), dim3(32,8)>>>(x, xt, T, F);
        else   k_gather_time<<<(unsigned)(((long)F*T*HID+255)/256), 256>>>(x, xt, T, F);
    }
    char p[256];
    snprintf(p, sizeof(p), "TSMamba.%d.time_mamba", idx);
    if (mamba_block_gpu(g, p, xt, yt, S_t, T)) return 1;
    if (getenv("REUSE_SKIP_RESID")) { } else if (naive_resid())
        k_resid_scatter_freq<<<(unsigned)(((long)F*T*HID+255)/256), 256>>>(yt, x, y, xf, T, F);
    else
        k_resid_scatter_freq_t<<<dim3((unsigned)((F+31)/32), HID/32, (unsigned)T), dim3(32,32)>>>(yt, x, y, xf, T, F);
        kp_mark("resid_freq", &klast);
    snprintf(p, sizeof(p), "TSMamba.%d.freq_mamba", idx);
    if (mamba_block_gpu(g, p, xf, yf, S_f, F)) return 1;

    if (getenv("REUSE_SKIP_RESID")) { } else if (naive_resid())
        k_resid_scatter_out<<<(unsigned)(((long)F*T*HID+255)/256), 256>>>(yf, xf, y, T, F);
    else
        k_resid_scatter_out_t<<<dim3((unsigned)((F+31)/32), HID/32, (unsigned)T), dim3(32,32)>>>(yf, xf, y, T, F);
        kp_mark("resid_out", &klast);
    return reuse_gpu_ok(g, "tfmamba");
}

// ================================================================= decoders

extern "C" int reuse_gpu_decoder_pre(reuse_gpu_t *g, const char *prefix, const float *x,
                                     float *y, int T, int F)
{
    g->ws_off = 0;
    char key[256], dbp[256];
    snprintf(dbp, sizeof(dbp), "%s.dense_block.dense_block", prefix);
    float *d = wallc(g, (size_t)HID*T*F);
    if (!d) return 1;
    if (dense_block_gpu(g, dbp, x, T, F, d, dectile())) return 1;

    gdump("d_db", d, (size_t)HID*T*F);

    float *u1  = wallc(g, (size_t)HID*T*(size_t)(F*2));
    float *u1n = wallc(g, (size_t)HID*T*(size_t)(F*2));
    if (!u1 || !u1n) return 1;
    snprintf(key, sizeof(key), "%s.up_conv1", prefix);
    if (sp_up(g, key, d, T, F, 2, u1)) return 1;

    gdump("d_u1", u1, (size_t)HID*T*(size_t)(F*2));
    {
        char nk[256], ak[256];
        snprintf(nk, sizeof(nk), "%s.up_conv1.1.weight", prefix);
        snprintf(ak, sizeof(ak), "%s.up_conv1.2.weight", prefix);
        inorm(g, u1, u1n, 1, HID, T, F*2, nk, ak);
    }

    gdump("d_u1n", u1n, (size_t)HID*T*(size_t)(F*2));
    // up_conv2 runs on the (T,F)-transposed tensor
    const int Tt = F*2, Ft = T;
    float *sw = wallc(g, (size_t)HID*Tt*Ft);
    if (!sw) return 1;
    launch_transpose(HID, T, F*2, u1n, sw);

    gdump("d_sw", sw, (size_t)HID*Tt*Ft);
    float *u2  = wallc(g, (size_t)HID*Tt*(size_t)(Ft*4));
    float *u2n = wallc(g, (size_t)HID*Tt*(size_t)(Ft*4));
    if (!u2 || !u2n) return 1;

    snprintf(key, sizeof(key), "%s.up_conv2", prefix);
    if (sp_up(g, key, sw, Tt, Ft, 4, u2)) return 1;

    gdump("d_u2", u2, (size_t)HID*Tt*(size_t)(Ft*4));
    {
        char nk[256], ak[256];
        snprintf(nk, sizeof(nk), "%s.up_conv2.1.weight", prefix);
        snprintf(ak, sizeof(ak), "%s.up_conv2.2.weight", prefix);
        inorm(g, u2, u2n, 1, HID, Tt, Ft*4, nk, ak);
    }
    launch_transpose(HID, Tt, Ft*4, u2n, y);

    gdump("d_pre", y, (size_t)HID*Tt*(size_t)(Ft*4));
    return g->bad;
}

extern "C" int reuse_gpu_conv1x1_to1(reuse_gpu_t *g, const float *x, const char *pfx,
                                     float *y, int C_in, int H, int W)
{
    char key[256];
    snprintf(key, sizeof(key), "%s.weight", pfx); const float *w = reuse_gpu_w(g, key);
    snprintf(key, sizeof(key), "%s.bias",   pfx); const float *b = reuse_gpu_w(g, key);
    if (g->bad) return 1;
    return launch_conv1x1(g, x, w, b, y, C_in, 1, (long)H*W);
}

// ================================================================= full pass

extern "C" int reuse_gpu_forward(reuse_gpu_t *g, const float *mag, const float *pha,
                                 int F, int T, float *amp_out, float *pha_out)
{
    prewarm(g);
    const int Tp = T + 2, Fp = F + 2;
    float *xin = (float*)dalloc((size_t)2*Tp*Fp);
    float *enc = (float*)dalloc((size_t)HID*((Tp-1)/4+1)*((Fp-3)/2+1));
    float *blk = (float*)dalloc((size_t)HID*((Tp-1)/4+1)*((Fp-3)/2+1));
    if (!xin || !enc || !blk) return 1;
    cudaMemset(xin, 0, (size_t)2*Tp*Fp*sizeof(float));
    k_build_input<<<(unsigned)(((long)T*F+255)/256), 256>>>(mag, pha, xin, T, F);
    if (getenv("REUSE_MEM")) { size_t fr=0,to=0; cudaMemGetInfo(&fr,&to);
        fprintf(stderr, "[mem] forward entry: free %zu MiB / %zu MiB\n", fr>>20, to>>20); }
    double t0 = gt_now();
    if (reuse_gpu_encoder(g, xin, enc, Tp, Fp)) return 1;
    cudaDeviceSynchronize(); double t1 = gt_now(); g_t_enc = t1-t0;
    const int T2 = (Tp-1)/4 + 1, F2 = (Fp-3)/2 + 1;
    // Ping-pong the tower buffers: the per-block blk->enc copy is HID*T2*F2
    // floats (10.5 MB at 8 kHz, 315 MB per forward), pure overhead. `cur`/`nxt`
    // swap so each block reads what the previous one wrote. The decoders read
    // whatever `cur` ends up being; with an even block count that is `blk`.
    const int NBLK = 30;
    float *cur = enc, *nxt = blk;
    if (getenv("REUSE_COPYBACK")) {           /* A/B: restore the copy form */
        for (int i = 0; i < NBLK; i++) {
            if (reuse_gpu_tfmamba(g, i, enc, blk, T2, F2)) return 1;
            cudaMemcpyAsync(enc, blk, (size_t)HID*T2*F2*sizeof(float), cudaMemcpyDeviceToDevice);
        }
    } else {
        static int tf = -1;
        if (tf < 0) tf = getenv("REUSE_TOWER_FUSE") ? atoi(getenv("REUSE_TOWER_FUSE")) : 1;
        float *xa = NULL, *xb = NULL;
        if (tf) {
            /* Keep the tower in the [F,T,C] layout: one gather before the loop,
               and the per-block [C,T,F] write + re-transpose (21 MB and two
               launches per block) collapses into the two flat adds above. */
            xa = (float*)dalloc((size_t)HID*T2*F2);
            xb = (float*)dalloc((size_t)HID*T2*F2);
            if (!xa || !xb) return 1;

            k_gather_time_t<<<dim3((unsigned)((F2+31)/32), (unsigned)((HID+31)/32), (unsigned)T2), dim3(32,8)>>>(cur, xa, T2, F2);

        }
        if (tf) {
            for (int i = 0; i < NBLK; i++) {
                /* the last block's [C,T,F] output goes into `enc`, which the gather
                   consumed at loop entry and nothing else reads, so the forward's
                   tail keeps freeing exactly xin/enc/blk as before */
                if (reuse_gpu_tfmamba_a(g, i, xa, (i == NBLK-1 ? enc : NULL),
                                        (i == NBLK-1 ? NULL : xb), T2, F2, i == NBLK-1)) return 1;
                float *sw = xa; xa = xb; xb = sw;
            }
            cur = enc;                                       /* B-layout output for decoders */
            dfree(xa); dfree(xb);
            xa = xb = NULL;
        } else {
            for (int i = 0; i < NBLK; i++) {
                if (reuse_gpu_tfmamba(g, i, cur, nxt, T2, F2)) return 1;
                float *sw = cur; cur = nxt; nxt = sw;
            }
        }
    }
    enc = cur;

    cudaDeviceSynchronize(); double t2 = gt_now(); g_t_blk = t2-t1;
    if (getenv("REUSE_MEM")) { size_t fr=0,to=0; cudaMemGetInfo(&fr,&to);
        fprintf(stderr, "[mem] after 30 blocks: free %zu MiB / %zu MiB\n", fr>>20, to>>20); }
    const int Tx = T2*4, Fx = F2*2;
    float *dm = (float*)dalloc((size_t)HID*Tx*Fx);
    float *dp = (float*)dalloc((size_t)HID*Tx*Fx);
    float *fa = (float*)dalloc((size_t)Tx*Fx);
    float *fr = (float*)dalloc((size_t)Tx*Fx);
    float *fi = (float*)dalloc((size_t)Tx*Fx);
    if (!dm || !dp || !fa || !fr || !fi) return 1;
    int dp_rc = reuse_gpu_decoders_pair(g, enc, dm, dp, T2, F2);
    if (dp_rc == 1) return 1;
    if (dp_rc == 2) {
        if (reuse_gpu_decoder_pre(g, "mask_decoder", enc, dm, T2, F2)) return 1;
        if (reuse_gpu_decoder_pre(g, "phase_decoder", enc, dp, T2, F2)) return 1;
    }
    cudaDeviceSynchronize(); double t3 = gt_now(); g_t_dec = t3-t2;
    if (reuse_gpu_conv1x1_to1(g, dm, "mask_decoder.final_conv", fa, HID, Tx, Fx)) return 1;
    if (reuse_gpu_conv1x1_to1(g, dp, "phase_decoder.phase_conv_r", fr, HID, Tx, Fx)) return 1;
    if (reuse_gpu_conv1x1_to1(g, dp, "phase_decoder.phase_conv_i", fi, HID, Tx, Fx)) return 1;
    k_atan2_head<<<(unsigned)(((long)T*F+255)/256), 256>>>(fr, fi, pha_out, Tx, Fx, T, F);
    // crop [Tx,Fx] -> [T,F] (amp)
    k_crop_ft<<<(unsigned)(((long)T*F+255)/256), 256>>>(fa, amp_out, T, F, Fx);
    k_sweep_mask<<<(unsigned)((T+255)/256), 256>>>(amp_out, T, F);
    cudaDeviceSynchronize(); g_t_head = gt_now()-t3;
    dfree(dm); dfree(dp); dfree(fa); dfree(fr); dfree(fi);
    dfree(xin); dfree(enc); dfree(blk);
    return reuse_gpu_ok(g, "forward");
}
