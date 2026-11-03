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
        if (cudaMemset(g->ws, 0, cap * sizeof(float)) != cudaSuccess) {
            cudaGetLastError(); cudaFree(g->ws); g->ws = NULL; continue;
        }
        g->ws_cap = cap;
        if (mb != want_mb) fprintf(stderr, "workspace: %zu MiB (requested %zu MiB)\n", mb, want_mb);
        break;
    }
    if (!g->ws) { fprintf(stderr, "gpu workspace OOM (requested %zu MiB)\n", want_mb); reuse_gpu_destroy(g); return NULL; }
    return g;
}

extern "C" const float *reuse_gpu_w(reuse_gpu_t *g, const char *name)
{
    entry key; key.name = name;
    entry *r = (entry*)bsearch(&key, g->ents, (size_t)g->n, sizeof(entry), cmp_str);
    if (!r) { fprintf(stderr, "MISSING GPU TENSOR: %s\n", name); g->bad = 1; return NULL; }
    return r->dptr;
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

static void *dalloc(size_t n)
{
    void *p = NULL;
    if (cudaMalloc(&p, n * sizeof(float)) != cudaSuccess) { fprintf(stderr, "gpu OOM %zu\n", n); return NULL; }
    return p;
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
template<int KC>
__global__ void k_in_conv_r(const float *x, const float *W, float *z, float *xc,
                            const float *Wc, const float *Bc,
                            int S, int L, int rev)
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
    const bool halo_thr = tid < 3*NT;
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
                const long m = m0 - 3 + r;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (q < KC && m >= 0 && m < SL) {
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
    __syncthreads();
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
#pragma unroll
        for (int b4 = 0; b4 < 4; b4++) xc[m*DINNER + c0 + b4] = dsilu(fv[b4]);
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
                             int S, int L, int rev)
{
    const int NT = 64, R = FUSE_MT2 + 3, RA = 8;
    __shared__ __align__(16) float Xs[R][KC+4], Ws[NT][KC+4], Hl[3][NT+1];
    const long SL = (long)S * L;
    const long m0 = (long)blockIdx.x * FUSE_MT2;
    const int n0 = blockIdx.y * NT;
    const int tx = threadIdx.x, ty = threadIdx.y;                 // (16,16)
    const int tid = ty*16 + tx;
    const bool halo_thr = tid < 3*NT;
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
                const long m = m0 - 3 + r;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (q < KC && m >= 0 && m < SL) {
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
    __syncthreads();
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

__global__ void k_transpose(const float *x, float *y, int C, int H, int W)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)C*H*W) return;
    int w = (int)(i % W), h = (int)((i / W) % H), c = (int)(i / ((long)H*W));
    y[((long)c*W + w)*(long)H + h] = x[i];
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
    double *xw  = (double*)wallc(g, (size_t)num_frames*n_fft*2);
    double *Cc  = (double*)wallc(g, (size_t)F*n_fft*2);
    double *Cs  = (double*)wallc(g, (size_t)F*n_fft*2);
    double *dre = (double*)wallc(g, (size_t)F*num_frames*2);
    double *dim = (double*)wallc(g, (size_t)F*num_frames*2);
    float  *w   = (float*) wallc(g, (size_t)n_fft);
    if (!xw || !Cc || !Cs || !dre || !dim || !w) return 1;
    float *hw = (float*)malloc((size_t)n_fft*sizeof(float));
    double *hc = (double*)malloc((size_t)F*n_fft*sizeof(double));
    double *hs = (double*)malloc((size_t)F*n_fft*sizeof(double));
    if (!hw || !hc || !hs) { free(hw); free(hc); free(hs); return 1; }
    for (int i = 0; i < n_fft; i++)
        hw[i] = (i < win) ? 0.5f*(1.0f - cosf(2.0f*(float)M_PI*i/(float)win)) : 0.0f;
    const double pi = 3.14159265358979323846;
    for (int k = 0; k < F; k++)
        for (int n = 0; n < n_fft; n++) {
            double ang = -2.0*pi*((double)n*(double)k)/(double)n_fft;
            hc[(size_t)k*n_fft+n] = cos(ang);
            hs[(size_t)k*n_fft+n] = sin(ang);
        }
    if (cudaMemcpy(w, hw, (size_t)n_fft*sizeof(float), cudaMemcpyHostToDevice) ||
        cudaMemcpy(Cc, hc, (size_t)F*n_fft*sizeof(double), cudaMemcpyHostToDevice) ||
        cudaMemcpy(Cs, hs, (size_t)F*n_fft*sizeof(double), cudaMemcpyHostToDevice)) {
        free(hw); free(hc); free(hs); return reuse_gpu_ok(g, "stft memcpy");
    }
    free(hw); free(hc); free(hs);
    long pn = (long)num_frames*n_fft;
    k_stft_prep<<<(unsigned)((pn+255)/256), 256>>>(d_wav, T, xw, n_fft, hop, n_fft/2, w);
    k_stft_dft<<<dim3((unsigned)((num_frames+15)/16), (unsigned)((F+15)/16)), dim3(16,16)>>>(
        xw, Cc, Cs, dre, dim, F, num_frames, n_fft);
    long on = (long)F*num_frames;
    k_stft_head<<<(unsigned)((on+255)/256), 256>>>(dre, dim, d_mag, d_pha, on);
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
__global__ void k_idft(const float *Cr, const float *Ci, const float *Cc, const float *Cs,
                       float *Y, int F, int T, int n_fft)
{
    // Y[n][t] = sum_k Cr[k][t]*Cc[n][k] + Ci[k][t]*Cs[n][k], accumulated in double
    // (the CPU reference sums in double; float accumulation would not match).
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int t0 = blockIdx.y * 16;
    if (n >= n_fft) return;
    double acc[16];
#pragma unroll
    for (int i2 = 0; i2 < 16; i2++) acc[i2] = 0.0;
    for (int k = 0; k < F; k++) {
        double cv = (double)Cc[(long)n*F + k], sv = (double)Cs[(long)n*F + k];
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
    // host-side tables in double, stored as float (matches the CPU reference)
    float *Cc = (float*)malloc((size_t)n_fft*F*sizeof(float));
    float *Cs = (float*)malloc((size_t)n_fft*F*sizeof(float));
    float *hw = (float*)malloc((size_t)n_fft*sizeof(float));
    for (int n = 0; n < n_fft; n++) {
        for (int k = 0; k < F; k++) {
            double m = (k == 0 || 2*k == n_fft) ? 1.0 : 2.0;
            double ang = 2.0 * M_PI * n * k / (double)n_fft;
            Cc[(long)n*F + k] = (float)( m * cos(ang) / n_fft);
            Cs[(long)n*F + k] = (float)(-m * sin(ang) / n_fft);
        }
    }
    for (int i = 0; i < n_fft; i++) hw[i] = 1.0f;
    for (int i = 0; i < win && i < n_fft; i++)
        hw[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / win));
    float *d_Cc, *d_Cs, *d_w, *d_Cr, *d_Ci, *d_Y, *d_wsum;
    size_t nf = (size_t)n_fft*F, ft = (size_t)F*T, nt = (size_t)n_fft*T;
    if (cudaMalloc(&d_Cc, nf*sizeof(float)) || cudaMalloc(&d_Cs, nf*sizeof(float)) ||
        cudaMalloc(&d_w, n_fft*sizeof(float)) || cudaMalloc(&d_Cr, ft*sizeof(float)) ||
        cudaMalloc(&d_Ci, ft*sizeof(float)) || cudaMalloc(&d_Y, nt*sizeof(float)) ||
        cudaMalloc(&d_wsum, (size_t)out_len*sizeof(float))) {
        fprintf(stderr, "istft: gpu OOM\n"); cudaGetLastError();
        free(Cc); free(Cs); free(hw); return 1;
    }
    cudaMemcpy(d_Cc, Cc, nf*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Cs, Cs, nf*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, hw, n_fft*sizeof(float), cudaMemcpyHostToDevice);
    // mag -> linear (expm1(relu)) then split phase
    long nsp = ft;
    k_mag_pha_split<<<(unsigned)((nsp+255)/256), 256>>>(d_mag, d_pha, d_Cr, d_Ci, nsp);
    k_idft<<<dim3((unsigned)((n_fft+127)/128), (unsigned)((T+15)/16)), 128>>>(
        d_Cr, d_Ci, d_Cc, d_Cs, d_Y, F, T, n_fft);
    cudaMemset(d_out, 0, (size_t)out_len*sizeof(float));
    cudaMemset(d_wsum, 0, (size_t)out_len*sizeof(float));
    k_ola<<<dim3((unsigned)((n_fft+127)/128), (unsigned)T), dim3(128, 1)>>>(
        d_Y, d_w, d_out, d_wsum, F, T, n_fft, hop, n_fft/2, out_len);
    k_norm_ola<<<(unsigned)((out_len+255)/256), 256>>>(d_out, d_wsum, out_len);
    int rc = reuse_gpu_ok(g, "istft_gpu");
    cudaFree(d_Cc); cudaFree(d_Cs); cudaFree(d_w); cudaFree(d_Cr); cudaFree(d_Ci);
    cudaFree(d_Y); cudaFree(d_wsum);
    free(Cc); free(Cs); free(hw);
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
template<int R>
__global__ void k_conv2d_r2(const float *x, const float *w, const float *b,
                            int C_in, int H, int W, int padH, int padW, int dilH,
                            float *y, int co0)
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
template<int R, int PCO, int NCB>
__global__ void k_conv2d_r8(const float *x, const float *w, const float *b,
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

static int conv_chunk(void)
{
    static int c = -1;
    if (c < 0) { const char *e = getenv("REUSE_CONV_CHUNK"); c = e ? atoi(e) : 32; if (c <= 0) c = 1 << 30; }
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
            if (regtile && dilH == 1)      k_conv2d_r8<RN,PC,NC><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0); \
            else if (regtile && dilH == 2) k_conv2d_r8<RN+2,PC,NC><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0); \
            else if (regtile && dilH == 4) k_conv2d_r8<RN+8,PC,NC><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0); \
            else if (regtile && dilH == 8) k_conv2d_r8<RN+24,PC,NC><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0); \
        } while (0)
#define R8LAUNCH(R1,R2,R4,R8) \
            if      (regtile && dilH == 1) { if (pco==4) k_conv2d_r8<R1,4,4><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0); else if (pco==8) k_conv2d_r8<R1,8,2><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0); else k_conv2d_r8<R1,16,2><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0); } \
            else if (regtile && dilH == 2) { if (pco==4) k_conv2d_r8<R2,4,4><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0); else if (pco==8) k_conv2d_r8<R2,8,2><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0); else k_conv2d_r8<R2,16,2><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0); } \
            else if (regtile && dilH == 4) { if (pco==4) k_conv2d_r8<R4,4,4><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0); else if (pco==8) k_conv2d_r8<R4,8,2><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0); else k_conv2d_r8<R4,16,2><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0); } \
            else if (regtile && dilH == 8) { if (pco==4) k_conv2d_r8<R8,4,4><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0); else if (pco==8) k_conv2d_r8<R8,8,2><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0); else k_conv2d_r8<R8,16,2><<<grdb,blkb>>>(x,w,b,C_in,H,W,padH,padW,dilH,y,co0); }

static int launch_conv2d(reuse_gpu_t *g, const float *x, const float *w, const float *b,
                         int C_in, int H, int W, int C_out, int kH, int kW,
                         int padH, int padW, int dilH, int dilW, int strideH, int strideW,
                         float *y)
{
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
        static int big = -1;
        if (big < 0) { const char *e = getenv("REUSE_CONV_TILE"); big = e ? atoi(e) : 82; }
        if (big >= 81 && big <= 83 && n % (big == 81 ? 4 : (big == 82 ? 8 : 16)) == 0) {
            const int pco = (big == 81) ? 4 : (big == 82 ? 8 : 16), ncb = (big == 81) ? 4 : 2;
            const int trh = (big == 83) ? 16 : 32;
            const dim3 grdb((unsigned)((W_out + 16*ncb - 1) / (16*ncb)), (unsigned)((H_out + trh - 1) / trh), (unsigned)(n/pco));
            const dim3 blkb(16, 16);
            R8LAUNCH(34,36,40,48)
            else k_conv2d<<<grd, blk>>>(x, w, b, C_in, H, W, C_out, kH, kW, padH, padW, dilH, dilW, strideH, strideW, y, co0);
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
            if (regtile && dilH == 1)      k_conv2d_r2<66><<<grdb, blkb>>>(x, w, b, C_in, H, W, padH, padW, dilH, y, co0);
            else if (regtile && dilH == 2) k_conv2d_r2<68><<<grdb, blkb>>>(x, w, b, C_in, H, W, padH, padW, dilH, y, co0);
            else if (regtile && dilH == 4) k_conv2d_r2<72><<<grdb, blkb>>>(x, w, b, C_in, H, W, padH, padW, dilH, y, co0);
            else if (regtile && dilH == 8) k_conv2d_r2<80><<<grdb, blkb>>>(x, w, b, C_in, H, W, padH, padW, dilH, y, co0);
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
                           int T, int F, float *out)
{
    const int cap = HID * (DEPTH + 1) * T * F;
    float *skip  = wallc(g, (size_t)cap);
    float *skip2 = wallc(g, (size_t)cap);
    float *tmp   = wallc(g, (size_t)HID*T*F);
    float *cur   = wallc(g, (size_t)HID*T*F);
    if (!skip || !skip2 || !tmp || !cur) return 1;
    cudaMemcpyAsync(skip, x_in, (size_t)HID*T*F*sizeof(float), cudaMemcpyDeviceToDevice);
    int c_skip = HID;
    char key[256], nkey[256], akey[256];
    for (int i = 0; i < DEPTH; i++) {
        snprintf(key, sizeof(key), "%s.%d.0.weight", prefix, i); const float *w = reuse_gpu_w(g, key);
        snprintf(key, sizeof(key), "%s.%d.0.bias",   prefix, i); const float *b = reuse_gpu_w(g, key);
        if (g->bad) return 1;
        int dil = 1 << i;
        launch_conv2d(g, skip, w, b, c_skip, T, F, HID, 3, 3, dil, 1, dil, 1, 1, 1, tmp);
        snprintf(nkey, sizeof(nkey), "%s.%d.1.weight", prefix, i);
        snprintf(akey, sizeof(akey), "%s.%d.2.weight", prefix, i);
        inorm(g, tmp, cur, 1, HID, T, F, nkey, akey);
        cudaMemcpyAsync(skip2, cur, (size_t)HID*T*F*sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(skip2 + (size_t)HID*T*F, skip, (size_t)c_skip*T*F*sizeof(float), cudaMemcpyDeviceToDevice);
        float *sw = skip; skip = skip2; skip2 = sw;
        c_skip += HID;
    }
    cudaMemcpyAsync(out, cur, (size_t)HID*T*F*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
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
        if (t8 < 0) t8 = getenv("REUSE_SPUP") ? atoi(getenv("REUSE_SPUP")) : 1;
        if (t8 && (64*r) % 8 == 0) {
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
        dense_block_gpu(g, "dense_encoder.dense_block.dense_block", x1n, Tp, Fp, xdb);
    gdump("db", xdb, (size_t)HID*Tp*Fp);
    if (!getenv("REUSE_SKIP_ENC2D"))
        launch_conv2d(g, xdb, w2, b2, HID, Tp, Fp, HID, 1, 3, 0, 0, 1, 1, 4, 2, x2);
    gdump("c2", x2, (size_t)HID*T2*F2);
    if (!getenv("REUSE_SKIP_IN2"))
        inorm(g, x2, enc, 1, HID, T2, F2, "dense_encoder.dense_conv_2.1.weight", "dense_encoder.dense_conv_2.2.weight");
    gdump("enc", enc, (size_t)HID*T2*F2);
    return g->bad;
}

// ================================================================= Mamba

static int mamba_dir_gpu(reuse_gpu_t *g, const char *pfx, const float *x, float *out,
                         int S, int L, int rev, cudaStream_t st)
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
        if (getenv("REUSE_SKIP_INPROJ")) { } else
        if (fuse_conv()) {
            const int kc = fuse_kc();
            if (fuse_tile() == 128) {          // 128 measured slower (0.638 vs 0.555 s):
                                               // the traffic model lost to 110 registers/block
                k_in_conv_r2<32><<<dim3((unsigned)((SL+FUSE_MT2-1)/FUSE_MT2), (2*DINNER)/64), dim3(16,16), 0, st>>>(
                    x, W_in, z, xc, W_c, B_c, S, L, rev);
            }
            else if (kc == 16)
                k_in_conv_r<16><<<dim3((unsigned)((SL+FUSE_MT-1)/FUSE_MT), (2*DINNER)/64), dim3(16,16), 0, st>>>(
                    x, W_in, z, xc, W_c, B_c, S, L, rev);
            else if (kc == 8)
                k_in_conv_r<8><<<dim3((unsigned)((SL+FUSE_MT-1)/FUSE_MT), (2*DINNER)/64), dim3(16,16), 0, st>>>(
                    x, W_in, z, xc, W_c, B_c, S, L, rev);
            else
                k_in_conv_r<32><<<dim3((unsigned)((SL+FUSE_MT-1)/FUSE_MT), (2*DINNER)/64), dim3(16,16), 0, st>>>(
                    x, W_in, z, xc, W_c, B_c, S, L, rev);
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
        if (tiled < 0) tiled = getenv("REUSE_NAIVE_XPROJ") ? 0 : 1;
        if (getenv("REUSE_SKIP_XPROJ")) { } else
        if (tiled) k_x_proj_t<<<dim3((XP+31)/32, (unsigned)((SL+63)/64)), dim3(16,16), 0, st>>>(xc, W_xp, xdbl, SL);
        else       k_x_proj<<<dim3((unsigned)((SL+255)/256), XP), b1, 0, st>>>(xc, W_xp, xdbl, SL);
    }
    kp_mark("x_proj", &klast);
    GDBG("gxdbl", xdbl, (size_t)SL*XP);
    {
        static int t = -1;
        if (t < 0) { const char *e = getenv("REUSE_DTPROJ"); t = e ? atoi(e) : (getenv("REUSE_NAIVE_DTPROJ") ? 0 : 2); }
        if (getenv("REUSE_SKIP_DTPROJ")) { } else
        if (t == 2) k_dt_proj_r<<<(unsigned)((SL+7)/8), 256, 0, st>>>(xdbl, W_dt, B_dt, dtt, SL);
        else if (t) k_dt_proj_t<<<(unsigned)((SL+7)/8), 256, 0, st>>>(xdbl, W_dt, B_dt, dtt, SL);
        else   k_dt_proj<<<dim3((unsigned)((SL+255)/256), DINNER), b1, 0, st>>>(xdbl, W_dt, B_dt, dtt, SL);
    }
    kp_mark("dt_proj", &klast);
    GDBG("gdt", dtt, (size_t)SL*DINNER);
    {
        static int t = -1; if (t < 0) t = getenv("REUSE_NAIVE_SCAN") ? 0 : 1;
        if (getenv("REUSE_SKIP_SCAN")) { } else
        if (t) k_scan_b<<<dim3((unsigned)((S+7)/8), DINNER/32), 256, 0, st>>>(xc, dtt, A_log, Dv, xdbl, y, S, L, rev);
        else   k_scan<<<dim3((unsigned)S, DINNER/32), 32, 0, st>>>(xc, dtt, A_log, Dv, xdbl, y, S, L, rev);
    }
    kp_mark("scan", &klast);
    GDBG("gscan", y, (size_t)SL*DINNER);
    {
        static int tiled = -1;
        if (tiled < 0) tiled = getenv("REUSE_NAIVE_OP") ? 0 : 1;
        if (tiled) {
            dim3 g((HID+GT-1)/GT, (unsigned)((SL+GT-1)/GT));
            k_out_proj_tiled<<<g, dim3(GT/2, GT/2), 0, st>>>(y, z, W_op, out, (int)SL);
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
        if (mamba_dir_gpu(g, p, x, fw, S, L, 0, 0)) return 1;
        snprintf(p, sizeof(p), "%s.backward_blocks", pfx);
        if (mamba_dir_gpu(g, p, x, bw, S, L, 1, st_side)) return 1;
        if (cudaEventRecord(ev2, st_side) || cudaStreamWaitEvent(0, ev2, 0)) { g->bad = 1; return 1; }
    } else {
        if (mamba_dir_gpu(g, p, x, fw, S, L, 0, 0)) return 1;
        snprintf(p, sizeof(p), "%s.backward_blocks", pfx);
        if (mamba_dir_gpu(g, p, x, bw, S, L, 1, 0)) return 1;
    }
    if (getenv("REUSE_SKIP_CONCAT")) {
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
    k_gather_time<<<(unsigned)(((long)F*T*HID+255)/256), 256>>>(x, xt, T, F);
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
    if (dense_block_gpu(g, dbp, x, T, F, d)) return 1;
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
    k_transpose<<<(unsigned)(((long)HID*T*F*2+255)/256), 256>>>(u1n, sw, HID, T, F*2);
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
    k_transpose<<<(unsigned)(((long)HID*Tt*Ft*4+255)/256), 256>>>(u2n, y, HID, Tt, Ft*4);
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
    for (int i = 0; i < 30; i++) {
        if (reuse_gpu_tfmamba(g, i, enc, blk, T2, F2)) return 1;
        cudaMemcpyAsync(enc, blk, (size_t)HID*T2*F2*sizeof(float), cudaMemcpyDeviceToDevice);
    }
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
    if (reuse_gpu_decoder_pre(g, "mask_decoder", enc, dm, T2, F2)) return 1;
    if (reuse_gpu_decoder_pre(g, "phase_decoder", enc, dp, T2, F2)) return 1;
    cudaDeviceSynchronize(); double t3 = gt_now(); g_t_dec = t3-t2;
    if (reuse_gpu_conv1x1_to1(g, dm, "mask_decoder.final_conv", fa, HID, Tx, Fx)) return 1;
    if (reuse_gpu_conv1x1_to1(g, dp, "phase_decoder.phase_conv_r", fr, HID, Tx, Fx)) return 1;
    if (reuse_gpu_conv1x1_to1(g, dp, "phase_decoder.phase_conv_i", fi, HID, Tx, Fx)) return 1;
    k_atan2_head<<<(unsigned)(((long)T*F+255)/256), 256>>>(fr, fi, pha_out, Tx, Fx, T, F);
    // crop [Tx,Fx] -> [T,F] (amp)
    k_crop_ft<<<(unsigned)(((long)T*F+255)/256), 256>>>(fa, amp_out, T, F, Fx);
    k_sweep_mask<<<(unsigned)((T+255)/256), 256>>>(amp_out, T, F);
    cudaDeviceSynchronize(); g_t_head = gt_now()-t3;
    cudaFree(dm); cudaFree(dp); cudaFree(fa); cudaFree(fr); cudaFree(fi);
    cudaFree(xin); cudaFree(enc); cudaFree(blk);
    return reuse_gpu_ok(g, "forward");
}
