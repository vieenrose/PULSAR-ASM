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
    int ho = blockIdx.x * blockDim.x + threadIdx.x;
    int wo = blockIdx.y * blockDim.y + threadIdx.y;
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

// [HID,T,F] -> [F,T,HID]
__global__ void k_gather_time(const float *x, float *xt, int T, int F)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)F*T*HID) return;
    int c = i % HID, t = (int)((i / HID) % T), f = (int)(i / ((long)HID*T));
    xt[i] = x[(long)c*T*F + (long)t*F + f];
}
// residual on the time block + resample into [T,F,HID] for the freq block
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
__global__ void k_sp_up(const float *x, const float *Ww, const float *b, float *y,
                        int H, int W, int r)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int w = blockIdx.y * blockDim.y + threadIdx.y;
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

static int conv_chunk(void)
{
    static int c = -1;
    if (c < 0) { const char *e = getenv("REUSE_CONV_CHUNK"); c = e ? atoi(e) : 8; if (c <= 0) c = 1 << 30; }
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
        dim3 grd((unsigned)((H_out + 15) / 16), (unsigned)((W_out + 15) / 16), (unsigned)n);
        // the kernel adds co0 itself, so pass the un-offset pointers
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
    dim3 grd((unsigned)((H + 15)/16), (unsigned)((W + 15)/16), (unsigned)(HID*r));
    k_sp_up<<<grd, blk>>>(x, w, b, y, H, W, r);
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
    launch_conv1x1(g, xin, w1, b1, x1, 2, HID, (long)Tp*Fp);
    gdump("c1", x1, (size_t)HID*Tp*Fp);
    inorm(g, x1, x1n, 1, HID, Tp, Fp, "dense_encoder.dense_conv_1.1.weight", "dense_encoder.dense_conv_1.2.weight");
    gdump("c1n", x1n, (size_t)HID*Tp*Fp);
    dense_block_gpu(g, "dense_encoder.dense_block.dense_block", x1n, Tp, Fp, xdb);
    gdump("db", xdb, (size_t)HID*Tp*Fp);
    launch_conv2d(g, xdb, w2, b2, HID, Tp, Fp, HID, 1, 3, 0, 0, 1, 1, 4, 2, x2);
    gdump("c2", x2, (size_t)HID*T2*F2);
    inorm(g, x2, enc, 1, HID, T2, F2, "dense_encoder.dense_conv_2.1.weight", "dense_encoder.dense_conv_2.2.weight");
    gdump("enc", enc, (size_t)HID*T2*F2);
    return g->bad;
}

// ================================================================= Mamba

static int mamba_dir_gpu(reuse_gpu_t *g, const char *pfx, const float *x, float *out,
                         int S, int L, int rev)
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
        static int tiled = -1;
        if (tiled < 0) tiled = getenv("REUSE_NAIVE_INPROJ") ? 0 : 1;
        if (tiled) k_in_proj_t<<<dim3((unsigned)((SL+MT32-1)/MT32), (2*DINNER)/NT64), dim3(16,16)>>>(x, W_in, x0, z, S, L, rev);
        else       k_in_proj<<<dim3((unsigned)((SL+255)/256), 2*DINNER), b1>>>(x, W_in, x0, z, S, L, rev);
    }
    kp_mark("in_proj", &klast);
    GDBG("x0", x0, (size_t)SL*DINNER); GDBG("z", z, (size_t)SL*DINNER);
    k_conv_causal<<<dim3((unsigned)((SL+255)/256), DINNER), b1>>>(x0, W_c, B_c, xc, S, L);
    kp_mark("conv", &klast);
    GDBG("gxact", xc, (size_t)SL*DINNER);
    k_x_proj<<<dim3((unsigned)((SL+255)/256), XP), b1>>>(xc, W_xp, xdbl, SL);
    kp_mark("x_proj", &klast);
    GDBG("gxdbl", xdbl, (size_t)SL*XP);
    k_dt_proj<<<dim3((unsigned)((SL+255)/256), DINNER), b1>>>(xdbl, W_dt, B_dt, dtt, SL);
    kp_mark("dt_proj", &klast);
    GDBG("gdt", dtt, (size_t)SL*DINNER);
    k_scan<<<dim3((unsigned)S, DINNER/32), 32>>>(xc, dtt, A_log, Dv, xdbl, y, S, L, rev);
    kp_mark("scan", &klast);
    GDBG("gscan", y, (size_t)SL*DINNER);
    {
        static int tiled = -1;
        if (tiled < 0) tiled = getenv("REUSE_NAIVE_OP") ? 0 : 1;
        if (tiled) {
            dim3 g((HID+GT-1)/GT, (unsigned)((SL+GT-1)/GT));
            k_out_proj_tiled<<<g, dim3(GT/2, GT/2)>>>(y, z, W_op, out, (int)SL);
        } else {
            k_gate_outproj<<<dim3((unsigned)((SL+255)/256), HID), b1>>>(y, z, W_op, out, SL);
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
    if (mamba_dir_gpu(g, p, x, fw, S, L, 0)) return 1;
    snprintf(p, sizeof(p), "%s.backward_blocks", pfx);
    if (mamba_dir_gpu(g, p, x, bw, S, L, 1)) return 1;
    k_concat<<<(unsigned)((SL+255)/256), 256>>>(fw, bw, x, cat, S, L);
        kp_mark("concat", &klast);
    char key[256];
    snprintf(key, sizeof(key), "%s.output_proj.weight", pfx); const float *W = reuse_gpu_w(g, key);
    snprintf(key, sizeof(key), "%s.output_proj.bias",   pfx); const float *B = reuse_gpu_w(g, key);
    snprintf(key, sizeof(key), "%s.norm.weight",        pfx); const float *LW = reuse_gpu_w(g, key);
    snprintf(key, sizeof(key), "%s.norm.bias",          pfx); const float *LB = reuse_gpu_w(g, key);
    if (g->bad) return 1;
    k_outproj_ln<<<(unsigned)SL, HID>>>(cat, W, B, LW, LB, out, SL);
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
    k_resid_scatter_freq<<<(unsigned)(((long)F*T*HID+255)/256), 256>>>(yt, x, y, xf, T, F);
        kp_mark("resid_freq", &klast);
    snprintf(p, sizeof(p), "TSMamba.%d.freq_mamba", idx);
    if (mamba_block_gpu(g, p, xf, yf, S_f, F)) return 1;
    k_resid_scatter_out<<<(unsigned)(((long)F*T*HID+255)/256), 256>>>(yf, xf, y, T, F);
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
