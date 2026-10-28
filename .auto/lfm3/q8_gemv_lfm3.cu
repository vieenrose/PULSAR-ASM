// q8_gemv_lfm3.cu -- Q8_0 GEMV at LFM3-Audio-2B's real shapes, sm_53.
//
// Why: decode is bandwidth-bound on the GEMVs. LFM3's text backbone streams
// 1.466 G weights per token, so Q8_0 is ~1.47 GB + ~92 MB of block scales =
// 1.56 GB per token. At the repo's measured Nano numbers that is the whole
// decode budget: 1.56/13.0 = 120 ms (8.3 tok/s), /17.1 = 91 ms (11 tok/s).
// Everything else (conv, rmsnorm, attention) is noise next to that, so the
// GEMV rate IS the tok/s target and it is worth measuring per shape.
//
// Two kernels, same arithmetic contract:
//
//  A) staged: x in shared, padded stride 33 (kills a 32-way bank conflict),
//     R rows per block amortizes the staging. Tuned for K=2048 (64 Q8_0
//     blocks, 256 threads = 4 rows x 64 blocks). At K=7680 this needs 30 KB
//     shared per block; sm_53 has 64 KB per SM and the Nano has ONE SM, so
//     the shared pool, not the registers, caps residency. That is the point of
//     kernel B.
//
//  B) kg ("K-generic"): no shared at all. x is read through __ldg so it lives
//     in L1 -- a GEMV reads the same x for every row, so it is the perfectly
//     cacheable side, and 30 KB fits L1. One warp per row, thread t accumulates
//     Q8_0 blocks t, t+32, t+64, ... ascending, then a 32-lane butterfly.
//     No shared -> occupancy is set by threads/registers, not by x's size.
//
// BIT-EXACTNESS: q8_ref_tree_kernel mirrors kernel B's reduction exactly
// (strided ascending per lane, then the butterfly), using __fmul_rn/__fadd_rn
// with no contraction -- the repo's own technique (pq1_swar_sm53.cu:19-22).
// int8 -> float is exact, so both sides see identical products; only the
// addition ORDER can differ, and the reference pins it. Built -fmad=false so
// nothing contracts anywhere.
//
// Build on the Nano (CUDA 10.2):
//   nvcc -gencode arch=compute_53,code=sm_53 -O3 -fmad=false \
//        -Xptxas -O3,-dlcm=ca -o q8_gemv_lfm3 q8_gemv_lfm3.cu
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cuda_fp16.h>

#define Q8_BLK 32
#define NT 256

static double ms_since(const struct timespec* a) {
  struct timespec b; clock_gettime(CLOCK_MONOTONIC, &b);
  return (b.tv_sec - a->tv_sec) * 1e3 + (b.tv_nsec - a->tv_nsec) * 1e-6;
}

// ---------------------------------------------------------------- kernel A
// K = 2048 exactly. x staged in shared, stride padded 32 -> 33.
__global__ void q8_gemv_2048(const int8_t* __restrict__ q, const __half* __restrict__ s,
                             const float* __restrict__ x, float* __restrict__ out,
                             int N, int R) {
  __shared__ float xs[64 * 33];
  const int tid = threadIdx.x;
  for (int i = tid; i < 2048; i += NT) {
    int b = i >> 5, j = i & 31;
    xs[b * 33 + j] = x[i];
  }
  __syncthreads();

  const int nb = 64;
  const int row_local = tid / nb;
  const int b = tid % nb;
  const int base = blockIdx.x * R;
  const int GR = R / (NT / nb);
  float acc[8];
  #pragma unroll
  for (int g = 0; g < GR; g++) {
    const int r = row_local + g * (NT / nb);
    const int row = base + r;
    acc[r] = 0.f;
    if (row >= N) continue;
    const int8_t* qb = q + (size_t)row * 2048 + (size_t)b * Q8_BLK;
    const float sc = __half2float(s[(size_t)row * nb + b]);
    const int4 a0 = *reinterpret_cast<const int4*>(qb);
    const int4 a1 = *reinterpret_cast<const int4*>(qb + 16);
    const int8_t* c0 = reinterpret_cast<const int8_t*>(&a0);
    const int8_t* c1 = reinterpret_cast<const int8_t*>(&a1);
    const float* xb = xs + (size_t)b * 33;
    float sum = 0.f;
    #pragma unroll
    for (int j = 0; j < 16; j++) sum += (float)c0[j] * xb[j];
    #pragma unroll
    for (int j = 0; j < 16; j++) sum += (float)c1[j] * xb[16 + j];
    acc[r] = sc * sum;
  }
  const int lane = tid & 31, wid = tid >> 5;
  __shared__ float part[8][2];
  #pragma unroll
  for (int g = 0; g < GR; g++) {
    const int r = row_local + g * (NT / nb);
    float v = acc[r];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(~0u, v, o);
    if (lane == 0) part[r][wid % 2] = v;
  }
  __syncthreads();
  if (tid < R && base + tid < N) out[base + tid] = part[tid][0] + part[tid][1];
}

// ---------------------------------------------------------------- kernel B
// Any K that is a multiple of 32. No shared: x through __ldg (L1).
// One warp per row; lane l sums Q8_0 blocks l, l+32, l+64, ... ascending,
// then a 32-lane butterfly.
//
// The x side is loaded as 8 float4, NOT as 32 scalar __ldg. The first cut did
// 32 scalar x-loads per Q8_0 block and measured 1.88 GB/s at every shape --
// 4.8x slower than the staged kernel, and flat across shapes, which is the
// signature of instruction-issue-bound, not bandwidth-bound. 2 int4 (codes) +
// 8 float4 (x) + 32 FMAs per block is the balance this wants.
__global__ void q8_gemv_kg(const int8_t* __restrict__ q, const __half* __restrict__ s,
                           const float* __restrict__ x, float* __restrict__ out,
                           int N, int K) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= N) return;
  const int nb = K >> 5;                          // Q8_0 blocks per row
  const int8_t* qr = q + (size_t)warp * K;
  const __half* sr = s + (size_t)warp * nb;
  float sum = 0.f;
  for (int b = lane; b < nb; b += 32) {
    const int8_t* qb = qr + (size_t)b * Q8_BLK;
    const int4 a0 = __ldg(reinterpret_cast<const int4*>(qb));
    const int4 a1 = __ldg(reinterpret_cast<const int4*>(qb + 16));
    const int8_t* c0 = reinterpret_cast<const int8_t*>(&a0);
    const int8_t* c1 = reinterpret_cast<const int8_t*>(&a1);
    const float4* xb = reinterpret_cast<const float4*>(x + (size_t)b * Q8_BLK);
    float4 v0 = __ldg(xb + 0), v1 = __ldg(xb + 1);
    float4 v2 = __ldg(xb + 2), v3 = __ldg(xb + 3);
    float4 v4 = __ldg(xb + 4), v5 = __ldg(xb + 5);
    float4 v6 = __ldg(xb + 6), v7 = __ldg(xb + 7);
    float p = 0.f;
    // 32 SEPARATE p += statements. p += A+B+C+D parses as
    // p + ((A+B)+C)+D, which is not the reference's ((p+A)+B)+C+D; with
    // -fmad=false the grouping is fully visible, so writing it as one
    // expression silently breaks bit-exactness (observed: kg went YES->NO
    // the moment the x loads were vectorized into grouped adds).
    p += (float)c0[0] * v0.x;
    p += (float)c0[1] * v0.y;
    p += (float)c0[2] * v0.z;
    p += (float)c0[3] * v0.w;
    p += (float)c0[4] * v1.x;
    p += (float)c0[5] * v1.y;
    p += (float)c0[6] * v1.z;
    p += (float)c0[7] * v1.w;
    p += (float)c0[8] * v2.x;
    p += (float)c0[9] * v2.y;
    p += (float)c0[10] * v2.z;
    p += (float)c0[11] * v2.w;
    p += (float)c0[12] * v3.x;
    p += (float)c0[13] * v3.y;
    p += (float)c0[14] * v3.z;
    p += (float)c0[15] * v3.w;
    p += (float)c1[0] * v4.x;
    p += (float)c1[1] * v4.y;
    p += (float)c1[2] * v4.z;
    p += (float)c1[3] * v4.w;
    p += (float)c1[4] * v5.x;
    p += (float)c1[5] * v5.y;
    p += (float)c1[6] * v5.z;
    p += (float)c1[7] * v5.w;
    p += (float)c1[8] * v6.x;
    p += (float)c1[9] * v6.y;
    p += (float)c1[10] * v6.z;
    p += (float)c1[11] * v6.w;
    p += (float)c1[12] * v7.x;
    p += (float)c1[13] * v7.y;
    p += (float)c1[14] * v7.z;
    p += (float)c1[15] * v7.w;
    sum += __half2float(__ldg(sr + b)) * p;
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(~0u, sum, o);
  if (lane == 0) out[warp] = sum;
}

// ------------------------------------------- bit-exact reference for kernel B
__global__ void q8_ref_tree_kernel(const int8_t* q, const __half* s, const float* x,
                                   float* y, int N, int K) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= N) return;
  const int nb = K >> 5;
  // lane l's strided ascending sum, then the butterfly -- operation for
  // operation what kernel B does.
  float lane_part[32];
  for (int l = 0; l < 32; l++) {
    float sum = 0.f;
    for (int b = l; b < nb; b += 32) {
      float p = 0.f;
      for (int j = 0; j < Q8_BLK; j++)
        p = __fadd_rn(p, __fmul_rn((float)q[(size_t)r * K + (size_t)b * Q8_BLK + j],
                                   x[(size_t)b * Q8_BLK + j]));
      sum = __fadd_rn(sum, __fmul_rn(__half2float(s[(size_t)r * nb + b]), p));
    }
    lane_part[l] = sum;
  }
  for (int o = 16; o > 0; o >>= 1)
    for (int i = 0; i < 32; i++)
      lane_part[i] = __fadd_rn(lane_part[i], lane_part[i + o < 32 ? i + o : i]);
  y[r] = lane_part[0];
}

// quantize a row to Q8_0 (absmax/127 scale, f16 stored)
static void quant_row(const float* w, int K, int8_t* q, __half* s) {
  const int nb = K / Q8_BLK;
  for (int b = 0; b < nb; b++) {
    float mx = 0.f;
    for (int j = 0; j < Q8_BLK; j++) mx = fmaxf(mx, fabsf(w[b * Q8_BLK + j]));
    float sc = mx / 127.f;
    if (sc == 0.f) { memset(q + (size_t)b * Q8_BLK, 0, Q8_BLK); s[b] = __float2half(0.f); continue; }
    for (int j = 0; j < Q8_BLK; j++) {
      float v = w[b * Q8_BLK + j] / sc;
      int iv = (int)lrintf(v);
      if (iv > 127) iv = 127; if (iv < -127) iv = -127;
      q[(size_t)b * Q8_BLK + j] = (int8_t)iv;
    }
    s[b] = __float2half(sc);
  }
}

struct Shape { int N, K; const char* name; };

int main(int argc, char** argv) {
  // LFM3-Audio-2B text backbone: H=2048, INTER=7680, 16 q heads x 128,
  // 4 kv heads x 128. 13 conv layers, 11 attention layers, 24 total.
  Shape shapes[] = {
    {6144, 2048, "conv_in_proj  (13x)"},
    {2048, 2048, "conv_out/o_proj (13x+11x)"},
    { 512, 2048, "w_k / w_v     (11x2)"},
    {2048, 2048, "w_q           (11x)"},
    {7680, 2048, "gate+up w1,w3 (24x2)"},
    {2048, 7680, "down  w2      (24x)"},
  };
  const int NS = (int)(sizeof shapes / sizeof *shapes);

  int maxN = 0, maxK = 0;
  for (int i = 0; i < NS; i++) { if (shapes[i].N > maxN) maxN = shapes[i].N;
                                 if (shapes[i].K > maxK) maxK = shapes[i].K; }
  size_t wmax = (size_t)maxN * maxK;

  float* hw = (float*)malloc(wmax * 4);
  int8_t* hq = (int8_t*)malloc(wmax);
  // hs is the HOST scale staging array; ds is the DEVICE copy. They were the
  // same device pointer here, so quant_row wrote host-visible memory through a
  // device pointer -- a wild write that segfaulted before any kernel ran.
  __half* hs = (__half*)malloc((size_t)maxN * (maxK / Q8_BLK) * 2);
  int8_t *dq; __half *ds; float *dx, *dy, *dr;
  for (size_t i = 0; i < wmax; i++) hw[i] = (float)(((int)(i * 2654435761u) >> 8) % 4001 - 2000) / 500.f;

  cudaMalloc(&dq, wmax);
  cudaMalloc(&ds, (size_t)maxN * (maxK / Q8_BLK) * 2);
  cudaMalloc(&dx, (size_t)maxK * 4);
  cudaMalloc(&dy, (size_t)maxN * 4);
  cudaMalloc(&dr, (size_t)maxN * 4);
  cudaMemset(dx, 0, (size_t)maxK * 4);

  printf("%-28s %6s %6s %8s %9s %9s %8s  %s\n",
         "shape", "N", "K", "MB", "ms", "GB/s", "kernel", "bit-exact");
  double total_ms = 0.0;
  for (int si = 0; si < NS; si++) {
    int N = shapes[si].N, K = shapes[si].K;
    const int nb = K / Q8_BLK;
    for (int r = 0; r < N; r++) quant_row(hw + (size_t)r * K, K, hq + (size_t)r * K, hs + (size_t)r * nb);
    cudaMemcpy(dq, hq, (size_t)N * K, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs, (size_t)N * nb * 2, cudaMemcpyHostToDevice);
    float hx[8192];
    for (int i = 0; i < K; i++) hx[i] = (float)(((int)(i * 40503u) >> 4) % 2001 - 1000) / 700.f;
    cudaMemcpy(dx, hx, (size_t)K * 4, cudaMemcpyHostToDevice);

    double mb = (N * K + (size_t)N * nb * 2 + (size_t)K * 4 + (size_t)N * 4) / 1e6;
    for (int kg = 0; kg < 2; kg++) {
      const char* kname = kg ? "kg" : "staged";
      int reps = 200;
      for (int w = 0; w < 20; w++) {
        if (kg) q8_gemv_kg<<<(N * 32 + NT - 1) / NT, NT>>>(dq, ds, dx, dy, N, K);
        else    q8_gemv_2048<<<(N + 3) / 4, NT>>>(dq, ds, dx, dy, N, 4);
      }
      cudaDeviceSynchronize();
      struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
      for (int r = 0; r < reps; r++) {
        if (kg) q8_gemv_kg<<<(N * 32 + NT - 1) / NT, NT>>>(dq, ds, dx, dy, N, K);
        else    q8_gemv_2048<<<(N + 3) / 4, NT>>>(dq, ds, dx, dy, N, 4);
      }
      cudaDeviceSynchronize();
      double ms = ms_since(&t0) / reps;

      // bit-exactness against the kernel-B tree reference (K=2048 only for A,
      // so A is checked the same way -- same contract, same tree)
      q8_ref_tree_kernel<<<(N + 255) / 256, 256>>>(dq, ds, dx, dr, N, K);
      float *ay = (float*)malloc((size_t)N * 4), *ar = (float*)malloc((size_t)N * 4);
      cudaMemcpy(ay, dy, (size_t)N * 4, cudaMemcpyDeviceToHost);
      cudaMemcpy(ar, dr, (size_t)N * 4, cudaMemcpyDeviceToHost);
      long bad = 0; double mx = 0;
      for (int i = 0; i < N; i++) { if (ay[i] != ar[i]) bad++;
        double d = fabs((double)ay[i] - ar[i]); if (d > mx) mx = d; }
      free(ay); free(ar);
      printf("%-28s %6d %6d %8.1f %9.3f %9.2f %8s  %s\n",
             shapes[si].name, N, K, mb, ms, mb * 1e-3 / (ms * 1e-3), kname,
             bad ? "NO" : "YES");
      if (!kg) total_ms += ms;
      cudaError_t e = cudaGetLastError();
      if (e) { printf("  cuda: %s\n", cudaGetErrorString(e)); }
    }
  }
  printf("\nsum over the 6 distinct shapes (once each): %.3f ms\n", total_ms);
  return 0;
}
