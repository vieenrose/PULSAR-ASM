// q8_gemv_sm53 — a real Q8_0 GEMV for Jetson Nano (Tegra X1, 1 SM, sm_53).
//
// Why this exists: the pq1_* kernel family on this branch streams 2-bit PQ1_0
// bit planes (pq1_swar_sm53.cu:1-30) and is hard-gated to K in {4096, 12288}
// (:1340). There is NO 8-bit GEMV in the Nano tree, so any 8-bit model -- and
// LFM3-Audio, whose accuracy floor is Q8_0 (doc/LFM3_AUDIO_PLAN.md §11) -- has
// no kernel at all. A throwaway probe reached 0.81 GB/s, ~20x below the 17.1
// GB/s this engine achieves, so the open question is whether Q8 at K=2048 can
// reach the bus ceiling at all. This file answers it with a serious kernel.
//
// Design (K = 2048; Q8_0 = 32 int8 codes + 1 f16 scale per block of 32):
//   * one thread owns exactly ONE Q8_0 block: 32 codes as two int4 loads
//     (32 B, coalesced across a warp -- 32 contiguous blocks = 1024 B) plus
//     one f16 scale; the 32 products are summed in INDEX ORDER by that thread;
//   * x lives in shared (K floats = 8 KB at K=2048), staged once per block;
//   * R rows per block. R=1 (the first cut) staged x for EVERY block, so half
//     of all memory traffic was x and not weights: N/4 blocks x 8 KB = 15.4 MB
//     of x on top of 15.7 MB of weights, which is why it read 3.1 GB/s.
//     Looping R rows amortizes the staging by R;
//   * no atomics -- the naive version's per-row atomicAdd was a real cost.
//
// BIT-EXACTNESS, stated honestly: the 64 block-partials of a row are combined
// by a warp shuffle, which is a TREE, not the sequential `acc += p[b]` of the
// host reference; fp add is not associative, so this agrees to a few ULP and
// NOT bit-for-bit (measured maxrel ~3e-4, inflated further by --use_fast_math
// contracting the 32-term sums into FMAs). The repo's own fix, from
// pq1_swar_sm53.cu:19-22 -- "the fp accumulation is made bit-exact too by
// reducing per-group products in GROUP ORDER in fp32 (__fmul_rn/__fadd_rn, no
// contraction)" -- applies here: make the reference mirror the kernel's tree
// and compile without contraction. Reported as maxabs/maxrel, not asserted.
//
// Build on the Nano:
//   /usr/local/cuda-10.2/bin/nvcc -gencode arch=compute_53,code=sm_53 -O3 \
//     --use_fast_math -Xptxas -O3,-dlcm=ca -o q8_gemv q8_gemv_sm53.cu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define Q8_BLK 32
#define NTHREADS 256

template <int R>
__global__ void q8_gemv_k2048(const int8_t* __restrict__ q, const __half* __restrict__ s,
                              const float* __restrict__ x, float* __restrict__ out,
                              int N, int nb) {
  // Padded row stride. Unpadded (xs[b*32 + j]) makes lanes 0..31 of a warp
  // read xs[0*32+j], xs[1*32+j], ... xs[31*32+j] -- stride 32 floats = the SAME
  // bank, a 32-way conflict on every one of the 32 shared loads. Stride 33
  // spreads them across banks; it is the whole reason for the 3x speedup here.
  __shared__ float xs[2048 + 64];
  const int tid = threadIdx.x;
  #pragma unroll
  for (int i = tid; i < 2048; i += NTHREADS) {
    const int b = i >> 5, j = i & 31;
    xs[b * 33 + j] = x[i];
  }
  __syncthreads();

  const int row_local = tid / nb;                // 0..3 (256 threads / 64 blocks)
  const int b         = tid % nb;                // Q8_0 block index within a row
  const int base = blockIdx.x * R;               // first row handled here
  const int GR = R / (NTHREADS / nb);             // row-groups each thread walks
  float acc[R];
  #pragma unroll
  for (int g = 0; g < GR; g++) {
    const int r = row_local + g * (NTHREADS / nb);
    const int row = base + r;
    acc[r] = 0.f;
    if (row >= N) continue;
    const int8_t* qb = q + (size_t)row * 2048 + (size_t)b * Q8_BLK;
    const float sc   = __half2float(s[(size_t)row * nb + b]);
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

  // each row's nb (64) partials live in the 2 warps that computed it: threads
  // with row_local == r % (NTHREADS/nb). Letting every thread reduce every row
  // would read acc[r] slots it never wrote.
  const int lane = tid & 31, wid = tid >> 5;
  const int WPB = nb / 32;                       // warps covering one row (2)
  const int RPB = NTHREADS / nb;                 // rows per block (4)
  __shared__ float part[R][2];
  #pragma unroll
  for (int g = 0; g < GR; g++) {
    const int r = row_local + g * RPB;
    float v = acc[r];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(~0u, v, o);
    if (lane == 0) part[r][wid % WPB] = v;
  }
  __syncthreads();
  if (tid < R && base + tid < N) out[base + tid] = part[tid][0] + part[tid][1];
}

// Sequential host reference: per block, index-order sum of 32 products, times
// the block scale, blocks accumulated in order.
static void q8_ref(const int8_t* q, const __half* s, const float* x, float* y,
                   int N, int K) {
  const int nb = K / Q8_BLK;
  for (int r = 0; r < N; r++) {
    float acc = 0.f;
    for (int b = 0; b < nb; b++) {
      float sum = 0.f;
      for (int j = 0; j < Q8_BLK; j++)
        sum += (float)q[(size_t)r * K + b * Q8_BLK + j] * x[b * Q8_BLK + j];
      acc += __half2float(s[(size_t)r * nb + b]) * sum;
    }
    y[r] = acc;
  }
}

// The bit-exactness reference: mirrors the KERNEL's arithmetic operation for
// operation, which is the repo's own technique (pq1_swar_sm53.cu:19-22, "the
// fp accumulation is made bit-exact too by reducing per-group products in
// GROUP ORDER in fp32 (__fmul_rn/__fadd_rn, no contraction)").
//
// Two things must match, and both were visible as maxrel ~3e-4 against the
// sequential reference above:
//   1. no FMA contraction of the 32-term sums -- compile with -fmad=false, and
//      use __fmul_rn/__fadd_rn here so the host side cannot contract either;
//   2. the inter-block reduction, which in the kernel is a warp-shuffle
//      butterfly (o = 16,8,4,2,1) over the 32 partials of each half-row, then
//      part[0] + part[1]. fp addition is not associative, so a sequential
//      acc += p[b] differs in the last bits.
// int8 -> float is exact, so the products are the same values on both sides.
static __device__ float butterfly32(const float* p) {
  float v[32];
  for (int i = 0; i < 32; i++) v[i] = p[i];
  for (int o = 16; o > 0; o >>= 1) {
    float t[32];
    for (int i = 0; i < 32; i++) t[i] = __fadd_rn(v[i], v[i + o < 32 ? i + o : i]);
    for (int i = 0; i < 32; i++) v[i] = t[i];
  }
  return v[0];
}

static __device__ void q8_ref_tree_one(const int8_t* q, const __half* s, const float* x,
                                       float* y, int r, int K) {
  const int nb = K / Q8_BLK;                     // 64 at K=2048
  float p[64];
  for (int b = 0; b < nb; b++) {
    float sum = 0.f;
    for (int j = 0; j < Q8_BLK; j++)
      sum = __fadd_rn(sum, __fmul_rn((float)q[(size_t)r * K + b * Q8_BLK + j],
                                     x[b * Q8_BLK + j]));
    p[b] = __fmul_rn(__half2float(s[(size_t)r * nb + b]), sum);
  }
  y[r] = __fadd_rn(butterfly32(p), butterfly32(p + 32));
}

// Runs the reference on the device: __fadd_rn/__fmul_rn are __device__
// intrinsics, so a host-side reference cannot use them (nvcc rejects the call
// outright). One thread per row; only used by the correctness check.
__global__ void q8_ref_tree_kernel(const int8_t* q, const __half* s, const float* x,
                                   float* y, int N, int K) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < N) q8_ref_tree_one(q, s, x, y, r, K);
}

static double ms_since(const struct timespec* a) {
  struct timespec b; clock_gettime(CLOCK_MONOTONIC, &b);
  return (b.tv_sec - a->tv_sec) * 1e3 + (b.tv_nsec - a->tv_nsec) * 1e-6;
}

int main() {
  const int K = 2048, nb = K / Q8_BLK;
  const int MAXN = 8192, MAXK = 4096;
  int shapes[3][2] = {{7680, 2048}, {6144, 2048}, {2048, 2048}};

  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  printf("%s cc%d.%d SMs=%d\n", p.name, p.major, p.minor, p.multiProcessorCount);

  int8_t* qh = (int8_t*)malloc((size_t)MAXN * MAXK);
  __half* sh = (__half*)malloc((size_t)MAXN * (MAXK / Q8_BLK) * 2);
  float*  xh = (float*)malloc((size_t)MAXK * 4);
  float*  yh = (float*)malloc((size_t)MAXN * 4);
  // deterministic pseudo-data, deliberately non-zero: zeros hide a kernel that
  // reads nothing and reports infinite bandwidth.
  unsigned st = 12345;
  auto rnd = [&]() { st = st * 1103515245u + 12345u; return (int)((st >> 16) & 0xff) - 128; };
  for (size_t i = 0; i < (size_t)MAXN * MAXK; i++) qh[i] = (int8_t)rnd();
  for (size_t i = 0; i < (size_t)MAXN * (MAXK / Q8_BLK); i++)
    sh[i] = __float2half((float)(rnd() % 64) / 64.f + 0.01f);
  for (int i = 0; i < MAXK; i++) xh[i] = (float)(rnd() % 255) / 128.f;

  int8_t *dq; __half *ds; float *dx, *dy, *dref;
  cudaMalloc(&dq, (size_t)MAXN * MAXK);
  cudaMalloc(&ds, (size_t)MAXN * (MAXK / Q8_BLK) * 2);
  cudaMalloc(&dx, (size_t)MAXK * 4);
  cudaMalloc(&dy, (size_t)MAXN * 4);
  cudaMalloc(&dref, (size_t)MAXN * 4);
  cudaMemcpy(dq, qh, (size_t)MAXN * MAXK, cudaMemcpyHostToDevice);
  cudaMemcpy(ds, sh, (size_t)MAXN * (MAXK / Q8_BLK) * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dx, xh, (size_t)MAXK * 4, cudaMemcpyHostToDevice);
  cudaError_t e = cudaGetLastError();
  if (e) { printf("setup: %s\n", cudaGetErrorString(e)); return 1; }

  printf("\n%-3s %-8s %6s %8s %9s %9s  %s\n",
         "R", "rows", "N", "MB", "ms", "GB/s", "vs sequential host ref");
  int RS[6] = {4, 8, 16, 32, 64, 128};
  for (int ri = 0; ri < 6; ri++) {
    const int R = RS[ri];
    for (auto& S : shapes) {
      const int N = S[0], k = S[1];
      if (k != K) continue;                        // K=2048 kernel only
      const int nblk = (N + R - 1) / R;
      auto launch = [&]() {
        switch (R) {
          case 4:   q8_gemv_k2048<4>  <<<nblk, NTHREADS>>>(dq, ds, dx, dy, N, nb); break;
          case 8:   q8_gemv_k2048<8>  <<<nblk, NTHREADS>>>(dq, ds, dx, dy, N, nb); break;
          case 16:  q8_gemv_k2048<16> <<<nblk, NTHREADS>>>(dq, ds, dx, dy, N, nb); break;
          case 32:  q8_gemv_k2048<32> <<<nblk, NTHREADS>>>(dq, ds, dx, dy, N, nb); break;
          case 64:  q8_gemv_k2048<64> <<<nblk, NTHREADS>>>(dq, ds, dx, dy, N, nb); break;
          default:  q8_gemv_k2048<128><<<nblk, NTHREADS>>>(dq, ds, dx, dy, N, nb); break;
        }
      };
      launch();                                    // correctness BEFORE timing
      cudaError_t e2 = cudaDeviceSynchronize();
      if (e2) { printf("  LAUNCH(R=%d): %s\n", R, cudaGetErrorString(e2)); return 1; }
      cudaMemcpy(yh, dy, (size_t)N * 4, cudaMemcpyDeviceToHost);
      q8_ref_tree_kernel<<<(N + 255) / 256, 256>>>(dq, ds, dx, dref, N, k);
      float* yt = (float*)malloc((size_t)N * 4);
      cudaMemcpy(yt, dref, (size_t)N * 4, cudaMemcpyDeviceToHost);
      float* yr = (float*)malloc((size_t)N * 4);
      q8_ref(qh, sh, xh, yr, N, k);
      int bad = 0, badt = 0; double maxabs = 0, maxrel = 0, maxrelt = 0;
      for (int i = 0; i < N; i++) {
        double a = fabs((double)yh[i] - yr[i]), r = a / (fabs((double)yr[i]) + 1e-9);
        double at = fabs((double)yh[i] - yt[i]);
        double rt = at / (fabs((double)yt[i]) + 1e-9);
        if (a > maxabs) maxabs = a;
        if (r > maxrel) maxrel = r;
        if (rt > maxrelt) maxrelt = rt;
        if (yh[i] != yr[i]) bad++;
        if (yh[i] != yt[i]) badt++;
      }
      free(yr); free(yt);
      for (int i = 0; i < 20; i++) launch();
      cudaDeviceSynchronize();
      struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
      for (int i = 0; i < 50; i++) launch();
      cudaDeviceSynchronize();
      double ms = ms_since(&t0) / 50;
      // Count what the kernel actually streams: 1 B/weight of codes PLUS the
      // f16 scale per 32-code block, so x1.0625. Codes-only flatters the
      // number by 6%.
      double bytes = (double)N * k * (1.0 + 2.0 / Q8_BLK);
      printf("%-4d %-8d %6d %8.1f %9.3f %9.2f  tree %-9s seq maxrel %.2g  tree maxrel %.2g\n",
             R, N, N, bytes / 1e6, ms, bytes / (ms * 1e6),
             badt ? "MISMATCH" : "BIT-EXACT", maxrel, maxrelt);
    }
  }
  printf("\nreference points: engine achieves 17.1 GB/s, measured ceiling 21.9 GB/s\n");
  printf("(2-bit PQ1_0 hs kernel at the same 7680x2048 shape: 0.398 ms)\n");
  printf("build for bit-exactness: -O3 -fmad=false (NO --use_fast_math)\n");
  cudaFree(dq); cudaFree(ds); cudaFree(dx); cudaFree(dy); cudaFree(dref);
  free(qh); free(sh); free(xh); free(yh);
  return 0;
}
