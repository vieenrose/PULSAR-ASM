// q8_sweep.cu -- rate sweep for the split-K Q8_0 GEMV on the two shapes that
// dominate LFM3-Audio decode: gate+up [7680,2048] (54% of decode GEMV) and
// down w2 [2048,7680] (22%).
//
// Where the headroom is. Best rate so far is 12.88 GB/s (5fd7e79) against the
// repo's measured 21.9 GB/s Nano ceiling, so ~1.7x is left, and 99e76b8 closed
// off the other route: quantization is Pareto-dominated, so the bytes are
// fixed and only kernel quality can move.
//
// Occupancy is NOT the open knob. Compute capability 5.3 caps 1536 threads/SM,
// so 6 blocks x 256 threads is the ceiling -- at KCHUNK=2048 shared is 8448 B
// -> 7 blocks, so the thread cap binds first. KCHUNK=1024 gives 4224 B -> 15
// blocks by shared, still 6 by threads. The knobs that do something are
// therefore R (rows amortized per x-stage -- x staging is the traffic kg
// could not amortize at all) and the TPB/nb relationship that sets rows per
// block, subject to: nb | NT and (NT/nb) | R.
//
// Bit-exactness is checked per config against ref_chunked, which mirrors the
// kernel's reduction tree for THAT nb:
//   per chunk: nb Q8_0 block partials, a 32-lane butterfly within each group
//   of 32, then the groups summed ascending; chunks then summed ascending.
// For nb=64 that is exactly the two-half tree of 5fd7e79, so the tuned
// configuration is verified by the same reference that certified it.
//
// Build on the Nano (CUDA 10.2):
//   nvcc -gencode arch=compute_53,code=sm_53 -O3 -fmad=false \
//        -Xptxas -O3,-dlcm=ca -o q8_sweep q8_sweep.cu
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cuda_fp16.h>

#define Q8_BLK 32
#define RMAX 32
#define WPMAX 8            // max warps covering one row (nb/32 <= 8 -> nb <= 256)

static double ms_since(const struct timespec* a) {
  struct timespec b; clock_gettime(CLOCK_MONOTONIC, &b);
  return (b.tv_sec - a->tv_sec) * 1e3 + (b.tv_nsec - a->tv_nsec) / 1e-6;
}

// pass 1: R rows x one KCHUNK-wide column chunk, x staged in shared with the
// 32->33 padded stride. nb = KCHUNK/32, rows per block = NT/nb, GR = R/(NT/nb).
__global__ void q8_partial(const int8_t* __restrict__ q, const __half* __restrict__ s,
                           const float* __restrict__ x, float* __restrict__ part,
                           int N, int K, int KCHUNK, int R, int nb) {
  extern __shared__ float xs[];                 // nb * 33 floats
  const int nchunk = K / KCHUNK;
  const int rowblk = blockIdx.x / nchunk;
  const int chunk  = blockIdx.x % nchunk;
  const int koff   = chunk * KCHUNK;
  const int NT     = blockDim.x;
  const int tid    = threadIdx.x;
  const float* xg = x + koff;
  for (int i = tid; i < KCHUNK; i += NT) {
    int b = i >> 5, j = i & 31;
    xs[b * 33 + j] = xg[i];
  }
  __syncthreads();

  __shared__ float sh[RMAX][WPMAX];
  const int base = rowblk * R;
  const int perrow = NT / nb;                   // rows handled per block pass
  const int row_local = tid / nb;
  const int b = tid % nb;
  const int GR = R / perrow;
  const int WPB = nb / 32;                      // warps covering one row
  float acc[RMAX];
  #pragma unroll 1
  for (int g = 0; g < GR; g++) {
    const int r = row_local + g * perrow;
    const int row = base + r;
    acc[r] = 0.f;
    if (row >= N) continue;
    const int8_t* qb = q + (size_t)row * K + koff + (size_t)b * Q8_BLK;
    const float sc = __half2float(s[(size_t)row * (K / Q8_BLK) + (size_t)chunk * nb + b]);
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
  #pragma unroll 1
  for (int g = 0; g < GR; g++) {
    const int r = row_local + g * perrow;
    float v = acc[r];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(~0u, v, o);
    if (lane == 0) sh[r][wid % WPB] = v;
  }
  __syncthreads();
  if (tid < R && base + tid < N) {
    float v = sh[tid][0];
    #pragma unroll 1
    for (int w = 1; w < WPB; w++) v += sh[tid][w];   // groups ascending
    part[(size_t)(base + tid) * nchunk + chunk] = v;
  }
}

__global__ void q8_combine(const float* __restrict__ part, float* __restrict__ out,
                           int N, int nchunk) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= N) return;
  float acc = 0.f;
  for (int c = 0; c < nchunk; c++) acc = __fadd_rn(acc, part[(size_t)r * nchunk + c]);
  out[r] = acc;
}

// bit-exact reference for a given KCHUNK (see file header for the tree)
__global__ void ref_chunked(const int8_t* q, const __half* s, const float* x,
                            float* y, int N, int K, int KCHUNK) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= N) return;
  const int nchunk = K / KCHUNK;
  const int nb = KCHUNK / Q8_BLK;
  float acc = 0.f;
  for (int c = 0; c < nchunk; c++) {
    const int koff = c * KCHUNK;
    for (int grp = 0; grp < nb / 32; grp++) {
      float v[32];
      for (int i = 0; i < 32; i++) {
        const int b = grp * 32 + i;
        float sum = 0.f;
        for (int j = 0; j < Q8_BLK; j++)
          sum = __fadd_rn(sum, __fmul_rn((float)q[(size_t)r * K + koff + (size_t)b * Q8_BLK + j],
                                         x[koff + (size_t)b * Q8_BLK + j]));
        v[i] = __fmul_rn(__half2float(s[(size_t)r * (K / Q8_BLK) + (size_t)c * nb + b]), sum);
      }
      for (int o = 16; o > 0; o >>= 1) {
        float t[32];
        for (int i = 0; i < 32; i++) t[i] = __fadd_rn(v[i], v[i + o < 32 ? i + o : i]);
        for (int i = 0; i < 32; i++) v[i] = t[i];
      }
      acc = __fadd_rn(acc, v[0]);                    // groups ascending
    }
  }
  y[r] = acc;
}

static void quant_row(const float* w, int K, int8_t* q, __half* s) {
  const int nb = K / Q8_BLK;
  for (int b = 0; b < nb; b++) {
    float mx = 0.f;
    for (int j = 0; j < Q8_BLK; j++) mx = fmaxf(mx, fabsf(w[b * Q8_BLK + j]));
    __half h = __float2half(mx / 127.f);
    float sc16 = __half2float(h);
    if (sc16 == 0.f) { memset(q + (size_t)b * Q8_BLK, 0, Q8_BLK); s[b] = __float2half(0.f); continue; }
    for (int j = 0; j < Q8_BLK; j++) {
      int iv = (int)lrintf(w[b * Q8_BLK + j] / sc16);
      if (iv > 127) iv = 127; if (iv < -127) iv = -127;
      q[(size_t)b * Q8_BLK + j] = (int8_t)iv;
    }
    s[b] = h;
  }
}

int main(void) {
  struct Shape { int N, K; const char* name; double share; };
  Shape shapes[] = {
    {7680, 2048, "gate+up w1,w3", 0.54},
    {2048, 7680, "down w2",       0.22},
    {6144, 2048, "conv_in_proj",  0.11},
  };
  const int NSH = (int)(sizeof shapes / sizeof *shapes);
  // KCHUNK >= 1024 REQUIRED: the reduction is a 32-lane butterfly, so a row
  // must span a full warp. KCHUNK=512 -> nb=16 -> WPB = nb/32 = 0 ->
  // `wid % WPB' divides by zero, kills the context, and every later config
  // then reports "unspecified launch failure".
  const int KCH[] = {1024, 2048};
  const int NKCH = (int)(sizeof KCH / sizeof *KCH);
  const int TPB[] = {128, 256, 512};
  const int NTPB = (int)(sizeof TPB / sizeof *TPB);
  const int RR[] = {4, 8, 16, 32};
  const int NRR = (int)(sizeof RR / sizeof *RR);

  int maxN = 0, maxK = 0;
  for (int i = 0; i < NSH; i++) { if (shapes[i].N > maxN) maxN = shapes[i].N;
                                  if (shapes[i].K > maxK) maxK = shapes[i].K; }
  size_t wmax = (size_t)maxN * maxK;
  const int maxnb = maxK / Q8_BLK;
  float* hw = (float*)malloc(wmax * 4);
  int8_t* hq = (int8_t*)malloc(wmax);
  __half* hs = (__half*)malloc((size_t)maxN * maxnb * 2);
  float* hx = (float*)malloc((size_t)maxK * 4);
  float* ay = (float*)malloc((size_t)maxN * 4);
  float* ar = (float*)malloc((size_t)maxN * 4);
  int8_t *dq; __half *ds; float *dx, *dy, *dpart, *dref;
  for (size_t i = 0; i < wmax; i++) hw[i] = (float)(((int)(i * 2654435761u) >> 8) % 4001 - 2000) / 500.f;
  cudaMalloc(&dq, wmax);
  cudaMalloc(&ds, (size_t)maxN * maxnb * 2);
  cudaMalloc(&dx, (size_t)maxK * 4);
  cudaMalloc(&dy, (size_t)maxN * 4);
  cudaMalloc(&dpart, (size_t)maxN * 32 * 4);
  cudaMalloc(&dref, (size_t)maxN * 4);
  int dev; cudaGetDevice(&dev);
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
  printf("GPU: %s  SMs=%d  maxThreadsPerSM=%d  sharedPerBlock=%zu  sharedPerSM=%zu\n\n",
         prop.name, prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor,
         prop.sharedMemPerBlock, prop.sharedMemPerMultiprocessor);

  for (int si = 0; si < NSH; si++) {
    const int N = shapes[si].N, K = shapes[si].K;
    for (int r = 0; r < N; r++)
      quant_row(hw + (size_t)r * K, K, hq + (size_t)r * K, hs + (size_t)r * (K / Q8_BLK));
    cudaMemcpy(dq, hq, (size_t)N * K, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs, (size_t)N * (K / Q8_BLK) * 2, cudaMemcpyHostToDevice);
    for (int i = 0; i < K; i++) hx[i] = (float)(((int)(i * 40503u) >> 4) % 2001 - 1000) / 700.f;
    cudaMemcpy(dx, hx, (size_t)K * 4, cudaMemcpyHostToDevice);
    const double mb = (N * (double)K + (size_t)N * (K / Q8_BLK) * 2 + (size_t)K * 4 + (size_t)N * 4) / 1e6;
    printf("=== %s  N=%d K=%d  (%.1f MB, %.0f%% of decode GEMV) ===\n",
           shapes[si].name, N, K, mb, 100 * shapes[si].share);
    printf("%7s %4s %4s %7s %9s %8s %7s  %s\n",
           "KCHUNK", "TPB", "R", "blocks", "ms", "GB/s", "thr/SM", "bit-exact");
    double bestms = 1e30; int bk = 0, bt = 0, br = 0;
    int nchecked = 0, nskip = 0, nfail = 0;

    for (int ki = 0; ki < NKCH; ki++) {
      const int KCHUNK = KCH[ki];
      if (KCHUNK > K || K % KCHUNK) continue;
      const int nb = KCHUNK / Q8_BLK;
      const int nchunk = K / KCHUNK;
      if (nb < 32 || nb / 32 > WPMAX) continue;
      ref_chunked<<<(N + 255) / 256, 256>>>(dq, ds, dx, dref, N, K, KCHUNK);
      cudaMemcpy(ar, dref, (size_t)N * 4, cudaMemcpyDeviceToHost);

      // collect the valid configs for this chunk width FIRST, then run each in
      // its own process: a divide-by-zero kernel kills the CUDA context and
      // every later launch in the same process reports "unspecified launch
      // failure", which turns one bad config into a whole bogus sweep.
      for (int ti = 0; ti < NTPB; ti++) {
        for (int ri = 0; ri < NRR; ri++) {
          const int NT = TPB[ti], R = RR[ri];
          if (NT % nb) { nskip++; continue; }
          const int perrow = NT / nb;
          if (perrow < 1 || R % perrow || perrow > R) { nskip++; continue; }
          if (R > RMAX) { nskip++; continue; }
          const size_t shm = (size_t)nb * 33 * 4;
          if (shm > prop.sharedMemPerBlock) { nskip++; continue; }
          const int nblk = ((N + R - 1) / R) * nchunk;
          const int reps = (int)(2e8 / (double)((size_t)N * K)) + 3;

          q8_partial<<<nblk, NT, shm>>>(dq, ds, dx, dpart, N, K, KCHUNK, R, nb);
          q8_combine<<<(N + 255) / 256, 256>>>(dpart, dy, N, nchunk);
          if (cudaGetLastError() != cudaSuccess) { nfail++; continue; }
          cudaDeviceSynchronize();
          if (cudaGetLastError() != cudaSuccess) { nfail++; continue; }

          for (int w = 0; w < 10; w++) {
            q8_partial<<<nblk, NT, shm>>>(dq, ds, dx, dpart, N, K, KCHUNK, R, nb);
            q8_combine<<<(N + 255) / 256, 256>>>(dpart, dy, N, nchunk);
          }
          cudaDeviceSynchronize();
          // CUDA events, not clock_gettime. On this board's nvcc/CUDA-10.2
          // host path clock_gettime was returning failure, so t0 stayed
          // UNINITIALIZED: the sweep printed 2.9e12 ms and, tellingly,
          // NEGATIVE times for a kernel that was measurably running (every
          // config was bit-exact YES, and the same kernels time at 1.3-2.0 ms
          // under cudaEvent in q8_gemv_splitk.cu). A clock that can fail
          // silently is not a stopwatch; events cannot.
          cudaEvent_t ev0, ev1;
          cudaEventCreate(&ev0); cudaEventCreate(&ev1);
          double best = 1e30;
          for (int t = 0; t < 3; t++) {
            cudaEventRecord(ev0);
            for (int r2 = 0; r2 < reps; r2++) {
              q8_partial<<<nblk, NT, shm>>>(dq, ds, dx, dpart, N, K, KCHUNK, R, nb);
              q8_combine<<<(N + 255) / 256, 256>>>(dpart, dy, N, nchunk);
            }
            cudaEventRecord(ev1);
            cudaEventSynchronize(ev1);
            float msf = 0.f; cudaEventElapsedTime(&msf, ev0, ev1);
            const double ms = msf / reps;
            if (ms < best) best = ms;
          }
          cudaEventDestroy(ev0); cudaEventDestroy(ev1);
          cudaMemcpy(ay, dy, (size_t)N * 4, cudaMemcpyDeviceToHost);
          long bad = 0;
          for (int i = 0; i < N; i++) if (ay[i] != ar[i]) bad++;

          const int cap_thr = (int)prop.maxThreadsPerMultiProcessor / NT;
          const int cap_shm = (int)(prop.sharedMemPerMultiprocessor / (shm + 2048));
          int blocks_res = cap_thr < cap_shm ? cap_thr : cap_shm;
          if (blocks_res < 1) blocks_res = 1;
          const int thr_sm = blocks_res * NT;
          const double gb = mb / (best * 1e-3) / 1e6 * 1e3;
          printf("%7d %4d %4d %7d %9.3f %8.2f %7d  %s\n",
                 KCHUNK, NT, R, nblk, best, gb, thr_sm, bad ? "NO" : "YES");
          nchecked++;
          if (!bad && best < bestms) { bestms = best; bk = KCHUNK; bt = NT; br = R; }
        }
      }
    }
    if (bestms < 1e29)
      printf("--> BEST %s: KCHUNK=%d TPB=%d R=%d  %.3f ms  %.2f GB/s   (%d checked, %d skipped, %d failed)\n\n",
             shapes[si].name, bk, bt, br, bestms, mb / (bestms * 1e-3) / 1e6 * 1e3,
             nchecked, nskip, nfail);
    else
      printf("--> %s: NO VALID CONFIG (checked %d, skipped %d, failed %d)\n\n",
             shapes[si].name, nchecked, nskip, nfail);
  }
  printf("SWEEP_DONE\n");
  return 0;
}
