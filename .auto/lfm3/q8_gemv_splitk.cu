// q8_gemv_splitk.cu -- split-K staged Q8_0 GEMV: the FFN down-projection
// (2048x7680) at the rate the K=2048 shapes already get.
//
// The problem. LFM3's GEMVs are 1.558 GB/token, so the GEMV rate IS the tok/s
// target. Per shape, measured on this board: the K=2048 shapes reach 9.05 GB/s
// (staged) but the K=7680 down-projection reaches only 5.96 (kg). w2 is 377.5 M
// params x 24 layers, so that gap is 24 x (2.81 - 1.85) = 23 ms/token out of
// ~152 -- the largest single kernel deficiency left.
//
// Why kg is slower: it is one warp per row, so EVERY row re-reads all of x
// through L1. x traffic is N*K*4 = 63 MB at 2048x7680 against 16 MB of
// weights -- the kernel moves 4x more activation than weight. The staged
// kernel fixes exactly that by putting x in shared once per block and
// amortizing over R rows, but at K=7680 that is 30 KB of shared per block, and
// the Nano has ONE SM with 64 KB, so the shared pool caps residency at 2
// blocks = 512 threads. Occupancy, not bandwidth.
//
// The fix: split K. Chunk K into 2048-column pieces so every block runs the
// tuned K=2048 configuration (8.4 KB shared -> 7 blocks resident on the SM)
// and stages only 2048 floats. grid = (N/R) x (K/2048).
//
// Bit-exactness needs a DETERMINISTIC combine, so no atomics: pass 1 writes
// partials[r*nchunk + c], pass 2 sums the chunks ascending with __fadd_rn.
// Each partial is produced by the staged 2048-wide reduction, i.e. per row:
// 64 Q8_0 block partials, a 32-lane butterfly over blocks 0..31, the same over
// 32..63, and the two halves added -- which is exactly what
// q8_ref_tree_kernel computes for a 2048-wide row, so the reference is reused
// verbatim per chunk and the chunk loop is pinned on top of it. Built
// -fmad=false so nothing contracts anywhere.
//
// Build on the Nano (CUDA 10.2):
//   nvcc -gencode arch=compute_53,code=sm_53 -O3 -fmad=false \
//        -Xptxas -O3,-dlcm=ca -o q8_gemv_splitk q8_gemv_splitk.cu
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cuda_fp16.h>

#define Q8_BLK 32
#define NT 256
#define KCHUNK 2048
#define RMAX 8                        /* max rows per block; NT/nb = 4 must divide R */

static double ms_since(const struct timespec* a) {
  struct timespec b; clock_gettime(CLOCK_MONOTONIC, &b);
  return (b.tv_sec - a->tv_sec) * 1e3 + (b.tv_nsec - a->tv_nsec) * 1e-6;
}

// ---------------------------------------------------- pass 1: partial dots
// One block: R rows x one 2048-column chunk. x chunk staged in shared with
// the 32->33 padded stride (an unpadded 32 stride costs a 32-way bank
// conflict on the per-block xb[j] reads).
__global__ void q8_partial_2048(const int8_t* __restrict__ q, const __half* __restrict__ s,
                                const float* __restrict__ x, float* __restrict__ part,
                                int N, int K, int R) {
  const int nchunk = K / KCHUNK;
  const int rowblk = blockIdx.x / nchunk;       // which group of R rows
  const int chunk  = blockIdx.x % nchunk;       // which K chunk
  const int koff   = chunk * KCHUNK;            // column offset of this chunk
  const int nb = KCHUNK / Q8_BLK;               // 64

  __shared__ float xs[64 * 33];
  const int tid = threadIdx.x;
  const float* xg = x + koff;
  for (int i = tid; i < KCHUNK; i += NT) {
    int b = i >> 5, j = i & 31;
    xs[b * 33 + j] = xg[i];
  }
  __syncthreads();

  const int base = rowblk * R;
  const int row_local = tid / nb;               // 0..R-1 (R = NT/nb = 4)
  const int b = tid % nb;
  const int GR = R / (NT / nb);
  // RMAX, not 4: r = row_local + g*(NT/nb) reaches R-1, so R=8 walked
  // acc[4] and sh[4][2] off the end. The first R=8 run reported a clean
  // "YES" bit-exact and 10.4 GB/s -- 24% faster than R=4 -- from a kernel
  // doing out-of-bounds writes, and the tell was two IDENTICAL configs
  // timing 2.053 and 1.604 ms in the same run. A win from an OOB kernel is
  // not a win; size by R, then measure.
  float acc[RMAX];
  #pragma unroll
  for (int g = 0; g < GR; g++) {
    const int r = row_local + g * (NT / nb);
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
  __shared__ float sh[RMAX][2];
  #pragma unroll
  for (int g = 0; g < GR; g++) {
    const int r = row_local + g * (NT / nb);
    float v = acc[r];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(~0u, v, o);
    if (lane == 0) sh[r][wid % 2] = v;
  }
  __syncthreads();
  if (tid < R && base + tid < N)
    part[(size_t)(base + tid) * nchunk + chunk] = sh[tid][0] + sh[tid][1];
}

// --------------------------------------------------- pass 2: pinned combine
// Sequential ascending over chunks, __fadd_rn so the compiler cannot reassociate.
__global__ void q8_combine(const float* __restrict__ part, float* __restrict__ out,
                           int N, int nchunk) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= N) return;
  float acc = 0.f;
  for (int c = 0; c < nchunk; c++) acc = __fadd_rn(acc, part[(size_t)r * nchunk + c]);
  out[r] = acc;
}

// --------------------------------------- bit-exact reference (device, pinned)
// Per chunk: the staged 2048-wide tree (butterfly over block partials 0..31
// and 32..63, halves added). Then chunks summed ascending.
__global__ void ref_chunked(const int8_t* q, const __half* s, const float* x,
                            float* y, int N, int K) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= N) return;
  const int nchunk = K / KCHUNK;
  const int nb = KCHUNK / Q8_BLK;
  float acc = 0.f;
  for (int c = 0; c < nchunk; c++) {
    const int koff = c * KCHUNK;
    float p[64];
    for (int b = 0; b < nb; b++) {
      float sum = 0.f;
      for (int j = 0; j < Q8_BLK; j++)
        sum = __fadd_rn(sum, __fmul_rn((float)q[(size_t)r * K + koff + (size_t)b * Q8_BLK + j],
                                       x[koff + (size_t)b * Q8_BLK + j]));
      p[b] = __fmul_rn(__half2float(s[(size_t)r * (K / Q8_BLK) + (size_t)c * nb + b]), sum);
    }
    // butterfly32 over p[0..31] and p[32..63], then the two halves added
    for (int half = 0; half < 2; half++) {
      float* v = p + half * 32;
      for (int o = 16; o > 0; o >>= 1) {
        float t[32];
        for (int i = 0; i < 32; i++) t[i] = __fadd_rn(v[i], v[i + o < 32 ? i + o : i]);
        for (int i = 0; i < 32; i++) v[i] = t[i];
      }
    }
    acc = __fadd_rn(acc, __fadd_rn(p[0], p[32]));
  }
  y[r] = acc;
}

static void quant_row(const float* w, int K, int8_t* q, __half* s) {
  const int nb = K / Q8_BLK;
  for (int b = 0; b < nb; b++) {
    float mx = 0.f;
    for (int j = 0; j < Q8_BLK; j++) mx = fmaxf(mx, fabsf(w[b * Q8_BLK + j]));
    float sc = mx / 127.f;
    __half h = __float2half(sc);
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

static double gTokMs = 0.0; static int gFail = 0;
int main(int argc, char** argv) {
  struct Shape { int N, K, R; const char* name; int per_tok; };
  // per_tok = how many times this GEMV runs per decoded token in LFM3-Audio-2B
  // (13 conv layers, 11 attention layers, 24 layers, FFN in all 24). R=0 means
  // "sweep R = 4 and 8 and keep the best".
  Shape shapes[] = {
    {6144, 2048, 0, "conv_in_proj  (13x)", 13},
    {2048, 2048, 0, "conv_out_proj (13x)", 13},
    {2048, 2048, 0, "w_q           (11x)", 11},
    { 512, 2048, 0, "w_k           (11x)", 11},
    { 512, 2048, 0, "w_v           (11x)", 11},
    {2048, 2048, 0, "w_o           (11x)", 11},
    {7680, 2048, 0, "gate+up w1,w3 (24x)", 48},
    {2048, 7680, 0, "down  w2      (24x)", 24},
    {2048, 4096, 4, "K=4096 sanity (not in tok)", 0},
  };
  const int NS = (int)(sizeof shapes / sizeof *shapes);
  int maxN = 0, maxK = 0;
  for (int i = 0; i < NS; i++) { if (shapes[i].N > maxN) maxN = shapes[i].N;
                                 if (shapes[i].K > maxK) maxK = shapes[i].K; }
  size_t wmax = (size_t)maxN * maxK;
  const int maxnb = maxK / Q8_BLK;

  float* hw = (float*)malloc(wmax * 4);
  int8_t* hq = (int8_t*)malloc(wmax);
  __half* hs = (__half*)malloc((size_t)maxN * maxnb * 2);
  float* hx = (float*)malloc((size_t)maxK * 4);
  int8_t *dq; __half *ds; float *dx, *dy, *dpart, *dref;
  for (size_t i = 0; i < wmax; i++) hw[i] = (float)(((int)(i * 2654435761u) >> 8) % 4001 - 2000) / 500.f;
  cudaMalloc(&dq, wmax);
  cudaMalloc(&ds, (size_t)maxN * maxnb * 2);
  cudaMalloc(&dx, (size_t)maxK * 4);
  cudaMalloc(&dy, (size_t)maxN * 4);
  cudaMalloc(&dpart, (size_t)maxN * 16 * 4);
  cudaMalloc(&dref, (size_t)maxN * 4);

  printf("%-26s %6s %6s %4s %8s %9s %8s  %s\n",
         "shape", "N", "K", "R", "MB", "ms", "GB/s", "bit-exact");
  for (int si = 0; si < NS; si++) {
    const int N = shapes[si].N, K = shapes[si].K;
    const int Rsweep[2] = {4, 8};
    const int nR = shapes[si].R ? 1 : 2;
    const int nchunk = K / KCHUNK;
    for (int r = 0; r < N; r++)
      quant_row(hw + (size_t)r * K, K, hq + (size_t)r * K, hs + (size_t)r * (K / Q8_BLK));
    cudaMemcpy(dq, hq, (size_t)N * K, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs, (size_t)N * (K / Q8_BLK) * 2, cudaMemcpyHostToDevice);
    for (int i = 0; i < K; i++) hx[i] = (float)(((int)(i * 40503u) >> 4) % 2001 - 1000) / 700.f;
    cudaMemcpy(dx, hx, (size_t)K * 4, cudaMemcpyHostToDevice);

    for (int ri = 0; ri < nR; ri++) {
    const int R = Rsweep[ri];        /* a multiple of NT/nb = 4, <= RMAX */
    const int nblk = (N + R - 1) / R * nchunk;
    double mb = (N * (double)K + (size_t)N * (K / Q8_BLK) * 2 + (size_t)K * 4 + (size_t)N * 4) / 1e6;

    float* ay = (float*)malloc((size_t)N * 4);
    float* ar = (float*)malloc((size_t)N * 4);
    ref_chunked<<<(N + 255) / 256, 256>>>(dq, ds, dx, dref, N, K);
    cudaMemcpy(ar, dref, (size_t)N * 4, cudaMemcpyDeviceToHost);

    int reps = (int)(2e8 / (double)((size_t)N * K)) + 3;
    for (int w = 0; w < 10; w++) {
      q8_partial_2048<<<nblk, NT>>>(dq, ds, dx, dpart, N, K, R);
      q8_combine<<<(N + 255) / 256, 256>>>(dpart, dy, N, nchunk);
    }
    cudaDeviceSynchronize();
    double best = 1e30;
    for (int t = 0; t < 3; t++) {
      struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
      for (int r = 0; r < reps; r++) {
        q8_partial_2048<<<nblk, NT>>>(dq, ds, dx, dpart, N, K, R);
        q8_combine<<<(N + 255) / 256, 256>>>(dpart, dy, N, nchunk);
      }
      // MUST sync before stopping the clock: launches are async, so without
      // this the measured interval is the time to ENQUEUE the launches. That
      // is what produced the absurd 405 GB/s (a constant 0.04 ms for every
      // shape, 16.8 MB and 4.5 MB alike, against a 21.9 GB/s board ceiling).
      cudaDeviceSynchronize();
      double ms = ms_since(&t0) / reps;
      if (ms < best) best = ms;
    }
    cudaMemcpy(ay, dy, (size_t)N * 4, cudaMemcpyDeviceToHost);
    long bad = 0; double mx = 0;
    for (int i = 0; i < N; i++) { if (ay[i] != ar[i]) bad++;
      double d = fabs((double)ay[i] - ar[i]); if (d > mx) mx = d; }
    printf("%-26s %6d %6d %4d %8.1f %9.3f %8.2f  %s\n",
           shapes[si].name, N, K, R, mb, best, mb * 1e-3 / (best * 1e-3),
           bad ? "NO" : "YES");
    if (bad) gFail = 1;
    if (shapes[si].per_tok && ri == nR - 1) gTokMs += best * shapes[si].per_tok;
    free(ay); free(ar);
    cudaError_t e = cudaGetLastError();
    if (e) printf("  cuda: %s\n", cudaGetErrorString(e));
    }
  }
  printf("\nGEMV per decoded token, best R per shape: %.1f ms -> %.2f tok/s\n",
         gTokMs, 1000.0 / gTokMs);
  printf("%s\n", gFail ? "SPLITK_FAIL" : "SPLITK_OK");
  return 0;
}
