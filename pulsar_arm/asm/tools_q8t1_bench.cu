// tools_q8t1_bench.cu -- standalone timing for the depth server's T=1 Q8 GEMVs.
// Baseline: k_q8g<1,1,2> (lfm3_q8g.cuh, row-major Q8, 128-thread blocks).
// Candidate: k_q8t1 (32-row tiled Q8, lane = row, PD groups in flight, MINB=4).
// Shapes: depth_linear [12288, 2048], head codebook [2017, 1024].
// Bit-identical by construction (per-row ascending fma chain, full K in one chain);
// the bench checks max abs diff (must be 0) and prints GB/s of codes+scales.
//   nvcc -std=c++11 -gencode arch=compute_53,code=sm_53 -O3 -fmad=false -Xptxas -O3,-dlcm=ca \
//        -o q8t1_bench tools_q8t1_bench.cu && ./q8t1_bench [iters]
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "lfm3_q8g.cuh"

// ---- candidate: T=1 Q8 on 32-row tiles, lane = row ---------------------------
// Tile layout per group of 32 codes along K: 8 int slots x 32 rows, then 32 scales.
// codes: for tile, for group g: 8 slots of 32 ints; slot k holds the 32 rows' k-th
//        code-int contiguously, so a warp's 32 lanes load 128 coalesced bytes.
// scales: for tile, for group g: 32 floats, one per row.
// Per row: acc = 0; groups ascending; in group codes ascending:
//   acc = fmaf(fmul((float)code, scale), x, acc) -- the k_q8g contract exactly
// (scale folded per code with one rounding, then fma: same two roundings as k_q8g).
template <int PD, int MINB>
__global__ void __launch_bounds__(256, MINB) k_q8t1(const int* __restrict__ q, const float* __restrict__ s,
                                                    const float* __restrict__ x, float* __restrict__ y,
                                                    int N, int K) {
  extern __shared__ float xs[];                  // x staged + nothing else (T=1)
  const int G = K / 32, tid = threadIdx.x;
  for (int i = tid; i < K; i += 256) xs[i] = x[i];
  __syncthreads();
  const int row = blockIdx.x * 256 + tid;
  if (row >= N) return;
  const size_t tile = (size_t)(row >> 5), lane = row & 31, tg = tile * (size_t)G;
  float yv = 0.f;
  int nw[PD * 8]; float ns[PD];
  #pragma unroll
  for (int u = 0; u < PD; u++) {
    const size_t b = (tg + u) * 256;
    #pragma unroll
    for (int k = 0; k < 8; k++) nw[u * 8 + k] = q[b + (size_t)k * 32 + lane];
    ns[u] = s[(tg + u) * 32 + lane];
  }
  for (int g = 0; g < G; g += PD) {             // G % PD == 0 (checked by caller)
    int w[PD * 8]; float sc[PD];
    #pragma unroll
    for (int u = 0; u < PD; u++) {
      #pragma unroll
      for (int k = 0; k < 8; k++) w[u * 8 + k] = nw[u * 8 + k];
      sc[u] = ns[u];
    }
    if (g + PD < G) {
      #pragma unroll
      for (int u = 0; u < PD; u++) {
        const size_t b = (tg + (size_t)g + PD + u) * 256;
        #pragma unroll
        for (int k = 0; k < 8; k++) nw[u * 8 + k] = q[b + (size_t)k * 32 + lane];
        ns[u] = s[(tg + (size_t)g + PD + u) * 32 + lane];
      }
    }
    #pragma unroll
    for (int u = 0; u < PD; u++) {
      #pragma unroll
      for (int k = 0; k < 8; k++) {
        const int v = w[u * 8 + k];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          const float wf = __fmul_rn((float)(signed char)(v >> (8 * j)), sc[u]);
          yv = __fmaf_rn(wf, xs[(size_t)(g + u) * 32 + k * 4 + j], yv);
        }
      }
    }
  }
  y[row] = yv;
}

static void tile_q8(const int8_t* q, const float* s, int* tq, float* ts, int N, int K) {
  const int G = K / 32;
  for (int t = 0; t < N; t += 32)
    for (int g = 0; g < G; g++)
      for (int r = 0; r < 32 && t + r < N; r++) {
        const int* src = (const int*)(q + (size_t)(t + r) * K + g * 32);
        int* d = tq + ((size_t)(t >> 5) * G + g) * 256 + r;
        for (int k = 0; k < 8; k++) d[k * 32] = src[k];
        ts[((size_t)(t >> 5) * G + g) * 32 + r] = s[(size_t)(t + r) * G + g];
      }
}

template <int PD, int MINB>
static float run_new(const int* q, const float* s, const float* x, float* y, int N, int K, int it) {
  if ((K / 32) % PD) return -1.f;
  dim3 gr((N + 255) / 256);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (int w = 0; w < 3; w++) k_q8t1<PD, MINB><<<gr, 256, (size_t)K * 4>>>(q, s, x, y, N, K);
  cudaEventRecord(a);
  for (int i = 0; i < it; i++) k_q8t1<PD, MINB><<<gr, 256, (size_t)K * 4>>>(q, s, x, y, N, K);
  cudaEventRecord(b); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms, a, b);
  if (cudaGetLastError() != cudaSuccess) return -2.f;
  return ms / it;
}

static float run_old(const int8_t* q, const float* s, const float* x, float* y, int N, int K, int it, int kb) {
  const unsigned nbk = (unsigned)((N + 127) / 128);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (int w = 0; w < 3; w++) {
    if (kb == 2) k_q8g<1, 1, 2><<<nbk, 128>>>(q, s, x, y, N, K, 0);
    else k_q8g<1, 1, 1><<<nbk, 128>>>(q, s, x, y, N, K, 0);
  }
  cudaEventRecord(a);
  for (int i = 0; i < it; i++) {
    if (kb == 2) k_q8g<1, 1, 2><<<nbk, 128>>>(q, s, x, y, N, K, 0);
    else k_q8g<1, 1, 1><<<nbk, 128>>>(q, s, x, y, N, K, 0);
  }
  cudaEventRecord(b); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms, a, b);
  if (cudaGetLastError() != cudaSuccess) return -2.f;
  return ms / it;
}

int main(int argc, char** argv) {
  const int it = argc > 1 ? atoi(argv[1]) : 40;
  const int shapes[2][2] = { {12288, 2048}, {2017, 1024} };
  for (auto& sh : shapes) {
    const int N = sh[0], K = sh[1], G = K / 32, Nt = (N + 31) & ~31;
    std::vector<int8_t> hq((size_t)N * K); for (size_t i = 0; i < hq.size(); i++) hq[i] = (int8_t)(rand() % 255 - 127);
    std::vector<float> hs((size_t)N * G); for (auto& v : hs) v = 0.01f * (rand() % 100 - 50);
    std::vector<float> hx(K); for (auto& v : hx) v = 0.1f * (rand() % 200 - 100);
    std::vector<int> ht((size_t)(Nt / 32) * G * 256, 0);
    std::vector<float> hts((size_t)(Nt / 32) * G * 32, 0);
    tile_q8(hq.data(), hs.data(), ht.data(), hts.data(), N, K);
    int8_t *q; float *s, *x, *y, *yr; int *tq; float *ts;
    cudaMalloc(&q, (size_t)N * K); cudaMalloc(&s, (size_t)N * G * 4);
    cudaMalloc(&x, (size_t)K * 4); cudaMalloc(&y, (size_t)N * 4); cudaMalloc(&yr, (size_t)N * 4);
    cudaMalloc(&tq, ht.size() * 4); cudaMalloc(&ts, hts.size() * 4);
    cudaMemcpy(q, hq.data(), (size_t)N * K, cudaMemcpyHostToDevice);
    cudaMemcpy(s, hs.data(), (size_t)N * G * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(x, hx.data(), (size_t)K * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(tq, ht.data(), ht.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(ts, hts.data(), hts.size() * 4, cudaMemcpyHostToDevice);
    const double mb = ((size_t)N * K + (size_t)N * G * 4) / 1e6;
    printf("N %5d K %5d (%.1f MB)\n", N, K, mb);
    float mo1 = run_old(q, s, x, y, N, K, it, 1);
    float mo2 = (K % 64 == 0) ? run_old(q, s, x, y, N, K, it, 2) : -1.f;
    if (mo1 > 0) printf("  old KB1        %8.1f us  %5.1f GB/s\n", mo1 * 1e3, mb / mo1);
    if (mo2 > 0) printf("  old KB2        %8.1f us  %5.1f GB/s\n", mo2 * 1e3, mb / mo2);
    // reference outputs for exactness
    std::vector<float> href(N); cudaMemcpy(href.data(), y, (size_t)N * 4, cudaMemcpyDeviceToHost);
#define R(P, M) { float ms = run_new<P, M>(tq, ts, x, yr, N, K, it); \
      if (ms > 0) { std::vector<float> h(N); cudaMemcpy(h.data(), yr, (size_t)N * 4, cudaMemcpyDeviceToHost); \
        double md = 0; for (int i = 0; i < N; i++) md = fmax(md, fabs(h[i] - href[i])); \
        printf("  new PD%d MINB%d %8.1f us  %5.1f GB/s  maxdiff %.3g\n", P, M, ms * 1e3, mb / ms, md); } }
    R(1, 8) R(2, 4) R(2, 6) R(4, 2)
    cudaFree(q); cudaFree(s); cudaFree(x); cudaFree(y); cudaFree(yr); cudaFree(tq); cudaFree(ts);
  }
  return 0;
}
