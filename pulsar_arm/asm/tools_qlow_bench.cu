// tools_qlow_bench.cu -- standalone timing of k_qlow (lfm3_qlow.cuh) on the LFM3 backbone's
// Q5 shapes, random payload, sweeping block size TPB, prefetch depth PD and K split KS.
// Reports GB/s of payload (codes + high plane + scales) per launch incl. k_qlow_combine.
//   nvcc -std=c++11 -gencode arch=compute_53,code=sm_53 -O3 -fmad=false -Xptxas -O3,-dlcm=ca \
//        -o qlow_bench tools_qlow_bench.cu && ./qlow_bench
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "lfm3_qlow.cuh"

template <int TPB, int PD, int MB = 2048 / TPB>
static float run(const uint4* q, const __half2* s, const unsigned short* x, float* part,
                 unsigned short* y, int N, int K, int KS, int it) {
  const int G = K / 32;
  if ((G / KS) % PD || G % KS) return -1.f;
  const int KL = G / KS * 32; const size_t sm = (KL + KL / 32) * 4;
  dim3 gr((N + TPB - 1) / TPB, KS);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (int w = 0; w < 3; w++) {
    k_qlow<5, TPB, PD, MB><<<gr, TPB, sm>>>(q, s, x, part, y, N, K);
    k_qlow_combine<<<(N + 255) / 256, 256>>>(part, y, N, KS);
  }
  cudaEventRecord(a);
  for (int i = 0; i < it; i++) {
    k_qlow<5, TPB, PD, MB><<<gr, TPB, sm>>>(q, s, x, part, y, N, K);
    k_qlow_combine<<<(N + 255) / 256, 256>>>(part, y, N, KS);
  }
  cudaEventRecord(b); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms, a, b);
  if (cudaGetLastError() != cudaSuccess) return -2.f;
  return ms / it;
}

int main(int argc, char** argv) {
  const int shapes[4][2] = { {7680, 2048}, {2048, 7680}, {6144, 2048}, {2048, 2048} };
  const int it = argc > 1 ? atoi(argv[1]) : 50;
  for (auto& sh : shapes) {
    const int N = sh[0], K = sh[1], G = K / 32;
    const size_t nq = (size_t)(N / 32) * G * 40, ns = (size_t)(N / 32) * G * 32;
    std::vector<uint32_t> hq(nq * 4); for (auto& v : hq) v = (uint32_t)rand() * 2654435761u;
    std::vector<__half2> hs(ns); for (auto& v : hs) v = __floats2half2_rn(0.01f, -0.3f);
    std::vector<unsigned short> hx(K); for (auto& v : hx) v = 0x3f00 + (rand() & 0xff);
    uint4* q; __half2* s; unsigned short *x, *y; float* part;
    cudaMalloc(&q, nq * 16); cudaMalloc(&s, ns * 4); cudaMalloc(&x, K * 2); cudaMalloc(&y, N * 2);
    cudaMalloc(&part, (size_t)16 * N * 4);
    cudaMemcpy(q, hq.data(), nq * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(s, hs.data(), ns * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(x, hx.data(), K * 2, cudaMemcpyHostToDevice);
    const double mb = (nq * 16 + ns * 4) / 1e6;
    printf("N %5d K %5d (%.1f MB)\n", N, K, mb);
    for (int KS : {2, 4, 8}) {
#define R(T, P, M) { float ms = run<T, P, M>(q, s, x, part, y, N, K, KS, it); \
      if (ms > 0) printf("  KS %d TPB %3d PD %d MINB %d  %7.1f us  %5.1f GB/s\n", KS, T, P, M, ms * 1e3, mb / ms); }
      R(256, 1, 8) R(256, 2, 6) R(256, 2, 5) R(256, 2, 4) R(256, 4, 4) R(256, 4, 3) R(256, 4, 2) R(128, 2, 8) R(128, 4, 6) R(128, 4, 4)
    }
    cudaFree(q); cudaFree(s); cudaFree(x); cudaFree(y); cudaFree(part);
  }
  return 0;
}
