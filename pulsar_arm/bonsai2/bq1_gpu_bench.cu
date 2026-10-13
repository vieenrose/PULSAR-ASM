/* bq1_gpu_bench.cu — Nano Gen1 roofline + parity bench, no model needed.
 * Compares bq1_gemv_sm53 (GPU) vs bq1_gemv_cpu (A57) on synthetic binary
 * weights, measures streamed GB/s and equivalent 8B tok/s.
 *
 * Build on Nano:
 *   nvcc -gencode arch=compute_53,code=sm_53 -O3 -maxrregcount=64 \
 *     --use_fast_math -std=c++11 bq1_gpu_bench.cu bq1_swar_sm53.o bq1_cpu.c \
 *     -o /tmp/bq1_gpu_bench -lcudart
 * Run:
 *   /tmp/bq1_gpu_bench [K=4096] [N=4096] [iters=200]
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "bq1_wrap.h"

extern int bq1_gemv_sm53_launch(const uint32_t*, const unsigned short*,
  const signed char*, float, float*, int, int, void*);

static double now_s(void) {
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
  int K = argc > 1 ? atoi(argv[1]) : 4096;
  int N = argc > 2 ? atoi(argv[2]) : 4096;
  int iters = argc > 3 ? atoi(argv[3]) : 200;
  if ((K & 255) || K <= 0 || N <= 0) { printf("K must be %%256==0\n"); return 2; }
  int K32 = K >> 5, K256 = K >> 8;

  uint32_t* wb = (uint32_t*)malloc((size_t)N * K32 * 4);
  uint16_t* ws = (uint16_t*)malloc((size_t)N * K256 * 2);
  float* x = (float*)malloc((size_t)K * 4);
  int8_t* xq = (int8_t*)malloc((size_t)K);
  float *y = (float*)malloc((size_t)N * 4), *yg = (float*)malloc((size_t)N * 4);
  srand(12345);
  for (int i = 0; i < N * K32; i++) wb[i] = (uint32_t)rand() ^ ((uint32_t)rand() << 16);
  for (int i = 0; i < N * K256; i++) ws[i] = bq1_f2h(0.5f + (rand() % 100) / 200.0f);
  for (int i = 0; i < K; i++) x[i] = ((rand() % 2001) - 1000) / 1500.0f;
  float xs = bq1_quant_row_i8(x, xq, K);

  /* device buffers (discrete cudaMalloc; on Tegra this is unified LPDDR4) */
  uint32_t *d_wb; unsigned short *d_ws; signed char *d_xq; float *d_y;
  cudaMalloc(&d_wb, (size_t)N * K32 * 4);
  cudaMalloc(&d_ws, (size_t)N * K256 * 2);
  cudaMalloc(&d_xq, (size_t)K);
  cudaMalloc(&d_y, (size_t)N * 4);
  cudaMemcpy(d_wb, wb, (size_t)N * K32 * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(d_ws, ws, (size_t)N * K256 * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(d_xq, xq, (size_t)K, cudaMemcpyHostToDevice);

  /* parity: 1 GPU run vs CPU */
  bq1_gemv_cpu(N, K, wb, ws, xq, xs, y);
  if (bq1_gemv_sm53_launch(d_wb, d_ws, d_xq, xs, d_y, K, N, 0) != 0) {
    printf("LAUNCH FAIL: %s\n", cudaGetErrorString(cudaGetLastError())); return 1;
  }
  cudaDeviceSynchronize();
  cudaMemcpy(yg, d_y, (size_t)N * 4, cudaMemcpyDeviceToHost);
  double md = 0; int exact = 0;
  for (int i = 0; i < N; i++) {
    if (!memcmp(&y[i], &yg[i], 4)) exact++;
    double d = fabs((double)y[i] - yg[i]) / fmax(fabs((double)y[i]), 1e-6);
    if (d > md) md = d;
  }
  printf("parity: %d/%d bit-exact, maxrel %.2e %s\n",
    exact, N, md, (exact == N) ? "BIT-EXACT" : (md < 1e-6 ? "CLOSE" : "MISMATCH"));

  /* timing */
  cudaDeviceSynchronize();
  double t0 = now_s();
  for (int it = 0; it < iters; it++)
    bq1_gemv_sm53_launch(d_wb, d_ws, d_xq, xs, d_y, K, N, 0);
  cudaDeviceSynchronize();
  double t1 = now_s();
  double ms = (t1 - t0) * 1e3 / iters;
  /* bytes streamed per GEMV: bits + scales + acts(in) + y(out) */
  double bytes = (double)N * K / 8 + (double)N * K256 * 2 + K + (double)N * 4;
  double gbs = bytes / (ms * 1e-3) / 1e9;
  printf("gemv %dx%d: %.3f ms/call, %.2f GB/s (of 25.6 ceiling = %.0f%%)\n",
    N, K, ms, gbs, gbs / 25.6 * 100.0);
  /* 8B-equivalent: 36 layers x (attn+mlp ~ 8 matrices of ~4096x4096 avg)?
   * Rough: 8B binary ~1.06GB/tok (see NANO_BRINGUP). tok/s = GB/s / 1.06. */
  printf("roofline tok/s @1.06GB/tok: %.1f (target 12-15)\n", gbs / 1.06);
  printf("baseline check: stock Q1_0 ~4-6 tok/s = this kernel must be >=2x\n");
  return (exact == N) ? 0 : 3;
}
