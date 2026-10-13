/* bq1_neon_parity.c — NEON vs CPU gate + threaded speed (runs on Nano). */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "bq1_wrap.h"

void bq1_gemv_neon(int rows, int cols, const uint32_t *,
                   const uint16_t *, const int8_t *, float, float *);

static double now_s(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(void) {
  int K = 4096, N = 1024;
  uint32_t *wb = malloc((size_t)N * (K >> 5) * 4);
  uint16_t *ws = malloc((size_t)N * (K >> 8) * 2);
  float *x = malloc((size_t)K * 4);
  int8_t *xq = malloc(K);
  float *y = malloc((size_t)N * 4), *yn = malloc((size_t)N * 4);
  srand(21);
  for (int i = 0; i < N * (K >> 5); i++) wb[i] = (uint32_t)rand() ^ ((uint32_t)rand() << 16);
  for (int i = 0; i < N * (K >> 8); i++) ws[i] = bq1_f2h(0.5f + (rand() % 100) / 200.0f);
  for (int i = 0; i < K; i++) x[i] = ((rand() % 2001) - 1000) / 1500.0f;
  float xs = bq1_quant_row_i8(x, xq, K);
  bq1_gemv_cpu(N, K, wb, ws, xq, xs, y);
  double t0 = now_s();
  bq1_gemv_neon(N, K, wb, ws, xq, xs, yn);
  double t1 = now_s();
  int ex = 0;
  double md = 0;
  for (int i = 0; i < N; i++) {
    if (!memcmp(&y[i], &yn[i], 4)) ex++;
    double d = fabs((double)y[i] - yn[i]) / fmax(fabs((double)y[i]), 1e-6);
    if (d > md) md = d;
  }
  printf("neon-vs-cpu: %d/%d exact, maxrel %.2e %s\n",
         ex, N, md, ex == N ? "BIT-EXACT" : "CHECK");
  double ms = (t1 - t0) * 1e3;
  double bytes = (double)N * K / 8 + N * (K >> 8) * 2 + K + (size_t)N * 4;
  printf("neon %dx%d threads=%s: %.2f ms %.2f GB/s\n", N, K,
         getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "?",
         ms, bytes / (ms * 1e-3) / 1e9);
  return ex == N ? 0 : 1;
}
