/* pq1_parity.c — Q1_0 gate: device kernel vs the fork-shaped host reference. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include "pq1_wrap.h"
#if defined(PQ1_REAL_CUDA)
#include <cuda_runtime.h>
#endif

static void mk_q8k(uint8_t *xq, int K, unsigned seed) {
  int nb = K / 256;
  srand(seed);
  for (int b = 0; b < nb; b++) {
    uint8_t *qb = xq + (size_t)b * 292;
    float d = (float)(1 + (rand() % 100)) / 100.0f;
    memcpy(qb, &d, 4);
    for (int i = 0; i < 256; i++)
      ((int8_t *)(qb + 4))[i] = (int8_t)(rand() % 255 - 127);
  }
}

int main(int argc, char **argv) {
  int rows = argc > 1 ? atoi(argv[1]) : 256;
  int K = argc > 2 ? atoi(argv[2]) : 4096;
  int nb = K / 128;
  size_t q1b = (size_t)rows * nb * 18;
  uint8_t *q1 = malloc(q1b);
  uint32_t *wb = malloc((size_t)rows * (K / 32) * 4);
  uint16_t *sc = malloc((size_t)rows * nb * 2);
  uint8_t *xq = malloc((size_t)(K / 256) * 292);
  float *yA = malloc((size_t)rows * 4), *yB = malloc((size_t)rows * 4);
  if (!q1 || !wb || !sc || !xq || !yA || !yB) { printf("oom\n"); return 2; }
  srand(31337);
  for (size_t i = 0; i < q1b; i++) q1[i] = (uint8_t)(rand() & 0xFF);
  for (int r = 0; r < rows; r++)
    for (int b = 0; b < nb; b++) {
      uint16_t h = (uint16_t)(0x2800 + (rand() % 0x1000));
      memcpy(q1 + ((size_t)r * nb + b) * 18, &h, 2);
    }
  mk_q8k(xq, K, 99);
  pq1_split_repack(q1, rows, K, wb, sc);
  pq1_gemv_cpu(rows, K, wb, sc, xq, yA);

  int repack_ok = 1;
  for (int r = 0; r < rows && repack_ok; r++)
    for (int b = 0; b < nb && repack_ok; b++) {
      const uint8_t *src = q1 + ((size_t)r * nb + b) * 18;
      if (memcmp(src + 2, (const uint8_t *)wb + ((size_t)r * (K / 32) + b * 4) * 4, 16))
        repack_ok = 0;
      if (memcmp(src, sc + (size_t)r * nb + b, 2)) repack_ok = 0;
    }
  printf("split-repack %s\n", repack_ok ? "OK" : "MISMATCH");
#if defined(PQ1_REAL_CUDA)
  uint8_t *d_wb, *d_xq;
  uint16_t *d_sc;
  float *d_y;
  cudaMalloc((void **)&d_wb, (size_t)rows * (K / 32) * 4);
  cudaMalloc((void **)&d_sc, (size_t)rows * nb * 2);
  cudaMalloc((void **)&d_xq, (size_t)(K / 256) * 292);
  cudaMalloc((void **)&d_y, (size_t)rows * 4);
  cudaMemcpy(d_wb, wb, (size_t)rows * (K / 32) * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(d_sc, sc, (size_t)rows * nb * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(d_xq, xq, (size_t)(K / 256) * 292, cudaMemcpyHostToDevice);
  int rc = pq1_gemv_sm53_launch(d_wb, d_sc, d_xq, d_y, K, rows, NULL);
  if (rc != 0) { printf("launch failed rc=%d\n", rc); return 1; }
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("sync: %s\n", cudaGetErrorString(e)); return 1; }
  cudaMemcpy(yB, d_y, (size_t)rows * 4, cudaMemcpyDeviceToHost);
  int exact = 0;
  double md = 0;
  for (int r = 0; r < rows; r++) {
    if (!memcmp(&yA[r], &yB[r], 4)) exact++;
    double d = fabs((double)yA[r] - yB[r]) / fmax(fabs((double)yA[r]), 1e-6);
    if (d > md) md = d;
  }
  printf("kernel vs host-ref: %d/%d bit-exact, maxrel %.2e %s\n", exact, rows, md,
         exact == rows ? "BIT-EXACT" : (md < 1e-5 ? "CLOSE" : "MISMATCH"));
  printf("sample: ref %.6f dev %.6f\n", yA[0], yB[0]);

  /* ---- T-MAC bit-sliced variant: same host reference, same gates ---------- */
  size_t plw = (size_t)(K / 256) * 73;            /* words, mirrors Q8_K stride */
  int ngr = K >> 7;                               /* 128-weight groups (Au) */
  uint32_t *planes = (uint32_t *)malloc(plw * 4);
  int *au_h = (int *)malloc((size_t)ngr * 4);
  float *yC = (float *)malloc((size_t)rows * 4);
  uint32_t *d_pl; int *d_au; float *d_y2;
  if (!planes || !au_h || !yC) { printf("oom\n"); return 2; }
  pq1_build_planes(xq, K, planes);
  pq1_build_au(xq, K, au_h);
  cudaMalloc((void **)&d_pl, plw * 4);
  cudaMalloc((void **)&d_au, (size_t)ngr * 4);
  cudaMalloc((void **)&d_y2, (size_t)rows * 4);
  cudaMemcpy(d_pl, planes, plw * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(d_au, au_h, (size_t)ngr * 4, cudaMemcpyHostToDevice);
  int rc2 = pq1_gemv_tm_launch(d_wb, d_sc, d_pl, d_au, d_y2, K, rows, NULL);
  if (rc2 != 0) { printf("tm launch failed rc=%d\n", rc2); return 1; }
  cudaError_t e2 = cudaDeviceSynchronize();
  if (e2 != cudaSuccess) { printf("tm sync: %s\n", cudaGetErrorString(e2)); return 1; }
  cudaMemcpy(yC, d_y2, (size_t)rows * 4, cudaMemcpyDeviceToHost);
  int exact_tm = 0;
  double mdm = 0;
  for (int r = 0; r < rows; r++) {
    if (!memcmp(&yA[r], &yC[r], 4)) exact_tm++;
    double d = fabs((double)yA[r] - (double)yC[r]) / fmax(fabs((double)yA[r]), 1e-6);
    if (d > mdm) mdm = d;
  }
  printf("tm kernel vs host-ref: %d/%d bit-exact, maxrel %.2e %s\n", exact_tm, rows,
         mdm, exact_tm == rows ? "BIT-EXACT" : (mdm < 1e-5 ? "CLOSE" : "MISMATCH"));

  /* Kernel-level A/B, interleaved in one process so board drift cancels:
   * PQ1_TIME=<iters> times N launches of each kernel alternately. Launch
   * overhead is inside both timings, so the ratio is conservative. */
  if (getenv("PQ1_TIME")) {
    int nit = atoi(getenv("PQ1_TIME"));
    /* PQ1_GAP=<us>: idle gap between launches, to reproduce the engine's
     * launch pattern (CPU work between GEMVs) instead of a hot back-to-back
     * loop. GPU devfreq cannot boost across gaps of ~ms. */
    long gap = getenv("PQ1_GAP") ? atol(getenv("PQ1_GAP")) : 0;
    for (int i = 0; i < 20; i++) { /* warm clocks and caches */
      pq1_gemv_sm53_launch(d_wb, d_sc, d_xq, d_y, K, rows, NULL);
      pq1_gemv_tm_launch(d_wb, d_sc, d_pl, d_au, d_y2, K, rows, NULL);
    }
    cudaDeviceSynchronize();
    double ta = 0, tb = 0;
    struct timespec s0, s1;
    for (int i = 0; i < nit; i++) {
      if (gap) usleep(gap);
      clock_gettime(CLOCK_MONOTONIC, &s0);
      pq1_gemv_sm53_launch(d_wb, d_sc, d_xq, d_y, K, rows, NULL);
      cudaDeviceSynchronize();
      clock_gettime(CLOCK_MONOTONIC, &s1);
      ta += (double)(s1.tv_sec - s0.tv_sec) * 1e3 + (double)(s1.tv_nsec - s0.tv_nsec) * 1e-6;
      if (gap) usleep(gap);
      clock_gettime(CLOCK_MONOTONIC, &s0);
      pq1_gemv_tm_launch(d_wb, d_sc, d_pl, d_au, d_y2, K, rows, NULL);
      cudaDeviceSynchronize();
      clock_gettime(CLOCK_MONOTONIC, &s1);
      tb += (double)(s1.tv_sec - s0.tv_sec) * 1e3 + (double)(s1.tv_nsec - s0.tv_nsec) * 1e-6;
    }
    printf("PQ1TIME iters=%d rows=%d K=%d gap=%ldus swar=%.4f ms  tm=%.4f ms  speedup=%.4fx\n",
           nit, rows, K, gap, ta / nit, tb / nit, ta / tb);
  }
  if (argc > 3 && !strcmp(argv[3], "tm"))
    return (exact == rows && exact_tm == rows && repack_ok) ? 0 : 1;
  return (exact == rows && repack_ok) ? 0 : 1;
#else
  printf("no CUDA launcher (host build)\n");
  printf("y[0]=%.6f y[%d]=%.6f\n", yA[0], rows - 1, yA[rows - 1]);
  return repack_ok ? 0 : 1;
#endif
}
