/* pq1t_parity.c — PQ1_TERN gate: the ternary hs kernel (pq1_tmx_launch on a
 * -DPQ1_TERN build) against the host reference pq1_gemv_cpu, bit for bit, on
 * random PQ2_0 rows (codes 0..3, so the +2 weight is exercised too) and random
 * Q8_K activations, through the device plane prep the engine uses. Also times
 * the kernel (cudaEvent, 50 launches) for tuning.
 *   ./pq1t_parity [rows K]...      default: the Ternary-Bonsai-4B decode shapes */
#include "pq1_wrap.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rs = 12345u;
static uint32_t rnd(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }

static int probe(int rows, int K) {
  const int nb = K / 128;
  uint8_t *q2 = malloc((size_t)rows * nb * 34);
  for (size_t i = 0; i < (size_t)rows * nb; i++) {
    uint8_t *b = q2 + i * 34;
    uint16_t h = (uint16_t)(0x1c00 + (rnd() % 0x1000));   /* ~2^-8 .. 2^-6 */
    memcpy(b, &h, 2);
    for (int j = 0; j < 32; j++) b[2 + j] = (uint8_t)rnd();
  }
  uint8_t *xq = calloc((size_t)(K / 256), 292);
  for (int b = 0; b < K / 256; b++) {
    float d = 0.001f + (float)(rnd() % 1000) * 1e-5f;
    memcpy(xq + (size_t)b * 292, &d, 4);
    for (int i = 0; i < 256; i++) xq[(size_t)b * 292 + 4 + i] = (uint8_t)(int8_t)((int)(rnd() % 255) - 127);
  }
  uint32_t *wb = malloc(PQ1_RW(K) * 4 * rows);
  uint16_t *sc = malloc((size_t)rows * nb * 2);
  pq1_split_repack(q2, rows, K, wb, sc);
  float *yr = malloc((size_t)rows * 4), *yd = malloc((size_t)rows * 4);
  pq1_gemv_cpu(rows, K, wb, sc, xq, yr);

  void *dc, *ds, *dx, *dy; unsigned int *dpl; int *dau;
  cudaMalloc(&dc, PQ1_RW(K) * 4 * rows); cudaMalloc(&ds, (size_t)rows * nb * 2);
  cudaMalloc(&dx, (size_t)(K / 256) * 292); cudaMalloc(&dy, (size_t)rows * 4);
  cudaMalloc((void **)&dpl, (size_t)(K / 256) * 292); cudaMalloc((void **)&dau, (size_t)nb * 4);
  cudaMemcpy(dc, wb, PQ1_RW(K) * 4 * rows, cudaMemcpyHostToDevice);
  cudaMemcpy(ds, sc, (size_t)rows * nb * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(dx, xq, (size_t)(K / 256) * 292, cudaMemcpyHostToDevice);
  cudaMemset(dy, 0xff, (size_t)rows * 4);
  int rc = pq1_tm_prep_launch(dx, dpl, dau, K, NULL);
  rc |= pq1_tmx_launch(dc, ds, dpl, dau, dy, K, rows, NULL);
  cudaDeviceSynchronize();
  cudaMemcpy(yd, dy, (size_t)rows * 4, cudaMemcpyDeviceToHost);
  int ok = 0;
  for (int r = 0; r < rows; r++) ok += !memcmp(&yr[r], &yd[r], 4);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0, 0);
  for (int i = 0; i < 50; i++) pq1_tmx_launch(dc, ds, dpl, dau, dy, K, rows, NULL);
  cudaEventRecord(e1, 0); cudaEventSynchronize(e1);
  float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
  const double us = ms * 1000.0 / 50, mb = (PQ1_RW(K) * 4.0 + nb * 2.0) * rows / 1e6;
  printf("rows=%6d K=%5d rc=%d %s %d/%d  %.1f us  %.2f GB/s\n", rows, K, rc,
         ok == rows ? "BIT-EXACT" : "MISMATCH", ok, rows, us, mb / us * 1e3);
  if (ok != rows)
    for (int r = 0, k = 0; r < rows && k < 3; r++)
      if (memcmp(&yr[r], &yd[r], 4)) { printf("  row %d ref %.9g dev %.9g\n", r, yr[r], yd[r]); k++; }
  cudaFree(dc); cudaFree(ds); cudaFree(dx); cudaFree(dy); cudaFree(dpl); cudaFree(dau);
  free(q2); free(xq); free(wb); free(sc); free(yr); free(yd);
  return ok == rows && rc == 0;
}

int main(int argc, char **argv) {
  int good = 1;
  if (argc >= 3) {
    for (int i = 1; i + 1 < argc; i += 2) good &= probe(atoi(argv[i]), atoi(argv[i + 1]));
  } else {
    static const int sh[][2] = {{6144, 2560}, {2560, 4096}, {19456, 2560}, {2560, 9728},
                                {151669, 2560}, {1001, 2560}, {37, 9728}};
    for (unsigned i = 0; i < sizeof sh / sizeof sh[0]; i++) good &= probe(sh[i][0], sh[i][1]);
  }
  printf("%s\n", good ? "PQ1T PARITY OK" : "PQ1T PARITY FAIL");
  return !good;
}
