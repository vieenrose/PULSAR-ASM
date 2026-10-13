/* pq2_parity.c — exact-PQ2 gate: device kernel vs fork-faithful host ref.
 * Device buffers are cudaMalloc'd (the launcher takes device pointers).
 * PQ2_DUMP=<path> also writes rows,K,pq2[],xq[] for an independent oracle. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "pq2_wrap.h"

#if defined(PQ2_REAL_CUDA)
#include <cuda_runtime.h>
#endif

static void mk_q8k(uint8_t *xq, int K, unsigned seed) {
  int nb = K / 256;
  srand(seed);
  for (int b = 0; b < nb; b++) {
    uint8_t *qb = xq + (size_t)b * 292;
    float d = (float)(1 + (rand() % 100)) / 100.0f;
    memcpy(qb, &d, 4);
    for (int i = 0; i < 256; i++) ((int8_t *)(qb + 4))[i] = (int8_t)(rand() % 255 - 127);
  }
}

int main(int argc, char **argv) {
  int rows = argc > 1 ? atoi(argv[1]) : 256;
  int K = argc > 2 ? atoi(argv[2]) : 4096;
  int ng = K / 128;
  size_t nbytes = (size_t)rows * ng * 34;
  uint8_t *pq2 = malloc(nbytes);
  uint8_t *cd = malloc((size_t)rows * ng * 32);
  uint16_t *sc = malloc((size_t)rows * ng * 2);
  uint8_t *xq = malloc((size_t)(K / 256) * 292);
  float *yA = malloc((size_t)rows * 4), *yB = malloc((size_t)rows * 4);
  if (!pq2 || !cd || !sc || !xq || !yA || !yB) { printf("oom\n"); return 2; }

  srand(4242);
  for (size_t i = 0; i < nbytes; i++) pq2[i] = (uint8_t)(rand() & 0xFF);
  for (size_t i = 0; i < (size_t)rows * ng; i++) {
    uint16_t h = (uint16_t)(0x2800 + (rand() % 0x1000));
    memcpy(pq2 + i * 34, &h, 2);
  }
  mk_q8k(xq, K, 77);

  pq2_split_repack(pq2, rows, K, cd, sc);
  if (getenv("PQ2_DUMP")) {
    FILE *f = fopen(getenv("PQ2_DUMP"), "wb");
    int hdr[2] = {rows, K};
    fwrite(hdr, 4, 2, f);
    fwrite(pq2, 1, nbytes, f);
    fwrite(xq, 1, (size_t)(K / 256) * 292, f);
    fclose(f);
    printf("dumped %s\n", getenv("PQ2_DUMP"));
  }
  pq2_gemv_cpu(rows, K, cd, sc, xq, yA);

  int repack_ok = 1;
  for (int r = 0; r < rows && repack_ok; r++)
    for (int g = 0; g < ng && repack_ok; g++) {
      const uint8_t *src = pq2 + (size_t)r * ng * 34 + (size_t)g * 34;
      if (memcmp(src + 2, cd + ((size_t)r * ng + g) * 32, 32) != 0) repack_ok = 0;
      if (memcmp(src, sc + (size_t)r * ng + g, 2) != 0) repack_ok = 0;
    }
  printf("split-repack %s\n", repack_ok ? "OK" : "MISMATCH");

#if defined(PQ2_REAL_CUDA)
  uint8_t *d_cd = NULL, *d_xq = NULL;
  uint16_t *d_sc = NULL;
  float *d_y = NULL;
  size_t scb = (size_t)rows * ng * 2, xqb = (size_t)(K / 256) * 292;
  cudaMalloc(&d_cd, (size_t)rows * ng * 32);
  cudaMalloc(&d_sc, scb);
  cudaMalloc(&d_xq, xqb);
  cudaMalloc(&d_y, (size_t)rows * 4);
  cudaMemcpy(d_cd, cd, (size_t)rows * ng * 32, cudaMemcpyHostToDevice);
  cudaMemcpy(d_sc, sc, scb, cudaMemcpyHostToDevice);
  cudaMemcpy(d_xq, xq, xqb, cudaMemcpyHostToDevice);
  int rc = pq2_gemv_sm53_launch(d_cd, d_sc, d_xq, d_y, K, rows, NULL);
  if (rc != 0) { printf("launch failed rc=%d\n", rc); return 1; }
  cudaError_t e1 = cudaDeviceSynchronize();
  if (e1 != cudaSuccess) printf("sync: %s\n", cudaGetErrorString(e1));
  cudaError_t e2 = cudaMemcpy(yB, d_y, (size_t)rows * 4, cudaMemcpyDeviceToHost);
  if (e2 != cudaSuccess) printf("memcpyD2H: %s\n", cudaGetErrorString(e2));
  cudaError_t e3 = cudaGetLastError();
  if (e3 != cudaSuccess) printf("lastError: %s\n", cudaGetErrorString(e3));
#else
  printf("no CUDA launcher (host build): host ref self-check only\n");
  printf("y[0]=%.6f y[%d]=%.6f\n", yA[0], rows - 1, yA[rows - 1]);
  return repack_ok ? 0 : 1;
#endif

  int exact = 0;
  double md = 0;
  for (int r = 0; r < rows; r++) {
    if (!memcmp(&yA[r], &yB[r], 4)) exact++;
    double d = fabs((double)yA[r] - yB[r]) / fmax(fabs((double)yA[r]), 1e-6);
    if (d > md) md = d;
  }
  printf("kernel vs fork-ref: %d/%d bit-exact, maxrel %.2e %s\n", exact, rows, md,
         exact == rows ? "BIT-EXACT" : (md < 1e-5 ? "CLOSE" : "MISMATCH"));
  printf("sample: ref %.6f dev %.6f\n", yA[0], yB[0]);
  return (exact == rows && repack_ok) ? 0 : 1;
}
