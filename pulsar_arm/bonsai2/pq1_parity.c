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

#if defined(PQ1_REAL_CUDA)
/* `pq1_parity fa`: the f16 flash-attention kernels against pq1_fa16_head on
 * random q/K/V (8 kv heads x 4 query heads, ctx 4096) at several depths, bit
 * for bit; plus the exhaustive pq1_r16 check */
static int fa_parity(void) {
  const int nh = 32, nkv = 8, ctx = 4096;
  const int Ts[] = {1, 2, 31, 32, 33, 97, 512, 1500, 4096};
  float *q = malloc((size_t)nh * 128 * 4), *o = malloc((size_t)nh * 128 * 4);
  float *od = malloc((size_t)nh * 128 * 4), *tmp = malloc((size_t)ctx * 128 * 4);
  uint16_t *K = malloc((size_t)nkv * ctx * 128 * 2), *V = malloc((size_t)nkv * ctx * 128 * 2);
  unsigned char par[PQ1_APAR_BYTES];
  float *d_q, *d_sc, *d_mv, *d_o;
  unsigned short *d_k, *d_v;
  unsigned char *d_par;
  unsigned long long *d_bad, bad[2] = {0, 0};
  cudaMalloc((void **)&d_q, (size_t)nh * 512); cudaMalloc((void **)&d_o, (size_t)nh * 512);
  cudaMalloc((void **)&d_k, (size_t)nkv * ctx * 256); cudaMalloc((void **)&d_v, (size_t)nkv * ctx * 256);
  cudaMalloc((void **)&d_sc, (size_t)nh * ctx * 4); cudaMalloc((void **)&d_mv, (size_t)nh * ctx * 8);
  cudaMalloc((void **)&d_par, sizeof par); cudaMalloc((void **)&d_bad, 16);
  srand(4242);
  /* K post-norm-like magnitudes; a few large q dims so that the scores span a
   * wide range (new maxima late in the sequence, tiny vs, exact ties) */
  for (size_t i = 0; i < (size_t)nkv * ctx * 128; i++) {
    float k = ((rand() % 20001) - 10000) / 4000.0f, v = ((rand() % 20001) - 10000) / 1000.0f;
    if (i % 977 == 5) k = 0.0f;
    if (i % 1231 == 7) v = -0.0f;
    pq1_f2h_row(&k, &K[i], 1);
    pq1_f2h_row(&v, &V[i], 1);
  }
  /* repeat a K row: identical scores (s > M is strict) */
  memcpy(K + 700 * 128, K + 300 * 128, 256);
  for (int i = 0; i < nh * 128; i++) q[i] = ((rand() % 20001) - 10000) / 2500.0f * (i % 16 == 3 ? 6.0f : 1.0f);
  cudaMemcpy(d_q, q, (size_t)nh * 512, cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, K, (size_t)nkv * ctx * 256, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, V, (size_t)nkv * ctx * 256, cudaMemcpyHostToDevice);
  int allok = 1;
  for (unsigned ti = 0; ti < sizeof Ts / sizeof Ts[0]; ti++) {
    const int T = Ts[ti], npos = T - 1;
    if (getenv("PQ1_FA_T") && atoi(getenv("PQ1_FA_T")) != T) continue;
    memset(par, 0, sizeof par);
    memcpy(par + PQ1_APAR_NPOS, &npos, 4);
    cudaMemcpy(d_par, par, sizeof par, cudaMemcpyHostToDevice);
    cudaMemset(d_o, 0xff, (size_t)nh * 512);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int rc = pq1_fa_launch(d_q, d_k, d_v, d_sc, d_mv, d_par, d_o, nh, nkv, ctx, 0);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    cudaMemcpy(od, d_o, (size_t)nh * 512, cudaMemcpyDeviceToHost);
    int badn = 0, first = -1;
    for (int h = 0; h < nh; h++) {
      pq1_fa16_head(q + h * 128, K + (size_t)(h / 4) * ctx * 128, V + (size_t)(h / 4) * ctx * 128, T,
                    0.0883883476f, o + h * 128);
      for (int d = 0; d < 128; d++)
        if (memcmp(&o[h * 128 + d], &od[h * 128 + d], 4)) { if (first < 0) first = h * 128 + d; badn++; }
    }
    printf("fa T=%4d rc=%d bad=%d/%d %s  (%.3f ms)", T, rc, badn, nh * 128, (rc == 0 && !badn) ? "OK" : "MISMATCH",
           (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) * 1e-6);
    if (first >= 0) printf("  first [%d] host %.9g dev %.9g", first, o[first], od[first]);
    printf("\n");
    if (rc || badn) allok = 0;
    if (getenv("PQ1_FA_TIME")) {   /* per-kernel time, 20 reps each */
      cudaEvent_t e0, e1;
      cudaEventCreate(&e0); cudaEventCreate(&e1);
      float ms3[3];
      for (int k = 0; k < 3; k++) {
        pq1_fa_mask = 1 << k;
        cudaEventRecord(e0, 0);
        for (int r = 0; r < 20; r++) pq1_fa_launch(d_q, d_k, d_v, d_sc, d_mv, d_par, d_o, nh, nkv, ctx, 0);
        cudaEventRecord(e1, 0);
        cudaEventSynchronize(e1);
        cudaEventElapsedTime(&ms3[k], e0, e1);
        ms3[k] /= 20;
      }
      pq1_fa_mask = 7;
      printf("   per layer: qk %.3f  sm %.3f  pv %.3f ms\n", ms3[0], ms3[1], ms3[2]);
    }
  }
  cudaMemset(d_bad, 0, 16);
  int rc = pq1_r16_test_launch(d_bad);
  cudaMemcpy(bad, d_bad, 16, cudaMemcpyDeviceToHost);
  printf("r16 all 2^32 inputs: rc=%d bad=%llu%s first=0x%08llx\n", rc, bad[0], bad[0] ? " MISMATCH" : " OK", bad[1]);
  if (rc || bad[0]) allok = 0;
  (void)tmp;
  printf("FA PARITY: %s\n", allok ? "PASS" : "FAIL");
  return allok ? 0 : 1;
}
/* `pq1_parity rms`: pq1_rms_q8k against fwd8's chain on the host (X += R,
 * rms(): serial double sum, sc = 1/sqrtf(float(s/n) + 1e-6f), y = (x*sc)*w,
 * then q8_K_quant with fwd8's fused magic add), byte for byte on X and on the
 * d + qs bytes of every Q8_K block. Each vector runs with the certified tree sum
 * and with the forced serial chain; a third of them are steered so that s/n
 * lands within 2^-38..2^-46 (relative) of a float rounding midpoint, where the
 * tree sum alone cannot decide the float and must fall back. */
static double rms_ser(const float *x, int n) {
  double s = 0;
  for (int i = 0; i < n; i++) s = fma((double)x[i], (double)x[i], s);
  return s;
}
static void rms_ref(float *x, const float *r, const float *w, int n, uint8_t *xq) {
  float y[4096];
  if (r) for (int i = 0; i < n; i++) x[i] = x[i] + r[i];
  const double s = rms_ser(x, n);
  const float sc = 1.0f / sqrtf((float)(s / n) + 1e-6f);
  for (int i = 0; i < n; i++) y[i] = (x[i] * sc) * w[i];
  for (int b = 0; b < n / 256; b++) {
    const float *yb = y + b * 256;
    uint8_t *qb = xq + (size_t)b * 292;
    float amax = 0, mx = 0;
    for (int j = 0; j < 256; j++) if (fabsf(yb[j]) > amax) { amax = fabsf(yb[j]); mx = yb[j]; }
    if (!amax) { memset(qb, 0, 260); continue; }
    const float iscale = -127.f / mx;
    for (int j = 0; j < 256; j++) {
      const float v = fmaf(yb[j], iscale, 12582912.f);
      int q;
      memcpy(&q, &v, 4);
      q = (q & 0x007fffff) - 0x00400000;
      ((int8_t *)(qb + 4))[j] = (int8_t)(q < 127 ? q : 127);
    }
    const float d = 1.0f / iscale;
    memcpy(qb, &d, 4);
  }
}
static float frand(void) { return ((rand() % 2000001) - 1000000) / 1000000.0f; }
/* steer the serial sum of x to T*(1+delta), T = n * (a float midpoint near s/n),
 * by choosing the last two elements (both small); returns 0 if it gave up */
static int steer(float *x, int n, double delta) {
  x[n - 2] = 0.0f; x[n - 1] = 0.0f;
  const double s0 = rms_ser(x, n);
  float f = (float)(s0 / n);
  const float f2 = nextafterf(f, INFINITY);
  const double T = ((double)f + (double)f2) * 0.5 * n * (1.0 + delta);
  double P = rms_ser(x, n - 2);
  float a = (float)sqrt(T - P);
  for (int it = 0; it < 64; it++) {
    const double P2 = P + (double)a * a;
    const double D2 = T - P2;
    if (D2 >= 0) {
      x[n - 2] = a;
      x[n - 1] = (float)sqrt(D2);
      return 1;
    }
    a = nextafterf(a, 0.0f);
  }
  return 0;
}
static int rms_parity(void) {
  const int n = 4096, NV = 600;
  float *X = malloc(n * 4), *R = malloc(n * 4), *W = malloc(n * 4), *Xh = malloc(n * 4), *Xd = malloc(n * 4);
  const size_t qb = (size_t)(n / 256) * 292;
  uint8_t *qh = malloc(qb), *qd = malloc(qb);
  float *d_X, *d_R, *d_W;
  uint8_t *d_q;
  int *d_nfb, nfb[2] = {0, 0};
  cudaMalloc((void **)&d_X, n * 4); cudaMalloc((void **)&d_R, n * 4); cudaMalloc((void **)&d_W, n * 4);
  cudaMalloc((void **)&d_q, qb); cudaMalloc((void **)&d_nfb, 8);
  cudaMemset(d_nfb, 0, 8);
  srand(777);
  int bad = 0, steered = 0, nres = 0;
  for (int v = 0; v < NV; v++) {
    const int kind = v % 6;
    const float sx = powf(2.0f, (float)(rand() % 24) - 12.0f);
    for (int i = 0; i < n; i++) {
      X[i] = frand() * sx * (i % 97 == 13 ? 60.0f : 1.0f);
      R[i] = frand() * sx;
      W[i] = 0.25f + 2.0f * fabsf(frand());
    }
    if (kind == 1) for (int i = 512; i < 768; i++) R[i] = -X[i];          /* a zero Q8_K block */
    if (kind == 2) { X[1030] = -1e4f * sx; X[1100] = 1e4f * sx; R[1030] = R[1100] = 0; } /* amax tie */
    if (kind == 3) for (int i = 0; i < 256; i++) { X[i] *= 1e-38f; R[i] *= 1e-38f; } /* subnormals */
    const int useR = (kind != 4);
    if (v % 3 == 0) {   /* steer x = X + R onto a midpoint: fold R into X first */
      for (int i = 0; i < n; i++) { X[i] = X[i] + R[i]; R[i] = 0.0f; }
      const double e = ldexp(1.0 + (rand() % 1000) / 1000.0, -(38 + rand() % 9)) * (rand() & 1 ? 1 : -1);
      steered += steer(X, n, e);
    }
    nres += useR;
    for (int mode = 0; mode < 2; mode++) {
      memcpy(Xh, X, n * 4);
      rms_ref(Xh, useR ? R : NULL, W, n, qh);
      cudaMemcpy(d_X, X, n * 4, cudaMemcpyHostToDevice);
      cudaMemcpy(d_R, R, n * 4, cudaMemcpyHostToDevice);
      cudaMemcpy(d_W, W, n * 4, cudaMemcpyHostToDevice);
      cudaMemset(d_q, 0xA5, qb);
      int rc = pq1_rms_q8k_launch(d_X, useR ? d_R : NULL, d_W, d_q, n, mode, d_nfb + mode, 0);
      cudaDeviceSynchronize();
      cudaMemcpy(Xd, d_X, n * 4, cudaMemcpyDeviceToHost);
      cudaMemcpy(qd, d_q, qb, cudaMemcpyDeviceToHost);
      int b1 = rc != 0 || memcmp(Xd, Xh, n * 4);
      for (int b = 0; b < n / 256; b++) b1 |= memcmp(qd + b * 292, qh + b * 292, 260) != 0;
      if (b1 && bad < 5) {
        for (int i = 0; i < n; i++)
          if (memcmp(&Xd[i], &Xh[i], 4)) { printf("  X[%d] host %08x dev %08x\n", i, *(unsigned *)&Xh[i], *(unsigned *)&Xd[i]); break; }
        for (size_t j = 0; j < qb; j++)
          if (j % 292 < 260 && qd[j] != qh[j]) { printf("  q byte %zu (blk %zu off %zu) host %02x dev %02x d host %08x dev %08x\n", j, j / 292, j % 292, qh[j], qd[j], *(unsigned *)(qh + j / 292 * 292), *(unsigned *)(qd + j / 292 * 292)); break; }
      }
      if (b1 && bad++ < 5) printf("rms vec %d kind %d mode %d MISMATCH rc=%d\n", v, kind, mode, rc);
    }
  }
  cudaMemcpy(nfb, d_nfb, 8, cudaMemcpyDeviceToHost);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0, 0);
  for (int r = 0; r < 200; r++) pq1_rms_q8k_launch(d_X, d_R, d_W, d_q, n, 0, NULL, 0);
  cudaEventRecord(e1, 0);
  cudaEventSynchronize(e1);
  float ms = 0;
  cudaEventElapsedTime(&ms, e0, e1);
  printf("rms_q8k n=%d: %d vectors x {tree, serial}, %d steered to a midpoint, %d with residual; "
         "tree fell back %d times (serial arm %d); bad=%d; %.2f us/call\n",
         n, NV, steered, nres, nfb[0], nfb[1], bad, ms * 1000 / 200);
  printf("RMS PARITY: %s\n", bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}
#endif

/* fpl: the quant kernels' fused plane emit against quant + pq1_tm_prep_launch,
 * byte for byte (X written back, Q8_K bytes, planes, per-group sums), on rms
 * (with and without residual), silu and plain q8k, with zero Q8_K blocks. */
static int fpl_parity(void) {
  const int NMAX = 12288, NV = 60;
  const size_t qb = (size_t)(NMAX / 256) * 292, pb = (size_t)(NMAX / 256) * 73 * 4, ab = (NMAX / 128) * 4;
  float *X = malloc(2 * NMAX * 4), *R = malloc(NMAX * 4), *W = malloc(NMAX * 4);
  uint8_t *h1 = malloc(qb + pb + ab + NMAX * 4), *h2 = malloc(qb + pb + ab + NMAX * 4);
  float *d_X, *d_R, *d_W;
  uint8_t *d_q;
  unsigned *d_pl;
  int *d_au;
  cudaMalloc((void **)&d_X, 2 * NMAX * 4); cudaMalloc((void **)&d_R, NMAX * 4);
  cudaMalloc((void **)&d_W, NMAX * 4); cudaMalloc((void **)&d_q, qb);
  cudaMalloc((void **)&d_pl, pb); cudaMalloc((void **)&d_au, ab);
  srand(4242);
  int bad = 0, ncase = 0;
  for (int v = 0; v < NV; v++) {
    const int kind = v % 4;   /* 0 rms+R, 1 rms, 2 silu, 3 q8k */
    const int n = kind < 2 ? 4096 : (v & 4 ? 12288 : 4096);
    const float sx = powf(2.0f, (float)(rand() % 24) - 12.0f);
    for (int i = 0; i < 2 * n; i++) X[i] = frand() * sx * (i % 97 == 13 ? 60.0f : 1.0f);
    for (int i = 0; i < n; i++) { R[i] = frand() * sx; W[i] = 0.25f + 2.0f * fabsf(frand()); }
    if (v % 3 == 0) for (int i = 256; i < 512; i++) { X[i] = 0.0f; R[i] = 0.0f; X[n + i] = 0.0f; }
    if (kind >= 2 && v % 5 == 2)   /* |v| ties of alternating sign: the first index decides */
      for (int i = 5, m = 0; i < n; i += 37, m++) { X[i] = 80.0f * sx * (m & 1 ? -1.0f : 1.0f); X[n + i] = 3.0f * sx * (m & 2 ? -1.0f : 1.0f); }
    for (int arm = 0; arm < 2; arm++) {
      uint8_t *h = arm ? h2 : h1;
      cudaMemcpy(d_X, X, 2 * n * 4, cudaMemcpyHostToDevice);
      cudaMemcpy(d_R, R, n * 4, cudaMemcpyHostToDevice);
      cudaMemcpy(d_W, W, n * 4, cudaMemcpyHostToDevice);
      cudaMemset(d_q, 0xA5, qb); cudaMemset(d_pl, 0x5A, pb); cudaMemset(d_au, 0x77, ab);
      int rc;
      if (!arm) {
        rc = kind < 2 ? pq1_rms_q8k_launch(d_X, kind ? NULL : d_R, d_W, d_q, n, 0, NULL, 0)
           : kind == 2 ? pq1_silu_q8k_launch(d_X, n, d_q, NULL, 0)
                       : pq1_q8k_launch(d_X, n, d_q, 0);
        rc |= pq1_tm_prep_launch(d_q, d_pl, d_au, n, 0);
      } else {
        rc = kind < 2 ? pq1_rms_q8k_pl_launch(d_X, kind ? NULL : d_R, d_W, d_q, d_pl, d_au, n, 0)
                      : pq1_silu_q8k_pl_launch(kind == 2, d_X, n, d_q, d_pl, d_au, 0);
      }
      cudaDeviceSynchronize();
      if (rc) bad++;
      cudaMemcpy(h, d_q, (size_t)(n / 256) * 292, cudaMemcpyDeviceToHost);
      cudaMemcpy(h + qb, d_pl, (size_t)(n / 256) * 73 * 4, cudaMemcpyDeviceToHost);
      cudaMemcpy(h + qb + pb, d_au, (size_t)(n / 128) * 4, cudaMemcpyDeviceToHost);
      cudaMemcpy(h + qb + pb + ab, d_X, (size_t)n * 4, cudaMemcpyDeviceToHost);
    }
    ncase++;
    const char *what[4] = {"q8k", "planes", "sums", "X"};
    const size_t off[4] = {0, qb, qb + pb, qb + pb + ab};
    const size_t len[4] = {(size_t)(n / 256) * 292, (size_t)(n / 256) * 73 * 4, (size_t)(n / 128) * 4, (size_t)n * 4};
    for (int k = 0; k < 4; k++)
      if (memcmp(h1 + off[k], h2 + off[k], len[k])) {
        if (bad < 8) printf("fpl case %d kind %d n %d: %s differ\n", v, kind, n, what[k]);
        bad++;
      }
  }
  {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    float t[6];
    for (int k = 0; k < 6; k++) {
      cudaEventRecord(e0, 0);
      for (int r = 0; r < 200; r++) {
        if (k == 0) pq1_rms_q8k_launch(d_X, d_R, d_W, d_q, 4096, 0, NULL, 0);
        if (k == 1) pq1_rms_q8k_pl_launch(d_X, d_R, d_W, d_q, d_pl, NULL, 4096, 0);
        if (k == 2) pq1_silu_q8k_launch(d_X, 12288, d_q, NULL, 0);
        if (k == 3) pq1_silu_q8k_pl_launch(1, d_X, 12288, d_q, d_pl, NULL, 0);
        if (k == 4) pq1_tm_prep_launch(d_q, d_pl, d_au, 4096, 0);
        if (k == 5) pq1_tm_prep_launch(d_q, d_pl, d_au, 12288, 0);
      }
      cudaEventRecord(e1, 0);
      cudaEventSynchronize(e1);
      cudaEventElapsedTime(&t[k], e0, e1);
    }
    printf("us/call: rms %.2f rms_pl %.2f | silu %.2f silu_pl %.2f | prep4096 %.2f prep12288 %.2f\n",
           t[0] * 5, t[1] * 5, t[2] * 5, t[3] * 5, t[4] * 5, t[5] * 5);
  }
  printf("fpl: %d cases (rms+R, rms, silu, q8k; n 4096/12288; zero blocks, amax ties), bad=%d\n", ncase, bad);
  printf("FPL PARITY: %s\n", bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}
/* exhaustive: every float through pq1_expf_ff vs pq1_expf_dev (glibc-2.27 expf) */
extern int pq1_expf_chk_launch(unsigned base, unsigned long long *c_d);
static int expf_parity(void) {
  unsigned long long *d_c, c[4];
  cudaMalloc((void **)&d_c, sizeof c);
  cudaMemset(d_c, 0, sizeof c);
  for (unsigned i = 0; i < 4096; i++) {
    if (pq1_expf_chk_launch(i << 20, d_c)) { printf("expf launch failed\n"); return 1; }
    if ((i & 63) == 63) cudaDeviceSynchronize();
  }
  int ce = (int)cudaMemcpy(c, d_c, sizeof c, cudaMemcpyDeviceToHost);
  if (ce) { printf("expf: cuda error %d\n", ce); return 1; }
  cudaFree(d_c);
  printf("expf: 2^32 inputs, mismatches %llu, certified-wrong %llu, fallbacks %llu (%llu of them fast-correct) -> %s\n",
         c[0], c[1], c[2], c[3], (c[0] || c[1]) ? "FAIL" : "PASS");
  return (c[0] || c[1]) ? 1 : 0;
}

int main(int argc, char **argv) {
#if defined(PQ1_REAL_CUDA)
  if (argc > 1 && !strcmp(argv[1], "fa")) return fa_parity();
  if (argc > 1 && !strcmp(argv[1], "rms")) return rms_parity();
  if (argc > 1 && !strcmp(argv[1], "fpl")) return fpl_parity();
  if (argc > 1 && !strcmp(argv[1], "expf")) return expf_parity();
#endif
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
  memset(planes, 0, plw * 4);   /* words 65..72 of each block are padding, never read */
  pq1_build_planes(xq, K, planes);
  pq1_build_au(xq, K, au_h);
  cudaMalloc((void **)&d_pl, plw * 4);
  cudaMalloc((void **)&d_au, (size_t)ngr * 4);
  cudaMalloc((void **)&d_y2, (size_t)rows * 4);
  cudaMemcpy(d_pl, planes, plw * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(d_au, au_h, (size_t)ngr * 4, cudaMemcpyHostToDevice);
  /* run #72: the engine builds the planes and group sums ON THE DEVICE from the
   * raw bytes in d_xq (pq1_tm_prep). They must be byte-identical to the host
   * build, and the tm gate below runs on the DEVICE-built ones. */
  {
    uint32_t *d_pl2; int *d_au2;
    uint32_t *pl2 = (uint32_t *)malloc(plw * 4);
    int *au2 = (int *)malloc((size_t)ngr * 4);
    if (!pl2 || !au2) { printf("oom\n"); return 2; }
    cudaMalloc((void **)&d_pl2, plw * 4);
    cudaMalloc((void **)&d_au2, (size_t)ngr * 4);
    cudaMemset(d_pl2, 0xA5, plw * 4);
    cudaMemset(d_au2, 0xA5, (size_t)ngr * 4);
    int rcp = pq1_tm_prep_launch(d_xq, d_pl2, d_au2, K, NULL);
    if (rcp != 0 || cudaDeviceSynchronize() != cudaSuccess) {
      printf("tm prep launch failed rc=%d: %s\n", rcp, cudaGetErrorString(cudaGetLastError()));
      return 1;
    }
    cudaMemcpy(pl2, d_pl2, plw * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(au2, d_au2, (size_t)ngr * 4, cudaMemcpyDeviceToHost);
    int okp = !memcmp(pl2, planes, plw * 4), oka = !memcmp(au2, au_h, (size_t)ngr * 4);
    if (!okp) {
      for (size_t i = 0; i < plw; i++)
        if (pl2[i] != planes[i]) {
          printf("  first plane mismatch word %zu (block %zu word %zu): dev %08x host %08x\n",
                 i, i / 73, i % 73, pl2[i], planes[i]);
          break;
        }
    }
    printf("tm device prep vs host build: planes %s, au %s\n",
           okp ? "identical" : "MISMATCH", oka ? "identical" : "MISMATCH");
    if (!okp || !oka) return 1;
    d_pl = d_pl2; d_au = d_au2;   /* the tm gate and timings use the device-built planes */
    free(pl2); free(au2);
  }
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

  /* ---- group-stationary Harley-Seal variant: same planes, same gate ------ */
  float *yH = (float *)malloc((size_t)rows * 4);
  float *d_y5 = NULL;
  if (!yH) { printf("oom\n"); return 2; }
  cudaMalloc((void **)&d_y5, (size_t)rows * 4);
  cudaMemset(d_y5, 0xFF, (size_t)rows * 4);
  int rc5 = pq1_gemv_hs_launch(d_wb, d_sc, d_pl, d_au, d_y5, K, rows, NULL);
  if (rc5 != 0) { printf("hs launch failed rc=%d\n", rc5); return 1; }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    printf("hs sync: %s\n", cudaGetErrorString(cudaGetLastError()));
    return 1;
  }
  cudaMemcpy(yH, d_y5, (size_t)rows * 4, cudaMemcpyDeviceToHost);
  int exact_hs = 0;
  double mdh = 0;
  for (int r = 0; r < rows; r++) {
    if (!memcmp(&yA[r], &yH[r], 4)) exact_hs++;
    double d = fabs((double)yA[r] - (double)yH[r]) / fmax(fabs((double)yA[r]), 1e-6);
    if (d > mdh) mdh = d;
  }
  printf("hs kernel vs host-ref: %d/%d bit-exact, maxrel %.2e %s\n", exact_hs, rows,
         mdh, exact_hs == rows ? "BIT-EXACT" : (mdh < 1e-5 ? "CLOSE" : "MISMATCH"));

  /* ---- 128-rows/block variant (run #45): same reference, same gate -------- */
  float *yD = (float *)malloc((size_t)rows * 4);
  float *d_y3 = NULL;
  if (!yD) { printf("oom\n"); return 2; }
  cudaMalloc((void **)&d_y3, (size_t)rows * 4);
  int rc3 = pq1_gemv_r128_launch(d_wb, d_sc, d_xq, d_y3, K, rows, NULL);
  if (rc3 != 0) { printf("r128 launch failed rc=%d\n", rc3); return 1; }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    printf("r128 sync: %s\n", cudaGetErrorString(cudaGetLastError()));
    return 1;
  }
  cudaMemcpy(yD, d_y3, (size_t)rows * 4, cudaMemcpyDeviceToHost);
  int exact_r = 0;
  double mdr = 0;
  for (int r = 0; r < rows; r++) {
    if (!memcmp(&yA[r], &yD[r], 4)) exact_r++;
    double d = fabs((double)yA[r] - (double)yD[r]) / fmax(fabs((double)yA[r]), 1e-6);
    if (d > mdr) mdr = d;
  }
  printf("r128 kernel vs host-ref: %d/%d bit-exact, maxrel %.2e %s\n", exact_r, rows,
         mdr, exact_r == rows ? "BIT-EXACT" : (mdr < 1e-5 ? "CLOSE" : "MISMATCH"));

  /* ---- 256-rows/block variant (run #46): same reference, same gate ------- */
  float *yE = (float *)malloc((size_t)rows * 4);
  float *d_y4 = NULL;
  if (!yE) { printf("oom\n"); return 2; }
  cudaMalloc((void **)&d_y4, (size_t)rows * 4);
  int rc4 = pq1_gemv_r256_launch(d_wb, d_sc, d_xq, d_y4, K, rows, NULL);
  if (rc4 != 0) { printf("r256 launch failed rc=%d\n", rc4); return 1; }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    printf("r256 sync: %s\n", cudaGetErrorString(cudaGetLastError()));
    return 1;
  }
  cudaMemcpy(yE, d_y4, (size_t)rows * 4, cudaMemcpyDeviceToHost);
  int exact_e = 0;
  double mde = 0;
  for (int r = 0; r < rows; r++) {
    if (!memcmp(&yA[r], &yE[r], 4)) exact_e++;
    double d = fabs((double)yA[r] - (double)yE[r]) / fmax(fabs((double)yA[r]), 1e-6);
    if (d > mde) mde = d;
  }
  printf("r256 kernel vs host-ref: %d/%d bit-exact, maxrel %.2e %s\n", exact_e, rows,
         mde, exact_e == rows ? "BIT-EXACT" : (mde < 1e-5 ? "CLOSE" : "MISMATCH"));

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
      pq1_gemv_r128_launch(d_wb, d_sc, d_xq, d_y3, K, rows, NULL);
      pq1_gemv_r256_launch(d_wb, d_sc, d_xq, d_y4, K, rows, NULL);
      pq1_gemv_hs_launch(d_wb, d_sc, d_pl, d_au, d_y5, K, rows, NULL);
    }
    cudaDeviceSynchronize();
    double ta = 0, tb = 0, tc = 0, td = 0, te = 0;
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
      pq1_gemv_r128_launch(d_wb, d_sc, d_xq, d_y3, K, rows, NULL);
      cudaDeviceSynchronize();
      clock_gettime(CLOCK_MONOTONIC, &s1);
      tc += (double)(s1.tv_sec - s0.tv_sec) * 1e3 + (double)(s1.tv_nsec - s0.tv_nsec) * 1e-6;
      if (gap) usleep(gap);
      clock_gettime(CLOCK_MONOTONIC, &s0);
      pq1_gemv_r256_launch(d_wb, d_sc, d_xq, d_y4, K, rows, NULL);
      cudaDeviceSynchronize();
      clock_gettime(CLOCK_MONOTONIC, &s1);
      td += (double)(s1.tv_sec - s0.tv_sec) * 1e3 + (double)(s1.tv_nsec - s0.tv_nsec) * 1e-6;
      if (gap) usleep(gap);
      clock_gettime(CLOCK_MONOTONIC, &s0);
      pq1_gemv_tm_launch(d_wb, d_sc, d_pl, d_au, d_y2, K, rows, NULL);
      cudaDeviceSynchronize();
      clock_gettime(CLOCK_MONOTONIC, &s1);
      tb += (double)(s1.tv_sec - s0.tv_sec) * 1e3 + (double)(s1.tv_nsec - s0.tv_nsec) * 1e-6;
      if (gap) usleep(gap);
      clock_gettime(CLOCK_MONOTONIC, &s0);
      pq1_gemv_hs_launch(d_wb, d_sc, d_pl, d_au, d_y5, K, rows, NULL);
      cudaDeviceSynchronize();
      clock_gettime(CLOCK_MONOTONIC, &s1);
      te += (double)(s1.tv_sec - s0.tv_sec) * 1e3 + (double)(s1.tv_nsec - s0.tv_nsec) * 1e-6;
    }
    printf("PQ1TIME iters=%d rows=%d K=%d gap=%ldus swar=%.4f ms  r128=%.4f ms (%.4fx)  r256=%.4f ms (%.4fx)  tm=%.4f ms  hs=%.4f ms (tm/hs %.4fx)\n",
           nit, rows, K, gap, ta / nit, tc / nit, ta / tc, td / nit, ta / td, tb / nit, te / nit, tb / te);
  }
  if (argc > 3 && !strcmp(argv[3], "all"))
    return (exact == rows && exact_tm == rows && exact_r == rows &&
            exact_e == rows && exact_hs == rows && repack_ok) ? 0 : 1;
  if (argc > 3 && !strcmp(argv[3], "hs"))
    return (exact == rows && exact_hs == rows && repack_ok) ? 0 : 1;
  if (argc > 3 && !strcmp(argv[3], "tm"))
    return (exact == rows && exact_tm == rows && repack_ok) ? 0 : 1;
  if (argc > 3 && !strcmp(argv[3], "r128"))
    return (exact == rows && exact_r == rows && repack_ok) ? 0 : 1;
  if (argc > 3 && !strcmp(argv[3], "r256"))
    return (exact == rows && exact_e == rows && repack_ok) ? 0 : 1;
  return (exact == rows && repack_ok) ? 0 : 1;
#else
  printf("no CUDA launcher (host build)\n");
  printf("y[0]=%.6f y[%d]=%.6f\n", yA[0], rows - 1, yA[rows - 1]);
  return repack_ok ? 0 : 1;
#endif
}
