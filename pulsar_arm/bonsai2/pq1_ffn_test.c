/* pq1_ffn_test.c - loop iter 2: the device silu(g)*u of pq1_silu_q8k against
 * the host pq1_silu_mul (NEON, glibc-2.27-exact expf), bit for bit.
 *   phase 1: EVERY float g (2^32), u = 1  (v = g*sg exactly)
 *   phase 2: 2^26 random (g, u) pairs, u over the full bit range (denormals)
 * NaN results compare equal when both sides are NaN (payloads are not part of
 * the contract; a NaN activation is a broken model either way).
 * Build (on the Nano): gcc -O2 -fopenmp -mcpu=cortex-a57 -ffp-contract=off
 *   -I$CUDAINC pq1_ffn_test.c pq1_host.o pq1_swar_sm53.o -lcudart -lm */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
int pq1_silu_q8k_launch(const float *gu_d, int n, unsigned char *xq_d, float *vout_d, void *stream);
void pq1_silu_mul(float *lu, const float *lg, int n);

static long cmp(const float *a, const float *b, const float *g, const float *u, int n, long *nan) {
    long bad = 0;
    for (int i = 0; i < n; i++) {
        uint32_t x, y; memcpy(&x, a + i, 4); memcpy(&y, b + i, 4);
        if (x == y) continue;
        if (isnan(a[i]) && isnan(b[i])) { (*nan)++; continue; }
        if (bad < 8) {
            uint32_t gb; memcpy(&gb, g + i, 4);
            fprintf(stderr, "MISMATCH g=%08x (%a) u=%a dev=%08x host=%08x\n", gb, g[i], u ? u[i] : 1.0f, x, y);
        }
        bad++;
    }
    return bad;
}
static void host_silu(float *ref, const float *g, const float *u, int n) {
#pragma omp parallel for schedule(static)
    for (int j = 0; j < n; j += 65536) {
        for (int i = 0; i < 65536; i++) ref[j + i] = u ? u[j + i] : 1.0f;
        pq1_silu_mul(ref + j, g + j, 65536);
    }
}
int main(int argc, char **argv) {
    const int n = 1 << 24;
    const int c0 = argc > 1 ? atoi(argv[1]) : 0, c1 = argc > 2 ? atoi(argv[2]) : 256;
    float *hg = malloc((size_t)n * 4), *hu = malloc((size_t)n * 4);
    float *ref = malloc((size_t)n * 4), *dev = malloc((size_t)n * 4);
    float *d_gu, *d_v; unsigned char *d_xq;
    if (cudaMalloc((void **)&d_gu, (size_t)n * 8) || cudaMalloc((void **)&d_v, (size_t)n * 4) ||
        cudaMalloc((void **)&d_xq, (size_t)n / 256 * 292)) { fprintf(stderr, "cudaMalloc\n"); return 2; }
    long bad = 0, nan = 0;
    for (int i = 0; i < n; i++) hu[i] = 1.0f;
    cudaMemcpy(d_gu + n, hu, (size_t)n * 4, cudaMemcpyHostToDevice);
    for (int c = c0; c < c1; c++) {
        for (int i = 0; i < n; i++) { uint32_t b = ((uint32_t)c << 24) | (uint32_t)i; memcpy(hg + i, &b, 4); }
        cudaMemcpy(d_gu, hg, (size_t)n * 4, cudaMemcpyHostToDevice);
        if (pq1_silu_q8k_launch(d_gu, n, d_xq, d_v, NULL) || cudaDeviceSynchronize()) { fprintf(stderr, "launch\n"); return 2; }
        cudaMemcpy(dev, d_v, (size_t)n * 4, cudaMemcpyDeviceToHost);
        host_silu(ref, hg, NULL, n);
        bad += cmp(dev, ref, hg, NULL, n, &nan);
    }
    printf("phase1 g in [%d,%d)<<24 x 2^24: mismatches=%ld nan_pairs=%ld\n", c0, c1, bad, nan);
    long bad2 = 0, nan2 = 0;
    uint64_t s = 0x9e3779b97f4a7c15ULL;
    for (int rep = 0; rep < 4; rep++) {
        for (int i = 0; i < n; i++) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            uint32_t gb = (uint32_t)s, ub = (uint32_t)(s >> 32);
            float g; memcpy(&g, &gb, 4);
            if (rep & 1) g = (float)((int32_t)gb) * 0x1p-27f;      /* |g| < 16: the live range */
            memcpy(hg + i, &g, 4); memcpy(hu + i, &ub, 4);
        }
        cudaMemcpy(d_gu, hg, (size_t)n * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_gu + n, hu, (size_t)n * 4, cudaMemcpyHostToDevice);
        if (pq1_silu_q8k_launch(d_gu, n, d_xq, d_v, NULL) || cudaDeviceSynchronize()) { fprintf(stderr, "launch\n"); return 2; }
        cudaMemcpy(dev, d_v, (size_t)n * 4, cudaMemcpyDeviceToHost);
        host_silu(ref, hg, hu, n);
        bad2 += cmp(dev, ref, hg, hu, n, &nan2);
    }
    printf("phase2 random (g,u) x 2^26: mismatches=%ld nan_pairs=%ld\n", bad2, nan2);
    printf("%s\n", bad || bad2 ? "SILU FAIL" : "SILU BIT-EXACT");
    return bad || bad2;
}
