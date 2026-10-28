/* q8_gemv_cpu.c -- price the CPU half of a hybrid Q8 GEMV on the Nano.
 *
 * Why this exists: the CUDA GEMV runs on ONE SM (measured 6.8 GB/s kg,
 * 9.05 GB/s staged) while all 4 Cortex-A57 cores sit idle. A hybrid split --
 * some layers' Q8_0 GEMV in NEON/C on the A57s, the rest on CUDA -- is the
 * natural way past the single-SM ceiling, but it is only worth building if
 * the CPU side moves real bytes. This measures that, at LFM3's real shapes,
 * before any of it is written into the engine.
 *
 * The inner loop is written as 32 SEPARATE acc += terms in ascending index
 * order, matching the CUDA kernel's contract, so a CPU kernel added later can
 * be bit-exact against the same reference. -ffp-contract=off keeps the
 * compiler from fusing them into FMAs and changing the rounding.
 *
 * Build on the Nano:
 *   gcc -O2 -mcpu=cortex-a57 -fopenmp -ffp-contract=off -o q8_gemv_cpu q8_gemv_cpu.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define Q8_BLK 32

/* row-major int8 codes [N,K], scales [N][K/32] as f32 (the f16 round-trip is
 * applied at quantize time so the reference sees the stored value). */
static void gemv_q8(const signed char* q, const float* s, const float* x,
                    float* y, int N, int K) {
    const int nb = K >> 5;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < N; r++) {
        const signed char* qr = q + (size_t)r * K;
        const float* sr = s + (size_t)r * nb;
        float acc = 0.f;
        for (int b = 0; b < nb; b++) {
            const signed char* qb = qr + (size_t)b * Q8_BLK;
            const float* xb = x + (size_t)b * Q8_BLK;
            float p = 0.f;
            p += (float)qb[0]  * xb[0];   p += (float)qb[1]  * xb[1];
            p += (float)qb[2]  * xb[2];   p += (float)qb[3]  * xb[3];
            p += (float)qb[4]  * xb[4];   p += (float)qb[5]  * xb[5];
            p += (float)qb[6]  * xb[6];   p += (float)qb[7]  * xb[7];
            p += (float)qb[8]  * xb[8];   p += (float)qb[9]  * xb[9];
            p += (float)qb[10] * xb[10];  p += (float)qb[11] * xb[11];
            p += (float)qb[12] * xb[12];  p += (float)qb[13] * xb[13];
            p += (float)qb[14] * xb[14];  p += (float)qb[15] * xb[15];
            p += (float)qb[16] * xb[16];  p += (float)qb[17] * xb[17];
            p += (float)qb[18] * xb[18];  p += (float)qb[19] * xb[19];
            p += (float)qb[20] * xb[20];  p += (float)qb[21] * xb[21];
            p += (float)qb[22] * xb[22];  p += (float)qb[23] * xb[23];
            p += (float)qb[24] * xb[24];  p += (float)qb[25] * xb[25];
            p += (float)qb[26] * xb[26];  p += (float)qb[27] * xb[27];
            p += (float)qb[28] * xb[28];  p += (float)qb[29] * xb[29];
            p += (float)qb[30] * xb[30];  p += (float)qb[31] * xb[31];
            acc += sr[b] * p;
        }
        y[r] = acc;
    }
}

static void quant_row(const float* w, int K, signed char* q, float* s) {
    const int nb = K / Q8_BLK;
    for (int b = 0; b < nb; b++) {
        float mx = 0.f;
        for (int j = 0; j < Q8_BLK; j++) mx = fabsf(w[b * Q8_BLK + j]) > mx ? fabsf(w[b * Q8_BLK + j]) : mx;
        float sc = mx / 127.f;
        if (sc == 0.f) { memset(q + (size_t)b * Q8_BLK, 0, Q8_BLK); s[b] = 0.f; continue; }
        /* store the scale as f16 does, so CPU and GPU see the same number */
        _Float16 h = (_Float16)sc;
        float sc16 = (float)h;
        if (sc16 == 0.f) { memset(q + (size_t)b * Q8_BLK, 0, Q8_BLK); s[b] = 0.f; continue; }
        for (int j = 0; j < Q8_BLK; j++) {
            int iv = (int)lrintf(w[b * Q8_BLK + j] / sc16);
            if (iv > 127) iv = 127; if (iv < -127) iv = -127;
            q[(size_t)b * Q8_BLK + j] = (signed char)iv;
        }
        s[b] = sc16;
    }
}

static double ms_since(const struct timespec* a) {
    struct timespec b; clock_gettime(CLOCK_MONOTONIC, &b);
    return (b.tv_sec - a->tv_sec) * 1e3 + (b.tv_nsec - a->tv_nsec) / 1e6;
}

struct Shape { int N, K; const char* name; };

int main(void) {
    struct Shape sh[] = {
        {6144, 2048, "conv_in_proj"}, {2048, 2048, "conv_out/o_proj"},
        { 512, 2048, "w_k/w_v"},      {7680, 2048, "gate+up w1,w3"},
        {2048, 7680, "down w2"},
    };
    const int NS = (int)(sizeof sh / sizeof *sh);
    int maxN = 0, maxK = 0;
    for (int i = 0; i < NS; i++) { if (sh[i].N > maxN) maxN = sh[i].N; if (sh[i].K > maxK) maxK = sh[i].K; }
    size_t wmax = (size_t)maxN * maxK;
    float* hw = (float*)malloc(wmax * 4);
    signed char* hq = (signed char*)malloc(wmax);
    float* hs = (float*)malloc((size_t)maxN * (maxK / Q8_BLK) * 4);
    float* hx = (float*)malloc((size_t)maxK * 4);
    float* hy = (float*)malloc((size_t)maxN * 4);
    for (size_t i = 0; i < wmax; i++) hw[i] = (float)(((int)(i * 2654435761u) >> 8) % 4001 - 2000) / 500.f;

    printf("threads=%d\n", omp_get_max_threads());
    printf("%-20s %6s %6s %8s %9s %9s\n", "shape", "N", "K", "MB", "ms", "GB/s");
    for (int si = 0; si < NS; si++) {
        int N = sh[si].N, K = sh[si].K; const int nb = K / Q8_BLK;
        for (int r = 0; r < N; r++) quant_row(hw + (size_t)r * K, K, hq + (size_t)r * K, hs + (size_t)r * nb);
        for (int i = 0; i < K; i++) hx[i] = (float)(((int)(i * 40503u) >> 4) % 2001 - 1000) / 700.f;
        double mb = (N * (double)K + (size_t)N * nb * 4 + (size_t)K * 4 + (size_t)N * 4) / 1e6;
        int reps = 30;
        for (int w = 0; w < 5; w++) gemv_q8(hq, hs, hx, hy, N, K);
        double best = 1e30;
        for (int t = 0; t < 3; t++) {
            struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
            for (int r = 0; r < reps; r++) gemv_q8(hq, hs, hx, hy, N, K);
            double ms = ms_since(&t0) / reps;
            if (ms < best) best = ms;
        }
        printf("%-20s %6d %6d %8.1f %9.3f %9.2f\n", sh[si].name, N, K, mb, best,
               mb * 1e-3 / (best * 1e-3));
    }
    return 0;
}
