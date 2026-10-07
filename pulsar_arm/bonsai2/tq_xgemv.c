/* tq_xgemv: Bonsai2 projection kernel = signs -> FWHT-1024 -> PTQ1_0 dot.
 * The composition every ternary GEMV in Bonsai 2 runs through (fork build_lora_mm).
 * Reference path = scalar FWHT + scalar staged decode + plain dot;
 * test path     = asm FWHT + same decode. Decoder itself is separately proven
 * bit-exact vs the fork (tq_probe). Correctness on real rows + timing.
 *
 * build: gcc -O2 -o tq_xgemv tq_xgemv.c fwht.S -lm
 */
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
void fwht1024_f32(float * x);
static const uint8_t P3[6] = {1, 3, 9, 27, 81, 243};
static float h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu, o;
    if (e == 0) { if (!m) return 0.0f; float f = (float)m * 5.9604645e-8f; memcpy(&o, &f, 4); o |= s; }
    else if (e == 31) o = s | 0x7f800000u | (m << 13);
    else o = s | ((e + 112u) << 23) | (m << 13);
    float r; memcpy(&r, &o, 4); return r;
}
/* staged PTQ1_0 group decode -> 128 trits*scale, exactly the fork traversal */
static void dec_grp(const uint8_t *b, float *o) {
    float sc = h2f((uint16_t)(b[26] | (b[27] << 8)));
    int k = 0;
    for (int q = 0; q < 2; q++) {
        int base = q ? 16 : 0, c = q ? 8 : 16;
        for (int nn = 0; nn < 5; nn++)
            for (int m = 0; m < c; m++) {
                uint8_t v = (uint8_t)(b[base + m] * P3[nn]);
                o[k++] = (float)(((uint16_t)v * 3) >> 8) - 1.0f;
            }
    }
    for (int nn = 0; nn < 4; nn++)
        for (int h = 0; h < 2; h++) {
            uint8_t v = (uint8_t)(b[24 + h] * P3[nn]);
            o[k++] = (float)(((uint16_t)v * 3) >> 8) - 1.0f;
        }
    for (int i = 0; i < 128; i++) o[i] *= sc;
}
static void scalar_fwht(float *a, int n) {
    /* prescale like the fork ggml_fwht: dst = src * 1/sqrt(n), then butterflies.
     * Dropping this made the reference off by exactly sqrt(1024)=32. */
    for (int i = 0; i < n; i++) a[i] *= 0.03125f;
    for (int len = 1; len < n; len <<= 1)
        for (int i = 0; i < n; i += 2 * len)
            for (int j = 0; j < len; j++) {
                float u = a[i + j], v = a[i + len + j];
                a[i + j] = u + v; a[i + len + j] = u - v;
            }
}
/* reference: signs -> scalar FWHT -> dot */
static void xgemv_ref(int rows, int cols, const uint8_t *w, const float *x,
                      float *y, const float *sgn) {
    static float t[17408];
    for (int i = 0; i < cols; i++) t[i] = x[i] * sgn[i];
    for (int b = 0; b < cols / 1024; b++) scalar_fwht(t + b * 1024, 1024);
    static float G[128];
    for (int r = 0; r < rows; r++) {
        const uint8_t *row = w + (size_t)r * (cols / 128) * 28;
        double acc = 0;
        for (int g = 0; g < cols / 128; g++) {
            dec_grp(row + g * 28, G);
            for (int i = 0; i < 128; i++) acc += (double)G[i] * (double)t[g * 128 + i];
        }
        y[r] = (float)acc;
    }
}
/* test path: signs -> asm FWHT -> dot */
static void xgemv_t(int rows, int cols, const uint8_t *w, const float *x,
                    float *y, const float *sgn) {
    static float t[17408];
    for (int i = 0; i < cols; i++) t[i] = x[i] * sgn[i];
    for (int b = 0; b < cols / 1024; b++) fwht1024_f32(t + b * 1024);
    static float G[128];
    for (int r = 0; r < rows; r++) {
        const uint8_t *row = w + (size_t)r * (cols / 128) * 28;
        double acc = 0;
        for (int g = 0; g < cols / 128; g++) {
            dec_grp(row + g * 28, G);
            for (int i = 0; i < 128; i++) acc += (double)G[i] * (double)t[g * 128 + i];
        }
        y[r] = (float)acc;
    }
}
#define COLS 5120
#define ROWS 17408
static float X[COLS], YA[ROWS], YB[ROWS];
#ifndef TQ_XGEMV_LIB
int main(int argc, char **argv) {
    int fd = open(argv[1], O_RDONLY); struct stat st; fstat(fd, &st);
    uint8_t *m = mmap(0, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    long ds = atol(argv[2]), gate = atol(argv[3]);
    const uint8_t *W = m + ds + gate;
    static float sgn[COLS];
    FILE *fs = fopen("/tmp/sign5120.bin", "rb");
    if (!fs || fread(sgn, 4, COLS, fs) != COLS) { printf("signs load fail\n"); return 1; }
    fclose(fs);
    for (int i = 0; i < COLS; i++) X[i] = ((i * 37) % 251) / 125.0f - 1.0f;
    /* correctness: 64 rows */
    int nchk = argc > 4 ? atoi(argv[4]) : 64;
    xgemv_ref(nchk, COLS, W, X, YA, sgn);
    xgemv_t(nchk, COLS, W, X, YB, sgn);
    double mx = 0, md = 0; size_t wi = 0;
    for (int r = 0; r < nchk; r++) {
        double d = fabs((double)YA[r] - (double)YB[r]) / fmax(1e-9, fabs((double)YA[r]));
        if (fabs(YA[r]) > mx) mx = fabs(YA[r]);
        if (d > md) { md = d; wi = r; }
    }
    printf("correctness(%d rows): max|y|=%.4f maxrel=%.3e (row %zu) %s\n",
           nchk, mx, md, wi, md < 1e-6 ? "MATCH" : "MISMATCH");
    /* timing: full matrix, both paths */
    struct timespec a, b;
    double br = 1e30, bt = 1e30;
    for (int it = 0; it < 3; it++) {
        clock_gettime(CLOCK_MONOTONIC, &a);
        xgemv_ref(ROWS, COLS, W, X, YA, sgn);
        clock_gettime(CLOCK_MONOTONIC, &b);
        double s = (b.tv_sec - a.tv_sec) + 1e-9 * (b.tv_nsec - a.tv_nsec);
        if (s < br) br = s;
        clock_gettime(CLOCK_MONOTONIC, &a);
        xgemv_t(ROWS, COLS, W, X, YB, sgn);
        clock_gettime(CLOCK_MONOTONIC, &b);
        s = (b.tv_sec - a.tv_sec) + 1e-9 * (b.tv_nsec - a.tv_nsec);
        if (s < bt) bt = s;
    }
    double n = (double)ROWS * COLS;
    printf("full ffn_gate %dx%d: ref %.1f ms (%.2f ns/w)  asm-fwht path %.1f ms (%.2f ns/w)\n",
           ROWS, COLS, br * 1e3, br * 1e9 / n, bt * 1e3, bt * 1e9 / n);
    /* full-matrix agreement: the timing loop already left YA/YB from the
     * two full passes */
    {
        double mx2 = 0, md2 = 0; size_t wi2 = 0;
        for (int r = 0; r < ROWS; r++) {
            double d = fabs((double)YA[r] - (double)YB[r]) / fmax(1e-9, fabs((double)YA[r]));
            if (fabs(YA[r]) > mx2) mx2 = fabs(YA[r]);
            if (d > md2) { md2 = d; wi2 = r; }
        }
        printf("full-matrix agreement(%d rows): max|y|=%.4f maxrel=%.3e (row %zu) %s\n",
               ROWS, mx2, md2, wi2, md2 < 1e-6 ? "MATCH" : "MISMATCH");
    }
    return 0;
}
#endif /* TQ_XGEMV_LIB */
