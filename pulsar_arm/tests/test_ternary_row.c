/* PULSAR-ARM | tests/test_ternary_row.c
 *
 * embed_row_b3 (asm) against ternary_row_b3 (C) against an independent fp64
 * dequantise. Self-contained on purpose: it packs its own B3_128 rows from a
 * deterministic LCG, so it needs neither the blob nor a PTGV fixture, and the
 * expectation comes from the trits it chose rather than from the format.
 *
 * The two things easy to get wrong in a row gather are the last code byte (26
 * bytes carry 130 slots but only 128 trits) and the row stride, so every case
 * uses rows > 1 with different scales per row, and at least one case uses
 * ng = 1 to sit exactly on the 3-trit boundary.
 *
 * This harness earned its keep: the first embed_row_b3 parked escale in s8,
 * which is callee-saved (d8-d15 are persistent in the AArch64 ABI), and gcc had
 * hoisted the loop-invariant (float)escale into d8 - so the C reference quietly
 * received the kernel's leftover scale. asm-vs-C bit-diff caught it immediately;
 * a comparison against the numbers alone would not have named the cause.
 *
 * Build and run on the Pi (the x86 box cannot assemble aarch64):
 *   gcc -O2 -o test_ternary_row tests/test_ternary_row.c \
 *       ../kernels/ternary_gemv.c ../asm/ternary_gemv.S -lm
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GROUP 128
#define GB    28                       /* 2-byte fp16 scale + 26 code bytes */

void embed_row_b3(float *out, const uint8_t *w, int row, int cols, float escale);
void ternary_row_b3(const uint8_t *w, int row, int cols, float escale, float *out);

static uint64_t st = 0x243F6A8885A308D3ull;
static uint32_t rnd(void) {
    st ^= st << 13; st ^= st >> 7; st ^= st << 17;
    return (uint32_t)(st >> 32);
}

/* independent fp16 -> fp64; deliberately shares no code with the reference */
static double f16_bits_to_double(uint16_t h) {
    unsigned sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1fu, man = h & 0x3ffu;
    double v;
    if (exp == 0) v = (double)man * pow(2.0, -24);
    else if (exp == 31) v = INFINITY;
    else v = pow(2.0, (int)exp - 15) * (1.0 + (double)man / 1024.0);
    return sign ? -v : v;
}

static int run_case(const char *name, int rows, int cols, double escale, double tol) {
    int ng = cols / GROUP;
    size_t n = (size_t)rows * cols;
    uint8_t *w = malloc((size_t)rows * ng * GB);
    double *want = malloc(n * sizeof(double));
    float *a = malloc(n * sizeof(float)), *c = malloc(n * sizeof(float));
    if (!w || !want || !a || !c) { printf("oom\n"); return 1; }

    for (int r = 0; r < rows; r++) {
        for (int g = 0; g < ng; g++) {
            uint16_t hb = (uint16_t)(((rnd() & 1u) << 15) |
                                     ((12u + rnd() % 8u) << 10) |
                                     (rnd() & 0x3ffu));
            double s = f16_bits_to_double(hb);
            uint8_t *blk = w + ((size_t)r * ng + g) * GB;
            blk[0] = (uint8_t)(hb & 0xff);
            blk[1] = (uint8_t)(hb >> 8);
            memset(blk + 2, 0, GB - 2);
            for (int i = 0; i < GROUP; i++) {
                unsigned u = rnd() % 10u;
                int t = u < 3 ? 0 : (u < 7 ? 1 : -1);      /* ~40 % nonzero */
                blk[2 + i / 5] += (uint8_t)((t + 1) * (unsigned[]){1, 3, 9, 27, 81}[i % 5]);
                want[(size_t)r * cols + g * GROUP + i] = (double)t * s * escale;
            }
        }
    }

    for (int r = 0; r < rows; r++) {
        embed_row_b3(a + (size_t)r * cols, w, r, cols, (float)escale);
        ternary_row_b3(w, r, cols, (float)escale, c + (size_t)r * cols);
    }

    int bitdiff = 0, bad = 0;
    double worst = 0.0;
    size_t wi = 0;
    for (size_t i = 0; i < n; i++) {
        if (memcmp(&a[i], &c[i], 4) != 0) bitdiff++;
        double d = fabs((double)a[i] - want[i]) / fmax(fabs(want[i]), 1e-9);
        if (d > worst) { worst = d; wi = i; }
        if (d > tol) bad++;
    }
    printf("%-12s rows=%2d cols=%5d ng=%2d escale=%-5g  asm-vs-C %-3d bit-diff"
           "  asm-vs-f64 rel=%.2e (idx %zu)  %s\n",
           name, rows, cols, ng, escale, bitdiff, worst, wi,
           (bitdiff || bad) ? "FAIL" : "ok");
    free(w); free(want); free(a); free(c);
    return (bitdiff || bad) ? 1 : 0;
}

void ternary_gemv_b3_v(int rows, int cols, const uint8_t *w,
                         const float *x, float *y);
void ternary_gemv_qfold(int rows, int cols, const uint8_t *w,
                        const float *x, float *y);

/* The vectorised kernel against its C mirror against fp64. Packs synthetic
 * B3_128 matrices the same way run_case packs rows (every trit chosen, last
 * byte only 3 wide), then checks the Q-fold algorithm three ways. The asm and
 * the C mirror share the math but not the summation order, so they agree to
 * ~1e-5, not bitwise; both should sit ~1e-6 from fp64. A failure here names the
 * fold, not the row gather (that is run_case's job). */
static int run_gemv_qfold(const char *name, int rows, int cols, double tol) {
    int ng = cols / GROUP;
    uint8_t *w = malloc((size_t)rows * ng * GB);
    float *x = malloc((size_t)cols * sizeof(float));
    float *ya = malloc((size_t)rows * sizeof(float));
    float *yc = malloc((size_t)rows * sizeof(float));
    double *want = malloc((size_t)rows * sizeof(double));
    double *xd = malloc((size_t)cols * sizeof(double));
    if (!w || !x || !ya || !yc || !want || !xd) { printf("oom\n"); return 1; }
    for (int i = 0; i < cols; i++) {
        xd[i] = ((double)(rnd() % 2001u) - 1000.0) / 500.0;
        x[i] = (float)xd[i];
    }
    for (int r = 0; r < rows; r++) {
        double acc = 0.0;
        for (int g = 0; g < ng; g++) {
            uint16_t hb = (uint16_t)(((rnd() & 1u) << 15) |
                                     ((12u + rnd() % 8u) << 10) | (rnd() & 0x3ffu));
            double s = f16_bits_to_double(hb);
            uint8_t *blk = w + ((size_t)r * ng + g) * GB;
            blk[0] = (uint8_t)(hb & 0xff);
            blk[1] = (uint8_t)(hb >> 8);
            memset(blk + 2, 0, GB - 2);
            for (int i = 0; i < GROUP; i++) {
                unsigned u = rnd() % 10u;
                int t = u < 3 ? 0 : (u < 7 ? 1 : -1);
                blk[2 + i / 5] += (uint8_t)((t + 1) * (unsigned[]){1, 3, 9, 27, 81}[i % 5]);
                acc += (double)t * s * xd[g * GROUP + i];
            }
        }
        want[r] = acc;
    }
    ternary_gemv_b3_v(rows, cols, w, x, ya);
    ternary_gemv_qfold(rows, cols, w, x, yc);
    int bad = 0;
    double wac = 0.0, waf = 0.0, wcf = 0.0;
    for (int r = 0; r < rows; r++) {
        double dac = fabs((double)ya[r] - (double)yc[r]) / fmax(fabs((double)yc[r]), 1e-9);
        double daf = fabs((double)ya[r] - want[r]) / fmax(fabs(want[r]), 1e-9);
        double dcf = fabs((double)yc[r] - want[r]) / fmax(fabs(want[r]), 1e-9);
        if (dac > wac) wac = dac;
        if (daf > waf) waf = daf;
        if (dcf > wcf) wcf = dcf;
        if (dac > tol || daf > tol) bad++;
    }
    printf("%-12s rows=%2d cols=%5d ng=%2d  asm-vs-C %.2e  asm-vs-f64 %.2e"
           "  C-vs-f64 %.2e  %s\n",
           name, rows, cols, ng, wac, waf, wcf, bad ? "FAIL" : "ok");
    free(w); free(x); free(ya); free(yc); free(want); free(xd);
    return bad ? 1 : 0;
}

int main(void) {
    struct { const char *name; int rows, cols; double escale; } cases[] = {
        { "one-group", 1, 128,  1.0 },      /* exactly on the 3-trit byte */
        { "two-group", 3, 256,  1.0 },
        { "hidden",    4, 2048, 1.0 },      /* 1.7B / 4B hidden width */
        { "inter",     2, 6144, 0.5 },      /* power of two: must stay exact */
        { "scaled",    2, 2048, 0.7 },      /* one rounding, still ~1e-7 */
    };
    int bad = 0;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
        bad += run_case(cases[i].name, cases[i].rows, cases[i].cols,
                        cases[i].escale, 1e-6);
    bad += run_gemv_qfold("qfold-256", 4, 256, 1e-4);
    bad += run_gemv_qfold("qfold-2048", 4, 2048, 1e-4);
    printf("%s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
