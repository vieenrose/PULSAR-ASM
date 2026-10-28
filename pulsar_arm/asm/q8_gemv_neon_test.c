/* q8_gemv_neon_test.c -- gate + benchmark for the NEON Q8_0 GEMV on A57.
 *
 * Build and run ON THE NANO:
 *   as -o q8_gemv_neon.o q8_gemv_neon.S
 *   gcc -O2 -mcpu=cortex-a57 -fopenmp -ffp-contract=off \
 *       -o q8_gemv_neon_test q8_gemv_neon_test.c q8_gemv_neon.o -lm
 *   ./q8_gemv_neon_test          # bit-exactness gate
 *   ./q8_gemv_neon_test bench    # per-shape GB/s, 1 and 4 threads
 *
 * -ffp-contract=off is required: the reference spells its two fused steps as
 * explicit fmaf() calls, and if the compiler also contracted the plain a*b+c
 * sites the reference and the kernel would fuse different sets of operations
 * and disagree in the last bits.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define Q8_BLK 32
float q8_dot_row(const signed char* q, const float* s, const float* x, long K);

/* Mirrors the kernel's pinned reduction order operation for operation:
 * 4 accumulator groups x 4 lanes, each a 2-term ascending sequence with the
 * second term fused; fold (a0+a2)+(a1+a3) elementwise, then (C0+C2)+(C1+C3)
 * across lanes; blocks sequential with the scale on the block sum, fused. */
static float ref_row(const signed char* q, const float* s, const float* x, int K) {
    const int nb = K / Q8_BLK;
    float acc = 0.f;
    for (int b = 0; b < nb; b++) {
        const signed char* qb = q + (size_t)b * Q8_BLK;
        const float* xb = x + (size_t)b * Q8_BLK;
        float a[4][4];
        for (int g = 0; g < 4; g++)
            for (int i = 0; i < 4; i++) {
                int j0 = 4 * g + i, j1 = 16 + 4 * g + i;
                a[g][i] = fmaf((float)qb[j1], xb[j1], (float)qb[j0] * xb[j0]);
            }
        float C[4];
        for (int i = 0; i < 4; i++)
            C[i] = (a[0][i] + a[2][i]) + (a[1][i] + a[3][i]);
        float D0 = C[0] + C[2], D1 = C[1] + C[3];
        float bs = D0 + D1;
        /* UNFUSED on purpose: binutils 2.30 has no scalar fmla, so the kernel
         * does fmul then fadd here. Spelling this fmaf() made the block-scale
         * step one rounding on the reference side and two on the kernel side
         * -- a last-ulp (3e-5) disagreement at every K > 32. The two
         * per-product steps above ARE fused in the kernel (vector fmla),
         * which is why those use fmaf(). */
        acc = acc + s[b] * bs;
    }
    return acc;
}

static unsigned st = 2463534242u;
static float rnd(void) {
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    return (float)((int)(st >> 8) % 20001 - 10000) / 800.0f;
}
static signed char rndq(void) {
    st ^= st << 13; st ^= st >> 17; st ^= st << 5;
    return (signed char)((int)(st >> 8) % 255 - 127);
}

static int run(int N, int K) {
    size_t n = (size_t)N * K;
    signed char* q = malloc(n);
    float* x = malloc((size_t)K * 4);
    int nb = K / Q8_BLK;
    float* s = malloc((size_t)N * nb * 4);
    float* y = malloc((size_t)N * 4);
    for (size_t i = 0; i < n; i++) q[i] = rndq();
    for (int i = 0; i < K; i++) x[i] = rnd();
    for (int i = 0; i < N * nb; i++) s[i] = rnd() * 1e-3f;

    long bad = 0; double maxabs = 0;
    for (int r = 0; r < N; r++) {
        float a = ref_row(q + (size_t)r * K, s + (size_t)r * nb, x, K);
        float b = q8_dot_row(q + (size_t)r * K, s + (size_t)r * nb, x, K);
        if (a != b) bad++;
        double d = fabs((double)a - b); if (d > maxabs) maxabs = d;
    }
    printf("N=%-6d K=%-6d %s  maxabs %.3g  (%ld/%d)\n", N, K,
           bad ? "MISMATCH" : "BIT-EXACT", maxabs, bad, N);
    free(q); free(x); free(s); free(y);
    return bad ? 1 : 0;
}

static void gemv(const signed char* q, const float* s, const float* x,
                 float* y, int N, int K, int nthr) {
    #pragma omp parallel for num_threads(nthr) schedule(static)
    for (int r = 0; r < N; r++)
        y[r] = q8_dot_row(q + (size_t)r * K, s + (size_t)r * (K / Q8_BLK), x, K);
}

static double ms_since(const struct timespec* a) {
    struct timespec b; clock_gettime(CLOCK_MONOTONIC, &b);
    return (b.tv_sec - a->tv_sec) * 1e3 + (b.tv_nsec - a->tv_nsec) / 1e6;
}

static void bench(int N, int K, const char* name, int nthr) {
    size_t n = (size_t)N * K;
    signed char* q = malloc(n);
    float* x = malloc((size_t)K * 4);
    int nb = K / Q8_BLK;
    float* s = malloc((size_t)N * nb * 4);
    float* y = malloc((size_t)N * 4);
    for (size_t i = 0; i < n; i++) q[i] = rndq();
    for (int i = 0; i < K; i++) x[i] = rnd();
    for (int i = 0; i < N * nb; i++) s[i] = rnd() * 1e-3f;
    double mb = (N * (double)K + (size_t)N * nb * 4 + (size_t)K * 4 + (size_t)N * 4) / 1e6;
    long reps = (long)(2e8 / (double)n) + 2;   /* long: on a 32-bit
                                                 * int this overflowed */
    for (int w = 0; w < 3; w++) gemv(q, s, x, y, N, K, nthr);
    double best = 1e30;
    for (int t = 0; t < 3; t++) {
        struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int r = 0; r < reps; r++) gemv(q, s, x, y, N, K, nthr);
        double ms = ms_since(&t0) / reps;
        if (ms < best) best = ms;
    }
    printf("  %-18s T%d  N=%5d K=%4d  %8.3f ms  %6.2f GB/s\n",
           name, nthr, N, K, best, mb * 1e-3 / (best * 1e-3));
    free(q); free(x); free(s); free(y);
}

int main(int argc, char** argv) {
    struct Shape { int N, K; const char* name; };
    struct Shape sh[] = {
        {6144, 2048, "conv_in_proj"}, {2048, 2048, "conv_out/o_proj"},
        { 512, 2048, "w_k/w_v"},      {2048, 2048, "w_q"},
        {7680, 2048, "gate+up w1,w3"},{2048, 7680, "down w2"},
    };
    const int NS = (int)(sizeof sh / sizeof *sh);
    if (argc > 1 && !strcmp(argv[1], "bench")) {
        printf("threads available: %d\n", omp_get_max_threads());
        /* ./q8_gemv_neon_test bench <shape-idx> [nthr] bisects a single shape,
         * which is how the libgomp 714 GB OOM got localized. */
        if (argc > 2) {
            int i = atoi(argv[2]);
            int t = (argc > 3) ? atoi(argv[3]) : 0;
            if (t) bench(sh[i].N, sh[i].K, sh[i].name, t);
            else { bench(sh[i].N, sh[i].K, sh[i].name, 1); bench(sh[i].N, sh[i].K, sh[i].name, 4); }
            return 0;
        }
        for (int i = 0; i < NS; i++) { fflush(stdout); bench(sh[i].N, sh[i].K, sh[i].name, 1);
                                       fflush(stdout); bench(sh[i].N, sh[i].K, sh[i].name, 4); }
        return 0;
    }
    int fail = 0;
    int ks[] = {2048, 7680, 64, 96, 32, 128, 4096};
    for (int i = 0; i < (int)(sizeof ks / sizeof *ks); i++) {
        int K = ks[i], N = (K <= 128) ? 256 : 97;   /* odd N: exercises edges */
        fail |= run(N, K);
    }
    printf("%s\n", fail ? "Q8_NEON_FAIL" : "Q8_NEON_OK");
    return fail;
}
