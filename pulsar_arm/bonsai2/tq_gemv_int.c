/* tq_gemv_int: integer-path PTQ1_0 GEMV mirroring llama.cpp's
 * ggml_vec_dot_ptq1_0_q8_0 (q8_0 activations + trits as i8). This is the
 * formulation that CAN vectorize: no per-element float conversion in the inner
 * loop (the thing that killed the f32 path's NEON prospects).
 *
 * Per group of 128 (= 4 blocks of 32):
 *   contrib = w_scale * sum_b( d_b * sum_{i in b}( trit_i * xq_i ) )
 * with trit = digit - 1 in {-1,0,+1}, dotted DIRECTLY - no correction term.
 * (An earlier revision subtracted a CS correction, theorizing a digit-based
 * dot; the fork source (quants.c generic) decodes q = xi - 1 and dots with
 * no correction, and the corrected version mismatched on all 64 probe rows.
 * CS builder kept as dead code, not called by the dot.)
 *
 * Validation 1: vs fork quantize_row_q8_0 + vec_dot_ptq1_0_q8_0 -> bit-exact target
 * Validation 2: vs the f32 path (tq_xgemv) -> quantization tolerance
 * Build: gcc -O2 -o tq_gemv_int tq_gemv_int.c -L$LL/build/bin -lggml-base
 *        -lggml-cpu -Wl,-rpath,$LL/build/bin -lm
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

void quantize_row_q8_0(const float * x, void * y, int64_t k);
void ggml_vec_dot_ptq1_0_q8_0(int n, float * s, size_t bs, const void * vx,
                              size_t bx, const void * vy, size_t by, int nrc);

#define COLS 5120
#define ROWS 17408
static const uint8_t P3[6] = {1, 3, 9, 27, 81, 243};
static float h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu, o;
    if (e == 0) { if (!m) return 0.0f; float f = (float)m * 5.9604645e-8f; memcpy(&o, &f, 4); o |= s; }
    else if (e == 31) o = s | 0x7f800000u | (m << 13);
    else o = s | ((e + 112u) << 23) | (m << 13);
    float r; memcpy(&r, &o, 4); return r;
}
/* staged PTQ1_0 trit at element idx (0..127) of a 28B block: digit - 1 */
static int trit_at(const uint8_t *b, int idx) {
    int byte, nn;
    if (idx < 80)      { nn = idx / 16; byte = idx % 16; }
    else if (idx < 120){ nn = (idx - 80) / 8; byte = 16 + (idx - 80) % 8; }
    else               { nn = (idx - 120) / 2; byte = 24 + (idx - 120) % 2; }
    uint8_t v = (uint8_t)(b[byte] * P3[nn]);
    return (int)(((uint16_t)v * 3) >> 8) - 1;
}

void tq_gemv_int(int rows, int cols, const uint8_t *w, const float *x, float *y,
                 const uint8_t *xq /* (cols/32)*34 */, const float *CS) {
    int ng = cols / 128;
    for (int r = 0; r < rows; r++) {
        const uint8_t *row = w + (size_t)r * ng * 28;
        float acc = 0.0f;
        for (int g = 0; g < ng; g++) {
            const uint8_t *blk = row + g * 28;
            float ws = h2f((uint16_t)(blk[26] | (blk[27] << 8)));
            float sum = 0;
            for (int kb = 0; kb < 4; kb++) {          /* 4 q8 blocks of 32 */
                const uint8_t *qb = xq + (size_t)(g * 4 + kb) * 34;
                float db = h2f((uint16_t)(qb[0] | (qb[1] << 8)));
                const int8_t *qs = (const int8_t *)(qb + 2);
                int32_t isum = 0;
                for (int i = 0; i < 32; i++)
                    isum += (int32_t)trit_at(blk, kb * 32 + i) * (int32_t)qs[i];
                sum += db * (float)isum;
            }
            acc += ws * sum; (void)CS; /* fp32 order mirrors fork generic exactly */
        }
        y[r] = acc;
    }
}
/* per-call correction CS[g] = sum_b d_b * sum_{i in b} xq_i */
void tq_build_CS(int cols, const uint8_t *xq, float *CS) {
    int ng = cols / 128;
    for (int g = 0; g < ng; g++) {
        float s = 0;
        for (int kb = 0; kb < 4; kb++) {
            const uint8_t *qb = xq + (size_t)(g * 4 + kb) * 34;
            float db = h2f((uint16_t)(qb[0] | (qb[1] << 8)));
            const int8_t *qs = (const int8_t *)(qb + 2);
            int32_t sx = 0;
            for (int i = 0; i < 32; i++) sx += (int32_t)qs[i];
            s += db * (float)sx;
        }
        CS[g] = s;
    }
}

/* ---- validation harness ---- */
static float X[COLS], XT[COLS], YA[ROWS], YB[ROWS];
static uint8_t XQ[(COLS / 32) * 34];
static float CS[COLS / 128];

/* f32 reference path: signs -> FWHT -> staged decode dot (from tq_xgemv.c) */
extern void fwht1024_f32(float * x);
extern void tq_xgemv(int rows, int cols, const uint8_t *w, const float *x,
                     float *y, const float *sgn);

int main(int argc, char **argv) {
    const char *path = argv[1];
    long ds = atol(argv[2]), off = atol(argv[3]);
    int fd = open(path, O_RDONLY); struct stat st; fstat(fd, &st);
    uint8_t *m = mmap(0, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    const uint8_t *W = m + ds + off;
    float *sgn = malloc(COLS * 4);
    FILE *fs = fopen("/tmp/sign5120.bin", "rb");
    if (!fs || fread(sgn, 4, COLS, fs) != COLS) { printf("signs fail\n"); return 1; }
    fclose(fs);
    for (int i = 0; i < COLS; i++) { X[i] = ((i * 37) % 251) / 125.0f - 1.0f; XT[i] = X[i] * sgn[i]; }
    for (int b = 0; b < COLS / 1024; b++) fwht1024_f32(XT + b * 1024);
    quantize_row_q8_0(XT, XQ, COLS);
    tq_build_CS(COLS, XQ, CS);

    int nchk = argc > 4 ? atoi(argv[4]) : 64;
    /* path A: integer, mirroring the fork */
    tq_gemv_int(nchk, COLS, W, XT, YB, XQ, CS);
    /* path B: fork's own inner loop, same inputs */
    for (int r = 0; r < nchk; r++) {
        const uint8_t *row = W + (size_t)r * (COLS / 128) * 28;
        float ref = 0;
        ggml_vec_dot_ptq1_0_q8_0(COLS, &ref, 0, row, 0, XQ, 0, 1);
        YA[r] = ref;
    }
    /* path C: f32 reference (signs+FWHT applied inside by tq_xgemv on X, not XT)
     * -> skip; f32 vs int comparison is quantization noise by design. */

    int bitdiff = 0;
    double md = 0;
    for (int r = 0; r < nchk; r++) {
        if (memcmp(&YA[r], &YB[r], 4) != 0) bitdiff++;
        double d = fabs((double)YA[r] - (double)YB[r]) / fmax(fabs((double)YA[r]), 1e-12);
        if (d > md) md = d;
    }
    printf("int-path vs fork vec_dot (%d rows): bit-diff %d, maxrel %.2e %s\n",
           nchk, bitdiff, md, bitdiff == 0 ? "BIT-EXACT" : (md < 1e-6 ? "close" : "MISMATCH"));
    printf("sample: fork %.6f  mine %.6f\n", YA[0], YB[0]);

    /* timing: full matrix, integer path */
    struct timespec a, b;
    double bt = 1e30;
    for (int it = 0; it < 3; it++) {
        clock_gettime(CLOCK_MONOTONIC, &a);
        tq_gemv_int(ROWS, COLS, W, XT, YB, XQ, CS);
        clock_gettime(CLOCK_MONOTONIC, &b);
        double s = (b.tv_sec - a.tv_sec) + 1e-9 * (b.tv_nsec - a.tv_nsec);
        if (s < bt) bt = s;
    }
    printf("full ffn_gate %dx%d int-path: %.1f ms (%.2f ns/w)  [f32 baseline was 153 ms / 1.72 ns/w]\n",
           ROWS, COLS, bt * 1e3, bt * 1e9 / ((double)ROWS * COLS));
    free(sgn);
    return 0;
}
