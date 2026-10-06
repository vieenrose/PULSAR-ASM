/* PULSAR-ARM | kernels/ternary_gemv.c
 *
 * Reference implementation of the ternary GEMV for the Bonsai checkpoints
 * (GGUF Q2_0 blocks, and the lossless base-3 repack the engine ships).
 *
 * Block layout, 128 weights per group:
 *   Q2_0    34 B: fp16 scale | 32 B of 2-bit codes,  w = (code - 1) * scale
 *   B3_128  28 B: fp16 scale | 26 B of base-3 codes, 5 trits per byte,
 *                 byte = sum(c_i * 3^i), c in {0,1,2}, w = (c - 1) * scale
 *
 * Two implementations are provided so the asm port has a recipe and a
 * numerical reference:
 *   ternary_gemv_scalar  - decode one group to floats, then a 4-lane FMA loop
 *   ternary_gemv_neon    - same algorithm, 8-wide NEON FMA; the intra-group
 *                          accumulation order differs, so it agrees with the
 *                          scalar path to ~1e-5 relative rather than exactly
 * Both accumulate per group in the order the oracle uses, then acc += scale*s,
 * so the result matches the numpy oracle to fp32 rounding.
 *
 * With -DTERNARY_MAIN this becomes a standalone fixture checker:
 *   ternary_gemv <file.tv ...>
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#define GROUP 128
#define B3_CODE_BYTES 26
#define Q20_CODE_BYTES 32

static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t man = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {                 /* subnormal or zero */
        if (man == 0) return sign ? -0.0f : 0.0f;
        float v = ldexpf((float)man, -24);
        return sign ? -v : v;
    }
    if (exp == 0x1f) {              /* inf / nan */
        bits = sign | 0x7f800000u | (man << 13);
        float f;
        memcpy(&f, &bits, 4);
        return f;
    }
    bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* decode one block's codes into trits in {-1, 0, +1} */
static inline void decode_block(int fmt, const uint8_t *codes, float *trit) {
    if (fmt) {                                   /* base-3, 5 trits per byte */
        for (int k = 0; k < B3_CODE_BYTES; k++) {
            unsigned v = codes[k];
            for (int i = 0; i < 5; i++) {
                int idx = k * 5 + i;
                if (idx >= GROUP) break;
                trit[idx] = (float)((int)(v % 3u) - 1);
                v /= 3u;
            }
        }
    } else {                                     /* 2-bit codes */
        for (int k = 0; k < Q20_CODE_BYTES; k++) {
            unsigned v = codes[k];
            for (int i = 0; i < 4; i++) {
                trit[k * 4 + i] = (float)((int)((v >> (2 * i)) & 3u) - 1);
            }
        }
    }
}

void ternary_gemv_scalar(int fmt, int rows, int cols, const uint8_t *w,
                                const float *x, float *y) {
    int ng = cols / GROUP;
    int gb = fmt ? 2 + B3_CODE_BYTES : 2 + Q20_CODE_BYTES;
    float trit[GROUP];
    for (int r = 0; r < rows; r++) {
        const uint8_t *row = w + (size_t)r * ng * gb;
        float acc = 0.0f;
        for (int g = 0; g < ng; g++) {
            const uint8_t *blk = row + (size_t)g * gb;
            float scale = f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
            const float *xg = x + g * GROUP;
            decode_block(fmt, blk + 2, trit);
            float s = 0.0f;
            for (int i = 0; i < GROUP; i++) s += trit[i] * xg[i];
            acc += scale * s;
        }
        y[r] = acc;
    }
}

void ternary_gemv_neon(int fmt, int rows, int cols, const uint8_t *w,
                              const float *x, float *y) {
    int ng = cols / GROUP;
    int gb = fmt ? 2 + B3_CODE_BYTES : 2 + Q20_CODE_BYTES;
    float trit[GROUP];
    for (int r = 0; r < rows; r++) {
        const uint8_t *row = w + (size_t)r * ng * gb;
        float acc = 0.0f;
        for (int g = 0; g < ng; g++) {
            const uint8_t *blk = row + (size_t)g * gb;
            float scale = f16_to_f32((uint16_t)(blk[0] | (blk[1] << 8)));
            const float *xg = x + g * GROUP;
            decode_block(fmt, blk + 2, trit);
            float s = 0.0f;
#ifdef __ARM_NEON
            float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
            int i = 0;
            for (; i + 8 <= GROUP; i += 8) {
                a0 = vfmaq_f32(a0, vld1q_f32(trit + i), vld1q_f32(xg + i));
                a1 = vfmaq_f32(a1, vld1q_f32(trit + i + 4), vld1q_f32(xg + i + 4));
            }
            s = vaddvq_f32(a0) + vaddvq_f32(a1);
            for (; i < GROUP; i++) s += trit[i] * xg[i];
#else
            for (int i = 0; i < GROUP; i++) s += trit[i] * xg[i];
#endif
            acc += scale * s;
        }
        y[r] = acc;
    }
}

#ifdef TERNARY_MAIN
#define MAGIC 0x50544756u           /* "PTGV" */

static int check(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }
    uint64_t magic;
    uint32_t ver, fmt, rows, cols, ng, gb;
    if (fread(&magic, 8, 1, f) != 1 || fread(&ver, 4, 1, f) != 1 ||
        fread(&fmt, 4, 1, f) != 1 || fread(&rows, 4, 1, f) != 1 ||
        fread(&cols, 4, 1, f) != 1 || fread(&ng, 4, 1, f) != 1 ||
        fread(&gb, 4, 1, f) != 1) {
        fprintf(stderr, "%s: bad header\n", path);
        return 1;
    }
    if (magic != MAGIC) { fprintf(stderr, "%s: bad magic\n", path); return 1; }
    if (ng * GROUP != cols) { fprintf(stderr, "%s: cols not a group multiple\n", path); return 1; }
    float *x = malloc((size_t)cols * 4);
    uint8_t *w = malloc((size_t)rows * ng * gb);
    float *y = malloc((size_t)rows * 4);
    float *ref = malloc((size_t)rows * 4);
    if (!x || !w || !y || !ref) { fprintf(stderr, "oom\n"); return 1; }
    if (fread(x, 4, cols, f) != cols || fread(w, gb, (size_t)rows * ng, f) != (size_t)rows * ng ||
        fread(ref, 4, rows, f) != rows) {
        fprintf(stderr, "%s: short file\n", path);
        return 1;
    }
    ternary_gemv_scalar(fmt, rows, cols, w, x, y);
    double worst = 0.0;
    int wi = 0;
    for (uint32_t r = 0; r < rows; r++) {
        double d = fabs((double)y[r] - (double)ref[r]) /
                   fmax(fabs((double)ref[r]), 1e-6);
        if (d > worst) { worst = d; wi = (int)r; }
    }
    float *y2 = malloc((size_t)rows * 4);
    ternary_gemv_neon(fmt, rows, cols, w, x, y2);
    double worst2 = 0.0;
    for (uint32_t r = 0; r < rows; r++) {
        double d = fabs((double)y2[r] - (double)y[r]) / fmax(fabs((double)y[r]), 1e-6);
        if (d > worst2) worst2 = d;
    }
    printf("%-52s fmt=%-6s rows=%4u cols=%5u  scalar-vs-oracle rel=%.2e (row %d)"
           "  neon-vs-scalar rel=%.2e\n",
           path, fmt ? "B3_128" : "Q2_0", rows, cols, worst, wi, worst2);
    int ok = worst < 5e-4 && worst2 < 1e-4;   /* fp32 order differences */
    free(x); free(w); free(y); free(ref); free(y2);
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    int bad = 0;
    if (argc < 2) { fprintf(stderr, "usage: %s <file.tv ...>\n", argv[0]); return 2; }
    for (int i = 1; i < argc; i++) bad += check(argv[i]);
    printf("%s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
#endif
