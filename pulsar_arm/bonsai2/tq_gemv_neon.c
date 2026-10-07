/* tq_gemv_neon: NEON integer-path PTQ1_0 GEMV.
 * Same formulation as tq_gemv_int (fork-faithful: trit = digit-1 dotted
 * directly, fp32 accumulation in fork order -> must stay BIT-EXACT vs the
 * fork vec_dot), but the two hot loops are vectorized:
 *   decode: 8 trits per ~7 NEON insns (widening u8 mul emulates the mod-256
 *           wrap exactly: vmull -> vmovn -> vmull -> vshrn -> sub 1)
 *   dot:    SDOT (vdotq_s32), 16 lanes per insn (needs ARMv8.2+dotprod)
 * Layout per 128-group, 4 kb chunks of 32 (same staged traversal as fork):
 *   kb0: qs[0..16)@nn0, qs[0..16)@nn1          (dec16 x2, direct)
 *   kb1: qs[0..16)@nn2, qs[0..16)@nn3          (dec16 x2, direct)
 *   kb2: qs[0..16)@nn4, qs[16..24)@nn0+nn1     (dec16 + 2x dec8 -> tmp)
 *   kb3: qs[16..24)@nn2,nn3,nn4, qh mixed-nn   (3x dec8 + 8 scalar -> tmp)
 * Build: gcc -O2 -march=armv8.2-a+dotprod -DTQ_XGEMV_LIB -o tq_gemv_neon
 *          tq_gemv_neon.c tq_xgemv.c fwht.S -L$LB -lggml-base -lggml-cpu
 *          -Wl,-rpath,$LB -lm
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
#include <arm_neon.h>

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

/* decode 8 trits from 8 bytes at one nn -> int8x8, exact scalar emulation */
static inline int8x8_t dec8(const uint8_t *b, uint8_t p3) {
    uint8x8_t bb = vld1_u8(b);
    uint16x8_t p = vmull_u8(bb, vdup_n_u8(p3));   /* full product, exact */
    uint8x8_t w = vmovn_u16(p);                    /* mod 256 wrap, exact */
    uint32x4_t t0 = vmull_u16(vget_low_u16(vmovl_u8(w)), vdup_n_u16(3));
    uint32x4_t t1 = vmull_u16(vget_high_u16(vmovl_u8(w)), vdup_n_u16(3));
    uint16x4_t s0 = vshrn_n_u32(t0, 8);
    uint16x4_t s1 = vshrn_n_u32(t1, 8);
    uint8x8_t d = vmovn_u16(vcombine_u16(s0, s1)); /* digit 0..2 */
    return vreinterpret_s8_u8(vsub_u8(d, vdup_n_u8(1)));
}

/* decode 16 trits from 16 bytes at one nn -> int8x16 */
static inline int8x16_t dec16(const uint8_t *b, uint8_t p3) {
    return vcombine_s8(dec8(b, p3), dec8(b + 8, p3));
}

/* scalar fallback for the 8 mixed-nn qh trits (negligible: 8 ops/group) */
static inline void dec_qh8(const uint8_t *qh, int8_t *o) {
    for (int nn = 0; nn < 4; nn++)
        for (int h = 0; h < 2; h++) {
            uint8_t v = (uint8_t)(qh[h] * P3[nn]);
            o[nn * 2 + h] = (int8_t)(((uint16_t)v * 3) >> 8) - 1;
        }
}

void tq_gemv_neon(int rows, int cols, const uint8_t *w, const float *x,
                  float *y, const uint8_t *xq) {
    (void)x;
    int ng = cols / 128;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        int8_t tmp[32]; /* private per thread: kb3 staging */
        const uint8_t *row = w + (size_t)r * ng * 28;
        float acc = 0.0f;
        for (int g = 0; g < ng; g++) {
            const uint8_t *blk = row + g * 28;
            float ws = h2f((uint16_t)(blk[26] | (blk[27] << 8)));
            float sum = 0.0f;
            /* kb0, kb1: pure dec16 pairs */
            for (int kb = 0; kb < 2; kb++) {
                const uint8_t *qb = xq + (size_t)(g * 4 + kb) * 34;
                float db = h2f((uint16_t)(qb[0] | (qb[1] << 8)));
                const int8_t *qs = (const int8_t *)(qb + 2);
                int8x16_t t0 = dec16(blk, P3[kb * 2]);
                int8x16_t t1 = dec16(blk, P3[kb * 2 + 1]);
                int32x4_t a = vdotq_s32(vdupq_n_s32(0), t0,
                                        vld1q_s8(qs));
                a = vdotq_s32(a, t1, vld1q_s8(qs + 16));
                sum += db * (float)vaddvq_s32(a);
            }
            /* kb2: qs[0..16)@nn4 then qs[16..24)@nn0,nn1 */
            {
                const uint8_t *qb = xq + (size_t)(g * 4 + 2) * 34;
                float db = h2f((uint16_t)(qb[0] | (qb[1] << 8)));
                const int8_t *qs = (const int8_t *)(qb + 2);
                int8x16_t t0 = dec16(blk, P3[4]);
                int8x16_t t1 = vcombine_s8(dec8(blk + 16, P3[0]),
                                            dec8(blk + 16, P3[1]));
                int32x4_t a = vdotq_s32(vdupq_n_s32(0), t0,
                                        vld1q_s8(qs));
                a = vdotq_s32(a, t1, vld1q_s8(qs + 16));
                sum += db * (float)vaddvq_s32(a);
            }
            /* kb3: qs[16..24)@nn2,nn3,nn4 then qh mixed */
            {
                const uint8_t *qb = xq + (size_t)(g * 4 + 3) * 34;
                float db = h2f((uint16_t)(qb[0] | (qb[1] << 8)));
                const int8_t *qs = (const int8_t *)(qb + 2);
                vst1_s8(tmp, dec8(blk + 16, P3[2]));
                vst1_s8(tmp + 8, dec8(blk + 16, P3[3]));
                vst1_s8(tmp + 16, dec8(blk + 16, P3[4]));
                dec_qh8(blk + 24, tmp + 24);
                int8x16_t t0 = vld1q_s8(tmp);
                int8x16_t t1 = vld1q_s8(tmp + 16);
                int32x4_t a = vdotq_s32(vdupq_n_s32(0), t0,
                                        vld1q_s8(qs));
                a = vdotq_s32(a, t1, vld1q_s8(qs + 16));
                sum += db * (float)vaddvq_s32(a);
            }
            acc += ws * sum;
        }
        y[r] = acc;
    }
}

/* ---- validation + timing harness (mirrors tq_gemv_int main) ---- */
static float X[COLS], XT[COLS], YA[ROWS], YB[ROWS];
static uint8_t XQ[(COLS / 32) * 34];

extern void fwht1024_f32(float * x);

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

    int nchk = argc > 4 ? atoi(argv[4]) : 64;
    tq_gemv_neon(nchk, COLS, W, XT, YB, XQ);
    for (int r = 0; r < nchk; r++) {
        const uint8_t *row = W + (size_t)r * (COLS / 128) * 28;
        float ref = 0;
        ggml_vec_dot_ptq1_0_q8_0(COLS, &ref, 0, row, 0, XQ, 0, 1);
        YA[r] = ref;
    }
    int bitdiff = 0;
    double md = 0;
    for (int r = 0; r < nchk; r++) {
        if (memcmp(&YA[r], &YB[r], 4) != 0) bitdiff++;
        double d = fabs((double)YA[r] - (double)YB[r]) / fmax(fabs((double)YA[r]), 1e-12);
        if (d > md) md = d;
    }
    printf("neon-path vs fork vec_dot (%d rows): bit-diff %d, maxrel %.2e %s\n",
           nchk, bitdiff, md, bitdiff == 0 ? "BIT-EXACT" : (md < 1e-6 ? "close" : "MISMATCH"));
    printf("sample: fork %.6f  mine %.6f\n", YA[0], YB[0]);

    struct timespec a, b;
    double bt = 1e30;
    for (int it = 0; it < 3; it++) {
        clock_gettime(CLOCK_MONOTONIC, &a);
        tq_gemv_neon(ROWS, COLS, W, XT, YB, XQ);
        clock_gettime(CLOCK_MONOTONIC, &b);
        double s = (b.tv_sec - a.tv_sec) + 1e-9 * (b.tv_nsec - a.tv_nsec);
        if (s < bt) bt = s;
    }
    printf("full ffn_gate %dx%d neon-path: %.1f ms (%.2f ns/w)  [scalar-int 139 ms / f32 153 ms]\n",
           ROWS, COLS, bt * 1e3, bt * 1e9 / ((double)ROWS * COLS));
    free(sgn);
    return 0;
}
