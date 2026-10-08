/* pq2_gemv: NEON PQ2_0 x Q8_K GEMV for the 8B port (no Hadamard transform
 * on this checkpoint - plain integer dot, mirroring the fork generic).
 * Block (34B): fp16 scale + 32B qs, 2 bits/weight, 128 weights. Value =
 * ((byte >> 2j) & 3) - 1, weight (b*4+j) of each 32-group from byte b.
 * Two PQ2_0 blocks share one Q8_K block (halves), single scale each side.
 * NEON: per 8B -> 4 shifted/masked s8x8 -> double-zip to order -> 2x SDOT.
 * Validation: vs fork ggml_vec_dot_pq2_0_q8_K -> bit-exact target.
 * Build: gcc -O2 -march=armv8.2-a+dotprod -o pq2_gemv pq2_gemv.c
 *          -L$LB -lggml-base -lggml-cpu -Wl,-rpath,$LB -lm
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
#include <pthread.h>

#ifndef STANDALONE
void quantize_row_q8_K(const float * x, void * y, int64_t k);
void ggml_vec_dot_pq2_0_q8_K(int n, float * s, size_t bs, const void * vx,
                             size_t bx, const void * vy, size_t by, int nrc);
#endif

#define COLS 4096
#define ROWS 12288
static float h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu, o;
    if (e == 0) { if (!m) return 0.0f; float f = (float)m * 5.9604645e-8f; memcpy(&o, &f, 4); o |= s; }
    else if (e == 31) o = s | 0x7f800000u | (m << 13);
    else o = s | ((e + 112u) << 23) | (m << 13);
    float r; memcpy(&r, &o, 4); return r;
}

/* decode 32 weights from 8 bytes -> two int8x16 in weight order */
static inline void dec32(const uint8_t *b, int8x16_t *o0, int8x16_t *o1) {
    uint8x8_t bb = vld1_u8(b);
    uint8x8_t m3 = vdup_n_u8(3), one = vdup_n_u8(1);
    int8x8_t c0 = vreinterpret_s8_u8(vsub_u8(vand_u8(bb, m3), one));
    int8x8_t c1 = vreinterpret_s8_u8(vsub_u8(vand_u8(vshr_n_u8(bb, 2), m3), one));
    int8x8_t c2 = vreinterpret_s8_u8(vsub_u8(vand_u8(vshr_n_u8(bb, 4), m3), one));
    int8x8_t c3 = vreinterpret_s8_u8(vsub_u8(vand_u8(vshr_n_u8(bb, 6), m3), one));
    int8x8x2_t z02 = vzip_s8(c0, c2);
    int8x8x2_t z13 = vzip_s8(c1, c3);
    int8x8x2_t w0 = vzip_s8(z02.val[0], z13.val[0]);
    int8x8x2_t w1 = vzip_s8(z02.val[1], z13.val[1]);
    *o0 = vcombine_s8(w0.val[0], w0.val[1]);
    *o1 = vcombine_s8(w1.val[0], w1.val[1]);
}

void pq2_gemv_range(int r0, int r1, int cols, const uint8_t *w,
                      const float *x, float *y, const uint8_t *xq) {
    int ng = cols / 128;
    for (int r = r0; r < r1; r++) {
        const uint8_t *row = w + (size_t)r * ng * 34;
        float acc = 0.0f;
        for (int g = 0; g < ng; g++) {
/* block (34B): fp16 scale FIRST, then 32B qs (struct order) */
            const uint8_t *blk = row + g * 34;
            float ws = h2f((uint16_t)(blk[0] | (blk[1] << 8)));
            const uint8_t *bqs = blk + 2;
            /* the Q8_K half covering this group: float scale + 256 quants */
            const uint8_t *q8b = xq + (size_t)(g >> 1) * 292;
            float db;
            memcpy(&db, q8b, 4);
            const int8_t *qs = (const int8_t *)(q8b + 4 + (g & 1) * 128);
            int32x4_t a = vdupq_n_s32(0);
            for (int k = 0; k < 4; k++) {
                int8x16_t t0, t1;
                dec32(bqs + k * 8, &t0, &t1);
                a = vdotq_s32(a, t0, vld1q_s8(qs + k * 32));
                a = vdotq_s32(a, t1, vld1q_s8(qs + k * 32 + 16));
            }
            acc += (ws * db) * (float)vaddvq_s32(a);
        }
        y[r] = acc;
    }
}

/* Row-parallel dispatch (bit-exact: each row is computed by exactly one
   thread with the identical serial code above; there is no cross-row
   reduction). Raw pthreads, not OpenMP, so the STANDALONE static binary
   (glibc or bionic) needs no extra runtime. Persistent pool: threads are
   created once and parked on a condvar; per-call dispatch is one broadcast
   + one completion wait (~5-10us), vs ~30us per pthread_create which would
   cost ~30ms/token across the ~250 GEMVs of a token. Count from
   PQ2_THREADS (default 4 = one phone big cluster); 1 = pure serial. */
static struct {
    pthread_t *tid;
    int nt;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    pthread_cond_t dv;
    int gen;
    int done;
    int shutdown;
    const uint8_t *w;
    const uint8_t *xq;
    const float *x;
    float *y;
    int rows, cols;
} PQ2P = { .nt = -1 };

static void *pq2_worker(void *arg) {
    long idx = (long)arg;
    int mygen = 0;
    for (;;) {
        pthread_mutex_lock(&PQ2P.mu);
        while (PQ2P.gen == mygen && !PQ2P.shutdown)
            pthread_cond_wait(&PQ2P.cv, &PQ2P.mu);
        if (PQ2P.shutdown) { pthread_mutex_unlock(&PQ2P.mu); return NULL; }
        mygen = PQ2P.gen;
        const uint8_t *w = PQ2P.w;
        const uint8_t *xq = PQ2P.xq;
        const float *x = PQ2P.x;
        float *y = PQ2P.y;
        int rows = PQ2P.rows, cols = PQ2P.cols, nt = PQ2P.nt;
        pthread_mutex_unlock(&PQ2P.mu);
        int r0 = (int)((long)rows * idx / nt);
        int r1 = (int)((long)rows * (idx + 1) / nt);
        if (r1 > r0) pq2_gemv_range(r0, r1, cols, w, x, y, xq);
        pthread_mutex_lock(&PQ2P.mu);
        if (++PQ2P.done == nt - 1) pthread_cond_signal(&PQ2P.dv);
        pthread_mutex_unlock(&PQ2P.mu);
    }
}

static int pq2_nthreads(void) {
    if (PQ2P.nt >= 0) return PQ2P.nt;
    int nt = 4;
    const char *e = getenv("PQ2_THREADS");
    if (!e) e = getenv("OMP_NUM_THREADS");
    if (e && atoi(e) > 0) nt = atoi(e);
    if (nt < 1) nt = 1;
    if (nt > 32) nt = 32;
    PQ2P.nt = nt;
    if (nt > 1) {
        pthread_mutex_init(&PQ2P.mu, NULL);
        pthread_cond_init(&PQ2P.cv, NULL);
        pthread_cond_init(&PQ2P.dv, NULL);
        PQ2P.gen = 0; PQ2P.done = 0; PQ2P.shutdown = 0;
        PQ2P.tid = malloc((size_t)(nt - 1) * sizeof(pthread_t));
        for (long i = 1; i < nt; i++)
            pthread_create(&PQ2P.tid[i - 1], NULL, pq2_worker, (void *)i);
    }
    return nt;
}

void pq2_gemv(int rows, int cols, const uint8_t *w, const float *x, float *y,
              const uint8_t *xq) {
    int nt = pq2_nthreads();
    if (nt == 1 || rows < nt) {
        /* serial: identical to the pre-thread code (single range 0..rows) */
        pq2_gemv_range(0, rows, cols, w, x, y, xq);
        return;
    }
    pthread_mutex_lock(&PQ2P.mu);
    PQ2P.w = w; PQ2P.xq = xq; PQ2P.x = x; PQ2P.y = y;
    PQ2P.rows = rows; PQ2P.cols = cols;
    PQ2P.done = 0;
    PQ2P.gen++;
    pthread_cond_broadcast(&PQ2P.cv);
    pthread_mutex_unlock(&PQ2P.mu);
    /* main thread takes range 0 while workers take theirs */
    int r0 = 0, r1 = (int)((long)rows * 1 / nt);
    pq2_gemv_range(r0, r1, cols, w, x, y, xq);
    pthread_mutex_lock(&PQ2P.mu);
    while (PQ2P.done < nt - 1)
        pthread_cond_wait(&PQ2P.dv, &PQ2P.mu);
    pthread_mutex_unlock(&PQ2P.mu);
}

/* Q8_K block layout needed: d at [0..1], qs at [8..263] (verify at runtime) */
static float X[COLS], YA[ROWS], YB[ROWS];
static uint8_t XQ[(COLS / 256) * 292];

/* self-test harness: needs libggml, compiled out of the standalone build */
#ifndef STANDALONE
int main(int argc, char **argv) {
    const char *path = argv[1];
    long ds = atol(argv[2]), off = atol(argv[3]);
    int fd = open(path, O_RDONLY); struct stat st; fstat(fd, &st);
    uint8_t *m = mmap(0, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    const uint8_t *W = m + ds + off;
    for (int i = 0; i < COLS; i++) X[i] = ((i * 37) % 251) / 125.0f - 1.0f;
    quantize_row_q8_K(X, XQ, COLS);
    /* layout probe: quantize a delta row and find the scale position */
    {
        static float d_[256]; static uint8_t q_[292];
        for (int i = 0; i < 256; i++) d_[i] = (i == 0);
        quantize_row_q8_K(d_, q_, 256);
        float dd;
        memcpy(&dd, q_, 4);
        printf("q8k delta probe: d=%.6g (expect ~1/127)\n", dd);
    }
    int nchk = argc > 4 ? atoi(argv[4]) : 64;
    pq2_gemv(nchk, COLS, W, X, YB, XQ);
    for (int r = 0; r < nchk; r++) {
        const uint8_t *row = W + (size_t)r * (COLS / 128) * 34;
        float ref = 0;
        ggml_vec_dot_pq2_0_q8_K(COLS, &ref, 0, row, 0, XQ, 0, 1);
        YA[r] = ref;
    }
    int bitdiff = 0;
    double md = 0;
    for (int r = 0; r < nchk; r++) {
        if (memcmp(&YA[r], &YB[r], 4) != 0) bitdiff++;
        double d = fabs((double)YA[r] - (double)YB[r]) / fmax(fabs((double)YA[r]), 1e-12);
        if (d > md) md = d;
    }
    printf("pq2-neon vs fork vec_dot (%d rows): bit-diff %d, maxrel %.2e %s\n",
           nchk, bitdiff, md, bitdiff == 0 ? "BIT-EXACT" : (md < 1e-6 ? "close" : "MISMATCH"));
    printf("sample: fork %.6f  mine %.6f\n", YA[0], YB[0]);
    struct timespec a, b;
    double bt = 1e30;
    for (int it = 0; it < 3; it++) {
        clock_gettime(CLOCK_MONOTONIC, &a);
        pq2_gemv(ROWS, COLS, W, X, YB, XQ);
        clock_gettime(CLOCK_MONOTONIC, &b);
        double s = (b.tv_sec - a.tv_sec) + 1e-9 * (b.tv_nsec - a.tv_nsec);
        if (s < bt) bt = s;
    }
    printf("full 8B ffn %dx%d pq2-neon: %.1f ms (%.2f ns/w)\n",
           ROWS, COLS, bt * 1e3, bt * 1e9 / ((double)ROWS * COLS));
    return 0;
}
#endif
