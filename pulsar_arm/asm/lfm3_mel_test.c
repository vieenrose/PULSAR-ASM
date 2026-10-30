/* lfm3_mel_test.c -- parity gate + benchmark for lfm3_mel.S (log-mel front end).

   Gate: BIT equality against the C reference below, which mirrors the kernel's
   documented accumulation order (l: lane, ascending j: block, then the house
   tree ((a0+a1)+(a2+a3))+((a4+a5)+(a6+a7))) -- not "a" sum. Tolerance checks
   are what let 5fd7e79's truncated K=7680 kernel print YES, so nothing in this
   file compares within an epsilon; the tolerance gate against HF lives in
   .auto/lfm3/checks_mel.py and is a separate question (torch uses pocketfft +
   BLAS, so HF parity is a tolerance, not a bit pattern).

   BUILD (Nano):
     as -o lfm3_mel_o lfm3_mel.S
     gcc -O2 -ffp-contract=off -o lfm3_mel_test lfm3_mel_test.c lfm3_mel_o -lm
   -ffp-contract=off is not optional: a fused a*b+c is a different real number
   than the kernel's fmul-then-fadd, and the gate asserts equality.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

void lfm3_mel_power(const float *x, const float *win, const float *twc,
                    const float *tws, float *pw, int n_bins);
void lfm3_mel_matmul(const float *pw, const float *filt, float *mel, int n_mels);
void lfm3_mel_norm(const float *logmel, float *out, int n, float floor_v, float offset,
                   float divisor);

#define NFFT 400
#define NBIN 201
#define NBPAD 208
#define NMEL 128

static inline int biteq(float a, float b) {
    return a == b || (a == 0.0f && b == 0.0f);   /* +0 and -0 are the same number */
}

/* ---- reference: mirrors lfm3_mel_power's documented order exactly ---- */
static void ref_power(const float *x, const float *w, const float *twc, const float *tws,
                      float *pw, int nb) {
    static float y[NFFT], cp[NFFT / 2], co[NFFT / 2];
    for (int n = 0; n < NFFT; n++) y[n] = x[n] * w[n];
    for (int n = 0; n < NFFT / 2; n++) { cp[n] = y[n] + y[n + NFFT / 2]; co[n] = y[n] - y[n + NFFT / 2]; }
    for (int k = 0; k < nb; k++) {
        const float *c = (k & 1) ? co : cp;
        const float *pc = twc + (size_t)k * (NFFT / 2), *ps = tws + (size_t)k * (NFFT / 2);
        float r[8] = {0,0,0,0,0,0,0,0}, ii[8] = {0,0,0,0,0,0,0,0};
        for (int j = 0; j < NFFT / 16; j++)                     /* 25 blocks of 8 */
            for (int l = 0; l < 8; l++) { int o = j * 8 + l; r[l] += c[o] * pc[o]; ii[l] += c[o] * ps[o]; }
        float tr = ((r[0]+r[1])+(r[2]+r[3])), ti = ((r[4]+r[5])+(r[6]+r[7])), re = tr + ti;
        tr = ((ii[0]+ii[1])+(ii[2]+ii[3])); ti = ((ii[4]+ii[5])+(ii[6]+ii[7])); float im = tr + ti;
        pw[k] = re * re + im * im;
    }
}

/* ---- reference: mirrors lfm3_mel_matmul's documented order exactly ---- */
static void ref_matmul(const float *pw, const float *filt, float *mel, int nm) {
    for (int m = 0; m < nm; m++) {
        const float *f = filt + (size_t)m * NBPAD;
        float a[8] = {0,0,0,0,0,0,0,0};
        for (int j = 0; j < NBPAD / 8; j++)                     /* 26 blocks of 8 */
            for (int l = 0; l < 8; l++) { int o = j * 8 + l; a[l] += f[o] * pw[o]; }
        mel[m] = ((a[0]+a[1])+(a[2]+a[3])) + ((a[4]+a[5])+(a[6]+a[7]));
    }
}

static void ref_norm(const float *in, float *out, int n, float fl, float off, float div) {
    for (int i = 0; i < n; i++) {
        float v = in[i] < fl ? fl : in[i];                      /* clamp(min=) */
        v = v + off;
        out[i] = v / div;
    }
}

static float frand(uint64_t *s) {                               /* uniform in [-1,1) */
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (int32_t)(*s >> 33) / 2147483648.0f;
}

int main(void) {
    float *x = malloc(NFFT * sizeof(float)), *w = malloc(NFFT * sizeof(float));
    float *twc = malloc((size_t)NBIN * (NFFT / 2) * sizeof(float));
    float *tws = malloc((size_t)NBIN * (NFFT / 2) * sizeof(float));
    float *filt = malloc((size_t)NMEL * NBPAD * sizeof(float));
    float *pa = calloc(NBPAD, sizeof(float)), *pr = calloc(NBPAD, sizeof(float));
    float *ma = malloc(NMEL * sizeof(float)), *mr = malloc(NMEL * sizeof(float));
    float *na = malloc(NMEL * sizeof(float)), *nr = malloc(NMEL * sizeof(float));
    uint64_t s = 0x243F6A8885A308D3ULL;
    long bad = 0, checks = 0;

    /* realistic window and mel filters; twiddles randomised to stress lane order */
    for (int i = 0; i < NFFT; i++) w[i] = 0.5f - 0.5f * cosf(2.0f * 3.14159265358979f * i / NFFT);
    for (size_t i = 0; i < (size_t)NBIN * (NFFT / 2); i++) {
        twc[i] = frand(&s); tws[i] = frand(&s);
    }
    for (int m = 0; m < NMEL; m++)
        for (int k = 0; k < NBPAD; k++)
            filt[(size_t)m * NBPAD + k] = (k >= NBIN) ? 0.0f : (k > m * 2 && k < m * 2 + 40 ? frand(&s) * 0.1f : 0.0f);

    for (int trial = 0; trial < 64; trial++) {
        for (int i = 0; i < NFFT; i++) x[i] = frand(&s) * (trial & 1 ? 1e3f : 1.0f);
        memset(pa, 0, NBPAD * sizeof(float)); memset(pr, 0, NBPAD * sizeof(float));
        lfm3_mel_power(x, w, twc, tws, pa, NBIN);
        ref_power(x, w, twc, tws, pr, NBIN);
        for (int k = 0; k < NBIN; k++) { checks++; if (!biteq(pa[k], pr[k])) { if (bad < 5) printf("POWER k=%d asm %.9g ref %.9g\n", k, pa[k], pr[k]); bad++; } }
        if (memcmp(pa + NBIN, pr + NBIN, (NBPAD - NBIN) * sizeof(float)) && !biteq(pa[NBIN], pr[NBIN])) bad++;

        lfm3_mel_matmul(pa, filt, ma, NMEL);
        ref_matmul(pa, filt, mr, NMEL);
        for (int m = 0; m < NMEL; m++) { checks++; if (!biteq(ma[m], mr[m])) { if (bad < 5) printf("MEL m=%d asm %.9g ref %.9g\n", m, ma[m], mr[m]); bad++; } }

        for (int i = 0; i < NMEL; i++) ma[i] = (i % 7 == 3) ? -1e9f : (i % 11 == 5 ? 0.0f : log10f(fabsf(ma[i]) + 1.0f));
        lfm3_mel_norm(ma, na, NMEL, -12.0f, 4.0f, 4.0f);
        ref_norm(ma, nr, NMEL, -12.0f, 4.0f, 4.0f);
        for (int i = 0; i < NMEL; i++) { checks++; if (!biteq(na[i], nr[i])) { if (bad < 5) printf("NORM i=%d asm %.9g ref %.9g\n", i, na[i], nr[i]); bad++; } }
    }

    /* the -inf case the silence frame produces */
    for (int i = 0; i < NMEL; i++) ma[i] = -INFINITY;
    lfm3_mel_norm(ma, na, NMEL, -12.0f, 4.0f, 4.0f);
    ref_norm(ma, nr, NMEL, -12.0f, 4.0f, 4.0f);
    for (int i = 0; i < NMEL; i++) { checks++; if (!biteq(na[i], nr[i])) { printf("NORM-INF i=%d asm %.9g ref %.9g\n", i, na[i], nr[i]); bad++; } }

    /* odd lengths through the norm tail */
    float ain[8], ao[8], bo[8];
    for (int n = 1; n <= 8; n++) {
        for (int i = 0; i < 8; i++) ain[i] = frand(&s) * 4.0f;
        lfm3_mel_norm(ain, ao, n, -12.0f, 4.0f, 4.0f);
        ref_norm(ain, bo, n, -12.0f, 4.0f, 4.0f);
        for (int i = 0; i < n; i++) { checks++; if (!biteq(ao[i], bo[i])) { printf("NORM-TAIL n=%d i=%d\n", n, i); bad++; } }
    }

    printf("parity: %ld checks, %ld mismatches -> %s\n", checks, bad, bad ? "NO" : "YES");

    /* bench: frames/s on 2000 frames of random audio */
    for (int i = 0; i < NFFT; i++) x[i] = frand(&s);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int f = 0; f < 2000; f++) {
        lfm3_mel_power(x, w, twc, tws, pa, NBIN);
        lfm3_mel_matmul(pa, filt, ma, NMEL);
        for (int i = 0; i < NMEL; i++) ma[i] = log10f(fabsf(ma[i]) + 1.0f);
        lfm3_mel_norm(ma, na, NMEL, -12.0f, 4.0f, 4.0f);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("bench: 2000 frames in %.1f ms = %.3f ms/frame (20 s of audio = 2000 frames)\n",
           ms, ms / 2000.0);
    return bad != 0;
}
