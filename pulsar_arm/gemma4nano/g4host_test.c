/* g4host_test.c - g4_host.S against g4ref.c, bit for bit, on the Nano.
 * Build: gcc -O2 -ffp-contract=off -o g4host_test g4host_test.c g4_host.S -lm
 */
#define G4REF_NO_MAIN
#include "g4ref.c"

#include "g4_hostglue.h"

static uint64_t rs = 88172645463325252ull;
static uint64_t xr(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static double ur(void) { return (double)(xr() >> 11) * 0x1p-53; }

static long bad;
static void cmpd(const char *what, double a, double b, double in) {
    if (memcmp(&a, &b, 8)) { if (bad < 10) printf("  %s in %.17g: asm %.17g ref %.17g\n", what, in, a, b); bad++; }
}

int main(void) {
    build_tab();
    /* --- exp: 16M random in [-800, 800], dense sweeps of the edges, specials --- */
    enum { N = 1 << 16 };
    static double x[N], y[N];
    long nexp = 0;
    for (int rep = 0; rep < 256; rep++) {
        for (int i = 0; i < N; i++) {
            int m = rep % 4;
            x[i] = m == 0 ? -800.0 + 1600.0 * ur()             /* whole range incl. out of range */
                 : m == 1 ? -750.0 + 50.0 * ur()                 /* subnormal tail, -745 edge */
                 : m == 2 ? 700.0 + 12.0 * ur()                  /* overflow edge */
                 : (ur() - 0.5) * 4.0;                           /* the common case */
        }
        if (rep == 0) {
            double sp[] = {NAN, INFINITY, -INFINITY, 0.0, -0.0, 709.0, -745.0, nextafter(709.0, 1e9),
                           nextafter(-745.0, -1e9), 0.34657359027997264, -0.34657359027997264, 1e-300, -1e-300};
            for (unsigned i = 0; i < sizeof sp / sizeof *sp; i++) x[i] = sp[i];
        }
        g4_exp_vec(x, y, N, TAB2);
        for (int i = 0; i < N; i++) cmpd("exp", y[i], g4_exp(x[i]), x[i]);
        nexp += N;
    }
    printf("g4_exp_vec: %ld values, %ld mismatches\n", nexp, bad);
    long b0 = bad;

    /* --- GeGLU: random floats over many magnitudes, plus specials --- */
    static float g[N], g2[N], u[N];
    long ngelu = 0;
    for (int rep = 0; rep < 64; rep++) {
        for (int i = 0; i < N; i++) {
            double mag = pow(10.0, -6.0 + 10.0 * ur());           /* 1e-6 .. 1e4 */
            g[i] = (float)((ur() < 0.5 ? -1 : 1) * mag);
            u[i] = (float)((ur() - 0.5) * 8.0);
        }
        if (rep == 0) { float sp[] = {NAN, INFINITY, -INFINITY, 0.0f, -0.0f, 1e30f, -1e30f, 7.0f, -7.0f, 30.0f, -30.0f};
                        for (unsigned i = 0; i < sizeof sp / sizeof *sp; i++) g[i] = sp[i]; }
        memcpy(g2, g, sizeof g);
        g4_geglu(g, u, N, TAB2);
        geglu_ref(g2, u, N);
        for (int i = 0; i < N; i++) if (memcmp(&g[i], &g2[i], 4)) { if (bad < 20) printf("  geglu i %d: asm %.9g ref %.9g\n", i, g[i], g2[i]); bad++; }
        ngelu += N;
    }
    printf("g4_geglu: %ld values, %ld mismatches\n", ngelu, bad - b0);
    long b1 = bad;

    /* --- attention scores and p.V at both head sizes, many lengths --- */
    static float K[MAXSEQ * HD_F], V[MAXSEQ * HD_F], q[HD_F], s1[MAXSEQ], s2[MAXSEQ], o1[HD_F], o2[HD_F], p[MAXSEQ];
    long nsc = 0, npv = 0;
    for (int hdi = 0; hdi < 2; hdi++) {
        int hd = hdi ? HD_F : HD_S;
        for (int i = 0; i < MAXSEQ * hd; i++) { K[i] = (float)(ur() - 0.5); V[i] = (float)((ur() - 0.5) * 3); }
        int lens[] = {1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 64, 100, 511, 512, 513, 1000, 2048};
        for (unsigned li = 0; li < sizeof lens / sizeof *lens; li++) {
            int n = lens[li];
            for (int d = 0; d < hd; d++) q[d] = (float)((ur() - 0.5) * 2);
            g4_attn_scores(q, K, n, hd, s1);
            for (int t = 0; t < n; t++) { float s = 0.0f; for (int d = 0; d < hd; d++) s = fmaf(K[t * hd + d], q[d], s); s2[t] = s; }
            for (int t = 0; t < n; t++) if (memcmp(&s1[t], &s2[t], 4)) { if (bad < 30) printf("  scores hd %d n %d t %d\n", hd, n, t); bad++; }
            nsc += n;
            for (int t = 0; t < n; t++) p[t] = (float)ur();
            g4_attn_pv(p, V, n, hd, o1);
            for (int d = 0; d < hd; d++) o2[d] = 0.0f;
            for (int t = 0; t < n; t++) for (int d = 0; d < hd; d++) o2[d] = fmaf(p[t], V[t * hd + d], o2[d]);
            for (int d = 0; d < hd; d++) if (memcmp(&o1[d], &o2[d], 4)) { if (bad < 30) printf("  pv hd %d n %d d %d\n", hd, n, d); bad++; }
            npv += hd;
        }
    }
    printf("g4_attn_scores: %ld scores; g4_attn_pv: %ld outputs; %ld mismatches\n", nsc, npv, bad - b1);
    long b2 = bad, nall = 0;
    /* all-heads path against the per-head reference, positions [lo, pos] */
    static float qa[NH * HD_F], oa1[NH * HD_F], oa2[NH * HD_F];
    for (int hdi = 0; hdi < 2; hdi++) {
        int hd = hdi ? HD_F : HD_S;
        for (int i = 0; i < MAXSEQ * hd; i++) { K[i] = (float)((ur() - 0.5) * 2); V[i] = (float)((ur() - 0.5) * 3); }
        int cases[][2] = {{0, 0}, {0, 1}, {0, 2}, {0, 4}, {0, 5}, {0, 7}, {0, 63}, {10, 600}, {0, 511}, {1, 512}, {300, 811}, {0, 2047}};
        for (unsigned ci = 0; ci < sizeof cases / sizeof *cases; ci++) {
            int lo = cases[ci][0], pos = cases[ci][1];
            for (int i = 0; i < NH * hd; i++) qa[i] = (float)((ur() - 0.5) * 4);
            attn_all_asm(qa, K, V, lo, pos, hd, oa1);
            for (int j = 0; j < NH; j++) attn_head_ref(qa + j * hd, K, V, lo, pos, hd, oa2 + j * hd);
            for (int i = 0; i < NH * hd; i++) if (memcmp(&oa1[i], &oa2[i], 4)) { if (bad < 40) printf("  attn_all hd %d lo %d pos %d i %d\n", hd, lo, pos, i); bad++; }
            nall += NH * hd;
        }
    }
    printf("attn_all_asm (8 heads, one pass): %ld outputs, %ld mismatches\n", nall, bad - b2);
    long b3 = bad, nrms = 0;
    {   /* RMS norm: sizes used by the model, magnitudes from tiny to huge */
        static float xr_[HD_F * NH], w_[H], y1[H > HD_F ? H : HD_F], y2[H > HD_F ? H : HD_F];
        int sizes[] = {256, 512, 1536};
        for (int rep = 0; rep < 2000; rep++) {
            int nn = sizes[rep % 3];
            double sc_ = pow(10.0, -20.0 + 40.0 * ur());
            for (int i = 0; i < nn; i++) { xr_[i] = (float)((ur() - 0.5) * sc_); w_[i] = (float)((ur() - 0.2) * 3); }
            rmsnorm(xr_, w_, y1, nn); g4_rmsnorm(xr_, w_, y2, nn);
            for (int i = 0; i < nn; i++) if (memcmp(&y1[i], &y2[i], 4)) { if (bad < 50) printf("  rmsnorm n %d i %d\n", nn, i); bad++; }
            memcpy(y1, xr_, nn * 4); memcpy(y2, xr_, nn * 4);
            rmsnorm_noscale(y1, nn); g4_rmsnorm_ns(y2, nn);
            for (int i = 0; i < nn; i++) if (memcmp(&y1[i], &y2[i], 4)) { if (bad < 50) printf("  rmsnorm_ns n %d i %d\n", nn, i); bad++; }
            nrms += 2 * nn;
        }
    }
    printf("g4_rmsnorm / _ns: %ld outputs, %ld mismatches\n", nrms, bad - b3);
    long b4 = bad, nq8 = 0;
    {   /* Q8_K: model sizes, zero blocks, +-ties of the max |x|, tiny and huge scales */
        static float xv[2 * INTER];
        static uint8_t q1[(2 * INTER / 256) * Q8B], q2[(2 * INTER / 256) * Q8B];
        int sizes[] = {256, 1536, 2048, 4096, 6144, 12288};
        for (int rep = 0; rep < 3000; rep++) {
            int nn = sizes[rep % 6];
            double sc_ = pow(10.0, -30.0 + 60.0 * ur());
            for (int i = 0; i < nn; i++) xv[i] = (float)((ur() - 0.5) * sc_);
            if (rep % 7 == 1) for (int i = 0; i < 256; i++) xv[i] = 0.0f;
            if (rep % 7 == 2) { xv[5] = -3.0f * (float)sc_; xv[77] = 3.0f * (float)sc_; }
            if (rep % 7 == 3) { xv[200] = 2.0f * (float)sc_; xv[13] = -2.0f * (float)sc_; }
            memset(q1, 0xAA, sizeof q1); memset(q2, 0x55, sizeof q2);
            q8k(xv, q1, nn); g4_q8k(xv, q2, nn);
            if (memcmp(q1, q2, (size_t)(nn / 256) * Q8B)) { if (bad < 60) printf("  q8k n %d rep %d differs\n", nn, rep); bad++; }
            nq8 += nn / 256;
        }
    }
    printf("g4_q8k: %ld blocks (292 bytes each, compared whole), %ld mismatches\n", nq8, bad - b4);
    printf(bad ? "HOST ASM TEST: FAIL (%ld)\n" : "HOST ASM TEST: OK\n", bad);
    return bad != 0;
}
