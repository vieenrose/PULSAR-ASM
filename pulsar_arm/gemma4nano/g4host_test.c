/* g4host_test.c - g4_host.S against g4ref.c, bit for bit, on the Nano.
 * Build: gcc -O2 -ffp-contract=off -o g4host_test g4host_test.c g4_host.S -lm
 */
#define G4REF_NO_MAIN
#include "g4ref.c"

extern void g4_exp_vec(const double *x, double *y, long n, const double *tab);
extern void g4_geglu(float *g, const float *u, long n, const double *tab);
extern void g4_attn_scores(const float *q, const float *K, long n, long hd, float *s);
extern void g4_attn_pv(const float *p, const float *V, long n, long hd, float *o);

static double TAB2[48];
static void build_tab(void) {
    double v[24] = {EXP_P0};
    for (int i = 0; i < 13; i++) v[1 + i] = EXP_C[i];
    v[14] = EXP_LOG2E; v[15] = EXP_SH; v[16] = EXP_LN2HI; v[17] = EXP_LN2LO;
    v[18] = 709.0; v[19] = -745.0; v[20] = INFINITY; v[21] = GELU_C1; v[22] = GELU_C3; v[23] = 1.0;
    for (int i = 0; i < 24; i++) TAB2[2 * i] = TAB2[2 * i + 1] = v[i];
}
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
    printf(bad ? "HOST ASM TEST: FAIL (%ld)\n" : "HOST ASM TEST: OK\n", bad);
    return bad != 0;
}
