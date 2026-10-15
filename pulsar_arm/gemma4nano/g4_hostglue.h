/* g4_hostglue.h - C side of the AArch64 host kernels (g4_host.S), shared by the
 * engine (g4eng.c) and the unit test (g4host_test.c). Include after g4ref.c. */
extern void g4_exp_vec(const double *x, double *y, long n, const double *tab);
extern void g4_geglu(float *g, const float *u, long n, const double *tab);
extern void g4_attn_scores(const float *q, const float *K, long n, long hd, float *s);
extern void g4_attn_pv(const float *p, const float *V, long n, long hd, float *o);
extern void g4_attn_scores8(const float *qT, const float *K, long n, long hd, float *S);
extern void g4_attn_pv8(const float *P, const float *V, long n, long hd, float *O);
extern double g4_rms_r(const float *x, long n);
extern void g4_rmsnorm(const float *x, const float *w, float *y, long n);
extern void g4_rmsnorm_ns(float *x, long n);
extern void g4_q8k(const float *x, uint8_t *y, long n);
static inline void q8k_asm(const float *x, uint8_t *y, int n) { g4_q8k(x, y, n); }
static inline void rms_asm(const float *x, const float *w, float *y, int n) { g4_rmsnorm(x, w, y, n); }
static inline void rmsns_asm(float *x, int n) { g4_rmsnorm_ns(x, n); }
static double TAB2[48];   /* exp/GELU constants from g4ref.c's own definitions, duplicated pairs */
static void build_tab(void) {
    double v[24] = {EXP_P0};
    for (int i = 0; i < 13; i++) v[1 + i] = EXP_C[i];
    v[14] = EXP_LOG2E; v[15] = EXP_SH; v[16] = EXP_LN2HI; v[17] = EXP_LN2LO;
    v[18] = 709.0; v[19] = -745.0; v[20] = INFINITY; v[21] = GELU_C1; v[22] = GELU_C3; v[23] = 1.0;
    for (int i = 0; i < 24; i++) TAB2[2 * i] = TAB2[2 * i + 1] = v[i];
}
static inline void geglu_asm(float *g, const float *u, int n) { g4_geglu(g, u, n, TAB2); }
/* attn_head_ref with the hot loops in assembly; same operations in the same order */
static inline void attn_head_asm(const float *q, const float *K, const float *V, int lo, int pos, int hd, float *o) {
    static double xin[MAXSEQ], e[MAXSEQ];
    static float pp[MAXSEQ];
    int n = pos - lo + 1;
    g4_attn_scores(q, K + (size_t)lo * hd, n, hd, sc + lo);
    float mx = -INFINITY;
    for (int t = lo; t <= pos; t++) if (sc[t] > mx) mx = sc[t];
    for (int i = 0; i < n; i++) xin[i] = (double)sc[lo + i] - (double)mx;
    int ne = n & ~1;
    if (ne) g4_exp_vec(xin, e, ne, TAB2);
    if (n & 1) e[n - 1] = g4_exp(xin[n - 1]);
    double sum = 0;
    for (int i = 0; i < n; i++) sum += e[i];
    for (int i = 0; i < n; i++) pp[i] = (float)(e[i] / sum);
    g4_attn_pv(pp, V + (size_t)lo * hd, n, hd, o);
}

/* All 8 heads in one pass over K and V (attn_all_ref's result, bit for bit):
   scores per head = the same fmaf chains; softmax per head as attn_head_ref;
   p.V per head and dim = the same fmaf chain over t. */
static void attn_all_asm(const float *q, const float *K, const float *V, int lo, int pos, int hd, float *o) {
    static float qT[HD_F * NH], S8[MAXSEQ * NH], P8[MAXSEQ * NH];
    static double xin[MAXSEQ], e[MAXSEQ];
    int n = pos - lo + 1;
    for (int h = 0; h < NH; h++) for (int d = 0; d < hd; d++) qT[d * NH + h] = q[h * hd + d];
    g4_attn_scores8(qT, K + (size_t)lo * hd, n, hd, S8);
    for (int h = 0; h < NH; h++) {
        float mx = -INFINITY;
        for (int t = 0; t < n; t++) if (S8[t * NH + h] > mx) mx = S8[t * NH + h];
        for (int t = 0; t < n; t++) xin[t] = (double)S8[t * NH + h] - (double)mx;
        int ne = n & ~1;
        if (ne) g4_exp_vec(xin, e, ne, TAB2);
        if (n & 1) e[n - 1] = g4_exp(xin[n - 1]);
        double sum = 0;
        for (int t = 0; t < n; t++) sum += e[t];
        for (int t = 0; t < n; t++) P8[t * NH + h] = (float)(e[t] / sum);
    }
    g4_attn_pv8(P8, V + (size_t)lo * hd, n, hd, o);
}
