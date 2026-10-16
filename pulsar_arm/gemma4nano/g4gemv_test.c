/* g4gemv_test.c - M3 gate: every GPU GEMV of the Gemma 4 E2B Nano engine must be
 * bit-identical to g4ref.c on the real weights.
 *
 * Covers all 275 Q4_0 layer tensors (K2's q4_gemv kernel, g4_ptx_q4.h), the
 * 262144-row Q6_K tied head (q6k_gemv) and the F16 per_layer_model_proj
 * (f16_gemv_t), each with three activation vectors (two Gaussian-like seeds and a
 * heavy-tailed one). The CPU side IS g4ref.c (included, main compiled out), so the
 * comparison is against the reference code itself, not a copy.
 *
 * Build on the Nano (driver only, no CUDA toolkit at runtime):
 *   gcc -O2 -fopenmp -ffp-contract=off -o g4gemv_test g4gemv_test.c -ldl -lm
 * Run:   ./g4gemv_test <gemma.gguf> <g4_kernels.ptx> [q4|head|f16|all|perf]
 */
#define G4REF_NO_MAIN
#include "g4ref.c"
#include "g4_ptx_q4.h"
#include <dlfcn.h>
#include <time.h>

typedef int CUresult;
static CUresult (*cuInit_)(unsigned);
static CUresult (*cuDeviceGet_)(int *, int);
static CUresult (*cuCtxCreate_)(void **, unsigned, int);
static CUresult (*cuModuleLoadData_)(void **, const void *);
static CUresult (*cuModuleGetFunction_)(void **, void *, const char *);
static CUresult (*cuMemAlloc_)(uint64_t *, size_t);
static CUresult (*cuMemFree_)(uint64_t);
static CUresult (*cuMemcpyHtoD_)(uint64_t, const void *, size_t);
static CUresult (*cuMemcpyDtoH_)(void *, uint64_t, size_t);
static CUresult (*cuLaunchKernel_)(void *, unsigned, unsigned, unsigned, unsigned, unsigned,
                                   unsigned, unsigned, void *, void **, void **);
static CUresult (*cuCtxSynchronize_)(void);

#define CK(x) do { CUresult r_ = (x); if (r_) { fprintf(stderr, "%s:%d %s -> %d\n", __FILE__, __LINE__, #x, r_); exit(3); } } while (0)
static void *sym(void *h, const char *n) { void *p = dlsym(h, n); if (!p) { fprintf(stderr, "no %s\n", n); exit(3); } return p; }
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static double urand(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (double)(rng >> 11) * 0x1p-53; }
static void make_x(float *x, int n, int kind) {
    for (int i = 0; i < n; i++) {
        double g = urand() + urand() + urand() + urand() - 2.0;     /* ~Gaussian, sd 0.58 */
        if (kind == 2 && urand() < 0.01) g *= 40.0;                 /* heavy tail: outliers */
        x[i] = (float)(g * (kind == 1 ? 3.0 : 1.0));
    }
}
static int first_diff(const float *a, const float *b, int n) { for (int i = 0; i < n; i++) if (memcmp(a + i, b + i, 4)) return i; return -1; }

static void *fq4, *fq6, *fq62, *fq63, *fq64, *ff16, *fgeglu, *fq8k, *fq4i, *fq6i, *fq4ib, *fq6ib, *fgeglub, *fargb, *fasc, *fapv, *ff162, *ff16b, *fgq2;
static uint64_t dWi, dSi, dSh;   /* q4i nibbles; scales f32 (v1-v4, b) and fp16 (v5, b2) */
static uint64_t dW, dS, dX, dY;

/* q4i layout: [tile of 32 rows][block][32 rows] x (16 B nibbles | f32 scale) */
static void repack_q4i(const tensor_t *t, uint8_t *nib, float *scl) {
    int nb = (int)t->dims[0] / 32, rows = (int)t->dims[1];
    for (int r = 0; r < rows; r++) for (int b = 0; b < nb; b++) {
        const uint8_t *blk = t->data + ((size_t)r * nb + b) * 18;
        size_t o = ((size_t)(r / 32) * nb + b) * 32 + (r % 32);
        scl[o] = h2f((uint16_t)(blk[0] | blk[1] << 8));
        memcpy(nib + o * 16, blk + 2, 16);
    }
}
/* repack t into dWi / dSi (f32) / dSh (fp16) */
static void upload_q4i(const tensor_t *t) {
    int nb = (int)t->dims[0] / 32, rows = (int)t->dims[1];
    uint8_t *nib = malloc((size_t)rows * nb * 16); float *scl = malloc((size_t)rows * nb * 4);
    uint16_t *sh = malloc((size_t)rows * nb * 2);
    repack_q4i(t, nib, scl);
    for (int r = 0; r < rows; r++) for (int b = 0; b < nb; b++) {
        const uint8_t *blk = t->data + ((size_t)r * nb + b) * 18;
        sh[((size_t)(r / 32) * nb + b) * 32 + (r % 32)] = (uint16_t)(blk[0] | blk[1] << 8);
    }
    CK(cuMemcpyHtoD_(dWi, nib, (size_t)rows * nb * 16)); CK(cuMemcpyHtoD_(dSi, scl, (size_t)rows * nb * 4));
    CK(cuMemcpyHtoD_(dSh, sh, (size_t)rows * nb * 2));
    free(nib); free(scl); free(sh);
}
static int q4i_v5(void) { return !getenv("Q4I") || getenv("Q4I")[0] == '5'; }   /* default: the engine's kernel */
static void launch_q4i(const tensor_t *t, uint64_t out) {
    int rows = (int)t->dims[1], cols = (int)t->dims[0];
    void *a[] = {&dWi, q4i_v5() ? &dSh : &dSi, &dX, &out, &rows, &cols};
    unsigned sm = (getenv("Q4I") && getenv("Q4I")[0] < '3') ? 0 : (unsigned)(cols * 2 + cols / 32 * 4 + cols / 256 * 4);
    CK(cuLaunchKernel_(fq4i, (rows + 255) / 256, 1, 1, 32, 8, 1, sm, NULL, a, NULL));
}
static int test_q4(const tensor_t *t) {
    int cols = (int)t->dims[0], rows = (int)t->dims[1], nb = cols / 32;
    /* device layout: nibbles [rows][nb][16], scales f32 [rows][nb] */
    uint8_t *nib = malloc((size_t)rows * nb * 16); float *scl = malloc((size_t)rows * nb * 4);
    for (size_t i = 0; i < (size_t)rows * nb; i++) {
        const uint8_t *blk = t->data + i * 18;
        scl[i] = h2f((uint16_t)(blk[0] | blk[1] << 8));
        memcpy(nib + i * 16, blk + 2, 16);
    }
    CK(cuMemcpyHtoD_(dW, nib, (size_t)rows * nb * 16));
    CK(cuMemcpyHtoD_(dS, scl, (size_t)rows * nb * 4));
    upload_q4i(t);
    static float x[2 * INTER], ycpu[2 * INTER], ygpu[2 * INTER];
    int bad = 0;
    for (int kind = 0; kind < 3; kind++) {
        make_x(x, cols, kind);
        q8k(x, XQ, cols);
        gemv_q4(t, XQ, ycpu);
        CK(cuMemcpyHtoD_(dX, XQ, (size_t)(cols / 256) * Q8B));
        uint32_t r0 = 0, r1 = rows, unb = nb, rf = rows;
        void *args[] = {&dW, &dS, &dX, &dY, &r0, &r1, &unb, &rf};
        CK(cuLaunchKernel_(fq4, (rows + 31) / 32, 1, 1, 32, 8, 1, 0, NULL, args, NULL));
        CK(cuCtxSynchronize_());
        CK(cuMemcpyDtoH_(ygpu, dY, (size_t)rows * 4));
        int d = first_diff(ycpu, ygpu, rows);
        if (d >= 0) { bad++; printf("  %s x%d: row %d cpu %.9g gpu %.9g\n", t->name, kind, d, ycpu[d], ygpu[d]); }
        launch_q4i(t, dY); CK(cuCtxSynchronize_());
        CK(cuMemcpyDtoH_(ygpu, dY, (size_t)rows * 4));
        d = first_diff(ycpu, ygpu, rows);
        if (d >= 0) { bad++; printf("  q4i %s x%d: row %d cpu %.9g gpu %.9g\n", t->name, kind, d, ycpu[d], ygpu[d]); }
    }
    free(nib); free(scl);
    return bad;
}

/* q6k_gemv2 layout: each 210-byte Q6_K block padded to 212 bytes (content unchanged) */
static uint8_t *repack_q6_212(const tensor_t *t) {
    size_t nblk = (size_t)t->dims[1] * (t->dims[0] / 256);
    uint8_t *o = calloc(nblk, 212);
    for (size_t i = 0; i < nblk; i++) memcpy(o + i * 212, t->data + i * 210, 210);
    return o;
}
/* q6i layout: [tile][block][chunk 0..13][32 lanes] x 16 B, blocks padded 210 -> 224 */
static uint8_t *repack_q6i(const tensor_t *t) {
    size_t rows = t->dims[1];
    uint8_t *o = calloc(rows * 6, 224);
    for (size_t r = 0; r < rows; r++) for (int b = 0; b < 6; b++) {
        const uint8_t *src = t->data + (r * 6 + b) * 210;
        uint8_t blk[224] = {0};
        memcpy(blk, src, 210);
        for (int k = 0; k < 14; k++)
            memcpy(o + ((((r / 32) * 6 + b) * 14 + k) * 32 + (r % 32)) * 16, blk + 16 * k, 16);
    }
    return o;
}
static int test_head(void) {
    const tensor_t *t = tens("token_embd.weight");
    int cols = (int)t->dims[0], rows = (int)t->dims[1], nb = cols / 256;
    uint64_t dH, dH2; CK(cuMemAlloc_(&dH, (size_t)rows * nb * 210)); CK(cuMemAlloc_(&dH2, (size_t)rows * nb * 212));
    CK(cuMemcpyHtoD_(dH, t->data, (size_t)rows * nb * 210));
    { uint8_t *rp = repack_q6_212(t); CK(cuMemcpyHtoD_(dH2, rp, (size_t)rows * nb * 212)); free(rp); }
    float *ygpu2 = malloc((size_t)rows * 4);
    float *ycpu = malloc((size_t)rows * 4), *ygpu = malloc((size_t)rows * 4);
    static float x[H];
    int bad = 0;
    for (int kind = 0; kind < 3; kind++) {
        make_x(x, cols, kind);
        q8k(x, XQ, cols);
        double t0 = now_ms();
        gemv_q6k_rows(t, XQ, ycpu, 0, rows);
        double t1 = now_ms();
        CK(cuMemcpyHtoD_(dX, XQ, (size_t)nb * Q8B));
        int irows = rows, inb = nb;
        void *args[] = {&dH, &dX, &dY, &irows, &inb};
        double t2 = now_ms();
        CK(cuLaunchKernel_(fq6, (rows + 7) / 8, 1, 1, 32, 8, 1, 0, NULL, args, NULL));
        CK(cuCtxSynchronize_());
        double t3 = now_ms();
        CK(cuMemcpyDtoH_(ygpu, dY, (size_t)rows * 4));
        int irows2 = rows;
        void *args2[] = {&dH2, &dX, &dY, &irows2};
        CK(cuLaunchKernel_(fq62, (rows + 31) / 32, 1, 1, 32, 8, 1, 0, NULL, args2, NULL));
        CK(cuCtxSynchronize_());
        CK(cuMemcpyDtoH_(ygpu2, dY, (size_t)rows * 4));
        int d = first_diff(ycpu, ygpu, rows), d2 = first_diff(ycpu, ygpu2, rows);
        CK(cuLaunchKernel_(fq63, (rows + 31) / 32, 1, 1, 32, 8, 1, 0, NULL, args2, NULL));
        CK(cuCtxSynchronize_());
        CK(cuMemcpyDtoH_(ygpu2, dY, (size_t)rows * 4));
        int d3 = first_diff(ycpu, ygpu2, rows);
        printf("  head x%d: v3 %s\n", kind, d3 < 0 ? "262144/262144 identical" : "MISMATCH");
        if (d3 >= 0) bad++;
        CK(cuLaunchKernel_(fq64, (rows + 31) / 32, 1, 1, 32, 8, 1, 0, NULL, args2, NULL));
        CK(cuCtxSynchronize_());
        CK(cuMemcpyDtoH_(ygpu2, dY, (size_t)rows * 4));
        int d4 = first_diff(ycpu, ygpu2, rows);
        printf("  head x%d: v4 %s\n", kind, d4 < 0 ? "262144/262144 identical" : "MISMATCH");
        if (d4 >= 0) bad++;
        {
            static uint64_t dHi = 0;
            if (!dHi) { uint8_t *rp = repack_q6i(t); CK(cuMemAlloc_(&dHi, (size_t)rows * 6 * 224)); CK(cuMemcpyHtoD_(dHi, rp, (size_t)rows * 6 * 224)); free(rp); }
            void *ai[] = {&dHi, &dX, &dY, &irows2};
            CK(cuLaunchKernel_(fq6i, (rows + 255) / 256, 1, 1, 32, 8, 1, 0, NULL, ai, NULL));
            CK(cuCtxSynchronize_());
            CK(cuMemcpyDtoH_(ygpu2, dY, (size_t)rows * 4));
            int di = first_diff(ycpu, ygpu2, rows);
            printf("  head x%d: q6i %s\n", kind, di < 0 ? "262144/262144 identical" : "MISMATCH");
            if (di >= 0) { bad++; printf("    q6i row %d cpu %.9g gpu %.9g\n", di, ycpu[di], ygpu2[di]); }
        }
        printf("  head x%d: v1 %s, v2 %s  (cpu ref %.0f ms)\n", kind,
               d < 0 ? "262144/262144 identical" : "MISMATCH", d2 < 0 ? "262144/262144 identical" : "MISMATCH", t1 - t0);
        (void)t2; (void)t3;
        if (d >= 0) { bad++; printf("    v1 row %d cpu %.9g gpu %.9g\n", d, ycpu[d], ygpu[d]); }
        if (d2 >= 0) { bad++; printf("    v2 row %d cpu %.9g gpu %.9g\n", d2, ycpu[d2], ygpu2[d2]); }
    }
    CK(cuMemFree_(dH)); CK(cuMemFree_(dH2)); free(ycpu); free(ygpu); free(ygpu2);
    return bad;
}


/* MLP stages: geglu_dev vs geglu_ref and q8k_dev vs q8k, bit for bit */
static int test_mlp_stages(void) {
    double v[24] = {EXP_P0};
    for (int i = 0; i < 13; i++) v[1 + i] = EXP_C[i];
    v[14] = EXP_LOG2E; v[15] = EXP_SH; v[16] = EXP_LN2HI; v[17] = EXP_LN2LO;
    v[18] = 709.0; v[19] = -745.0; v[20] = INFINITY; v[21] = GELU_C1; v[22] = GELU_C3; v[23] = 1.0;
    uint64_t dTab, dIn, dOut, dQ;
    const int n = 2 * INTER;
    CK(cuMemAlloc_(&dTab, sizeof v)); CK(cuMemcpyHtoD_(dTab, v, sizeof v));
    CK(cuMemAlloc_(&dIn, (size_t)2 * n * 4)); CK(cuMemAlloc_(&dOut, (size_t)n * 4)); CK(cuMemAlloc_(&dQ, (size_t)(n / 256) * Q8B));
    static float y[2 * 2 * INTER], g[2 * INTER], hg[2 * INTER];
    static uint8_t qc[(2 * INTER / 256) * Q8B], qg[(2 * INTER / 256) * Q8B];
    int bad = 0; long ng = 0, nq = 0;
    for (int rep = 0; rep < 40; rep++) {
        for (int i = 0; i < 2 * n; i++) {
            double mag = pow(10.0, -5.0 + 9.0 * urand());
            y[i] = (float)((urand() < 0.5 ? -1 : 1) * mag);
        }
        /* Edge values, all finite: GEMV outputs are always finite, and the contract
           does not cover non-finite inputs (ARM and the GPU produce NaNs with
           different payloads, e.g. gelu(-inf) = -inf/inf). Huge magnitudes still
           drive the exp overflow and underflow branches. */
        if (rep == 0) { float sp[] = {0.0f, -0.0f, 1e30f, -1e30f, 3e38f, -3e38f, 7.0f, -7.0f, 30.0f, -30.0f, 1e-30f, -1e-30f};
                        for (unsigned i = 0; i < sizeof sp / sizeof *sp; i++) y[i] = sp[i]; }
        /* GeGLU */
        for (int i = 0; i < n; i++) g[i] = y[i];
        geglu_ref(g, y + n, n);
        CK(cuMemcpyHtoD_(dIn, y, (size_t)2 * n * 4));
        int in = n;
        void *a1[] = {&dIn, &dOut, &in, &dTab};
        CK(cuLaunchKernel_(fgeglu, (n + 255) / 256, 1, 1, 256, 1, 1, 0, NULL, a1, NULL));
        CK(cuCtxSynchronize_());
        CK(cuMemcpyDtoH_(hg, dOut, (size_t)n * 4));
        for (int i = 0; i < n; i++) if (memcmp(&g[i], &hg[i], 4)) { if (bad < 10) printf("  geglu i %d: gpu %.9g ref %.9g\n", i, hg[i], g[i]); bad++; }
        ng += n;
        /* Q8_K on the GeGLU output, with crafted blocks: all-zero, and +-ties of |x| */
        if (rep == 1) for (int j = 0; j < 256; j++) g[j] = 0.0f;
        if (rep == 2) { g[256 + 10] = -5.0f; g[256 + 40] = 5.0f; for (int j = 0; j < 256; j++) if (j != 10 && j != 40) g[256 + j] = 1.0f; }
        if (rep == 3) { g[512 + 3] = 7.0f; g[512 + 200] = -7.0f; }
        q8k(g, qc, n);
        CK(cuMemcpyHtoD_(dOut, g, (size_t)n * 4));
        void *a2[] = {&dOut, &dQ};
        CK(cuLaunchKernel_(fq8k, n / 256, 1, 1, 256, 1, 1, 0, NULL, a2, NULL));
        CK(cuCtxSynchronize_());
        CK(cuMemcpyDtoH_(qg, dQ, (size_t)(n / 256) * Q8B));
        for (int b = 0; b < n / 256; b++) if (memcmp(qc + b * Q8B, qg + b * Q8B, 260)) { if (bad < 20) printf("  q8k block %d differs\n", b); bad++; }
        nq += n / 256;
    }
    printf("  GeGLU on GPU: %ld values; Q8_K on GPU: %ld blocks; %d mismatches\n", ng, nq, bad);
    return bad;
}

/* batched kernels (prefill): every token of a B = 8 batch must equal the single-token
   CPU reference, bit for bit */
static int test_batch(void) {
    enum { B = 8 };
    int bad = 0;
    static uint8_t xqb[B * (2 * INTER / 256) * Q8B];
    static float x[2 * INTER], ycpu[2 * INTER], yb[B * 2 * INTER];
    uint64_t dXb, dYb;
    CK(cuMemAlloc_(&dXb, sizeof xqb)); CK(cuMemAlloc_(&dYb, (size_t)B * VOCAB * 4));
    const char *names[] = {"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_output.weight", "blk.4.attn_output.weight",
                           "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight", "blk.20.ffn_gate.weight", "blk.20.ffn_down.weight",
                           "blk.0.inp_gate.weight", "blk.0.proj.weight"};
    int ntest = 0;
    for (unsigned k = 0; k < sizeof names / sizeof *names; k++) {
        const tensor_t *t = tens(names[k]);
        int cols = (int)t->dims[0], rows = (int)t->dims[1], nb = cols / 32, xs = (cols / 256) * Q8B;
        uint8_t *nib = malloc((size_t)rows * nb * 16); float *scl = malloc((size_t)rows * nb * 4);
        upload_q4i(t);
        free(nib); free(scl);
        for (int b = 0; b < B; b++) { make_x(x, cols, b % 3); q8k(x, xqb + (size_t)b * xs, cols); }
        CK(cuMemcpyHtoD_(dXb, xqb, (size_t)B * xs));
        int ib = B;
        void *a[] = {&dWi, getenv("Q4B1") ? &dSi : &dSh, &dXb, &dYb, &rows, &cols, &ib, &xs, &rows};
        CK(cuLaunchKernel_(fq4ib, (rows + 255) / 256, 1, 1, 32, 8, 1, 0, NULL, a, NULL));
        CK(cuCtxSynchronize_());
        CK(cuMemcpyDtoH_(yb, dYb, (size_t)B * rows * 4));
        for (int b = 0; b < B; b++) {
            gemv_q4(t, xqb + (size_t)b * xs, ycpu);
            int d = first_diff(ycpu, yb + (size_t)b * rows, rows);
            if (d >= 0) { bad++; printf("  q4i_b %s token %d row %d\n", names[k], b, d); }
        }
        ntest++;
    }
    printf("  q4i_gemv_b: %d shapes x 8 tokens, %d mismatches\n", ntest, bad);
    int b1 = bad;
    {
        const tensor_t *t = tens("token_embd.weight");
        int rows = (int)t->dims[1], xs = 6 * Q8B;
        uint8_t *rp = repack_q6i(t);
        uint64_t dHi; CK(cuMemAlloc_(&dHi, (size_t)rows * 6 * 224)); CK(cuMemcpyHtoD_(dHi, rp, (size_t)rows * 6 * 224)); free(rp);
        for (int b = 0; b < B; b++) { make_x(x, H, b % 3); q8k(x, xqb + (size_t)b * xs, H); }
        CK(cuMemcpyHtoD_(dXb, xqb, (size_t)B * xs));
        for (int t0 = 0; t0 < B; t0 += 4) {   /* q6i_gemv_b: 4 tokens per pass */
            int ib = 4; uint64_t xt = dXb + (uint64_t)t0 * xs, yt = dYb + (uint64_t)t0 * rows * 4;
            void *a[] = {&dHi, &xt, &yt, &rows, &ib, &xs, &rows};
            CK(cuLaunchKernel_(fq6ib, (rows + 255) / 256, 1, 1, 32, 8, 1, 0, NULL, a, NULL));
        }
        CK(cuCtxSynchronize_());
        float *hb = malloc((size_t)B * rows * 4), *hc = malloc((size_t)rows * 4);
        CK(cuMemcpyDtoH_(hb, dYb, (size_t)B * rows * 4));
        uint64_t dOut; CK(cuMemAlloc_(&dOut, 64));
        void *aa[] = {&dYb, &rows, &dOut};
        CK(cuLaunchKernel_(fargb, B, 1, 1, 1024, 1, 1, 0, NULL, aa, NULL));
        CK(cuCtxSynchronize_());
        unsigned am[2 * B]; CK(cuMemcpyDtoH_(am, dOut, sizeof am));
        for (int b = 0; b < B; b++) {
            gemv_q6k_rows(t, xqb + (size_t)b * xs, hc, 0, rows);
            int d = first_diff(hc, hb + (size_t)b * rows, rows);
            if (d >= 0) { bad++; printf("  q6i_b token %d row %d\n", b, d); }
            int top = 0; for (int v = 1; v < rows; v++) if (hc[v] > hc[top]) top = v;
            float tv; memcpy(&tv, &am[2 * b], 4);
            if ((int)am[2 * b + 1] != top || memcmp(&tv, &hc[top], 4)) { bad++; printf("  argmax_b token %d: gpu %u cpu %d\n", b, am[2 * b + 1], top); }
        }
        free(hb); free(hc);
        printf("  q6i_gemv_b + argmax_b: 8 tokens x 262144 rows, %d mismatches\n", bad - b1);
    }
    int b2 = bad;
    {
        double v[24] = {EXP_P0};
        for (int i = 0; i < 13; i++) v[1 + i] = EXP_C[i];
        v[14] = EXP_LOG2E; v[15] = EXP_SH; v[16] = EXP_LN2HI; v[17] = EXP_LN2LO;
        v[18] = 709.0; v[19] = -745.0; v[20] = INFINITY; v[21] = GELU_C1; v[22] = GELU_C3; v[23] = 1.0;
        uint64_t dTab; CK(cuMemAlloc_(&dTab, sizeof v)); CK(cuMemcpyHtoD_(dTab, v, sizeof v));
        const int n = INTER;
        static float yy[B * 2 * INTER], g[INTER], hgo[B * INTER];
        for (int i = 0; i < B * 2 * n; i++) yy[i] = (float)((urand() - 0.5) * 20.0);
        uint64_t dG; CK(cuMemAlloc_(&dG, (size_t)B * 2 * n * 4));
        CK(cuMemcpyHtoD_(dG, yy, (size_t)B * 2 * n * 4));
        int in = n, ib = B;
        void *a[] = {&dG, &dYb, &in, &ib, &dTab};
        CK(cuLaunchKernel_(fgeglub, (B * n + 255) / 256, 1, 1, 256, 1, 1, 0, NULL, a, NULL));
        CK(cuCtxSynchronize_());
        CK(cuMemcpyDtoH_(hgo, dYb, (size_t)B * n * 4));
        for (int b = 0; b < B; b++) {
            memcpy(g, yy + (size_t)b * 2 * n, n * 4);
            geglu_ref(g, yy + (size_t)b * 2 * n + n, n);
            if (first_diff(g, hgo + (size_t)b * n, n) >= 0) { bad++; printf("  geglu_b token %d\n", b); }
        }
        {   /* geglu_q8k2 on the same 8 tokens ([gate|up] layout) vs geglu_ref + q8k */
            static uint8_t qc[INTER / 256 * Q8B], qg[8 * INTER / 256 * Q8B];
            uint64_t dQ; CK(cuMemAlloc_(&dQ, sizeof qg));
            int gs = 2 * n; uint64_t up = dG + (uint64_t)n * 4;
            void *a[] = {&dG, &gs, &up, &gs, &dQ, &in, &dTab};
            CK(cuLaunchKernel_(fgq2, B * n / 256, 1, 1, 256, 1, 1, 0, NULL, a, NULL));
            CK(cuCtxSynchronize_());
            CK(cuMemcpyDtoH_(qg, dQ, sizeof qg));
            for (int b = 0; b < B; b++) {
                memcpy(g, yy + (size_t)b * 2 * n, n * 4);
                geglu_ref(g, yy + (size_t)b * 2 * n + n, n);
                q8k(g, qc, n);
                if (memcmp(qc, qg + (size_t)b * (n / 256) * Q8B, (size_t)(n / 256) * Q8B)) { bad++; printf("  geglu_q8k2 token %d\n", b); }
            }
            CK(cuMemFree_(dQ));
        }
        printf("  geglu_b + geglu_q8k2: 8 tokens x %d, %d mismatches\n", n, bad - b2);
    }
    return bad;
}


/* decode attention kernels (attn_scores + attn_softmax_pv) against attn_head_ref,
   all 8 heads, bit for bit; scores spread like real ones (up to ~+-30) */
static int test_attn(void) {
    double v[24] = {EXP_P0};
    for (int i = 0; i < 13; i++) v[1 + i] = EXP_C[i];
    v[14] = EXP_LOG2E; v[15] = EXP_SH; v[16] = EXP_LN2HI; v[17] = EXP_LN2LO;
    v[18] = 709.0; v[19] = -745.0; v[20] = INFINITY; v[21] = GELU_C1; v[22] = GELU_C3; v[23] = 1.0;
    uint64_t dTab, dQ, dK, dV, dS, dO;
    CK(cuMemAlloc_(&dTab, sizeof v)); CK(cuMemcpyHtoD_(dTab, v, sizeof v));
    const int NP = 640;
    float *K = malloc((size_t)NP * HD_F * 4), *V = malloc((size_t)NP * HD_F * 4);
    static float q[NH * HD_F], o1[NH * HD_F], o2[NH * HD_F];
    CK(cuMemAlloc_(&dQ, sizeof q)); CK(cuMemAlloc_(&dK, (size_t)NP * HD_F * 4)); CK(cuMemAlloc_(&dV, (size_t)NP * HD_F * 4));
    CK(cuMemAlloc_(&dS, (size_t)NP * NH * 4)); CK(cuMemAlloc_(&dO, sizeof o2));
    int bad = 0, ncase = 0;
    const int cases[][3] = {{256, 0, 0}, {256, 0, 1}, {256, 0, 36}, {256, 0, 299}, {512, 0, 352}, {512, 0, 5},
                            {256, 100, 611}, {512, 37, 200}, {256, 0, 255}, {512, 0, 256}};
    for (unsigned c = 0; c < sizeof cases / sizeof *cases; c++) {
        const int hd = cases[c][0], lo = cases[c][1], pos = cases[c][2], n = pos - lo + 1;
        for (int i = 0; i < NP * hd; i++) { K[i] = (float)((urand() - 0.5) * 2.0); V[i] = (float)((urand() - 0.5) * 4.0); }
        for (int i = 0; i < NH * hd; i++) q[i] = (float)((urand() - 0.5) * (c & 1 ? 1.0 : 6.0));
        for (int j = 0; j < NH; j++) attn_head_ref(q + j * hd, K, V, lo, pos, hd, o1 + j * hd);
        CK(cuMemcpyHtoD_(dQ, q, (size_t)NH * hd * 4));
        CK(cuMemcpyHtoD_(dK, K, (size_t)NP * hd * 4)); CK(cuMemcpyHtoD_(dV, V, (size_t)NP * hd * 4));
        int ipos = pos, win = lo > 0 ? pos + 1 - lo : 0, ihd = hd, sstr = 0;
        void *a1[] = {&dQ, &dK, &dS, &ipos, &win, &ihd, &sstr};
        CK(cuLaunchKernel_(fasc, (n * NH + 255) / 256, 1, 1, 256, 1, 1, (unsigned)(NH * hd * 4), NULL, a1, NULL));
        void *a2[] = {&dS, &dV, &dO, &ipos, &win, &ihd, &dTab, &sstr};
        CK(cuLaunchKernel_(fapv, NH, 1, 1, 256, 1, 1, (unsigned)(n * 12), NULL, a2, NULL));
        CK(cuCtxSynchronize_());
        CK(cuMemcpyDtoH_(o2, dO, (size_t)NH * hd * 4));
        int d = first_diff(o1, o2, NH * hd);
        if (d >= 0) { bad++; printf("  attn hd %d lo %d pos %d: first diff at %d (%a vs %a)\n", hd, lo, pos, d, o1[d], o2[d]); }
        ncase++;
    }
    /* batched: 8 tokens at pos0..pos0+7, each attending its own causal (windowed) range */
    {
        const int bc[][3] = {{256, 300, 512}, {512, 200, 0}, {256, 508, 512}, {256, 0, 512}};
        static float qb[8 * NH * HD_F], ob[8 * NH * HD_F];
        uint64_t dQ8, dS8, dO8;
        CK(cuMemAlloc_(&dQ8, sizeof qb)); CK(cuMemAlloc_(&dS8, (size_t)8 * NP * NH * 4)); CK(cuMemAlloc_(&dO8, sizeof ob));
        for (unsigned c = 0; c < sizeof bc / sizeof *bc; c++) {
            const int hd = bc[c][0], pos0 = bc[c][1], win = bc[c][2], B = 8;
            for (int i = 0; i < NP * hd; i++) { K[i] = (float)((urand() - 0.5) * 2.0); V[i] = (float)((urand() - 0.5) * 4.0); }
            for (int i = 0; i < B * NH * hd; i++) qb[i] = (float)((urand() - 0.5) * 6.0);
            CK(cuMemcpyHtoD_(dQ8, qb, (size_t)B * NH * hd * 4));
            CK(cuMemcpyHtoD_(dK, K, (size_t)NP * hd * 4)); CK(cuMemcpyHtoD_(dV, V, (size_t)NP * hd * 4));
            const int last = pos0 + B - 1, nmax = last - ((win > 0 && last + 1 > win) ? last + 1 - win : 0) + 1;
            int ip = pos0, iw = win, ihd = hd, sstr = NP * NH;
            void *a1[] = {&dQ8, &dK, &dS8, &ip, &iw, &ihd, &sstr};
            CK(cuLaunchKernel_(fasc, (nmax * NH + 255) / 256, B, 1, 256, 1, 1, (unsigned)(NH * hd * 4), NULL, a1, NULL));
            void *a2[] = {&dS8, &dV, &dO8, &ip, &iw, &ihd, &dTab, &sstr};
            CK(cuLaunchKernel_(fapv, NH, B, 1, 256, 1, 1, (unsigned)(nmax * 12), NULL, a2, NULL));
            CK(cuCtxSynchronize_());
            CK(cuMemcpyDtoH_(ob, dO8, (size_t)B * NH * hd * 4));
            for (int t = 0; t < B; t++) {
                const int pos = pos0 + t, lo = (win > 0 && pos + 1 > win) ? pos + 1 - win : 0;
                for (int j = 0; j < NH; j++) attn_head_ref(qb + (size_t)t * NH * hd + j * hd, K, V, lo, pos, hd, o1 + j * hd);
                int d = first_diff(o1, ob + (size_t)t * NH * hd, NH * hd);
                if (d >= 0) { bad++; printf("  attn batched hd %d pos %d (win %d): first diff at %d\n", hd, pos, win, d); }
            }
            ncase += B;
        }
        CK(cuMemFree_(dQ8)); CK(cuMemFree_(dS8)); CK(cuMemFree_(dO8));
    }
    printf("  attention: %d cases (n 1..512, hd 256/512, lo > 0, batched 4 x 8 tokens incl. window crossing), %d mismatches\n", ncase, bad);
    if (getenv("ATTNPERF")) {   /* warm timing at decode shapes (n = 300) */
        for (int hd = 256; hd <= 512; hd *= 2) {
            int ip = 299, iw = 0, ihd = hd, sstr = 0;
            void *a1[] = {&dQ, &dK, &dS, &ip, &iw, &ihd, &sstr};
            void *a2[] = {&dS, &dV, &dO, &ip, &iw, &ihd, &dTab, &sstr};
            double t0 = now_ms();
            while (now_ms() - t0 < 1500) { CK(cuLaunchKernel_(fasc, (300 * NH + 255) / 256, 1, 1, 256, 1, 1, (unsigned)(NH * hd * 4), NULL, a1, NULL)); CK(cuCtxSynchronize_()); }
            for (int rep = 0; rep < 2; rep++) {
                double t1 = now_ms();
                for (int i = 0; i < 100; i++) CK(cuLaunchKernel_(fasc, (300 * NH + 255) / 256, 1, 1, 256, 1, 1, (unsigned)(NH * hd * 4), NULL, a1, NULL));
                CK(cuCtxSynchronize_()); double t2 = now_ms();
                for (int i = 0; i < 100; i++) CK(cuLaunchKernel_(fapv, NH, 1, 1, 256, 1, 1, 300 * 12, NULL, a2, NULL));
                CK(cuCtxSynchronize_()); double t3 = now_ms();
                printf("  warm n=300 hd=%d: scores %.1f us, softmax+pv %.1f us\n", hd, (t2 - t1) * 10, (t3 - t2) * 10);
            }
        }
    }
    free(K); free(V);
    return bad;
}

static int test_f16(void) {
    const tensor_t *t = tens("per_layer_model_proj.weight");
    int cols = (int)t->dims[0], rows = (int)t->dims[1];
    const uint16_t *w = (const uint16_t *)t->data;
    uint16_t *wt = malloc((size_t)rows * cols * 2);
    for (int r = 0; r < rows; r++) for (int k = 0; k < cols; k++) wt[(size_t)k * rows + r] = w[(size_t)r * cols + k];
    uint64_t dT, dXf; CK(cuMemAlloc_(&dT, (size_t)rows * cols * 2)); CK(cuMemAlloc_(&dXf, (size_t)cols * 4));
    CK(cuMemcpyHtoD_(dT, wt, (size_t)rows * cols * 2));
    static float x[H], ycpu[NL * PLE], ygpu[NL * PLE];
    int bad = 0;
    for (int kind = 0; kind < 3; kind++) {
        make_x(x, cols, kind);
        gemv_f16(t, x, ycpu);
        CK(cuMemcpyHtoD_(dXf, x, (size_t)cols * 4));
        int irows = rows, icols = cols;
        void *args[] = {&dT, &dXf, &dY, &irows, &icols};
        CK(cuLaunchKernel_(ff16, (rows + 127) / 128, 1, 1, 128, 1, 1, 0, NULL, args, NULL));
        CK(cuCtxSynchronize_());
        CK(cuMemcpyDtoH_(ygpu, dY, (size_t)rows * 4));
        int d = first_diff(ycpu, ygpu, rows);
        printf("  per_layer_model_proj x%d: %s\n", kind, d < 0 ? "8960/8960 rows identical" : "MISMATCH");
        if (d >= 0) { bad++; printf("    row %d cpu %.9g gpu %.9g\n", d, ycpu[d], ygpu[d]); }
        void *a2[] = {&dT, &dXf, &dY, &irows, &icols};
        CK(cuLaunchKernel_(ff162, (rows / 2 + 255) / 256, 1, 1, 256, 1, 1, (unsigned)cols * 4, NULL, a2, NULL));
        CK(cuCtxSynchronize_());
        CK(cuMemcpyDtoH_(ygpu, dY, (size_t)rows * 4));
        d = first_diff(ycpu, ygpu, rows);
        if (d >= 0) { bad++; printf("    f16_gemv_t2 row %d cpu %.9g gpu %.9g\n", d, ycpu[d], ygpu[d]); }
    }
    {   /* batched: 8 tokens (and 5, partial), each against gemv_f16 */
        static float xb[8 * H], yb[8 * NL * PLE];
        uint64_t dXb; CK(cuMemAlloc_(&dXb, sizeof xb));
        for (int B = 8; B >= 5; B -= 3) {
            for (int b = 0; b < B; b++) make_x(xb + b * cols, cols, b % 3);
            CK(cuMemcpyHtoD_(dXb, xb, (size_t)B * cols * 4));
            int irows = rows, icols = cols, ib = B;
            void *a[] = {&dT, &dXb, &dY, &irows, &icols, &ib};
            CK(cuLaunchKernel_(ff16b, (rows / 2 + 255) / 256, 1, 1, 256, 1, 1, 0, NULL, a, NULL));
            CK(cuCtxSynchronize_());
            CK(cuMemcpyDtoH_(yb, dY, (size_t)B * rows * 4));
            for (int b = 0; b < B; b++) {
                gemv_f16(t, xb + b * cols, ycpu);
                int d = first_diff(ycpu, yb + (size_t)b * rows, rows);
                if (d >= 0) { bad++; printf("    f16_gemv_tb B %d token %d row %d\n", B, b, d); }
            }
        }
        printf("  f16_gemv_t2 + f16_gemv_tb (8 and 5 tokens): %s\n", bad ? "MISMATCH" : "identical");
        if (getenv("F16PERF")) {
            int irows = rows, icols = cols, ib = 8;
            void *a1[] = {&dT, &dXf, &dY, &irows, &icols};
            void *a[] = {&dT, &dXb, &dY, &irows, &icols, &ib};
            for (int rep = 0; rep < 2; rep++) {
                double t0 = now_ms();
                for (int i = 0; i < 20; i++) CK(cuLaunchKernel_(ff16, (rows + 127) / 128, 1, 1, 128, 1, 1, 0, NULL, a1, NULL));
                CK(cuCtxSynchronize_()); double t1 = now_ms();
                for (int i = 0; i < 20; i++) CK(cuLaunchKernel_(ff162, (rows / 2 + 255) / 256, 1, 1, 256, 1, 1, (unsigned)cols * 4, NULL, a1, NULL));
                CK(cuCtxSynchronize_()); double t2 = now_ms();
                for (int i = 0; i < 20; i++) CK(cuLaunchKernel_(ff16b, (rows / 2 + 255) / 256, 1, 1, 256, 1, 1, 0, NULL, a, NULL));
                CK(cuCtxSynchronize_()); double t3 = now_ms();
                printf("  f16 time: v1 %.3f ms, t2 %.3f ms, tb %.3f ms per 8 tokens (%.3f per token)\n", (t1 - t0) / 20, (t2 - t1) / 20, (t3 - t2) / 20, (t3 - t2) / 160);
            }
        }
        CK(cuMemFree_(dXb));
    }
    CK(cuMemFree_(dT)); CK(cuMemFree_(dXf)); free(wt);
    return bad;
}


/* perf mode: kernel-only time per shape (20 launches after a warm-up), and the
   per-token GPU estimate from the layer schedule. Not a gate. */
static unsigned smem_bytes;
static double time_launch(void *f, unsigned gx, unsigned bx, unsigned by, void **args) {
    CK(cuLaunchKernel_(f, gx, 1, 1, bx, by, 1, smem_bytes, NULL, args, NULL)); CK(cuCtxSynchronize_());
    double t0 = now_ms();
    for (int i = 0; i < 20; i++) CK(cuLaunchKernel_(f, gx, 1, 1, bx, by, 1, smem_bytes, NULL, args, NULL));
    CK(cuCtxSynchronize_());
    return (now_ms() - t0) / 20;
}
static int use_q4i;
static double perf_q4(const char *name) {
    const tensor_t *t = tens(name);
    uint32_t r0 = 0, r1 = (uint32_t)t->dims[1], nb = (uint32_t)t->dims[0] / 32, rf = r1;
    double ms;
    if (use_q4i) { int rows = (int)r1, cols = (int)nb * 32; void *a[] = {&dWi, q4i_v5() ? &dSh : &dSi, &dX, &dY, &rows, &cols};
                   smem_bytes = (getenv("Q4I") && getenv("Q4I")[0] < '3') ? 0 : (unsigned)(cols * 2 + cols / 32 * 4 + cols / 256 * 4);
                   ms = time_launch(fq4i, (rows + 255) / 256, 32, 8, a); smem_bytes = 0; }
    else { void *args[] = {&dW, &dS, &dX, &dY, &r0, &r1, &nb, &rf}; ms = time_launch(fq4, (r1 + 31) / 32, 32, 8, args); }
    double gb = (double)r1 * nb * 20 / 1e9;   /* 16 B nibbles + 4 B f32 scale per block */
    printf("  %-28s %6u x %-6u %7.3f ms  %5.2f GB/s\n", name, r1, nb * 32, ms, gb / (ms / 1e3));
    return ms;
}
static void perf(void) {
    printf("kernel times (weights only; contents irrelevant for timing):\n");
    double s_att = perf_q4("blk.0.attn_q.weight") + perf_q4("blk.0.attn_k.weight") + perf_q4("blk.0.attn_v.weight") + perf_q4("blk.0.attn_output.weight");
    double f_att = perf_q4("blk.4.attn_q.weight") + perf_q4("blk.4.attn_k.weight") + perf_q4("blk.4.attn_v.weight") + perf_q4("blk.4.attn_output.weight");
    double sh_s = perf_q4("blk.15.attn_q.weight") + perf_q4("blk.15.attn_output.weight");
    double sh_f = perf_q4("blk.19.attn_q.weight") + perf_q4("blk.19.attn_output.weight");
    double mlp1 = perf_q4("blk.0.ffn_gate.weight") * 2 + perf_q4("blk.0.ffn_down.weight");
    double mlp2 = perf_q4("blk.20.ffn_gate.weight") * 2 + perf_q4("blk.20.ffn_down.weight");
    double pl = perf_q4("blk.0.inp_gate.weight") + perf_q4("blk.0.proj.weight");
    const tensor_t *hd = tens("token_embd.weight");
    uint64_t dH; CK(cuMemAlloc_(&dH, (size_t)hd->dims[1] * 6 * 210)); CK(cuMemcpyHtoD_(dH, hd->data, (size_t)hd->dims[1] * 6 * 210));
    int irows = (int)hd->dims[1], inb = 6;
    void *ah[] = {&dH, &dX, &dY, &irows, &inb};
    double head1 = time_launch(fq6, (irows + 7) / 8, 32, 8, ah);
    printf("  %-28s %6d x %-6d %7.3f ms  %5.2f GB/s\n", "head Q6_K v1", irows, 1536, head1, (double)irows * 1260 / 1e9 / (head1 / 1e3));
    CK(cuMemFree_(dH));
    CK(cuMemAlloc_(&dH, (size_t)irows * 6 * 212));
    { uint8_t *rp = repack_q6_212(hd); CK(cuMemcpyHtoD_(dH, rp, (size_t)irows * 6 * 212)); free(rp); }
    void *ah2[] = {&dH, &dX, &dY, &irows};
    double head2 = time_launch(fq62, (irows + 31) / 32, 32, 8, ah2);
    printf("  %-28s %6d x %-6d %7.3f ms  %5.2f GB/s\n", "head Q6_K v2", irows, 1536, head2, (double)irows * 1272 / 1e9 / (head2 / 1e3));
    double head3 = time_launch(fq63, (irows + 31) / 32, 32, 8, ah2);
    printf("  %-28s %6d x %-6d %7.3f ms  %5.2f GB/s\n", "head Q6_K v3", irows, 1536, head3, (double)irows * 1272 / 1e9 / (head3 / 1e3));
    double head = time_launch(fq64, (irows + 31) / 32, 32, 8, ah2);
    printf("  %-28s %6d x %-6d %7.3f ms  %5.2f GB/s\n", "head Q6_K v4", irows, 1536, head, (double)irows * 1272 / 1e9 / (head / 1e3));
    CK(cuMemFree_(dH));
    CK(cuMemAlloc_(&dH, (size_t)irows * 6 * 224));
    { uint8_t *rp = repack_q6i(hd); CK(cuMemcpyHtoD_(dH, rp, (size_t)irows * 6 * 224)); free(rp); }
    void *ahi[] = {&dH, &dX, &dY, &irows};
    double headi = time_launch(fq6i, (irows + 255) / 256, 32, 8, ahi);
    printf("  %-28s %6d x %-6d %7.3f ms  %5.2f GB/s\n", "head Q6_K q6i", irows, 1536, headi, (double)irows * 1344 / 1e9 / (headi / 1e3));
    CK(cuMemFree_(dH));
    /* schedule: layers 0-14 own KV (12 sliding, 3 full); 15-34 shared (16 sliding, 4 full) */
    double tot = 12 * s_att + 3 * f_att + 16 * sh_s + 4 * sh_f + 15 * mlp1 + 20 * mlp2 + 35 * pl + head;
    printf("per-token GPU GEMV estimate: %.1f ms (layers %.1f + head %.1f), %.1f tok/s ceiling from GEMVs alone\n",
           tot, tot - head, head, 1000 / tot);
}

/* perfb: batched (B = 8) kernel time vs the single-token kernel, per token */
static void perfb(void) {
    enum { B = 8 };
    static uint8_t xqb[B * (2 * INTER / 256) * Q8B];
    static float x[2 * INTER];
    uint64_t dXb, dYb; CK(cuMemAlloc_(&dXb, sizeof xqb)); CK(cuMemAlloc_(&dYb, (size_t)B * VOCAB * 4));
    const char *names[] = {"blk.0.attn_q.weight", "blk.0.attn_output.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight",
                           "blk.20.ffn_gate.weight", "blk.20.ffn_down.weight", "blk.0.proj.weight"};
    double s1 = 0, sb = 0;
    for (unsigned k = 0; k < sizeof names / sizeof *names; k++) {
        const tensor_t *t = tens(names[k]);
        int cols = (int)t->dims[0], rows = (int)t->dims[1], nb = cols / 32, xs = (cols / 256) * Q8B;
        uint8_t *nib = malloc((size_t)rows * nb * 16); float *scl = malloc((size_t)rows * nb * 4);
        upload_q4i(t);
        free(nib); free(scl);
        for (int b = 0; b < B; b++) { make_x(x, cols, 0); q8k(x, xqb + (size_t)b * xs, cols); }
        CK(cuMemcpyHtoD_(dXb, xqb, (size_t)B * xs));
        CK(cuMemcpyHtoD_(dX, xqb, (size_t)xs));
        int ib = B;
        void *a[] = {&dWi, getenv("Q4B1") ? &dSi : &dSh, &dXb, &dYb, &rows, &cols, &ib, &xs, &rows};
        double tb = time_launch(fq4ib, (rows + 255) / 256, 32, 8, a);
        void *a1[] = {&dWi, q4i_v5() ? &dSh : &dSi, &dX, &dY, &rows, &cols};
        smem_bytes = (unsigned)(cols * 2 + cols / 32 * 4 + cols / 256 * 4);
        double t1 = time_launch(fq4i, (rows + 255) / 256, 32, 8, a1);
        smem_bytes = 0;
        printf("  %-26s single %.3f ms/token, batched %.3f ms/token (x%.2f)\n", names[k], t1, tb / B, t1 / (tb / B));
        s1 += t1; sb += tb / B;
    }
    const tensor_t *t = tens("token_embd.weight");
    int rows = (int)t->dims[1], xs = 6 * Q8B, ib = B;
    uint8_t *rp = repack_q6i(t);
    uint64_t dHi; CK(cuMemAlloc_(&dHi, (size_t)rows * 6 * 224)); CK(cuMemcpyHtoD_(dHi, rp, (size_t)rows * 6 * 224)); free(rp);
    for (int b = 0; b < B; b++) { make_x(x, H, 0); q8k(x, xqb + (size_t)b * xs, H); }
    CK(cuMemcpyHtoD_(dXb, xqb, (size_t)B * xs)); CK(cuMemcpyHtoD_(dX, xqb, (size_t)xs));
    void *a[] = {&dHi, &dXb, &dYb, &rows, &ib, &xs, &rows};
    double tb = time_launch(fq6ib, (rows + 255) / 256, 32, 8, a);
    void *a1[] = {&dHi, &dX, &dY, &rows};
    double t1 = time_launch(fq6i, (rows + 255) / 256, 32, 8, a1);
    printf("  %-26s single %.3f ms/token, batched %.3f ms/token (x%.2f)\n", "head", t1, tb / B, t1 / (tb / B));
}


int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <gguf> <g4_kernels.ptx> [q4|head|f16|all|perf]\n", argv[0]); return 2; }
    const char *what = argc > 3 ? argv[3] : "all";
    gguf_open(argv[1]);
    void *h = dlopen("libcuda.so.1", RTLD_NOW);
    if (!h) { fprintf(stderr, "dlopen libcuda: %s\n", dlerror()); return 3; }
    cuInit_ = sym(h, "cuInit"); cuDeviceGet_ = sym(h, "cuDeviceGet"); cuCtxCreate_ = sym(h, "cuCtxCreate_v2");
    cuModuleLoadData_ = sym(h, "cuModuleLoadData"); cuModuleGetFunction_ = sym(h, "cuModuleGetFunction");
    cuMemAlloc_ = sym(h, "cuMemAlloc_v2"); cuMemFree_ = sym(h, "cuMemFree_v2");
    cuMemcpyHtoD_ = sym(h, "cuMemcpyHtoD_v2"); cuMemcpyDtoH_ = sym(h, "cuMemcpyDtoH_v2");
    cuLaunchKernel_ = sym(h, "cuLaunchKernel"); cuCtxSynchronize_ = sym(h, "cuCtxSynchronize");
    int dev; void *ctx, *m4, *m6;
    CK(cuInit_(0)); CK(cuDeviceGet_(&dev, 0)); CK(cuCtxCreate_(&ctx, 0, dev));
    CK(cuModuleLoadData_(&m4, G4_PTX_Q4)); CK(cuModuleGetFunction_(&fq4, m4, "q4_gemv"));
    { FILE *f = fopen(argv[2], "rb"); if (!f) { perror(argv[2]); return 2; }
      static char ptx[1 << 20]; size_t n = fread(ptx, 1, sizeof ptx - 1, f); fclose(f); ptx[n] = 0;
      CK(cuModuleLoadData_(&m6, ptx)); }
    CK(cuModuleGetFunction_(&fq6, m6, "q6k_gemv")); CK(cuModuleGetFunction_(&fq62, m6, "q6k_gemv2")); CK(cuModuleGetFunction_(&fq63, m6, "q6k_gemv3")); CK(cuModuleGetFunction_(&fq64, m6, "q6k_gemv4")); CK(cuModuleGetFunction_(&fq6i, m6, "q6i_gemv")); CK(cuModuleGetFunction_(&fq4ib, m6, getenv("Q4B1") ? "q4i_gemv_b" : "q4i_gemv_b2")); CK(cuModuleGetFunction_(&fq6ib, m6, "q6i_gemv_b")); CK(cuModuleGetFunction_(&fgeglub, m6, "geglu_b")); CK(cuModuleGetFunction_(&fargb, m6, "argmax_b")); CK(cuModuleGetFunction_(&fasc, m6, "attn_scores")); CK(cuModuleGetFunction_(&fgq2, m6, "geglu_q8k2")); CK(cuModuleGetFunction_(&ff162, m6, "f16_gemv_t2")); CK(cuModuleGetFunction_(&ff16b, m6, "f16_gemv_tb")); CK(cuModuleGetFunction_(&fapv, m6, "attn_softmax_pv")); CK(cuModuleGetFunction_(&fq4i, m6, getenv("Q4I") ? (getenv("Q4I")[0] == '1' ? "q4i_gemv" : getenv("Q4I")[0] == '2' ? "q4i_gemv2" : getenv("Q4I")[0] == '3' ? "q4i_gemv3" : getenv("Q4I")[0] == '5' ? "q4i_gemv5" : "q4i_gemv4") : "q4i_gemv5")); CK(cuModuleGetFunction_(&fgeglu, m6, "geglu_dev")); CK(cuModuleGetFunction_(&fq8k, m6, "q8k_dev")); CK(cuModuleGetFunction_(&ff16, m6, "f16_gemv_t"));
    CK(cuMemAlloc_(&dW, (size_t)2 * INTER * (H / 32) * 16)); CK(cuMemAlloc_(&dS, (size_t)2 * INTER * (H / 32) * 4));
    CK(cuMemAlloc_(&dX, sizeof XQ));
    CK(cuMemAlloc_(&dWi, (size_t)2 * INTER * (H / 32) * 16)); CK(cuMemAlloc_(&dSi, (size_t)2 * INTER * (H / 32) * 4)); CK(cuMemAlloc_(&dSh, (size_t)2 * INTER * (H / 32) * 2)); CK(cuMemAlloc_(&dY, (size_t)VOCAB * 4));
    int fails = 0;
    if (!strcmp(what, "q4") || !strcmp(what, "all")) {
        int n = 0, bad = 0; double t0 = now_ms();
        for (int i = 0; i < NT; i++) if (T[i].type == 2) { n++; int b = test_q4(&T[i]); bad += b != 0; }
        printf("Q4_0: %d/%d tensors bit-identical on 3 inputs each (%.0f s)\n", n - bad, n, (now_ms() - t0) / 1e3);
        fails += bad;
    }
    if (!strcmp(what, "head") || !strcmp(what, "all")) fails += test_head();
    if (!strcmp(what, "f16") || !strcmp(what, "all")) fails += test_f16();
    if (!strcmp(what, "mlp") || !strcmp(what, "all")) fails += test_mlp_stages();
    if (!strcmp(what, "batch") || !strcmp(what, "all")) fails += test_batch();
    if (!strcmp(what, "attn") || !strcmp(what, "all")) fails += test_attn();
    if (!strcmp(what, "perf")) { perf(); return 0; }
    if (!strcmp(what, "perfi")) { use_q4i = 1; perf(); return 0; }
    if (!strcmp(what, "perfb")) { perfb(); perfb(); return 0; }
    printf(fails ? "M3 GATE: FAIL (%d)\n" : "M3 GATE: OK\n", fails);
    return fails ? 1 : 0;
}
