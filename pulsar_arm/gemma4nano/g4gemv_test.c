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
 * Run:   ./g4gemv_test <gemma.gguf> <g4_kernels.ptx> [q4|head|f16|all]
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

static void *fq4, *fq6, *ff16;
static uint64_t dW, dS, dX, dY;

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
    }
    free(nib); free(scl);
    return bad;
}

static int test_head(void) {
    const tensor_t *t = tens("token_embd.weight");
    int cols = (int)t->dims[0], rows = (int)t->dims[1], nb = cols / 256;
    uint64_t dH; CK(cuMemAlloc_(&dH, (size_t)rows * nb * 210));
    CK(cuMemcpyHtoD_(dH, t->data, (size_t)rows * nb * 210));
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
        int d = first_diff(ycpu, ygpu, rows);
        printf("  head x%d: %s  (cpu ref %.0f ms, gpu kernel %.1f ms)\n", kind,
               d < 0 ? "262144/262144 rows identical" : "MISMATCH", t1 - t0, t3 - t2);
        if (d >= 0) { bad++; printf("    row %d cpu %.9g gpu %.9g\n", d, ycpu[d], ygpu[d]); }
    }
    CK(cuMemFree_(dH)); free(ycpu); free(ygpu);
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
    }
    CK(cuMemFree_(dT)); CK(cuMemFree_(dXf)); free(wt);
    return bad;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <gguf> <g4_kernels.ptx> [q4|head|f16|all]\n", argv[0]); return 2; }
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
    CK(cuModuleGetFunction_(&fq6, m6, "q6k_gemv")); CK(cuModuleGetFunction_(&ff16, m6, "f16_gemv_t"));
    CK(cuMemAlloc_(&dW, (size_t)2 * INTER * (H / 32) * 16)); CK(cuMemAlloc_(&dS, (size_t)2 * INTER * (H / 32) * 4));
    CK(cuMemAlloc_(&dX, sizeof XQ)); CK(cuMemAlloc_(&dY, (size_t)VOCAB * 4));
    int fails = 0;
    if (!strcmp(what, "q4") || !strcmp(what, "all")) {
        int n = 0, bad = 0; double t0 = now_ms();
        for (int i = 0; i < NT; i++) if (T[i].type == 2) { n++; int b = test_q4(&T[i]); bad += b != 0; }
        printf("Q4_0: %d/%d tensors bit-identical on 3 inputs each (%.0f s)\n", n - bad, n, (now_ms() - t0) / 1e3);
        fails += bad;
    }
    if (!strcmp(what, "head") || !strcmp(what, "all")) fails += test_head();
    if (!strcmp(what, "f16") || !strcmp(what, "all")) fails += test_f16();
    printf(fails ? "M3 GATE: FAIL (%d)\n" : "M3 GATE: OK\n", fails);
    return fails ? 1 : 0;
}
