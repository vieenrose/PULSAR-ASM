/* g4eng.c - Gemma 4 E2B (text only) on the Jetson Nano: host + CUDA (driver API).
 *
 * The forward pass is g4ref.c's own forward() (included), with its three GEMV
 * hooks pointed at the GPU kernels that g4gemv_test proved bit-identical to the
 * reference (K2's q4_gemv, q6k_gemv4, f16_gemv_t). So the engine's output must
 * equal golden_en.txt / golden_zh.txt byte for byte; that is the M4 gate.
 *
 * This first version keeps every CPU stage in C and uses plain synchronous
 * copies per GEMV: correctness first, the K2 driver-path tricks and assembly come
 * after profiling.
 *
 * Build (Nano, driver only):
 *   gcc -O2 -fopenmp -ffp-contract=off -o g4eng g4eng.c -ldl -lm
 * Run:   ./g4eng <gemma.gguf> <rope.bin> <ids...> [--gen N]
 */
#define G4REF_NO_MAIN
#include "g4ref.c"
#include "g4_ptx_q4.h"
#include "g4_ptx_k.h"
#include <dlfcn.h>
#include <time.h>

typedef int CUresult;
static CUresult (*cuInit_)(unsigned);
static CUresult (*cuDeviceGet_)(int *, int);
static CUresult (*cuCtxCreate_)(void **, unsigned, int);
static CUresult (*cuModuleLoadData_)(void **, const void *);
static CUresult (*cuModuleGetFunction_)(void **, void *, const char *);
static CUresult (*cuMemAlloc_)(uint64_t *, size_t);
static CUresult (*cuMemcpyHtoD_)(uint64_t, const void *, size_t);
static CUresult (*cuMemcpyDtoH_)(void *, uint64_t, size_t);
static CUresult (*cuLaunchKernel_)(void *, unsigned, unsigned, unsigned, unsigned, unsigned,
                                   unsigned, unsigned, void *, void **, void **);
static CUresult (*cuCtxSynchronize_)(void);
static CUresult (*cuMemGetInfo_)(size_t *, size_t *);

#define CK(x) do { CUresult r_ = (x); if (r_) { fprintf(stderr, "%s:%d %s -> %d\n", __FILE__, __LINE__, #x, r_); exit(3); } } while (0)
static void *sym(void *h, const char *n) { void *p = dlsym(h, n); if (!p) { fprintf(stderr, "no %s\n", n); exit(3); } return p; }
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

static void *fq4, *fq6, *ff16;
static uint64_t DW[600], DS[600];          /* per tensor index: nibbles, f32 scales */
static uint64_t dX, dXf, dY, dHead, dF16;
static size_t dev_bytes;
/* G4PROF=1: per-hook wall time split (H2D, launch+sync, D2H), printed at exit */
static int prof;
static double P_h2d[3], P_k[3], P_d2h[3]; static long P_n[3];

static void drop_pages(const void *p, size_t n) {   /* the GGUF copy is on the GPU now */
    long ps = sysconf(_SC_PAGESIZE);
    uintptr_t a = ((uintptr_t)p + ps - 1) & ~(uintptr_t)(ps - 1), e = ((uintptr_t)p + n) & ~(uintptr_t)(ps - 1);
    if (e > a) madvise((void *)a, e - a, MADV_DONTNEED);
}
static uint64_t dalloc(size_t n) { uint64_t d; CK(cuMemAlloc_(&d, n)); dev_bytes += n; return d; }

/* ---------------------------------------------------------------- hooks */
static void gpu_q4(const tensor_t *t, const uint8_t *xq, float *y) {
    int i = (int)(t - T);
    uint32_t cols = (uint32_t)t->dims[0], r0 = 0, r1 = (uint32_t)t->dims[1], nb = cols / 32, rf = r1;
    double a_ = prof ? now_ms() : 0;
    CK(cuMemcpyHtoD_(dX, xq, (size_t)(cols / 256) * Q8B));
    void *args[] = {&DW[i], &DS[i], &dX, &dY, &r0, &r1, &nb, &rf};
    double b_ = prof ? now_ms() : 0;
    CK(cuLaunchKernel_(fq4, (r1 + 31) / 32, 1, 1, 32, 8, 1, 0, NULL, args, NULL));
    CK(cuCtxSynchronize_());
    double c_ = prof ? now_ms() : 0;
    CK(cuMemcpyDtoH_(y, dY, (size_t)r1 * 4));
    if (prof) { double d_ = now_ms(); P_h2d[0] += b_ - a_; P_k[0] += c_ - b_; P_d2h[0] += d_ - c_; P_n[0]++; }
}
/* n tensors laid out back to back on the device (q+k+v, gate+up): one launch over
   the concatenated rows, one D2H, then split. Same kernel and arithmetic per row. */
static float YCAT[2 * 2 * INTER];
static void gpu_q4n(int n, const tensor_t **ts, const uint8_t *xq, float **ys) {
    uint32_t rows = 0;
    for (int k = 0; k < n; k++) {
        int i = (int)(ts[k] - T);
        size_t nb_k = ts[k]->dims[0] / 32;
        if (ts[k]->dims[0] != ts[0]->dims[0] ||
            DW[i] != DW[ts[0] - T] + (size_t)rows * nb_k * 16 || DS[i] != DS[ts[0] - T] + (size_t)rows * nb_k * 4) {
            for (int j = 0; j < n; j++) gpu_q4(ts[j], xq, ys[j]);    /* not contiguous: one by one */
            return;
        }
        rows += (uint32_t)ts[k]->dims[1];
    }
    int i0 = (int)(ts[0] - T);
    uint32_t cols = (uint32_t)ts[0]->dims[0], r0 = 0, r1 = rows, nb = cols / 32, rf = rows;
    double a_ = prof ? now_ms() : 0;
    CK(cuMemcpyHtoD_(dX, xq, (size_t)(cols / 256) * Q8B));
    void *args[] = {&DW[i0], &DS[i0], &dX, &dY, &r0, &r1, &nb, &rf};
    double b_ = prof ? now_ms() : 0;
    CK(cuLaunchKernel_(fq4, (r1 + 31) / 32, 1, 1, 32, 8, 1, 0, NULL, args, NULL));
    CK(cuCtxSynchronize_());
    double c_ = prof ? now_ms() : 0;
    CK(cuMemcpyDtoH_(YCAT, dY, (size_t)rows * 4));
    for (int k = 0, off = 0; k < n; off += (int)ts[k]->dims[1], k++) memcpy(ys[k], YCAT + off, ts[k]->dims[1] * 4);
    if (prof) { double d_ = now_ms(); P_h2d[0] += b_ - a_; P_k[0] += c_ - b_; P_d2h[0] += d_ - c_; P_n[0]++; }
}
static void gpu_head(const tensor_t *t, const uint8_t *xq, float *y) {
    int rows = (int)t->dims[1];
    double a_ = prof ? now_ms() : 0;
    CK(cuMemcpyHtoD_(dX, xq, (size_t)(t->dims[0] / 256) * Q8B));
    void *args[] = {&dHead, &dX, &dY, &rows};
    double b_ = prof ? now_ms() : 0;
    CK(cuLaunchKernel_(fq6, (rows + 31) / 32, 1, 1, 32, 8, 1, 0, NULL, args, NULL));
    CK(cuCtxSynchronize_());
    double c_ = prof ? now_ms() : 0;
    CK(cuMemcpyDtoH_(y, dY, (size_t)rows * 4));
    if (prof) { double d_ = now_ms(); P_h2d[1] += b_ - a_; P_k[1] += c_ - b_; P_d2h[1] += d_ - c_; P_n[1]++; }
}
static void gpu_f16(const tensor_t *t, const float *x, float *y) {
    int rows = (int)t->dims[1], cols = (int)t->dims[0];
    double a_ = prof ? now_ms() : 0;
    CK(cuMemcpyHtoD_(dXf, x, (size_t)cols * 4));
    void *args[] = {&dF16, &dXf, &dY, &rows, &cols};
    double b_ = prof ? now_ms() : 0;
    CK(cuLaunchKernel_(ff16, (rows + 127) / 128, 1, 1, 128, 1, 1, 0, NULL, args, NULL));
    CK(cuCtxSynchronize_());
    double c_ = prof ? now_ms() : 0;
    CK(cuMemcpyDtoH_(y, dY, (size_t)rows * 4));
    if (prof) { double d_ = now_ms(); P_h2d[2] += b_ - a_; P_k[2] += c_ - b_; P_d2h[2] += d_ - c_; P_n[2]++; }
}

/* ----------------------------------------------------------------- load */
static void gpu_init(void) {
    double t0 = now_ms();
    void *h = dlopen("libcuda.so.1", RTLD_NOW);
    if (!h) { fprintf(stderr, "dlopen libcuda: %s\n", dlerror()); exit(3); }
    cuInit_ = sym(h, "cuInit"); cuDeviceGet_ = sym(h, "cuDeviceGet"); cuCtxCreate_ = sym(h, "cuCtxCreate_v2");
    cuModuleLoadData_ = sym(h, "cuModuleLoadData"); cuModuleGetFunction_ = sym(h, "cuModuleGetFunction");
    cuMemAlloc_ = sym(h, "cuMemAlloc_v2"); cuMemcpyHtoD_ = sym(h, "cuMemcpyHtoD_v2"); cuMemcpyDtoH_ = sym(h, "cuMemcpyDtoH_v2");
    cuLaunchKernel_ = sym(h, "cuLaunchKernel"); cuCtxSynchronize_ = sym(h, "cuCtxSynchronize");
    cuMemGetInfo_ = sym(h, "cuMemGetInfo_v2");
    int dev; void *ctx, *m4, *mk;
    CK(cuInit_(0)); CK(cuDeviceGet_(&dev, 0)); CK(cuCtxCreate_(&ctx, 0, dev));
    CK(cuModuleLoadData_(&m4, G4_PTX_Q4)); CK(cuModuleGetFunction_(&fq4, m4, "q4_gemv"));
    CK(cuModuleLoadData_(&mk, G4_PTX_K));
    CK(cuModuleGetFunction_(&fq6, mk, "q6k_gemv4")); CK(cuModuleGetFunction_(&ff16, mk, "f16_gemv_t"));
    dX = dalloc(sizeof XQ); dXf = dalloc((size_t)H * 4); dY = dalloc((size_t)VOCAB * 4);
    /* Q4_0 layers: nibbles [rows][nb][16] + f32 scales [rows][nb] (K2 layout).
       q+k+v and gate+up of each layer get one allocation each, back to back, so
       gpu_q4n can cover them with a single launch. */
    int nq4 = 0;
    for (int il = 0; il < NL; il++) {
        const layer_t *L = &LT[il];
        const tensor_t *grp[6][3] = {{L->q, L->k, L->v}, {L->o}, {L->gate, L->up}, {L->down}, {L->ig}, {L->proj}};
        for (int g = 0; g < 6; g++) {
            size_t tot = 0;
            for (int k = 0; k < 3 && grp[g][k]; k++) tot += (size_t)grp[g][k]->dims[1] * (grp[g][k]->dims[0] / 32);
            uint64_t bw = dalloc(tot * 16), bs = dalloc(tot * 4);
            size_t off = 0;
            for (int k = 0; k < 3 && grp[g][k]; k++) {
                const tensor_t *t = grp[g][k];
                int i = (int)(t - T);
                size_t nblk = (size_t)t->dims[1] * (t->dims[0] / 32);
                uint8_t *nib = malloc(nblk * 16); float *scl = malloc(nblk * 4);
                for (size_t b = 0; b < nblk; b++) {
                    const uint8_t *blk = t->data + b * 18;
                    scl[b] = h2f((uint16_t)(blk[0] | blk[1] << 8));
                    memcpy(nib + b * 16, blk + 2, 16);
                }
                DW[i] = bw + off * 16; DS[i] = bs + off * 4;
                CK(cuMemcpyHtoD_(DW[i], nib, nblk * 16)); CK(cuMemcpyHtoD_(DS[i], scl, nblk * 4));
                free(nib); free(scl);
                drop_pages(t->data, nblk * 18);
                off += nblk; nq4++;
            }
        }
    }
    /* Q6_K head, blocks padded 210 -> 212 bytes (q6k_gemv4 layout) */
    {
        const tensor_t *t = tens("token_embd.weight");
        size_t nblk = (size_t)t->dims[1] * (t->dims[0] / 256);
        uint8_t *rp = calloc(nblk, 212);
        for (size_t b = 0; b < nblk; b++) memcpy(rp + b * 212, t->data + b * 210, 210);
        dHead = dalloc(nblk * 212); CK(cuMemcpyHtoD_(dHead, rp, nblk * 212)); free(rp);
        /* not dropped: the embedding row of each token is still read on the host */
    }
    /* F16 per_layer_model_proj, transposed */
    {
        const tensor_t *t = tens("per_layer_model_proj.weight");
        int cols = (int)t->dims[0], rows = (int)t->dims[1];
        const uint16_t *w = (const uint16_t *)t->data;
        uint16_t *wt = malloc((size_t)rows * cols * 2);
        for (int r = 0; r < rows; r++) for (int k = 0; k < cols; k++) wt[(size_t)k * rows + r] = w[(size_t)r * cols + k];
        dF16 = dalloc((size_t)rows * cols * 2); CK(cuMemcpyHtoD_(dF16, wt, (size_t)rows * cols * 2)); free(wt);
        drop_pages(t->data, (size_t)rows * cols * 2);
    }
    size_t fr = 0, tot = 0; cuMemGetInfo_(&fr, &tot);
    fprintf(stderr, "GPU: %d Q4_0 tensors + Q6_K head + F16 proj, %.1f MB on device, %.0f MB free of %.0f; load %.0f ms\n",
            nq4, dev_bytes / 1048576.0, fr / 1048576.0, tot / 1048576.0, now_ms() - t0);
    G_Q4 = gpu_q4; G_Q4N = gpu_q4n; G_HEAD = gpu_head; G_F16 = gpu_f16;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <gguf> <rope.bin> <ids...> [--gen N]\n", argv[0]); return 2; }
    int gen = 0, n = 0;
    static int ids[MAXSEQ];
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--gen") && i + 1 < argc) { gen = atoi(argv[++i]); continue; }
        ids[n++] = atoi(argv[i]);
    }
    gguf_open(argv[1]);
    { int c = 0; for (int i = 0; i < NL; i++) c += !SLIDING[i];
      if (c != 7) { fprintf(stderr, "sliding pattern: %d full layers, want 7\n", c); return 2; } }
    layer_table();
    { size_t want = (size_t)MAXSEQ * (128 + 256) * 2 * 4;
      ROPE = malloc(want); FILE *f = fopen(argv[2], "rb");
      if (!f || fread(ROPE, 1, want, f) != want) { fprintf(stderr, "rope table %s: want %zu bytes\n", argv[2], want); return 2; }
      fclose(f); }
    prof = getenv("G4PROF") && getenv("G4PROF")[0] == '1';
    gpu_init();
    int pos = 0, n_prompt = n;
    double tp = 0, tg = 0;
    while (pos < MAXSEQ) {
        int tok = ids[pos];
        double t0 = now_ms();
        forward(tok, pos);
        int top = 0;
        for (int v = 1; v < VOCAB; v++) if (logits[v] > logits[top]) top = v;
        double dt = now_ms() - t0;
        if (pos < n_prompt) tp += dt; else tg += dt;
        printf("step %d id_in=%d top=%d logit=%.4f\n", pos, tok, top, capped(logits[top]));
        fflush(stdout);
        pos++;
        if (pos >= n) {
            if (n - n_prompt >= gen || top == 1 || top == 106) break;
            ids[n++] = top;
        }
    }
    int ng = pos - n_prompt;
    if (prof) {
        const char *nm[3] = {"q4", "head", "f16"};
        double tg_all = tp + tg, hk = 0;
        for (int k = 0; k < 3; k++) {
            hk += P_h2d[k] + P_k[k] + P_d2h[k];
            fprintf(stderr, "PROF %-4s calls %6ld  h2d %8.0f ms  kernel+sync %8.0f ms  d2h %8.0f ms  (%.2f ms/token)\n",
                    nm[k], P_n[k], P_h2d[k], P_k[k], P_d2h[k], (P_h2d[k] + P_k[k] + P_d2h[k]) / pos);
        }
        fprintf(stderr, "PROF forward total %.0f ms over %d steps: GPU hooks %.1f ms/token, everything else (CPU stages) %.1f ms/token\n",
                tg_all, pos, hk / pos, (tg_all - hk) / pos);
    }
    fprintf(stderr, "prefill %d tokens %.0f ms (%.2f tok/s); decode %d steps %.0f ms (%.2f tok/s)\n",
            n_prompt, tp, n_prompt * 1000.0 / tp, ng, tg, ng > 0 ? ng * 1000.0 / tg : 0.0);
    return 0;
}
