/* g4eng.c - Gemma 4 E2B (text only) on the Jetson Nano: host + CUDA (driver API),
 * hot host loops in AArch64 assembly (g4_host.S).
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
 *   gcc -O2 -fopenmp -ffp-contract=off -o g4eng g4eng.c g4_host.S -ldl -lm
 * Run:   ./g4eng <gemma.gguf> <rope.bin> <ids...> [--gen N] [--batch B] [--sample temp topk topp seed]
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
static CUresult (*cuStreamCreate_)(void **, unsigned);
static CUresult (*cuEventCreate_)(void **, unsigned);
static CUresult (*cuEventRecord_)(void *, void *);
static CUresult (*cuEventQuery_)(void *);
static CUresult (*cuMemHostAlloc_)(void **, size_t, unsigned);
static CUresult (*cuMemHostGetDevicePointer_)(uint64_t *, void *, unsigned);

#define CK(x) do { CUresult r_ = (x); if (r_) { fprintf(stderr, "%s:%d %s -> %d\n", __FILE__, __LINE__, #x, r_); exit(3); } } while (0)
static void *sym(void *h, const char *n) { void *p = dlsym(h, n); if (!p) { fprintf(stderr, "no %s\n", n); exit(3); } return p; }
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

static void *fq4, *fq6, *ff16, *fxc, *fgeglu, *fq8k, *fq4i, *fq4ib, *fgeglub, *fargb, *fq6ib, *fasc, *fapv;
/* decode attention on the GPU: device copies of the kv caches (rows [0, dsync[i]) valid),
   sized for the run's positions (G_CAP) */
static uint64_t dKC[FIRST_SHARED], dVC[FIRST_SHARED], dQa, dSa, dOa;
static int dsync[FIRST_SHARED], G_CAP = MAXSEQ, pf_on = 1, gpu_attn;
static uint64_t dYdevB, dHdevB, dX2B, dLogB;   /* batched prefill intermediates (BATCH_MAX tokens) */
/* per tensor: 0 = K2 q4_gemv layout, 1 = q4i_gemv3 (q4i_gemv4: row-per-lane, tile-interleaved).
   Chosen per weight group at load: q4i where it measured faster (>= 1536 rows and
   <= 6144 columns), K2 elsewhere (k/v alone, inp_gate, down at 12288 columns). */
static unsigned char KIND[600];
static int use_q4i = 1;
static uint64_t dYdev, dHdev, dX2, dTab;    /* device-only MLP intermediates */
/* fast driver path (G4FAST, default on), the K2 Exp93-96 recipe: one non-blocking
   stream; inputs written to DEVICEMAP-pinned host memory and moved on the GPU by
   K2's xq_copy kernel; GEMV results written by the kernel straight into pinned host
   memory; completion = a timing-free event polled by the host; results read back
   with four interleaved NEON streams (pinned memory is uncached for the CPU). Only
   the transport changes; every kernel and its arithmetic are the same. */
static int fast = 1;
static void *stream, *event;
static void *h_in, *h_out;
static uint64_t h_in_dev, h_out_dev;
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


static void wait_stream(void) {
    CK(cuEventRecord_(event, stream));
    if ((PF_N[0] | PF_N[1]) && pf_on) {   /* use the GPU wait: pull the next attention's K/V into L2 */
        for (int k = 0; k < 2; k++)
            for (size_t o = 0; o < PF_N[k]; o += 64) __builtin_prefetch(PF_P[k] + o, 0, 2);
        PF_N[0] = PF_N[1] = 0;
    }
    CUresult r;
    while ((r = cuEventQuery_(event)) == 600) { }
    if (r) { fprintf(stderr, "event query %d\n", r); exit(3); }
}
static void copy_from_pinned(void *dst, const void *src, size_t n) {
    size_t q = (n / 256) * 64;
    char *d = (char *)dst; const char *s = (const char *)src;
#if !defined(__aarch64__)
    (void)d; (void)s; memcpy(dst, src, n); return;   /* host-side compile check only */
#endif
    if (q < 64) { memcpy(dst, src, n); return; }
    for (size_t i = 0; i < q; i += 64) {
#if defined(__aarch64__)
        __asm__ volatile(
            "ld1 {v0.16b-v3.16b}, [%4]\n\t" "ld1 {v4.16b-v7.16b}, [%5]\n\t"
            "ld1 {v16.16b-v19.16b}, [%6]\n\t" "ld1 {v20.16b-v23.16b}, [%7]\n\t"
            "st1 {v0.16b-v3.16b}, [%0]\n\t" "st1 {v4.16b-v7.16b}, [%1]\n\t"
            "st1 {v16.16b-v19.16b}, [%2]\n\t" "st1 {v20.16b-v23.16b}, [%3]\n\t"
            :: "r"(d + i), "r"(d + q + i), "r"(d + 2 * q + i), "r"(d + 3 * q + i),
               "r"(s + i), "r"(s + q + i), "r"(s + 2 * q + i), "r"(s + 3 * q + i)
            : "v0","v1","v2","v3","v4","v5","v6","v7","v16","v17","v18","v19","v20","v21","v22","v23","memory");
#endif
    }
    if (4 * q < n) memcpy(d + 4 * q, s + 4 * q, n - 4 * q);
}
/* stage n bytes (multiple of 4) of input into dX through the pinned buffer */
static void stage_in(const void *src, size_t n, uint64_t dst) {
    memcpy(h_in, src, n);
    uint32_t nw = (uint32_t)(n / 4);
    void *a[] = {&h_in_dev, &dst, &nw};
    CK(cuLaunchKernel_(fxc, (nw + 255) / 256, 1, 1, 256, 1, 1, 0, stream, a, NULL));
}
/* launch the Q4_0 GEMV of a weight group on the stream (K2 or q4i layout) */
static void launch_q4(int kind, uint64_t *w, uint64_t *s, uint64_t xq_dev, uint64_t y_dev, uint32_t cols, uint32_t rows) {
    if (kind) {
        int irows = (int)rows, icols = (int)cols;
        unsigned sm = cols * 2 + cols / 32 * 4 + cols / 256 * 4;
        void *a[] = {w, s, &xq_dev, &y_dev, &irows, &icols};
        CK(cuLaunchKernel_(fq4i, (rows + 255) / 256, 1, 1, 32, 8, 1, sm, stream, a, NULL));
    } else {
        uint32_t r0 = 0, r1 = rows, nb = cols / 32, rf = rows;
        void *a[] = {w, s, &xq_dev, &y_dev, &r0, &r1, &nb, &rf};
        CK(cuLaunchKernel_(fq4, (r1 + 31) / 32, 1, 1, 32, 8, 1, 0, stream, a, NULL));
    }
}
/* one Q4_0 launch over rows [0, rows) of the weights at (w, s), y -> pinned host */
static void q4_fast(int kind, uint64_t *w, uint64_t *s, uint32_t cols, uint32_t rows, const uint8_t *xq, float *y) {
    double a_ = prof ? now_ms() : 0;
    stage_in(xq, (size_t)(cols / 256) * Q8B, dX);
    double b_ = prof ? now_ms() : 0;
    launch_q4(kind, w, s, dX, h_out_dev, cols, rows);
    wait_stream();
    double c_ = prof ? now_ms() : 0;
    copy_from_pinned(y, h_out, (size_t)rows * 4);
    if (prof) { double d_ = now_ms(); P_h2d[0] += b_ - a_; P_k[0] += c_ - b_; P_d2h[0] += d_ - c_; P_n[0]++; }
}
#include "g4_hostglue.h"

/* MLP on the GPU: gate+up -> geglu_dev -> q8k_dev -> down, one wait at the end.
   XQ already holds the quantized ffn_norm output. */
static void gpu_mlp(const layer_t *L, int inter, float *out) {
    if (!fast) { mlp_ref(L, inter, out); return; }
    int ig = (int)(L->gate - T), id = (int)(L->down - T);
    double a_ = prof ? now_ms() : 0;
    stage_in(XQ, (size_t)(H / 256) * Q8B, dX);
    launch_q4(KIND[ig], &DW[ig], &DS[ig], dX, dYdev, H, 2 * inter);
    int n = inter;
    void *a2[] = {&dYdev, &dHdev, &n, &dTab};
    CK(cuLaunchKernel_(fgeglu, (n + 255) / 256, 1, 1, 256, 1, 1, 0, stream, a2, NULL));
    void *a3[] = {&dHdev, &dX2};
    CK(cuLaunchKernel_(fq8k, n / 256, 1, 1, 256, 1, 1, 0, stream, a3, NULL));
    launch_q4(KIND[id], &DW[id], &DS[id], dX2, h_out_dev, (uint32_t)inter, H);
    double b_ = prof ? now_ms() : 0;
    wait_stream();
    double c_ = prof ? now_ms() : 0;
    copy_from_pinned(out, h_out, (size_t)H * 4);
    if (prof) { double d_ = now_ms(); P_h2d[0] += b_ - a_; P_k[0] += c_ - b_; P_d2h[0] += d_ - c_; P_n[0]++; }
}

/* attention + o projection of one token on the GPU, one wait. Rows of the host kv
   cache not yet on the device are copied first (the prompt's rows, written by the
   batched prefill, go up once at the first decode step); the newest row (pos, just
   stored by forward()) travels with q in the staging buffer. Kernels: attn_scores,
   attn_softmax_pv (attn_head_ref bit for bit), q8k_dev, the o GEMV. */
static void gpu_attno(const layer_t *L, int src, const float *q, int lo, int pos, int hd, float *out) {
    const int io = (int)(L->o - T);
    if (!fast || pos >= G_CAP || !KIND[io]) { attno_ref(L, src, q, lo, pos, hd, out); return; }
    double a_ = prof ? now_ms() : 0;
    static float pk[NH * HD_F + 2 * HD_F];
    int newrow = 0;
    if (dsync[src] < pos) {   /* catch-up: rows [dsync, pos) */
        size_t r0 = (size_t)dsync[src], nr = (size_t)pos - r0;
        CK(cuMemcpyHtoD_(dKC[src] + r0 * hd * 4, KC[src] + r0 * hd, nr * hd * 4));
        CK(cuMemcpyHtoD_(dVC[src] + r0 * hd * 4, VC[src] + r0 * hd, nr * hd * 4));
        dsync[src] = pos;
    }
    memcpy(pk, q, (size_t)NH * hd * 4);
    if (dsync[src] == pos) {   /* the newest row */
        memcpy(pk + NH * hd, KC[src] + (size_t)pos * hd, (size_t)hd * 4);
        memcpy(pk + NH * hd + hd, VC[src] + (size_t)pos * hd, (size_t)hd * 4);
        newrow = 1;
    }
    stage_in(pk, (size_t)(NH * hd + (newrow ? 2 * hd : 0)) * 4, dQa);
    if (newrow) {
        uint64_t ks = dQa + (uint64_t)NH * hd * 4, vs = ks + (uint64_t)hd * 4;
        uint64_t kd = dKC[src] + (uint64_t)pos * hd * 4, vd = dVC[src] + (uint64_t)pos * hd * 4;
        uint32_t nw = (uint32_t)hd;
        void *c1[] = {&ks, &kd, &nw}, *c2[] = {&vs, &vd, &nw};
        CK(cuLaunchKernel_(fxc, (nw + 255) / 256, 1, 1, 256, 1, 1, 0, stream, c1, NULL));
        CK(cuLaunchKernel_(fxc, (nw + 255) / 256, 1, 1, 256, 1, 1, 0, stream, c2, NULL));
        dsync[src] = pos + 1;
    }
    int n = pos - lo + 1, ihd = hd;
    uint64_t kl = dKC[src] + (uint64_t)lo * hd * 4, vl = dVC[src] + (uint64_t)lo * hd * 4;
    void *a1[] = {&dQa, &kl, &dSa, &n, &ihd};
    CK(cuLaunchKernel_(fasc, (n * NH + 255) / 256, 1, 1, 256, 1, 1, (unsigned)(NH * hd * 4), stream, a1, NULL));
    void *a2[] = {&dSa, &vl, &dOa, &n, &ihd, &dTab};
    CK(cuLaunchKernel_(fapv, NH, 1, 1, 256, 1, 1, (unsigned)(n * 12), stream, a2, NULL));
    void *a3[] = {&dOa, &dX2};
    CK(cuLaunchKernel_(fq8k, NH * hd / 256, 1, 1, 256, 1, 1, 0, stream, a3, NULL));
    launch_q4(1, &DW[io], &DS[io], dX2, h_out_dev, (uint32_t)(NH * hd), H);
    double b_ = prof ? now_ms() : 0;
    wait_stream();
    double c_ = prof ? now_ms() : 0;
    copy_from_pinned(out, h_out, (size_t)H * 4);
    if (prof) { double d_ = now_ms(); P_h2d[0] += b_ - a_; P_k[0] += c_ - b_; P_d2h[0] += d_ - c_; P_n[0]++; }
}

/* ---------------------------------------------------- batched prefill hooks */
static float YCATB[BATCH_MAX * NH * HD_F + BATCH_MAX * 2 * HD_F];
/* q4i: one q4i_gemv_b launch for the whole group and all B tokens; K2-layout
   tensors: one single-token launch per token on the stream; one wait either way */
static void gpu_q4n_b(int n, const tensor_t **ts, const uint8_t *xq, int xs, int B, float **ys) {
    int i0 = (int)(ts[0] - T);
    uint32_t rows = 0, cols = (uint32_t)ts[0]->dims[0];
    for (int k = 0; k < n; k++) {
        if (KIND[ts[k] - T] != KIND[i0] || (KIND[i0] && DW[ts[k] - T] != DW[i0]) || (!KIND[i0] && n > 1)) { q4n_b_seq(n, ts, xq, xs, B, ys); return; }
        rows += (uint32_t)ts[k]->dims[1];
    }
    if (!fast || (size_t)B * xs > 65536 || (size_t)B * rows > sizeof YCATB / 4) { q4n_b_seq(n, ts, xq, xs, B, ys); return; }
    double a_ = prof ? now_ms() : 0;
    stage_in(xq, (size_t)B * xs, dX);
    double b_ = prof ? now_ms() : 0;
    if (KIND[i0]) {
        int irows = (int)rows, icols = (int)cols, ib = B, ixs = xs, iys = (int)rows;
        void *a[] = {&DW[i0], &DS[i0], &dX, &h_out_dev, &irows, &icols, &ib, &ixs, &iys};
        CK(cuLaunchKernel_(fq4ib, (rows + 255) / 256, 1, 1, 32, 8, 1, 0, stream, a, NULL));
    } else {
        for (int t = 0; t < B; t++) {
            uint64_t xt = dX + (uint64_t)t * xs, yt = h_out_dev + (uint64_t)t * rows * 4;
            uint32_t r0 = 0, r1 = rows, nb = cols / 32, rf = rows;
            void *a[] = {&DW[i0], &DS[i0], &xt, &yt, &r0, &r1, &nb, &rf};
            CK(cuLaunchKernel_(fq4, (r1 + 31) / 32, 1, 1, 32, 8, 1, 0, stream, a, NULL));
        }
    }
    wait_stream();
    double c_ = prof ? now_ms() : 0;
    copy_from_pinned(YCATB, h_out, (size_t)B * rows * 4);
    for (int t = 0; t < B; t++)
        for (int k = 0, off = 0; k < n; off += (int)ts[k]->dims[1], k++)
            memcpy(ys[k] + (size_t)t * ts[k]->dims[1], YCATB + (size_t)t * rows + off, ts[k]->dims[1] * 4);
    if (prof) { double d_ = now_ms(); P_h2d[0] += b_ - a_; P_k[0] += c_ - b_; P_d2h[0] += d_ - c_; P_n[0]++; }
}
/* MLP for B tokens on the GPU: gate+up (q4i_gemv_b), geglu_b, q8k_dev, down (q4i_gemv_b
   or per-token K2 launches), one wait */
static void gpu_mlp_b(const layer_t *L, int inter, const uint8_t *xq, int xs, int B, float *out) {
    if (!fast || !KIND[L->gate - T]) { mlp_b_seq(L, inter, xq, xs, B, out); return; }
    int ig = (int)(L->gate - T), id = (int)(L->down - T);
    double a_ = prof ? now_ms() : 0;
    stage_in(xq, (size_t)B * xs, dX);
    int r2 = 2 * inter, cH = H, ib = B, ixs = xs, iys = 2 * inter;
    void *a1[] = {&DW[ig], &DS[ig], &dX, &dYdevB, &r2, &cH, &ib, &ixs, &iys};
    CK(cuLaunchKernel_(fq4ib, (r2 + 255) / 256, 1, 1, 32, 8, 1, 0, stream, a1, NULL));
    int n = inter;
    void *a2[] = {&dYdevB, &dHdevB, &n, &ib, &dTab};
    CK(cuLaunchKernel_(fgeglub, (B * n + 255) / 256, 1, 1, 256, 1, 1, 0, stream, a2, NULL));
    void *a3[] = {&dHdevB, &dX2B};
    CK(cuLaunchKernel_(fq8k, B * n / 256, 1, 1, 256, 1, 1, 0, stream, a3, NULL));
    int xs2 = (inter / 256) * Q8B;
    if (KIND[id]) {
        int rH = H, ci = inter, iyH = H;
        void *a4[] = {&DW[id], &DS[id], &dX2B, &h_out_dev, &rH, &ci, &ib, &xs2, &iyH};
        CK(cuLaunchKernel_(fq4ib, (H + 255) / 256, 1, 1, 32, 8, 1, 0, stream, a4, NULL));
    } else {
        for (int t = 0; t < B; t++) {
            uint64_t xt = dX2B + (uint64_t)t * xs2, yt = h_out_dev + (uint64_t)t * H * 4;
            uint32_t r0 = 0, r1 = H, nb = (uint32_t)inter / 32, rf = H;
            void *a4[] = {&DW[id], &DS[id], &xt, &yt, &r0, &r1, &nb, &rf};
            CK(cuLaunchKernel_(fq4, (r1 + 31) / 32, 1, 1, 32, 8, 1, 0, stream, a4, NULL));
        }
    }
    double b_ = prof ? now_ms() : 0;
    wait_stream();
    double c_ = prof ? now_ms() : 0;
    copy_from_pinned(out, h_out, (size_t)B * H * 4);
    if (prof) { double d_ = now_ms(); P_h2d[0] += b_ - a_; P_k[0] += c_ - b_; P_d2h[0] += d_ - c_; P_n[0]++; }
}
/* head for B tokens: B q6i launches into device logits, argmax_b per token; only the
   index and raw value come back, plus token `keep`'s full logits for the sampler */
static void gpu_head_b(const tensor_t *emb, const uint8_t *xq, int xs, int B, int keep) {
    if (!fast) { head_b_seq(emb, xq, xs, B, keep); return; }
    int rows = (int)emb->dims[1];
    double a_ = prof ? now_ms() : 0;
    stage_in(xq, (size_t)B * xs, dX);
    double b_ = prof ? now_ms() : 0;
    /* one q6i_gemv_b launch: each weight block read once for all B tokens (paired
       6-bit decode made it faster than B single launches); G4HEAD1=1: B launches */
    if (!getenv("G4HEAD1")) {
        int ib = B, ixs = xs, iys = VOCAB;
        void *a[] = {&dHead, &dX, &dLogB, &rows, &ib, &ixs, &iys};
        CK(cuLaunchKernel_(fq6ib, (rows + 255) / 256, 1, 1, 32, 8, 1, 0, stream, a, NULL));
    } else for (int t = 0; t < B; t++) {
        uint64_t xt = dX + (uint64_t)t * xs, yt = dLogB + (uint64_t)t * VOCAB * 4;
        void *a[] = {&dHead, &xt, &yt, &rows};
        CK(cuLaunchKernel_(fq6, (rows + 255) / 256, 1, 1, 32, 8, 1, 0, stream, a, NULL));
    }
    void *aa[] = {&dLogB, &rows, &h_out_dev};
    CK(cuLaunchKernel_(fargb, B, 1, 1, 1024, 1, 1, 0, stream, aa, NULL));
    wait_stream();
    double c_ = prof ? now_ms() : 0;
    unsigned r[2 * BATCH_MAX];
    memcpy(r, h_out, (size_t)8 * B);
    for (int t = 0; t < B; t++) { memcpy(&TOPV[t], &r[2 * t], 4); TOPI[t] = (int)r[2 * t + 1]; }
    if (keep >= 0) CK(cuMemcpyDtoH_(logits, dLogB + (uint64_t)keep * VOCAB * 4, (size_t)VOCAB * 4));
    if (prof) { double d_ = now_ms(); P_h2d[1] += b_ - a_; P_k[1] += c_ - b_; P_d2h[1] += d_ - c_; P_n[1]++; }
}

/* ---------------------------------------------------------------- hooks */
static void gpu_q4(const tensor_t *t, const uint8_t *xq, float *y) {
    int i = (int)(t - T);
    if (fast) { q4_fast(KIND[i], &DW[i], &DS[i], (uint32_t)t->dims[0], (uint32_t)t->dims[1], xq, y); return; }
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
    if (KIND[ts[0] - T]) {   /* q4i group: one interleaved image, launched as a whole */
        for (int k = 0; k < n; k++) {
            if (!KIND[ts[k] - T] || DW[ts[k] - T] != DW[ts[0] - T]) { fprintf(stderr, "q4i group mismatch\n"); exit(3); }
            rows += (uint32_t)ts[k]->dims[1];
        }
        int i0 = (int)(ts[0] - T);
        q4_fast(1, &DW[i0], &DS[i0], (uint32_t)ts[0]->dims[0], rows, xq, YCAT);
        for (int k = 0, off = 0; k < n; off += (int)ts[k]->dims[1], k++) memcpy(ys[k], YCAT + off, ts[k]->dims[1] * 4);
        return;
    }
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
    if (fast) {
        q4_fast(KIND[i0], &DW[i0], &DS[i0], (uint32_t)ts[0]->dims[0], rows, xq, YCAT);
        for (int k = 0, off = 0; k < n; off += (int)ts[k]->dims[1], k++) memcpy(ys[k], YCAT + off, ts[k]->dims[1] * 4);
        return;
    }
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
    if (fast) {
        double a_ = prof ? now_ms() : 0;
        stage_in(xq, (size_t)(t->dims[0] / 256) * Q8B, dX);
        void *args[] = {&dHead, &dX, &h_out_dev, &rows};
        double b_ = prof ? now_ms() : 0;
        CK(cuLaunchKernel_(fq6, (rows + 255) / 256, 1, 1, 32, 8, 1, 0, stream, args, NULL));   /* q6i_gemv */
        wait_stream();
        double c_ = prof ? now_ms() : 0;
        copy_from_pinned(y, h_out, (size_t)rows * 4);
        if (prof) { double d_ = now_ms(); P_h2d[1] += b_ - a_; P_k[1] += c_ - b_; P_d2h[1] += d_ - c_; P_n[1]++; }
        return;
    }
    double a_ = prof ? now_ms() : 0;
    CK(cuMemcpyHtoD_(dX, xq, (size_t)(t->dims[0] / 256) * Q8B));
    void *args[] = {&dHead, &dX, &dY, &rows};
    double b_ = prof ? now_ms() : 0;
    CK(cuLaunchKernel_(fq6, (rows + 255) / 256, 1, 1, 32, 8, 1, 0, NULL, args, NULL));   /* q6i_gemv */
    CK(cuCtxSynchronize_());
    double c_ = prof ? now_ms() : 0;
    CK(cuMemcpyDtoH_(y, dY, (size_t)rows * 4));
    if (prof) { double d_ = now_ms(); P_h2d[1] += b_ - a_; P_k[1] += c_ - b_; P_d2h[1] += d_ - c_; P_n[1]++; }
}
static void gpu_f16(const tensor_t *t, const float *x, float *y) {
    int rows = (int)t->dims[1], cols = (int)t->dims[0];
    if (fast) {
        double a_ = prof ? now_ms() : 0;
        stage_in(x, (size_t)cols * 4, dXf);
        void *args[] = {&dF16, &dXf, &h_out_dev, &rows, &cols};
        double b_ = prof ? now_ms() : 0;
        CK(cuLaunchKernel_(ff16, (rows + 127) / 128, 1, 1, 128, 1, 1, 0, stream, args, NULL));
        wait_stream();
        double c_ = prof ? now_ms() : 0;
        copy_from_pinned(y, h_out, (size_t)rows * 4);
        if (prof) { double d_ = now_ms(); P_h2d[2] += b_ - a_; P_k[2] += c_ - b_; P_d2h[2] += d_ - c_; P_n[2]++; }
        return;
    }
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
    cuStreamCreate_ = sym(h, "cuStreamCreate"); cuEventCreate_ = sym(h, "cuEventCreate");
    cuEventRecord_ = sym(h, "cuEventRecord"); cuEventQuery_ = sym(h, "cuEventQuery");
    cuMemHostAlloc_ = sym(h, "cuMemHostAlloc"); cuMemHostGetDevicePointer_ = sym(h, "cuMemHostGetDevicePointer_v2");
    int dev; void *ctx, *m4, *mk;
    CK(cuInit_(0)); CK(cuDeviceGet_(&dev, 0)); CK(cuCtxCreate_(&ctx, 0, dev));
    CK(cuModuleLoadData_(&m4, G4_PTX_Q4)); CK(cuModuleGetFunction_(&fq4, m4, "q4_gemv"));
    CK(cuModuleGetFunction_(&fxc, m4, "xq_copy"));

    fast = !(getenv("G4FAST") && getenv("G4FAST")[0] == '0');
    use_q4i = fast && !(getenv("G4Q4I") && getenv("G4Q4I")[0] == '0');   /* q4i needs the fast path */
    if (fast) {
        CK(cuStreamCreate_(&stream, 1));                       /* non-blocking */
        CK(cuEventCreate_(&event, 2));                         /* CU_EVENT_DISABLE_TIMING */
        CK(cuMemHostAlloc_(&h_in, 65536, 1 | 2));              /* PORTABLE | DEVICEMAP */
        CK(cuMemHostAlloc_(&h_out, (size_t)VOCAB * 4, 1 | 2));
        CK(cuMemHostGetDevicePointer_(&h_in_dev, h_in, 0));
        CK(cuMemHostGetDevicePointer_(&h_out_dev, h_out, 0));
    }
    CK(cuModuleLoadData_(&mk, G4_PTX_K));
    CK(cuModuleGetFunction_(&ff16, mk, "f16_gemv_t"));
    CK(cuModuleGetFunction_(&fgeglu, mk, "geglu_dev")); CK(cuModuleGetFunction_(&fq8k, mk, "q8k_dev"));
    CK(cuModuleGetFunction_(&fq4i, mk, getenv("G4Q4I4") ? "q4i_gemv4" : "q4i_gemv5"));   /* G4Q4I4=1: previous kernel (A/B) */
    CK(cuModuleGetFunction_(&fq6, mk, "q6i_gemv"));
    CK(cuModuleGetFunction_(&fq4ib, mk, getenv("G4Q4I4") ? "q4i_gemv_b" : "q4i_gemv_b2")); CK(cuModuleGetFunction_(&fgeglub, mk, "geglu_b"));
    CK(cuModuleGetFunction_(&fargb, mk, "argmax_b")); CK(cuModuleGetFunction_(&fq6ib, mk, "q6i_gemv_b"));
    CK(cuModuleGetFunction_(&fasc, mk, "attn_scores")); CK(cuModuleGetFunction_(&fapv, mk, "attn_softmax_pv"));
    dYdev = dalloc((size_t)4 * INTER * 4); dHdev = dalloc((size_t)2 * INTER * 4); dX2 = dalloc(sizeof XQ);
    dYdevB = dalloc((size_t)BATCH_MAX * 4 * INTER * 4); dHdevB = dalloc((size_t)BATCH_MAX * 2 * INTER * 4);
    dX2B = dalloc((size_t)BATCH_MAX * sizeof XQ); dLogB = dalloc((size_t)BATCH_MAX * VOCAB * 4);
    {   /* exp/GELU constants for geglu_dev: g4ref.c's own definitions */
        double v[24] = {EXP_P0};
        for (int i = 0; i < 13; i++) v[1 + i] = EXP_C[i];
        v[14] = EXP_LOG2E; v[15] = EXP_SH; v[16] = EXP_LN2HI; v[17] = EXP_LN2LO;
        v[18] = 709.0; v[19] = -745.0; v[20] = INFINITY; v[21] = GELU_C1; v[22] = GELU_C3; v[23] = 1.0;
        dTab = dalloc(sizeof v); CK(cuMemcpyHtoD_(dTab, v, sizeof v));
    }
    dX = dalloc(65536); dXf = dalloc((size_t)H * 4); dY = dalloc((size_t)VOCAB * 4);   /* dX: up to BATCH_MAX inputs */
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
            size_t grows = 0;
            for (int k = 0; k < 3 && grp[g][k]; k++) grows += grp[g][k]->dims[1];
            int gcols = (int)grp[g][0]->dims[0];
            /* q4i for every group since the paired-nibble kernels (v5/b2): faster than
               K2 also for 256-row inp_gate and 12288-column down; 12288 = Q4I_MAXCOLS */
            int kind = use_q4i && gcols <= 12288 && grows % 32 == 0;
            if (kind) {   /* q4i: tile-interleave the whole group: [tile][block][32 rows] */
                int nbq = gcols / 32;
                uint8_t *nib = malloc(tot * 16); float *scl = malloc(tot * 4);
                size_t r = 0;
                for (int k = 0; k < 3 && grp[g][k]; k++) {
                    const tensor_t *t = grp[g][k];
                    for (size_t rr = 0; rr < t->dims[1]; rr++, r++)
                        for (int b = 0; b < nbq; b++) {
                            const uint8_t *blk = t->data + (rr * nbq + b) * 18;
                            size_t o = ((r / 32) * nbq + b) * 32 + (r % 32);
                            scl[o] = h2f((uint16_t)(blk[0] | blk[1] << 8));
                            memcpy(nib + o * 16, blk + 2, 16);
                        }
                    int i = (int)(t - T);
                    DW[i] = bw; DS[i] = bs; KIND[i] = 1;   /* only the group start is launched */
                    drop_pages(t->data, (size_t)t->dims[1] * nbq * 18);
                    nq4++;
                }
                CK(cuMemcpyHtoD_(bw, nib, tot * 16)); CK(cuMemcpyHtoD_(bs, scl, tot * 4));
                free(nib); free(scl);
                continue;
            }
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
    /* Q6_K head, q6i_gemv layout: rows in tiles of 32, each 210-byte block padded to
       224 and stored as [tile][block][16-byte chunk][32 rows] (warp loads 512 B) */
    {
        const tensor_t *t = tens("token_embd.weight");
        size_t rows = t->dims[1];
        if (t->dims[0] != 1536 || rows % 32) { fprintf(stderr, "head shape unexpected\n"); exit(2); }
        uint8_t *rp = calloc(rows * 6, 224);
        for (size_t r = 0; r < rows; r++) for (int b = 0; b < 6; b++) {
            uint8_t blk[224] = {0};
            memcpy(blk, t->data + (r * 6 + b) * 210, 210);
            for (int k = 0; k < 14; k++)
                memcpy(rp + ((((r / 32) * 6 + b) * 14 + k) * 32 + (r % 32)) * 16, blk + 16 * k, 16);
        }
        dHead = dalloc(rows * 6 * 224); CK(cuMemcpyHtoD_(dHead, rp, rows * 6 * 224)); free(rp);
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
    if (!(getenv("G4GPUATTN") && getenv("G4GPUATTN")[0] == '0')) {   /* G4GPUATTN=0: CPU attention */
        for (int i = 0; i < FIRST_SHARED; i++) {
            size_t b = (size_t)G_CAP * (SLIDING[i] ? HD_S : HD_F) * 4;
            dKC[i] = dalloc(b); dVC[i] = dalloc(b);
        }
        dQa = dalloc((size_t)(NH * HD_F + 2 * HD_F) * 4); dSa = dalloc((size_t)G_CAP * NH * 4); dOa = dalloc((size_t)NH * HD_F * 4);
        gpu_attn = 1;
        fprintf(stderr, "decode attention: on the GPU (kv cache %d positions)\n", G_CAP);
    }
    size_t fr = 0, tot = 0; cuMemGetInfo_(&fr, &tot);
    if (fast) fprintf(stderr, "GPU driver path: fast (stream, timing-free event, mapped pinned I/O)\n");
    { int nk = 0; for (int i = 0; i < NT; i++) nk += KIND[i];
      fprintf(stderr, "Q4_0 kernels: %d tensors on q4i_gemv5, %d on K2 q4_gemv\n", nk, nq4 - nk); }
    fprintf(stderr, "GPU: %d Q4_0 tensors + Q6_K head + F16 proj, %.1f MB on device, %.0f MB free of %.0f; load %.0f ms\n",
            nq4, dev_bytes / 1048576.0, fr / 1048576.0, tot / 1048576.0, now_ms() - t0);
    G_Q4 = gpu_q4; G_Q4N = gpu_q4n; G_HEAD = gpu_head; G_F16 = gpu_f16;
    G_Q4N_B = gpu_q4n_b; G_MLP_B = gpu_mlp_b; G_HEAD_B = gpu_head_b;
    if (gpu_attn) { G_ATTNO = gpu_attno; pf_on = 0; }
    if (!(getenv("G4GPUMLP") && getenv("G4GPUMLP")[0] == '0')) { G_MLP = gpu_mlp; fprintf(stderr, "MLP: on the GPU (gate+up, GeGLU, Q8_K, down)\n"); }
    /* G4ASM=0 keeps the C host loops (A/B and fallback) */
    if (!(getenv("G4ASM") && getenv("G4ASM")[0] == '0')) {
        build_tab(); G_GEGLU = geglu_asm; G_ATTN = attn_head_asm; G_ATTNALL = attn_all_asm; G_RMS = rms_asm; G_RMSNS = rmsns_asm; G_Q8K = q8k_asm;
        fprintf(stderr, "host kernels: assembly (g4_host.S)\n");
    }
    G_PAR = !(getenv("G4PAR") && getenv("G4PAR")[0] == '0') && G_ATTNALL == attn_all_asm;   /* thread-safe host kernels only */
}

/* G4PROF: time the CPU-side hooks of single-token forwards (calls made outside a
   parallel region, i.e. decode); [0] rms [1] rmsns [2] q8k [3] attention [4] geglu */
static double PC_t[5]; static long PC_n[5];
static void (*o_rms)(const float *, const float *, float *, int);
static void (*o_rmsns)(float *, int);
static void (*o_q8k)(const float *, uint8_t *, int);
static void (*o_attn)(const float *, const float *, const float *, int, int, int, float *);
static void (*o_geglu)(float *, const float *, int);
#define PC_WRAP(k, call) do { if (omp_in_parallel()) { call; break; } double t_ = now_ms(); call; PC_t[k] += now_ms() - t_; PC_n[k]++; } while (0)
static void w_rms(const float *x, const float *w, float *y, int n) { PC_WRAP(0, o_rms(x, w, y, n)); }
static void w_rmsns(float *x, int n) { PC_WRAP(1, o_rmsns(x, n)); }
static void w_q8k(const float *x, uint8_t *y, int n) { PC_WRAP(2, o_q8k(x, y, n)); }
static void w_attn(const float *q, const float *K, const float *V, int lo, int pos, int hd, float *o) { PC_WRAP(3, o_attn(q, K, V, lo, pos, hd, o)); }
static void w_geglu(float *g, const float *u, int n) { PC_WRAP(4, o_geglu(g, u, n)); }

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <gguf> <rope.bin> <ids...> [--gen N] [--batch B] [--sample temp topk topp seed]\n", argv[0]); return 2; }
    /* OpenMP workers must not busy-wait between parallel regions: libgomp's default
       spin (~300k iterations) keeps 3 cores hot while the main thread waits on the
       GPU and slows decode ~6% (shared DRAM / power budget). libgomp reads its
       environment once at load, so set a short spin and re-exec once; a user value
       of GOMP_SPINCOUNT or OMP_WAIT_POLICY is respected. */
    if (!getenv("GOMP_SPINCOUNT") && !getenv("OMP_WAIT_POLICY")) {
        setenv("GOMP_SPINCOUNT", "10000", 1);
        execv("/proc/self/exe", argv);   /* returns only on failure: run with the default */
    }
    int gen = 0, n = 0, batch = BATCH_MAX;   /* prefill in chunks of 8 by default; --batch 1 = token by token */
    static int ids[MAXSEQ];
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--gen") && i + 1 < argc) { gen = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--batch") && i + 1 < argc) { batch = atoi(argv[++i]); if (batch > BATCH_MAX) batch = BATCH_MAX; if (batch < 1) batch = 1; continue; }
        if (parse_sample(argc, argv, &i)) continue;
        ids[n++] = atoi(argv[i]);
    }
    gguf_open(argv[1]);
    { int c = 0; for (int i = 0; i < NL; i++) c += !SLIDING[i];
      if (c != 7) { fprintf(stderr, "sliding pattern: %d full layers, want 7\n", c); return 2; } }
    layer_table();
    { /* mapped read-only, not copied: only the pages of the positions actually used
         become resident (the table holds 2048 positions, 6.3 MB) */
      size_t want = (size_t)MAXSEQ * (128 + 256) * 2 * 4;
      int fd = open(argv[2], O_RDONLY); struct stat st;
      if (fd < 0 || fstat(fd, &st) || (size_t)st.st_size < want) { fprintf(stderr, "rope table %s: want %zu bytes\n", argv[2], want); return 2; }
      void *m = mmap(NULL, want, PROT_READ, MAP_PRIVATE, fd, 0);
      if (m == MAP_FAILED) { fprintf(stderr, "rope table %s: mmap failed\n", argv[2]); return 2; }
      close(fd); ROPE = m; }
    prof = getenv("G4PROF") && getenv("G4PROF")[0] == '1';
    G_CAP = n + gen + 1 < MAXSEQ ? n + gen + 1 : MAXSEQ;   /* positions this run can reach */
    gpu_init();
    if (prof) {
        o_rms = G_RMS; G_RMS = w_rms; o_rmsns = G_RMSNS; G_RMSNS = w_rmsns; o_q8k = G_Q8K; G_Q8K = w_q8k;
        o_attn = G_ATTNALL; G_ATTNALL = w_attn; o_geglu = G_GEGLU; G_GEGLU = w_geglu;
    }
    int n_prompt = n;
    double tp = 0, tg = 0;
    G_CLOCK = now_ms;
    int pos = run_generation(ids, n, gen, batch, &tp, &tg);
    int ng = pos - n_prompt;
    if (prof) {
        const char *nm[3] = {"q4", "head", "f16"};
        double tg_all = tp + tg, hk = 0;
        for (int k = 0; k < 3; k++) {
            hk += P_h2d[k] + P_k[k] + P_d2h[k];
            fprintf(stderr, "PROF %-4s calls %6ld  h2d %8.0f ms  kernel+sync %8.0f ms  d2h %8.0f ms  (%.2f ms/token)\n",
                    nm[k], P_n[k], P_h2d[k], P_k[k], P_d2h[k], (P_h2d[k] + P_k[k] + P_d2h[k]) / pos);
        }
        const char *cn[5] = {"rms", "rmsns", "q8k", "attn", "geglu"};
        for (int k = 0; k < 5; k++) fprintf(stderr, "PROF cpu %-5s calls %7ld  %8.1f ms  (%.2f ms per decode step)\n", cn[k], PC_n[k], PC_t[k], ng > 0 ? PC_t[k] / ng : 0.0);
        fprintf(stderr, "PROF forward total %.0f ms over %d steps: GPU hooks %.1f ms/token, everything else (CPU stages) %.1f ms/token\n",
                tg_all, pos, hk / pos, (tg_all - hk) / pos);
    }
    fprintf(stderr, "prefill %d tokens %.0f ms (%.2f tok/s); decode %d steps %.0f ms (%.2f tok/s)\n",
            n_prompt, tp, n_prompt * 1000.0 / tp, ng, tg, ng > 0 ? ng * 1000.0 / tg : 0.0);
    return 0;
}
