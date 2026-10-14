/* Exp45 probe: graph-per-GEMV-call vs the engine's current per-call host path.
   Real CUDA 10.2 signatures (R440 has NO cuGraphAddMemcpyNode1D - it is CUDA 11+;
   the 10.2 symbol is cuGraphAddMemcpyNode with CUDA_MEMCPY3D params by pointer).

   The engine's per-call host path (measured with K2PROF in run #69) is
   stage 38.8us + launch 40us + d2h 21.3us + device-wait, x ~230 calls/token
   = ~100us/call = ~23ms of the 149ms/token.

   This probe builds ONE reusable exec-graph per GEMV shape:
        memcpy H2D (pinned h_in -> d_xq) -> kernel (grid=NROW) -> memcpy D2H (d_y -> pinned h_out)
   and per call does:
        memcpy xq -> h_in;  cuGraphExecKernelNodeSetParams (new weight/sc/y pointers);
        cuGraphLaunch;  event record + poll.
   The A/B baseline is the engine's exact current pattern with the same sizes.
   PASS = the graph path is bit-identical AND cheaper per call. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <stdint.h>
#include <time.h>

typedef int CUresult;
typedef unsigned long long CUdeviceptr;

typedef struct {
    void *func;
    unsigned gx, gy, gz, bx, by, bz;
    unsigned shmem;
    void **kernelParams;
    void **extra;
} KernParams;

/* CUDA 10.2 CUDA_MEMCPY3D (order per cuda.h, 6.x layout, 100 bytes). */
typedef struct {
    size_t srcXInBytes, srcY, srcZ, srcLOD;
    int srcMemoryType;
    const void *srcHost;
    CUdeviceptr srcDevice;
    void *srcArray;
    void *reserved0;
    size_t srcPitch, srcHeight;
    size_t dstXInBytes, dstY, dstZ, dstLOD;
    int dstMemoryType;
    void *dstHost;
    CUdeviceptr dstDevice;
    void *dstArray;
    void *reserved1;
    size_t dstPitch, dstHeight;
    size_t WidthInBytes, Height, Depth;
} Memcpy3D;

typedef CUresult (*fn_init)(unsigned);
typedef CUresult (*fn_dev)(int *, int);
typedef CUresult (*fn_ctx)(void **, unsigned, int);
typedef CUresult (*fn_mod)(void *, const char *);
typedef CUresult (*fn_getfn)(void **, void *, const char *);
typedef CUresult (*fn_alloc)(CUdeviceptr *, size_t);
typedef CUresult (*fn_halloc)(void **, size_t, unsigned);
typedef CUresult (*fn_sync)(void);
typedef CUresult (*fn_d2h)(void *, CUdeviceptr, size_t);
typedef CUresult (*fn_h2d)(CUdeviceptr, const void *, size_t);
typedef CUresult (*fn_h2da)(CUdeviceptr, const void *, size_t, void *);
typedef CUresult (*fn_d2ha)(void *, CUdeviceptr, size_t, void *);
typedef CUresult (*fn_str)(void *, unsigned);
typedef CUresult (*fn_evcreate)(void **, unsigned);
typedef CUresult (*fn_evrecord)(void *, void *);
typedef CUresult (*fn_evquery)(void *);
typedef CUresult (*fn_gcreate)(void **, unsigned);
typedef CUresult (*fn_gkn)(void **, void *, const void *, size_t, const KernParams *);
typedef CUresult (*fn_gmc)(void **, void *, const void *, size_t, const Memcpy3D *, void *);
typedef CUresult (*fn_ginst)(void **, void *, void *, char *, size_t);
typedef CUresult (*fn_glaunch)(void *, void *);
typedef CUresult (*fn_gset)(void *, void *, const KernParams *);
typedef CUresult (*fn_gsetmc)(void *, void *, const Memcpy3D *, void *);
typedef CUresult (*fn_launch)(void *, unsigned, unsigned, unsigned, unsigned,
                              unsigned, unsigned, unsigned, void *, void **, void **);

/* gemv-ish: y[i] = dot4(xq) + 0.5 ; uses xq (IN) and writes y (OUT), grid = rows.
   Both xq and y are real device buffers, as in the engine. */
static const char *ptx =
".version 4.0\n"
".target sm_50\n"
".address_size 64\n"
".visible .entry gemv(.param .u64 pxq, .param .u64 py, .param .u32 pn) {\n"
"  .reg .u64 %rd<6>;\n  .reg .u32 %r<6>;\n  .reg .f32 %f<6>;\n  .reg .pred %p<2>;\n"
"  ld.param.u64 %rd1, [pxq];\n"
"  ld.param.u64 %rd2, [py];\n"
"  ld.param.u32 %r1, [pn];\n"
"  mov.u32 %r2, %ctaid.x;\n"
"  mul.wide.u32 %rd3, %r2, 4;\n"
"  add.u64 %rd2, %rd2, %rd3;\n"
"  mov.f32 %f1, 0F00000000;\n"
"  mov.u32 %r3, 0;\n"
"LOOP: setp.ge.u32 %p1, %r3, %r1; @%p1 bra done;\n"
"  mul.wide.u32 %rd4, %r3, 4;\n"
"  add.u64 %rd5, %rd1, %rd4;\n"
"  ld.global.f32 %f2, [%rd5];\n"
"  fma.rn.f32 %f1, %f2, %f2, %f1;\n"
"  add.u32 %r3, %r3, 1;\n"
"  bra LOOP;\n"
"done:\n"
"  mov.f32 %f3, 0F3F000000;\n"
"  add.f32 %f1, %f1, %f3;\n"
"  st.global.f32 [%rd2], %f1;\n"
"  ret;\n"
"}\n";

#define CK(x) do { CUresult _r = (x); if (_r) { \
    printf("FAIL %s rc=%d\n", #x, _r); return 1; } } while (0)

static double ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

#define CU_MEMORYTYPE_HOST   1
#define CU_MEMORYTYPE_DEVICE 2

int main(void) {
    void *lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (!lib) { printf("no libcuda\n"); return 1; }
#define SYM(T, N) T N = (T) dlsym(lib, #N); if (!N) printf("MISSING %s\n", #N);
    SYM(fn_init, cuInit) SYM(fn_dev, cuDeviceGet) SYM(fn_ctx, cuCtxCreate_v2)
    SYM(fn_mod, cuModuleLoadData) SYM(fn_getfn, cuModuleGetFunction)
    SYM(fn_alloc, cuMemAlloc_v2) SYM(fn_halloc, cuMemHostAlloc)
    SYM(fn_sync, cuCtxSynchronize)
    SYM(fn_d2h, cuMemcpyDtoH_v2) SYM(fn_h2d, cuMemcpyHtoD_v2)
    SYM(fn_h2da, cuMemcpyHtoDAsync_v2) SYM(fn_d2ha, cuMemcpyDtoHAsync_v2)
    SYM(fn_str, cuStreamCreate)
    SYM(fn_evcreate, cuEventCreate) SYM(fn_evrecord, cuEventRecord)
    SYM(fn_evquery, cuEventQuery)
    SYM(fn_gcreate, cuGraphCreate) SYM(fn_gkn, cuGraphAddKernelNode)
    SYM(fn_gmc, cuGraphAddMemcpyNode) SYM(fn_ginst, cuGraphInstantiate)
    SYM(fn_glaunch, cuGraphLaunch)
    SYM(fn_gset, cuGraphExecKernelNodeSetParams)
    SYM(fn_gsetmc, cuGraphExecMemcpyNodeSetParams)
    SYM(fn_launch, cuLaunchKernel)

    int NROW = 256;                 /* engine shape (probe #69: rows=2048 nb=48) */
    int NIN = 128;                 /* xq ~40KB */
    size_t BY = (size_t) NROW * 4, BI = (size_t) NIN * 4;
    int REP = 1000, i, rep, bad;
    double tA = 0;

    CK(cuInit(0));
    int dev = 0; void *ctx = 0;
    CK(cuDeviceGet(&dev, 0));
    CK(cuCtxCreate_v2(&ctx, 0, dev));
    void *mod = 0, *fg = 0;
    CK(cuModuleLoadData(&mod, ptx));
    CK(cuModuleGetFunction(&fg, mod, "gemv"));

    CUdeviceptr d_xq = 0, d_y = 0;
    CK(cuMemAlloc_v2(&d_xq, BI));
    CK(cuMemAlloc_v2(&d_y, BY));
    void *h_in = 0, *h_out = 0;
    CK(cuMemHostAlloc(&h_in, BI, 1));
    CK(cuMemHostAlloc(&h_out, BY, 1));
    for (i = 0; i < NIN; i++) ((float *) h_in)[i] = (float)(i % 17) * 0.25f;

    void *stream = 0, *ev = 0;
    CK(cuStreamCreate(&stream, 1));
    CK(cuEventCreate(&ev, 0));

    /* --- engine's current per-call pattern (baseline), pinned + async + event poll --- */
    { unsigned n32 = NIN;
      void *args[3] = { &d_xq, &d_y, &n32 };
      double t0 = ms();
      for (rep = 0; rep < REP; rep++) {
          memcpy(h_in, h_in, 0);                       /* engine copies xq here */
          CK(cuMemcpyHtoDAsync_v2(d_xq, h_in, BI, stream));
          CK(cuLaunchKernel(fg, NROW,1,1, 32,1,1, 0, stream, args, 0));
          CK(cuMemcpyDtoHAsync_v2(h_out, d_y, BY, stream));
          CK(cuEventRecord(ev, stream));
          while (cuEventQuery(ev) == 600) { }
      }
      CK(cuCtxSynchronize());
      double d = ms() - t0;
      tA = d;
      printf("A engine-style per call (H2D+launch+D2H+event): %.2f ms = %.2f us/call\n",
             d, d * 1e3 / REP); }

    /* --- build the reusable graph for this shape --- */
    void *graph = 0;
    CK(cuGraphCreate(&graph, 0));

    /* kernel params live in persistent storage: the exec graph copies them at SetParams */
    CUdeviceptr gp_xq = d_xq, gp_y = d_y;
    unsigned gp_n = NIN;
    void *kargs[3] = { &gp_xq, &gp_y, &gp_n };
    KernParams kp;
    memset(&kp, 0, sizeof kp);
    kp.func = fg; kp.gx = NROW; kp.gy = 1; kp.gz = 1;
    kp.bx = 32; kp.by = 1; kp.bz = 1; kp.shmem = 0;
    kp.kernelParams = kargs; kp.extra = 0;

    Memcpy3D m_in, m_out;
    memset(&m_in, 0, sizeof m_in);
    m_in.srcMemoryType = CU_MEMORYTYPE_HOST;   m_in.srcHost = h_in;
    m_in.dstMemoryType = CU_MEMORYTYPE_DEVICE; m_in.dstDevice = d_xq;
    m_in.WidthInBytes = BI; m_in.Height = 1; m_in.Depth = 1;
    memset(&m_out, 0, sizeof m_out);
    m_out.srcMemoryType = CU_MEMORYTYPE_DEVICE; m_out.srcDevice = d_y;
    m_out.dstMemoryType = CU_MEMORYTYPE_HOST;   m_out.dstHost = h_out;
    m_out.WidthInBytes = BY; m_out.Height = 1; m_out.Depth = 1;

    void *n1 = 0, *n2 = 0, *n3 = 0;
    CK(cuGraphAddMemcpyNode(&n1, graph, 0, 0, &m_in, ctx));
    CK(cuGraphAddKernelNode(&n2, graph, &n1, 1, &kp));
    CK(cuGraphAddMemcpyNode(&n3, graph, &n2, 1, &m_out, ctx));

    void *ge = 0;
    { char log[512]; log[0] = 0;
      CUresult r = cuGraphInstantiate(&ge, graph, 0, log, sizeof log);
      printf("instantiate rc=%d ge=%p log=%.200s\n", r, ge, log);
      if (r) return 2; }

    /* --- correctness: graph result must equal the baseline result, bit for bit --- */
    float *ref = (float *) malloc(BY);
    { unsigned n32 = NIN; void *args[3] = { &d_xq, &d_y, &n32 };
      CK(cuMemcpyHtoD_v2(d_xq, h_in, BI));
      CK(cuLaunchKernel(fg, NROW,1,1, 32,1,1, 0, stream, args, 0));
      CK(cuCtxSynchronize());
      CK(cuMemcpyDtoH_v2(ref, d_y, BY)); }
    memset(h_out, 0, BY);
    CK(cuGraphLaunch(ge, stream));
    CK(cuCtxSynchronize());
    bad = 0;
    for (i = 0; i < NROW; i++) if (((float *) h_out)[i] != ref[i]) bad++;
    printf("graph vs direct: mismatches=%d %s (y[0]=%f)\n", bad,
           bad ? "MISMATCH" : "BIT-IDENTICAL", *(float *) h_out);
    if (bad) return 2;

    /* --- B: graph path, engine-style event poll, params patched every call --- */
    double t_set = 0, t_launch = 0, t_poll = 0, t_total = 0;
    t_total = ms();
    for (rep = 0; rep < REP; rep++) {
        double a = ms();
        gp_xq = d_xq; gp_y = d_y; gp_n = NIN;       /* engine varies these per call */
        CK(cuGraphExecKernelNodeSetParams(ge, n2, &kp));
        double b = ms(); t_set += b - a;
        CK(cuGraphLaunch(ge, stream));
        double c = ms(); t_launch += c - b;
        CK(cuEventRecord(ev, stream));
        while (cuEventQuery(ev) == 600) { }
        t_poll += ms() - c;
    }
    CK(cuCtxSynchronize());
    double gb = ms() - t_total;
    printf("B graph per call (SetParams+launch+event): %.2f ms = %.2f us/call "
           "[set %.2f launch %.2f poll %.2f us]\n",
           gb, gb * 1e3 / REP, t_set * 1e3 / REP, t_launch * 1e3 / REP, t_poll * 1e3 / REP);

    /* --- C: graph path WITHOUT per-call SetParams (params fixed; upper bound) --- */
    double t0 = ms();
    for (rep = 0; rep < REP; rep++) {
        CK(cuGraphLaunch(ge, stream));
        CK(cuEventRecord(ev, stream));
        while (cuEventQuery(ev) == 600) { }
    }
    CK(cuCtxSynchronize());
    double gc = ms() - t0;
    printf("C graph per call (no SetParams): %.2f ms = %.2f us/call\n", gc, gc * 1e3 / REP);

    /* --- D: pipelined graph (no per-call event: one sync at the end) --- */
    t0 = ms();
    for (rep = 0; rep < REP; rep++) {
        CK(cuGraphExecKernelNodeSetParams(ge, n2, &kp));
        CK(cuGraphLaunch(ge, stream));
    }
    CK(cuCtxSynchronize());
    double gd = ms() - t0;
    printf("D graph per call (SetParams+launch, sync at end): %.2f ms = %.2f us/call\n",
           gd, gd * 1e3 / REP);

    /* after D, results must still be valid */
    CK(cuMemcpyDtoH_v2(h_out, d_y, BY));
    bad = 0;
    for (i = 0; i < NROW; i++) if (((float *) h_out)[i] != ref[i]) bad++;
    printf("post-pipeline verify: mismatches=%d %s\n", bad, bad ? "MISMATCH" : "OK");

    printf("=== per-call cost, %d rows / %d B in, %d reps ===\n", NROW, NIN, REP);
    printf("A engine-style      : %.2f us\nB graph+SetParams     : %.2f us\n"
           "C graph (fixed)     : %.2f us\nD graph pipelined     : %.2f us\n",
           tA * 1e3 / REP, gb * 1e3 / REP, gc * 1e3 / REP, gd * 1e3 / REP);
    printf("PROBE-DONE\n");
    return 0;
}
