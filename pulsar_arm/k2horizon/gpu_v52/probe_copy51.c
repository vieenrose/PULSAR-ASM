/* Exp46 probe: which per-call host sequence is cheapest on R440 at ENGINE sizes?
   K2PROF (run #69) says the engine pays stage 38.8 + launch 40.0 + d2h 21.3 us per call
   (~100us x ~230 calls/token = 23ms of the 149ms/token). The xq operand is tiny
   (xq_bytes = ceil(nb32/8)*292+64 -> 1816 B for the common nb32=48 shape), so the
   38.8us is fixed DRIVER SUBMISSION cost, not bandwidth. h2dp6 measured a SYNCHRONOUS
   467MB cuMemcpyHtoD at 5.99 GB/s vs depth-1 async 1.05 GB/s, hinting the sync entry
   point has a much cheaper host path than the async one on this Tegra driver.
   This probe measures, at engine sizes, every legal combination:
     H2D  = cuMemcpyHtoD_v2 (sync)        H2DA = cuMemcpyHtoDAsync_v2 (stream)
     D2H  = cuMemcpyDtoH_v2 (sync)        D2HA = cuMemcpyDtoHAsync_v2 + event poll
   plus isolated fixed-overhead numbers (0-byte copies, launch-only). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <stdint.h>
#include <time.h>

typedef int CUresult;
typedef unsigned long long CUdeviceptr;
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
typedef CUresult (*fn_launch)(void *, unsigned, unsigned, unsigned, unsigned,
                              unsigned, unsigned, unsigned, void *, void **, void **);

/* tiny gemv: y[i] = sum_j xq[j]^2 + 0.5  (grid=rows, 32 threads; engine-like shape) */
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

#define TIME_BLOCK(label, REP_, BODY) do {                        \
    double _t0 = ms();                                            \
    for (int _r = 0; _r < (REP_); _r++) { BODY; }                 \
    double _d = ms() - _t0;                                       \
    printf("%-34s %8.2f us/call  (total %.1f ms)\n",              \
           label, _d * 1e3 / (REP_), _d);                         \
} while (0)

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
    SYM(fn_launch, cuLaunchKernel)

    const int NROW = 2048, NIN = 454;        /* engine: nb32=48 -> 1816 B xq, 8 KB y */
    size_t BY = (size_t) NROW * 4, BI = (size_t) NIN * 4;
    const int R = 500;
    int i;

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

    unsigned n32 = NIN;
    void *args[3] = { &d_xq, &d_y, &n32 };

    printf("=== sizes: xq=%zu B  y=%zu B  rows=%d  reps=%d ===\n", BI, BY, NROW, R);

    /* fixed-overhead isolation */
    TIME_BLOCK("0B H2D async (submission)", R,
               CK(cuMemcpyHtoDAsync_v2(d_xq, h_in, 0, stream));
               CK(cuEventRecord(ev, stream));
               while (cuEventQuery(ev) == 600) {});
    TIME_BLOCK("0B H2D sync", R, CK(cuMemcpyHtoD_v2(d_xq, h_in, 0)));
    TIME_BLOCK("launch only + event poll", R,
               CK(cuLaunchKernel(fg, NROW,1,1, 32,1,1, 0, stream, args, 0));
               CK(cuEventRecord(ev, stream));
               while (cuEventQuery(ev) == 600) {});
    TIME_BLOCK("H2DA(1816B) + event poll", R,
               CK(cuMemcpyHtoDAsync_v2(d_xq, h_in, BI, stream));
               CK(cuEventRecord(ev, stream));
               while (cuEventQuery(ev) == 600) {});
    TIME_BLOCK("H2D sync(1816B)", R, CK(cuMemcpyHtoD_v2(d_xq, h_in, BI)));
    TIME_BLOCK("D2HA(8KB) + event poll", R,
               CK(cuMemcpyDtoHAsync_v2(h_out, d_y, BY, stream));
               CK(cuEventRecord(ev, stream));
               while (cuEventQuery(ev) == 600) {});
    TIME_BLOCK("D2H sync(8KB)", R, CK(cuMemcpyDtoH_v2(h_out, d_y, BY)));

    /* full per-call sequences (engine shapes), each ending in a completion signal */
    TIME_BLOCK("P1 H2DA+launch+D2HA+ev (engine)", R,
               CK(cuMemcpyHtoDAsync_v2(d_xq, h_in, BI, stream));
               CK(cuLaunchKernel(fg, NROW,1,1, 32,1,1, 0, stream, args, 0));
               CK(cuMemcpyDtoHAsync_v2(h_out, d_y, BY, stream));
               CK(cuEventRecord(ev, stream));
               while (cuEventQuery(ev) == 600) {});
    TIME_BLOCK("P2 H2Dsync+launch+D2HA+ev", R,
               CK(cuMemcpyHtoD_v2(d_xq, h_in, BI));
               CK(cuLaunchKernel(fg, NROW,1,1, 32,1,1, 0, stream, args, 0));
               CK(cuMemcpyDtoHAsync_v2(h_out, d_y, BY, stream));
               CK(cuEventRecord(ev, stream));
               while (cuEventQuery(ev) == 600) {});
    TIME_BLOCK("P3 H2Dsync+launch+D2Hsync", R,
               CK(cuMemcpyHtoD_v2(d_xq, h_in, BI));
               CK(cuLaunchKernel(fg, NROW,1,1, 32,1,1, 0, stream, args, 0));
               CK(cuMemcpyDtoH_v2(h_out, d_y, BY)));
    TIME_BLOCK("P4 H2DA+launch+D2Hsync", R,
               CK(cuMemcpyHtoDAsync_v2(d_xq, h_in, BI, stream));
               CK(cuLaunchKernel(fg, NROW,1,1, 32,1,1, 0, stream, args, 0));
               CK(cuMemcpyDtoH_v2(h_out, d_y, BY)));
    TIME_BLOCK("P5 launch+D2Hsync (no H2D)", R,
               CK(cuLaunchKernel(fg, NROW,1,1, 32,1,1, 0, stream, args, 0));
               CK(cuMemcpyDtoH_v2(h_out, d_y, BY)));

    /* correctness of the sync sequence (P3) vs P1 output */
    { float *ref = (float *) malloc(BY);
      CK(cuMemcpyHtoD_v2(d_xq, h_in, BI));
      CK(cuLaunchKernel(fg, NROW,1,1, 32,1,1, 0, stream, args, 0));
      CK(cuCtxSynchronize());
      CK(cuMemcpyDtoH_v2(ref, d_y, BY));
      CK(cuMemcpyHtoD_v2(d_xq, h_in, BI));
      CK(cuLaunchKernel(fg, NROW,1,1, 32,1,1, 0, stream, args, 0));
      CK(cuMemcpyDtoH_v2(h_out, d_y, BY));
      int bad = 0;
      for (i = 0; i < NROW; i++) if (((float *) h_out)[i] != ref[i]) bad++;
      printf("sync-sequence verify: mismatches=%d %s\n", bad, bad ? "MISMATCH" : "OK");
      free(ref); }

    printf("PROBE-DONE\n");
    return 0;
}
