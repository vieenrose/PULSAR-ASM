/* Exp43 probe v3c: ZERO-COPY staging on R440 Tegra (non-UVA driver).
   The engine pays ~39us H2D + ~21us D2H per GEMV call (~60us x ~230 calls/token =
   14ms of the 149ms/token). On Tegra the CPU and GPU share LPDDR4, so a pinned
   host buffer should be readable AND writable by kernels through the device pointer
   returned by cuMemHostGetDevicePointer - no copy in either direction.
   Tests with data verification in both directions:
     1. cuMemHostGetDevicePointer on a cuMemHostAlloc buffer: rc + pointer value
     2. WRITE path: kernel writes a pattern through the pinned device pointer;
        CPU reads it back through the HOST pointer (GPU -> host visibility).
     3. READ path: CPU writes a pattern into the pinned buffer; kernel reads it
        through the device pointer, doubles it into a normal device buffer, D2H
        back to host, CPU verifies (host -> GPU visibility).
   PTX is ISA 4.0 / sm_50 (R440's ptxas rejects .version 6.0). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <stdint.h>
#include <time.h>

typedef int CUresult;
typedef CUresult (*fn_init)(unsigned);
typedef CUresult (*fn_dev)(int *, int);
typedef CUresult (*fn_ctx)(void **, unsigned, int);
typedef CUresult (*fn_mod)(void *, const char *);
typedef CUresult (*fn_getfn)(void **, void *, const char *);
typedef CUresult (*fn_alloc)(uint64_t *, size_t);
typedef CUresult (*fn_halloc)(void **, size_t, unsigned);
typedef CUresult (*fn_sync)(void);
typedef CUresult (*fn_d2h)(void *, uint64_t, size_t);
typedef CUresult (*fn_h2d)(uint64_t, const void *, size_t);
typedef CUresult (*fn_launch)(void *, unsigned, unsigned, unsigned, unsigned,
                              unsigned, unsigned, unsigned, void *, void **, void **);
typedef CUresult (*fn_h2dptr)(uint64_t *, void *, unsigned);
typedef CUresult (*fn_set32)(uint64_t, unsigned, size_t);

/* wr: dst[i] = 1234.5 + i        (dst = pinned host buffer, passed as DEVICE ptr)
   rd2x: out[i] = src[i]*2        (src = pinned host buffer, passed as DEVICE ptr) */
static const char *ptx =
".version 4.0\n"
".target sm_50\n"
".address_size 64\n"
".visible .entry wr(.param .u64 pd) {\n"
"  .reg .u64 %rd<4>;\n  .reg .u32 %r<4>;\n  .reg .f32 %f<4>;\n"
"  ld.param.u64 %rd1, [pd];\n"
"  mov.u32 %r1, %ctaid.x;\n"
"  mul.wide.u32 %rd2, %r1, 4;\n"
"  add.u64 %rd1, %rd1, %rd2;\n"
"  mov.f32 %f1, 0F441A4000;\n"
"  cvt.rn.f32.u32 %f2, %r1;\n"
"  add.f32 %f1, %f1, %f2;\n"
"  st.global.f32 [%rd1], %f1;\n"
"  ret;\n"
"}\n"
".visible .entry rd2x(.param .u64 ps, .param .u64 pd) {\n"
"  .reg .u64 %rd<4>;\n  .reg .u32 %r<4>;\n  .reg .f32 %f<4>;\n"
"  ld.param.u64 %rd1, [ps];\n"
"  ld.param.u64 %rd2, [pd];\n"
"  mov.u32 %r1, %ctaid.x;\n"
"  mul.wide.u32 %rd3, %r1, 4;\n"
"  add.u64 %rd1, %rd1, %rd3;\n"
"  add.u64 %rd2, %rd2, %rd3;\n"
"  ld.global.f32 %f1, [%rd1];\n"
"  add.f32 %f1, %f1, %f1;\n"
"  st.global.f32 [%rd2], %f1;\n"
"  ret;\n"
"}\n";

#define CK(x) do { CUresult _r = (x); if (_r) { \
    printf("FAIL %s rc=%d\n", #x, _r); return 1; } } while (0)

int main(void) {
    void *lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (!lib) { printf("no libcuda\n"); return 1; }
#define SYM(T, N) T N = (T) dlsym(lib, #N); if (!N) printf("MISSING %s\n", #N);
    SYM(fn_init, cuInit) SYM(fn_dev, cuDeviceGet) SYM(fn_ctx, cuCtxCreate_v2)
    SYM(fn_mod, cuModuleLoadData) SYM(fn_getfn, cuModuleGetFunction)
    SYM(fn_alloc, cuMemAlloc_v2) SYM(fn_halloc, cuMemHostAlloc)
    SYM(fn_sync, cuCtxSynchronize)
    SYM(fn_d2h, cuMemcpyDtoH_v2) SYM(fn_h2d, cuMemcpyHtoD_v2)
    SYM(fn_launch, cuLaunchKernel)
    SYM(fn_set32, cuMemsetD32_v2)

    fn_h2dptr pGetDevPtr = (fn_h2dptr) dlsym(lib, "cuMemHostGetDevicePointer_v2");
    if (!pGetDevPtr) pGetDevPtr = (fn_h2dptr) dlsym(lib, "cuMemHostGetDevicePointer");
    printf("cuMemHostGetDevicePointer resolved=%p\n", (void *) pGetDevPtr);
    if (!pGetDevPtr) { printf("NO ZERO-COPY API\n"); return 2; }

    CK(cuInit(0));
    int dev = 0; void *ctx = 0;
    CK(cuDeviceGet(&dev, 0));
    CK(cuCtxCreate_v2(&ctx, 0, dev));
    void *mod = 0, *fwr = 0, *frd = 0;
    CK(cuModuleLoadData(&mod, ptx));
    CK(cuModuleGetFunction(&fwr, mod, "wr"));
    CK(cuModuleGetFunction(&frd, mod, "rd2x"));

    const int N = 4096;                       /* 16 KB: engine xq/y are ~40KB/256KB */
    size_t BYTES = (size_t) N * 4;
    int i, rep;

    uint64_t d_dev = 0;
    CK(cuMemAlloc_v2(&d_dev, BYTES));
    void *h_pin = 0;
    CK(cuMemHostAlloc(&h_pin, BYTES, 1));      /* CU_MEMHOSTALLOC_PORTABLE */

    uint64_t d_pin = 0;
    CUresult rp = pGetDevPtr(&d_pin, h_pin, 0);
    printf("HostGetDevicePointer rc=%d  h_pin=%p d_pin=0x%llx\n",
           rp, h_pin, (unsigned long long) d_pin);
    if (rp) { printf("NO ZERO-COPY POINTER (rc=%d)\n", rp); return 2; }

    /* ---- test 2: GPU writes through the pinned device pointer, CPU reads host view */
    { void *args[1] = { &d_pin };
      CK(cuLaunchKernel(fwr, N,1,1, 1,1,1, 0, 0, args, 0));
      CK(cuCtxSynchronize()); }
    { int bad = 0, first = -1;
      for (i = 0; i < N; i++) {
          float e = 1234.5f + (float) i;
          if (((float *) h_pin)[i] != e) { bad++; if (first < 0) first = i; }
      }
      printf("WRITE-through (GPU->pinned, CPU read): mismatches=%d %s first=%d got=%f\n",
             bad, bad ? "NOT VISIBLE" : "OK zero-copy WRITE path",
             first, first >= 0 ? *(float *) ((char *) h_pin + 4 * first) : 0.0f); }

    /* ---- test 3: CPU writes pinned, GPU reads via d_pin, writes device buf, D2H */
    for (i = 0; i < N; i++) ((float *) h_pin)[i] = (float)(i + 7);
    if (cuMemsetD32_v2) CK(cuMemsetD32_v2(d_dev, 0, N));
    { void *args[2] = { &d_pin, &d_dev };
      CK(cuLaunchKernel(frd, N,1,1, 1,1,1, 0, 0, args, 0));
      CK(cuCtxSynchronize()); }
    { float *back = (float *) malloc(BYTES);
      CK(cuMemcpyDtoH_v2(back, d_dev, BYTES));
      int bad = 0, first = -1;
      for (i = 0; i < N; i++)
          if (back[i] != (float)(i + 7) * 2.0f) { bad++; if (first < 0) first = i; }
      printf("READ-through (CPU->pinned, GPU read): mismatches=%d %s first=%d got=%f want=%f\n",
             bad, bad ? "GPU SEES STALE DATA (H2D still required)" : "OK zero-copy READ path",
             first, back[first < 0 ? 0 : first], (float)((first < 0 ? 0 : first) + 7) * 2.0f);
      free(back); }

    printf("PROBE-DONE\n");
    return 0;
}
