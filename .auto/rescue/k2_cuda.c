/* k2_cuda host runtime, Phase 1 (Q4 GEMV on GPU, CPU does rest, bit-exact).
 * Device-dependency-free: libc + dlopen(L4T libcuda) only, hand PTX sm_53
 * JITed by driver. No toolkit, no installs.
 * GEMV bit-exact vs CPU scalar: int32 dots exact; per-block fmul(ws,db) rn
 * then fma rn accumulate, blocks sequential, 1 thread/row. Tiled cooperative
 * loads (weights+sc, u64 addresses via cvt); xq shared-vector via L2.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>

typedef int CUresult;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUdev_st *CUdevice;
#define CUDA_SUCCESS 0
#define CU_MEMHOSTREGISTER_PORTABLE 0x01
static void *libcuda;
static CUresult (*pInit)(unsigned);
static CUresult (*pDevGet)(CUdevice *, int);
static CUresult (*pCtxCreate_v2)(CUcontext *, unsigned, CUdevice);
static CUresult (*pModLoadData)(CUmodule *, const void *);
static CUresult (*pModLoadDataEx)(CUmodule *, const void *, unsigned, int *,
                                  void **);
static char jit_log[4096];
static CUresult (*pModGetFunc)(CUfunction *, CUmodule, const char *);
static CUresult (*pMemHostRegister_v2)(void *, size_t, unsigned);
static CUresult (*pMemAllocManaged)(void **, size_t, unsigned);
static CUresult (*pMemAlloc)(uint64_t *, size_t);
static CUresult (*pMemcpyDtoH)(void *, uint64_t, size_t);
static CUresult (*pMemcpyHtoD)(uint64_t, const void *, size_t);
static CUresult (*pLaunch)(CUfunction, unsigned, unsigned, unsigned,
                           unsigned, unsigned, unsigned, unsigned, void *,
                           void **, void **);
static CUresult (*pCtxSync)(void);
static const char *(*pErrStr)(CUresult, const char **);
extern char __bss_start, _end, b_KCAbuf, b_VCAbuf;
extern uint64_t g_LBASE, g_O_HEAD;
extern uint32_t g_vocab;
extern uint64_t tbl_koff[9];
extern uint32_t tbl_krows[9], tbl_kcols[9];
extern uint64_t g_task_sc;
extern void k2q4_gemv_range(uint32_t, uint32_t, uint32_t, uint64_t, uint64_t,
                            uint64_t, uint64_t);
static CUmodule mod;
static CUfunction k_gemv;
static volatile int cuda_ok;
static uint64_t blob_bytes;

static uint32_t rows_from_w(uint64_t w) {
    uint64_t lb = g_LBASE, oh = g_O_HEAD;
    if (w >= oh && w < oh + (uint64_t)0x20000000)
        return g_vocab;
    if (w < lb) return 0;
    {
        uint64_t stride = 17707008ULL, il = (w - lb) / stride;
        uint64_t base = lb + il * stride;
        int k;
        for (k = 1; k <= 8; k++) {
            uint64_t ks;
            if (!tbl_kcols[k]) continue;
            ks = base + tbl_koff[k];
            if (w >= ks &&
                w < ks + (uint64_t)tbl_krows[k] * (tbl_kcols[k] / 32) * 18ULL)
                return tbl_krows[k];
        }
    }
    return 0;
}

static const char *ptx =
".version 6.0\n.target sm_53\n.address_size 64\n"
".visible .entry q4_gemv(\n"
".param .u64 w,.param .u64 sc,.param .u64 xq,.param .u64 y,\n"
".param .u32 r0,.param .u32 r1,.param .u32 nb32,.param .u32 rowsf)\n{\n"
".reg .u64 %rd<16>; .reg .u32 %r<48>; .reg .f32 %f<12>; .reg .pred %p<8>;\n"
"ld.param.u64 %rd1,[w]; ld.param.u64 %rd2,[sc];\n"
"ld.param.u64 %rd3,[xq]; ld.param.u64 %rd4,[y];\n"
"ld.param.u32 %r10,[r0]; ld.param.u32 %r11,[r1];\n"
"ld.param.u32 %r12,[nb32];\n"
"mov.u32 %r5,%tid.x; mov.u32 %r6,%ctaid.x;\n"
"mad.lo.u32 %r7,%r6,32,%r10;\n"
"add.u32 %r7,%r7,%r5;\n"
"setp.lt.u32 %p1,%r7,%r11;\n"
"@!%p1 bra BB_EXIT;\n"
"mul.lo.u32 %r8,%r12,18;\n"
"mul.lo.u32 %r8,%r7,%r8;\n"
"cvt.u64.u32 %rd5,%r8; add.u64 %rd5,%rd5,%rd1;\n"
"mul.lo.u32 %r8,%r7,%r12;\n"
"mul.lo.u32 %r8,%r8,4;\n"
"cvt.u64.u32 %rd6,%r8; add.u64 %rd6,%rd6,%rd2;\n"
"mov.u64 %rd9,2; mov.u64 %rd10,18; mov.u64 %rd11,1; mov.u64 %rd12,4;\n"
"add.u64 %rd7,%rd5,%rd9;\n"
"mov.f32 %f1,0F00000000;\n"
"mov.u32 %r8,0;\n"
"BB_BLK:\nsetp.ge.u32 %p2,%r8,%r12; @%p2 bra BB_BLK_DONE;\n"
"shr.u32 %r9,%r8,3;\n"
"mul.lo.u32 %r9,%r9,292;\n"
"cvt.u64.u32 %rd8,%r9; add.u64 %rd8,%rd8,%rd3;\n"
"ld.global.f32 %f2,[%rd8];\n"
"and.b32 %r9,%r8,7;\n"
"mul.lo.u32 %r9,%r9,32;\n"
"add.u64 %rd8,%rd8,%rd12;\n"
"cvt.u64.u32 %rd13,%r9; add.u64 %rd8,%rd8,%rd13;\n"
"ld.global.u32 %r9,[%rd6];\n"
"mov.b32 %f3,%r9;\n"
"add.u64 %rd6,%rd6,%rd12;\n"
"mov.u32 %r9,0;\n"
"mov.u32 %r10,0;\n"
"BB_BYTE:\nsetp.ge.u32 %p3,%r10,16; @%p3 bra BB_BYTE_DONE;\n"
"ld.global.u8 %r5,[%rd7];\n"
"ld.global.s8 %r6,[%rd8];\n"
"ld.global.s8 %r13,[%rd8+16];\n"
"and.b32 %r14,%r5,15;\n"
"sub.s32 %r14,%r14,8;\n"
"mad.lo.s32 %r9,%r14,%r6,%r9;\n"
"shr.u32 %r5,%r5,4;\n"
"sub.s32 %r5,%r5,8;\n"
"mad.lo.s32 %r9,%r5,%r13,%r9;\n"
"add.u64 %rd7,%rd7,%rd11;\n"
"add.u64 %rd8,%rd8,%rd11;\n"
"add.u32 %r10,%r10,1; bra BB_BYTE;\n"
"BB_BYTE_DONE:\n"
"cvt.rn.f32.s32 %f4,%r9;\n"
"mul.f32 %f5,%f3,%f2;\n"
"fma.rn.f32 %f1,%f5,%f4,%f1;\n"
"add.u64 %rd7,%rd7,%rd9;\n"
"add.u32 %r8,%r8,1; bra BB_BLK;\n"
"BB_BLK_DONE:\n"
"mul.lo.u32 %r8,%r7,4;\n"
"cvt.u64.u32 %rd5,%r8; add.u64 %rd5,%rd5,%rd4;\n"
"st.global.f32 [%rd5],%f1;\n"
"BB_EXIT:\n"
"ret;\n}\n";

static void ck(CUresult r, const char *m) {
    if (r) {
        const char *s = "?";
        if (pErrStr) pErrStr(r, &s);
        printf("CUDA %s: %d %s\n", m, r, s);
        fflush(stdout);
    }
}

static uint64_t blob_vaddr; static size_t blob_bytes_g;
static uint64_t dev_base, dev_xq, dev_sc, dev_y; /* device-resident buffers */
static int sc_staged = 0;
#define SC_REGION 134217728ULL
#define SC_CAP 16777216u
#define XQ_CAP 65536u
#define Y_CAP 524288u
static CUcontext g_ctx;
static CUresult (*pCtxSetCurrent)(CUcontext);
static int gpu_lock;

void cuda_init(void) {
    uint32_t h[10];
    size_t n0 = 0, n1 = 0;
    setvbuf(stdout, NULL, _IONBF, 0);
    if (cuda_ok) return;
    libcuda = dlopen("libcuda.so.1", RTLD_NOW);
    if (!libcuda) { printf("no libcuda, CPU only\n"); return; }
#define SYM(v, s) p##v = dlsym(libcuda, "cu" s); if (!p##v) { printf("no cu" s "\n"); return; }
    SYM(Init, "Init") SYM(DevGet, "DeviceGet") SYM(CtxCreate_v2, "CtxCreate_v2")
    SYM(ModLoadData, "ModuleLoadData") SYM(ModGetFunc, "ModuleGetFunction")
    SYM(MemHostRegister_v2, "MemHostRegister_v2")
    pMemAllocManaged = dlsym(libcuda, "cuMemAllocManaged");
    pMemAlloc = (CUresult (*)(uint64_t *, size_t))dlsym(libcuda, "cuMemAlloc_v2");
    pMemcpyDtoH = (CUresult (*)(void *, uint64_t, size_t))dlsym(libcuda, "cuMemcpyDtoH_v2");
    pMemcpyHtoD = dlsym(libcuda, "cuMemcpyHtoD_v2");
    pModLoadDataEx = dlsym(libcuda, "cuModuleLoadDataEx");
    pLaunch = dlsym(libcuda, "cuLaunchKernel");
    pCtxSync = dlsym(libcuda, "cuCtxSynchronize");
    pErrStr = dlsym(libcuda, "cuGetErrorString");
    if (!pLaunch || !pCtxSync) { printf("no launch/sync\n"); return; }
    { CUresult r = pInit(0); if (r) { ck(r, "init"); return; } }
    { CUdevice d; CUcontext c;
      if (pDevGet(&d, 0)) return;
      if (pCtxCreate_v2(&c, 0, d)) return;
      g_ctx = c;
      pCtxSetCurrent = (CUresult (*)(CUcontext))dlsym(libcuda, "cuCtxSetCurrent");
      if (pModLoadDataEx) {
          int jit_opt[] = { 1 /*ERROR_LOG_BUFFER*/, 2 /*LOG_VERBOSE*/, 0 };
          void *jit_val[] = { jit_log, (void *)1, NULL };
          jit_log[0] = 0; jit_log[sizeof(jit_log)-1] = 0;
          { CUresult r = pModLoadDataEx(&mod, ptx, 2, jit_opt, jit_val);
            if (r) { printf("PTX JIT failed %d: %.2000s\n", r, jit_log); return; } }
      } else if (pModLoadData(&mod, ptx)) { printf("PTX JIT failed (no log)\n"); return; }
      if (pModGetFunc(&k_gemv, mod, "q4_gemv")) { printf("no kernel\n"); return; } }
    memcpy(h, (void *)(uintptr_t)g_LBASE, 40);
    { /* exact mapped size (any blob): find mapping containing LBASE */
      FILE *m = fopen("/proc/self/maps", "r");
      uint64_t lo = 0, hi = 0, lb = g_LBASE;
      char line[512];
      blob_bytes = 0;
      if (m) {
        while (fgets(line, sizeof(line), m)) {
          if (sscanf(line, "%lx-%lx", &lo, &hi) == 2 && lb >= lo && lb < hi) {
            blob_bytes = (size_t)(hi - lo);
            break;
          }
        }
        fclose(m);
      }
      printf("CUDA blob mapping %llu\n", (unsigned long long)blob_bytes);
      if (!blob_bytes) { printf("no blob mapping\n"); return; }
      blob_vaddr = g_LBASE; blob_bytes_g = blob_bytes;
      { int mr = mlock((void *)(uintptr_t)g_LBASE, blob_bytes);
        if (mr) printf("mlock blob failed (errno %d)\n", errno); }
      { /* touch the data buffers GPU writes (activations b_SC, logits
           b_HEADB) so pages are resident before kernel touches them.
           Volatile byte loop: FORTIFY-safe, touches ONLY kernel-touched
           buffers (bss header globals + dlsym table stay untouched). */
        extern char b_SC[], b_HEADB[];
        volatile char *tp; size_t n, i;
        tp = (volatile char *)b_SC;   n = 134217728;
        for (i = 0; i < n; i += 4096) tp[i] = 0;
        tp = (volatile char *)b_HEADB; n = 4u * 1024 * 1024;
        for (i = 0; i < n; i += 4096) tp[i] = 0;
        printf("CUDA bss touched (b_SC 128M + b_HEADB 4M)\n"); }
      { /* non-UVA driver: kernels can only dereference cuMemAlloc'd device
           memory (proven: host pointers fault 719). Stage weights once. */
        CUresult r;
        if (!pMemAlloc || !pMemcpyDtoH) { printf("no memalloc/dtoH\n"); return; }
        r = pMemAlloc(&dev_base, blob_bytes);
        if (r) { ck(r, "memalloc-blob"); return; }
        r = pMemcpyHtoD(dev_base, (void *)(uintptr_t)g_LBASE, blob_bytes);
        if (r) { ck(r, "stage-blob"); return; }
        r = pMemAlloc(&dev_xq, XQ_CAP);
        if (r) { ck(r, "memalloc-xq"); return; }
        r = pMemAlloc(&dev_y, Y_CAP);
        if (r) { ck(r, "memalloc-y"); return; }
        printf("CUDA weights staged on device\n");
      }
      cuda_ok = 1;
      printf("CUDA ok device-staged sm_53\n");
    }
}

extern uint64_t g_task_sc;
extern void k2q4_gemv_range(uint32_t, uint32_t, uint32_t, uint64_t, uint64_t,
                            uint64_t, uint64_t);
__attribute__((visibility("default")))
int cuda_is_ok(void) { return cuda_ok; }

static int selftest_remaining = 8;
static int dbg = -1;

/* Independent C reference for Q4 GEMV (no .S): bit-exact with engine by
   construction — int32 dot is associative; fp path per block = ONE mul
   (ws*db) + ONE fma accumulating in block order (same order as .S). */
static void cuda_q4_ref(uint32_t rows, uint32_t cols, const uint8_t *w,
                        const uint64_t scb, const int8_t *xq, float *out) {
    uint32_t nb32 = cols >> 5;
    uint32_t r, b;
    for (r = 0; r < rows; r++) {
        const uint8_t *wp = w + (uint64_t)r * nb32 * 18 + 2;
        const float *sp = (const float *)(uintptr_t)scb + (uint64_t)r * nb32;
        float acc = 0.0f;
        for (b = 0; b < nb32; b++) {
            const int8_t *xb = xq + (uint64_t)(b >> 3) * 292 + 4 + (b & 7) * 32;
            int32_t dot = 0;
            int k;
            for (k = 0; k < 16; k++) {
                int lo = (wp[k] & 15) - 8;
                int hi = (wp[k] >> 4) - 8;
                dot += lo * (int)xb[k] + hi * (int)xb[16 + k];
            }
            { float db = ((const float *)(uintptr_t)xq)[(b >> 3) * 73];
              acc = (sp[b] * db) * (float)dot + acc; }
            wp += 18;
        }
        out[r] = acc;
    }
}

__attribute__((visibility("default")))
void cuda_gemv(uint32_t r0, uint32_t r1, uint32_t cols, uint64_t w,
               uint64_t x_act_unused, uint64_t y, uint64_t xq) {
    uint32_t nb32 = cols >> 5;
    uint32_t rowsf = rows_from_w(w);
    uint64_t dev_w, d_sc = dev_sc, d_xq = dev_xq, d_y = dev_y, scb;
    size_t nrows, sc_bytes, xq_bytes;
    void *args[] = { &dev_w, &d_sc, &d_xq, &d_y, (void *)&r0, (void *)&r1,
                     (void *)&nb32, (void *)&rowsf };
    unsigned grid;
    CUresult r;
    extern uint32_t g_nthreads;
    if (!cuda_ok || !rowsf || g_nthreads != 1) {
        /* CPU path (bit-exact): GPU stays threads-1 only — at threads>1
           per-call staging + one global lock serialize the workers. */
        k2q4_gemv_range(r0, r1, cols, w, x_act_unused, y, xq);
        return;
    }
    grid = (r1 > r0) ? (r1 - r0 + 31) / 32 : 0;
    if (!grid) return;
    nrows = (size_t)(r1 - r0);
    if (w < blob_vaddr || w - blob_vaddr >= blob_bytes_g) {
        k2q4_gemv_range(r0, r1, cols, w, x_act_unused, y, xq); return; }
    dev_w = dev_base + (w - blob_vaddr);
    scb = g_task_sc;
    { extern char b_SC[];
      uint64_t sc_lo = (uint64_t)(uintptr_t)b_SC;
      if (!sc_staged) {
          if (pMemAlloc(&dev_sc, SC_REGION)) {
              printf("CUDA sc-mirror alloc failed\n"); cuda_ok = 0;
              k2q4_gemv_range(r0, r1, cols, w, x_act_unused, y, xq); return;
          }
          if (pMemcpyHtoD(dev_sc, (void *)(uintptr_t)b_SC, SC_REGION)) {
              printf("CUDA sc-mirror copy failed\n"); cuda_ok = 0;
              k2q4_gemv_range(r0, r1, cols, w, x_act_unused, y, xq); return;
          }
          sc_staged = 1;
          printf("CUDA sc mirror staged (128MB)\n");
      }
      if (scb < sc_lo || scb + (size_t)rowsf * nb32 * 4 > sc_lo + SC_REGION) {
          k2q4_gemv_range(r0, r1, cols, w, x_act_unused, y, xq); return; }
      d_sc = dev_sc + (scb - sc_lo);
    }
    sc_bytes = nrows * (size_t)nb32 * 4;
    xq_bytes = ((size_t)nb32 / 8 + 1) * 292 + 64;
    if (xq_bytes > XQ_CAP || nrows * 4 > Y_CAP) {
        k2q4_gemv_range(r0, r1, cols, w, x_act_unused, y, xq); return; }
    while (__atomic_exchange_n(&gpu_lock, 1, __ATOMIC_ACQUIRE))
        __asm__ volatile("yield");
    if (pCtxSetCurrent) pCtxSetCurrent(g_ctx);
    /* stage inputs to device (absolute-row layout preserved at r0 offsets) */
    (void)sc_bytes;
    r = pMemcpyHtoD(d_xq, (void *)(uintptr_t)xq, xq_bytes);
    if (!r) r = pLaunch(k_gemv, grid, 1, 1, 32, 1, 1, 0, NULL, args, NULL);
    if (!r) r = pCtxSync();
    if (!r) r = pMemcpyDtoH((void *)(uintptr_t)(y + (uint64_t)r0 * 4),
                            d_y + (uint64_t)r0 * 4, nrows * 4);
    if (r) { ck(r, "gpu-path"); cuda_ok = 0; }
    /* Self-test v3: first 8 GEMV calls, full row range, INSIDE gpu_lock,
       dedicated host+device scratch — engine y is NEVER touched (peer #6:
       writing y clobbered caller state -> SIGSEGV). */
    if (cuda_ok && selftest_remaining > 0) {
        static float s_gpu[65536], s_cpu[65536];
        static uint64_t dev_s = 0;
        uint32_t t_r0 = 0, t_r1 = rowsf;
        size_t full = (size_t)rowsf * 4;
        void *targs[] = { &dev_w, &d_sc, &d_xq, &dev_s,
                          &t_r0, &t_r1, &nb32, &rowsf };
        CUresult r2 = 0;
        int pass = 1;
        selftest_remaining--;
        if (full > sizeof(s_gpu)) pass = 0;
        if (pass && !dev_s) {
            if (pMemAlloc(&dev_s, sizeof(s_gpu))) {
                printf("CUDA scratch-alloc failed -> CPU mode\n");
                cuda_ok = 0; pass = 0;
            }
        }
        if (pass) {
            r2 = pLaunch(k_gemv, (rowsf + 31) / 32, 1, 1, 32, 1, 1,
                         0, NULL, targs, NULL);
            if (!r2) r2 = pCtxSync();
            if (!r2) r2 = pMemcpyDtoH(s_gpu, dev_s, full);
            if (r2) { ck(r2, "selftest-path"); pass = 0; }
        }
        if (pass) {
            cuda_q4_ref(rowsf, cols, (const uint8_t *)(uintptr_t)w,
                        scb, (const int8_t *)(uintptr_t)xq, s_cpu);
            if (memcmp(s_gpu, s_cpu, full)) {
                size_t i, nf = full / 4;
                for (i = 0; i < nf; i++)
                    if (s_gpu[i] != s_cpu[i]) break;
                printf("CUDA selftest diff idx=%u gpu=%f cpu=%f\n",
                       (unsigned)i, (double)s_gpu[i], (double)s_cpu[i]);
                pass = 0;
            }
        }
        if (!pass) { printf("CUDA self-test MISMATCH -> CPU mode\n"); cuda_ok = 0; }
        else if (selftest_remaining == 0)
            printf("CUDA self-test OK (8/8 bit-exact)\n");
    }
    if (dbg < 0) dbg = getenv("K2DBG") ? 1 : 0;
    if (dbg && cuda_ok && selftest_remaining >= 4) {
        static float u_gpu[4], u_cpu[4];
        uint32_t u_r0 = 0, u_r1 = 1, u_nb = 1, u_rf = 1;
        void *uargs[] = { &dev_w, &d_sc, &d_xq, &d_y,
                          &u_r0, &u_r1, &u_nb, &u_rf };
        CUresult ru = pLaunch(k_gemv, 1, 1, 1, 32, 1, 1, 0, NULL, uargs, NULL);
        if (!ru) ru = pCtxSync();
        if (!ru) ru = pMemcpyDtoH(u_gpu, d_y, 4);
        u_cpu[0] = 0;
        cuda_q4_ref(1, 32, (const uint8_t *)(uintptr_t)w,
                    scb, (const int8_t *)(uintptr_t)xq, u_cpu);
        printf("CUDA DBG woff=%llx gpu=%f cpu=%f ru=%d\n",
               (unsigned long long)(w - blob_vaddr), (double)u_gpu[0],
               (double)u_cpu[0], (int)ru);
    }
    if (pCtxSetCurrent) pCtxSetCurrent(0);
    __atomic_store_n(&gpu_lock, 0, __ATOMIC_RELEASE);
    if (!cuda_ok) k2q4_gemv_range(r0, r1, cols, w, x_act_unused, y, xq);
}
