/* shfltest2: per-form JIT+behavior for shfl.idx on sm_53. Each form is its own
 * module: lane t holds t, shuffles lane 12 out, stores 32 results. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
typedef int CUresult;
static CUresult (*pInit)(unsigned), (*pDev)(void *, int), (*pCtx)(void *, unsigned, void *);
static CUresult (*pMod)(void *, const void *), (*pFn)(void *, void *, const char *);
static CUresult (*pMA)(unsigned long long *, unsigned long long);
static CUresult (*pLU)(void *, int, int, int, int, int, int, unsigned, void *, void **, void **);
static CUresult (*pSync)(void), (*pDtoH)(void *, unsigned long long, unsigned long long);
static CUresult (*pErr)(CUresult, const char **);
static CUresult (*pModEx)(void *, const void *, unsigned, int *, void **);
static const char *HDR =
".version 4.0\n.target sm_50\n.address_size 64\n"
".visible .entry tk(.param .u64 po) {\n.reg .u32 %r<4>; .reg .u64 %rd<2>;\n"
"ld.param.u64 %rd1,[po]; mov.u32 %r1,%tid.x;\n";
static const char *TLR =
"mul.wide.u32 %rd2,%r1,4; add.u64 %rd2,%rd2,%rd1; st.global.u32 [%rd2],%r2; ret; }\n";
static const char *FORMS[3] = {
    "mov.u32 %r2,%r1;",
    "add.u32 %r2,%r1,100;",
    "mov.u32 %r3,12; shfl.idx.b32 %r2,%r1,%r3,31;",
};
int main(void) {
    void *h = dlopen("libcuda.so.1", RTLD_NOW);
    void *dev, *ctx;
    int k, i;
    if (!h) return 1;
    pInit = dlsym(h, "cuInit"); pDev = dlsym(h, "cuDeviceGet");
    pCtx = dlsym(h, "cuCtxCreate_v2"); pMod = dlsym(h, "cuModuleLoadData");
    pFn = dlsym(h, "cuModuleGetFunction"); pMA = dlsym(h, "cuMemAlloc_v2");
    pLU = dlsym(h, "cuLaunchKernel"); pSync = dlsym(h, "cuCtxSynchronize");
    pDtoH = dlsym(h, "cuMemcpyDtoH_v2");
    pErr = dlsym(h, "cuGetErrorString");
    pModEx = dlsym(h, "cuModuleLoadDataEx");
    if (pInit(0) || pDev(&dev, 0) || pCtx(&ctx, 0, dev)) { printf("SETUP FAIL\n"); return 1; }
    for (k = 0; k < 3; k++) {
        char ptx[1024];
        void *mod, *fn;
        unsigned long long d = 0;
        unsigned out[32];
        void *args[1];
        { FILE *df = fopen("/tmp/dbg0.ptx", "w"); fputs(HDR, df); fputs(FORMS[0], df); fputs("\n", df); fputs(TLR, df); fclose(df); }
        { const char *pf = getenv("PTX_FILE");
          if (pf) { FILE *ff = fopen(pf, "rb"); size_t zn = fread(ptx, 1, sizeof(ptx)-1, ff);
              ptx[zn] = 0; fclose(ff); }
          else snprintf(ptx, sizeof(ptx), "%s%s\n%s", HDR, FORMS[k], TLR); }
        { CUresult e; char log[4096]; int lv = 5; void *vals[3]; int opts[3];
          log[0] = 0;
          opts[0] = 5; vals[0] = log;
          opts[1] = 6; vals[1] = (void *)(unsigned long)sizeof(log);
          e = pModEx(&mod, ptx, 2, opts, vals);
          if (e) { printf("form%d [%s] JIT-FAIL log: %s\n", k, FORMS[k], log); continue; } }
        if (pFn(&fn, mod, "tk") || pMA(&d, sizeof(out))) { printf("form%d LOADFAIL\n", k); continue; }
        args[0] = &d;
        memset(out, 0xAA, sizeof(out));
        if (pLU(fn, 1, 1, 1, 32, 1, 1, 0, NULL, args, NULL) || pSync() ||
            pDtoH(out, d, sizeof(out))) { printf("form%d RUNFAIL\n", k); continue; }
        printf("form%d [%s]:", k, FORMS[k]);
        for (i = 0; i < 32; i++) printf(" %u", out[i]);
        printf("\n");
    }
    return 0;
}
