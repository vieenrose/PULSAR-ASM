/* regprobe: read-only introspection of a PTX kernel's resource usage.
 * Loads PTX via dlopened libcuda, prints NUM_REGS / SHARED_SIZE / MAX_THREADS.
 * No code change, no timing impact. Usage: regprobe kernel.ptx func */
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
typedef int CUresult; typedef void *CUdevice; typedef void *CUcontext;
typedef void *CUmodule; typedef void *CUfunction;
#define CU_FUNC_ATTRIBUTE_NUM_REGS 4
#define CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES 8
#define CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK 0
int main(int argc, char **argv) {
    void *h;
    CUresult (*pInit)(unsigned), (*pDevGet)(CUdevice *, int), (*pCtx)(CUcontext *, unsigned, CUdevice);
    CUresult (*pMod)(CUmodule *, const void *), (*pFn)(CUfunction *, CUmodule, const char *);
    CUresult (*pAttr)(int *, int, CUfunction);
    CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
    FILE *f; long sz; char *buf; int v;
    const int attrs[] = {4, 8, 0};
    const char *names[] = {"NUM_REGS", "SHARED_SIZE_BYTES", "MAX_THREADS_PER_BLOCK"};
    int i;
    if (argc < 3) return 2;
    h = dlopen("libcuda.so.1", RTLD_NOW);
    if (!h) h = dlopen("libcuda.so", RTLD_NOW);
    if (!h) { printf("no libcuda\n"); return 1; }
    pInit = dlsym(h, "cuInit"); pDevGet = dlsym(h, "cuDeviceGet");
    pCtx = dlsym(h, "cuCtxCreate_v2"); pMod = dlsym(h, "cuModuleLoadData");
    pFn = dlsym(h, "cuModuleGetFunction"); pAttr = dlsym(h, "cuFuncGetAttribute");
    f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); sz = ftell(f); fseek(f, 0, SEEK_SET);
    buf = malloc(sz + 1); fread(buf, 1, sz, f); buf[sz] = 0; fclose(f);
    if (pInit(0) || pDevGet(&dev, 0) || pCtx(&ctx, 0, dev) || pMod(&mod, buf)) {
        printf("cuda setup failed\n"); return 1; }
    if (pFn(&fn, mod, argv[2])) { printf("getfunc failed\n"); return 1; }
    for (i = 0; i < 3; i++) {
        if (pAttr(&v, attrs[i], fn)) { printf("%s ?) failed\n", names[i]); return 1; }
        printf("%s %d\n", names[i], v);
    }
    return 0;
}
