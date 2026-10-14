/* Exp42 pre-bench probe #2: why does cuMemcpyHtoDAsync fail with rc=1 after a few
   enqueues? h2dp5 showed slots 0-3 OK then rc=1 (INVALID_VALUE) at slot 4 in every
   state, with a 2-deep queue (sync every 2 enqueues). Hypothesis: R440 limits the
   number of queued async copies per stream. Test DEPTH-1 (enqueue, then sync before
   the next enqueue) vs the engine's real 467MB footprint, and measure throughput at
   several slot sizes so the ring size in v49 can be chosen from data. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <time.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

typedef int CUresult;
static CUresult (*pInit)(unsigned);
static CUresult (*pDevGet)(int *, int);
static CUresult (*pCtxCreate)(void **, unsigned, int);
static CUresult (*pMemAlloc)(unsigned long long *, size_t);
static CUresult (*pHostAlloc)(void **, size_t, unsigned);
static CUresult (*pH2DA)(unsigned long long, const void *, size_t, void *);
static CUresult (*pH2D)(unsigned long long, const void *, size_t);
static CUresult (*pStreamCreate)(void **, unsigned);
static CUresult (*pStreamSync)(void *);
static CUresult (*pFree)(unsigned long long, size_t);

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

#define TOTAL ((size_t)467 << 20)

/* mode 0: sync after every enqueue (queue depth 1). mode 1: sync every k enqueues. */
static void ring_test(const char *tag, unsigned long long d, size_t slot, int sync_every) {
    void *s[2]; void *st = 0; int i, rc, n = 0;
    double t0, tprod = 0, tcopy = 0, tsync = 0, a;
    int slots = (int)(TOTAL / slot);
    if (pHostAlloc(&s[0], slot, 0) || pHostAlloc(&s[1], slot, 0)) { printf("  pin fail\n"); return; }
    memset(s[0], 1, slot); memset(s[1], 2, slot);
    pStreamCreate(&st, 0);
    t0 = now_ms();
    for (i = 0; i < slots; i++) {
        a = now_ms();
        memset(s[i & 1], i, slot);                     /* producer (repack stand-in) */
        tprod += now_ms() - a;
        a = now_ms();
        rc = pH2DA(d + (size_t)i * slot, s[i & 1], slot, st);
        tcopy += now_ms() - a;
        if (rc) { printf("  %s h2da rc=%d at slot %d\n", tag, rc, i); goto out; }
        if (sync_every && (i % sync_every) == sync_every - 1) {
            a = now_ms(); pStreamSync(st); tsync += now_ms() - a; n++;
        }
    }
    a = now_ms(); pStreamSync(st); tsync += now_ms() - a;
    printf("  %-14s slot=%2luMB sync_every=%d: total %.0f ms (%.2f GB/s)  prod %.0f copy %.0f sync %.0f\n",
           tag, (unsigned long)(slot >> 20), sync_every, now_ms() - t0,
           (double)TOTAL / (now_ms() - t0) / 1e6, tprod, tcopy, tsync);
out:
    pStreamSync(st);
    if (pFree) pFree(d, TOTAL);
}

int main(int argc, char **argv) {
    void *L = dlopen("libcuda.so.1", RTLD_NOW);
    int dev; void *ctx = 0; unsigned long long d = 0;
    int m = argc > 1 ? atoi(argv[1]) : 0;
    if (!L) { printf("no libcuda\n"); return 1; }
    pInit = dlsym(L, "cuInit");
    pDevGet = dlsym(L, "cuDeviceGet");
    pCtxCreate = (CUresult (*)(void **, unsigned, int))dlsym(L, "cuCtxCreate_v2");
    pMemAlloc = (CUresult (*)(unsigned long long *, size_t))dlsym(L, "cuMemAlloc_v2");
    pHostAlloc = (CUresult (*)(void **, size_t, unsigned))dlsym(L, "cuMemHostAlloc");
    pH2DA = (CUresult (*)(unsigned long long, const void *, size_t, void *))dlsym(L, "cuMemcpyHtoDAsync_v2");
    pH2D = (CUresult (*)(unsigned long long, const void *, size_t))dlsym(L, "cuMemcpyHtoD_v2");
    pStreamCreate = (CUresult (*)(void **, unsigned))dlsym(L, "cuStreamCreate");
    pStreamSync = (CUresult (*)(void *))dlsym(L, "cuStreamSynchronize");
    pFree = (CUresult (*)(unsigned long long, size_t))dlsym(L, "cuMemFree_v2");
    pInit(0); pDevGet(&dev, 0); pCtxCreate(&ctx, 0, dev);

    { int rc = pMemAlloc(&d, TOTAL);
      printf("device alloc 467MB rc=%d addr=%llx\n", rc, (unsigned long long)d);
      if (rc) return 1; }

    if (m == 0) {
        printf("-- depth-1 (sync every enqueue)\n");
        ring_test("depth1", d, (size_t)32 << 20, 1);
        pMemAlloc(&d, TOTAL);
        printf("-- depth-1, 8MB slots\n");
        ring_test("d1-8MB", d, (size_t)8 << 20, 1);
        pMemAlloc(&d, TOTAL);
        printf("-- depth-1, 4MB slots\n");
        ring_test("d1-4MB", d, (size_t)4 << 20, 1);
        pMemAlloc(&d, TOTAL);
        printf("-- depth-2 (h2dp5 pattern, reproduce)\n");
        ring_test("depth2", d, (size_t)32 << 20, 2);
        pMemAlloc(&d, TOTAL);
        printf("-- serial reference: one big cuMemcpyHtoD of 467MB\n");
        { void *big = 0; double t;
          int rc = pHostAlloc(&big, TOTAL, 0);
          printf("  pin 467MB rc=%d\n", rc);
          if (!rc) {
              memset(big, 7, TOTAL);
              t = now_ms();
              rc = pH2D(d, big, TOTAL);
              printf("  serial H2D 467MB rc=%d %.0f ms (%.2f GB/s)\n",
                     rc, now_ms() - t, (double)TOTAL / (now_ms() - t) / 1e6);
          } }
    }
    return 0;
}
