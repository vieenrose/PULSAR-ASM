/* k2_cuda host runtime v12 (Exp6: 4-row blocking on single-copy 16B repack).
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
#include <sched.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>

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
static CUresult (*pMemcpyHtoDAsync)(uint64_t, const void *, size_t, void *);
static CUresult (*pMemcpyDtoHAsync)(void *, uint64_t, size_t, void *);
static CUresult (*pMemHostAlloc)(void **, size_t, unsigned);
static CUresult (*pMemFreeHost)(void *);
static CUresult (*pStreamSync)(void *);
static void *g_stream;   /* tentative; defined with the event-completion state below */

/* Exp42 (run #69): init-phase timing. The metric window includes process init
   (T1 is taken before exec); engine init is 1.74 s of a 34 s window = 5.1%, so
   init is a real lever. Print unconditionally: bench stdout goes to a file. */
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}
/* NEVER call cuGetErrorString with rc==0 - it SEGFAULTS on this R440 driver. */
/* Exp42 (run #69): the repack is 29.2M x memcpy(dst, src+2, 16). repackprobe2 on the
   Nano: glibc memcpy costs ~1007 ms per 910 MB destination (call overhead dominates a
   16-byte copy: 0.58 GB/s) while a packed-struct 16-byte assignment runs the SAME
   byte-for-byte image in 287 ms - 3.5x faster, verified byte-identical at -O2 and -O3
   over 910 MB. That is ~700 ms of the 981 ms "repack + H2D" init phase. */
typedef struct { uint64_t a, b; } __attribute__((packed, aligned(1))) q4blk;

static const char *k2err(int rc) {
    switch (rc) {
    case 0:   return "OK";
    case 1:   return "INVALID_VALUE";
    case 2:   return "OUT_OF_MEMORY";
    case 713: return "ADDR_NOT_MAPPED";
    case 719: return "LAUNCH_OUT_OF_RESOURCES";
    default:  return "ERR";
    }
}
#define CU_MEMHOSTALLOC_PORTABLE 0x01
static void *h_in, *h_out;   /* persistent PAGE-LOCKED staging for xq/y */
/* Exp48 (run #72) xq STAGE DEDUP.
   K2PROF (run #69) split the engine's per-call host cost at engine sizes:
   stage 38.8us + launch 40.0us + d2h 21.3us over ~230 calls/token = 23 ms of the
   149 ms/token. probe_copy51 isolated the fixed driver submission cost on this R440:
   a ZERO-BYTE cuMemcpyHtoDAsync + event round trip costs 47.55 us, while the same
   zero-byte call through the SYNCHRONOUS entry point costs 0.28 us - so the async
   entry point's cost is submission overhead, not bandwidth (the sync variant of the
   whole sequence was 109us/call cheaper in isolation, but benching it LOST 19% because
   synchronous copies destroy the CPU/GPU overlap: run #71).
   So the copy must stay async - but its BYTES are frequently redundant. K2ARGS capture
   of a full 216-token generation (42552 calls): 41.2% of consecutive GEMV calls pass the
   SAME xq pointer, and at the dominant shape (rows=512/2048, nb=48) 1750/2000 consecutive
   calls repeat the identical xq buffer - q/k/v (and the FFN pair) of a layer consume ONE
   quantized activation, which the caller re-quantizes to the SAME b_XQQ region and the
   engine re-stages to the SAME d_xq. Re-uploading byte-identical data to the same device
   address is a provable no-op, so: when (d_xq, length) match the previous stage AND the
   device-staged bytes are byte-identical to the new source, skip BOTH the host memcpy
   into pinned memory and the H2D submission. Bit-exact by construction (device memory
   is left exactly as the engine's own last stage left it).
   K2DEDUP=0 disables (A/B). Cost when it misses: one memcmp (~1.5us at 1.8KB). */
static uint64_t xq_last_dst;      /* d_xq of the last completed stage */
static uint64_t xq_last_src;      /* source the CPU read from, to cheaply detect a moved xq */
static size_t   xq_last_len;      /* its byte length */
static int      xq_staged;        /* a stage completed => h_in mirrors d_xq */
static long dd_calls, dd_skip; static long long dd_bytes;
static int dd_on = -1;
static void dd_dump(void) {
    printf("DEDUP calls=%ld skipped=%ld (%.1f%%) MB_saved=%lld\n",
           dd_calls, dd_skip,
           dd_calls ? 100.0 * (double)dd_skip / (double)dd_calls : 0.0,
           (long long)(dd_bytes >> 20));
    fflush(stdout);
}
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

static uint64_t blob_vaddr; static size_t blob_bytes_g;   /* fwd: defined with the CUDA state below */
static uint64_t dev_q4;          /* all Q4 code, repacked once to 16B blocks */
static uint64_t q4_off[9];       /* repacked byte offset of kind k within a layer */
static uint64_t q4_layer_dev;    /* repacked bytes per layer */
static uint64_t q4_head_off;     /* repacked offset of the lm_head tensor */
static uint32_t q4_layers;
#define Q4_STRIDE 17707008ULL
#define Q4_HID 1536u
static int q4_ready = 0;

/* Classify a host weight pointer -> (tensor rows, device base of its repacked
   code). Returns 0 (caller takes the CPU path) when w is not a Q4 tensor.
   Same arithmetic as the old rows_from_w; only the device mapping is new. */
static uint32_t q4_classify(uint64_t w, uint64_t *dev_out) {
    uint64_t lb = g_LBASE, oh = g_O_HEAD;
    if (!q4_ready) return 0;
    *dev_out = 0;
    if (w >= oh && w < oh + (uint64_t)0x20000000) {
        *dev_out = dev_q4 + q4_head_off; return g_vocab; }
    if (w < lb) return 0;
    {
        uint64_t il = (w - lb) / Q4_STRIDE;
        uint64_t base = lb + il * Q4_STRIDE;
        int k;
        for (k = 1; k <= 8; k++) {
            uint64_t ks, bend;
            if (!tbl_kcols[k]) continue;
            ks = base + tbl_koff[k];
            bend = ks + (uint64_t)tbl_krows[k] * (tbl_kcols[k] >> 5) * 18ULL;
            if (w >= ks && w < bend) {
                *dev_out = dev_q4 + il * q4_layer_dev + q4_off[k];
                return tbl_krows[k];
            }
        }
    }
    return 0;
}

/* Exp42 (run #69): pinned-ring streaming init.
   The serial path is `malloc(467MB) -> repack -> cuMemcpyHtoD`: the repack pays
   ~115k first-touch faults on the fresh buffer and the copy runs afterwards
   (981 ms measured for repack+H2D, h2dp3). The ring streams instead: the repack
   writes one 32MB PINNED slot (driver-pinned, pre-touched once), cuMemcpyHtoDAsync
   pushes it on g_stream while the CPU fills the next slot; NSLOT=2 keeps one slot
   free for writes, and the producer waits on pStreamSync when both are in flight.
   WHY PINNED: cuMemcpyHtoD from pageable memory stages through a driver bounce
   buffer (~1.2 GB/s); from pinned it runs 4.0-4.5 GB/s (h2dp3, idle board).
   WHY A RING AND NOT ONE BIG BUFFER: the run #69 probe cuMemHostAlloc'd 467MB in
   a single buffer and the OOM killer took the process down (dmesg: "Out of memory:
   Kill process (k2_gexp48) total-vm:9521780kB") - on the 4 GB Tegra a 467MB pinned
   buffer + the 467MB device alloc + the 714MB blob mmap do not fit. 2 x 32MB is
   the working set. Any allocation failure falls back to the serial path.
   NOTE: free() on a cuMemHostAlloc pointer is a double free (SIGSEGV seen at
   cuda_init+3396), so slots go back ONLY through cuMemFreeHost. */
#define Q4_SLOT ((size_t)32 << 20)
#define Q4_NSLOT 3   /* 3 slots: repack one while TWO copies are in flight */

/* Shared repack body: 18B-stride rows -> 16B blocks into DST, in the exact device
   order (per-layer k matrices, then the head). The SAME code runs in the serial
   path, so ring and serial produce byte-identical device images by construction. */
#define Q4_REPACK_INTO(DST) \
    do { \
        int i_, k_; \
        for (i_ = 0; i_ < (int)q4_layers; i_) \
            for (k_ = 1; k_ <= 8; k_) { \
                uint32_t nb = tbl_kcols[k_] >> 5, r, b; \
                const uint8_t *src; \
                uint8_t *d = (DST) + (size_t)i_ * q4_layer_dev + q4_off[k_]; \
                if (!tbl_kcols[k_]) continue; \
                src = (const uint8_t *)(uintptr_t)(lb + (uint64_t)i_ * Q4_STRIDE + tbl_koff[k_]); \
                for (r = 0; r < tbl_krows[k_]; r++) \
                    for (b = 0; b < nb; b++) \
                        *(q4blk *)(d + ((size_t)r * nb + b) * 16) = \
                            *(const q4blk *)(src + (size_t)r * (size_t)nb * 18 + 2 + (size_t)b * 18); \
            } \
        { const uint8_t *src = (const uint8_t *)(uintptr_t)oh; \
          uint32_t nb = Q4_HID >> 5, r, b; \
          uint8_t *d = (DST) + (size_t)q4_head_off; \
          for (r = 0; r < g_vocab; r++) \
              for (b = 0; b < nb; b++) \
                  *(q4blk *)(d + ((size_t)r * nb + b) * 16) = \
                      *(const q4blk *)(src + (size_t)r * (size_t)nb * 18 + 2 + (size_t)b * 18); } \
    } while (0)

static void q4_ring_fini(uint8_t **slot) {
    int i;
    for (i = 0; i < Q4_NSLOT; i++)
        if (slot[i]) {
            int rc = pMemFreeHost ? pMemFreeHost(slot[i]) : -1;
            if (rc) printf("INIT unpin rc=%d(%s) - slot leaked\n", rc, k2err(rc));
            slot[i] = 0;
        }
}

/* K2VERIFY=1: build a serial reference repack on the host, read the DEVICE image
   back, compare byte-for-byte. The 216-step gate only samples model output, so a
   streaming-layout bug could hide inside it; this proves the device bytes are the
   proven serial layout. Diagnostic path only (needs 2 x 467MB host allocations). */
static void q4_verify(uint64_t lb, uint64_t oh, size_t total, uint64_t drp) {
    uint8_t *ref = (uint8_t *)malloc(total);
    uint8_t *back = (uint8_t *)malloc(total);
    size_t i, bad = 0;
    int rc;
    if (!ref || !back || !pMemcpyDtoH) {
        printf("VERIFY skipped (alloc/dtoH)\n"); free(ref); free(back); return;
    }
    Q4_REPACK_INTO(ref);
    rc = pMemcpyDtoH(back, drp, total);
    if (rc) { printf("VERIFY dtoH rc=%d(%s)\n", rc, k2err(rc)); free(ref); free(back); return; }
    for (i = 0; i < total; i++)
        if (ref[i] != back[i]) {
            if (bad < 5) printf("VERIFY diff at %llu: ref %02x dev %02x\n",
                                (unsigned long long)i, ref[i], back[i]);
            bad++;
        }
    printf("VERIFY %s (%llu bytes, %llu diffs)\n", bad ? "FAIL" : "OK",
           (unsigned long long)total, (unsigned long long)bad);
    free(ref); free(back);
}

static int q4_ring_flush(uint64_t drp, uint64_t *out, uint8_t **slot, int *si, int *npend) {
    int rc = pMemcpyHtoDAsync(drp + *out, slot[*si], Q4_SLOT, g_stream);
    if (rc) { printf("INIT ring h2da rc=%d(%s)\n", rc, k2err(rc)); return 0; }
    *out += Q4_SLOT;
    *si = (*si + 1) % Q4_NSLOT;
    /* Queue DEPTH 2 (two copies in flight, third slot being written). h2dp6 on the
       board: depth 1 = 1.05 GB/s, depth 2 = 2.86 GB/s for the 467MB stream. h2dp5
       showed depth 2 is also what triggers cuMemcpyHtoDAsync rc=1 when a SECOND
       batch of pinned slots is allocated in the same process, so the engine pins its
       ring exactly once (q4_ring_fini frees via cuMemFreeHost before any other pin). */
    if (++(*npend) >= 2) {
        if (pStreamSync(g_stream)) return 0;
        *npend = 0;
    }
    return 1;
}

static int q4_repack_ring(uint64_t lb, uint64_t oh, size_t total, uint64_t drp) {
    uint8_t *slot[Q4_NSLOT];
    int si = 0, npend = 0, rc, i2, k2;
    size_t fill = 0, out = 0;
    uint8_t *wp;
    int wrem;                       /* BYTES left in the current slot (16B blocks) */
    double t0, t1, t2;
    if (!pMemHostAlloc || !pMemFreeHost || !pMemcpyHtoDAsync || !pStreamSync || !g_stream)
        return 0;
    memset(slot, 0, sizeof(slot));
    t0 = now_ms();
    for (i2 = 0; i2 < Q4_NSLOT; i2++) {
        void *p = 0;
        rc = pMemHostAlloc(&p, Q4_SLOT, 0);
        if (rc) {
            printf("INIT pin slot %d rc=%d(%s) -> serial fallback\n", i2, rc, k2err(rc));
            q4_ring_fini(slot);
            return 0;
        }
        slot[i2] = (uint8_t *)p;
        memset(p, 0, Q4_SLOT);      /* pre-touch: faults cost ~7 ms/MB on pinned pages */
    }
    t1 = now_ms();
    printf("INIT ring pin+pretouch %d x %llu MB %.0f ms\n",
           Q4_NSLOT, (unsigned long long)(Q4_SLOT >> 20), t1 - t0);
    wp = slot[0];
    wrem = (int)Q4_SLOT;
    for (i2 = 0; i2 < (int)q4_layers; i2++)
        for (k2 = 1; k2 <= 8; k2++) {
            uint32_t nb = tbl_kcols[k2] >> 5, r, b;
            const uint8_t *src;
            if (!tbl_kcols[k2]) continue;
            src = (const uint8_t *)(uintptr_t)(lb + (uint64_t)i2 * Q4_STRIDE + tbl_koff[k2]);
            for (r = 0; r < tbl_krows[k2]; r++)
                for (b = 0; b < nb; b++) {
                    *(q4blk *)wp = *(const q4blk *)(src + (size_t)r * (size_t)nb * 18 + 2 + (size_t)b * 18);
                    wp += 16; fill += 16;
                    if ((wrem -= 16) == 0) {
                        if (!q4_ring_flush(drp, &out, slot, &si, &npend)) { q4_ring_fini(slot); return 0; }
                        wp = slot[si]; fill = 0; wrem = (int)Q4_SLOT;
                    }
                }
        }
    { const uint8_t *src = (const uint8_t *)(uintptr_t)oh;
      uint32_t nb = Q4_HID >> 5, r, b;
      for (r = 0; r < g_vocab; r++)
          for (b = 0; b < nb; b++) {
              *(q4blk *)wp = *(const q4blk *)(src + (size_t)r * (size_t)nb * 18 + 2 + (size_t)b * 18);
              wp += 16; fill += 16;
              if ((wrem -= 16) == 0) {
                  if (!q4_ring_flush(drp, &out, slot, &si, &npend)) { q4_ring_fini(slot); return 0; }
                  wp = slot[si]; fill = 0; wrem = (int)Q4_SLOT;
              }
          } }
    t2 = now_ms();
    printf("INIT ring repack %.0f ms (%llu MB streamed)\n", t2 - t1, (unsigned long long)(out >> 20));
    if (fill) {                       /* 467MB % 32MB tail */
        if (pMemcpyHtoDAsync(drp + out, slot[si], fill, g_stream)) { q4_ring_fini(slot); return 0; }
        out += fill;
    }
    if (out != total) {
        printf("INIT ring streamed %llu != total %llu -> serial fallback\n",
               (unsigned long long)out, (unsigned long long)total);
        q4_ring_fini(slot); return 0;
    }
    if (pStreamSync(g_stream)) { q4_ring_fini(slot); return 0; }
    printf("INIT ring h2d+sync %.0f ms (ring total %.0f ms)\n", now_ms() - t2, now_ms() - t0);
    q4_ring_fini(slot);
    return 1;
}

/* One-time host-side repack of every Q4 tensor: 18B-stride rows -> 16B blocks,
   into ONE device buffer (no raw-blob mirror on device). Bit-exact by
   construction: identical bytes, identical order, kernel math untouched. */
static int q4_build(void) {
    uint64_t lb = g_LBASE, oh = g_O_HEAD;
    uint64_t tot = 0, head_bytes;
    size_t total;
    uint8_t *tmp;
    uint64_t drp = 0;
    int k; uint32_t i;
    if (!tbl_kcols[1]) return 0;
    for (k = 1; k <= 8; k++) {
        if (!tbl_kcols[k]) continue;
        q4_off[k] = tot;
        tot += (uint64_t)tbl_krows[k] * (tbl_kcols[k] >> 5) * 16;
    }
    q4_layer_dev = tot;
    if (oh <= lb) return 0;
    /* Exp5b fix (run #31): (oh-lb)/stride counts the fp16 embed region as Q4
       layer-slots (gave 39). The engine packs exactly 28 Q4 layers; the embed
       gap between the layer array and oh is NOT Q4. */
    q4_layers = 28;
    if (oh < lb + (uint64_t)q4_layers * Q4_STRIDE) { printf("q4_build: head overlaps layer array\n"); return 0; }
    q4_head_off = (size_t)tot * q4_layers;
    head_bytes = (size_t)g_vocab * (Q4_HID >> 5) * 16;
    total = (size_t)q4_head_off + head_bytes;
    if (total > (size_t)blob_bytes_g) { printf("q4_build: total %llu > blob %llu\n",
        (unsigned long long)total, (unsigned long long)blob_bytes_g); return 0; }
    { /* Device buffer FIRST: the ring streams into it, and the serial path
         reuses it. (Staged v49 initially called the ring with drp==0 - every
         cuMemcpyHtoDAsync then returned INVALID_VALUE and the run silently fell
         back to serial. Caught by reading the code before the bench.) */
      double ta = now_ms();
      int rc = pMemAlloc(&drp, total);
      printf("INIT dev-alloc %llu MB rc=%d(%s) %.0f ms\n",
             (unsigned long long)(total >> 20), rc, k2err(rc), now_ms() - ta);
      if (rc) { printf("q4_build: memalloc failed\n"); return 0; } }
    if (q4_repack_ring(lb, oh, total, drp)) {
        dev_q4 = drp;
        printf("CUDA q4 repacked single-copy RING %llu MB, %u layers\n",
               (unsigned long long)(total >> 20), q4_layers);
        if (getenv("K2VERIFY")) q4_verify(lb, oh, total, drp);
        q4_ready = 1;
        return 1;
    }
    printf("q4_build: ring unavailable -> serial path\n");
    tmp = (uint8_t *)malloc(total);
    if (!tmp) { printf("q4_build: malloc %llu MB failed\n", (unsigned long long)(total>>20)); return 0; }
    for (i = 0; i < q4_layers; i++)
        for (k = 1; k <= 8; k++) {
            uint32_t nb = tbl_kcols[k] >> 5, r;
            const uint8_t *src;
            if (!tbl_kcols[k]) continue;
            src = (const uint8_t *)(uintptr_t)(lb + (uint64_t)i * Q4_STRIDE + tbl_koff[k]);
            for (r = 0; r < tbl_krows[k]; r++) { uint32_t b;
                for (b = 0; b < nb; b++)
                    memcpy(tmp + (size_t)i * q4_layer_dev + q4_off[k] + ((size_t)r * nb + b) * 16,
                           src + (size_t)r * (size_t)nb * 18 + 2 + (size_t)b * 18, 16); }
        }
    { const uint8_t *src = (const uint8_t *)(uintptr_t)oh;
      uint32_t nb = Q4_HID >> 5, r;
      for (r = 0; r < g_vocab; r++) { uint32_t b;
          for (b = 0; b < nb; b++)
              memcpy(tmp + (size_t)q4_head_off + ((size_t)r * nb + b) * 16,
                     src + (size_t)r * (size_t)nb * 18 + 2 + (size_t)b * 18, 16); } }
    { double tb = now_ms();
      int rc = pMemcpyHtoD(drp, tmp, total);
      printf("INIT serial h2d %.0f ms rc=%d(%s)\n", now_ms() - tb, rc, k2err(rc));
      if (rc) { free(tmp); printf("q4_build: h2d failed\n"); return 0; } }
    free(tmp);
    dev_q4 = drp;
    printf("CUDA q4 repacked single-copy %llu MB, %u layers\n",
           (unsigned long long)(total >> 20), q4_layers);
    q4_ready = 1;
    return 1;
}

static const char *ptx =
	"// K2 Q4 GEMV v29 (Exp23: 4-lane replay — row q chain owned by lane q; 96->16 replay instrs/iter)\n"
	".version 4.0\n"
	".target sm_50\n"
	".address_size 64\n"
	".visible .entry q4_gemv(.param .u64 pw, .param .u64 psc, .param .u64 pxq, .param .u64 py, .param .u32 pr0, .param .u32 pr1, .param .u32 pnb, .param .u32 prf) {\n"
	".reg .u64 %rd<24>; .reg .u32 %r<128>; .reg .f32 %f<32>; .reg .pred %p<8>;\n"
	".shared .align 4 .b32 shbuf[4096];\n"
	"ld.param.u64 %rd1,[pw]; ld.param.u64 %rd2,[psc]; ld.param.u64 %rd3,[pxq]; ld.param.u64 %rd4,[py];\n"
	"ld.param.u32 %r10,[pr0]; ld.param.u32 %r11,[pr1]; ld.param.u32 %r12,[pnb];\n"
	"mov.u32 %r1,%tid.x; mov.u32 %r2,%tid.y; mov.u32 %r3,%ctaid.x; add.u32 %r104,%r2,1;\n"
	"shl.b32 %r3,%r3,3; add.u32 %r3,%r3,%r2; shl.b32 %r2,%r3,2; add.u32 %r2,%r2,%r10;\n"
	"add.u32 %r7,%r2,1; add.u32 %r70,%r2,2; add.u32 %r71,%r2,3;\n"
	"setp.ge.u32 %p1,%r2,%r11; @%p1 bra $done;\n"
	"setp.ge.u32 %p4,%r7,%r11;\n"
	"selp.b32 %r7,%r2,%r7,%p4;\n"
	"setp.ge.u32 %p5,%r70,%r11;\n"
	"selp.b32 %r70,%r2,%r70,%p5;\n"
	"setp.ge.u32 %p6,%r71,%r11;\n"
	"selp.b32 %r71,%r2,%r71,%p6;\n"
	"shr.u32 %r33,%r1,2;\n"
	"and.b32 %r34,%r1,3;\n"
	"mul.lo.u32 %r4,%r2,%r12; shl.b32 %r4,%r4,4;\n"
	"shl.b32 %r5,%r33,4; shl.b32 %r6,%r34,2; add.u32 %r4,%r4,%r5; add.u32 %r4,%r4,%r6;\n"
	"cvt.u64.u32 %rd5,%r4; add.u64 %rd5,%rd5,%rd1;\n"
	"mul.lo.u32 %r4,%r7,%r12; shl.b32 %r4,%r4,4;\n"
	"shl.b32 %r5,%r33,4; shl.b32 %r6,%r34,2; add.u32 %r4,%r4,%r5; add.u32 %r4,%r4,%r6;\n"
	"cvt.u64.u32 %rd6,%r4; add.u64 %rd6,%rd6,%rd1;\n"
	"mul.lo.u32 %r4,%r70,%r12; shl.b32 %r4,%r4,4;\n"
	"shl.b32 %r5,%r33,4; shl.b32 %r6,%r34,2; add.u32 %r4,%r4,%r5; add.u32 %r4,%r4,%r6;\n"
	"cvt.u64.u32 %rd22,%r4; add.u64 %rd22,%rd22,%rd1;\n"
	"mul.lo.u32 %r4,%r71,%r12; shl.b32 %r4,%r4,4;\n"
	"shl.b32 %r5,%r33,4; shl.b32 %r6,%r34,2; add.u32 %r4,%r4,%r5; add.u32 %r4,%r4,%r6;\n"
	"cvt.u64.u32 %rd23,%r4; add.u64 %rd23,%rd23,%rd1;\n"
	"shl.b32 %r5,%r33,5; shl.b32 %r6,%r34,2; add.u32 %r5,%r5,%r6; add.u32 %r5,%r5,4;\n"
	"cvt.u64.u32 %rd17,%r5; add.u64 %rd17,%rd17,%rd3;\n"
	"mul.lo.u32 %r4,%r2,%r12; shl.b32 %r4,%r4,2; cvt.u64.u32 %rd18,%r4; add.u64 %rd18,%rd18,%rd2;\n"
	"mul.lo.u32 %r4,%r7,%r12; shl.b32 %r4,%r4,2; cvt.u64.u32 %rd21,%r4; add.u64 %rd21,%rd21,%rd2;\n"
	"mul.lo.u32 %r4,%r70,%r12; shl.b32 %r4,%r4,2; cvt.u64.u32 %rd7,%r4; add.u64 %rd7,%rd7,%rd2;\n"
	"mul.lo.u32 %r4,%r71,%r12; shl.b32 %r4,%r4,2; cvt.u64.u32 %rd8,%r4; add.u64 %rd8,%rd8,%rd2;\n"
	"shl.b32 %r5,%r33,2; cvt.u64.u32 %rd19,%r5;\n"
	"add.u64 %rd18,%rd18,%rd19; add.u64 %rd21,%rd21,%rd19; add.u64 %rd7,%rd7,%rd19; add.u64 %rd8,%rd8,%rd19;\n"
	"mov.u64 %rd20,%rd3;\n"
	"add.u32 %r39,%r12,7; shr.u32 %r39,%r39,3;\n"
	"setp.eq.u32 %p0,%r1,0;\n"
	"mov.b32 %r61,shbuf; mov.u32 %r63,%tid.y; shl.b32 %r63,%r63,10; add.u32 %r65,%r61,%r63;\n"
	"mul.lo.u32 %r60,%r1,4; add.u32 %r62,%r65,%r60;\n"
	"mul.lo.u32 %r66,%r33,16;\n"
	"and.b32 %r90,%r39,0xFFFFFFFE;\n"
	"and.b32 %r91,%r39,1;\n"
	"add.u32 %r79,%r65,8192; add.u32 %r92,%r65,8448; add.u32 %r93,%r65,8704;\n"
	"mov.f32 %f1,0F00000000; mov.u32 %r13,0;\n"
	"shr.u32 %r76,%r1,3; and.b32 %r77,%r1,7;\n"
	"shl.b32 %r98,%r76,8; add.u32 %r98,%r98,%r65;\n"
	"shl.b32 %r77,%r77,4; add.u32 %r98,%r98,%r77;\n"
	"shl.b32 %r76,%r76,2; add.u32 %r77,%r77,%r76; shl.b32 %r77,%r77,1;\n"
	"add.u32 %r99,%r77,%r79;\n"
	"shl.b32 %r103,%r1,3; add.u32 %r103,%r103,%r79;\n"
	"setp.lt.u32 %p0,%r1,4;\n"
	"ld.global.u32 %r20,[%rd5+0];\n"
	"ld.global.u32 %r22,[%rd6+0];\n"
	"ld.global.u32 %r26,[%rd22+0];\n"
	"ld.global.u32 %r27,[%rd23+0];\n"
	"ld.global.u32 %r24,[%rd17+0];\n"
	"ld.global.u32 %r28,[%rd17+16];\n"
	"ld.global.f32 %f2,[%rd18+0]; ld.global.f32 %f4,[%rd21+0]; ld.global.f32 %f5,[%rd7+0]; ld.global.f32 %f9,[%rd8+0]; ld.global.f32 %f3,[%rd20+0];\n"
	"$LIT:\n"
	"ld.global.u32 %r80,[%rd5+128];\n"
	"ld.global.u32 %r81,[%rd6+128];\n"
	"ld.global.u32 %r82,[%rd22+128];\n"
	"ld.global.u32 %r83,[%rd23+128];\n"
	"ld.global.u32 %r84,[%rd17+292];\n"
	"ld.global.u32 %r85,[%rd17+308];\n"
	"ld.global.f32 %f26,[%rd18+32]; ld.global.f32 %f27,[%rd21+32]; ld.global.f32 %f28,[%rd7+32]; ld.global.f32 %f29,[%rd8+32]; ld.global.f32 %f30,[%rd20+292];\n"
	"shl.b32 %r14,%r13,3; add.u32 %r14,%r14,%r33; setp.ge.u32 %p2,%r14,%r12;\n"
	"xor.b32 %r97,%r20,0x88888888; xor.b32 %r100,%r22,0x88888888;\n"
	"xor.b32 %r101,%r26,0x88888888; xor.b32 %r102,%r27,0x88888888;\n"
	"bfe.s32 %r43,%r24,0,8;\n"
	"bfe.s32 %r44,%r28,0,8;\n"
	"bfe.s32 %r41,%r97,0,4;\n"
	"bfe.s32 %r42,%r97,4,4;\n"
	"mad.lo.s32 %r46,%r41,%r43,0;\n"
	"mad.lo.s32 %r46,%r42,%r44,%r46;\n"
	"bfe.s32 %r41,%r100,0,4;\n"
	"bfe.s32 %r42,%r100,4,4;\n"
	"mad.lo.s32 %r49,%r41,%r43,0;\n"
	"mad.lo.s32 %r49,%r42,%r44,%r49;\n"
	"bfe.s32 %r41,%r101,0,4;\n"
	"bfe.s32 %r42,%r101,4,4;\n"
	"mad.lo.s32 %r35,%r41,%r43,0;\n"
	"mad.lo.s32 %r35,%r42,%r44,%r35;\n"
	"bfe.s32 %r41,%r102,0,4;\n"
	"bfe.s32 %r42,%r102,4,4;\n"
	"mad.lo.s32 %r36,%r41,%r43,0;\n"
	"mad.lo.s32 %r36,%r42,%r44,%r36;\n"
	"bfe.s32 %r43,%r24,8,8;\n"
	"bfe.s32 %r44,%r28,8,8;\n"
	"bfe.s32 %r41,%r97,8,4;\n"
	"bfe.s32 %r42,%r97,12,4;\n"
	"mad.lo.s32 %r46,%r41,%r43,%r46;\n"
	"mad.lo.s32 %r46,%r42,%r44,%r46;\n"
	"bfe.s32 %r41,%r100,8,4;\n"
	"bfe.s32 %r42,%r100,12,4;\n"
	"mad.lo.s32 %r49,%r41,%r43,%r49;\n"
	"mad.lo.s32 %r49,%r42,%r44,%r49;\n"
	"bfe.s32 %r41,%r101,8,4;\n"
	"bfe.s32 %r42,%r101,12,4;\n"
	"mad.lo.s32 %r35,%r41,%r43,%r35;\n"
	"mad.lo.s32 %r35,%r42,%r44,%r35;\n"
	"bfe.s32 %r41,%r102,8,4;\n"
	"bfe.s32 %r42,%r102,12,4;\n"
	"mad.lo.s32 %r36,%r41,%r43,%r36;\n"
	"mad.lo.s32 %r36,%r42,%r44,%r36;\n"
	"bfe.s32 %r43,%r24,16,8;\n"
	"bfe.s32 %r44,%r28,16,8;\n"
	"bfe.s32 %r41,%r97,16,4;\n"
	"bfe.s32 %r42,%r97,20,4;\n"
	"mad.lo.s32 %r46,%r41,%r43,%r46;\n"
	"mad.lo.s32 %r46,%r42,%r44,%r46;\n"
	"bfe.s32 %r41,%r100,16,4;\n"
	"bfe.s32 %r42,%r100,20,4;\n"
	"mad.lo.s32 %r49,%r41,%r43,%r49;\n"
	"mad.lo.s32 %r49,%r42,%r44,%r49;\n"
	"bfe.s32 %r41,%r101,16,4;\n"
	"bfe.s32 %r42,%r101,20,4;\n"
	"mad.lo.s32 %r35,%r41,%r43,%r35;\n"
	"mad.lo.s32 %r35,%r42,%r44,%r35;\n"
	"bfe.s32 %r41,%r102,16,4;\n"
	"bfe.s32 %r42,%r102,20,4;\n"
	"mad.lo.s32 %r36,%r41,%r43,%r36;\n"
	"mad.lo.s32 %r36,%r42,%r44,%r36;\n"
	"bfe.s32 %r43,%r24,24,8;\n"
	"bfe.s32 %r44,%r28,24,8;\n"
	"bfe.s32 %r41,%r97,24,4;\n"
	"bfe.s32 %r42,%r97,28,4;\n"
	"mad.lo.s32 %r46,%r41,%r43,%r46;\n"
	"mad.lo.s32 %r46,%r42,%r44,%r46;\n"
	"bfe.s32 %r41,%r100,24,4;\n"
	"bfe.s32 %r42,%r100,28,4;\n"
	"mad.lo.s32 %r49,%r41,%r43,%r49;\n"
	"mad.lo.s32 %r49,%r42,%r44,%r49;\n"
	"bfe.s32 %r41,%r101,24,4;\n"
	"bfe.s32 %r42,%r101,28,4;\n"
	"mad.lo.s32 %r35,%r41,%r43,%r35;\n"
	"mad.lo.s32 %r35,%r42,%r44,%r35;\n"
	"bfe.s32 %r41,%r102,24,4;\n"
	"bfe.s32 %r42,%r102,28,4;\n"
	"mad.lo.s32 %r36,%r41,%r43,%r36;\n"
	"mad.lo.s32 %r36,%r42,%r44,%r36;\n"
	"cvt.rn.f32.s32 %f6,%r46; mul.f32 %f7,%f2,%f3; selp.f32 %f7,0F00000000,%f7,%p2;\n"
	"cvt.rn.f32.s32 %f17,%r49; mul.f32 %f18,%f4,%f3;\n"
	"or.pred %p7,%p2,%p4; selp.f32 %f18,0F00000000,%f18,%p7;\n"
	"cvt.rn.f32.s32 %f20,%r35; mul.f32 %f21,%f5,%f3;\n"
	"or.pred %p7,%p2,%p5; selp.f32 %f21,0F00000000,%f21,%p7;\n"
	"cvt.rn.f32.s32 %f23,%r36; mul.f32 %f24,%f9,%f3;\n"
	"or.pred %p7,%p2,%p6; selp.f32 %f24,0F00000000,%f24,%p7;\n"
	"st.shared.f32 [%r62],%f7; st.shared.f32 [%r62+128],%f6;\n"
	"st.shared.f32 [%r62+256],%f18; st.shared.f32 [%r62+384],%f17;\n"
	"st.shared.f32 [%r62+512],%f21; st.shared.f32 [%r62+640],%f20;\n"
	"st.shared.f32 [%r62+768],%f24; st.shared.f32 [%r62+896],%f23;\n"
	"// v43: bar.sync dropped (warp-synchronous STS->LDS; all 32 lanes reach every point)\n"
	"ld.shared.v4.f32 {%f10,%f11,%f12,%f13},[%r98+128];\n"
	"add.f32 %f8,%f10,%f11;\n"
	"add.f32 %f8,%f8,%f12;\n"
	"add.f32 %f8,%f8,%f13;\n"
	"ld.shared.f32 %f14,[%r98+0];\n"
	"add.u32 %r77,%r99,0;\n"
	"st.shared.v2.f32 [%r77],{%f14,%f8};\n"
	"// v43: bar.sync dropped (warp-synchronous STS->LDS; all 32 lanes reach every point)\n"
	"add.u32 %r96,%r103,0;\n"
	"@%p0 ld.shared.v2.f32 {%f14,%f8},[%r96+0];\n"
	"@%p0 fma.rn.f32 %f1,%f14,%f8,%f1;\n"
	"@%p0 ld.shared.v2.f32 {%f14,%f8},[%r96+32];\n"
	"@%p0 fma.rn.f32 %f1,%f14,%f8,%f1;\n"
	"@%p0 ld.shared.v2.f32 {%f14,%f8},[%r96+64];\n"
	"@%p0 fma.rn.f32 %f1,%f14,%f8,%f1;\n"
	"@%p0 ld.shared.v2.f32 {%f14,%f8},[%r96+96];\n"
	"@%p0 fma.rn.f32 %f1,%f14,%f8,%f1;\n"
	"@%p0 ld.shared.v2.f32 {%f14,%f8},[%r96+128];\n"
	"@%p0 fma.rn.f32 %f1,%f14,%f8,%f1;\n"
	"@%p0 ld.shared.v2.f32 {%f14,%f8},[%r96+160];\n"
	"@%p0 fma.rn.f32 %f1,%f14,%f8,%f1;\n"
	"@%p0 ld.shared.v2.f32 {%f14,%f8},[%r96+192];\n"
	"@%p0 fma.rn.f32 %f1,%f14,%f8,%f1;\n"
	"@%p0 ld.shared.v2.f32 {%f14,%f8},[%r96+224];\n"
	"@%p0 fma.rn.f32 %f1,%f14,%f8,%f1;\n"
	"ld.global.u32 %r20,[%rd5+256];\n"
	"ld.global.u32 %r22,[%rd6+256];\n"
	"ld.global.u32 %r26,[%rd22+256];\n"
	"ld.global.u32 %r27,[%rd23+256];\n"
	"ld.global.u32 %r24,[%rd17+584];\n"
	"ld.global.u32 %r28,[%rd17+600];\n"
	"ld.global.f32 %f2,[%rd18+64]; ld.global.f32 %f4,[%rd21+64]; ld.global.f32 %f5,[%rd7+64]; ld.global.f32 %f9,[%rd8+64]; ld.global.f32 %f3,[%rd20+584];\n"
	"add.u32 %r14,%r13,1; shl.b32 %r14,%r14,3; add.u32 %r14,%r14,%r33; setp.ge.u32 %p2,%r14,%r12;\n"
	"xor.b32 %r97,%r80,0x88888888; xor.b32 %r100,%r81,0x88888888;\n"
	"xor.b32 %r101,%r82,0x88888888; xor.b32 %r102,%r83,0x88888888;\n"
	"bfe.s32 %r43,%r84,0,8;\n"
	"bfe.s32 %r44,%r85,0,8;\n"
	"bfe.s32 %r41,%r97,0,4;\n"
	"bfe.s32 %r42,%r97,4,4;\n"
	"mad.lo.s32 %r86,%r41,%r43,0;\n"
	"mad.lo.s32 %r86,%r42,%r44,%r86;\n"
	"bfe.s32 %r41,%r100,0,4;\n"
	"bfe.s32 %r42,%r100,4,4;\n"
	"mad.lo.s32 %r87,%r41,%r43,0;\n"
	"mad.lo.s32 %r87,%r42,%r44,%r87;\n"
	"bfe.s32 %r41,%r101,0,4;\n"
	"bfe.s32 %r42,%r101,4,4;\n"
	"mad.lo.s32 %r88,%r41,%r43,0;\n"
	"mad.lo.s32 %r88,%r42,%r44,%r88;\n"
	"bfe.s32 %r41,%r102,0,4;\n"
	"bfe.s32 %r42,%r102,4,4;\n"
	"mad.lo.s32 %r89,%r41,%r43,0;\n"
	"mad.lo.s32 %r89,%r42,%r44,%r89;\n"
	"bfe.s32 %r43,%r84,8,8;\n"
	"bfe.s32 %r44,%r85,8,8;\n"
	"bfe.s32 %r41,%r97,8,4;\n"
	"bfe.s32 %r42,%r97,12,4;\n"
	"mad.lo.s32 %r86,%r41,%r43,%r86;\n"
	"mad.lo.s32 %r86,%r42,%r44,%r86;\n"
	"bfe.s32 %r41,%r100,8,4;\n"
	"bfe.s32 %r42,%r100,12,4;\n"
	"mad.lo.s32 %r87,%r41,%r43,%r87;\n"
	"mad.lo.s32 %r87,%r42,%r44,%r87;\n"
	"bfe.s32 %r41,%r101,8,4;\n"
	"bfe.s32 %r42,%r101,12,4;\n"
	"mad.lo.s32 %r88,%r41,%r43,%r88;\n"
	"mad.lo.s32 %r88,%r42,%r44,%r88;\n"
	"bfe.s32 %r41,%r102,8,4;\n"
	"bfe.s32 %r42,%r102,12,4;\n"
	"mad.lo.s32 %r89,%r41,%r43,%r89;\n"
	"mad.lo.s32 %r89,%r42,%r44,%r89;\n"
	"bfe.s32 %r43,%r84,16,8;\n"
	"bfe.s32 %r44,%r85,16,8;\n"
	"bfe.s32 %r41,%r97,16,4;\n"
	"bfe.s32 %r42,%r97,20,4;\n"
	"mad.lo.s32 %r86,%r41,%r43,%r86;\n"
	"mad.lo.s32 %r86,%r42,%r44,%r86;\n"
	"bfe.s32 %r41,%r100,16,4;\n"
	"bfe.s32 %r42,%r100,20,4;\n"
	"mad.lo.s32 %r87,%r41,%r43,%r87;\n"
	"mad.lo.s32 %r87,%r42,%r44,%r87;\n"
	"bfe.s32 %r41,%r101,16,4;\n"
	"bfe.s32 %r42,%r101,20,4;\n"
	"mad.lo.s32 %r88,%r41,%r43,%r88;\n"
	"mad.lo.s32 %r88,%r42,%r44,%r88;\n"
	"bfe.s32 %r41,%r102,16,4;\n"
	"bfe.s32 %r42,%r102,20,4;\n"
	"mad.lo.s32 %r89,%r41,%r43,%r89;\n"
	"mad.lo.s32 %r89,%r42,%r44,%r89;\n"
	"bfe.s32 %r43,%r84,24,8;\n"
	"bfe.s32 %r44,%r85,24,8;\n"
	"bfe.s32 %r41,%r97,24,4;\n"
	"bfe.s32 %r42,%r97,28,4;\n"
	"mad.lo.s32 %r86,%r41,%r43,%r86;\n"
	"mad.lo.s32 %r86,%r42,%r44,%r86;\n"
	"bfe.s32 %r41,%r100,24,4;\n"
	"bfe.s32 %r42,%r100,28,4;\n"
	"mad.lo.s32 %r87,%r41,%r43,%r87;\n"
	"mad.lo.s32 %r87,%r42,%r44,%r87;\n"
	"bfe.s32 %r41,%r101,24,4;\n"
	"bfe.s32 %r42,%r101,28,4;\n"
	"mad.lo.s32 %r88,%r41,%r43,%r88;\n"
	"mad.lo.s32 %r88,%r42,%r44,%r88;\n"
	"bfe.s32 %r41,%r102,24,4;\n"
	"bfe.s32 %r42,%r102,28,4;\n"
	"mad.lo.s32 %r89,%r41,%r43,%r89;\n"
	"mad.lo.s32 %r89,%r42,%r44,%r89;\n"
	"cvt.rn.f32.s32 %f6,%r86; mul.f32 %f7,%f26,%f30; selp.f32 %f7,0F00000000,%f7,%p2;\n"
	"cvt.rn.f32.s32 %f17,%r87; mul.f32 %f18,%f27,%f30;\n"
	"or.pred %p7,%p2,%p4; selp.f32 %f18,0F00000000,%f18,%p7;\n"
	"cvt.rn.f32.s32 %f20,%r88; mul.f32 %f21,%f28,%f30;\n"
	"or.pred %p7,%p2,%p5; selp.f32 %f21,0F00000000,%f21,%p7;\n"
	"cvt.rn.f32.s32 %f23,%r89; mul.f32 %f24,%f29,%f30;\n"
	"or.pred %p7,%p2,%p6; selp.f32 %f24,0F00000000,%f24,%p7;\n"
	"st.shared.f32 [%r62],%f7; st.shared.f32 [%r62+128],%f6;\n"
	"st.shared.f32 [%r62+256],%f18; st.shared.f32 [%r62+384],%f17;\n"
	"st.shared.f32 [%r62+512],%f21; st.shared.f32 [%r62+640],%f20;\n"
	"st.shared.f32 [%r62+768],%f24; st.shared.f32 [%r62+896],%f23;\n"
	"// v43: bar.sync dropped (warp-synchronous STS->LDS; all 32 lanes reach every point)\n"
	"ld.shared.v4.f32 {%f10,%f11,%f12,%f13},[%r98+128];\n"
	"add.f32 %f8,%f10,%f11;\n"
	"add.f32 %f8,%f8,%f12;\n"
	"add.f32 %f8,%f8,%f13;\n"
	"ld.shared.f32 %f14,[%r98+0];\n"
	"add.u32 %r77,%r99,256;\n"
	"st.shared.v2.f32 [%r77],{%f14,%f8};\n"
	"// v43: bar.sync dropped (warp-synchronous STS->LDS; all 32 lanes reach every point)\n"
	"add.u32 %r96,%r103,256;\n"
	"@%p0 ld.shared.v2.f32 {%f14,%f8},[%r96+0];\n"
	"@%p0 fma.rn.f32 %f1,%f14,%f8,%f1;\n"
	"@%p0 ld.shared.v2.f32 {%f14,%f8},[%r96+32];\n"
	"@%p0 fma.rn.f32 %f1,%f14,%f8,%f1;\n"
	"@%p0 ld.shared.v2.f32 {%f14,%f8},[%r96+64];\n"
	"@%p0 fma.rn.f32 %f1,%f14,%f8,%f1;\n"
	"@%p0 ld.shared.v2.f32 {%f14,%f8},[%r96+96];\n"
	"@%p0 fma.rn.f32 %f1,%f14,%f8,%f1;\n"
	"@%p0 ld.shared.v2.f32 {%f14,%f8},[%r96+128];\n"
	"@%p0 fma.rn.f32 %f1,%f14,%f8,%f1;\n"
	"@%p0 ld.shared.v2.f32 {%f14,%f8},[%r96+160];\n"
	"@%p0 fma.rn.f32 %f1,%f14,%f8,%f1;\n"
	"@%p0 ld.shared.v2.f32 {%f14,%f8},[%r96+192];\n"
	"@%p0 fma.rn.f32 %f1,%f14,%f8,%f1;\n"
	"@%p0 ld.shared.v2.f32 {%f14,%f8},[%r96+224];\n"
	"@%p0 fma.rn.f32 %f1,%f14,%f8,%f1;\n"
	"add.u32 %r13,%r13,2;\n"
	"add.u64 %rd5,%rd5,256; add.u64 %rd6,%rd6,256; add.u64 %rd22,%rd22,256; add.u64 %rd23,%rd23,256;\n"
	"add.u64 %rd17,%rd17,584; add.u64 %rd18,%rd18,64; add.u64 %rd21,%rd21,64; add.u64 %rd7,%rd7,64; add.u64 %rd8,%rd8,64; add.u64 %rd20,%rd20,584;\n"
	"setp.lt.u32 %p3,%r13,%r90; @%p3 bra $LIT;\n"
	"setp.ne.u32 %p7,%r91,0; @%p7 bra $epi; bra $stor;\n"
	"$epi:\n"
	"shl.b32 %r14,%r13,3; add.u32 %r14,%r14,%r33; setp.ge.u32 %p2,%r14,%r12;\n"
	"xor.b32 %r97,%r20,0x88888888; xor.b32 %r100,%r22,0x88888888;\n"
	"xor.b32 %r101,%r26,0x88888888; xor.b32 %r102,%r27,0x88888888;\n"
	"bfe.s32 %r43,%r24,0,8;\n"
	"bfe.s32 %r44,%r28,0,8;\n"
	"bfe.s32 %r41,%r97,0,4;\n"
	"bfe.s32 %r42,%r97,4,4;\n"
	"mad.lo.s32 %r46,%r41,%r43,0;\n"
	"mad.lo.s32 %r46,%r42,%r44,%r46;\n"
	"bfe.s32 %r41,%r100,0,4;\n"
	"bfe.s32 %r42,%r100,4,4;\n"
	"mad.lo.s32 %r49,%r41,%r43,0;\n"
	"mad.lo.s32 %r49,%r42,%r44,%r49;\n"
	"bfe.s32 %r41,%r101,0,4;\n"
	"bfe.s32 %r42,%r101,4,4;\n"
	"mad.lo.s32 %r35,%r41,%r43,0;\n"
	"mad.lo.s32 %r35,%r42,%r44,%r35;\n"
	"bfe.s32 %r41,%r102,0,4;\n"
	"bfe.s32 %r42,%r102,4,4;\n"
	"mad.lo.s32 %r36,%r41,%r43,0;\n"
	"mad.lo.s32 %r36,%r42,%r44,%r36;\n"
	"bfe.s32 %r43,%r24,8,8;\n"
	"bfe.s32 %r44,%r28,8,8;\n"
	"bfe.s32 %r41,%r97,8,4;\n"
	"bfe.s32 %r42,%r97,12,4;\n"
	"mad.lo.s32 %r46,%r41,%r43,%r46;\n"
	"mad.lo.s32 %r46,%r42,%r44,%r46;\n"
	"bfe.s32 %r41,%r100,8,4;\n"
	"bfe.s32 %r42,%r100,12,4;\n"
	"mad.lo.s32 %r49,%r41,%r43,%r49;\n"
	"mad.lo.s32 %r49,%r42,%r44,%r49;\n"
	"bfe.s32 %r41,%r101,8,4;\n"
	"bfe.s32 %r42,%r101,12,4;\n"
	"mad.lo.s32 %r35,%r41,%r43,%r35;\n"
	"mad.lo.s32 %r35,%r42,%r44,%r35;\n"
	"bfe.s32 %r41,%r102,8,4;\n"
	"bfe.s32 %r42,%r102,12,4;\n"
	"mad.lo.s32 %r36,%r41,%r43,%r36;\n"
	"mad.lo.s32 %r36,%r42,%r44,%r36;\n"
	"bfe.s32 %r43,%r24,16,8;\n"
	"bfe.s32 %r44,%r28,16,8;\n"
	"bfe.s32 %r41,%r97,16,4;\n"
	"bfe.s32 %r42,%r97,20,4;\n"
	"mad.lo.s32 %r46,%r41,%r43,%r46;\n"
	"mad.lo.s32 %r46,%r42,%r44,%r46;\n"
	"bfe.s32 %r41,%r100,16,4;\n"
	"bfe.s32 %r42,%r100,20,4;\n"
	"mad.lo.s32 %r49,%r41,%r43,%r49;\n"
	"mad.lo.s32 %r49,%r42,%r44,%r49;\n"
	"bfe.s32 %r41,%r101,16,4;\n"
	"bfe.s32 %r42,%r101,20,4;\n"
	"mad.lo.s32 %r35,%r41,%r43,%r35;\n"
	"mad.lo.s32 %r35,%r42,%r44,%r35;\n"
	"bfe.s32 %r41,%r102,16,4;\n"
	"bfe.s32 %r42,%r102,20,4;\n"
	"mad.lo.s32 %r36,%r41,%r43,%r36;\n"
	"mad.lo.s32 %r36,%r42,%r44,%r36;\n"
	"bfe.s32 %r43,%r24,24,8;\n"
	"bfe.s32 %r44,%r28,24,8;\n"
	"bfe.s32 %r41,%r97,24,4;\n"
	"bfe.s32 %r42,%r97,28,4;\n"
	"mad.lo.s32 %r46,%r41,%r43,%r46;\n"
	"mad.lo.s32 %r46,%r42,%r44,%r46;\n"
	"bfe.s32 %r41,%r100,24,4;\n"
	"bfe.s32 %r42,%r100,28,4;\n"
	"mad.lo.s32 %r49,%r41,%r43,%r49;\n"
	"mad.lo.s32 %r49,%r42,%r44,%r49;\n"
	"bfe.s32 %r41,%r101,24,4;\n"
	"bfe.s32 %r42,%r101,28,4;\n"
	"mad.lo.s32 %r35,%r41,%r43,%r35;\n"
	"mad.lo.s32 %r35,%r42,%r44,%r35;\n"
	"bfe.s32 %r41,%r102,24,4;\n"
	"bfe.s32 %r42,%r102,28,4;\n"
	"mad.lo.s32 %r36,%r41,%r43,%r36;\n"
	"mad.lo.s32 %r36,%r42,%r44,%r36;\n"
	"cvt.rn.f32.s32 %f6,%r46; mul.f32 %f7,%f2,%f3; selp.f32 %f7,0F00000000,%f7,%p2;\n"
	"cvt.rn.f32.s32 %f17,%r49; mul.f32 %f18,%f4,%f3;\n"
	"or.pred %p7,%p2,%p4; selp.f32 %f18,0F00000000,%f18,%p7;\n"
	"cvt.rn.f32.s32 %f20,%r35; mul.f32 %f21,%f5,%f3;\n"
	"or.pred %p7,%p2,%p5; selp.f32 %f21,0F00000000,%f21,%p7;\n"
	"cvt.rn.f32.s32 %f23,%r36; mul.f32 %f24,%f9,%f3;\n"
	"or.pred %p7,%p2,%p6; selp.f32 %f24,0F00000000,%f24,%p7;\n"
	"st.shared.f32 [%r62],%f7; st.shared.f32 [%r62+128],%f6;\n"
	"st.shared.f32 [%r62+256],%f18; st.shared.f32 [%r62+384],%f17;\n"
	"st.shared.f32 [%r62+512],%f21; st.shared.f32 [%r62+640],%f20;\n"
	"st.shared.f32 [%r62+768],%f24; st.shared.f32 [%r62+896],%f23;\n"
	"// v43: bar.sync dropped (warp-synchronous STS->LDS; all 32 lanes reach every point)\n"
	"ld.shared.v4.f32 {%f10,%f11,%f12,%f13},[%r98+128];\n"
	"add.f32 %f8,%f10,%f11;\n"
	"add.f32 %f8,%f8,%f12;\n"
	"add.f32 %f8,%f8,%f13;\n"
	"ld.shared.f32 %f14,[%r98+0];\n"
	"add.u32 %r77,%r99,512;\n"
	"st.shared.v2.f32 [%r77],{%f14,%f8};\n"
	"// v43: bar.sync dropped (warp-synchronous STS->LDS; all 32 lanes reach every point)\n"
	"add.u32 %r96,%r103,512;\n"
	"@%p0 ld.shared.v2.f32 {%f14,%f8},[%r96+0];\n"
	"@%p0 fma.rn.f32 %f1,%f14,%f8,%f1;\n"
	"@%p0 ld.shared.v2.f32 {%f14,%f8},[%r96+32];\n"
	"@%p0 fma.rn.f32 %f1,%f14,%f8,%f1;\n"
	"@%p0 ld.shared.v2.f32 {%f14,%f8},[%r96+64];\n"
	"@%p0 fma.rn.f32 %f1,%f14,%f8,%f1;\n"
	"@%p0 ld.shared.v2.f32 {%f14,%f8},[%r96+96];\n"
	"@%p0 fma.rn.f32 %f1,%f14,%f8,%f1;\n"
	"@%p0 ld.shared.v2.f32 {%f14,%f8},[%r96+128];\n"
	"@%p0 fma.rn.f32 %f1,%f14,%f8,%f1;\n"
	"@%p0 ld.shared.v2.f32 {%f14,%f8},[%r96+160];\n"
	"@%p0 fma.rn.f32 %f1,%f14,%f8,%f1;\n"
	"@%p0 ld.shared.v2.f32 {%f14,%f8},[%r96+192];\n"
	"@%p0 fma.rn.f32 %f1,%f14,%f8,%f1;\n"
	"@%p0 ld.shared.v2.f32 {%f14,%f8},[%r96+224];\n"
	"@%p0 fma.rn.f32 %f1,%f14,%f8,%f1;\n"
	"add.u32 %r13,%r13,1;\n"
	"add.u64 %rd5,%rd5,128; add.u64 %rd6,%rd6,128; add.u64 %rd22,%rd22,128; add.u64 %rd23,%rd23,128;\n"
	"add.u64 %rd17,%rd17,292; add.u64 %rd18,%rd18,32; add.u64 %rd21,%rd21,32; add.u64 %rd7,%rd7,32; add.u64 %rd8,%rd8,32; add.u64 %rd20,%rd20,292;\n"
	"$stor:\n"
	"add.u32 %r96,%r2,%r1;\n"
	"setp.ge.u32 %p7,%r1,4;\n"
	"@%p7 mov.u32 %r96,%r11;\n"
	"setp.lt.u32 %p0,%r96,%r11;\n"
	"mul.wide.u32 %rd12,%r96,4; add.u64 %rd16,%rd4,%rd12;\n"
	"@%p0 st.global.f32 [%rd16],%f1;\n"
	"$done:\n"
	"ret;\n"
	"}\n";
        static void ck(CUresult r, const char *m) {
    const char *s = 0;
    if (r) {
    if (pErrStr) pErrStr(r, &s);
        printf("CUDA %s: %d %s\n", m, r, s);
        fflush(stdout);
    }
}

static uint64_t dev_xq, dev_sc, dev_y; /* device-resident buffers */
static int sc_staged = 0;
#define SC_REGION 134217728ULL
#define SC_CAP 16777216u
#define XQ_CAP 65536u
#define Y_CAP 524288u
static CUcontext g_ctx;
static int ctx_current;
/* v38 (Exp30): completion via event record + query poll on a dedicated
 * non-blocking stream. Measured on this R440 Tegra driver: cuCtxSynchronize
 * costs ~150-250us per call (launch+ctxSync 512us vs launch+eventPoll 359us
 * at rows=1536, with y bit-identical). EventQuery returns 600 while pending. */
static void *g_stream; static void *g_event;
static CUresult (*pStreamCreate)(void **, unsigned);
static CUresult (*pEventCreate)(void **, unsigned);
static CUresult (*pEventRecord)(void *, void *);
static CUresult (*pEventQuery)(void *);
static CUresult (*pCtxSetCurrent)(CUcontext);
static int gpu_lock;

void cuda_init(void) {
    uint32_t h[10];
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
    pMemcpyHtoDAsync = (CUresult (*)(uint64_t, const void *, size_t, void *))
        dlsym(libcuda, "cuMemcpyHtoDAsync_v2");
    pMemcpyDtoHAsync = (CUresult (*)(void *, uint64_t, size_t, void *))
        dlsym(libcuda, "cuMemcpyDtoHAsync_v2");
    pMemHostAlloc = (CUresult (*)(void **, size_t, unsigned))
        dlsym(libcuda, "cuMemHostAlloc");
    pMemFreeHost = (CUresult (*)(void *))dlsym(libcuda, "cuMemFreeHost");
    pModLoadDataEx = dlsym(libcuda, "cuModuleLoadDataEx");
    pLaunch = dlsym(libcuda, "cuLaunchKernel");
    pCtxSync = dlsym(libcuda, "cuCtxSynchronize");
    pErrStr = dlsym(libcuda, "cuGetErrorString");
    pStreamCreate = (CUresult (*)(void **, unsigned))dlsym(libcuda, "cuStreamCreate");
    pStreamSync = (CUresult (*)(void *))dlsym(libcuda, "cuStreamSynchronize");
    pEventCreate = (CUresult (*)(void **, unsigned))dlsym(libcuda, "cuEventCreate");
    pEventRecord = (CUresult (*)(void *, void *))dlsym(libcuda, "cuEventRecord");
    pEventQuery = (CUresult (*)(void *))dlsym(libcuda, "cuEventQuery");
    if (!pLaunch || !pCtxSync) { printf("no launch/sync\n"); return; }
    { CUresult r = pInit(0); if (r) { ck(r, "init"); return; } }
    { CUdevice d; CUcontext c;
      if (pDevGet(&d, 0)) return;
      if (pCtxCreate_v2(&c, 0, d)) return;
      g_ctx = c;
      pCtxSetCurrent = (CUresult (*)(CUcontext))dlsym(libcuda, "cuCtxSetCurrent");
      if (pCtxSetCurrent && !pCtxSetCurrent(g_ctx)) ctx_current = 1;
      if (pMemHostAlloc && pMemcpyHtoDAsync && pMemcpyDtoHAsync) {
          if (!pMemHostAlloc(&h_in, XQ_CAP, CU_MEMHOSTALLOC_PORTABLE) &&
              !pMemHostAlloc(&h_out, Y_CAP, CU_MEMHOSTALLOC_PORTABLE))
              printf("CUDA pinned staging %u+%u B\n",
                     (unsigned)XQ_CAP, (unsigned)Y_CAP);
          else { h_in = h_out = 0; printf("pinned staging unavailable (pageable)\n"); }
      }
      if (pStreamCreate && pEventCreate && pEventRecord && pEventQuery) {
          if (pStreamCreate(&g_stream, 1) == 0 && pEventCreate(&g_event, 0) == 0)
              printf("CUDA event-completion path\n");
          else { g_stream = 0; g_event = 0; }
      }
      if (pModLoadDataEx && getenv("K2DBG")) {
          int jit_opt[] = { 5 /*ERROR_LOG_BUFFER*/, 6 /*ERROR_LOG_BUFFER_SIZE*/,
                            9 /*LOG_VERBOSE*/ };
          void *jit_val[] = { jit_log, (void *)4096, (void *)1 };
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
           memory (proven: host pointers fault 719). Stage weights ONCE, in the
           v11 single-copy layout: Q4 code repacked to 16B blocks (q4_build),
           NO raw 714MB blob mirror (that double-stage OOMed the Nano: run #30). */
        CUresult r;
        if (!pMemAlloc || !pMemcpyDtoH) { printf("no memalloc/dtoH\n"); return; }
        if (!q4_build()) { printf("q4 build failed -> CPU only\n"); return; }
        r = pMemAlloc(&dev_xq, XQ_CAP);
        if (r) { ck(r, "memalloc-xq"); return; }
        r = pMemAlloc(&dev_y, Y_CAP);
        if (r) { ck(r, "memalloc-y"); return; }
        printf("CUDA weights staged on device (single-copy repack)\n");
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
static int dbg = -2;

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

/* Exp93: copy out of the pinned (CPU-UNCACHED on this Tegra) y staging buffer.
   Measured: libc memcpy reads it at 0.72 GB/s, one 64 B ld1 stream 0.84, FOUR
   interleaved 64 B streams 1.14-1.17 GB/s (2 and 8 streams do not help). About 2 MB
   of results per token are read back, so the copy is worth ~1 ms/token. The four
   streams are the four quarters of the buffer; the (<256 B) remainder uses memcpy. */
static void copy_from_pinned(void *dst, const void *src, size_t n) {
    size_t q = (n / 256) * 64;            /* bytes per stream, multiple of 64 */
    char *d = (char *)dst; const char *s = (const char *)src;
    size_t i;
    if (q < 64) { memcpy(dst, src, n); return; }
    for (i = 0; i < q; i += 64) {
        __asm__ volatile(
            "ld1 {v0.16b-v3.16b}, [%4]\n\t"
            "ld1 {v4.16b-v7.16b}, [%5]\n\t"
            "ld1 {v16.16b-v19.16b}, [%6]\n\t"
            "ld1 {v20.16b-v23.16b}, [%7]\n\t"
            "st1 {v0.16b-v3.16b}, [%0]\n\t"
            "st1 {v4.16b-v7.16b}, [%1]\n\t"
            "st1 {v16.16b-v19.16b}, [%2]\n\t"
            "st1 {v20.16b-v23.16b}, [%3]\n\t"
            :: "r"(d + i), "r"(d + q + i), "r"(d + 2 * q + i), "r"(d + 3 * q + i),
               "r"(s + i), "r"(s + q + i), "r"(s + 2 * q + i), "r"(s + 3 * q + i)
            : "v0","v1","v2","v3","v4","v5","v6","v7",
              "v16","v17","v18","v19","v20","v21","v22","v23","memory");
    }
    if (4 * q < n) memcpy(d + 4 * q, s + 4 * q, n - 4 * q);
}

__attribute__((visibility("default")))
void cuda_gemv(uint32_t r0, uint32_t r1, uint32_t cols, uint64_t w,
               uint64_t x_act_unused, uint64_t y, uint64_t xq) {
    uint32_t nb32 = cols >> 5;
    uint64_t dev_w = 0;
    uint32_t rowsf = q4_classify(w, &dev_w);
    /* Exp70: rowsf from classify() is the row count of the SINGLE tensor that
       contains w; a merged-group call (kind 9/10) asks for more rows. Keep the
       classify value for the self-test's CPU reference (which can only rebuild
       one tensor), and use the requested count (r1-r0) for the real launch. */
    uint32_t rows_cls = rowsf;
    uint64_t d_sc = dev_sc, d_xq = dev_xq, d_y = dev_y, scb;
    size_t nrows, sc_bytes, xq_bytes;
    void *args[] = { &dev_w, &d_sc, &d_xq, &d_y, (void *)&r0, (void *)&r1,
                     (void *)&nb32, (void *)&rowsf };
    unsigned grid;
    CUresult r;
    extern uint32_t g_nthreads;
    if (!cuda_ok || !rowsf || g_nthreads != 1) {
        /* CPU path (bit-exact): GPU stays threads-1 only — at threads>1
           per-call staging + one global lock serialize the workers.
           NOTE (Exp70): a merged-group call (rows=3072 q+k+v / 10240 gate+up)
           is NOT valid on this fallback, which walks the SOURCE 18B-stride
           layout where the tensors are separated by the fp16/fp32 norms. The
           GPU image is compact (q4_build running-sum) so only the GPU path can
           span tensors. A GPU loss therefore fails the golden gate loudly. */
        k2q4_gemv_range(r0, r1, cols, w, x_act_unused, y, xq);
        return;
    }
    /* Exp70: merged tensor groups request MORE rows than q4_classify() matched
       (it returns the row count of the single tensor containing w). The kernel
       ignores prf, and the only other use of rowsf is the sc-region bounds test
       below, which must see the true requested row count. For every pre-existing
       call r1-r0 == tbl_krows[kind] == the classify result, so this is a no-op
       there. */
    rowsf = r1 - r0;
    grid = (r1 > r0) ? (r1 - r0 + 31) / 32 : 0;
    if (!grid) return;
    nrows = (size_t)(r1 - r0);
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
    xq_bytes = (((size_t)nb32 + 7) / 8) * 292 + 64;
    if (xq_bytes > XQ_CAP || nrows * 4 > Y_CAP) {
        k2q4_gemv_range(r0, r1, cols, w, x_act_unused, y, xq); return; }
    while (__atomic_exchange_n(&gpu_lock, 1, __ATOMIC_ACQUIRE))
        __asm__ volatile("yield");
    if (pCtxSetCurrent && !ctx_current) {
        if (!pCtxSetCurrent(g_ctx)) ctx_current = 1;
    }
    /* stage inputs to device (absolute-row layout preserved at r0 offsets) */
    (void)sc_bytes;
    if (dd_on < 0) dd_on = (getenv("K2DEDUP") && getenv("K2DEDUP")[0] == '0') ? 0 : 1;
    if (dd_on) {
        int skip = 0;
        if (xq_staged && d_xq == xq_last_dst && xq_bytes == xq_last_len &&
            xq == xq_last_src &&
            !memcmp(h_in, (const void *)(uintptr_t)xq, xq_bytes))
            skip = 1;
        if (skip) {
            dd_skip++; dd_bytes += (long long)xq_bytes;
            r = CUDA_SUCCESS;
        } else {
            memcpy(h_in, (void *)(uintptr_t)xq, xq_bytes);
            r = pMemcpyHtoDAsync(d_xq, h_in, xq_bytes, g_stream);
            if (!r) { xq_last_dst = d_xq; xq_last_len = xq_bytes;
                      xq_last_src = xq; xq_staged = 1; }
        }
        dd_calls++;
        if ((dd_calls & 4095) == 0) dd_dump();
    } else if (h_in) {
        memcpy(h_in, (void *)(uintptr_t)xq, xq_bytes);
        r = pMemcpyHtoDAsync(d_xq, h_in, xq_bytes, g_stream);
    } else if (pMemcpyHtoDAsync)
        r = pMemcpyHtoDAsync(d_xq, (void *)(uintptr_t)xq, xq_bytes, g_stream);
    else
        r = pMemcpyHtoD(d_xq, (void *)(uintptr_t)xq, xq_bytes);
    { static int ngc = 0;
      if (dbg != 0 && getenv("K2ARGS") && ngc < 3) {
          printf("ARGS%d r0=%u r1=%u nb=%u rowsf=%u woff=%llx dsc=%llx dxq=%llx dy=%llx grid=%u\n",
                 ngc, r0, r1, nb32, rowsf, (unsigned long long)(w - blob_vaddr),
                 (unsigned long long)d_sc, (unsigned long long)d_xq,
                 (unsigned long long)d_y, grid); fflush(stdout); ngc++; } }
    if (!r) r = pLaunch(k_gemv, grid, 1, 1, 32, 8, 1, 0, g_stream, args, NULL);
    if (!r && h_out)
        r = pMemcpyDtoHAsync(h_out, d_y + (uint64_t)r0 * 4, nrows * 4, g_stream);
    else if (!r && pMemcpyDtoHAsync)
        r = pMemcpyDtoHAsync((void *)(uintptr_t)(y + (uint64_t)r0 * 4),
                             d_y + (uint64_t)r0 * 4, nrows * 4, g_stream);
    if (!r) {
        if (g_event) {
            r = pEventRecord(g_event, g_stream);
            if (!r) {
                long spins = 0;
                while (pEventQuery(g_event) == 600) {
                    if (++spins == 50000) sched_yield();
                    if (spins > 5000000) { printf("CUDA event poll timeout\n"); break; }
                }
            }
            if (r) { printf("CUDA event record failed %d -> CPU mode\n", r);
                     cuda_ok = 0; g_event = 0; }
        } else r = pCtxSync();
    }
    if (!r && h_out)
        copy_from_pinned((void *)(uintptr_t)(y + (uint64_t)r0 * 4), h_out, nrows * 4);
    if (!r && !pMemcpyDtoHAsync)
        r = pMemcpyDtoH((void *)(uintptr_t)(y + (uint64_t)r0 * 4),
                        d_y + (uint64_t)r0 * 4, nrows * 4);
    if (r) { ck(r, "gpu-path"); cuda_ok = 0; }
    /* Self-test v3: first 8 GEMV calls, full row range, INSIDE gpu_lock,
       dedicated host+device scratch — engine y is NEVER touched (peer #6:
       writing y clobbered caller state -> SIGSEGV). */
    if (cuda_ok && selftest_remaining > 0) {
        static float s_gpu[65536], s_cpu[65536];
        static uint64_t dev_s = 0;
        uint32_t t_r0 = 0, t_r1 = rows_cls;
        size_t full = (size_t)rows_cls * 4;
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
            r2 = pLaunch(k_gemv, (rows_cls + 15) / 16, 1, 1, 32, 8, 1,
                         0, NULL, targs, NULL);
            if (!r2) r2 = pCtxSync();
            if (!r2) r2 = pMemcpyDtoH(s_gpu, dev_s, full);
            if (r2) { ck(r2, "selftest-path"); pass = 0; }
        }
        if (pass) {
            cuda_q4_ref(rows_cls, cols, (const uint8_t *)(uintptr_t)w,
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
    if (dbg == -2) dbg = getenv("K2DBG") ? 1 : 0;
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
    if (dd_on && (dd_calls % 5000) == 0) dd_dump();
    __atomic_store_n(&gpu_lock, 0, __ATOMIC_RELEASE);
    if (!cuda_ok) k2q4_gemv_range(r0, r1, cols, w, x_act_unused, y, xq);
}
