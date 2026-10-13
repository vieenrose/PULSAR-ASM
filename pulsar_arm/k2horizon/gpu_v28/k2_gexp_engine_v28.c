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
    if (pMemAlloc(&drp, total)) { free(tmp); printf("q4_build: memalloc failed\n"); return 0; }
    if (pMemcpyHtoD(drp, tmp, total)) { free(tmp); printf("q4_build: h2d failed\n"); return 0; }
    free(tmp);
    dev_q4 = drp;
    printf("CUDA q4 repacked single-copy %llu MB, %u layers\n",
           (unsigned long long)(total >> 20), q4_layers);
    q4_ready = 1;
    return 1;
}

static const char *ptx =
	"// K2 Q4 GEMV v28 (Exp22: xor-once 0x88888888, v4 exchange load, hoisted exchange/replay index bases)\n"
	"// K2 Q4 GEMV v28 (Exp22: 4-lane replay — row q chain owned by lane q; 96->16 replay instrs/iter)\n"
	".version 4.0\n"
	".target sm_50\n"
	".address_size 64\n"
	".visible .entry q4_gemv(.param .u64 pw, .param .u64 psc, .param .u64 pxq, .param .u64 py, .param .u32 pr0, .param .u32 pr1, .param .u32 pnb, .param .u32 prf) {\n"
	".reg .u64 %rd<24>; .reg .u32 %r<128>; .reg .f32 %f<32>; .reg .pred %p<8>;\n"
	".shared .align 4 .b32 shbuf[4096];\n"
	"ld.param.u64 %rd1,[pw]; ld.param.u64 %rd2,[psc]; ld.param.u64 %rd3,[pxq]; ld.param.u64 %rd4,[py];\n"
	"ld.param.u32 %r10,[pr0]; ld.param.u32 %r11,[pr1]; ld.param.u32 %r12,[pnb];\n"
	"mov.u32 %r1,%tid.x; mov.u32 %r2,%tid.y; mov.u32 %r3,%ctaid.x;\n"
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
	"bar.sync 0;\n"
	"ld.shared.v4.f32 {%f10,%f11,%f12,%f13},[%r98+128];\n"
	"add.f32 %f8,%f10,%f11;\n"
	"add.f32 %f8,%f8,%f12;\n"
	"add.f32 %f8,%f8,%f13;\n"
	"ld.shared.f32 %f14,[%r98+0];\n"
	"add.u32 %r77,%r99,0;\n"
	"st.shared.v2.f32 [%r77],{%f14,%f8};\n"
	"bar.sync 0;\n"
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
	"bar.sync 0;\n"
	"ld.shared.v4.f32 {%f10,%f11,%f12,%f13},[%r98+128];\n"
	"add.f32 %f8,%f10,%f11;\n"
	"add.f32 %f8,%f8,%f12;\n"
	"add.f32 %f8,%f8,%f13;\n"
	"ld.shared.f32 %f14,[%r98+0];\n"
	"add.u32 %r77,%r99,256;\n"
	"st.shared.v2.f32 [%r77],{%f14,%f8};\n"
	"bar.sync 0;\n"
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
	"bar.sync 0;\n"
	"ld.shared.v4.f32 {%f10,%f11,%f12,%f13},[%r98+128];\n"
	"add.f32 %f8,%f10,%f11;\n"
	"add.f32 %f8,%f8,%f12;\n"
	"add.f32 %f8,%f8,%f13;\n"
	"ld.shared.f32 %f14,[%r98+0];\n"
	"add.u32 %r77,%r99,512;\n"
	"st.shared.v2.f32 [%r77],{%f14,%f8};\n"
	"bar.sync 0;\n"
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

__attribute__((visibility("default")))
void cuda_gemv(uint32_t r0, uint32_t r1, uint32_t cols, uint64_t w,
               uint64_t x_act_unused, uint64_t y, uint64_t xq) {
    uint32_t nb32 = cols >> 5;
    uint64_t dev_w = 0;
    uint32_t rowsf = q4_classify(w, &dev_w);
    uint64_t d_sc = dev_sc, d_xq = dev_xq, d_y = dev_y, scb;
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
    if (pCtxSetCurrent) pCtxSetCurrent(g_ctx);
    /* stage inputs to device (absolute-row layout preserved at r0 offsets) */
    (void)sc_bytes;
    r = pMemcpyHtoD(d_xq, (void *)(uintptr_t)xq, xq_bytes);
    { static int ngc = 0;
      if (dbg != 0 && getenv("K2ARGS") && ngc < 3) {
          printf("ARGS%d r0=%u r1=%u nb=%u rowsf=%u woff=%llx dsc=%llx dxq=%llx dy=%llx grid=%u\n",
                 ngc, r0, r1, nb32, rowsf, (unsigned long long)(w - blob_vaddr),
                 (unsigned long long)d_sc, (unsigned long long)d_xq,
                 (unsigned long long)d_y, grid); fflush(stdout); ngc++; } }
    if (!r) r = pLaunch(k_gemv, grid, 1, 1, 32, 8, 1, 0, NULL, args, NULL);
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
            r2 = pLaunch(k_gemv, (rowsf + 15) / 16, 1, 1, 32, 8, 1,
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
    if (pCtxSetCurrent) pCtxSetCurrent(0);
    __atomic_store_n(&gpu_lock, 0, __ATOMIC_RELEASE);
    if (!cuda_ok) k2q4_gemv_range(r0, r1, cols, w, x_act_unused, y, xq);
}
