/* fwd8.c: Ternary-Bonsai-8B (Qwen3-8B dense, PQ2_0) single-token forward.
 * 36 layers, HID 4096, INTER 12288, 32 heads x 128, 8 kv heads, vocab 151669.
 * All full attention (no SSM). YaRN rope (factor 4, orig 16384, base 1e6).
 * NO Hadamard transform on this checkpoint (no prism keys): plain PQ2_0
 * integer dots with Q8_K activations (mirrors fork generic exactly).
 * Q/K per-head RMS-128 norms before rope (mirror of 27B pattern; the greedy
 * diff judges - drop them if it mismatches).
 * Validation: greedy argmax stream vs :8091 server temp-0 on same ids.
 *
 * Build: gcc -O2 -fopenmp -march=armv8.2-a+dotprod -Dmain=pq2_main
 *          -c pq2_gemv.c -o fwd8_pq2.o
 *        gcc -O2 -fopenmp -march=armv8.2-a+dotprod -c fwd8.c -o fwd8.o
 *        gcc -O2 -fopenmp -o fwd8 fwd8.o fwd8_pq2.o
 *          -L$LB -lggml-base -lggml-cpu -Wl,-rpath,$LB -lm
 * Run: OMP_NUM_THREADS=20 taskset -c 0-19 ./fwd8 <gguf> <ids...> --gen G
 *      [--sample temp topp topk seed [minp]] [--trace]
 */
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <arm_neon.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define HID 4096
#define INTER 12288
#define NHEAD 32
#define HDIM 128
#define NKV 8
#define NLAYER 36
#define VOCAB 151669
#ifdef PQ1_BUILD
/* f16 K/V cache and the fork's flash-attention decode arithmetic (pq1_host.c
 * pq1_fa16_head); the device cache is sized per run (KVCTX) */
#define CTX 16384   /* compile-time ceiling; runs default to 4096 (PQ1_CTX), cache KVCTX */
#else
#define CTX 512
#endif
#define YARN_FACTOR 4.0f
#define YARN_ORIG 16384
#define FREQ_BASE 1000000.0f

#ifndef STANDALONE
void quantize_row_q8_K(const float * x, void * y, int64_t k);
#endif
/* Self-contained q8_K activation quantizer: textually identical to the
   fork's quantize_row_q8_K_ref + nearest_int (and the arm arch file just
   forwards to the ref), so the engine links no libggml on Android.
   Verified byte-for-byte against the library on random and real data. */
static inline int nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}
static void q8_K_quant(const float *x, uint8_t *yv, int64_t k) {
    const int nb = (int)(k / 256);
    for (int i = 0; i < nb; i++, x += 256) {
        uint8_t *yb = yv + (size_t)i * 292;
        int8_t *qs = (int8_t *)(yb + 4);
        int16_t *bsums = (int16_t *)(yb + 4 + 256);
        /* amax = max |x| (order-free); max = x at the FIRST index reaching
           it, which is what the serial strict-> scan selects. */
        float32x4_t am = vdupq_n_f32(0);
        for (int j = 0; j < 256; j += 4) am = vmaxq_f32(am, vabsq_f32(vld1q_f32(x + j)));
        const float amax = vmaxvq_f32(am);
        if (!amax) {
            /* the fork zeroes only d and qs here, leaving bsums stale; the
               kernel never reads bsums, but matching it exactly is what makes
               the byte-for-byte verification against the library meaningful */
            memset(yb, 0, 4);
            memset(yb + 4, 0, 256);
            continue;
        }
        /* first index with |x| == amax, 16 lanes per probe (#74): same
           element the serial strict-> scan selects */
        float max = 0;
        {
            const float32x4_t am4 = vdupq_n_f32(amax);
            int j = 0;
            for (; j < 256; j += 16) {
                uint32x4_t e = vorrq_u32(vorrq_u32(vceqq_f32(vabsq_f32(vld1q_f32(x + j)), am4),
                                                   vceqq_f32(vabsq_f32(vld1q_f32(x + j + 4)), am4)),
                                         vorrq_u32(vceqq_f32(vabsq_f32(vld1q_f32(x + j + 8)), am4),
                                                   vceqq_f32(vabsq_f32(vld1q_f32(x + j + 12)), am4)));
                if (vmaxvq_u32(e)) break;
            }
            for (int jj = j; jj < j + 16; ++jj) if (fabsf(x[jj]) == amax) { max = x[jj]; break; }
        }
        const float iscale = -127.f / max;
        /* nearest_int(iscale*x) per lane: one IEEE multiply, the +2^23*1.5
           magic add (FPCR round-to-nearest-even, as the scalar FPU), the
           mantissa bit tricks on the integer image, then MIN(127, v). */
        const float32x4_t is4 = vdupq_n_f32(iscale), magic = vdupq_n_f32(12582912.f);
        const int32x4_t m23 = vdupq_n_s32(0x007fffff), b22 = vdupq_n_s32(0x00400000);
        const int32x4_t c127 = vdupq_n_s32(127);
#define Q8K_LANE4(off) vminq_s32(vsubq_s32(vandq_s32(vreinterpretq_s32_f32( \
            vaddq_f32(vmulq_f32(vld1q_f32(x + (off)), is4), magic)), m23), b22), c127)
        /* bsums: 8 int16 group sums per half block, reduced in registers by
           a pairwise-add tree and stored as one vector (#74); integer sums
           of 16 int8 are exact in any order, |sum| <= 16*127 */
        for (int j = 0; j < 256; j += 128) {
            int16x8_t sm[8];
            for (int g = 0; g < 8; g++) {
                const int jg = j + 16 * g;
                /* narrowing by uzp1 (low halves of the lane pairs) = xtn+ins
                   in one op; the values fit in int8 so the halves are exact */
                int16x8_t h01 = vuzp1q_s16(vreinterpretq_s16_s32(Q8K_LANE4(jg)),
                                           vreinterpretq_s16_s32(Q8K_LANE4(jg + 4)));
                int16x8_t h23 = vuzp1q_s16(vreinterpretq_s16_s32(Q8K_LANE4(jg + 8)),
                                           vreinterpretq_s16_s32(Q8K_LANE4(jg + 12)));
                vst1q_s8(qs + jg, vuzp1q_s8(vreinterpretq_s8_s16(h01), vreinterpretq_s8_s16(h23)));
                sm[g] = vaddq_s16(h01, h23);
            }
            int16x8_t p01 = vpaddq_s16(sm[0], sm[1]), p23 = vpaddq_s16(sm[2], sm[3]);
            int16x8_t p45 = vpaddq_s16(sm[4], sm[5]), p67 = vpaddq_s16(sm[6], sm[7]);
            vst1q_s16(bsums + (j >> 4), vpaddq_s16(vpaddq_s16(p01, p23), vpaddq_s16(p45, p67)));
        }
#undef Q8K_LANE4
        float d = 1.0f / iscale;
        memcpy(yb, &d, 4);
    }
}
void pq2_gemv(int rows, int cols, const uint8_t *w, const float *x, float *y,
              const uint8_t *xq);

#ifdef BQ1_TIME
/* Harness instrumentation only (no math): per-step wall time so the
 * autoresearch loop can separate prefill from decode. Guarded so other
 * builds (Spark/phone) are byte-identical to before. */
#include <time.h>
static double bq1_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
#define BQ1_T0() double _t0 = bq1_now()
#define BQ1_T1(step)     printf("BQ1TIME step=%d ms=%.3f\n", step, (bq1_now() - _t0) * 1e3), fflush(stdout)
/* Pass-level CPU profile (PQ1_CPUTIME=1): time between consecutive marks is
 * charged to the category of the mark, so every pass of the step loop is
 * accounted for once. ~600 clock_gettime calls per token; no math touched. */
enum { CT_SAMPLE, CT_EMBED, CT_RMS, CT_QUANT, CT_GEMV, CT_HNORM, CT_ROPE,
       CT_KVCPY, CT_ATTN, CT_RESID, CT_SILU, CT_ARGMAX, CT_N };
static const char *CT_NAME[CT_N] = {"sample", "embed", "rms", "quant", "gemv",
    "headnorm", "rope", "kvcopy", "attn", "resid", "silu", "argmax"};
static double CT_ACC[CT_N], CT_LAST;
static int CT_ON = -1, CT_STEPS;
static inline void ct_mark(int cat) {
    if (CT_ON <= 0) return;
    double n = bq1_now();
    CT_ACC[cat] += n - CT_LAST;
    CT_LAST = n;
}
static void ct_begin(int step, int n0) {
    if (CT_ON < 0) CT_ON = getenv("PQ1_CPUTIME") ? 1 : 0;
    if (CT_ON <= 0) return;
    if (step == n0) { memset(CT_ACC, 0, sizeof CT_ACC); CT_STEPS = 0; }
    if (step >= n0) CT_STEPS++;
    CT_LAST = bq1_now();
}
static void ct_dump(void) {
    if (CT_ON <= 0 || !CT_STEPS) return;
    double tot = 0;
    for (int i = 0; i < CT_N; i++) tot += CT_ACC[i];
    fprintf(stderr, "PQ1CPU decode steps=%d total=%.1f ms/tok\n", CT_STEPS, tot * 1e3 / CT_STEPS);
    for (int i = 0; i < CT_N; i++)
        fprintf(stderr, "PQ1CPU %-9s %8.2f ms/tok %5.1f%%\n", CT_NAME[i],
                CT_ACC[i] * 1e3 / CT_STEPS, 100.0 * CT_ACC[i] / tot);
}
#else
#define BQ1_T0() ((void)0)
#define BQ1_T1(step) ((void)0)
#define ct_mark(c) ((void)0)
#define ct_begin(s, n) ((void)0)
#define ct_dump() ((void)0)
#endif

static int TRACE = 0;
#define MAG(nm, v, n) do { double _s = 0; int _n = (n); \
    for (int _i = 0; _i < _n; _i++) _s += (double)(v)[_i] * (v)[_i]; \
    printf("  %s rms=%.4g\n", nm, sqrt(_s / _n)); } while (0)
#define V8(nm, v) do { printf("  %s v8:", nm); \
    for (int _i = 0; _i < 8; _i++) printf(" %.6g", (double)(v)[_i]); \
    printf("\n"); } while (0)
#define V8R(nm, v, s) do { printf("  %s [%d]:", nm, (s)); \
    for (int _i = 0; _i < 8; _i++) printf(" %.6g", (double)(v)[(s) + _i]); \
    printf("\n"); } while (0)

static uint8_t *G;
static char TN[500][64];
static uint64_t TO[500];
static int NT;

static void gguf_table(const char *path) {
    int fd = open(path, O_RDONLY);
    struct stat st; fstat(fd, &st);
    G = mmap(0, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    uint8_t *p = G;
    if (memcmp(p, "GGUF", 4)) { printf("not gguf\n"); exit(1); }
    p += 4 + 4;
    uint64_t nt = *(uint64_t *)p, nkv = *(uint64_t *)(p + 8); p += 16;
    for (uint64_t i = 0; i < nkv; i++) {
        uint64_t ln = *(uint64_t *)p; p += 8; p += ln;
        int t = *(int *)p; p += 4;
        if (t == 0 || t == 4 || t == 10) p += (t == 0 ? 1 : t == 4 ? 4 : 8);
        else if (t == 1 || t == 3 || t == 5 || t == 11) p += (t == 1 ? 1 : t == 3 ? 2 : t == 5 ? 4 : 8);
        else if (t == 2) p += 2;
        else if (t == 6 || t == 12) p += (t == 6 ? 4 : 8);
        else if (t == 7) p += 1;
        else if (t == 8) { uint64_t n = *(uint64_t *)p; p += 8; p += n; }
        else if (t == 9) {
            int et = *(int *)p; uint64_t n = *(uint64_t *)(p + 4); p += 12;
            if (et == 8) { for (uint64_t k = 0; k < n; k++) { uint64_t m = *(uint64_t *)p; p += 8 + m; } }
            else { int sz = et <= 1 ? 1 : et <= 3 ? 2 : et <= 6 ? 4 : et == 7 ? 1 : 8; p += n * sz; }
        } else { printf("bad kv type %d\n", t); exit(1); }
    }
    NT = (int)nt;
    for (int i = 0; i < NT; i++) {
        uint64_t ln = *(uint64_t *)p; p += 8;
        memcpy(TN[i], p, ln < 63 ? ln : 63); TN[i][ln < 63 ? ln : 63] = 0; p += ln;
        int nd = *(int *)p; p += 4;
        for (int d = 0; d < nd; d++) p += 8;
        p += 4;
        uint64_t off = *(uint64_t *)p; p += 8;
        TO[i] = off;
    }
    uint64_t ds = (((uint64_t)(p - G)) + 31) & ~31ull;
    for (int i = 0; i < NT; i++) TO[i] += ds;
}
static uint8_t *tw(const char *name) {
    for (int i = 0; i < NT; i++)
        if (!strcmp(TN[i], name)) return G + TO[i];
    printf("tensor missing: %s\n", name);
    exit(1);
}
static char NB[128];
static char *tn(int il, const char *kind, char *o) {
    sprintf(o, "blk.%d.%s", il, kind);
    return o;
}
#define TW(il, kind) tw(tn(il, kind, NB))

static float h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu, o;
    if (e == 0) { if (!m) return 0.0f; float f = (float)m * 5.9604645e-8f; memcpy(&o, &f, 4); o |= s; }
    else if (e == 31) o = s | 0x7f800000u | (m << 13);
    else o = s | ((e + 112u) << 23) | (m << 13);
    float r; memcpy(&r, &o, 4); return r;
}

/* PQ2_0 row dequant (00=-1,01=0,10=+1,11=+2), scale-first 34B blocks */
static void dec_row(const uint8_t *W, int row, int cols, float *o) {
    int ng = cols / 128;
    const uint8_t *r = W + (size_t)row * ng * 34;
    for (int g = 0; g < ng; g++) {
        const uint8_t *b = r + g * 34;
        float sc = h2f((uint16_t)(b[0] | (b[1] << 8)));
        const uint8_t *q = b + 2;
        for (int j = 0; j < 128; j++)
            o[g * 128 + j] = (float)(((q[j / 4] >> ((j % 4) * 2)) & 3) - 1) * sc;
    }
}

static uint8_t *XQ;   /* (12288/256)*292 max */
static void gemvP(const uint8_t *W, int rows, int K, const float *x, float *y) {
    q8_K_quant(x, XQ, K);
    ct_mark(CT_QUANT);
    pq2_gemv(rows, K, W, x, y, XQ);
    ct_mark(CT_GEMV);
}

/* One launch for the projections that consume the same activation vector
   (q/k/v share XN, gate/up share the post-attention norm). Run #53's phase
   timing measured the per-call API cost at ~40us H2D + ~45us launch + ~100us
   D2H, so 253 launches/token pay 46.6ms = 12.8% of the decode step in fixed
   overhead. The batch entry is a weak reference: builds that do not link it
   (the 2-bit path) take the scalar loop, and a non-zero return (tensor not
   repacked yet, no CUDA, launch failure) falls back to the same loop, so the
   results are the scalar path's bit for bit. */
static void gemvP_multi(const uint8_t *const *W, const int *rows, int n, int K,
                        const float *x, float *const *Y) {
    q8_K_quant(x, XQ, K);
    ct_mark(CT_QUANT);
    if (pq2_gemv_batch && pq2_gemv_batch(W, rows, n, K, Y, XQ) == 0) { ct_mark(CT_GEMV); return; }
    for (int i = 0; i < n; i++) pq2_gemv(rows[i], K, W[i], x, Y[i], XQ);
    ct_mark(CT_GEMV);
}

static void rms(const float *x, const float *w, int n, float *y) {
    double s = 0;
    for (int i = 0; i < n; i++) s += (double)x[i] * x[i];
    float sc = 1.0f / sqrtf((float)(s / n) + 1e-6f);
    for (int i = 0; i < n; i++) y[i] = x[i] * sc * w[i];
}
static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float silu(float x) { return x * sigmoid(x); }

/* YaRN rope, full-dim NEOX pairs (j, j+64), n_dims=128.
 * Runtime params verified against the live model: freq_scale = 1/factor
 * (=0.25), ext = 1, corr from beta 32/1, theta_scale = base^(-2/128),
 * and magnitude scale 1+0.1*ln(4) = 1.13863 (the rope op applies it;
 * fork k_cache rms / Kcur rms = 1.13863 exactly). */
static float CORR[2];
static int CORR_INIT = 0;
static float yarn_corr_dim(int n_dims, int n_ctx, float rot, float base) {
    return n_dims * logf(n_ctx / (rot * 2 * (float)M_PI)) / (2 * logf(base));
}
static void yarn_init(void) {
    float st = floorf(yarn_corr_dim(128, YARN_ORIG, 32.0f, FREQ_BASE));
    float en = ceilf(yarn_corr_dim(128, YARN_ORIG, 1.0f, FREQ_BASE));
    CORR[0] = st > 0 ? st : 0;
    CORR[1] = en < 127 ? en : 127;
    CORR_INIT = 1;
}
/* The 64 (cos, sin) pairs depend only on pos, not on the head or the layer,
 * so they are computed once per position (same expressions, same order) and
 * reused by the 40 heads x 36 layers of that token: 92k cosf/sinf per token
 * become 128. The per-element rotation is untouched. */
static float RC_C[64], RC_S[64];
static int RC_POS = -1;
static void rope_table(int pos) {
    if (!CORR_INIT) yarn_init();
    const float fs = 1.0f / YARN_FACTOR, msc = 1.0f + 0.1f * logf(YARN_FACTOR);
    const float tscale = powf(FREQ_BASE, -2.0f / 128.0f);
    float theta = (float)pos;
    for (int j = 0; j < 64; j++, theta *= tscale) {
        float y = (j - CORR[0]) / fmaxf(0.001f, CORR[1] - CORR[0]);
        float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
        float ti = fs * theta, te = theta;
        float th = ti * (1 - ramp) + te * ramp;
        RC_C[j] = cosf(th) * msc; RC_S[j] = sinf(th) * msc;
    }
    RC_POS = pos;
}
static void rope_yarn_apply(float *x, int heads, int pos) {
    if (RC_POS != pos) rope_table(pos);
    for (int h = 0; h < heads; h++) {
        float *xh = x + h * 128;
        for (int j = 0; j < 64; j++) {
            float c = RC_C[j], s = RC_S[j];
            float a = xh[j], b = xh[j + 64];
            xh[j] = a * c - b * s;
            xh[j + 64] = a * s + b * c;
        }
    }
}

static float X[HID], XN[HID], AO[HID], FO[HID];
static float QF[32 * 128], KF[8 * 128], VF[8 * 128];
static float ATTO[32 * 128], LG[INTER], LU[INTER];
#ifdef PQ1_BUILD
/* PQ1_FFN_CHECK=1: after every fused FFN graph, rerun the host-driven path
 * (gemvP_multi, pq1_silu_mul, gemvP) and compare, bit for bit, the device's
 * Q8_K down activation (d + qs of every block) and the FFN output. */
static long FFNCHK_N = 0, FFNCHK_XQ = 0, FFNCHK_FO = 0;
static void ffn_check_report(void) {
    fprintf(stderr, "FFNCHK layers=%ld xq_blocks_bad=%ld fo_bad=%ld\n", FFNCHK_N, FFNCHK_XQ, FFNCHK_FO);
}
static int ffn_check_on(void) {
    static int on = -1;
    if (on < 0) {
        on = getenv("PQ1_FFN_CHECK") && atoi(getenv("PQ1_FFN_CHECK")) && pq1_ffn_dbg_xq;
        if (on) atexit(ffn_check_report);
    }
    return on;
}
static void gemvP_multi(const uint8_t *const *W, const int *rows, int n, int K,
                        const float *x, float *const *Y);
static void gemvP(const uint8_t *W, int rows, int K, const float *x, float *y);
static void ffn_check(const uint8_t *const *Wg, const uint8_t *Wd) {
    static float FOF[HID];
    static uint8_t XQF[(INTER / 256) * 292];
    memcpy(FOF, FO, sizeof FOF);
    pq1_ffn_dbg_xq(XQF, sizeof XQF);
    const int Rg[2] = {INTER, INTER};
    float *Yg[2] = {LG, LU};
    gemvP_multi(Wg, Rg, 2, HID, XN, Yg);
    pq1_silu_mul(LU, LG, INTER);
    gemvP(Wd, HID, INTER, LU, FO);
    for (int b = 0; b < INTER / 256; b++)
        if (memcmp(XQF + (size_t)b * 292, XQ + (size_t)b * 292, 260)) FFNCHK_XQ++;
    for (int i = 0; i < HID; i++)
        if (memcmp(&FOF[i], &FO[i], 4)) FFNCHK_FO++;
    FFNCHK_N++;
}
#endif
static float HEAD[VOCAB];
/* K cache TRANSPOSED (run #74): [layer][kv head][dim][position], so that the
 * q.k scores of 4/8/16 consecutive positions are one NEON fma chain per lane
 * (lane = position) with the per-position arithmetic unchanged: lane t of
 * vfmaq_f32(acc, dup(q[i]), K^T[i][t..]) is exactly fmaf(q[i], K[t][i], acc)
 * in the same i order as the serial reference. Same 75 MB as the row layout. */
#define KTB 16                       /* positions per contiguous K^T tile */
#ifdef PQ1_BUILD
/* f16 K/V rows, [layer][kv head][position][dim] (the fork's cache type and the
 * device layout), position stride KVCTX. Allocated on first use by the host
 * attention (CPU fallback, PQ1_ATT_CHECK): the fused GPU path never touches
 * them, and a static [CTX] array would reserve 2.4 GB of BSS at CTX 16384. */
static uint16_t *K16b, *V16b;
static int KVCTX = CTX;   /* device cache positions for this run */
#define K16(il, kh, t) (K16b + (((size_t)(il) * NKV + (kh)) * KVCTX + (t)) * HDIM)
#define V16(il, kh, t) (V16b + (((size_t)(il) * NKV + (kh)) * KVCTX + (t)) * HDIM)
void pq1_f2h_row(const float *x, uint16_t *h, int n);
void pq1_fa16_head(const float *q, const uint16_t *K, const uint16_t *V, int T, float scale,
                   float *out);
#else
static float KTC[36][NKV][CTX / KTB][HDIM][KTB];   /* [layer][kh][tile][i][t%16] */
/* V cache kv-head major (#74): a head's p.v walks a contiguous 512 B/position
 * slab (22 KB at 44 positions) that stays in L1 across its 4 passes and the 4
 * query heads sharing the kv head; the row layout strode 4 KB (L1 set alias). */
static float VCA[36][NKV][CTX][HDIM];
#endif
static int NPOS;

static void embed(int id) {
    uint8_t *W = tw("token_embd.weight");
    int ng = HID / 128;
#ifdef PQ1_BUILD
    /* Real Q1_0 (1-bit) checkpoint: 18-byte blocks = fp16 scale + 128
     * packed bits, bit=1 -> +1. Selected by the shim for the 1-bit build;
     * the 2-bit path below is unchanged. */
    const uint8_t *row = W + (size_t)id * ng * 18;
    for (int g = 0; g < ng; g++) {
        const uint8_t *b = row + g * 18;
        float sc = h2f((uint16_t)(b[0] | (b[1] << 8)));
        const uint8_t *q = b + 2;
        for (int j = 0; j < 128; j++)
            X[g * 128 + j] = ((q[j >> 3] >> (j & 7)) & 1) ? sc : -sc;
    }
#else
    const uint8_t *row = W + (size_t)id * ng * 34;
    for (int g = 0; g < ng; g++) {
        const uint8_t *b = row + g * 34;
        float sc = h2f((uint16_t)(b[0] | (b[1] << 8)));
        const uint8_t *q = b + 2;
        for (int j = 0; j < 128; j++)
            X[g * 128 + j] = (float)(((q[j / 4] >> ((j % 4) * 2)) & 3) - 1) * sc;
    }
#endif
}

static int argmax(const float *v, int n) {
    int bi = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[bi]) bi = i;
    return bi;
}

/* sampler (same as fwd.c, VOCAB-sized) */
static uint64_t RS;
static float frand01(void) {
    RS = RS * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((RS >> 11) * (1.0 / 9007199254740992.0));
}
/* sampler, fork order: top-k -> top-p -> min-p on RAW logits, temp scale
   last, then softmax draw (llama.cpp default chain). */
static int sample_topkpp(const float *lg, int n, float temp, float topp, int topk, float minp) {
    static float sc[1024];
    static int si[1024];
    if (temp <= 0.0f) return argmax(lg, n);
    if (topk <= 0 || topk > 1024) topk = 1024;
    if (topk > n) topk = n;
    int m = 0;
    for (int i = 0; i < n; i++) {
        float v = lg[i];
        /* full list and v <= its last (smallest) entry: the walk below would
         * pass every slot and reject v; skip it (NaN fails <= and walks) */
        if (m == topk && v <= sc[topk - 1]) continue;
        int k = 0;
        while (k < m && v <= sc[k]) k++;
        if (k >= topk) continue;
        for (int j = (m < topk ? m : topk - 1); j > k; j--) { sc[j] = sc[j-1]; si[j] = si[j-1]; }
        sc[k] = v; si[k] = i;
        if (m < topk) m++;
    }
    /* min-p on raw, then top-p on raw */
    float mx = sc[0];
    if (minp > 0.0f) {
        float thr = mx + logf(minp);
        while (m > 1 && sc[m-1] < thr) m--;
    }
    double es = 0;
    for (int k = 0; k < m; k++) es += expf(sc[k] - mx);
    double acc = 0;
    int keep = m;
    for (int k = 0; k < m; k++) {
        acc += expf(sc[k] - mx) / es;
        if (acc >= topp) { keep = k + 1; break; }
    }
    /* temp scale last, softmax, draw */
    double es2 = 0;
    for (int k = 0; k < keep; k++) es2 += expf((sc[k] - mx) / temp);
    double r = frand01() * es2, a2 = 0;
    for (int k = 0; k < keep; k++) {
        a2 += expf((sc[k] - mx) / temp);
        if (r < a2 || k == keep - 1) return si[k];
    }
    return si[keep-1];
}
static float STEMP = 0, STOPP = 1, SMINP = 0;
static int STopK = 0;

/* The attention block of layer il on the host: q/k/v GEMV, head norms, rope,
 * K/V cache store, attention, attn_output GEMV -> AO. put = also mirror the new
 * K/V rows into the device cache (loop iter 4; 0 under PQ1_ATT_CHECK, where the
 * device path has already written them). */
static void attn_cpu(int il, int put) {
    /* attention */
    { /* one launch for q+k+v: same K=4096 activation (XN) */
        const uint8_t *Wq[3] = {TW(il, "attn_q.weight"),
                                TW(il, "attn_k.weight"),
                                TW(il, "attn_v.weight")};
        const int Rq[3] = {NHEAD * HDIM, NKV * HDIM, NKV * HDIM};
        float *Yq[3] = {QF, KF, VF};
        gemvP_multi(Wq, Rq, 3, HID, XN, Yq);
    }
    float *qnw = (float *)TW(il, "attn_q_norm.weight");
    float *knw = (float *)TW(il, "attn_k_norm.weight");
    for (int h = 0; h < NHEAD; h++) rms(QF + h * 128, qnw, 128, QF + h * 128);
    for (int h = 0; h < NKV; h++) rms(KF + h * 128, knw, 128, KF + h * 128);
    ct_mark(CT_HNORM);
    if (TRACE) { V8("qnorm", QF); V8("knorm", KF); V8("vcur", VF); }
    rope_yarn_apply(QF, NHEAD, NPOS);
    rope_yarn_apply(KF, NKV, NPOS);
    ct_mark(CT_ROPE);
#ifdef PQ1_BUILD
    {
        static uint16_t kr[NKV * HDIM], vr[NKV * HDIM];
        if (!K16b) {
            K16b = calloc((size_t)36 * NKV * KVCTX * HDIM, 2);
            V16b = calloc((size_t)36 * NKV * KVCTX * HDIM, 2);
            if (!K16b || !V16b) { fprintf(stderr, "host K/V cache: out of memory\n"); exit(1); }
        }
        pq1_f2h_row(KF, kr, NKV * HDIM);
        pq1_f2h_row(VF, vr, NKV * HDIM);
        for (int kh = 0; kh < NKV; kh++) {
            memcpy(K16(il, kh, NPOS), kr + kh * HDIM, HDIM * 2);
            memcpy(V16(il, kh, NPOS), vr + kh * HDIM, HDIM * 2);
        }
        /* loop iter 4: keep the device K/V cache complete for the fused path */
        if (put && pq1_attn_kv_put) pq1_attn_kv_put(il, NLAYER, NKV, KVCTX, NPOS, kr, vr);
    }
    ct_mark(CT_KVCPY);
    for (int h = 0; h < NHEAD; h++)
        pq1_fa16_head(QF + h * 128, K16(il, h / 4, 0), V16(il, h / 4, 0), NPOS + 1, 0.0883883476f,
                      ATTO + h * 128);
#else
    (void)put;
    for (int kh = 0; kh < NKV; kh++)
        for (int i = 0; i < HDIM; i++) KTC[il][kh][NPOS / KTB][i][NPOS % KTB] = KF[kh * 128 + i];
    for (int kh = 0; kh < NKV; kh++) memcpy(VCA[il][kh][NPOS], VF + kh * HDIM, HDIM * 4);
    ct_mark(CT_KVCPY);
    /* Same arithmetic as the serial reference (kept in
     * fwd8_cpu_parity.c), reorganised for the A57:
     *  - q.k: gcc fuses the reference's `s += q*k` into fmadd, so each
     *    score is one fused chain over i; eight positions' chains are
     *    interleaved (independent accumulators) to hide the 4-cycle
     *    fmadd latency. Per-chain order is unchanged.
     *  - softmax: expf(sc-mx) once per score, reused in the sum.
     *  - p.v: lane i accumulates o[i] = fma(p, v[i], o[i]) over t in
     *    the reference order; four lanes per NEON fma. */
    /* Serial on purpose: an OpenMP parallel-for over the 32 heads read
     * 11.6 ms/tok (passive) / 9.7 (active) vs 8.7 serial, and the active
     * spin cost the GPU side 5% on top (#74). */
    for (int h = 0; h < NHEAD; h++) {
        const int kh = h / 4;
        const float *q = QF + h * 128;
        float mx = -1e30f, sc[CTX], pr[CTX];
        const float32x4_t scl = vdupq_n_f32(0.0883883476f);
        int t;
        /* q.k over 16-position tiles (8 KB contiguous each): lane =
         * position, fma chain over i in the reference order (#74).
         * The last tile is computed in full; lanes past NPOS are
         * never read (static storage, always finite). */
        for (t = 0; t <= NPOS; t += KTB) {
            float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
            float32x4_t a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
            const float *kp = &KTC[il][kh][t / KTB][0][0];
            for (int i = 0; i < 128; i++, kp += KTB) {
                const float32x4_t qi = vdupq_n_f32(q[i]);
                a0 = vfmaq_f32(a0, qi, vld1q_f32(kp));
                a1 = vfmaq_f32(a1, qi, vld1q_f32(kp + 4));
                a2 = vfmaq_f32(a2, qi, vld1q_f32(kp + 8));
                a3 = vfmaq_f32(a3, qi, vld1q_f32(kp + 12));
            }
            vst1q_f32(sc + t, vmulq_f32(a0, scl));
            vst1q_f32(sc + t + 4, vmulq_f32(a1, scl));
            vst1q_f32(sc + t + 8, vmulq_f32(a2, scl));
            vst1q_f32(sc + t + 12, vmulq_f32(a3, scl));
        }
        for (t = 0; t <= NPOS; t++) if (sc[t] > mx) mx = sc[t];
        double es = 0;
        for (t = 0; t <= NPOS; t++) { sc[t] = expf(sc[t] - mx); es += sc[t]; }
        for (t = 0; t <= NPOS; t++) pr[t] = sc[t] / (float)es;
        /* p.v: o[i] = fma(p_t, v_t[i], o[i]) over t in order; 16 lane
         * groups kept in registers per pass, two passes over i (#74) */
        float *o = ATTO + h * 128;
        for (int ib = 0; ib < 128; ib += 32) {
            /* 8 named accumulators: gcc 7 -O2 keeps a float32x4_t[]
             * in memory (no unroll), which cost 8.8 ms/tok (#74) */
            float32x4_t c0 = vdupq_n_f32(0), c1 = c0, c2 = c0, c3 = c0;
            float32x4_t c4 = c0, c5 = c0, c6 = c0, c7 = c0;
            const float *Vt = &VCA[il][kh][0][ib];
            for (t = 0; t <= NPOS; t++, Vt += HDIM) {
                const float32x4_t p4 = vdupq_n_f32(pr[t]);
                c0 = vfmaq_f32(c0, vld1q_f32(Vt), p4);
                c1 = vfmaq_f32(c1, vld1q_f32(Vt + 4), p4);
                c2 = vfmaq_f32(c2, vld1q_f32(Vt + 8), p4);
                c3 = vfmaq_f32(c3, vld1q_f32(Vt + 12), p4);
                c4 = vfmaq_f32(c4, vld1q_f32(Vt + 16), p4);
                c5 = vfmaq_f32(c5, vld1q_f32(Vt + 20), p4);
                c6 = vfmaq_f32(c6, vld1q_f32(Vt + 24), p4);
                c7 = vfmaq_f32(c7, vld1q_f32(Vt + 28), p4);
            }
            vst1q_f32(o + ib, c0);      vst1q_f32(o + ib + 4, c1);
            vst1q_f32(o + ib + 8, c2);  vst1q_f32(o + ib + 12, c3);
            vst1q_f32(o + ib + 16, c4); vst1q_f32(o + ib + 20, c5);
            vst1q_f32(o + ib + 24, c6); vst1q_f32(o + ib + 28, c7);
        }
    }
#endif
    ct_mark(CT_ATTN);
    gemvP(TW(il, "attn_output.weight"), HID, HID, ATTO, AO);
    if (TRACE) { V8("attnout", AO); V8R("attnout", AO, 1000); }
}

#ifdef PQ1_BUILD
/* PQ1_ATT_CHECK=1: after every fused attention graph, rerun the host path and
 * compare, bit for bit, the device's K and V rows, attention output and AO. */
static long ATTCHK_N = 0, ATTCHK_K = 0, ATTCHK_V = 0, ATTCHK_Q = 0, ATTCHK_O = 0, ATTCHK_AO = 0;
static void att_check_report(void) {
    fprintf(stderr, "ATTCHK layers=%ld k_bad=%ld v_bad=%ld q_bad=%ld atto_bad=%ld ao_bad=%ld\n",
            ATTCHK_N, ATTCHK_K, ATTCHK_V, ATTCHK_Q, ATTCHK_O, ATTCHK_AO);
}
static int att_check_on(void) {
    static int on = -1;
    if (on < 0) {
        on = getenv("PQ1_ATT_CHECK") && atoi(getenv("PQ1_ATT_CHECK")) && pq1_attn_dbg;
        if (on) atexit(att_check_report);
    }
    return on;
}
static void att_check(int il) {
    static float AOF[HID], QD[NHEAD * HDIM], OD[NHEAD * HDIM];
    static uint16_t KD[NKV * HDIM], VD[NKV * HDIM];
    memcpy(AOF, AO, sizeof AOF);
    pq1_attn_dbg(il, NPOS, KD, VD, QD, OD, NHEAD);
    attn_cpu(il, 0);
    for (int i = 0; i < NKV * HDIM; i++) {
        if (KD[i] != K16(il, i / HDIM, NPOS)[i % HDIM]) ATTCHK_K++;
        if (VD[i] != V16(il, i / HDIM, NPOS)[i % HDIM]) ATTCHK_V++;
    }
    for (int i = 0; i < NHEAD * HDIM; i++) {
        if (memcmp(&QD[i], &QF[i], 4)) ATTCHK_Q++;
        if (memcmp(&OD[i], &ATTO[i], 4)) ATTCHK_O++;
    }
    for (int i = 0; i < HID; i++)
        if (memcmp(&AOF[i], &AO[i], 4)) ATTCHK_AO++;
    ATTCHK_N++;
}
#endif

int main(int argc, char **argv) {
    gguf_table(argv[1]);
    XQ = malloc(((size_t)INTER / 256) * 292);
    static int ids[CTX + 8];
    int nids = 0, gen = 0;
    /* run context: 4096 positions unless PQ1_CTX asks for more (up to the
     * compile-time CTX; beyond 4096 the device cache must be TurboQuant) */
    int ctxcap = CTX;
#ifdef PQ1_BUILD
    ctxcap = getenv("PQ1_CTX") ? atoi(getenv("PQ1_CTX")) : 4096;
    if (ctxcap < 1) ctxcap = 4096;
    ctxcap = (ctxcap + 63) / 64 * 64;   /* the device cache is 64-aligned */
    if (ctxcap > CTX) ctxcap = CTX;
#endif
    for (int a = 2; a < argc; a++) {
        if (!strcmp(argv[a], "--gen")) gen = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--trace")) TRACE = 1;
        else if (!strcmp(argv[a], "--sample")) {
            STEMP = atof(argv[++a]); STOPP = atof(argv[++a]);
            STopK = atoi(argv[++a]); RS = strtoull(argv[++a], 0, 10);
            if (a + 1 < argc && argv[a+1][0] != '-') SMINP = atof(argv[++a]);
        }
        else if (nids < ctxcap) ids[nids++] = atoi(argv[a]);
        else { fprintf(stderr, "prompt longer than the %d-position context\n", ctxcap); return 2; }
    }
    int n0 = nids;
    if (n0 + gen > ctxcap) {
        fprintf(stderr, "context %d: generation capped at %d tokens\n", ctxcap, ctxcap - n0);
        gen = ctxcap - n0;
    }
#ifdef PQ1_BUILD
    /* the device K/V cache holds this run's positions only (64-aligned): the
     * arena grows by the excess over the old 75.5 MB f32 ctx-512 cache only */
    KVCTX = (n0 + gen + 63) / 64 * 64;
    if (KVCTX > ctxcap) KVCTX = ctxcap;
    if (pq1_kv_shape) pq1_kv_shape(NLAYER, NKV, KVCTX);
#endif
    NPOS = 0;
#ifdef PQ1_BUILD
    if (pq1_head_need) pq1_head_need(STEMP > 0 || getenv("TOP40") != NULL);
#endif
    int top = -1;   /* the previous step's greedy argmax of HEAD */
    for (int step = 0; step < n0 + gen; step++) {
        BQ1_T0();
        ct_begin(step, n0);
        int id = step < n0 ? ids[step]
               : (STEMP > 0 ? sample_topkpp(HEAD, VOCAB, STEMP, STOPP, STopK, SMINP)
                            : top >= 0 ? top : argmax(HEAD, VOCAB));
        if (step >= n0) ids[nids++] = id;
        ct_mark(CT_SAMPLE);
        embed(id);
        ct_mark(CT_EMBED);
        if (TRACE) { MAG("emb", X, HID); V8("emb", X); }
        int tfused = 0;
#ifdef PQ1_BUILD
        /* loop iter 14: every layer, output_norm and the head as ONE device
         * graph (residual adds, rms norms and Q8_K included, X kept on the
         * device); declines on the first token (merged tensors and the device
         * K/V cache are built by it), under TRACE and the per-layer checks */
        if (!TRACE && pq1_token_fused && !att_check_on() && !ffn_check_on()) {
            static const unsigned char **WT;
            static const float **NW;
            if (!WT) {
                static const char *const wk[7] = {"attn_q.weight", "attn_k.weight", "attn_v.weight",
                                                  "attn_output.weight", "ffn_gate.weight",
                                                  "ffn_up.weight", "ffn_down.weight"};
                static const char *const nk[4] = {"attn_norm.weight", "ffn_norm.weight",
                                                  "attn_q_norm.weight", "attn_k_norm.weight"};
                WT = malloc((7 * (size_t)NLAYER + 1) * sizeof *WT);
                NW = malloc((4 * (size_t)NLAYER + 1) * sizeof *NW);
                for (int il = 0; il < NLAYER; il++) {
                    for (int j = 0; j < 7; j++) WT[7 * il + j] = TW(il, wk[j]);
                    for (int j = 0; j < 4; j++) NW[4 * il + j] = (const float *)TW(il, nk[j]);
                }
                WT[7 * NLAYER] = tw("output.weight");
                NW[4 * NLAYER] = (const float *)tw("output_norm.weight");
            }
            if (RC_POS != NPOS) rope_table(NPOS);
            tfused = pq1_token_fused(NLAYER, WT, NW, NHEAD, NKV, HID, INTER, VOCAB, KVCTX, NPOS,
                                     RC_C, RC_S, X, HEAD) == 0;
            if (tfused) ct_mark(CT_GEMV);
        }
#endif
        for (int il = 0; il < NLAYER && !tfused; il++) {
            float *anw = (float *)TW(il, "attn_norm.weight");
            rms(X, anw, HID, XN);
            ct_mark(CT_RMS);
            int afused = 0;
#ifdef PQ1_BUILD
            /* loop iter 4: q+k+v, head norms, rope, K/V store, attention, Q8_K
             * and attn_output as one device graph; declines on the first token
             * (merged q+k+v not built yet) and under TRACE */
            if (!TRACE && pq1_attn_fused) {
                if (RC_POS != NPOS) rope_table(NPOS);
                q8_K_quant(XN, XQ, HID);
                ct_mark(CT_QUANT);
                afused = pq1_attn_fused(il, TW(il, "attn_q.weight"), TW(il, "attn_k.weight"),
                                        TW(il, "attn_v.weight"), TW(il, "attn_output.weight"),
                                        (float *)TW(il, "attn_q_norm.weight"),
                                        (float *)TW(il, "attn_k_norm.weight"), NHEAD, NKV, HID,
                                        KVCTX, NPOS, RC_C, RC_S, XQ, AO) == 0;
                if (afused) { ct_mark(CT_GEMV); if (att_check_on()) att_check(il); }
            }
#endif
            if (!afused) attn_cpu(il, 1);
            for (int i = 0; i < HID; i++) X[i] += AO[i];
            ct_mark(CT_RESID);
            /* FFN (8B names its second norm ffn_norm, not post_attention_norm) */
            float *pnw = (float *)TW(il, "ffn_norm.weight");
            rms(X, pnw, HID, XN);
            ct_mark(CT_RMS);
            {
                const uint8_t *Wg[2] = {TW(il, "ffn_gate.weight"),
                                        TW(il, "ffn_up.weight")};
                const uint8_t *Wd = TW(il, "ffn_down.weight");
                int fused = 0;
#ifdef PQ1_BUILD
                /* loop iter 2: the whole tail (gate+up, silu*u, Q8_K of the
                 * product, down) as one device graph when the glue has it;
                 * declines on the first token (tensors not merged yet) and
                 * under TRACE (LG/LU stay on the device). */
                if (!TRACE && pq1_ffn_fused) {
                    q8_K_quant(XN, XQ, HID);
                    ct_mark(CT_QUANT);
                    fused = pq1_ffn_fused(Wg[0], Wg[1], Wd, INTER, HID, XQ, FO) == 0;
                    if (fused) { ct_mark(CT_GEMV); if (ffn_check_on()) ffn_check(Wg, Wd); }
                }
#endif
                if (!fused) {
                    /* one launch for gate+up: same K=4096 activation (XN) */
                    const int Rg[2] = {INTER, INTER};
                    float *Yg[2] = {LG, LU};
                    gemvP_multi(Wg, Rg, 2, HID, XN, Yg);
#ifdef PQ1_BUILD
                    /* silu(g)*u four lanes at a time (pq1_host.c, built with
                     * -ffp-contract=off): expf via the glibc-2.27-exact NEON port in
                     * pq1_expf_neon.h, then 1/(1+e), *g, *u in the reference order. */
                    pq1_silu_mul(LU, LG, INTER);
#else
                    for (int i = 0; i < INTER; i++) LU[i] = silu(LG[i]) * LU[i];
#endif
                    ct_mark(CT_SILU);
                    gemvP(Wd, HID, INTER, LU, FO);
                }
            }
            if (TRACE) { V8("ffnout", FO); V8R("gate", LG, 1000); V8R("up", LU, 1000); V8R("ffnout", FO, 1000); }
            for (int i = 0; i < HID; i++) X[i] += FO[i];
            ct_mark(CT_RESID);
        }
        if (!tfused) {
            float *onw = (float *)tw("output_norm.weight");
            rms(X, onw, HID, XN);
            ct_mark(CT_RMS);
            gemvP(tw("output.weight"), VOCAB, HID, XN, HEAD);
        }
        float topv;
        top = -1;
#ifdef PQ1_BUILD
        if (tfused && pq1_head_top) top = pq1_head_top(&topv);
#endif
        if (top < 0) { top = argmax(HEAD, VOCAB); topv = HEAD[top]; }
        ct_mark(CT_ARGMAX);
        printf("step %d id_in=%d top=%d logit=%.4f\n", step, id, top, topv);
        BQ1_T1(step);
        if (getenv("TOP40") && step == 22) {
            /* print top-40 (id:prob) under the configured sampler for comparison
               with server n_probs at the same position */
            float t = STEMP > 0 ? STEMP : 1.0f;
            int tk = STopK > 0 ? STopK : 40;
            if (tk > 40) tk = 40;
            static float s2[1024]; static int ix[1024];
            int m2 = 0;
            for (int i = 0; i < VOCAB; i++) {
                float v = HEAD[i] / t;
                int k = 0;
                while (k < m2 && v <= s2[k]) k++;
                if (k >= tk) continue;
                for (int j = (m2 < tk ? m2 : tk - 1); j > k; j--) { s2[j] = s2[j-1]; ix[j] = ix[j-1]; }
                s2[k] = v; ix[k] = i;
                if (m2 < tk) m2++;
            }
            double e2 = 0;
            for (int k = 0; k < m2; k++) e2 += expf(s2[k] - s2[0]);
            for (int k = 0; k < 40 && k < m2; k++)
                printf("  P %d %.6f\n", ix[k], expf(s2[k] - s2[0]) / e2);
        }
        fflush(stdout);
        if (step >= n0 && id == 151645) break; /* EOS */
        NPOS++;
    }
    ct_dump();
    return 0;
}
