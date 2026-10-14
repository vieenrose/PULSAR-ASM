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
#define CTX 512
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
        float max = 0;
        for (int j = 0; j < 256; ++j) if (fabsf(x[j]) == amax) { max = x[j]; break; }
        const float iscale = -127.f / max;
        /* nearest_int(iscale*x) per lane: one IEEE multiply, the +2^23*1.5
           magic add (FPCR round-to-nearest-even, as the scalar FPU), the
           mantissa bit tricks on the integer image, then MIN(127, v). */
        const float32x4_t is4 = vdupq_n_f32(iscale), magic = vdupq_n_f32(12582912.f);
        const int32x4_t m23 = vdupq_n_s32(0x007fffff), b22 = vdupq_n_s32(0x00400000);
        const int32x4_t c127 = vdupq_n_s32(127);
#define Q8K_LANE4(off) vminq_s32(vsubq_s32(vandq_s32(vreinterpretq_s32_f32( \
            vaddq_f32(vmulq_f32(vld1q_f32(x + j + (off)), is4), magic)), m23), b22), c127)
        for (int j = 0; j < 256; j += 16) {
            int16x8_t h01 = vcombine_s16(vmovn_s32(Q8K_LANE4(0)), vmovn_s32(Q8K_LANE4(4)));
            int16x8_t h23 = vcombine_s16(vmovn_s32(Q8K_LANE4(8)), vmovn_s32(Q8K_LANE4(12)));
            int8x16_t q = vcombine_s8(vmovn_s16(h01), vmovn_s16(h23));
            vst1q_s8(qs + j, q);
            bsums[j >> 4] = vaddlvq_s8(q);   /* exact: |sum| <= 16*127 */
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
static float HEAD[VOCAB];
static float KCA[36][512][8 * 128];
static float VCA[36][512][8 * 128];
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

int main(int argc, char **argv) {
    gguf_table(argv[1]);
    XQ = malloc(((size_t)INTER / 256) * 292);
    int ids[1200], nids = 0, gen = 0;
    for (int a = 2; a < argc; a++) {
        if (!strcmp(argv[a], "--gen")) gen = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--trace")) TRACE = 1;
        else if (!strcmp(argv[a], "--sample")) {
            STEMP = atof(argv[++a]); STOPP = atof(argv[++a]);
            STopK = atoi(argv[++a]); RS = strtoull(argv[++a], 0, 10);
            if (a + 1 < argc && argv[a+1][0] != '-') SMINP = atof(argv[++a]);
        }
        else ids[nids++] = atoi(argv[a]);
    }
    int n0 = nids;
    NPOS = 0;
    for (int step = 0; step < n0 + gen; step++) {
        BQ1_T0();
        ct_begin(step, n0);
        int id = step < n0 ? ids[step]
               : (STEMP > 0 ? sample_topkpp(HEAD, VOCAB, STEMP, STOPP, STopK, SMINP)
                            : argmax(HEAD, VOCAB));
        if (step >= n0) ids[nids++] = id;
        ct_mark(CT_SAMPLE);
        embed(id);
        ct_mark(CT_EMBED);
        if (TRACE) { MAG("emb", X, HID); V8("emb", X); }
        for (int il = 0; il < NLAYER; il++) {
            float *anw = (float *)TW(il, "attn_norm.weight");
            rms(X, anw, HID, XN);
            ct_mark(CT_RMS);
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
            memcpy(KCA[il][NPOS], KF, NKV * HDIM * 4);
            memcpy(VCA[il][NPOS], VF, NKV * HDIM * 4);
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
            for (int h = 0; h < NHEAD; h++) {
                const int kh = h / 4;
                const float *q = QF + h * 128;
                float mx = -1e30f, sc[CTX];
                int t = 0;
                for (; t + 8 <= NPOS + 1; t += 8) {
                    const float *K0 = (float *)KCA[il][t + 0] + kh * 128;
                    const float *K1 = (float *)KCA[il][t + 1] + kh * 128;
                    const float *K2 = (float *)KCA[il][t + 2] + kh * 128;
                    const float *K3 = (float *)KCA[il][t + 3] + kh * 128;
                    const float *K4 = (float *)KCA[il][t + 4] + kh * 128;
                    const float *K5 = (float *)KCA[il][t + 5] + kh * 128;
                    const float *K6 = (float *)KCA[il][t + 6] + kh * 128;
                    const float *K7 = (float *)KCA[il][t + 7] + kh * 128;
                    float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
                    for (int i = 0; i < 128; i++) {
                        const float qi = q[i];
                        s0 = fmaf(qi, K0[i], s0); s1 = fmaf(qi, K1[i], s1);
                        s2 = fmaf(qi, K2[i], s2); s3 = fmaf(qi, K3[i], s3);
                        s4 = fmaf(qi, K4[i], s4); s5 = fmaf(qi, K5[i], s5);
                        s6 = fmaf(qi, K6[i], s6); s7 = fmaf(qi, K7[i], s7);
                    }
                    sc[t + 0] = s0 * 0.0883883476f; sc[t + 1] = s1 * 0.0883883476f;
                    sc[t + 2] = s2 * 0.0883883476f; sc[t + 3] = s3 * 0.0883883476f;
                    sc[t + 4] = s4 * 0.0883883476f; sc[t + 5] = s5 * 0.0883883476f;
                    sc[t + 6] = s6 * 0.0883883476f; sc[t + 7] = s7 * 0.0883883476f;
                }
                for (; t <= NPOS; t++) {
                    const float *Kt = (float *)KCA[il][t] + kh * 128;
                    float s = 0;
                    for (int i = 0; i < 128; i++) s = fmaf(q[i], Kt[i], s);
                    sc[t] = s * 0.0883883476f;
                }
                for (t = 0; t <= NPOS; t++) if (sc[t] > mx) mx = sc[t];
                double es = 0;
                for (t = 0; t <= NPOS; t++) { sc[t] = expf(sc[t] - mx); es += sc[t]; }
                float *o = ATTO + h * 128;
                for (int i = 0; i < 128; i += 4) vst1q_f32(o + i, vdupq_n_f32(0));
                for (t = 0; t <= NPOS; t++) {
                    const float32x4_t p4 = vdupq_n_f32(sc[t] / (float)es);
                    const float *Vt = (float *)VCA[il][t] + kh * 128;
                    for (int i = 0; i < 128; i += 4)
                        vst1q_f32(o + i, vfmaq_f32(vld1q_f32(o + i), vld1q_f32(Vt + i), p4));
                }
            }
            ct_mark(CT_ATTN);
            gemvP(TW(il, "attn_output.weight"), HID, HID, ATTO, AO);
            if (TRACE) { V8("attnout", AO); V8R("attnout", AO, 1000); }
            for (int i = 0; i < HID; i++) X[i] += AO[i];
            ct_mark(CT_RESID);
            /* FFN (8B names its second norm ffn_norm, not post_attention_norm) */
            float *pnw = (float *)TW(il, "ffn_norm.weight");
            rms(X, pnw, HID, XN);
            ct_mark(CT_RMS);
            { /* one launch for gate+up: same K=4096 activation (XN) */
                const uint8_t *Wg[2] = {TW(il, "ffn_gate.weight"),
                                        TW(il, "ffn_up.weight")};
                const int Rg[2] = {INTER, INTER};
                float *Yg[2] = {LG, LU};
                gemvP_multi(Wg, Rg, 2, HID, XN, Yg);
            }
            for (int i = 0; i < INTER; i++) LU[i] = silu(LG[i]) * LU[i];
            ct_mark(CT_SILU);
            gemvP(TW(il, "ffn_down.weight"), HID, INTER, LU, FO);
            if (TRACE) { V8("ffnout", FO); V8R("gate", LG, 1000); V8R("up", LU, 1000); V8R("ffnout", FO, 1000); }
            for (int i = 0; i < HID; i++) X[i] += FO[i];
            ct_mark(CT_RESID);
        }
        float *onw = (float *)tw("output_norm.weight");
        rms(X, onw, HID, XN);
        ct_mark(CT_RMS);
        gemvP(tw("output.weight"), VOCAB, HID, XN, HEAD);
        int top = argmax(HEAD, VOCAB);
        ct_mark(CT_ARGMAX);
        printf("step %d id_in=%d top=%d logit=%.4f\n", step, id, top, HEAD[top]);
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
