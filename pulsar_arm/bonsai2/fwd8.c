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

void quantize_row_q8_K(const float * x, void * y, int64_t k);
void pq2_gemv(int rows, int cols, const uint8_t *w, const float *x, float *y,
              const uint8_t *xq);

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
    quantize_row_q8_K(x, XQ, K);
    pq2_gemv(rows, K, W, x, y, XQ);
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
static void rope_yarn_apply(float *x, int heads, int pos) {
    if (!CORR_INIT) yarn_init();
    const float fs = 1.0f / YARN_FACTOR, msc = 1.0f + 0.1f * logf(YARN_FACTOR);
    const float tscale = powf(FREQ_BASE, -2.0f / 128.0f);
    for (int h = 0; h < heads; h++) {
        float *xh = x + h * 128;
        float theta = (float)pos;
        for (int j = 0; j < 64; j++, theta *= tscale) {
            float y = (j - CORR[0]) / fmaxf(0.001f, CORR[1] - CORR[0]);
            float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
            float ti = fs * theta, te = theta;
            float th = ti * (1 - ramp) + te * ramp;
            float c = cosf(th) * msc, s = sinf(th) * msc;
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
    const uint8_t *row = W + (size_t)id * ng * 34;
    for (int g = 0; g < ng; g++) {
        const uint8_t *b = row + g * 34;
        float sc = h2f((uint16_t)(b[0] | (b[1] << 8)));
        const uint8_t *q = b + 2;
        for (int j = 0; j < 128; j++)
            X[g * 128 + j] = (float)(((q[j / 4] >> ((j % 4) * 2)) & 3) - 1) * sc;
    }
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
        int id = step < n0 ? ids[step]
               : (STEMP > 0 ? sample_topkpp(HEAD, VOCAB, STEMP, STOPP, STopK, SMINP)
                            : argmax(HEAD, VOCAB));
        if (step >= n0) ids[nids++] = id;
        embed(id);
        if (TRACE) { MAG("emb", X, HID); V8("emb", X); }
        for (int il = 0; il < NLAYER; il++) {
            float *anw = (float *)TW(il, "attn_norm.weight");
            rms(X, anw, HID, XN);
            /* attention */
            gemvP(TW(il, "attn_q.weight"), NHEAD * HDIM, HID, XN, QF);
            gemvP(TW(il, "attn_k.weight"), NKV * HDIM, HID, XN, KF);
            gemvP(TW(il, "attn_v.weight"), NKV * HDIM, HID, XN, VF);
            float *qnw = (float *)TW(il, "attn_q_norm.weight");
            float *knw = (float *)TW(il, "attn_k_norm.weight");
            for (int h = 0; h < NHEAD; h++) rms(QF + h * 128, qnw, 128, QF + h * 128);
            for (int h = 0; h < NKV; h++) rms(KF + h * 128, knw, 128, KF + h * 128);
            if (TRACE) { V8("qnorm", QF); V8("knorm", KF); V8("vcur", VF); }
            rope_yarn_apply(QF, NHEAD, NPOS);
            rope_yarn_apply(KF, NKV, NPOS);
            memcpy(KCA[il][NPOS], KF, NKV * HDIM * 4);
            memcpy(VCA[il][NPOS], VF, NKV * HDIM * 4);
            for (int h = 0; h < NHEAD; h++) {
                int kh = h / 4;
                float mx = -1e30f, sc[CTX];
                for (int t = 0; t <= NPOS; t++) {
                    float s = 0;
                    float *Kt = (float *)KCA[il][t] + kh * 128;
                    for (int i = 0; i < 128; i++) s += QF[h * 128 + i] * Kt[i];
                    sc[t] = s * 0.0883883476f;
                    if (sc[t] > mx) mx = sc[t];
                }
                double es = 0;
                for (int t = 0; t <= NPOS; t++) es += expf(sc[t] - mx);
                float *o = ATTO + h * 128;
                for (int i = 0; i < 128; i++) o[i] = 0;
                for (int t = 0; t <= NPOS; t++) {
                    float p = expf(sc[t] - mx) / (float)es;
                    float *Vt = (float *)VCA[il][t] + kh * 128;
                    for (int i = 0; i < 128; i++) o[i] += p * Vt[i];
                }
            }
            gemvP(TW(il, "attn_output.weight"), HID, HID, ATTO, AO);
            if (TRACE) { V8("attnout", AO); V8R("attnout", AO, 1000); }
            for (int i = 0; i < HID; i++) X[i] += AO[i];
            /* FFN (8B names its second norm ffn_norm, not post_attention_norm) */
            float *pnw = (float *)TW(il, "ffn_norm.weight");
            rms(X, pnw, HID, XN);
            gemvP(TW(il, "ffn_gate.weight"), INTER, HID, XN, LG);
            gemvP(TW(il, "ffn_up.weight"), INTER, HID, XN, LU);
            for (int i = 0; i < INTER; i++) LU[i] = silu(LG[i]) * LU[i];
            gemvP(TW(il, "ffn_down.weight"), HID, INTER, LU, FO);
            if (TRACE) { V8("ffnout", FO); V8R("gate", LG, 1000); V8R("up", LU, 1000); V8R("ffnout", FO, 1000); }
            for (int i = 0; i < HID; i++) X[i] += FO[i];
        }
        float *onw = (float *)tw("output_norm.weight");
        rms(X, onw, HID, XN);
        gemvP(tw("output.weight"), VOCAB, HID, XN, HEAD);
        int top = argmax(HEAD, VOCAB);
        printf("step %d id_in=%d top=%d logit=%.4f\n", step, id, top, HEAD[top]);
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
    return 0;
}
