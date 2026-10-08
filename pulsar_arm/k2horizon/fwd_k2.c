/* fwd_k2.c: IFM K2-Horizon-0.9B single-token forward (dense decoder).
 * 28 layers, HID 1536, INTER 5120, 32 heads x 64 (8 kv, GQA group 4),
 * vocab 64256, untied head. Plain RMS norms (weight *, NO fold), SwiGLU
 * MLP, NeoX rope from baked fp32 tables, YaRN baked at convert time.
 * Stage 1: fp32-reference GEMV (fp16 weights converted scalar) for oracle
 * validation; kernelize (Q8/integer) after the forward is proven.
 * Blob: K2H1 (see k2_convert.py): canonical order per layer
 *   in_norm(F32,1536), q(2048x1536), k(512x1536), v(512x1536),
 *   o(1536x2048), post_norm(F32,1536), gate(5120x1536), up(5120x1536),
 *   down(1536x5120); then embed(64256x1536), final norm, head(64256x1536);
 *   then cos/sin tables (npos,32) fp32.
 * Build (Spark, validation): gcc -O2 -march=armv8.2-a+fp16 -o fwd_k2
 *          fwd_k2.c -lm
 * Run: ./fwd_k2 <blob> <ids...> --gen G [--sample temp topp topk seed]
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

#define HID 1536
#define INTER 5120
#define NHEAD 32
#define HDIM 64
#define NKV 8
#define VOCAB 64256
#define CTX 512

static int NL = 28;   /* overwritten by blob header, assert-equal */
static int WTYPE = 0;     /* 0 = fp16 weights, 2 = Q4_0 blocks */
static int HEAD_KIND = 0;
static uint8_t *G;
/* per-layer tensor descriptors: byte offset, rows, cols, kind
   (0 = fp16, 1 = fp32 norm, 2 = Q4_0). Row stride derives from kind. */
static size_t LOFF[28][9];
static int LROWS[28][9], LCOLS[28][9], LKIND[28][9];
static size_t O_EMB, O_NORM, O_HEAD, O_COS, O_SIN;
static int NPOS;

static void blob_load(const char *path) {
    int fd = open(path, O_RDONLY);
    struct stat st; fstat(fd, &st);
    G = mmap(0, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (memcmp(G, "K2H1", 4)) { printf("not k2h1\n"); exit(1); }
    uint32_t *h = (uint32_t *)(G + 4);
    NL = (int)h[0];
    if (NL != 28) { printf("layer count %d != 28 (static caches)\n", NL); exit(1); }
    if (h[1] != HID || h[2] != INTER || h[3] != NHEAD || h[4] != NKV ||
        h[5] != HDIM || h[6] != VOCAB) {
        printf("geometry mismatch\n"); exit(1);
    }
    int npos = (int)h[8];
    WTYPE = (int)h[7];
    size_t off = 40;
    size_t showr[9] = {HID, NHEAD * HDIM, NKV * HDIM, NKV * HDIM, HID,
                       HID, INTER, INTER, HID};
    size_t showc[9] = {0, HID, HID, HID, NHEAD * HDIM, 0, HID, HID, INTER};
    for (int il = 0; il < NL; il++) {
        for (int k = 0; k < 9; k++) {
            int is_norm = (k == 0 || k == 5);
            int kind = is_norm ? 1 : (WTYPE == 2 ? 2 : 0);
            size_t n = showr[k] * (is_norm ? 1 : showc[k]);
            size_t stride;
            if (kind == 1) stride = n * 4;
            else if (kind == 2) stride = (n / 32) * 18;
            else stride = n * 2;
            LOFF[il][k] = off;
            LROWS[il][k] = (int)showr[k];
            LCOLS[il][k] = is_norm ? 0 : (int)showc[k];
            LKIND[il][k] = kind;
            off += stride;
        }
    }
    O_EMB = off; off += (size_t)VOCAB * HID * 2;   /* embed always fp16 */
    O_NORM = off; off += HID * 4;
    O_HEAD = off;
    HEAD_KIND = (WTYPE == 2) ? 2 : 0;
    off += (WTYPE == 2) ? ((size_t)VOCAB * HID / 32) * 18
                        : (size_t)VOCAB * HID * 2;
    O_COS = off; off += (size_t)npos * 32 * 4;
    O_SIN = off;
    NPOS = npos;
}

/* fp16 bit -> fp32 scalar (validation path; vectorize later) */
static float h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu, o;
    if (e == 0) {
        if (!m) return 0.0f;
        float f = (float)m * 5.9604645e-8f;
        memcpy(&o, &f, 4); o |= s;
    } else if (e == 31) o = s | 0x7f800000u | (m << 13);
    else o = s | ((e + 112u) << 23) | (m << 13);
    float r; memcpy(&r, &o, 4); return r;
}

/* GEMV dispatch: fp16 reference (wtype 0, oracle validation) vs Q4 integer
   path (wtype 2, via Q8_K activations). XQQ sized for max cols (5120). */
void k2q4_gemv_range(int r0, int r1, int cols, const uint8_t *w,
                      const float *x, float *y, const uint8_t *xq);
void k2_q8_quant(const float *x, uint8_t *yv, int64_t k);
static void gemv_ref(const uint16_t *W, int rows, int cols, const float *x, float *y);
static uint8_t XQQ[(5120 / 256) * 292];
static void gemvW(int il, int k, const float *x, float *y) {
    const uint8_t *W = G + LOFF[il][k];
    int rows = LROWS[il][k], cols = LCOLS[il][k];
    if (LKIND[il][k] == 2) {
        k2_q8_quant(x, XQQ, cols);
        k2q4_gemv_range(0, rows, cols, W, x, y, XQQ);
    } else {
        gemv_ref((const uint16_t *)W, rows, cols, x, y);
    }
}
static void gemv_ref(const uint16_t *W, int rows, int cols, const float *x, float *y) {
    for (int r = 0; r < rows; r++) {
        double acc = 0;
        const uint16_t *row = W + (size_t)r * cols;
        for (int c = 0; c < cols; c++) acc += (double)h2f(row[c]) * x[c];
        y[r] = (float)acc;
    }
}

static void rms(const float *x, const float *w, int n, float *y) {
    double s = 0;
    for (int i = 0; i < n; i++) s += (double)x[i] * x[i];
    float sc = 1.0f / sqrtf((float)(s / n) + 1e-6f);
    for (int i = 0; i < n; i++) y[i] = x[i] * sc * w[i];
}
static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

static float X[HID], XN[HID], AO[HID], FO[HID];
static float QF[NHEAD * HDIM], KF[NKV * HDIM], VF[NKV * HDIM];
static float ATTO[NHEAD * HDIM], LG[INTER], LU[INTER];
static float HEAD[VOCAB];
static float KCA[28][512][8 * 64];
static float VCA[28][512][8 * 64];
static int NPOS_TOK;

static void embed(int id) {
    const uint16_t *W = (const uint16_t *)(G + O_EMB);
    const uint16_t *row = W + (size_t)id * HID;
    for (int i = 0; i < HID; i++) X[i] = h2f(row[i]);
}

static int argmax(const float *v, int n) {
    int bi = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[bi]) bi = i;
    return bi;
}

static uint64_t RS;
static int TRACE = 0;
#define TRACEL(il) do { if (TRACE) { double _s = 0; \
    for (int _i = 0; _i < HID; _i++) _s += (double)X[_i] * X[_i]; \
    fprintf(stderr, "L%d rms=%.6g v8:", il, sqrt(_s / HID)); \
    for (int _i = 0; _i < 8; _i++) fprintf(stderr, " %.6g", (double)X[_i]); \
    fprintf(stderr, "\n"); } } while (0)
static float frand01(void) {
    RS = RS * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((RS >> 11) * (1.0 / 9007199254740992.0));
}
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
    blob_load(argv[1]);
    const float *COS = (const float *)(G + O_COS);
    const float *SIN = (const float *)(G + O_SIN);
    int ids[1200], nids = 0, gen = 0;
    for (int a = 2; a < argc; a++) {
        if (!strcmp(argv[a], "--gen")) gen = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--sample")) {
            STEMP = atof(argv[++a]); STOPP = atof(argv[++a]);
            STopK = atoi(argv[++a]); RS = strtoull(argv[++a], 0, 10);
            if (a + 1 < argc && argv[a+1][0] != '-') SMINP = atof(argv[++a]);
        }
        else ids[nids++] = atoi(argv[a]);
    }
    int n0 = nids;
    NPOS_TOK = 0;
    if (getenv("K2TRACE")) TRACE = 1;
    for (int step = 0; step < n0 + gen; step++) {
        int id = step < n0 ? ids[step]
               : (STEMP > 0 ? sample_topkpp(HEAD, VOCAB, STEMP, STOPP, STopK, SMINP)
                            : argmax(HEAD, VOCAB));
        if (step >= n0) ids[nids++] = id;
        embed(id);
        for (int il = 0; il < NL; il++) {
            float *anw = (float *)(G + LOFF[il][0]);
            rms(X, anw, HID, XN);
            gemvW(il, 1, XN, QF);
            gemvW(il, 2, XN, KF);
            gemvW(il, 3, XN, VF);
            /* NeoX rope, pairs (j, j+32) of each 64-head */
            const float *C = COS + (size_t)NPOS_TOK * 32;
            const float *S = SIN + (size_t)NPOS_TOK * 32;
            for (int h = 0; h < NHEAD; h++) {
                float *xh = QF + h * HDIM;
                for (int j = 0; j < 32; j++) {
                    float a = xh[j], b = xh[j + 32];
                    xh[j] = a * C[j] - b * S[j];
                    xh[j + 32] = a * S[j] + b * C[j];
                }
            }
            for (int h = 0; h < NKV; h++) {
                float *xh = KF + h * HDIM;
                for (int j = 0; j < 32; j++) {
                    float a = xh[j], b = xh[j + 32];
                    xh[j] = a * C[j] - b * S[j];
                    xh[j + 32] = a * S[j] + b * C[j];
                }
            }
            memcpy(KCA[il][NPOS_TOK], KF, NKV * HDIM * 4);
            memcpy(VCA[il][NPOS_TOK], VF, NKV * HDIM * 4);
            for (int h = 0; h < NHEAD; h++) {
                int kh = h / 4;
                float mx = -1e30f, sc[CTX];
                for (int t = 0; t <= NPOS_TOK; t++) {
                    float s = 0;
                    float *Kt = (float *)KCA[il][t] + kh * HDIM;
                    for (int i = 0; i < HDIM; i++) s += QF[h * HDIM + i] * Kt[i];
                    sc[t] = s * 0.125f;
                    if (sc[t] > mx) mx = sc[t];
                }
                double es = 0;
                for (int t = 0; t <= NPOS_TOK; t++) es += expf(sc[t] - mx);
                float *o = ATTO + h * HDIM;
                for (int i = 0; i < HDIM; i++) o[i] = 0;
                for (int t = 0; t <= NPOS_TOK; t++) {
                    float p = expf(sc[t] - mx) / (float)es;
                    float *Vt = (float *)VCA[il][t] + kh * HDIM;
                    for (int i = 0; i < HDIM; i++) o[i] += p * Vt[i];
                }
            }
            gemvW(il, 4, ATTO, AO);
            for (int i = 0; i < HID; i++) X[i] += AO[i];
            float *pnw = (float *)(G + LOFF[il][5]);
            rms(X, pnw, HID, XN);
            gemvW(il, 6, XN, LG);
            gemvW(il, 7, XN, LU);
            for (int i = 0; i < INTER; i++) {
                float g = LG[i];
                LU[i] = (g / (1.0f + expf(-g))) * LU[i];
            }
            gemvW(il, 8, LU, FO);
            for (int i = 0; i < HID; i++) X[i] += FO[i];
            if (step == n0 - 1) TRACEL(il);
        }
        float *onw = (float *)(G + O_NORM);
        rms(X, onw, HID, XN);
        if (HEAD_KIND == 2) {
            k2_q8_quant(XN, XQQ, HID);
            k2q4_gemv_range(0, VOCAB, HID, G + O_HEAD, XN, HEAD, XQQ);
        } else {
            gemv_ref((const uint16_t *)(G + O_HEAD), VOCAB, HID, XN, HEAD);
        }
        int top = argmax(HEAD, VOCAB);
        printf("step %d id_in=%d top=%d logit=%.4f\n", step, id, top, HEAD[top]);
        fflush(stdout);
        if (step >= n0 && id == 1) break; /* EOS */
        NPOS_TOK++;
    }
    return 0;
}
