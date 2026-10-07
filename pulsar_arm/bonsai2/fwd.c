/* fwd.c: Bonsai-2-27B single-token forward, greedy oracle-diff vs the fork server.
 * Every kernel below is transcribed from fork source and validated before:
 *   PTQ decode (tq_probe bit-exact) | FWHT (fwht.S bit-exact) | signs table
 *   GEMV (tq_gemv_neon bit-exact) | GDN step (gdn_test fp32) | rope (fp32)
 * New composition only: embed (FWHT-then-signs per build_embd_rows),
 * per-head L2 over 128 (NOT joint-4096), raw-gate fused GDN, SwiGLU FFN.
 * Validation: feed prompt ids, argmax each step, diff vs server temp-0 stream.
 *
 * Build: gcc -O2 -march=armv8.2-a+dotprod -DTQ_XGEMV_LIB
 *          -Dmain=tq_neon_main -c tq_gemv_neon.c
 *        gcc -O2 -march=armv8.2-a+dotprod -c fwd.c
 *        gcc -O2 -o fwd fwd.o tq_gemv_neon.o fwht.S -L$LB \
 *          -lggml-base -lggml-cpu -Wl,-rpath,$LB -lm
 * Run: taskset -c 3 ./fwd <gguf> <id0> <id1> ... --gen G
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

#define HID 5120
#define INTER 17408
#define NHEAD 24
#define HDIM 256
#define NKV 4
#define NLAYER 64
#define VOCAB 248320
#define DSTATE 128
#define NKH 16
#define NVH 48
#define QKVW 10240
#define ZW 6144
#define CTX 512

void quantize_row_q8_0(const float * x, void * y, int64_t k);
void ggml_vec_dot_ptq1_0_q8_0(int n, float * s, size_t bs, const void * vx,
                              size_t bx, const void * vy, size_t by, int nrc);
static int XCHECK = 0;
void tq_gemv_neon(int rows, int cols, const uint8_t *w, const float *x,
                  float *y, const uint8_t *xq);
void s8_gemv(int rows, int cols, const uint8_t *w8, float *y,
             const uint8_t *xq);
void fwht1024_f32(float * x);

#define V8(nm, v) do { if (TRACE) { printf("  %s v8:", nm); \
    for (int _i = 0; _i < 8; _i++) printf(" %.6g", (double)(v)[_i]); \
    printf("\n"); } } while (0)
#define V8R(nm, v, s) do { if (TRACE) { printf("  %s [%d]:", nm, (s)); \
    for (int _i = 0; _i < 8; _i++) printf(" %.6g", (double)(v)[(s) + _i]); \
    printf("\n"); } } while (0)
static int TRACE = 0;
#define MAG(nm, v, n) do { if (TRACE) { double _s = 0; int _n = (n); \
    for (int _i = 0; _i < _n; _i++) _s += (double)(v)[_i] * (v)[_i]; \
    printf("  %s rms=%.4g\n", nm, sqrt(_s / _n)); } } while (0)

static uint8_t *G;                 /* mmap base */
static char TN[900][64];           /* tensor names */
static uint64_t TO[900];           /* absolute offsets */
static int64_t TD0[900], TD1[900]; /* dims */
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
        TD0[i] = TD1[i] = 1;
        for (int d = 0; d < nd; d++) {
            uint64_t v = *(uint64_t *)p; p += 8;
            if (d == 0) TD0[i] = (int64_t)v; else if (d == 1) TD1[i] = (int64_t)v;
        }
        p += 4; /* type */
        uint64_t off = *(uint64_t *)p; p += 8;
        TO[i] = off; /* relative; fixed below */
    }
    uint64_t ds = (((uint64_t)(p - G)) + 31) & ~31ull;
    printf("tensors=%d data_start=%llu\n", NT, (unsigned long long)ds);
    for (int i = 0; i < NT; i++) TO[i] += ds;
}
static uint8_t *tw(const char *name) {
    for (int i = 0; i < NT; i++)
        if (!strcmp(TN[i], name)) return G + TO[i];
    printf("tensor missing: %s\n", name);
    exit(1);
}
static int tw_idx(const char *name) {
    for (int i = 0; i < NT; i++)
        if (!strcmp(TN[i], name)) return i;
    printf("tensor missing: %s\n", name);
    exit(1);
}
/* transposed s8 cache: per 128-group [128B raw trits][4B fp32 scale].
   Decoded once at first use (OpenMP over rows); the per-token dot then
   streams trits with no digit decode. Same math/order as the staged path
   (bit-exact preserved). */
static uint8_t *S8CACHE[900];
static float h2f(uint16_t h);
static const uint8_t P3T[6] = {1, 3, 9, 27, 81, 243};
static void transpose_tensor(int idx) {
    int64_t K = TD0[idx], rows = TD1[idx];
    int ng = (int)(K / 128);
    uint8_t *src = G + TO[idx];
    uint8_t *dst = malloc((size_t)rows * ng * 132);
    if (!dst) { printf("s8 malloc fail\n"); exit(1); }
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; r++) {
        for (int g = 0; g < ng; g++) {
            const uint8_t *b = src + ((size_t)r * ng + g) * 28;
            int8_t *o = (int8_t *)(dst + ((size_t)r * ng + g) * 132);
            int k = 0;
            for (int q = 0; q < 2; q++) {
                int base = q ? 16 : 0, c = q ? 8 : 16;
                for (int nn = 0; nn < 5; nn++)
                    for (int m = 0; m < c; m++) {
                        uint8_t v = (uint8_t)(b[base + m] * P3T[nn]);
                        o[k++] = (int8_t)(((uint16_t)v * 3) >> 8) - 1;
                    }
            }
            for (int nn = 0; nn < 4; nn++)
                for (int h = 0; h < 2; h++) {
                    uint8_t v = (uint8_t)(b[24 + h] * P3T[nn]);
                    o[k++] = (int8_t)(((uint16_t)v * 3) >> 8) - 1;
                }
            { float sc = h2f((uint16_t)(b[26] | (b[27] << 8))); memcpy(o + 128, &sc, 4); }
        }
    }
    S8CACHE[idx] = dst;
}
static uint8_t *tw_s8(const char *name) {
    int idx = tw_idx(name);
    if (!S8CACHE[idx]) transpose_tensor(idx);
    return S8CACHE[idx];
}
static void dump_off(const char *name) {
    for (int i = 0; i < NT; i++)
        if (!strcmp(TN[i], name)) {
            printf("OFF %s = %llu dims=%lld x %lld\n", name,
                   (unsigned long long)TO[i], (long long)TD0[i], (long long)TD1[i]);
            return;
        }
    printf("OFF %s MISSING\n", name);
}

static float h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu, o;
    if (e == 0) { if (!m) return 0.0f; float f = (float)m * 5.9604645e-8f; memcpy(&o, &f, 4); o |= s; }
    else if (e == 31) o = s | 0x7f800000u | (m << 13);
    else o = s | ((e + 112u) << 23) | (m << 13);
    float r; memcpy(&r, &o, 4); return r;
}
static float bf16f(uint16_t h) {
    uint32_t o = (uint32_t)h << 16; float r; memcpy(&r, &o, 4); return r;
}
static const uint8_t P3[6] = {1, 3, 9, 27, 81, 243};

/* PTQ group decode -> 128 floats incl scale (proven in tq_xgemv) */
static void dec_grp(const uint8_t *b, float *o) {
    float sc = h2f((uint16_t)(b[26] | (b[27] << 8)));
    int k = 0;
    for (int q = 0; q < 2; q++) {
        int base = q ? 16 : 0, c = q ? 8 : 16;
        for (int nn = 0; nn < 5; nn++)
            for (int m = 0; m < c; m++) {
                uint8_t v = (uint8_t)(b[base + m] * P3[nn]);
                o[k++] = (float)(((uint16_t)v * 3) >> 8) - 1.0f;
            }
    }
    for (int nn = 0; nn < 4; nn++)
        for (int h = 0; h < 2; h++) {
            uint8_t v = (uint8_t)(b[24 + h] * P3[nn]);
            o[k++] = (float)(((uint16_t)v * 3) >> 8) - 1.0f;
        }
    for (int i = 0; i < 128; i++) o[i] *= sc;
}

static float *SGN5120, *SGN6144, *SGN17408;
static void load_sign(const char *path, float **o, int n) {
    FILE *f = fopen(path, "rb");
    *o = malloc(n * 4);
    if (!f || fread(*o, 4, n, f) != (size_t)n) { printf("signs fail %s\n", path); exit(1); }
    fclose(f);
}

/* activation transform: x * signs -> FWHT/1024 -> Q8 (mirrors build_lora_mm) */
static uint8_t *XQ;   /* max (248320/32)*34 */
static float *TT;     /* max 248320 */
static void xform(const float *x, int K, const float *sgn, uint8_t *xq) {
    for (int i = 0; i < K; i++) TT[i] = x[i] * sgn[i];
    for (int b = 0; b < K / 1024; b++) fwht1024_f32(TT + b * 1024);
    quantize_row_q8_0(TT, xq, K);
}
static void gemvT(const uint8_t *W, int rows, int K, const float *sgn,
                  const float *x, float *y) {
    xform(x, K, sgn, XQ);
    tq_gemv_neon(rows, K, W, x, y, XQ);
    if (XCHECK) {
        static int ncall = 0;
        /* L0 ffn_gate (3rd gemvT) and ffn_up (4th) */
        if (ncall == 3 || ncall == 4) {
            printf(" xchk %s:\n", ncall == 3 ? "ffn_gate" : "ffn_up");
            int ng = K / 128;
            for (int r = 1000; r < 1008; r++) {
                float ref = 0;
                ggml_vec_dot_ptq1_0_q8_0(K, &ref, 0, W + (size_t)r * ng * 28, 0, XQ, 0, 1);
                printf("  xchk row %d mine=%.6g fork=%.6g\n", r, y[r], ref);
            }
        }
        ncall++;
    }
}
/* transposed-weight path: same xform, sdot-only dot over cached s8 rows */
static void gemvS(const char *name, int rows, int K, const float *sgn,
                  const float *x, float *y) {
    uint8_t *W8 = tw_s8(name);
    xform(x, K, sgn, XQ);
    s8_gemv(rows, K, W8, y, XQ);
}
/* shared-transform pairs/triples: one signs/FWHT/Q8 for several dots over
   the same activation (bit-exact: xform is deterministic, dots unchanged).
   Kinds (not TWN results: the single NB buffer can't hold two names). */
static void gemvS2(int il, const char *k1, const char *k2,
                   int r1, int r2, int K,
                   const float *sgn, const float *x, float *y1, float *y2) {
    char n1[64], n2[64];
    sprintf(n1, "blk.%d.%s", il, k1); sprintf(n2, "blk.%d.%s", il, k2);
    uint8_t *W1 = tw_s8(n1), *W2 = tw_s8(n2);
    xform(x, K, sgn, XQ);
    s8_gemv(r1, K, W1, y1, XQ);
    s8_gemv(r2, K, W2, y2, XQ);
}
static void gemvS3(int il, const char *k1, const char *k2, const char *k3,
                   int r1, int r2, int r3, int K,
                   const float *sgn, const float *x,
                   float *y1, float *y2, float *y3) {
    char n1[64], n2[64], n3[64];
    sprintf(n1, "blk.%d.%s", il, k1);
    sprintf(n2, "blk.%d.%s", il, k2);
    sprintf(n3, "blk.%d.%s", il, k3);
    uint8_t *W1 = tw_s8(n1), *W2 = tw_s8(n2), *W3 = tw_s8(n3);
    xform(x, K, sgn, XQ);
    s8_gemv(r1, K, W1, y1, XQ);
    s8_gemv(r2, K, W2, y2, XQ);
    s8_gemv(r3, K, W3, y3, XQ);
}

static void rms(const float *x, const float *w, int n, float *y) {
    double s = 0;
    for (int i = 0; i < n; i++) s += (double)x[i] * x[i];
    float sc = 1.0f / sqrtf((float)(s / n) + 1e-6f);
    for (int i = 0; i < n; i++) y[i] = x[i] * sc * w[i];
}
static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float silu(float x) { return x * sigmoid(x); }
static float softplus_g(float x) { return x > 20.0f ? x : log1pf(expf(x)); }
/* per-128 L2, scale = 1/max(sqrt(sum),eps) - ggml_l2_norm exact */
static void l2_128(float *x) {
    for (int h = 0; h < 16; h++) {
        double s = 0;
        for (int i = 0; i < 128; i++) s += (double)x[h * 128 + i] * x[h * 128 + i];
        float sc = 1.0f / fmaxf(sqrtf((float)s), 1e-6f);
        for (int i = 0; i < 128; i++) x[h * 128 + i] *= sc;
    }
}
/* MRoPE partial NEOX: pairs (j,j+32) j<32, ang = p*1e7^(-2j/64), rest verbatim */
static void rope_apply(float *x, int heads, int pos) {
    for (int h = 0; h < heads; h++) {
        float *xh = x + h * 256;
        for (int j = 0; j < 32; j++) {
            float ang = (float)pos * powf(1e7f, -2.0f * j / 64.0f);
            float c = cosf(ang), s = sinf(ang);
            float a = xh[j], b = xh[j + 32];
            xh[j] = a * c - b * s;
            xh[j + 32] = a * s + b * c;
        }
    }
}
/* GDN step, transcribed from validated gdn_test */
static void gdn_step(const float *q, const float *k, const float *v,
                     float beta, float g, float *buf, float *out) {
    float delta[128];
    float dec = expf(g);
    for (int j = 0; j < 128 * 128; j++) buf[j] *= dec;
    for (int j = 0; j < 128; j++) {
        float s = 0.0f;
        for (int i = 0; i < 128; i++) s += buf[j * 128 + i] * k[i];
        delta[j] = (v[j] - s) * beta;
    }
    for (int j = 0; j < 128; j++)
        for (int i = 0; i < 128; i++)
            buf[j * 128 + i] += k[i] * delta[j];
    for (int j = 0; j < 128; j++) {
        float s = 0.0f;
        for (int i = 0; i < 128; i++) s += buf[j * 128 + i] * q[i];
        out[j] = s * (1.0f / sqrtf(128.0f));
    }
}

/* scratch (single token) */
static float X[HID], XN[HID], AO[HID], FO[HID];
static float QKV[QKVW], ZV[ZW], B48[48], A48[48];
static float RING[3][QKVW], CIN[4][QKVW], CONV[QKVW];
static float QH_[16 * 128], KH_[16 * 128], VH_[48 * 128];
static float GOUT[48 * 128], ZN[48 * 128], GDN[48 * 128], FOUT[ZW];
static float QF[24 * 256], GF[24 * 256], KF[4 * 256], VF[4 * 256];
static float QFULL[12288];
static float ATTO[24 * 256], LG[INTER], LU[INTER];
static float HEAD[VOCAB];
static float SSM[48][48][128 * 128];   /* 150 MB */
static float CRING[48][3][QKVW];       /* 5.9 MB */
static float PM[ZW];                   /* gdn_v_grouped permute for ssm_out */
static float KCA[16][512][4 * 256];     /* full-layer KV cache */
static float VCA[16][512][4 * 256];
static int NPOS;

static char NB[128];
static char *tn(int il, const char *kind, char *o) {
    sprintf(o, "blk.%d.%s", il, kind);
    return o;
}
#define TW(il, kind) tw(tn(il, kind, NB))
#define TWN(il, kind) tn(il, kind, NB)  /* name only, no table lookup */

/* plain BF16 matvec 48 outs (ssm_alpha/beta are NOT transformed) */
static void bfgemv(const uint8_t *W, const float *x, float *y) {
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < 48; r++) {
        const uint16_t *row = (const uint16_t *)(W + (size_t)r * HID * 2);
        double s = 0;
        for (int i = 0; i < HID; i++) s += (double)bf16f(row[i]) * x[i];
        y[r] = (float)s;
    }
}

static void layer_linear(int il, int li) {
    gemvS2(il, "attn_qkv.weight", "attn_gate.weight", QKVW, ZW, HID, SGN5120, XN, QKV, ZV);
    bfgemv(TW(il, "ssm_beta.weight"), XN, B48);
    bfgemv(TW(il, "ssm_alpha.weight"), XN, A48);
    float *dtb = (float *)TW(il, "ssm_dt.bias");
    float *amp = (float *)TW(il, "ssm_a");
    /* conv over [cached s-3,s-2,s-1, current], THEN roll the ring for next step */
    float *ker = (float *)TW(il, "ssm_conv1d.weight");
    /* kernel layout [t + c*4] (ne=[4,10240]): element (t,c) at c*4+t */
    for (int c = 0; c < QKVW; c++) {
        double s = 0;
        for (int t = 0; t < 3; t++) s += (double)CRING[li][t][c] * ker[c * 4 + t];
        s += (double)QKV[c] * ker[c * 4 + 3];
        CONV[c] = silu((float)s);
    }
    memcpy(CRING[li][0], CRING[li][1], 2 * QKVW * 4);
    memcpy(CRING[li][2], QKV, QKVW * 4);
    memcpy(QH_, CONV, 2048 * 4);
    memcpy(KH_, CONV + 2048, 2048 * 4);
    memcpy(VH_, CONV + 4096, 6144 * 4);
    MAG("qkv", QKV, QKVW); MAG("conv", CONV, QKVW);
    V8("qkv", QKV); V8("conv", CONV); V8("z", ZV);
    V8R("qkv", QKV, 1000); V8R("conv", CONV, 1000);
    V8R("qkv", QKV, 8000); V8R("conv", CONV, 8000);
    /* per-head L2 over 128 on q (16 heads) and k (16 heads) */
    l2_128(QH_);
    l2_128(KH_);
    /* GDN over 48 independent heads: row-split parallel (each head owns its
       state slice; fp order within a head unchanged -> bit-exact streams) */
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < 48; h++) {
        float beta = sigmoid(B48[h]);
        float g0 = amp[h] * softplus_g(A48[h] + dtb[h]);
        gdn_step(QH_ + (h % 16) * 128, KH_ + (h % 16) * 128, VH_ + h * 128,
                 beta, g0, (float *)SSM[li][h], GOUT + h * 128);
    }
    /* gated norm: silu(z) * rmsnorm(gdn, ssm_norm[128]) */
    float *snw = (float *)TW(il, "ssm_norm.weight");
    for (int h = 0; h < 48; h++) {
        double s = 0;
        for (int i = 0; i < 128; i++) s += (double)GOUT[h * 128 + i] * GOUT[h * 128 + i];
        float sc = 1.0f / sqrtf((float)(s / 128) + 1e-6f);
        float *z = ZV + h * 128;
        for (int i = 0; i < 128; i++)
            ZN[h * 128 + i] = silu(z[i]) * (GOUT[h * 128 + i] * sc * snw[i]);
    }
    memcpy(FOUT, ZN, ZW * 4);
    MAG("gout", GOUT, ZW);
    V8("final", ZN);
    V8R("final", ZN, 1000);
    /* gdn_v_grouped=1: [hd=128,nk=16,rep=3] -> [hd,rep,nk] before signs/FWHT:
       P[i + 128*(h/16) + 384*(h%16)] = FOUT[h*128 + i] */
    for (int h = 0; h < 48; h++)
        for (int i = 0; i < 128; i++)
            PM[i + 128 * (h / 16) + 384 * (h % 16)] = FOUT[h * 128 + i];
    gemvS(TWN(il, "ssm_out.weight"), HID, ZW, SGN6144, PM, AO);
    V8("attnout", AO);
    V8R("attnout", AO, 1000);
    for (int i = 0; i < HID; i++) X[i] += AO[i];
    V8("xres", X);
}

static void layer_full(int il, int fi) {
    gemvS3(il, "attn_q.weight", "attn_k.weight", "attn_v.weight",
             12288, 1024, 1024, HID, SGN5120, XN, QFULL, KF, VF);
    V8R("qfull", QFULL, 1000); V8R("qfull", QFULL, 8000);
    /* fused layout is interleaved per head: [Q0 G0 Q1 G1 ...], 256-wide
       halves - NOT first-half/second-half (graph strides nb1 = 512 elems) */
    for (int h = 0; h < 24; h++) {
        memcpy(QF + h * 256, QFULL + h * 512, 256 * 4);
        memcpy(GF + h * 256, QFULL + h * 512 + 256, 256 * 4);
    }
    float *qnw = (float *)TW(il, "attn_q_norm.weight");
    for (int h = 0; h < 24; h++) rms(QF + h * 256, qnw, 256, QF + h * 256);
    float *knw = (float *)TW(il, "attn_k_norm.weight");
    for (int h = 0; h < 4; h++) rms(KF + h * 256, knw, 256, KF + h * 256);
    rope_apply(QF, 24, NPOS);
    rope_apply(KF, 4, NPOS);
    V8R("kcur", KF, 512); V8R("vcur", VF, 512);
    memcpy(KCA[fi][NPOS], KF, 1024 * 4);
    memcpy(VCA[fi][NPOS], VF, 1024 * 4);
    for (int h = 0; h < 24; h++) {
        int kh = h / 6;
        float mx = -1e30f, sc[CTX];
        for (int t = 0; t <= NPOS; t++) {
            float s = 0;
            float *Kt = (float *)KCA[fi][t] + kh * 256;
            for (int i = 0; i < 256; i++) s += QF[h * 256 + i] * Kt[i];
            sc[t] = s * 0.0625f;
            if (sc[t] > mx) mx = sc[t];
        }
        double es = 0;
        for (int t = 0; t <= NPOS; t++) es += expf(sc[t] - mx);
        float *o = ATTO + h * 256;
        for (int i = 0; i < 256; i++) o[i] = 0;
        for (int t = 0; t <= NPOS; t++) {
            float p = expf(sc[t] - mx) / (float)es;
            float *Vt = (float *)VCA[fi][t] + kh * 256;
            for (int i = 0; i < 256; i++) o[i] += p * Vt[i];
        }
    }
    /* gate: sigmoid per element over 6144 then fold into heads */
    for (int i = 0; i < 6144; i++) GF[i] = sigmoid(GF[i]);
    for (int i = 0; i < 6144; i++) ATTO[i] *= GF[i];
    MAG("atto", ATTO, 6144);
    V8R("atto", ATTO, 1000);
    gemvS(TWN(il, "attn_output.weight"), HID, 6144, SGN6144, ATTO, AO);
    for (int i = 0; i < HID; i++) X[i] += AO[i];
}

static void layer_ffn(int il) {
    float *pnw = (float *)TW(il, "post_attention_norm.weight");
    rms(X, pnw, HID, XN);
    MAG("pn", XN, HID); V8("pn", XN);
    gemvS2(il, "ffn_gate.weight", "ffn_up.weight", INTER, INTER, HID, SGN5120, XN, LG, LU);
    for (int i = 0; i < INTER; i++) LU[i] = silu(LG[i]) * LU[i];
    gemvS(TWN(il, "ffn_down.weight"), HID, INTER, SGN17408, LU, FO);
    MAG("ffn", FO, HID);
    V8("ffnout", FO);
    V8("gate", LG); V8("up", LU);
    V8R("gate", LG, 1000); V8R("up", LU, 1000); V8R("ffnout", FO, 1000);
    for (int i = 0; i < HID; i++) X[i] += FO[i];
}

/* embed: latent row -> dequant -> FWHT -> signs */
static void embed(int id) {
    uint8_t *W = tw("token_embd.weight");
    const uint8_t *row = W + (size_t)id * 40 * 28;
    static float tmp[128];
    for (int g = 0; g < 40; g++) dec_grp(row + g * 28, tmp), memcpy(X + g * 128, tmp, 128 * 4);
    for (int b = 0; b < 5; b++) fwht1024_f32(X + b * 1024);
    for (int i = 0; i < HID; i++) X[i] *= SGN5120[i];
}

static int argmax(const float *v, int n);
/* sampler, fork order: top-k -> top-p -> min-p on RAW logits, temp scale
   last, then softmax draw (llama.cpp default chain). */
static uint64_t RS;
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
static float STEMP = 0;
static float STOPP = 1;
static float SMINP = 0;
static int STopK = 0;
static int argmax(const float *v, int n) {
    int bi = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[bi]) bi = i;
    return bi;
}

int main(int argc, char **argv) {
    gguf_table(argv[1]);
    load_sign("/tmp/sign5120.bin", &SGN5120, 5120);
    load_sign("/tmp/sign6144.bin", &SGN6144, 6144);
    load_sign("/tmp/sign17408.bin", &SGN17408, 17408);
    XQ = malloc(((size_t)VOCAB / 32) * 34);
    TT = malloc((size_t)VOCAB * 4);
    if (getenv("DUMPOFF")) {
        dump_off("blk.0.ffn_gate.weight");
        dump_off("blk.0.ffn_up.weight");
        dump_off("blk.0.ffn_down.weight");
        dump_off("blk.0.attn_qkv.weight");
        dump_off("blk.0.ssm_conv1d.weight");
        dump_off("token_embd.weight");
        dump_off("output.weight");
        exit(0);
    }
    memset(SSM, 0, sizeof(SSM));
    memset(CRING, 0, sizeof(CRING));
    int ids[1200], nids = 0, gen = 0;
    for (int a = 2; a < argc; a++) {
        if (!strcmp(argv[a], "--trace")) TRACE = 1;
        else if (!strcmp(argv[a], "--xcheck")) XCHECK = 1;
        else if (!strcmp(argv[a], "--sample")) {
            STEMP = atof(argv[++a]); STOPP = atof(argv[++a]);
            STopK = atoi(argv[++a]); RS = strtoull(argv[++a], 0, 10);
            SMINP = (a + 1 < argc && argv[a+1][0] != '-') ? atof(argv[++a]) : 0.0f;
        }
        else if (!strcmp(argv[a], "--gen")) gen = atoi(argv[++a]);
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
        MAG("emb", X, HID);
        V8("emb", X);
        int fi = 0, li = 0;
        for (int il = 0; il < NLAYER; il++) {
            float *anw = (float *)TW(il, "attn_norm.weight");
            rms(X, anw, HID, XN);
            if ((il + 1) % 4 == 0) layer_full(il, fi++);
            else layer_linear(il, li++);
            layer_ffn(il);
            if (TRACE) { double _s = 0; for (int i = 0; i < HID; i++) _s += (double)X[i] * X[i];
                printf("L%d Xrms=%.4g\n", il, sqrt(_s / HID)); }
        }
        float *onw = (float *)tw("output_norm.weight");
        rms(X, onw, HID, XN);
        gemvS("output.weight", VOCAB, HID, SGN5120, XN, HEAD);
        int top = argmax(HEAD, VOCAB);
        printf("step %d id_in=%d top=%d logit=%.4f logit271=%.4f\n", step, id, top, HEAD[top], HEAD[271]);
        /* top-5 */
        { float c[5] = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f}; int bi[5] = {0,0,0,0,0};
          for (int i = 0; i < VOCAB; i++) { float v = HEAD[i];
            for (int k = 0; k < 5; k++) if (v > c[k]) {
              for (int m = 4; m > k; m--) { c[m] = c[m-1]; bi[m] = bi[m-1]; }
              c[k] = v; bi[k] = i; break; } }
          printf("  top5: %d:%.3f %d:%.3f %d:%.3f %d:%.3f %d:%.3f\n",
                 bi[0],c[0],bi[1],c[1],bi[2],c[2],bi[3],c[3],bi[4],c[4]); }
        fflush(stdout);
        if (step >= n0 && id == 248046) break; /* EOS */
        NPOS++;
    }
    return 0;
}
