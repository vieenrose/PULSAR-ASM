/* g4ref.c - bit-exact scalar reference of Gemma 4 E2B (text only) on Google's
 * Q4_0 QAT GGUF. This file DEFINES the arithmetic the Jetson Nano engine (GPU
 * GEMVs + AArch64 host) must reproduce bit for bit; it is not tuned for speed.
 *
 * Arithmetic contract (build with -ffp-contract=off: every fused op is explicit):
 *  - Activations entering a Q4_0 / Q6_K GEMV are quantized to Q8_K blocks of 256
 *    (f32 d, 256 int8; the K2 quantizer: signed max, iscale = -127/max,
 *    q = nearest_int(fmaf(iscale, x, 0x1.8p23)), clamp < 127, d = 1/iscale).
 *  - Q4_0 GEMV (layers): per row, blocks of 32 in ascending order,
 *    acc = fmaf(d_w * d_a, (float)isum, acc), isum exact int32. Same as K2's kernel.
 *  - Q6_K GEMV (tied head): per super-block of 256 in ascending order,
 *    isum = sum_j sc_j * sum_l (q - 32) * q8 (exact int32), acc = fmaf(d_w * d_a, (float)isum, acc).
 *  - F16 GEMV (per_layer_model_proj): per row acc = fmaf(w_k, x_k, acc), k ascending.
 *  - Norms, rope, GELU, softmax normalisation, soft-cap: float64 with + - * / sqrt
 *    only, sequential sums, and g4_exp() below (no libm transcendental), rounded
 *    to float32 between stages. Attention dots and the p.V sum: float32 fmaf
 *    chains in ascending order.
 *  - Rope cos/sin come from a baked table (g4_rope.py), not computed here.
 *
 * Usage: g4ref <gemma.gguf> <rope.bin> <ids...> [--gen N]
 * Prints: step <n> id_in=<id> top=<id> logit=<x.xxxx>   (logit after soft-cap)
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define H 1536
#define NL 35
#define NH 8
#define PLE 256
#define VOCAB 262144
#define HD_S 256
#define HD_F 512
#define FIRST_SHARED 15
#define WINDOW 512
#define MAXSEQ 2048
#define EPS 1e-6
#define SOFTCAP 30.0
#define INTER 6144

/* ------------------------------------------------------------------ GGUF */
typedef struct { char name[96]; uint32_t type; uint64_t dims[4]; int nd; const uint8_t *data; } tensor_t;
static tensor_t T[600];
static int NT;
static int SLIDING[NL];          /* from gemma4.attention.sliding_window_pattern */

static const uint8_t *gp;
static uint64_t rd64(void) { uint64_t v; memcpy(&v, gp, 8); gp += 8; return v; }
static uint32_t rd32(void) { uint32_t v; memcpy(&v, gp, 4); gp += 4; return v; }
static void skipval(uint32_t t, const char *key);
static void readstr(char *dst, size_t cap) {
    uint64_t n = rd64();
    if (dst) { size_t m = n < cap - 1 ? n : cap - 1; memcpy(dst, gp, m); dst[m] = 0; }
    gp += n;
}
static const int SCSZ[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
static void skipval(uint32_t t, const char *key) {
    if (t == 8) { readstr(NULL, 0); return; }
    if (t == 9) {
        uint32_t at = rd32(); uint64_t n = rd64();
        int want = key && !strcmp(key, "gemma4.attention.sliding_window_pattern");
        for (uint64_t i = 0; i < n; i++) {
            if (want && at == 7 && i < NL) SLIDING[i] = gp[0];
            skipval(at, NULL);
        }
        return;
    }
    gp += SCSZ[t];
}
static const tensor_t *tens(const char *name) {
    for (int i = 0; i < NT; i++) if (!strcmp(T[i].name, name)) return &T[i];
    fprintf(stderr, "missing tensor %s\n", name); exit(2);
}
static void gguf_open(const char *path) {
    int fd = open(path, O_RDONLY); struct stat s;
    if (fd < 0 || fstat(fd, &s)) { perror(path); exit(2); }
    const uint8_t *base = mmap(NULL, s.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); exit(2); }
    gp = base;
    if (memcmp(gp, "GGUF", 4)) { fprintf(stderr, "not gguf\n"); exit(2); }
    gp += 4;
    if (rd32() != 3) { fprintf(stderr, "gguf v3 only\n"); exit(2); }
    uint64_t nt = rd64(), nkv = rd64();
    for (uint64_t i = 0; i < nkv; i++) { char key[128]; readstr(key, sizeof key); uint32_t t = rd32(); skipval(t, key); }
    uint64_t offs[600];
    NT = (int)nt;
    for (int i = 0; i < NT; i++) {
        readstr(T[i].name, sizeof T[i].name);
        T[i].nd = (int)rd32();
        for (int d = 0; d < T[i].nd; d++) T[i].dims[d] = rd64();
        T[i].type = rd32();
        offs[i] = rd64();
    }
    uint64_t start = ((uint64_t)(gp - base) + 31) / 32 * 32;
    for (int i = 0; i < NT; i++) T[i].data = base + start + offs[i];
}

/* ------------------------------------------------------------- numerics */
static float h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, o;
    if (e == 0) {
        if (!m) o = s;
        else { /* subnormal half: m * 2^-24, exact in float */
            float f = (float)m * 0x1p-24f; uint32_t b; memcpy(&b, &f, 4); o = s | b; }
    } else if (e == 31) o = s | 0x7f800000u | (m << 13);
    else o = s | ((e + 112u) << 23) | (m << 13);
    float r; memcpy(&r, &o, 4); return r;
}

/* exp(x) in double from + - * and exponent assembly only: Cody-Waite reduction
 * by ln2, degree-13 Taylor polynomial in Horner form, scale by 2^k in two
 * steps so subnormal results stay exact-rounded twice at worst. Deterministic
 * on every IEEE double machine; accurate to a few ulp, which is all the model
 * needs. */
static double pow2i(int k) { uint64_t b = (uint64_t)(k + 1023) << 52; double d; memcpy(&d, &b, 8); return d; }
/* exp constants at file scope: g4_host.S receives a table built from these same
   definitions (g4_exp_table), so the assembly uses identical bits */
static const double EXP_LOG2E = 1.4426950408889634, EXP_LN2HI = 6.93147180369123816490e-01,
                    EXP_LN2LO = 1.90821492927058770002e-10, EXP_SH = 6755399441055744.0; /* 1.5*2^52 */
static const double EXP_P0 = 1.0 / 6227020800.0;            /* 1/13! */
static const double EXP_C[13] = {1.0 / 479001600.0, 1.0 / 39916800.0, 1.0 / 3628800.0,
    1.0 / 362880.0, 1.0 / 40320.0, 1.0 / 5040.0, 1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0,
    1.0 / 6.0, 0.5, 1.0, 1.0};
static const double GELU_C1 = 1.5957691216057308, GELU_C3 = 0.07135481627462202;
static double g4_exp(double x) {
    if (x != x) return x;
    if (x > 709.0) return INFINITY;
    if (x < -745.0) return 0.0;
    double kd = (x * EXP_LOG2E + EXP_SH) - EXP_SH;          /* round to nearest integer */
    int k = (int)kd;
    double r = (x - kd * EXP_LN2HI) - kd * EXP_LN2LO;       /* |r| <= ~0.347 */
    double p = EXP_P0;
    for (int i = 0; i < 13; i++) p = p * r + EXP_C[i];
    int k1 = k / 2, k2 = k - k1;                /* each in [-538, 512] */
    return (p * pow2i(k1)) * pow2i(k2);
}
static double g4_tanh(double y) {
    double a = y < 0 ? -y : y;
    if (a > 20.0) return y < 0 ? -1.0 : 1.0;
    double e = g4_exp(-2.0 * a);
    double t = (1.0 - e) / (1.0 + e);
    return y < 0 ? -t : t;
}

/* ------------------------------------------------------- Q8_K activations */
#define Q8B 292   /* f32 d + 256 int8 + 32 bytes unused (K2's stride) */
static int nearest_int(float x, float iscale) {
    float v = fmaf(iscale, x, 12582912.f);
    int i; memcpy(&i, &v, 4);
    return (i & 0x007fffff) - 0x00400000;
}
static void q8k(const float *x, uint8_t *y, int n) {
    for (int b = 0; b < n / 256; b++, x += 256) {
        uint8_t *yb = y + (size_t)b * Q8B;
        int8_t *q = (int8_t *)(yb + 4);
        float mx = 0, amax = 0;
        for (int j = 0; j < 256; j++) { float a = fabsf(x[j]); if (a > amax) { amax = a; mx = x[j]; } }
        memset(yb + 260, 0, 32);
        if (amax == 0) { memset(yb, 0, 260); continue; }
        float iscale = -127.f / mx;
        for (int j = 0; j < 256; j++) { int v = nearest_int(x[j], iscale); q[j] = (int8_t)(v < 127 ? v : 127); }
        float d = 1.0f / iscale; memcpy(yb, &d, 4);
    }
}

/* ---------------------------------------------------------------- GEMVs */
static void gemv_q4(const tensor_t *w, const uint8_t *xq, float *y) {
    int cols = (int)w->dims[0], rows = (int)w->dims[1], nb = cols / 32;
    if (w->type != 2) { fprintf(stderr, "%s not Q4_0\n", w->name); exit(2); }
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const uint8_t *row = w->data + (size_t)r * nb * 18;
        float acc = 0.0f;
        for (int b = 0; b < nb; b++) {
            const uint8_t *blk = row + b * 18;
            float ws = h2f((uint16_t)(blk[0] | blk[1] << 8));
            const uint8_t *q8b = xq + (size_t)(b / 8) * Q8B;
            float db; memcpy(&db, q8b, 4);
            const int8_t *qs = (const int8_t *)(q8b + 4 + (b % 8) * 32);
            int32_t s = 0;
            for (int j = 0; j < 16; j++) {
                s += ((int)(blk[2 + j] & 15) - 8) * qs[j];
                s += ((int)(blk[2 + j] >> 4) - 8) * qs[j + 16];
            }
            acc = fmaf(ws * db, (float)s, acc);
        }
        y[r] = acc;
    }
}
/* index of value t (0..255) inside a Q6_K block, ggml dequantize_row_q6_K order:
   half n = t/128, within it quarter k = (t%128)/32, l = t%32 */
static int q6_value(const uint8_t *blk, int t) {
    const uint8_t *ql = blk, *qh = blk + 128;
    int n = t / 128, k = (t % 128) / 32, l = t % 32;
    const uint8_t *L = ql + 64 * n, *Hh = qh + 32 * n;
    int lo = (k & 1) ? L[l + 32] : L[l];
    lo = (k & 2) ? lo >> 4 : lo & 15;
    int hi = (Hh[l] >> (2 * k)) & 3;
    return (lo | hi << 4) - 32;
}
static int q6_scale_index(int t) { int n = t / 128, k = (t % 128) / 32, l = t % 32; return 8 * n + 2 * k + l / 16; }

/* Per-block integer form: isum = sum over the 16 sub-blocks j of
   sc_j * sum_{l in j} q * q8. Sub-block j holds the 16 values t with
   q6_scale_index(t) == j (ggml's own grouping). */
static void gemv_q6k_rows(const tensor_t *w, const uint8_t *xq, float *y, int r0, int r1) {
    int cols = (int)w->dims[0], nb = cols / 256;
    if (w->type != 14) { fprintf(stderr, "%s not Q6_K\n", w->name); exit(2); }
    #pragma omp parallel for schedule(static)
    for (int r = r0; r < r1; r++) {
        const uint8_t *row = w->data + (size_t)r * nb * 210;
        float acc = 0.0f;
        for (int b = 0; b < nb; b++) {
            const uint8_t *blk = row + b * 210;
            const int8_t *sc = (const int8_t *)(blk + 192);
            float dw = h2f((uint16_t)(blk[208] | blk[209] << 8));
            const uint8_t *q8b = xq + (size_t)b * Q8B;
            float da; memcpy(&da, q8b, 4);
            const int8_t *qs = (const int8_t *)(q8b + 4);
            int32_t sub[16] = {0};
            for (int t = 0; t < 256; t++) sub[q6_scale_index(t)] += q6_value(blk, t) * qs[t];
            int32_t s = 0;
            for (int j = 0; j < 16; j++) s += sc[j] * sub[j];
            acc = fmaf(dw * da, (float)s, acc);
        }
        y[r] = acc;
    }
}
static void row_q6k(const tensor_t *w, int r, float *out) {   /* ggml: (d * sc) * q */
    int cols = (int)w->dims[0], nb = cols / 256;
    const uint8_t *row = w->data + (size_t)r * nb * 210;
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = row + b * 210;
        const int8_t *sc = (const int8_t *)(blk + 192);
        float d = h2f((uint16_t)(blk[208] | blk[209] << 8));
        for (int t = 0; t < 256; t++)
            out[b * 256 + t] = (d * (float)sc[q6_scale_index(t)]) * (float)q6_value(blk, t);
    }
}
static void gemv_f16(const tensor_t *w, const float *x, float *y) {
    int cols = (int)w->dims[0], rows = (int)w->dims[1];
    if (w->type != 1) { fprintf(stderr, "%s not F16\n", w->name); exit(2); }
    const uint16_t *wd = (const uint16_t *)w->data;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        float acc = 0.0f;
        for (int k = 0; k < cols; k++) acc = fmaf(h2f(wd[(size_t)r * cols + k]), x[k], acc);
        y[r] = acc;
    }
}
static const float *f32t(const tensor_t *t) { if (t->type != 0) { fprintf(stderr, "%s not F32\n", t->name); exit(2); } return (const float *)t->data; }

/* GEMV hooks: the reference uses the CPU functions above; the Nano engine
   (g4eng.c) points them at its GPU kernels, which M3 proved bit-identical, so the
   rest of forward() is literally shared code. */
static void head_cpu(const tensor_t *w, const uint8_t *xq, float *y) { gemv_q6k_rows(w, xq, y, 0, (int)w->dims[1]); }
static void (*G_Q4)(const tensor_t *, const uint8_t *, float *) = gemv_q4;
static void (*G_HEAD)(const tensor_t *, const uint8_t *, float *) = head_cpu;
static void (*G_F16)(const tensor_t *, const float *, float *) = gemv_f16;
/* n GEMVs sharing one quantized input (q+k+v, gate+up); the engine runs them as one launch */
static void q4n_seq(int n, const tensor_t **ts, const uint8_t *xq, float **ys) { for (int i = 0; i < n; i++) G_Q4(ts[i], xq, ys[i]); }
static void (*G_Q4N)(int, const tensor_t **, const uint8_t *, float **) = q4n_seq;

/* per-layer tensor table, filled once (the names never change) */
typedef struct {
    const tensor_t *q, *k, *v, *o, *gate, *up, *down, *ig, *proj;
    const float *n_attn, *n_q, *n_k, *n_post_attn, *n_ffn, *n_post_ffw, *n_post;
    float scale;
} layer_t;
static layer_t LT[NL];
static void layer_table(void) {
    char b[96];
#define W(kind) (snprintf(b, sizeof b, "blk.%d.%s.weight", il, kind), tens(b))
    for (int il = 0; il < NL; il++) {
        layer_t *L = &LT[il];
        L->q = W("attn_q"); L->o = W("attn_output"); L->gate = W("ffn_gate"); L->up = W("ffn_up");
        L->down = W("ffn_down"); L->ig = W("inp_gate"); L->proj = W("proj");
        L->k = il < FIRST_SHARED ? W("attn_k") : NULL; L->v = il < FIRST_SHARED ? W("attn_v") : NULL;
        L->n_attn = f32t(W("attn_norm")); L->n_q = f32t(W("attn_q_norm"));
        L->n_k = il < FIRST_SHARED ? f32t(W("attn_k_norm")) : NULL;
        L->n_post_attn = f32t(W("post_attention_norm")); L->n_ffn = f32t(W("ffn_norm"));
        L->n_post_ffw = f32t(W("post_ffw_norm")); L->n_post = f32t(W("post_norm"));
        L->scale = f32t(W("layer_output_scale"))[0];
    }
#undef W
}

/* --------------------------------------------------------------- stages */
static double rms_r(const float *x, int n) {
    double ss = 0; for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    return sqrt(ss / n + EPS);
}
static void rmsnorm(const float *x, const float *w, float *y, int n) {
    double r = rms_r(x, n);
    for (int i = 0; i < n; i++) y[i] = (float)((double)x[i] / r * (double)w[i]);
}
static void rmsnorm_noscale(float *x, int n) {
    double r = rms_r(x, n);
    for (int i = 0; i < n; i++) x[i] = (float)((double)x[i] / r);
}
/* norm hooks (the engine points them at g4_host.S) */
static void (*G_RMS)(const float *, const float *, float *, int) = rmsnorm;
static void (*G_RMSNS)(float *, int) = rmsnorm_noscale;
static float gelu(float xf) {
    double x = xf;
    return (float)(x / (1.0 + g4_exp(-(GELU_C1 * x + GELU_C3 * (x * x * x)))));
}
static void rope(float *x, const float *c, const float *s, int hd) {
    int h = hd / 2;
    for (int i = 0; i < h; i++) {
        double a = x[i], b = x[i + h], cc = c[i], ss = s[i];
        x[i] = (float)(a * cc - b * ss);
        x[i + h] = (float)(b * cc + a * ss);
    }
}

static float *ROPE;   /* slide cos | slide sin | full cos | full sin */
static const float *rope_cs(int full, int sin_, int pos) {
    size_t sl = (size_t)MAXSEQ * 128, fu = (size_t)MAXSEQ * 256;
    if (!full) return ROPE + (sin_ ? sl : 0) + (size_t)pos * 128;
    return ROPE + 2 * sl + (sin_ ? fu : 0) + (size_t)pos * 256;
}

static float KC[FIRST_SHARED][MAXSEQ * HD_F], VC[FIRST_SHARED][MAXSEQ * HD_F];
static float xs[H], hb[H], qb[NH * HD_F], kb[HD_F], vb[HD_F], ab[NH * HD_F],
             gb[2 * INTER], ub[2 * INTER], pin[NL * PLE], projb[NL * PLE], pleg[PLE],
             rowe[NL * PLE], sc[MAXSEQ], logits[VOCAB];
static uint8_t XQ[(2 * INTER / 256) * Q8B];

/* One query head over cached positions [lo, pos]: scores as float fmaf chains over
   d ascending, max, exp in float64 (g4_exp) with a sequential float64 sum,
   p = (float)(e / sum), then o[d] = fmaf chain over t ascending. Scaling is 1.0. */
static void attn_head_ref(const float *q, const float *K, const float *V, int lo, int pos, int hd, float *o) {
    static double e[MAXSEQ];
    float mx = -INFINITY;
    for (int t = lo; t <= pos; t++) {
        const float *k = K + (size_t)t * hd;
        float s = 0.0f;
        for (int d = 0; d < hd; d++) s = fmaf(k[d], q[d], s);
        sc[t] = s; if (s > mx) mx = s;
    }
    double sum = 0;
    for (int t = lo; t <= pos; t++) { e[t] = g4_exp((double)sc[t] - (double)mx); sum += e[t]; }
    for (int d = 0; d < hd; d++) o[d] = 0.0f;
    for (int t = lo; t <= pos; t++) {
        float p = (float)(e[t] / sum);
        const float *v = V + (size_t)t * hd;
        for (int d = 0; d < hd; d++) o[d] = fmaf(p, v[d], o[d]);
    }
}
/* g[i] = gelu(g[i]) * u[i]  (GeGLU of the MLP, and of the per-layer-input gate) */
static void geglu_ref(float *g, const float *u, int n) { for (int i = 0; i < n; i++) g[i] = gelu(g[i]) * u[i]; }
static void (*G_ATTN)(const float *, const float *, const float *, int, int, int, float *) = attn_head_ref;
static void (*G_GEGLU)(float *, const float *, int) = geglu_ref;
/* all NH query heads against the single kv head; the engine does them in one pass */
static void attn_all_ref(const float *q, const float *K, const float *V, int lo, int pos, int hd, float *o) {
    for (int j = 0; j < NH; j++) G_ATTN(q + j * hd, K, V, lo, pos, hd, o + j * hd);
}
static void (*G_ATTNALL)(const float *, const float *, const float *, int, int, int, float *) = attn_all_ref;
/* MLP body on the quantized input in XQ: gate+up, GeGLU, Q8_K, down -> out. The
   engine runs the whole chain on the GPU (one synchronization); same arithmetic. */
static void mlp_ref(const layer_t *L, int inter, float *out) {
    const tensor_t *ts[2] = {L->gate, L->up};
    float *ys[2] = {gb, ub};
    G_Q4N(2, ts, XQ, ys);
    G_GEGLU(gb, ub, inter);
    q8k(gb, XQ, inter);
    G_Q4(L->down, XQ, out);
}
static void (*G_MLP)(const layer_t *, int, float *) = mlp_ref;

static void __attribute__((unused)) forward(int tok, int pos) {
    /* embedding and per-layer inputs */
    const tensor_t *emb = tens("token_embd.weight");
    row_q6k(emb, tok, xs);
    for (int i = 0; i < H; i++) xs[i] = (float)((double)xs[i] * sqrt(1536.0));
    G_F16(tens("per_layer_model_proj.weight"), xs, projb);
    row_q6k(tens("per_layer_token_embd.weight"), tok, rowe);
    const float *pnw = f32t(tens("per_layer_proj_norm.weight"));
    for (int l = 0; l < NL; l++) {
        double sl[PLE], ss = 0;
        for (int i = 0; i < PLE; i++) { sl[i] = (double)projb[l * PLE + i] * (1.0 / sqrt(1536.0)); ss += sl[i] * sl[i]; }
        double r = sqrt(ss / PLE + EPS);
        for (int i = 0; i < PLE; i++) {
            double nrm = sl[i] / r * (double)pnw[i];
            double tk = (double)rowe[l * PLE + i] * 16.0;
            pin[l * PLE + i] = (float)((nrm + tk) / sqrt(2.0));
        }
    }
    int store_s = -1, store_f = -1;
    for (int i = 0; i < FIRST_SHARED; i++) { if (SLIDING[i]) store_s = i; else store_f = i; }

    for (int il = 0; il < NL; il++) {
        int full = !SLIDING[il], hd = full ? HD_F : HD_S, inter = il >= FIRST_SHARED ? 2 * INTER : INTER;
        const layer_t *L = &LT[il];
        const float *cs = rope_cs(full, 0, pos), *sn = rope_cs(full, 1, pos);
        /* attention */
        G_RMS(xs, L->n_attn, hb, H);
        q8k(hb, XQ, H);
        if (il < FIRST_SHARED) {               /* q, k, v share the quantized input */
            const tensor_t *ts[3] = {L->q, L->k, L->v};
            float *ys[3] = {qb, kb, vb};
            G_Q4N(3, ts, XQ, ys);
        } else G_Q4(L->q, XQ, qb);
        const float *qn = L->n_q;
        for (int j = 0; j < NH; j++) { float t[HD_F]; G_RMS(qb + j * hd, qn, t, hd); memcpy(qb + j * hd, t, hd * 4); rope(qb + j * hd, cs, sn, hd); }
        int src = il < FIRST_SHARED ? il : (full ? store_f : store_s);
        if (il < FIRST_SHARED) {
            float t[HD_F];
            G_RMS(kb, L->n_k, t, hd); rope(t, cs, sn, hd);
            memcpy(KC[il] + (size_t)pos * hd, t, hd * 4);
            G_RMSNS(vb, hd);
            memcpy(VC[il] + (size_t)pos * hd, vb, hd * 4);
        }
        int lo = full ? 0 : (pos + 1 > WINDOW ? pos + 1 - WINDOW : 0);
        G_ATTNALL(qb, KC[src], VC[src], lo, pos, hd, ab);
        q8k(ab, XQ, NH * hd);
        G_Q4(L->o, XQ, hb);
        { float t[H]; G_RMS(hb, L->n_post_attn, t, H); for (int i = 0; i < H; i++) xs[i] = xs[i] + t[i]; }
        /* MLP */
        G_RMS(xs, L->n_ffn, hb, H);
        q8k(hb, XQ, H);
        G_MLP(L, inter, hb);
        { float t[H]; G_RMS(hb, L->n_post_ffw, t, H); for (int i = 0; i < H; i++) xs[i] = xs[i] + t[i]; }
        /* per-layer input */
        q8k(xs, XQ, H);
        G_Q4(L->ig, XQ, pleg);
        G_GEGLU(pleg, pin + il * PLE, PLE);
        q8k(pleg, XQ, PLE);
        G_Q4(L->proj, XQ, hb);
        { float t[H]; G_RMS(hb, L->n_post, t, H); for (int i = 0; i < H; i++) xs[i] = xs[i] + t[i]; }
        for (int i = 0; i < H; i++) xs[i] = xs[i] * L->scale;
    }
    G_RMS(xs, f32t(tens("output_norm.weight")), hb, H);
    q8k(hb, XQ, H);
    G_HEAD(emb, XQ, logits);   /* raw logits; soft-cap applied by the caller */
}
static float __attribute__((unused)) capped(float l) { return (float)(SOFTCAP * g4_tanh((double)l / SOFTCAP)); }

#ifndef G4REF_NO_MAIN   /* g4gemv_test.c includes this file for its numerics */
int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <gguf> <rope.bin> <ids...> [--gen N]\n", argv[0]); return 2; }
    int gen = 0, n = 0;
    static int ids[MAXSEQ];
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--gen") && i + 1 < argc) { gen = atoi(argv[++i]); continue; }
        ids[n++] = atoi(argv[i]);
    }
    gguf_open(argv[1]);
    { int c = 0; for (int i = 0; i < NL; i++) c += !SLIDING[i];
      if (c != 7) { fprintf(stderr, "sliding pattern: %d full layers, want 7\n", c); return 2; } }
    layer_table();
    { size_t want = (size_t)MAXSEQ * (128 + 256) * 2 * 4;
      ROPE = malloc(want); FILE *f = fopen(argv[2], "rb");
      if (!f || fread(ROPE, 1, want, f) != want) { fprintf(stderr, "rope table %s: want %zu bytes\n", argv[2], want); return 2; }
      fclose(f); }
    int pos = 0, n_prompt = n;
    while (pos < MAXSEQ) {
        int tok = ids[pos];
        forward(tok, pos);
        int top = 0;
        for (int v = 1; v < VOCAB; v++) if (logits[v] > logits[top]) top = v;
        /* greedy: argmax of the RAW logits (first max wins); the cap is monotone,
           and applying it first could merge distinct values in float32 */
        printf("step %d id_in=%d top=%d logit=%.4f\n", pos, tok, top, capped(logits[top]));
        fflush(stdout);
        pos++;
        if (pos >= n) {
            if (n - n_prompt >= gen || top == 1 || top == 106) break;   /* gen = tokens generated */
            ids[n++] = top;
        }
    }
    return 0;
}
#endif
