// fwd35_ref.c: portable host reference for Ternary-Bonsai-2 (arch qwen35, PTQ1_0 weights).
// It mirrors the PrismML fork's x86 CPU arithmetic op for op (llama-simple style greedy,
// f16 K/V cache, flash attention), so its greedy stream equals the fork's bit for bit and
// it can serve as the parity oracle for the Nano GPU engine.
//
//   build: gcc -O2 -fopenmp -ffp-contract=off -o fwd35_ref fwd35_ref.c -lm
//   usage: fwd35_ref model.gguf N id id ...      (prints N greedy ids, one per line)
//   env:   R_MARGIN=1  "# step s id logit margin (2nd)" lines on stderr
//          R_DUMP=f    raw f32 logits of every step appended to f
//
// The prompt is evaluated as one batch, like llama_decode on the whole prompt: every op is
// per token and in token order, except the F32 ssm_alpha/ssm_beta projections, which the
// fork runs through tinyBLAS when the batch has >= 2 rows and through ggml_vec_dot_f32 for
// a single row. Those two dot orders differ, so the prompt and decode paths are kept apart.
//
// Every fused multiply-add below is an explicit fmaf() at a place where the fork's x86
// build emits vfmadd (SIMD intrinsics or compiler contraction); everything else is
// separately rounded, which is why the file is built with -ffp-contract=off. libm calls go
// through glibc_mathf.h, so the reference returns the same bits on x86 and on the Nano.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "glibc_mathf.h"   // libm expf/logf/sinf/cosf, bit-identical on every platform

#ifndef ROPE_DST1_X0SIN
#define ROPE_DST1_X0SIN 0   // 0: dst1 = fma(x0, sin, round(x1*cos)), as the x86 build compiles it (verified)
#endif
#ifndef CONV_FMA
#define CONV_FMA 0          // ssm_conv: products rounded, then summed
#endif
#ifndef FA_S_FMA
#define FA_S_FMA 1          // flash attention: S = fma(S, ms, vs)
#endif

static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float bitsf(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

/* ---------------- fp16 (IEEE binary16, round to nearest even) ---------------- */
static float f16f(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e == 0) {
        if (!m) return bitsf(s);
        return bitsf(fbits(ldexpf((float)m, -24)) | s);
    }
    if (e == 31) return bitsf(s | 0x7f800000u | (m << 13));
    return bitsf(s | ((e + 112) << 23) | (m << 13));
}
static uint16_t ff16(float ff) {
    const uint32_t f32infty = 255u << 23, f16max = (127u + 16) << 23;
    const uint32_t denorm_magic = ((127u - 15) + (23 - 10) + 1) << 23;
    uint32_t f = fbits(ff), sign = f & 0x80000000u;
    uint16_t o;
    f ^= sign;
    if (f >= f16max) o = (f > f32infty) ? 0x7e00 : 0x7c00;
    else if (f < (113u << 23)) {
        float t = bitsf(f) + bitsf(denorm_magic);
        o = (uint16_t)(fbits(t) - denorm_magic);
    } else {
        uint32_t mant_odd = (f >> 13) & 1;
        f += ((uint32_t)(15 - 127) << 23) + 0xfff;
        f += mant_odd;
        o = (uint16_t)(f >> 13);
    }
    return o | (uint16_t)(sign >> 16);
}

/* ---------------- GGUF ---------------- */
typedef struct { const char *k; uint64_t kl; uint32_t t; const uint8_t *v; } KV;
typedef struct { const char *n; uint64_t nl; uint32_t nd, type; int64_t ne[4]; uint64_t off; const uint8_t *d; } TI;
static const uint8_t *G; static size_t GN;
static KV *kvs; static uint64_t nkv;
static TI *tis; static uint64_t nti;

static uint64_t rd64(const uint8_t **p) { uint64_t v; memcpy(&v, *p, 8); *p += 8; return v; }
static uint32_t rd32(const uint8_t **p) { uint32_t v; memcpy(&v, *p, 4); *p += 4; return v; }
static const size_t gsz[13] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
static void skipv(const uint8_t **p, uint32_t t) {
    if (t == 8) { uint64_t l = rd64(p); *p += l; return; }
    if (t == 9) { uint32_t at = rd32(p); uint64_t n = rd64(p);
        for (uint64_t i = 0; i < n; i++) skipv(p, at);
        return; }
    *p += gsz[t];
}
static void gguf_open(const char *path) {
    int fd = open(path, O_RDONLY); struct stat st;
    if (fd < 0 || fstat(fd, &st)) { perror(path); exit(1); }
    GN = st.st_size;
    G = mmap(NULL, GN, PROT_READ, MAP_PRIVATE, fd, 0);
    if (G == MAP_FAILED) { perror("mmap"); exit(1); }
    const uint8_t *p = G;
    if (memcmp(p, "GGUF", 4)) { fprintf(stderr, "not gguf\n"); exit(1); }
    p += 8;
    nti = rd64(&p); nkv = rd64(&p);
    kvs = calloc(nkv, sizeof(KV)); tis = calloc(nti, sizeof(TI));
    for (uint64_t i = 0; i < nkv; i++) {
        kvs[i].kl = rd64(&p); kvs[i].k = (const char *)p; p += kvs[i].kl;
        kvs[i].t = rd32(&p); kvs[i].v = p; skipv(&p, kvs[i].t);
    }
    for (uint64_t i = 0; i < nti; i++) {
        TI *t = &tis[i];
        t->nl = rd64(&p); t->n = (const char *)p; p += t->nl;
        t->nd = rd32(&p);
        for (int j = 0; j < 4; j++) t->ne[j] = 1;
        for (uint32_t j = 0; j < t->nd; j++) t->ne[j] = (int64_t)rd64(&p);
        t->type = rd32(&p); t->off = rd64(&p);
    }
    uint64_t align = 32;
    for (uint64_t i = 0; i < nkv; i++)
        if (kvs[i].kl == 17 && !memcmp(kvs[i].k, "general.alignment", 17)) align = *(const uint32_t *)kvs[i].v;
    size_t base = ((size_t)(p - G) + align - 1) / align * align;
    for (uint64_t i = 0; i < nti; i++) tis[i].d = G + base + tis[i].off;
}
static const KV *kv_get(const char *k) {
    size_t l = strlen(k);
    for (uint64_t i = 0; i < nkv; i++) if (kvs[i].kl == l && !memcmp(kvs[i].k, k, l)) return &kvs[i];
    return NULL;
}
static int64_t kv_int(const char *k, int64_t def) {
    const KV *e = kv_get(k); if (!e) return def;
    switch (e->t) {
        case 0: return *(const uint8_t *)e->v;  case 1: return *(const int8_t *)e->v;
        case 4: return *(const uint32_t *)e->v; case 5: return *(const int32_t *)e->v;
        case 7: return *(const uint8_t *)e->v;
        case 10: return (int64_t)*(const uint64_t *)e->v; case 11: return *(const int64_t *)e->v;
    }
    fprintf(stderr, "kv %s: type %u\n", k, e->t); exit(1);
}
static float kv_f32(const char *k, float def) {
    const KV *e = kv_get(k); if (!e) return def;
    if (e->t != 6) { fprintf(stderr, "kv %s not f32\n", k); exit(1); }
    return *(const float *)e->v;
}
// int32 array: returns element pointer and count
static const int32_t *kv_i32arr(const char *k, uint64_t *n) {
    const KV *e = kv_get(k); if (!e || e->t != 9) return NULL;
    const uint8_t *p = e->v; uint32_t at = rd32(&p); *n = rd64(&p);
    if (at != 5 && at != 4) { fprintf(stderr, "kv %s: array type %u\n", k, at); exit(1); }
    return (const int32_t *)p;
}
static const TI *tget(const char *n, int must) {
    size_t l = strlen(n);
    for (uint64_t i = 0; i < nti; i++) if (tis[i].nl == l && !memcmp(tis[i].n, n, l)) return &tis[i];
    if (must) { fprintf(stderr, "missing tensor %s\n", n); exit(1); }
    return NULL;
}

/* ---------------- model ---------------- */
#define T_F32 0
#define T_PTQ1 143
#define PTQ_BYTES 28   /* qs[24] qh[2] d(fp16) per 128 weights */
typedef struct { const uint8_t *w; int K, N; } PT;
typedef struct {
    int full;
    const float *attn_norm, *post_norm;
    PT qkv, gate, ssm_out, wq, wk, wv, wo, ffn_gate, ffn_up, ffn_down;
    const float *ssm_a, *ssm_alpha, *ssm_beta, *conv1d, *dt_bias, *ssm_norm, *q_norm, *k_norm;
} Layer;

static int NL, E, FF, NH, NKV, HD = 256, NV, NK, DS = 128, INNER, CH, NROT, VOCAB, BS, GROUPED, TIED, CTX = 4096;
static float EPS, ROPE_BASE;
static Layer *Ls;
static PT tok_embd, out_w;
static const float *out_norm;
static int nsign; static int sign_w[8]; static float *sign_v[8];
static int8_t LUT[256][5];

static PT pt(const char *n) {
    const TI *t = tget(n, 1);
    if (t->type != T_PTQ1) { fprintf(stderr, "%s: type %u, want PTQ1_0\n", n, t->type); exit(1); }
    PT p = { t->d, (int)t->ne[0], (int)t->ne[1] }; return p;
}
static const float *f32t(const char *n, int must) {
    const TI *t = tget(n, must); if (!t) return NULL;
    if (t->type != T_F32) { fprintf(stderr, "%s: type %u, want F32\n", n, t->type); exit(1); }
    return (const float *)t->d;
}
static const float *signs_for(int w) {
    for (int i = 0; i < nsign; i++) if (sign_w[i] == w) return sign_v[i];
    fprintf(stderr, "no sign vector for width %d\n", w); exit(1);
}

static void load(void) {
    char a[64];
    NL = kv_int("qwen35.block_count", 0); E = kv_int("qwen35.embedding_length", 0);
    FF = kv_int("qwen35.feed_forward_length", 0);
    NH = kv_int("qwen35.attention.head_count", 0); NKV = kv_int("qwen35.attention.head_count_kv", 0);
    HD = kv_int("qwen35.attention.key_length", 256);
    NK = kv_int("qwen35.ssm.group_count", 0); NV = kv_int("qwen35.ssm.time_step_rank", 0);
    DS = kv_int("qwen35.ssm.state_size", 128); INNER = kv_int("qwen35.ssm.inner_size", 0);
    CH = INNER + 2 * NK * DS;
    NROT = kv_int("qwen35.rope.dimension_count", 64);
    EPS = kv_f32("qwen35.attention.layer_norm_rms_epsilon", 1e-6f);
    ROPE_BASE = kv_f32("qwen35.rope.freq_base", 1e7f);
    int fai = kv_int("qwen35.full_attention_interval", 4);
    BS = kv_int("prism.hadamard.block_size", 0);
    GROUPED = kv_int("prism.hadamard.gdn_v_grouped", 0);
    TIED = kv_int("prism.hadamard.tied_output", 0);
    if (!BS || kv_int("qwen35.ssm.conv_kernel", 4) != 4 || HD != 256 || DS != 128 || INNER != NV * DS) {
        fprintf(stderr, "unsupported config\n"); exit(1);
    }
    uint64_t nw, nv; const int32_t *sw = kv_i32arr("prism.hadamard.sign_widths", &nw);
    const int32_t *sv = kv_i32arr("prism.hadamard.sign_values", &nv);
    uint64_t o = 0;
    for (uint64_t i = 0; i < nw; i++) {
        int32_t swi; memcpy(&swi, sw + i, 4); sign_w[nsign] = swi; sign_v[nsign] = malloc(sizeof(float) * swi);
        for (int j = 0; j < swi; j++) { int32_t t; memcpy(&t, sv + o + j, 4); sign_v[nsign][j] = (float)t; }
        o += swi; nsign++;
    }
    if (o != nv) { fprintf(stderr, "sign_values length\n"); exit(1); }
    tok_embd = pt("token_embd.weight"); VOCAB = tok_embd.N;
    out_w = TIED ? tok_embd : pt("output.weight");
    out_norm = f32t("output_norm.weight", 1);
    Ls = calloc(NL, sizeof(Layer));
    for (int l = 0; l < NL; l++) {
        Layer *L = &Ls[l];
        L->full = ((l + 1) % fai) == 0;
#define N_(s) (snprintf(a, sizeof a, "blk.%d." s, l), a)
        L->attn_norm = f32t(N_("attn_norm.weight"), 1);
        L->post_norm = f32t(N_("post_attention_norm.weight"), 1);
        L->ffn_gate = pt(N_("ffn_gate.weight")); L->ffn_up = pt(N_("ffn_up.weight"));
        L->ffn_down = pt(N_("ffn_down.weight"));
        if (L->full) {
            L->wq = pt(N_("attn_q.weight")); L->wk = pt(N_("attn_k.weight"));
            L->wv = pt(N_("attn_v.weight")); L->wo = pt(N_("attn_output.weight"));
            L->q_norm = f32t(N_("attn_q_norm.weight"), 1); L->k_norm = f32t(N_("attn_k_norm.weight"), 1);
        } else {
            L->qkv = pt(N_("attn_qkv.weight")); L->gate = pt(N_("attn_gate.weight"));
            L->ssm_out = pt(N_("ssm_out.weight"));
            L->ssm_a = f32t(N_("ssm_a"), 1); L->ssm_alpha = f32t(N_("ssm_alpha.weight"), 1);
            L->ssm_beta = f32t(N_("ssm_beta.weight"), 1); L->conv1d = f32t(N_("ssm_conv1d.weight"), 1);
            L->dt_bias = f32t(N_("ssm_dt.bias"), 1); L->ssm_norm = f32t(N_("ssm_norm.weight"), 1);
        }
#undef N_
    }
    static const uint8_t pow3[5] = {1, 3, 9, 27, 81};
    for (int b = 0; b < 256; b++)
        for (int n = 0; n < 5; n++) {
            uint8_t v = (uint8_t)(b * pow3[n]);
            LUT[b][n] = (int8_t)((((uint16_t)v * 3) >> 8) - 1);
        }
}

/* ---------------- arithmetic primitives (fork x86 order) ---------------- */
// ggml_vec_dot_f32 / ggml_vec_dot_f16 AVX2: 4 accumulators x 8 lanes, then the F32x8 reduce.
static float dotg(const float *x, const float *y, int n) {
    float a[4][8] = {{0}};
    for (int i = 0; i < n; i += 32)
        for (int j = 0; j < 4; j++)
            for (int l = 0; l < 8; l++) a[j][l] = fmaf(x[i + 8 * j + l], y[i + 8 * j + l], a[j][l]);
    float t[4];
    for (int l = 0; l < 8; l++) { a[0][l] += a[2][l]; a[1][l] += a[3][l]; a[0][l] += a[1][l]; }
    for (int l = 0; l < 4; l++) t[l] = a[0][l] + a[0][l + 4];
    return (t[0] + t[1]) + (t[2] + t[3]);
}
// tinyBLAS<8, __m256> F32 gemm element: one 8-lane fma accumulator, hsum.
static float dott(const float *a, const float *b, int k) {
    float c[8] = {0}, h[4];
    for (int l = 0; l < k; l += 8)
        for (int i = 0; i < 8; i++) c[i] = fmaf(a[l + i], b[l + i], c[i]);
    for (int i = 0; i < 4; i++) h[i] = c[i + 4] + c[i];
    return (h[0] + h[2]) + (h[1] + h[3]);
}
// ggml_v_expf (AVX2+FMA), one lane
static float vexpf(float x) {
    const float r = 0x1.8p23f;
    const float z = fmaf(x, 0x1.715476p+0f, r);
    const float n = z - r;
    const float b = fmaf(-n, 0x1.7f7d1cp-20f, fmaf(-n, 0x1.62e4p-1f, x));
    const uint32_t e = fbits(z) << 23;
    const float k = bitsf(e + fbits(1.0f));
    const float u = b * b;
    const float j = fmaf(fmaf(fmaf(0x1.0e4020p-7f, b, 0x1.573e2ep-5f), u,
                              fmaf(0x1.555e66p-3f, b, 0x1.fffdb6p-2f)),
                         u, 0x1.ffffecp-1f * b);
    if (!(fabsf(n) > 126.0f)) return fmaf(j, k, k);
    const uint32_t g = (n <= 0.0f) ? 0x82000000u : 0u;
    const float s1 = bitsf(g + 0x7f000000u);
    const float s2 = bitsf(e - g);
    if (fabsf(n) > 192.0f) return s1 * s1;
    return fmaf(s2, j, s2) * s1;
}
static float vsilu(float x) { return x / (1.0f + vexpf(0.0f - x)); }
static float sigm(float x) { return 1.0f / (1.0f + gm_expf(-x)); }

// rms_norm + mul: double sum of rounded squares, (x*scale)*w
static void rmsnorm(const float *x, const float *w, float *y, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)(x[i] * x[i]);
    const float mean = (float)(s / n);
    const float sc = 1.0f / sqrtf(mean + EPS);
    for (int i = 0; i < n; i++) y[i] = (x[i] * sc) * w[i];
}
static void l2norm(float *x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)(x[i] * x[i]);
    const float sc = 1.0f / fmaxf(sqrtf((float)s), EPS);
    for (int i = 0; i < n; i++) x[i] = x[i] * sc;
}
// normalized Sylvester-Walsh-Hadamard per block: scale first, then butterflies
static void fwht(float *x, int n) {
    const float sc = 1.0f / sqrtf((float)BS);
    for (int b0 = 0; b0 < n; b0 += BS) {
        float *v = x + b0;
        for (int j = 0; j < BS; j++) v[j] = v[j] * sc;
        for (int len = 1; len < BS; len <<= 1)
            for (int i = 0; i < BS; i += 2 * len)
                for (int j = 0; j < len; j++) {
                    const float u = v[i + j], w = v[i + len + j];
                    v[i + j] = u + w; v[i + len + j] = u - w;
                }
    }
}
// activation side of a Hadamard-folded matmul: [GDN head regroup], signs, FWHT
static void xform(const float *x, int n, float *o, int perm) {
    if (perm) {
        const int rep = NV / NK;
        for (int k = 0; k < NK; k++)
            for (int r = 0; r < rep; r++)
                memcpy(o + DS * (r + rep * k), x + DS * (k + NK * r), sizeof(float) * DS);
    } else memcpy(o, x, sizeof(float) * n);
    const float *s = signs_for(n);
    for (int i = 0; i < n; i++) o[i] = o[i] * s[i];
    fwht(o, n);
}
// quantize_row_q8_0 (AVX): d = amax/127 as fp16, q = round_half_even(x * 127/amax)
typedef struct { int8_t *q; float *d; } Q8;
static void q8(const float *x, int n, Q8 *o) {
    for (int b = 0; b < n / 32; b++) {
        float amax = 0.0f;
        for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(x[32 * b + j]));
        const float d = amax / 127.0f;
        o->d[b] = f16f(ff16(d));
        const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
        for (int j = 0; j < 32; j++) o->q[32 * b + j] = (int8_t)rintf(x[32 * b + j] * id);
    }
}
// ggml_vec_dot_ptq1_0_q8_0 (x86): exact int sums per 32, fma chain per block, fma over blocks
static float dot_ptq(const uint8_t *row, int nb, const Q8 *y) {
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = row + PTQ_BYTES * i;
        const int8_t *q = y->q + 128 * i;
        int s[4] = {0, 0, 0, 0};
        for (int n = 0; n < 5; n++)
            for (int m = 0; m < 16; m++) { const int ix = n * 16 + m; s[ix >> 5] += LUT[b[m]][n] * q[ix]; }
        for (int n = 0; n < 5; n++)
            for (int m = 0; m < 8; m++) { const int ix = 80 + n * 8 + m; s[ix >> 5] += LUT[b[16 + m]][n] * q[ix]; }
        for (int n = 0; n < 4; n++)
            for (int h = 0; h < 2; h++) { const int ix = 120 + n * 2 + h; s[3] += LUT[b[24 + h]][n] * q[ix]; }
        float sumi = 0.0f;
        for (int k = 0; k < 4; k++) sumi = fmaf(y->d[4 * i + k], (float)s[k], sumi);
        uint16_t dh; memcpy(&dh, b + 26, 2);
        sumf = fmaf(f16f(dh), sumi, sumf);
    }
    return sumf;
}
static Q8 qa;   // scratch for the quantized activation
static void gemv(const PT *W, const float *xt, float *y) {   // xt: already transformed
    q8(xt, W->K, &qa);
    const int nb = W->K / 128;
    const size_t rb = (size_t)nb * PTQ_BYTES;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < W->N; r++) y[r] = dot_ptq(W->w + rb * r, nb, &qa);
}

/* ---------------- state ---------------- */
static float **conv_st;   // per linear layer: [3][CH], oldest first
static float **ssm_st;    // per linear layer: [NV][128][128], row j = column j of S
static uint16_t **kc, **vc; // per full layer: [CTX][NKV*HD] f16

static void embed(int id, float *x) {
    const int nb = E / 128;
    const uint8_t *row = tok_embd.w + (size_t)id * nb * PTQ_BYTES;
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = row + PTQ_BYTES * i;
        uint16_t dh; memcpy(&dh, b + 26, 2);
        const float d = f16f(dh);
        float *y = x + 128 * i; int o = 0;
        for (int n = 0; n < 5; n++) for (int m = 0; m < 16; m++) y[o++] = (float)LUT[b[m]][n] * d;
        for (int n = 0; n < 5; n++) for (int m = 0; m < 8; m++) y[o++] = (float)LUT[b[16 + m]][n] * d;
        for (int n = 0; n < 4; n++) for (int h = 0; h < 2; h++) y[o++] = (float)LUT[b[24 + h]][n] * d;
    }
    fwht(x, E);
    const float *s = signs_for(E);
    for (int i = 0; i < E; i++) x[i] = x[i] * s[i];
}

// R_TDUMP=file R_TSTEP=s: dump named intermediates of the forward that yields step s's
// logits, in fork_greedy's FG_TDUMP record format, to localize a divergence.
static FILE *tdf; static int tcur = -1, tstep = -2;
static void tdump(const char *fmt, int il, const float *v, int n) {
    if (!tdf || tcur != tstep) return;
    char nm[64]; snprintf(nm, sizeof nm, fmt, il);
    uint32_t nl = strlen(nm); uint64_t ne[4] = {(uint64_t)n, 1, 1, 1}, nn = n;
    fwrite(&nl, 4, 1, tdf); fwrite(nm, 1, nl, tdf); fwrite(ne, 8, 4, tdf); fwrite(&nn, 8, 1, tdf); fwrite(v, 4, n, tdf);
}

static float *xn, *xt, *t1, *t2, *t3, *t4, *t5, *ffa, *ffb;

static void layer_linear(int l, Layer *L, float *x, int batch) {
    float *qkv = t1, *z = t2, *o = t3;
    float beta[64], alpha[64];
    rmsnorm(x, L->attn_norm, xn, E);
    xform(xn, E, xt, 0);
    gemv(&L->qkv, xt, qkv);
    gemv(&L->gate, xt, z);
    for (int h = 0; h < NV; h++) {
        beta[h]  = batch ? dott(L->ssm_beta + (size_t)h * E, xn, E)  : dotg(L->ssm_beta + (size_t)h * E, xn, E);
        alpha[h] = batch ? dott(L->ssm_alpha + (size_t)h * E, xn, E) : dotg(L->ssm_alpha + (size_t)h * E, xn, E);
    }
    // causal conv over the window [s0 s1 s2 x], then silu
    float *cs = conv_st[l], *cv = t4;
    for (int c = 0; c < CH; c++) {
        const float *k = L->conv1d + 4 * c;
        const float w0 = cs[c], w1 = cs[CH + c], w2 = cs[2 * CH + c], w3 = qkv[c];
#if CONV_FMA
        float s = fmaf(w0, k[0], 0.0f); s = fmaf(w1, k[1], s); s = fmaf(w2, k[2], s); s = fmaf(w3, k[3], s);
#else
        float s = 0.0f; s += w0 * k[0]; s += w1 * k[1]; s += w2 * k[2]; s += w3 * k[3];
#endif
        cv[c] = vsilu(s);
        cs[c] = w1; cs[CH + c] = w2; cs[2 * CH + c] = w3;
    }
    tdump("conv_output_silu-%d", l, cv, CH);
    tdump("alpha-%d", l, alpha, NV); tdump("beta-%d", l, beta, NV);
    for (int h = 0; h < 2 * NK; h++) l2norm(cv + DS * h, DS);
    const float *qh = cv, *kh = cv + NK * DS, *vh = cv + 2 * NK * DS;
    const float scale = 1.0f / sqrtf((float)DS);
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < NV; h++) {
        float *M = ssm_st[l] + (size_t)h * DS * DS, delta[128];
        const float *q = qh + DS * (h % NK), *k = kh + DS * (h % NK), *v = vh + DS * h;
        const float bt = 1.0f / (1.0f + gm_expf(-beta[h]));
        const float xg = alpha[h] + L->dt_bias[h];
        const float g = L->ssm_a[h] * ((xg > 20.0f) ? xg : gm_logf(1.0f + gm_expf(xg)));
        const float eg = gm_expf(g);
        for (int i = 0; i < DS * DS; i++) M[i] = M[i] * eg;
        for (int j = 0; j < DS; j++) delta[j] = (v[j] - dotg(M + j * DS, k, DS)) * bt;
        for (int j = 0; j < DS; j++) for (int i = 0; i < DS; i++) M[j * DS + i] = fmaf(k[i], delta[j], M[j * DS + i]);
        for (int j = 0; j < DS; j++) o[h * DS + j] = dotg(M + j * DS, q, DS) * scale;
    }
    tdump("attn_output-%d", l, o, INNER);
    // gated norm: silu(z) * rms(o, ssm_norm), per v head
    for (int h = 0; h < NV; h++) {
        float nrm[128];
        rmsnorm(o + h * DS, L->ssm_norm, nrm, DS);
        for (int i = 0; i < DS; i++) o[h * DS + i] = vsilu(z[h * DS + i]) * nrm[i];
    }
    xform(o, INNER, xt, GROUPED && NV > NK);
    tdump("final_output-%d", l, o, INNER);
    gemv(&L->ssm_out, xt, t5);
    tdump("linear_attn_out-%d", l, t5, E);
    for (int i = 0; i < E; i++) x[i] = t5[i] + x[i];
    tdump("attn_residual-%d", l, x, E);
}

static float ROPE_TS;   // powf(base, -2/n_rot), printed in hex at start-up to compare platforms
static void rope(float *v, int pos) {
    const float ts = ROPE_TS;
    float th = (float)pos;
    const int half = NROT / 2;
    for (int i = 0; i < half; i++) {
        const float c = gm_cosf(th), s = gm_sincos_sinf(th);  // fork: GCC-fused sincosf
        const float x0 = v[i], x1 = v[i + half];
        v[i] = fmaf(x0, c, -(x1 * s));
#if ROPE_DST1_X0SIN
        v[i + half] = fmaf(x1, c, x0 * s);
#else
        v[i + half] = fmaf(x0, s, x1 * c);
#endif
        th *= ts;
    }
}

static void layer_full(int l, Layer *L, float *x, int pos) {
    const int fi = l / 4;
    float *qf = t1, *kk = t2, *vv = t3, *att = t4;
    rmsnorm(x, L->attn_norm, xn, E);
    xform(xn, E, xt, 0);
    gemv(&L->wq, xt, qf);
    gemv(&L->wk, xt, kk);
    gemv(&L->wv, xt, vv);
    float *q = ffa;   // NH*HD normed + roped queries
    for (int h = 0; h < NH; h++) { rmsnorm(qf + h * 2 * HD, L->q_norm, q + h * HD, HD); rope(q + h * HD, pos); }
    tdump("Qcur-%d", l, q, NH * HD);
    float kdump[2048];
    for (int h = 0; h < NKV; h++) {
        float kn[256]; rmsnorm(kk + h * HD, L->k_norm, kn, HD); rope(kn, pos);
        memcpy(kdump + h * HD, kn, sizeof kn);
        for (int d = 0; d < HD; d++) { kc[fi][(size_t)pos * NKV * HD + h * HD + d] = ff16(kn[d]);
                                       vc[fi][(size_t)pos * NKV * HD + h * HD + d] = ff16(vv[h * HD + d]); }
    }
    tdump("k_cache_in-%d", l, kdump, NKV * HD);
    static float fao[8192];
    const float scale = 1.0f / sqrtf((float)HD);
    const int grp = NH / NKV;
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < NH; h++) {
        float qv[256], kv[256], acc[256];
        uint16_t vk[256];
        for (int d = 0; d < HD; d++) qv[d] = f16f(ff16(q[h * HD + d]));
        memset(vk, 0, sizeof vk);
        float S = 0.0f, M = -INFINITY;
        const int g = h / grp;
        for (int ic = 0; ic <= pos; ic++) {
            const uint16_t *kr = kc[fi] + (size_t)ic * NKV * HD + g * HD;
            const uint16_t *vr = vc[fi] + (size_t)ic * NKV * HD + g * HD;
            for (int d = 0; d < HD; d++) kv[d] = f16f(kr[d]);
            float s = dotg(kv, qv, HD);
            s = s * scale;
            const float Mold = M;
            float ms = 1.0f, vs = 1.0f;
            if (s > M) {
                M = s; ms = gm_expf(Mold - M);
                for (int d = 0; d < HD; d++) vk[d] = ff16(f16f(vk[d]) * ms);
            } else vs = gm_expf(s - M);
            for (int d = 0; d < HD; d++) vk[d] = ff16(fmaf(f16f(vr[d]), vs, f16f(vk[d])));
#if FA_S_FMA
            S = fmaf(S, ms, vs);
#else
            S = S * ms + vs;
#endif
        }
        const float Si = S == 0.0f ? 0.0f : 1.0f / S;
        for (int d = 0; d < HD; d++) acc[d] = f16f(vk[d]) * Si;
        memcpy(fao + h * HD, acc, sizeof acc);
        const float *gt = qf + h * 2 * HD + HD;
        for (int d = 0; d < HD; d++) att[h * HD + d] = acc[d] * sigm(gt[d]);
    }
    xform(att, NH * HD, xt, 0);
    tdump("fa_out-%d", l, fao, NH * HD);
    tdump("attn_gated-%d", l, att, NH * HD);
    gemv(&L->wo, xt, t5);
    for (int i = 0; i < E; i++) x[i] = t5[i] + x[i];
    tdump("attn_residual-%d", l, x, E);
}

static int ffn_il;
static void ffn(Layer *L, float *x) {
    rmsnorm(x, L->post_norm, xn, E);
    xform(xn, E, xt, 0);
    gemv(&L->ffn_gate, xt, ffa);
    gemv(&L->ffn_up, xt, ffb);
    for (int i = 0; i < FF; i++) ffa[i] = vsilu(ffa[i]) * ffb[i];
    xform(ffa, FF, xt, 0);
    gemv(&L->ffn_down, xt, t5);
    tdump("ffn_out-%d", ffn_il, t5, E);
    for (int i = 0; i < E; i++) x[i] = t5[i] + x[i];
    tdump("l_out-%d", ffn_il, x, E);
}

static float *logits;
static void forward(int id, int pos, int batch, int want_logits) {
    static float *x;
    if (!x) x = malloc(sizeof(float) * E);
    embed(id, x);
    for (int l = 0; l < NL; l++) {
        Layer *L = &Ls[l];
        if (L->full) layer_full(l, L, x, pos); else layer_linear(l, L, x, batch);
        ffn_il = l; ffn(L, x);
    }
    if (!want_logits) return;
    rmsnorm(x, out_norm, xn, E);
    xform(xn, E, xt, 0);
    gemv(&out_w, xt, logits);
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model.gguf N id id ...\n", argv[0]); return 1; }
    gguf_open(argv[1]);
    load();
    const int N = atoi(argv[2]), np = argc - 3;
    const int mx = E > FF ? (E > CH ? (E > FF ? E : FF) : CH) : FF;
    int big = FF; if (CH > big) big = CH; if (2 * NH * HD > big) big = 2 * NH * HD; if (INNER > big) big = INNER;
    (void)mx;
    xn = malloc(sizeof(float) * big); xt = malloc(sizeof(float) * big);
    t1 = malloc(sizeof(float) * big); t2 = malloc(sizeof(float) * big); t3 = malloc(sizeof(float) * big);
    t4 = malloc(sizeof(float) * big); t5 = malloc(sizeof(float) * big);
    ffa = malloc(sizeof(float) * big); ffb = malloc(sizeof(float) * big);
    qa.q = malloc(big > VOCAB ? big : big); qa.d = malloc(sizeof(float) * big / 32);
    logits = malloc(sizeof(float) * VOCAB);
    conv_st = calloc(NL, sizeof(float *)); ssm_st = calloc(NL, sizeof(float *));
    kc = calloc(NL, sizeof(uint16_t *)); vc = calloc(NL, sizeof(uint16_t *));
    for (int l = 0; l < NL; l++) {
        if (Ls[l].full) {
            kc[l / 4] = calloc((size_t)CTX * NKV * HD, 2); vc[l / 4] = calloc((size_t)CTX * NKV * HD, 2);
        } else {
            conv_st[l] = calloc((size_t)3 * CH, sizeof(float));
            ssm_st[l] = calloc((size_t)NV * DS * DS, sizeof(float));
        }
    }
    ROPE_TS = powf(ROPE_BASE, -2.0f / NROT);
    fprintf(stderr, "fwd35_ref: rope theta_scale %a\n", ROPE_TS);
    fprintf(stderr, "fwd35_ref: L=%d E=%d FF=%d H=%d/%d NV=%d NK=%d BS=%d grouped=%d tied=%d vocab=%d\n",
            NL, E, FF, NH, NKV, NV, NK, BS, GROUPED, TIED, VOCAB);
    const char *dump = getenv("R_DUMP");
    const int marg = getenv("R_MARGIN") != NULL;
    if (getenv("R_TDUMP")) { tdf = fopen(getenv("R_TDUMP"), "wb"); tstep = atoi(getenv("R_TSTEP")); }
    FILE *df = dump ? fopen(dump, "wb") : NULL;
    int pos = 0;
    const int batch = np >= 2;
    tcur = 0;
    for (int i = 0; i < np; i++, pos++) forward(atoi(argv[3 + i]), pos, batch, i == np - 1);
    for (int s = 0; s < N; s++) {
        int a = 0;
        for (int i = 1; i < VOCAB; i++) if (logits[i] > logits[a]) a = i;
        if (df) fwrite(logits, 4, VOCAB, df);
        printf("%d\n", a); fflush(stdout);
        if (marg) {
            int a2 = a == 0 ? 1 : 0;
            for (int i = 0; i < VOCAB; i++) if (i != a && logits[i] > logits[a2]) a2 = i;
            fprintf(stderr, "# step %d id %d logit %.4f margin %.4f (2nd %d)\n", s, a, logits[a], logits[a] - logits[a2], a2);
        }
        tcur = s + 1;
        if (s + 1 < N) forward(a, pos++, 0, 1);
    }
    if (df) fclose(df);
    if (tdf) fclose(tdf);
    return 0;
}
