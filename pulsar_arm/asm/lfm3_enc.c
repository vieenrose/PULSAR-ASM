/* lfm3_enc.c -- LFM3 audio encoder, C reference (S3).
 *
 * Reads the encoder part of an .lfm3q8 blob (tools/mkaudio.py), dequantizes the Q8
 * linears once at load (code * scale per 32-element block), and runs the encoder
 * in fp32 with the spec's operation order (.auto/lfm3/enc_ref/enc_numpy.py):
 *   downsampler (3 stride-2 2-D convs, SiLU, flatten c*16+h, lin_out)
 *   17 pre-norm LFM2 layers, full causal attention (offline), RoPE over 96 dims
 *   embedding_norm, then the projector (LayerNorm, Linear, GELU exact, Linear).
 *
 * Usage: lfm3_enc <blob> <input_128xT_f32.bin> <T> <outdir>
 * Writes raw little-endian f32 stage files into outdir: ds_out, layer_00..layer_16,
 * enc_out, proj_out. Build: cc -O2 -std=c99 -o lfm3_enc lfm3_enc.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define HID 768
#define FFN 1792
#define NH 8
#define NKV 4
#define HD 96
#define NLAYER 17
#define PROJ_OUT 2048
#define EPS 1e-5f
#define THETA 1000000.0

typedef struct { char name[128]; int dtype; int ne[4]; int nd; size_t off, ncodes, nscales; } Ent;
typedef struct { Ent *e; size_t n; unsigned char *blob; size_t base; size_t size; } Blob;


/* ---- blob reading ----------------------------------------------------- */
static void die(const char *m) { fprintf(stderr, "lfm3_enc: %s\n", m); exit(2); }

static int jfind(const char *s, const char *key, const char *end) {
    const char *p = strstr(s, key);
    return (p && p < end) ? (int)(p - s) : -1;
}

static Blob load_blob(const char *path) {
    Blob b; memset(&b, 0, sizeof b);
    FILE *f = fopen(path, "rb"); if (!f) die("cannot open blob");
    fseek(f, 0, SEEK_END); b.size = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    b.blob = malloc(b.size); if (!b.blob) die("oom blob");
    if (fread(b.blob, 1, b.size, f) != b.size) die("short read");
    fclose(f);
    if (memcmp(b.blob, "LFM3Q8", 6) != 0) die("bad magic");
    uint32_t hl; memcpy(&hl, b.blob + 8, 4);
    char *h = malloc(hl + 1); memcpy(h, b.blob + 12, hl); h[hl] = 0;
    b.base = 12 + hl;
    /* table: one object per tensor, keys in fixed order from json.dumps */
    size_t cap = 512; b.e = malloc(sizeof(Ent) * cap); b.n = 0;
    const char *t = strstr(h, "\"table\":[");
    if (!t) die("no table");
    const char *p = t + 9;
    while (*p) {
        const char *o = strchr(p, '{'); if (!o) break;
        const char *c = strchr(o, '}'); if (!c) die("bad table");
        const char *nm = strstr(o, "\"name\":\""); if (!nm || nm > c) break;
        if (b.n == cap) { cap *= 2; b.e = realloc(b.e, sizeof(Ent) * cap); }
        Ent *e = &b.e[b.n++]; memset(e, 0, sizeof *e);
        nm += 8; size_t L = strcspn(nm, "\""); if (L >= sizeof e->name) die("name too long");
        memcpy(e->name, nm, L); e->name[L] = 0;
        int i = jfind(o, "\"dtype\":", c); e->dtype = atoi(o + i + 8);
        i = jfind(o, "\"ne\":[", c); { const char *q = o + i + 6; char *endp; e->nd = 0;
            while (*q && *q != ']') { e->ne[e->nd++] = (int)strtol(q, &endp, 10); q = endp; while (*q == ',') q++; } }
        i = jfind(o, "\"off\":", c); e->off = (size_t)strtoull(o + i + 6, NULL, 10);
        i = jfind(o, "\"ncodes\":", c); e->ncodes = (size_t)strtoull(o + i + 9, NULL, 10);
        i = jfind(o, "\"nscales\":", c); e->nscales = (size_t)strtoull(o + i + 10, NULL, 10);
        p = c + 1;
        if (*p == ']') break;
    }
    free(h);
    return b;
}

/* Dequantize (Q8) or copy (F32) the named tensor into a float array. */
static float *get(Blob *b, const char *name, int want_nd, size_t *count) {
    for (size_t i = 0; i < b->n; i++) {
        Ent *e = &b->e[i];
        if (strcmp(e->name, name) != 0) continue;
        if (e->nd != want_nd && !(want_nd == 2 && e->nd == 3)) die(name);
        size_t n = 1; for (int d = 0; d < e->nd; d++) n *= (size_t)e->ne[d];
        float *w = malloc(n * sizeof(float));
        if (e->dtype == 1) {                       /* Q8: int8 codes, fp32 scale per 32 */
            const int8_t *q = (const int8_t *)(b->blob + b->base + e->off);
            const float *s = (const float *)(b->blob + b->base + e->off + e->ncodes);
            for (size_t j = 0; j < n; j++) w[j] = (float)q[j] * s[j / 32];
        } else if (e->dtype == 2) {                /* F32 verbatim */
            memcpy(w, b->blob + b->base + e->off, n * sizeof(float));
        } else die("unsupported dtype");
        if (count) *count = n;
        return w;
    }
    fprintf(stderr, "missing tensor %s\n", name); exit(2);
}

/* ---- kernels (fp32, spec order) ----------------------------------------- */
static void rmsnorm(float *x, int T, int D, const float *w) {       /* in place */
    for (int t = 0; t < T; t++) {
        float *r = x + (size_t)t * D; double v = 0;
        for (int i = 0; i < D; i++) v += (double)r[i] * r[i];
        float inv = 1.0f / sqrtf((float)(v / D) + EPS);
        for (int i = 0; i < D; i++) r[i] = w[i] * (r[i] * inv);
    }
}

/* y[T,O] = x[T,I] * W[O,I]^T (+ bias) */
static void linear(const float *x, int T, int I, const float *W, int O, const float *bias, float *y) {
    for (int t = 0; t < T; t++) {
        const float *xr = x + (size_t)t * I; float *yr = y + (size_t)t * O;
        for (int o = 0; o < O; o++) {
            const float *wr = W + (size_t)o * I; float acc = 0.f;
            for (int i = 0; i < I; i++) acc += xr[i] * wr[i];
            yr[o] = bias ? acc + bias[o] : acc;
        }
    }
}

static float silu(float v) { return v / (1.0f + expf(-v)); }

/* 2-D conv on (C,H,W) layout, weight [O, C/groups, k, k], stride s, pad p */
static void conv2d(const float *x, int C, int H, int Wd, const float *w, const float *b,
                   int O, int k, int s, int p, int groups, float *y, int *Ho_out, int *Wo_out) {
    int Ho = (H + 2 * p - k) / s + 1, Wo = (Wd + 2 * p - k) / s + 1;
    int cig = C / groups, cog = O / groups;
    for (int o = 0; o < O; o++) {
        int g = o / cog;
        for (int yy = 0; yy < Ho; yy++)
            for (int xx = 0; xx < Wo; xx++) {
                float acc = b[o];
                for (int ic = 0; ic < cig; ic++) {
                    int c = g * cig + ic;
                    for (int i = 0; i < k; i++) {
                        int iy = yy * s + i - p; if (iy < 0 || iy >= H) continue;
                        for (int j = 0; j < k; j++) {
                            int ix = xx * s + j - p; if (ix < 0 || ix >= Wd) continue;
                            acc += w[(((size_t)o * cig + ic) * k + i) * k + j] * x[((size_t)c * H + iy) * Wd + ix];
                        }
                    }
                }
                y[((size_t)o * Ho + yy) * Wo + xx] = acc;
            }
    }
    *Ho_out = Ho; *Wo_out = Wo;
}

/* ---- model ------------------------------------------------------------ */
static float *W_get(Blob *b, const char *fmt, int idx, const char *suffix, int nd) {
    char nm[256]; snprintf(nm, sizeof nm, fmt, idx); strcat(nm, suffix);
    return get(b, nm, nd, NULL);
}

static void rope(float *q, float *k, int T) {
    /* rotate-half over all 96 dims: out = x*cos + rot_half(x)*sin, cos/sin per (t, d) */
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < HD / 2; d++) {
            double inv = pow(THETA, -(double)(2 * d) / HD);
            double ang = (double)t * inv;
            float c = (float)cos(ang), s = (float)sin(ang);
            /* half-split: pair (d, d+48); cos/sin cover both halves with the same angle */
            for (int h = 0; h < NH; h++) {
                float *v = q + ((size_t)t * NH + h) * HD;
                float a = v[d], bb = v[d + HD / 2];
                v[d] = a * c - bb * s; v[d + HD / 2] = bb * c + a * s;
            }
            for (int h = 0; h < NKV; h++) {
                float *v = k + ((size_t)t * NKV + h) * HD;
                float a = v[d], bb = v[d + HD / 2];
                v[d] = a * c - bb * s; v[d + HD / 2] = bb * c + a * s;
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 5) { fprintf(stderr, "usage: %s blob input.bin T outdir\n", argv[0]); return 2; }
    int Tm = atoi(argv[3]);
    const char *outdir = argv[4];
    Blob b = load_blob(argv[1]);

    /* input (1,128,T) f32 */
    float *feat = malloc(sizeof(float) * 128 * (size_t)Tm);
    FILE *fi = fopen(argv[2], "rb"); if (!fi) die("input");
    if (fread(feat, sizeof(float), 128 * (size_t)Tm, fi) != 128 * (size_t)Tm) die("input short");
    fclose(fi);

    /* ---- downsampler ---- */
    const char *D = "encoder.downsample.";
    char nm[256];
#define DN(s) (snprintf(nm, sizeof nm, "%s%s", D, s), get(&b, nm, 4, NULL))
#define DB(s) (snprintf(nm, sizeof nm, "%s%s", D, s), get(&b, nm, 1, NULL))
    float *w_in = DN("conv_in.weight"), *b_in = DB("conv_in.bias");
    float *w_mid = DN("conv_mid.weight"), *b_mid = DB("conv_mid.bias");
    float *w_lmid = DN("lin_mid.weight"), *b_lmid = DB("lin_mid.bias");
    float *w_end = DN("conv_end.weight"), *b_end = DB("conv_end.bias");
    float *w_lend = DN("lin_end.weight"), *b_lend = DB("lin_end.bias");
    size_t nout_w = 0; float *w_out = get(&b, "encoder.downsample.lin_out.weight", 2, &nout_w);
    float *b_out = DB("lin_out.bias");

    int H = 128, Wd = Tm, Ho, Wo;
    float *x1 = malloc(sizeof(float) * 256 * 64 * (size_t)((Wd + 2 - 3) / 2 + 1));
    conv2d(feat, 1, H, Wd, w_in, b_in, 256, 3, 2, 1, 1, x1, &Ho, &Wo);
    for (size_t i = 0; i < (size_t)256 * Ho * Wo; i++) x1[i] = silu(x1[i]);
    float *x2 = malloc(sizeof(float) * 256 * (size_t)Ho * Wo);
    int H2, W2;
    conv2d(x1, 256, Ho, Wo, w_mid, b_mid, 256, 3, 2, 1, 256, x2, &H2, &W2);
    float *x2b = malloc(sizeof(float) * 256 * (size_t)H2 * W2);
    conv2d(x2, 256, H2, W2, w_lmid, b_lmid, 256, 1, 1, 0, 1, x2b, &H2, &W2);
    for (size_t i = 0; i < (size_t)256 * H2 * W2; i++) x2b[i] = silu(x2b[i]);
    float *x3 = malloc(sizeof(float) * 256 * (size_t)((H2 + 2 - 3) / 2 + 1) * (size_t)((W2 + 2 - 3) / 2 + 1));
    int H3, W3;
    conv2d(x2b, 256, H2, W2, w_end, b_end, 256, 3, 2, 1, 256, x3, &H3, &W3);
    float *x3b = malloc(sizeof(float) * 256 * (size_t)H3 * W3);
    conv2d(x3, 256, H3, W3, w_lend, b_lend, 256, 1, 1, 0, 1, x3b, &H3, &W3);
    for (size_t i = 0; i < (size_t)256 * H3 * W3; i++) x3b[i] = silu(x3b[i]);
    /* flatten(1,2): (C=256, H=16, W=T') -> row t, feature c*16+h */
    int Tt = W3;
    float *flat = malloc(sizeof(float) * (size_t)Tt * 4096);
    for (int t = 0; t < Tt; t++)
        for (int c = 0; c < 256; c++)
            for (int h = 0; h < H3; h++)
                flat[(size_t)t * 4096 + c * H3 + h] = x3b[((size_t)c * H3 + h) * W3 + t];
    float *ds = malloc(sizeof(float) * (size_t)Tt * HID);
    linear(flat, Tt, 4096, w_out, HID, b_out, ds);

    #define DUMP(name, ptr, n) do { char _p[512]; snprintf(_p, sizeof _p, "%s/%s", outdir, name); \
        FILE *o = fopen(_p, "wb"); if (!o) die("dump"); fwrite(ptr, sizeof(float), (n), o); fclose(o); } while (0)
    DUMP("ds_out", ds, (size_t)Tt * HID);

    /* ---- 17 layers ---- */
    float *x = ds;
    float *h = malloc(sizeof(float) * (size_t)Tt * HID);
    float *out = malloc(sizeof(float) * (size_t)Tt * HID);
    float *tmp = malloc(sizeof(float) * (size_t)Tt * 2 * HID * 3);
    static const char types[NLAYER] = {0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,1,1}; /* 1 = attention */
    for (int L = 0; L < NLAYER; L++) {
        char pre[128]; snprintf(pre, sizeof pre, "encoder.lfm.lfm.layers.%d.", L);
        float *opn = W_get(&b, "encoder.lfm.lfm.layers.%d.", L, "operator_norm.weight", 1);
        memcpy(h, x, sizeof(float) * (size_t)Tt * HID);
        rmsnorm(h, Tt, HID, opn);
        if (types[L] == 0) {
            /* conv block: in_proj [3C,C] -> B, C, X ; y = C * conv3(B*X) ; out_proj */
            float *Wi = W_get(&b, "encoder.lfm.lfm.layers.%d.", L, "conv.in_proj.weight", 2);
            float *Wo2 = W_get(&b, "encoder.lfm.lfm.layers.%d.", L, "conv.out_proj.weight", 2);
            float *cw = W_get(&b, "encoder.lfm.lfm.layers.%d.", L, "conv.conv.weight", 2); /* [C,1,3] */
            float *bc = tmp;                                   /* T x 3C */
            linear(h, Tt, HID, Wi, 3 * HID, NULL, bc);
            float *bx = malloc(sizeof(float) * (size_t)Tt * HID);
            float *cg = malloc(sizeof(float) * (size_t)Tt * HID);
            for (int t = 0; t < Tt; t++)
                for (int c = 0; c < HID; c++) {
                    bx[(size_t)t * HID + c] = bc[(size_t)t * 3 * HID + c] * bc[(size_t)t * 3 * HID + 2 * HID + c];
                    cg[(size_t)t * HID + c] = bc[(size_t)t * 3 * HID + HID + c];
                }
            /* causal depthwise conv width 3: y[t,c] = sum_k w[c,k] * bx[t-2+k, c] */
            float *cy = malloc(sizeof(float) * (size_t)Tt * HID);
            for (int t = 0; t < Tt; t++)
                for (int c = 0; c < HID; c++) {
                    float acc = 0.f;
                    for (int k = 0; k < 3; k++) {
                        int tt = t - 2 + k; if (tt < 0) continue;
                        acc += cw[(size_t)c * 3 + k] * bx[(size_t)tt * HID + c];
                    }
                    cy[(size_t)t * HID + c] = cg[(size_t)t * HID + c] * acc;
                }
            linear(cy, Tt, HID, Wo2, HID, NULL, out);
            free(bx); free(cg); free(cy); free(Wi); free(Wo2); free(cw);
        } else {
            /* full causal attention, 8 q heads / 4 kv heads, RoPE, q/k rmsnorm */
            float *Wq = W_get(&b, "encoder.lfm.lfm.layers.%d.", L, "self_attn.q_proj.weight", 2);
            float *Wk = W_get(&b, "encoder.lfm.lfm.layers.%d.", L, "self_attn.k_proj.weight", 2);
            float *Wv = W_get(&b, "encoder.lfm.lfm.layers.%d.", L, "self_attn.v_proj.weight", 2);
            float *Wo2 = W_get(&b, "encoder.lfm.lfm.layers.%d.", L, "self_attn.out_proj.weight", 2);
            float *qn = W_get(&b, "encoder.lfm.lfm.layers.%d.", L, "self_attn.q_layernorm.weight", 1);
            float *kn = W_get(&b, "encoder.lfm.lfm.layers.%d.", L, "self_attn.k_layernorm.weight", 1);
            float *q = malloc(sizeof(float) * (size_t)Tt * NH * HD);
            float *k = malloc(sizeof(float) * (size_t)Tt * NKV * HD);
            float *v = malloc(sizeof(float) * (size_t)Tt * NKV * HD);
            linear(h, Tt, HID, Wq, NH * HD, NULL, q);
            linear(h, Tt, HID, Wk, NKV * HD, NULL, k);
            linear(h, Tt, HID, Wv, NKV * HD, NULL, v);
            /* per-head rmsnorm over HD, then RoPE */
            for (int t = 0; t < Tt; t++) {
                for (int hh = 0; hh < NH; hh++) rmsnorm(q + ((size_t)t * NH + hh) * HD, 1, HD, qn);
                for (int hh = 0; hh < NKV; hh++) rmsnorm(k + ((size_t)t * NKV + hh) * HD, 1, HD, kn);
            }
            rope(q, k, Tt);
            float *ao = malloc(sizeof(float) * (size_t)Tt * NH * HD);
            float *sc = malloc(sizeof(float) * (size_t)Tt);
            float scale = 1.0f / sqrtf((float)HD);
            for (int hh = 0; hh < NH; hh++) {
                int kv = hh / (NH / NKV);
                for (int i = 0; i < Tt; i++) {
                    float *qi = q + ((size_t)i * NH + hh) * HD;
                    float mx = -INFINITY;
                    for (int j = 0; j <= i; j++) {
                        const float *kj = k + ((size_t)j * NKV + kv) * HD;
                        float s2 = 0.f; for (int d = 0; d < HD; d++) s2 += qi[d] * kj[d];
                        sc[j] = s2 * scale; if (sc[j] > mx) mx = sc[j];
                    }
                    float den = 0.f;
                    for (int j = 0; j <= i; j++) { sc[j] = expf(sc[j] - mx); den += sc[j]; }
                    float *oi = ao + ((size_t)i * NH + hh) * HD;
                    for (int d = 0; d < HD; d++) oi[d] = 0.f;
                    for (int j = 0; j <= i; j++) {
                        float p = sc[j] / den;
                        const float *vj = v + ((size_t)j * NKV + kv) * HD;
                        for (int d = 0; d < HD; d++) oi[d] += p * vj[d];
                    }
                }
            }
            linear(ao, Tt, NH * HD, Wo2, HID, NULL, out);
            free(q); free(k); free(v); free(ao); free(sc);
            free(Wq); free(Wk); free(Wv); free(Wo2); free(qn); free(kn);
        }
        for (size_t i = 0; i < (size_t)Tt * HID; i++) out[i] += x[i];       /* x = x + op */
        float *fn = W_get(&b, "encoder.lfm.lfm.layers.%d.", L, "ffn_norm.weight", 1);
        memcpy(h, out, sizeof(float) * (size_t)Tt * HID);
        rmsnorm(h, Tt, HID, fn);
        float *w1 = W_get(&b, "encoder.lfm.lfm.layers.%d.", L, "feed_forward.w1.weight", 2);
        float *w3 = W_get(&b, "encoder.lfm.lfm.layers.%d.", L, "feed_forward.w3.weight", 2);
        float *w2 = W_get(&b, "encoder.lfm.lfm.layers.%d.", L, "feed_forward.w2.weight", 2);
        float *g = malloc(sizeof(float) * (size_t)Tt * FFN), *u = malloc(sizeof(float) * (size_t)Tt * FFN);
        linear(h, Tt, HID, w1, FFN, NULL, g);
        linear(h, Tt, HID, w3, FFN, NULL, u);
        for (size_t i = 0; i < (size_t)Tt * FFN; i++) g[i] = silu(g[i]) * u[i];
        float *dn = malloc(sizeof(float) * (size_t)Tt * HID);
        linear(g, Tt, FFN, w2, HID, NULL, dn);
        for (size_t i = 0; i < (size_t)Tt * HID; i++) out[i] += dn[i];      /* x = x + ffn */
        char lname[32]; snprintf(lname, sizeof lname, "layer_%02d", L);
        DUMP(lname, out, (size_t)Tt * HID);
        float *tswap = x; x = out; out = tswap;
        free(g); free(u); free(dn); free(w1); free(w2); free(w3); free(opn); free(fn);
    }
    /* embedding_norm + output */
    float *en = get(&b, "encoder.lfm.lfm.embedding_norm.weight", 1, NULL);
    rmsnorm(x, Tt, HID, en);
    DUMP("enc_out", x, (size_t)Tt * HID);

    /* ---- projector: LayerNorm(768) -> Linear 768->2048 -> GELU -> Linear 2048->2048 ---- */
    float *pw0 = get(&b, "mlp.model.0.weight", 1, NULL), *pb0 = get(&b, "mlp.model.0.bias", 1, NULL);
    float *pw1 = get(&b, "mlp.model.1.weight", 2, NULL), *pb1 = get(&b, "mlp.model.1.bias", 1, NULL);
    float *pw3 = get(&b, "mlp.model.3.weight", 2, NULL), *pb3 = get(&b, "mlp.model.3.bias", 1, NULL);
    float *ln = malloc(sizeof(float) * (size_t)Tt * HID);
    for (int t = 0; t < Tt; t++) {
        const float *r = x + (size_t)t * HID; double mu = 0, var = 0;
        for (int i = 0; i < HID; i++) mu += r[i];
        mu /= HID;
        for (int i = 0; i < HID; i++) var += (r[i] - mu) * (r[i] - mu);
        var /= HID; float inv = 1.0f / sqrtf((float)var + 1e-5f);
        for (int i = 0; i < HID; i++) ln[(size_t)t * HID + i] = (r[i] - (float)mu) * inv * pw0[i] + pb0[i];
    }
    float *p1 = malloc(sizeof(float) * (size_t)Tt * PROJ_OUT), *p2 = malloc(sizeof(float) * (size_t)Tt * PROJ_OUT);
    linear(ln, Tt, HID, pw1, PROJ_OUT, pb1, p1);
    for (size_t i = 0; i < (size_t)Tt * PROJ_OUT; i++) p1[i] = 0.5f * p1[i] * (1.0f + erff(p1[i] / 1.41421356237f));
    linear(p1, Tt, PROJ_OUT, pw3, PROJ_OUT, pb3, p2);
    DUMP("proj_out", p2, (size_t)Tt * PROJ_OUT);
    printf("lfm3_enc: T=%d tokens=%d -> %s\n", Tm, Tt, outdir);
    return 0;
}
