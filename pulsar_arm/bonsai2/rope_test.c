/* Rope validation: fork ggml_rope_multi (MROPE path) vs scalar ref transcribed
 * from ggml_mrope_cache_init + rotate_pairs. Bonsai 2 text config:
 * n_dims=64 (partial 0.25 of head_dim 256), NEOX pairing (i, i+32),
 * theta_j = p * freq_base^(-2j/64), freq_base 1e7, sections {11,11,10,0}
 * (moot for text: all four position tracks equal), yarn off (ext_factor 0),
 * remainder dims copied verbatim.
 * Build: gcc -O2 -o rope_test rope_test.c -I$LL/ggml/include -L$LL/build/bin
 *        -lggml -lggml-base -lggml-cpu -Wl,-rpath,$LL/build/bin -lm
 */
#include "ggml.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NE0 256     /* head_dim */
#define ND  64      /* n_dims (rotated) */
#define NH  3
#define T   4
#define BASE 1e7f

static uint32_t rng = 0x12345678u;
static float frand(void) {
    rng = rng * 1664525u + 1013904223u;
    return (float)((double)(rng >> 8) / 16777216.0 * 6.0 - 3.0);
}
/* scalar ref: angle_j = p * BASE^(-2j/ND), pair (j, j+32) for j<32, rest copied */
static void ref_rope(const float *in, float *out, int p) {
    for (int j = 0; j < ND / 2; j++) {
        float ang = (float)p * powf(BASE, -2.0f * j / (float)ND);
        float c = cosf(ang), s = sinf(ang);
        float x0 = in[j], x1 = in[j + ND / 2];
        out[j] = x0 * c - x1 * s;
        out[j + ND / 2] = x0 * s + x1 * c;
    }
    for (int i = ND; i < NE0; i++) out[i] = in[i];
}
int main(void) {
    struct ggml_init_params ip = { 1u << 26, NULL, false };
    struct ggml_context *ctx = ggml_init(ip);
    struct ggml_tensor *a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, NE0, NH, T);
    struct ggml_tensor *pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T * 4);
    for (int i = 0; i < ggml_nelements(a); i++) ((float *)a->data)[i] = frand();
    for (int i = 0; i < T * 4; i++) ((int32_t *)pos->data)[i] = i % T;  /* text: 4 equal tracks */
    int sections[4] = { 11, 11, 10, 0 };
    struct ggml_tensor *out = ggml_rope_multi(ctx, a, pos, NULL, ND, sections,
        GGML_ROPE_TYPE_MROPE, 262144, BASE, 1.0f, 0.0f, 1.0f, 256, 32);
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    const float *o = (const float *)out->data;
    const float *i = (const float *)a->data;
    double m = 0; long wi = 0;
    static float r[NE0];
    for (int t = 0; t < T; t++)
        for (int h = 0; h < NH; h++) {
            ref_rope(i + (long)(h + NH * t) * NE0, r, t);
            for (int d = 0; d < NE0; d++) {
                double dd = fabs((double)o[(long)(h + NH * t) * NE0 + d] - (double)r[d]);
                if (dd > m) { m = dd; wi = (long)(h * T + t) * NE0 + d; }
            }
        }
    printf("rope maxabs diff: %.3e (idx %ld) %s\n", m, wi, m < 1e-5 ? "MATCH" : "MISMATCH");
    /* dump t=1,h=0 pair 0 to identify the fork's angle formula */
    {
        const float *ii = i + (long)(0 + NH * 1) * NE0; /* h=0,t=1: (h+NH*t) */
        const float *oo = o + (long)(0 + NH * 1) * NE0;
        printf("t=1: in[0]=%.6f in[32]=%.6f\n", ii[0], ii[32]);
        printf("  fork out[0]=%.6f out[32]=%.6f\n", oo[0], oo[32]);
        printf("  mine out[0]=%.6f out[32]=%.6f\n", r[0], r[32]);
        float ts = powf(BASE, -2.0f/ND), tsn = powf(BASE, -2.0f/NE0);
        float cands[4] = {1.0f, ts, tsn, 1.0f*ts*ts};
        for (int cI = 0; cI < 4; cI++) {
            float ang = cands[cI];
            float c = cosf(ang), sn = sinf(ang);
            printf("  cand%d ang=%.9f -> out0=%.6f out32=%.6f\n", cI, ang,
                   ii[0]*c - ii[32]*sn, ii[0]*sn + ii[32]*c);
        }
    }
    /* sanity: token 0 has zero angle -> must be identity */
    double zi = 0;
    for (int d = 0; d < ND; d++)
        if (fabs(o[d] - i[d]) > zi) zi = fabs(o[d] - i[d]);
    printf("p=0 rotated-dims identity check: %.3e\n", zi);
    return m < 1e-5 ? 0 : 1;
}
