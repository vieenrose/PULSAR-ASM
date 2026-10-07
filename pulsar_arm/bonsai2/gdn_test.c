/* GDN validation: fork ggml_gated_delta_net vs scalar ref transcribed from the
 * fork's CPU implementation (ggml-cpu/ops.cpp:10941). Config matches Bonsai 2:
 * S_v=128, 48 v-heads, 16 q/k heads (q[idx%16]), scalar gate, raw-gates mode,
 * K=1 snapshot, T=3 tokens, 1 seq. Compares BOTH outputs: attn and final state.
 * Build: gcc -O2 -o gdn_test gdn_test.c -I$LL/ggml/include -I$LL/ggml/src/ggml-cpu
 *        -L$LL/build/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,... -lm
 */
#include "ggml.h"
#include "ggml-cpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SV 128
#define H 48
#define QH 16
#define T 3

static uint32_t rng = 0x9e3779b9u;
static float frand(void) {
    rng = rng * 1664525u + 1013904223u;
    return (float)((double)(rng >> 8) / 16777216.0 * 4.0 - 2.0);
}

/* scalar reference, transcribed from the CPU op:
 * state is transposed: buf[j*SV+i] = S[i][j]; scalar g per head. */
static void gdn_ref_step(const float *q, const float *k, const float *v,
                         float beta, float g, float *buf, float *out) {
    float delta[SV];
    float dec = expf(g);
    for (int j = 0; j < SV * SV; j++) buf[j] *= dec;
    for (int j = 0; j < SV; j++) {
        float s = 0.0f;
        for (int i = 0; i < SV; i++) s += buf[j * SV + i] * k[i];
        delta[j] = (v[j] - s) * beta;
    }
    for (int j = 0; j < SV; j++)
        for (int i = 0; i < SV; i++)
            buf[j * SV + i] += k[i] * delta[j];
    for (int j = 0; j < SV; j++) {
        float s = 0.0f;
        for (int i = 0; i < SV; i++) s += buf[j * SV + i] * q[i];
        out[j] = s * (1.0f / sqrtf((float)SV));
    }
}
static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

int main(int argc, char **argv) {
    const int TT = argc > 1 ? atoi(argv[1]) : 3;
    const int ZERO = argc > 2;
    struct ggml_init_params ip = { 1u << 28, NULL, false };   /* 256MB ctx */
    struct ggml_context *ctx = ggml_init(ip);
    struct ggml_tensor *q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, SV, QH, TT, 1);
    struct ggml_tensor *k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, SV, QH, TT, 1);
    struct ggml_tensor *v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, SV, H,  TT, 1);
    struct ggml_tensor *g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, TT, 1);
    struct ggml_tensor *b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, TT, 1);
    struct ggml_tensor *s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, SV, SV, H, 1);
    struct ggml_tensor *dtb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    struct ggml_tensor *am  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    /* deterministic inputs */
    for (int i = 0; i < ggml_nelements(q); i++) ((float *)q->data)[i] = frand();
    for (int i = 0; i < ggml_nelements(k); i++) ((float *)k->data)[i] = frand();
    for (int i = 0; i < ggml_nelements(v); i++) ((float *)v->data)[i] = frand();
    for (int i = 0; i < ggml_nelements(g); i++) ((float *)g->data)[i] = frand() * 0.5f;
    for (int i = 0; i < ggml_nelements(b); i++) ((float *)b->data)[i] = frand() * 0.5f;
    for (int i = 0; i < ggml_nelements(s); i++) ((float *)s->data)[i] = ZERO ? 0.0f : frand() * 0.25f;
    for (int i = 0; i < H; i++) { ((float *)dtb->data)[i] = frand() * 0.3f;
                                  ((float *)am->data)[i] = 0.5f + (i % 7) * 0.1f; }
    struct ggml_tensor *gdn = ggml_gated_delta_net(ctx, q, k, v, g, b, s, 1);
    ggml_gated_delta_net_set_raw_gates(gdn, dtb, am);
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, gdn);
    ggml_graph_compute_with_ctx(ctx, gf, 20);

    /* fork results: attn region + state region of gdn->data */
    const float *dst = (const float *)gdn->data;
    const long attn_elems = (long)SV * H * TT;
    /* my ref: per head */
    static float state[H][SV * SV], attn[T][H * SV];
    (void)TT;
    for (int h = 0; h < H; h++) {
        memcpy(state[h], s->data, sizeof(state[0])); /* shared initial state? no:
           fork reads state at iv3*stride + iv1*Sv*Sv -> per-head slice */
        const float *sp = (const float *)s->data + (long)h * SV * SV;
        memcpy(state[h], sp, sizeof(state[0]));
    }
    for (int h = 0; h < H; h++) {
        for (int t = 0; t < TT; t++) {
            const float *qd = (const float *)q->data + (long)(h % QH) * SV + (long)t * SV * QH;
            const float *kd = (const float *)k->data + (long)(h % QH) * SV + (long)t * SV * QH;
            const float *vd = (const float *)v->data + (long)h * SV + (long)t * SV * H;
            float beta_raw = ((const float *)b->data)[(long)t * H + h];
            float g_raw    = ((const float *)g->data)[(long)t * H + h];
            float beta = sigmoid(beta_raw);
            float x = g_raw + ((const float *)dtb->data)[h];
            float g0 = ((const float *)am->data)[h] * (x > 20.0f ? x : log1pf(expf(x)));
            gdn_ref_step(qd, kd, vd, beta, g0, state[h], attn[t] + h * SV);
        }
    }
    /* compare attn */
    double ma = 0; long wa = 0;
    double fa = 0;
    for (int t = 0; t < TT; t++)
        for (int i = 0; i < H * SV; i++) {
            if (fabs(dst[t * H * SV + i]) > fa) fa = fabs(dst[t * H * SV + i]);
            double d = fabs(dst[t * H * SV + i] - attn[t][i]);
            if (d > ma) { ma = d; wa = t * (long)H * SV + i; }
        }
    /* compare final state (fork writes K=1 slot after attn) */
    const float *dst_s = dst + attn_elems;
    double ms = 0;
    for (int i = 0; i < H * SV * SV; i++) {
        double d = fabs(dst_s[i] - (double)((float *)state)[i]);
        if (d > ms) ms = d;
    }
    printf("T=%d zero=%d max|attn_fork|=%.4f\n", TT, ZERO, fa);
    printf("attn  maxabs diff: %.3e (%ld) %s\n", ma, wa, ma < 1e-5 ? "MATCH" : "MISMATCH");
    printf("state maxabs diff: %.3e %s\n", ms, ms < 1e-4 ? "MATCH" : "MISMATCH");
    return (ma < 1e-5 && ms < 1e-4) ? 0 : 1;
}
