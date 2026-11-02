// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
#include "blob.h"
#include "tfmamba.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>

#define HID 64
#define DINNER 256
#define DSTATE 16
#define DCONV 4
#define DTRANK 4
#define XP (DTRANK + 2*DSTATE)

static int g_rev = 0;
static const char *g_pfx = "";
static char xbuf[64];
static const char *xdbg_tag(const char *n){ snprintf(xbuf,sizeof(xbuf),"%s_%s", g_rev?"bw":"fw", n); return xbuf; }
static const float *tget(const reuse_blob_t *b, const char *pfx, const char *sfx) {
    char key[256];
    snprintf(key, sizeof(key), "%s.%s", pfx, sfx);
    const reuse_tensor_t *t = reuse_blob_find(b, key);
    if (!t) { fprintf(stderr, "MISSING TENSOR: %s\n", key); exit(2); }
    return t->data;
}

static void ddrop(const char *tag, const float *p, size_t n) {
    const char *base = getenv("REUSE_DEBUG");
    if (!base) return;
    if (strncmp(g_pfx, "TSMamba.0.", 10) != 0) return;
    if (strstr(g_pfx, "time_mamba") == NULL) return;
    char path[512]; snprintf(path, sizeof(path), "%s_c_%s.raw", base, tag);
    FILE *f = fopen(path, "wb"); if (!f) return; fwrite(p, sizeof(float), n, f); fclose(f);
}
static float silu(float v) { return v / (1.0f + expf(-v)); }
static float softplus_f(float v) { return v > 20.0f ? v : log1pf(expf(v)); }

// x:[S,L,HID] -> out:[S,L,HID]; rev=0 forward, rev=1 backward (sequence reversed).
static void mamba_dir(const reuse_blob_t *blob, const char *pfx,
                      const float *x, float *out, int S, int L, int rev) {
    g_rev = rev;
    g_pfx = pfx;
    const float *W_in   = tget(blob, pfx, "in_proj.weight");
    const float *W_conv = tget(blob, pfx, "conv1d.weight");
    const float *B_conv = tget(blob, pfx, "conv1d.bias");
    const float *W_xp   = tget(blob, pfx, "x_proj.weight");
    const float *W_dt   = tget(blob, pfx, "dt_proj.weight");
    const float *B_dt   = tget(blob, pfx, "dt_proj.bias");
    const float *A_log  = tget(blob, pfx, "A_log");
    const float *Dv     = tget(blob, pfx, "D");
    const float *W_out  = tget(blob, pfx, "out_proj.weight");

    const size_t SL = (size_t)S * L;
    float *A = (float*)malloc((size_t)DINNER * DSTATE * sizeof(float));
    for (int i = 0; i < DINNER * DSTATE; i++) A[i] = -expf(A_log[i]);
    float *x0 = (float*)malloc(SL * DINNER * sizeof(float));
    float *z  = (float*)malloc(SL * DINNER * sizeof(float));
    float *xc = (float*)malloc(SL * DINNER * sizeof(float));
    float *xdbl = (float*)malloc(SL * XP * sizeof(float));
    float *dtt = (float*)malloc(SL * DINNER * sizeof(float));
    float *y  = (float*)malloc(SL * DINNER * sizeof(float));

    #pragma omp parallel for schedule(static)
    for (int s = 0; s < S; s++) {
        for (int t = 0; t < L; t++) {
            const int ts = rev ? (L - 1 - t) : t;
            const float *xi = x + ((size_t)s * L + ts) * HID;
            float *xo = x0 + ((size_t)s * L + t) * DINNER;
            float *zo = z  + ((size_t)s * L + t) * DINNER;
            for (int o = 0; o < 2*DINNER; o++) {
                const float *wr = W_in + (size_t)o * HID;
                float acc = 0.0f;
                for (int c = 0; c < HID; c++) acc += xi[c] * wr[c];
                if (o < DINNER) xo[o] = acc; else zo[o - DINNER] = acc;
            }
        }
    }

    ddrop(xdbg_tag("xz"), x0, SL*DINNER); ddrop(xdbg_tag("z"), z, SL*DINNER);
    // causal depthwise conv1d (k=4) + SiLU
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < S; s++) {
        for (int c = 0; c < DINNER; c++) {
            const float *wc = W_conv + (size_t)c * DCONV;
            for (int t = 0; t < L; t++) {
                float acc = B_conv[c];
                for (int k = 0; k < DCONV; k++) {
                    int tt = t - (DCONV - 1) + k;
                    if (tt < 0) continue;
                    acc += x0[((size_t)s * L + tt) * DINNER + c] * wc[k];
                }
                xc[((size_t)s * L + t) * DINNER + c] = silu(acc);
            }
        }
    }

    ddrop(xdbg_tag("xact"), xc, SL*DINNER);
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < S; s++) {
        for (int t = 0; t < L; t++) {
            const float *xi = xc + ((size_t)s * L + t) * DINNER;
            float *yo = xdbl + ((size_t)s * L + t) * XP;
            for (int o = 0; o < XP; o++) {
                const float *wr = W_xp + (size_t)o * DINNER;
                float acc = 0.0f;
                for (int c = 0; c < DINNER; c++) acc += xi[c] * wr[c];
                yo[o] = acc;
            }
        }
    }
    ddrop(xdbg_tag("xdbl"), xdbl, SL*XP);
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < S; s++) {
        for (int t = 0; t < L; t++) {
            const float *di = xdbl + ((size_t)s * L + t) * XP;
            float *yo = dtt + ((size_t)s * L + t) * DINNER;
            for (int o = 0; o < DINNER; o++) {
                const float *wr = W_dt + (size_t)o * DTRANK;
                float acc = B_dt[o];
                for (int c = 0; c < DTRANK; c++) acc += di[c] * wr[c];
                yo[o] = softplus_f(acc);
            }
        }
    }

    ddrop(xdbg_tag("dt"), dtt, SL*DINNER);
    // selective scan; Bm/Cm live inside xdbl with row stride XP
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < S; s++) {
        for (int c = 0; c < DINNER; c++) {
            float h[DSTATE];
            for (int n = 0; n < DSTATE; n++) h[n] = 0.0f;
            const float *Ac = A + (size_t)c * DSTATE;
            for (int ii = 0; ii < L; ii++) {
                // After the causal conv, x0/xc/dtt/xdbl all live on the scan
                // axis (reversed for rev=1), so index them by ii, not by t.
                const size_t sl = (size_t)s * L + ii;
                const float d = dtt[sl * DINNER + c];
                const float ut = xc[sl * DINNER + c];
                const float *Bt = xdbl + sl * XP + DTRANK;
                const float *Ct = Bt + DSTATE;
                float acc = 0.0f;
                for (int n = 0; n < DSTATE; n++) {
                    h[n] = expf(d * Ac[n]) * h[n] + d * Bt[n] * ut;
                    acc += h[n] * Ct[n];
                }
                y[sl * DINNER + c] = acc + ut * Dv[c];
            }
        }
    }

    ddrop(xdbg_tag("scan"), y, SL*DINNER);
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < S; s++) {
        for (int t = 0; t < L; t++) {
            float *yi = y + ((size_t)s * L + t) * DINNER;
            const float *zi = z + ((size_t)s * L + t) * DINNER;
            for (int c = 0; c < DINNER; c++) yi[c] *= silu(zi[c]);
            float *oo = out + ((size_t)s * L + t) * HID;
            for (int o = 0; o < HID; o++) {
                const float *wr = W_out + (size_t)o * DINNER;
                float acc = 0.0f;
                for (int c = 0; c < DINNER; c++) acc += yi[c] * wr[c];
                oo[o] = acc;
            }
        }
    }
    ddrop(xdbg_tag("yg"), y, SL*DINNER);
    ddrop(xdbg_tag("out"), out, SL*HID);
    free(A); free(x0); free(z); free(xc); free(xdbl); free(dtt); free(y);
}

// MambaBlock: bidirectional + concat + output_proj + LayerNorm.
static void mamba_block(const reuse_blob_t *blob, const char *pfx,
                        const float *x, float *out /*[S,L,HID]*/, int S, int L) {
    char p[192];
    float *fw = (float*)malloc((size_t)S*L*HID*sizeof(float));
    float *bw = (float*)malloc((size_t)S*L*HID*sizeof(float));
    float *cat = (float*)malloc((size_t)S*L*2*HID*sizeof(float));
    const size_t SL = (size_t)S * L;

    snprintf(p, sizeof(p), "%s.forward_blocks", pfx);
    mamba_dir(blob, p, x, fw, S, L, 0);
    snprintf(p, sizeof(p), "%s.backward_blocks", pfx);
    mamba_dir(blob, p, x, bw, S, L, 1);   // bw output already indexed in un-reversed order

    // PyTorch: out_bw = flip( bw(flip(x)) + flip(x) ); bw[] is indexed on the
    // reversed axis, so un-flip it before adding the residual.
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < S; s++) {
        for (int t = 0; t < L; t++) {
            const size_t i  = (size_t)s*L + t;
            const size_t ir = (size_t)s*L + (L - 1 - t);
            for (int c = 0; c < HID; c++) {
                cat[i*2*HID + c]        = fw[i*HID + c] + x[i*HID + c];
                cat[i*2*HID + HID + c]  = bw[ir*HID + c] + x[i*HID + c];
            }
        }
    }
    const float *W_op = tget(blob, pfx, "output_proj.weight");   // [HID, 2*HID]
    const float *B_op = tget(blob, pfx, "output_proj.bias");
    const float *LW   = tget(blob, pfx, "norm.weight");
    const float *LB   = tget(blob, pfx, "norm.bias");

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < SL; i++) {
        float tmp[HID];
        for (int o = 0; o < HID; o++) {
            const float *wr = W_op + (size_t)o * 2*HID;
            float acc = B_op[o];
            for (int c = 0; c < 2*HID; c++) acc += cat[i*2*HID + c] * wr[c];
            tmp[o] = acc;
        }
        float mean = 0.0f;
        for (int o = 0; o < HID; o++) mean += tmp[o];
        mean /= HID;
        float var = 0.0f;
        for (int o = 0; o < HID; o++) { float d = tmp[o]-mean; var += d*d; }
        var /= HID;
        const float inv = 1.0f / sqrtf(var + 1e-5f);
        for (int o = 0; o < HID; o++) out[i*HID + o] = (tmp[o]-mean)*inv*LW[o] + LB[o];
    }
    free(fw); free(bw); free(cat);
}

// x, out: [B,HID,T,F] channel-major, matching PyTorch [B,C,T,F].
void reuse_tfmamba_block(const reuse_blob_t *blob, int idx,
                         const float *x, float *out, int B, int T, int F) {
    const int S_t = B*F, S_f = B*T;
    float *xt = (float*)malloc((size_t)S_t*T*HID*sizeof(float));
    float *yt = (float*)malloc((size_t)S_t*T*HID*sizeof(float));
    float *xf = (float*)malloc((size_t)S_f*F*HID*sizeof(float));
    float *yf = (float*)malloc((size_t)S_f*F*HID*sizeof(float));
    char p[192];

    // [B,C,T,F] -> [(b,f), t, c]
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++)
      for (int f = 0; f < F; f++)
        for (int t = 0; t < T; t++)
          for (int c = 0; c < HID; c++)
            xt[(((size_t)b*F + f)*T + t)*HID + c] = x[(((size_t)b*HID + c)*T + t)*F + f];

    ddrop("xt", xt, (size_t)S_t*T*HID);
    snprintf(p, sizeof(p), "TSMamba.%d.time_mamba", idx);
    mamba_block(blob, p, xt, yt, S_t, T);

    // time residual, then gather freq sequences [(b,t), f, c]
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++)
      for (int f = 0; f < F; f++)
        for (int t = 0; t < T; t++)
          for (int c = 0; c < HID; c++) {
            float r = yt[(((size_t)b*F + f)*T + t)*HID + c] + xt[(((size_t)b*F + f)*T + t)*HID + c];
            xf[(((size_t)b*T + t)*F + f)*HID + c] = r;
          }

    snprintf(p, sizeof(p), "TSMamba.%d.freq_mamba", idx);
    mamba_block(blob, p, xf, yf, S_f, F);

    // freq residual, scatter back to [B,C,T,F]
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++)
      for (int t = 0; t < T; t++)
        for (int f = 0; f < F; f++)
          for (int c = 0; c < HID; c++)
            out[(((size_t)b*HID + c)*T + t)*F + f] =
                yf[(((size_t)b*T + t)*F + f)*HID + c] + xf[(((size_t)b*T + t)*F + f)*HID + c];

    free(xt); free(yt); free(xf); free(yf);
}
