// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
// Full RE-USE forward: DenseEncoder -> 30x TFMambaBlock -> Mag/PhaseDecoder -> sweep mask.
#include "blob.h"
#include "model.h"
#include "conv2d_full.cuh"
#include "instancenorm.cuh"
#include "tfmamba.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HID 64
#define DEPTH 4

static const float *wt(const reuse_blob_t *b, const char *name) {
    const reuse_tensor_t *t = reuse_blob_find(b, name);
    if (!t) { fprintf(stderr, "MISSING TENSOR: %s\n", name); exit(2); }
    return t->data;
}
static float *alloc_n(size_t n) {
    float *p = (float*)malloc(n * sizeof(float));
    if (!p) { fprintf(stderr, "OOM %zu floats\n", n); exit(3); }
    return p;
}

// PyTorch DenseBlock: skip=x; for i: x = IN+PReLU(conv_i(cat-skip, dil=2^i)); skip=cat([x,skip])
static void dense_block(const reuse_blob_t *blob, const char *prefix,
                        const float *x_in, int N, int T, int F, int hid, float *out) {
    const int cap = hid * (DEPTH + 1) * T * F;
    float *skip = alloc_n((size_t)cap), *skip2 = alloc_n((size_t)cap);
    float *tmp = alloc_n((size_t)hid*T*F), *cur = alloc_n((size_t)hid*T*F);
    memcpy(skip, x_in, (size_t)hid*T*F*sizeof(float));
    int c_skip = hid;
    for (int i = 0; i < DEPTH; i++) {
        char key[256];
        snprintf(key, sizeof(key), "%s.%d.0.weight", prefix, i); const float *w = wt(blob, key);
        snprintf(key, sizeof(key), "%s.%d.0.bias",   prefix, i); const float *b = wt(blob, key);
        snprintf(key, sizeof(key), "%s.%d.1.weight", prefix, i); const float *nw = wt(blob, key);
        snprintf(key, sizeof(key), "%s.%d.1.bias",   prefix, i); const float *nb = wt(blob, key);
        snprintf(key, sizeof(key), "%s.%d.2.weight", prefix, i); const float *pa = wt(blob, key);
        int dil = 1 << i;
        conv2d_general(skip, N, c_skip, T, F, w, b, hid, 3, 3, dil, 1, dil, 1, 1, 1, tmp);
        instancenorm2d_prelu(tmp, cur, N, hid, T, F, nw, nb, pa, 1e-5f);
        memcpy(skip2, cur, (size_t)hid*T*F*sizeof(float));
        memcpy(skip2 + (size_t)hid*T*F, skip, (size_t)c_skip*T*F*sizeof(float));
        float *sw = skip; skip = skip2; skip2 = sw;
        c_skip += hid;
    }
    memcpy(out, cur, (size_t)hid*T*F*sizeof(float));
    free(skip); free(skip2); free(tmp); free(cur);
}

// SPConvTranspose2d(in=hid, out=hid, k=(1,3), r): zero-pad W by 1, conv hid->hid*r, sub-pixel.
// x:[N,hid,H,W] -> y:[N,hid,H,W*r]
static void sp_up(const reuse_blob_t *blob, const char *prefix,
                  const float *x, int N, int H, int W, int r, float *y) {
    char key[256];
    snprintf(key, sizeof(key), "%s.0.conv.weight", prefix); const float *w = wt(blob, key);
    snprintf(key, sizeof(key), "%s.0.conv.bias",   prefix); const float *b = wt(blob, key);
    const int Wp = W + 2;
    float *xp = alloc_n((size_t)HID*H*Wp);
    for (int c = 0; c < HID; c++)
      for (int h = 0; h < H; h++) {
        xp[(size_t)c*H*Wp + (size_t)h*Wp + 0] = 0.0f;
        memcpy(xp + (size_t)c*H*Wp + (size_t)h*Wp + 1,
               x + ((size_t)c*H + h)*W, (size_t)W*sizeof(float));
        xp[(size_t)c*H*Wp + (size_t)h*Wp + Wp - 1] = 0.0f;
      }
    float *cv = alloc_n((size_t)HID*r*H*W);
    conv2d_general(xp, N, HID, H, Wp, w, b, HID*r, 1, 3, 0, 0, 1, 1, 1, 1, cv);
    free(xp);
    // out[b, c, h, w*r + a] = cv[b, a*hid + c, h, w]
    for (int c = 0; c < HID; c++)
      for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
          for (int a = 0; a < r; a++)
            y[((size_t)c*H + h)*(size_t)(W*r) + (size_t)w*r + a] =
                cv[(((size_t)a*HID + c)*H + h)*W + w];
    free(cv);
}

static void transpose_HW(const float *x, int N, int C, int H, int W, float *y) {
    for (int n = 0; n < N; n++)
      for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
          for (int w = 0; w < W; w++)
            y[((size_t)n*C + c)*(size_t)(W*H) + (size_t)w*H + h] =
                x[((size_t)n*C + c)*(size_t)H*W + (size_t)h*W + w];
}

// Mag/Phase decoder body: DenseBlock -> up_conv1(r=2 on F) -> up_conv2(r=4 on T) .
// x:[N,hid,T,F] -> y:[N,hid,T*4,F*2]
static void decoder_pre(const reuse_blob_t *blob, const char *prefix,
                        const float *x, int N, int T, int F, float *y) {
    char key[256];
    float *d = alloc_n((size_t)HID*T*F);
    snprintf(key, sizeof(key), "%s.dense_block.dense_block", prefix);
    dense_block(blob, key, x, N, T, F, HID, d);

    float *u1 = alloc_n((size_t)HID*T*(size_t)(F*2));
    sp_up(blob, (char*)strcat(strcpy(key, prefix), ".up_conv1"), d, N, T, F, 2, u1);
    float *u1n = alloc_n((size_t)HID*T*(size_t)(F*2));
    snprintf(key, sizeof(key), "%s.up_conv1.1.weight", prefix); const float *nw = wt(blob, key);
    snprintf(key, sizeof(key), "%s.up_conv1.1.bias",   prefix); const float *nb = wt(blob, key);
    snprintf(key, sizeof(key), "%s.up_conv1.2.weight", prefix); const float *pa = wt(blob, key);
    instancenorm2d_prelu(u1, u1n, N, HID, T, F*2, nw, nb, pa, 1e-5f);
    free(u1); free(d);

    // up_conv2 runs on the (T,F)-transposed tensor, then transposed back.
    const int Tt = F*2, Ft = T;
    float *sw = alloc_n((size_t)HID*Tt*Ft);
    transpose_HW(u1n, N, HID, T, F*2, sw);
    free(u1n);
    float *u2 = alloc_n((size_t)HID*Tt*(size_t)(Ft*4));
    sp_up(blob, (char*)strcat(strcpy(key, prefix), ".up_conv2"), sw, N, Tt, Ft, 4, u2);
    free(sw);
    float *u2n = alloc_n((size_t)HID*Tt*(size_t)(Ft*4));
    snprintf(key, sizeof(key), "%s.up_conv2.1.weight", prefix); const float *nw2 = wt(blob, key);
    snprintf(key, sizeof(key), "%s.up_conv2.1.bias",   prefix); const float *nb2 = wt(blob, key);
    snprintf(key, sizeof(key), "%s.up_conv2.2.weight", prefix); const float *pa2 = wt(blob, key);
    instancenorm2d_prelu(u2, u2n, N, HID, Tt, Ft*4, nw2, nb2, pa2, 1e-5f);
    free(u2);
    transpose_HW(u2n, N, HID, Tt, Ft*4, y);   // -> [N,hid,T,F*2]
    free(u2n);
}

void reuse_forward(const reuse_blob_t *blob, const float *mag_in, const float *pha_in,
                   int B, int F_in, int T_in, float *out_mag, float *out_pha) {
    const int Tp = T_in + 2, Fp = F_in + 2;
    // encoder input: [B,2,T,F] with 2 zero freq cols and 2 zero time rows
    float *xin = (float*)calloc((size_t)2*Tp*Fp, sizeof(float));
    for (int t = 0; t < T_in; t++)
        for (int f = 0; f < F_in; f++) {
            xin[(size_t)t*Fp + f] = mag_in[(size_t)f*T_in + t];
            xin[(size_t)Tp*Fp + (size_t)t*Fp + f] = pha_in[(size_t)f*T_in + t];
        }

    float *x1 = alloc_n((size_t)HID*Tp*Fp);
    conv2d_general(xin, 1, 2, Tp, Fp,
                   wt(blob, "dense_encoder.dense_conv_1.0.weight"),
                   wt(blob, "dense_encoder.dense_conv_1.0.bias"),
                   HID, 1, 1, 0, 0, 1, 1, 1, 1, x1);
    free(xin);
    float *x1n = alloc_n((size_t)HID*Tp*Fp);
    instancenorm2d_prelu(x1, x1n, 1, HID, Tp, Fp,
                   wt(blob, "dense_encoder.dense_conv_1.1.weight"),
                   wt(blob, "dense_encoder.dense_conv_1.1.bias"),
                   wt(blob, "dense_encoder.dense_conv_1.2.weight"), 1e-5f);
    free(x1);

    float *xdb = alloc_n((size_t)HID*Tp*Fp);
    dense_block(blob, "dense_encoder.dense_block.dense_block", x1n, 1, Tp, Fp, HID, xdb);
    free(x1n);

    const int T2 = (Tp - 1)/4 + 1, F2 = (Fp - 3)/2 + 1;
    float *x2 = alloc_n((size_t)HID*T2*F2);
    conv2d_general(xdb, 1, HID, Tp, Fp,
                   wt(blob, "dense_encoder.dense_conv_2.0.weight"),
                   wt(blob, "dense_encoder.dense_conv_2.0.bias"),
                   HID, 1, 3, 0, 0, 1, 1, 4, 2, x2);
    free(xdb);
    float *enc = alloc_n((size_t)HID*T2*F2);
    instancenorm2d_prelu(x2, enc, 1, HID, T2, F2,
                   wt(blob, "dense_encoder.dense_conv_2.1.weight"),
                   wt(blob, "dense_encoder.dense_conv_2.1.bias"),
                   wt(blob, "dense_encoder.dense_conv_2.2.weight"), 1e-5f);
    free(x2);

    float *blk = alloc_n((size_t)HID*T2*F2);
    for (int i = 0; i < 30; i++) {
        reuse_tfmamba_block(blob, i, enc, blk, 1, T2, F2);
        memcpy(enc, blk, (size_t)HID*T2*F2*sizeof(float));
    }
    free(blk);

    // decoders produce [N,hid,T2*4,F2*2]; crop to [F_in, T_in]
    const int Tx = T2*4, Fx = F2*2;
    float *dm = alloc_n((size_t)HID*Tx*Fx);
    decoder_pre(blob, "mask_decoder", enc, 1, T2, F2, dm);
    float *dp = alloc_n((size_t)HID*Tx*Fx);
    decoder_pre(blob, "phase_decoder", enc, 1, T2, F2, dp);
    free(enc);

    float *fa = alloc_n((size_t)Tx*Fx), *fr = alloc_n((size_t)Tx*Fx), *fi = alloc_n((size_t)Tx*Fx);
    conv2d_general(dm, 1, HID, Tx, Fx,
                   wt(blob, "mask_decoder.final_conv.weight"),
                   wt(blob, "mask_decoder.final_conv.bias"),
                   1, 1, 1, 0, 0, 1, 1, 1, 1, fa);
    conv2d_general(dp, 1, HID, Tx, Fx,
                   wt(blob, "phase_decoder.phase_conv_r.weight"),
                   wt(blob, "phase_decoder.phase_conv_r.bias"),
                   1, 1, 1, 0, 0, 1, 1, 1, 1, fr);
    conv2d_general(dp, 1, HID, Tx, Fx,
                   wt(blob, "phase_decoder.phase_conv_i.weight"),
                   wt(blob, "phase_decoder.phase_conv_i.bias"),
                   1, 1, 1, 0, 0, 1, 1, 1, 1, fi);
    free(dm); free(dp);

    for (int t = 0; t < T_in; t++)
        for (int f = 0; f < F_in; f++) {
            out_mag[(size_t)f*T_in + t] = fa[(size_t)t*Fx + f];
            out_pha[(size_t)f*T_in + t] = atan2f(fi[(size_t)t*Fx + f], fr[(size_t)t*Fx + f]);
        }
    free(fa); free(fr); free(fi);

    // "strange sweep artifact" removal, matching inference.py
    float *mz = alloc_n((size_t)F_in*T_in);
    for (int t = 0; t < T_in; t++) {
        int nz = 0;
        for (int f = 0; f < F_in; f++) {
            float a = out_mag[(size_t)f*T_in + t];
            float m = a > 0.0f ? expm1f(a) : 0.0f;
            mz[(size_t)f*T_in + t] = m;
            if (m == 0.0f) nz++;
        }
        if ((float)nz / (float)F_in > 0.5f)
            for (int f = 0; f < F_in; f++) out_mag[(size_t)f*T_in + t] = 0.0f;
    }
    free(mz);
}
