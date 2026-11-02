// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
#ifndef REUSE_GPU_H
#define REUSE_GPU_H
#include "blob.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct reuse_gpu_s reuse_gpu_t;

// Upload every blob tensor into one device buffer. Returns NULL on failure.
reuse_gpu_t *reuse_gpu_create(const reuse_blob_t *blob);
void         reuse_gpu_destroy(reuse_gpu_t *g);
void         reuse_gpu_set_skips(void);
void         reuse_gpu_get_times(double *enc, double *blk, double *dec, double *head);
const float *reuse_gpu_w(reuse_gpu_t *g, const char *name);   // device pointer
int          reuse_gpu_ok(reuse_gpu_t *g, const char *where); // print+report CUDA errors

// Encoder: xin [1,2,Tp,Fp] (device) -> enc [1,64,T2,F2] (device), T2=(Tp-1)/4+1, F2=(Fp-3)/2+1
int reuse_gpu_encoder(reuse_gpu_t *g, const float *xin, float *enc, int Tp, int Fp);

// One TFMamba block: x [1,64,T,F] -> y [1,64,T,F]
int reuse_gpu_tfmamba(reuse_gpu_t *g, int idx, const float *x, float *y, int T, int F);

// Mag/Phase decoder pre-stage: x [1,64,T,F] -> y [1,64,T*4,F*2]
int reuse_gpu_decoder_pre(reuse_gpu_t *g, const char *prefix, const float *x, float *y, int T, int F);

// Full forward: mag/pha [F,T] (device) -> amp/pha [F,T] (device)
int reuse_gpu_forward(reuse_gpu_t *g, const float *mag, const float *pha,
                      int F, int T, float *amp_out, float *pha_out);

// 1x1 conv C_in -> 1 : x [C_in,H,W] -> y [H,W]
int reuse_gpu_conv1x1_to1(reuse_gpu_t *g, const float *x, const char *pfx, float *y, int C_in, int H, int W);

#ifdef __cplusplus
}
#endif
#endif
