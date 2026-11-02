// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
#include "istft_dft.cuh"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void build_hann(float *w, int n) {
    for (int i = 0; i < n; i++) {
        w[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / n));
    }
}

void reuse_istft(const float *mag, const float *pha, int F, int num_frames,
                 int n_fft, int hop_size, int win_size,
                 float *out, int out_len) {
    float *w = (float *)malloc(n_fft * sizeof(float));
    for (int i = 0; i < n_fft; i++) w[i] = 1.0f;
    for (int i = 0; i < win_size; i++) {
        w[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / win_size));
    }
    int pad = n_fft / 2;
    memset(out, 0, out_len * sizeof(float));
    float *window_sum = (float *)calloc(out_len, sizeof(float));
    for (int t = 0; t < num_frames; t++) {
        // inverse DFT
        float frame[4096];
        memset(frame, 0, n_fft * sizeof(float));
        for (int n = 0; n < n_fft; n++) {
            float re = 0.0f;
            for (int k = 0; k < F; k++) {
                int off = k * num_frames + t;
                float magv = expf(mag[off]) - 1.0f; // log1p inverse for relu_log1p
                if (magv < 0.0f) magv = 0.0f;
                float phav = pha[off];
                float cv = magv * cosf(phav);
                float sv = magv * sinf(phav);
                float ang = 2.0f * (float)M_PI * n * k / (float)n_fft;
                re += cv * cosf(ang) - sv * sinf(ang);
            }
            frame[n] = re / (float)n_fft;
        }
        int start = t * hop_size - pad;
        for (int n = 0; n < n_fft && n < win_size; n++) {
            int idx = start + n;
            if (idx >= 0 && idx < out_len) {
                out[idx] += frame[n] * w[n];
                window_sum[idx] += w[n] * w[n];
            }
        }
    }
    for (int i = 0; i < out_len; i++) {
        if (window_sum[i] > 1e-12f) out[i] /= window_sum[i];
    }
    free(w);
    free(window_sum);
}
