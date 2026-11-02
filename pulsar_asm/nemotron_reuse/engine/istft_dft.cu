// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
#include "istft_dft.cuh"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

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
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < num_frames; t++) {
        // inverse DFT
        float frame[4096];
        memset(frame, 0, n_fft * sizeof(float));
        // Real-valued inverse DFT from the one-sided spectrum: interior bins
        // carry the conjugate pair and must be counted twice.
        for (int n = 0; n < n_fft; n++) {
            double re = 0.0;
            for (int k = 0; k < F; k++) {
                int off = k * num_frames + t;
                double magv = exp((double)mag[off]) - 1.0; // relu_log1p inverse
                if (magv < 0.0) magv = 0.0;
                double phav = (double)pha[off];
                double cv = magv * cos(phav);
                double sv = magv * sin(phav);
                double ang = 2.0 * M_PI * n * k / (double)n_fft;
                double m = (k == 0 || 2*k == n_fft) ? 1.0 : 2.0;
                re += m * (cv * cos(ang) - sv * sin(ang));
            }
            frame[n] = (float)(re / (double)n_fft);
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
