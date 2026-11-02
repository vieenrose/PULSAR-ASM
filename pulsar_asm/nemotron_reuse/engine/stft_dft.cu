// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
#include "stft_dft.cuh"
#include <math.h>
#include <stdlib.h>

static void build_hann(float *w, int n) {
    for (int i = 0; i < n; i++) {
        w[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / n));
    }
}

void reuse_stft_mag_pha(const float *x, int T, int n_fft, int hop_size, int win_size,
                        float *mag_out, float *pha_out, int F, int num_frames) {
    float *w = (float *)malloc(n_fft * sizeof(float));
    for (int i = 0; i < n_fft; i++) w[i] = 1.0f;
    for (int i = 0; i < win_size; i++) {
        w[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / win_size));
    }
    int pad = n_fft / 2;
    for (int t = 0; t < num_frames; t++) {
        int start = t * hop_size - pad;
        for (int k = 0; k < F; k++) {
            float re = 0.0f, im = 0.0f;
            for (int n = 0; n < n_fft; n++) {
                int idx = start + n;
                float val = 0.0f;
                if (idx < 0) val = x[-idx - 1];
                else if (idx >= T) val = x[2*T - 2 - idx];
                else val = x[idx];
                val *= w[n];
                float ang = -2.0f * (float)M_PI * n * k / (float)n_fft;
                re += val * cosf(ang);
                im += val * sinf(ang);
            }
            int off = k * num_frames + t;
            float mag = sqrtf(re*re + im*im);
            mag_out[off] = log1pf(mag);
            pha_out[off] = atan2f(im, re);
        }
    }
    free(w);
}
