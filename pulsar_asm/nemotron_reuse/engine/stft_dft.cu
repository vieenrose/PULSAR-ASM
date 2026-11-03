// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
#include "stft_dft.cuh"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

// DFT-GEMM front end. Twiddles are precomputed once in double and the
// accumulators are double: near the negative-real axis the sign of Im decides
// the sign of atan2, so a single-precision accumulation flips whole bins by 2*pi.
void reuse_stft_mag_pha(const float *x, int T, int n_fft, int hop_size, int win_size,
                        float *mag_out, float *pha_out, int F, int num_frames) {
    float *w = (float *)malloc((size_t)n_fft * sizeof(float));
    for (int i = 0; i < n_fft; i++) {
        w[i] = (i < win_size)
             ? 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (float)win_size))
             : 0.0f;
    }
    // twiddle[k][n] = exp(-2*pi*i*n*k/n_fft)
    double *tw = (double *)malloc((size_t)F * n_fft * 2 * sizeof(double));
    if (!tw) { free(w); return; }
    const double pi = 3.14159265358979323846;
#pragma omp parallel for schedule(static)
    for (int k = 0; k < F; k++) {
        double *tc = tw + (size_t)k * n_fft * 2;
        for (int n = 0; n < n_fft; n++) {
            double ang = -2.0 * pi * ((double)n * (double)k) / (double)n_fft;
            tc[2*n]   = cos(ang);
            tc[2*n+1] = sin(ang);
        }
    }
    const int pad = n_fft / 2;
    // Bin-major with the frame loop inside: every output bin keeps its own
    // double accumulation order, so this is bit-identical to the frame-major
    // form, but the 5 KB twiddle row for bin k stays in L1 for the whole call.
    // Frame-major re-read all 824 KB of twiddles for every one of the 1021
    // frames (824 MB of RAM reads per call, i.e. memory bound at ~26 GB/s).
    // Frames are independent, so the parallelism sits on the bin loop.
#pragma omp parallel for schedule(static)
    for (int k = 0; k < F; k++) {
        const double *tc = tw + (size_t)k * n_fft * 2;
        for (int t = 0; t < num_frames; t++) {
            const int start = t * hop_size - pad;
            double re = 0.0, im = 0.0;
            for (int n = 0; n < n_fft; n++) {
                int idx = start + n;
                double val;
                if (idx < 0)         val = x[-idx - 1];
                else if (idx >= T)   val = x[2*T - 2 - idx];
                else                 val = x[idx];
                val *= w[n];
                re += val * tc[2*n];
                im += val * tc[2*n+1];
            }
            const int off = k * num_frames + t;
            mag_out[off] = (float)log1p(sqrt(re*re + im*im));
            pha_out[off] = (float)atan2(im, re);
        }
    }
    free(tw);
    free(w);
}
