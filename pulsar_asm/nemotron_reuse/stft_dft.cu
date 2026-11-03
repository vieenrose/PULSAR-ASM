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
        const double *tc = tw + (size_t)k * n_fft * 2;
        for (int t = 0; t < num_frames; t++) {
            const int start = t * hop_size - win_size / 2;
            // Four accumulator pairs and a branch-free interior span: the
            // mirrored-pad tests (idx < 0, idx >= T) only apply to the edge
            // samples, and a loop whose body carries those branches does not
            // vectorise. Splitting them out lets gcc turn the interior into
            // packed-double FMAs. Splitting the sum 4 ways changes the
            // rounding at ~1e-16 relative, far inside the 1e-4 gate (the
            // fp32 phase-sign sensitivity noted above is a single-precision
            // concern; this accumulator is double).
            double re[4] = { 0.0, 0.0, 0.0, 0.0 }, im[4] = { 0.0, 0.0, 0.0, 0.0 };
            const int lo = (start < 0) ? -start : 0;
            const int hi = (start + n_fft > T) ? (T - start) : n_fft;
            const int ilo = lo, ihi = hi > lo ? hi : lo;
            for (int n = ilo; n < ihi; n++) {
                const double val = (double)x[start + n] * (double)w[n];
                re[n & 3] += val * tc[2*n];
                im[n & 3] += val * tc[2*n+1];
            }
            for (int n = 0; n < ilo; n++) {
                int idx = start + n;
                double val;
                if (idx < 0)         val = x[-idx - 1];
                else if (idx >= T)   val = x[2*T - 2 - idx];
                else                 val = x[idx];
                val *= w[n];
                re[n & 3] += val * tc[2*n];
                im[n & 3] += val * tc[2*n+1];
            }
            for (int n = ihi; n < n_fft; n++) {
                int idx = start + n;
                double val;
                if (idx < 0)         val = x[-idx - 1];
                else if (idx >= T)   val = x[2*T - 2 - idx];
                else                 val = x[idx];
                val *= w[n];
                re[n & 3] += val * tc[2*n];
                im[n & 3] += val * tc[2*n+1];
            }
            const double re_ = (re[0] + re[1]) + (re[2] + re[3]);
            const double im_ = (im[0] + im[1]) + (im[2] + im[3]);
            const int off = k * num_frames + t;
            mag_out[off] = (float)log1p(sqrt(re_*re_ + im_*im_));
            pha_out[off] = (float)atan2(im_, re_);
        }
    }
    free(tw);
    free(w);
}
