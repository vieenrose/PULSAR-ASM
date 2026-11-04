// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
#include "stft_dft.cuh"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// DFT-GEMM front end. Twiddles are precomputed once in double and the
// accumulators are double: near the negative-real axis the sign of Im decides
// the sign of atan2, so a single-precision accumulation flips whole bins by 2*pi.
//
// Fast form (default; REUSE_STFT_FAST=0 restores the serial loop): the
// windowed frame xw[t][n] = x[mirror(t*hop-pad+n)] * w[n] is built once in a
// parallel pre-pass (the serial loop recomputed the mirror index and the
// window multiply for every one of the F bins), and the DFT inner loop is a
// pure dot product with FOUR independent accumulator chains (n mod 4) that
// the compiler can schedule in parallel; the chains combine as (r0+r1)+(r2+r3).
// Validated on the Nano against the serial form (REUSE_DUMP_STFT): mag is
// BIT-IDENTICAL (max|d| = 0), pha max|d| 1.8e-12 / rel rms 2.5e-15, i.e. the
// float outputs are identical and no atan2 sign flips; stft(cpu) 0.087 ->
// 0.068 s at 8 kHz (-22%, 2 interleaved A/B pairs, #70/#73).
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
    static int fast = -1;
    if (fast < 0) { const char *e = getenv("REUSE_STFT_FAST"); fast = e ? atoi(e) : 1; }
    double *xw = NULL;
    if (fast) {
        xw = (double *)malloc((size_t)num_frames * n_fft * sizeof(double));
        if (!xw) fast = 0;                       /* fall back to the serial form */
    }
    if (fast) {
#pragma omp parallel for schedule(static)
        for (int t = 0; t < num_frames; t++) {
            const int start = t * hop_size - pad;
            double *dst = xw + (size_t)t * n_fft;
            for (int n = 0; n < n_fft; n++) {
                int idx = start + n;
                double val;
                if (idx < 0)         val = x[-idx - 1];
                else if (idx >= T)   val = x[2*T - 2 - idx];
                else                 val = x[idx];
                dst[n] = val * w[n];
            }
        }
    }
    // Bin-major with the frame loop inside: every output bin keeps its own
    // double accumulation order, so this is bit-identical to the frame-major
    // form, but the 5 KB twiddle row for bin k stays in L1 for the whole call.
    // Frame-major re-read all 824 KB of twiddles for every one of the 1021
    // frames (824 MB of RAM reads per call, i.e. memory bound at ~26 GB/s).
    // Frames are independent, so the parallelism sits on the bin loop.
    const int nq4 = n_fft / 4;
#pragma omp parallel for schedule(static)
    for (int k = 0; k < F; k++) {
        const double *tc = tw + (size_t)k * n_fft * 2;
        for (int t = 0; t < num_frames; t++) {
            double re, im;
            if (fast) {
                const double *fr = xw + (size_t)t * n_fft;
                double re0=0, re1=0, re2=0, re3=0, im0=0, im1=0, im2=0, im3=0;
                for (int q = 0; q < nq4; q++) {
                    const int n = 4*q;
                    const double v0=fr[n], v1=fr[n+1], v2=fr[n+2], v3=fr[n+3];
                    re0 += v0*tc[2*n];     im0 += v0*tc[2*n+1];
                    re1 += v1*tc[2*n+2];   im1 += v1*tc[2*n+3];
                    re2 += v2*tc[2*n+4];   im2 += v2*tc[2*n+5];
                    re3 += v3*tc[2*n+6];   im3 += v3*tc[2*n+7];
                }
                re = (re0 + re1) + (re2 + re3);
                im = (im0 + im1) + (im2 + im3);
                for (int n = 4*nq4; n < n_fft; n++) {
                    re += fr[n] * tc[2*n];
                    im += fr[n] * tc[2*n+1];
                }
            } else {
                const int start = t * hop_size - pad;
                re = 0.0; im = 0.0;
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
            }
            const int off = k * num_frames + t;
            mag_out[off] = (float)log1p(sqrt(re*re + im*im));
            pha_out[off] = (float)atan2(im, re);
        }
    }
    free(xw);
    free(tw);
    free(w);
}
