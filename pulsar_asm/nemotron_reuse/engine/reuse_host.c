// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
#include "blob.h"
#include "wav_io.h"
#include "stft_dft.cuh"
#include "istft_dft.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int make_even(int v){ v = (int)round(v); return v%2==0?v:v+1; }

void conv2d_1x1(const float *x, int N, int C_in, int H, int W,
                const float *w, const float *b,
                float *y, int C_out);
void instancenorm2d_prelu(const float *x, float *y, int N, int C, int H, int W,
                          const float *w, const float *b, const float *alpha, float eps);
void mamba_scan_fwd(const float *x, const float *dt, const float *A_log, const float *D,
                    const float *B, const float *C,
                    float *y, int Bsz, int L, int D_in, int N);

int main(int argc, char **argv) {
    if (argc < 5) {
        printf("usage: %s <model.reuse> <input.wav> <output.wav> <mode:full|chunk>\n", argv[0]);
        return 1;
    }
    reuse_blob_t blob;
    if (reuse_blob_load(argv[1], &blob) != 0) {
        printf("failed to load blob\n");
        return 1;
    }
    printf("loaded %d tensors\n", blob.ntensors);
    float *wav = NULL; int nch=0,sr=0,nf=0;
    if (wav_read_f32(argv[2], &wav, &nch, &sr, &nf) != 0) {
        printf("failed to read wav\n");
        return 1;
    }
    printf("sr=%d nch=%d nf=%d\n", sr, nch, nf);
    int n_fft=320,hp=40,ws=320,cfg_sr=8000;
    int nfft=make_even(n_fft*sr/cfg_sr), hop=make_even(hp*sr/cfg_sr), wsize=make_even(ws*sr/cfg_sr);
    int F = nfft/2+1;
    int num_frames = 1 + ((nf + 2*(nfft/2)) - nfft)/hop;
    if (num_frames < 1) num_frames = 1;
    printf("nfft=%d hop=%d F=%d num_frames=%d\n", nfft, hop, F, num_frames);
    float *mag = (float*)malloc(F*num_frames*sizeof(float));
    float *pha = (float*)malloc(F*num_frames*sizeof(float));
    reuse_stft_mag_pha(wav, nf, nfft, hop, wsize, mag, pha, F, num_frames);
    printf("stft done\n");
    // TODO: full model forward pass
    float *out = (float*)calloc(nf, sizeof(float));
    reuse_istft(mag, pha, F, num_frames, nfft, hop, wsize, out, nf);
    printf("istft done\n");
    // write simple raw
    FILE *fo = fopen(argv[3], "wb");
    for (int i=0;i<nf;i++) { short s = (short)(out[i]*32768.0f); fwrite(&s,2,1,fo); }
    fclose(fo);
    free(mag); free(pha); free(wav); free(out);
    reuse_blob_free(&blob);
    return 0;
}
#include "model.h"
