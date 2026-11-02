// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
#include "blob.h"
#include "wav_io.h"
#include "stft_dft.cuh"
#include "istft_dft.cuh"
#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int make_even(int v){ v = (int)round(v); return v%2==0?v:v+1; }

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
    if (nch > 1) {
        float *mono = (float*)malloc((size_t)nf*sizeof(float));
        for (int i=0;i<nf;i++){ float s=0; for(int c=0;c<nch;c++) s+=wav[i*nch+c]; mono[i]=s/nch; }
        free(wav); wav=mono; nch=1;
    }
    printf("sr=%d nch=%d nf=%d\n", sr, nch, nf);
    int n_fft=320,hp=40,ws=320,cfg_sr=8000;
    int nfft=make_even(n_fft*sr/cfg_sr), hop=make_even(hp*sr/cfg_sr), wsize=make_even(ws*sr/cfg_sr);
    int F = nfft/2+1;
    int num_frames = 1 + ((nf + 2*(nfft/2)) - nfft)/hop;
    if (num_frames < 1) num_frames = 1;
    printf("nfft=%d hop=%d F=%d num_frames=%d\n", nfft, hop, F, num_frames);
    float *mag = (float*)malloc((size_t)F*num_frames*sizeof(float));
    float *pha = (float*)malloc((size_t)F*num_frames*sizeof(float));
    const char *mfp = getenv("REUSE_MAG"), *pfp = getenv("REUSE_PHA");
    if (mfp && pfp) {
        FILE *f = fopen(mfp, "rb");
        if (fread(mag, sizeof(float), (size_t)F*num_frames, f) != (size_t)F*num_frames) { printf("mag short\n"); return 1; }
        fclose(f);
        f = fopen(pfp, "rb");
        if (fread(pha, sizeof(float), (size_t)F*num_frames, f) != (size_t)F*num_frames) { printf("pha short\n"); return 1; }
        fclose(f);
        printf("stft loaded from %s\n", mfp);
    } else {
        reuse_stft_mag_pha(wav, nf, nfft, hop, wsize, mag, pha, F, num_frames);
    }
    const char *dip = getenv("REUSE_DUMP_INPHA");
    if (dip) { FILE *f = fopen(dip, "wb"); fwrite(pha, sizeof(float), (size_t)F*num_frames, f); fclose(f); }
    const char *dim = getenv("REUSE_DUMP_INMAG");
    if (dim) { FILE *f = fopen(dim, "wb"); fwrite(mag, sizeof(float), (size_t)F*num_frames, f); fclose(f); }
    float *mag_o = (float*)malloc((size_t)F*num_frames*sizeof(float));
    float *pha_o = (float*)malloc((size_t)F*num_frames*sizeof(float));
    int full = (argc >= 5 && strcmp(argv[4], "full") == 0);
    if (full) {
        reuse_forward(&blob, mag, pha, 1, F, num_frames, mag_o, pha_o);
    } else {
        memcpy(mag_o, mag, (size_t)F*num_frames*sizeof(float));
        memcpy(pha_o, pha, (size_t)F*num_frames*sizeof(float));
    }
    const char *dap = getenv("REUSE_DUMP_AMP");
    if (dap) { FILE *f = fopen(dap, "wb"); fwrite(mag_o, sizeof(float), (size_t)F*num_frames, f); fclose(f); }
    const char *dpp = getenv("REUSE_DUMP_PHA");
    if (dpp) { FILE *f = fopen(dpp, "wb"); fwrite(pha_o, sizeof(float), (size_t)F*num_frames, f); fclose(f); }
    float *out = (float*)calloc((size_t)nf, sizeof(float));
    reuse_istft(mag_o, pha_o, F, num_frames, nfft, hop, wsize, out, nf);
    FILE *fo = fopen(argv[3], "wb");
    if (fo) {
        for (int i=0;i<nf;i++) { short s = (short)fmaxf(-32768.0f, fminf(32767.0f, out[i]*32768.0f)); fwrite(&s,2,1,fo); }
        fclose(fo);
    }
    const char *drw = getenv("REUSE_DUMP_AUDIO");
    if (drw) { FILE *f = fopen(drw, "wb"); fwrite(out, sizeof(float), (size_t)nf, f); fclose(f); }
    free(mag); free(pha); free(mag_o); free(pha_o); free(wav); free(out);
    reuse_blob_free(&blob);
    return 0;
}
