// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
// Write the engine STFT of a WAV as in_mag.raw / in_pha.raw, so the model gate
// can be fed the reference STFT (isolates model+iSTFT from STFT conditioning).
#include "wav_io.h"
#include "stft_dft.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
static int make_even(int v){ v=(int)round((double)v); return v%2==0?v:v+1; }
int main(int argc, char **argv) {
    if (argc < 2) { printf("usage: %s <in.wav> <out_prefix>\n", argv[0]); return 1; }
    float *wav=NULL; int nch=0,sr=0,nf=0;
    if (wav_read_f32(argv[1], &wav, &nch, &sr, &nf)) { printf("wav fail\n"); return 1; }
    if (nch > 1) { float*m=malloc((size_t)nf*sizeof(float));
        for(int i=0;i<nf;i++){float s=0;for(int c=0;c<nch;c++)s+=wav[i*nch+c]; m[i]=s/nch;} free(wav); wav=m; }
    int nfft=make_even(320*sr/8000), hop=make_even(40*sr/8000), win=make_even(320*sr/8000);
    int F=nfft/2+1, T=1+((nf+2*(nfft/2))-nfft)/hop;
    float *mag=malloc((size_t)F*T*sizeof(float)), *pha=malloc((size_t)F*T*sizeof(float));
    reuse_stft_mag_pha(wav, nf, nfft, hop, win, mag, pha, F, T);
    char p[512];
    mkdir(argv[2], 0777);
    snprintf(p,sizeof(p),"%s/in_mag.raw", argv[2]); FILE*f=fopen(p,"wb");
    if (!f) { printf("cannot write %s\n", p); return 1; }
    fwrite(mag,sizeof(float),(size_t)F*T,f); fclose(f);
    snprintf(p,sizeof(p),"%s/in_pha.raw", argv[2]); f=fopen(p,"wb");
    if (!f) { printf("cannot write %s\n", p); return 1; }
    fwrite(pha,sizeof(float),(size_t)F*T,f); fclose(f);
    printf("sr=%d nfft=%d hop=%d F=%d T=%d\n", sr, nfft, hop, F, T);
    return 0;
}
