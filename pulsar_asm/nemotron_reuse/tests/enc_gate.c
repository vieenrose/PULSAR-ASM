// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
// Encoder-only gate: STFT -> pad -> DenseEncoder -> dump. Compare vs PyTorch dump_enc.py
#include "blob.h"
#include "wav_io.h"
#include "stft_dft.cuh"
#include "conv2d_full.cuh"
#include "instancenorm.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const float *find_tensor(const reuse_blob_t *blob, const char *name) {
    const reuse_tensor_t *t = reuse_blob_find(blob, name);
    if (!t) { fprintf(stderr, "MISSING TENSOR: %s\n", name); exit(2); }
    return t->data;
}

static int make_even(int v){ v = (int)round((double)v); return v%2==0?v:v+1; }

// DenseBlock: PyTorch semantics.
//   skip = x
//   for i: x = IN+PReLU(conv(skip[i+1*hid -> hid], dil=(1<<i)))  ; skip = cat([x, skip])
//   returns x  (hid channels)
static void dense_block(const reuse_blob_t *blob, const char *prefix,
                        const float *x_in, int N, int T, int F, int hid,
                        float *out /* hid channels */) {
    int cap = hid*5*T*F;
    float *skip = (float*)malloc((size_t)cap*sizeof(float));
    float *skip2 = (float*)malloc((size_t)cap*sizeof(float));
    float *tmp = (float*)malloc((size_t)hid*T*F*sizeof(float));
    float *cur = (float*)malloc((size_t)hid*T*F*sizeof(float));
    memcpy(skip, x_in, (size_t)hid*T*F*sizeof(float));
    int c_skip = hid;
    for (int i = 0; i < 4; i++) {
        char key[160];
        sprintf(key, "%s.%d.0.weight", prefix, i); const float *w = find_tensor(blob, key);
        sprintf(key, "%s.%d.0.bias",   prefix, i); const float *b = find_tensor(blob, key);
        sprintf(key, "%s.%d.1.weight", prefix, i); const float *nw = find_tensor(blob, key);
        sprintf(key, "%s.%d.1.bias",   prefix, i); const float *nb = find_tensor(blob, key);
        sprintf(key, "%s.%d.2.weight", prefix, i); const float *pa = find_tensor(blob, key);
        int dil = 1 << i;
        conv2d_general(skip, N, c_skip, T, F, w, b, hid, 3, 3, dil, 1, dil, 1, 1, 1, tmp);
        instancenorm2d_prelu(tmp, cur, N, hid, T, F, nw, nb, pa, 1e-5f);
        // new skip = [cur, skip]
        memcpy(skip2, cur, (size_t)hid*T*F*sizeof(float));
        memcpy(skip2 + (size_t)hid*T*F, skip, (size_t)c_skip*T*F*sizeof(float));
        float *sw = skip; skip = skip2; skip2 = sw;
        c_skip += hid;
    }
    memcpy(out, cur, (size_t)hid*T*F*sizeof(float));
    free(skip); free(skip2); free(tmp); free(cur);
}

int main(int argc, char **argv) {
    if (argc < 4) { printf("usage: %s <blob> <in.wav> <out.raw> [nfft hop win]\n", argv[0]); return 1; }
    reuse_blob_t blob;
    if (reuse_blob_load(argv[1], &blob) != 0) { printf("blob fail\n"); return 1; }
    float *wav=NULL; int nch=0,sr=0,nf=0;
    if (wav_read_f32(argv[2], &wav, &nch, &sr, &nf) != 0) { printf("wav fail\n"); return 1; }
    if (nch > 1) { float*m=(float*)malloc((size_t)nf*sizeof(float));
        for(int i=0;i<nf;i++){float s=0;for(int c=0;c<nch;c++)s+=wav[i*nch+c];m[i]=s/nch;} free(wav); wav=m; }
    int nfft=320,hop=40,win=320;
    if (argc >= 7) { nfft=atoi(argv[4]); hop=atoi(argv[5]); win=atoi(argv[6]); }
    int F = nfft/2+1;
    int T = 1 + ((nf + 2*(nfft/2)) - nfft)/hop;
    const int hid = 64;
    printf("sr=%d nf=%d nfft=%d hop=%d F=%d T=%d\n", sr, nf, nfft, hop, F, T);
    fflush(stdout);

    float *mag=(float*)malloc((size_t)F*T*sizeof(float));
    float *pha=(float*)malloc((size_t)F*T*sizeof(float));
    const char *mfp = getenv("REUSE_MAG"), *pfp = getenv("REUSE_PHA");
    if (mfp && pfp) {
        FILE *f=fopen(mfp,"rb"); if(fread(mag,sizeof(float),(size_t)F*T,f)!=(size_t)F*T){printf("mag short\n");return 1;} fclose(f);
        f=fopen(pfp,"rb"); if(fread(pha,sizeof(float),(size_t)F*T,f)!=(size_t)F*T){printf("pha short\n");return 1;} fclose(f);
        printf("stft loaded from %s\n", mfp); fflush(stdout);
    } else {
        reuse_stft_mag_pha(wav, nf, nfft, hop, win, mag, pha, F, T);
        printf("stft done\n"); fflush(stdout);
    }

    // pad: T+2 rows, F+2 cols (zeros), 2 channels
    int Tp=T+2, Fp=F+2;
    float *xin=(float*)calloc((size_t)2*Tp*Fp, sizeof(float));
    for (int t=0;t<T;t++) for (int f=0;f<F;f++) {
        xin[0*Tp*Fp + t*Fp + f] = mag[f*T + t];   // [C,T,F] from [F,T]
        xin[1*Tp*Fp + t*Fp + f] = pha[f*T + t];
    }
    free(mag); free(pha);

    // dense_conv_1: 2 -> hid, 1x1, IN+PReLU
    float *x1=(float*)malloc((size_t)hid*Tp*Fp*sizeof(float));
    conv2d_general(xin, 1, 2, Tp, Fp,
                   find_tensor(&blob,"dense_encoder.dense_conv_1.0.weight"),
                   find_tensor(&blob,"dense_encoder.dense_conv_1.0.bias"),
                   hid, 1,1, 0,0, 1,1, 1,1, x1);
    free(xin);
    float *x1n=(float*)malloc((size_t)hid*Tp*Fp*sizeof(float));
    instancenorm2d_prelu(x1, x1n, 1, hid, Tp, Fp,
                   find_tensor(&blob,"dense_encoder.dense_conv_1.1.weight"),
                   find_tensor(&blob,"dense_encoder.dense_conv_1.1.bias"),
                   find_tensor(&blob,"dense_encoder.dense_conv_1.2.weight"), 1e-5f);
    free(x1);
    printf("conv1 done\n"); fflush(stdout);

    // DenseBlock
    float *xdb=(float*)malloc((size_t)hid*Tp*Fp*sizeof(float));
    dense_block(&blob, "dense_encoder.dense_block.dense_block", x1n, 1, Tp, Fp, hid, xdb);
    free(x1n);
    printf("dense_block done\n"); fflush(stdout);

    // dense_conv_2: (1,3) stride (4,2) + IN + PReLU
    int T2=(Tp-1)/4+1, F2=(Fp-3)/2+1;
    float *x2=(float*)malloc((size_t)hid*T2*F2*sizeof(float));
    conv2d_general(xdb, 1, hid, Tp, Fp,
                   find_tensor(&blob,"dense_encoder.dense_conv_2.0.weight"),
                   find_tensor(&blob,"dense_encoder.dense_conv_2.0.bias"),
                   hid, 1,3, 0,0, 1,1, 4,2, x2);
    free(xdb);
    float *enc=(float*)malloc((size_t)hid*T2*F2*sizeof(float));
    instancenorm2d_prelu(x2, enc, 1, hid, T2, F2,
                   find_tensor(&blob,"dense_encoder.dense_conv_2.1.weight"),
                   find_tensor(&blob,"dense_encoder.dense_conv_2.1.bias"),
                   find_tensor(&blob,"dense_encoder.dense_conv_2.2.weight"), 1e-5f);
    free(x2);
    printf("encoder out [%d,%d,%d] -> %s\n", hid, T2, F2, argv[3]); fflush(stdout);
    FILE *fp=fopen(argv[3],"wb"); fwrite(enc,sizeof(float),(size_t)hid*T2*F2,fp); fclose(fp);
    free(enc); free(wav);
    reuse_blob_free(&blob);
    return 0;
}
