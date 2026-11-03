// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
// GPU driver: CPU STFT -> GPU model -> CPU iSTFT, with RTF timing.
#include "blob.h"
#include "wav_io.h"
#include "stft_dft.cuh"
#include "istft_dft.cuh"
#include "reuse_gpu.h"
#include "reuse_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

static int make_even(int v){ v = (int)round((double)v); return v%2==0?v:v+1; }
static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec + 1e-9*ts.tv_nsec; }

int main(int argc, char **argv)
{
    if (argc < 4) { printf("usage: %s <blob> <in.wav> <out.pcm> [reps]\n", argv[0]); return 1; }
    int reps = argc > 4 ? atoi(argv[4]) : 3;
    reuse_blob_t blob;
    if (reuse_blob_load(argv[1], &blob)) { printf("blob fail\n"); return 1; }
    float *wav = NULL; int nch=0, sr=0, nf=0;
    if (wav_read_f32(argv[2], &wav, &nch, &sr, &nf)) { printf("wav fail\n"); return 1; }
    if (nch > 1) {
        float *m = (float*)malloc((size_t)nf*sizeof(float));
        for (int i=0;i<nf;i++){ float s=0; for(int c=0;c<nch;c++) s+=wav[i*nch+c]; m[i]=s/nch; }
        free(wav); wav = m; nch = 1;
    }
    int n_fft=320, hp=40, ws=320, cfg_sr=8000;
    int nfft=make_even(n_fft*sr/cfg_sr), hop=make_even(hp*sr/cfg_sr), wsize=make_even(ws*sr/cfg_sr);
    int F = nfft/2+1;
    int T = 1 + ((nf + 2*(nfft/2)) - nfft)/hop;
    if (T < 1) T = 1;
    printf("sr=%d nf=%d nfft=%d hop=%d F=%d T=%d\n", sr, nf, nfft, hop, F, T);
    fflush(stdout);

    reuse_gpu_t *g = reuse_gpu_create(&blob);
    if (!g) return 1;
    // The weights now live on the GPU; on the Nano's 3 GiB shared-memory
    // device every host allocation matters, so drop the 38 MiB blob copy.
    reuse_blob_free(&blob);
    if (getenv("REUSE_MEM")) {
        size_t fr=0, to=0; cudaMemGetInfo(&fr,&to);
        printf("[mem] free %zu MiB / total %zu MiB\n", fr>>20, to>>20); fflush(stdout);
    }

    float *d_mag, *d_pha, *d_amp, *d_phaout;
    size_t nsp = (size_t)F*T;
    cudaMalloc(&d_mag, nsp*sizeof(float));
    cudaMalloc(&d_pha, nsp*sizeof(float));
    cudaMalloc(&d_amp, nsp*sizeof(float));
    cudaMalloc(&d_phaout, nsp*sizeof(float));
    float *h_mag = (float*)malloc(nsp*sizeof(float));
    float *h_pha = (float*)malloc(nsp*sizeof(float));

    const char *mfp = getenv("REUSE_MAG"), *pfp = getenv("REUSE_PHA");
    double t_stft = 0;
    if (mfp && pfp) {
        FILE *f = fopen(mfp, "rb");
        if (!f) { printf("cannot open %s\n", mfp); return 1; }
        if (fread(h_mag, sizeof(float), nsp, f) != nsp) { printf("short read %s\n", mfp); return 1; }
        fclose(f);
        f = fopen(pfp, "rb");
        if (!f) { printf("cannot open %s\n", pfp); return 1; }
        if (fread(h_pha, sizeof(float), nsp, f) != nsp) { printf("short read %s\n", pfp); return 1; }
        fclose(f);
        printf("using reference STFT\n");
    } else {
        double t0 = now_s();
        reuse_stft_mag_pha(wav, nf, nfft, hop, wsize, h_mag, h_pha, F, T);
        t_stft = now_s() - t0;
        printf("cpu stft %.3f s\n", t_stft);
    }
    fflush(stdout);
    cudaMemcpy(d_mag, h_mag, nsp*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pha, h_pha, nsp*sizeof(float), cudaMemcpyHostToDevice);

    float *h_amp = (float*)malloc(nsp*sizeof(float));
    float *h_phaout = (float*)malloc(nsp*sizeof(float));
    float *out = (float*)calloc((size_t)nf, sizeof(float));

    double best = 1e30, t_load = 0;
    for (int r = 0; r < reps; r++) {
        double t0 = now_s();
        if (reuse_gpu_forward(g, d_mag, d_pha, F, T, d_amp, d_phaout)) { printf("forward fail rep %d\n", r); return 1; }
        cudaDeviceSynchronize();
        double dt = now_s() - t0;
        if (r > 0 && dt < best) best = dt;
        printf("rep %d: %.3f s\n", r, dt); fflush(stdout);
    }
    double t1 = now_s();
    cudaMemcpy(h_amp, d_amp, nsp*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_phaout, d_phaout, nsp*sizeof(float), cudaMemcpyDeviceToHost);
    t_load = now_s() - t1;

    double t_istft = 0;
    if (getenv("REUSE_GPU_ISTFT")) {
        float *d_out = NULL;
        if (cudaMalloc(&d_out, (size_t)nf*sizeof(float))) { printf("istft gpu OOM\n"); return 1; }
        double t0 = now_s();
        if (reuse_gpu_istft(g, d_amp, d_phaout, F, T, nfft, hop, wsize, d_out, nf)) {
            printf("gpu istft fail\n"); return 1;
        }
        cudaDeviceSynchronize();
        t_istft = now_s() - t0;
        cudaMemcpy(out, d_out, (size_t)nf*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_out);
    } else {
        double t0 = now_s();
        reuse_istft(h_amp, h_phaout, F, T, nfft, hop, wsize, out, nf);
        t_istft = now_s() - t0;
    }

    const char *da = getenv("REUSE_DUMP_AMP");
    if (da) { FILE *f = fopen(da, "wb"); fwrite(h_amp, sizeof(float), nsp, f); fclose(f); }
    const char *dp = getenv("REUSE_DUMP_PHA");
    if (dp) { FILE *f = fopen(dp, "wb"); fwrite(h_phaout, sizeof(float), nsp, f); fclose(f); }
    const char *dw = getenv("REUSE_DUMP_AUDIO");
    if (dw) { FILE *f = fopen(dw, "wb"); fwrite(out, sizeof(float), (size_t)nf, f); fclose(f); }

    FILE *fo = fopen(argv[3], "wb");
    if (fo) {
        for (int i=0;i<nf;i++){ short s = (short)fmaxf(-32768.0f, fminf(32767.0f, out[i]*32768.0f)); fwrite(&s,2,1,fo); }
        fclose(fo);
    }
    double audio_s = (double)nf / sr;
    reuse_gpu_print_kprof();
    double te, tb, td, th; reuse_gpu_get_times(&te, &tb, &td, &th);
    printf("phases: encoder %.3f s | 30xTFMamba %.3f s | decoders %.3f s | heads %.3f s\n", te, tb, td, th);
    printf("=== RTF (model GPU, STFT CPU, iSTFT %s) ===\n", getenv("REUSE_GPU_ISTFT") ? "GPU" : "CPU");
    printf("audio      %.4f s\n", audio_s);
    printf("model best %.4f s   RTF_model %.4f\n", best, best/audio_s);
    printf("stft(cpu)  %.4f s   RTF %.4f\n", t_stft, t_stft/audio_s);
    printf("istft(cpu) %.4f s   RTF %.4f\n", t_istft, t_istft/audio_s);
    printf("total(best+stft+istft) %.4f s  RTF_total %.4f\n", best+t_stft+t_istft, (best+t_stft+t_istft)/audio_s);
    printf("d2h %.4f s\n", t_load);
    fflush(stdout);

    cudaFree(d_mag); cudaFree(d_pha); cudaFree(d_amp); cudaFree(d_phaout);
    free(h_mag); free(h_pha); free(h_amp); free(h_phaout); free(wav); free(out);
    reuse_gpu_destroy(g); reuse_blob_free(&blob);
    return 0;
}
