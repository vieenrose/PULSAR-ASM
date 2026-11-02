#include "blob.h"
#include "reuse_gpu.h"
#include "wav_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
int main(int argc, char **argv) {
    if (argc < 3) { printf("usage: %s <blob> <enc_ref.raw> [Tp Fp]\n", argv[0]); return 1; }
    reuse_blob_t blob; if (reuse_blob_load(argv[1], &blob)) return 1;
    int Tp = argc > 3 ? atoi(argv[3]) : 1025, Fp = argc > 4 ? atoi(argv[4]) : 165;
    int hid = 64, T2 = (Tp-1)/4+1, F2 = (Fp-3)/2+1;
    size_t nin = (size_t)2*Tp*Fp;
    float *h_in = (float*)malloc(nin*sizeof(float));
    // build encoder input from the reference STFT: [B,2,T,F] padded
    const char *mfp = getenv("REUSE_MAG"), *pfp = getenv("REUSE_PHA");
    int F = Fp-2, T = Tp-2;
    float *mag = (float*)malloc((size_t)F*T*sizeof(float)), *pha = (float*)malloc((size_t)F*T*sizeof(float));
    FILE *f = fopen(mfp, "rb"); fread(mag, sizeof(float), (size_t)F*T, f); fclose(f);
    f = fopen(pfp, "rb"); fread(pha, sizeof(float), (size_t)F*T, f); fclose(f);
    memset(h_in, 0, nin*sizeof(float));
    for (int t = 0; t < T; t++) for (int ff = 0; ff < F; ff++) {
        h_in[0*Tp*Fp + t*Fp + ff] = mag[(size_t)ff*T + t];
        h_in[1*Tp*Fp + t*Fp + ff] = pha[(size_t)ff*T + t];
    }
    reuse_gpu_t *g = reuse_gpu_create(&blob);
    if (!g) return 1;
    float *d_in, *d_enc;
    cudaMalloc(&d_in, nin*sizeof(float)); cudaMalloc(&d_enc, (size_t)hid*T2*F2*sizeof(float));
    cudaMemcpy(d_in, h_in, nin*sizeof(float), cudaMemcpyHostToDevice);
    if (reuse_gpu_encoder(g, d_in, d_enc, Tp, Fp)) { printf("encoder fail\n"); return 1; }
    cudaDeviceSynchronize();
    float *h_out = (float*)malloc((size_t)hid*T2*F2*sizeof(float));
    cudaMemcpy(h_out, d_enc, (size_t)hid*T2*F2*sizeof(float), cudaMemcpyDeviceToHost);
    float *ref = (float*)malloc((size_t)hid*T2*F2*sizeof(float));
    f = fopen(argv[2], "rb"); size_t got = fread(ref, sizeof(float), (size_t)hid*T2*F2, f); fclose(f);
    double sr = 0, sr2 = 0;
    for (size_t i = 0; i < (size_t)hid*T2*F2; i++) { double d = ref[i]-h_out[i]; sr += d*d; sr2 += (double)ref[i]*ref[i]; }
    printf("GPU encoder vs ref: n=%zu rel_rms=%.3e\n", got, sqrt(sr/(double)got)/sqrt(sr2/(double)got));
    const char *dw = getenv("REUSE_DUMP_ENC");
    if (dw) { f = fopen(dw, "wb"); fwrite(h_out, sizeof(float), (size_t)hid*T2*F2, f); fclose(f); }
    cudaFree(d_in); cudaFree(d_enc); free(h_in); free(h_out); free(ref); free(mag); free(pha);
    reuse_gpu_destroy(g); reuse_blob_free(&blob);
    return 0;
}
