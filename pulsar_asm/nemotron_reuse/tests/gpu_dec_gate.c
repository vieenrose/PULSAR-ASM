#include "blob.h"
#include "reuse_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
int main(int argc, char **argv) {
    if (argc < 4) { printf("usage: %s <blob> <enc.raw> <ref.raw> [T F] [which]\n", argv[0]); return 1; }
    int T = argc > 4 ? atoi(argv[4]) : 256, F = argc > 5 ? atoi(argv[5]) : 81;
    const char *which = argc > 6 ? argv[6] : "mask";
    reuse_blob_t b; if (reuse_blob_load(argv[1], &b)) return 1;
    float *h = malloc((size_t)64*T*F*sizeof(float));
    FILE *f = fopen(argv[2], "rb"); if (fread(h, sizeof(float), (size_t)64*T*F, f) != (size_t)64*T*F) return 1; fclose(f);
    reuse_gpu_t *g = reuse_gpu_create(&b); if (!g) return 1;
    float *dx = NULL, *dy = NULL;
    cudaMalloc(&dx, (size_t)64*T*F*sizeof(float));
    cudaMalloc(&dy, (size_t)64*T*4*F*2*sizeof(float));
    cudaMemcpy(dx, h, (size_t)64*T*F*sizeof(float), cudaMemcpyHostToDevice);
    if (reuse_gpu_decoder_pre(g, which, dx, dy, T, F)) { printf("decoder fail\n"); return 1; }
    cudaDeviceSynchronize();
    size_t n = (size_t)64*T*4*F*2;
    float *o = malloc(n*sizeof(float));
    cudaMemcpy(o, dy, n*sizeof(float), cudaMemcpyDeviceToHost);
    float *r = malloc(n*sizeof(float));
    f = fopen(argv[3], "rb"); fread(r, sizeof(float), n, f); fclose(f);
    double s=0, sr=0;
    for (size_t i=0;i<n;i++){ double dd=r[i]-o[i]; s+=dd*dd; sr+=(double)r[i]*r[i]; }
    printf("GPU decoder_pre(%s) rel_rms=%.3e  n=%zu\n", which, sqrt(s/(double)n)/sqrt(sr/(double)n), n);
    return 0;
}
