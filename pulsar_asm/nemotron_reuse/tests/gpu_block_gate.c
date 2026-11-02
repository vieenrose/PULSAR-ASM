#include "blob.h"
#include "reuse_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
int main(int argc, char **argv) {
    if (argc < 4) { printf("usage: %s <blob> <in.raw> <ref.raw> [T F] [block]\n", argv[0]); return 1; }
    int T = argc > 4 ? atoi(argv[4]) : 256, F = argc > 5 ? atoi(argv[5]) : 81;
    int blk = argc > 6 ? atoi(argv[6]) : 0;
    size_t n = (size_t)64*T*F;
    reuse_blob_t b; if (reuse_blob_load(argv[1], &b)) return 1;
    float *h = malloc(n*sizeof(float));
    FILE *f = fopen(argv[2], "rb"); if (fread(h, sizeof(float), n, f) != n) { printf("short\n"); return 1; } fclose(f);
    reuse_gpu_t *g = reuse_gpu_create(&b); if (!g) return 1;
    float *dx, *dy; cudaMalloc(&dx, n*sizeof(float)); cudaMalloc(&dy, n*sizeof(float));
    cudaMemcpy(dx, h, n*sizeof(float), cudaMemcpyHostToDevice);
    if (reuse_gpu_tfmamba(g, blk, dx, dy, T, F)) { printf("tfmamba fail\n"); return 1; }
    cudaDeviceSynchronize();
    float *o = malloc(n*sizeof(float));
    cudaMemcpy(o, dy, n*sizeof(float), cudaMemcpyDeviceToHost);
    float *r = malloc(n*sizeof(float));
    f = fopen(argv[3], "rb"); fread(r, sizeof(float), n, f); fclose(f);
    double s=0, sr=0;
    for (size_t i=0;i<n;i++){ double d=r[i]-o[i]; s+=d*d; sr+=(double)r[i]*r[i]; }
    printf("GPU block %d rel_rms=%.3e\n", blk, sqrt(s/(double)n)/sqrt(sr/(double)n));
    const char *dw = getenv("GPU_DUMP");
    if (dw) { f = fopen(dw, "wb"); fwrite(o, sizeof(float), n, f); fclose(f); }
    return 0;
}
