#include "blob.h"
#include "reuse_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+1e-9*ts.tv_nsec; }
int main(int argc, char **argv) {
    int T = 256, F = 81, iters = argc > 1 ? atoi(argv[1]) : 20, blk = 0;
    size_t n = (size_t)64*T*F;
    reuse_blob_t b; if (reuse_blob_load("/tmp/pulsar-asm/pulsar_asm/nemotron_reuse/tests/golden/reuse.blob", &b)) return 1;
    reuse_gpu_t *g = reuse_gpu_create(&b); if (!g) return 1;
    float *dx, *dy; cudaMalloc(&dx, n*sizeof(float)); cudaMalloc(&dy, n*sizeof(float));
    cudaMemset(dx, 0, n*sizeof(float));
    if (!getenv("REUSE_NOSKIPS")) reuse_gpu_set_skips();
    for (int i = 0; i < iters; i++) {
        double t0 = now_s();
        if (reuse_gpu_tfmamba(g, blk, dx, dy, T, F)) { printf("fail: %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }
        cudaDeviceSynchronize();
        printf("iter %d: %.4f s\n", i, now_s()-t0); fflush(stdout);
    }
    if (!getenv("REUSE_NOSKIPS")) reuse_gpu_set_skips();
    reuse_gpu_print_ktimes();
    return 0;
}
