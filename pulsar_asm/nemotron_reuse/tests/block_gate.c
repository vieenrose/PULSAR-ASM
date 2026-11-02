#include "blob.h"
#include "tfmamba.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    if (argc < 4) { printf("usage: %s <blob> <in.raw> <out.raw> [T F]\n", argv[0]); return 1; }
    reuse_blob_t blob;
    if (reuse_blob_load(argv[1], &blob)) { printf("blob fail\n"); return 1; }
    int T = argc > 4 ? atoi(argv[4]) : 256;
    int F = argc > 5 ? atoi(argv[5]) : 81;
    size_t n = (size_t)64*T*F;
    float *x = (float*)malloc(n*sizeof(float));
    float *o = (float*)malloc(n*sizeof(float));
    FILE *f = fopen(argv[2], "rb");
    if (fread(x, sizeof(float), n, f) != n) { printf("in short\n"); return 1; }
    fclose(f);
    reuse_tfmamba_block(&blob, 0, x, o, 1, T, F);
    f = fopen(argv[3], "wb"); fwrite(o, sizeof(float), n, f); fclose(f);
    printf("block done T=%d F=%d\n", T, F);
    free(x); free(o); reuse_blob_free(&blob);
    return 0;
}
