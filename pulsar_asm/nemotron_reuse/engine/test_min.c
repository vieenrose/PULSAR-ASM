#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pulsar_asm/nemotron_reuse/engine/blob.h"
#include "pulsar_asm/nemotron_reuse/engine/wav_io.h"
#include "pulsar_asm/nemotron_reuse/engine/stft_dft.cuh"
#include "pulsar_asm/nemotron_reuse/engine/istft_dft.cuh"
#include "pulsar_asm/nemotron_reuse/engine/model.h"
#include <math.h>

static int make_even(int v){ v = (int)round(v); return v%2==0?v:v+1; }

int main(int argc, char **argv) {
    reuse_blob_t blob;
    if (reuse_blob_load(argv[1], &blob) != 0) { printf("bad blob\n"); return 1; }
    printf("ok\n");
    reuse_blob_free(&blob);
    return 0;
}
