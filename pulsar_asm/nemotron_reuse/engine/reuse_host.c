// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
#include "blob.h"
#include "wav_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    reuse_blob_free(&blob);
    printf("stub: not yet implemented\n");
    return 0;
}
