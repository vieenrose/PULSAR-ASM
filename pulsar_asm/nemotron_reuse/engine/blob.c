// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
#include "blob.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int reuse_blob_load(const char *path, reuse_blob_t *blob) {
    memset(blob, 0, sizeof(*blob));
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char magic[8];
    if (fread(magic, 1, 8, f) != 8) { fclose(f); return -2; }
    if (memcmp(magic, "REUSE001", 8) != 0) { fclose(f); return -3; }
    uint64_t hlen = 0;
    if (fread(&hlen, 1, 8, f) != 8) { fclose(f); return -4; }
    blob->header_json = (char *)malloc(hlen + 1);
    if (fread(blob->header_json, 1, hlen, f) != hlen) { fclose(f); return -5; }
    blob->header_json[hlen] = 0;
    // count tensors by scanning? easier to read until eof by reading sequentially
    // but we don't know count; read all
    long start = ftell(f);
    fseek(f, 0, SEEK_END);
    long end = ftell(f);
    fseek(f, start, SEEK_SET);
    // naive: count by iterating
    int count = 0;
    long pos = start;
    while (pos < end) {
        uint32_t nlen = 0;
        if (fread(&nlen, 1, 4, f) != 4) break;
        fseek(f, nlen, SEEK_CUR);
        uint32_t ndim = 0;
        if (fread(&ndim, 1, 4, f) != 4) break;
        fseek(f, ndim * 4, SEEK_CUR);
        uint32_t dtype = 0;
        if (fread(&dtype, 1, 4, f) != 4) break;
        uint64_t dlen = 0;
        if (fread(&dlen, 1, 8, f) != 8) break;
        fseek(f, dlen, SEEK_CUR);
        count++;
        pos = ftell(f);
    }
    blob->ntensors = count;
    blob->tensors = (reuse_tensor_t *)calloc(count, sizeof(reuse_tensor_t));
    fseek(f, start, SEEK_SET);
    for (int i = 0; i < count; i++) {
        uint32_t nlen = 0; fread(&nlen, 1, 4, f);
        blob->tensors[i].name = (char *)malloc(nlen + 1);
        fread(blob->tensors[i].name, 1, nlen, f);
        blob->tensors[i].name[nlen] = 0;
        uint32_t ndim = 0; fread(&ndim, 1, 4, f);
        blob->tensors[i].ndim = ndim;
        blob->tensors[i].shape = (int *)malloc(ndim * sizeof(int));
        size_t nelem = 1;
        for (uint32_t d = 0; d < ndim; d++) {
            uint32_t s = 0; fread(&s, 1, 4, f);
            blob->tensors[i].shape[d] = s;
            nelem *= s;
        }
        blob->tensors[i].nelem = nelem;
        uint32_t dtype = 0; fread(&dtype, 1, 4, f);
        (void)dtype;
        uint64_t dlen = 0; fread(&dlen, 1, 8, f);
        blob->tensors[i].data = (float *)malloc(dlen);
        fread(blob->tensors[i].data, 1, dlen, f);
    }
    fclose(f);
    return 0;
}

void reuse_blob_free(reuse_blob_t *blob) {
    if (!blob) return;
    free(blob->header_json);
    for (int i = 0; i < blob->ntensors; i++) {
        free(blob->tensors[i].name);
        free(blob->tensors[i].shape);
        free(blob->tensors[i].data);
    }
    free(blob->tensors);
    memset(blob, 0, sizeof(*blob));
}

const reuse_tensor_t *reuse_blob_find(const reuse_blob_t *blob, const char *name) {
    for (int i = 0; i < blob->ntensors; i++) {
        if (strcmp(blob->tensors[i].name, name) == 0) return &blob->tensors[i];
    }
    return NULL;
}
