// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
#include "blob.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int reuse_blob_load(const char *path, reuse_blob_t *blob) {
    memset(blob, 0, sizeof(*blob));
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char magic[8];
    if (fread(magic, 1, 8, f) != 8) { fclose(f); return -2; }
    if (memcmp(magic, "REUSE001", 8) != 0) { fclose(f); return -3; }
    uint64_t hlen = 0;
    if (fread(&hlen, 1, 8, f) != 8) { fclose(f); return -4; }
    blob->header_json = (char *)malloc((size_t)hlen + 1);
    if (fread(blob->header_json, 1, (size_t)hlen, f) != (size_t)hlen) { fclose(f); free(blob->header_json); blob->header_json=NULL; return -5; }
    blob->header_json[hlen] = 0;
    long start = ftell(f);
    fseek(f, 0, SEEK_END);
    long end = ftell(f);
    fseek(f, start, SEEK_SET);
    int count = 0;
    long pos = start;
    while (pos < end) {
        uint32_t nlen = 0;
        if (fread(&nlen, 1, 4, f) != 4) break;
        if (fseek(f, (long)nlen, SEEK_CUR)) break;
        uint32_t ndim = 0;
        if (fread(&ndim, 1, 4, f) != 4) break;
        if (fseek(f, (long)ndim * 4, SEEK_CUR)) break;
        uint32_t dtype = 0;
        if (fread(&dtype, 1, 4, f) != 4) break;
        uint64_t dlen = 0;
        if (fread(&dlen, 1, 8, f) != 8) break;
        if (fseek(f, (long)dlen, SEEK_CUR)) break;
        count++;
        pos = ftell(f);
    }
    blob->ntensors = count;
    blob->tensors = (reuse_tensor_t *)calloc((size_t)count, sizeof(reuse_tensor_t));
    fseek(f, start, SEEK_SET);
    for (int i = 0; i < count; i++) {
        uint32_t nlen = 0;
        if (fread(&nlen, 1, 4, f) != 4) break;
        blob->tensors[i].name = (char *)calloc((size_t)nlen + 1, 1);
        if (fread(blob->tensors[i].name, 1, (size_t)nlen, f) != (size_t)nlen) {}
        uint32_t ndim = 0;
        if (fread(&ndim, 1, 4, f) != 4) break;
        blob->tensors[i].ndim = (int)ndim;
        blob->tensors[i].shape = (int *)malloc((size_t)ndim * sizeof(int));
        size_t nelem = 1;
        for (uint32_t d = 0; d < ndim; d++) {
            uint32_t s = 1;
            if (fread(&s, 1, 4, f) != 4) s = 1;
            blob->tensors[i].shape[d] = (int)s;
            nelem *= (size_t)s;
        }
        blob->tensors[i].nelem = nelem;
        uint32_t dtype = 0;
        if (fread(&dtype, 1, 4, f) != 4) break;
        (void)dtype;
        uint64_t dlen = 0;
        if (fread(&dlen, 1, 8, f) != 8) break;
        blob->tensors[i].data = (float *)malloc((size_t)dlen);
        if (fread(blob->tensors[i].data, 1, (size_t)dlen, f) != (size_t)dlen) {
            memset(blob->tensors[i].data, 0, (size_t)dlen);
        }
    }
    fclose(f);
    return 0;
}

void reuse_blob_free(reuse_blob_t *blob) {
    if (!blob) return;
    if (blob->header_json) { free(blob->header_json); blob->header_json = NULL; }
    if (blob->tensors) {
        for (int i = 0; i < blob->ntensors; i++) {
            if (blob->tensors[i].name) { free(blob->tensors[i].name); blob->tensors[i].name = NULL; }
            if (blob->tensors[i].shape) { free(blob->tensors[i].shape); blob->tensors[i].shape = NULL; }
            if (blob->tensors[i].data) { free(blob->tensors[i].data); blob->tensors[i].data = NULL; }
        }
        free(blob->tensors); blob->tensors = NULL;
    }
    blob->ntensors = 0;
}

const reuse_tensor_t *reuse_blob_find(const reuse_blob_t *blob, const char *name) {
    if (!blob || !name || !blob->tensors) return NULL;
    for (int i = 0; i < blob->ntensors; i++) {
        if (blob->tensors[i].name && strcmp(blob->tensors[i].name, name) == 0) return &blob->tensors[i];
    }
    return NULL;
}
