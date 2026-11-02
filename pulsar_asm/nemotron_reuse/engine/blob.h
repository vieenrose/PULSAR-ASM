// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct {
    char *name;
    int ndim;
    int *shape;
    size_t nelem;
    float *data; // host copy for now
} reuse_tensor_t;

typedef struct {
    char *header_json;
    int ntensors;
    reuse_tensor_t *tensors;
} reuse_blob_t;

int reuse_blob_load(const char *path, reuse_blob_t *blob);
void reuse_blob_free(reuse_blob_t *blob);
const reuse_tensor_t *reuse_blob_find(const reuse_blob_t *blob, const char *name);
