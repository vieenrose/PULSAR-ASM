// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
#include "model.h"
#include <string.h>

void reuse_forward(const reuse_blob_t *blob, const float *mag_in, const float *pha_in,
                   int B, int F_in, int T_in,
                   float *out_mag, float *out_pha) {
    (void)blob;
    memcpy(out_mag, mag_in, (size_t)B*(size_t)F_in*(size_t)T_in*sizeof(float));
    memcpy(out_pha, pha_in, (size_t)B*(size_t)F_in*(size_t)T_in*sizeof(float));
}
