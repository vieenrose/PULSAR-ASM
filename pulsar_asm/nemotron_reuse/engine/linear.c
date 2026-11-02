// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1

void linear_fwd(const float *x, const float *w, const float *b,
                 float *y, int B, int In, int Out) {
    for (int i = 0; i < B; i++) {
        for (int o = 0; o < Out; o++) {
            float acc = b ? b[o] : 0.0f;
            for (int j = 0; j < In; j++) {
                acc += x[i*In + j] * w[o*In + j];
            }
            y[i*Out + o] = acc;
        }
    }
}
