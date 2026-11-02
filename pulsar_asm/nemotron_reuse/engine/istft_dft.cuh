// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void reuse_istft(const float *mag, const float *pha, int F, int num_frames,
                 int n_fft, int hop_size, int win_size,
                 float *out, int out_len);

#ifdef __cplusplus
}
#endif
