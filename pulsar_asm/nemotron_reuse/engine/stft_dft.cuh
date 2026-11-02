// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void reuse_stft_mag_pha(const float *x, int T, int n_fft, int hop_size, int win_size,
                        float *mag_out, float *pha_out, int F, int num_frames);

#ifdef __cplusplus
}
#endif
