// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
#pragma once

// Fused selective scan for Mamba
// h in registers, parallel over batch*seq or per direction
// supports time and freq directions via stride
