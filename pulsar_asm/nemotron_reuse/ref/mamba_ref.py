# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MambaRef(nn.Module):
    def __init__(self, d_model, cfg):
        super().__init__()
        d_state = cfg['model_cfg']['d_state']
        d_conv = cfg['model_cfg']['d_conv']
        expand = cfg['model_cfg']['expand']
        d_inner = int(expand * d_model)
        dt_rank = d_model // 16
        if dt_rank < 1:
            dt_rank = 1
        self.d_inner = d_inner
        self.dt_rank = dt_rank
        self.d_state = d_state
        self.d_conv = d_conv

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[..., :L]
        x = x.transpose(1, 2)
        x = F.silu(x)

        x_dbl = self.x_proj(x)
        dt, Bm, Cm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)
        dt = F.softplus(dt)

        A = -torch.exp(self.A_log.float())
        y = self.selective_scan(x, dt, A, Bm, Cm, self.D.float())
        y = y * F.silu(z)
        out = self.out_proj(y)
        return out

    def selective_scan(self, u, delta, A, B, C, D):
        Bsz, L, D_in = u.shape
        N = A.shape[-1]
        h = torch.zeros((Bsz, D_in, N), device=u.device, dtype=u.dtype)
        ys = []
        for i in range(L):
            delta_t = delta[:, i, :]
            u_t = u[:, i, :]
            B_t = B[:, i, :]
            C_t = C[:, i, :]
            deltaA = torch.exp(delta_t.unsqueeze(-1) * A)
            deltaB_u = (delta_t.unsqueeze(-1) * B_t.unsqueeze(1)) * u_t.unsqueeze(-1)
            h = deltaA * h + deltaB_u
            y_t = torch.einsum('b d n, b n -> b d', h, C_t)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        y = y + u * D
        return y
