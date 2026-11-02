#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import os, sys, argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_mag', required=True)
    parser.add_argument('--ref_pha', required=True)
    parser.add_argument('--cand_mag', required=True)
    parser.add_argument('--cand_pha', required=True)
    parser.add_argument('--thresh', type=float, default=1e-3)
    args = parser.parse_args()
    rm = torch.load(args.ref_mag).float().reshape(-1)
    rp = torch.load(args.ref_pha).float().reshape(-1)
    cm = torch.load(args.cand_mag).float().reshape(-1)
    cp = torch.load(args.cand_pha).float().reshape(-1)
    n = min(rm.numel(), cm.numel())
    rc = rm[:n] * torch.cos(rp[:n])
    rs = rm[:n] * torch.sin(rp[:n])
    cc = cm[:n] * torch.cos(cp[:n])
    cs = cm[:n] * torch.sin(cp[:n])
    err = ((rc - cc)**2 + (rs - cs)**2).sqrt().mean().item()
    print(f'complex_err={err:.3e}')
    sys.exit(0 if err < args.thresh else 1)


if __name__ == '__main__':
    main()
