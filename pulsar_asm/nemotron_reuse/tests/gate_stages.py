#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import os, sys, json, argparse
import torch
import numpy as np


def rel_rms(a, b, eps=1e-12):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    diff = (a - b).norm() ** 2
    ref = (b).norm() ** 2
    if ref < eps:
        return float(diff.sqrt())
    return float((diff / ref).sqrt())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', required=True)
    parser.add_argument('--cand', required=True)
    parser.add_argument('--thresh', type=float, default=1e-4)
    args = parser.parse_args()
    ref = torch.load(args.ref, map_location='cpu')
    cand = torch.load(args.cand, map_location='cpu')
    ok = True
    for k in ref:
        if k not in cand:
            print('MISSING', k)
            ok = False
            continue
        r = rel_rms(ref[k], cand[k])
        print(f'{k}: rel_rms={r:.3e}')
        if r > args.thresh:
            ok = False
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
