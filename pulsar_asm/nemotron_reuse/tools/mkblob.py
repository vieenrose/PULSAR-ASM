#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import os
import json
import struct
import argparse
import torch
from safetensors import safe_open
from pathlib import Path


MAGIC = b'REUSE001'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--safetensors', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    cfg = json.load(open(args.config))
    tensors = {}
    with safe_open(args.safetensors, framework='pt') as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k).cpu()

    with open(args.out, 'wb') as f:
        f.write(MAGIC)
        header = {'config': cfg, 'tensors': list(tensors.keys()), 'dtype': 'f32'}
        h = json.dumps(header).encode('utf-8')
        f.write(struct.pack('<Q', len(h)))
        f.write(h)
        for k in tensors:
            t = tensors[k]
            data = t.numpy().tobytes()
            f.write(struct.pack('<I', len(k)))
            f.write(k.encode('utf-8'))
            f.write(struct.pack('<I', t.ndim))
            for s in t.shape:
                f.write(struct.pack('<I', s))
            f.write(struct.pack('<I', 0))  # f32
            f.write(struct.pack('<Q', len(data)))
            f.write(data)

    print('wrote', args.out, len(tensors), 'tensors')


if __name__ == '__main__':
    main()
