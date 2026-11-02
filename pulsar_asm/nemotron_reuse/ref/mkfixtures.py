#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import os
import json
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--rates', default='8000,16000,44100,48000')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rates = [int(x) for x in args.rates.split(',')]
    src = args.src
    for sr in rates:
        out = os.path.join(args.outdir, f'mic_test2_{sr}.wav')
        if os.path.exists(out):
            continue
        cmd = ['ffmpeg', '-y', '-i', src, '-ar', str(sr), '-ac', '1', out]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(os.path.join(args.outdir, 'fixtures.json'), 'w') as f:
        json.dump({'src': src, 'rates': rates}, f, indent=2)


if __name__ == '__main__':
    main()
