#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/home/louis/PULSAR-ASM/pulsar_asm/lfm3')
from runtime.lfm3_model import LFM3
from runtime import pulsar_abi as abi
import json

abi.require_avx2_fma()
m = LFM3('weights.bin', max_seq=512, verbose=True)
print('SMOKE_OK')
m.close()
