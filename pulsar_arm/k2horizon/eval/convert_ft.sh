#!/bin/bash
# Convert the K2 meeting fine-tune (k2-qat4 safetensors, F32, vocab 69312)
# into native blobs and validate numerically.
set -e
cd ~/k2
SF=finetune/safetensors/k2-qat4
[ -f "$SF/model.safetensors" ] || { echo "no safetensors"; exit 1; }

# 1) Q4 blob (asm engine path) with the 1536-position rope tables
python3 k2_convert.py "$SF" k2h_ft_q4.blob k2rope1536.npz --q4
ls -la k2h_ft_q4.blob

# 2) fp16 blob (C engine reference path, gemv_ref)
python3 k2_convert.py "$SF" k2h_ft_fp16.blob k2rope1536.npz
ls -la k2h_ft_fp16.blob

# 3) vocab check from header
python3 -c "
import struct
h = struct.unpack('<4s9I', open('k2h_ft_q4.blob','rb').read(40))
print('nl,hid,inter,nhead,nkv,hdim,vocab,wtype,npos =', h[1:])
assert h[7] == 2 and h[8] == 1536 and h[6] == 69312, 'header mismatch'
print('header OK')
"
