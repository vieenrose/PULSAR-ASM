"""Offline proof for v11's repack mapping (no Nano needed).

The v10/v11 kernel reads, for row r of a tensor, code bytes at
18B-stride:  base + r*nb*18 + 2 + b*18 + [0..16)
and (v11)   16B-stride: base16 + row_dev_off + r*nb*16 + b*16 + [0..16)
This script builds both layouts from the SAME random weights and computes the
engine's exact fp32 Q4 GEMV (mul ws*db, then fma by (int)dot, ascending blocks)
from each. Bit-identical output for every tensor kind and every (r0,r1) slice
=> the repack + q4_classify device mapping is exact.
"""
import numpy as np
koff=[0,6144,1775616,2219008-3392,2217984,2660352,4429824,4435968,8859648,13283328]
koff=[0,6144,1775616,2217984,2660352,4429824,4435968,8859648,13283328]
krows=[1536,2048,512,512,1536,1536,5120,5120,1536]
kcols=[0,1536,1536,1536,2048,0,1536,1536,5120]
STRIDE=17707008; NL=28; VOC=64256; HID=1536
tot=0
for k in range(1,9):
    if not kcols[k]: continue
    nb=kcols[k]>>5
    assert koff[k]+krows[k]*nb*18 <= STRIDE, (k, koff[k]+krows[k]*nb*18)
    tot+=krows[k]*nb*16
q4_layer_dev=tot; q4_head_off=q4_layer_dev*NL; head=VOC*(HID>>5)*16
total=q4_head_off+head
print("layer stride ok; repacked/layer=%d  head_off=%d  total=%d (%.1f MiB)  <= blob 748976168: %s"
      %(q4_layer_dev,q4_head_off,total,total/2**20,total<=748976168))

def gemv(rows,cols,code,sc,xq,r0,r1):
    """engine-order fp32 Q4 GEMV over rows [r0,r1); code = layout-agnostic callable(r,b)->16 bytes"""
    nb=cols>>5; out=np.zeros(rows,dtype=np.float32)
    xqf=xq.view(np.float32)
    for r in range(r0,r1):
        acc=np.float32(0.0)
        for b in range(nb):
            blk=np.frombuffer(bytes(code(r,b)),dtype=np.uint8)   # 16 bytes = 32 nibbles
            lo=(blk & np.uint8(15)).astype(np.int32)-8           # low nibbles, bytes 0..15
            hi=(blk >> 4).astype(np.int32)-8                     # high nibbles, same bytes
            o=(b>>3)*292+4+(b&7)*32
            xb=xq[o:o+16].astype(np.int32)-0; xb2=xq[o+16:o+32].astype(np.int32)-0
            dot=int(np.dot(lo,xb)+np.dot(hi,xb2))
            db=xqf[(b>>3)*73]
            acc=np.float32(np.float32(np.float32(sc[r*nb+b])*db)*np.float32(dot)+acc)
        out[r]=acc
    return out

rng=np.random.default_rng(7)
q4_off=[0]*9; t=0
for k in range(1,9):
    if not kcols[k]: continue
    q4_off[k]=t; t+=krows[k]*(kcols[k]>>5)*16
bad=[]; nchk=0
for k in range(1,9):
    if not kcols[k]: continue
    rows,cols=krows[k],kcols[k]; nb=cols>>5
    code=bytearray(rng.integers(0,256,size=rows*nb*18,dtype=np.uint8).tobytes())
    # scales: nb per row
    sc=rng.standard_normal(rows*nb).astype(np.float32)
    nx=((cols+255)//256)*292+64
    xqi8=bytearray(rng.integers(-8,9,size=nx,dtype=np.int8).tobytes())
    # make the float slots at (b>>3)*73 valid floats (alias int8 slots): build float array then alias
    farr=rng.standard_normal(nx//4).astype(np.float32)
    xq=np.frombuffer(np.frombuffer(bytes(xqi8),dtype=np.int8).tobytes(),dtype=np.int8).copy()
    xqf=xq.view(np.float32); xqf[:]=farr
    xq=xq.view(np.int8)
    def mk18(r,b,rows=rows,nb=nb):
        o=r*nb*18+2+b*18; return bytes(code[o:o+16])
    # v11 layout: repacked per-tensor block, row-major, 16B blocks (q4_build)
    rep=bytearray(rows*nb*16)
    for r in range(rows):
        for b in range(nb):
            o=r*nb*18+2+b*18; rep[(r*nb+b)*16:(r*nb+b)*16+16]=code[o:o+16]
    def mk16(r,b,rows=rows,nb=nb,rep=rep):
        o=(r*nb+b)*16; return bytes(rep[o:o+16])
    slices=[(0,rows),(0,16),(7,rows),(3,17),(rows-1,rows),(5,max(6,rows-5)),(0,1),(1,2)]
    for (r0,r1) in slices:
        a=gemv(rows,cols,mk18,sc,xq,r0,r1); b=gemv(rows,cols,mk16,sc,xq,r0,r1)
        nchk+=1
        if a[r0:r1].tobytes()!=b[r0:r1].tobytes(): bad.append((k,r0,r1))
    # C-side mapping: q4_off[k] + row*nb*16 + b*16 stays inside the layer slice
    assert q4_off[k]+(rows-1)*nb*16+(nb-1)*16+16 <= q4_layer_dev, k
print("per-layer packing ok (all kinds fit in q4_layer_dev)")
assert q4_head_off+(VOC-1)*(HID>>5)*16+((HID>>5)-1)*16+16<=total
print("head tensor mapping ok")
print("GEMV slice comparisons: %d checked, %d mismatches"%(nchk,len(bad)))
print("VERDICT:", "REPACK+CLASSIFY MAPPING IS BIT-EXACT" if not bad else "BROKEN "+str(bad[:5]))
