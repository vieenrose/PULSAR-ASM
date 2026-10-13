import numpy as np
rng = np.random.default_rng(7)
koff  = [0,6144,1775616,2217984,2660352,4429824,4435968,8859648,13283328]
krows = [1536,2048,512,512,1536,1536,5120,5120,1536]
kcols = [0,1536,1536,1536,2048,0,1536,1536,5120]
VOC, HID = 64256, 1536
def repack_v10(tbase, rows, nb):  # PROVEN formula (v10, gate-green on device)
    out = np.zeros(rows*nb*16, np.uint8)
    for r in range(rows):
        for b in range(nb):
            out[(r*nb+b)*16:(r*nb+b)*16+16] = tbase[r*nb*18+2+b*18:r*nb*18+2+b*18+16]
    return out
def repack_v11c(tbase, rows, nb):  # literal transcription of v11c C loop
    out = np.zeros(rows*nb*16, np.uint8)
    for r in range(rows):
        for b in range(nb):
            out[(r*nb+b)*16:(r*nb+b)*16+16] = tbase[r*nb*18+2+b*18:r*nb*18+2+b*18+16]
    return out
bad = 0
for k in range(1,9):
    if not kcols[k]: continue
    nb = kcols[k]>>5; rows = krows[k]
    t = rng.integers(0,256,rows*nb*18,dtype=np.uint8)
    if not np.array_equal(repack_v10(t,rows,nb), repack_v11c(t,rows,nb)):
        print("MISMATCH kind",k); bad += 1
nb = HID>>5
t = rng.integers(0,256,VOC*nb*18,dtype=np.uint8)
if not np.array_equal(repack_v10(t,VOC,nb), repack_v11c(t,VOC,nb)): print("MISMATCH head"); bad += 1
print("kinds+head checked, mismatches:",bad)
print("VERDICT:", "V11C PACK == V10 PROVEN PACK" if bad==0 else "FAIL")
