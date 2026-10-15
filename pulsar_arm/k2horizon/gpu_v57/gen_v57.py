#!/usr/bin/env python3
"""K2 Q4 GEMV v57 (Exp62): 32-BYTE-PER-ROW-LOAD kernel.

Motivation (probe91b, run #80): at engine item geometry, 16B/lane vector loads run
2.7x faster than 4B/lane scalar loads (10.66 vs 3.96 GB/s) with bit-identical output.
v52 loads one 4B word per lane per row per iteration (10 loads/iteration/lane).
v57 loads 32 contiguous bytes per lane per row (2 x ld.global.v4.f32 = 8 words = 8 sv),
so each lane covers ONE (row q, v52-iteration i') fully: q = k mod 4, i' = k div 4 + 8i.

Bit-exactness: the int dots are exact int32 (order-free). The float chain for row q is
fma(scprod_sv, dot_sv, f) for sv=0..7 within an iteration, iterations ascending. Each lane
computes the full 8-FMA chain for one (q, i') locally (level 1), then a cross-lane sequential
chain transfer (level 2) applies chain_{i'=0} (lane 4q), chain_{1} (lane 4q+4), ... in i'
order through shbuf, so lane 4q ends with the exact v52 chain value.
"""
import sys

P = []
A = P.append
A("// K2 Q4 GEMV v57 (Exp62: 32B/lane v4 loads + cross-lane chain transfer)")
A(".version 4.0")
A(".target sm_50")
A(".address_size 64")
A(".visible .entry q4_gemv("
  ".param .u64 pw, .param .u64 psc, .param .u64 pxq, .param .u64 py, "
  ".param .u32 pr0, .param .u32 pr1, .param .u32 pnb, .param .u32 prf) {")
A(".reg .u64 %rd<16>; .reg .b32 %r<80>; .reg .f32 %f<80>; .reg .pred %p<16>;")
A(".shared .align 4 .b32 shbuf[4096];")

# ---- prologue -------------------------------------------------------------
A("ld.param.u64 %rd1,[pw]; ld.param.u64 %rd2,[psc]; ld.param.u64 %rd3,[pxq]; ld.param.u64 %rd4,[py];")
A("ld.param.u32 %r10,[pr0]; ld.param.u32 %r11,[pr1]; ld.param.u32 %r12,[pnb];")
A("mov.u32 %r1,%tid.x; mov.u32 %r2,%tid.y; mov.u32 %r3,%ctaid.x;")
# base row of this warp: ((ctaid*8)+ty)*4 + r0   (identical to v52)
A("shl.b32 %r3,%r3,3; add.u32 %r3,%r3,%r2; shl.b32 %r2,%r3,2; add.u32 %r2,%r2,%r10;")
A("setp.ge.u32 %p1,%r2,%r11; @%p1 bra $done;")
# row for this lane: row = base + (k mod 4); clamp to base when >= r1 (no OOB reads)
A("shr.u32 %r33,%r1,1; and.b32 %r34,%r33,3;")           # q = (k>>1)&3
A("and.b32 %r35,%r1,1;")                                  # b = k&1 (sv half)
A("shr.u32 %r36,%r1,3;")                                  # j = k>>3
A("add.u32 %r37,%r2,%r34;")                              # row = base + q
A("setp.ge.u32 %p2,%r37,%r11;")                          # row invalid (mask products)
A("selp.u32 %r37,%r2,%r37,%p2;")                         # clamp address to base
# weight base: weights + row*nb32*16 + 32*k
A("mul.lo.u32 %r4,%r37,%r12; shl.b32 %r4,%r4,4;")
A("shl.b32 %r5,%r1,5; add.u32 %r4,%r4,%r5;")
A("cvt.u64.u32 %rd5,%r4; add.u64 %rd5,%rd5,%rd1;")
# xq base: pxq + 32k + 4 + 32*(k&1) + 292*(k>>3)
A("shl.b32 %r6,%r1,5; add.u32 %r6,%r6,4;")
A("shl.b32 %r7,%r35,5; add.u32 %r6,%r6,%r7;")
A("mul.lo.u32 %r7,%r36,292; add.u32 %r6,%r6,%r7;")
A("cvt.u64.u32 %rd6,%r6; add.u64 %rd6,%rd6,%rd3;")
# scale base: psc + row*nb32*4 + 4k   ;  db base: pxq + 292*(k>>3)
A("mul.lo.u32 %r4,%r37,%r12; shl.b32 %r4,%r4,2; add.u32 %r4,%r4,%r1; shl.b32 %r4,%r4,2;")
A("cvt.u64.u32 %rd7,%r4; add.u64 %rd7,%rd7,%rd2;")
A("mov.u64 %rd8,%rd3;")
A("mul.lo.u32 %r4,%r36,292; cvt.u64.u32 %rd9,%r4; add.u64 %rd8,%rd8,%rd9;")
# loop bound: iters_v4 = ceil(ceil(nb32/8)/4) = (nb32+31)>>5
A("add.u32 %r13,%r12,31; shr.u32 %r13,%r13,5;")
# shared-memory base for this warp + final-accumulator area
A("mov.b32 %r61,shbuf; mov.u32 %r62,%tid.y; shl.b32 %r62,%r62,10; add.u32 %r65,%r61,%r62;")
# exchange offsets: 256*j + 32*(k&1) + 8*((k>>1)&3)
A("shl.b32 %r63,%r36,8; shl.b32 %r64,%r35,5; add.u32 %r63,%r63,%r64;")
A("shl.b32 %r64,%r34,3; add.u32 %r63,%r63,%r64;")
A("add.u32 %r63,%r63,%r65;")
# loop counter r66 = 0; block index r67 = k
A("mov.u32 %r66,0; mov.u32 %r67,%r1;")
A("mov.f32 %f1,0F00000000;")
A("mov.f32 %f78,0F00000000;")
A("setp.eq.u32 %p8,%r1,0;")
A("setp.eq.u32 %p9,%r1,4;")
A("setp.eq.u32 %p10,%r1,8;")
A("setp.eq.u32 %p11,%r1,12;")
A("setp.eq.u32 %p12,%r1,16;")
A("setp.eq.u32 %p13,%r1,20;")
A("setp.eq.u32 %p14,%r1,24;")
A("setp.eq.u32 %p15,%r1,28;")
A("$LIT:")

# ---- loads ----------------------------------------------------------------
# 8 weight v4 loads: 2 per row (rows 0..3 at rd5 + 0/16, +64/80, +128/144, +192/208)
w = 2  # f2..
for row in range(4):
    base = 64*row
    for k_ in range(2):
        off = base + 16*k_
        A(f"ld.global.v4.f32 {{%f{w},%f{w+1},%f{w+2},%f{w+3}}},[%rd5+{off}];")
        w += 4
# xq: 5 v4 loads (20 regs) at byte 32k+4+32(k&1)+2192i -> registers f34..f53
xq = 34
for t in range(5):
    A(f"ld.global.v4.f32 {{%f{xq},%f{xq+1},%f{xq+2},%f{xq+3}}},[%rd6+{16*t}];")
    xq += 4
# scales: one row scale + db (single f32 loads, identical values to v52)
A("ld.global.f32 %f70,[%rd7+0];")
A("ld.global.f32 %f71,[%rd8+0];")
# word-valid mask: block index b = k + 32*i  >= nb32
A("setp.ge.u32 %p3,%r67,%r12;")
A("mul.lo.u32 %r40,%r66,32; add.u32 %r40,%r40,%r1;")   # b = 32*i + k
A("setp.ge.u32 %p3,%r40,%r12;")

# ---- unpack: 8 words (m=0..7) x 4 rows = 32 dots --------------------------
# word m: weight reg index (row r): wreg = 2 + 8*r + 4*(m div 4) + (m mod 4)
# xq X_m = A_m (registers xq + 4*t + 2*(m mod 2) ...):
#   X_m = int32 slot (8m+4)//4 = 2m+1 relative to the lane's first xq int32 (f34 = byte 0..3)
#   register index = (2m+1) // 4 ... each v4 load gives 4 consecutive int32.
#   int32 slot s (0..19) -> f register xq + (s div 4)*4 + (s mod 4)
def xqreg(slot):
    return xq + (slot // 4) * 4 + (slot % 4)

n = 4  # int dot regs f74.. not needed; use int regs r40..: dots in r40..r71
dot = 40
for r in range(4):          # row index 0..3
    for m in range(8):      # word within the lane's 32B block
        wreg = 2 + 8*r + 4*(m // 4) + (m % 4)
        xs = 2*m + 1          # int32 slot of X (byte 8m+4 -> word 2m+1)
        ys = 2*m + 5          # Y = X + 4 words
        X, Y = xqreg(xs), xqreg(ys)
        # sign-bias trick identical to v52: xor 0x88888888 makes bfe.s32(width4) = (n-8)
        A(f"xor.b32 %r38,%f{wreg},0x88888888;")
        first = True
        for rr in range(4):  # 4 int8 bytes of X / Y, paired with nibbles 2rr, 2rr+1
            pos = 8*rr
            A(f"bfe.s32 %r41,%f{X},{pos},8;")
            A(f"bfe.s32 %r42,%f{Y},{pos},8;")
            A(f"bfe.s32 %r43,%r38,{pos},4;")
            A(f"bfe.s32 %r44,%r38,{pos+4},4;")
            A(f"mad.lo.s32 %r{dot},%r43,%r41,0;")
            A(f"mad.lo.s32 %r{dot},%r44,%r42,%r{dot};")
        # cvt + scale product + mask (mask = word-invalid OR row-invalid)
        f_cvt = 40 + 8*r + m
        A(f"cvt.rn.f32.s32 %f{f_cvt},%r{dot};")
        dot += 1
# scale*db once (same value for all 32 dots: one 16B block = one scale)
A("mul.f32 %f72,%f70,%f71;")
# 32 selp + 32 stores of the dot, 8 stores of the (identical per row) scale product
dot = 40
for r in range(4):
    A(f"or.pred %p4,%p3,%p2;")
    for m in range(8):
        f_cvt = 40 + 8*r + m
        A(f"@%p4 selp.f32 %f{f_cvt},0F00000000,%f{f_cvt},%p4;")
    # 8 dots at offsets 256*j + 32*(k&1) + 8*q + 4m -> contiguous 32B = 2 v4 stores
    for t in range(2):
        off = 4 + 16*t
        A(f"st.shared.v4.f32 [%r63+{off}],{{%f{40+8*r+4*t},%f{40+8*r+4*t+1},%f{40+8*r+4*t+2},%f{40+8*r+4*t+3}}};")
    # scale product for row r: same value for all 8 sv -> 2 stores (offsets 0 and 4)
    A(f"@%p4 selp.f32 %f73,0F00000000,%f72,%p4;")
    A(f"st.shared.v2.f32 [%r63],{{%f73,%f73}};")
    dot += 8
# ---- level-1 chain: 8 FMAs (sv = 0..7) on the lane's (q, i') chain ---------
# f1 = fma(scprod, dot, f1); scprod = the stored selp'd scale product = %f73 (last row) --
# must be per row: redo per row in the transfer below.
for r in range(4):
    A(f"@%p{8+r} st.shared.f32 [%r65+{64+r*4}],%f73;") if r == 0 else None
A("// level 2 chain transfer: lane 4q+j applies its 8-FMA chain in order")
# chain transfer: lane 4q reads accumulator, applies 8 FMAs, writes back; j = 0..7
# (rows: lane 4q+j has row q = j mod 4 -> for a fixed owner lane 4q the chain order is
#  j = 0..7 = lanes 4q, 4q+4, ..., 4q+28)
# registers: %f74 = the scprod value for this lane's row (reload from the staging we wrote:
# the lane's own scprod for row r is at shbuf offset 40+... -> we stored %f73 (row 3) only.
