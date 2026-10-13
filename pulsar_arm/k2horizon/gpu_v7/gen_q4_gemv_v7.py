#!/usr/bin/env python3
"""Generate the K2 Q4_0 warp-per-row GEMV PTX (v7h) + optional C embed."""
import sys

P = []
A = P.append
A("// K2 Q4 GEMV v7h: warp/row, 8 blocks/iter, 4 bytes/lane, pure pointer walks")
A(".version 4.0")
A(".target sm_50")
A(".address_size 64")
A(".visible .entry q4_gemv("
  ".param .u64 pw, .param .u64 psc, .param .u64 pxq, .param .u64 py, "
  ".param .u32 pr0, .param .u32 pr1, .param .u32 pnb, .param .u32 prf) {")
A(".reg .u64 %rd<24>; .reg .u32 %r<80>; .reg .f32 %f<20>; .reg .pred %p<8>;")
A(".shared .align 4 .b32 shbuf[4096];")
A("ld.param.u64 %rd1,[pw]; ld.param.u64 %rd2,[psc]; ld.param.u64 %rd3,[pxq]; ld.param.u64 %rd4,[py];")
A("ld.param.u32 %r10,[pr0]; ld.param.u32 %r11,[pr1]; ld.param.u32 %r12,[pnb];")
A("mov.u32 %r1,%tid.x; mov.u32 %r2,%tid.y; mov.u32 %r3,%ctaid.x;")
# row = ctaid*8 + tid.y + row0 ; exit if >= row1
A("shl.b32 %r3,%r3,3; add.u32 %r2,%r3,%r2; add.u32 %r2,%r2,%r10;")
A("setp.ge.u32 %p1,%r2,%r11; @%p1 bra $done;")
# lane constants: s = t>>2 (block within iter), q = t&3 (byte within block half)
A("shr.u32 %r33,%r1,2;")
A("and.b32 %r34,%r1,3;")
# code base: w + row*nb32*18 + 2 + 18s + q
A("mul.lo.u32 %r4,%r2,%r12; mul.lo.u32 %r4,%r4,18; add.u32 %r4,%r4,2;")
A("mul.lo.u32 %r5,%r33,18; add.u32 %r4,%r4,%r5; add.u32 %r4,%r4,%r34;")
A("cvt.u64.u32 %rd5,%r4; add.u64 %rd5,%rd5,%rd1;")
# x8 base: xq + 32s + q + 4   (int16 = +16)
A("shl.b32 %r5,%r33,5; add.u32 %r5,%r5,%r34; add.u32 %r5,%r5,4;")
A("cvt.u64.u32 %rd17,%r5; add.u64 %rd17,%rd17,%rd3;")
# sc base: sc + row*nb32*4 + 4s
A("mul.lo.u32 %r4,%r2,%r12; shl.b32 %r4,%r4,2; cvt.u64.u32 %rd18,%r4; add.u64 %rd18,%rd18,%rd2;")
A("shl.b32 %r5,%r33,2; cvt.u64.u32 %rd19,%r5; add.u64 %rd18,%rd18,%rd19;")
# db base: xq (scale float at (b>>3)*292 ; b>>3 = it)
A("mov.u64 %rd20,%rd3;")
# iterations = ceil(nb32/8)
A("add.u32 %r39,%r12,7; shr.u32 %r39,%r39,3;")
A("setp.eq.u32 %p5,%r1,0;")
# shared layout: row stride 512B: f7 slots 0..127, f6 slots 256..383 (byte offsets)
A("mov.b32 %r61,shbuf; mov.u32 %r63,%tid.y; shl.b32 %r63,%r63,9; add.u32 %r65,%r61,%r63;")
A("mul.lo.u32 %r60,%r1,4; add.u32 %r62,%r65,%r60;")
A("mul.lo.u32 %r66,%r33,16; add.u32 %r67,%r65,%r66;")          # lane0: f7 slot byte base (16s)
A("add.u32 %r69,%r65,256; add.u32 %r69,%r69,%r66;")            # lane0: f6 slot byte base (256+16s)
A("mov.f32 %f1,0F00000000; mov.u32 %r13,0;")
A("$LIT:")
# b = 8*it + s ; invalid if b >= nb32
A("shl.b32 %r14,%r13,3; add.u32 %r14,%r14,%r33; setp.ge.u32 %p2,%r14,%r12;")
# 4 code bytes (j = q, q+4, q+8, q+12)
A("ld.global.u8 %r20,[%rd5];")
A("ld.global.u8 %r21,[%rd5+4];")
A("ld.global.u8 %r22,[%rd5+8];")
A("ld.global.u8 %r23,[%rd5+12];")
# 4 x8 bytes + 4 x16 bytes
A("ld.global.s8 %r24,[%rd17]; ld.global.s8 %r25,[%rd17+16];")
A("ld.global.s8 %r26,[%rd17+4]; ld.global.s8 %r27,[%rd17+20];")
A("ld.global.s8 %r28,[%rd17+8]; ld.global.s8 %r29,[%rd17+24];")
A("ld.global.s8 %r30,[%rd17+12]; ld.global.s8 %r31,[%rd17+28];")
A("ld.global.f32 %f2,[%rd18]; ld.global.f32 %f3,[%rd20];")
for i, (c, x8, x16) in enumerate([(20, 24, 25), (21, 26, 27), (22, 28, 29), (23, 30, 31)]):
    A(f"and.b32 %r40,%r{c},15; sub.s32 %r40,%r40,8;")
    A(f"shr.u32 %r41,%r{c},4; and.b32 %r41,%r41,15; sub.s32 %r41,%r41,8;")
    A(f"mul.lo.s32 %r42,%r40,%r{x8}; mul.lo.s32 %r43,%r41,%r{x16};")
    if i == 0:
        A("add.s32 %r46,%r42,%r43;")
    else:
        A(f"add.s32 %r44,%r42,%r43; add.s32 %r46,%r46,%r44;")
A("cvt.rn.f32.s32 %f6,%r46; mul.f32 %f7,%f2,%f3; selp.f32 %f7,0F00000000,%f7,%p2;")
A("st.shared.f32 [%r62],%f7; st.shared.f32 [%r62+256],%f6;")
A("bar.sync 0;")
# full-row chain on ALL lanes (unpredicated; lane-independent base %r65).
# every lane builds an identical %f1; lane 0 stores it. f8_s int-exact per block, fma ascending.
for sv in range(8):
    base = 256 + 16 * sv
    A(f"ld.shared.f32 %f10,[%r65+{base+0}];")
    A(f"ld.shared.f32 %f11,[%r65+{base+4}];")
    A(f"ld.shared.f32 %f12,[%r65+{base+8}];")
    A(f"ld.shared.f32 %f13,[%r65+{base+12}];")
    A("add.f32 %f8,%f10,%f11;")
    A("add.f32 %f8,%f8,%f12;")
    A("add.f32 %f8,%f8,%f13;")
    A(f"ld.shared.f32 %f14,[%r65+{16*sv}];")
    A("fma.rn.f32 %f1,%f14,%f8,%f1;")
A("bar.sync 0;")
A("add.u32 %r13,%r13,1;")
A("add.u64 %rd5,%rd5,144; add.u64 %rd17,%rd17,292; add.u64 %rd18,%rd18,32; add.u64 %rd20,%rd20,292;")
A("setp.lt.u32 %p3,%r13,%r39; @%p3 bra $LIT;")
A("setp.eq.u32 %p3,%r1,0; mul.wide.u32 %rd12,%r2,4; add.u64 %rd16,%rd4,%rd12; @%p3 st.global.f32 [%rd16],%f1;")
A("$done:")
A("ret;")
A("}")

ptx = "\n".join(P) + "\n"
open('/tmp/k2v7.ptx', 'w').write(ptx)
print("wrote /tmp/k2v7.ptx", len(P), "lines")

if '--embed' in sys.argv:
    clines = ['\t"' + ln.replace('\\', '\\\\').replace('"', '\\"') + '\\n"' for ln in P]
    ckernel = "static const char *ptx =\n" + "\n".join(clines) + ";"
    s = open('/tmp/k2_cuda.c').read()
    i0 = s.index('static const char *ptx =')
    j = s.index('static void ck', i0)
    i1 = s.rindex('";', i0, j)
    s = s[:i0] + ckernel + s[i1 + 2:]
    open('/tmp/k2_cuda.c', 'w').write(s)
    print("embedded into /tmp/k2_cuda.c")
