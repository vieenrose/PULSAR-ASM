#!/usr/bin/env python3
"""Generate the K2 Q4_0 warp GEMV PTX (v22: 4-row blocking, LANE-SPLIT exchange, single-copy 16B code).
Warp covers 4 rows (r2..r2+3); x8/x16/db loaded once, dotted vs 4 code/sc rows.
Per-row products/order identical to v10/v11 => bit-exact by construction.
Grid: 32 rows/block (8 warps). Shmem: 8x128B dense arrays per warp, 1024B stride.
"""
import sys

P = []
A = P.append
A("// K2 Q4 GEMV v12 (Exp6: 4-row blocking on single-copy 16B code): warp covers 4 rows, 8 blocks/iter, pure pointer walks")
A(".version 4.0")
A(".target sm_50")
A(".address_size 64")
A(".visible .entry q4_gemv("
  ".param .u64 pw, .param .u64 psc, .param .u64 pxq, .param .u64 py, "
  ".param .u32 pr0, .param .u32 pr1, .param .u32 pnb, .param .u32 prf) {")
A(".reg .u64 %rd<24>; .reg .u32 %r<128>; .reg .f32 %f<32>; .reg .pred %p<8>;")
A(".shared .align 4 .b32 shbuf[4096];")
A("ld.param.u64 %rd1,[pw]; ld.param.u64 %rd2,[psc]; ld.param.u64 %rd3,[pxq]; ld.param.u64 %rd4,[py];")
A("ld.param.u32 %r10,[pr0]; ld.param.u32 %r11,[pr1]; ld.param.u32 %r12,[pnb];")
A("mov.u32 %r1,%tid.x; mov.u32 %r2,%tid.y; mov.u32 %r3,%ctaid.x;")
# row0 = (ctaid*8 + tid.y)*4 + row0 ; rows 1..3 follow; clamp OOB rows to row0
A("shl.b32 %r3,%r3,3; add.u32 %r3,%r3,%r2; shl.b32 %r2,%r3,2; add.u32 %r2,%r2,%r10;")
A("add.u32 %r7,%r2,1; add.u32 %r70,%r2,2; add.u32 %r71,%r2,3;")
A("setp.ge.u32 %p1,%r2,%r11; @%p1 bra $done;")
A("setp.ge.u32 %p4,%r7,%r11;")
A("selp.b32 %r7,%r2,%r7,%p4;")
A("setp.ge.u32 %p5,%r70,%r11;")
A("selp.b32 %r70,%r2,%r70,%p5;")
A("setp.ge.u32 %p6,%r71,%r11;")
A("selp.b32 %r71,%r2,%r71,%p6;")
# lane constants: s = t>>2 (block within iter), q = t&3 (byte within block)
A("shr.u32 %r33,%r1,2;")
A("and.b32 %r34,%r1,3;")
# code bases: w + row*nb32*16 + 16s + 4q  (single-copy 16B blocks)
for dst, row in (("%rd5", "%r2"), ("%rd6", "%r7"), ("%rd22", "%r70"), ("%rd23", "%r71")):
    A(f"mul.lo.u32 %r4,{row},%r12; shl.b32 %r4,%r4,4;")
    A("shl.b32 %r5,%r33,4; shl.b32 %r6,%r34,2; add.u32 %r4,%r4,%r5; add.u32 %r4,%r4,%r6;")
    A(f"cvt.u64.u32 {dst.replace('rd', 'rd')},%r4; add.u64 {dst},{dst},%rd1;")
# x8 base: xq + 32s + q + 4
A("shl.b32 %r5,%r33,5; shl.b32 %r6,%r34,2; add.u32 %r5,%r5,%r6; add.u32 %r5,%r5,4;")
A("cvt.u64.u32 %rd17,%r5; add.u64 %rd17,%rd17,%rd3;")
# sc bases: sc + row*nb32*4 + 4s
for dst, row in (("%rd18", "%r2"), ("%rd21", "%r7"), ("%rd7", "%r70"), ("%rd8", "%r71")):
    A(f"mul.lo.u32 %r4,{row},%r12; shl.b32 %r4,%r4,2; cvt.u64.u32 {dst},%r4; add.u64 {dst},{dst},%rd2;")
A("shl.b32 %r5,%r33,2; cvt.u64.u32 %rd19,%r5;")
A("add.u64 %rd18,%rd18,%rd19; add.u64 %rd21,%rd21,%rd19; add.u64 %rd7,%rd7,%rd19; add.u64 %rd8,%rd8,%rd19;")
# db base: xq scale float (row-independent)
A("mov.u64 %rd20,%rd3;")
# iterations = ceil(nb32/8)
A("add.u32 %r39,%r12,7; shr.u32 %r39,%r39,3;")
A("setp.eq.u32 %p0,%r1,0;")
# shared layout: dense 8x128B per warp, 1024B stride
# row0: sc@0 dot@128 | row1: sc@256 dot@384 | row2: sc@512 dot@640 | row3: sc@768 dot@896
A("mov.b32 %r61,shbuf; mov.u32 %r63,%tid.y; shl.b32 %r63,%r63,10; add.u32 %r65,%r61,%r63;")
# NOTE: line56 (r61/r65) is INSIDE $LIT (recomputed per iter); dist base derived fresh per iter in r79
A("mul.lo.u32 %r60,%r1,4; add.u32 %r62,%r65,%r60;")
A("mul.lo.u32 %r66,%r33,16;")
A("mov.f32 %f1,0F00000000; mov.f32 %f16,0F00000000; mov.f32 %f22,0F00000000; mov.f32 %f25,0F00000000; mov.u32 %r13,0;")
A("$LIT:")
# b = 8*it + s ; invalid if b >= nb32
A("shl.b32 %r14,%r13,3; add.u32 %r14,%r14,%r33; setp.ge.u32 %p2,%r14,%r12;")
# vector loads: code u32 x4 (one per row), x8 u32, x16 u32
A("ld.global.u32 %r20,[%rd5];")
A("ld.global.u32 %r22,[%rd6];")
A("ld.global.u32 %r26,[%rd22];")
A("ld.global.u32 %r27,[%rd23];")
A("ld.global.u32 %r24,[%rd17];")
A("ld.global.u32 %r28,[%rd17+16];")
A("ld.global.f32 %f2,[%rd18]; ld.global.f32 %f4,[%rd21]; ld.global.f32 %f5,[%rd7]; ld.global.f32 %f9,[%rd8]; ld.global.f32 %f3,[%rd20];")
for r in range(4):
    sh = 8 * r
    # x nibbles (shared across rows; computed once here)
    A(f"shl.b32 %r43,%r24,{24-sh}; shr.s32 %r43,%r43,24;")
    A(f"shl.b32 %r44,%r28,{24-sh}; shr.s32 %r44,%r44,24;")
    # row0
    A(f"shr.u32 %r40,%r20,{sh}; and.b32 %r40,%r40,255;")
    A("and.b32 %r41,%r40,15; sub.s32 %r41,%r41,8;")
    A("shr.u32 %r42,%r40,4; and.b32 %r42,%r42,15; sub.s32 %r42,%r42,8;")
    A("mul.lo.s32 %r45,%r41,%r43; mul.lo.s32 %r47,%r42,%r44; add.s32 %r48,%r45,%r47;")
    # row1
    A(f"shr.u32 %r50,%r22,{sh}; and.b32 %r50,%r50,255;")
    A("and.b32 %r51,%r50,15; sub.s32 %r51,%r51,8;")
    A("shr.u32 %r52,%r50,4; and.b32 %r52,%r52,15; sub.s32 %r52,%r52,8;")
    A("mul.lo.s32 %r53,%r51,%r43; mul.lo.s32 %r54,%r52,%r44; add.s32 %r55,%r53,%r54;")
    # row2
    A(f"shr.u32 %r56,%r26,{sh}; and.b32 %r56,%r56,255;")
    A("and.b32 %r57,%r56,15; sub.s32 %r57,%r57,8;")
    A("shr.u32 %r58,%r56,4; and.b32 %r58,%r58,15; sub.s32 %r58,%r58,8;")
    A("mul.lo.s32 %r15,%r57,%r43; mul.lo.s32 %r16,%r58,%r44; add.s32 %r59,%r15,%r16;")
    # row3
    A(f"shr.u32 %r29,%r27,{sh}; and.b32 %r29,%r29,255;")
    A("and.b32 %r30,%r29,15; sub.s32 %r30,%r30,8;")
    A("shr.u32 %r31,%r29,4; and.b32 %r31,%r31,15; sub.s32 %r31,%r31,8;")
    A("mul.lo.s32 %r37,%r30,%r43; mul.lo.s32 %r38,%r31,%r44; add.s32 %r32,%r37,%r38;")
    if r == 0:
        A("mov.b32 %r46,%r48; mov.b32 %r49,%r55; mov.b32 %r35,%r59; mov.b32 %r36,%r32;")
    else:
        A("add.s32 %r46,%r46,%r48; add.s32 %r49,%r49,%r55; add.s32 %r35,%r35,%r59; add.s32 %r36,%r36,%r32;")
# int dots -> fp32, scale, mask invalid (block OOB or row OOB-clamped)
A("cvt.rn.f32.s32 %f6,%r46; mul.f32 %f7,%f2,%f3; selp.f32 %f7,0F00000000,%f7,%p2;")
A("cvt.rn.f32.s32 %f17,%r49; mul.f32 %f18,%f4,%f3;")
A("or.pred %p7,%p2,%p4; selp.f32 %f18,0F00000000,%f18,%p7;")
A("cvt.rn.f32.s32 %f20,%r35; mul.f32 %f21,%f5,%f3;")
A("or.pred %p7,%p2,%p5; selp.f32 %f21,0F00000000,%f21,%p7;")
A("cvt.rn.f32.s32 %f23,%r36; mul.f32 %f24,%f9,%f3;")
A("or.pred %p7,%p2,%p6; selp.f32 %f24,0F00000000,%f24,%p7;")
# publish to shmem (all lanes)
A("st.shared.f32 [%r62],%f7; st.shared.f32 [%r62+128],%f6;")
A("st.shared.f32 [%r62+256],%f18; st.shared.f32 [%r62+384],%f17;")
A("st.shared.f32 [%r62+512],%f21; st.shared.f32 [%r62+640],%f20;")
A("st.shared.f32 [%r62+768],%f24; st.shared.f32 [%r62+896],%f23;")
A("bar.sync 0;")
# LANE-SPLIT EXCHANGE: lane L=(q*8+sv), q=tid.x>>3, sv=tid.x&7 reads ONLY its
# row-q sv-partial (5 shmem reads vs 160). All lanes write (sc,dotsum) uniformly
# to exchange area; bar; lane0 replays the 32 FMAs in exact v12 order => bit-exact.
A("shr.u32 %r76,%r1,3;")
A("and.b32 %r77,%r1,7;")
A("shl.b32 %r78,%r76,8; add.u32 %r78,%r78,%r65;")
A("shl.b32 %r77,%r77,4; add.u32 %r78,%r78,%r77;")
A("ld.shared.f32 %f10,[%r78+128];")
A("ld.shared.f32 %f11,[%r78+132];")
A("ld.shared.f32 %f12,[%r78+136];")
A("ld.shared.f32 %f13,[%r78+140];")
A("add.f32 %f8,%f10,%f11;")
A("add.f32 %f8,%f8,%f12;")
A("add.f32 %f8,%f8,%f13;")
A("ld.shared.f32 %f14,[%r78+0];")
A("shl.b32 %r76,%r76,2;")
A("add.u32 %r77,%r77,%r76;")
A("shl.b32 %r77,%r77,1;")
A("add.u32 %r79,%r65,8192; add.u32 %r77,%r77,%r79;")
A("st.shared.f32 [%r77+0],%f14;")
A("st.shared.f32 [%r77+4],%f8;")
A("bar.sync 0;")
A("setp.eq.u32 %p0,%r1,0;")
A("add.u32 %r79,%r65,8192;")
for (_q, _acc) in ((0, "%f1"), (1, "%f16"), (2, "%f22"), (3, "%f25")):
    for _sv in range(8):
        A(f"@%p0 ld.shared.f32 %f14,[%r79+{8*(_sv*4+_q)+0}];")
        A(f"@%p0 ld.shared.f32 %f8,[%r79+{8*(_sv*4+_q)+4}];")
        A(f"@%p0 fma.rn.f32 {_acc},%f14,%f8,{_acc};")
A("bar.sync 0;")
A("add.u32 %r13,%r13,1;")
A("add.u64 %rd5,%rd5,128; add.u64 %rd6,%rd6,128; add.u64 %rd22,%rd22,128; add.u64 %rd23,%rd23,128;")
A("add.u64 %rd17,%rd17,292; add.u64 %rd18,%rd18,32; add.u64 %rd21,%rd21,32; add.u64 %rd7,%rd7,32; add.u64 %rd8,%rd8,32; add.u64 %rd20,%rd20,292;")
A("setp.lt.u32 %p3,%r13,%r39; @%p3 bra $LIT;")
# lane0 stores (rows 1..3 guarded by validity)
A("setp.eq.u32 %p0,%r1,0; mul.wide.u32 %rd12,%r2,4; add.u64 %rd16,%rd4,%rd12; @%p0 st.global.f32 [%rd16],%f1;")
A("@%p4 bra $skip1;")
A("mul.wide.u32 %rd12,%r7,4; add.u64 %rd16,%rd4,%rd12; @%p0 st.global.f32 [%rd16],%f16;")
A("$skip1:")
A("@%p5 bra $skip2;")
A("mul.wide.u32 %rd12,%r70,4; add.u64 %rd16,%rd4,%rd12; @%p0 st.global.f32 [%rd16],%f22;")
A("$skip2:")
A("@%p6 bra $skip3;")
A("mul.wide.u32 %rd12,%r71,4; add.u64 %rd16,%rd4,%rd12; @%p0 st.global.f32 [%rd16],%f25;")
A("$skip3:")
A("$done:")
A("ret;")
A("}")

ptx = "\n".join(P) + "\n"
open('/tmp/k2v22.ptx', 'w').write(ptx)
print("wrote /tmp/k2v22.ptx", len(P), "lines")
