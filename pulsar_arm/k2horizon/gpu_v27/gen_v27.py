#!/usr/bin/env python3
"""Generate the K2 Q4_0 warp GEMV PTX (v27: v26 + CROSS-BODY LOAD PIPELINING (loads for body N+1 issued a full body early).
Trip = 2 iters (A then B). B-loads issue before A-compute (22 loads in flight, covering
L2-miss latency). A/B/pang exchange areas (256B each, upper shbuf) drop the post-replay bar.
Bit-exact: identical per-(iter,row,sv) FMA/leaf order; unrolling only reorders across iters.
"""
import sys

P = []
A = P.append
A("// K2 Q4 GEMV v25 (Exp19: 4-lane replay — row q chain owned by lane q; 96->16 replay instrs/iter)")
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
A("shl.b32 %r3,%r3,3; add.u32 %r3,%r3,%r2; shl.b32 %r2,%r3,2; add.u32 %r2,%r2,%r10;")
A("add.u32 %r7,%r2,1; add.u32 %r70,%r2,2; add.u32 %r71,%r2,3;")
A("setp.ge.u32 %p1,%r2,%r11; @%p1 bra $done;")
A("setp.ge.u32 %p4,%r7,%r11;")
A("selp.b32 %r7,%r2,%r7,%p4;")
A("setp.ge.u32 %p5,%r70,%r11;")
A("selp.b32 %r70,%r2,%r70,%p5;")
A("setp.ge.u32 %p6,%r71,%r11;")
A("selp.b32 %r71,%r2,%r71,%p6;")
A("shr.u32 %r33,%r1,2;")
A("and.b32 %r34,%r1,3;")
for dst, row in (("%rd5", "%r2"), ("%rd6", "%r7"), ("%rd22", "%r70"), ("%rd23", "%r71")):
    A(f"mul.lo.u32 %r4,{row},%r12; shl.b32 %r4,%r4,4;")
    A("shl.b32 %r5,%r33,4; shl.b32 %r6,%r34,2; add.u32 %r4,%r4,%r5; add.u32 %r4,%r4,%r6;")
    A(f"cvt.u64.u32 {dst},%r4; add.u64 {dst},{dst},%rd1;")
A("shl.b32 %r5,%r33,5; shl.b32 %r6,%r34,2; add.u32 %r5,%r5,%r6; add.u32 %r5,%r5,4;")
A("cvt.u64.u32 %rd17,%r5; add.u64 %rd17,%rd17,%rd3;")
for dst, row in (("%rd18", "%r2"), ("%rd21", "%r7"), ("%rd7", "%r70"), ("%rd8", "%r71")):
    A(f"mul.lo.u32 %r4,{row},%r12; shl.b32 %r4,%r4,2; cvt.u64.u32 {dst},%r4; add.u64 {dst},{dst},%rd2;")
A("shl.b32 %r5,%r33,2; cvt.u64.u32 %rd19,%r5;")
A("add.u64 %rd18,%rd18,%rd19; add.u64 %rd21,%rd21,%rd19; add.u64 %rd7,%rd7,%rd19; add.u64 %rd8,%rd8,%rd19;")
A("mov.u64 %rd20,%rd3;")
A("add.u32 %r39,%r12,7; shr.u32 %r39,%r39,3;")
A("setp.eq.u32 %p0,%r1,0;")
A("mov.b32 %r61,shbuf; mov.u32 %r63,%tid.y; shl.b32 %r63,%r63,10; add.u32 %r65,%r61,%r63;")
A("mul.lo.u32 %r60,%r1,4; add.u32 %r62,%r65,%r60;")
A("mul.lo.u32 %r66,%r33,16;")
# trip bounds: r90 = iters&~1 (r13 terminal), r91 = iters&1 (epilogue flag)
A("and.b32 %r90,%r39,0xFFFFFFFE;")
A("and.b32 %r91,%r39,1;")
# per-warp exchange areas (upper shbuf): ping A=r79, pong B=r92, pang epi=r93
A("add.u32 %r79,%r65,8192; add.u32 %r92,%r65,8448; add.u32 %r93,%r65,8704;")
A("mov.f32 %f1,0F00000000; mov.u32 %r13,0;")


def body(c0, c1, c2, c3, x8, x16, sc0, sc1, sc2, sc3, db, d0, d1, d2, d3, area, it1, p_loads=1, p_rest=1, delta=None):
    # cN/x/sc/db: load-dest regs; dN: intdot accum regs; area: exchange base reg; it1: 0/1
    if p_loads:
        DC, DX, DS = delta if delta else ((128, 292, 32) if it1 else (0, 0, 0))
        A(f"ld.global.u32 %{c0},[%rd5+{DC}];")
        A(f"ld.global.u32 %{c1},[%rd6+{DC}];")
        A(f"ld.global.u32 %{c2},[%rd22+{DC}];")
        A(f"ld.global.u32 %{c3},[%rd23+{DC}];")
        A(f"ld.global.u32 %{x8},[%rd17+{DX}];")
        A(f"ld.global.u32 %{x16},[%rd17+{16+DX}];")
        A(f"ld.global.f32 %{sc0},[%rd18+{DS}]; ld.global.f32 %{sc1},[%rd21+{DS}]; ld.global.f32 %{sc2},[%rd7+{DS}]; ld.global.f32 %{sc3},[%rd8+{DS}]; ld.global.f32 %{db},[%rd20+{DX}];")
    if p_rest:
        if it1:
            A("add.u32 %r14,%r13,1; shl.b32 %r14,%r14,3; add.u32 %r14,%r14,%r33; setp.ge.u32 %p2,%r14,%r12;")
        else:
            A("shl.b32 %r14,%r13,3; add.u32 %r14,%r14,%r33; setp.ge.u32 %p2,%r14,%r12;")
        for r in range(4):
            pos = 8 * r
            # int8 x bytes: bfe.s32(width 8) == shl/shr.s32 sign-extend of byte r
            A(f"bfe.s32 %r43,%{x8},{pos},8;")
            A(f"bfe.s32 %r44,%{x16},{pos},8;")
            # int4 code nibbles: xor 0x88<<8r flips nibble sign bits, bfe.s32(width 4)
            # gives exactly (n-8) signed; mad.lo.s32 fuses product+accumulate (int = exact).
            for (_c, _acc) in ((c0, d0), (c1, d1), (c2, d2), (c3, d3)):
                A(f"xor.b32 %r97,%{_c},0x{(0x88 << pos) & 0xFFFFFFFF:08X};")
                A(f"bfe.s32 %r41,%r97,{pos},4;")
                A(f"bfe.s32 %r42,%r97,{pos + 4},4;")
                if r == 0:
                    A(f"mad.lo.s32 %{_acc},%r41,%r43,0;")
                else:
                    A(f"mad.lo.s32 %{_acc},%r41,%r43,%{_acc};")
                A(f"mad.lo.s32 %{_acc},%r42,%r44,%{_acc};")
        A(f"cvt.rn.f32.s32 %f6,%{d0}; mul.f32 %f7,%{sc0},%{db}; selp.f32 %f7,0F00000000,%f7,%p2;")
        A(f"cvt.rn.f32.s32 %f17,%{d1}; mul.f32 %f18,%{sc1},%{db};")
        A("or.pred %p7,%p2,%p4; selp.f32 %f18,0F00000000,%f18,%p7;")
        A(f"cvt.rn.f32.s32 %f20,%{d2}; mul.f32 %f21,%{sc2},%{db};")
        A("or.pred %p7,%p2,%p5; selp.f32 %f21,0F00000000,%f21,%p7;")
        A(f"cvt.rn.f32.s32 %f23,%{d3}; mul.f32 %f24,%{sc3},%{db};")
        A("or.pred %p7,%p2,%p6; selp.f32 %f24,0F00000000,%f24,%p7;")
        A("st.shared.f32 [%r62],%f7; st.shared.f32 [%r62+128],%f6;")
        A("st.shared.f32 [%r62+256],%f18; st.shared.f32 [%r62+384],%f17;")
        A("st.shared.f32 [%r62+512],%f21; st.shared.f32 [%r62+640],%f20;")
        A("st.shared.f32 [%r62+768],%f24; st.shared.f32 [%r62+896],%f23;")
        A("bar.sync 0;")
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
        A(f"add.u32 %r77,%r77,%{area};")
        A("st.shared.v2.f32 [%r77],{%f14,%f8};")
        A("bar.sync 0;")
        # 4-LANE PARALLEL REPLAY: lane q (0..3) owns row q's chain; reads its 8 (sc,dot)
        # pairs at stride 32 (base 8q) and FMAs them in sv order => identical order => bit-exact.
        A("setp.lt.u32 %p0,%r1,4;")
        A(f"shl.b32 %r96,%r1,3; add.u32 %r96,%r96,%{area};")
        for _sv in range(8):
            A(f"@%p0 ld.shared.v2.f32 {{%f14,%f8}},[%r96+{32*_sv}];")
            A("@%p0 fma.rn.f32 %f1,%f14,%f8,%f1;")


# prologue: A values for the first body, loaded before $LIT
body("r20", "r22", "r26", "r27", "r24", "r28", "f2", "f4", "f5", "f9", "f3",
     "r46", "r49", "r35", "r36", "r79", 0, p_loads=1, p_rest=0)
A("$LIT:")
# CROSS-BODY PIPELINING: B loads issue first (cover = A rest ~120 instrs),
# then A compute, then NEXT-TRIP A loads (cover = B rest), then B compute.
body("r80", "r81", "r82", "r83", "r84", "r85", "f26", "f27", "f28", "f29", "f30",
     "r86", "r87", "r88", "r89", "r92", 1, p_loads=1, p_rest=0)
body("r20", "r22", "r26", "r27", "r24", "r28", "f2", "f4", "f5", "f9", "f3",
     "r46", "r49", "r35", "r36", "r79", 0, p_loads=0, p_rest=1)
body("r20", "r22", "r26", "r27", "r24", "r28", "f2", "f4", "f5", "f9", "f3",
     "r46", "r49", "r35", "r36", "r79", 0, p_loads=1, p_rest=0, delta=(256, 584, 64))
body("r80", "r81", "r82", "r83", "r84", "r85", "f26", "f27", "f28", "f29", "f30",
     "r86", "r87", "r88", "r89", "r92", 1, p_loads=0, p_rest=1)
A("add.u32 %r13,%r13,2;")
A("add.u64 %rd5,%rd5,256; add.u64 %rd6,%rd6,256; add.u64 %rd22,%rd22,256; add.u64 %rd23,%rd23,256;")
A("add.u64 %rd17,%rd17,584; add.u64 %rd18,%rd18,64; add.u64 %rd21,%rd21,64; add.u64 %rd7,%rd7,64; add.u64 %rd8,%rd8,64; add.u64 %rd20,%rd20,584;")
A("setp.lt.u32 %p3,%r13,%r90; @%p3 bra $LIT;")
# odd epilogue (it=r13, pang r93), then join stores
A("setp.ne.u32 %p7,%r91,0; @%p7 bra $epi; bra $stor;")
A("$epi:")
# odd tail: uses A regs loaded at the end of the last trip (no loads here)
body("r20", "r22", "r26", "r27", "r24", "r28", "f2", "f4", "f5", "f9", "f3",
     "r46", "r49", "r35", "r36", "r93", 0, p_loads=0, p_rest=1)
A("add.u32 %r13,%r13,1;")
A("add.u64 %rd5,%rd5,128; add.u64 %rd6,%rd6,128; add.u64 %rd22,%rd22,128; add.u64 %rd23,%rd23,128;")
A("add.u64 %rd17,%rd17,292; add.u64 %rd18,%rd18,32; add.u64 %rd21,%rd21,32; add.u64 %rd7,%rd7,32; add.u64 %rd8,%rd8,32; add.u64 %rd20,%rd20,292;")
A("$stor:")
# lane q<4 stores row r2+q; force invalid address for lanes>=4 via predicated mov
A("add.u32 %r96,%r2,%r1;")
A("setp.ge.u32 %p7,%r1,4;")
A("@%p7 mov.u32 %r96,%r11;")
A("setp.lt.u32 %p0,%r96,%r11;")
A("mul.wide.u32 %rd12,%r96,4; add.u64 %rd16,%rd4,%rd12;")
A("@%p0 st.global.f32 [%rd16],%f1;")
A("$done:")
A("ret;")
A("}")

ptx = "\n".join(P) + "\n"
open('/tmp/k2v27.ptx', 'w').write(ptx)
print("wrote /tmp/k2v27.ptx", len(P), "lines")
