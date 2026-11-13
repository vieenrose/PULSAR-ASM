// lfm3_cpurows.cpp -- NEON rows of the batched-prefill GEMV, in q8_partial_TLf/TLg's exact order
// (see LFM_CPUCHK in lfm3_nanode.cu). Separate TU: nvcc's front end cannot parse arm_neon.h.
// Build: g++ -O3 -mcpu=cortex-a57 -ffp-contract=off -c lfm3_cpurows.cpp
#include <arm_neon.h>
#include <stdint.h>
#include <stddef.h>
extern "C" void lfm3_neon_rows(const int8_t* hq, const float* hs, const float* xT, int nb, int G, int r_lo, int r_hi, float* out) {
  const size_t tq = (size_t)nb * 64 * 16;
  for (int rr = r_lo; rr < r_hi; rr++) {
    const int8_t* tq8 = hq + (size_t)(rr >> 5) * tq; const float* tsc = hs + (size_t)(rr >> 5) * nb * 32;
    float32x4_t st[5][8], pv[8];
    for (int k = 0; k < 32; k++) {
      const int i = ((k & 1) << 4) | ((k & 2) << 2) | (k & 4) | ((k & 8) >> 2) | ((k & 16) >> 4);
      float32x4_t acc[8];
      for (int g = 0; g < 8; g++) acc[g] = vdupq_n_f32(0.f);
      for (int b = i; b < nb; b += 32) {
        const int8_t* c0 = tq8 + ((size_t)b * 64 + (rr & 31)) * 16;
        const int8_t* c1 = tq8 + ((size_t)b * 64 + 32 + (rr & 31)) * 16;
        float32x4_t dot[8];
        for (int g = 0; g < 8; g++) dot[g] = vdupq_n_f32(0.f);
        const float* xb = xT + (size_t)b * 32 * G * 4;
        for (int j = 0; j < 32; j++) {
          const float32x4_t c = vdupq_n_f32((float)(j < 16 ? c0[j] : c1[j - 16]));
          const float* xj = xb + (size_t)j * G * 4;
          if (G == 8) { for (int g = 0; g < 8; g++) dot[g] = vfmaq_f32(dot[g], c, vld1q_f32(xj + 4 * g)); }
          else for (int g = 0; g < G; g++) dot[g] = vfmaq_f32(dot[g], c, vld1q_f32(xj + 4 * g));
        }
        const float32x4_t sc = vdupq_n_f32(tsc[(size_t)b * 32 + (rr & 31)]);
        for (int g = 0; g < G; g++) acc[g] = vfmaq_f32(acc[g], sc, dot[g]);
      }
      for (int g = 0; g < G; g++) pv[g] = acc[g];
      for (int l = 0; l < 5; l++) {
        if ((k >> l) & 1) { for (int g = 0; g < G; g++) pv[g] = vaddq_f32(st[l][g], pv[g]); }
        else { for (int g = 0; g < G; g++) st[l][g] = pv[g]; break; }
      }
    }
    for (int g = 0; g < G; g++) vst1q_f32(out + ((size_t)(rr - r_lo) * G + g) * 4, pv[g]);
  }
}

// v2 (run #297): weights row-major (codes [R][K] int8, scales [R][nb]), two rows per pass sharing
// each x vector, codes widened 4 at a time (vmovl x2 + vcvtq), products by vfmaq_laneq_f32 (one
// fused fma per product, same operands and order as v1 / the GPU leaf). G must be 8 (32 positions)
// or less; the per-g loops are unrolled for G == 8.
template <int G>
static void rows2(const int8_t* q, const float* sc, const float* xT, int K, int nb, int r_lo, int r_hi, float* out) {
  for (int r = r_lo; r < r_hi; r += 2) {
    const int nr = r + 1 < r_hi ? 2 : 1;
    const int8_t* qa = q + (size_t)r * K; const int8_t* qb = q + (size_t)(r + nr - 1) * K;
    const float* sa = sc + (size_t)r * nb; const float* sb = sc + (size_t)(r + nr - 1) * nb;
    float32x4_t sta[5][G], stb[5][G], pa[G], pb[G];
    for (int k = 0; k < 32; k++) {
      const int i = ((k & 1) << 4) | ((k & 2) << 2) | (k & 4) | ((k & 8) >> 2) | ((k & 16) >> 4);
      float32x4_t aa[G], ab[G];
      for (int g = 0; g < G; g++) { aa[g] = vdupq_n_f32(0.f); ab[g] = vdupq_n_f32(0.f); }
      for (int b = i; b < nb; b += 32) {
        float32x4_t da[G], db[G];
        for (int g = 0; g < G; g++) { da[g] = vdupq_n_f32(0.f); db[g] = vdupq_n_f32(0.f); }
        const float* xb = xT + (size_t)b * 32 * G * 4;
        for (int j0 = 0; j0 < 32; j0 += 8) {
          const int16x8_t wa = vmovl_s8(vld1_s8(qa + b * 32 + j0)), wb = vmovl_s8(vld1_s8(qb + b * 32 + j0));
          const float32x4_t ca0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(wa))), ca1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(wa)));
          const float32x4_t cb0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(wb))), cb1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(wb)));
#define STEP(J, CA, CB, L) { const float* xj = xb + (size_t)(j0 + J) * G * 4; \
          for (int g = 0; g < G; g++) { const float32x4_t xv = vld1q_f32(xj + 4 * g); \
            da[g] = vfmaq_laneq_f32(da[g], xv, CA, L); db[g] = vfmaq_laneq_f32(db[g], xv, CB, L); } }
          STEP(0, ca0, cb0, 0) STEP(1, ca0, cb0, 1) STEP(2, ca0, cb0, 2) STEP(3, ca0, cb0, 3)
          STEP(4, ca1, cb1, 0) STEP(5, ca1, cb1, 1) STEP(6, ca1, cb1, 2) STEP(7, ca1, cb1, 3)
#undef STEP
        }
        const float32x4_t va = vdupq_n_f32(sa[b]), vb = vdupq_n_f32(sb[b]);
        for (int g = 0; g < G; g++) { aa[g] = vfmaq_f32(aa[g], va, da[g]); ab[g] = vfmaq_f32(ab[g], vb, db[g]); }
      }
      for (int g = 0; g < G; g++) { pa[g] = aa[g]; pb[g] = ab[g]; }
      for (int l = 0; l < 5; l++) {
        if ((k >> l) & 1) { for (int g = 0; g < G; g++) { pa[g] = vaddq_f32(sta[l][g], pa[g]); pb[g] = vaddq_f32(stb[l][g], pb[g]); } }
        else { for (int g = 0; g < G; g++) { sta[l][g] = pa[g]; stb[l][g] = pb[g]; } break; }
      }
    }
    for (int g = 0; g < G; g++) vst1q_f32(out + ((size_t)(r - r_lo) * G + g) * 4, pa[g]);
    if (nr == 2) for (int g = 0; g < G; g++) vst1q_f32(out + ((size_t)(r + 1 - r_lo) * G + g) * 4, pb[g]);
  }
}
extern "C" void lfm3_neon_rows2(const int8_t* q, const float* sc, const float* xT, int K, int nb, int G,
                                int r_lo, int r_hi, float* out) {
  switch (G) {
    case 8: rows2<8>(q, sc, xT, K, nb, r_lo, r_hi, out); break;
    case 7: rows2<7>(q, sc, xT, K, nb, r_lo, r_hi, out); break;
    case 6: rows2<6>(q, sc, xT, K, nb, r_lo, r_hi, out); break;
    case 5: rows2<5>(q, sc, xT, K, nb, r_lo, r_hi, out); break;
    case 4: rows2<4>(q, sc, xT, K, nb, r_lo, r_hi, out); break;
    case 3: rows2<3>(q, sc, xT, K, nb, r_lo, r_hi, out); break;
    case 2: rows2<2>(q, sc, xT, K, nb, r_lo, r_hi, out); break;
    default: rows2<1>(q, sc, xT, K, nb, r_lo, r_hi, out); break;
  }
}
