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
