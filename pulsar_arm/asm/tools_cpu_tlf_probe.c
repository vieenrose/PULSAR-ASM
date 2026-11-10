// tools_cpu_tlf_probe.c -- throughput probe for a CPU share of the LFM3 prefill GEMM (TLf
// shape: K = 2048 int8 codes per row, 64 Q8 blocks of 32, one fp32 scale per block, 32
// prompt positions per chunk). Per row and block the dot is a sequential fp32 fma chain per
// position (positions in NEON lanes, 8 x float32x4 = 32 positions), folded with the block
// scale; this is the arithmetic shape an exact CPU row share would run. Timing only.
//   gcc -O3 -mcpu=cortex-a57 -fopenmp -o cpu_tlf_probe tools_cpu_tlf_probe.c   (-O3: 5.08 vs 3.34 GMAC/s at -O2, 3 threads)
//   ./cpu_tlf_probe <rows> <reps> [threads]
#include <arm_neon.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define K 2048
#define NB (K / 32)
#define P 32

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

// Blocked form: R rows share every x load; G float32x4 position vectors per pass (P/(4G)
// passes per block); weights multiplied by element (fmla v, x, w.s[e]) from the converted
// vector. Per row and position the chain is the naive kernel's (same codes, same order).
#define DEFK(R, G) \
static void kern_##R##_##G(const int8_t* q, const float* s, const float* x, float* y, int rows) { \
  _Pragma("omp parallel for schedule(static)") \
  for (int n0 = 0; n0 < rows; n0 += R) { \
    for (int i = 0; i < R * P; i++) y[(size_t)n0 * P + i] = 0.f; \
    for (int b = 0; b < NB; b++) { \
      for (int g0 = 0; g0 < P / 4; g0 += G) { \
        float32x4_t d[R][G]; \
        for (int r = 0; r < R; r++) for (int g = 0; g < G; g++) d[r][g] = vdupq_n_f32(0.f); \
        for (int c = 0; c < 32; c += 8) { \
          float32x4_t wl[R], wh[R]; \
          for (int r = 0; r < R; r++) { \
            const int16x8_t w16 = vmovl_s8(vld1_s8(q + (size_t)(n0 + r) * K + b * 32 + c)); \
            wl[r] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16))); \
            wh[r] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16))); } \
          for (int e = 0; e < 8; e++) { \
            const float* xk = x + (size_t)(b * 32 + c + e) * P + 4 * g0; \
            float32x4_t xv[G]; for (int g = 0; g < G; g++) xv[g] = vld1q_f32(xk + 4 * g); \
            for (int r = 0; r < R; r++) for (int g = 0; g < G; g++) { \
              switch (e) { \
                case 0: d[r][g] = vfmaq_laneq_f32(d[r][g], xv[g], wl[r], 0); break; \
                case 1: d[r][g] = vfmaq_laneq_f32(d[r][g], xv[g], wl[r], 1); break; \
                case 2: d[r][g] = vfmaq_laneq_f32(d[r][g], xv[g], wl[r], 2); break; \
                case 3: d[r][g] = vfmaq_laneq_f32(d[r][g], xv[g], wl[r], 3); break; \
                case 4: d[r][g] = vfmaq_laneq_f32(d[r][g], xv[g], wh[r], 0); break; \
                case 5: d[r][g] = vfmaq_laneq_f32(d[r][g], xv[g], wh[r], 1); break; \
                case 6: d[r][g] = vfmaq_laneq_f32(d[r][g], xv[g], wh[r], 2); break; \
                default: d[r][g] = vfmaq_laneq_f32(d[r][g], xv[g], wh[r], 3); } } } \
        } \
        for (int r = 0; r < R; r++) { const float sc = s[(size_t)(n0 + r) * NB + b]; \
          for (int g = 0; g < G; g++) { float* yp = y + (size_t)(n0 + r) * P + 4 * (g0 + g); \
            vst1q_f32(yp, vfmaq_n_f32(vld1q_f32(yp), d[r][g], sc)); } } \
      } \
    } \
  } \
}
DEFK(1, 8) DEFK(2, 4) DEFK(2, 8) DEFK(3, 4) DEFK(4, 4) DEFK(4, 2) DEFK(6, 2)

int main(int argc, char** argv) {
  const int rows = argc > 1 ? atoi(argv[1]) : 1024, reps = argc > 2 ? atoi(argv[2]) : 20;
  if (argc > 3) omp_set_num_threads(atoi(argv[3]));
  int8_t* q = aligned_alloc(64, (size_t)rows * K);
  float* s = aligned_alloc(64, (size_t)rows * NB * 4);
  float* x = aligned_alloc(64, (size_t)K * P * 4);          // [k][position]
  float* y = aligned_alloc(64, (size_t)rows * P * 4);
  for (size_t i = 0; i < (size_t)rows * K; i++) q[i] = (int8_t)(rand() & 0xff);
  for (size_t i = 0; i < (size_t)rows * NB; i++) s[i] = 1e-3f * (rand() & 1023);
  for (size_t i = 0; i < (size_t)K * P; i++) x[i] = 1e-2f * ((rand() & 2047) - 1024);
  // reference checksum from the naive kernel, then every blocked variant must match bitwise
  float* yref = aligned_alloc(64, (size_t)rows * P * 4);
  double best = 1e30;
  for (int r = 0; r < reps; r++) {
    const double t0 = now();
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < rows; n++) {
      float32x4_t acc[8];
      for (int g = 0; g < 8; g++) acc[g] = vdupq_n_f32(0.f);
      const int8_t* qr = q + (size_t)n * K;
      for (int b = 0; b < NB; b++) {
        float32x4_t d[8];
        for (int g = 0; g < 8; g++) d[g] = vdupq_n_f32(0.f);
        for (int c = 0; c < 32; c += 8) {
          const int16x8_t w16 = vmovl_s8(vld1_s8(qr + b * 32 + c));
          const float32x4_t lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16)));
          const float32x4_t hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16)));
          const float wv[8] = { vgetq_lane_f32(lo, 0), vgetq_lane_f32(lo, 1), vgetq_lane_f32(lo, 2), vgetq_lane_f32(lo, 3),
                                vgetq_lane_f32(hi, 0), vgetq_lane_f32(hi, 1), vgetq_lane_f32(hi, 2), vgetq_lane_f32(hi, 3) };
          for (int e = 0; e < 8; e++) {
            const float* xk = x + (size_t)(b * 32 + c + e) * P;
            for (int g = 0; g < 8; g++) d[g] = vfmaq_n_f32(d[g], vld1q_f32(xk + 4 * g), wv[e]);
          }
        }
        const float sc = s[(size_t)n * NB + b];
        for (int g = 0; g < 8; g++) acc[g] = vfmaq_n_f32(acc[g], d[g], sc);
      }
      for (int g = 0; g < 8; g++) vst1q_f32(y + (size_t)n * P + 4 * g, acc[g]);
    }
    const double dt = now() - t0;
    if (dt < best) best = dt;
  }
  const double gmac = (double)rows * K * P / best * 1e-9;
  double cs = 0; for (size_t i = 0; i < (size_t)rows * P; i++) cs += y[i];
  printf("CPU_TLF naive rows=%d threads=%d best %.2f ms  %.2f GMAC/s  (checksum %.3g)\n",
         rows, omp_get_max_threads(), best * 1e3, gmac, cs);
  for (size_t i = 0; i < (size_t)rows * P; i++) yref[i] = y[i];
  struct { const char* nm; void (*f)(const int8_t*, const float*, const float*, float*, int); } V[] = {
    {"1x8", kern_1_8}, {"2x4", kern_2_4}, {"2x8", kern_2_8}, {"3x4", kern_3_4}, {"4x4", kern_4_4},
    {"4x2", kern_4_2}, {"6x2", kern_6_2} };
  for (int v = 0; v < 7; v++) {
    double bt = 1e30;
    for (int r = 0; r < reps; r++) { const double t0 = now(); V[v].f(q, s, x, y, rows); const double dt = now() - t0; if (dt < bt) bt = dt; }
    size_t bad = 0; for (size_t i = 0; i < (size_t)rows * P; i++) bad += memcmp(&y[i], &yref[i], 4) != 0;
    printf("CPU_TLF %s best %.2f ms  %.2f GMAC/s  mismatches %zu\n", V[v].nm, bt * 1e3, (double)rows * K * P / bt * 1e-9, bad);
  }
  return 0;
}
