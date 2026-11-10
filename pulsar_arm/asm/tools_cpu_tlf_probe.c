// tools_cpu_tlf_probe.c -- throughput probe for a CPU share of the LFM3 prefill GEMM (TLf
// shape: K = 2048 int8 codes per row, 64 Q8 blocks of 32, one fp32 scale per block, 32
// prompt positions per chunk). Per row and block the dot is a sequential fp32 fma chain per
// position (positions in NEON lanes, 8 x float32x4 = 32 positions), folded with the block
// scale; this is the arithmetic shape an exact CPU row share would run. Timing only.
//   gcc -O2 -mcpu=cortex-a57 -fopenmp -o cpu_tlf_probe tools_cpu_tlf_probe.c
//   ./cpu_tlf_probe <rows> <reps> [threads]
#include <arm_neon.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define K 2048
#define NB (K / 32)
#define P 32

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

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
  printf("CPU_TLF rows=%d threads=%d best %.2f ms  %.2f GMAC/s  (checksum %.3g)\n",
         rows, omp_get_max_threads(), best * 1e3, gmac, cs);
  return 0;
}
