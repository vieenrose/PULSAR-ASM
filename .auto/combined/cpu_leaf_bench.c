// .auto/combined/cpu_leaf_bench.c -- can the idle A57s take a share of the ASR prefill GEMV rows?
//
// Prices the CPU half of a CPU/GPU row split of the backbone's batched prefill (32-position
// chunks). Each output (row r, position p) is the GPU leaf's chain, kept bit-for-bit:
//   per 32-code block b:  dot = 0; for jj in 0..31: dot = fma(c[jj], x[p][b*32+jj], dot)
//                         acc = fma(scale[b], dot, acc)        (blocks in order)
// (the GPU additionally combines leaves in a pairwise tree; that is O(K/32) adds per output and
// does not change the cost picture -- it is NOT reproduced here, this bench prices throughput.)
// NEON form: positions are the vector lanes, so each lane runs exactly the scalar chain above with
// vfmaq_f32 (fused, round-to-nearest, no flush: same IEEE result as fmaf). 8 positions per code
// broadcast (two q registers) amortise the int8 -> float conversion. x is staged transposed,
// [K][8], once per position group; the weights stay row-major int8 + one float scale per 32.
// Checks every output against a scalar fmaf reference (must be bitwise equal), then times
// N rows x K x T positions at 1 and 4 threads.
// Build: gcc -O3 -mcpu=cortex-a57 -pthread -o cpu_leaf_bench cpu_leaf_bench.c -lm
#include <arm_neon.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int N = 1024, K = 2048, T = 32;
static int8_t* W; static float* S; static float* X; static float* XT; static float* Y;

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

// XT layout: [T/8 groups][K][8]
static void stage_x(void) {
  for (int g = 0; g < T / 8; g++)
    for (int k = 0; k < K; k++)
      for (int l = 0; l < 8; l++) XT[((size_t)g * K + k) * 8 + l] = X[(size_t)(g * 8 + l) * K + k];
}

static void rows(int r0, int r1) {
  const int nb = K / 32;
  for (int g = 0; g < T / 8; g++) {
    const float* xg = XT + (size_t)g * K * 8;
    for (int r = r0; r < r1; r++) {
      const int8_t* w = W + (size_t)r * K; const float* s = S + (size_t)r * nb;
      float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
      for (int b = 0; b < nb; b++) {
        float32x4_t d0 = vdupq_n_f32(0.f), d1 = vdupq_n_f32(0.f);
        const float* xb = xg + (size_t)b * 32 * 8;
        const int8_t* wb = w + b * 32;
        for (int h = 0; h < 32; h += 8) {
          const int16x8_t w16 = vmovl_s8(vld1_s8(wb + h));
          const float32x4_t clo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16)));
          const float32x4_t chi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16)));
          const float* xh = xb + h * 8;
#define STEP(cv, lane, j) { const float32x4_t c = vdupq_laneq_f32(cv, lane); \
            d0 = vfmaq_f32(d0, c, vld1q_f32(xh + (j) * 8)); d1 = vfmaq_f32(d1, c, vld1q_f32(xh + (j) * 8 + 4)); }
          STEP(clo, 0, 0) STEP(clo, 1, 1) STEP(clo, 2, 2) STEP(clo, 3, 3)
          STEP(chi, 0, 4) STEP(chi, 1, 5) STEP(chi, 2, 6) STEP(chi, 3, 7)
#undef STEP
        }
        const float32x4_t sb = vdupq_n_f32(s[b]);
        a0 = vfmaq_f32(a0, sb, d0); a1 = vfmaq_f32(a1, sb, d1);
      }
      float* y = Y + (size_t)r * T + g * 8;
      vst1q_f32(y, a0); vst1q_f32(y + 4, a1);
    }
  }
}

// Two rows per pass share every x load (register blocking): half the x traffic per MAC. Each
// output's chain is unchanged -- every (row, lane) still runs its own sequential fma order.
static void rows2(int r0, int r1) {
  const int nb = K / 32;
  for (int g = 0; g < T / 8; g++) {
    const float* xg = XT + (size_t)g * K * 8;
    int r = r0;
    for (; r + 1 < r1; r += 2) {
      const int8_t* wA = W + (size_t)r * K; const int8_t* wB = wA + K;
      const float* sA = S + (size_t)r * nb; const float* sB = sA + nb;
      float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f), b0 = vdupq_n_f32(0.f), b1 = vdupq_n_f32(0.f);
      for (int b = 0; b < nb; b++) {
        float32x4_t d0 = vdupq_n_f32(0.f), d1 = vdupq_n_f32(0.f), e0 = vdupq_n_f32(0.f), e1 = vdupq_n_f32(0.f);
        const float* xb = xg + (size_t)b * 32 * 8;
        for (int h = 0; h < 32; h += 8) {
          const int16x8_t vA = vmovl_s8(vld1_s8(wA + b * 32 + h)), vB = vmovl_s8(vld1_s8(wB + b * 32 + h));
          const float32x4_t Alo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vA))), Ahi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vA)));
          const float32x4_t Blo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vB))), Bhi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vB)));
          const float* xh = xb + h * 8;
#define STEP2(ca, cb, lane, j) { const float32x4_t x0 = vld1q_f32(xh + (j) * 8), x1 = vld1q_f32(xh + (j) * 8 + 4); \
            const float32x4_t c = vdupq_laneq_f32(ca, lane), e = vdupq_laneq_f32(cb, lane); \
            d0 = vfmaq_f32(d0, c, x0); d1 = vfmaq_f32(d1, c, x1); e0 = vfmaq_f32(e0, e, x0); e1 = vfmaq_f32(e1, e, x1); }
          STEP2(Alo, Blo, 0, 0) STEP2(Alo, Blo, 1, 1) STEP2(Alo, Blo, 2, 2) STEP2(Alo, Blo, 3, 3)
          STEP2(Ahi, Bhi, 0, 4) STEP2(Ahi, Bhi, 1, 5) STEP2(Ahi, Bhi, 2, 6) STEP2(Ahi, Bhi, 3, 7)
#undef STEP2
        }
        const float32x4_t sa = vdupq_n_f32(sA[b]), sb = vdupq_n_f32(sB[b]);
        a0 = vfmaq_f32(a0, sa, d0); a1 = vfmaq_f32(a1, sa, d1);
        b0 = vfmaq_f32(b0, sb, e0); b1 = vfmaq_f32(b1, sb, e1);
      }
      float* y = Y + (size_t)r * T + g * 8;
      vst1q_f32(y, a0); vst1q_f32(y + 4, a1); vst1q_f32(y + T, b0); vst1q_f32(y + T + 4, b1);
    }
  }
  if ((r1 - r0) & 1) rows(r1 - 1, r1);
}
static int g_two = 0;
typedef struct { int r0, r1; } Job;
static void* worker(void* p) { Job* j = (Job*)p; if (g_two) rows2(j->r0, j->r1); else rows(j->r0, j->r1); return NULL; }
static void run(int nth) {
  pthread_t th[8]; Job jb[8];
  for (int i = 0; i < nth; i++) { jb[i].r0 = N * i / nth; jb[i].r1 = N * (i + 1) / nth; }
  for (int i = 1; i < nth; i++) pthread_create(&th[i], NULL, worker, &jb[i]);
  worker(&jb[0]);
  for (int i = 1; i < nth; i++) pthread_join(th[i], NULL);
}

int main(int argc, char** argv) {
  if (argc > 1) N = atoi(argv[1]);
  if (argc > 2) K = atoi(argv[2]);
  if (argc > 3) T = atoi(argv[3]);
  if (argc > 4) g_two = atoi(argv[4]);
  const int nb = K / 32;
  W = malloc((size_t)N * K); S = malloc((size_t)N * nb * 4); X = malloc((size_t)T * K * 4);
  XT = malloc((size_t)T * K * 4); Y = malloc((size_t)N * T * 4);
  srand(7);
  for (size_t i = 0; i < (size_t)N * K; i++) W[i] = (int8_t)(rand() % 255 - 127);
  for (size_t i = 0; i < (size_t)N * nb; i++) S[i] = ldexpf((float)(rand() % 1000 + 1), -17);
  for (size_t i = 0; i < (size_t)T * K; i++) X[i] = ((float)rand() / RAND_MAX - 0.5f) * 4.f;
  stage_x();
  run(4);
  long bad = 0;
  for (int r = 0; r < N; r++)
    for (int p = 0; p < T; p++) {
      float acc = 0.f;
      for (int b = 0; b < nb; b++) {
        float dot = 0.f;
        for (int jj = 0; jj < 32; jj++) dot = fmaf((float)W[(size_t)r * K + b * 32 + jj], X[(size_t)p * K + b * 32 + jj], dot);
        acc = fmaf(S[(size_t)r * nb + b], dot, acc);
      }
      uint32_t u, v; float y = Y[(size_t)r * T + p]; memcpy(&u, &acc, 4); memcpy(&v, &y, 4);
      if (u != v) bad++;
    }
  printf("rows/pass=%d N=%d K=%d T=%d  bitwise vs scalar fmaf chain: %ld of %ld differ\n", g_two ? 2 : 1, N, K, T, bad, (long)N * T);
  for (int nth = 1; nth <= 4; nth *= 4) {
    double best = 1e9;
    for (int it = 0; it < 5; it++) { double t0 = now(); run(nth); double t = now() - t0; if (t < best) best = t; }
    const double mac = (double)N * K * T;
    printf("  threads=%d  %.2f ms  %.2f GMAC/s\n", nth, best * 1e3, mac / best * 1e-9);
  }
  return bad ? 1 : 0;
}
