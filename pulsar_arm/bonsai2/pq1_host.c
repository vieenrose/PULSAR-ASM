/* pq1_host.c — Q1_0 split repack + fork-shaped host reference (no CUDA). */
#include "pq1_wrap.h"
#include "pq1_expf_neon.h"
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <pthread.h>
#include <sched.h>

float pq1_h2f(uint16_t h) {
  uint32_t s = (uint32_t)(h & 0x8000u) << 16;
  uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu, o;
  if (e == 0) {
    if (!m) return 0.0f;
    float f = (float)m * 5.9604645e-8f;
    memcpy(&o, &f, 4); o |= s;
  } else if (e == 31) o = s | 0x7f800000u | (m << 13);
  else o = s | ((e + 112u) << 23) | (m << 13);
  float r; memcpy(&r, &o, 4); return r;
}

void pq1_split_repack(const uint8_t *q1, int rows, int K, uint32_t *w_bits,
                      uint16_t *w_scales_h) {
  int nb = K / 128;                 /* 18B blocks per row */
#ifdef _OPENMP
  /* Pure data movement: each row writes a disjoint slice, so the loop is a
   * partition of the output and the result is bit-identical to the serial loop,
   * at any thread count. The repack is the dominant part of the engine's startup
   * (it walks all 7.57G weights), so parallelising it is a pure startup win. */
#pragma omp parallel for schedule(static)
#endif
  for (int r = 0; r < rows; r++) {
    const uint8_t *row = q1 + (size_t)r * nb * 18;
    uint32_t *wb = w_bits + (size_t)r * (K / 32);
    uint16_t *sc = w_scales_h + (size_t)r * nb;
    for (int b = 0; b < nb; b++) {
      memcpy(wb + b * 4, row + b * 18 + 2, 16); /* 128 bits, LSB-first */
      memcpy(sc + b, row + b * 18, 2);          /* fp16 scale, verbatim */
    }
  }
}

void pq1_gemv_cpu(int rows, int K, const uint32_t *w_bits,
                  const uint16_t *w_scales_h, const uint8_t *xq, float *y) {
  int nb = K / 128;
  for (int r = 0; r < rows; r++) {
    const uint32_t *wb = w_bits + (size_t)r * (K / 32);
    float acc = 0.0f;
    for (int b = 0; b < nb; b++) {
      const uint8_t *qb = xq + (size_t)(b >> 1) * 292;
      float db; memcpy(&db, qb, 4);
      const int8_t *qs = (const int8_t *)(qb + 4 + (b & 1) * 128);
      float ws = pq1_h2f(w_scales_h[(size_t)r * nb + b]);
      const uint32_t *w4 = wb + b * 4;
      int isum = 0;
      for (int i = 0; i < 128; i++) {
        int bit = (w4[i >> 5] >> (i & 31)) & 1;
        isum += (bit ? 1 : -1) * (int)qs[i];
      }
      acc += (ws * db) * (float)isum;
    }
    y[r] = acc;
  }
}

/* T-MAC bit-plane repack of the Q8_K activations (run #41 experiment).
 *
 * For every 32 activations (one 32-bit code word = 32 weights) build the 8
 * bit-plane words: plane word p has bit a = bit p of the unsigned biased byte
 * x' = (uint8)(x + 128) of activation (32c + a). Then for a code word W
 *   Pu = sum_p 2^p * popc(plane[c][p] & W)
 * is EXACTLY the sum of x' over the selected weights: each activation
 * contributes its own byte value once (bit a of plane p carries bit p of that
 * byte, weighted by 2^p). Integer, so order-independent and bit-exact.
 *
 * Size: 8 words per 32 activations = 1 byte per activation - byte-identical
 * traffic to the int8 array it replaces, and the block stride (73 words =
 * 292B) mirrors Q8_K so the kernel's staging code and addresses are unchanged
 * (word 0 = the fp32 scale d, words 1..64 = 64 plane words). */
void pq1_build_planes(const uint8_t *xq, int K, uint32_t *planes) {
  int nb = K >> 8;
  for (int b = 0; b < nb; b++) {
    const uint8_t *src = xq + (size_t)b * 292;
    uint32_t *dst = planes + (size_t)b * 73;
    memcpy(dst, src, 4);            /* fp32 scale d, verbatim (word 0) */
    memset(dst + 1, 0, 64 * 4);
    const uint8_t *act = src + 4;   /* 256 int8 activations */
    for (int i = 0; i < 256; i++) {
      unsigned v = (unsigned)act[i] ^ 0x80u;   /* x' = x + 128, unsigned */
      uint32_t *pw = dst + 1 + (i >> 5) * 8;
      uint32_t m = 1u << (i & 31);
      if (v & 1u) pw[0] |= m;
      if (v & 2u) pw[1] |= m;
      if (v & 4u) pw[2] |= m;
      if (v & 8u) pw[3] |= m;
      if (v & 16u) pw[4] |= m;
      if (v & 32u) pw[5] |= m;
      if (v & 64u) pw[6] |= m;
      if (v & 128u) pw[7] |= m;
    }
  }
}

/* Au[g] = sum of the unsigned biased bytes x' over the group's 128 activations
 * (= sum(int8) + 16384). Row-independent, so the host computes it once per
 * token; the kernel's isum = 2*Pu - Au + 16384 - 256*pc is unchanged. */
void pq1_build_au(const uint8_t *xq, int K, int *au) {
  int nb = K >> 7;
  for (int b = 0; b < nb; b++) {
    const int8_t *qs = (const int8_t *)(xq + (size_t)(b >> 1) * 292 + 4 +
                                         (b & 1) * 128);
    int s = 0;
    for (int i = 0; i < 128; i++) s += (int)qs[i] + 128;
    au[b] = s;
  }
}

#ifndef PQ1_REAL_CUDA
int pq1_gemv_sm53_launch(const unsigned char *a, const unsigned short *b,
                         const unsigned char *c, float *d, int e, int f,
                         void *g) {
  (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g;
  return -99;
}
int pq1_gemv_tm_launch(const unsigned char *a, const unsigned short *b,
                       const unsigned int *c, const int *d, float *e, int f,
                       int g, void *h) {
  (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; (void)h;
  return -99;
}
int pq1_tm_prep_launch(const unsigned char *a, unsigned int *b, int *c, int d,
                       void *e) {
  (void)a; (void)b; (void)c; (void)d; (void)e;
  return -99;
}
#endif

/* silu(g)*u in place: LU[i] = (g * (1/(1+expf(-g)))) * LU[i], four lanes at
 * a time. Lives here because this file is built with -ffp-contract=off: in
 * fwd8.c's default-contract build gcc fused one of the header's double ops
 * and 2 of 4.28e9 inputs came out 1 ulp off (exhaustive test). */
void pq1_silu_mul(float *lu, const float *lg, int n) {
    int i = 0;
    /* two independent 4-lane chains per iteration (#74): the expf core is a
     * dependent chain of ~20 double ops plus a divide, so one chain at a time
     * left the A57's second FP pipe idle. Per-lane arithmetic unchanged. */
    const int n8 = n & ~7;
    /* Serial on purpose: OpenMP over 4 cores read 9.4 (passive) / 10.5 ms/tok
     * (active) against 7.5 serial; 36 fork/joins per token cost more than the
     * work they split (#74). */
    for (int j = 0; j < n8; j += 8) {
        const float32x4_t g0 = vld1q_f32(lg + j), u0 = vld1q_f32(lu + j);
        const float32x4_t g1 = vld1q_f32(lg + j + 4), u1 = vld1q_f32(lu + j + 4);
        const float32x4_t e0 = pq1_expf4(vnegq_f32(g0));
        const float32x4_t e1 = pq1_expf4(vnegq_f32(g1));
        const float32x4_t sg0 = vdivq_f32(vdupq_n_f32(1.0f), vaddq_f32(vdupq_n_f32(1.0f), e0));
        const float32x4_t sg1 = vdivq_f32(vdupq_n_f32(1.0f), vaddq_f32(vdupq_n_f32(1.0f), e1));
        vst1q_f32(lu + j, vmulq_f32(vmulq_f32(g0, sg0), u0));
        vst1q_f32(lu + j + 4, vmulq_f32(vmulq_f32(g1, sg1), u1));
    }
    i = n8;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t g = vld1q_f32(lg + i), u = vld1q_f32(lu + i);
        const float32x4_t e = pq1_expf4(vnegq_f32(g));
        const float32x4_t sg = vdivq_f32(vdupq_n_f32(1.0f), vaddq_f32(vdupq_n_f32(1.0f), e));
        vst1q_f32(lu + i, vmulq_f32(vmulq_f32(g, sg), u));
    }
    for (; i < n; i++) lu[i] = (lg[i] * (1.0f / (1.0f + expf(-lg[i])))) * lu[i];
}

/* ---- run #75: hybrid CPU rows ------------------------------------------ */
void pq1_hyb_gsum(const uint8_t *xq, int K, int *gsum) {
  int nb = K >> 7;
  for (int b = 0; b < nb; b++) {
    const int8_t *qs = (const int8_t *)(xq + (size_t)(b >> 1) * 292 + 4 + (b & 1) * 128);
    int32x4_t a = vdupq_n_s32(0);
    for (int i = 0; i < 128; i += 16) a = vpadalq_s16(a, vpaddlq_s8(vld1q_s8(qs + i)));
    gsum[b] = vaddvq_s32(a);
  }
}

void pq1_hyb_rows(const uint32_t *codes, const float *scales_f, int rows, int K,
                  const uint8_t *xq, const int *gsum, float *y) {
  const int nb = K >> 7;
  /* lane l of (bytes {2s x8, 2s+1 x8} TST bitm) = bit (l&7) of byte 2s+(l>>3)
   * = weight 16s+l of the group (LSB-first words, little-endian bytes). */
  static const uint8_t BITM[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
  static const uint8_t IDX[8][16] = {
    {0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1}, {2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3},
    {4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5}, {6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7},
    {8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9}, {10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11},
    {12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13}, {14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15}};
  const uint8x16_t bitm = vld1q_u8(BITM);
  const uint8x16_t i0 = vld1q_u8(IDX[0]), i1 = vld1q_u8(IDX[1]), i2 = vld1q_u8(IDX[2]), i3 = vld1q_u8(IDX[3]);
  const uint8x16_t i4 = vld1q_u8(IDX[4]), i5 = vld1q_u8(IDX[5]), i6 = vld1q_u8(IDX[6]), i7 = vld1q_u8(IDX[7]);
  const size_t rb = (size_t)K / 8;   /* bytes per row of codes */
  int r = 0;
  /* four rows at a time: one activation load feeds four independent
   * accumulate chains (the A57 cannot overlap one row's 8-step padal chain
   * with its own scalar epilogue; four rows give it the ILP) */
  for (; r + 4 <= rows; r += 4) {
    const uint8_t *w0 = (const uint8_t *)codes + (size_t)r * rb;
    const uint8_t *w1 = w0 + rb, *w2 = w1 + rb, *w3 = w2 + rb;
    const float *ws0 = scales_f + (size_t)r * nb, *ws1 = ws0 + nb, *ws2 = ws1 + nb, *ws3 = ws2 + nb;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int b = 0; b < nb; b++) {
      const uint8_t *qb = xq + (size_t)(b >> 1) * 292;
      float db; memcpy(&db, qb, 4);
      const int8_t *qs = (const int8_t *)(qb + 4 + (b & 1) * 128);
      const uint8x16_t c0 = vld1q_u8(w0 + b * 16), c1 = vld1q_u8(w1 + b * 16);
      const uint8x16_t c2 = vld1q_u8(w2 + b * 16), c3 = vld1q_u8(w3 + b * 16);
      int16x8_t s0 = vdupq_n_s16(0), s1 = vdupq_n_s16(0), s2 = vdupq_n_s16(0), s3 = vdupq_n_s16(0);
#define HYB4(ix, off) { const int8x16_t x = vld1q_s8(qs + (off)); \
      s0 = vpadalq_s8(s0, vandq_s8(x, vreinterpretq_s8_u8(vtstq_u8(vqtbl1q_u8(c0, ix), bitm)))); \
      s1 = vpadalq_s8(s1, vandq_s8(x, vreinterpretq_s8_u8(vtstq_u8(vqtbl1q_u8(c1, ix), bitm)))); \
      s2 = vpadalq_s8(s2, vandq_s8(x, vreinterpretq_s8_u8(vtstq_u8(vqtbl1q_u8(c2, ix), bitm)))); \
      s3 = vpadalq_s8(s3, vandq_s8(x, vreinterpretq_s8_u8(vtstq_u8(vqtbl1q_u8(c3, ix), bitm)))); }
      HYB4(i0, 0); HYB4(i1, 16); HYB4(i2, 32); HYB4(i3, 48);
      HYB4(i4, 64); HYB4(i5, 80); HYB4(i6, 96); HYB4(i7, 112);
#undef HYB4
      const int g = gsum[b];
      a0 += (ws0[b] * db) * (float)(2 * vaddlvq_s16(s0) - g);
      a1 += (ws1[b] * db) * (float)(2 * vaddlvq_s16(s1) - g);
      a2 += (ws2[b] * db) * (float)(2 * vaddlvq_s16(s2) - g);
      a3 += (ws3[b] * db) * (float)(2 * vaddlvq_s16(s3) - g);
    }
    y[r] = a0; y[r + 1] = a1; y[r + 2] = a2; y[r + 3] = a3;
  }
  for (; r < rows; r++) {
    const uint8_t *w0 = (const uint8_t *)codes + (size_t)r * rb;
    const float *ws0 = scales_f + (size_t)r * nb;
    float a0 = 0.0f;
    for (int b = 0; b < nb; b++) {
      const uint8_t *qb = xq + (size_t)(b >> 1) * 292;
      float db; memcpy(&db, qb, 4);
      const int8_t *qs = (const int8_t *)(qb + 4 + (b & 1) * 128);
      const uint8x16_t c0 = vld1q_u8(w0 + b * 16);
      int16x8_t s0 = vdupq_n_s16(0);
#define HYB1(ix, off) s0 = vpadalq_s8(s0, vandq_s8(vld1q_s8(qs + (off)), \
          vreinterpretq_s8_u8(vtstq_u8(vqtbl1q_u8(c0, ix), bitm))))
      HYB1(i0, 0); HYB1(i1, 16); HYB1(i2, 32); HYB1(i3, 48);
      HYB1(i4, 64); HYB1(i5, 80); HYB1(i6, 96); HYB1(i7, 112);
#undef HYB1
      a0 += (ws0[b] * db) * (float)(2 * vaddlvq_s16(s0) - gsum[b]);
    }
    y[r] = a0;
  }
}

/* spinning pool: workers wait on a generation counter, run their row slice of
 * the current job and count themselves done. They spin (yield) between jobs,
 * i.e. during the GPU phases too; PQ1_HYB_THREADS=1 keeps everything on the
 * calling thread. */
typedef struct {
  const uint32_t *codes; const float *scales_f; const uint8_t *xq; const int *gsum;
  float *y; int rows, K;
} pq1_hyb_job;
static pq1_hyb_job HJ;
static atomic_int HGEN = 0, HDONE = 0;
static int HNT = 0;
static int HMAXT = 3;   /* 3 workers; the 4th core launches and spins in CUDA */
static void pq1_hyb_slice(int t, int nt) {
  int r0 = (int)((long)HJ.rows * t / nt), r1 = (int)((long)HJ.rows * (t + 1) / nt);
  if (r1 > r0)
    pq1_hyb_rows(HJ.codes + (size_t)r0 * (HJ.K / 32), HJ.scales_f + (size_t)r0 * (HJ.K / 128),
                 r1 - r0, HJ.K, HJ.xq, HJ.gsum, HJ.y + r0);
}
static void *pq1_hyb_worker(void *arg) {
  int t = (int)(intptr_t)arg, seen = 0;
  for (;;) {
    while (atomic_load_explicit(&HGEN, memory_order_acquire) == seen) __asm__ volatile("yield");
    seen = atomic_load_explicit(&HGEN, memory_order_acquire);
    pq1_hyb_slice(t, HNT);
    atomic_fetch_add_explicit(&HDONE, 1, memory_order_release);
  }
  return NULL;
}
void pq1_hyb_threads(int n) { HMAXT = n < 1 ? 1 : n; }
/* Run #75: the slice runs on HNT worker threads only; the caller's thread is
 * the one that launches the CUDA graph, and on this platform (CUDA 10.2,
 * Tegra, spin scheduling) cudaGraphLaunch does not return until the GPU is
 * done (hsync 7.8 us per call with 120 ms of GPU work), so the caller cannot
 * compute anything while the GPU runs. Start the workers, launch, then wait. */
void pq1_hyb_start(const uint32_t *codes, const float *scales_f, int rows, int K,
                   const uint8_t *xq, float *y) {
  static int gsum[4096];
  if (K / 128 > 4096) { rows = 0; }   /* K > 512K: not a shape this engine has */
  pq1_hyb_gsum(xq, K, gsum);
  if (!HNT) {
    const char *e = getenv("PQ1_HYB_THREADS");
    HNT = e ? atoi(e) : HMAXT;
    if (HNT < 1) HNT = 1;
    for (int t = 0; t < HNT; t++) {
      pthread_t th;
      if (pthread_create(&th, NULL, pq1_hyb_worker, (void *)(intptr_t)t) != 0) { HNT = t; break; }
      pthread_detach(th);
    }
    if (!HNT) HNT = -1;   /* no workers: the caller computes in pq1_hyb_wait */
  }
  HJ.codes = codes; HJ.scales_f = scales_f; HJ.xq = xq; HJ.gsum = gsum;
  HJ.y = y; HJ.rows = rows; HJ.K = K;
  atomic_store_explicit(&HDONE, 0, memory_order_relaxed);
  atomic_fetch_add_explicit(&HGEN, 1, memory_order_release);
}
void pq1_hyb_wait(void) {
  if (HNT < 0) { pq1_hyb_slice(0, 1); return; }
  while (atomic_load_explicit(&HDONE, memory_order_acquire) < HNT) __asm__ volatile("yield");
}
void pq1_hyb_run(const uint32_t *codes, const float *scales_f, int rows, int K,
                 const uint8_t *xq, float *y) {
  pq1_hyb_start(codes, scales_f, rows, K, xq, y);
  pq1_hyb_wait();
}
