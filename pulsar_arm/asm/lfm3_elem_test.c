/* lfm3_elem_test.c -- parity gate + benchmark for lfm3_elem.S (LFM3-Audio
   elementwise kernels, Jetson Nano / Cortex-A57).
   Gate: bit-exact equality against the C reference below, which MIRRORS the
   kernel's documented reduction order (see lfm3_elem.S) rather than computing
   "a" sum. Tolerance checks are what let 5fd7e79's truncated K=7680 kernel
   print YES, so nothing here compares within an epsilon.
   Build (on the Nano):
     as -o lfm3_elem.o lfm3_elem.S
     gcc -O2 -ffp-contract=off -o lfm3_elem_test lfm3_elem_test.c lfm3_elem.o -lm
   -ffp-contract=off is not optional: a fused a*b+c in the reference is a
   different real number than the kernel's mul-then-add.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <arm_acle.h>

void lfm3_rmsnorm_bf16(const uint16_t* x, const float* w, uint16_t* y, int H, float eps);
void lfm3_mul_bf16(const uint16_t* a, const uint16_t* b, uint16_t* y, int n);
void lfm3_add_bf16(const uint16_t* a, const uint16_t* b, uint16_t* y, int n);

/* ------------------------------------------------ bf16, RN-even ---------- */
static uint16_t f2bf(float f) {
  uint32_t u; memcpy(&u, &f, 4);
  if (((u >> 23) & 0xffu) == 0xffu) return (uint16_t)(u >> 16);   /* inf/nan */
  const uint32_t r = u + 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(r >> 16);
}
static float bf2f(uint16_t b) { uint32_t u = ((uint32_t)b) << 16; float f; memcpy(&f, &u, 4); return f; }

/* mirrors lfm3_elem.S exactly: 8 accumulators, chunks ascending, then
   ((a0+a1)+(a2+a3)) + ((a4+a5)+(a6+a7)), then 1/sqrtf, then the two rounds */
static void ref_rmsnorm(const uint16_t* x, const float* w, uint16_t* y, int H, float eps) {
  float A[4] = {0.f, 0.f, 0.f, 0.f}, B[4] = {0.f, 0.f, 0.f, 0.f};
  for (int c = 0; c < H / 8; c++)
    for (int j = 0; j < 4; j++) {
      const float v0 = bf2f(x[c * 8 + j]);
      const float v1 = bf2f(x[c * 8 + 4 + j]);
      A[j] = A[j] + v0 * v0;
      B[j] = B[j] + v1 * v1;
    }
  const float tA = (A[0] + A[1]) + (A[2] + A[3]);
  const float tB = (B[0] + B[1]) + (B[2] + B[3]);
  const float total = tA + tB;
  const float r = 1.0f / sqrtf(total / (float)H + eps);
  for (int i = 0; i < H; i++) {
    const float xr = bf2f(f2bf(bf2f(x[i]) * r));      /* .to(input_dtype) */
    y[i] = f2bf(w[i] * xr);                            /* weight * that */
  }
}
static void ref_mul(const uint16_t* a, const uint16_t* b, uint16_t* y, int n) {
  for (int i = 0; i < n; i++) y[i] = f2bf(bf2f(a[i]) * bf2f(b[i]));
}
static void ref_add(const uint16_t* a, const uint16_t* b, uint16_t* y, int n) {
  for (int i = 0; i < n; i++) y[i] = f2bf(bf2f(a[i]) + bf2f(b[i]));
}

static double ms_since(const struct timespec* a) {
  struct timespec b; clock_gettime(CLOCK_MONOTONIC, &b);
  return (b.tv_sec - a->tv_sec) * 1e3 + (b.tv_nsec - a->tv_nsec) / 1e6;
}
static uint32_t rng = 12345u;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static uint16_t rand_bf(void) { return f2bf(((float)(rnd() % 20000) - 10000.f) / 3000.f); }

/* AAPCS64: d8-d15 are callee-saved. Poison them, call, verify. A clobbered
   d8 in q8_gemv_neon.S caused a 714 GB OpenMP allocation with zero symptom
   under a direct-call gate, so this check is part of the gate. */
static int check_callee_saved(void (*fn)(void), const char* what) {
  register double d8 asm("d8"), d9 asm("d9"), d10 asm("d10"), d11 asm("d11");
  register double d12 asm("d12"), d13 asm("d13"), d14 asm("d14"), d15 asm("d15");
  d8 = 1.5; d9 = 2.5; d10 = 3.5; d11 = 4.5; d12 = 5.5; d13 = 6.5; d14 = 7.5; d15 = 8.5;
  asm volatile("" ::: "d8","d9","d10","d11","d12","d13","d14","d15");
  fn();
  int bad = 0;
  if (d8 != 1.5 || d9 != 2.5 || d10 != 3.5 || d11 != 4.5) bad = 1;
  if (d12 != 5.5 || d13 != 6.5 || d14 != 7.5 || d15 != 8.5) bad = 1;
  if (bad) printf("REGUSTER CLOBBER: %s modified d8-d15\n", what);
  return bad;
}

#define H 2048
#define INTER 7680
#define MAXH 4096                  /* the widest H the parity list tests */
static uint16_t *x, *a, *b, *y, *yr, *w16;
static float* w;

static void call_rms(void) { lfm3_rmsnorm_bf16(x, w, y, H, 1e-5f); }
static void call_mul(void) { lfm3_mul_bf16(a, b, y, INTER); }
static void call_add(void) { lfm3_add_bf16(a, b, y, INTER); }

int main(void) {
  /* sized for the widest test, not for H: the parity list runs H=4096 and
     x/w were first allocated for 2048, which overran the heap and segfaulted
     the whole test -- the symptom pointed at the asm, the cause was here. */
  x = malloc(MAXH * 2); a = malloc(INTER * 2); b = malloc(INTER * 2);
  y = malloc(INTER * 2); yr = malloc(INTER * 2); w16 = malloc(MAXH * 2);
  w = malloc(MAXH * 4);
  int fails = 0;

  /* ---- rmsnorm: bit-exact at every H that is a multiple of 8 ---------- */
  printf("rmsnorm (H multiple of 8, eps=1e-5):\n");
  const int HS[] = {8, 16, 24, 32, 64, 128, 512, 1024, 2048, 4096};
  for (size_t hi = 0; hi < sizeof HS / sizeof *HS; hi++) {
    const int hh = HS[hi];
    for (int i = 0; i < hh; i++) { x[i] = rand_bf(); w[i] = (float)((int)(rnd() % 2000) - 1000) / 1000.f; }
    memset(y, 0, hh * 2); memset(yr, 0, hh * 2);
    lfm3_rmsnorm_bf16(x, w, y, hh, 1e-5f);
    ref_rmsnorm(x, w, yr, hh, 1e-5f);
    long bad = 0; int first = -1;
    for (int i = 0; i < hh; i++) if (y[i] != yr[i]) { if (first < 0) first = i; bad++; }
    printf("  H=%5d  %s", hh, bad ? "FAIL" : "ok");
    if (bad) { printf("  (%ld/%d differ, first i=%d: asm %04x ref %04x)", bad, hh, first, y[first], yr[first]); fails++; }
    printf("\n");
  }
  /* every model norm runs at H=2048; check several seeds so the gate is not
     one lucky draw, and eps values in use (norm_eps 1e-5) */
  long total_bad = 0;
  for (int seed = 0; seed < 40; seed++) {
    rng = 1u + seed * 7919u;
    for (int i = 0; i < H; i++) { x[i] = rand_bf(); w[i] = (float)((int)(rnd() % 2000) - 1000) / 1000.f; }
    lfm3_rmsnorm_bf16(x, w, y, H, 1e-5f);
    ref_rmsnorm(x, w, yr, H, 1e-5f);
    for (int i = 0; i < H; i++) if (y[i] != yr[i]) total_bad++;
  }
  printf("  40 seeds at H=2048: %s (%ld mismatching elements)\n",
         total_bad ? "FAIL" : "bit-exact", total_bad);
  if (total_bad) fails++;

  /* ---- mul / add ------------------------------------------------------ */
  printf("mul_bf16 / add_bf16:\n");
  for (int t = 0; t < 2; t++) {
    const int n = t ? INTER : H;
    for (int i = 0; i < n; i++) { a[i] = rand_bf(); b[i] = rand_bf(); }
    /* separate output buffers: the first version ran mul into y, then add
       into y, then printed y for the mul failure index -- so the mul
       diagnostic showed two identical bit patterns and looked like a false
       positive while the real mul failures were being overwritten. */
    lfm3_mul_bf16(a, b, y, n); ref_mul(a, b, yr, n);
    long m1 = 0; int f1 = -1;
    for (int i = 0; i < n; i++) if (y[i] != yr[i]) { if (f1 < 0) f1 = i; m1++; }
    lfm3_add_bf16(a, b, y, n); ref_add(a, b, yr, n);
    long m2 = 0; int f2 = -1;
    for (int i = 0; i < n; i++) if (y[i] != yr[i]) { if (f2 < 0) f2 = i; m2++; }
    printf("  n=%5d  mul %s (%ld)  add %s (%ld)\n", n,
           m1 ? "FAIL" : "bit-exact", m1, m2 ? "FAIL" : "bit-exact", m2);
    /* print BOTH bit patterns: printing the ref's float value made asm and
       ref look identical (4.5625 vs 0x4092) and hid which bits disagree. */
    if (m1) printf("       mul i=%d a=%04x b=%04x  asm=%04x(%.7g) ref=%04x(%.7g) prod=%.9g\n",
                   f1, a[f1], b[f1], y[f1], bf2f(y[f1]), yr[f1], bf2f(yr[f1]),
                   (double)(bf2f(a[f1]) * bf2f(b[f1])));
    if (m2) printf("       add i=%d a=%04x b=%04x  asm=%04x(%.7g) ref=%04x(%.7g) sum=%.9g\n",
                   f2, a[f2], b[f2], y[f2], bf2f(y[f2]), yr[f2], bf2f(yr[f2]),
                   (double)(bf2f(a[f2]) + bf2f(b[f2])));
    fails += (m1 != 0) + (m2 != 0);
  }

  /* ---- callee-saved register contract -------------------------------- */
  /* Only valid in a non-optimizing build. The checker holds its poisoned
     values in explicit-register locals across an INDIRECT call and lists
     d8-d15 as clobbered; at -O2 gcc reloads them from its own saved copies,
     so all three kernels get reported as clobberers. The -O0 build (the
     authoritative run for this check) says ok.
     The integer-register half of the contract is NOT visible to this check
     at any level -- that is how the x26 clobber in lfm3_rmsnorm_bf16 got
     through (it surfaced as a SIGSEGV in main at -O2, not as a gate line),
     so the -O2 build is run too, as the real integer-register detector. */
#if defined(__OPTIMIZE__)
  printf("AAPCS64 d8-d15 contract:\n  skipped in the optimized build (needs -O0); "
         "run lfm3_elem_t0 for this check\n");
#else
  printf("AAPCS64 d8-d15 contract:\n");
  int c1 = check_callee_saved(call_rms, "lfm3_rmsnorm_bf16");
  int c2 = check_callee_saved(call_mul, "lfm3_mul_bf16");
  int c3 = check_callee_saved(call_add, "lfm3_add_bf16");
  printf("  %s\n", (c1 | c2 | c3) ? "FAIL" : "ok");
  fails += c1 + c2 + c3;
#endif

  /* ---- benchmark ------------------------------------------------------ */
  printf("benchmark (1 A57 core):\n");
  for (int i = 0; i < H; i++) { x[i] = rand_bf(); w[i] = (float)((int)(rnd() % 2000) - 1000) / 1000.f; }
  for (int i = 0; i < INTER; i++) { a[i] = rand_bf(); b[i] = rand_bf(); }
  const int REPS = 20000;
  struct timespec t0; double best;
  /* 96 rmsnorms per token (4 x 24 layers), 26 mul, 48 add */
  best = 1e30;
  for (int trial = 0; trial < 3; trial++) {
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < REPS; r++) lfm3_rmsnorm_bf16(x, w, y, H, 1e-5f);
    double ms = ms_since(&t0) / REPS; if (ms < best) best = ms;
  }
  printf("  rmsnorm H=2048      %7.3f us  x96/token = %6.2f ms/token\n", best * 1e3, best * 96);
  double bmul = 1e30, badd = 1e30;
  for (int trial = 0; trial < 3; trial++) {
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < REPS; r++) lfm3_mul_bf16(a, b, y, INTER);
    double ms = ms_since(&t0) / REPS; if (ms < bmul) bmul = ms;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < REPS; r++) lfm3_add_bf16(a, b, y, INTER);
    ms = ms_since(&t0) / REPS; if (ms < badd) badd = ms;
  }
  printf("  mul   n=7680        %7.3f us  x26/token = %6.2f ms/token\n", bmul * 1e3, bmul * 26);
  printf("  add   n=7680        %7.3f us  x48/token = %6.2f ms/token\n", badd * 1e3, badd * 48);
  printf("  elementwise total (these 3 ops) = %.2f ms/token\n", best * 96 + bmul * 26 + badd * 48);

  printf("%s\n", fails ? "LFM3_ELEM_FAIL" : "LFM3_ELEM_OK");
  return fails ? 1 : 0;
}
