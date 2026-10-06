/* PULSAR-ARM | tools/bench_fma.c
 *
 * Peak 4-lane fp32 fmla rate. Any GEMV - ternary or not - pays at least one
 * add per weight, so this divides straight into a floor per token:
 * measured 7.0 G fma-lanes/s (3.9/cycle at 1.8 GHz, A72 has one vector FMA
 * pipe, so that is the machine peak) => 243 ms/token for a 1.7B model and
 * 571 ms for 4B. The blob streams at 3.93 GB/s in 96 ms, so ternary is
 * COMPUTE-bound; gemma is bandwidth-bound. Do not quote 99 ms for 1.7B.
 *
 * Build: gcc -O2 -march=armv8-a+fp -o bench_fma bench_fma.c && taskset -c 3 ./bench_fma
 */
#include <stdio.h>
#include <arm_neon.h>
#include <time.h>
int main(void) {
    float32x4_t a0 = vdupq_n_f32(1e-6f), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    float32x4_t b = vdupq_n_f32(1.0000001f), c = vdupq_n_f32(0.9999999f);
    const long N = 200000000;                 /* iterations x 8 fma = 1.6e9 fma */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (long i = 0; i < N; i++) {
        a0 = vfmaq_f32(a0, b, c); a1 = vfmaq_f32(a1, b, c);
        a2 = vfmaq_f32(a2, b, c); a3 = vfmaq_f32(a3, b, c);
        a4 = vfmaq_f32(a4, b, c); a5 = vfmaq_f32(a5, b, c);
        a6 = vfmaq_f32(a6, b, c); a7 = vfmaq_f32(a7, b, c);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double s = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
    double fma = (double)N * 8 * 4;
    float r = a0[0] + a1[0] + a2[0] + a3[0] + a4[0] + a5[0] + a6[0] + a7[0];
    printf("fmla: %.1f G fp32 fma-lanes/s  (%.3f s for %.2f G)\n", fma / s / 1e9, s, fma / 1e9);
    printf("=> 1.7e9 weights at 1 fma each: %.0f ms    4e9 (4B): %.0f ms   [%g]\n",
           1.7e9 / (fma / s) * 1e3, 4e9 / (fma / s) * 1e3, r);
    return 0;
}
