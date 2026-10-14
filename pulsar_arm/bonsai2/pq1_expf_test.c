/* exhaustive check of pq1_expf4 against libm expf on the board */
#include "pq1_expf_neon.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
int main(void) {
    uint64_t mism = 0, tested = 0, shown = 0;
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint64_t base = 0; base < (1ULL << 32); base += 4) {
        uint32_t u[4] = { (uint32_t)base, (uint32_t)base + 1, (uint32_t)base + 2, (uint32_t)base + 3 };
        float x[4]; memcpy(x, u, 16);
        float y[4]; vst1q_f32(y, pq1_expf4(vld1q_f32(x)));
        for (int i = 0; i < 4; i++) {
            float ref = expf(x[i]);
            uint32_t a, b; memcpy(&a, &y[i], 4); memcpy(&b, &ref, 4);
            if (ref != ref) continue;
            tested++;
            if (a != b) { mism++; if (shown++ < 10) printf("MISMATCH x=%a (0x%08x) got=%a ref=%a\n", x[i], u[i], y[i], ref); }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("tested=%llu mismatches=%llu secs=%.1f\n", (unsigned long long)tested, (unsigned long long)mism,
           (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec));
    /* throughput: 12288 silu-like expf per call, 1000 calls */
    static float buf[12288]; for (int i = 0; i < 12288; i++) buf[i] = (i % 97) * 0.37f - 17.f;
    float acc = 0; clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int k = 0; k < 1000; k++) for (int i = 0; i < 12288; i += 4) { float o[4]; vst1q_f32(o, pq1_expf4(vld1q_f32(buf + i))); acc += o[0]; }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("neon  12288 expf: %.3f ms/call (acc %g)\n", ((t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec)), acc);
    acc = 0; clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int k = 0; k < 1000; k++) for (int i = 0; i < 12288; i++) acc += expf(buf[i]);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("libm  12288 expf: %.3f ms/call (acc %g)\n", ((t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec)), acc);
    return 0;
}
