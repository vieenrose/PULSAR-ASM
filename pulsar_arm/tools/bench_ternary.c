/* PULSAR-ARM | tools/bench_ternary.c
 *
 * Throughput probe for ternary_gemv_b3 on REAL blob bytes. The point is iteration
 * speed: a full token on the blob costs ~7 s, this takes ~20 ms per iteration and
 * predicts the token time to within 3% (3.98 ns/weight projected 6834 ms vs 7079
 * ms measured, 2026-10-06), because the ternary GEMV is ~97% of the forward.
 *
 * Build on the Pi:
 *   gcc -O2 -o bench_ternary bench_ternary.c ../asm/ternary_gemv.S
 *   taskset -c 3 ./bench_ternary ~/bonsai_b3.bin 7
 *
 * It reads one contiguous 2048x2048 q_proj block, so the access pattern is the
 * one a real forward produces. ns/weight is the number to compare across runs.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
void ternary_gemv_b3(int rows, int cols, const uint8_t *w, const float *x, float *y);
#define ROWS 2048
#define COLS 2048
static float X[COLS], Y[ROWS];
int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/home/luigi/bonsai_b3.bin";
    int iters = argc > 2 ? atoi(argv[2]) : 5;
    int fd = open(path, O_RDONLY); struct stat st; fstat(fd, &st);
    uint8_t *m = mmap(0, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    uint64_t hl; memcpy(&hl, m, 8);
    char *json = strndup((char *)m + 8, hl);
    /* a real q_proj: contiguous, 2048x2048, hot in a real forward */
    char *p = strstr(json, "\"model.layers.5.self_attn.q_proj.weight\"");
    char *q = strchr(strstr(p, "\"data_offsets\""), '[');
    long o0, o1; sscanf(q + 1, " %ld , %ld", &o0, &o1);
    const uint8_t *W = m + 8 + hl + o0;
    printf("block bytes %ld (expect %d)\n", o1 - o0, ROWS * COLS / 128 * 28);
    for (int i = 0; i < COLS; i++) X[i] = ((i * 37) % 251) / 125.0f - 1.0f;
    ternary_gemv_b3(ROWS, COLS, W, X, Y);
    double best = 1e30;
    struct timespec a, b;
    for (int t = 0; t < iters; t++) {
        clock_gettime(CLOCK_MONOTONIC, &a);
        ternary_gemv_b3(ROWS, COLS, W, X, Y);
        clock_gettime(CLOCK_MONOTONIC, &b);
        double s = (b.tv_sec - a.tv_sec) + 1e-9 * (b.tv_nsec - a.tv_nsec);
        if (s < best) best = s;
    }
    double bytes = (double)ROWS * COLS / 128 * 28;
    printf("best %.2f ms  -> %.2f GB/s stored, %.2f G trits/s, %.2f ns/weight\n",
           best * 1e3, bytes / best / 1e9, ROWS * (double)COLS / best / 1e9,
           best * 1e9 / (ROWS * (double)COLS));
    printf("projected blob token (376 MB): %.2f ms   y[0]=%g\n", 376e6 / (bytes / best) * 1e3, Y[0]);
    return 0;
}
