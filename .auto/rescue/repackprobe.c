/* Exp42 pre-bench probe #3: the repack itself.
   Engine measurements (v32 init instrumentation): repack+H2D = 981 ms. h2dp6 just
   measured the pure serial copy at 82 ms (5.99 GB/s), so ~900 ms of the 981 ms is
   the CPU repack: 29.2M x memcpy(dst, src+2, 16) over an 18-byte-stride source.
   900 ms / 29.2M = 31 ns per 16-byte copy = glibc memcpy call overhead + short-copy
   dispatch, NOT bandwidth (467MB at 6 GB/s is 78 ms). Test variants for bit-identical
   output (byte-for-byte the same destination image):
     V0  memcpy(d, s+2, 16)                      [current]
     V1  two unaligned u64 loads/stores
     V2  one unaligned 16-byte struct load/store
     V3  V2 + 4-block unrolled (72B read -> 64B written per iteration)
   Also measures the cost of writing into a FRESH malloc buffer (first-touch faults)
   vs a pre-touched one, since that is the other init cost candidate. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

typedef struct { uint64_t a, b; } U16;   /* unaligned-capable on arm64/x86 */

#define ROWS 14336u          /* k3 rows: 14336 x 1152 cols -> nb = 36 blocks/row */
#define NB   36u
#define BLK  (1ull << 30)    /* ~1 GB source = 8.9 layers of k3 = 1.28 GB dest */

int main(void) {
    size_t src_bytes = BLK;
    uint8_t *src = malloc(src_bytes);
    size_t dst_bytes = src_bytes / 18 * 16;
    uint8_t *d1 = malloc(dst_bytes);       /* fresh (untouched) */
    uint8_t *d2 = malloc(dst_bytes);
    size_t nblk = dst_bytes / 16;
    size_t i; double t;
    uint64_t chk = 0;
    if (!src || !d1 || !d2) { printf("alloc fail\n"); return 1; }
    memset(src, 0x5a, src_bytes);
    memset(d2, 0, dst_bytes);              /* pre-touch d2 */

    printf("blocks = %llu (16B each = %llu MB dest)\n",
           (unsigned long long)nblk, (unsigned long long)(dst_bytes >> 20));

    /* --- V0: memcpy(dst, src+2, 16) per block (current engine code) --- */
    t = now_ms();
    for (i = 0; i < nblk; i++)
        memcpy(d1 + i * 16, src + i * 18 + 2, 16);
    printf("V0 memcpy      : %.0f ms  (%.2f GB/s out)\n", now_ms() - t,
           (double)dst_bytes / (now_ms() - t) / 1e6);

    /* --- V1: two unaligned u64 --- */
    t = now_ms();
    for (i = 0; i < nblk; i++) {
        const uint8_t *s = src + i * 18 + 2;
        uint64_t a, b;
        memcpy(&a, s, 8); memcpy(&b, s + 8, 8);
        memcpy(d1 + i * 16, &a, 8); memcpy(d1 + i * 16 + 8, &b, 8);
    }
    printf("V1 2x u64      : %.0f ms\n", now_ms() - t);

    /* --- V2: one unaligned 16B struct --- */
    t = now_ms();
    for (i = 0; i < nblk; i++)
        *(U16 *)(d1 + i * 16) = *(const U16 *)(src + i * 18 + 2);
    printf("V2 16B struct  : %.0f ms\n", now_ms() - t);

    /* --- V3: 4 blocks per step (72B in, 64B out) --- */
    t = now_ms();
    for (i = 0; i + 3 < nblk; i += 4) {
        const uint8_t *s = src + i * 18 + 2;
        U16 b0 = *(const U16 *)(s);
        U16 b1 = *(const U16 *)(s + 18);
        U16 b2 = *(const U16 *)(s + 36);
        U16 b3 = *(const U16 *)(s + 54);
        *(U16 *)(d1 + i * 16) = b0;
        *(U16 *)(d1 + i * 16 + 16) = b1;
        *(U16 *)(d1 + i * 16 + 32) = b2;
        *(U16 *)(d1 + i * 16 + 48) = b3;
    }
    printf("V3 struct x4   : %.0f ms\n", now_ms() - t);

    /* correctness of V2/V3 vs V0 reference (rebuild reference in d2 via memcpy) */
    {
        size_t m = (nblk / 4) * 4;
        memset(d2, 0, dst_bytes);
        for (i = 0; i < m; i++) memcpy(d2 + i * 16, src + i * 18 + 2, 16);
        for (i = 0; i < m * 16; i++) if (d1[i] != d2[i]) { chk++; if (chk < 4) printf("diff @%llu\n", (unsigned long long)i); }
        printf("V3 vs V0 byte compare: %s (%llu diffs)\n", chk ? "MISMATCH" : "IDENTICAL", (unsigned long long)chk);
    }

    /* --- first-touch cost: fresh vs pre-touched destination --- */
    t = now_ms();
    memset(d1, 0, dst_bytes);
    printf("memset fresh dest (untouched since malloc, first pass): %.0f ms\n", now_ms() - t);
    t = now_ms();
    memset(d1, 0, dst_bytes);
    printf("memset warm dest (second pass)                        : %.0f ms\n", now_ms() - t);
    printf("RSS check via /proc: see VmRSS line below\n");
    { FILE *f = fopen("/proc/self/status", "r"); char l[256];
      while (f && fgets(l, sizeof l, f)) if (!strncmp(l, "VmRSS", 5)) printf("  %s", l);
      if (f) fclose(f); }
    return 0;
}
