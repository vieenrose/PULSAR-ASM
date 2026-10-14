/* Exp42 probe #4: pick the fastest BIT-IDENTICAL 16-byte repack form.
   repackprobe (measured on the Nano, 910MB dest):
     V0 glibc memcpy(dst,src+2,16)  1643 ms  <- what the engine does today (0.58 GB/s)
     V1 2x u64 via memcpy            305 ms
     V2 16B struct lvalue copy       300 ms
     V3 V2 unrolled x4               468 ms (slower: register pressure)
   All byte-identical. Now test strict-aliasing-clean forms that a -O2 compiler can
   inline: V4 __builtin_memcpy with constant 16, V5 packed-struct assignment,
   V6 __builtin_memcpy through __restrict pointers. Whichever wins becomes the engine
   repack body. Verify byte-identical against V0. */
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

typedef struct { uint64_t a, b; } __attribute__((packed, aligned(1))) U16P;

#define BLK (1ull << 30)

int main(void) {
    size_t src_bytes = BLK;
    size_t dst_bytes = src_bytes / 18 * 16;
    size_t nblk = dst_bytes / 16, i;
    uint8_t *src = malloc(src_bytes);
    uint8_t *ref = malloc(dst_bytes);
    uint8_t *d = malloc(dst_bytes);
    double t; unsigned long long diffs;
    if (!src || !ref || !d) { printf("alloc fail\n"); return 1; }
    memset(src, 0x5a, src_bytes);

    /* reference: V0 (current engine form) */
    for (i = 0; i < nblk; i++) memcpy(ref + i * 16, src + i * 18 + 2, 16);
    printf("nblk=%llu (%llu MB dest)\n", (unsigned long long)nblk, (unsigned long long)(dst_bytes >> 20));

#define TIMEV(name, ...) do {                          \
        t = now_ms();                                   \
        for (i = 0; i < nblk; i++) { __VA_ARGS__; }     \
        printf("%-22s %6.0f ms\n", name, now_ms() - t); \
        diffs = 0;                                      \
        for (i = 0; i < nblk * 16; i++) if (d[i] != ref[i]) diffs++; \
        printf("   vs V0: %s (%llu diffs)\n", diffs ? "MISMATCH" : "IDENTICAL", diffs); \
    } while (0)

    TIMEV("V0 glibc memcpy", memcpy(d + i * 16, src + i * 18 + 2, 16));
    TIMEV("V4 __builtin_memcpy", __builtin_memcpy(d + i * 16, src + i * 18 + 2, 16));
    TIMEV("V5 packed struct", *(U16P *)(d + i * 16) = *(const U16P *)(src + i * 18 + 2));
    TIMEV("V7 u64 pair locals", {
        const uint8_t *s = src + i * 18 + 2;
        uint64_t a, b;
        __builtin_memcpy(&a, s, 8); __builtin_memcpy(&b, s + 8, 8);
        __builtin_memcpy(d + i * 16, &a, 8); __builtin_memcpy(d + i * 16 + 8, &b, 8);
    });
    /* V8: 4 blocks with 72B -> 64B, packed struct, restrict-hinted */
    t = now_ms();
    { const uint8_t *__restrict s = src + 2;
      uint8_t *__restrict dd = d;
      for (i = 0; i + 3 < nblk; i += 4) {
          U16P b0 = *(const U16P *)(s), b1 = *(const U16P *)(s + 18);
          U16P b2 = *(const U16P *)(s + 36), b3 = *(const U16P *)(s + 54);
          s += 72;
          *(U16P *)(dd) = b0; *(U16P *)(dd + 16) = b1;
          *(U16P *)(dd + 32) = b2; *(U16P *)(dd + 48) = b3;
          dd += 64;
      } }
    printf("%-22s %6.0f ms\n", "V8 packed x4 restrict", now_ms() - t);
    diffs = 0;
    for (i = 0; i + 3 < nblk; i++) if (d[i] != ref[i]) diffs++;
    printf("   vs V0: %s (%llu diffs, prefix)\n", diffs ? "MISMATCH" : "IDENTICAL", diffs);
    return 0;
}
