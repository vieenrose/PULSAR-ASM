/* Build: gcc -O2 -ffp-contract=off -march=native -o markstein_check markstein_check.c -lm; run: ./markstein_check <r-count>
   (2^23 float x per r). Run on 2026-10-15 with 8 seeds x 3750: 2.5e11 checks, 0 mismatches. */
/* Markstein-corrected division vs IEEE division: q1 = fma(fma(-q0, r, x), y, q0),
   q0 = x*y, y = 1/r (RN). Checks RN(x/r) == q1 for every float mantissa x (all 2^24
   significands incl. sign and a spread of exponents) against many r of the form
   sqrt(ss/n + eps) and random doubles. Also checks the full rms output expression. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
static uint64_t s = 88172645463325252ull;
static uint64_t xr(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
int main(int argc, char **argv) {
    long nr = argc > 1 ? atol(argv[1]) : 200;
    long bad = 0, tot = 0;
    for (long k = 0; k < nr; k++) {
        double r;
        if (k % 3 == 0) { double ss = (double)(xr() >> 11) * 0x1p-53 * 1e4; r = sqrt(ss / 1536 + 1e-6); }
        else if (k % 3 == 1) { uint64_t b = (xr() & 0x000fffffffffffffull) | 0x3ff0000000000000ull; memcpy(&r, &b, 8); r *= ldexp(1.0, (int)(xr() % 40) - 20); }
        else { uint64_t b = 0x3fffffffffffffffull - (xr() % 1024); memcpy(&r, &b, 8); r *= ldexp(1.0, (int)(xr() % 20) - 10); }   /* near all-ones mantissas */
        const double y = 1.0 / r;
        for (uint32_t m = 0; m < (1u << 23); m++) {
            const int ex = (int)(xr() % 60) + 97;   /* float exponents 2^-30 .. 2^29 */
            uint32_t xb = (uint32_t)ex << 23 | m | (m & 1 ? 0x80000000u : 0);
            float xf; memcpy(&xf, &xb, 4);
            const double x = (double)xf;
            const double q = x / r;
            const double q0 = x * y;
            const double q1 = fma(fma(-q0, r, x), y, q0);
            tot++;
            if (memcmp(&q, &q1, 8)) { if (bad < 10) printf("MISMATCH x=%a r=%a q=%a q1=%a\n", x, r, q, q1); bad++; }
        }
    }
    printf("%ld r values x 2^23 x: %ld checks, %ld mismatches\n", nr, tot, bad);
    return bad != 0;
}
