/* Verify the wrap-free digit identity: digit_nn(byte) = (3t>>8) - 3*(t>>8)
 * where t = byte * 3^nn, vs the fork's ((uint8_t)(byte*3^nn) * 3) >> 8.
 * If this holds for all bytes and nn, the NEON decode needs no u8 truncation
 * step at all - 5 vector ops instead of 7. */
#include <stdio.h>
#include <stdint.h>
static const int P[6] = {1, 3, 9, 27, 81, 243};
int main(void) {
    int bad = 0;
    for (int nn = 0; nn < 5; nn++)
        for (int b = 0; b < 256; b++) {
            int fork = (((uint8_t)(b * P[nn])) * 3) >> 8;
            int t = b * P[nn];
            int mine = ((3 * t) >> 8) - 3 * (t >> 8);
            if (fork != mine) { if (bad < 3) printf("nn=%d b=%d fork=%d mine=%d\n", nn, b, fork, mine); bad++; }
        }
    printf("wrap-free identity: %s (%d violations of 1280)\n", bad ? "FAIL" : "EXACT", bad);
    /* also confirm the widest intermediate fits u16: 3*t for nn=4: 3*255*81 */
    printf("max 3*t = %d (fits u16: %s)\n", 3 * 255 * 81, (3 * 255 * 81) < 65536 ? "yes" : "NO");
    return bad ? 1 : 0;
}
