/* PTGV fixture checker: asm ternary GEMV vs the C reference and the oracle. */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void ternary_gemv_b3(int rows, int cols, const uint8_t *w, const float *x, float *y);
void ternary_gemv_scalar(int fmt, int rows, int cols, const uint8_t *w, const float *x, float *y);

#define MAGIC 0x50544756u

int main(int argc, char **argv) {
    int bad = 0;
    for (int a = 1; a < argc; a++) {
        FILE *f = fopen(argv[a], "rb");
        if (!f) { perror(argv[a]); bad++; continue; }
        uint64_t magic; uint32_t ver, fmt, rows, cols, ng, gb;
        fread(&magic, 8, 1, f); fread(&ver, 4, 1, f); fread(&fmt, 4, 1, f);
        fread(&rows, 4, 1, f); fread(&cols, 4, 1, f);
        fread(&ng, 4, 1, f); fread(&gb, 4, 1, f);
        if (magic != MAGIC) { printf("%s: bad magic\n", argv[a]); bad++; fclose(f); continue; }
        float *x = malloc(cols * 4), *yref = malloc(rows * 4), *yasm = malloc(rows * 4),
              *yc = malloc(rows * 4);
        uint8_t *w = malloc((size_t)rows * ng * gb);
        fread(x, 4, cols, f); fread(w, gb, (size_t)rows * ng, f); fread(yref, 4, rows, f);
        fclose(f);
        int fmt0 = fmt;
        if (fmt) {                      /* base-3: the asm kernel handles this */
            ternary_gemv_b3(rows, cols, w, x, yasm);
        } else {                        /* Q2_0: reference C path only */
            ternary_gemv_scalar(0, rows, cols, w, x, yasm);
        }
        ternary_gemv_scalar(fmt0, rows, cols, w, x, yc);
        double wc = 0, wa = 0;
        int iwc = 0, iwa = 0;
        for (uint32_t r = 0; r < rows; r++) {
            double d1 = fabs((double)yc[r] - (double)yref[r]) / fmax(fabs((double)yref[r]), 1e-6);
            double d2 = fabs((double)yasm[r] - (double)yref[r]) / fmax(fabs((double)yref[r]), 1e-6);
            if (d1 > wc) { wc = d1; iwc = r; }
            if (d2 > wa) { wa = d2; iwa = r; }
        }
        int ok = (wc < 5e-4) && (wa < 5e-4);
        printf("%-30s %-6s %4ux%-5u  C-vs-oracle %.2e (row %d)  ASM-vs-oracle %.2e (row %d)  %s\n",
               strrchr(argv[a], '/') ? strrchr(argv[a], '/') + 1 : argv[a],
               fmt ? "B3_128" : "Q2_0", rows, cols, wc, iwc, wa, iwa, ok ? "ok" : "FAIL");
        if (!ok) bad++;
        free(x); free(yref); free(yasm); free(yc); free(w);
    }
    printf("%s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
