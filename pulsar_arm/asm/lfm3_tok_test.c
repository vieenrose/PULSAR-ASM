/* lfm3_tok_test.c -- S1 gate driver. Reads fixtures (text + expected ids), encodes
   with lfm3_tok.c, compares id-for-id. Exit 0 iff every fixture matches exactly. */
#include "lfm3_tok.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s tok.bin fixdir\n", argv[0]); return 2; }
    LTok t;
    if (ltok_load(&t, argv[1])) { fprintf(stderr, "load %s failed\n", argv[1]); return 2; }
    /* bijectivity: the byte map must be a permutation (else encode/decode disagree) */
    {
        int seen[512] = {0}, bad = 0;
        for (int b = 0; b < 256; b++) {
            if (t.bmap[b] >= 512 || seen[t.bmap[b]]) bad = 1;
            seen[t.bmap[b]] = 1;
        }
        if (bad) { fprintf(stderr, "byte map not bijective\n"); return 2; }
    }
    DIR *d = opendir(argv[2]);
    if (!d) { fprintf(stderr, "opendir %s failed\n", argv[2]); return 2; }
    int nfix = 0, nbad = 0;
    long tot_ids = 0;
    struct dirent *e;
    while ((e = readdir(d))) {
        size_t L = strlen(e->d_name);
        if (L < 5 || memcmp(e->d_name + L - 4, ".txt", 4)) continue;
        char pb[1024], qb[1024];
        snprintf(pb, sizeof pb, "%s/%s", argv[2], e->d_name);
        snprintf(qb, sizeof qb, "%s/%.*s.ids", argv[2], (int)(L - 4), e->d_name);
        FILE *f = fopen(pb, "rb");
        if (!f) { fprintf(stderr, "open %s failed\n", pb); return 2; }
        fseek(f, 0, SEEK_END); long nl = ftell(f); fseek(f, 0, SEEK_SET);
        uint8_t *txt = malloc(nl ? nl : 1);
        if (nl && fread(txt, 1, nl, f) != (size_t)nl) { fclose(f); return 2; }
        fclose(f);
        f = fopen(qb, "rb");
        if (!f) { fprintf(stderr, "open %s failed\n", qb); return 2; }
        fseek(f, 0, SEEK_END); long ml = ftell(f); fseek(f, 0, SEEK_SET);
        uint32_t *want = malloc(ml ? ml : 4);
        if (ml && fread(want, 1, ml, f) != (size_t)ml) { fclose(f); return 2; }
        fclose(f);
        size_t nw = ml / 4;
        uint32_t *got = malloc((nl * 2 + 16) * sizeof(uint32_t));
        int ng = ltok_encode(&t, txt, nl, got, (int)(nl * 2 + 16));
        nfix++;
        tot_ids += nw;
        if (ng < 0 || (size_t)ng != nw || memcmp(got, want, ml)) {
            nbad++;
            printf("MISMATCH %s: want %zu ids, got %d\n", e->d_name, nw, ng);
            if (ng >= 0) {
                size_t m = nw < (size_t)ng ? nw : (size_t)ng;
                for (size_t i = 0; i < m && i < 12; i++)
                    if (got[i] != want[i]) { printf("  first diff at %zu: want %u got %u\n", i, want[i], got[i]); break; }
            }
            if (nbad > 5) { printf("  ...stopping after 6 mismatches\n"); break; }
        }
        free(txt); free(want); free(got);
    }
    closedir(d);
    printf("tok gate: %d/%d fixtures exact (%ld ids)\n", nfix - nbad, nfix, tot_ids);
    return nbad ? 1 : 0;
}
