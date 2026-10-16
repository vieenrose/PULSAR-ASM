// Exhaustive check of glibc_mathf.h against the system libm: all 2^32 float inputs.
//   gcc -O2 -fopenmp -ffp-contract=off -o glibc_mathf_test glibc_mathf_test.c -lm
//   ./glibc_mathf_test [expf|logf|sinf|cosf ...]   (default: all four)
// Prints the mismatch count per function (NaN results compare equal to any NaN).
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "glibc_mathf.h"

typedef float (*ff)(float);
static float (*volatile L_expf)(float) = expf, (*volatile L_logf)(float) = logf,
             (*volatile L_sinf)(float) = sinf, (*volatile L_cosf)(float) = cosf;

static void run(const char *nm, ff mine, ff lib) {
    unsigned long long bad = 0; uint32_t first[8]; int nf = 0;
    #pragma omp parallel for schedule(dynamic, 1) reduction(+:bad)
    for (int64_t hi = 0; hi < 65536; hi++) {
        for (uint32_t lo = 0; lo < 65536; lo++) {
            const uint32_t u = (uint32_t)(hi << 16) | lo;
            const float x = gm_asf(u);
            const float a = mine(x), b = lib(x);
            if (gm_asu(a) != gm_asu(b) && !(a != a && b != b)) {
                bad++;
                #pragma omp critical
                if (nf < 8) first[nf++] = u;
            }
        }
    }
    printf("%s: %llu mismatches", nm, bad);
    for (int i = 0; i < nf; i++) {
        const float x = gm_asf(first[i]);
        printf("  [%a: %a vs %a]", x, mine(x), lib(x));
    }
    printf("\n"); fflush(stdout);
}
static float m_expf(float x) { return gm_expf(x); }
static float m_logf(float x) { return gm_logf(x); }
static float m_sinf(float x) { return gm_sinf(x); }
static float m_cosf(float x) { return gm_cosf(x); }
static float w_expf(float x) { return L_expf(x); }
static float w_logf(float x) { return L_logf(x); }
static float w_sinf(float x) { return L_sinf(x); }
static float w_cosf(float x) { return L_cosf(x); }

int main(int argc, char **argv) {
    const char *all[] = {"expf", "logf", "sinf", "cosf"};
    const char **w = argc > 1 ? (const char **)argv + 1 : all;
    const int n = argc > 1 ? argc - 1 : 4;
    for (int i = 0; i < n; i++) {
        if (!strcmp(w[i], "expf")) run("expf", m_expf, w_expf);
        if (!strcmp(w[i], "logf")) run("logf", m_logf, w_logf);
        if (!strcmp(w[i], "sinf")) run("sinf", m_sinf, w_sinf);
        if (!strcmp(w[i], "cosf")) run("cosf", m_cosf, w_cosf);
    }
    return 0;
}
