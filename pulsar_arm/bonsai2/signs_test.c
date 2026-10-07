#include <math.h>
#include <stdio.h>
/* signs + FWHT combined: out[i] = FWHT(x[i] * s[i]) for 1024 elems.
 * Validates the activation-side transform composition. s holds +-1.0f. */
void fwht1024_f32(float * x);
static float X[1024], S[1024], R[1024];
static void ref(float * x, float * s) {
    for (int i = 0; i < 1024; i++) x[i] *= s[i];
    for (int i = 0; i < 1024; i++) x[i] *= 0.03125f;
    for (int len = 1; len < 1024; len <<= 1)
        for (int i = 0; i < 1024; i += 2 * len)
            for (int j = 0; j < len; j++) {
                float u = x[i + j], v = x[i + len + j];
                x[i + j] = u + v; x[i + len + j] = u - v;
            }
}
int main(void) {
    for (int i = 0; i < 1024; i++) {
        X[i] = ((i * 37) % 251) / 125.0f - 1.0f;
        S[i] = (i % 3 == 0) ? -1.0f : 1.0f;
        R[i] = X[i];
    }
    ref(R, S);
    /* asm path: apply signs with a vector mul, then fwht */
    for (int i = 0; i < 1024; i++) X[i] *= S[i];
    fwht1024_f32(X);
    double md = 0;
    for (int i = 0; i < 1024; i++) {
        double d = fabs((double)X[i] - (double)R[i]);
        if (d > md) md = d;
    }
    printf("signs+fwht vs ref: maxdiff=%.3e %s\n", md, md < 1e-4 ? "MATCH" : "MISMATCH");
    return 0;
}
