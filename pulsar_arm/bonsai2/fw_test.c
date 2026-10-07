#include <math.h>
#include <stdio.h>
static float A[1024], B[1024];
void fwht1024_f32(float * x);
/* scalar reference: fork's algorithm, prescale then butterflies */
static void ref(float * a) {
    for (int i = 0; i < 1024; i++) a[i] *= 0.03125f;
    for (int len = 1; len < 1024; len <<= 1)
        for (int i = 0; i < 1024; i += 2 * len)
            for (int j = 0; j < len; j++) {
                float u = a[i + j], v = a[i + len + j];
                a[i + j] = u + v; a[i + len + j] = u - v;
            }
}
int main(void) {
    for (int i = 0; i < 1024; i++) {
        A[i] = ((i * 37) % 251) / 125.0f - 1.0f;
        B[i] = A[i];
    }
    ref(B);
    fwht1024_f32(A);
    double mx = 0, md = 0;
    for (int i = 0; i < 1024; i++) {
        double d = fabs((double)A[i] - (double)B[i]);
        if (fabs(B[i]) > mx) mx = fabs(B[i]);
        if (d > md) md = d;
    }
    printf("vs scalar-ref: maxdiff=%.3e %s\n", md, md < 1e-5 ? "MATCH" : "MISMATCH");
    fwht1024_f32(A); /* involution: second application must restore */
    md = 0;
    for (int i = 0; i < 1024; i++) {
        float want = ((i * 37) % 251) / 125.0f - 1.0f;
        double d = fabs((double)A[i] - (double)want);
        if (d > md) md = d;
    }
    printf("involution maxdiff=%.3e %s\n", md, md < 1e-4 ? "MATCH" : "MISMATCH");
    return 0;
}
