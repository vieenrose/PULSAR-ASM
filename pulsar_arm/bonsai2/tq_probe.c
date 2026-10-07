/* PTQ1_0 decoder probe: fork's dequantize_row_ptq1_0 vs a from-spec decoder,
 * on real rows from the partial download. Must agree bit-for-bit (same fp32
 * arithmetic order is NOT guaranteed, so tolerance is 1e-6 relative). */
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
void dequantize_row_ptq1_0(const void * x, float * y, int64_t k);
static const uint8_t P3[6] = {1, 3, 9, 27, 81, 243};
static float f16(float b0, float b1) { (void)b0; (void)b1; return 0; }
static float h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu, o;
    if (e == 0) { if (!m) return s ? -0.0f : 0.0f; float f = (float)m * 5.9604645e-8f; memcpy(&o, &f, 4); o |= s; }
    else if (e == 31) o = s | 0x7f800000u | (m << 13);
    else o = s | ((e + 112u) << 23) | (m << 13);
    float r; memcpy(&r, &o, 4); return r;
}
/* one 28-byte group -> 128 floats. Mirrors the fork traversal exactly:
 * 16 bytes digit-major (80), 8 bytes digit-major (40), qh 2 bytes x4 (8). */
static void my_decode_group(const uint8_t *blk, float *out) {
    float scale = h2f((uint16_t)(blk[26] | (blk[27] << 8)));
    int o = 0;
    for (int q = 0; q < 2; q++) {
        int base = q ? 16 : 0, c = q ? 8 : 16;
        for (int nn = 0; nn < 5; nn++)
            for (int m = 0; m < c; m++) {
                uint8_t v = (uint8_t)(blk[base + m] * P3[nn]);
                int xi = ((uint16_t)v * 3) >> 8;
                out[o++] = (float)(xi - 1) * scale;
            }
    }
    for (int nn = 0; nn < 4; nn++)
        for (int h = 0; h < 2; h++) {
            uint8_t v = (uint8_t)(blk[24 + h] * P3[nn]);
            int xi = ((uint16_t)v * 3) >> 8;
            out[o++] = (float)(xi - 1) * scale;
        }
}
int main(int argc, char **argv) {
    const char *path = argv[1];
    long data_start = atol(argv[2]), emb_off = atol(argv[3]);
    int rows = argc > 4 ? atoi(argv[4]) : 4, cols = 5120;
    int fd = open(path, O_RDONLY); struct stat st; fstat(fd, &st);
    uint8_t *m = mmap(0, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    static float A[5120], B[5120];
    double mx = 0, md = 0;
    for (int r = 0; r < rows; r++) {
        const uint8_t *w = m + data_start + emb_off + (size_t)r * (cols / 128) * 28;
        dequantize_row_ptq1_0(w, A, cols);
        for (int g = 0; g < cols / 128; g++)
            my_decode_group(w + g * 28, B + g * 128);
        for (int i = 0; i < cols; i++) {
            double d = fabs((double)A[i] - (double)B[i]);
            if (fabs(A[i]) > mx) mx = fabs(A[i]);
            if (d > md) md = d;
        }
    }
    printf("rows=%d max|fork|=%.6f maxdiff=%.3e rel=%.2e %s\n",
           rows, mx, md, md / (mx + 1e-30), md / (mx + 1e-30) < 1e-6 ? "MATCH" : "MISMATCH");
    printf("row0[0..5] fork: %.6f %.6f %.6f %.6f %.6f %.6f\n", A[0], A[1], A[2], A[3], A[4], A[5]);
    return 0;
}
