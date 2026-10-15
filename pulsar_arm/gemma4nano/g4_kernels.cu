// g4_kernels.cu - Gemma 4 E2B GEMV kernels for the Jetson Nano (sm_53), besides
// the K2 Q4_0 kernel (g4_ptx_q4.h). Compiled to PTX once on a machine with the
// toolkit and embedded; the engine itself only needs the driver to JIT it:
//   nvcc -arch=compute_53 -ptx -fmad=false -O3 g4_kernels.cu -o g4_kernels.ptx
// -fmad=false: no implicit contraction; the fused ops below are explicit (__fmaf_rn),
// and each kernel reproduces the matching g4ref.c function bit for bit.
#include <cuda_fp16.h>
#include <stdint.h>

// Q6_K tied head: y[r] = g4ref.c gemv_q6k_rows. One warp per row.
// Lane l owns value index l of each of the 8 quarters (2 halves x 4) of a 256-value
// block: t = 128n + 32k + l, scale index 8n + 2k + l/16 (ggml's order). Integer
// partial sums are exact, so the warp reduction order cannot matter; lane 0 then
// applies acc = fma(dw * da, (float)isum, acc) over the blocks in ascending order.
extern "C" __global__ void q6k_gemv(const uint8_t *__restrict__ w,
                                    const uint8_t *__restrict__ xq,
                                    float *__restrict__ y, int rows, int nb) {
    const int lane = threadIdx.x;
    const int r = blockIdx.x * blockDim.y + threadIdx.y;
    if (r >= rows) return;
    const uint8_t *row = w + (size_t)r * nb * 210;
    float acc = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = row + b * 210;
        const int8_t *sc = (const int8_t *)(blk + 192);
        const uint8_t *q8b = xq + (size_t)b * 292;
        const int8_t *qs = (const int8_t *)(q8b + 4);
        int s = 0;
        #pragma unroll
        for (int n = 0; n < 2; n++) {
            const int L0 = blk[64 * n + lane], L1 = blk[64 * n + lane + 32];
            const int Hh = blk[128 + 32 * n + lane];
            const int q1 = ((L0 & 15) | ((Hh & 3) << 4)) - 32;
            const int q2 = ((L1 & 15) | (((Hh >> 2) & 3) << 4)) - 32;
            const int q3 = ((L0 >> 4) | (((Hh >> 4) & 3) << 4)) - 32;
            const int q4 = ((L1 >> 4) | (((Hh >> 6) & 3) << 4)) - 32;
            const int is = 8 * n + (lane >> 4);
            s += sc[is + 0] * (q1 * qs[128 * n + lane]);
            s += sc[is + 2] * (q2 * qs[128 * n + 32 + lane]);
            s += sc[is + 4] * (q3 * qs[128 * n + 64 + lane]);
            s += sc[is + 6] * (q4 * qs[128 * n + 96 + lane]);
        }
        #pragma unroll
        for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) {
            const float dw = __half2float(*(const __half *)(blk + 208));
            const float da = *(const float *)q8b;
            acc = __fmaf_rn(__fmul_rn(dw, da), __int2float_rn(s), acc);
        }
    }
    if (lane == 0) y[r] = acc;
}

// F16 GEMV for per_layer_model_proj: y[r] = g4ref.c gemv_f16. One thread per row,
// acc = fma(w[r][k], x[k], acc) for k ascending. The weights are stored transposed
// (wt[k * rows + r]) so a warp's 32 rows read 64 contiguous bytes per k.
extern "C" __global__ void f16_gemv_t(const __half *__restrict__ wt,
                                      const float *__restrict__ x,
                                      float *__restrict__ y, int rows, int cols) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float acc = 0.0f;
    for (int k = 0; k < cols; k++)
        acc = __fmaf_rn(__half2float(wt[(size_t)k * rows + r]), x[k], acc);
    y[r] = acc;
}
