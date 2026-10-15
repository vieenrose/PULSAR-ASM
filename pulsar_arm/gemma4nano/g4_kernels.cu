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

// Q6_K tied head, v2: same arithmetic as q6k_gemv, faster data path.
// Layout (repacked once at load): every 210-byte Q6_K block padded to 212 bytes,
// so each block starts 4-byte aligned; content unchanged.
// Lane = (g, m): g = lane/8 picks the 32-byte ql run [32g, 32g+32) of the block,
// i.e. half n = g/2 and quarter pair (k0,k1) = (g&1, (g&1)+2) (low and high
// nibbles of the same bytes); m = lane%8 picks positions p = 4m..4m+3 inside the
// quarter. So each lane loads one u32 of ql and one u32 of qh per block, covering
// 8 values: t = 128n + 32k + p for k in {k0,k1}, p in {4m..4m+3}. Their sub-block
// scales are sc[8n + 2k + m/4]. Activations of all 6 blocks are loaded into
// registers once and reused for the ROWS_PER_WARP rows the warp walks.
// Integer sums are exact, so per-block isum equals g4ref's; lane 0 then applies
// acc = fma(dw * da, (float)isum, acc) over blocks in ascending order.
#define Q6B2 212
#define ROWS_PER_WARP 4
extern "C" __global__ void q6k_gemv2(const uint8_t *__restrict__ w,
                                     const uint8_t *__restrict__ xq,
                                     float *__restrict__ y, int rows) {
    const int lane = threadIdx.x, g = lane >> 3, m = lane & 7;
    const int n = g >> 1, k0 = g & 1, k1 = k0 + 2;
    const int warp = blockIdx.x * blockDim.y + threadIdx.y;
    int a0[6], a1[6];              // packed int8x4 activations for quarters k0, k1
    float da[6];
    #pragma unroll
    for (int b = 0; b < 6; b++) {
        const uint8_t *q8b = xq + b * 292;
        da[b] = *(const float *)q8b;
        a0[b] = *(const int *)(q8b + 4 + 128 * n + 32 * k0 + 4 * m);
        a1[b] = *(const int *)(q8b + 4 + 128 * n + 32 * k1 + 4 * m);
    }
    for (int i = 0; i < ROWS_PER_WARP; i++) {
        const int r = warp * ROWS_PER_WARP + i;
        if (r >= rows) return;
        const uint8_t *row = w + (size_t)r * 6 * Q6B2;
        float acc = 0.0f;
        #pragma unroll
        for (int b = 0; b < 6; b++) {
            const uint8_t *blk = row + b * Q6B2;
            const unsigned L = *(const unsigned *)(blk + 32 * g + 4 * m);         // ql bytes
            const unsigned Hh = *(const unsigned *)(blk + 128 + 32 * n + 4 * m);  // qh bytes
            const int8_t *sc = (const int8_t *)(blk + 192);
            const int s0 = sc[8 * n + 2 * k0 + (m >> 2)], s1 = sc[8 * n + 2 * k1 + (m >> 2)];
            int p0 = 0, p1 = 0;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const int lb = (L >> (8 * j)) & 0xff, hb = (Hh >> (8 * j)) & 0xff;
                const int v0 = ((lb & 15) | (((hb >> (2 * k0)) & 3) << 4)) - 32;
                const int v1 = ((lb >> 4) | (((hb >> (2 * k1)) & 3) << 4)) - 32;
                p0 += v0 * (int)(int8_t)(a0[b] >> (8 * j));
                p1 += v1 * (int)(int8_t)(a1[b] >> (8 * j));
            }
            int s = s0 * p0 + s1 * p1;
            #pragma unroll
            for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
            if (lane == 0) {
                const float dw = __half2float(*(const __half *)(blk + 208));
                acc = __fmaf_rn(__fmul_rn(dw, da[b]), __int2float_rn(s), acc);
            }
        }
        if (lane == 0) y[r] = acc;
    }
}

// Q6_K tied head, v3: q6k_gemv2's data path, but per row all 12 weight words are
// loaded up front and the 6 block sums are reduced interleaved (6 independent
// shuffle chains instead of 6 serial ones). Lane 0 still applies the 6 fmas in
// ascending block order, so the result is the same bits.
extern "C" __global__ void q6k_gemv3(const uint8_t *__restrict__ w,
                                     const uint8_t *__restrict__ xq,
                                     float *__restrict__ y, int rows) {
    const int lane = threadIdx.x, g = lane >> 3, m = lane & 7;
    const int n = g >> 1, k0 = g & 1, k1 = k0 + 2;
    const int i0 = 8 * n + 2 * k0 + (m >> 2), i1 = 8 * n + 2 * k1 + (m >> 2);
    const int warp = blockIdx.x * blockDim.y + threadIdx.y;
    int a0[6], a1[6];
    float da[6];
    #pragma unroll
    for (int b = 0; b < 6; b++) {
        const uint8_t *q8b = xq + b * 292;
        da[b] = *(const float *)q8b;
        a0[b] = *(const int *)(q8b + 4 + 128 * n + 32 * k0 + 4 * m);
        a1[b] = *(const int *)(q8b + 4 + 128 * n + 32 * k1 + 4 * m);
    }
    for (int i = 0; i < ROWS_PER_WARP; i++) {
        const int r = warp * ROWS_PER_WARP + i;
        if (r >= rows) return;
        const uint8_t *row = w + (size_t)r * 6 * Q6B2;
        unsigned L[6], Hh[6];
        int S0[6], S1[6];
        #pragma unroll
        for (int b = 0; b < 6; b++) {
            const uint8_t *blk = row + b * Q6B2;
            L[b] = *(const unsigned *)(blk + 32 * g + 4 * m);
            Hh[b] = *(const unsigned *)(blk + 128 + 32 * n + 4 * m);
            S0[b] = ((const int8_t *)(blk + 192))[i0];
            S1[b] = ((const int8_t *)(blk + 192))[i1];
        }
        int s[6];
        #pragma unroll
        for (int b = 0; b < 6; b++) {
            int p0 = 0, p1 = 0;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const int lb = (L[b] >> (8 * j)) & 0xff, hb = (Hh[b] >> (8 * j)) & 0xff;
                const int v0 = ((lb & 15) | (((hb >> (2 * k0)) & 3) << 4)) - 32;
                const int v1 = ((lb >> 4) | (((hb >> (2 * k1)) & 3) << 4)) - 32;
                p0 += v0 * (int)(int8_t)(a0[b] >> (8 * j));
                p1 += v1 * (int)(int8_t)(a1[b] >> (8 * j));
            }
            s[b] = S0[b] * p0 + S1[b] * p1;
        }
        #pragma unroll
        for (int o = 16; o; o >>= 1) {
            #pragma unroll
            for (int b = 0; b < 6; b++) s[b] += __shfl_xor_sync(0xffffffffu, s[b], o);
        }
        if (lane == 0) {
            float acc = 0.0f;
            #pragma unroll
            for (int b = 0; b < 6; b++) {
                const float dw = __half2float(*(const __half *)(row + b * Q6B2 + 208));
                acc = __fmaf_rn(__fmul_rn(dw, da[b]), __int2float_rn(s[b]), acc);
            }
            y[r] = acc;
        }
    }
}

// Q6_K tied head, v4: fewer instructions per value (the head is issue-bound on the
// Nano's single SM). Same lane mapping and layout as v2/v3.
//  - 4 values per u32 decoded at once:  q = (ql nibbles) | (qh 2-bit pairs << 4), as
//    unsigned bytes 0..63, with byte masks instead of per-value shifts.
//  - the -32 offset leaves the inner loop: sum (q-32)*a = sum q*a - 32 * sum a,
//    and sum a depends only on the activations, so it is computed once per warp.
//  - activation bytes are sign-extended once into registers (48 per lane).
// All sums are exact int32, so each block's isum is unchanged; lane 0 applies the
// 6 fmas in ascending block order as before.
extern "C" __global__ void __launch_bounds__(256) q6k_gemv4(const uint8_t *__restrict__ w,
                                     const uint8_t *__restrict__ xq,
                                     float *__restrict__ y, int rows) {
    const int lane = threadIdx.x, g = lane >> 3, m = lane & 7;
    const int n = g >> 1, k0 = g & 1, k1 = k0 + 2;
    const int i0 = 8 * n + 2 * k0 + (m >> 2), i1 = 8 * n + 2 * k1 + (m >> 2);
    const int sh0 = 4 - 2 * k0;          // qh pair for k0 -> bits 4..5 (left shift)
    const int sh1 = 2 * k1 - 4;          // qh pair for k1 -> bits 4..5 (right shift)
    const int warp = blockIdx.x * blockDim.y + threadIdx.y;
    int A0[6][4], A1[6][4], C0[6], C1[6];
    float da[6];
    #pragma unroll
    for (int b = 0; b < 6; b++) {
        const uint8_t *q8b = xq + b * 292;
        da[b] = *(const float *)q8b;
        const int w0 = *(const int *)(q8b + 4 + 128 * n + 32 * k0 + 4 * m);
        const int w1 = *(const int *)(q8b + 4 + 128 * n + 32 * k1 + 4 * m);
        C0[b] = 0; C1[b] = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            A0[b][j] = (int)(int8_t)(w0 >> (8 * j)); C0[b] += A0[b][j];
            A1[b][j] = (int)(int8_t)(w1 >> (8 * j)); C1[b] += A1[b][j];
        }
    }
    for (int i = 0; i < ROWS_PER_WARP; i++) {
        const int r = warp * ROWS_PER_WARP + i;
        if (r >= rows) return;
        const uint8_t *row = w + (size_t)r * 6 * Q6B2;
        unsigned L[6], Hh[6];
        int S0[6], S1[6];
        #pragma unroll
        for (int b = 0; b < 6; b++) {
            const uint8_t *blk = row + b * Q6B2;
            L[b] = *(const unsigned *)(blk + 32 * g + 4 * m);
            Hh[b] = *(const unsigned *)(blk + 128 + 32 * n + 4 * m);
            S0[b] = ((const int8_t *)(blk + 192))[i0];
            S1[b] = ((const int8_t *)(blk + 192))[i1];
        }
        int s[6];
        #pragma unroll
        for (int b = 0; b < 6; b++) {
            const unsigned q0 = (L[b] & 0x0F0F0F0Fu) | ((Hh[b] << sh0) & 0x30303030u);
            const unsigned q1 = ((L[b] >> 4) & 0x0F0F0F0Fu) | ((Hh[b] >> sh1) & 0x30303030u);
            int p0 = -32 * C0[b], p1 = -32 * C1[b];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                p0 += (int)((q0 >> (8 * j)) & 0xffu) * A0[b][j];
                p1 += (int)((q1 >> (8 * j)) & 0xffu) * A1[b][j];
            }
            s[b] = S0[b] * p0 + S1[b] * p1;
        }
        #pragma unroll
        for (int o = 16; o; o >>= 1) {
            #pragma unroll
            for (int b = 0; b < 6; b++) s[b] += __shfl_xor_sync(0xffffffffu, s[b], o);
        }
        if (lane == 0) {
            float acc = 0.0f;
            #pragma unroll
            for (int b = 0; b < 6; b++) {
                const float dw = __half2float(*(const __half *)(row + b * Q6B2 + 208));
                acc = __fmaf_rn(__fmul_rn(dw, da[b]), __int2float_rn(s[b]), acc);
            }
            y[r] = acc;
        }
    }
}
