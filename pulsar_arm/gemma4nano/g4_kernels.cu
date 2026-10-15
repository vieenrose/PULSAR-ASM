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

// ------------------------------------------------------------------------------
// MLP stages on the GPU, so gate+up -> GeGLU -> Q8_K -> down needs no CPU round
// trip. Both reproduce g4ref.c bit for bit with explicit IEEE ops (no contraction).

// g4ref.c g4_exp, operation by operation. tab: the same constant table the host
// assembly uses (24 doubles: 1/13!, EXP_C[0..12], LOG2E, SH, LN2HI, LN2LO, ...).
__device__ __forceinline__ double g4_exp_dev(double x, const double *__restrict__ tab) {
    if (x != x) return x;
    if (x > 709.0) return __longlong_as_double(0x7ff0000000000000ll);
    if (x < -745.0) return 0.0;
    const double kd = __dsub_rn(__dadd_rn(__dmul_rn(x, tab[14]), tab[15]), tab[15]);
    const int k = (int)kd;
    const double r = __dsub_rn(__dsub_rn(x, __dmul_rn(kd, tab[16])), __dmul_rn(kd, tab[17]));
    double p = tab[0];
    #pragma unroll
    for (int i = 1; i <= 13; i++) p = __dadd_rn(__dmul_rn(p, r), tab[i]);
    const int k1 = k / 2, k2 = k - k1;
    const double s1 = __longlong_as_double((long long)(k1 + 1023) << 52);
    const double s2 = __longlong_as_double((long long)(k2 + 1023) << 52);
    return __dmul_rn(__dmul_rn(p, s1), s2);
}

// h[i] = (float)gelu(g[i]) * u[i]; g = y[0..n), u = y[n..2n) (the merged gate+up output)
extern "C" __global__ void geglu_dev(const float *__restrict__ y, float *__restrict__ h,
                                     int n, const double *__restrict__ tab) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double x = (double)y[i];
    const double x3 = __dmul_rn(__dmul_rn(x, x), x);
    const double t = -__dadd_rn(__dmul_rn(tab[21], x), __dmul_rn(tab[22], x3));
    const double e = g4_exp_dev(t, tab);
    const float gl = __double2float_rn(__ddiv_rn(x, __dadd_rn(1.0, e)));
    h[i] = __fmul_rn(gl, y[n + i]);
}

// Q8_K quantization of n floats (n % 256 == 0), one 256-thread block per 256 values,
// output 292-byte blocks (f32 d, 256 int8, 32 bytes unused) as g4ref.c q8k:
// signed max = the FIRST element with the largest |x| (C: if (a > amax)), found as
// max |x| then the smallest index holding it; iscale = -127/max;
// q = nearest_int(fmaf(iscale, x, 0x1.8p23)) clamped below 127; d = 1/iscale.
extern "C" __global__ void q8k_dev(const float *__restrict__ x, uint8_t *__restrict__ xq) {
    __shared__ float sa[256];
    __shared__ int si[256];
    const int j = threadIdx.x, b = blockIdx.x;
    const float v = x[b * 256 + j];
    const float a = fabsf(v);
    sa[j] = a;
    __syncthreads();
    for (int s = 128; s; s >>= 1) { if (j < s) sa[j] = fmaxf(sa[j], sa[j + s]); __syncthreads(); }
    const float amax = sa[0];
    si[j] = (a == amax) ? j : 256;
    __syncthreads();
    for (int s = 128; s; s >>= 1) { if (j < s) si[j] = min(si[j], si[j + s]); __syncthreads(); }
    uint8_t *yb = xq + (size_t)b * 292;
    if (j < 32) yb[260 + j] = 0;
    if (!(amax > 0.0f)) {                       // all zero: d = 0, q = 0
        if (j < 4) yb[j] = 0;
        yb[4 + j] = 0;
        return;
    }
    const float mx = x[b * 256 + si[0]];
    const float iscale = __fdiv_rn(-127.f, mx);
    const float f = __fmaf_rn(iscale, v, 12582912.f);
    int q = (__float_as_int(f) & 0x007fffff) - 0x00400000;
    q = q < 127 ? q : 127;
    yb[4 + j] = (uint8_t)(int8_t)q;
    if (j == 0) { const float d = __fdiv_rn(1.0f, iscale); *(float *)yb = d; }
}

// ------------------------------------------------------------------------------
// Q4_0 GEMV, row-per-lane ("q4i"): the K2 kernel's arithmetic with a different
// data path. Per row: isum_b = sum over the 32 weights of block b of (n-8)*a,
// exact int32; acc = fma(ws_b * d_a(b/8), (float)isum_b, acc) for b ascending.
// That is gemv_q4 in g4ref.c, so the result is the same bits.
// Layout (built once at load): rows in tiles of 32; for tile t and block b the
// 32 rows' 16 nibble bytes are contiguous (512 B), scales f32 as [t][b][32].
// A lane owns one row and walks its blocks in order, so the float chain stays in
// one thread (no cross-lane reduction); a warp reads 512 contiguous bytes per
// block. Activations go to shared memory once per CTA, sign-extended to int16,
// with each 32-block's sum precomputed so (n-8)*a = n*a - 8*sum(a) per block.
#define Q4I_MAXCOLS 12288
extern "C" __global__ void __launch_bounds__(256) q4i_gemv(const uint4 *__restrict__ w,
        const float *__restrict__ sc, const uint8_t *__restrict__ xq,
        float *__restrict__ y, int rows, int cols) {
    __shared__ short A[Q4I_MAXCOLS];
    __shared__ int SA[Q4I_MAXCOLS / 32];
    __shared__ float DA[Q4I_MAXCOLS / 256];
    const int nb = cols >> 5, tid = threadIdx.y * 32 + threadIdx.x;
    for (int i = tid; i < cols; i += 256) A[i] = (short)(int8_t)xq[(i >> 8) * 292 + 4 + (i & 255)];
    for (int i = tid; i < (cols >> 8); i += 256) DA[i] = *(const float *)(xq + i * 292);
    __syncthreads();
    for (int b = tid; b < nb; b += 256) {
        int s = 0;
        for (int j = 0; j < 32; j++) s += A[b * 32 + j];
        SA[b] = s;
    }
    __syncthreads();
    const int tile = blockIdx.x * 8 + threadIdx.y;
    const int r = tile * 32 + threadIdx.x;
    if (r >= rows) return;
    const uint4 *wt = w + (size_t)tile * nb * 32 + threadIdx.x;
    const float *st = sc + (size_t)tile * nb * 32 + threadIdx.x;
    float acc = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint4 q = wt[(size_t)b * 32];
        const float ws = st[(size_t)b * 32];
        const short *a = A + b * 32;
        int s = -8 * SA[b];
        const unsigned qw[4] = {q.x, q.y, q.z, q.w};
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                s += (int)((qw[k] >> (8 * j)) & 15u) * (int)a[4 * k + j];
                s += (int)((qw[k] >> (8 * j + 4)) & 15u) * (int)a[16 + 4 * k + j];
            }
        }
        acc = __fmaf_rn(__fmul_rn(ws, DA[b >> 3]), __int2float_rn(s), acc);
    }
    y[r] = acc;
}

// q4i v2: same layout and arithmetic as q4i_gemv; the 32 int16 activations of a
// block are read as four 16-byte shared-memory vectors (instead of ~34 scalar
// loads, which made v1 load-unit bound) and split into halves in registers.
extern "C" __global__ void __launch_bounds__(256) q4i_gemv2(const uint4 *__restrict__ w,
        const float *__restrict__ sc, const uint8_t *__restrict__ xq,
        float *__restrict__ y, int rows, int cols) {
    __shared__ __align__(16) short A[Q4I_MAXCOLS];
    __shared__ int SA[Q4I_MAXCOLS / 32];
    __shared__ float DA[Q4I_MAXCOLS / 256];
    const int nb = cols >> 5, tid = threadIdx.y * 32 + threadIdx.x;
    for (int i = tid; i < cols; i += 256) A[i] = (short)(int8_t)xq[(i >> 8) * 292 + 4 + (i & 255)];
    for (int i = tid; i < (cols >> 8); i += 256) DA[i] = *(const float *)(xq + i * 292);
    __syncthreads();
    for (int b = tid; b < nb; b += 256) {
        int s = 0;
        for (int j = 0; j < 32; j++) s += A[b * 32 + j];
        SA[b] = s;
    }
    __syncthreads();
    const int tile = blockIdx.x * 8 + threadIdx.y;
    const int r = tile * 32 + threadIdx.x;
    if (r >= rows) return;
    const uint4 *wt = w + (size_t)tile * nb * 32 + threadIdx.x;
    const float *st = sc + (size_t)tile * nb * 32 + threadIdx.x;
    const uint4 *A4 = (const uint4 *)A;
    float acc = 0.0f;
    for (int b = 0; b < nb; b++) {
        const uint4 q = wt[(size_t)b * 32];
        const float ws = st[(size_t)b * 32];
        const uint4 a0 = A4[b * 4 + 0], a1 = A4[b * 4 + 1], a2 = A4[b * 4 + 2], a3 = A4[b * 4 + 3];
        // a0: x0..x7, a1: x8..x15 (low nibbles' partners); a2: x16..x23, a3: x24..x31
        const unsigned lo[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        const unsigned hi[8] = {a2.x, a2.y, a2.z, a2.w, a3.x, a3.y, a3.z, a3.w};
        const unsigned qw[4] = {q.x, q.y, q.z, q.w};
        int s = -8 * SA[b];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const int e = 4 * k + j;                       // weight byte index 0..15
                const unsigned pl = lo[e >> 1], ph = hi[e >> 1];
                const int al = (e & 1) ? ((int)pl >> 16) : (int)(short)(pl & 0xffffu);
                const int ah = (e & 1) ? ((int)ph >> 16) : (int)(short)(ph & 0xffffu);
                s += (int)((qw[k] >> (8 * j)) & 15u) * al;
                s += (int)((qw[k] >> (8 * j + 4)) & 15u) * ah;
            }
        }
        acc = __fmaf_rn(__fmul_rn(ws, DA[b >> 3]), __int2float_rn(s), acc);
    }
    y[r] = acc;
}

// q4i v3: v2 plus (a) dynamic shared memory sized to the real column count, so
// more CTAs (warps) are resident to hide memory latency, and (b) the next block's
// weights and scale are loaded before the current block is computed.
// Shared memory bytes = cols*2 (int16 activations) + nb*4 (block sums) + (cols/256)*4.
extern "C" __global__ void __launch_bounds__(256) q4i_gemv3(const uint4 *__restrict__ w,
        const float *__restrict__ sc, const uint8_t *__restrict__ xq,
        float *__restrict__ y, int rows, int cols) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int nb = cols >> 5, tid = threadIdx.y * 32 + threadIdx.x;
    short *A = (short *)smem;
    int *SA = (int *)(smem + cols * 2);
    float *DA = (float *)(smem + cols * 2 + nb * 4);
    for (int i = tid; i < cols; i += 256) A[i] = (short)(int8_t)xq[(i >> 8) * 292 + 4 + (i & 255)];
    for (int i = tid; i < (cols >> 8); i += 256) DA[i] = *(const float *)(xq + i * 292);
    __syncthreads();
    for (int b = tid; b < nb; b += 256) {
        int s = 0;
        for (int j = 0; j < 32; j++) s += A[b * 32 + j];
        SA[b] = s;
    }
    __syncthreads();
    const int tile = blockIdx.x * 8 + threadIdx.y;
    const int r = tile * 32 + threadIdx.x;
    if (r >= rows) return;
    const uint4 *wt = w + (size_t)tile * nb * 32 + threadIdx.x;
    const float *st = sc + (size_t)tile * nb * 32 + threadIdx.x;
    const uint4 *A4 = (const uint4 *)A;
    float acc = 0.0f;
    uint4 qn = wt[0];
    float wsn = st[0];
    for (int b = 0; b < nb; b++) {
        const uint4 q = qn;
        const float ws = wsn;
        if (b + 1 < nb) { qn = wt[(size_t)(b + 1) * 32]; wsn = st[(size_t)(b + 1) * 32]; }
        const uint4 a0 = A4[b * 4 + 0], a1 = A4[b * 4 + 1], a2 = A4[b * 4 + 2], a3 = A4[b * 4 + 3];
        const unsigned lo[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        const unsigned hi[8] = {a2.x, a2.y, a2.z, a2.w, a3.x, a3.y, a3.z, a3.w};
        const unsigned qw[4] = {q.x, q.y, q.z, q.w};
        int s = -8 * SA[b];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const int e = 4 * k + j;
                const unsigned pl = lo[e >> 1], ph = hi[e >> 1];
                const int al = (e & 1) ? ((int)pl >> 16) : (int)(short)(pl & 0xffffu);
                const int ah = (e & 1) ? ((int)ph >> 16) : (int)(short)(ph & 0xffffu);
                s += (int)((qw[k] >> (8 * j)) & 15u) * al;
                s += (int)((qw[k] >> (8 * j + 4)) & 15u) * ah;
            }
        }
        acc = __fmaf_rn(__fmul_rn(ws, DA[b >> 3]), __int2float_rn(s), acc);
    }
    y[r] = acc;
}

// q4i v4: v3 with the weight/scale loads issued PF blocks ahead (a register ring),
// for more memory-level parallelism when few rows (warps) exist, e.g. down
// (1536 rows) at 12288 columns. Same arithmetic, same block order per row.
#define Q4I_PF 4
extern "C" __global__ void __launch_bounds__(256) q4i_gemv4(const uint4 *__restrict__ w,
        const float *__restrict__ sc, const uint8_t *__restrict__ xq,
        float *__restrict__ y, int rows, int cols) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int nb = cols >> 5, tid = threadIdx.y * 32 + threadIdx.x;
    short *A = (short *)smem;
    int *SA = (int *)(smem + cols * 2);
    float *DA = (float *)(smem + cols * 2 + nb * 4);
    for (int i = tid; i < cols; i += 256) A[i] = (short)(int8_t)xq[(i >> 8) * 292 + 4 + (i & 255)];
    for (int i = tid; i < (cols >> 8); i += 256) DA[i] = *(const float *)(xq + i * 292);
    __syncthreads();
    for (int b = tid; b < nb; b += 256) {
        int s = 0;
        for (int j = 0; j < 32; j++) s += A[b * 32 + j];
        SA[b] = s;
    }
    __syncthreads();
    const int tile = blockIdx.x * 8 + threadIdx.y;
    const int r = tile * 32 + threadIdx.x;
    if (r >= rows) return;
    const uint4 *wt = w + (size_t)tile * nb * 32 + threadIdx.x;
    const float *st = sc + (size_t)tile * nb * 32 + threadIdx.x;
    const uint4 *A4 = (const uint4 *)A;
    uint4 qr[Q4I_PF];
    float sr[Q4I_PF];
    #pragma unroll
    for (int p = 0; p < Q4I_PF; p++) if (p < nb) { qr[p] = wt[(size_t)p * 32]; sr[p] = st[(size_t)p * 32]; }
    float acc = 0.0f;
    for (int b0 = 0; b0 < nb; b0 += Q4I_PF) {
        #pragma unroll
        for (int p = 0; p < Q4I_PF; p++) {
            const int b = b0 + p;
            if (b < nb) {
                const uint4 q = qr[p];
                const float ws = sr[p];
                if (b + Q4I_PF < nb) { qr[p] = wt[(size_t)(b + Q4I_PF) * 32]; sr[p] = st[(size_t)(b + Q4I_PF) * 32]; }
                const uint4 a0 = A4[b * 4 + 0], a1 = A4[b * 4 + 1], a2 = A4[b * 4 + 2], a3 = A4[b * 4 + 3];
                const unsigned lo[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                const unsigned hi[8] = {a2.x, a2.y, a2.z, a2.w, a3.x, a3.y, a3.z, a3.w};
                const unsigned qw[4] = {q.x, q.y, q.z, q.w};
                int s = -8 * SA[b];
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const int e = 4 * k + j;
                        const unsigned pl = lo[e >> 1], ph = hi[e >> 1];
                        const int al = (e & 1) ? ((int)pl >> 16) : (int)(short)(pl & 0xffffu);
                        const int ah = (e & 1) ? ((int)ph >> 16) : (int)(short)(ph & 0xffffu);
                        s += (int)((qw[k] >> (8 * j)) & 15u) * al;
                        s += (int)((qw[k] >> (8 * j + 4)) & 15u) * ah;
                    }
                }
                acc = __fmaf_rn(__fmul_rn(ws, DA[b >> 3]), __int2float_rn(s), acc);
            }
        }
    }
    y[r] = acc;
}

// ------------------------------------------------------------------------------
// Q6_K tied head, row-per-lane ("q6i"): same arithmetic as gemv_q6k_rows in
// g4ref.c - per 256-block, isum = sum_j sc_j * sum_{t in sub-block j} (q_t-32)*a_t
// (exact int32), then acc = fma(d_w * d_a, (float)isum, acc) over blocks in order.
// Layout (built at load): rows in tiles of 32; for tile t, block b, the block's
// 210 bytes (padded to 224 = 14 x 16) are stored as [chunk k = 0..13][lane 0..31]
// x 16 B, so warp load k reads 512 contiguous bytes. A lane owns one row.
// Values are decoded four per u32 (one ql word, one qh word -> 4 six-bit bytes);
// the four values of a word share one sub-block. The -32 offset is applied per
// sub-block with precomputed activation sums.
extern "C" __global__ void __launch_bounds__(256) q6i_gemv(const uint4 *__restrict__ w,
        const uint8_t *__restrict__ xq, float *__restrict__ y, int rows) {
    __shared__ __align__(16) short A[1536];
    __shared__ int SA[96];                   // 6 blocks x 16 sub-blocks
    __shared__ float DA[6];
    const int tid = threadIdx.y * 32 + threadIdx.x;
    for (int i = tid; i < 1536; i += 256) A[i] = (short)(int8_t)xq[(i >> 8) * 292 + 4 + (i & 255)];
    if (tid < 6) DA[tid] = *(const float *)(xq + tid * 292);
    __syncthreads();
    if (tid < 96) {
        // sub-block j of block b holds t = 128n + 32k + l with j = 8n + 2k + l/16
        const int b = tid >> 4, j = tid & 15, n = j >> 3, k = (j >> 1) & 3, lh = j & 1;
        int s = 0;
        for (int l = 16 * lh; l < 16 * lh + 16; l++) s += A[b * 256 + 128 * n + 32 * k + l];
        SA[tid] = s;
    }
    __syncthreads();
    const int tile = blockIdx.x * 8 + threadIdx.y;
    const int r = tile * 32 + threadIdx.x;
    if (r >= rows) return;
    const uint4 *wt = w + (size_t)tile * 6 * 14 * 32 + threadIdx.x;
    float acc = 0.0f;
    for (int b = 0; b < 6; b++) {
        uint4 c[14];
        #pragma unroll
        for (int k = 0; k < 14; k++) c[k] = wt[((size_t)b * 14 + k) * 32];
        const unsigned *u = (const unsigned *)c;      // the block's bytes as 56 words
        // ql: words 0..31 (128 B), qh: words 32..47 (64 B), scales: words 48..51, d: word 52 low half
        int p[16];
        #pragma unroll
        for (int j = 0; j < 16; j++) p[j] = 0;
        const uint2 *A2 = (const uint2 *)(A + b * 256);
        #pragma unroll
        for (int n = 0; n < 2; n++) {
            #pragma unroll
            for (int lg = 0; lg < 8; lg++) {
                const unsigned L0 = u[16 * n + lg], L1 = u[16 * n + 8 + lg], Hh = u[32 + 8 * n + lg];
                const unsigned qk[4] = {
                    (L0 & 0x0F0F0F0Fu) | ((Hh << 4) & 0x30303030u),
                    (L1 & 0x0F0F0F0Fu) | ((Hh << 2) & 0x30303030u),
                    ((L0 >> 4) & 0x0F0F0F0Fu) | (Hh & 0x30303030u),
                    ((L1 >> 4) & 0x0F0F0F0Fu) | ((Hh >> 2) & 0x30303030u)};
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    const uint2 av = A2[(128 * n + 32 * k + 4 * lg) >> 2];   // 4 int16 activations
                    const int a0 = (int)(short)(av.x & 0xffffu), a1 = (int)av.x >> 16;
                    const int a2 = (int)(short)(av.y & 0xffffu), a3 = (int)av.y >> 16;
                    const unsigned q = qk[k];
                    const int j = 8 * n + 2 * k + (lg >> 2);
                    p[j] += (int)(q & 0xffu) * a0 + (int)((q >> 8) & 0xffu) * a1
                          + (int)((q >> 16) & 0xffu) * a2 + (int)(q >> 24) * a3;
                }
            }
        }
        const int8_t *scb = (const int8_t *)(u + 48);
        int s = 0;
        #pragma unroll
        for (int j = 0; j < 16; j++) s += (int)scb[j] * (p[j] - 32 * SA[b * 16 + j]);
        const float dw = __half2float(*(const __half *)(u + 52));
        acc = __fmaf_rn(__fmul_rn(dw, DA[b]), __int2float_rn(s), acc);
    }
    y[r] = acc;
}

// ==============================================================================
// Batched kernels for the prefill: B <= 8 tokens per weight pass. For each token
// the arithmetic is exactly the single-token kernel's (same integer sums, same
// fma chain over blocks in ascending order); only the weight reads are shared.
#define BMAX 8

// q4i_gemv_b: q4i layout. xq: token t at xq + t*xstride. y: token t at y + t*ystride.
// Activations are staged in shared memory one 1024-column chunk at a time
// (B x 1024 int16 + block sums), so any column count fits; the per-token float
// chain simply continues across chunks. Nibbles are decoded once per block and
// reused for all B tokens.
#define Q4B_KC 1024
extern "C" __global__ void __launch_bounds__(256) q4i_gemv_b(const uint4 *__restrict__ w,
        const float *__restrict__ sc, const uint8_t *__restrict__ xq, float *__restrict__ y,
        int rows, int cols, int B, int xstride, int ystride) {
    __shared__ __align__(16) short A[BMAX * Q4B_KC];
    __shared__ int SA[BMAX * (Q4B_KC / 32)];
    __shared__ float DA[BMAX * (Q4I_MAXCOLS / 256)];
    const int nb = cols >> 5, nd = cols >> 8, tid = threadIdx.y * 32 + threadIdx.x;
    for (int i = tid; i < B * nd; i += 256) DA[i] = *(const float *)(xq + (i / nd) * xstride + (i % nd) * 292);
    const int tile = blockIdx.x * 8 + threadIdx.y;
    const int r = tile * 32 + threadIdx.x;
    const bool live = r < rows;
    const uint4 *wt = w + (size_t)tile * nb * 32 + threadIdx.x;
    const float *st = sc + (size_t)tile * nb * 32 + threadIdx.x;
    float acc[BMAX];
    #pragma unroll
    for (int t = 0; t < BMAX; t++) acc[t] = 0.0f;
    for (int c0 = 0; c0 < cols; c0 += Q4B_KC) {
        const int kc = min(Q4B_KC, cols - c0), kb = kc >> 5;
        __syncthreads();
        for (int i = tid; i < B * kc; i += 256) {
            const int t = i / kc, c = c0 + i % kc;
            A[t * Q4B_KC + (i % kc)] = (short)(int8_t)xq[t * xstride + (c >> 8) * 292 + 4 + (c & 255)];
        }
        __syncthreads();
        for (int i = tid; i < B * kb; i += 256) {
            const int t = i / kb, bl = i % kb;
            int s = 0;
            for (int j = 0; j < 32; j++) s += A[t * Q4B_KC + bl * 32 + j];
            SA[t * (Q4B_KC / 32) + bl] = s;
        }
        __syncthreads();
        if (live) {
            for (int bl = 0; bl < kb; bl++) {
                const int b = (c0 >> 5) + bl;
                const uint4 q = wt[(size_t)b * 32];
                const float ws = st[(size_t)b * 32];
                const unsigned qw[4] = {q.x, q.y, q.z, q.w};
                int nl[16], nh[16];
                #pragma unroll
                for (int e = 0; e < 16; e++) {
                    nl[e] = (int)((qw[e >> 2] >> (8 * (e & 3))) & 15u);
                    nh[e] = (int)((qw[e >> 2] >> (8 * (e & 3) + 4)) & 15u);
                }
                #pragma unroll
                for (int t = 0; t < BMAX; t++) {
                    if (t < B) {
                        const uint4 *A4 = (const uint4 *)(A + t * Q4B_KC + bl * 32);
                        const uint4 a0 = A4[0], a1 = A4[1], a2 = A4[2], a3 = A4[3];
                        const unsigned lo[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                        const unsigned hi[8] = {a2.x, a2.y, a2.z, a2.w, a3.x, a3.y, a3.z, a3.w};
                        int s = -8 * SA[t * (Q4B_KC / 32) + bl];
                        #pragma unroll
                        for (int e = 0; e < 16; e++) {
                            const unsigned pl = lo[e >> 1], ph = hi[e >> 1];
                            const int al = (e & 1) ? ((int)pl >> 16) : (int)(short)(pl & 0xffffu);
                            const int ah = (e & 1) ? ((int)ph >> 16) : (int)(short)(ph & 0xffffu);
                            s += nl[e] * al + nh[e] * ah;
                        }
                        acc[t] = __fmaf_rn(__fmul_rn(ws, DA[t * nd + (b >> 3)]), __int2float_rn(s), acc[t]);
                    }
                }
            }
        }
    }
    if (live) {
        #pragma unroll
        for (int t = 0; t < BMAX; t++) if (t < B) y[(size_t)t * ystride + r] = acc[t];
    }
}

// q6i_gemv_b: the q6i head for B tokens. Per block, each sub-block's 16 values are
// decoded once (four words) and dotted with every token's activations; per token
// isum = sum_j sc_j * (p_j - 32 * sum a_j), then acc = fma(dw * da, (float)isum, acc).
extern "C" __global__ void __launch_bounds__(256) q6i_gemv_b(const uint4 *__restrict__ w,
        const uint8_t *__restrict__ xq, float *__restrict__ y, int rows, int B, int xstride, int ystride) {
    __shared__ __align__(16) short A[BMAX * 1536];
    __shared__ int SA[BMAX * 96];
    __shared__ float DA[BMAX * 6];
    const int tid = threadIdx.y * 32 + threadIdx.x;
    for (int i = tid; i < B * 1536; i += 256) {
        const int t = i / 1536, c = i % 1536;
        A[i] = (short)(int8_t)xq[t * xstride + (c >> 8) * 292 + 4 + (c & 255)];
    }
    for (int i = tid; i < B * 6; i += 256) DA[i] = *(const float *)(xq + (i / 6) * xstride + (i % 6) * 292);
    __syncthreads();
    for (int i = tid; i < B * 96; i += 256) {
        const int t = i / 96, bj = i % 96, b = bj >> 4, j = bj & 15, n = j >> 3, k = (j >> 1) & 3, lh = j & 1;
        int s = 0;
        for (int l = 16 * lh; l < 16 * lh + 16; l++) s += A[t * 1536 + b * 256 + 128 * n + 32 * k + l];
        SA[i] = s;
    }
    __syncthreads();
    const int tile = blockIdx.x * 8 + threadIdx.y;
    const int r = tile * 32 + threadIdx.x;
    if (r >= rows) return;
    const uint4 *wt = w + (size_t)tile * 6 * 14 * 32 + threadIdx.x;
    float acc[BMAX];
    #pragma unroll
    for (int t = 0; t < BMAX; t++) acc[t] = 0.0f;
    for (int b = 0; b < 6; b++) {
        uint4 c[14];
        #pragma unroll
        for (int k = 0; k < 14; k++) c[k] = wt[((size_t)b * 14 + k) * 32];
        const unsigned *u = (const unsigned *)c;
        const int8_t *scb = (const int8_t *)(u + 48);
        int s[BMAX];
        #pragma unroll
        for (int t = 0; t < BMAX; t++) s[t] = 0;
        #pragma unroll
        for (int n = 0; n < 2; n++) {
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                #pragma unroll
                for (int lh = 0; lh < 2; lh++) {
                    const int j = 8 * n + 2 * k + lh;
                    unsigned q[4];
                    #pragma unroll
                    for (int g = 0; g < 4; g++) {
                        const int lg = 4 * lh + g;
                        const unsigned L0 = u[16 * n + lg], L1 = u[16 * n + 8 + lg], Hh = u[32 + 8 * n + lg];
                        q[g] = k == 0 ? ((L0 & 0x0F0F0F0Fu) | ((Hh << 4) & 0x30303030u))
                             : k == 1 ? ((L1 & 0x0F0F0F0Fu) | ((Hh << 2) & 0x30303030u))
                             : k == 2 ? (((L0 >> 4) & 0x0F0F0F0Fu) | (Hh & 0x30303030u))
                                      : (((L1 >> 4) & 0x0F0F0F0Fu) | ((Hh >> 2) & 0x30303030u));
                    }
                    #pragma unroll
                    for (int t = 0; t < BMAX; t++) {
                        if (t < B) {
                            const uint2 *A2 = (const uint2 *)(A + t * 1536 + b * 256 + 128 * n + 32 * k + 16 * lh);
                            int p = 0;
                            #pragma unroll
                            for (int g = 0; g < 4; g++) {
                                const uint2 av = A2[g];
                                p += (int)(q[g] & 0xffu) * (int)(short)(av.x & 0xffffu)
                                   + (int)((q[g] >> 8) & 0xffu) * ((int)av.x >> 16)
                                   + (int)((q[g] >> 16) & 0xffu) * (int)(short)(av.y & 0xffffu)
                                   + (int)(q[g] >> 24) * ((int)av.y >> 16);
                            }
                            s[t] += (int)scb[j] * (p - 32 * SA[t * 96 + b * 16 + j]);
                        }
                    }
                }
            }
        }
        const float dw = __half2float(*(const __half *)(u + 52));
        #pragma unroll
        for (int t = 0; t < BMAX; t++) if (t < B) acc[t] = __fmaf_rn(__fmul_rn(dw, DA[t * 6 + b]), __int2float_rn(s[t]), acc[t]);
    }
    #pragma unroll
    for (int t = 0; t < BMAX; t++) if (t < B) y[(size_t)t * ystride + r] = acc[t];
}

// GeGLU for B tokens: y is [token][gate (n) | up (n)], h is [token][n].
extern "C" __global__ void geglu_b(const float *__restrict__ y, float *__restrict__ h,
                                   int n, int B, const double *__restrict__ tab) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * B) return;
    const int t = i / n, j = i % n;
    const float g = y[(size_t)t * 2 * n + j], uu = y[(size_t)t * 2 * n + n + j];
    const double x = (double)g;
    const double x3 = __dmul_rn(__dmul_rn(x, x), x);
    const double tt = -__dadd_rn(__dmul_rn(tab[21], x), __dmul_rn(tab[22], x3));
    const double e = g4_exp_dev(tt, tab);
    const float gl = __double2float_rn(__ddiv_rn(x, __dadd_rn(1.0, e)));
    h[i] = __fmul_rn(gl, uu);
}

// Per-token argmax of the raw logits (block t handles y + t*n); same first-max rule
// as the host loop. out[2t] = value bits, out[2t+1] = index.
extern "C" __global__ void __launch_bounds__(1024) argmax_b(const float *__restrict__ y, int n,
                                                            unsigned *__restrict__ out) {
    __shared__ float sv[1024];
    __shared__ int si[1024];
    const int t = threadIdx.x;
    const float *yy = y + (size_t)blockIdx.x * n;
    float best = -__int_as_float(0x7f800000);
    int bi = 0x7fffffff;
    for (int i = t; i < n; i += 1024) { const float v = yy[i]; if (v > best) { best = v; bi = i; } }
    sv[t] = best; si[t] = bi;
    __syncthreads();
    for (int s = 512; s; s >>= 1) {
        if (t < s) {
            const float a = sv[t], b = sv[t + s];
            const int ia = si[t], ib = si[t + s];
            if (b > a || (b == a && ib < ia)) { sv[t] = b; si[t] = ib; }
        }
        __syncthreads();
    }
    if (t == 0) { out[2 * blockIdx.x] = __float_as_uint(sv[0]); out[2 * blockIdx.x + 1] = (unsigned)si[0]; }
}
