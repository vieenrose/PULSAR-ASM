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

// q4i v5: v4 with paired nibble extraction. A word masked with 0x000F000F holds two
// nibbles, one per 16-bit half, so 4 masks (3 shifts) yield the 8 nibbles of a word
// (v4: one shift/BFE per nibble). The activations are stored permuted so that each
// 32-bit smem word pairs, half by half, with one masked word: for weight word k
// (bytes 4k..4k+3) the words are (a[4k], a[4k+2]), (a[16+4k], a[16+4k+2]),
// (a[4k+1], a[4k+3]), (a[17+4k], a[19+4k]). Each product q*a is the same integer as
// in v4 and int32 addition is exact in any order, so s (and y) are bit-identical.
// s + lo16(m)*lo16(a) + hi16(m)*hi16(a), signed 16-bit halves (exact int32):
// mad.wide.s16 on register halves maps to XMAD with half selects, no unpacking
__device__ __forceinline__ int dot2_s16(unsigned m, unsigned a, int s) {
    asm("{\n\t.reg .s16 a0, a1, q0, q1;\n\tmov.b32 {a0, a1}, %1;\n\tmov.b32 {q0, q1}, %2;\n\t"
        "mad.wide.s16 %0, q0, a0, %0;\n\tmad.wide.s16 %0, q1, a1, %0;\n\t}" : "+r"(s) : "r"(a), "r"(m));
    return s;
}
// Activations are staged Q4I5_KC columns at a time (8 KB of permuted pairs): with the
// whole row staged, a 12288-column down projection took 24.6 KB of shared memory and
// only two blocks fit on the SM. The weight prefetch ring runs on across chunks; each
// row's float chain still visits the blocks in ascending order. nb % Q4I_PF == 0.
#define Q4I5_KC 4096
extern "C" __global__ void __launch_bounds__(256) q4i_gemv5(const uint4 *__restrict__ w,
        const unsigned short *__restrict__ sc, const uint8_t *__restrict__ xq,
        float *__restrict__ y, int rows, int cols) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int nb = cols >> 5, tid = threadIdx.y * 32 + threadIdx.x;
    const int kcw = min(cols, Q4I5_KC), kbw = kcw >> 5;
    unsigned *AW = (unsigned *)smem;                  /* kbw * 16 permuted int16 pairs */
    int *SA = (int *)(smem + kcw * 2);
    float *DA = (float *)(smem + kcw * 2 + kbw * 4);
    for (int i = tid; i < (cols >> 8); i += 256) DA[i] = *(const float *)(xq + i * 292);
    const int tile = blockIdx.x * 8 + threadIdx.y;
    const int r = tile * 32 + threadIdx.x;
    const bool live = r < rows;
    const uint4 *wt = w + (size_t)tile * nb * 32 + threadIdx.x;
    const unsigned short *st = sc + (size_t)tile * nb * 32 + threadIdx.x;   /* fp16 block scales */
    const uint4 *A4 = (const uint4 *)AW;
    uint4 qr[Q4I_PF];
    unsigned short sr[Q4I_PF];
    #pragma unroll
    for (int p = 0; p < Q4I_PF; p++) if (live && p < nb) { qr[p] = wt[(size_t)p * 32]; sr[p] = st[(size_t)p * 32]; }
    float acc = 0.0f;
    for (int cb = 0; cb < nb; cb += kbw) {            /* chunk of blocks [cb, cb + kb) */
        const int kb = min(kbw, nb - cb);
        __syncthreads();
        for (int i = tid; i < kb * 16; i += 256) {
            const int bl = i >> 4, k = (i >> 2) & 3, c = i & 3;
            const int e0 = (cb + bl) * 32 + ((c & 1) ? 16 : 0) + 4 * k + (c >> 1);
            const int a0 = (int8_t)xq[(e0 >> 8) * 292 + 4 + (e0 & 255)];
            const int a1 = (int8_t)xq[((e0 + 2) >> 8) * 292 + 4 + ((e0 + 2) & 255)];
            AW[i] = (unsigned)(a0 & 0xffff) | ((unsigned)a1 << 16);
        }
        __syncthreads();
        for (int bl = tid; bl < kb; bl += 256) {
            int s = 0;
            for (int j = 0; j < 16; j++) { const unsigned v = AW[bl * 16 + j]; s += (int)(short)(v & 0xffffu) + ((int)v >> 16); }
            SA[bl] = s;
        }
        __syncthreads();
        if (!live) continue;
        for (int b0 = cb; b0 < cb + kb; b0 += Q4I_PF) {
            #pragma unroll
            for (int p = 0; p < Q4I_PF; p++) {
                const int b = b0 + p, bl = b - cb;
                const uint4 q = qr[p];
                const float ws = __half2float(__ushort_as_half(sr[p]));   /* exact */
                if (b + Q4I_PF < nb) { qr[p] = wt[(size_t)(b + Q4I_PF) * 32]; sr[p] = st[(size_t)(b + Q4I_PF) * 32]; }
                const unsigned qw[4] = {q.x, q.y, q.z, q.w};
                int s = -8 * SA[bl];
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    const uint4 a = A4[bl * 4 + k];
                    const unsigned m0 = qw[k] & 0x000F000Fu, m1 = (qw[k] >> 4) & 0x000F000Fu;
                    const unsigned m2 = (qw[k] >> 8) & 0x000F000Fu, m3 = (qw[k] >> 12) & 0x000F000Fu;
                    s = dot2_s16(m0, a.x, s);
                    s = dot2_s16(m1, a.y, s);
                    s = dot2_s16(m2, a.z, s);
                    s = dot2_s16(m3, a.w, s);
                }
                acc = __fmaf_rn(__fmul_rn(ws, DA[b >> 3]), __int2float_rn(s), acc);
            }
        }
    }
    if (live) y[r] = acc;
}

// q6i activation order in shared memory: element i of each group of 4 stored at
// position (a0, a2, a1, a3); sub-block sums (16 aligned elements) are unchanged
#define Q6P(i) (((i) & ~3) | (((i) & 1) << 1) | (((i) >> 1) & 1))
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
    // stored permuted within each group of 4 as (a0, a2, a1, a3): word x of a uint2
    // pairs with bytes 0, 2 of a value word, word y with bytes 1, 3 (dot2_s16)
    for (int i = tid; i < 1536; i += 256) A[Q6P(i)] = (short)(int8_t)xq[(i >> 8) * 292 + 4 + (i & 255)];
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
                    const uint2 av = A2[(128 * n + 32 * k + 4 * lg) >> 2];   // (a0,a2), (a1,a3)
                    const unsigned q = qk[k];
                    const int j = 8 * n + 2 * k + (lg >> 2);
                    p[j] = dot2_s16(q & 0x00FF00FFu, av.x, p[j]);
                    p[j] = dot2_s16((q >> 8) & 0x00FF00FFu, av.y, p[j]);
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

// q4i_gemv_b2: q4i_gemv_b with v5's paired nibbles. Per block the 16 masked words
// (two nibbles each, see q4i_gemv5) are computed once and dotted with every token's
// permuted activation words (two XMADs per word); same integer products, exact int32
// sums, same float chain per token, so y is bit-identical to q4i_gemv_b.
extern "C" __global__ void __launch_bounds__(256) q4i_gemv_b2(const uint4 *__restrict__ w,
        const unsigned short *__restrict__ sc, const uint8_t *__restrict__ xq, float *__restrict__ y,
        int rows, int cols, int B, int xstride, int ystride) {
    __shared__ __align__(16) unsigned AW[BMAX * Q4B_KC / 2];
    __shared__ int SA[BMAX * (Q4B_KC / 32)];
    __shared__ float DA[BMAX * (Q4I_MAXCOLS / 256)];
    const int nb = cols >> 5, nd = cols >> 8, tid = threadIdx.y * 32 + threadIdx.x;
    // slots t >= B are computed on zeros (and never stored): the token loop below then has
    // no data-dependent branch, so the 8 independent XMAD chains can interleave
    for (int i = tid; i < BMAX * nd; i += 256) DA[i] = i < B * nd ? *(const float *)(xq + (i / nd) * xstride + (i % nd) * 292) : 0.0f;
    const int tile = blockIdx.x * 8 + threadIdx.y;
    const int r = tile * 32 + threadIdx.x;
    const bool live = r < rows;
    const uint4 *wt = w + (size_t)tile * nb * 32 + threadIdx.x;
    const unsigned short *st = sc + (size_t)tile * nb * 32 + threadIdx.x;   /* fp16 block scales */
    float acc[BMAX];
    #pragma unroll
    for (int t = 0; t < BMAX; t++) acc[t] = 0.0f;
    for (int c0 = 0; c0 < cols; c0 += Q4B_KC) {
        const int kc = min(Q4B_KC, cols - c0), kb = kc >> 5, kw = kc >> 1;
        __syncthreads();
        for (int i = tid; i < BMAX * kw; i += 256) {
            const int t = i / kw, wi = i % kw, bl = wi >> 4, k = (wi >> 2) & 3, cc = wi & 3;
            if (t >= B) { AW[t * (Q4B_KC / 2) + wi] = 0u; continue; }
            const int e0 = c0 + bl * 32 + ((cc & 1) ? 16 : 0) + 4 * k + (cc >> 1), e1 = e0 + 2;
            const uint8_t *xt = xq + t * xstride;
            const int a0 = (int8_t)xt[(e0 >> 8) * 292 + 4 + (e0 & 255)];
            const int a1 = (int8_t)xt[(e1 >> 8) * 292 + 4 + (e1 & 255)];
            AW[t * (Q4B_KC / 2) + wi] = (unsigned)(a0 & 0xffff) | ((unsigned)a1 << 16);
        }
        __syncthreads();
        for (int i = tid; i < BMAX * kb; i += 256) {
            const int t = i / kb, bl = i % kb;
            int s = 0;
            for (int j = 0; j < 16; j++) { const unsigned v = AW[t * (Q4B_KC / 2) + bl * 16 + j]; s += (int)(short)(v & 0xffffu) + ((int)v >> 16); }
            SA[t * (Q4B_KC / 32) + bl] = s;
        }
        __syncthreads();
        if (live) {
            for (int bl = 0; bl < kb; bl++) {
                const int b = (c0 >> 5) + bl;
                const uint4 q = wt[(size_t)b * 32];
                const float ws = __half2float(__ushort_as_half(st[(size_t)b * 32]));   /* exact */
                const unsigned qw[4] = {q.x, q.y, q.z, q.w};
                unsigned m[16];
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    m[4 * k + 0] = qw[k] & 0x000F000Fu;         m[4 * k + 1] = (qw[k] >> 4) & 0x000F000Fu;
                    m[4 * k + 2] = (qw[k] >> 8) & 0x000F000Fu;  m[4 * k + 3] = (qw[k] >> 12) & 0x000F000Fu;
                }
                #pragma unroll
                for (int t = 0; t < BMAX; t++) {
                    const uint4 *A4 = (const uint4 *)(AW + t * (Q4B_KC / 2) + bl * 16);
                    int s = -8 * SA[t * (Q4B_KC / 32) + bl];
                    #pragma unroll
                    for (int k = 0; k < 4; k++) {
                        const uint4 a = A4[k];
                        s = dot2_s16(m[4 * k + 0], a.x, s); s = dot2_s16(m[4 * k + 1], a.y, s);
                        s = dot2_s16(m[4 * k + 2], a.z, s); s = dot2_s16(m[4 * k + 3], a.w, s);
                    }
                    acc[t] = __fmaf_rn(__fmul_rn(ws, DA[t * nd + (b >> 3)]), __int2float_rn(s), acc[t]);
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
    // slots t >= B hold zeros and are never stored: no data-dependent branch in the
    // token loops, so the per-token chains interleave (as q4i_gemv_b2)
    for (int i = tid; i < BMAX * 1536; i += 256) {
        const int t = i / 1536, c = i % 1536;
        if (t >= B) { A[i] = 0; continue; }
        A[t * 1536 + Q6P(c)] = (short)(int8_t)xq[t * xstride + (c >> 8) * 292 + 4 + (c & 255)];   // as q6i_gemv
    }
    for (int i = tid; i < BMAX * 6; i += 256) DA[i] = i < B * 6 ? *(const float *)(xq + (i / 6) * xstride + (i % 6) * 292) : 0.0f;
    __syncthreads();
    for (int i = tid; i < BMAX * 96; i += 256) {
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
                        const uint2 *A2 = (const uint2 *)(A + t * 1536 + b * 256 + 128 * n + 32 * k + 16 * lh);
                        int p = 0;
                        #pragma unroll
                        for (int g = 0; g < 4; g++) {
                            const uint2 av = A2[g];
                            p = dot2_s16(q[g] & 0x00FF00FFu, av.x, p);
                            p = dot2_s16((q[g] >> 8) & 0x00FF00FFu, av.y, p);
                        }
                        s[t] += (int)scb[j] * (p - 32 * SA[t * 96 + b * 16 + j]);
                    }
                }
            }
        }
        const float dw = __half2float(*(const __half *)(u + 52));
        #pragma unroll
        for (int t = 0; t < BMAX; t++) acc[t] = __fmaf_rn(__fmul_rn(dw, DA[t * 6 + b]), __int2float_rn(s[t]), acc[t]);
    }
    #pragma unroll
    for (int t = 0; t < BMAX; t++) if (t < B) y[(size_t)t * ystride + r] = acc[t];
}

// f16_gemv_t2: f16_gemv_t with two adjacent rows per thread (one half2 load per k
// covers both: a warp reads 128 contiguous bytes) and x staged in shared memory.
// Each row keeps f16_gemv_t's chain: acc = fma(w[k][r], x[k], acc), k ascending.
// rows must be even.
extern "C" __global__ void __launch_bounds__(256) f16_gemv_t2(const __half2 *__restrict__ wt,
        const float *__restrict__ x, float *__restrict__ y, int rows, int cols) {
    extern __shared__ float xs[];
    for (int k = threadIdx.x; k < cols; k += 256) xs[k] = x[k];
    __syncthreads();
    const int p = blockIdx.x * 256 + threadIdx.x, hr = rows >> 1;   // p: row pair
    if (p >= hr) return;
    float a0 = 0.0f, a1 = 0.0f;
    #pragma unroll 8
    for (int k = 0; k < cols; k++) {
        const float2 w = __half22float2(wt[(size_t)k * hr + p]);
        a0 = __fmaf_rn(w.x, xs[k], a0);
        a1 = __fmaf_rn(w.y, xs[k], a1);
    }
    y[2 * p] = a0; y[2 * p + 1] = a1;
}
// f16_gemv_tb: f16_gemv_t2 for B <= 8 tokens (x: token t at x + t*cols, y: token t at
// y + t*rows); x staged 512 columns at a time; slots t >= B run on zeros, never stored
#define F16B_KC 512
extern "C" __global__ void __launch_bounds__(256) f16_gemv_tb(const __half2 *__restrict__ wt,
        const float *__restrict__ x, float *__restrict__ y, int rows, int cols, int B) {
    __shared__ float xs[BMAX * F16B_KC];
    const int p = blockIdx.x * 256 + threadIdx.x, hr = rows >> 1;
    float a0[BMAX], a1[BMAX];
    #pragma unroll
    for (int t = 0; t < BMAX; t++) { a0[t] = 0.0f; a1[t] = 0.0f; }
    for (int c0 = 0; c0 < cols; c0 += F16B_KC) {
        const int kc = min(F16B_KC, cols - c0);
        __syncthreads();
        for (int i = threadIdx.x; i < BMAX * F16B_KC; i += 256) {
            const int t = i / F16B_KC, k = i % F16B_KC;
            xs[i] = (t < B && k < kc) ? x[(size_t)t * cols + c0 + k] : 0.0f;
        }
        __syncthreads();
        if (p < hr) {
            for (int k = 0; k < kc; k++) {
                const float2 w = __half22float2(wt[(size_t)(c0 + k) * hr + p]);
                #pragma unroll
                for (int t = 0; t < BMAX; t++) {
                    a0[t] = __fmaf_rn(w.x, xs[t * F16B_KC + k], a0[t]);
                    a1[t] = __fmaf_rn(w.y, xs[t * F16B_KC + k], a1[t]);
                }
            }
        }
    }
    if (p < hr) {
        #pragma unroll
        for (int t = 0; t < BMAX; t++) if (t < B) { y[(size_t)t * rows + 2 * p] = a0[t]; y[(size_t)t * rows + 2 * p + 1] = a1[t]; }
    }
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

// ==============================================================================
// Attention (all NH query heads against one kv head), exactly attn_head_ref in g4ref.c:
// s_t = fmaf chain over d ascending of k[d]*q[d] from 0; mx = max s_t;
// e_t = g4_exp((double)s_t - (double)mx); sum = serial double sum over t ascending;
// p_t = (float)(e_t / sum); o[d] = fmaf chain over t ascending of p_t*v[d] from 0.
// Token dimension blockIdx.y: token j sits at pos = pos0 + j and attends rows lo..pos,
// lo = pos + 1 - win when win > 0 and pos + 1 > win (sliding window), else 0. K and V
// point at row 0 of the cache; q, S and O are per token (strides 8*hd, sstride, 8*hd).
// Decode is the B = 1 case.
__device__ __forceinline__ void attn_range(int pos0, int win, int &pos, int &lo, int &n) {
    pos = pos0 + (int)blockIdx.y;
    lo = (win > 0 && pos + 1 > win) ? pos + 1 - win : 0;
    n = pos - lo + 1;
}
// attn_scores: one thread per (t, h); S[t * NH + h]. q is staged in shared memory
// transposed as float4 [d/4][h] (the 8 heads of a warp read 128 contiguous bytes, no
// bank conflicts) and K is read as float4; the fmaf chain stays d-ascending.
extern "C" __global__ void __launch_bounds__(256) attn_scores(const float *__restrict__ q,
        const float *__restrict__ K, float *__restrict__ S, int pos0, int win, int hd, int sstride) {
    extern __shared__ __align__(16) float4 q4[];
    int pos, lo, n;
    attn_range(pos0, win, pos, lo, n);
    const float *qt = q + (size_t)blockIdx.y * 8 * hd;
    for (int i = threadIdx.x; i < 2 * hd; i += 256) {   // i = (d/4) * 8 + h
        const int h = i & 7, d4 = i >> 3;
        q4[i] = *(const float4 *)(qt + h * hd + 4 * d4);
    }
    __syncthreads();
    const int idx = blockIdx.x * 256 + threadIdx.x, t = idx >> 3, h = idx & 7;
    if (t >= n) return;
    const float4 *k = (const float4 *)(K + (size_t)(lo + t) * hd);
    float s = 0.0f;
    for (int d4 = 0; d4 < hd / 4; d4++) {
        const float4 kv = __ldg(k + d4), qv = q4[d4 * 8 + h];
        s = __fmaf_rn(kv.x, qv.x, s); s = __fmaf_rn(kv.y, qv.y, s);
        s = __fmaf_rn(kv.z, qv.z, s); s = __fmaf_rn(kv.w, qv.w, s);
    }
    S[(size_t)blockIdx.y * sstride + idx] = s;
}
// attn_softmax_pv: one 256-thread block per (head, token). Shared: e[n] doubles, p[n] floats.
extern "C" __global__ void __launch_bounds__(256) attn_softmax_pv(const float *__restrict__ S,
        const float *__restrict__ V, float *__restrict__ O, int pos0, int win, int hd,
        const double *__restrict__ tab, int sstride) {
    extern __shared__ __align__(16) unsigned char sm[];
    int pos, lo, n;
    attn_range(pos0, win, pos, lo, n);
    S += (size_t)blockIdx.y * sstride;
    double *e = (double *)sm;
    float *p = (float *)(sm + (size_t)n * 8);
    __shared__ float red[256];
    __shared__ double sum_s;
    const int h = blockIdx.x, tid = threadIdx.x;
    float m = -INFINITY;
    for (int t = tid; t < n; t += 256) { const float s = S[t * 8 + h]; if (s > m) m = s; }
    red[tid] = m;
    __syncthreads();
    for (int w = 128; w > 0; w >>= 1) {
        if (tid < w && red[tid + w] > red[tid]) red[tid] = red[tid + w];
        __syncthreads();
    }
    const double mx = (double)red[0];
    for (int t = tid; t < n; t += 256) e[t] = g4_exp_dev(__dsub_rn((double)S[t * 8 + h], mx), tab);
    __syncthreads();
    if (tid == 0) { double s = 0.0; for (int t = 0; t < n; t++) s = __dadd_rn(s, e[t]); sum_s = s; }
    __syncthreads();
    const double sum = sum_s;
    for (int t = tid; t < n; t += 256) p[t] = __double2float_rn(__ddiv_rn(e[t], sum));
    __syncthreads();
    float *o = O + (size_t)blockIdx.y * 8 * hd + h * hd;
    for (int d = tid; d < hd; d += 256) {
        float acc = 0.0f;
        const float *v = V + (size_t)lo * hd + d;
        for (int t = 0; t < n; t++) acc = __fmaf_rn(p[t], __ldg(v + (size_t)t * hd), acc);
        o[d] = acc;
    }
}
