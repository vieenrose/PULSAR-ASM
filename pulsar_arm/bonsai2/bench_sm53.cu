// bench_sm53.cu: Jetson Nano GPU rates (sm_53, one 128-lane SM, so per SM = whole GPU): stream read,
// popc, lop3, iadd, hfma2, ffma, f32->f16->f32 round trips, i2f, dfma, shfl, lds.128.
// Ceilings for the Bonsai-2 limits in .auto/HANDOFF_B2_4K.md; run under jetson_clocks (921.6 MHz).
//   nvcc -O3 -gencode arch=compute_53,code=sm_53 -o bench_sm53 bench_sm53.cu
#include <cstdio>
#include <cuda_fp16.h>
#include <cstdint>
#define CK(x) do{cudaError_t e=(x); if(e){printf("err %s line %d\n",cudaGetErrorString(e),__LINE__);return 1;}}while(0)
__global__ void k_read(const uint4 *p, size_t n, uint32_t *out) {
    uint32_t a = 0; size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, s = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += 4 * s) {
        uint4 v0 = p[i], v1 = i + s < n ? p[i + s] : make_uint4(0,0,0,0), v2 = i + 2*s < n ? p[i + 2*s] : make_uint4(0,0,0,0), v3 = i + 3*s < n ? p[i + 3*s] : make_uint4(0,0,0,0);
        a ^= v0.x ^ v0.y ^ v0.z ^ v0.w ^ v1.x ^ v1.y ^ v1.z ^ v1.w ^ v2.x ^ v2.w ^ v3.x ^ v3.y ^ v2.y ^ v2.z ^ v3.z ^ v3.w;
    }
    if (a == 0x12345678) out[0] = a;
}
#define NI 4096
__global__ void k_popc(uint32_t *o, uint32_t s) {
    uint32_t a0 = s + threadIdx.x, a1 = a0 * 3, a2 = a0 * 5, a3 = a0 * 7, a4 = a0 * 9, a5 = a0 * 11, a6 = a0 * 13, a7 = a0 * 17;
    #pragma unroll 16
    for (int i = 0; i < NI; i++) { a0 = __popc(a0) ^ a7; a1 = __popc(a1) ^ a0; a2 = __popc(a2) ^ a1; a3 = __popc(a3) ^ a2; a4 = __popc(a4) ^ a3; a5 = __popc(a5) ^ a4; a6 = __popc(a6) ^ a5; a7 = __popc(a7) ^ a6; }
    if ((a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7) == 0x1234567) o[0] = 1;
}
__device__ __forceinline__ uint32_t l3(uint32_t a, uint32_t b, uint32_t c) { uint32_t r; asm volatile("lop3.b32 %0,%1,%2,%3,0x96;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
__global__ void k_lop3(uint32_t *o, uint32_t s) {
    uint32_t a0 = s + threadIdx.x, a1 = a0 * 3, a2 = a0 * 5, a3 = a0 * 7, a4 = a0 * 9, a5 = a0 * 11, a6 = a0 * 13, a7 = a0 * 17;
    #pragma unroll 16
    for (int i = 0; i < NI; i++) { a0 = l3(a0, a7, a1); a1 = l3(a1, a0, a2); a2 = l3(a2, a1, a3); a3 = l3(a3, a2, a4); a4 = l3(a4, a3, a5); a5 = l3(a5, a4, a6); a6 = l3(a6, a5, a7); a7 = l3(a7, a6, a0); }
    if ((a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7) == 0x1234567) o[0] = 1;
}
__global__ void k_iadd(uint32_t *o, uint32_t s) {
    uint32_t a0 = s + threadIdx.x, a1 = a0 * 3, a2 = a0 * 5, a3 = a0 * 7, a4 = a0 * 9, a5 = a0 * 11, a6 = a0 * 13, a7 = a0 * 17;
    #pragma unroll 16
    for (int i = 0; i < NI; i++) { a0 += a7; a1 += a0; a2 += a1; a3 += a2; a4 += a3; a5 += a4; a6 += a5; a7 += a6; }
    if ((a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7) == 0x1234567) o[0] = 1;
}
__global__ void k_hfma2(uint32_t *o, float s) {
    __half2 a0 = __float2half2_rn(s + threadIdx.x * 1e-3f), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0, m = __float2half2_rn(0.999f), c = __float2half2_rn(1e-3f);
    #pragma unroll 16
    for (int i = 0; i < NI; i++) { a0 = __hfma2(a0, m, c); a1 = __hfma2(a1, m, c); a2 = __hfma2(a2, m, c); a3 = __hfma2(a3, m, c); a4 = __hfma2(a4, m, c); a5 = __hfma2(a5, m, c); a6 = __hfma2(a6, m, c); a7 = __hfma2(a7, m, c); }
    __half2 t = __hadd2(__hadd2(__hadd2(a0, a1), __hadd2(a2, a3)), __hadd2(__hadd2(a4, a5), __hadd2(a6, a7)));
    if (__low2float(t) == 1234.5f) o[0] = 1;
}
__global__ void k_ffma(uint32_t *o, float s) {
    float a0 = s + threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
    #pragma unroll 16
    for (int i = 0; i < NI; i++) { a0 = fmaf(a0, 0.999f, 1e-3f); a1 = fmaf(a1, 0.999f, 1e-3f); a2 = fmaf(a2, 0.999f, 1e-3f); a3 = fmaf(a3, 0.999f, 1e-3f); a4 = fmaf(a4, 0.999f, 1e-3f); a5 = fmaf(a5, 0.999f, 1e-3f); a6 = fmaf(a6, 0.999f, 1e-3f); a7 = fmaf(a7, 0.999f, 1e-3f); }
    if (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 == 1234.5f) o[0] = 1;
}
// f32 -> f16 -> f32 round trip: two conversions per step (F2F.F16.F32 + HADD2.F32 or F2F.F32.F16)
__global__ void k_f2f(uint32_t *o, float s) {
    float a0 = s + threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
    #pragma unroll 16
    for (int i = 0; i < NI; i++) { a0 = __half2float(__float2half_rn(a0 + a7)); a1 = __half2float(__float2half_rn(a1 + a0)); a2 = __half2float(__float2half_rn(a2 + a1)); a3 = __half2float(__float2half_rn(a3 + a2)); a4 = __half2float(__float2half_rn(a4 + a3)); a5 = __half2float(__float2half_rn(a5 + a4)); a6 = __half2float(__float2half_rn(a6 + a5)); a7 = __half2float(__float2half_rn(a7 + a6)); }
    if (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 == 1234.5f) o[0] = 1;
}
__global__ void k_i2f(uint32_t *o, int s) {
    int a0 = s + threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
    #pragma unroll 16
    for (int i = 0; i < NI; i++) { a0 = __float_as_int((float)(a0 ^ a7)); a1 = __float_as_int((float)(a1 ^ a0)); a2 = __float_as_int((float)(a2 ^ a1)); a3 = __float_as_int((float)(a3 ^ a2)); a4 = __float_as_int((float)(a4 ^ a3)); a5 = __float_as_int((float)(a5 ^ a4)); a6 = __float_as_int((float)(a6 ^ a5)); a7 = __float_as_int((float)(a7 ^ a6)); }
    if ((a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7) == 0x1234567) o[0] = 1;
}
__global__ void k_dfma(uint32_t *o, double s) {
    double a0 = s + threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3;
    #pragma unroll 16
    for (int i = 0; i < NI / 4; i++) { a0 = fma(a0, 0.999, 1e-3); a1 = fma(a1, 0.999, 1e-3); a2 = fma(a2, 0.999, 1e-3); a3 = fma(a3, 0.999, 1e-3); }
    if (a0 + a1 + a2 + a3 == 1234.5) o[0] = 1;
}
__global__ void k_shfl(uint32_t *o, uint32_t s) {
    uint32_t a0 = s + threadIdx.x, a1 = a0 * 3, a2 = a0 * 5, a3 = a0 * 7;
    #pragma unroll 16
    for (int i = 0; i < NI / 4; i++) { a0 = __shfl_xor_sync(~0u, a0, 1) + a3; a1 = __shfl_xor_sync(~0u, a1, 2) + a0; a2 = __shfl_xor_sync(~0u, a2, 4) + a1; a3 = __shfl_xor_sync(~0u, a3, 8) + a2; }
    if ((a0 ^ a1 ^ a2 ^ a3) == 0x1234567) o[0] = 1;
}
__global__ __launch_bounds__(1024) void k_lds(uint32_t *o, uint32_t s) {
    __shared__ uint4 sm[2048];
    for (int i = threadIdx.x; i < 2048; i += blockDim.x) sm[i] = make_uint4(i, i * 3, i * 5, i * 7);
    __syncthreads();
    uint32_t a = s, b = 0; int ix = threadIdx.x;
    #pragma unroll 16
    for (int i = 0; i < NI / 4; i++) { uint4 v = sm[(ix + i * 32) & 2047]; uint4 w = sm[(ix + i * 32 + 1024) & 2047]; a ^= v.x ^ v.y ^ w.z ^ w.w; b += v.z ^ v.w ^ w.x ^ w.y; }
    if ((a ^ b) == 0x1234567) o[0] = 1;
}
int main() {
    uint32_t *o; CK(cudaMalloc(&o, 64));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); float ms;
    // stream read: 512 MB, measured twice after warmup
    size_t bytes = 512ull << 20, n = bytes / 16; uint4 *p; CK(cudaMalloc(&p, bytes)); CK(cudaMemset(p, 1, bytes));
    for (int bl : {2, 4, 8}) for (int tpb : {256, 512, 1024}) {
        k_read<<<bl, tpb>>>(p, n, o); cudaDeviceSynchronize();
        float best = 1e9;
        for (int r = 0; r < 3; r++) { cudaEventRecord(e0); k_read<<<bl, tpb>>>(p, n, o); cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); if (ms < best) best = ms; }
        printf("read blocks=%d tpb=%d: %.2f GB/s\n", bl, tpb, bytes / best / 1e6);
    }
    const double clk = 921.6e6;
    int B = 8, T = 1024;
    auto rep = [&](const char *nm, double ops_per_thread_iter, double tot_iters) {
        cudaEventElapsedTime(&ms, e0, e1);
        double ops = ops_per_thread_iter * tot_iters * B * T;
        printf("%-6s %.1f ops/clk/SM  (%.3f ms)\n", nm, ops / (ms * 1e-3) / clk, ms);
    };
#define RUN(k, arg, nm, per, it) k<<<B, T>>>(o, arg); cudaDeviceSynchronize(); cudaEventRecord(e0); k<<<B, T>>>(o, arg); cudaEventRecord(e1); cudaEventSynchronize(e1); rep(nm, per, it);
    RUN(k_popc, 7u, "popc", 8, NI)      // + 8 xor; xor is cheap but shares issue
    RUN(k_lop3, 7u, "lop3", 8, NI)
    RUN(k_iadd, 7u, "iadd", 8, NI)
    RUN(k_hfma2, 0.5f, "hfma2", 8, NI)  // half2 ops (x2 for scalar FMA)
    RUN(k_ffma, 0.5f, "ffma", 8, NI)
    RUN(k_f2f, 0.5f, "f2frt", 8, NI)    // round trips (fadd + 2 cvt)
    RUN(k_i2f, 7, "i2f", 8, NI)          // + xor
    RUN(k_dfma, 0.5, "dfma", 4, NI / 4)
    RUN(k_shfl, 7u, "shfl", 4, NI / 4)  // + iadd
    RUN(k_lds, 7u, "lds128", 2, NI / 4) // 128-bit loads per thread
    CK(cudaGetLastError());
    return 0;
}
