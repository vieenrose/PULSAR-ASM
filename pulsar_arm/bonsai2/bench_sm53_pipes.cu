// bench_sm53_pipes.cu: Jetson Nano GPU pipe sharing / dual issue (sm_53): popc+lop3 mixes,
// popc+i2f, popc+f2f, hfma2+ffma, hfma2 latency. See .auto/HANDOFF_B2_4K.md.
//   nvcc -O3 -gencode arch=compute_53,code=sm_53 -o bench_sm53_pipes bench_sm53_pipes.cu
#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>
#define NI 2048
__device__ __forceinline__ uint32_t l3(uint32_t a, uint32_t b, uint32_t c) { uint32_t r; asm volatile("lop3.b32 %0,%1,%2,%3,0x96;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
__device__ __forceinline__ uint32_t pc(uint32_t a) { uint32_t r; asm volatile("popc.b32 %0,%1;" : "=r"(r) : "r"(a)); return r; }
// LP lop3 per popc; 4 independent chains of (popc ; LP lop3)
template <int LP>
__global__ void k_mix(uint32_t *o, uint32_t s) {
    uint32_t p0 = s + threadIdx.x, p1 = p0 * 3, p2 = p0 * 5, p3 = p0 * 7;
    uint32_t l0 = p0 * 9, l1 = p0 * 11, l2 = p0 * 13, l3_ = p0 * 17, l4 = p0 * 19, l5 = p0 * 23, l6 = p0 * 29, l7 = p0 * 31;
    #pragma unroll 8
    for (int i = 0; i < NI; i++) {
        p0 = pc(p0 ^ l0); p1 = pc(p1 ^ l1); p2 = pc(p2 ^ l2); p3 = pc(p3 ^ l3_);
        #pragma unroll
        for (int j = 0; j < LP; j++) { l0 = l3(l0, l1, p0); l1 = l3(l1, l2, p1); l2 = l3(l2, l3_, p2); l3_ = l3(l3_, l4, p3); l4 = l3(l4, l5, l0); l5 = l3(l5, l6, l1); l6 = l3(l6, l7, l2); l7 = l3(l7, l0, l3_); }
    }
    if ((p0 ^ p1 ^ p2 ^ p3 ^ l0 ^ l1 ^ l2 ^ l3_ ^ l4 ^ l5 ^ l6 ^ l7) == 0x1234567) o[0] = 1;
}
// popc and i2f interleaved, independent
__global__ void k_pi(uint32_t *o, uint32_t s) {
    uint32_t a0 = s + threadIdx.x, a1 = a0 * 3, a2 = a0 * 5, a3 = a0 * 7; int b0 = a0, b1 = a1, b2 = a2, b3 = a3;
    #pragma unroll 8
    for (int i = 0; i < NI; i++) { a0 = pc(a0) ^ a3; a1 = pc(a1) ^ a0; a2 = pc(a2) ^ a1; a3 = pc(a3) ^ a2;
        b0 = __float_as_int((float)b0) ^ b3; b1 = __float_as_int((float)b1) ^ b0; b2 = __float_as_int((float)b2) ^ b1; b3 = __float_as_int((float)b3) ^ b2; }
    if ((a0 ^ a1 ^ a2 ^ a3 ^ b0 ^ b1 ^ b2 ^ b3) == 0x1234567) o[0] = 1;
}
// popc and f32->f16 cvt interleaved
__global__ void k_pf(uint32_t *o, uint32_t s) {
    uint32_t a0 = s + threadIdx.x, a1 = a0 * 3, a2 = a0 * 5, a3 = a0 * 7; float b0 = a0, b1 = a1, b2 = a2, b3 = a3;
    #pragma unroll 8
    for (int i = 0; i < NI; i++) { a0 = pc(a0) ^ a3; a1 = pc(a1) ^ a0; a2 = pc(a2) ^ a1; a3 = pc(a3) ^ a2;
        b0 = __half2float(__float2half_rn(b0 + b3)); b1 = __half2float(__float2half_rn(b1 + b0)); }
    if ((a0 ^ a1 ^ a2 ^ a3) + b0 + b1 + b2 + b3 == 1234.5f) o[0] = 1;
}
// hfma2 and ffma interleaved
__global__ void k_hf(uint32_t *o, float s) {
    __half2 h0 = __float2half2_rn(s), h1 = h0, h2 = h0, h3 = h0, m = __float2half2_rn(0.999f), c = __float2half2_rn(1e-3f);
    float f0 = s, f1 = s + 1, f2 = s + 2, f3 = s + 3;
    #pragma unroll 8
    for (int i = 0; i < NI; i++) { h0 = __hfma2(h0, m, c); h1 = __hfma2(h1, m, c); h2 = __hfma2(h2, m, c); h3 = __hfma2(h3, m, c);
        f0 = fmaf(f0, 0.999f, 1e-3f); f1 = fmaf(f1, 0.999f, 1e-3f); f2 = fmaf(f2, 0.999f, 1e-3f); f3 = fmaf(f3, 0.999f, 1e-3f); }
    if (__low2float(__hadd2(__hadd2(h0, h1), __hadd2(h2, h3))) + f0 + f1 + f2 + f3 == 1234.5f) o[0] = 1;
}
// hfma2 alone with 12 independent chains (latency check)
__global__ void k_h12(uint32_t *o, float s) {
    __half2 h[12]; for (int j = 0; j < 12; j++) h[j] = __float2half2_rn(s + j * 1e-3f);
    __half2 m = __float2half2_rn(0.999f), c = __float2half2_rn(1e-3f);
    #pragma unroll 4
    for (int i = 0; i < NI; i++) {
        #pragma unroll
        for (int j = 0; j < 12; j++) h[j] = __hfma2(h[j], m, c);
    }
    __half2 t = h[0]; for (int j = 1; j < 12; j++) t = __hadd2(t, h[j]);
    if (__low2float(t) == 1234.5f) o[0] = 1;
}
int main() {
    uint32_t *o; cudaMalloc(&o, 64);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); float ms;
    const double clk = 921.6e6; const int B = 8, T = 1024;
    // 4 popc + 4 xor + 8*LP lop3 per iter
#define MIX(LP) { k_mix<LP><<<B, T>>>(o, 7); cudaDeviceSynchronize(); cudaEventRecord(e0); k_mix<LP><<<B, T>>>(o, 7); cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); \
    double c = ms * 1e-3 * clk / ((double)NI * B * T); double alu = (4 + 8 * LP) / 128.0, pp = 4 / 32.0; \
    printf("mix LP=%d: %.4f clk/thread-iter | issue-sum %.4f | dual max %.4f\n", LP, c, alu + 4 / 128.0, alu > pp ? alu : pp); }
    MIX(0) MIX(1) MIX(2) MIX(3) MIX(4) MIX(6)
#define ONE(k, arg, nm, a, b) { k<<<B, T>>>(o, arg); cudaDeviceSynchronize(); cudaEventRecord(e0); k<<<B, T>>>(o, arg); cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); \
    double c = ms * 1e-3 * clk / ((double)NI * B * T); printf("%s: %.4f clk/thread-iter (shared-pipe %.4f, separate %.4f)\n", nm, c, a, b); }
    ONE(k_pi, 7u, "popc4+i2f4 (+8 xor)", 8 / 32.0, 4 / 32.0)
    ONE(k_pf, 7u, "popc4+f2frt2 (+4xor,2fadd)", 8 / 32.0, 4 / 32.0)
    ONE(k_hf, 0.5f, "hfma2x4+ffma4", 8 / 128.0, 4 / 128.0)
    ONE(k_h12, 0.5f, "hfma2x12", 12 / 128.0, 0.0)
    return 0;
}
