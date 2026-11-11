// lfm3_qlow.cuh -- low-bit (Q4 / Q5) backbone GEMV for one position (the speech server's
// decode). NOT bit-exact to the Q8 path: a different weight format, chosen in
// .auto/lfm3/quant_loss.py / gptq_lfm3.py and gated by KL vs HF.
//
// Format per tensor [N, K] (K % 32 == 0, N % 32 == 0), groups of 32 weights along K:
//   w = d * q + mn,  q in [0, 2^B - 1],  d, mn fp16 (stored as d' = 128 * d: exact, power of 2)
// The exporter writes the payload already tiled (32-row tiles; lane = row in a warp):
//   codes  : for tile, for group g: 32 x uint4 (row r's 32 nibbles: byte b of word i holds
//            weight 8i+b in its low nibble and weight 8i+4+b in its high nibble)
//            [B = 5] then 32 x uint32 high planes: bit 8*b + p = bit 4 of the weight in
//            quad p (= 2*i + half, half 0: low nibbles, 1: high), byte b
//   scales : for tile, for group g: 32 x half2 (d', mn)
// Decode: a code becomes the float 1 + q / 128 exactly (exponent 0x3F8, q in mantissa
// bits 16..20) with one byte permute, so a group's chain sums (1 + q/128) x and
//   y += d' * (acc - sx) + mn * sx,   sx = sum of the group's x (once per call, in smem).
#pragma once
#include <cuda_fp16.h>
#include <cstdint>

// A code q (B bits) sits in the low bits of byte 2 of the float 0x3F80_0000 | q << 16, i.e.
// exactly 1 + q / 128; low nibbles need no shift (w & 0x0F0F0F0F), high nibbles one.
// [B = 5] bit 4 of q comes from the high plane (shift so quad p's bits land on byte bit 4).
template <int B>
__device__ __forceinline__ void qlow_group(const uint4 w4, const uint32_t hp, const float* xg,
                                           float& a0, float& a1, float& a2, float& a3) {
  const uint32_t w[4] = { w4.x, w4.y, w4.z, w4.w };
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    #pragma unroll
    for (int hf = 0; hf < 2; hf++) {
      uint32_t c = 0x80808080u;
      if (B == 5) {
        const int p = 2 * i + hf;
        const uint32_t hs = p <= 4 ? (hp << (4 - p)) : (hp >> (p - 4));
        c |= hs & 0x10101010u;
      }
      const uint32_t m = ((hf ? (w[i] >> 4) : w[i]) & 0x0F0F0F0Fu) | c;
      const float4 xv = *reinterpret_cast<const float4*>(xg + 8 * i + 4 * hf);
      const float f0 = __uint_as_float(__byte_perm(m, 0x3F00u, 0x5044));
      const float f1 = __uint_as_float(__byte_perm(m, 0x3F00u, 0x5144));
      const float f2 = __uint_as_float(__byte_perm(m, 0x3F00u, 0x5244));
      const float f3 = __uint_as_float(__byte_perm(m, 0x3F00u, 0x5344));
      a0 = __fmaf_rn(f0, xv.x, a0); a1 = __fmaf_rn(f1, xv.y, a1);
      a2 = __fmaf_rn(f2, xv.z, a2); a3 = __fmaf_rn(f3, xv.w, a3);
    }
  }
}

// one row per thread. Block (x, y) = 256 rows x K slice y of KS: x (bf16) slice staged as f32
// in smem with per-group sums; KS == 1 writes bf16 y (yb, not in place), else partial
// sums to part[y][row] and k_qlow_combine adds the slices in order. PD groups per iteration
// (all loads of the next PD groups issued before this iteration's math).
// MINB resident blocks per SM (launch bounds; default full occupancy = 32 registers, which
// spills for PD > 1).
template <int B, int TPB, int PD, int MINB = 2048 / TPB>
__global__ void __launch_bounds__(TPB, MINB) k_qlow(const uint4* __restrict__ q, const __half2* __restrict__ s,
                                                     const unsigned short* __restrict__ x, float* __restrict__ part,
                                                     unsigned short* __restrict__ yb, int N, int K) {
  extern __shared__ float xs[];                  // [KL] then sx[KL/32]
  const int G = K / 32, KS = gridDim.y, GL = G / KS, KL = GL * 32, g0 = blockIdx.y * GL, tid = threadIdx.x;
  float* sx = xs + KL;
  for (int i = tid; i < KL; i += TPB) xs[i] = __uint_as_float((uint32_t)x[g0 * 32 + i] << 16);
  __syncthreads();
  for (int g = tid; g < GL; g += TPB) {
    float t = 0.f;
    #pragma unroll 8
    for (int k = 0; k < 32; k++) t += xs[g * 32 + k];
    sx[g] = t;
  }
  __syncthreads();
  const int row = blockIdx.x * TPB + tid;
  if (row >= N) return;
  constexpr int GW = (B == 4) ? 32 : 40;         // uint4 slots per group per tile (u32 plane = 8 uint4)
  const size_t tile = (size_t)(row >> 5), lane = row & 31;
  const uint4* qt = q + (tile * G + g0) * GW + lane;
  const uint32_t* qh = reinterpret_cast<const uint32_t*>(q + (tile * G + g0) * GW + 32) + lane;
  const __half2* st = s + (tile * G + g0) * 32 + lane;
  float y = 0.f;
  uint4 nw[PD]; uint32_t nh[PD]; __half2 nd[PD];
  #pragma unroll
  for (int u = 0; u < PD; u++) {
    nw[u] = qt[(size_t)u * GW]; nh[u] = B == 5 ? qh[(size_t)u * GW * 4] : 0u; nd[u] = st[(size_t)u * 32];
  }
  for (int g = 0; g < GL; g += PD) {             // GL % PD == 0 (checked by the caller)
    uint4 w[PD]; uint32_t h[PD]; __half2 dm[PD];
    #pragma unroll
    for (int u = 0; u < PD; u++) { w[u] = nw[u]; h[u] = nh[u]; dm[u] = nd[u]; }
    if (g + PD < GL) {
      #pragma unroll
      for (int u = 0; u < PD; u++) {
        const size_t gg = g + PD + u;
        nw[u] = qt[gg * GW]; if (B == 5) nh[u] = qh[gg * GW * 4]; nd[u] = st[gg * 32];
      }
    }
    #pragma unroll
    for (int u = 0; u < PD; u++) {
      float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
      qlow_group<B>(w[u], h[u], xs + (g + u) * 32, a0, a1, a2, a3);
      const float acc = (a0 + a1) + (a2 + a3), sg = sx[g + u];
      const float2 f = __half22float2(dm[u]);
      y = __fmaf_rn(f.x, acc - sg, __fmaf_rn(f.y, sg, y));
    }
  }
  if (KS > 1 || part) part[(size_t)blockIdx.y * N + row] = y;
  else { const uint32_t u = __float_as_uint(y); yb[row] = (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
}
__global__ void k_qlow_combine(const float* __restrict__ part, unsigned short* __restrict__ yb, int N, int KS) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= N) return;
  float y = 0.f;
  for (int k = 0; k < KS; k++) y += part[(size_t)k * N + row];
  const uint32_t u = __float_as_uint(y); yb[row] = (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
