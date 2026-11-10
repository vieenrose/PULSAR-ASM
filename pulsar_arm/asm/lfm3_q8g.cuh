// lfm3_q8g.cuh -- the Q8 GEMM kernel shared by the depth head and the codec (GPU path).
// Contract (both engines' NEON kernel, lfm3_depth_neon.c): per output acc = 0; for k
// ascending: acc = fma((float)code_k * scale_{k/32}, x_k, acc) -- so outputs are
// bit-identical to the NEON engine whatever the thread mapping.
#pragma once
// ---- GPU GEMMs (LFM_DEPTH_GPU, default 1; 0 = the NEON kernels) -------------------------
// k_q8g<T>: one thread per output row, T accumulators (one per position). Per output the
// arithmetic is the NEON contract exactly: acc = 0; for k ascending: acc = fma(w_k, x_k, acc)
// with w_k = (float)code_k * scale_{k/32} rounded once. Only the schedule differs: each
// 32-code Q8 block of 128 rows is staged through shared memory with coalesced loads (8
// threads read one row's 32 bytes; rows padded to 9 words so per-row reads are
// conflict-free), and the block's 32 activations of all T positions sit in shared too.
// Grouped launch (gridDim.y > 1, the 12 codebook heads): group g uses rows g*N.. of q/s,
// activation row g of X and output row g of Y, i.e. one GEMV per group in one launch.
// Activations are staged k-major (xs[k][t], T padded to a multiple of 4), so the T values
// of one k arrive as T/4 broadcast LDS.128 instead of T scalar loads.
template <int T, int RR = 1>
__global__ void __launch_bounds__(128) k_q8g(const int8_t* __restrict__ q, const float* __restrict__ s,
                                             const float* __restrict__ X, float* __restrict__ Y,
                                             int N, int K, int tile_mode) {
  // tile_mode 0: grid y = groups (weights rows g*N.., X row g, Y row g -- the depth heads).
  // tile_mode 1: grid y = position tiles of T over one weight matrix (X/Y advance by T rows).
  // RR rows per thread (rows r0 + r*128 + tid): the T activations of one k are loaded
  // once from shared and feed RR*T independent chains; each chain is the contract above.
  constexpr int TP = (T + 3) & ~3, BR = 128 * RR, NX = (T * 32 + 127) / 128;
  __shared__ int wq[BR][9];
  __shared__ float ws[BR];
  __shared__ __align__(16) float xs[32][TP];
  const int nb = K / 32;
  { const int g = blockIdx.y;   // group / tile offsets (0 for the plain GEMM)
    if (!tile_mode) { q += (size_t)g * N * K; s += (size_t)g * N * nb; }
    X += (size_t)g * K * T; Y += (size_t)g * N * T; }
  const int r0 = blockIdx.x * BR, tid = threadIdx.x;
  float acc[RR][T];
  #pragma unroll
  for (int r = 0; r < RR; r++)
    #pragma unroll
    for (int t = 0; t < T; t++) acc[r][t] = 0.f;
  for (int b = 0; b < nb; b++) {
    // staging: loads into registers first (unrolled), then shared stores
    int pw[8 * RR]; float psc[RR]; float px[NX];
    #pragma unroll
    for (int i = 0; i < 8 * RR; i++) {
      const int idx = tid + i * 128, row = idx >> 3, w = idx & 7, gr = r0 + row;
      pw[i] = gr < N ? reinterpret_cast<const int*>(q + (size_t)gr * K + (size_t)b * 32)[w] : 0;
    }
    #pragma unroll
    for (int r = 0; r < RR; r++) { const int n = r0 + r * 128 + tid; psc[r] = n < N ? s[(size_t)n * nb + b] : 0.f; }
    #pragma unroll
    for (int ii = 0; ii < NX; ii++) {
      const int i = tid + ii * 128;
      px[ii] = i < T * 32 ? X[(size_t)(i >> 5) * K + (size_t)b * 32 + (i & 31)] : 0.f;
    }
    #pragma unroll
    for (int i = 0; i < 8 * RR; i++) { const int idx = tid + i * 128; wq[idx >> 3][idx & 7] = pw[i]; }
    #pragma unroll
    for (int r = 0; r < RR; r++) ws[r * 128 + tid] = psc[r];
    #pragma unroll
    for (int ii = 0; ii < NX; ii++) { const int i = tid + ii * 128; if (i < T * 32) xs[i & 31][i >> 5] = px[ii]; }
    __syncthreads();
    float sc[RR];
    #pragma unroll
    for (int r = 0; r < RR; r++) sc[r] = ws[r * 128 + tid];
    #pragma unroll
    for (int w = 0; w < 8; w++) {
      int v[RR];
      #pragma unroll
      for (int r = 0; r < RR; r++) v[r] = wq[r * 128 + tid][w];
      #pragma unroll
      for (int j = 0; j < 4; j++) {
        float wf[RR];
        #pragma unroll
        for (int r = 0; r < RR; r++) wf[r] = __fmul_rn((float)(signed char)(v[r] >> (8 * j)), sc[r]);
        const int k = w * 4 + j;
        if (T % 4 == 0) {
          #pragma unroll
          for (int t4 = 0; t4 < T / 4; t4++) {
            const float4 xv = *reinterpret_cast<const float4*>(&xs[k][t4 * 4]);
            #pragma unroll
            for (int r = 0; r < RR; r++) {
              acc[r][t4 * 4 + 0] = __fmaf_rn(wf[r], xv.x, acc[r][t4 * 4 + 0]);
              acc[r][t4 * 4 + 1] = __fmaf_rn(wf[r], xv.y, acc[r][t4 * 4 + 1]);
              acc[r][t4 * 4 + 2] = __fmaf_rn(wf[r], xv.z, acc[r][t4 * 4 + 2]);
              acc[r][t4 * 4 + 3] = __fmaf_rn(wf[r], xv.w, acc[r][t4 * 4 + 3]);
            }
          }
        } else {
          #pragma unroll
          for (int t = 0; t < T; t++) {
            const float xv = xs[k][t];
            #pragma unroll
            for (int r = 0; r < RR; r++) acc[r][t] = __fmaf_rn(wf[r], xv, acc[r][t]);
          }
        }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int r = 0; r < RR; r++) {
    const int n = r0 + r * 128 + tid;
    if (n < N) {
      #pragma unroll
      for (int t = 0; t < T; t++) Y[(size_t)t * N + n] = acc[r][t];
    }
  }
}
