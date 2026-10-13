/* pq2_swar_sm53.cu — EXACT PQ2_0 GEMV for Maxwell sm_53 (Nano Gen1).
 *
 * Why: BQ1 sign-projection of PQ2_0 ({0,1,2,3}->{-1,+1,+1,+1} + averaged
 * scales) saturates activations (sampled runs loop one token, logit ~179).
 * No true 1-bit 8B checkpoint exists (only 27B ships PTQ1_0), so exact PQ2
 * math is the only honest 8B path. Fork-faithful AND zero-IMUL in the dot.
 *
 * Math (exact, fork order): code c = b0 + 2*b1, value v = c-1
 *   v = 2*b1 + (b0 - 1)   =>  isum = 2*mpos_b1 + S_pos(b0) - S_all
 *   acc += (ws * db) * (float)isum        (per 128-group, group order)
 * Integer partials are exact (|isum| <= 32512), so the SWAR decomposition
 * equals the fork's direct dot bit-for-bit; the fp accumulation is made
 * bit-exact too by reducing per-group products in GROUP ORDER in fp32
 * (__fmul_rn/__fadd_rn, no contraction) — see "ordered reduce".
 *
 * Layout (split, from pq2_split_repack):
 *   w_codes  [N, K/4] uint8  (byte b = weights 4b..4b+3, w_j = (q>>2j)&3)
 *   w_scales [N, K/128] fp16 (native PQ2 scale, no averaging)
 *   xq       Q8_K as fwd8 builds it: (K/256)*292B, halves shared per group
 *   K in {4096, 12288}; N arbitrary.
 * Mapping: 16 lanes/row (half-warp), full-K Q8_K tile in shared (<=14KB)
 * loaded once per block with int4, 16 rows/block share it.
 */
#include <cuda_fp16.h>
#include <stdint.h>

#define PQ2_BLOCK_ROWS 16  /* 256 threads = 16 rows x 16 lanes */
#define PQ2_MAXK 12288
#define PQ2_MAXG (PQ2_MAXK / 128)   /* 96 groups max */
#define Q8K_NB 292

__device__ __forceinline__ int pq2_iadd3(int a, int b, int c) {
  return a + b + c; /* fuses to IADD3 */
}

/* One code byte (4 weights) -> b0/b1 plane nibbles (bit k = plane k). */
__device__ __forceinline__ void pq2_split4(uint8_t cb, uint32_t *b0,
                                           uint32_t *b1) {
  uint32_t t0 = cb & 0x55u;
  t0 = (t0 | (t0 >> 1)) & 0x33u;
  t0 = (t0 | (t0 >> 2)) & 0x0Fu;
  uint32_t t1 = ((uint32_t)cb >> 1) & 0x55u;
  t1 = (t1 | (t1 >> 1)) & 0x33u;
  t1 = (t1 | (t1 >> 2)) & 0x0Fu;
  *b0 = t0;
  *b1 = t1;
}

/* 4 one-bit flags -> 4 keep bytes (0xFF keep / 0x00 drop), no IMUL-free. */
__device__ __forceinline__ uint32_t pq2_keep4(uint32_t bits4) {
  uint32_t spread = (bits4 * 0x00204081u) & 0x01010101u;
  return spread * 255u;
}

__device__ __forceinline__ int pq2_hsum4(uint32_t apack) {
  int s0 = (int)(apack << 24) >> 24;
  int s1 = (int)(apack << 16) >> 24;
  int s2 = (int)(apack << 8) >> 24;
  int s3 = (int)((int32_t)apack) >> 24;
  return pq2_iadd3(s0, s1, s2) + s3;
}

/* masked 4-lane byte sum: bytes of apack where keep byte == 0xFF. */
__device__ __forceinline__ int pq2_dot4(uint32_t apack, uint32_t keep,
                                        int acc) {
  uint32_t mk = apack & keep;
  int s0 = (int)(mk << 24) >> 24;
  int s1 = (int)(mk << 16) >> 24;
  int s2 = (int)(mk << 8) >> 24;
  int s3 = (int)((int32_t)mk) >> 24;
  acc = pq2_iadd3(s0, s1, acc);
  return pq2_iadd3(s2, s3, acc);
}

__global__ __launch_bounds__(256, 3)
void pq2_gemv_sm53(const uint8_t* __restrict__ w_codes,
                   const __half* __restrict__ w_scales,
                   const uint8_t* __restrict__ xq, float* __restrict__ y,
                   int K, int N) {
  __shared__ uint8_t sh_x[PQ2_MAXK / 256 * Q8K_NB]; /* <=14016B */
  __shared__ float sh_prod[PQ2_BLOCK_ROWS][PQ2_MAXG];

  int row = blockIdx.x * PQ2_BLOCK_ROWS + (threadIdx.x >> 4);
  int sub = threadIdx.x & 15;
  if (blockIdx.x * PQ2_BLOCK_ROWS >= N) return; /* block-uniform */
  bool active = row < N;

  int K128 = K >> 7;
  int GperThr = K128 >> 4;   /* 2 (K=4096) or 6 (K=12288) */
  int my_g0 = sub * GperThr;

  /* 1. Q8_K tile: cooperative LDG.E.128 once per block. */
  int nvec = (K / 256 * Q8K_NB) >> 4;
  for (int i = (int)threadIdx.x; i < nvec; i += (int)blockDim.x) {
    int4 v = __ldg(reinterpret_cast<const int4*>(xq + i * 16));
    *reinterpret_cast<int4*>(sh_x + i * 16) = v;
  }
  __syncthreads();

  /* 2. Per-group integer dot, fp32 product, in per-lane registers. */
  const uint8_t *cd = w_codes + (size_t)(active ? row : 0) * (K >> 2);
  for (int s = 0; s < GperThr; s++) {
    int g = my_g0 + s;
    const uint8_t *qb = sh_x + (size_t)(g >> 1) * Q8K_NB;
    float db;
    __builtin_memcpy(&db, qb, 4);
    const int8_t *qs = (const int8_t *)(qb + 4 + (g & 1) * 128);
    const uint8_t *c = cd + g * 32;
    int pos0 = 0, mpos = 0, all = 0;
#pragma unroll
    for (int b = 0; b < 32; b += 4) {
      uint32_t cw;
      __builtin_memcpy(&cw, c + b, 4);
      uint32_t a0, a1, a2, a3;
      __builtin_memcpy(&a0, qs + b * 4, 4);
      __builtin_memcpy(&a1, qs + b * 4 + 4, 4);
      __builtin_memcpy(&a2, qs + b * 4 + 8, 4);
      __builtin_memcpy(&a3, qs + b * 4 + 12, 4);
      uint32_t ap[4] = {a0, a1, a2, a3};
#pragma unroll
      for (int j = 0; j < 4; j++) {
        uint8_t cb = (uint8_t)(cw >> (j * 8));
        uint32_t f0, f1;
        pq2_split4(cb, &f0, &f1);
        pos0 = pq2_dot4(ap[j], pq2_keep4(f0), pos0);
        mpos = pq2_dot4(ap[j], pq2_keep4(f1), mpos);
        all = pq2_iadd3(pq2_hsum4(ap[j]), 0, all);
      }
    }
    int isum = (mpos << 1) + pos0 - all;
    float ws = active ? __half2float(
        __ldg(w_scales + (size_t)(active ? row : 0) * K128 + g)) : 0.0f;
    /* (ws*db)*isum with no contraction, matching the C reference exactly */
    sh_prod[threadIdx.x >> 4][g] =
        __fmul_rn(__fmul_rn(ws, db), (float)isum);
  }

  /* 3. Ordered reduce: lane 0 sums groups 0..K128-1 in order in fp32.
   * That is the fork's accumulation order, so the row result is
   * bit-identical to the serial reference (no tree, no shuffles). */
  __syncwarp();
  if (sub == 0 && active) {
    float acc = 0.0f;
    const float *p = sh_prod[threadIdx.x >> 4];
    for (int g = 0; g < K128; g++) acc = __fadd_rn(acc, p[g]);
    y[row] = acc;
  }
}

extern "C" int pq2_gemv_sm53_launch(const uint8_t *w_codes,
                                    const unsigned short *w_scales_h,
                                    const uint8_t *xq, float *y, int K, int N,
                                    void *stream) {
  if ((K & 127) != 0) return -2;
  if (K <= 0 || K > PQ2_MAXK || N <= 0) return -2;
  dim3 block(256, 1, 1);
  dim3 grid((N + PQ2_BLOCK_ROWS - 1) / PQ2_BLOCK_ROWS, 1, 1);
  cudaStream_t s = (cudaStream_t)stream;
  pq2_gemv_sm53<<<grid, block, 0, s>>>(
      w_codes, reinterpret_cast<const __half *>(w_scales_h), xq, y, K, N);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}
