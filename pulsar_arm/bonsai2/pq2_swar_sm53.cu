/* pq2_swar_sm53.cu — EXACT PQ2_0 GEMV for Maxwell sm_53 (Nano Gen1).
 *
 * Why: BQ1 sign-projection of PQ2_0 ({0,1,2,3}->{-1,+1,+1,+1} + averaged
 * scales) saturates activations (sampled runs loop one token, logit ~179).
 * No true 1-bit 8B checkpoint exists (only 27B ships PTQ1_0), so exact PQ2
 * math is the only honest 8B path. Fork-faithful AND zero-IMUL in the dot.
 *
 * Math (exact, fork order): code c = b0 + 2*b1, value v = c-1
 *   v = 2*b1 + (b0 - 1)   =>  isum = 2*mpos_b1 + S_pos(b0) - S_all
 * Unsigned kernel form (v2): activations are stored biased x' = x XOR 0x80
 * so every masked partial sum needs only and/shift/add (no sign extension):
 *   isum = Pu(b0) + 2*Mu(b1) - Au + corr[row][g]
 * with Au (row-independent) precomputed per block into shared, and corr
 * (codes-only) precomputed at repack time into int16. Both are exact, so
 * the integer result is unchanged.
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
#define PQ2_K128_MAX (PQ2_MAXG)     /* per-block group count bound */
#define Q8K_NB 292                  /* Q8_K block: float d + 256 i8 */
#define Q8K_TILE_STRIDE 272         /* shared tile stride: 256 acts (16B
                                     * aligned) + d + pad, 17 u32 words */

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

/* masked 4-lane UNSIGNED byte sum (8 ops): bytes of apack where the keep
 * byte is 0xFF. Two-stage SWAR fold, no sign extension, no IMUL. */
__device__ __forceinline__ int pq2_usum4(uint32_t apack, uint32_t keep,
                                         int acc) {
  uint32_t mk = apack & keep;
  uint32_t t = (mk & 0x00FF00FFu) + ((mk >> 8) & 0x00FF00FFu);
  uint32_t s = (t & 0xFFFFu) + (t >> 16);
  return acc + (int)s;
}

__global__ __launch_bounds__(256, 3)
void pq2_gemv_sm53(const uint8_t* __restrict__ w_codes,
                   const __half* __restrict__ w_scales,
                   const int16_t* __restrict__ w_corr,
                   const uint8_t* __restrict__ xq, float* __restrict__ y,
                   int K, int N) {
  __shared__ __align__(16) uint8_t sh_x[PQ2_MAXK / 256 * Q8K_TILE_STRIDE];
  __shared__ float sh_prod[PQ2_BLOCK_ROWS][PQ2_MAXG];
  __shared__ int sh_au[PQ2_MAXG];  /* unsigned sum of all 128 acts per group */

  int row = blockIdx.x * PQ2_BLOCK_ROWS + (threadIdx.x >> 4);
  int sub = threadIdx.x & 15;
  if (blockIdx.x * PQ2_BLOCK_ROWS >= N) return; /* block-uniform */
  bool active = row < N;

  int K128 = K >> 7;
  int GperThr = K128 >> 4;   /* 2 (K=4096) or 6 (K=12288) */
  int my_g0 = sub * GperThr;

  /* 1. Q8_K tile with a 272B stride so every 128-act group starts on a
   * 16-byte boundary (LDS.128 in the inner loop). Source blocks are 292B
   * (73 u32: d + 64 act words + 8 bsum words we skip), so the copy reads
   * 65 words per block and biases the acts to unsigned (x' = x^0x80) in
   * the same pass — no alignment hazard, and 11% less global traffic. */
  {
    const uint32_t *x32 = reinterpret_cast<const uint32_t *>(xq);
    uint32_t *t32 = reinterpret_cast<uint32_t *>(sh_x);
    int nb = K / 256;
    int nw = nb * 65;
    for (int i = (int)threadIdx.x; i < nw; i += (int)blockDim.x) {
      int b = i / 65, w = i - b * 65;
      uint32_t v = __ldg(x32 + b * 73 + w);
      if (w == 0) t32[b * 68 + 64] = v; /* fp32 d, verbatim */
      else t32[b * 68 + (w - 1)] = v ^ 0x80808080u;
    }
  }
  __syncthreads();

  /* 1b. Au (unsigned sum of all 128 acts per group) is row-independent, so
   * one thread per group computes it once for the whole block. */
  if ((int)threadIdx.x < K128) {
    int g = (int)threadIdx.x;
    const uint32_t *p = reinterpret_cast<const uint32_t *>(sh_x) +
                        (g >> 1) * 68 + (g & 1) * 32;
    uint32_t acc = 0;
#pragma unroll
    for (int q = 0; q < 32; q++) {
      uint32_t m = p[q];
      uint32_t t = (m & 0x00FF00FFu) + ((m >> 8) & 0x00FF00FFu);
      acc += (t & 0xFFFFu) + (t >> 16);
    }
    sh_au[g] = (int)acc;
  }
  __syncthreads();

  /* 2. Per-group integer dot, fp32 product, in per-lane registers. */
  const uint8_t *cd = w_codes + (size_t)(active ? row : 0) * (K >> 2);
  for (int s = 0; s < GperThr; s++) {
    int g = my_g0 + s;
    const uint32_t *gb = reinterpret_cast<const uint32_t *>(sh_x) +
                         (g >> 1) * 68;
    float db;
    __builtin_memcpy(&db, gb + 64, 4);
    const uint32_t *qs = gb + (g & 1) * 32; /* 32 u32 = 128 act bytes */
    const uint8_t *c = cd + g * 32;
    /* vector loads: 32 code bytes as 2x LDG.E.128 (16B-aligned: row stride
     * K/4 and group stride 32 are both multiples of 16), and each 16-weight
     * act slice as one LDS.128. Cuts inner-loop loads from 40 to 10 per
     * group. Two accumulator pairs (even/odd word) give the chains ILP. */
    const uint4 *cv = reinterpret_cast<const uint4 *>(c);
    uint4 cw0 = __ldg(cv);
    uint4 cw1 = __ldg(cv + 1);
    uint32_t cws[8];
    cws[0] = (uint32_t)cw0.x; cws[1] = (uint32_t)cw0.y;
    cws[2] = (uint32_t)cw0.z; cws[3] = (uint32_t)cw0.w;
    cws[4] = (uint32_t)cw1.x; cws[5] = (uint32_t)cw1.y;
    cws[6] = (uint32_t)cw1.z; cws[7] = (uint32_t)cw1.w;
    const uint4 *av = reinterpret_cast<const uint4 *>(qs);
    int pu0 = 0, mu0 = 0, pu1 = 0, mu1 = 0;
#pragma unroll
    for (int w = 0; w < 8; w++) {
      uint32_t cw = cws[w];
      uint4 av4 = av[w]; /* LDS.128: 16 act bytes */
      uint32_t ap[4] = {(uint32_t)av4.x, (uint32_t)av4.y, (uint32_t)av4.z,
                        (uint32_t)av4.w};
#pragma unroll
      for (int j = 0; j < 4; j++) {
        uint8_t cb = (uint8_t)(cw >> (j * 8));
        uint32_t f0, f1;
        pq2_split4(cb, &f0, &f1);
        if (w & 1) {
          pu1 = pq2_usum4(ap[j], pq2_keep4(f0), pu1);
          mu1 = pq2_usum4(ap[j], pq2_keep4(f1), mu1);
        } else {
          pu0 = pq2_usum4(ap[j], pq2_keep4(f0), pu0);
          mu0 = pq2_usum4(ap[j], pq2_keep4(f1), mu0);
        }
      }
    }
    int pu = pu0 + pu1, mu = mu0 + mu1;
    /* exact: isum = Pu + 2*Mu - Au + corr (bias corrections are exact
     * integers, codes-only, precomputed at repack time) */
    int isum = (mu << 1) + pu - sh_au[g] + (int)__ldg(w_corr + (size_t)(active ? row : 0) * K128 + g);
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
                                    const short *w_corr,
                                    const uint8_t *xq, float *y, int K, int N,
                                    void *stream) {
  if ((K & 127) != 0) return -2;
  if (K <= 0 || K > PQ2_MAXK || N <= 0) return -2;
  dim3 block(256, 1, 1);
  dim3 grid((N + PQ2_BLOCK_ROWS - 1) / PQ2_BLOCK_ROWS, 1, 1);
  cudaStream_t s = (cudaStream_t)stream;
  pq2_gemv_sm53<<<grid, block, 0, s>>>(
      w_codes, reinterpret_cast<const __half *>(w_scales_h),
      w_corr, xq, y, K, N);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}
