/* bq1_swar_sm53.cu v4r — half-warp-row, single-tile Maxwell SWAR GEMV.
 * Nano Gen1 (sm_53). Best measured: 1.87-1.90 GB/s (4096^2: 1.20ms).
 * Tried and reverted: v2 LUT+dual-acc (1.06ms REGRESS), v5 popc diet
 * (1.81, no gain), 2-way even/odd-k split (wash -1.6%), v6 cooperative
 * S_all precompute (1.65ms REGRESS: separate pass costs more than the
 * fused hsum it removed). Lesson: keep one fused integer loop; the
 * compiler's ILP slots are already full.
 *
 * Mapping: 16 threads/row (half-warp); K=4096 -> 1 thread owns exactly 1
 * full 256-superblock; K=12288 -> 3 contiguous supers. Single activation
 * tile in shared (<=12KB), cooperative int4 load once, 8-16 rows share it.
 * IMUL-spread mask (pipe unsaturated, parity holds). S_pos/S_all split:
 * dot = 2*S_pos - S_all, 1 FMA per 256 weights, fork-faithful fp order.
 */
#include <cuda_fp16.h>
#include <stdint.h>

#define BQ1_BLOCK_ROWS 16  /* 256 threads = 16 rows x 16 lanes */
#define BQ1_MAXK 12288

__device__ __forceinline__ int bq1_iadd3(int a, int b, int c) {
  return a + b + c; /* fuses to IADD3; cuobjdump to confirm */
}

__device__ __forceinline__ int bq1_hsum4(uint32_t apack) {
  int s0 = (int)(apack << 24) >> 24;
  int s1 = (int)(apack << 16) >> 24;
  int s2 = (int)(apack << 8) >> 24;
  int s3 = (int)((int32_t)apack) >> 24;
  return bq1_iadd3(s0, s1, s2) + s3;
}

__global__ __launch_bounds__(256, 3)
void bq1_gemv_sm53(const uint32_t* __restrict__ w_bits,
                   const __half* __restrict__ w_scales,
                   const int8_t* __restrict__ x_q, float x_scale,
                   float* __restrict__ y, int K, int N) {
  __shared__ int8_t sh_x[BQ1_MAXK];

  int row = blockIdx.x * BQ1_BLOCK_ROWS + (threadIdx.x >> 4);
  int sub = threadIdx.x & 15;
  unsigned hmask = 0xFFFFu << ((threadIdx.x >> 4 & 1) * 16);
  if (blockIdx.x * BQ1_BLOCK_ROWS >= N) return; /* block-uniform */
  bool active = row < N;

  int K32 = K >> 5, K256 = K >> 8;
  int sb_per_thr = K256 >> 4; /* 1 (K=4096) or 3 (K=12288) */
  int my_sb0 = sub * sb_per_thr;

  int nvec = K >> 4;
  for (int i = (int)threadIdx.x; i < nvec; i += (int)blockDim.x) {
    int4 v = __ldg(reinterpret_cast<const int4*>(x_q + i * 16));
    *reinterpret_cast<int4*>(sh_x + i * 16) = v;
  }
  __syncthreads();

  float acc_f = 0.0f;
  const uint32_t *x32 = reinterpret_cast<const uint32_t*>(sh_x);
  for (int s = 0; s < sb_per_thr; s++) {
    int sb = my_sb0 + s;
    int w0 = sb * 8;
    int pos = 0, all = 0;
    int4 q0 = {0, 0, 0, 0}, q1 = {0, 0, 0, 0};
    if (active) {
      const int4 *wp = reinterpret_cast<const int4 *>(
          w_bits + (size_t)row * K32 + w0);
      q0 = __ldg(wp); q1 = __ldg(wp + 1);
    }
    uint32_t wv[8];
    wv[0] = (uint32_t)q0.x; wv[1] = (uint32_t)q0.y;
    wv[2] = (uint32_t)q0.z; wv[3] = (uint32_t)q0.w;
    wv[4] = (uint32_t)q1.x; wv[5] = (uint32_t)q1.y;
    wv[6] = (uint32_t)q1.z; wv[7] = (uint32_t)q1.w;
    int pbase = sb * 64;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      uint32_t w32 = wv[k];
#pragma unroll
      for (int j = 0; j < 8; j++) {
        int p = pbase + k * 8 + j;
        uint32_t ap = x32[p];
        uint32_t w4 = (w32 >> (j * 4)) & 0xFu;
        uint32_t spread = (w4 * 0x00204081u) & 0x01010101u;
        uint32_t keep = spread * 255u;
        uint32_t mk = ap & keep;
        int s0 = (int)(mk << 24) >> 24;
        int s1 = (int)(mk << 16) >> 24;
        int s2 = (int)(mk << 8) >> 24;
        int s3 = (int)((int32_t)mk) >> 24;
        pos = bq1_iadd3(s0, s1, pos);
        pos = bq1_iadd3(s2, s3, pos);
        all = bq1_iadd3(bq1_hsum4(ap), 0, all);
      }
    }
    float ws = active ? __half2float(
        __ldg(w_scales + (size_t)row * K256 + sb)) : 0.0f;
    acc_f += (float)((pos << 1) - all) * ws;
  }

  acc_f += __shfl_down_sync(hmask, acc_f, 8);
  acc_f += __shfl_down_sync(hmask, acc_f, 4);
  acc_f += __shfl_down_sync(hmask, acc_f, 2);
  acc_f += __shfl_down_sync(hmask, acc_f, 1);
  if (sub == 0 && active) y[row] = acc_f * x_scale;
}

extern "C" int bq1_gemv_sm53_launch(const uint32_t* w_bits,
                                    const unsigned short* w_scales_h,
                                    const signed char* x_q, float x_scale,
                                    float* y, int K, int N, void* stream) {
  if ((K & 255) != 0) return -2;
  if (K <= 0 || K > BQ1_MAXK || N <= 0) return -2;
  dim3 block(256, 1, 1);
  dim3 grid((N + BQ1_BLOCK_ROWS - 1) / BQ1_BLOCK_ROWS, 1, 1);
  cudaStream_t s = (cudaStream_t)stream;
  bq1_gemv_sm53<<<grid, block, 0, s>>>(
      w_bits, reinterpret_cast<const __half*>(w_scales_h),
      reinterpret_cast<const int8_t*>(x_q), x_scale, y, K, N);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}
