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
#define Q8K_NB 292                  /* Q8_K block: float d + 256 i8 */
#define Q8K_TILE_STRIDE 272         /* shared tile stride: 256 acts (16B
                                     * aligned) + d + pad, 17 u32 words */
/* v4 (E3): process K in chunks of 8 Q8_K blocks (16 groups) so the
 * activation tile is 2.2KB instead of 13KB. Shared per block drops from
 * ~20KB to ~3.4KB, lifting occupancy from 3 blocks/SM (shared-limited) to
 * the 4-block register cap (32 warps vs 24). Each chunk gives every lane
 * exactly one group, so the ordered fp reduce is a short 16-add pass per
 * chunk instead of one long tail at the end. */
#define PQ2_CHUNK_BLOCKS 8
#define PQ2_CHUNK_GROUPS (PQ2_CHUNK_BLOCKS * 2)
#define PQ2_CHUNK_BYTES (PQ2_CHUNK_BLOCKS * Q8K_TILE_STRIDE)
#define PQ2_MAXCHUNK ((PQ2_MAXK / 256) / PQ2_CHUNK_BLOCKS)

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

/* 4 one-bit flags -> 4 keep bytes (0xFF keep / 0x00 drop). */
__device__ __forceinline__ uint32_t pq2_keep4(uint32_t bits4) {
  uint32_t spread = (bits4 * 0x00204081u) & 0x01010101u;
  return spread * 255u;
}

/* masked 4-lane SIGNED byte sum: bytes of apack where the keep byte is
 * 0xFF, sign-extended and added to acc. The signed form is what removes the
 * per-(row,group) correction array: with isum = 2*mu + pu - au computed
 * directly, no bias bookkeeping is needed, which is 132MB less on the GPU
 * for the 8B (6.5% of the arena) plus that array's read traffic. */
__device__ __forceinline__ int pq2_sdot4(uint32_t apack, uint32_t keep,
                                         int acc) {
  uint32_t mk = apack & keep;
  int s0 = (int)(mk << 24) >> 24;
  int s1 = (int)(mk << 16) >> 24;
  int s2 = (int)(mk << 8) >> 24;
  int s3 = (int)((int32_t)mk) >> 24;
  acc = pq2_iadd3(s0, s1, acc);
  return pq2_iadd3(s2, s3, acc);
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

/* full 4-byte SIGNED sum (row-independent Au needs this form). */
__device__ __forceinline__ int pq2_ssum4(uint32_t apack) {
  int s0 = (int)(apack << 24) >> 24;
  int s1 = (int)(apack << 16) >> 24;
  int s2 = (int)(apack << 8) >> 24;
  int s3 = (int)((int32_t)apack) >> 24;
  return pq2_iadd3(s0, s1, s2) + s3;
}

__global__ __launch_bounds__(256, 5)
void pq2_gemv_sm53(const uint8_t* __restrict__ w_codes,
                   const __half* __restrict__ w_scales,
                   const uint8_t* __restrict__ xq, float* __restrict__ y,
                   int K, int N) {
  __shared__ __align__(16) uint8_t sh_x[PQ2_CHUNK_BYTES];
  __shared__ int sh_au[PQ2_CHUNK_GROUPS];
  __shared__ float sh_prod[PQ2_BLOCK_ROWS][PQ2_CHUNK_GROUPS];

  int row = blockIdx.x * PQ2_BLOCK_ROWS + (threadIdx.x >> 4);
  int sub = threadIdx.x & 15;
  if (blockIdx.x * PQ2_BLOCK_ROWS >= N) return; /* block-uniform */
  bool active = row < N;
  int rrow = active ? row : 0;

  int K128 = K >> 7;
  int nchunk = K128 / PQ2_CHUNK_GROUPS;
  if (nchunk > PQ2_MAXCHUNK || nchunk <= 0) return;

  float acc = 0.0f; /* lane 0 only; groups accumulated in order */

  for (int ch = 0; ch < nchunk; ch++) {
    /* 1. stage the chunk: d verbatim, acts biased to unsigned (x^0x80), at
     * a 272B stride so every 128-act group is 16B-aligned (LDS.128). */
    const uint32_t *x32 = reinterpret_cast<const uint32_t *>(xq) +
                          (size_t)ch * PQ2_CHUNK_BLOCKS * 73;
    uint32_t *t32 = reinterpret_cast<uint32_t *>(sh_x);
    {
      const int nw = PQ2_CHUNK_BLOCKS * 65;
      for (int i = (int)threadIdx.x; i < nw; i += (int)blockDim.x) {
        int b = i / 65, w = i - b * 65;
        uint32_t v = __ldg(x32 + b * 73 + w);
        if (w == 0) t32[b * 68 + 64] = v;      /* fp32 d, verbatim */
        else t32[b * 68 + (w - 1)] = v ^ 0x80808080u; /* x' = x + 128 */
      }
    }
    __syncthreads();

    /* 2. Au per group (row-independent): one thread per group. */
    if ((int)threadIdx.x < PQ2_CHUNK_GROUPS) {
      int gg = (int)threadIdx.x;
      const uint32_t *p = t32 + (gg >> 1) * 68 + (gg & 1) * 32;
      uint32_t a2 = 0;
#pragma unroll
      for (int q = 0; q < 32; q++) {
        uint32_t m = p[q];
        uint32_t t = (m & 0x00FF00FFu) + ((m >> 8) & 0x00FF00FFu);
        a2 += (t & 0xFFFFu) + (t >> 16);
      }
      sh_au[gg] = (int)a2;
    }
    __syncthreads();

    /* 3. one group per lane (128 weights per group, 32 code bytes) */
    int g = sub;
    const uint8_t *c =
        w_codes + (size_t)rrow * (K >> 2) +
        (size_t)(ch * PQ2_CHUNK_GROUPS + g) * 32;
    const uint32_t *gb = t32 + (g >> 1) * 68;
    float db;
    __builtin_memcpy(&db, gb + 64, 4);
    const uint32_t *qs = gb + (g & 1) * 32;

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
    /* Bias-correction counts, derived from the code bytes in the same pass:
     * with x' = x+128, isum = Pu + 2*Mu - Au + 16384 - 128*pc0 - 256*pc1.
     * Computing this here instead of reading a per-(row,group) int16 array
     * saves 132MB of GPU RAM (the array was as large as the scales) and its
     * read traffic; the 2 POPC per code byte cost ~+6 ALU ops per 4 weights,
     * which the SASS experiment showed we have slack for. */
    int pc0 = 0, pc1 = 0;
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
        pc0 += __popc((uint32_t)cb & 0x55u);
        pc1 += __popc(((uint32_t)cb >> 1) & 0x55u);
        if (w & 1) {
          pu1 = pq2_usum4(ap[j], pq2_keep4(f0), pu1);
          mu1 = pq2_usum4(ap[j], pq2_keep4(f1), mu1);
        } else {
          pu0 = pq2_usum4(ap[j], pq2_keep4(f0), pu0);
          mu0 = pq2_usum4(ap[j], pq2_keep4(f1), mu0);
        }
      }
    }
    int isum = ((mu0 + mu1) << 1) + (pu0 + pu1) - sh_au[g] + 16384 -
               (pc0 << 7) - (pc1 << 8);
    float ws = active ? __half2float(__ldg(w_scales + (size_t)rrow * K128 +
                                          (ch * PQ2_CHUNK_GROUPS + g)))
                      : 0.0f;
    sh_prod[threadIdx.x >> 4][g] =
        __fmul_rn(__fmul_rn(ws, db), (float)isum);
    __syncwarp();

    /* 4. ordered reduce for this chunk: lane 0 of each row adds the 16
     * products in group order (chunks are processed in order too). */
    if (sub == 0) {
      const float *p = sh_prod[threadIdx.x >> 4];
#pragma unroll
      for (int k = 0; k < PQ2_CHUNK_GROUPS; k++) acc = __fadd_rn(acc, p[k]);
    }
    __syncthreads(); /* the tile is reused by the next chunk */
  }

  if (sub == 0 && active) y[row] = acc;
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
