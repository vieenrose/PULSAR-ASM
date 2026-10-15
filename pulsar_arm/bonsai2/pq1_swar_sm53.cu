/* pq1_swar_sm53.cu — EXACT PQ1_0 GEMV for Maxwell sm_53 (Nano Gen1).
 *
 * Why: BQ1 sign-projection of PQ1_0 ({0,1,2,3}->{-1,+1,+1,+1} + averaged
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
 * Layout (split, from pq1_split_repack):
 *   w_bits   [N, K/32] u32   (bit j of word w = weight w*32+j; 1 -> +1)
 *   w_scales [N, K/128] fp16 (native PQ2 scale, no averaging)
 *   xq       Q8_K as fwd8 builds it: (K/256)*292B, halves shared per group
 *   K in {4096, 12288}; N arbitrary.
 * Mapping: 16 lanes/row (half-warp), full-K Q8_K tile in shared (<=14KB)
 * loaded once per block with int4, 16 rows/block share it.
 */
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdlib.h>

#define PQ1_BLOCK_ROWS 64  /* 256 threads = 32 rows x 8 lanes: the act tile
                            * is still staged by all 256 threads (16 groups,
                            * 16 threads each) and the barrier is still an
                            * 8-warp barrier, but the tile now feeds 32 rows
                            * instead of 16, halving the staging+barrier cost
                            * per unit of work. The 512-thread version of this
                            * idea failed because a 16-warp barrier converges
                            * worse; this keeps the barrier size fixed. */
#define PQ1_LANES 4        /* lanes per row; each lane owns four groups */
#define PQ1_MAXK 12288
#define Q8K_NB 292                  /* Q8_K block: float d + 256 i8 */
#define Q8K_TILE_STRIDE 272         /* shared tile stride: 256 acts (16B
                                     * aligned) + d + pad, 17 u32 words */
/* v4 (E3): process K in chunks of 8 Q8_K blocks (16 groups) so the
 * activation tile is 2.2KB instead of 13KB. Shared per block drops from
 * ~20KB to ~3.4KB, lifting occupancy from 3 blocks/SM (shared-limited) to
 * the 4-block register cap (32 warps vs 24). Each chunk gives every lane
 * exactly one group, so the ordered fp reduce is a short 16-add pass per
 * chunk instead of one long tail at the end. */
#define PQ1_CHUNK_BLOCKS 8
#define PQ1_CHUNK_GROUPS (PQ1_CHUNK_BLOCKS * 2)
#define PQ1_CHUNK_BYTES (PQ1_CHUNK_BLOCKS * Q8K_TILE_STRIDE)
#define PQ1_MAXCHUNK ((PQ1_MAXK / 256) / PQ1_CHUNK_BLOCKS)

__device__ __forceinline__ int pq1_iadd3(int a, int b, int c) {
  return a + b + c; /* fuses to IADD3 */
}

/* One code byte (4 weights) -> b0/b1 plane nibbles (bit k = plane k). */
__device__ __forceinline__ void pq1_split4(uint8_t cb, uint32_t *b0,
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
__device__ __forceinline__ uint32_t pq1_keep4(uint32_t bits4) {
  uint32_t spread = (bits4 * 0x00204081u) & 0x01010101u;
  return spread * 255u;
}

/* masked 4-lane SIGNED byte sum: bytes of apack where the keep byte is
 * 0xFF, sign-extended and added to acc. The signed form is what removes the
 * per-(row,group) correction array: with isum = 2*mu + pu - au computed
 * directly, no bias bookkeeping is needed, which is 132MB less on the GPU
 * for the 8B (6.5% of the arena) plus that array's read traffic. */
__device__ __forceinline__ int pq1_sdot4(uint32_t apack, uint32_t keep,
                                         int acc) {
  uint32_t mk = apack & keep;
  int s0 = (int)(mk << 24) >> 24;
  int s1 = (int)(mk << 16) >> 24;
  int s2 = (int)(mk << 8) >> 24;
  int s3 = (int)((int32_t)mk) >> 24;
  acc = pq1_iadd3(s0, s1, acc);
  return pq1_iadd3(s2, s3, acc);
}

/* Masked 4-lane UNSIGNED byte sum (8 ops): bytes of apack where the keep
 * byte is 0xFF. Two-stage SWAR fold, no sign extension, no IMUL. Superseded
 * in the inner loop by the packed IMAD accumulator (see pq1_gemv_sm53), which
 * computes the same integer in 1 IMAD + 4 folding ops. */
__device__ __forceinline__ int pq1_usum4(uint32_t apack, uint32_t keep,
                                         int acc) {
  uint32_t mk = apack & keep;
  uint32_t t = (mk & 0x00FF00FFu) + ((mk >> 8) & 0x00FF00FFu);
  uint32_t s = (t & 0xFFFFu) + (t >> 16);
  return acc + (int)s;
}

/* full 4-byte SIGNED sum (row-independent Au needs this form). */
__device__ __forceinline__ int pq1_ssum4(uint32_t apack) {
  int s0 = (int)(apack << 24) >> 24;
  int s1 = (int)(apack << 16) >> 24;
  int s2 = (int)(apack << 8) >> 24;
  int s3 = (int)((int32_t)apack) >> 24;
  return pq1_iadd3(s0, s1, s2) + s3;
}

__global__ __launch_bounds__(256, 5)
void pq1_gemv_sm53(const uint8_t* __restrict__ w_codes,
                   const __half* __restrict__ w_scales,
                   const uint8_t* __restrict__ xq, float* __restrict__ y,
                   int K, int N) {
  __shared__ __align__(16) uint8_t sh_x[PQ1_CHUNK_BYTES];
  __shared__ int sh_au[PQ1_CHUNK_GROUPS];
  __shared__ uint32_t sh_keep[16]; /* run #67: 16-entry nibble -> 0xFF byte mask */
  __shared__ float sh_prod[PQ1_BLOCK_ROWS][PQ1_CHUNK_GROUPS];

  int row = blockIdx.x * PQ1_BLOCK_ROWS + (threadIdx.x >> 2);
  int sub = threadIdx.x & 3;
  if (blockIdx.x * PQ1_BLOCK_ROWS >= N) return; /* block-uniform */
  bool active = row < N;
  int rrow = active ? row : 0;

  int K128 = K >> 7;
  int nchunk = K128 / PQ1_CHUNK_GROUPS;
  if (nchunk > PQ1_MAXCHUNK || nchunk <= 0) return;

  float acc = 0.0f; /* lane 0 only; groups accumulated in order */

  for (int ch = 0; ch < nchunk; ch++) {
    /* 1. Fused staging + Au: 16 threads per group (256 threads cover all
     * 16 groups), each copying its own 2 act words into the 272B-strided
     * tile and folding them locally, then a 4-step half-warp shuffle
     * reduces to the group's Au. This removes a barrier per chunk and
     * replaces a 32-iteration serial fold on 16 threads (240 threads idle
     * at the barrier) with ~22 cycles on all of them. */
    const uint32_t *x32 = reinterpret_cast<const uint32_t *>(xq) +
                          (size_t)ch * PQ1_CHUNK_BLOCKS * 73;
    uint32_t *t32 = reinterpret_cast<uint32_t *>(sh_x);
    {
      /* 256 threads / 16 groups = 16 threads per group: gid picks the
       * group, slot picks which 2 of its 32 act words this thread owns. */
      int gid = (int)threadIdx.x >> 4;   /* 0..15 group in this chunk */
      int slot = (int)threadIdx.x & 15;  /* 0..15 within the group */
      int bb = gid >> 1;                 /* its Q8_K block */
      int wb = (gid & 1) * 32 + slot * 2;
      const uint32_t *src = x32 + bb * 73 + 1 + wb;
      uint32_t v0 = __ldg(src);
      uint32_t v1 = __ldg(src + 1);
      uint32_t m0 = v0 ^ 0x80808080u, m1 = v1 ^ 0x80808080u;
      t32[bb * 68 + wb] = m0;
      t32[bb * 68 + wb + 1] = m1;
      if ((int)threadIdx.x < PQ1_CHUNK_BLOCKS) {
        uint32_t dv = __ldg(x32 + (int)threadIdx.x * 73);
        t32[(int)threadIdx.x * 68 + 64] = dv; /* fp32 d, verbatim */
      }
      uint32_t t0 = (m0 & 0x00FF00FFu) + ((m0 >> 8) & 0x00FF00FFu);
      uint32_t t1 = (m1 & 0x00FF00FFu) + ((m1 >> 8) & 0x00FF00FFu);
      int a2 = (int)((t0 & 0xFFFFu) + (t0 >> 16) + (t1 & 0xFFFFu) +
                     (t1 >> 16));
      /* the 16 threads of a group are lanes gid*16..gid*16+15, i.e. one
       * half-warp each, so a stride-8..1 shuffle reduces within the group */
#pragma unroll
      for (int o = 8; o > 0; o >>= 1)
        a2 += __shfl_down_sync(0xFFFFFFFFu, a2, o);
      if (slot == 0) sh_au[gid] = a2;
    }
    /* Run #67: the keep mask depends only on the 4-bit nibble value, so it is a
     * 16-entry table lookup instead of a chain of constant multiplies. ptxas
     * emits each 32-bit constant multiply as 3 XMAD-family ops, so the source
     * `((b4*0x00204081)&0x01010101)*255` costs ~6 SASS ops per nibble; this is
     * 1 LDS.32 on a 64-byte table (word j -> 0xFF in byte j iff bit j set),
     * which #15/#30 showed is not a scarce resource on this part. Built before
     * the staging barrier so every thread sees it afterwards. */
    if (threadIdx.x < 16) {
      uint32_t kb = (uint32_t)threadIdx.x;
      sh_keep[kb] = ((kb * 0x00204081u) & 0x01010101u) * 255u;
    }
    __syncthreads();

    /* 3. two groups per lane (the 8 lanes of a row cover the chunk's 16
     * groups); unroll 1 keeps the inner loop the same size as the single
     * group-per-lane version, so this tests the amortisation alone. */
#pragma unroll 1
    for (int half = 0; half < 4; half++) {
    int g = sub + (half << 2);
    /* one 128-weight group: codes are 128 bits = 4 u32 (one 16B LDG.128,
     * rows are 16B-aligned because the row stride is K/8 bytes); acts are
     * 128 bytes = 8 LDS.128. Per 4 weights: one bit-spread + one masked
     * unsigned sum (~13 ops) versus ~27 for the 2-bit two-plane form. */
    const uint8_t *c =
        w_codes + (size_t)rrow * (K >> 3) +
        (size_t)(ch * PQ1_CHUNK_GROUPS + g) * 16;
    /* this group's activation half: Q8_K block (g>>1), half (g&1) */
    const uint32_t *gb = t32 + (g >> 1) * 68;
    float db;
    __builtin_memcpy(&db, gb + 64, 4);
    const uint32_t *qs = gb + (g & 1) * 32;
    const uint4 *cv = reinterpret_cast<const uint4 *>(c);
    uint4 cw = __ldg(cv);
    uint32_t cws[4] = {(uint32_t)cw.x, (uint32_t)cw.y, (uint32_t)cw.z,
                       (uint32_t)cw.w};
    const uint32_t *av = qs; /* one act word (4 bytes) per 4-weight nibble */
    /* Packed accumulator: pu0/pu1 hold (sum << 16). Each nibble contributes
     * (x0+x1+x2+x3) to the TOP 16 bits through a single IMAD by 0x00010001, so
     * the whole 4-byte masked sum costs one multiply-accumulate instead of the
     * 9-op SWAR fold (pq1_usum4). Per group an accumulator takes 16 nibbles at
     * <= 1020 each, so the packed value stays <= 16320 in the top 16 bits
     * (1.07e9 < 2^31) and never overflows. Integer addition is associative and
     * exact, so this computes the SAME integer Pu as the per-nibble SWAR sum -
     * bit-identical by construction, no rounding, no reassociation question.
     * The float conversion below is untouched. */
    uint32_t pu0 = 0, pu1 = 0;
    int pc = 0;
#pragma unroll
    for (int w = 0; w < 4; w++) {
      uint32_t bits = cws[w];
      pc += __popc(bits);
#pragma unroll
      for (int kk = 0; kk < 8; kk++) {
        uint32_t b4 = (bits >> (kk * 4)) & 0xFu;
        /* 4 one-bit weights -> 4 keep bytes (0xFF keep / 0x00 drop) */
        uint32_t keep = sh_keep[b4];
        uint32_t ap = av[w * 8 + kk]; /* the matching 4 act bytes */
        /* 3-input LOP3 form: folding the selection AND into both byte-mask ANDs
         * lets ptxas emit two LOP32I ops instead of three LOPs plus an LOP for
         * the mk temporary - 1 SASS op per nibble, ~32 in the unrolled body.
         * (ap & keep & M1) takes the low pair of bytes; (ap & keep & 0xFF00FF00)
         * >> 8 takes the high pair into the same 16-bit lanes, which is exactly
         * the previous (mk >> 8) & 0x00FF00FF. Same boolean function, integer
         * sums untouched: bit-identical by construction. */
        uint32_t fold = (ap & keep & 0x00FF00FFu) +
                        ((ap & keep & 0xFF00FF00u) >> 8);
        if (kk & 1) pu1 += fold * 0x00010001u;
        else        pu0 += fold * 0x00010001u;
      }
    }
    /* x' = x+128 unsigned, so with Pu = sum of x' where the bit is 1 and
     * Au/pc from staging: sum(w*x) = 2*Pu - Au + 16384 - 256*pc, exact.
     * pu0/pu1 are packed (<< 16); shifting them down recovers Pu exactly. */
    int isum = (int)(((pu0 >> 16) + (pu1 >> 16)) << 1) - sh_au[g] + 16384 -
               (pc << 8);
    float ws = active ? __half2float(__ldg(w_scales + (size_t)rrow * K128 +
                                          (ch * PQ1_CHUNK_GROUPS + g)))
                      : 0.0f;
    sh_prod[threadIdx.x >> 2][g] =
        __fmul_rn(__fmul_rn(ws, db), (float)isum);
    __syncwarp();

    } /* end of the two-groups-per-lane loop */

    /* 4. ordered reduce for this chunk: lane 0 of each row adds the 16
     * products in group order (chunks are processed in order too). */
    if (sub == 0) {
      const float *p = sh_prod[threadIdx.x >> 2];
#pragma unroll
      for (int k = 0; k < PQ1_CHUNK_GROUPS; k++) acc = __fadd_rn(acc, p[k]);
    }
    __syncthreads(); /* the tile is reused by the next chunk */
  }

  if (sub == 0 && active) y[row] = acc;
}
/* ---- Run #45 experiment: 128 rows per block (2 lanes/row) -----------------
 * Continues the session's only validated scaling trend: #20 halved the number of
 * blocks by going 32 -> 64 rows/block (measured +4.2% real in #26), because every
 * block re-stages the same activation chunk (2.2KB) from global and pays the
 * barrier; more rows per block amortises that over more work. #20's 512-thread
 * version (32 rows... 256 threads / 8 lanes) REGRESSED -9.1% - a 16-warp barrier
 * converges worse - so this keeps the block at 8 warps and takes the rows by
 * HALVING the lanes per row: 2 lanes/row x 128 rows = 256 threads, each lane
 * owning 8 of the chunk's 16 groups. The barrier size is unchanged; the staging
 * traffic per row halves (N/64 -> N/128 blocks per matrix). Numerically identical
 * by construction: the per-group products and the group-order reduce are the same
 * integers and the same fp32 sequence. */
#define PQ1_BLOCK_ROWS2 128

__global__ __launch_bounds__(256, 5)
void pq1_gemv_r128(const uint8_t* __restrict__ w_codes,
                   const __half* __restrict__ w_scales,
                   const uint8_t* __restrict__ xq, float* __restrict__ y,
                   int K, int N) {
  __shared__ __align__(16) uint8_t sh_x[PQ1_CHUNK_BYTES];
  __shared__ int sh_au[PQ1_CHUNK_GROUPS];
  __shared__ uint32_t sh_keep[16]; /* run #67: 16-entry nibble -> 0xFF byte mask */
  __shared__ float sh_prod[PQ1_BLOCK_ROWS2][PQ1_CHUNK_GROUPS];

  int row = blockIdx.x * PQ1_BLOCK_ROWS2 + (threadIdx.x >> 1);
  int sub = threadIdx.x & 1;
  if (blockIdx.x * PQ1_BLOCK_ROWS2 >= N) return; /* block-uniform */
  bool active = row < N;
  int rrow = active ? row : 0;

  int K128 = K >> 7;
  int nchunk = K128 / PQ1_CHUNK_GROUPS;
  if (nchunk > PQ1_MAXCHUNK || nchunk <= 0) return;

  float acc = 0.0f;

  for (int ch = 0; ch < nchunk; ch++) {
    const uint32_t* x32 = reinterpret_cast<const uint32_t*>(xq) +
                          (size_t)ch * PQ1_CHUNK_BLOCKS * 73;
    uint32_t* t32 = reinterpret_cast<uint32_t*>(sh_x);
    {
      int gid = (int)threadIdx.x >> 4;
      int slot = (int)threadIdx.x & 15;
      int bb = gid >> 1;
      int wb = (gid & 1) * 32 + slot * 2;
      const uint32_t* src = x32 + bb * 73 + 1 + wb;
      uint32_t v0 = __ldg(src);
      uint32_t v1 = __ldg(src + 1);
      uint32_t m0 = v0 ^ 0x80808080u, m1 = v1 ^ 0x80808080u;
      t32[bb * 68 + wb] = m0;
      t32[bb * 68 + wb + 1] = m1;
      if ((int)threadIdx.x < PQ1_CHUNK_BLOCKS) {
        uint32_t dv = __ldg(x32 + (int)threadIdx.x * 73);
        t32[(int)threadIdx.x * 68 + 64] = dv;
      }
      uint32_t t0 = (m0 & 0x00FF00FFu) + ((m0 >> 8) & 0x00FF00FFu);
      uint32_t t1 = (m1 & 0x00FF00FFu) + ((m1 >> 8) & 0x00FF00FFu);
      int a2 = (int)((t0 & 0xFFFFu) + (t0 >> 16) + (t1 & 0xFFFFu) +
                     (t1 >> 16));
#pragma unroll
      for (int o = 8; o > 0; o >>= 1)
        a2 += __shfl_down_sync(0xFFFFFFFFu, a2, o);
      if (slot == 0) sh_au[gid] = a2;
    }
    /* Run #67: the keep mask depends only on the 4-bit nibble value, so it is a
     * 16-entry table lookup instead of a chain of constant multiplies. ptxas
     * emits each 32-bit constant multiply as 3 XMAD-family ops, so the source
     * `((b4*0x00204081)&0x01010101)*255` costs ~6 SASS ops per nibble; this is
     * 1 LDS.32 on a 64-byte table (word j -> 0xFF in byte j iff bit j set),
     * which #15/#30 showed is not a scarce resource on this part. Built before
     * the staging barrier so every thread sees it afterwards. */
    if (threadIdx.x < 16) {
      uint32_t kb = (uint32_t)threadIdx.x;
      sh_keep[kb] = ((kb * 0x00204081u) & 0x01010101u) * 255u;
    }
    __syncthreads();

#pragma unroll 1
    for (int half = 0; half < 8; half++) {
      int g = sub + (half << 1);
      const uint8_t* c =
          w_codes + (size_t)rrow * (K >> 3) +
          (size_t)(ch * PQ1_CHUNK_GROUPS + g) * 16;
      const uint32_t* gb = t32 + (g >> 1) * 68;
      float db;
      __builtin_memcpy(&db, gb + 64, 4);
      const uint32_t* av = gb + (g & 1) * 32;
      const uint4* cv = reinterpret_cast<const uint4*>(c);
      uint4 cw = __ldg(cv);
      uint32_t cws[4] = {(uint32_t)cw.x, (uint32_t)cw.y, (uint32_t)cw.z,
                         (uint32_t)cw.w};
      uint32_t pu0 = 0, pu1 = 0;
      int pc = 0;
#pragma unroll
      for (int w = 0; w < 4; w++) {
        uint32_t bits = cws[w];
        pc += __popc(bits);
#pragma unroll
        for (int kk = 0; kk < 8; kk++) {
          uint32_t b4 = (bits >> (kk * 4)) & 0xFu;
          uint32_t keep = sh_keep[b4];
          uint32_t ap = av[w * 8 + kk];
          uint32_t fold = (ap & keep & 0x00FF00FFu) +
                          ((ap & keep & 0xFF00FF00u) >> 8);
          if (kk & 1) pu1 += fold * 0x00010001u;
          else        pu0 += fold * 0x00010001u;
        }
      }
      int isum = (int)(((pu0 >> 16) + (pu1 >> 16)) << 1) - sh_au[g] + 16384 -
                 (pc << 8);
      float ws = active ? __half2float(__ldg(w_scales + (size_t)rrow * K128 +
                                            (size_t)(ch * PQ1_CHUNK_GROUPS + g)))
                        : 0.0f;
      sh_prod[threadIdx.x >> 1][g] = __fmul_rn(__fmul_rn(ws, db), (float)isum);
      __syncwarp();
    }

    if (sub == 0) {
      const float* p = sh_prod[threadIdx.x >> 1];
#pragma unroll
      for (int k = 0; k < PQ1_CHUNK_GROUPS; k++) acc = __fadd_rn(acc, p[k]);
    }
    __syncthreads();
  }

  if (sub == 0 && active) y[row] = acc;
}

extern "C" int pq1_gemv_r128_launch(const uint8_t* w_codes,
                                    const unsigned short* w_scales_h,
                                    const uint8_t* xq, float* y, int K, int N,
                                    void* stream) {
  if ((K & 127) != 0) return -2;
  if (K <= 0 || K > PQ1_MAXK || N <= 0) return -2;
  dim3 block(256, 1, 1);
  dim3 grid((N + PQ1_BLOCK_ROWS2 - 1) / PQ1_BLOCK_ROWS2, 1, 1);
  cudaStream_t s = (cudaStream_t)stream;
  pq1_gemv_r128<<<grid, block, 0, s>>>(
      w_codes, reinterpret_cast<const __half*>(w_scales_h), xq, y, K, N);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

/* ---- Run #46 experiment: 256 rows per block (1 lane/row) -------------------
 * One more step of the amortisation trend that paid at every halving so far:
 * 16 -> 32 rows = +2.0% (#17), 32 -> 64 = +4.2% (#26), 64 -> 128 = +2.8% (#45).
 * With ONE lane per row a block covers 256 rows, the activation chunk is staged
 * once per 256 rows, and the epilogue disappears: the lane that owns a row can
 * accumulate its own 16 groups in registers, in ascending group order, which is
 * exactly the fp32 order the 64- and 128-row kernels use (their lane 0 sums
 * sh_prod[row][0..15] in ascending order). So no sh_prod array is needed at all:
 * shared memory drops to the 2.2KB tile + 64B of Au, and each row's result is
 * written by the single lane that computed it. Bit-exact by construction. */
#define PQ1_BLOCK_ROWS3 256

__global__ __launch_bounds__(256, 5)
void pq1_gemv_r256(const uint8_t* __restrict__ w_codes,
                   const __half* __restrict__ w_scales,
                   const uint8_t* __restrict__ xq, float* __restrict__ y,
                   int K, int N) {
  __shared__ __align__(16) uint8_t sh_x[PQ1_CHUNK_BYTES];
  __shared__ int sh_au[PQ1_CHUNK_GROUPS];
  __shared__ uint32_t sh_keep[16]; /* run #67: 16-entry nibble -> 0xFF byte mask */

  int row = blockIdx.x * PQ1_BLOCK_ROWS3 + threadIdx.x;
  if (blockIdx.x * PQ1_BLOCK_ROWS3 >= N) return; /* block-uniform */
  bool active = row < N;
  int rrow = active ? row : 0;

  int K128 = K >> 7;
  int nchunk = K128 / PQ1_CHUNK_GROUPS;
  if (nchunk > PQ1_MAXCHUNK || nchunk <= 0) return;

  float acc = 0.0f; /* this row's running sum, groups in ascending order */

  for (int ch = 0; ch < nchunk; ch++) {
    const uint32_t* x32 = reinterpret_cast<const uint32_t*>(xq) +
                          (size_t)ch * PQ1_CHUNK_BLOCKS * 73;
    uint32_t* t32 = reinterpret_cast<uint32_t*>(sh_x);
    {
      int gid = (int)threadIdx.x >> 4;
      int slot = (int)threadIdx.x & 15;
      int bb = gid >> 1;
      int wb = (gid & 1) * 32 + slot * 2;
      const uint32_t* src = x32 + bb * 73 + 1 + wb;
      uint32_t v0 = __ldg(src);
      uint32_t v1 = __ldg(src + 1);
      uint32_t m0 = v0 ^ 0x80808080u, m1 = v1 ^ 0x80808080u;
      t32[bb * 68 + wb] = m0;
      t32[bb * 68 + wb + 1] = m1;
      if ((int)threadIdx.x < PQ1_CHUNK_BLOCKS) {
        uint32_t dv = __ldg(x32 + (int)threadIdx.x * 73);
        t32[(int)threadIdx.x * 68 + 64] = dv;
      }
      uint32_t t0 = (m0 & 0x00FF00FFu) + ((m0 >> 8) & 0x00FF00FFu);
      uint32_t t1 = (m1 & 0x00FF00FFu) + ((m1 >> 8) & 0x00FF00FFu);
      int a2 = (int)((t0 & 0xFFFFu) + (t0 >> 16) + (t1 & 0xFFFFu) +
                     (t1 >> 16));
#pragma unroll
      for (int o = 8; o > 0; o >>= 1)
        a2 += __shfl_down_sync(0xFFFFFFFFu, a2, o);
      if (slot == 0) sh_au[gid] = a2;
    }
    /* Run #67: the keep mask depends only on the 4-bit nibble value, so it is a
     * 16-entry table lookup instead of a chain of constant multiplies. ptxas
     * emits each 32-bit constant multiply as 3 XMAD-family ops, so the source
     * `((b4*0x00204081)&0x01010101)*255` costs ~6 SASS ops per nibble; this is
     * 1 LDS.32 on a 64-byte table (word j -> 0xFF in byte j iff bit j set),
     * which #15/#30 showed is not a scarce resource on this part. Built before
     * the staging barrier so every thread sees it afterwards. */
    if (threadIdx.x < 16) {
      uint32_t kb = (uint32_t)threadIdx.x;
      sh_keep[kb] = ((kb * 0x00204081u) & 0x01010101u) * 255u;
    }
    __syncthreads();

#pragma unroll 1
    for (int g = 0; g < PQ1_CHUNK_GROUPS; g++) {
      const uint8_t* c =
          w_codes + (size_t)rrow * (K >> 3) + (size_t)(ch * PQ1_CHUNK_GROUPS + g) * 16;
      const uint32_t* gb = t32 + (g >> 1) * 68;
      float db;
      __builtin_memcpy(&db, gb + 64, 4);
      const uint32_t* av = gb + (g & 1) * 32;
      const uint4* cv = reinterpret_cast<const uint4*>(c);
      uint4 cw = __ldg(cv);
      uint32_t cws[4] = {(uint32_t)cw.x, (uint32_t)cw.y, (uint32_t)cw.z,
                         (uint32_t)cw.w};
      uint32_t pu0 = 0, pu1 = 0;
      int pc = 0;
#pragma unroll
      for (int w = 0; w < 4; w++) {
        uint32_t bits = cws[w];
        pc += __popc(bits);
#pragma unroll
        for (int kk = 0; kk < 8; kk++) {
          uint32_t b4 = (bits >> (kk * 4)) & 0xFu;
          uint32_t keep = sh_keep[b4];
          uint32_t ap = av[w * 8 + kk];
          uint32_t fold = (ap & keep & 0x00FF00FFu) +
                          ((ap & keep & 0xFF00FF00u) >> 8);
          if (kk & 1) pu1 += fold * 0x00010001u;
          else        pu0 += fold * 0x00010001u;
        }
      }
      int isum = (int)(((pu0 >> 16) + (pu1 >> 16)) << 1) - sh_au[g] + 16384 -
                 (pc << 8);
      float ws = active ? __half2float(__ldg(w_scales + (size_t)rrow * K128 +
                                            (size_t)(ch * PQ1_CHUNK_GROUPS + g)))
                        : 0.0f;
      acc = __fadd_rn(acc, __fmul_rn(__fmul_rn(ws, db), (float)isum));
    }
    __syncthreads();
  }

  if (active) y[row] = acc;
}

extern "C" int pq1_gemv_r256_launch(const uint8_t* w_codes,
                                    const unsigned short* w_scales_h,
                                    const uint8_t* xq, float* y, int K, int N,
                                    void* stream) {
  if ((K & 127) != 0) return -2;
  if (K <= 0 || K > PQ1_MAXK || N <= 0) return -2;
  dim3 block(256, 1, 1);
  dim3 grid((N + PQ1_BLOCK_ROWS3 - 1) / PQ1_BLOCK_ROWS3, 1, 1);
  cudaStream_t s = (cudaStream_t)stream;
  pq1_gemv_r256<<<grid, block, 0, s>>>(
      w_codes, reinterpret_cast<const __half*>(w_scales_h), xq, y, K, N);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

/* ---- Run #41 experiment: T-MAC bit-sliced inner loop -----------------------
 * Why: SWAR's masked sum costs ~2.5 lane-ops/weight organised as a ~7-deep
 * dependent chain per 4 weights. The measured slope on this kernel is 0.51% of
 * decode time per 1% of that chain, and the chain is ~51% of decode time - so
 * SWAR's whole family is capped at ~2x. Bit-slicing amortises the work over 32
 * weights per instruction instead of 4:
 *   Pu(code word W) = sum_p 2^p * popc(plane_p & W)
 * is an EXACT integer identity for "sum the unsigned biased bytes x' = x+128 of
 * the selected activations": plane word p bit a carries bit p of activation a,
 * so activation a contributes sum_p 2^p * bit_p(x'_a) = x'_a, once. Integer and
 * order-independent, hence bit-identical to the SWAR sum (no reassociation, no
 * rounding) - the same integer isum feeds the same fp epilogue.
 * Cost: 8 AND + 8 POPC + 8 accumulate per 32 weights = 0.75 ALU + 0.25 XU ops
 * per weight, against 2.5 ALU per weight. The activation side is read as 8
 * plane words per 32 weights = 1 byte/weight: IDENTICAL load count to today.
 * The planes are built once per token on the host (pq1_build_planes); they
 * mirror the Q8_K block layout (73 words/block: word 0 = fp32 d, words 1..64 =
 * 64 plane words) so the kernel's staging code and every address is unchanged. */
#define TM_GROUP_WORDS 32 /* 4 32-weight blocks x 8 planes per 128-weight group */
/* Run #71: TWO rows per thread (128 rows per block). Knockout timing at
 * 24576x4096 (PQ1_TIME hot loop, 2.27 ms): removing the weight LDG -2.5%,
 * replacing POPC by LOP -1.5%, removing the plane LDS.128s -35%. The plane
 * words were re-read from shared for every row (8 LDS.128 per 128 weights,
 * each a 4-phase 512-byte warp transaction), so the kernel is LDS-bound, not
 * XU- or DRAM-bound. Each thread now applies one plane load to the code words
 * of two rows (row and row+64): plane LDS per weight halves, POPC count is
 * unchanged. Every row's dot product keeps its own pu/pc/isum, its own fp
 * epilogue and its own in-order 16-group sum, so the result is bit-identical
 * to the one-row kernel (pq1_parity at every engine shape). */
#define TM_ROWS 128
#define TM_HALF 64

/* Loop iter 3: pc folded into plane 7 (V=1, default; PQ1_TMV=0 is the A/B arm).
 * pu = S7 + 128*popc(P7&m) with popc(P7&m) = pc - popc(~P7&m), so
 * 2pu - 256pc = 2*(S7 - 128*popc(~P7&m)): the 4 POPC + 3 adds of pc per row
 * and group disappear (POPC 72 -> 64 per 2-row group iteration, same isum, so
 * the fp epilogue and the result are unchanged). Kernel, PQ1_TIME 200 iters,
 * two alternations: 24576x4096 1.430/1.423 -> 1.373/1.373 ms, 4096x12288
 * 0.809/0.805 -> 0.788/0.794, 6144x4096 0.417/0.416 -> 0.411/0.414, 4096x4096
 * 0.299/0.297 -> 0.284/0.295. Measured out alongside: a carry-save adder across
 * code words 0..2 per plane (popc(x0)+popc(x1)+popc(x2) = popc(x0^x1^x2) +
 * 2*popc(maj)), POPC 64 -> 48 at equal instruction count: plane-major order
 * spills at 48 regs (1.64 ms) and is still slower at 64 regs/no spill (1.44
 * ms): the kernel is issue-bound, POPC only paid where instructions went too. */
template <int V>
__global__ __launch_bounds__(256, 5)
void pq1_gemv_tm(const uint8_t* __restrict__ w_codes,
                 const __half* __restrict__ w_scales,
                 const uint32_t* __restrict__ xpl, const int* __restrict__ xau,
                 float* __restrict__ y, int K, int N) {
  __shared__ __align__(16) uint8_t sh_x[PQ1_CHUNK_BYTES];
  __shared__ int sh_au[PQ1_CHUNK_GROUPS];
  __shared__ float sh_prod[TM_ROWS][PQ1_CHUNK_GROUPS];

  int lrow = threadIdx.x >> 2;                   /* 0..63 */
  int row0 = blockIdx.x * TM_ROWS + lrow;
  int row1 = row0 + TM_HALF;
  int sub = threadIdx.x & 3;
  if (blockIdx.x * TM_ROWS >= N) return; /* block-uniform */
  bool act0 = row0 < N, act1 = row1 < N;
  int rrow0 = act0 ? row0 : 0;
  int rrow1 = act1 ? row1 : 0;

  int K128 = K >> 7;
  int nchunk = K128 / PQ1_CHUNK_GROUPS;
  if (nchunk > PQ1_MAXCHUNK || nchunk <= 0) return;

  float acc0 = 0.0f, acc1 = 0.0f; /* lane 0 only; groups accumulated in order */
  uint32_t* t32 = reinterpret_cast<uint32_t*>(sh_x);
  const uint8_t* wrow0 = w_codes + (size_t)rrow0 * (size_t)(K >> 3);
  const uint8_t* wrow1 = w_codes + (size_t)rrow1 * (size_t)(K >> 3);
  const __half* srow0 = w_scales + (size_t)rrow0 * (size_t)K128;
  const __half* srow1 = w_scales + (size_t)rrow1 * (size_t)K128;

  for (int ch = 0; ch < nchunk; ch++) {
    /* Stage this chunk's plane words (and the fp32 scales) into shared: same
     * 73-word-per-block source layout and 68-word-per-block tile layout as
     * the SWAR kernel; the host applied the 0x80 bias while building the
     * planes and supplies Au. */
    const uint32_t* x32 = reinterpret_cast<const uint32_t*>(xpl) +
                          (size_t)ch * PQ1_CHUNK_BLOCKS * 73;
    {
      int gid = (int)threadIdx.x >> 4;  /* 0..15 group in this chunk */
      int slot = (int)threadIdx.x & 15; /* 0..15 within the group */
      int bb = gid >> 1;                /* its Q8_K block */
      int wb = (gid & 1) * 32 + slot * 2;
      const uint32_t* src = x32 + bb * 73 + 1 + wb;
      t32[bb * 68 + wb] = __ldg(src);
      t32[bb * 68 + wb + 1] = __ldg(src + 1);
      if ((int)threadIdx.x < PQ1_CHUNK_BLOCKS) {
        uint32_t dv = __ldg(x32 + (int)threadIdx.x * 73);
        t32[(int)threadIdx.x * 68 + 64] = dv; /* fp32 d, verbatim */
      }
      if (slot == 0) sh_au[gid] = xau[ch * PQ1_CHUNK_GROUPS + gid];
    }
    __syncthreads();

#pragma unroll 1
    for (int half = 0; half < 4; half++) {
      int g = sub + (half << 2);
      size_t goff = (size_t)(ch * PQ1_CHUNK_GROUPS + g);
      /* 128 weights = 4 code words (one uint4) = 32 plane words (8 uint4) */
      const uint4* pv =
          reinterpret_cast<uint4*>(t32) + (g >> 1) * 17 + (g & 1) * 8;
      uint4 cw0 = __ldg(reinterpret_cast<const uint4*>(wrow0 + goff * 16));
      uint4 cw1 = __ldg(reinterpret_cast<const uint4*>(wrow1 + goff * 16));
      int pc0 = V ? 0 : __popc(cw0.x) + __popc(cw0.y) + __popc(cw0.z) + __popc(cw0.w);
      int pc1 = V ? 0 : __popc(cw1.x) + __popc(cw1.y) + __popc(cw1.z) + __popc(cw1.w);
      int pu0 = 0, pu1 = 0;
#pragma unroll
      for (int ci = 0; ci < 4; ci++) {
        uint32_t m0 = (ci == 0) ? cw0.x : (ci == 1) ? cw0.y : (ci == 2) ? cw0.z : cw0.w;
        uint32_t m1 = (ci == 0) ? cw1.x : (ci == 1) ? cw1.y : (ci == 2) ? cw1.z : cw1.w;
        uint4 lo4 = pv[ci * 2];
        uint4 hi4 = pv[ci * 2 + 1];
        int lo0 = __popc(lo4.x & m0) + 2 * __popc(lo4.y & m0) +
                  4 * __popc(lo4.z & m0) + 8 * __popc(lo4.w & m0);
        int hi0 = 16 * __popc(hi4.x & m0) + 32 * __popc(hi4.y & m0) +
                  64 * __popc(hi4.z & m0) +
                  (V ? -128 * __popc(~hi4.w & m0) : 128 * __popc(hi4.w & m0));
        pu0 += lo0 + hi0;
        int lo1 = __popc(lo4.x & m1) + 2 * __popc(lo4.y & m1) +
                  4 * __popc(lo4.z & m1) + 8 * __popc(lo4.w & m1);
        int hi1 = 16 * __popc(hi4.x & m1) + 32 * __popc(hi4.y & m1) +
                  64 * __popc(hi4.z & m1) +
                  (V ? -128 * __popc(~hi4.w & m1) : 128 * __popc(hi4.w & m1));
        pu1 += lo1 + hi1;
      }
      int au = sh_au[g];
      /* identical to the SWAR kernel's epilogue: same integer isum, same fp */
      int isum0 = (pu0 << 1) - au + 16384 - (pc0 << 8);
      int isum1 = (pu1 << 1) - au + 16384 - (pc1 << 8);
      float db;
      __builtin_memcpy(&db, t32 + (g >> 1) * 68 + 64, 4);
      float ws0 = act0 ? __half2float(__ldg(srow0 + goff)) : 0.0f;
      float ws1 = act1 ? __half2float(__ldg(srow1 + goff)) : 0.0f;
      sh_prod[lrow][g] = __fmul_rn(__fmul_rn(ws0, db), (float)isum0);
      sh_prod[lrow + TM_HALF][g] = __fmul_rn(__fmul_rn(ws1, db), (float)isum1);
      __syncwarp();
    }

    if (sub == 0) {
      const float* p0 = sh_prod[lrow];
      const float* p1 = sh_prod[lrow + TM_HALF];
#pragma unroll
      for (int k = 0; k < PQ1_CHUNK_GROUPS; k++) acc0 = __fadd_rn(acc0, p0[k]);
#pragma unroll
      for (int k = 0; k < PQ1_CHUNK_GROUPS; k++) acc1 = __fadd_rn(acc1, p1[k]);
    }
    __syncthreads(); /* the tile is reused by the next chunk */
  }

  if (sub == 0 && act0) y[row0] = acc0;
  if (sub == 0 && act1) y[row1] = acc1;
}

extern "C" int pq1_gemv_tm_launch(const uint8_t* w_codes,
                                  const unsigned short* w_scales_h,
                                  const uint32_t* xpl, const int* xau, float* y,
                                  int K, int N, void* stream) {
  if ((K & 127) != 0) return -2;
  if (K <= 0 || K > PQ1_MAXK || N <= 0) return -2;
  dim3 block(256, 1, 1);
  dim3 grid((N + TM_ROWS - 1) / TM_ROWS, 1, 1);
  cudaStream_t s = (cudaStream_t)stream;
  static int tmv = -1;
  if (tmv < 0) { const char* e = getenv("PQ1_TMV"); tmv = e ? atoi(e) : 1; }
  if (tmv == 1)
    pq1_gemv_tm<1><<<grid, block, 0, s>>>(
        w_codes, reinterpret_cast<const __half*>(w_scales_h), xpl, xau, y, K, N);
  else
    pq1_gemv_tm<0><<<grid, block, 0, s>>>(
        w_codes, reinterpret_cast<const __half*>(w_scales_h), xpl, xau, y, K, N);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}


/* Group-stationary Harley-Seal kernel (PQ1_HS=1). The T-MAC kernel above is
 * issue-bound on re-reading the plane words from shared memory for every row
 * (8 LDS.128 per 128 weights at ~7.4 transactions each). Here a lane owns ONE
 * 128-weight group of the activation for the whole launch: its 32 plane words
 * live in registers, a warp covers 32 groups (K = 4096; K = 12288 uses a team
 * of WR = 3 warps), and the weight rows stream through it, 512 contiguous
 * bytes per warp per row (one LDG.128 per lane).
 *
 * Per (row, group) the 32 masked plane words are reduced by a Harley-Seal
 * carry-save network (LOP3 0x96 sum / 0xE8 majority, carries rippling up one
 * level, ONE popc per level, 10 levels): 78 LOP3 + 10 POPC + 9 shift-adds
 * instead of 32 AND + 32 POPC + 31 shift-adds, measured 1.26 vs 0.97
 * word-planes/clk on this SM in a register-only microbenchmark. The plane-7
 * term -128*popc(~P7&m) is rewritten with positive weights only:
 * sum_j popc(~P7_j & m_j) = 128 - sum_j popc(P7_j | ~m_j), so with
 *   S = sum_{q<7} 2^q popc(P_q & m) + 128 popc(P_7 | ~m)   (4 words)
 * the V=1 isum (pu<<1) - au + 16384 equals 2*S - au - 16384: the same integer
 * for every input, hence the same product and the same fp result.
 *
 * Epilogue: prod = (ws * db) * (float)isum, written to a padded shared tile
 * P[row][group] for a batch of RB rows; lanes 0..RB-1 then each add one
 * row's G products from 0.0f in ascending group order, the reference's order. */
#ifndef HS_TPB1
#define HS_TPB1 8
#endif
#ifndef HS_MINB1
#define HS_MINB1 4
#endif
#ifndef HS_TPB3
#define HS_TPB3 2
#endif
#ifndef HS_MINB3
#define HS_MINB3 1
#endif
__device__ __forceinline__ uint32_t hs_x3(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}
__device__ __forceinline__ uint32_t hs_mj(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0xE8;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}
/* S over one 128-weight group: p[q][j] = plane q of 32-activation word j */
__device__ __forceinline__ int hs_group(const uint32_t (&p)[8][4], uint4 w) {
  const uint32_t m0 = w.x, m1 = w.y, m2 = w.z, m3 = w.w;
  uint32_t a0, a1, a2, s, t, ca, cb, cc, na, nb, nc, lv;
  int acc;
  /* level 0: 4 words -> 1 + carries ca, cb */
  a0 = p[0][0] & m0; a1 = p[0][1] & m1; a2 = p[0][2] & m2;
  s = hs_x3(a0, a1, a2); ca = hs_mj(a0, a1, a2);
  t = p[0][3] & m3;
  lv = s ^ t; cb = s & t;
  acc = __popc(lv);
  /* level 1: 4 words + 2 carries -> 1 + carries na, nb, nc */
  a0 = p[1][0] & m0; a1 = p[1][1] & m1; a2 = p[1][2] & m2;
  s = hs_x3(a0, a1, a2); na = hs_mj(a0, a1, a2);
  t = hs_x3(s, ca, cb); nb = hs_mj(s, ca, cb);
  a0 = p[1][3] & m3;
  lv = t ^ a0; nc = t & a0;
  acc += __popc(lv) << 1;
  ca = na; cb = nb; cc = nc;
  /* levels 2..6: 4 words + 3 carries -> 1 + 3 carries */
#pragma unroll
  for (int q = 2; q < 7; q++) {
    a0 = p[q][0] & m0; a1 = p[q][1] & m1; a2 = p[q][2] & m2;
    s = hs_x3(a0, a1, a2); na = hs_mj(a0, a1, a2);
    t = hs_x3(s, ca, cb); nb = hs_mj(s, ca, cb);
    a0 = p[q][3] & m3;
    lv = hs_x3(t, cc, a0); nc = hs_mj(t, cc, a0);
    acc += __popc(lv) << q;
    ca = na; cb = nb; cc = nc;
  }
  /* level 7: inputs P7 | ~m (weight +128, see above) */
  a0 = p[7][0] | ~m0; a1 = p[7][1] | ~m1; a2 = p[7][2] | ~m2;
  s = hs_x3(a0, a1, a2); na = hs_mj(a0, a1, a2);
  t = hs_x3(s, ca, cb); nb = hs_mj(s, ca, cb);
  a0 = p[7][3] | ~m3;
  lv = hs_x3(t, cc, a0); nc = hs_mj(t, cc, a0);
  acc += __popc(lv) << 7;
  /* level 8: 3 carries -> 1 + carry to level 9 */
  lv = hs_x3(na, nb, nc); s = hs_mj(na, nb, nc);
  acc += __popc(lv) << 8;
  acc += __popc(s) << 9;
  return acc;
}

/* Sign-magnitude form (PQ1_HSV=1, default). Q8_K quants lie in [-127, 127]
 * (|iscale * x| <= 127 before nearest_int; -128 never occurs), so with the
 * sign plane N and |x| in 7 planes Q_q, x = (1 - 2N)|x|, and with e = m ^ N
 * (set where the product is positive) x*w = |x|(2e - 1), hence
 *   isum = 2 * sum_q 2^q popc((m ^ N) & Q_q) - A,   A = sum |x|,
 * the same integer as the offset form above. A masked input is one LOP3
 * (0x28) and there are 28 instead of 32; the planes are converted once per
 * launch in registers (bit-sliced conditional negate), so the activation
 * prep and its host reference are unchanged.
 *
 * The tree is rebalanced for this SM: POPC runs on the quarter-rate pipe and
 * issues beside the LOP3s, while a full adder costs 2 ALU slots to save one
 * POPC, so the all-adder tree above over-compresses. One adder on even levels
 * (popc its sum, the 4th word and the two incoming carries), two chained
 * adders on odd levels (popc the final sum, two carries out): 10 adders and
 * 18 POPC for the 28 inputs. Same S, so the same isum. */
#define HS_LS(q, a, b, c, d)                                              \
  a = hs_mk(m0, n[0], p[q][0]); b = hs_mk(m1, n[1], p[q][1]);             \
  c = hs_mk(m2, n[2], p[q][2]); d = hs_mk(m3, n[3], p[q][3]);
__device__ __forceinline__ uint32_t hs_mk(uint32_t m, uint32_t n, uint32_t q) {
  uint32_t r; /* (m ^ n) & q */
  asm("lop3.b32 %0, %1, %2, %3, 0x28;" : "=r"(r) : "r"(m), "r"(n), "r"(q));
  return r;
}
__device__ __forceinline__ int hs_group_s(const uint32_t (&p)[8][4], const uint32_t (&n)[4],
                                          uint4 w) {
  const uint32_t m0 = w.x, m1 = w.y, m2 = w.z, m3 = w.w;
  uint32_t a, b, c, d, s, t, k0, k1, k2;
  HS_LS(0, a, b, c, d);
  s = hs_x3(a, b, c); k0 = hs_mj(a, b, c);
  int acc = __popc(s) + __popc(d);
#pragma unroll
  for (int q = 1; q < 7; q += 2) {
    /* odd level q: 4 words + carry k0 -> popc(t), carries k1, k2 */
    HS_LS(q, a, b, c, d);
    s = hs_x3(a, b, c); k1 = hs_mj(a, b, c);
    t = hs_x3(s, d, k0); k2 = hs_mj(s, d, k0);
    acc += __popc(t) << q;
    /* even level q+1: 4 words + carries k1, k2 -> 4 popc, carry k0 */
    HS_LS(q + 1, a, b, c, d);
    s = hs_x3(a, b, c); k0 = hs_mj(a, b, c);
    acc += (__popc(s) + __popc(d) + __popc(k1) + __popc(k2)) << (q + 1);
  }
  return acc + (__popc(k0) << 7);
}

#ifndef HS_RB1
#define HS_RB1 6
#endif
#ifndef HS_PD1
#define HS_PD1 3
#endif
#ifndef HS_RB3
#define HS_RB3 8
#endif
#ifndef HS_PD3
#define HS_PD3 2
#endif

/* step j of the segmented reduction (below): warp wt adds its 32 partials of
 * batch b = j - wt (if 0 <= b <= jlast) onto warp wt-1's sum; the last warp
 * (batch j - WR + 1, row base hrow) stores y. */
template <int WR, int RB, int PS>
__device__ __forceinline__ void hs_seg_step(float (*tiles)[RB * PS], float (*sacc)[RB],
                                            float* __restrict__ y, int j, int jlast, int wt,
                                            int lane, int hrow) {
  const int b = j - wt;
  if (b < 0 || b > jlast || lane >= RB) return;
  const float4* src = reinterpret_cast<const float4*>(tiles[b & 3] + lane * PS + wt * 32);
  float acc = wt == 0 ? 0.0f : sacc[b & 3][lane];
#pragma unroll
  for (int k = 0; k < 8; k++) {
    const float4 v = src[k];
    acc = __fadd_rn(acc, v.x);
    acc = __fadd_rn(acc, v.y);
    acc = __fadd_rn(acc, v.z);
    acc = __fadd_rn(acc, v.w);
  }
  if (wt == WR - 1) y[hrow + lane] = acc;
  else sacc[b & 3][lane] = acc;
}

/* SEG (WR > 1, PQ1_HSG=1 default): the row chain over the G group partials is
 * split by warp segment instead of run whole by warp 0. After batch j's
 * barrier warp w adds its own 32 partials of batch j-w onto the running sum
 * warp w-1 left for that batch (sacc), and the last warp stores y. Same chain,
 * same order; the reduction issue is spread evenly over the team instead of
 * making warp 0 the straggler at every barrier (the whole chain on warp 0 cost
 * the K=12288 launch ~69 us of 472). A tile lives 4 batches (NBUF 4). */
/* RB rows per batch, PD rows in flight per lane (a ring of PD load slots
 * that must restart aligned at every batch: RB % PD == 0). RB 6 / PD 3 at
 * K = 4096: deeper prefetch in the same 64 registers (offset form: 8 / 2). */
template <int WR, int TPB, int MINB, int RB, int PD, bool SM, bool SEG = false>
__global__ __launch_bounds__(32 * WR * TPB, MINB)
void pq1_gemv_hs(const uint8_t* __restrict__ w_codes,
                 const __half* __restrict__ w_scales,
                 const uint32_t* __restrict__ xpl, const int* __restrict__ xau,
                 float* __restrict__ y, int N) {
  static_assert(RB % PD == 0, "load ring must realign at each batch");
  constexpr int G = 32 * WR;   /* 128-weight groups per row */
  constexpr int PS = G + 4;    /* tile stride: 16-byte rows, conflict-free */
  static_assert(!SEG || WR > 1, "segmented reduction needs WR > 1");
  constexpr int NBUF = SEG ? 4 : WR > 1 ? 2 : 1;
  __shared__ __align__(16) float sh_p[TPB][NBUF][RB * PS];
  __shared__ float sacc[TPB][SEG ? 4 : 1][RB];
  const int warp = (int)threadIdx.x >> 5, lane = (int)threadIdx.x & 31;
  const int team = warp / WR, wt = warp - team * WR;
  const int g = wt * 32 + lane;

  uint32_t p[8][4];
  {
    const uint32_t* pb = xpl + (g >> 1) * 73 + 1 + (g & 1) * 32;
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int q = 0; q < 8; q++) p[q][j] = __ldg(pb + j * 8 + q);
  }
  const float db = __int_as_float((int)__ldg(xpl + (g >> 1) * 73));
  uint32_t n[4] = {0u, 0u, 0u, 0u};
  int cg;
  if (SM) {
    /* u = x + 128 planes -> N = ~P7, |x| = N ? (u ^ 127) + 1 : u & 127 */
    int asum = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      n[j] = ~p[7][j];
      uint32_t c = n[j];
#pragma unroll
      for (int q = 0; q < 7; q++) {
        const uint32_t t = p[q][j] ^ n[j];
        p[q][j] = t ^ c;
        c &= t;
        asum += __popc(p[q][j]) << q;
      }
    }
    cg = -asum;
  } else {
    cg = -__ldg(xau + g) - 16384;
  }

  /* Batches of RB consecutive rows; the last batch is moved back to end at
   * row N-1 (it recomputes a few rows of its neighbour and stores the same
   * bits), so every row address is in range and no load needs a clamp. */
  const int nbatch = (N + RB - 1) / RB;
  const int stride = (int)gridDim.x * TPB;
  int bi = (int)blockIdx.x * TPB + team;
  if (bi >= nbatch) return; /* team-uniform */
  const uint4* wq = reinterpret_cast<const uint4*>(w_codes) + g;
  const __half* sq = w_scales + g;

  int r0 = min(bi * RB, N - RB);
  const uint4* wb = wq + (size_t)r0 * G;
  const __half* sb = sq + (size_t)r0 * G;
  uint4 cwr[PD];
  __half wsr[PD];
#pragma unroll
  for (int i = 0; i < PD; i++) { cwr[i] = __ldg(wb + i * G); wsr[i] = __ldg(sb + i * G); }
  int buf = 0;
  int jb = 0, rh1 = 0, rh2 = 0;   /* SEG: batch index, row bases of batches jb-1, jb-2 */
  for (;;) {
    const int bnext = bi + stride;
    const int rn0 = bnext < nbatch ? min(bnext * RB, N - RB) : r0;
#ifdef HS_EXP_HOT
    const uint4* wn = wq + (size_t)(team & 3) * RB * G;
    const __half* sn = sq + (size_t)(team & 3) * RB * G;
#else
    const uint4* wn = wq + (size_t)rn0 * G;
    const __half* sn = sq + (size_t)rn0 * G;
#endif
    float* P = sh_p[team][buf];
#pragma unroll
    for (int rr = 0; rr < RB; rr++) {
      const uint4 cw = cwr[rr % PD];
      const __half wsh = wsr[rr % PD];
#ifdef HS_EXP_NOSC
      if (rr + PD < RB) {
        cwr[rr % PD] = __ldg(wb + (rr + PD) * G);
      } else {
        cwr[rr % PD] = __ldg(wn + (rr + PD - RB) * G);
      }
#else
      if (rr + PD < RB) {
        cwr[rr % PD] = __ldg(wb + (rr + PD) * G);
        wsr[rr % PD] = __ldg(sb + (rr + PD) * G);
      } else {
        cwr[rr % PD] = __ldg(wn + (rr + PD - RB) * G);
        wsr[rr % PD] = __ldg(sn + (rr + PD - RB) * G);
      }
#endif
#ifdef HS_EXP_NOCOMP
      const int isum = __popc(cw.x ^ cw.y ^ cw.z ^ cw.w) + cg;
#else
      const int isum = 2 * (SM ? hs_group_s(p, n, cw) : hs_group(p, cw)) + cg;
#endif
      P[rr * PS + g] = __fmul_rn(__fmul_rn(__half2float(wsh), db), (float)isum);
    }
    if (WR == 1) __syncwarp();
    else asm volatile("bar.sync %0, %1;" ::"r"(team + 1), "r"(WR * 32));
    if (SEG) {
      hs_seg_step<WR, RB, PS>(sh_p[team], sacc[team], y, jb, jb, wt, lane, WR == 3 ? rh2 : rh1);
      if (bnext >= nbatch) {
        /* drain: the last WR-1 segments of the last WR-1 batches */
#pragma unroll
        for (int d = 1; d < WR; d++) {
          asm volatile("bar.sync %0, %1;" ::"r"(team + 1), "r"(WR * 32));
          const int bl = WR - 1 - d;   /* the storing warp's batch is jb - bl */
          hs_seg_step<WR, RB, PS>(sh_p[team], sacc[team], y, jb + d, jb, wt, lane,
                                  bl == 2 ? rh2 : bl == 1 ? rh1 : r0);
        }
        break;
      }
      rh2 = rh1; rh1 = r0; jb++;
      buf = jb & 3;
      bi = bnext; r0 = rn0; wb = wn; sb = sn;
      continue;
    }
#ifdef HS_ROT
    if (wt == (WR > 1 ? (bi / stride) % WR : 0) && lane < RB) {
#else
    if (wt == 0 && lane < RB) {
#endif
      const float4* src = reinterpret_cast<const float4*>(P + lane * PS);
      float acc = 0.0f;
#ifdef HS_EXP_NORED
      for (int k = 0; k < 1; k++) {
#else
#pragma unroll
      for (int k = 0; k < G / 4; k++) {
#endif
        float4 v = src[k];
        acc = __fadd_rn(acc, v.x);
        acc = __fadd_rn(acc, v.y);
        acc = __fadd_rn(acc, v.z);
        acc = __fadd_rn(acc, v.w);
      }
      y[r0 + lane] = acc;
    }
    if (bnext >= nbatch) break;
    if (WR == 1) __syncwarp(); /* single tile: read before the next batch writes */
    else buf ^= 1;              /* two tiles: the next barrier orders the reuse */
    bi = bnext; r0 = rn0; wb = wn; sb = sn;
  }
}

template <int WR, int TPB, int MINB, int RB, int PD, bool SM, bool SEG = false>
static int hs_launch(const uint8_t* w_codes, const __half* w_scales, const uint32_t* xpl,
                     const int* xau, float* y, int N, cudaStream_t s) {
  static int nres = 0;
  if (nres == 0) {
    int per = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, pq1_gemv_hs<WR, TPB, MINB, RB, PD, SM, SEG>,
                                                  32 * WR * TPB, 0);
    int dev = 0, nsm = 1;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    nres = (per > 0 ? per : 1) * nsm;
  }
  int nbatch = (N + RB - 1) / RB;
  int nblk = (nbatch + TPB - 1) / TPB;
  if (nblk > nres) nblk = nres;
  pq1_gemv_hs<WR, TPB, MINB, RB, PD, SM, SEG><<<nblk, 32 * WR * TPB, 0, s>>>(w_codes, w_scales, xpl,
                                                                         xau, y, N);
  return 0;
}

/* PQ1_HSV=0 is the A/B arm: the offset-form tree at RB 8 / PD 2, the kernel
 * of loop-261014-2216 row 12. */
extern "C" int pq1_gemv_hs_launch(const uint8_t* w_codes,
                                  const unsigned short* w_scales_h,
                                  const uint32_t* xpl, const int* xau, float* y,
                                  int K, int N, void* stream) {
  static int hv = -1;
  if (hv < 0) { const char* e = getenv("PQ1_HSV"); hv = e ? atoi(e) : 1; }
  static int hg = -1;
  if (hg < 0) { const char* e = getenv("PQ1_HSG"); hg = e ? atoi(e) : 1; }
  if (N < 16) return pq1_gemv_tm_launch(w_codes, w_scales_h, xpl, xau, y, K, N, stream);
  const __half* ws = reinterpret_cast<const __half*>(w_scales_h);
  cudaStream_t s = (cudaStream_t)stream;
  if (K == 4096) {
    if (hv) hs_launch<1, HS_TPB1, HS_MINB1, HS_RB1, HS_PD1, true>(w_codes, ws, xpl, xau, y, N, s);
    else hs_launch<1, 8, 4, 8, 2, false>(w_codes, ws, xpl, xau, y, N, s);
  } else if (K == 12288) {
    if (hv && hg) hs_launch<3, HS_TPB3, HS_MINB3, HS_RB3, HS_PD3, true, true>(w_codes, ws, xpl, xau, y, N, s);
    else if (hv) hs_launch<3, HS_TPB3, HS_MINB3, HS_RB3, HS_PD3, true>(w_codes, ws, xpl, xau, y, N, s);
    else hs_launch<3, 2, 1, 8, 2, false>(w_codes, ws, xpl, xau, y, N, s);
  } else {
    return pq1_gemv_tm_launch(w_codes, w_scales_h, xpl, xau, y, K, N, stream);
  }
  return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

/* 1 when pq1_tmx_launch(K, N) reads the per-group sums xau: the sign-magnitude
 * hs kernels rebuild them from the planes, tm and the offset-form arm do not. */
extern "C" int pq1_tmx_uses_au(int K, int N) {
  const char* e = getenv("PQ1_HS");
  const char* v = getenv("PQ1_HSV");
  if ((e && !atoi(e)) || (v && !atoi(v))) return 1;
  return N < 16 || (K != 4096 && K != 12288);
}

/* Engine entry: every T-MAC GEMV goes to the hs kernel above (default since
 * loop-261014-2216 row 12: +30.7% decode tps, ab_pair both orders);
 * PQ1_HS=0 restores the bit-serial tm. */
extern "C" int pq1_tmx_launch(const uint8_t* w_codes, const unsigned short* w_scales_h,
                              const uint32_t* xpl, const int* xau, float* y, int K, int N,
                              void* stream) {
  static int hs = -1;
  if (hs < 0) { const char* e = getenv("PQ1_HS"); hs = e ? atoi(e) : 1; }
  return hs ? pq1_gemv_hs_launch(w_codes, w_scales_h, xpl, xau, y, K, N, stream)
            : pq1_gemv_tm_launch(w_codes, w_scales_h, xpl, xau, y, K, N, stream);
}


/* Device-side T-MAC activation prep (run #72). The engine used to build the
 * bit planes and the per-group sums on the host (pq1_build_planes/_au: 0.21 ms
 * at K=4096, 0.64 ms at K=12288 per call) and then needed five API calls
 * (2 H2D, launch, sync, D2H) that the SWAR path had folded into one CUDA
 * graph. That overhead is why T-MAC lost on every shape but gate+up and the
 * head even after #71 made the kernel itself the fastest at every shape.
 * This kernel is the same integer transform, one thread per 128-activation
 * group (K/128 threads: 32..96), so the raw Q8_K bytes can be uploaded as for
 * the SWAR kernels and {H2D, prep, tm, D2H} captured in one graph. The planes
 * and sums are integers built from the same bytes, so the result is identical
 * to the host build (pq1_parity checks the two byte-for-byte). */
__global__ void pq1_tm_prep(const uint8_t* __restrict__ xq, uint32_t* __restrict__ pl,
                            int* __restrict__ au, int ngroups) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= ngroups) return;
  int b = g >> 1, half = g & 1;
  const uint8_t* src = xq + (size_t)b * 292 + 4 + half * 128;
  uint32_t* dst = pl + (size_t)b * 73 + 1 + half * 32;
  if (half == 0)
    pl[(size_t)b * 73] = *reinterpret_cast<const uint32_t*>(xq + (size_t)b * 292);
  else {
    /* words 65..72 mirror Q8_K's bsums area: unread by the kernel, zeroed so the
     * device build is byte-identical to a zero-initialised host build */
#pragma unroll
    for (int w = 65; w < 73; w++) pl[(size_t)b * 73 + w] = 0u;
  }
  int s = 0;
#pragma unroll
  for (int w = 0; w < 4; w++) {
    uint32_t p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0, p5 = 0, p6 = 0, p7 = 0;
#pragma unroll
    for (int a = 0; a < 32; a++) {
      unsigned v = (unsigned)src[w * 32 + a] ^ 0x80u;
      s += (int)v;
      p0 |= ((v >> 0) & 1u) << a;
      p1 |= ((v >> 1) & 1u) << a;
      p2 |= ((v >> 2) & 1u) << a;
      p3 |= ((v >> 3) & 1u) << a;
      p4 |= ((v >> 4) & 1u) << a;
      p5 |= ((v >> 5) & 1u) << a;
      p6 |= ((v >> 6) & 1u) << a;
      p7 |= ((v >> 7) & 1u) << a;
    }
    dst[w * 8 + 0] = p0; dst[w * 8 + 1] = p1; dst[w * 8 + 2] = p2; dst[w * 8 + 3] = p3;
    dst[w * 8 + 4] = p4; dst[w * 8 + 5] = p5; dst[w * 8 + 6] = p6; dst[w * 8 + 7] = p7;
  }
  au[g] = s;
}

/* Loop iter 6: the same transform with one warp per group (PQ1_TMPW=1,
 * default; 0 is the A/B arm). Lane a holds activation byte a of each 32-byte
 * word, plane word b = __ballot_sync(bit b): lane a's bit lands at bit a, as
 * in the serial loop above. au is an integer sum, so the shuffle tree gives
 * the identical value. 32..96 serial threads -> 32..96 warps. */
__global__ __launch_bounds__(128) void pq1_tm_prep_w(const uint8_t* __restrict__ xq,
                                                    uint32_t* __restrict__ pl,
                                                    int* __restrict__ au, int ngroups) {
  int g = blockIdx.x * 4 + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  if (g >= ngroups) return; /* warp-uniform */
  int b = g >> 1, half = g & 1;
  const uint8_t* src = xq + (size_t)b * 292 + 4 + half * 128;
  uint32_t* dst = pl + (size_t)b * 73 + 1 + half * 32;
  if (half == 0) {
    if (lane == 0) pl[(size_t)b * 73] = *reinterpret_cast<const uint32_t*>(xq + (size_t)b * 292);
  } else if (lane < 8) {
    pl[(size_t)b * 73 + 65 + lane] = 0u;
  }
  int s = 0;
#pragma unroll
  for (int w = 0; w < 4; w++) {
    unsigned v = (unsigned)src[w * 32 + lane] ^ 0x80u;
    s += (int)v;
    uint32_t pb[8];
#pragma unroll
    for (int q = 0; q < 8; q++) pb[q] = __ballot_sync(0xffffffffu, (v >> q) & 1u);
    if (lane < 8) {
      uint32_t o = pb[0];
#pragma unroll
      for (int q = 1; q < 8; q++) if (lane == q) o = pb[q];
      dst[w * 8 + lane] = o;
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if (lane == 0) au[g] = s;
}

extern "C" int pq1_tm_prep_launch(const uint8_t* xq_d, uint32_t* pl_d, int* au_d,
                                  int K, void* stream) {
  if ((K & 255) != 0 || K <= 0 || K > PQ1_MAXK) return -2;
  int ng = K >> 7;
  static int pw = -1;
  if (pw < 0) { const char* e = getenv("PQ1_TMPW"); pw = e ? atoi(e) : 1; }
  dim3 block(128, 1, 1);
  if (pw) {
    dim3 gridw((ng + 3) / 4, 1, 1);
    pq1_tm_prep_w<<<gridw, block, 0, (cudaStream_t)stream>>>(xq_d, pl_d, au_d, ng);
  } else {
    dim3 grid((ng + 127) / 128, 1, 1);
    pq1_tm_prep<<<grid, block, 0, (cudaStream_t)stream>>>(xq_d, pl_d, au_d, ng);
  }
  return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

extern "C" int pq1_gemv_sm53_launch(const uint8_t *w_codes,
                                    const unsigned short *w_scales_h,
                                    const uint8_t *xq, float *y, int K, int N,
                                    void *stream) {
  if ((K & 127) != 0) return -2;
  if (K <= 0 || K > PQ1_MAXK || N <= 0) return -2;
  dim3 block(256, 1, 1);
  dim3 grid((N + PQ1_BLOCK_ROWS - 1) / PQ1_BLOCK_ROWS, 1, 1);
  cudaStream_t s = (cudaStream_t)stream;
  pq1_gemv_sm53<<<grid, block, 0, s>>>(
      w_codes, reinterpret_cast<const __half *>(w_scales_h), xq, y, K, N);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

/* ---- Run #54: batched (multi-tensor) launches -----------------------------
 * Run #53's phase timing (PQ1_KTIME=3) measured the engine's per-call API cost:
 * ~40us H2D + ~45us launch + ~100us D2H, i.e. 46.6ms per token = 12.8% of the
 * 365ms decode step. The engine calls one GEMV per projection - 253 launches per
 * token - and several of them consume the SAME activation vector: q, k, v all
 * read XN, and gate and up both read the post-attention norm. Folding each such
 * group into ONE launch pays the H2D and the launch once for the whole group
 * (the D2H must stay per output buffer: the CPU reads QF/KF/VF separately).
 *
 * Numerics: each output row's dot product is computed by exactly the same code,
 * in the same order, with the same accumulator as the single-tensor kernels - the
 * only addition is a three-way select of the base pointers from the global row
 * index. Bit-exactness is verified by pq1_parity against the host reference at
 * every shape the batch path is used for.
 *
 * LANES = 1 -> 256 rows/block (the r256 epilogue: the lane owns its row's 16
 * groups and writes y itself); LANES = 2 -> 128 rows/block (the r128 epilogue:
 * two lanes per row, combined through sh_prod, lane 0 summing the 16 groups in
 * ascending group order). The row counts of every tensor in this model (3072,
 * 4096, 1024, 12288, 151669) are multiples of 256 except the 151669-row head,
 * which is never batched, so a block never straddles two tensors. */
#define PQ1_BN 3

template <int LANES>
__global__ __launch_bounds__(256, 5)
void pq1_gemv_batch(const uint8_t* __restrict__ c0, const uint8_t* __restrict__ c1,
                    const uint8_t* __restrict__ c2,
                    const __half* __restrict__ s0, const __half* __restrict__ s1,
                    const __half* __restrict__ s2,
                    float* __restrict__ y0, float* __restrict__ y1,
                    float* __restrict__ y2,
                    int n0, int n1, int n2,
                    const uint8_t* __restrict__ xq, int K, int N) {
  const int RPB = 256 / LANES;
  __shared__ __align__(16) uint8_t sh_x[PQ1_CHUNK_BYTES];
  __shared__ int sh_au[PQ1_CHUNK_GROUPS];
  __shared__ uint32_t sh_keep[16]; /* run #67: 16-entry nibble -> 0xFF byte mask */
  __shared__ float sh_prod[RPB][PQ1_CHUNK_GROUPS];

  int row = blockIdx.x * RPB + (LANES == 1 ? threadIdx.x : threadIdx.x >> 1);
  int sub = (LANES == 2) ? (threadIdx.x & 1) : 0;
  if (blockIdx.x * RPB >= N) return; /* block-uniform */
  bool active = row < N;

  /* one-time row -> tensor resolution (row is a multiple of RPB <= every row count) */
  const uint8_t* codes = c0;
  const __half* sc = s0;
  float* y = y0;
  int rrow = row;
  if (row >= n0) {
    if (row < n0 + n1) { codes = c1; sc = s1; y = y1; rrow = row - n0; }
    else               { codes = c2; sc = s2; y = y2; rrow = row - n0 - n1; }
  }

  int K128 = K >> 7;
  int nchunk = K128 / PQ1_CHUNK_GROUPS;
  if (nchunk > PQ1_MAXCHUNK || nchunk <= 0) return;

  float acc = 0.0f;

  for (int ch = 0; ch < nchunk; ch++) {
    const uint32_t* x32 = reinterpret_cast<const uint32_t*>(xq) +
                          (size_t)ch * PQ1_CHUNK_BLOCKS * 73;
    uint32_t* t32 = reinterpret_cast<uint32_t*>(sh_x);
    {
      int gid = (int)threadIdx.x >> 4;
      int slot = (int)threadIdx.x & 15;
      int bb = gid >> 1;
      int wb = (gid & 1) * 32 + slot * 2;
      const uint32_t* src = x32 + bb * 73 + 1 + wb;
      uint32_t v0 = __ldg(src);
      uint32_t v1 = __ldg(src + 1);
      uint32_t m0 = v0 ^ 0x80808080u, m1 = v1 ^ 0x80808080u;
      t32[bb * 68 + wb] = m0;
      t32[bb * 68 + wb + 1] = m1;
      if ((int)threadIdx.x < PQ1_CHUNK_BLOCKS) {
        uint32_t dv = __ldg(x32 + (int)threadIdx.x * 73);
        t32[(int)threadIdx.x * 68 + 64] = dv;
      }
      uint32_t t0 = (m0 & 0x00FF00FFu) + ((m0 >> 8) & 0x00FF00FFu);
      uint32_t t1 = (m1 & 0x00FF00FFu) + ((m1 >> 8) & 0x00FF00FFu);
      int a2 = (int)((t0 & 0xFFFFu) + (t0 >> 16) + (t1 & 0xFFFFu) +
                     (t1 >> 16));
#pragma unroll
      for (int o = 8; o > 0; o >>= 1)
        a2 += __shfl_down_sync(0xFFFFFFFFu, a2, o);
      if (slot == 0) sh_au[gid] = a2;
    }
    /* Run #67: the keep mask depends only on the 4-bit nibble value, so it is a
     * 16-entry table lookup instead of a chain of constant multiplies. ptxas
     * emits each 32-bit constant multiply as 3 XMAD-family ops, so the source
     * `((b4*0x00204081)&0x01010101)*255` costs ~6 SASS ops per nibble; this is
     * 1 LDS.32 on a 64-byte table (word j -> 0xFF in byte j iff bit j set),
     * which #15/#30 showed is not a scarce resource on this part. Built before
     * the staging barrier so every thread sees it afterwards. */
    if (threadIdx.x < 16) {
      uint32_t kb = (uint32_t)threadIdx.x;
      sh_keep[kb] = ((kb * 0x00204081u) & 0x01010101u) * 255u;
    }
    __syncthreads();

#pragma unroll 1
    for (int g = 0; g < PQ1_CHUNK_GROUPS; g++) {
      const uint8_t* c =
          codes + (size_t)rrow * (K >> 3) +
          (size_t)(ch * PQ1_CHUNK_GROUPS + g) * 16;
      const uint32_t* gb = t32 + (g >> 1) * 68;
      float db;
      __builtin_memcpy(&db, gb + 64, 4);
      const uint32_t* av = gb + (g & 1) * 32;
      const uint4* cv = reinterpret_cast<const uint4*>(c);
      uint4 cw = __ldg(cv);
      uint32_t cws[4] = {(uint32_t)cw.x, (uint32_t)cw.y, (uint32_t)cw.z,
                         (uint32_t)cw.w};
      uint32_t pu0 = 0, pu1 = 0;
      int pc = 0;
#pragma unroll
      for (int w = 0; w < 4; w++) {
        uint32_t bits = cws[w];
        pc += __popc(bits);
#pragma unroll
        for (int kk = 0; kk < 8; kk++) {
          uint32_t b4 = (bits >> (kk * 4)) & 0xFu;
          uint32_t keep = sh_keep[b4];
          uint32_t ap = av[w * 8 + kk];
          uint32_t fold = (ap & keep & 0x00FF00FFu) +
                          ((ap & keep & 0xFF00FF00u) >> 8);
          if (kk & 1) pu1 += fold * 0x00010001u;
          else        pu0 += fold * 0x00010001u;
        }
      }
      int isum = (int)(((pu0 >> 16) + (pu1 >> 16)) << 1) - sh_au[g] + 16384 -
                 (pc << 8);
      float ws = active ? __half2float(__ldg(sc + (size_t)rrow * K128 +
                                            (size_t)(ch * PQ1_CHUNK_GROUPS + g)))
                        : 0.0f;
      if (LANES == 1) {
        acc = __fadd_rn(acc, __fmul_rn(__fmul_rn(ws, db), (float)isum));
      } else {
        sh_prod[threadIdx.x >> 1][g] = __fmul_rn(__fmul_rn(ws, db), (float)isum);
        __syncwarp();
      }
    }

    /* Epilogues are kept fully compile-time separate per instantiation: taking
     * the address of `acc` (e.g. `const float* p = &acc`) forces the accumulator
     * into LOCAL MEMORY, which made the shared epilogue of the first batch draft
     * 4x slower per launch than the specialised kernels (run #54: sync 4.6ms per
     * q+k+v batch vs 1.1ms for the three separate launches, at identical 40-register
     * occupancy). LANES is a template parameter, so exactly one branch survives. */
    if (LANES == 2 && sub == 0) {
#pragma unroll
      for (int k = 0; k < PQ1_CHUNK_GROUPS; k++)
        acc = __fadd_rn(acc, sh_prod[threadIdx.x >> 1][k]);
    }
    __syncthreads();
  }

  if (active && (LANES == 1 || sub == 0)) y[rrow] = acc;
}

extern "C" int pq1_gemv_batch_launch(int use256, const uint8_t* const* codes,
                                     const unsigned short* const* scales,
                                     float* const* ys, const int* rows, int n,
                                     int K, int N, const uint8_t* xq,
                                     void* stream) {
  if ((K & 127) != 0) return -2;
  if (K <= 0 || K > PQ1_MAXK || N <= 0 || n < 1 || n > PQ1_BN) return -2;
  const uint8_t* c[PQ1_BN] = {codes[0], n > 1 ? codes[1] : codes[0],
                              n > 2 ? codes[2] : codes[0]};
  const __half* s[PQ1_BN];
  float* y[PQ1_BN];
  int rn[PQ1_BN];
  for (int i = 0; i < PQ1_BN; i++) {
    s[i] = reinterpret_cast<const __half*>(scales[i]);
    y[i] = ys[i];
    rn[i] = (i < n) ? rows[i] : 0;
  }
  dim3 block(256, 1, 1);
  cudaStream_t st = (cudaStream_t)stream;
  int rc;
  if (use256) {
    dim3 grid((N + PQ1_BLOCK_ROWS3 - 1) / PQ1_BLOCK_ROWS3, 1, 1);
    pq1_gemv_batch<1><<<grid, block, 0, st>>>(c[0], c[1], c[2], s[0], s[1], s[2],
                                             y[0], y[1], y[2], rn[0], rn[1], rn[2],
                                             xq, K, N);
    rc = (cudaGetLastError() == cudaSuccess) ? 0 : -1;
  } else {
    dim3 grid((N + PQ1_BLOCK_ROWS2 - 1) / PQ1_BLOCK_ROWS2, 1, 1);
    pq1_gemv_batch<2><<<grid, block, 0, st>>>(c[0], c[1], c[2], s[0], s[1], s[2],
                                             y[0], y[1], y[2], rn[0], rn[1], rn[2],
                                             xq, K, N);
    rc = (cudaGetLastError() == cudaSuccess) ? 0 : -1;
  }
  return rc;
}

/* ---- loop iter 2: fused FFN tail, silu(g)*u + Q8_K quant on the device ----
 * One block per 256-activation Q8_K block, one thread per element. The value
 * is fwd8's CPU chain bit for bit: e = glibc-2.27 expf(-g) (the table+poly
 * algorithm of pq1_expf_neon.h in double, plus glibc's |x|>=88 special cases
 * that the NEON port hands to libm), sg = 1/(1+e), v = (g*sg)*u; then
 * q8_K_quant: amax = max|v|, max = v at the FIRST index reaching it,
 * iscale = -127/max, q = MIN(127, mantissa(fma(v, iscale, 1.5*2^23)) - 2^22)
 * (gcc contracts fwd8.c's vmulq+vaddq to fmla), d = 1/iscale.
 * Every f32 op is inline PTX WITHOUT .ftz: this file is built --use_fast_math
 * (ftz=true), the A57 keeps denormals, and the _rn intrinsics would inherit
 * the flush. Double ops are never flushed or contracted as __d*_rn. */
__constant__ unsigned long long PQ1_EXP2F_TAB_D[32] = {
    0x3ff0000000000000ULL, 0x3fefd9b0d3158574ULL, 0x3fefb5586cf9890fULL, 0x3fef9301d0125b51ULL,
    0x3fef72b83c7d517bULL, 0x3fef54873168b9aaULL, 0x3fef387a6e756238ULL, 0x3fef1e9df51fdee1ULL,
    0x3fef06fe0a31b715ULL, 0x3feef1a7373aa9cbULL, 0x3feedea64c123422ULL, 0x3feece086061892dULL,
    0x3feebfdad5362a27ULL, 0x3feeb42b569d4f82ULL, 0x3feeab07dd485429ULL, 0x3feea47eb03a5585ULL,
    0x3feea09e667f3bcdULL, 0x3fee9f75e8ec5f74ULL, 0x3feea11473eb0187ULL, 0x3feea589994cce13ULL,
    0x3feeace5422aa0dbULL, 0x3feeb737b0cdc5e5ULL, 0x3feec49182a3f090ULL, 0x3feed503b23e255dULL,
    0x3feee89f995ad3adULL, 0x3feeff76f2fb5e47ULL, 0x3fef199bdd85529cULL, 0x3fef3720dcef9069ULL,
    0x3fef5818dcfba487ULL, 0x3fef7c97337b9b5fULL, 0x3fefa4afa2a490daULL, 0x3fefd0765b6e4540ULL,
};
__device__ __forceinline__ float pq1_fadd(float a, float b) { float r; asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float pq1_fsub(float a, float b) { float r; asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float pq1_fmul(float a, float b) { float r; asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float pq1_fdiv(float a, float b) { float r; asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float pq1_ffma(float a, float b, float c) { float r; asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r; }
/* q8_K's nearest_int on the image of fma(v, iscale, 1.5*2^23), capped at 127.
 * 0 * inf (iscale overflows when a block has 0 < amax < 127/FLT_MAX and an exact
 * zero) is the default NaN 0x7fc00000 on the A57 (q = 0) but 0x7fffffff on
 * sm_53 (q = 127): every NaN maps to the A57's. */
__device__ __forceinline__ int pq1_q8n(float v, float iscale) {
  unsigned u = __float_as_uint(pq1_ffma(v, iscale, 12582912.0f));
  if ((u & 0x7fffffffu) > 0x7f800000u) u = 0x7fc00000u;
  return min((int)(u & 0x007fffffu) - 0x00400000, 127);
}
__device__ __forceinline__ double pq1_f2d(float a) { double r; asm("cvt.f64.f32 %0, %1;" : "=d"(r) : "f"(a)); return r; }
__device__ __forceinline__ float pq1_d2f(double a) { float r; asm("cvt.rn.f32.f64 %0, %1;" : "=f"(r) : "d"(a)); return r; }

__device__ __forceinline__ float pq1_expf_dev(float x) {
  const uint32_t ux = __float_as_uint(x), abstop = (ux >> 20) & 0x7ffu;
  if (abstop >= 0x42bu) {                          /* |x| >= 88 or NaN: glibc's branches */
    if (ux == 0xff800000u) return 0.0f;            /* -inf */
    if (abstop >= 0x7f8u) return pq1_fadd(x, x);   /* +inf, NaN */
    if ((int32_t)ux > 0 && ux > 0x42b17217u) return __uint_as_float(0x7f800000u); /* oflow */
    if ((int32_t)ux < 0 && ux > 0xc2cff1b4u) return 0.0f;                         /* uflow */
  }
  const double z = __dmul_rn(__longlong_as_double(0x40471547652b82feLL), pq1_f2d(x));
  const double SH = __longlong_as_double(0x4338000000000000LL);
  double kd = __dadd_rn(z, SH);
  const unsigned long long ki = (unsigned long long)__double_as_longlong(kd);
  kd = __dsub_rn(kd, SH);
  const double r = __dsub_rn(z, kd);
  const double s = __longlong_as_double((long long)(PQ1_EXP2F_TAB_D[ki & 31] + (ki << 47)));
  const double zz = __fma_rn(__longlong_as_double(0x3ebc6af84b912394LL), r,
                             __longlong_as_double(0x3f2ebfce50fac4f3LL));
  const double r2 = __dmul_rn(r, r);
  double y = __fma_rn(__longlong_as_double(0x3f962e42ff0c52d6LL), r, 1.0);
  y = __fma_rn(zz, r2, y);
  return pq1_d2f(__dmul_rn(y, s));
}

/* pq1_expf_ff (loop-261014-2216 row 20): pq1_expf_dev's result from fp32
 * float-float arithmetic, for silu. sm_53 issues fp64 at 1/32 of the fp32
 * rate, and the 11 double ops of pq1_expf_dev were ~2/3 of pq1_silu_q8k<1>.
 * For -86.5 < x < 88 it evaluates the same formula with pairs (h, l):
 * z = x * InvLn2N, k = rint(z), r = z - k, y = 1 + C2 r + (C1 + C0 r) r^2,
 * m = y * 2^((k&31)/32) in [0.98, 2.03), then rounds m to float and scales by
 * 2^(k>>5) (exact: the result stays normal). Its error against the double
 * D = y_d * s_d that glibc rounds is far below 2^-36 (residuals of the split
 * constants <= 2^-49, z exact to ~2^-36 absolute in a |r| <= 0.5 interval
 * worth ln2/32 per unit). When |m - f| is not at least 2^-36 inside half an
 * ulp of f (and r is not within 2^-20 of +-0.5, where glibc's k could differ)
 * the result is not certified and it falls back to pq1_expf_dev. Outside the
 * interval it is pq1_expf_dev. Verified exhaustively against pq1_expf_dev over
 * all 2^32 inputs on the Nano (pq1_parity expf). */
__constant__ uint32_t PQ1_EXP2F_FF[64] = {   /* 2^(j/32) = hi + lo, |residual| <= 2^-49.2 rel */
    0x3f800000u, 0x00000000u, 0x3f82cd87u, 0xb34ea7a9u, 0x3f85aac3u, 0x334f9891u, 0x3f88980fu, 0xb37eda4bu,
    0x3f8b95c2u, 0xb260aba1u, 0x3f8ea43au, 0xb3697465u, 0x3f91c3d3u, 0x33675624u, 0x3f94f4f0u, 0xb32e0212u,
    0x3f9837f0u, 0x33231b71u, 0x3f9b8d3au, 0xb30c5563u, 0x3f9ef532u, 0x33412342u, 0x3fa27043u, 0x30c3125au,
    0x3fa5fed7u, 0xb32c9d5eu, 0x3fa9a15bu, 0xb3162b08u, 0x3fad583fu, 0xb22deaf6u, 0x3fb123f6u, 0xb37c5aa8u,
    0x3fb504f3u, 0x32cfe77au, 0x3fb8fbafu, 0x330ec5f7u, 0x3fbd08a4u, 0xb3414fe8u, 0x3fc12c4du, 0xb2d6663eu,
    0x3fc5672au, 0x320aa837u, 0x3fc9b9beu, 0xb37323a2u, 0x3fce248cu, 0x3228fc24u, 0x3fd2a81eu, 0xb35c1daau,
    0x3fd744fdu, 0xb2d4a58au, 0x3fdbfbb8u, 0xb3504a1cu, 0x3fe0ccdfu, 0xb21eab59u, 0x3fe5b907u, 0xb2441be6u,
    0x3feac0c7u, 0xb24116deu, 0x3fefe4bau, 0xb348464au, 0x3ff5257du, 0x32292436u, 0x3ffa83b3u, 0xb2923758u,
};
/* certified fast path; *ok = 0 means the caller must use pq1_expf_dev */
/* tab: the 2^(j/32) pairs as th[0..31], tl[0..31] in shared memory (divergent
 * __constant__ reads serialize per distinct address) */
__device__ __forceinline__ float pq1_expf_ffc(float x, bool* ok, const float* __restrict__ tab) {
  const float CH = __uint_as_float(0x4238aa3bu), CL = __uint_as_float(0x35257060u);
  const float C2H = __uint_as_float(0x3cb17218u), C2L = __uint_as_float(0xae73ad2au);
  const float C1H = __uint_as_float(0x3975fe73u), C1L = __uint_as_float(0xacf053b1u);
  const float C0H = __uint_as_float(0x35e357c2u);
  const float zh = pq1_fmul(x, CH);
  float zl = pq1_ffma(x, CH, -zh);
  zl = pq1_ffma(x, CL, zl);
  const float kf = rintf(zh);
  const float rh0 = pq1_fsub(zh, kf);                 /* exact */
  const float rh = pq1_fadd(rh0, zl);                 /* fast two-sum, |rh0| >= |zl| or rh0 = 0 */
  const float rl = pq1_fsub(zl, pq1_fsub(rh, rh0));
  /* C2 r */
  const float ph = pq1_fmul(C2H, rh);
  float pl = pq1_ffma(C2H, rh, -ph);
  pl = pq1_ffma(C2H, rl, pl);
  pl = pq1_ffma(C2L, rh, pl);
  /* r^2 */
  const float sh = pq1_fmul(rh, rh);
  float sl = pq1_ffma(rh, rh, -sh);
  sl = pq1_ffma(pq1_fadd(rh, rh), rl, sl);
  /* w = C1 + C0 r */
  const float t = pq1_fmul(C0H, rh);
  const float wh = pq1_fadd(C1H, t);
  float wl = pq1_fadd(pq1_fsub(C1H, wh), t);
  wl = pq1_fadd(wl, pq1_ffma(C0H, rh, -t));
  wl = pq1_ffma(C0H, rl, pq1_fadd(wl, C1L));
  /* q = w r^2 */
  const float qh = pq1_fmul(wh, sh);
  float ql = pq1_ffma(wh, sh, -qh);
  ql = pq1_ffma(wh, sl, ql);
  ql = pq1_ffma(wl, sh, ql);
  /* y = 1 + (p + q); |p| <= 0.011, |q| <= 6e-5 */
  const float ah = pq1_fadd(ph, qh);
  const float bb = pq1_fsub(ah, ph);
  float al = pq1_fadd(pq1_fsub(ph, pq1_fsub(ah, bb)), pq1_fsub(qh, bb));
  al = pq1_fadd(al, pq1_fadd(pl, ql));
  const float yh = pq1_fadd(1.0f, ah);
  const float yl = pq1_fadd(pq1_fsub(pq1_fsub(1.0f, yh), -ah), al);
  /* m = y T_j */
  const int k = (int)kf;
  const float th = tab[k & 31], tl = tab[32 + (k & 31)];
  const float mh = pq1_fmul(yh, th);
  float ml = pq1_ffma(yh, th, -mh);
  ml = pq1_ffma(yh, tl, ml);
  ml = pq1_ffma(yl, th, ml);
  const float f = pq1_fadd(mh, ml);
  const float err = pq1_fadd(pq1_fsub(mh, f), ml);    /* m - f */
  const float ulpd = pq1_fsub(f, __uint_as_float(__float_as_uint(f) - 1u));  /* ulp below f (<= above) */
  *ok = (fabsf(err) < pq1_ffma(0.5f, ulpd, -1.4551915e-11f /* 2^-36 */)) &&
        (fabsf(rh) < 0.49999905f /* 0.5 - 2^-20 */);
  return pq1_fmul(f, __int_as_float(((k >> 5) + 127) << 23));
}
/* fill tab (64 floats) from PQ1_EXP2F_FF; the caller syncs */
__device__ __forceinline__ void pq1_expf_tab(float* tab, int t) {
  if (t < 64) tab[t] = __uint_as_float(PQ1_EXP2F_FF[2 * (t & 31) + (t >> 5)]);
}
__device__ __forceinline__ float pq1_expf_ff(float x, const float* __restrict__ tab) {
  bool ok = false;
  float e = 0.0f;
  if (x > -86.5f && x < 88.0f) e = pq1_expf_ffc(x, &ok, tab);
  return ok ? e : pq1_expf_dev(x);
}

/* Fused T-MAC plane emit (loop-261014-2216 row 17): the quant kernels write
 * pq1_tm_prep_w's layout (planes of v = byte ^ 0x80, per-group sums) themselves,
 * so its 145 launches/token go away. */
/* Vote-free form (VOTE issues slowly here: 1024 ballots cost the rms block
 * ~5 us): plane q of a 32-byte word from its 8 little-endian words, bit 4i+k =
 * bit q of byte k of word i, of v = byte ^ 0x80 (the xor flips plane 7 only). */
__device__ __forceinline__ uint32_t pq1_plane_tr(const uint4 a, const uint4 b, int q) {
  const uint32_t u[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  uint32_t o = 0u;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    uint32_t x = (u[i] >> q) & 0x01010101u;   /* bits 0, 8, 16, 24 */
    x |= x >> 7;                              /* bits 0,1 and 16,17 */
    x |= x >> 14;                             /* bits 0..3 */
    o |= (x & 0xfu) << (4 * i);
  }
  return q == 7 ? ~o : o;
}
/* sum over 32 bytes of (byte ^ 0x80) */
__device__ __forceinline__ int pq1_bsum_x80(const uint4 a, const uint4 b) {
  const uint32_t u[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  uint32_t s2 = 0u;   /* two 16-bit lanes, each < 16 * 255 */
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const uint32_t v = u[i] ^ 0x80808080u;
    s2 += (v & 0x00ff00ffu) + ((v >> 8) & 0x00ff00ffu);
  }
  return (int)((s2 & 0xffffu) + (s2 >> 16));
}

/* SILU=1: v = silu(gu[i]) * gu[n+i] (the FFN tail); SILU=2 the same through
 * pq1_expf_ff (PQ1_EXF=0 keeps 1, the A/B arm). SILU=0 (loop iter 4): v = gu[i],
 * i.e. plain q8_K_quant of a device vector (the attention output). */
template <int SILU>
__global__ __launch_bounds__(256) void pq1_silu_q8k(const float* __restrict__ gu, int n,
                                                    uint8_t* __restrict__ xq,
                                                    float* __restrict__ vout,
                                                    uint32_t* __restrict__ pl,
                                                    int* __restrict__ au) {
  __shared__ float sv[256];
  __shared__ __align__(16) uint8_t sqb[256];
  __shared__ unsigned sm[8];
  __shared__ int si[8];
  __shared__ float stab[SILU == 2 ? 64 : 1];
  const int t = threadIdx.x, i = blockIdx.x * 256 + t;
  float v;
  if (SILU == 2) {
    pq1_expf_tab(stab, t);
    __syncthreads();
  }
  if (SILU) {
    const float g = gu[i], u = gu[n + i];
    const float mg = __uint_as_float(__float_as_uint(g) ^ 0x80000000u);
    const float e = SILU == 2 ? pq1_expf_ff(mg, stab) : pq1_expf_dev(mg);
    const float sg = pq1_fdiv(1.0f, pq1_fadd(1.0f, e));
    v = pq1_fmul(pq1_fmul(g, sg), u);
  } else {
    v = gu[i];
  }
  if (vout) vout[i] = v;
  sv[t] = v;
  const unsigned a = __float_as_uint(v) & 0x7fffffffu;   /* |v| bits: integer order == float order */
  unsigned m = a;
#pragma unroll
  for (int o = 16; o; o >>= 1) m = max(m, __shfl_xor_sync(0xffffffffu, m, o));
  if ((t & 31) == 0) sm[t >> 5] = m;
  __syncthreads();
  unsigned am = sm[0];
#pragma unroll
  for (int k = 1; k < 8; k++) am = max(am, sm[k]);
  uint8_t* yb = xq + (size_t)blockIdx.x * 292;
  unsigned qb = 0u, dbits = 0u;
  if (am == 0u) {                                   /* fork: d = 0, qs = 0 (block-uniform) */
    if (t == 0) *reinterpret_cast<uint32_t*>(yb) = 0u;
    yb[4 + t] = 0;
  } else {
    int idx = (a == am) ? t : 256;
#pragma unroll
    for (int o = 16; o; o >>= 1) idx = min(idx, __shfl_xor_sync(0xffffffffu, idx, o));
    if ((t & 31) == 0) si[t >> 5] = idx;
    __syncthreads();
    int first = si[0];
#pragma unroll
    for (int k = 1; k < 8; k++) first = min(first, si[k]);
    const float iscale = pq1_fdiv(-127.0f, sv[first]);
    qb = (uint8_t)(int8_t)pq1_q8n(v, iscale);
    yb[4 + t] = (uint8_t)qb;
    dbits = __float_as_uint(pq1_fdiv(1.0f, iscale));
    if (t == 0) *reinterpret_cast<uint32_t*>(yb) = dbits;
  }
  if (pl) {                                         /* block b = Q8_K block, warp = word */
    /* one plane word per lane of warps 0-1: t = 8 * word + plane (the 64 words
     * of this Q8_K block in order; warp 0 = group 2b, warp 1 = group 2b+1) */
    uint32_t* dst = pl + (size_t)blockIdx.x * 73;
    sqb[t] = (uint8_t)qb;
    __syncthreads();
    if (t < 64) {                                   /* warp-uniform */
      const uint4* src = reinterpret_cast<const uint4*>(sqb + (t >> 3) * 32);
      const uint4 a0 = src[0], a1 = src[1];
      dst[1 + t] = pq1_plane_tr(a0, a1, t & 7);
      if (au) {
        int sum = (t & 7) ? 0 : pq1_bsum_x80(a0, a1);
        sum += __shfl_xor_sync(0xffffffffu, sum, 8);
        sum += __shfl_xor_sync(0xffffffffu, sum, 16);
        if ((t & 31) == 0) au[blockIdx.x * 2 + (t >> 5)] = sum;
      }
    } else if (t < 72) {
      dst[65 + t - 64] = 0u;
    } else if (t == 72) {
      dst[0] = dbits;
    }
  }
}

/* Warp-per-block form of pq1_silu_q8k with planes (loop-261014-2216 row 21):
 * one warp per 256-element Q8_K block, lane l owning elements l + 32 j
 * (j = 0..7), shuffle-only reductions. The 256-thread form ran one Q8_K
 * block per CTA: 48 CTAs = 6 waves on the one SM, each a chain of loads,
 * two __syncthreads reductions, three divides and the plane build. Here
 * the 48 warps of the FFN tail are resident at once. Same values: amax over
 * |v| bits, first index by integer min, iscale/d/q as pq1_silu_q8k, planes by
 * pq1_plane_tr over the same 32-byte words (word j = elements 32j..32j+31),
 * group sums as integers. PQ1_QW=0 keeps the 256-thread kernel. */
template <int SILU>
__global__ __launch_bounds__(256) void pq1_q8k_w(const float* __restrict__ gu, int n,
                                                 uint8_t* __restrict__ xq,
                                                 uint32_t* __restrict__ pl,
                                                 int* __restrict__ au) {
  __shared__ float stab[SILU == 2 ? 64 : 1];
  __shared__ __align__(16) uint8_t sqb[8][256];
  const int t = threadIdx.x, l = t & 31, w = t >> 5;
  if (SILU == 2) {
    pq1_expf_tab(stab, t);
    __syncthreads();
  }
  const int b = blockIdx.x * 8 + w;
  if (b * 256 >= n) return;                         /* warp-uniform; no CTA barrier follows */
  const float* src = gu + (size_t)b * 256 + l;
  float v[8];
  unsigned a[8], am = 0u;
#pragma unroll
  for (int j = 0; j < 8; j++) {
    const float x = src[32 * j];
    if (SILU) {
      const float u = src[n + 32 * j];
      const float mg = __uint_as_float(__float_as_uint(x) ^ 0x80000000u);
      const float e = SILU == 2 ? pq1_expf_ff(mg, stab) : pq1_expf_dev(mg);
      const float sg = pq1_fdiv(1.0f, pq1_fadd(1.0f, e));
      v[j] = pq1_fmul(pq1_fmul(x, sg), u);
    } else {
      v[j] = x;
    }
    a[j] = __float_as_uint(v[j]) & 0x7fffffffu;
    am = max(am, a[j]);
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) am = max(am, __shfl_xor_sync(0xffffffffu, am, o));
  uint8_t* yb = xq + (size_t)b * 292;
  uint32_t q[8];
  unsigned dbits = 0u;
  if (am == 0u) {                                   /* fork: d = 0, qs = 0 (warp-uniform) */
#pragma unroll
    for (int j = 0; j < 8; j++) q[j] = 0u;
  } else {
    int idx = 256;
    float vf = 0.0f;
#pragma unroll
    for (int j = 7; j >= 0; j--)                    /* ends on the lowest j */
      if (a[j] == am) { idx = l + 32 * j; vf = v[j]; }
    int first = idx;
#pragma unroll
    for (int o = 16; o; o >>= 1) first = min(first, __shfl_xor_sync(0xffffffffu, first, o));
    const float vmax = __shfl_sync(0xffffffffu, vf, first & 31);
    const float iscale = pq1_fdiv(-127.0f, vmax);
#pragma unroll
    for (int j = 0; j < 8; j++) q[j] = (uint8_t)(int8_t)pq1_q8n(v[j], iscale);
    dbits = __float_as_uint(pq1_fdiv(1.0f, iscale));
  }
  if (l == 0) *reinterpret_cast<uint32_t*>(yb) = dbits;
#pragma unroll
  for (int j = 0; j < 8; j++) { yb[4 + l + 32 * j] = (uint8_t)q[j]; sqb[w][l + 32 * j] = (uint8_t)q[j]; }
  __syncwarp();
  uint32_t* dst = pl + (size_t)b * 73;
#pragma unroll
  for (int h = 0; h < 2; h++) {                     /* plane word wd = 8 * word + plane */
    const int wd = l + 32 * h;
    const uint4* s4 = reinterpret_cast<const uint4*>(&sqb[w][(wd >> 3) * 32]);
    dst[1 + wd] = pq1_plane_tr(s4[0], s4[1], wd & 7);
  }
  if (l < 8) dst[65 + l] = 0u;
  if (l == 0) dst[0] = dbits;
  if (au) {
    int s0 = 0, s1 = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) { s0 += (int)(q[j] ^ 0x80u); s1 += (int)(q[j + 4] ^ 0x80u); }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
      s0 += __shfl_xor_sync(0xffffffffu, s0, o);
      s1 += __shfl_xor_sync(0xffffffffu, s1, o);
    }
    if (l == 0) { au[2 * b] = s0; au[2 * b + 1] = s1; }
  }
}

/* pq1_parity expf: pq1_expf_ff and the certified core against pq1_expf_dev for
 * inputs base .. base + 2^20 - 1. c[0] final mismatches, c[1] certified but
 * wrong, c[2] in-range fallbacks, c[3] fallbacks whose fast value was right. */
__global__ void pq1_expf_chk(uint32_t base, unsigned long long* __restrict__ c) {
  __shared__ float tab[64];
  pq1_expf_tab(tab, threadIdx.x);
  __syncthreads();
  const uint32_t u = base + blockIdx.x * blockDim.x + threadIdx.x;
  const float x = __uint_as_float(u);
  bool ok = false, in = x > -86.5f && x < 88.0f;
  float e = 0.0f;
  if (in) e = pq1_expf_ffc(x, &ok, tab);
  const uint32_t r = __float_as_uint(pq1_expf_dev(x)), f = __float_as_uint(pq1_expf_ff(x, tab));
  const unsigned m0 = __ballot_sync(0xffffffffu, f != r);
  const unsigned m1 = __ballot_sync(0xffffffffu, ok && __float_as_uint(e) != r);
  const unsigned m2 = __ballot_sync(0xffffffffu, in && !ok);
  const unsigned m3 = __ballot_sync(0xffffffffu, in && !ok && __float_as_uint(e) == r);
  if ((threadIdx.x & 31) == 0) {
    if (m0) atomicAdd(&c[0], (unsigned long long)__popc(m0));
    if (m1) atomicAdd(&c[1], (unsigned long long)__popc(m1));
    if (m2) atomicAdd(&c[2], (unsigned long long)__popc(m2));
    if (m3) atomicAdd(&c[3], (unsigned long long)__popc(m3));
  }
}
extern "C" int pq1_expf_chk_launch(unsigned base, unsigned long long* c_d) {
  pq1_expf_chk<<<(1 << 20) / 256, 256>>>(base, c_d);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

static int pq1_exf(void) {
  static int v = -1;
  if (v < 0) { const char* e = getenv("PQ1_EXF"); v = e ? atoi(e) : 1; }
  return v;
}

extern "C" int pq1_silu_q8k_launch(const float* gu_d, int n, uint8_t* xq_d, float* vout_d,
                                   void* stream) {
  if (n <= 0 || (n & 255) != 0) return -2;
  if (pq1_exf())
    pq1_silu_q8k<2><<<n / 256, 256, 0, (cudaStream_t)stream>>>(gu_d, n, xq_d, vout_d, NULL, NULL);
  else
    pq1_silu_q8k<1><<<n / 256, 256, 0, (cudaStream_t)stream>>>(gu_d, n, xq_d, vout_d, NULL, NULL);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

extern "C" int pq1_q8k_launch(const float* x_d, int n, uint8_t* xq_d, void* stream) {
  if (n <= 0 || (n & 255) != 0) return -2;
  pq1_silu_q8k<0><<<n / 256, 256, 0, (cudaStream_t)stream>>>(x_d, n, xq_d, NULL, NULL, NULL);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

/* the two launches above plus pq1_tm_prep_launch's planes/sums in one kernel */
extern "C" int pq1_silu_q8k_pl_launch(int silu, const float* x_d, int n, uint8_t* xq_d,
                                      uint32_t* pl_d, int* au_d, void* stream) {
  if (n <= 0 || (n & 255) != 0 || n > PQ1_MAXK) return -2;
  static int qw = -1;
  if (qw < 0) { const char* e = getenv("PQ1_QW"); qw = e ? atoi(e) : 1; }
  if (qw) {
    const int nb = (n / 256 + 7) / 8;
    if (silu && pq1_exf())
      pq1_q8k_w<2><<<nb, 256, 0, (cudaStream_t)stream>>>(x_d, n, xq_d, pl_d, au_d);
    else if (silu)
      pq1_q8k_w<1><<<nb, 256, 0, (cudaStream_t)stream>>>(x_d, n, xq_d, pl_d, au_d);
    else
      pq1_q8k_w<0><<<nb, 256, 0, (cudaStream_t)stream>>>(x_d, n, xq_d, pl_d, au_d);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
  }
  if (silu && pq1_exf())
    pq1_silu_q8k<2><<<n / 256, 256, 0, (cudaStream_t)stream>>>(x_d, n, xq_d, NULL, pl_d, au_d);
  else if (silu)
    pq1_silu_q8k<1><<<n / 256, 256, 0, (cudaStream_t)stream>>>(x_d, n, xq_d, NULL, pl_d, au_d);
  else
    pq1_silu_q8k<0><<<n / 256, 256, 0, (cudaStream_t)stream>>>(x_d, n, xq_d, NULL, pl_d, au_d);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

/* ---- loop iter 4: the attention block on the device ------------------------
 * Head rms + rope as fwd8's host path, bit for bit (its object code was read to
 * pin every contraction: gcc 7.5 -O2 -mcpu=cortex-a57, default -ffp-contract=fast):
 *  head rms   s = fma(double(x), double(x), s) serially over the 128 dims (the
 *             products are exact), sc = 1/sqrtf(float(s/128) + 1e-6f),
 *             y = (sc*x)*w                         [rms(): fmadd d; fdiv d; fcvt;
 *                                                   fadd; fsqrt; fdiv; fmul; fmul]
 *  rope       x[j] = fma(a, c, -(b*s)), x[j+64] = fma(a, s, b*c)
 *                                                  [rope_yarn_apply: fmul; fnmsub;
 *                                                   fmul; fmadd]
 * 4k-context rework: K and V rows are stored as f16 (RNE) and the attention is
 * the fork's flash-attention decode (pq1_fa16_head in pq1_host.c is the bit-exact
 * host reference; the arithmetic is documented there), split in three kernels
 * whose grids depend on ctx only, so one captured graph serves every position:
 *  pq1_fa_qk  s[h][t] = vec_dot_f16(K_t, f16(q_h)) * scale (16 chains, fork reduce)
 *  pq1_fa_sm  per (h, t): the running max M before t is an exclusive prefix max;
 *             s > M ? (ms = expf(M - s), vs = 1) : (ms = 1, vs = expf(s - M))
 *  pq1_fa_pv  per (h, d), t in order: VKQ = r16(VKQ * ms) when ms != 1, then
 *             VKQ = r16(fma(V, vs, VKQ)); S = fma(S, ms, vs); out = VKQ * (1/S)
 * As in pq1_silu_q8k every f32 op and compare is inline PTX without .ftz. The
 * cos/sin table, npos and the Q8_K activation arrive in one H2D (layout
 * PQ1_APAR_*, pq1_wrap.h); K/V caches are [kv head][ctx][128] halves per layer. */
#ifndef PQ1_APAR_ROPE   /* per-token parameter block layout: the same values as pq1_wrap.h */
#define PQ1_APAR_ROPE 0
#define PQ1_APAR_NPOS 512
#define PQ1_APAR_XQ 576
#endif
__device__ __forceinline__ float pq1_fsqrt(float a) { float r; asm("sqrt.rn.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float pq1_fneg(float a) { return __uint_as_float(__float_as_uint(a) ^ 0x80000000u); }
/* (s > m) ? s : m with an IEEE compare (no ftz): NaN s never taken */
__device__ __forceinline__ float pq1_fgtsel(float m, float s) {
  float r;
  asm("{ .reg .pred p; setp.gt.f32 p, %2, %1; selp.f32 %0, %2, %1, p; }" : "=f"(r) : "f"(m), "f"(s));
  return r;
}
/* a > b, IEEE (no ftz): false on NaN */
__device__ __forceinline__ int pq1_fgt(float a, float b) {
  int r;
  asm("{ .reg .pred p; setp.gt.f32 p, %1, %2; selp.s32 %0, 1, 0, p; }" : "=r"(r) : "f"(a), "f"(b));
  return r;
}
/* f32 -> f16 bits, round to nearest even (overflow -> inf, f16 subnormals kept) */
__device__ __forceinline__ unsigned short pq1_f2h(float a) {
  unsigned short r;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ float pq1_h2f(unsigned short h) {
  float r;
  asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(h));
  return r;
}
/* pq1_h2f(pq1_f2h(x)) for every non-NaN x, without the two conversions (which
 * issue at 1/4 the FP32 rate on sm_53): x + C - C with C = 1.5 * 2^(E+13), E =
 * max(exponent(x), -14), rounds x to a multiple of the f16 ulp 2^(E-10), ties
 * to even (C / ulp is even); the sign is OR-ed back so that values rounding to
 * zero keep it (-0, as the conversion gives), and |x| >= 65520 becomes inf.
 * Checked against the conversion pair for all 2^32 inputs (pq1_r16_test). */
__device__ __forceinline__ float pq1_r16(float x) {
  const unsigned ux = __float_as_uint(x);
  const unsigned e = max(ux & 0x7f800000u, 0x38800000u);
  const float C = __uint_as_float(e + 0x06C00000u);
  int big;   /* |x| >= 65520 (false for NaN): IEEE compare on |x|, no ftz */
  asm("{ .reg .pred p; setp.ge.f32 p, %1, 0f477FF000; selp.s32 %0, 1, 0, p; }" : "=r"(big) : "f"(fabsf(x)));
  const unsigned r = big ? 0x7f800000u : __float_as_uint(pq1_fsub(pq1_fadd(x, C), C));
  return __uint_as_float(r | (ux & 0x80000000u));
}

__device__ __forceinline__ float pq1_rcv(float x) { return pq1_h2f(pq1_f2h(x)); }

/* one warp per head: warps [0, nh+nkv) are q then k heads (rms + rope; k goes to
 * the cache row npos), warps [nh+nkv, nh+2nkv) copy the v heads to their row */
__global__ __launch_bounds__(128) void pq1_attn_pre(const float* __restrict__ qkv,
                                                    const float* __restrict__ nw,
                                                    const uint8_t* __restrict__ par,
                                                    float* __restrict__ qo,
                                                    unsigned short* __restrict__ kc,
                                                    unsigned short* __restrict__ vc, int nh,
                                                    int nkv, int ctx) {
  __shared__ float sx[4][128];
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  const int hh = blockIdx.x * 4 + w;
  if (hh >= nh + 2 * nkv) return;   /* warp-uniform */
  const int npos = *reinterpret_cast<const int*>(par + PQ1_APAR_NPOS);
  const float* src = qkv + (size_t)hh * 128;
  if (hh >= nh + nkv) {
    unsigned short* dst = vc + ((size_t)(hh - nh - nkv) * ctx + npos) * 128;
#pragma unroll
    for (int k = 0; k < 4; k++) dst[l + 32 * k] = pq1_f2h(src[l + 32 * k]);
    return;
  }
  float x[4];
#pragma unroll
  for (int k = 0; k < 4; k++) { x[k] = src[l + 32 * k]; sx[w][l + 32 * k] = x[k]; }
  __syncwarp();
  float sc = 0.0f;
  if (l == 0) {
    double s = 0.0;
    for (int j = 0; j < 128; j++) { const double d = pq1_f2d(sx[w][j]); s = __fma_rn(d, d, s); }
    sc = pq1_fdiv(1.0f, pq1_fsqrt(pq1_fadd(pq1_d2f(__ddiv_rn(s, 128.0)), 1e-6f)));
  }
  sc = __shfl_sync(0xffffffffu, sc, 0);
  const float* wn = nw + (hh < nh ? 0 : 128);
  float y[4];
#pragma unroll
  for (int k = 0; k < 4; k++) y[k] = pq1_fmul(pq1_fmul(sc, x[k]), wn[l + 32 * k]);
  const float* rc = reinterpret_cast<const float*>(par + PQ1_APAR_ROPE); /* cos[64] | sin[64] */
  float o[4];
#pragma unroll
  for (int k = 0; k < 2; k++) {   /* pairs (j, j+64), j = l + 32k */
    const float c = rc[l + 32 * k], sn = rc[64 + l + 32 * k], a = y[k], b = y[k + 2];
    o[k] = pq1_ffma(a, c, pq1_fneg(pq1_fmul(b, sn)));
    o[k + 2] = pq1_ffma(a, sn, pq1_fmul(b, c));
  }
  if (hh < nh) {
#pragma unroll
    for (int k = 0; k < 4; k++) qo[(size_t)hh * 128 + l + 32 * k] = o[k];
  } else {
    unsigned short* dst = kc + ((size_t)(hh - nh) * ctx + npos) * 128;
#pragma unroll
    for (int k = 0; k < 4; k++) dst[l + 32 * k] = pq1_f2h(o[k]);
  }
}

/* Loop iter 7 (PQ1_APRE=1, default; 0 is the A/B arm): the same per-head math
 * in one 256-thread block. The q/k head RMS chain (128 serial double fma) used
 * to run on lane 0 of one warp per head: on sm_53 a one-lane DP instruction
 * still costs a whole warp's DP issue (8 clk), so 40 heads x 128 x 2 DP ops
 * serialized (~100 us). Here thread t runs head t's chain, so one DP warp
 * instruction serves 32 heads; each chain is unchanged (same j order, same
 * fma, same scale formula), and the rope/scale stage is the same per element. */
#define PQ1_APRE_MAXH 64
__global__ __launch_bounds__(256) void pq1_attn_pre1(const float* __restrict__ qkv,
                                                    const float* __restrict__ nw,
                                                    const uint8_t* __restrict__ par,
                                                    float* __restrict__ qo,
                                                    unsigned short* __restrict__ kc,
                                                    unsigned short* __restrict__ vc, int nh,
                                                    int nkv, int ctx) {
  extern __shared__ float sx[]; /* [nqk][129] */
  __shared__ float ssc[PQ1_APRE_MAXH];
  const int nqk = nh + nkv, t = threadIdx.x;
  const int npos = *reinterpret_cast<const int*>(par + PQ1_APAR_NPOS);
  for (int i = t; i < nqk * 128; i += 256) sx[(i >> 7) * 129 + (i & 127)] = qkv[i];
  for (int i = t; i < nkv * 128; i += 256) {
    const int h = i >> 7, d = i & 127;
    vc[((size_t)h * ctx + npos) * 128 + d] = pq1_f2h(qkv[(size_t)(nqk + h) * 128 + d]);
  }
  __syncthreads();
  if (t < nqk) {
    const float* r = sx + t * 129;
    double sd = 0.0;
    for (int j = 0; j < 128; j++) { const double d = pq1_f2d(r[j]); sd = __fma_rn(d, d, sd); }
    ssc[t] = pq1_fdiv(1.0f, pq1_fsqrt(pq1_fadd(pq1_d2f(__ddiv_rn(sd, 128.0)), 1e-6f)));
  }
  __syncthreads();
  const float* rc = reinterpret_cast<const float*>(par + PQ1_APAR_ROPE); /* cos[64] | sin[64] */
  for (int i = t; i < nqk * 64; i += 256) {   /* pair (j, j+64) of head hh */
    const int hh = i >> 6, j = i & 63;
    const float sc = ssc[hh];
    const float* wn = nw + (hh < nh ? 0 : 128);
    const float a = pq1_fmul(pq1_fmul(sc, sx[hh * 129 + j]), wn[j]);
    const float b = pq1_fmul(pq1_fmul(sc, sx[hh * 129 + j + 64]), wn[j + 64]);
    const float c = rc[j], sn = rc[64 + j];
    const float o0 = pq1_ffma(a, c, pq1_fneg(pq1_fmul(b, sn)));
    const float o1 = pq1_ffma(a, sn, pq1_fmul(b, c));
    if (hh < nh) {
      qo[(size_t)hh * 128 + j] = o0;
      qo[(size_t)hh * 128 + j + 64] = o1;
    } else {
      unsigned short* dst = kc + ((size_t)(hh - nh) * ctx + npos) * 128;
      dst[j] = pq1_f2h(o0);
      dst[j + 64] = pq1_f2h(o1);
    }
  }
}

/* Loop iter 27 (PQ1_APRE=2, default; 1 is the A/B arm): pq1_attn_pre1's math with
 * its latency taken out. pre1 (33 us/layer in the engine) is dominated by
 * dependent waits, not arithmetic: its qkv copy loop issues one global load at
 * a time (24 per thread), the rope stage re-reads wn/cos/sin from global per
 * pair, and each head's 128 serial double fma run on one thread. Here every
 * thread issues its 6 float4 qkv loads at once (q/k rows to shared, v rows to
 * the cache as 8-byte f16 stores), wn and cos/sin are staged in shared, and
 * the head RMS is certified instead of chained:
 *   warp w takes heads w, w+8, ...; lane l forms the exact squares of dims l,
 *   l+32, l+64, l+96 as fp32 pairs (TwoProd), adds them and then the 5 shuffle
 *   levels in pair arithmetic (TwoSum + one rounded tail add, Fast2Sum
 *   renormalization): (H, L) is the sum within ~2^-44 relative (all terms
 *   >= 0, 8 levels of <= 2^-47), and pre1's serial s within 127u (u = 2^-53).
 *   So s lies in [(H+L)(1-e), (H+L)(1+e)] with e = 2^-40, and when that whole
 *   interval is inside H's float rounding interval (half an ulp up, half the
 *   ulp below H down: a quarter ulp of H at a power of two), rn_f32(s/128) =
 *   H/128 exactly (H >= 2^-100: the /128 and the ulp are exact). Otherwise
 *   (non-finite or tiny H, near a boundary, or serial != 0: the parity gate)
 *   lane 0 runs pre1's double chain over the shared row. nfb counts those.
 * The rope/scale stage and the K/V values are pre1's, element for element. */
#define PQ1_AP2_S 132   /* q/k row stride in shared: 16-byte rows */
__device__ __forceinline__ void pq1_two_sum(float a, float b, float& s, float& e) {
  s = pq1_fadd(a, b);
  const float bb = pq1_fsub(s, a);
  e = pq1_fadd(pq1_fsub(a, pq1_fsub(s, bb)), pq1_fsub(b, bb));
}
/* (ah, al) += (bh, bl), both normalized and >= 0 */
__device__ __forceinline__ void pq1_ff_add(float& ah, float& al, float bh, float bl) {
  float s, e;
  pq1_two_sum(ah, bh, s, e);
  e = pq1_fadd(e, pq1_fadd(al, bl));
  ah = pq1_fadd(s, e);
  al = pq1_fsub(e, pq1_fsub(ah, s));
}
__global__ __launch_bounds__(256) void pq1_attn_pre2(const float* __restrict__ qkv,
                                                    const float* __restrict__ nw,
                                                    const uint8_t* __restrict__ par,
                                                    float* __restrict__ qo,
                                                    unsigned short* __restrict__ kc,
                                                    unsigned short* __restrict__ vc, int nh,
                                                    int nkv, int ctx, int serial,
                                                    int* __restrict__ nfb) {
  extern __shared__ __align__(16) float sx[]; /* [nqk][PQ1_AP2_S] */
  __shared__ float ssc[PQ1_APRE_MAXH];
  __shared__ __align__(16) float swn[256], src_[128];
  const int nqk = nh + nkv, t = threadIdx.x, wp = t >> 5, l = t & 31;
  const int nq4 = (nqk + nkv) * 32;
  const int npos = *reinterpret_cast<const int*>(par + PQ1_APAR_NPOS);
  {
    float4 v[8];
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const int i = t + 256 * k;
      if (i < nq4) v[k] = reinterpret_cast<const float4*>(qkv)[i];
    }
    float4 aux;
    if (t < 64) aux = reinterpret_cast<const float4*>(nw)[t];
    else if (t < 96) aux = reinterpret_cast<const float4*>(par + PQ1_APAR_ROPE)[t - 64];
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const int i = t + 256 * k;
      if (i >= nq4) break;
      const int h = i >> 5, d = (i & 31) * 4;
      if (h < nqk) {
        *reinterpret_cast<float4*>(sx + h * PQ1_AP2_S + d) = v[k];
      } else {
        const uint32_t lo = pq1_f2h(v[k].x) | ((uint32_t)pq1_f2h(v[k].y) << 16);
        const uint32_t hi = pq1_f2h(v[k].z) | ((uint32_t)pq1_f2h(v[k].w) << 16);
        *reinterpret_cast<uint2*>(vc + ((size_t)(h - nqk) * ctx + npos) * 128 + d) = make_uint2(lo, hi);
      }
    }
    if (t < 64) reinterpret_cast<float4*>(swn)[t] = aux;
    else if (t < 96) reinterpret_cast<float4*>(src_)[t - 64] = aux;
  }
  __syncthreads();
  for (int h = wp; h < nqk; h += 8) {   /* warp-uniform */
    const float* r = sx + h * PQ1_AP2_S;
    float H, L;
    {
      const float x0 = r[l], x1 = r[l + 32], x2 = r[l + 64], x3 = r[l + 96];
      H = pq1_fmul(x0, x0); L = pq1_ffma(x0, x0, pq1_fneg(H));
      float bh = pq1_fmul(x1, x1), bl = pq1_ffma(x1, x1, pq1_fneg(bh));
      pq1_ff_add(H, L, bh, bl);
      bh = pq1_fmul(x2, x2); bl = pq1_ffma(x2, x2, pq1_fneg(bh));
      pq1_ff_add(H, L, bh, bl);
      bh = pq1_fmul(x3, x3); bl = pq1_ffma(x3, x3, pq1_fneg(bh));
      pq1_ff_add(H, L, bh, bl);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
      const float oh = __shfl_xor_sync(0xffffffffu, H, o), ol = __shfl_xor_sync(0xffffffffu, L, o);
      pq1_ff_add(H, L, oh, ol);
    }
    if (l == 0) {
      const uint32_t hb = __float_as_uint(H);
      /* finite and >= 2^-100 (sign clear: all terms >= 0, NaN fails too) */
      int ok = !serial && hb >= 0x0d800000u && hb < 0x7f800000u;
      if (ok) {
        const float ulp = __uint_as_float((hb & 0x7f800000u) - (23u << 23));
        const float hup = pq1_fmul(ulp, 0.5f);
        const float hdn = (hb & 0x007fffffu) ? hup : pq1_fmul(ulp, 0.25f);
        const float eh = pq1_fmul(H, 1.0f / 1099511627776.0f);   /* e H, e = 2^-40 */
        /* margin 1 - 2^-20: the two sums below are within 2^-24 relative of exact */
        ok = pq1_fgt(pq1_fmul(hup, 0.99999904632568359375f), pq1_fadd(L, eh)) &&
             pq1_fgt(pq1_fmul(hdn, 0.99999904632568359375f), pq1_fsub(eh, L));
      }
      float f = pq1_fmul(H, 0.0078125f);
      if (!ok) {
        double sd = 0.0;
        for (int j = 0; j < 128; j++) { const double e = pq1_f2d(r[j]); sd = __fma_rn(e, e, sd); }
        f = pq1_d2f(__ddiv_rn(sd, 128.0));
        if (nfb) atomicAdd(nfb, 1);
      }
      ssc[h] = pq1_fdiv(1.0f, pq1_fsqrt(pq1_fadd(f, 1e-6f)));
    }
  }
  __syncthreads();
  for (int i = t; i < nqk * 64; i += 256) {   /* pair (j, j+64) of head hh */
    const int hh = i >> 6, j = i & 63;
    const float sc = ssc[hh];
    const float* wn = swn + (hh < nh ? 0 : 128);
    const float a = pq1_fmul(pq1_fmul(sc, sx[hh * PQ1_AP2_S + j]), wn[j]);
    const float b = pq1_fmul(pq1_fmul(sc, sx[hh * PQ1_AP2_S + j + 64]), wn[j + 64]);
    const float c = src_[j], sn = src_[64 + j];
    const float o0 = pq1_ffma(a, c, pq1_fneg(pq1_fmul(b, sn)));
    const float o1 = pq1_ffma(a, sn, pq1_fmul(b, c));
    if (hh < nh) {
      qo[(size_t)hh * 128 + j] = o0;
      qo[(size_t)hh * 128 + j + 64] = o1;
    } else {
      unsigned short* dst = kc + ((size_t)(hh - nh) * ctx + npos) * 128;
      dst[j] = pq1_f2h(o0);
      dst[j + 64] = pq1_f2h(o1);
    }
  }
}
/* pq1_parity apre: ver 1 = pq1_attn_pre1, 2 = pq1_attn_pre2 (serial forces its chain) */
extern "C" int pq1_attn_pre_launch(int ver, int serial, int* nfb_d, const float* qkv_d,
                                   const float* nw_d, const uint8_t* par_d, float* qo_d,
                                   unsigned short* kc_d, unsigned short* vc_d, int nh, int nkv,
                                   int ctx, void* stream) {
  if (nh + nkv > PQ1_APRE_MAXH) return -2;
  cudaStream_t s = (cudaStream_t)stream;
  const size_t sh = (nh + nkv) * (ver == 2 ? PQ1_AP2_S : 129) * sizeof(float);
  if (ver == 2)
    pq1_attn_pre2<<<1, 256, sh, s>>>(qkv_d, nw_d, par_d, qo_d, kc_d, vc_d, nh, nkv, ctx, serial, nfb_d);
  else
    pq1_attn_pre1<<<1, 256, sh, s>>>(qkv_d, nw_d, par_d, qo_d, kc_d, vc_d, nh, nkv, ctx);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

/* q.k scores. Block = (kv head, tile stride), 32*G threads (G = nh/nkv query
 * heads, one warp each); a tile is 32 positions, lane = position. The K tile is
 * widened to f32 in shared memory, dim-major ([dim][pos], padded), so lane reads
 * are conflict-free and q (f16-rounded, f32 in smem) is a broadcast. */
#define PQ1_FA_QB 16
__global__ __launch_bounds__(256) void pq1_fa_qk(const float* __restrict__ qo,
                                                 const unsigned short* __restrict__ kc,
                                                 const uint8_t* __restrict__ par,
                                                 float* __restrict__ sc, int nh, int nkv,
                                                 int ctx) {
  __shared__ float sk[128][33];
  __shared__ __align__(16) float sq[8][128];
  const int G = nh / nkv, kh = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
  const int w = tid >> 5, l = tid & 31;
  const int T = *reinterpret_cast<const int*>(par + PQ1_APAR_NPOS) + 1;
  if ((int)blockIdx.y * 32 >= T) return;   /* block-uniform, before any barrier */
  for (int i = tid; i < G * 128; i += nt)
    sq[i >> 7][i & 127] = pq1_h2f(pq1_f2h(qo[(size_t)kh * G * 128 + i]));
  const unsigned short* kb = kc + (size_t)kh * ctx * 128;
  for (int t0 = blockIdx.y * 32; t0 < T; t0 += gridDim.y * 32) {
    __syncthreads();
    for (int idx = tid; idx < 512; idx += nt) {
      const int p = idx >> 4, c = idx & 15;
      uint4 u = make_uint4(0u, 0u, 0u, 0u);
      if (t0 + p < T) u = __ldg(reinterpret_cast<const uint4*>(kb + (size_t)(t0 + p) * 128) + c);
      const unsigned wv[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
      for (int e = 0; e < 4; e++) {
        sk[c * 8 + 2 * e][p] = pq1_h2f((unsigned short)(wv[e] & 0xffffu));
        sk[c * 8 + 2 * e + 1][p] = pq1_h2f((unsigned short)(wv[e] >> 16));
      }
    }
    __syncthreads();
    float a[16];
#pragma unroll
    for (int c = 0; c < 16; c++) a[c] = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; j++)
#pragma unroll
      for (int c = 0; c < 16; c++) a[c] = pq1_ffma(sk[16 * j + c][l], sq[w][16 * j + c], a[c]);
    float r[4];
#pragma unroll
    for (int q = 0; q < 4; q++) r[q] = pq1_fadd(pq1_fadd(a[q], a[8 + q]), pq1_fadd(a[4 + q], a[12 + q]));
    const float s = pq1_fadd(pq1_fadd(r[0], r[1]), pq1_fadd(r[2], r[3]));
    if (t0 + l < T) sc[(size_t)(kh * G + w) * ctx + t0 + l] = pq1_fmul(s, 0.0883883476f);
  }
}

/* q.k scores for G = 4 (Bonsai-8B: 32 q / 8 kv heads), q held in registers.
 * Lane = (position half hf, chain c): the fork's chain c sums dims 16j + c,
 * j = 0..7, so a thread keeps q[h][16j + c] of the 4 heads (32 registers,
 * f16-rounded) and every K element it reads from shared memory feeds 4 fmas.
 * The 16 chains of a position live in 16 lanes and the fork's reduction tree
 *   r_l = (a_l + a_{8+l}) + (a_{4+l} + a_{12+l}),  s = (r0 + r1) + (r2 + r3)
 * is a butterfly xor 8, xor 4, xor 1, xor 2; the first two stages also split
 * the heads between the partners (each lane forwards the half it drops), so
 * the 4 heads cost 5 shuffles instead of 16 (fadd is commutative bitwise).
 * Block = (kv head, tile stride), 128 threads, tiles of 16 K rows widened to
 * f32 in shared memory with a 144-float row pitch (the two half-warps' rows
 * land on disjoint banks). */
#define PQ1_QK4_TR 16
#define PQ1_QK4_QB 12
__global__ __launch_bounds__(128) void pq1_fa_qk4(const float* __restrict__ qo,
                                                  const unsigned short* __restrict__ kc,
                                                  const uint8_t* __restrict__ par,
                                                  float* __restrict__ sc, int ctx) {
  __shared__ __align__(16) float sk[PQ1_QK4_TR][144];
  const int kh = blockIdx.x, tid = threadIdx.x, w = tid >> 5, l = tid & 31;
  const int hf = l >> 4, c = l & 15;
  const int T = *reinterpret_cast<const int*>(par + PQ1_APAR_NPOS) + 1;
  if ((int)blockIdx.y * PQ1_QK4_TR >= T) return;   /* block-uniform, before any barrier */
  float q[4][8];
#pragma unroll
  for (int h = 0; h < 4; h++)
#pragma unroll
    for (int j = 0; j < 8; j++) q[h][j] = pq1_rcv(qo[(size_t)(kh * 4 + h) * 128 + 16 * j + c]);
  const unsigned short* kb = kc + (size_t)kh * ctx * 128;
  float* so = sc + (size_t)kh * 4 * ctx;
  const bool lo8 = !(c & 8), lo4 = !(c & 4);
  const int hw = c >> 2;   /* the head a lane holds after the two splitting stages */
  uint4 pu[2];
  auto fetch = [&](int t0) {
    const int n = min(PQ1_QK4_TR, T - t0);
#pragma unroll
    for (int i = 0; i < 2; i++) {   /* 16 rows x 16 uint4 = 256 loads, 2 per thread */
      const int idx = tid + i * 128, p = idx >> 4, cc = idx & 15;
      pu[i] = make_uint4(0u, 0u, 0u, 0u);
      if (p < n) pu[i] = __ldg(reinterpret_cast<const uint4*>(kb + (size_t)(t0 + p) * 128) + cc);
    }
  };
  fetch(blockIdx.y * PQ1_QK4_TR);
  for (int t0 = blockIdx.y * PQ1_QK4_TR; t0 < T; t0 += gridDim.y * PQ1_QK4_TR) {
    const int n = min(PQ1_QK4_TR, T - t0);
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 2; i++) {
      const int idx = tid + i * 128, p = idx >> 4, cc = idx & 15;
      const unsigned uu[4] = {pu[i].x, pu[i].y, pu[i].z, pu[i].w};
      float f[8];
#pragma unroll
      for (int e = 0; e < 4; e++) {
        f[2 * e] = pq1_h2f((unsigned short)(uu[e] & 0xffffu));
        f[2 * e + 1] = pq1_h2f((unsigned short)(uu[e] >> 16));
      }
      reinterpret_cast<float4*>(sk[p])[2 * cc] = make_float4(f[0], f[1], f[2], f[3]);
      reinterpret_cast<float4*>(sk[p])[2 * cc + 1] = make_float4(f[4], f[5], f[6], f[7]);
    }
    __syncthreads();
    if (t0 + gridDim.y * PQ1_QK4_TR < T) fetch(t0 + gridDim.y * PQ1_QK4_TR);
#pragma unroll
    for (int st = 0; st < PQ1_QK4_TR / 8; st++) {   /* 4 warps x 2 positions per step */
      const int p = st * 8 + w * 2 + hf;
      float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const float k = sk[p][16 * j + c];
#pragma unroll
        for (int h = 0; h < 4; h++) a[h] = pq1_ffma(k, q[h][j], a[h]);
      }
      /* xor 8: lanes c < 8 keep heads 0,1, lanes c >= 8 keep heads 2,3 */
      const float s0 = __shfl_xor_sync(0xffffffffu, lo8 ? a[2] : a[0], 8);
      const float s1 = __shfl_xor_sync(0xffffffffu, lo8 ? a[3] : a[1], 8);
      const float b0 = pq1_fadd(lo8 ? a[0] : a[2], s0), b1 = pq1_fadd(lo8 ? a[1] : a[3], s1);
      /* xor 4: lanes with c & 4 clear keep the first of their two heads */
      const float s2 = __shfl_xor_sync(0xffffffffu, lo4 ? b1 : b0, 4);
      const float d = pq1_fadd(lo4 ? b0 : b1, s2);   /* r_(c & 3) of head hw */
      const float e = pq1_fadd(d, __shfl_xor_sync(0xffffffffu, d, 1));
      const float f = pq1_fadd(e, __shfl_xor_sync(0xffffffffu, e, 2));
      if (!(c & 3) && p < n) so[(size_t)hw * ctx + t0 + p] = pq1_fmul(f, 0.0883883476f);
    }
  }
}

/* online-softmax parameters, one block per query head: each thread owns a
 * contiguous run of positions; the running max before the run is an exclusive
 * prefix of the per-run maxima (the same strict-> fold, so the same value; a
 * +0/-0 tie can flip the sign of a zero M, which changes no ms or vs below). */
template <bool SCAN>
__global__ __launch_bounds__(256) void pq1_fa_sm(const float* __restrict__ sc,
                                                 const uint8_t* __restrict__ par,
                                                 float2* __restrict__ mv, int ctx) {
  __shared__ float sm[256];
  const int h = blockIdx.x, tid = threadIdx.x;
  const int T = *reinterpret_cast<const int*>(par + PQ1_APAR_NPOS) + 1;
  const int per = (T + 255) >> 8, b = tid * per, e = min(T, b + per);
  const float* s = sc + (size_t)h * ctx;
  const float NINF = __uint_as_float(0xff800000u);
  float m = NINF;
  for (int t = b; t < e; t++) m = pq1_fgtsel(m, s[t]);
  /* Exclusive prefix of the run maxima as a shuffle scan (was a 256-step serial
   * loop on thread 0: ~11 us per wave of 8 blocks). Combining (a, b) ->
   * fgtsel(a, b) keeps the left value unless the right is strictly greater;
   * no summary is ever NaN (the fold starts at -inf and never takes a NaN),
   * so the combine is associative and the scan equals the serial fold. */
  float M = NINF;
  if (SCAN) {
    const int lane = tid & 31, w = tid >> 5;
    float v = m;
  #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const float u = __shfl_up_sync(0xffffffffu, v, d);
      if (lane >= d) v = pq1_fgtsel(u, v);
    }
    if (lane == 31) sm[w] = v;
    __syncthreads();
    for (int k = 0; k < w; k++) M = pq1_fgtsel(M, sm[k]);
    const float vl = __shfl_up_sync(0xffffffffu, v, 1);
    if (lane) M = pq1_fgtsel(M, vl);
  } else {   /* PQ1_FAS=0: the serial fold on thread 0 (A/B arm) */
    sm[tid] = m;
    __syncthreads();
    if (tid == 0) {
      float run = NINF;
      for (int k = 0; k < 256; k++) { const float x = sm[k]; sm[k] = run; run = pq1_fgtsel(run, x); }
    }
    __syncthreads();
    M = sm[tid];
  }
  for (int t = b; t < e; t++) {
    const float x = s[t];
    float ms = 1.0f, vs = 1.0f;
    if (pq1_fgt(x, M)) { ms = pq1_expf_dev(pq1_fsub(M, x)); M = x; }
    else vs = pq1_expf_dev(pq1_fsub(x, M));
    mv[(size_t)h * ctx + t] = make_float2(ms, vs);
  }
}

/* V accumulation: block = kv head, 32*G threads, thread = (query head w, dims
 * 4l..4l+3): four independent chains share each (ms, vs) and V load. Positions
 * strictly in order (each step rounds VKQ to f16). V tiles of PQ1_PV_TR rows are
 * widened to f32 once in shared memory, the next tile is prefetched into
 * registers while the current one runs, and a tile in which no head rescales
 * (every ms == 1, the common case after the first positions) takes a branch-free
 * path. The f16 rounding of chain i < 4 - NALU uses the cvt pair (the conversion
 * pipe), the rest pq1_r16 (the ALU pipe); the two are bitwise equal. */
#define PQ1_PV_TR 8   /* 8 blocks x (4 KB + G*64 B) fit the one SM (64 KB) */
template <int G, int NALU>
__global__ __launch_bounds__(32 * G) void pq1_fa_pv(const unsigned short* __restrict__ vc,
                                                     const float2* __restrict__ mv,
                                                     const uint8_t* __restrict__ par,
                                                     float* __restrict__ atto, int ctx) {
  constexpr int NT = 32 * G, NV = (PQ1_PV_TR * 16 + NT - 1) / NT, NM = (G * PQ1_PV_TR + NT - 1) / NT;
  __shared__ __align__(16) float sv[PQ1_PV_TR][128];
  __shared__ float2 smv[G][PQ1_PV_TR];
  const int kh = blockIdx.x, tid = threadIdx.x, w = tid >> 5, l = tid & 31, h = kh * G + w;
  const int T = *reinterpret_cast<const int*>(par + PQ1_APAR_NPOS) + 1;
  const unsigned short* vb = vc + (size_t)kh * ctx * 128;
  const float2* mb = mv + (size_t)kh * G * ctx;
  uint4 pu[NV];
  float2 pm[NM];
  auto fetch = [&](int t0) {
    const int n = min(PQ1_PV_TR, T - t0);
#pragma unroll
    for (int i = 0; i < NV; i++) {
      const int idx = tid + i * NT, p = idx >> 4, c = idx & 15;
      pu[i] = make_uint4(0u, 0u, 0u, 0u);
      if (idx < PQ1_PV_TR * 16 && p < n)
        pu[i] = __ldg(reinterpret_cast<const uint4*>(vb + (size_t)(t0 + p) * 128) + c);
    }
#pragma unroll
    for (int i = 0; i < NM; i++) {
      const int idx = tid + i * NT, hw = idx / PQ1_PV_TR, p = idx % PQ1_PV_TR;
      pm[i] = make_float2(1.0f, 0.0f);
      if (idx < G * PQ1_PV_TR && p < n) pm[i] = __ldg(mb + (size_t)hw * ctx + t0 + p);
    }
  };
  float a[4] = {0.0f, 0.0f, 0.0f, 0.0f}, S = 0.0f;
  if (T > 0) fetch(0);
  for (int t0 = 0; t0 < T; t0 += PQ1_PV_TR) {
    const int n = min(PQ1_PV_TR, T - t0);
    __syncthreads();
    int resc = 0;
#pragma unroll
    for (int i = 0; i < NV; i++) {
      const int idx = tid + i * NT, p = idx >> 4, c = idx & 15;
      if (idx < PQ1_PV_TR * 16) {
        const unsigned u[4] = {pu[i].x, pu[i].y, pu[i].z, pu[i].w};
        float f[8];
#pragma unroll
        for (int e = 0; e < 4; e++) {
          f[2 * e] = pq1_h2f((unsigned short)(u[e] & 0xffffu));
          f[2 * e + 1] = pq1_h2f((unsigned short)(u[e] >> 16));
        }
        reinterpret_cast<float4*>(sv[p])[2 * c] = make_float4(f[0], f[1], f[2], f[3]);
        reinterpret_cast<float4*>(sv[p])[2 * c + 1] = make_float4(f[4], f[5], f[6], f[7]);
      }
    }
#pragma unroll
    for (int i = 0; i < NM; i++) {
      const int idx = tid + i * NT;
      if (idx < G * PQ1_PV_TR) {
        smv[idx / PQ1_PV_TR][idx % PQ1_PV_TR] = pm[i];
        resc |= __float_as_uint(pm[i].x) != 0x3f800000u;
      }
    }
    resc = __syncthreads_or(resc);
    if (t0 + PQ1_PV_TR < T) fetch(t0 + PQ1_PV_TR);
    if (!resc && n == PQ1_PV_TR) {
#pragma unroll
      for (int p = 0; p < PQ1_PV_TR; p++) {
        const float2 m = smv[w][p];
        const float4 v = reinterpret_cast<const float4*>(sv[p])[l];
        const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int i = 0; i < 4; i++) {
          const float y = pq1_ffma(vv[i], m.y, a[i]);
          a[i] = (i < 4 - NALU) ? pq1_rcv(y) : pq1_r16(y);
        }
        S = pq1_ffma(S, m.x, m.y);
      }
    } else {
      for (int p = 0; p < n; p++) {
        const float2 m = smv[w][p];
        const float4 v = reinterpret_cast<const float4*>(sv[p])[l];
        const float vv[4] = {v.x, v.y, v.z, v.w};
        if (__float_as_uint(m.x) != 0x3f800000u) {
#pragma unroll
          for (int i = 0; i < 4; i++) a[i] = pq1_rcv(pq1_fmul(a[i], m.x));
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
          const float y = pq1_ffma(vv[i], m.y, a[i]);
          a[i] = (i < 4 - NALU) ? pq1_rcv(y) : pq1_r16(y);
        }
        S = pq1_ffma(S, m.x, m.y);
      }
    }
  }
  const float si = (__float_as_uint(S) & 0x7fffffffu) ? pq1_fdiv(1.0f, S) : 0.0f;
  reinterpret_cast<float4*>(atto + (size_t)h * 128)[l] =
      make_float4(pq1_fmul(a[0], si), pq1_fmul(a[1], si), pq1_fmul(a[2], si), pq1_fmul(a[3], si));
}

template <int G>
static void pq1_fa_pv_g(const unsigned short* vc, const float2* mv, const uint8_t* par, float* atto,
                        int nkv, int ctx, cudaStream_t s) {
  static int nalu = -1;
  if (nalu < 0) { const char* e = getenv("PQ1_PV_ALU"); nalu = e ? atoi(e) : 0; }
  switch (nalu) {
    case 0: pq1_fa_pv<G, 0><<<nkv, 32 * G, 0, s>>>(vc, mv, par, atto, ctx); break;
    case 2: pq1_fa_pv<G, 2><<<nkv, 32 * G, 0, s>>>(vc, mv, par, atto, ctx); break;
    case 4: pq1_fa_pv<G, 4><<<nkv, 32 * G, 0, s>>>(vc, mv, par, atto, ctx); break;
    default: pq1_fa_pv<G, 1><<<nkv, 32 * G, 0, s>>>(vc, mv, par, atto, ctx); break;
  }
}

extern "C" int pq1_fa_launch(const float* qo_d, const unsigned short* kc_d,
                             const unsigned short* vc_d, float* sc_d, float* mv_d,
                             const uint8_t* par_d, float* atto_d, int nh, int nkv, int ctx,
                             void* stream);
extern "C" int pq1_fa_mask = 7;   /* pq1_parity timing: which of qk|sm|pv to launch */
extern "C" int pq1_attn_launch(const float* qkv_d, const float* nw_d, const uint8_t* par_d,
                               float* qo_d, unsigned short* kc_d, unsigned short* vc_d,
                               float* sc_d, float* mv_d, float* atto_d, int nh, int nkv, int ctx,
                               void* stream) {
  if (nh <= 0 || nkv <= 0 || (nh % nkv) || nh / nkv > 8 || ((nh + 2 * nkv) & 3) || ctx <= 0)
    return -2;
  cudaStream_t s = (cudaStream_t)stream;
  static int apre = -1;
  if (apre < 0) { const char* e = getenv("PQ1_APRE"); apre = e ? atoi(e) : 2; }
  if (apre == 2 && nh + nkv <= PQ1_APRE_MAXH)
    pq1_attn_pre2<<<1, 256, (nh + nkv) * PQ1_AP2_S * sizeof(float), s>>>(qkv_d, nw_d, par_d, qo_d, kc_d,
                                                                   vc_d, nh, nkv, ctx, 0, NULL);
  else if (apre && nh + nkv <= PQ1_APRE_MAXH)
    pq1_attn_pre1<<<1, 256, (nh + nkv) * 129 * sizeof(float), s>>>(qkv_d, nw_d, par_d, qo_d, kc_d,
                                                                   vc_d, nh, nkv, ctx);
  else
    pq1_attn_pre<<<(nh + 2 * nkv) / 4, 128, 0, s>>>(qkv_d, nw_d, par_d, qo_d, kc_d, vc_d, nh, nkv, ctx);
  if (cudaGetLastError() != cudaSuccess) return -1;
  return pq1_fa_launch(qo_d, kc_d, vc_d, sc_d, mv_d, par_d, atto_d, nh, nkv, ctx, stream);
}

extern "C" int pq1_fa_launch(const float* qo_d, const unsigned short* kc_d,
                             const unsigned short* vc_d, float* sc_d, float* mv_d,
                             const uint8_t* par_d, float* atto_d, int nh, int nkv, int ctx,
                             void* stream) {
  if (nh <= 0 || nkv <= 0 || (nh % nkv) || nh / nkv > 8 || ctx <= 0) return -2;
  cudaStream_t s = (cudaStream_t)stream;
  const int G = nh / nkv, qb = min((ctx + 31) / 32, PQ1_FA_QB);
  static int qk4 = -1;
  if (qk4 < 0) { const char* e = getenv("PQ1_QK4"); qk4 = e ? atoi(e) : 1; }
  if (pq1_fa_mask & 1) {
    if (G == 4 && qk4)
      pq1_fa_qk4<<<dim3(nkv, min((ctx + PQ1_QK4_TR - 1) / PQ1_QK4_TR, PQ1_QK4_QB)), 128, 0, s>>>(
          qo_d, kc_d, par_d, sc_d, ctx);
    else
      pq1_fa_qk<<<dim3(nkv, qb), 32 * G, 0, s>>>(qo_d, kc_d, par_d, sc_d, nh, nkv, ctx);
  }
  static int fas = -1;
  if (fas < 0) { const char* e = getenv("PQ1_FAS"); fas = e ? atoi(e) : 1; }
  if (pq1_fa_mask & 2) {
    if (fas) pq1_fa_sm<true><<<nh, 256, 0, s>>>(sc_d, par_d, reinterpret_cast<float2*>(mv_d), ctx);
    else pq1_fa_sm<false><<<nh, 256, 0, s>>>(sc_d, par_d, reinterpret_cast<float2*>(mv_d), ctx);
  }
  if (pq1_fa_mask & 4)
    switch (G) {
#define PQ1_PVG(g) case g: pq1_fa_pv_g<g>(vc_d, reinterpret_cast<const float2*>(mv_d), par_d, atto_d, nkv, ctx, s); break;
      PQ1_PVG(1) PQ1_PVG(2) PQ1_PVG(3) PQ1_PVG(4) PQ1_PVG(5) PQ1_PVG(6) PQ1_PVG(7) PQ1_PVG(8)
#undef PQ1_PVG
    }
  return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

/* ---- whole-token graph: residual add + RMS norm + Q8_K in one kernel --------
 * One block of n/4 threads (n = 4096 for the 8B: 1024 threads), thread t owns
 * x[4t..4t+3]. x = X (+ R: fwd8's X[i] += AO[i] / FO[i], one rn add, stored
 * back to X), then fwd8's rms():
 *   s = fma(double(x_i), double(x_i), s) serially in index order (each product
 *   is exact in double), sc = 1/sqrtf(float(s/n) + 1e-6f), y = (x*sc)*w,
 * then q8_K_quant(y) exactly as pq1_silu_q8k<0> (Q8_K block b = threads
 * [64b, 64b+64), qs stored as one word per thread).
 * The serial double sum is the only order-dependent step, and 4096 dependent DP
 * adds on one lane would take ~200 us. Instead the block sums the same exact
 * squares as a tree (4-term chains, then 5 + 5 shuffle levels), S. All terms
 * are >= 0, so the serial s and S are within (n-1)u and 13u (u = 2^-53,
 * relative) of the exact sum, hence s is in [S(1-2^-40), S(1+2^-40)] for
 * n <= 4096 ((4095+13)u < 2^-41.0). float(rn(s/n)) is monotone in s, so when
 * both ends of that interval (rounded outward) give the same float, that float
 * IS fwd8's; otherwise (~2^-15 of calls) thread 0 runs the serial chain over
 * the shared copy of x. serial=1 forces the chain (parity gate); nfb counts
 * the fallbacks when not NULL. */
/* FAST (loop-261014-2216 row 22, PQ1_RMSV=0 keeps the old tail): lanes 0 and 1
 * form flo and fhi at once, and for n a power of two s/n is __dmul_rn(s, 1/n),
 * the same correctly rounded value of the same exact quotient (1/n exact), so
 * the two serial fp64 divide subroutines leave the critical path. */
template <bool FAST>
__global__ __launch_bounds__(1024) void pq1_rms_q8k(float* __restrict__ X,
                                                    const float* __restrict__ R,
                                                    const float* __restrict__ w,
                                                    uint8_t* __restrict__ xq, int n, int serial,
                                                    int* __restrict__ nfb,
                                                    uint32_t* __restrict__ pl,
                                                    int* __restrict__ au) {
  __shared__ float sx[4096];
  __shared__ uint32_t sq[1024];
  __shared__ double sw[32];
  __shared__ float ssc;
  __shared__ unsigned sam[32];
  __shared__ int sid[32];
  const int t = threadIdx.x, wp = t >> 5, l = t & 31, nw = blockDim.x >> 5;
  float4 v = reinterpret_cast<const float4*>(X)[t];
  if (R) {
    const float4 r = reinterpret_cast<const float4*>(R)[t];
    v.x = pq1_fadd(v.x, r.x); v.y = pq1_fadd(v.y, r.y);
    v.z = pq1_fadd(v.z, r.z); v.w = pq1_fadd(v.w, r.w);
    reinterpret_cast<float4*>(X)[t] = v;
  }
  reinterpret_cast<float4*>(sx)[t] = v;
  double s;
  {
    const double a = pq1_f2d(v.x), b = pq1_f2d(v.y), c = pq1_f2d(v.z), d = pq1_f2d(v.w);
    s = __fma_rn(d, d, __fma_rn(c, c, __fma_rn(b, b, __dmul_rn(a, a))));
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) s = __dadd_rn(s, __shfl_xor_sync(0xffffffffu, s, o));
  if (l == 0) sw[wp] = s;
  __syncthreads();
  if (wp == 0) {
    double S = l < nw ? sw[l] : 0.0;
#pragma unroll
    for (int o = 16; o; o >>= 1) S = __dadd_rn(S, __shfl_xor_sync(0xffffffffu, S, o));
    const double dn = (double)n;
    const bool p2 = FAST && (n & (n - 1)) == 0;
    const double rn = 1.0 / dn;                     /* exact when p2 */
    float flo, fhi;
    if (FAST) {
      const double Sd = l ? __dmul_ru(S, 1.0 + 1.0 / 1099511627776.0)
                          : __dmul_rd(S, 1.0 - 1.0 / 1099511627776.0);
      const float fl = pq1_d2f(p2 ? __dmul_rn(Sd, rn) : __ddiv_rn(Sd, dn));
      flo = fl;
      fhi = __shfl_sync(0xffffffffu, fl, 1);
    }
    if (l == 0) {
      if (!FAST) {
        flo = pq1_d2f(__ddiv_rn(__dmul_rd(S, 1.0 - 1.0 / 1099511627776.0), dn));
        fhi = pq1_d2f(__ddiv_rn(__dmul_ru(S, 1.0 + 1.0 / 1099511627776.0), dn));
      }
      float f = flo;
      if (serial || __float_as_uint(flo) != __float_as_uint(fhi) ||
          (__float_as_uint(flo) & 0x7f800000u) == 0x7f800000u) {
        double q = 0.0;
        for (int j = 0; j < n; j++) { const double e = pq1_f2d(sx[j]); q = __fma_rn(e, e, q); }
        f = pq1_d2f(p2 ? __dmul_rn(q, rn) : __ddiv_rn(q, dn));
        if (nfb) atomicAdd(nfb, 1);
      }
      ssc = pq1_fdiv(1.0f, pq1_fsqrt(pq1_fadd(f, 1e-6f)));
    }
  }
  __syncthreads();
  const float sc = ssc;
  const float4 wv = reinterpret_cast<const float4*>(w)[t];
  float y[4];
  y[0] = pq1_fmul(pq1_fmul(v.x, sc), wv.x);
  y[1] = pq1_fmul(pq1_fmul(v.y, sc), wv.y);
  y[2] = pq1_fmul(pq1_fmul(v.z, sc), wv.z);
  y[3] = pq1_fmul(pq1_fmul(v.w, sc), wv.w);
  /* sx is free again (the serial chain finished before the barrier) */
  reinterpret_cast<float4*>(sx)[t] = make_float4(y[0], y[1], y[2], y[3]);
  unsigned a[4], m = 0u;
#pragma unroll
  for (int k = 0; k < 4; k++) { a[k] = __float_as_uint(y[k]) & 0x7fffffffu; m = max(m, a[k]); }
#pragma unroll
  for (int o = 16; o; o >>= 1) m = max(m, __shfl_xor_sync(0xffffffffu, m, o));
  if (l == 0) sam[wp] = m;
  __syncthreads();
  const int b = t >> 6, lt = t & 63;
  const unsigned am = max(sam[2 * b], sam[2 * b + 1]);
  int idx = 256;
#pragma unroll
  for (int k = 3; k >= 0; k--) if (a[k] == am) idx = 4 * lt + k;
#pragma unroll
  for (int o = 16; o; o >>= 1) idx = min(idx, __shfl_xor_sync(0xffffffffu, idx, o));
  if (l == 0) sid[wp] = idx;
  __syncthreads();
  uint8_t* yb = xq + (size_t)b * 292;
  uint32_t word = 0u, dbits = 0u;                   /* am == 0: fork's d = 0, qs = 0 */
  if (am != 0u) {
    const int first = min(sid[2 * b], sid[2 * b + 1]);
    const float iscale = pq1_fdiv(-127.0f, sx[b * 256 + first]);
#pragma unroll
    for (int k = 0; k < 4; k++) word |= (uint32_t)(uint8_t)(int8_t)pq1_q8n(y[k], iscale) << (8 * k);
    dbits = __float_as_uint(pq1_fdiv(1.0f, iscale));
  }
  reinterpret_cast<uint32_t*>(yb + 4)[lt] = word;
  if (lt == 0) *reinterpret_cast<uint32_t*>(yb) = dbits;
  if (pl) {                                         /* warp wp = 128-group wp */
    sq[t] = word;
    if (lt == 0) pl[(size_t)b * 73] = dbits;
    if (l < 8 && (wp & 1)) pl[(size_t)b * 73 + 65 + l] = 0u;
    __syncthreads();
    /* lane l: word w = l >> 3 of this warp's group, plane q = l & 7 */
    const uint4* src = reinterpret_cast<const uint4*>(sq + wp * 32 + (l >> 3) * 8);
    const uint4 a0 = src[0], a1 = src[1];
    pl[(size_t)b * 73 + 1 + (wp & 1) * 32 + l] = pq1_plane_tr(a0, a1, l & 7);
    if (au) {                                       /* block-uniform */
      int sum = (l & 7) ? 0 : pq1_bsum_x80(a0, a1);
      sum += __shfl_xor_sync(0xffffffffu, sum, 8);
      sum += __shfl_xor_sync(0xffffffffu, sum, 16);
      if (l == 0) au[wp] = sum;
    }
  }
}

static int pq1_rmsv(void) {
  static int v = -1;
  if (v < 0) { const char* e = getenv("PQ1_RMSV"); v = e ? atoi(e) : 1; }
  return v;
}

extern "C" int pq1_rms_q8k_launch(float* X_d, const float* R_d, const float* w_d, uint8_t* xq_d,
                                  int n, int serial, int* nfb_d, void* stream) {
  if (n < 256 || n > 4096 || (n & 255) != 0) return -2;
  if (pq1_rmsv())
    pq1_rms_q8k<true><<<1, n / 4, 0, (cudaStream_t)stream>>>(X_d, R_d, w_d, xq_d, n, serial, nfb_d,
                                                           NULL, NULL);
  else
    pq1_rms_q8k<false><<<1, n / 4, 0, (cudaStream_t)stream>>>(X_d, R_d, w_d, xq_d, n, serial, nfb_d,
                                                            NULL, NULL);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

/* pq1_rms_q8k_launch plus pq1_tm_prep_launch's planes/sums in one kernel */
extern "C" int pq1_rms_q8k_pl_launch(float* X_d, const float* R_d, const float* w_d,
                                     uint8_t* xq_d, uint32_t* pl_d, int* au_d, int n,
                                     void* stream) {
  if (n < 256 || n > 4096 || (n & 255) != 0) return -2;
  if (pq1_rmsv())
    pq1_rms_q8k<true><<<1, n / 4, 0, (cudaStream_t)stream>>>(X_d, R_d, w_d, xq_d, n, 0, NULL, pl_d,
                                                           au_d);
  else
    pq1_rms_q8k<false><<<1, n / 4, 0, (cudaStream_t)stream>>>(X_d, R_d, w_d, xq_d, n, 0, NULL, pl_d,
                                                            au_d);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

/* exhaustive check of pq1_r16 against the conversion pair: every non-NaN f32
 * bit pattern; bad[0] counts mismatches, bad[1] holds the first one */
__global__ void pq1_r16_test(unsigned long long base, unsigned long long* bad) {
  const unsigned long long i = base + (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  const float x = __uint_as_float((unsigned)i);
  if ((((unsigned)i) & 0x7fffffffu) > 0x7f800000u) return;   /* NaN */
  const unsigned a = __float_as_uint(pq1_r16(x)), b = __float_as_uint(pq1_h2f(pq1_f2h(x)));
  if (a != b && atomicAdd(&bad[0], 1ull) == 0ull) bad[1] = i;
}
extern "C" int pq1_r16_test_launch(unsigned long long* bad_d) {
  for (unsigned long long base = 0; base < (1ull << 32); base += (1ull << 24)) {
    pq1_r16_test<<<(1 << 24) / 256, 256>>>(base, bad_d);
    if (cudaGetLastError() != cudaSuccess) return -1;
  }
  return cudaDeviceSynchronize() == cudaSuccess ? 0 : -1;
}

/* greedy argmax of the head logits inside the token graph: fwd8's argmax()
 * (bi = 0; v[i] > v[bi] -> bi = i) is the first index of the largest non-NaN
 * value (+0/-0 tie: the earlier one), and 0 when v[0] is NaN. Each thread
 * scans an ascending stride with the same strict >, then (value, index) pairs
 * merge by larger value, ties to the smaller index; NaN never enters a pair.
 * out[0] = index, out[1] = the bits of v[index]. One block of 1024. */
__global__ void __launch_bounds__(1024) pq1_argmax_k(const float* __restrict__ v, int n,
                                                      int* __restrict__ out) {
  __shared__ float sv[32];
  __shared__ int si[32];
  const int t = threadIdx.x;
  float bv = 0.f;
  int bi = 0x7fffffff;
#pragma unroll 4
  for (int i = t; i < n; i += 1024) {
    const float x = __ldg(v + i);
    if (x == x && (bi == 0x7fffffff || x > bv)) { bv = x; bi = i; }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
    const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (oi != 0x7fffffff && (bi == 0x7fffffff || ov > bv || (ov == bv && oi < bi))) { bv = ov; bi = oi; }
  }
  if ((t & 31) == 0) { sv[t >> 5] = bv; si[t >> 5] = bi; }
  __syncthreads();
  if (t < 32) {
    bv = sv[t]; bi = si[t];
#pragma unroll
    for (int o = 16; o; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (oi != 0x7fffffff && (bi == 0x7fffffff || ov > bv || (ov == bv && oi < bi))) { bv = ov; bi = oi; }
    }
    if (t == 0) {
      const float v0 = v[0];
      if (v0 != v0 || bi == 0x7fffffff) bi = 0;
      out[0] = bi;
      out[1] = __float_as_int(v[bi]);
    }
  }
}
extern "C" int pq1_argmax_launch(const float* v_d, int n, int* out_d, void* stream) {
  if (n <= 0) return -1;
  pq1_argmax_k<<<1, 1024, 0, (cudaStream_t)stream>>>(v_d, n, out_d);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}
