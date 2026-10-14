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

__global__ __launch_bounds__(256, 5)
void pq1_gemv_tm(const uint8_t* __restrict__ w_codes,
                 const __half* __restrict__ w_scales,
                 const uint32_t* __restrict__ xpl, const int* __restrict__ xau,
                 float* __restrict__ y, int K, int N) {
  __shared__ __align__(16) uint8_t sh_x[PQ1_CHUNK_BYTES];
  __shared__ int sh_au[PQ1_CHUNK_GROUPS];
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
  uint32_t* t32 = reinterpret_cast<uint32_t*>(sh_x);

  for (int ch = 0; ch < nchunk; ch++) {
    /* Stage this chunk's plane words (and the fp32 scales) into shared. Same
     * 73-word-per-block source layout, same 68-word-per-block tile layout and
     * the same 2-words-per-thread split as the SWAR kernel; only the XOR 0x80
     * is gone (the host applied the bias while building the planes) and Au now
     * comes from the host instead of a shuffle reduction. */
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
      /* 128 weights = 4 code words (one uint4) = 32 plane words (8 uint4) */
      const uint4* cv = reinterpret_cast<const uint4*>(
          w_codes + (size_t)rrow * (K >> 3) +
          (size_t)(ch * PQ1_CHUNK_GROUPS + g) * 16);
      const uint4* pv =
          reinterpret_cast<uint4*>(t32) + (g >> 1) * 17 + (g & 1) * 8;
      uint4 cw = __ldg(cv);
      int pc = __popc(cw.x) + __popc(cw.y) + __popc(cw.z) + __popc(cw.w);
      int pu = 0;
#pragma unroll
      for (int ci = 0; ci < 4; ci++) {
        uint32_t m = (ci == 0)   ? cw.x
                     : (ci == 1) ? cw.y
                     : (ci == 2) ? cw.z
                                 : cw.w;
        uint4 lo4 = pv[ci * 2];
        uint4 hi4 = pv[ci * 2 + 1];
        int lo = __popc(lo4.x & m) + 2 * __popc(lo4.y & m) +
                 4 * __popc(lo4.z & m) + 8 * __popc(lo4.w & m);
        int hi = 16 * __popc(hi4.x & m) + 32 * __popc(hi4.y & m) +
                 64 * __popc(hi4.z & m) + 128 * __popc(hi4.w & m);
        pu += lo + hi;
      }
      float db;
      __builtin_memcpy(&db, t32 + (g >> 1) * 68 + 64, 4);
      /* identical to the SWAR kernel's epilogue: same integer isum, same fp */
      int isum = (pu << 1) - sh_au[g] + 16384 - (pc << 8);
      float ws = active ? __half2float(__ldg(w_scales + (size_t)rrow * K128 +
                                            (size_t)(ch * PQ1_CHUNK_GROUPS + g)))
                        : 0.0f;
      sh_prod[threadIdx.x >> 2][g] =
          __fmul_rn(__fmul_rn(ws, db), (float)isum);
      __syncwarp();
    }

    if (sub == 0) {
      const float* p = sh_prod[threadIdx.x >> 2];
#pragma unroll
      for (int k = 0; k < PQ1_CHUNK_GROUPS; k++) acc = __fadd_rn(acc, p[k]);
    }
    __syncthreads(); /* the tile is reused by the next chunk */
  }

  if (sub == 0 && active) y[row] = acc;
}

extern "C" int pq1_gemv_tm_launch(const uint8_t* w_codes,
                                  const unsigned short* w_scales_h,
                                  const uint32_t* xpl, const int* xau, float* y,
                                  int K, int N, void* stream) {
  if ((K & 127) != 0) return -2;
  if (K <= 0 || K > PQ1_MAXK || N <= 0) return -2;
  dim3 block(256, 1, 1);
  dim3 grid((N + PQ1_BLOCK_ROWS - 1) / PQ1_BLOCK_ROWS, 1, 1);
  cudaStream_t s = (cudaStream_t)stream;
  pq1_gemv_tm<<<grid, block, 0, s>>>(
      w_codes, reinterpret_cast<const __half*>(w_scales_h), xpl, xau, y, K, N);
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
