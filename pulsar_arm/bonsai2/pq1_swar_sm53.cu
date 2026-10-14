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
      int pc0 = __popc(cw0.x) + __popc(cw0.y) + __popc(cw0.z) + __popc(cw0.w);
      int pc1 = __popc(cw1.x) + __popc(cw1.y) + __popc(cw1.z) + __popc(cw1.w);
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
                  64 * __popc(hi4.z & m0) + 128 * __popc(hi4.w & m0);
        pu0 += lo0 + hi0;
        int lo1 = __popc(lo4.x & m1) + 2 * __popc(lo4.y & m1) +
                  4 * __popc(lo4.z & m1) + 8 * __popc(lo4.w & m1);
        int hi1 = 16 * __popc(hi4.x & m1) + 32 * __popc(hi4.y & m1) +
                  64 * __popc(hi4.z & m1) + 128 * __popc(hi4.w & m1);
        pu1 += lo1 + hi1;
      }
      float db;
      __builtin_memcpy(&db, t32 + (g >> 1) * 68 + 64, 4);
      int au = sh_au[g];
      /* identical to the SWAR kernel's epilogue: same integer isum, same fp */
      int isum0 = (pu0 << 1) - au + 16384 - (pc0 << 8);
      int isum1 = (pu1 << 1) - au + 16384 - (pc1 << 8);
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
  pq1_gemv_tm<<<grid, block, 0, s>>>(
      w_codes, reinterpret_cast<const __half*>(w_scales_h), xpl, xau, y, K, N);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
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

extern "C" int pq1_tm_prep_launch(const uint8_t* xq_d, uint32_t* pl_d, int* au_d,
                                  int K, void* stream) {
  if ((K & 255) != 0 || K <= 0 || K > PQ1_MAXK) return -2;
  int ng = K >> 7;
  dim3 block(128, 1, 1);
  dim3 grid((ng + 127) / 128, 1, 1);
  pq1_tm_prep<<<grid, block, 0, (cudaStream_t)stream>>>(xq_d, pl_d, au_d, ng);
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
__device__ __forceinline__ float pq1_fmul(float a, float b) { float r; asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float pq1_fdiv(float a, float b) { float r; asm("div.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float pq1_ffma(float a, float b, float c) { float r; asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c)); return r; }
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

__global__ __launch_bounds__(256) void pq1_silu_q8k(const float* __restrict__ gu, int n,
                                                    uint8_t* __restrict__ xq,
                                                    float* __restrict__ vout) {
  __shared__ float sv[256];
  __shared__ unsigned sm[8];
  __shared__ int si[8];
  const int t = threadIdx.x, i = blockIdx.x * 256 + t;
  const float g = gu[i], u = gu[n + i];
  const float e = pq1_expf_dev(__uint_as_float(__float_as_uint(g) ^ 0x80000000u));
  const float sg = pq1_fdiv(1.0f, pq1_fadd(1.0f, e));
  const float v = pq1_fmul(pq1_fmul(g, sg), u);
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
  if (am == 0u) {                                   /* fork: d = 0, qs = 0 */
    if (t == 0) *reinterpret_cast<uint32_t*>(yb) = 0u;
    yb[4 + t] = 0;
    return;
  }
  int idx = (a == am) ? t : 256;
#pragma unroll
  for (int o = 16; o; o >>= 1) idx = min(idx, __shfl_xor_sync(0xffffffffu, idx, o));
  if ((t & 31) == 0) si[t >> 5] = idx;
  __syncthreads();
  int first = si[0];
#pragma unroll
  for (int k = 1; k < 8; k++) first = min(first, si[k]);
  const float iscale = pq1_fdiv(-127.0f, sv[first]);
  int q = (int)(__float_as_uint(pq1_ffma(v, iscale, 12582912.0f)) & 0x007fffffu) - 0x00400000;
  yb[4 + t] = (uint8_t)(int8_t)min(q, 127);
  if (t == 0) *reinterpret_cast<float*>(yb) = pq1_fdiv(1.0f, iscale);
}

extern "C" int pq1_silu_q8k_launch(const float* gu_d, int n, uint8_t* xq_d, float* vout_d,
                                   void* stream) {
  if (n <= 0 || (n & 255) != 0) return -2;
  pq1_silu_q8k<<<n / 256, 256, 0, (cudaStream_t)stream>>>(gu_d, n, xq_d, vout_d);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}
