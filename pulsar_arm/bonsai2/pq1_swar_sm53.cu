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

#define PQ1_BLOCK_ROWS 16  /* 256 threads = 16 rows x 16 lanes */
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

/* masked 4-lane UNSIGNED byte sum (8 ops): bytes of apack where the keep
 * byte is 0xFF. Two-stage SWAR fold, no sign extension, no IMUL. */
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
  __shared__ float sh_prod[PQ1_BLOCK_ROWS][PQ1_CHUNK_GROUPS];

  int row = blockIdx.x * PQ1_BLOCK_ROWS + (threadIdx.x >> 4);
  int sub = threadIdx.x & 15;
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
    __syncthreads();

    /* 3. one group per lane (128 weights per group, 32 code bytes) */
    int g = sub;
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
    int pu0 = 0, pu1 = 0, pc = 0;
#pragma unroll
    for (int w = 0; w < 4; w++) {
      uint32_t bits = cws[w];
      pc += __popc(bits);
#pragma unroll
      for (int kk = 0; kk < 8; kk++) {
        uint32_t b4 = (bits >> (kk * 4)) & 0xFu;
        /* 4 one-bit weights -> 4 keep bytes (0xFF keep / 0x00 drop) */
        uint32_t keep = ((b4 * 0x00204081u) & 0x01010101u) * 255u;
        uint32_t ap = av[w * 8 + kk]; /* the matching 4 act bytes */
        if (kk & 1) pu1 = pq1_usum4(ap, keep, pu1);
        else        pu0 = pq1_usum4(ap, keep, pu0);
      }
    }
    /* x' = x+128 unsigned, so with Pu = sum of x' where the bit is 1 and
     * Au/pc from staging: sum(w*x) = 2*Pu - Au + 16384 - 256*pc, exact. */
    int isum = ((pu0 + pu1) << 1) - sh_au[g] + 16384 - (pc << 8);
    float ws = active ? __half2float(__ldg(w_scales + (size_t)rrow * K128 +
                                          (ch * PQ1_CHUNK_GROUPS + g)))
                      : 0.0f;
    sh_prod[threadIdx.x >> 4][g] =
        __fmul_rn(__fmul_rn(ws, db), (float)isum);
    __syncwarp();

    /* 4. ordered reduce for this chunk: lane 0 of each row adds the 16
     * products in group order (chunks are processed in order too). */
    if (sub == 0) {
      const float *p = sh_prod[threadIdx.x >> 4];
#pragma unroll
      for (int k = 0; k < PQ1_CHUNK_GROUPS; k++) acc = __fadd_rn(acc, p[k]);
    }
    __syncthreads(); /* the tile is reused by the next chunk */
  }

  if (sub == 0 && active) y[row] = acc;
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
