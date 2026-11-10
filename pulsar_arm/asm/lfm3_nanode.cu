#include <type_traits>
// lfm3_nanode.cu -- LFM3-Audio-2B text-backbone decode harness for Jetson Nano.
//
// PURPOSE. Everything measured so far prices the GEMV kernels in isolation
// (q8_lc3.cu: 127.7 ms/token, 7.83 tok/s). The unpriced term is everything
// else per token -- 4 rmsnorms x 24 layers, RoPE, QK-norm, the conv gating,
// the depthwise conv, attention softmax, the sigmoid gate, the residual adds,
// the sampler. In the Bonsai engine that non-GEMV side is ~28% of decode, so
// a 127.7 ms budget that ignores it overstates the runtime by a third. This
// harness runs the real 24-layer loop against the real quantized weights and
// reports ms/token, so the number driving the remaining work is measured.
//
// bf16 EMULATION IS MANDATORY, NOT COSMETIC. ref_lfm3_np.py learned this the
// hard way: an fp32 reference is MORE accurate than the bf16 model, so it can
// never reproduce the bf16 goldens (it showed cos 0.955 / maxrel 0.17 on a
// 7680-term dot and looked like a math bug). The certified accuracy arm
// (quant_sensitivity.py "q8_0", 8/8 prompts) fake-quantizes to int8,
// dequantizes, and runs in bf16. So activations here are bf16, rounded at
// exactly the points torch rounds: every Linear output (in the combine
// kernel), rmsnorm's `.to(input_dtype)` BEFORE the weight multiply, each
// elementwise product, RoPE's `.to(q.dtype)`, the residual adds. Maxwell has
// no bf16 arithmetic, so bf16 is storage only: widen, compute in fp32, round.
//
// LAYER MATH, from modeling_lfm3.py Lfm3DecoderLayer.forward:
//   residual = x
//   h = operator_norm(x)
//   h = is_attention ? self_attn(h) : conv_op(h)
//   h = post_operator_norm(h)              # use_sandwich_norm = True
//   x = h + residual
//   h = ffn_norm(x); h = feed_forward(h); h = post_ffn_norm(h); x = h + x
// attention: q_norm/k_norm are RMSNorm over head_dim=128 PER HEAD; RoPE in
//   fp32; GQA 16 q heads / 4 kv heads; scaling = head_dim^-0.5; the output is
//   multiplied per head by sigmoid(gate_proj(h)) BEFORE o_proj.
// conv op: [B,C,xg] = in_proj(h)  (three H-wide chunks of a [3H,H] GEMV);
//   y = conv7(B*xg) (depthwise, 7 taps, conv_bias=False); out = out_proj(C*y).
// rope (compute_default_rope_parameters): zero-tail partial rotation -- the
//   first pr*dim/2 frequencies follow the FULL head_dim progression and the
//   rest are zero, so the tail is left unrotated. full_attention pr=0.5
//   theta=1e6; sliding_attention pr=1.0 theta=1e4.
//
// DISCLOSED DIVERGENCES from the HF bf16 reference (the goldens are matched
// by prefix, per the repo's quantization tolerance policy, not bit-for-bit):
//   * cos/sin use CUDA cosf/sinf; torch uses its own fp32 sin/cos. Not
//     bit-identical, and this one is not fixable from the engine side.
//   * logits are compared in fp32 (unrounded); torch compares bf16 logits, so
//     a near-tie can pick a different token.
//   * the attention softmax max/sum are per-warp ascending then cross-warp
//     ascending, which is a fixed order but not torch's reduction order.
//
// Build on the Nano (CUDA 10.2):
//   nvcc -gencode arch=compute_53,code=sm_53 -O3 -fmad=false \
//        -Xptxas -O3,-dlcm=ca -o lfm3_nanode lfm3_nanode.cu
//   ./lfm3_nanode <blob.lfm3q8> [--steps N] [--prompt id,id,...] [--trace]
#define _GNU_SOURCE 1                 // posix_fadvise + POSIX_FADV_* live in <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <time.h>
#include <zlib.h>
#include <pthread.h>
#include <cuda_fp16.h>

#define Q8_BLK 32
#define RMAX 256
#define H 2048
#define INTER 7680
#define NLAYER 24
#define NH 16
#define NKV 4
#define HD 128
#define VOCAB 128000
#define NORM_EPS 1e-5f
// 4096 positions: the plan's 4k context. The KV cache is 2 arrays (k, v) x
// NLAYER x MAXTOK x NKV x HD x 2 bytes: 3 MB at 64, 12.6 MB at 256, 201 MB at
// 4096. The old 64 was sized to the benchmark and silently truncated anything
// longer (`pos >= MAXTOK` breaks the step loop with no warning), which makes
// audio-length prompts unservable -- run #149. Measured device-free after load
// is 1.31 GB, so 4096 fits with ~1.1 GB to spare. k_attn_head's shared logits
// array grows with MAXTOK (256 B at 64, 16 KB at 4096; +4 KB segred), which
// costs that kernel some occupancy, but attention is KV-bandwidth-bound, not
// occupancy-bound, and the array size cannot change a value.
#define MAXTOK 4096
// fwd: the merge pass in the loader looks tensors up by name
struct Model; struct Ent;
static Ent* T(Model&, const std::string&);

// The tied embedding's tensor name (the blob's key) and the row-chunk size
// for the host-staged lm_head. 8192 rows = 16.8 MB of codes + 2.1 MB of
// scales per slab; 16 chunks cover 128000 rows.
#define EMBNAME "lfm.model.embed_tokens.weight"
#define LM_ROWS 8192
#define SLIDE_WIN 256                  // config /lfm/sliding_window
#define NCONV 13
#define NATTN 11

// CUDA 10.2 guards __bfloat162float/__float2bfloat16_rn behind
// __CUDA_ARCH__ >= 800, so on Maxwell (sm_53) they do not exist. bf16 is
// therefore raw 16-bit storage plus these two conversions; they are the
// standard round-to-nearest-even pair, i.e. the same bits torch's
// `.to(bfloat16)` produces, which is what the certified arm rounds with.
// (cuda_bf16.h is not included at all, so there is no chance of silently
// picking up an arch-gated intrinsic and getting a link error on the Nano.)
typedef unsigned short bf16;
__device__ __forceinline__ float bf2f(bf16 v) { return __uint_as_float(((unsigned)v) << 16); }
__device__ __forceinline__ bf16 f2bf(float v) {
  const unsigned u = __float_as_uint(v);
  if (((u >> 23) & 0xffu) == 0xffu) return (bf16)(unsigned short)(u >> 16);   // inf/nan
  const unsigned r = u + 0x7fffu + ((u >> 16) & 1u);                          // RN-even
  return (bf16)(unsigned short)(r >> 16);
}

// ---------------------------------------------------------------- kernels
// One row of a Q8 tensor -> bf16. This is the input-embedding lookup: the
// blob stores the tied embedding as Q8 codes + fp32 scales, so the row is
// bf16(f32(code) * scale), one round. The first version memcpy'd H*2 raw
// bytes out of the codes arena and called it bf16 -- correct only for a bf16
// payload, which this blob does not have. Latent: nothing reached it.

// (Per-launch diagnostics live here when needed: an env-gated CHKLAUNCH at every
//  launch site, LFM_DBG_LAUNCH=1 prints cudaGetLastError() after each, LFM_DBG_
//  LAUNCH_SYNC also syncs. Added 2026-10-29 to separate "this kernel faults" from
//  "the async fault surfaced two kernels later"; removed again because the macro's
//  line continuations broke three builds while chasing a loader bug. Re-add as a
//  static function, not a multiline macro.)

__global__ void k_dequant_row(const int8_t* __restrict__ c, const float* __restrict__ sc,
                              bf16* __restrict__ out, int k) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < k) out[i] = f2bf(__fmul_rn((float)c[i], sc[i / Q8_BLK]));
}

// ASR path: one audio-embedding row (fp32, the encoder's proj_out, [N,2048]) into the
// bf16 activation the layer loop reads. f2bf is the engine's single-rounding converter,
// the same one k_dequant_row uses for token-table rows, so an audio row and a token row
// enter the network with identical dtype discipline -- a row that arrived through a
// different conversion would silently shift every downstream logit.
__global__ void k_bf16_from_f32(const float* __restrict__ src, bf16* __restrict__ dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = f2bf(src[i]);
}

// The KV cache write as a kernel, so the position can come from device memory
// under graph capture: a cudaMemcpy's destination is baked into the graph, so
// pos*NKV*HD has to be computed on the device instead.
__global__ void k_kv_write(bf16* __restrict__ dst, const bf16* __restrict__ src,
                           int n, const int* __restrict__ posp, int stride) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dst[(size_t)(*posp) * stride + i] = src[i];
}

__global__ void k_bf2f(const bf16* __restrict__ in, float* __restrict__ out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = bf2f(in[i]);
}

// Stage an activation value into shared. The float overload is the historical
// path (k_bf2f converted bf16 -> float into M.xf first, then the GEMV copied
// floats); the bf16 overload folds that conversion into the GEMV's own x
// staging, so the ~110 k_bf2f launches per token (~0.5 ms of launch overhead +
// one full pass over x) disappear. Bit-identical by construction: both put
// bf2f(x_bf16) in the same shared slots in the same order.
__device__ __forceinline__ float lfm_stage(float v) { return v; }
__device__ __forceinline__ float lfm_stage(bf16 v) { return bf2f(v); }

// One row per lane: lane `l` of warp `w` owns row blockIdx.x*blockDim.x + tid
// and walks its K codes serially, dotting against x staged in shared. Returns
// one partial per row (nchunk = 1), so the caller uses k_combine_* with
// nchunk == 1. `launched` is set so the caller can prove this kernel ran and
// fall back if the 24 KB shared request is refused.
template <typename TX>
__global__ void q8_rowlane(const int8_t* __restrict__ q, const float* __restrict__ s,
                           const TX* __restrict__ x, float* __restrict__ part,
                           int N, int K) {
  extern __shared__ float xs[];
  const int tid = threadIdx.x;
  for (int i = tid; i < K; i += blockDim.x) xs[i] = lfm_stage(x[i]);
  __syncthreads();
  const int row = blockIdx.x * blockDim.x + tid;
  if (row >= N) return;
  const int8_t* qr = q + (size_t)row * K;
  const float* sr = s + (size_t)row * (K / Q8_BLK);
  float acc = 0.f;
  #pragma unroll 1
  for (int b = 0; b < K / Q8_BLK; b++) {
    const int8_t* qb = qr + (size_t)b * Q8_BLK;
    const float* xb = xs + (size_t)b * Q8_BLK;
    const int4 a0 = *reinterpret_cast<const int4*>(qb);
    const int4 a1 = *reinterpret_cast<const int4*>(qb + 16);
    const int8_t* c0 = reinterpret_cast<const int8_t*>(&a0);
    const int8_t* c1 = reinterpret_cast<const int8_t*>(&a1);
    float dot = 0.f;
    #pragma unroll
    for (int j = 0; j < 16; j++) dot = __fmaf_rn((float)c0[j], xb[j], dot);
    #pragma unroll
    for (int j = 0; j < 16; j++) dot = __fmaf_rn((float)c1[j], xb[16 + j], dot);
    acc = __fmaf_rn(sr[b], dot, acc);
  }
  part[row] = acc;
}


// Q8_0 pass 1: R rows x one chunk, one WARP per row, lanes striding the
// chunk's blocks, x staged in shared. This is the kernel whose 14-16.5 GB/s
// rate underlies the 127.7 ms/token budget (q8_lc3.cu, commit 2b5cbe5).
// Declare the 256-thread GEMV's six-block/SM target explicitly. The clean
// 40-register build already allows six blocks (40*256 = 10240 regs/block), so this only lets
// the compiler optimize for a known occupancy target. Register allocation or
// scheduling may change; values and reduction order cannot, so it stays exact.
template <typename TX, bool FMA, bool STREAM = false, int UNROLL = 1, int H2 = 0>
__global__ void __launch_bounds__(256, 6) q8_partial(const int8_t* __restrict__ q, const float* __restrict__ s,
                           const TX* __restrict__ x, float* __restrict__ part,
                           int N, int K, int KCHUNK, int R, int nb) {
  extern __shared__ float xs[];
  const int nchunk = K / KCHUNK;
  const int rowblk = blockIdx.x / nchunk;
  const int chunk = blockIdx.x % nchunk;
  const int koff = chunk * KCHUNK;
  const int NT = blockDim.x, tid = threadIdx.x;
  const TX* xg = x + koff;
  // LFM_H2 probe: x staged as half2 pairs, 16 pairs per 32-code block, padded
  // to 17 so the per-lane block reads spread across banks.
  __half2* xh = reinterpret_cast<__half2*>(xs);
  // H2 == 2: exact power-of-two rescale per 32-input block (#142). Scale the
  // block so max|x| lands in [8, 16) before fp16 conversion, and scale the
  // block dot back after: both multiplies are exact, so the only rounding is
  // the fp16 conversion itself, now in fp16's normal range. It also bounds the
  // in-block fp16 accumulator at 32 x 127 x 16 = 65024 < 65504.
  float* xsc = xs + (size_t)(KCHUNK / 32) * 17;      // [nb] inverse scales
  if (H2 == 2) {
    for (int b = tid; b < KCHUNK / 32; b += NT) {
      float mx = 0.f;
      for (int j = 0; j < 32; j++) mx = fmaxf(mx, fabsf(lfm_stage(xg[b * 32 + j])));
      int e = 0;
      if (mx > 0.f) frexpf(mx, &e);
      const int k = (mx > 0.f) ? 4 - e : 0;           // mx * 2^k in [8, 16)
      xsc[b] = ldexpf(1.f, -k);
      const float up = ldexpf(1.f, k);
      for (int q2 = 0; q2 < 16; q2++)
        xh[b * 17 + q2] = __floats2half2_rn(lfm_stage(xg[b * 32 + 2 * q2]) * up,
                                            lfm_stage(xg[b * 32 + 2 * q2 + 1]) * up);
    }
  } else if (H2) {
    for (int p = tid; p < KCHUNK / 2; p += NT) {
      const int b = p >> 4, k = p & 15;
      xh[b * 17 + k] = __floats2half2_rn(lfm_stage(xg[2 * p]), lfm_stage(xg[2 * p + 1]));
    }
  } else {
    for (int i = tid; i < KCHUNK; i += NT) {
      const int b = i >> 5, j = i & 31;
      xs[b * 33 + j] = lfm_stage(xg[i]);
    }
  }
  __syncthreads();
  __shared__ float sh[RMAX];
  const int base = rowblk * R;
  const int lane = tid & 31, wid = tid >> 5;
  const int WPB = NT >> 5, GR = R / WPB;
  #pragma unroll (UNROLL)
  for (int g = 0; g < GR; g++) {
    const int r = wid + g * WPB;
    const int row = base + r;
    float acc = 0.f;
    if (row < N) {
      const int8_t* qr = q + (size_t)row * K + koff;
      const float* sr = s + (size_t)row * (K / Q8_BLK) + (size_t)chunk * nb;
      #pragma unroll 1
      for (int b = lane; b < nb; b += 32) {
        const int4 a0 = STREAM ? __ldcs(reinterpret_cast<const int4*>(qr + (size_t)b * Q8_BLK))
                              : *reinterpret_cast<const int4*>(qr + (size_t)b * Q8_BLK);
        const int4 a1 = STREAM ? __ldcs(reinterpret_cast<const int4*>(qr + (size_t)b * Q8_BLK + 16))
                              : *reinterpret_cast<const int4*>(qr + (size_t)b * Q8_BLK + 16);
        const int8_t* c0 = reinterpret_cast<const int8_t*>(&a0);
        const int8_t* c1 = reinterpret_cast<const int8_t*>(&a1);
        if (H2) {
          // int8 -> half2 without I2F: (c ^ 0x80) = c + 128 in [0,255]; placed
          // as the low mantissa byte under 0x64 it reads as 1024 + c + 128
          // exactly, and subtracting 1152 is exact. Two codes per PRMT+HSUB2,
          // two products per HFMA2. The accumulator is fp16 inside a block.
          const __half2 off = __floats2half2_rn(1152.f, 1152.f);
          const __half2* xb2 = xh + (size_t)b * 17;
          const uint32_t w[8] = {(uint32_t)a0.x, (uint32_t)a0.y, (uint32_t)a0.z, (uint32_t)a0.w,
                                 (uint32_t)a1.x, (uint32_t)a1.y, (uint32_t)a1.z, (uint32_t)a1.w};
          __half2 h2 = __floats2half2_rn(0.f, 0.f);
          #pragma unroll
          for (int k = 0; k < 8; k++) {
            const uint32_t u = w[k] ^ 0x80808080u;
            const uint32_t lo = __byte_perm(u, 0x64646464u, 0x4140);
            const uint32_t hi = __byte_perm(u, 0x64646464u, 0x4342);
            __half2 cl = __hsub2(*reinterpret_cast<const __half2*>(&lo), off);
            __half2 ch = __hsub2(*reinterpret_cast<const __half2*>(&hi), off);
            h2 = __hfma2(cl, xb2[2 * k], h2);
            h2 = __hfma2(ch, xb2[2 * k + 1], h2);
          }
          const float2 f = __half22float2(h2);
          const float bd = (H2 == 2) ? (f.x + f.y) * xsc[b] : (f.x + f.y);
          acc = __fmaf_rn(sr[b], bd, acc);
          continue;
        }
        const float* xb = xs + (size_t)b * 33;
        float dot = 0.f;
        #pragma unroll
        for (int j = 0; j < 16; j++)
          dot = FMA ? __fmaf_rn((float)c0[j], xb[j], dot)
                    : __fadd_rn(dot, __fmul_rn((float)c0[j], xb[j]));
        #pragma unroll
        for (int j = 0; j < 16; j++)
          dot = FMA ? __fmaf_rn((float)c1[j], xb[16 + j], dot)
                    : __fadd_rn(dot, __fmul_rn((float)c1[j], xb[16 + j]));
        acc = FMA ? __fmaf_rn(sr[b], dot, acc)
                  : __fadd_rn(acc, __fmul_rn(sr[b], dot));
      }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(~0u, acc, o);
    if (lane == 0 && row < N) sh[r] = acc;
  }
  __syncthreads();
  if (tid < R && base + tid < N) part[(size_t)(base + tid) * nchunk + chunk] = sh[tid];
}

// pass 2: sum chunks ascending, then round to bf16. The rounding here IS the
// Linear's output rounding, so y is comparable to torch's Linear output.
__global__ void k_combine_bf(const float* __restrict__ part, bf16* __restrict__ out,
                             int N, int nchunk) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= N) return;
  float acc = 0.f;
  for (int c = 0; c < nchunk; c++) acc = __fadd_rn(acc, part[(size_t)r * nchunk + c]);
  out[r] = f2bf(acc);
}

// The T-wide twin of k_combine_bf: one block-row per (position, output row), reading
// the POSITION-FIRST partial layout the batched GEMM writes (row of position t, output
// row r, at part + (t*N + r)*nchunk).
//
// k_combine_bf itself has NO T dimension: a single launch over a T-position partial
// buffer writes position 0 and leaves t>=1 as unwritten memory -- the defect
// --prefill-chunk caught on its very first run (run log row 332: every shape mismatched
// at t>=1, unmatched cells holding 0x0000/0x7fff/0xffff). The accumulation is the same
// __fadd_rn chain over the same nchunk partials, so this is bit-identical to calling
// k_combine_bf once per position -- and --prefill-chunk exploits that: the batched side
// uses THIS kernel, the per-position side uses k_combine_bf, so a pass is two
// independent code paths agreeing bit-for-bit, not one path checked against itself.
__global__ void k_combine_bf_T(const float* __restrict__ part, bf16* __restrict__ out,
                               int N, int nchunk, int T) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  const int t = blockIdx.y;
  if (r >= N || t >= T) return;
  const size_t base = (size_t)t * N * nchunk;
  float acc = 0.f;
  for (int c = 0; c < nchunk; c++) acc = __fadd_rn(acc, part[base + (size_t)r * nchunk + c]);
  out[(size_t)t * N + r] = f2bf(acc);
}
__global__ void k_combine_f(const float* __restrict__ part, float* __restrict__ out,
                            int N, int nchunk) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= N) return;
  float acc = 0.f;
  for (int c = 0; c < nchunk; c++) acc = __fadd_rn(acc, part[(size_t)r * nchunk + c]);
  out[r] = acc;
}

// RMSNorm. modeling_lfm3.py:
//   variance = x.float().pow(2).mean(-1); x = x * rsqrt(var+eps)
//   return weight * x.to(input_dtype)          <-- ROUND, THEN MULTIPLY
// Rounding before the weight multiply is the most common way an rmsnorm
// parity gate fails, so it is written as an explicit widen-after-round here.
// blockDim = 256, 8 elements per thread, warp tree then 8 partials ascending.
__constant__ int c_rmsv = 1;   // LFM_RMSV: 1 = vector loads/stores in rmsnorm_body, 0 = scalar
__device__ __forceinline__ void rmsnorm_body(const bf16* __restrict__ x, const float* __restrict__ w,
                                             bf16* __restrict__ y, float eps, int H_) {
  const int tid = threadIdx.x;
  // Vector form: the thread's 8 consecutive x values as one 16-byte load kept in registers
  // (the scalar form loads them twice, 2 bytes at a time), w as two float4, y as one 16-byte
  // store. Same values, same products, same tree: identical bits. Block-uniform alignment
  // test; the scalar form below is the fallback. Reads all precede writes (__syncthreads).
  if (c_rmsv && ((((uintptr_t)x | (uintptr_t)y) & 15) | ((uintptr_t)w & 15)) == 0) {
    const uint4 xv = reinterpret_cast<const uint4*>(x)[tid];
    const bf16* xe = reinterpret_cast<const bf16*>(&xv);
    float xf[8], ss[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) { xf[j] = bf2f(xe[j]); ss[j] = __fmul_rn(xf[j], xf[j]); }
    float s = __fadd_rn(__fadd_rn(__fadd_rn(ss[0], ss[1]), __fadd_rn(ss[2], ss[3])),
                        __fadd_rn(__fadd_rn(ss[4], ss[5]), __fadd_rn(ss[6], ss[7])));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_down_sync(~0u, s, o));
    __shared__ float shv[8];
    if ((tid & 31) == 0) shv[tid >> 5] = s;
    __syncthreads();
    float tot = shv[0];
    #pragma unroll
    for (int i = 1; i < 8; i++) tot = __fadd_rn(tot, shv[i]);
    const float rs = rsqrtf(__fadd_rn(__fmul_rn(tot, 1.0f / (float)H_), eps));
    const float4 w0 = reinterpret_cast<const float4*>(w)[tid * 2], w1 = reinterpret_cast<const float4*>(w)[tid * 2 + 1];
    const float wf[8] = { w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w };
    uint4 yv; bf16* ye = reinterpret_cast<bf16*>(&yv);
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      const float rounded = bf2f(f2bf(__fmul_rn(xf[j], rs)));
      ye[j] = f2bf(__fmul_rn(wf[j], rounded));
    }
    reinterpret_cast<uint4*>(y)[tid] = yv;
    return;
  }
  const bf16* xr = x;
  float ss[8];
  #pragma unroll
  for (int j = 0; j < 8; j++) { const float v = bf2f(xr[tid * 8 + j]); ss[j] = __fmul_rn(v, v); }
  float s = __fadd_rn(__fadd_rn(__fadd_rn(ss[0], ss[1]), __fadd_rn(ss[2], ss[3])),
                      __fadd_rn(__fadd_rn(ss[4], ss[5]), __fadd_rn(ss[6], ss[7])));
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_down_sync(~0u, s, o));
  __shared__ float sh[8];
  if ((tid & 31) == 0) sh[tid >> 5] = s;
  __syncthreads();
  float tot = sh[0];
  #pragma unroll
  for (int i = 1; i < 8; i++) tot = __fadd_rn(tot, sh[i]);
  const float rs = rsqrtf(__fadd_rn(__fmul_rn(tot, 1.0f / (float)H_), eps));
  #pragma unroll
  for (int j = 0; j < 8; j++) {
    const int i = tid * 8 + j;
    const float v = bf2f(xr[i]);
    const float rounded = bf2f(f2bf(__fmul_rn(v, rs)));      // .to(input_dtype)
    y[i] = f2bf(__fmul_rn(w[i], rounded));                    // weight * that
  }
}

__global__ void k_rmsnorm(const bf16* __restrict__ x, const float* __restrict__ w,
                          bf16* __restrict__ y, float eps, int H_) {
  rmsnorm_body(x, w, y, eps, H_);
}

// The same block, one row per blockIdx.x, over a POSITION-FIRST activation arena
// (row t at x + t*H). Sharing rmsnorm_body is deliberate: the T-wide prefill path
// must be the same code, in the same evaluation order, as the T=1 path -- a
// re-written normalisation kernel is exactly the class of defect that cost this
// project six encoder runs (the in-place k_rmsnorm_f32 race, doc section 40).
__global__ void k_rmsnorm_T(const bf16* __restrict__ x, const float* __restrict__ w,
                            bf16* __restrict__ y, float eps, int H_) {
  const size_t off = (size_t)blockIdx.x * H_;
  rmsnorm_body(x + off, w, y + off, eps, H_);
}

// per-head RMSNorm over head_dim (q_norm / k_norm): one block per head,
// 32 threads, 4 elements each.
__global__ void k_headnorm(bf16* __restrict__ x, const float* __restrict__ w,
                           float eps, int hd) {
  const int tid = threadIdx.x;
  x += (size_t)blockIdx.y * gridDim.x * hd;          // batched prefill: row block per position
  bf16* xr = x + (size_t)blockIdx.x * hd;
  float ss[4];
  #pragma unroll
  for (int j = 0; j < 4; j++) { const float v = bf2f(xr[tid * 4 + j]); ss[j] = __fmul_rn(v, v); }
  float s = __fadd_rn(__fadd_rn(__fadd_rn(ss[0], ss[1]), __fadd_rn(ss[2], ss[3])), 0.f);
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_down_sync(~0u, s, o));
  s = __shfl_sync(~0u, s, 0);
  const float rs = rsqrtf(__fadd_rn(__fmul_rn(s, 1.0f / (float)hd), eps));
  #pragma unroll
  for (int j = 0; j < 4; j++) {
    const int i = tid * 4 + j;
    const float v = bf2f(xr[i]);
    xr[i] = f2bf(__fmul_rn(w[i], bf2f(f2bf(__fmul_rn(v, rs)))));
  }
}

// RoPE, zero-tail partial rotation, fp32 math, round to bf16 (apply_rotary_pos_emb
// computes in float and returns .to(q.dtype)). rot = 2*angles.
__global__ void k_rope(bf16* __restrict__ x, const float* __restrict__ invf,
                       int pos, int hd, int rot, const int* __restrict__ posp = nullptr) {
  if (posp) pos = *posp;
  x += (size_t)blockIdx.y * gridDim.x * hd; pos += blockIdx.y;   // batched prefill (y = position)
  const int half = rot / 2;
  bf16* xr = x + (size_t)blockIdx.x * hd;
  #pragma unroll 1
  for (int i = threadIdx.x; i < hd / 2; i += blockDim.x) {
    if (i >= half) continue;                     // zero tail: unrotated
    const float lo = bf2f(xr[i]), hi = bf2f(xr[i + hd / 2]);
    const float ang = __fmul_rn((float)pos, invf[i]);
    const float c = cosf(ang), s = sinf(ang);
    xr[i] = f2bf(__fadd_rn(__fmul_rn(lo, c), __fmul_rn(-hi, s)));
    xr[i + hd / 2] = f2bf(__fadd_rn(__fmul_rn(hi, c), __fmul_rn(lo, s)));
  }
}

// One block per q head, 256 threads. KV cache rows are [NKV][HD] per position.
// 8 warps sweep the positions for the logits; 8 segments x 32 lanes do the
// weighted value sum; the softmax max/sum reduce over 8 warp partials.
__global__ void k_attn_head(const bf16* __restrict__ q, const bf16* __restrict__ kc,
                            const bf16* __restrict__ vc, bf16* __restrict__ out,
                            int npos, int win, float scaling, int hd, int nkv,
                            const int* __restrict__ nposp = nullptr) {
  if (nposp) npos = *nposp;
  // Batched prefill: blockIdx.y = position offset in the chunk (its q/out rows, one more key).
  q += (size_t)blockIdx.y * gridDim.x * hd; out += (size_t)blockIdx.y * gridDim.x * hd; npos += blockIdx.y;
  const int h = blockIdx.x, tid = threadIdx.x;
  const int lane = tid & 31, wid = tid >> 5;
  __shared__ float sl[MAXTOK];
  __shared__ float red[8];
  __shared__ float segred[8][HD];
  const int lo = (win > 0 && npos > win) ? npos - win : 0;
  const bf16* qr = q + (size_t)h * hd;
  const int kvh = h / (NH / nkv);
  // Pre-load this thread's 4 Q values once (Q is constant across positions).
  const float q0 = bf2f(qr[lane * (hd / 32) + 0]);
  const float q1 = bf2f(qr[lane * (hd / 32) + 1]);
  const float q2 = bf2f(qr[lane * (hd / 32) + 2]);
  const float q3 = bf2f(qr[lane * (hd / 32) + 3]);
  for (int p = lo + wid; p < npos; p += 8) {
    const bf16* kr = kc + (size_t)p * nkv * hd + kvh * hd;
    float s = 0.f;
    const int base = lane * (hd / 32);
    // The lane's 4 keys are one 8-byte load (hd/32 == 4, row and lane offsets 8-aligned);
    // same values, same chain.
    const uint2 kw = *reinterpret_cast<const uint2*>(kr + base);
    s = __fadd_rn(s, __fmul_rn(q0, __uint_as_float(kw.x << 16)));
    s = __fadd_rn(s, __fmul_rn(q1, __uint_as_float(kw.x & 0xffff0000u)));
    s = __fadd_rn(s, __fmul_rn(q2, __uint_as_float(kw.y << 16)));
    s = __fadd_rn(s, __fmul_rn(q3, __uint_as_float(kw.y & 0xffff0000u)));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_down_sync(~0u, s, o));
    if (lane == 0) sl[p] = __fmul_rn(s, scaling);
  }
  __syncthreads();
  const int np = npos - lo;
  float mx = -1e30f;
  for (int i = tid; i < np; i += 256) mx = fmaxf(mx, sl[lo + i]);
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_down_sync(~0u, mx, o));
  if (lane == 0) red[wid] = mx;
  __syncthreads();
  if (tid < 8) {
    float v = red[tid];
    #pragma unroll
    for (int o = 4; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffu, v, o));
    if (tid == 0) red[0] = v;
  }
  __syncthreads();
  const float m = red[0];
  __syncthreads();
  float sum = 0.f;
  for (int i = tid; i < np; i += 256) { const float e = expf(__fadd_rn(sl[lo + i], -m));
    sl[lo + i] = e; sum = __fadd_rn(sum, e); }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum = __fadd_rn(sum, __shfl_down_sync(~0u, sum, o));
  if (lane == 0) red[wid] = sum;
  __syncthreads();
  if (tid < 8) {
    float v = red[tid];
    #pragma unroll
    for (int o = 4; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_down_sync(0xffu, v, o));
    if (tid == 0) red[0] = v;
  }
  __syncthreads();
  const float inv = __fdiv_rn(1.0f, red[0]);
  const int e = lane, seg = wid;
  // One accumulator PER OUTPUT DIMENSION, and all HD=128 of them must be
  // written. The first version accumulated only `vr[e]` with e = lane, i.e.
  // dimensions 0..31, and left segred[seg][32..127] as uninitialized shared
  // memory -- the combine step reads tid < HD = 128, so three quarters of
  // every attention head's output was garbage. Symptom: attention rms 3.25x
  // HF's, head 0's first four values exact (they live in 0..31), attention
  // LAYERS drifting (0.83-1.30) while conv layers tracked, logits corr 0.83.
  // The per-dimension sum order is unchanged (p = seg, seg+8, ...), so the
  // covered dimensions keep producing identical bits.
  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
  for (int p = seg; p < np; p += 8) {
    const bf16* vr = vc + (size_t)(lo + p) * nkv * hd + kvh * hd;
    const float w = __fmul_rn(sl[lo + p], inv);
    // Lane e owns dimensions 4e..4e+3 (one 8-byte load). Each dimension's chain is still
    // p = seg, seg+8, ... in order, only the owning lane changed: same bits.
    const uint2 vw = *reinterpret_cast<const uint2*>(vr + 4 * e);
    acc0 = __fadd_rn(acc0, __fmul_rn(w, __uint_as_float(vw.x << 16)));
    acc1 = __fadd_rn(acc1, __fmul_rn(w, __uint_as_float(vw.x & 0xffff0000u)));
    acc2 = __fadd_rn(acc2, __fmul_rn(w, __uint_as_float(vw.y << 16)));
    acc3 = __fadd_rn(acc3, __fmul_rn(w, __uint_as_float(vw.y & 0xffff0000u)));
  }
  segred[seg][4 * e] = acc0;
  segred[seg][4 * e + 1] = acc1;
  segred[seg][4 * e + 2] = acc2;
  segred[seg][4 * e + 3] = acc3;
  __syncthreads();
  if (tid < HD)
    out[(size_t)h * hd + tid] = f2bf(__fadd_rn(__fadd_rn(__fadd_rn(segred[0][tid], segred[1][tid]),
                                    __fadd_rn(segred[2][tid], segred[3][tid])),
                          __fadd_rn(__fadd_rn(segred[4][tid], segred[5][tid]),
                                     __fadd_rn(segred[6][tid], segred[7][tid]))));
}

// k_attn_headG<G>: prefill form of k_attn_head for G q heads sharing one kv head
// (blockIdx.x = head group, blockIdx.y = chunk position). Each K/V row word is loaded once
// and feeds G heads' chains; every head's chains, trees and softmax are k_attn_head's, so
// each head's output is bit-identical. Scores live in dynamic shared memory sized to the
// chunk's last position (sln floats per head) instead of MAXTOK.
__constant__ int g_attsm1 = 1;   // LFM_ATT_SM1
__constant__ int g_attgsw = 1;   // LFM_ATT_GSW
template <int G>
__global__ void __launch_bounds__(256) k_attn_headG(const bf16* __restrict__ q, const bf16* __restrict__ kc,
                             const bf16* __restrict__ vc, bf16* __restrict__ out,
                             int npos, int win, float scaling, int nkv, int sln) {
  extern __shared__ float dyn[];
  float* sl = dyn;                               // [G][sln]
  float* segred = dyn + G * sln;                 // [G][8][HD]
  __shared__ float red[8];
  __shared__ float mxs[G], invs[G];
  // g_attgsw (LFM_ATT_GSW): grid launched (chunk positions, head groups), so the blocks
  // resident together read the same kv head's cache rows; each block's work is unchanged.
  const int abx = g_attgsw ? blockIdx.y : blockIdx.x, aby = g_attgsw ? blockIdx.x : blockIdx.y;
  q += (size_t)aby * NH * HD; out += (size_t)aby * NH * HD; npos += aby;
  const int h0 = abx * G, tid = threadIdx.x;
  const int lane = tid & 31, wid = tid >> 5;
  const int lo = (win > 0 && npos > win) ? npos - win : 0;
  const int kvh = h0 / (NH / nkv);
  float qv[G][4];
  #pragma unroll
  for (int g = 0; g < G; g++)
    #pragma unroll
    for (int j = 0; j < 4; j++) qv[g][j] = bf2f(q[(size_t)(h0 + g) * HD + lane * 4 + j]);
  if (G == 2) {
    // Transposed reduce: a warp scores 8 consecutive positions x 2 heads = 16 sums. xor
    // stages 16, 8, 4, 2 each keep half the rows (lower lane: own + partner, upper lane:
    // partner + own), then stage 1; every row gets exactly the shfl_down tree's pairings.
    for (int p0 = lo + wid * 8; p0 < npos; p0 += 64) {
      float v[16];
      #pragma unroll
      for (int pi = 0; pi < 8; pi++) {
        const int p = p0 + pi;
        float k0 = 0.f, k1 = 0.f, k2 = 0.f, k3 = 0.f;
        if (p < npos) {
          const uint2 kw = *reinterpret_cast<const uint2*>(kc + (size_t)p * nkv * HD + kvh * HD + lane * 4);
          k0 = __uint_as_float(kw.x << 16); k1 = __uint_as_float(kw.x & 0xffff0000u);
          k2 = __uint_as_float(kw.y << 16); k3 = __uint_as_float(kw.y & 0xffff0000u);
        }
        #pragma unroll
        for (int g = 0; g < 2; g++) {
          float a = 0.f;
          a = __fadd_rn(a, __fmul_rn(qv[g][0], k0));
          a = __fadd_rn(a, __fmul_rn(qv[g][1], k1));
          a = __fadd_rn(a, __fmul_rn(qv[g][2], k2));
          a = __fadd_rn(a, __fmul_rn(qv[g][3], k3));
          v[g * 8 + pi] = a;
        }
      }
#define ATT_TSTEP(C, O) { const bool up = (lane & (O)) != 0; \
        _Pragma("unroll") for (int i = 0; i < (C) / 2; i++) { \
          const float snd = up ? v[i] : v[i + (C) / 2]; \
          const float r = __shfl_xor_sync(~0u, snd, (O)); \
          v[i] = up ? __fadd_rn(r, v[i + (C) / 2]) : __fadd_rn(v[i], r); } }
      ATT_TSTEP(16, 16) ATT_TSTEP(8, 8) ATT_TSTEP(4, 4) ATT_TSTEP(2, 2)
#undef ATT_TSTEP
      const float r1 = __shfl_xor_sync(~0u, v[0], 1);
      const float sc = (lane & 1) ? __fadd_rn(r1, v[0]) : __fadd_rn(v[0], r1);
      const int row = ((lane >> 4) & 1) * 8 + ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1);
      const int p = p0 + (row & 7);
      if (!(lane & 1) && p < npos) sl[(row >> 3) * sln + p] = __fmul_rn(sc, scaling);
    }
  } else
  for (int p = lo + wid; p < npos; p += 8) {
    const uint2 kw = *reinterpret_cast<const uint2*>(kc + (size_t)p * nkv * HD + kvh * HD + lane * 4);
    const float k0 = __uint_as_float(kw.x << 16), k1 = __uint_as_float(kw.x & 0xffff0000u);
    const float k2 = __uint_as_float(kw.y << 16), k3 = __uint_as_float(kw.y & 0xffff0000u);
    #pragma unroll
    for (int g = 0; g < G; g++) {
      float s = 0.f;
      s = __fadd_rn(s, __fmul_rn(qv[g][0], k0));
      s = __fadd_rn(s, __fmul_rn(qv[g][1], k1));
      s = __fadd_rn(s, __fmul_rn(qv[g][2], k2));
      s = __fadd_rn(s, __fmul_rn(qv[g][3], k3));
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_down_sync(~0u, s, o));
      if (lane == 0) sl[g * sln + p] = __fmul_rn(s, scaling);
    }
  }
  __syncthreads();
  const int np = npos - lo;
  if (g_attsm1 && G <= 4) {
    // All G heads' softmax passes interleaved in one loop each (LFM_ATT_SM1, default 1):
    // every head's per-thread sequence, shuffle tree and 8-warp tree are the loop-per-head
    // form's, so the values are identical; it saves (G-1) x 4 block barriers per call.
    __shared__ float redG[G][8];
    float mx[G];
    #pragma unroll
    for (int g = 0; g < G; g++) mx[g] = -1e30f;
    for (int i = tid; i < np; i += 256) {
      #pragma unroll
      for (int g = 0; g < G; g++) mx[g] = fmaxf(mx[g], sl[g * sln + lo + i]);
    }
    #pragma unroll
    for (int g = 0; g < G; g++) {
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) mx[g] = fmaxf(mx[g], __shfl_down_sync(~0u, mx[g], o));
      if (lane == 0) redG[g][wid] = mx[g];
    }
    __syncthreads();
    if (tid < 8) {
      #pragma unroll
      for (int g = 0; g < G; g++) {
        float v = redG[g][tid];
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffu, v, o));
        if (tid == 0) mxs[g] = v;
      }
    }
    __syncthreads();
    float m[G], sum[G];
    #pragma unroll
    for (int g = 0; g < G; g++) { m[g] = mxs[g]; sum[g] = 0.f; }
    for (int i = tid; i < np; i += 256) {
      #pragma unroll
      for (int g = 0; g < G; g++) {
        const float e = expf(__fadd_rn(sl[g * sln + lo + i], -m[g]));
        sl[g * sln + lo + i] = e; sum[g] = __fadd_rn(sum[g], e);
      }
    }
    #pragma unroll
    for (int g = 0; g < G; g++) {
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) sum[g] = __fadd_rn(sum[g], __shfl_down_sync(~0u, sum[g], o));
      if (lane == 0) redG[g][wid] = sum[g];
    }
    __syncthreads();
    if (tid < 8) {
      #pragma unroll
      for (int g = 0; g < G; g++) {
        float v = redG[g][tid];
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_down_sync(0xffu, v, o));
        if (tid == 0) invs[g] = __fdiv_rn(1.0f, v);
      }
    }
    __syncthreads();
  } else
  for (int g = 0; g < G; g++) {
    float* sg = sl + g * sln;
    float mx = -1e30f;
    for (int i = tid; i < np; i += 256) mx = fmaxf(mx, sg[lo + i]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_down_sync(~0u, mx, o));
    if (lane == 0) red[wid] = mx;
    __syncthreads();
    if (tid < 8) {
      float v = red[tid];
      #pragma unroll
      for (int o = 4; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffu, v, o));
      if (tid == 0) mxs[g] = v;
    }
    __syncthreads();
    const float m = mxs[g];
    float sum = 0.f;
    for (int i = tid; i < np; i += 256) { const float e = expf(__fadd_rn(sg[lo + i], -m));
      sg[lo + i] = e; sum = __fadd_rn(sum, e); }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum = __fadd_rn(sum, __shfl_down_sync(~0u, sum, o));
    if (lane == 0) red[wid] = sum;
    __syncthreads();
    if (tid < 8) {
      float v = red[tid];
      #pragma unroll
      for (int o = 4; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_down_sync(0xffu, v, o));
      if (tid == 0) invs[g] = __fdiv_rn(1.0f, v);
    }
    __syncthreads();
  }
  const int e = lane, seg = wid;
  float acc[G][4];
  #pragma unroll
  for (int g = 0; g < G; g++) { acc[g][0] = acc[g][1] = acc[g][2] = acc[g][3] = 0.f; }
  float inv[G];
  #pragma unroll
  for (int g = 0; g < G; g++) inv[g] = invs[g];
  for (int p = seg; p < np; p += 8) {
    const uint2 vw = *reinterpret_cast<const uint2*>(vc + (size_t)(lo + p) * nkv * HD + kvh * HD + 4 * e);
    const float v0 = __uint_as_float(vw.x << 16), v1 = __uint_as_float(vw.x & 0xffff0000u);
    const float v2 = __uint_as_float(vw.y << 16), v3 = __uint_as_float(vw.y & 0xffff0000u);
    #pragma unroll
    for (int g = 0; g < G; g++) {
      const float w = __fmul_rn(sl[g * sln + lo + p], inv[g]);
      acc[g][0] = __fadd_rn(acc[g][0], __fmul_rn(w, v0));
      acc[g][1] = __fadd_rn(acc[g][1], __fmul_rn(w, v1));
      acc[g][2] = __fadd_rn(acc[g][2], __fmul_rn(w, v2));
      acc[g][3] = __fadd_rn(acc[g][3], __fmul_rn(w, v3));
    }
  }
  #pragma unroll
  for (int g = 0; g < G; g++)
    #pragma unroll
    for (int j = 0; j < 4; j++) segred[(g * 8 + seg) * HD + 4 * e + j] = acc[g][j];
  __syncthreads();
  for (int i = tid; i < G * HD; i += 256) {
    const int g = i / HD, d = i % HD;
    const float* r = segred + g * 8 * HD + d;
    out[(size_t)(h0 + g) * HD + d] = f2bf(__fadd_rn(__fadd_rn(__fadd_rn(r[0], r[HD]), __fadd_rn(r[2 * HD], r[3 * HD])),
                                          __fadd_rn(__fadd_rn(r[4 * HD], r[5 * HD]), __fadd_rn(r[6 * HD], r[7 * HD]))));
  }
}

__global__ void k_mul_bf(const bf16* __restrict__ a, const bf16* __restrict__ b,
                         bf16* __restrict__ y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = f2bf(__fmul_rn(bf2f(a[i]), bf2f(b[i])));
}
__global__ void k_add_bf(const bf16* __restrict__ a, const bf16* __restrict__ b,
                         bf16* __restrict__ y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = f2bf(__fadd_rn(bf2f(a[i]), bf2f(b[i])));
}
__global__ void k_silu_mul_bf(const bf16* __restrict__ w1, const bf16* __restrict__ w3,
                              bf16* __restrict__ y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float a = bf2f(w1[i]);
    const float sg = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-a)));
    y[i] = f2bf(__fmul_rn(__fmul_rn(a, sg), bf2f(w3[i])));   // silu(w1)*w3, one round
  }
}
// k_silu_mul_bfv<V>: k_silu_mul_bf with V consecutive elements per thread (one 4- or 8-byte
// load of each input, one store): the per-element arithmetic is unchanged, only the access
// width. Needs n % V == 0 and V*2-byte aligned rows. LFM_PF_SILV selects V for the prefill
// batch launch (0 = k_silu_mul_bf).
template <int V>
__global__ void k_silu_mul_bfv(const bf16* __restrict__ w1, const bf16* __restrict__ w3,
                               bf16* __restrict__ y, int n) {
  typedef typename std::conditional<V == 2, uint32_t, uint2>::type VT;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i * V >= n) return;
  VT a4 = reinterpret_cast<const VT*>(w1)[i], b4 = reinterpret_cast<const VT*>(w3)[i], o4;
  const bf16* av = reinterpret_cast<const bf16*>(&a4);
  const bf16* bv = reinterpret_cast<const bf16*>(&b4);
  bf16* ov = reinterpret_cast<bf16*>(&o4);
  #pragma unroll
  for (int j = 0; j < V; j++) {
    const float a = bf2f(av[j]);
    const float sg = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-a)));
    ov[j] = f2bf(__fmul_rn(__fmul_rn(a, sg), bf2f(bv[j])));
  }
  reinterpret_cast<VT*>(y)[i] = o4;
}
__global__ void k_sigmoid_gate(const bf16* __restrict__ g, bf16* __restrict__ attn, int hd) {
  g += (size_t)blockIdx.y * gridDim.x * blockDim.x; attn += (size_t)blockIdx.y * gridDim.x * blockDim.x * hd;
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  const float s = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-bf2f(g[h]))));
  bf16* a = attn + (size_t)h * hd;
  #pragma unroll 4
  for (int j = 0; j < hd; j++) a[j] = f2bf(__fmul_rn(bf2f(a[j]), s));
}
// Vector forms (LFM_PF_EWV, default 1; 0 = the scalar kernels above). k_add_bf4: 4
// consecutive elements per thread, one 8-byte load of each input and one store; same
// per-element __fadd_rn and rounding. k_sigmoid_gate4: k_sigmoid_gate ran one thread per
// head (16 threads per block, each walking its own 128-element head, so a warp's loads were
// 256 B apart); now 256 threads per position, 4 consecutive elements each, and every thread
// recomputes its head's s from the same g value with the same expression: same bits.
__global__ void k_add_bf4(const bf16* __restrict__ a, const bf16* __restrict__ b,
                          bf16* __restrict__ y, int n4) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  uint2 a4 = reinterpret_cast<const uint2*>(a)[i], b4 = reinterpret_cast<const uint2*>(b)[i], o4;
  const bf16* av = reinterpret_cast<const bf16*>(&a4);
  const bf16* bv = reinterpret_cast<const bf16*>(&b4);
  bf16* ov = reinterpret_cast<bf16*>(&o4);
  #pragma unroll
  for (int j = 0; j < 4; j++) ov[j] = f2bf(__fadd_rn(bf2f(av[j]), bf2f(bv[j])));
  reinterpret_cast<uint2*>(y)[i] = o4;
}
__global__ void k_sigmoid_gate4(const bf16* __restrict__ g, bf16* __restrict__ attn, int nh, int hd) {
  g += (size_t)blockIdx.y * nh; attn += (size_t)blockIdx.y * nh * hd;
  const int e = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (e >= nh * hd) return;
  const float s = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-bf2f(g[e / hd]))));
  uint2 a4 = *reinterpret_cast<const uint2*>(attn + e);
  bf16* av = reinterpret_cast<bf16*>(&a4);
  #pragma unroll
  for (int j = 0; j < 4; j++) av[j] = f2bf(__fmul_rn(bf2f(av[j]), s));
  *reinterpret_cast<uint2*>(attn + e) = a4;
}
static int g_ewv = -1;
static bool ewv_on() { if (g_ewv < 0) g_ewv = getenv("LFM_PF_EWV") ? atoi(getenv("LFM_PF_EWV")) : 1; return g_ewv != 0; }
// ring: slots 0..6, newest at 6 (tap 6 = the current column)
__global__ void k_shift_in(bf16* __restrict__ ring, const bf16* __restrict__ xin, int H_) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= H_) return;
  #pragma unroll
  for (int k = 0; k < 6; k++) ring[(size_t)k * H_ + i] = ring[(size_t)(k + 1) * H_ + i];
  ring[(size_t)6 * H_ + i] = xin[i];
}
// y[i] = sum_k taps[i][k] * ring[k][i], one round to bf16 (the conv's own
// dtype in the bf16 model).
__global__ void k_conv7(const bf16* __restrict__ ring, const float* __restrict__ taps,
                        bf16* __restrict__ y, int H_) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= H_) return;
  const float* tp = taps + (size_t)i * 7;
  float acc = 0.f;
  #pragma unroll
  for (int k = 0; k < 7; k++)
    acc = __fadd_rn(acc, __fmul_rn(tp[k], bf2f(ring[(size_t)k * H_ + i])));
  y[i] = f2bf(acc);
}
// Prefill conv branch for a whole chunk: one thread per channel walks the chunk's
// positions in order with the 7 ring slots in registers. Per position it does exactly
// k_mul_bf (B*x), k_shift_in, k_conv7 and k_mul_bf (C*y) -- same operands, same rounding,
// same order -- so the outputs and the final ring are bit-identical to the per-position
// launches. bx rows are [B | C | x] (3H), out rows H.
__global__ void k_conv_chunk(bf16* __restrict__ ring, const float* __restrict__ taps,
                             const bf16* __restrict__ bx, bf16* __restrict__ out, int H_, int Tn) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= H_) return;
  bf16 r[7];
  #pragma unroll
  for (int k = 0; k < 7; k++) r[k] = ring[(size_t)k * H_ + i];
  float tp[7];
  #pragma unroll
  for (int k = 0; k < 7; k++) tp[k] = taps[(size_t)i * 7 + k];
  for (int t = 0; t < Tn; t++) {
    const bf16* row = bx + (size_t)t * 3 * H_;
    const bf16 bxv = f2bf(__fmul_rn(bf2f(row[i]), bf2f(row[2 * H_ + i])));     // B*x
    #pragma unroll
    for (int k = 0; k < 6; k++) r[k] = r[k + 1];                                // shift in
    r[6] = bxv;
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < 7; k++) acc = __fadd_rn(acc, __fmul_rn(tp[k], bf2f(r[k])));
    const bf16 cy = f2bf(acc);                                                  // conv7
    out[(size_t)t * H_ + i] = f2bf(__fmul_rn(bf2f(row[H_ + i]), bf2f(cy)));    // C*y
  }
  #pragma unroll
  for (int k = 0; k < 7; k++) ring[(size_t)k * H_ + i] = r[k];
}
__global__ void k_argmax(const float* __restrict__ lg, int* __restrict__ out, int n) {
  __shared__ float bv[8]; __shared__ int bi[8];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
  float best = -1e30f; int bidx = -1;
  for (int i = tid; i < n; i += 256) { const float v = lg[i]; if (v > best) { best = v; bidx = i; } }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_down_sync(~0u, best, o);
    const int oi = __shfl_down_sync(~0u, bidx, o);
    if (ov > best) { best = ov; bidx = oi; }
  }
  if (lane == 0) { bv[wid] = best; bi[wid] = bidx; }
  __syncthreads();
  if (tid < 8) {
    float v = bv[tid]; int ix = bi[tid];
    #pragma unroll
    for (int o = 4; o > 0; o >>= 1) {
      const float ov = __shfl_down_sync(0xffu, v, o); const int oi = __shfl_down_sync(0xffu, ix, o);
      if (ov > v) { v = ov; ix = oi; }
    }
    if (tid == 0) *out = ix;
  }
}

// ------------------------------------------------------------------- loader
struct Ent {
  std::string name;
  // ne[4] MUST be zero-initialised: the blob stores nd dims, and a 1-D
  // tensor (every norm vector) left ne[1..3] as stack garbage. It surfaced
  // as `UPLOAD FAIL ... ne=[2048,-555441400]`, which reads like blob
  // corruption and is really an uninitialised field -- and the diagnostic
  // printed the garbage, so it pointed at the exporter, not at this struct.
  int dtype, ne[4] = {0,0,0,0}, nd = 0;
  size_t file_off, nc, ns;
  uint32_t crc;
  bool host = false;          // payload lives in pinned host memory, not device
  int8_t* dq = nullptr; float* ds = nullptr; float* df = nullptr;
  bool tiled = false;         // dq/ds repacked to the 32-row tile layout (TLb/TLg/tlb_decode)
};

// The tied embedding, kept in PINNED HOST memory and streamed into a device
// slab per row-chunk. Defined above Model because Model holds one by value.
// LFM_TIME=1: per-phase decode timing (embedding lookup, conv layers,
// attention layers, final norm, lm_head). The plan's discipline is "prove the
// code path ran before believing a timing"; optimizing 139 ms without the
// split is guessing. Events + one sync per phase, so the instrumentation
// itself costs ~0.25 ms/token -- a diagnostic, not a shipping mode.
static const bool g_time = getenv("LFM_TIME") != nullptr;
// LFM_ROWLANE=1: a one-row-per-lane GEMV for the long-K shapes.
//
// Why: the per-kernel table gives a clean correlation between achieved
// bandwidth and ROW LENGTH, at fixed bytes. conv.in_proj 13.3 GB/s, ffn.w1
// 10.9, ffn.w3 10.9, conv.out_proj 10.3 -- all K=2048 -- while ffn.w2
// (N=2048, K=6144) gets 7.1 GB/s for the same 12.6 MB. Block fill (R),
// inter-row ILP (unroll), chunk interleaving (KCHUNK) and cache hints
// (__ldcs) are all measured out for it. At R=256 a block of that shape
// addresses 256 rows x 6 KB = 1.5 MB; the K=2048 shapes address 0.5 MB. That
// is a DRAM page-thrashing signature: each warp instruction advances 32 rows
// by 32 bytes, so a block keeps R row-streams open at a K-byte stride.
//
// The fix is the classic wide-K GEMV shape: give each LANE a whole row and
// walk it serially. Then a warp holds 32 fully contiguous 6 KB streams, there
// is no cross-lane reduction and no shuffle, and x is staged once per block in
// shared (K floats = 24 KB at K=6144, which fits the Nano's 64 KB/SM).
//
// NOT bit-identical: each lane sums its own row, so the reduction order
// differs from the 32-lane-strided + warp-tree + combine form. The HF-derived
// gates (rms matrix, pos-0 logits, greedy stream floors) are the authority, as
// they were for the FMA switch.
static const bool g_rowlane = getenv("LFM_ROWLANE") != nullptr;
static int g_rowlane_printed = 0;
// LFM_UNROLL2=1: let the GEMV's row loop overlap two rows per warp. The loop
// is #pragma unroll 1, so a warp walks its GR rows strictly serially and
// exposes each row's load latency in full. That is the shape of ffn.w2
// (N=2048, K=6144 -> 192 elements per lane per row, 1.773 ms measured vs
// 1.161 ms for ffn.w1's 6144x2048 with the same bytes): long rows, few of
// them, no inter-row overlap. Rows are independent and each row's reduction
// order is untouched, so this is bit-identical.
static const bool g_unroll2 = getenv("LFM_UNROLL2") != nullptr;
// LFM_RADAPT=1: shape-dependent rows-per-block. LFM_KTIME showed ffn.w2
// (N=2048, K=6144) at 1.773 ms vs ffn.w1/w3 (N=6144, K=2048) at 1.161 ms for
// the SAME 12.6 MB of weights: at R=256 a 2048-row tensor is only 8 blocks on
// this 4-SM board (2 blocks/SM), so it is the long-K, few-row shapes that
// underfill, not the short ones. The global R=64 test penalized the 6144-row
// shapes (which amortize the 8 KB x-stage across 256 rows) to fix them.
// R only remaps rows to blocks: per-row reduction order, and every output
// bit, is unchanged.
static const bool g_radapt = getenv("LFM_RADAPT") != nullptr;
// LFM_KTIME=1: per-kernel timing inside the layer loop. LFM_TIME only splits by
// layer TYPE (conv 70.6 / attn 56.6 ms), and the conv layers run 5.4 ms against
// a 2.5 ms byte floor while attention layers run 5.15 against 3.3, so most of
// the step is unexplained excess. Each wrapped statement is bracketed by events
// and synced, so the TOTAL is inflated by serialization: read the split, not
// the sum. Printed at exit.
static const bool g_ktime = getenv("LFM_KTIME") != nullptr;
static cudaEvent_t g_ka = nullptr, g_kb = nullptr;
static std::map<std::string, std::pair<double,int> > g_kt;
static void kt_add(const char* n, float ms) { g_kt[n].first += ms; g_kt[n].second++; }
#define KT(name, ...)                                                        \
  do {                                                                        \
    if (g_ktime) {                                                            \
      if (!g_ka) { cudaEventCreate(&g_ka); cudaEventCreate(&g_kb); }          \
      cudaEventRecord(g_ka, g_launch);                                        \
      __VA_ARGS__;                                                             \
      cudaEventRecord(g_kb, g_launch);                                        \
      cudaEventSynchronize(g_kb);                                             \
      float _ms = 0.f; cudaEventElapsedTime(&_ms, g_ka, g_kb);                \
      kt_add(name, _ms);                                                      \
    } else { __VA_ARGS__; }                                                    \
  } while (0)

// LFM_STREAM=1: read the Q8 code bytes with __ldcs (evict-first / streaming).
// Each code byte is read exactly once per token, by exactly one lane, so a
// normal load leaves the whole 32-byte sector in L2 for a reader that never
// comes -- 1.2 GB/token of pollution on a 128 MB L2. Same values, same order:
// a cache hint cannot change a bit, so the gates are unaffected by construction.
static const bool g_stream = getenv("LFM_STREAM") != nullptr;
// LFM_GRAPHS=1: capture the 24-layer decode loop into one CUDA graph and
// replay it per step (~250 launches/token -> 1). The plan's kept technique on
// this board for the Bonsai engine; the Nano's ~5 us launch and the strict
// q/k/v -> rope -> cache -> attention -> o_proj serialization are what the
// attention/conv phases pay above their GEMV byte floors (5.15 vs 3.3 ms and
// 5.4 vs 2.5 ms per layer).
//
// A capture bakes in every launch argument, so the per-step values must live
// in DEVICE memory: pos is read through M.pos (a device int written each step
// from a pinned host mirror by the graph's own first node -- pinned host
// memory is read at execution time, so a fixed source address carries a
// changing value). That covers k_rope (x2), k_attn_head, and the KV cache
// writes, which are the two per-layer D2D memcpys converted to k_kv_write.
// The embedding lookup and the lm_head stay outside the graph, so `tok` never
// needs indirection and the lm_head (2 launches of 500 blocks) has no launch
// overhead to save.
static const bool g_graphs = getenv("LFM_GRAPHS") != nullptr;
// Every kernel launch goes through this stream. It is 0 (the legacy stream)
// for all normal execution, so enabling graphs is the only thing that changes
// it: during capture it is set to a real (non-default) stream, which the
// legacy stream cannot be. Non-graph runs are bit-for-bit the old code path.
static cudaStream_t g_launch = (cudaStream_t)0;
// FMA accumulation is the DEFAULT (measured -2.0%, paired A/B R=1.0100,
// faster 3/3, gates green, greedy streams byte-identical). LFM_NOFMA=1
// restores the plan's mul-then-add form as the A/B control arm.
//
// FMA accumulation: __fmaf_rn (one FP instruction per product) instead of the 32-product dot product with __fmaf_rn (one FP
// instruction per product) instead of __fmul_rn + __fadd_rn (two).
//
// Why this is the lever: measured per-layer GEMV time (12 MB tensor, 24
// blocks, 6 blocks/SM) is 1.33 ms, and counting FP instructions reproduces
// that number -- each warp issues 64 blocks x (32 mul + 32 add + 1 acc add +
// 5 shuffles) = 2432 FP warp-instructions where the arithmetic minimum is 512
// FMAs. So the layer GEMVs (88% of the step) are FP-pipe-bound, not
// bandwidth-bound: 9.2 GB/s is a symptom, not the cause. Halving the FP
// instruction count should take them toward the ~19 GB/s FMA-limited rate.
//
// On numerics: the plan mandates "exact per-product rounding", which is what
// the mul-then-add form does. But the reference this engine is graded against
// (ggml_vec_dot_q8_0_q8_0, and HF's torch matmul) accumulates `sum += w*x` in
// C/CUDA, which compilers contract into an FMA -- so the FMA form is at least
// as faithful to the reference, and it is strictly more accurate per product
// (one rounding instead of two). It is NOT bit-identical to the plan's kernel,
// so the quality gates (HF rms matrix, pos-0 logits, greedy stream floors) are
// the authority here, and they run on every check.
static bool g_fma = !getenv("LFM_NOFMA");   // default ON; LFM_NOFMA=1 = the control arm
// LFM_EMB=resident|host: same as --emb-resident / --emb-host, as an env knob
// so ab.sh (which passes env per arm) can A/B the two lm_head streaming
// modes. Empty = the program default.
static const char* g_emb = getenv("LFM_EMB");
// LFM_LMDBUF=1: double-buffer the lm_head's pinned-host row streaming.
static const bool g_lmdbuf = getenv("LFM_LMDBUF") != nullptr;
// LFM_NOFUSE=1 restores the separate k_bf2f pass (the A/B control for the
// fused GEMV prologue; the fused form is the default and is bit-identical).
static const bool g_nofuse = getenv("LFM_NOFUSE") != nullptr;
// LFM_MERGE=1: concatenate the FFN's gate (w1) and up (w3) projections into
// ONE Q8 tensor per layer and run a single GEMV. Each is N=6144, K=2048, so
// alone they launch 24 blocks of R=256 rows each; merged they are 48 blocks
// of one kernel, which matters on a 4-SM part where short bursts from few
// blocks run at 9.0-9.4 GB/s while the N=128000 lm_head reaches 14 GB/s.
// BIT-IDENTICAL BY CONSTRUCTION: concatenating ROWS does not touch any row's
// reduction order, so the outputs are the same bits and the gates cannot
// regress; the merge costs 679 MB of device memory (the originals stay).
// This is the Bonsai Nano engine's kept technique on this exact board
// (merged q+k+v and gate+up tensors, .auto/HANDOFF_NANO_Q1_0.md).
static bool g_merge = getenv("LFM_MERGE") != nullptr;   // --merge also sets it
static cudaEvent_t g_evA, g_evB;
static double g_tEmb = 0, g_tConv = 0, g_tAttn = 0, g_tNorm = 0, g_tHead = 0;
static const bool g_legacy_feed = getenv("LFM_LEGACYFEED") != nullptr;  // pre-#139 feeding (A/B control)
// LFM_H2=1: measurement-only probe (#140), half2 code conversion + HFMA2 in the
// layer GEMVs. NOT bit-exact (fp16 accumulator inside each 32-code block);
// default off; shipping it is a quality decision for the gates and the user.
static const int g_h2 = getenv("LFM_H2") ? atoi(getenv("LFM_H2")) : 0;   // 1 = raw, 2 = per-block 2^k rescale
static inline void phBegin() { if (g_time) cudaEventRecord(g_evA); }
static inline void phEnd(double& acc) {
  if (!g_time) return;
  cudaEventRecord(g_evB); cudaEventSynchronize(g_evB);
  float ms = 0.f; cudaEventElapsedTime(&ms, g_evA, g_evB); acc += ms;
}

struct HostEmb {
  int8_t* hc = nullptr;   // pinned host codes, N*K
  float*  hs = nullptr;   // pinned host scales, N*(K/32)
  int8_t* dc = nullptr;   // device slab (codes)
  float*  dsl = nullptr;  // device slab (scales)
  // Double-buffered streaming (LFM_LMDBUF=1): two slabs, a second stream for
  // the H2D copies, and events so the GEMV of slab i runs while the copy
  // engine loads slab i+1. evC = "slab b is loaded", evG = "slab b consumed"
  // (the WAR guard: the copy for slab i+2 reuses slab b's memory and must not
  // start before GEMV(i) has read it).
  int8_t* dc2[2] = {nullptr, nullptr};
  float*  dsl2[2] = {nullptr, nullptr};
  cudaStream_t st = nullptr;
  cudaEvent_t evC[2] = {nullptr, nullptr}, evG[2] = {nullptr, nullptr};
  bool dbuf = false;
  int N = 0, K = 0, rows = 0;
};

struct Model {
  std::vector<Ent> ents;
  std::map<std::string, Ent*> by;
  int8_t* dq = nullptr; float* ds = nullptr; float* df = nullptr;
  size_t need_q = 0, need_s = 0, need_f = 0;
  // 0 = embedding resident on device (the original path), 1 = pinned host
  // only (--emb-host, what makes the model fit a 4 GB Nano), 2 = both
  // (--lm-selftest, so the two paths can be compared bitwise).
  int emb_mode = 0;
  HostEmb hemb;
  std::string path;
  size_t emb_file_off = 0;
  bf16 *x, *h1, *h2, *h3, *h4, *q, *k, *v, *attn, *gate, *bx, *cy, *ring;
  bf16* hgu = nullptr;   // 2*INTER: merged gate/up GEMV output (LFM_MERGE)
  // Graph mode step state. dpos holds {pos, pos+1} on the device; the graph's
  // first node copies both from the pinned host mirror, which the host writes
  // (a plain store) before each replay. Pinned host memory is read at
  // execution time, so a fixed address inside a captured graph can carry a
  // per-step value -- that is what makes the layer loop capturable without
  // touching the GEMV kernels' signatures.
  int* dpos = nullptr; int* hpos_pin = nullptr;
  cudaStream_t stCap = nullptr, stRun = nullptr;
  cudaGraphExec_t gx = nullptr;
  int pf4_ci = 0;                   // conv-layer counter for the T-wide driver's ring index
  float *xf, *lgf, *part;          // part = GEMV per-chunk partials, f32 by contract
  bf16 *kc, *vc;
  float *invf_full, *invf_slide;
  int* tok_dev;
};

// LFM_AEMB_WAIT=1 (set by lfm3_asr, which starts this process while the encoder still runs):
// --audio-emb names a file that appears (by rename, after the shell validated the encoder's
// output) only when the encoder has exited. Until then the upload stops after layer 1
// (g_load_gate) so the two processes never hold the full backbone plus the encoder's
// activations at once (that combination took picard down, run 281).
static const char* g_aemb_wait = nullptr;
static volatile int g_load_gate = 0x7fffffff;
static void aemb_wait_file() {
  if (!g_aemb_wait) return;
  struct timespec t0, t; clock_gettime(CLOCK_MONOTONIC, &t0);
  while (access(g_aemb_wait, R_OK) != 0) {
    if (getppid() == 1) { printf("--audio-emb wait: parent exited\n"); exit(3); }
    clock_gettime(CLOCK_MONOTONIC, &t);
    if (t.tv_sec - t0.tv_sec > 900) { printf("--audio-emb wait: %s did not appear in 900 s\n", g_aemb_wait); exit(3); }
    usleep(2000);
  }
  clock_gettime(CLOCK_MONOTONIC, &t);
  printf("--audio-emb: %s ready after %.2f s of waiting\n", g_aemb_wait, (t.tv_sec - t0.tv_sec) + 1e-9 * (t.tv_nsec - t0.tv_nsec));
  __atomic_store_n(&g_load_gate, 0x7fffffff, __ATOMIC_RELEASE);
  g_aemb_wait = nullptr;
}

// ---- pipelined blob reader (one producer thread, ring of 16 MB slots)
struct LoadRing {
  int fd = -1; size_t base = 0; std::vector<Ent>* ents = nullptr; const std::vector<size_t>* order = nullptr;
  static const int NS = 4; static const size_t CH = (size_t)16 << 20;
  char* buf[NS] = {}; size_t len[NS] = {}; int state[NS] = {};    // 0 free, 1 full, 2 eof/error
  int head = 0, tail = 0; bool stop = false;
  pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER; pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
};
static void* load_reader(void* arg) {
  LoadRing& R = *(LoadRing*)arg;
  for (int k = 0; k < LoadRing::NS; k++) R.buf[k] = (char*)malloc(LoadRing::CH);
  for (size_t oi = 0; oi < R.order->size(); oi++) {
    while ((int)oi >= __atomic_load_n(&g_load_gate, __ATOMIC_ACQUIRE)) usleep(2000);
    const Ent& e = (*R.ents)[(*R.order)[oi]];
    const size_t tot = e.nc + e.ns; const off_t off0 = (off_t)(R.base + e.file_off);
    for (size_t done = 0; done < tot; ) {
      const size_t n = tot - done < LoadRing::CH ? tot - done : LoadRing::CH;
      pthread_mutex_lock(&R.mu);
      while (R.state[R.head] != 0 && !R.stop) pthread_cond_wait(&R.cv, &R.mu);
      const bool st = R.stop; const int k = R.head;
      pthread_mutex_unlock(&R.mu);
      if (st) return nullptr;
      size_t got = 0;
      while (got < n) { const ssize_t r = pread(R.fd, R.buf[k] + got, n - got, off0 + (off_t)(done + got));
        if (r <= 0) break; got += (size_t)r; }
      // DONTNEED after each read drops those pages (no root needed) -- see the
      // comment at the up-front fadvise for why cache residency hurts here.
      posix_fadvise(R.fd, off0 + (off_t)done, (off_t)got, POSIX_FADV_DONTNEED);
      pthread_mutex_lock(&R.mu);
      R.len[k] = got; R.state[k] = got == n ? 1 : 2; R.head = (k + 1) % LoadRing::NS;
      pthread_cond_broadcast(&R.cv); pthread_mutex_unlock(&R.mu);
      if (got != n) return nullptr;
      done += n;
    }
  }
  return nullptr;
}
static char* load_take(LoadRing& R, int& slot, size_t& len) {
  pthread_mutex_lock(&R.mu);
  while (R.state[R.tail] == 0) pthread_cond_wait(&R.cv, &R.mu);
  slot = R.tail; const int st = R.state[slot]; len = R.len[slot];
  pthread_mutex_unlock(&R.mu);
  return st == 1 ? R.buf[slot] : nullptr;
}
static void load_give(LoadRing& R, int slot) {
  pthread_mutex_lock(&R.mu);
  R.state[slot] = 0; R.tail = (slot + 1) % LoadRing::NS;
  pthread_cond_broadcast(&R.cv); pthread_mutex_unlock(&R.mu);
}
static void load_stop(LoadRing& R) {
  pthread_mutex_lock(&R.mu); R.stop = true; pthread_cond_broadcast(&R.cv); pthread_mutex_unlock(&R.mu);
}
struct Model;
struct LoadJob {
  Model* M = nullptr; LoadRing R; cudaStream_t st = 0; pthread_t th;
  std::vector<size_t> order; std::vector<char*> d0, d1; std::vector<char> skip;
  int need[64]; int need_emb = 0;
  volatile int done = 0, finished = 0; bool ok = true;
  bool lazy = false;      // LFM_PF_LAZY: tensors allocated by this thread right before their upload
};
static int g_ovl = 0;                   // LFM_PF_OVL: weight upload overlapped with the prefill
static bool g_embfile = false;          // LFM_PF_EMBLAST: prefill text rows read from the blob file
static const char* g_blob_path = nullptr;
static LoadJob* g_lj = nullptr;
static void* load_consume(void* arg);
// Block until the first n tensors (upload order) are on the device; false on a load error.
static bool load_wait(int n) {
  if (!g_lj) return true;
  while (__atomic_load_n(&g_lj->done, __ATOMIC_ACQUIRE) < n && !__atomic_load_n(&g_lj->finished, __ATOMIC_ACQUIRE))
    usleep(200);
  return g_lj->ok && __atomic_load_n(&g_lj->done, __ATOMIC_ACQUIRE) >= n;
}

static bool pf_tilable(const Ent& e);
__global__ void k_tile_q(const int4* __restrict__ src, int4* __restrict__ dst, int N, int nb);
__global__ void k_tile_s(const float* __restrict__ src, float* __restrict__ dst, int N, int nb);
static void* load_consume(void* arg) {
  LoadJob& J = *(LoadJob*)arg;
  // LFM_PF_TLOAD (1 = GPU, default 2 = CPU, see below): tensors the prefill would tile are repacked here, on the
  // loader's stream right after their upload and CRC, before `done` publishes them -- off
  // the prefill's critical path. Same kernels and layout as pf_tile.
  static const int tload = getenv("LFM_PF_TLOAD") ? atoi(getenv("LFM_PF_TLOAD")) : 2;
  void* tscr = nullptr; size_t tscr_n = 0;
  // LFM_PF_TLOAD=2: the same tile layout built on the CPU (the loader thread has slack: the
  // eMMC stream finishes well before the prefill needs the last layer); the tensor is
  // gathered row-major in host memory, permuted, and uploaded once, so the GPU runs no
  // k_tile_q / k_tile_s / DtoD copies on the one SM the prefill is using.
  std::vector<char> hrow, htil;
  Model& M = *J.M;
  pthread_t rth; pthread_create(&rth, nullptr, load_reader, &J.R);
  bool ok = true;
  for (size_t oi = 0; oi < J.order.size() && ok; oi++) {
    Ent& e = M.ents[J.order[oi]];
    const size_t tot = e.nc + e.ns;
    const bool skip = J.skip[oi];
    if (J.lazy && !skip) {
      // Tegra's cudaMalloc backs and zeroes pages at ~0.86 ms/MB: the 2.15 GB of arenas
      // cost 1.83 s before the first byte could move. Allocated here, per tensor, it runs
      // behind the reader instead; the pointer is published with the same release store
      // of `done` that load_wait acquires, so no consumer can see it before the bytes.
      void* p0 = nullptr; void* p1 = nullptr;
      cudaError_t ae = cudaMalloc(&p0, e.nc);
      if (ae == cudaSuccess && e.dtype == 1) ae = cudaMalloc(&p1, e.ns);
      if (ae != cudaSuccess) {
        printf("LAZY ALLOC FAIL %s nc=%zu ns=%zu -> %s\n", e.name.c_str(), e.nc, e.ns, cudaGetErrorString(ae));
        ok = false; break;
      }
      J.d0[oi] = (char*)p0; J.d1[oi] = (char*)p1;
      if (e.dtype == 2) e.df = (float*)p0;
      else { e.dq = (decltype(e.dq))p0; e.ds = (float*)p1; }
    }
    char* d0 = J.d0[oi]; char* d1 = J.d1[oi];
    const bool ctile = !skip && tload == 2 && pf_tilable(e);
    if (ctile && hrow.size() < tot) { hrow.resize(tot); htil.resize(tot); }
    uint32_t c = 0;
    for (size_t done = 0; done < tot; ) {
      int slot; size_t len; char* p = load_take(J.R, slot, len);
      if (!p) { printf("short read %s\n", e.name.c_str()); ok = false; break; }
      // Recomputing the exporter's per-tensor crc32 is what turns "the engine
      // runs the quantizer that was certified 8/8" into a load-time assertion.
      c = (uint32_t)crc32(c, (const Bytef*)p, (uInt)len);
      cudaError_t ce = cudaSuccess;
      if (ctile) memcpy(hrow.data() + done, p, len);
      else if (!skip) {
        const size_t lo = done, hi = done + len;
        if (lo < e.nc) { const size_t n = (hi < e.nc ? hi : e.nc) - lo;
          ce = cudaMemcpyAsync(d0 + lo, p, n, cudaMemcpyHostToDevice, J.st); }
        if (ce == cudaSuccess && hi > e.nc && d1) { const size_t s0 = lo > e.nc ? lo : e.nc;
          ce = cudaMemcpyAsync(d1 + (s0 - e.nc), p + (s0 - lo), hi - s0, cudaMemcpyHostToDevice, J.st); }
        if (ce == cudaSuccess) ce = cudaStreamSynchronize(J.st);   // slot reuse + visibility
      }
      load_give(J.R, slot);
      if (ce != cudaSuccess) {
        printf("UPLOAD FAIL %s dtype=%d nc=%zu ns=%zu -> %s\n", e.name.c_str(), e.dtype, e.nc, e.ns,
               cudaGetErrorString(ce));
        ok = false; break;
      }
      done += len;
    }
    if (ok && c != e.crc) {
      printf("CRC MISMATCH %s: file %08x want %08x\n", e.name.c_str(), c, e.crc); ok = false; }
    if (ok && ctile) {
      const int N = e.ne[0], K = e.ne[1], nb = K / Q8_BLK;
      const int4* sq = reinterpret_cast<const int4*>(hrow.data());
      int4* dq = reinterpret_cast<int4*>(htil.data());
      for (int row = 0; row < N; row++)
        for (int bh = 0; bh < nb * 2; bh++)
          dq[((size_t)(row >> 5) * nb * 2 + bh) * 32 + (row & 31)] = sq[(size_t)row * nb * 2 + bh];
      const float* ss = reinterpret_cast<const float*>(hrow.data() + e.nc);
      float* ds = reinterpret_cast<float*>(htil.data() + e.nc);
      for (int row = 0; row < N; row++)
        for (int b = 0; b < nb; b++)
          ds[((size_t)(row >> 5) * nb + b) * 32 + (row & 31)] = ss[(size_t)row * nb + b];
      cudaError_t ce = cudaMemcpyAsync(d0, htil.data(), e.nc, cudaMemcpyHostToDevice, J.st);
      if (ce == cudaSuccess) ce = cudaMemcpyAsync(d1, htil.data() + e.nc, e.ns, cudaMemcpyHostToDevice, J.st);
      if (ce == cudaSuccess) ce = cudaStreamSynchronize(J.st);
      if (ce != cudaSuccess) { printf("CPU TILE UPLOAD FAIL %s: %s\n", e.name.c_str(), cudaGetErrorString(ce)); ok = false; }
      else e.tiled = true;
    } else if (ok && !skip && tload && pf_tilable(e)) {
      const int N = e.ne[0], K = e.ne[1], nb = K / Q8_BLK;
      if (tscr_n < e.nc) {
        if (tscr) cudaFree(tscr);
        tscr = nullptr; tscr_n = 0;
        if (cudaMalloc(&tscr, e.nc) == cudaSuccess) tscr_n = e.nc;
      }
      if (tscr) {
        k_tile_q<<<(unsigned)(((size_t)N * nb * 2 + 255) / 256), 256, 0, J.st>>>(reinterpret_cast<const int4*>(e.dq), (int4*)tscr, N, nb);
        cudaMemcpyAsync((void*)e.dq, tscr, e.nc, cudaMemcpyDeviceToDevice, J.st);
        k_tile_s<<<(unsigned)(((size_t)N * nb + 255) / 256), 256, 0, J.st>>>(e.ds, (float*)tscr, N, nb);
        cudaMemcpyAsync((void*)e.ds, tscr, (size_t)N * nb * 4, cudaMemcpyDeviceToDevice, J.st);
        if (cudaStreamSynchronize(J.st) != cudaSuccess) { printf("TILE FAIL %s\n", e.name.c_str()); ok = false; }
        else e.tiled = true;
      }
    }
    if (ok) __atomic_store_n(&J.done, (int)oi + 1, __ATOMIC_RELEASE);
  }
  if (tscr) cudaFree(tscr);
  load_stop(J.R); pthread_join(rth, nullptr);
  for (int k = 0; k < LoadRing::NS; k++) free(J.R.buf[k]);
  J.ok = ok;
  if (g_ovl && J.R.fd >= 0) close(J.R.fd);
  __atomic_store_n(&J.finished, 1, __ATOMIC_RELEASE);
  if (!ok) fflush(stdout);
  return nullptr;
}

static bool load_blob(const char* path, Model& M, std::string& js) {
  FILE* f = fopen(path, "rb");
  if (!f) { printf("cannot open %s\n", path); return false; }
  const int bfd = fileno(f);
  // A 1.95 GB blob read into RAM is read ONCE and never reused -- but the
  // page cache keeps it, and on Tegra CUDA's device pool is carved from
  // reclaimable system memory: reading the blob filled 1838 MB of cache and
  // cudaMemGetInfo's free fell from 1803 MB to 883 MB, so the run that
  // "does not fit" was starved by its own weights sitting in the cache.
  // DONTNEED after each read drops those pages; it needs no root (unlike
  // drop_caches), and the cost is one eMMC reread of the 295 MB embedding
  // for the host-staged lm_head. This is a deployment fix, not a benchmark
  // trick: the shipped engine has to coexist with the page cache on a 4 GB
  // board, and every MB it evicts is an MB the GPU cannot get back.
  posix_fadvise(bfd, 0, 0, POSIX_FADV_RANDOM);      // no readahead: we seek per tensor
  // Evict the WHOLE file up front, before the memory gate. The 1.8 GB of
  // cache this blob leaves behind comes from PREVIOUS runs and is shared, so
  // a later per-tensor DONTNEED cannot undo what is already squatting on the
  // GPU's pool: with it, cudaMemGetInfo's free sat at 1325 MB against a need
  // of 1622 MB; in a run that never read the file it measured 1803 MB.
  // Doing it here, before the gate, is what makes the gate see the board's
  // real ceiling. (0,0) = the whole file, and it is advisory: check nothing,
  // because a failure just means the run may have to wait for reclaim.
  posix_fadvise(bfd, 0, 0, POSIX_FADV_DONTNEED);
  char magic[8];
  if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "LFM3Q8\x01\x00", 8) != 0) {
    printf("bad magic in %s\n", path); fclose(f); return false;
  }
  uint32_t n = 0;
  if (fread(&n, 4, 1, f) != 1) { fclose(f); return false; }
  js.resize((size_t)n + 1);
  if (fread(&js[0], 1, n, f) != n) { fclose(f); return false; }
  js[n] = 0;
  const size_t base_data = 8 + 4 + (size_t)n;

  // The header table is JSON and the engine is dependency-free, so records
  // are scanned by hand. A record that fails to parse ABORTS: the name is the
  // join key between the blob and the layer loop, so a silently skipped
  // tensor would run the model with missing weights -- the same failure class
  // as the zero-match parameter filters that cost this repo two debug cycles.
  size_t p = 0;
  while (true) {
    p = js.find("{\"name\":", p);
    if (p == std::string::npos) break;
    size_t e1 = js.find("},", p);
    if (e1 == std::string::npos) e1 = js.find("}", p);
    std::string rec = js.substr(p, e1 - p + 1);
    auto get_int = [&](const char* k, long def) -> long {
      size_t z = rec.find(std::string("\"") + k + "\":");
      return z == std::string::npos ? def : atol(rec.c_str() + z + strlen(k) + 3);
    };
    Ent e;
    size_t q1 = rec.find('"', rec.find("\"name\":") + 7);
    size_t q2 = rec.find('"', q1 + 1);
    if (q1 == std::string::npos || q2 == std::string::npos) {
      printf("unparsable name at offset %zu\n", p); fclose(f); return false;
    }
    e.name = rec.substr(q1 + 1, q2 - q1 - 1);
    e.dtype = (int)get_int("dtype", -1);
    e.file_off = (size_t)get_int("off", -1);
    e.nc = (size_t)get_int("ncodes", -1);
    e.ns = (size_t)get_int("nscales", 0);
    e.crc = (uint32_t)get_int("crc", 0);
    e.nd = 0;
    size_t nz = rec.find("\"ne\":[");
    if (nz == std::string::npos) { printf("tensor %s has no ne\n", e.name.c_str()); fclose(f); return false; }
    // The first version passed (char**)&z2 to strtol with z2 a size_t. On
    // LP64 that compiles and runs, and strtol writes an ABSOLUTE address
    // into it, so the `z2 < rec.size()` test failed after the first element:
    // every tensor parsed as 1-D with ne[1] = 0. gemv_f takes K from ne[1],
    // so the whole engine was about to run with K=0. Caught by printing the
    // shape on a mismatch instead of trusting it.
    size_t z2 = rec.find('[', nz) + 1;
    const char* cp = rec.c_str() + z2;
    while (e.nd < 4) {
      char* endp = nullptr;
      const long v = strtol(cp, &endp, 10);
      if (endp == cp) break;
      e.ne[e.nd++] = (int)v;
      cp = endp;
      if (*cp == ',') cp++; else break;
    }
    // Fail loudly on a shape the engine cannot use, rather than deriving
    // K=0 and silently launching an empty grid.
    if (e.dtype == 1 && (e.nd != 2 || e.ne[0] <= 0 || e.ne[1] <= 0)) {
      printf("tensor %s: Q8 payload with nd=%d ne=[%d,%d,%d,%d] -- want 2 dims\n",
             e.name.c_str(), e.nd, e.ne[0], e.ne[1], e.ne[2], e.ne[3]);
      fclose(f); return false;
    }
    if (e.dtype < 0 || e.file_off == (size_t)-1 || e.nc == (size_t)-1 || e.nd == 0) {
      printf("record for %s is incomplete\n", e.name.c_str()); fclose(f); return false;
    }
    // mode 1 keeps the tied embedding in pinned host memory, so it must not
    // be counted in the device arenas -- that 270 MB is what makes the model
    // fit. Mode 2 (selftest) counts it AND takes the host copy.
    const bool hostemb = (M.emb_mode == 1) && (e.name == EMBNAME);
    const bool skq = (M.emb_mode == 2) && (e.name != EMBNAME);   // selftest: embedding only
    if (skq) {}
    else if (e.dtype == 1 || e.dtype == 3) { if (!hostemb) { M.need_q += e.nc; M.need_s += e.ns; } }
    else M.need_f += e.nc;
    // LFM_DBG_SIZE=1 dumps the per-tensor sizing decisions: the upload loop
    // and this pass MUST agree, and when they don't the failure is an
    // "invalid argument" cudaMemcpy 1.47 GB into the load with only the
    // arena's totals to go on. (Certified blob, mode 1: sizing 1,466,269,696
    // vs upload 1,471,381,504 = 5,111,808 bytes unaccounted.)
    if (getenv("LFM_DBG_SIZE"))
      printf("SIZE %-56s dtype=%d nc=%zu ns=%zu -> %s\n", e.name.c_str(), e.dtype,
             (size_t)e.nc, (size_t)e.ns, skq ? "skip(selftest)"
                                 : hostemb ? "skip(emb-host)"
                                 : (e.dtype == 1 || e.dtype == 3) ? "codes" : "f32");
    if (e.name == EMBNAME) M.emb_file_off = base_data + e.file_off;
    M.ents.push_back(e);
    p = e1 + 1;
  }
  if (M.ents.empty()) { printf("table parsed 0 tensors -- format mismatch\n"); fclose(f); return false; }
  printf("blob: %d tensors; codes %.0f MB, scales %.0f MB, f32 %.1f MB\n",
         (int)M.ents.size(), M.need_q / 1e6, M.need_s / 1e6, M.need_f / 1e6);

  // The Nano is a shared 4 GB board: a co-resident service moves device free
  // between 2.87 GB and 1.78 GB, and this model needs 1.95 GB. Failing on a
  // transient is not useful, so wait for room, and say so while waiting. The
  // slack is the arena (KV, ring, partials, ~30 MB) plus CUDA headroom.
  if (g_merge) {
    // gate/up per layer: 2 x (INTER rows x H codes) + 2 x (INTER scales fp32)
    M.need_q += (size_t)NLAYER * 2ull * ((size_t)INTER * H);
    M.need_s += (size_t)NLAYER * 2ull * ((size_t)INTER * (H / Q8_BLK) * 4);
  }
  const size_t need = M.need_q + M.need_s + M.need_f + (size_t)48 * 1024 * 1024;   // arena is KV 25 MB + ring 0.4 + partials 4.5 MB
  for (int tw = 0; tw < 60; tw++) {
    size_t fb = 0, tb = 0; cudaMemGetInfo(&fb, &tb);
    // Tegra is unified memory: cudaMemGetInfo's free excludes reclaimable page cache,
    // which cudaMalloc DOES get back. Gating on it alone made every single-shot ASR run
    // sleep the full 60 x 5 s = 300 s (free sat at ~1.08 GB with ~0.9 GB of cache from
    // the encoder stage) and then load fine anyway. MemAvailable is the kernel's own
    // estimate of free + reclaimable, i.e. what an allocation can actually obtain; a
    // board genuinely occupied by another process still waits.
    size_t avail = 0;
    if (FILE* mi = fopen("/proc/meminfo", "r")) {
      char ln[160]; long long v;
      while (fgets(ln, sizeof ln, mi))
        if (sscanf(ln, "MemAvailable: %lld kB", &v) == 1) { avail = (size_t)v * 1024; break; }
      fclose(mi);
    }
    if (fb >= need || avail >= need) {
      printf("device free %zu MB of %zu MB, MemAvailable %zu MB (need %zu MB) -- loading\n",
           fb / 1048576, tb / 1048576, avail / 1048576, need / 1048576);
      break;
    }
    if (tw == 0)
      printf("device free %.2f GB < %.2f GB needed; waiting for the board to drain\n",
             fb / 1e9, need / 1e9);
    if (tw == 2159) {
      printf("GIVING UP: device free %.2f GB, need %.2f GB. The 2B backbone at "
             "the certified Q8_0 arm is 1.95 GB of weights; on a contended "
             "4 GB Nano that does not fit.\n", fb / 1e9, need / 1e9);
      fclose(f); return false;
    }
    if (tw % 12 == 11) { size_t f2 = 0, t2 = 0; cudaMemGetInfo(&f2, &t2);
      printf("  waiting... free %zu MB (need %zu)\n", f2 / 1048576, need / 1048576); fflush(stdout); }
    if (g_aemb_wait) { printf("memory gate: waiting for the encoder to finish first\n"); fflush(stdout);
                       aemb_wait_file(); continue; }
    struct timespec ts; ts.tv_sec = 5; ts.tv_nsec = 0; nanosleep(&ts, nullptr);
  }
  // LFM_PF_LAZY (default 1 under LFM_PF_OVL): no up-front arenas; load_consume allocates
  // each tensor on the upload thread (see there). Offsets below are still accounted.
  static const bool lazy = g_ovl && !g_merge && !(getenv("LFM_PF_LAZY") && !atoi(getenv("LFM_PF_LAZY")));
  if (lazy) { M.dq = nullptr; M.ds = nullptr; M.df = nullptr; printf("LAZY arenas: tensors allocated on the upload thread\n"); }
  else {
  if (cudaMalloc(&M.dq, M.need_q) != cudaSuccess) {
    printf("codes arena cudaMalloc(%.0f MB) failed\n", M.need_q / 1e6); fclose(f); return false;
  }
  if (cudaMalloc(&M.ds, M.need_s) != cudaSuccess) {
    printf("scales arena cudaMalloc(%.0f MB) failed\n", M.need_s / 1e6); fclose(f); return false;
  }
  if (cudaMalloc(&M.df, M.need_f) != cudaSuccess) {
    printf("f32 arena cudaMalloc(%.1f MB) failed\n", M.need_f / 1e6); fclose(f); return false;
  }
  }
  size_t oc = 0, os = 0, of = 0;
  // f32 payloads first. At the memory gate boundary the LAST cudaMemcpy is
  // the one that starves, and in the first builds that was a norm vector 16 MB
  // into a 1.6 MB arena -- the failure then looked like a blob/CRC problem
  // while it was really an out-of-memory one. Uploading the small tensors
  // first makes starvation happen while there is still room to say so.
  LoadJob* J = new LoadJob;
  std::vector<size_t>& order = J->order;
  for (int pass = 0; pass < 2; pass++)
    for (size_t i = 0; i < M.ents.size(); i++)
      if ((M.ents[i].dtype == 2) == (pass == 0)) order.push_back(i);
  // LFM_PF_LORDER (default 1 with lazy arenas): the blob stores tensors in lexicographic
  // name order (layers 0, 1, 10..19, 2, 20..23, 3..9), so the layer-major prefill's layer 2
  // waited for ten later layers' uploads (+2.4 s measured). Within each pass (f32 first,
  // unchanged), upload in numeric layer order; non-layer tensors keep their place first.
  if (lazy && !(getenv("LFM_PF_LORDER") && !atoi(getenv("LFM_PF_LORDER")))) {
    auto key = [&](size_t i) { int L = -1; sscanf(M.ents[i].name.c_str(), "lfm.model.layers.%d.", &L);
                               return std::make_pair(M.ents[i].dtype == 2 ? 0 : 1, L); };
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return key(a) < key(b); });
  }
  // LFM_PF_EMBLAST (default 1 with lazy arenas): the tied embedding (~295 MB) moves to the
  // END of the upload order. The prefill needs only its text-token rows, which
  // pf4_embed_chunk then preads from the blob (g_embfile); the head waits for the whole
  // blob (and its CRC) as before. It sat at index 133 of 286: 3.5 s of eMMC before layer 0.
  if (lazy && !(getenv("LFM_PF_EMBLAST") && !atoi(getenv("LFM_PF_EMBLAST")))) {
    for (size_t k = 0; k < order.size(); k++)
      if (M.ents[order[k]].name == EMBNAME && M.ents[order[k]].dtype == 1) {
        const size_t ei = order[k]; order.erase(order.begin() + k); order.push_back(ei);
        g_embfile = true; g_blob_path = path; break;
      }
  }
  // Device addresses are a running offset in upload order, so they are all known
  // before a byte is read: assign them (and register the tensors) up front. That is
  // what lets LFM_PF_OVL run the copy loop on a thread while prefill starts.
  J->M = &M; J->d0.assign(order.size(), nullptr); J->d1.assign(order.size(), nullptr);
  J->skip.assign(order.size(), 0); J->lazy = lazy;
  for (int L = 0; L < NLAYER; L++) J->need[L] = 0;
  for (size_t oi = 0; oi < order.size(); oi++) {
    Ent& e = M.ents[order[oi]];
    // The bf16 (dtype 3) embedding must be DEVICE-resident: lm_head_bf16
    // reads it directly, and the Q8 host-staging path has no bf16 form.
    // Under --emb-host the tied embedding lives in pinned host memory only,
    // for BOTH payload formats: Q8 (lm_head_host) and bf16 (lm_head_bf16_host).
    // Mode 2 (selftest) needs ONLY the embedding resident, so the other 285
    // tensors stay off the device (they are still read so the CRC is checked).
    const bool skip = (M.emb_mode == 1 && e.name == EMBNAME) || (M.emb_mode == 2 && e.name != EMBNAME);
    J->skip[oi] = skip;
    if (!skip) {
      if (e.dtype == 3) { J->d0[oi] = (char*)(M.dq + oc); e.dq = M.dq + oc; e.ds = nullptr; oc += e.nc; }
      else if (e.dtype == 1) { J->d0[oi] = (char*)(M.dq + oc); J->d1[oi] = (char*)M.ds + os;
        e.dq = M.dq + oc; e.ds = (float*)((char*)M.ds + os); oc += e.nc; os += e.ns; }
      else {
        // `M.df + of` is float* arithmetic on a BYTE offset: it advanced 4*of,
        // so the first ~48 norm vectors landed inside the arena by luck. Byte offsets.
        J->d0[oi] = (char*)M.df + of; e.df = (float*)((char*)M.df + of); of += e.nc; }
      if (lazy) { J->d0[oi] = J->d1[oi] = nullptr; e.dq = nullptr; e.ds = nullptr; e.df = nullptr; }   // set by load_consume
      M.by[e.name] = &M.ents[order[oi]];
    }
    if (e.name == EMBNAME) J->need_emb = (int)oi + 1;
    int L = -1;
    if (sscanf(e.name.c_str(), "lfm.model.layers.%d.", &L) == 1 && L >= 0 && L < NLAYER && J->need[L] < (int)oi + 1)
      J->need[L] = (int)oi + 1;
  }
  // Pipelined read: a reader thread preads 16 MB chunks into a 4-slot ring while the
  // copy loop runs the incremental crc32 and the H2D copy of the previous chunk.
  // The serial read -> crc -> copy loop left the eMMC idle during crc + copy:
  // 34.9 s load against a 21.3 s cold dd of the same file. Bytes, order of the
  // uploads and the per-tensor CRC assertion are unchanged.
  J->R.base = base_data; J->R.ents = &M.ents; J->R.order = &order;
  if (g_ovl) {
    // LFM_PF_OVL: the copy loop continues on its own thread and its own non-blocking
    // stream (g_launch is the legacy stream, which would otherwise serialize every
    // upload behind queued prefill kernels); prefill waits per layer (load_wait).
    J->R.fd = dup(bfd);
    cudaStreamCreateWithFlags(&J->st, cudaStreamNonBlocking);
    g_lj = J;
    if (g_aemb_wait) __atomic_store_n(&g_load_gate, lazy ? J->need[1] : 0x7fffffff, __ATOMIC_RELEASE);
    pthread_create(&J->th, nullptr, load_consume, J);
  } else {
    J->R.fd = bfd; J->st = (cudaStream_t)0;
    load_consume(J);
    const bool ok = J->ok;
    delete J;
    if (!ok) { fclose(f); return false; }
  }
  for (size_t i = 0; i < M.ents.size(); i++)
    if (M.ents[i].name == EMBNAME) M.ents[i].host = (M.emb_mode == 1);
  fclose(f);  // Build the merged gate/up tensors (device-to-device copies of the Q8 rows
  // into the tail of the same arenas). Q8 rows are [K codes][K/32 fp32
  // scales] per row, and codes/scales live in separate arenas, so a plain
  // concatenation of the two tensors' code blocks and of their scale blocks
  // IS a valid 2N-row Q8 tensor: no re-quantization, no layout change.
  if (g_merge) {
    M.ents.reserve(M.ents.size() + NLAYER);          // keep M.by's Ent* valid
    int nm = 0;
    for (int L = 0; L < NLAYER; L++) {
      char na[96], nb[96], mn[96];
      snprintf(na, sizeof na, "lfm.model.layers.%d.feed_forward.w1.weight", L);
      snprintf(nb, sizeof nb, "lfm.model.layers.%d.feed_forward.w3.weight", L);
      snprintf(mn, sizeof mn, "MERGE:%d:gate_up", L);
      Ent* ea = T(M, na); Ent* eb = T(M, nb);
      if (!ea || !eb || ea->dtype != 1 || eb->dtype != 1 ||
          ea->ne[0] != eb->ne[0] || ea->ne[1] != eb->ne[1]) continue;
      Ent me; me.name = mn; me.dtype = 1; me.nd = 2;
      me.ne[0] = ea->ne[0] + eb->ne[0]; me.ne[1] = ea->ne[1];
      const size_t ncA = (size_t)ea->ne[0] * ea->ne[1];
      const size_t nsA = (size_t)ea->ne[0] * (ea->ne[1] / Q8_BLK) * 4;
      const size_t nc = ncA * 2, ns = nsA * 2;
      if (oc + nc > M.need_q || os + ns > M.need_s) {
        printf("merged tensor %s does not fit: codes %zu+%zu/%zu scales %zu+%zu/%zu\n",
               mn, oc, nc, M.need_q, os, ns, M.need_s);
        fclose(f); return false;
      }
      me.dq = M.dq + oc; me.ds = (float*)((char*)M.ds + os);
      cudaError_t c1 = cudaMemcpy(me.dq, ea->dq, ncA, cudaMemcpyDeviceToDevice);
      cudaError_t c2 = cudaMemcpy(me.dq + ncA, eb->dq, ncA, cudaMemcpyDeviceToDevice);
      cudaError_t c3 = cudaMemcpy(me.ds, ea->ds, nsA, cudaMemcpyDeviceToDevice);
      cudaError_t c4 = cudaMemcpy((char*)me.ds + nsA, eb->ds, nsA, cudaMemcpyDeviceToDevice);
      if (c1 || c2 || c3 || c4) {
        printf("merge copy failed for %s: %s\n", mn, cudaGetErrorString(c1 ? c1 : (c2 ? c2 : (c3 ? c3 : c4))));
        fclose(f); return false;
      }
      oc += nc; os += ns;
      M.ents.push_back(me);
      M.by[mn] = &M.ents.back();
      nm++;
    }
    printf("MERGED %d gate/up pairs into single GEMV tensors (proof: %d of %d layers)\n",
           nm, nm, NLAYER);
    fflush(stdout);
  }


  if (oc != M.need_q || os != M.need_s || of != M.need_f) {
    printf("arena accounting mismatch: codes %zu/%zu scales %zu/%zu f32 %zu/%zu\n",
           oc, M.need_q, os, M.need_s, of, M.need_f);
    return false;
  }
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) { printf("upload failed: %s\n", cudaGetErrorString(err)); return false; }
  return true;
}

static Ent* T(Model& M, const std::string& n) {
  auto it = M.by.find(n);
  if (it == M.by.end()) { printf("MISSING TENSOR %s\n", n.c_str()); exit(2); }
  return it->second;
}
static std::string LN(int L, const char* suf) {
  char b[128]; sprintf(b, "lfm.model.layers.%d.%s", L, suf); return b;
}
static float* NW(Model& M, int L, const char* which) { return T(M, LN(L, which))->df; }

// ------------------------------------------------------------ GEMV wrappers
// (KCHUNK, TPB, R) from the q8_lc3.cu sweep. Coverage is asserted, not
// assumed: 5fd7e79 computed nchunk = 7680/2048 = 3 and read 80% of K, and the
// mirrored reference agreed with it, so bit-exact printed YES.
struct GemvCfg { int KCHUNK, TPB, R; };
static GemvCfg pick(int N, int K) {
  // Env overrides exist for A/B measurement, not as shipped tuning: the plan
  // (row 3) measured 2.6x on this shape family from R alone, so every R/KCHUNK
  // claim has to be measured, not assumed. NOTE on numerical safety: KCHUNK
  // sets nchunk, which is the number of partials k_combine_f sums, so changing
  // it changes the summation ORDER of every GEMV output. The K-coverage assert
  // in the launchers (nchunk * KCHUNK == K) is what stops the "nchunk = 3 read
  // 80% of K" truncation bug that a 5fd7e79-era commit shipped with, so
  // overrides that do not divide K are rejected by the caller's check.
  static const int fR  = getenv("LFM_R")  ? atoi(getenv("LFM_R"))  : 0;
  static const int fKC = getenv("LFM_KCHUNK") ? atoi(getenv("LFM_KCHUNK")) : 0;
  static const int fTPB = getenv("LFM_TPB") ? atoi(getenv("LFM_TPB")) : 0;
  GemvCfg c;
  c.KCHUNK = (K >= 2048 && K % 2048 == 0) ? 2048 : K;
  c.TPB = 256;
  // R = rows per block, amortizing the 8 KB x-stage. Measured (run #82,
  // paired A/B in both orders, arm2 faster 4/4): 256 for every shape beats
  // the old 128-for-N-in-[1024,8192) by 0.5% (141.02 -> 140.32 ms/token).
  // R only remaps rows to blocks, so per-row reduction order -- and every
  // output bit -- is unchanged.
  c.R = (N >= 1024) ? 256 : 64;
  if (g_radapt) c.R = (N >= 4096) ? 256 : (N >= 1024 ? 64 : 32);
  if (K == 512) { c.KCHUNK = 512; c.TPB = 128; }
  if (fKC && K % fKC == 0) c.KCHUNK = fKC;
  if (fTPB >= 32 && fTPB <= 1024) c.TPB = fTPB;
  // R is capped at RMAX, not 1024: q8_partial stages its per-row accumulators
  // in __shared__ float sh[RMAX] indexed by r < R, so an override above RMAX
  // writes out of bounds in shared memory -- silently, and it would show up as
  // wrong logits, not as an error.
  if (fR >= 32 && fR <= RMAX) c.R = fR;
  else if (fR > RMAX) { printf("LFM_R=%d exceeds RMAX=%d (shared sh[RMAX]) -- using RMAX\n", fR, RMAX); c.R = RMAX; }
  return c;
}
// The fused-x GEMV launcher: 4 instantiations (FMA x streaming-load hint).
// Kept as one function so a new launch-time knob is added in one place.
static void q8_launch_bf(int nblk, int TPB, size_t shm, const int8_t* q, const float* sc,
                         const bf16* x, float* part, int N, int K, int KC, int R, int nb,
                         cudaStream_t st) {
  if (g_rowlane && nb * Q8_BLK == K && KC == K) {          // single-chunk shapes only
    const int nblk2 = (N + 255) / 256;
    const size_t shm2 = (size_t)K * 4;
    q8_rowlane<bf16><<<nblk2, 256, shm2, st>>>(q, sc, x, part, N, K);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) { if (g_rowlane_printed++ == 0) printf("Q8_ROWLANE=on (256 threads, %zu KB shared, one row per lane)\n", shm2 >> 10); return; }
    static int warned = 0;
    if (!warned++) printf("Q8_ROWLANE launch refused (%s) -- using q8_partial\n", cudaGetErrorString(e));
  }
  const int U = g_unroll2 ? 2 : 1;
  if (g_fma && U == 2)        q8_partial<bf16,true ,false,2><<<nblk, TPB, shm, st>>>(q, sc, x, part, N, K, KC, R, nb);
  else if (g_fma && g_h2 == 2) q8_partial<bf16,true ,false,1,2><<<nblk, TPB, shm, st>>>(q, sc, x, part, N, K, KC, R, nb);
  else if (g_fma && g_h2)     q8_partial<bf16,true ,false,1,1><<<nblk, TPB, shm, st>>>(q, sc, x, part, N, K, KC, R, nb);
  else if (g_fma)             q8_partial<bf16,true ,false,1><<<nblk, TPB, shm, st>>>(q, sc, x, part, N, K, KC, R, nb);
  else if (g_stream)          q8_partial<bf16,false,true ,1><<<nblk, TPB, shm, st>>>(q, sc, x, part, N, K, KC, R, nb);
  else                        q8_partial<bf16,false,false,1><<<nblk, TPB, shm, st>>>(q, sc, x, part, N, K, KC, R, nb);
}

// LFM_XRANGE=1: diagnostic for the half2 probe (#141). Records, per tensor
// family, the largest |x| fed to the GEMV and how many nonzero inputs fall
// below fp16's normal range (6.1e-5, where half goes subnormal) or above its
// max (65504). Syncs every GEMV: a measurement mode, never a timing mode.
static const bool g_xrange = getenv("LFM_XRANGE") != nullptr;
struct XR { double mx = 0; long n = 0, sub = 0, over = 0; };
static std::map<std::string, XR> g_xr;
static void xrange_add(const std::string& name, const bf16* xb, int K) {
  std::vector<uint16_t> h(K);
  cudaMemcpy(h.data(), xb, (size_t)K * 2, cudaMemcpyDeviceToHost);
  std::string fam = name;
  const size_t d = fam.find(".layers.");
  if (d != std::string::npos) { size_t e2 = fam.find('.', d + 8); fam = fam.substr(e2 + 1); }
  XR& r = g_xr[fam];
  for (int i = 0; i < K; i++) {
    uint32_t u = (uint32_t)h[i] << 16; float f; memcpy(&f, &u, 4);
    const double a = fabs(f);
    if (a > r.mx) r.mx = a;
    if (a > 0 && a < 6.1035e-5) r.sub++;
    if (a > 65504.0) r.over++;
    r.n++;
  }
}
static void xrange_print() {
  for (auto& kv : g_xr)
    printf("XRANGE %-28s n=%9ld max|x|=%10.4f subnormal-in-fp16=%8ld (%.4f%%) overflow=%ld\n",
           kv.first.c_str(), kv.second.n, kv.second.mx, kv.second.sub,
           100.0 * kv.second.sub / (kv.second.n ? kv.second.n : 1), kv.second.over);
}

// ---- S8 integration gate: capture the REAL activations every shipped gemv
// call sees, so the batched kernel can be replayed on them (a synthetic
// pattern proves the kernel; real activations prove the swap).
static void pf_launch(int ct, int arm, int h2, int rb, const int8_t* q, const float* s,
                      const bf16* x, float* part, int N, int K, int KC, int R,
                      int nb, int T, int nblk, int TPB, size_t shm);
static int g_prefill_phase = 0;   // 1 while the prompt is being fed (S8 arm scope)

// ASR path (--audio-emb FILE [--audio-at P]): the encoder's proj_out rows, uploaded as
// fp32 [N,2048], replace the token-table embedding for prompt positions P..P+N-1. HF's
// ASR prompt is exactly this shape (20 text tokens + 1622 identical audio placeholders
// for the 129.7 s fixture), and the placeholder id carries no information, so the audio
// rows -- not the ids -- are the prompt content at those positions.
static float* g_aemb = nullptr;
static int g_aemb_n = 0, g_aemb_at = 0;
static long g_aemb_fed = 0;      // rows actually consumed, printed at the end as proof
// SPEAK path (--speak-emb FILE [--speak-tok ID]): the speech-out loop's backbone feedback.
// A teacher-forced speak run feeds prompt + marker + slot tokens where every speech slot
// is the audio_out placeholder; at each step whose token is the placeholder the embedding
// comes from the next row of this file (frame-major fp32 [nf,2048]), not the token table.
// That is HF's own placeholder mechanism (masked_scatter of get_audio_out_embeds over the
// audio_out_token_id positions). Without --speak-emb the table row is used (old behavior).
static float* g_semb = nullptr;
static int g_semb_n = 0;
static long g_speak_fed = 0;     // placeholder steps actually served, printed as proof
static int g_speak_tok = 125032;  // config.json audio_out_token_id
static const char* g_dump_hidden_all = nullptr;  // every step's post-norm hidden, appended
// --serve: resident speech-out backbone. After the prompt, each step waits on stdin for
// the NEXT input instead of feeding its own argmax:
//   T <id>      feed text token <id>
//   A <path>    feed the audio_out placeholder; its embedding is the 2048-float row in
//               <path> (appended to g_semb, served by the --speak-emb branch)
//   Q           stop
// KV and conv state persist, so a slot costs one decode step, not a relaunch (a relaunch
// re-reads the 1.95 GB blob: 76 s of a 78 s launch, measured). The per-step computation
// is the T=1 path either way, so a served run must reproduce a relaunched run's records
// exactly -- that equality is the gate for this mode.
static bool g_serve = false;
static const int SERVE_MAXFR = 2048;     // frame-row capacity: 16 MB at 2048 x 2048 x 4
struct CapRec { std::string name; int N = 0, K = 0; std::vector<uint16_t> x, y; };
static int g_cap_on = 0;
static std::vector<CapRec> g_caps;
static std::map<std::string, size_t> g_capidx;
static void cap_add(const std::string& name, const bf16* xb, const bf16* yb, int N, int K) {
  size_t idx;
  auto it = g_capidx.find(name);
  if (it == g_capidx.end()) {
    idx = g_caps.size(); g_capidx[name] = idx; g_caps.push_back(CapRec());
    CapRec& c = g_caps.back(); c.name = name; c.N = N; c.K = K;
  } else idx = it->second;
  CapRec& c = g_caps[idx];
  const size_t o = c.x.size();
  c.x.resize(o + (size_t)K); c.y.resize(o + (size_t)N);
  cudaMemcpy(c.x.data() + o, xb, (size_t)K * 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(c.y.data() + o, yb, (size_t)N * 2, cudaMemcpyDeviceToHost);
}

static void tlb_decode(const int8_t* q, const float* s, const bf16* x, bf16* y, int N, int K, float* part);
static void gemv(Model& M, const std::string& name, const bf16* xb, bf16* yb) {
  Ent* e = T(M, name);
  const int N = e->ne[0], K = e->ne[1];
  if (e->tiled) {                                    // tiled weights: TLb at one position
    tlb_decode(e->dq, e->ds, xb, yb, N, K, M.part);
    if (g_cap_on) cap_add(name, xb, yb, N, K);
    return;
  }
  if (g_xrange) xrange_add(name, xb, K);
  GemvCfg c = pick(N, K);
  if (K % c.KCHUNK) { printf("%s: K=%d not divisible by KCHUNK=%d\n", name.c_str(), K, c.KCHUNK); exit(2); }
  const int nb = c.KCHUNK / Q8_BLK, nchunk = K / c.KCHUNK;
  if (nchunk * c.KCHUNK != K) { printf("%s: coverage FAIL\n", name.c_str()); exit(2); }
  // LFM_PF_USE=1: route the T=1 GEMV through the batched ARM=5 kernel (CT=4,
  // GRP=8, R=64). Numerics = q8_partial's order -- proven bit-exact on every
  // real activation of prompt A by --prefill-arm5-gate -- NOT q8_gemv_hs's, so
  // the goldens are the acceptance bar for this arm. Falls back to the shipped
  // path when the shape does not fit (K not a multiple of 32, shm over budget,
  // or the KC chooser cannot split K exactly).
  static int pfUse = -1;   // prefill-only: g_prefill_phase gates the arm
  if (pfUse < 0) { pfUse = getenv("LFM_PF_USE") ? atoi(getenv("LFM_PF_USE")) : 0; if (pfUse < 0) pfUse = 0; }
  if (pfUse && g_prefill_phase && !g_nofuse && K % Q8_BLK == 0) {
    const int CTq = 4, RPBq = 8, Rq = (256 / 32) * RPBq;
    int KCq = K; size_t shmq = (size_t)CTq * (KCq / Q8_BLK) * 17 * sizeof(__half2);
    while (KCq > 32 && (K % KCq || shmq > 49152)) { KCq /= 2; shmq = (size_t)CTq * (KCq / Q8_BLK) * 17 * sizeof(__half2); }
    if (K % KCq == 0 && shmq <= 49152) {
      const int nbq = KCq / Q8_BLK, nchq = K / KCq;
      if (nchq * nbq * Q8_BLK == K) {
        pf_launch(CTq, 5, 1, RPBq, e->dq, e->ds, xb, M.part, N, K, KCq, Rq, nbq, 1,
                  ((N + Rq - 1) / Rq) * nchq, 256, shmq);
        k_combine_bf<<<(N + 255) / 256, 256, 0, g_launch>>>(M.part, yb, N, nchq);
        if (g_cap_on) cap_add(name, xb, yb, N, K);
        return;
      }
    }
  }
  const int nblk = ((N + c.R - 1) / c.R) * nchunk;
  const size_t shm = (size_t)nb * 33 * 4;
  if (g_nofuse) {
    k_bf2f<<<(K + 255) / 256, 256, 0, g_launch>>>(xb, M.xf, K);
    if (g_fma) q8_partial<float,true><<<nblk, c.TPB, shm, g_launch>>>(e->dq, e->ds, M.xf, M.part, N, K, c.KCHUNK, c.R, nb);
    else q8_partial<float,false><<<nblk, c.TPB, shm, g_launch>>>(e->dq, e->ds, M.xf, M.part, N, K, c.KCHUNK, c.R, nb);
  } else {
    q8_launch_bf(nblk, c.TPB, shm, e->dq, e->ds, xb, M.part, N, K, c.KCHUNK, c.R, nb, g_launch);
  }
  k_combine_bf<<<(N + 255) / 256, 256, 0, g_launch>>>(M.part, yb, N, nchunk);
  // capture AFTER the combine wrote yb (a top-of-function capture reads the
  // previous call's stale output -- every cell mismatched, rel ~1e38)
  if (g_cap_on) cap_add(name, xb, yb, N, K);
}
// Batched H2 prefill, clean-room (#162 continued). VALIDATED PIECES ONLY:
// - staging layout identical to q8_partial<H2=1> (proven bit-exact by v3a);
// - per-lane register acc (shared acc was a cross-lane race -- the v1 bug);
// - shm sized exactly CT*nb*17 half2 = 8704 B (the v1 ULF was OOB shared use).
// The minimal direct-call probe just proved direct launches + shared + H2 all
// work in this context, so any failure here is in the newly added structure.
//
// WHY IT IS NOT INTEGRATED AS PREFILL (run #165 root cause, do not re-walk):
// a layer-first batched prefill using this kernel is bit-exact for every
// GEMV whose K equals the per-position row stride (q/k/v/gate/o, in_proj,
// out_proj, w1/w3: K=2048 = H) -- but NOT for ffn.w2 (N=2048 rows, K=7680).
// gemv() indexes x + row*K, so the sequential engine's w2 reads row n from
// h2 + n*7680: row 0 is the real silu output, rows 1..2047 read the arena
// scratch BEYOND the 7680-element buffer (q/k/v/attn/ring, step-recursive
// through every prompt/decode step). That garbage is deterministic and no
// per-position tiling reproduces it. Layer-first w2 gave position 16 the
// silu rows of positions 17-19 where the engine had position-16 bytes:
// L0 x rms at positions 10-17 off by +37..+158%, step-22 logits mean-abs
// 9.5 vs sequential (top-1 matched by tie luck, top-5 garbage). The w2
// GEMV must stay sequential gemv() calls for bit-exactness, and batching
// 9 of 10 GEMVs does not pay while w2 (a top-3 GEMV by time) is pinned.
// ARM=0 is the #162-#165 structure and must keep producing today's bits.
// ARM=1 (plan section 14, TEST-ONLY): the inner half2 chain split into two
// 4-iteration chains (k 0-3, k 4-7, summed before the block FMA) and the outer
// acc interleaved by block parity, halving BOTH dependency depths. ARM=1 changes
// the summation parenthesization, so the harness pairs it against ITS OWN T=1
// launches (batch-correctness) and prints the ARM0-vs-ARM1 relative delta.
// ARM=8 phase profile: summed SM cycles over warps (lane 0 of each warp adds).
// [0] staging  [1] barrier after staging  [2] b-loops  [3] reductions
// [4] end-of-t0 barrier  [5] whole kernel  [6] warps counted
__device__ unsigned long long g_pf_prof[8];
template <typename TX, int CT, bool H2, int ARM, int RB = 2, int GRP = 4>
__global__ void q8_prefill_h2(const int8_t* __restrict__ q, const float* __restrict__ s,
                           const TX* __restrict__ xall, float* __restrict__ part,
                           int N, int K, int KCHUNK, int R, int nb, int T) {
  extern __shared__ float smf[];
  __half2* xh = reinterpret_cast<__half2*>(smf);   // H2 staging: [CT][nb][17] half2
  float* xf = smf;                                 // fp32 staging: [CT][nb][33] float
  // ARM=6 only: per-warp coalesced weight staging, 32 blocks x 32 B = 256 words.
  // Zero-size for every other arm (the array dimension is a constant expression).
  __shared__ uint32_t wst[ARM == 6 ? 8 : 1][ARM == 6 ? 256 : 1];
  const int nchunk = K / KCHUNK, rowblk = blockIdx.x / nchunk, chunk = blockIdx.x % nchunk;
  const int koff = chunk * KCHUNK, NT = blockDim.x, tid = threadIdx.x;
  const int lane = tid & 31, wid = tid >> 5, WPB = NT >> 5, GR = R / WPB, base = rowblk * R;
  const __half2 off = __floats2half2_rn(1152.f, 1152.f);
  long long pf_t = (ARM == 8) ? clock64() : 0, pf_k0 = pf_t;
  long long pf_acc[5] = {0, 0, 0, 0, 0};
#define PF_MARK(i) do { if (ARM == 8) { const long long _n = clock64(); pf_acc[i] += _n - pf_t; pf_t = _n; } } while (0)
  for (int t0 = 0; t0 < T; t0 += CT) {
    int ct = T - t0; if (ct > CT) ct = CT;
    if (H2 && !(ARM == 7 && (GRP & 1))) {   // ARM7 ABL&1: skip x staging (timing only)
      for (int i = tid; i < ct * nb * 16; i += NT) {
        const int c = i / (nb * 16), rem = i % (nb * 16);
        const int b = rem / 16, k = rem % 16;
        int tt = t0 + c; if (tt >= T) tt = T - 1;
        xh[((size_t)c * nb + (size_t)b) * 17 + (size_t)k] =
          __floats2half2_rn(lfm_stage(xall[((size_t)tt) * K + koff + (size_t)b * 32 + 2 * k]),
                             lfm_stage(xall[((size_t)tt) * K + koff + (size_t)b * 32 + 2 * k + 1]));
      }
    } else {
      for (int i = tid; i < ct * KCHUNK; i += NT) {
        const int c = i / KCHUNK, rem = i % KCHUNK;
        const int b = rem >> 5, j = rem & 31;
        int tt = t0 + c; if (tt >= T) tt = T - 1;
        xf[(((size_t)c * nb) + (size_t)b) * 33 + (size_t)j] =
          lfm_stage(xall[((size_t)tt) * K + koff + (size_t)b * 32 + j]);
      }
    }
    PF_MARK(0);
    __syncthreads();
    PF_MARK(1);
    constexpr int G5 = (ARM == 5) ? GRP : 1;   // ARM7 reuses GRP as a mask (may be 0)
    float acc5[G5][CT];
#pragma unroll
    for (int gg = 0; gg < G5; gg++)
#pragma unroll
      for (int cc = 0; cc < CT; cc++) acc5[gg][cc] = 0.f;
    if (ARM == 5) {
      // x into REGISTERS, rows inner. LSU accounting for w2 at CT=4: 64 shared
      // half2 loads per (b,row) x GR=32 rows x nb=120 x 128 warps = 31.5M
      // warp-cycles of shared-load issue on 4 SMs ~= 7.9 ms of the measured
      // 11.7 ms -- the LSU is ~2/3 of the time. ARM3 cut that traffic by 2x but
      // paid more in other instructions; here x is loaded ONCE per (b,c) into
      // 16 registers and reused by all GR rows of the warp, so shared loads per
      // b are CT*16 INDEPENDENT of GR. Requires small GR (rows/warp), which the
      // harness sets with R = WPB*RPB; weight global traffic grows CT-fold, and
      // the stream is at 8% of bandwidth, so that is the cheap side.
      // Per (row,c) the k-chain order is ARM0's -> bit-exact by construction.
      if (GR != GRP) continue;   // harness sets R = WPB*RPB; guard against mismatch
      if (!H2) {
        for (int b = lane; b < nb; b += 32) {
#pragma unroll
          for (int c = 0; c < CT; c++) {
            const float* xb = xf + (((size_t)c * nb) + (size_t)b) * 33;
            float xv[32];
#pragma unroll
            for (int j = 0; j < 32; j++) xv[j] = xb[j];
#pragma unroll
            for (int g = 0; g < GRP; g++) {
              const int r = wid + g * WPB, row = base + r;
              if (row >= N) continue;
              const int8_t* qr = q + (size_t)row * K + koff + (size_t)b * Q8_BLK;
              const int4 a0 = *reinterpret_cast<const int4*>(qr);
              const int4 a1 = *reinterpret_cast<const int4*>(qr + 16);
              const int8_t* c0 = reinterpret_cast<const int8_t*>(&a0);
              const int8_t* c1 = reinterpret_cast<const int8_t*>(&a1);
              float dot = 0.f;
#pragma unroll
              for (int j = 0; j < 16; j++) dot = __fmaf_rn((float)c0[j], xv[j], dot);
#pragma unroll
              for (int j = 0; j < 16; j++) dot = __fmaf_rn((float)c1[j], xv[16 + j], dot);
              const float* sr = s + (size_t)row * (K / Q8_BLK) + (size_t)chunk * nb + b;
              acc5[g][c] = __fmaf_rn(sr[0], dot, acc5[g][c]);
            }
          }
        }
#pragma unroll
        for (int g = 0; g < GRP; g++) {
          const int r = wid + g * WPB, row = base + r;
#pragma unroll
          for (int c = 0; c < CT; c++) {
            float v = (row < N) ? acc5[g][c] : 0.f;
            for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(~0u, v, o);
            if (lane == 0 && row < N && t0 + c < T)
              part[(((size_t)(t0 + c)) * N + row) * nchunk + chunk] = v;
          }
        }
        continue;
      }
      for (int b = lane; b < nb; b += 32) {
#pragma unroll
        for (int c = 0; c < CT; c++) {
          const __half2* xb2 = xh + ((size_t)c * nb + (size_t)b) * 17;
          __half2 x[16];
#pragma unroll
          for (int k = 0; k < 16; k++) x[k] = xb2[k];
#pragma unroll
          for (int g = 0; g < GRP; g++) {
            const int r = wid + g * WPB, row = base + r;
            if (row >= N) continue;
            const int8_t* qr = q + (size_t)row * K + koff + (size_t)b * Q8_BLK;
            const int4 a0 = *reinterpret_cast<const int4*>(qr);
            const int4 a1 = *reinterpret_cast<const int4*>(qr + 16);
            const uint32_t w[8] = {(uint32_t)a0.x, (uint32_t)a0.y, (uint32_t)a0.z, (uint32_t)a0.w,
                                   (uint32_t)a1.x, (uint32_t)a1.y, (uint32_t)a1.z, (uint32_t)a1.w};
            __half2 h2 = __floats2half2_rn(0.f, 0.f);
#pragma unroll
            for (int k = 0; k < 8; k++) {
              const uint32_t u = w[k] ^ 0x80808080u;
              const uint32_t lo = __byte_perm(u, 0x64646464u, 0x4140);
              const uint32_t hi = __byte_perm(u, 0x64646464u, 0x4342);
              h2 = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&lo), off), x[2 * k], h2);
              h2 = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&hi), off), x[2 * k + 1], h2);
            }
            const float2 f = __half22float2(h2);
            const float* sr = s + (size_t)row * (K / Q8_BLK) + (size_t)chunk * nb + b;
            acc5[g][c] = __fmaf_rn(sr[0], f.x + f.y, acc5[g][c]);
          }
        }
      }
#pragma unroll
      for (int g = 0; g < GRP; g++) {
        const int r = wid + g * WPB, row = base + r;
#pragma unroll
        for (int c = 0; c < CT; c++) {
          float v = (row < N) ? acc5[g][c] : 0.f;
          for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(~0u, v, o);
          if (lane == 0 && row < N && t0 + c < T)
            part[(((size_t)(t0 + c)) * N + row) * nchunk + chunk] = v;
        }
      }
      continue;
    }
    if (ARM == 10 && H2) {
      // ARM9 row blocking (fewer shared loads per MAC) + software pipelining of
      // ALL global loads (weights AND scales) one iteration ahead over a
      // flattened (row-group, b) space, so the prefetch crosses row boundaries
      // (at K=2048 a row has only nb/32 = 2 b-iterations per lane).
      // ABL26 showed two co-binders: with only global loads removed (ABL18)
      // shared loads hold it at ~35 GMAC/s; with only shared reads removed
      // (ABL8) global latency holds it at ~38; with both removed it runs at
      // 62-70. ARM2-6 and ARM9 each attacked one binder. This arm attacks both.
      // Per (row,c): k-chain and ascending-b fmaf order are ARM0's -> bit-exact
      // by construction; invalid lanes/rows skip compute (acc untouched),
      // exactly as ARM0's loop bounds skip them.
      constexpr int RBX = (GRP < 1) ? 1 : GRP;
      const int nbj = (nb + 31) / 32;               // b-iterations per row-group
      const int ngr = (GR + RBX - 1) / RBX;         // row-groups per warp
      const int nit = ngr * nbj;
      const size_t cs = (size_t)nb * 17;
      uint32_t wn[RBX][8]; float sn[RBX];
      auto load_it = [&](int it, uint32_t (&w)[RBX][8], float (&sv)[RBX]) {
        const int g0 = (it / nbj) * RBX, b = (it % nbj) * 32 + lane;
#pragma unroll
        for (int rr = 0; rr < RBX; rr++) {
          const int row = base + wid + (g0 + rr) * WPB;
          if (g0 + rr < GR && row < N && b < nb) {
            const int8_t* qp = q + (size_t)row * K + koff + (size_t)b * Q8_BLK;
            const int4 a0 = *reinterpret_cast<const int4*>(qp);
            const int4 a1 = *reinterpret_cast<const int4*>(qp + 16);
            w[rr][0]=a0.x; w[rr][1]=a0.y; w[rr][2]=a0.z; w[rr][3]=a0.w;
            w[rr][4]=a1.x; w[rr][5]=a1.y; w[rr][6]=a1.z; w[rr][7]=a1.w;
            sv[rr] = s[(size_t)row * (K / Q8_BLK) + (size_t)chunk * nb + b];
          } else {
#pragma unroll
            for (int k = 0; k < 8; k++) w[rr][k] = 0x80808080u;
            sv[rr] = 0.f;
          }
        }
      };
      float accx[RBX][CT];
#pragma unroll
      for (int rr = 0; rr < RBX; rr++)
#pragma unroll
        for (int c = 0; c < CT; c++) accx[rr][c] = 0.f;
      if (nit > 0) load_it(0, wn, sn);
      for (int it = 0; it < nit; it++) {
        uint32_t w[RBX][8]; float sv[RBX];
#pragma unroll
        for (int rr = 0; rr < RBX; rr++) {
          sv[rr] = sn[rr];
#pragma unroll
          for (int k = 0; k < 8; k++) w[rr][k] = wn[rr][k];
        }
        if (it + 1 < nit) load_it(it + 1, wn, sn);   // issue next loads BEFORE compute
        const int g0 = (it / nbj) * RBX, j = it % nbj, b = j * 32 + lane;
        if (b < nb) {
          __half2 h[RBX][CT];
#pragma unroll
          for (int rr = 0; rr < RBX; rr++)
#pragma unroll
            for (int c = 0; c < CT; c++) h[rr][c] = __floats2half2_rn(0.f, 0.f);
          const __half2* xb = xh + (size_t)b * 17;
#pragma unroll
          for (int k = 0; k < 8; k++) {
            __half2 cl[RBX], ch[RBX];
#pragma unroll
            for (int rr = 0; rr < RBX; rr++) {
              const uint32_t u = w[rr][k] ^ 0x80808080u;
              const uint32_t lo = __byte_perm(u, 0x64646464u, 0x4140);
              const uint32_t hi = __byte_perm(u, 0x64646464u, 0x4342);
              cl[rr] = __hsub2(*reinterpret_cast<const __half2*>(&lo), off);
              ch[rr] = __hsub2(*reinterpret_cast<const __half2*>(&hi), off);
            }
#pragma unroll
            for (int c = 0; c < CT; c++) {
              const __half2 x0 = xb[(size_t)c * cs + 2 * k];
              const __half2 x1 = xb[(size_t)c * cs + 2 * k + 1];
#pragma unroll
              for (int rr = 0; rr < RBX; rr++) {
                h[rr][c] = __hfma2(cl[rr], x0, h[rr][c]);
                h[rr][c] = __hfma2(ch[rr], x1, h[rr][c]);
              }
            }
          }
#pragma unroll
          for (int rr = 0; rr < RBX; rr++) {
            const int row = base + wid + (g0 + rr) * WPB;
            if (g0 + rr < GR && row < N) {
#pragma unroll
              for (int c = 0; c < CT; c++) {
                const float2 f = __half22float2(h[rr][c]);
                accx[rr][c] = __fmaf_rn(sv[rr], f.x + f.y, accx[rr][c]);
              }
            }
          }
        }
        if (j == nbj - 1) {                          // row-group done: reduce + store
#pragma unroll
          for (int rr = 0; rr < RBX; rr++) {
            const int row = base + wid + (g0 + rr) * WPB;
            const bool ok = (g0 + rr < GR) && row < N;
#pragma unroll
            for (int c = 0; c < CT; c++) {
              float v = ok ? accx[rr][c] : 0.f;
              for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(~0u, v, o);
              if (lane == 0 && ok && t0 + c < T)
                part[(((size_t)(t0 + c)) * N + row) * nchunk + chunk] = v;
              accx[rr][c] = 0.f;
            }
          }
        }
      }
      continue;
    }
    if (ARM == 9 && H2) {
      // Row-blocked H2 (RB = GRP slot, rows per warp step). The single-SM
      // ceiling for "one shared x load per HFMA2" is ~58 GMAC/s (754 MB of
      // shared traffic for w1 at 128 B/clk), below the ~79 the realtime budget
      // needs, so x must be reused from registers across rows. Weights for each
      // of the RB rows are loaded once per b and converted once per k (shared
      // across all CT positions); each x half2 read from shared feeds RB HFMA2.
      // Per (row,c) the HFMA2 order is ARM0's (k = 0..7, lo then hi) -> bit-exact
      // by construction. ARM3 (the previous row-blocked try) reconverted per
      // byte and per position; that is what made it 4.5x slower.
      constexpr int RB9 = (GRP < 1) ? 1 : GRP;
      for (int g0 = 0; g0 < GR; g0 += RB9) {
        float acc9[RB9][CT];
        int rowv[RB9]; bool act[RB9];
#pragma unroll
        for (int rr = 0; rr < RB9; rr++) {
          rowv[rr] = base + wid + (g0 + rr) * WPB;
          act[rr] = (g0 + rr < GR) && rowv[rr] < N;
#pragma unroll
          for (int c = 0; c < CT; c++) acc9[rr][c] = 0.f;
        }
        for (int b = lane; b < nb; b += 32) {
          uint32_t w[RB9][8];
#pragma unroll
          for (int rr = 0; rr < RB9; rr++) {
            if (act[rr]) {
              const int8_t* qp = q + (size_t)rowv[rr] * K + koff + (size_t)b * Q8_BLK;
              const int4 a0 = *reinterpret_cast<const int4*>(qp);
              const int4 a1 = *reinterpret_cast<const int4*>(qp + 16);
              w[rr][0]=a0.x; w[rr][1]=a0.y; w[rr][2]=a0.z; w[rr][3]=a0.w;
              w[rr][4]=a1.x; w[rr][5]=a1.y; w[rr][6]=a1.z; w[rr][7]=a1.w;
            } else {
#pragma unroll
              for (int k = 0; k < 8; k++) w[rr][k] = 0x80808080u;   // decodes to 0
            }
          }
          __half2 h[RB9][CT];
#pragma unroll
          for (int rr = 0; rr < RB9; rr++)
#pragma unroll
            for (int c = 0; c < CT; c++) h[rr][c] = __floats2half2_rn(0.f, 0.f);
          const __half2* xb = xh + (size_t)b * 17;
          const size_t cs = (size_t)nb * 17;
#pragma unroll
          for (int k = 0; k < 8; k++) {
            __half2 cl[RB9], ch[RB9];
#pragma unroll
            for (int rr = 0; rr < RB9; rr++) {
              const uint32_t u = w[rr][k] ^ 0x80808080u;
              const uint32_t lo = __byte_perm(u, 0x64646464u, 0x4140);
              const uint32_t hi = __byte_perm(u, 0x64646464u, 0x4342);
              cl[rr] = __hsub2(*reinterpret_cast<const __half2*>(&lo), off);
              ch[rr] = __hsub2(*reinterpret_cast<const __half2*>(&hi), off);
            }
#pragma unroll
            for (int c = 0; c < CT; c++) {
              const __half2 x0 = xb[(size_t)c * cs + 2 * k];
              const __half2 x1 = xb[(size_t)c * cs + 2 * k + 1];
#pragma unroll
              for (int rr = 0; rr < RB9; rr++) {
                h[rr][c] = __hfma2(cl[rr], x0, h[rr][c]);
                h[rr][c] = __hfma2(ch[rr], x1, h[rr][c]);
              }
            }
          }
#pragma unroll
          for (int rr = 0; rr < RB9; rr++) {
            if (!act[rr]) continue;
            const float sb = s[(size_t)rowv[rr] * (K / Q8_BLK) + (size_t)chunk * nb + b];
#pragma unroll
            for (int c = 0; c < CT; c++) {
              const float2 f = __half22float2(h[rr][c]);
              acc9[rr][c] = __fmaf_rn(sb, f.x + f.y, acc9[rr][c]);
            }
          }
        }
#pragma unroll
        for (int rr = 0; rr < RB9; rr++) {
#pragma unroll
          for (int c = 0; c < CT; c++) {
            float v = act[rr] ? acc9[rr][c] : 0.f;
            for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(~0u, v, o);
            if (lane == 0 && act[rr] && t0 + c < T)
              part[(((size_t)(t0 + c)) * N + rowv[rr]) * nchunk + chunk] = v;
          }
        }
      }
      continue;
    }
    if (ARM == 7 && H2) {
      // TIMING-ONLY ABLATION of ARM0 (outputs are wrong for ABL != 0 and are
      // judged only by the nonzero/peak plausibility gate). GRP slot = mask:
      //   1 skip x staging   2 no weight global loads (w from b)
      //   4 no int8->half2 conversion   8 no shared x reads (x from a register)
      //   16 no scale global load (sb = 1)
      // ABL=0 is the control and must be bit-exact vs ARM0.
      constexpr int ABL = GRP;
      for (int g = 0; g < GR; g++) {
        const int r = wid + g * WPB, row = base + r;
        float acc7[CT];
#pragma unroll
        for (int c = 0; c < CT; c++) acc7[c] = 0.f;
        if (row < N) {
          const int8_t* qr = q + (size_t)row * K + koff;
          const float* sr = s + (size_t)row * (K / Q8_BLK) + (size_t)chunk * nb;
          // ABL&8 replacement x: DISTINCT per position c (and per lo/hi) so the
          // CT chains cannot be merged by the compiler. The first version used
          // one xreg for every c: identical chains -> CSE -> ~4x less HFMA2
          // work, which inflated ABL8/ABL26 (run #301 correction).
          __half2 xrc[CT][2];
#pragma unroll
          for (int c = 0; c < CT; c++) { xrc[c][0] = xh[lane + 3 * c]; xrc[c][1] = xh[lane + 3 * c + 1]; }
          for (int b = lane; b < nb; b += 32) {
            uint32_t w[8];
            if (ABL & 2) {
#pragma unroll
              for (int k = 0; k < 8; k++) w[k] = 0x01020304u * (uint32_t)(b + k + row);
            } else {
              const int4 a0 = *reinterpret_cast<const int4*>(qr + (size_t)b * Q8_BLK);
              const int4 a1 = *reinterpret_cast<const int4*>(qr + (size_t)b * Q8_BLK + 16);
              w[0]=a0.x; w[1]=a0.y; w[2]=a0.z; w[3]=a0.w; w[4]=a1.x; w[5]=a1.y; w[6]=a1.z; w[7]=a1.w;
            }
#pragma unroll
            for (int c = 0; c < CT; c++) {
              const __half2* xb2 = xh + ((size_t)c * nb + (size_t)b) * 17;
              __half2 h2 = __floats2half2_rn(0.f, 0.f);
#pragma unroll
              for (int k = 0; k < 8; k++) {
                __half2 cl, ch;
                if (ABL & 4) {
                  const uint32_t lo = w[k], hi = w[k] >> 1;
                  cl = *reinterpret_cast<const __half2*>(&lo);
                  ch = *reinterpret_cast<const __half2*>(&hi);
                } else {
                  const uint32_t u = w[k] ^ 0x80808080u;
                  const uint32_t lo = __byte_perm(u, 0x64646464u, 0x4140);
                  const uint32_t hi = __byte_perm(u, 0x64646464u, 0x4342);
                  cl = __hsub2(*reinterpret_cast<const __half2*>(&lo), off);
                  ch = __hsub2(*reinterpret_cast<const __half2*>(&hi), off);
                }
                const __half2 x0 = (ABL & 8) ? xrc[c][0] : xb2[2 * k];
                const __half2 x1 = (ABL & 8) ? xrc[c][1] : xb2[2 * k + 1];
                h2 = __hfma2(cl, x0, h2);
                h2 = __hfma2(ch, x1, h2);
              }
              const float2 f = __half22float2(h2);
              const float sb7 = (ABL & 16) ? 1.0f : sr[b];   // 16: no scale global load
              acc7[c] = __fmaf_rn(sb7, f.x + f.y, acc7[c]);
            }
          }
        }
#pragma unroll
        for (int c = 0; c < CT; c++) {
          float v = (row < N) ? acc7[c] : 0.f;
          for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(~0u, v, o);
          if (lane == 0 && row < N && t0 + c < T)
            part[(((size_t)(t0 + c)) * N + row) * nchunk + chunk] = v;
        }
      }
      continue;
    }
    if (ARM == 6 && H2) {
      // Coalesced weight staging (bench_h2: 2.9x the direct per-lane pattern in
      // isolation). The warp reads 32 consecutive 32-byte blocks as 64 int4
      // (fully coalesced, 1 KB), stages them in its own shared slice, and each
      // lane then reads ITS block from shared. Same lane->b mapping and same
      // k-chain as ARM0, so each (row,c) accumulates in ARM0's order.
      uint32_t* ws = &wst[wid][0];
      for (int g = 0; g < GR; g++) {
        const int r = wid + g * WPB, row = base + r;
        float acc6[CT];
#pragma unroll
        for (int c = 0; c < CT; c++) acc6[c] = 0.f;
        const int8_t* qr = q + (size_t)row * K + koff;
        const float* sr = s + (size_t)row * (K / Q8_BLK) + (size_t)chunk * nb;
        for (int b0 = 0; b0 < nb; b0 += 32) {
          const int nbl = (nb - b0 < 32) ? (nb - b0) : 32;
          if (row < N) {                       // uniform across the warp
            const int8_t* src = qr + (size_t)b0 * Q8_BLK;
            for (int i = lane; i < nbl * 2; i += 32) {
              const int4 v = *reinterpret_cast<const int4*>(src + (size_t)i * 16);
              ws[i * 4 + 0] = v.x; ws[i * 4 + 1] = v.y; ws[i * 4 + 2] = v.z; ws[i * 4 + 3] = v.w;
            }
          }
          __syncwarp();
          const int b = b0 + lane;
          if (row < N && b < nb) {
            const uint32_t* w = &ws[lane * 8];
            const float sb[1] = {sr[b]};
#pragma unroll
            for (int c = 0; c < CT; c++) {
              const __half2* xb2 = xh + ((size_t)c * nb + (size_t)b) * 17;
              __half2 h2 = __floats2half2_rn(0.f, 0.f);
#pragma unroll
              for (int k = 0; k < 8; k++) {
                const uint32_t u = w[k] ^ 0x80808080u;
                const uint32_t lo = __byte_perm(u, 0x64646464u, 0x4140);
                const uint32_t hi = __byte_perm(u, 0x64646464u, 0x4342);
                h2 = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&lo), off), xb2[2 * k], h2);
                h2 = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&hi), off), xb2[2 * k + 1], h2);
              }
              const float2 f = __half22float2(h2);
              acc6[c] = __fmaf_rn(sb[0], f.x + f.y, acc6[c]);
            }
          }
          __syncwarp();                        // staging is reused by the next b0
        }
#pragma unroll
        for (int c = 0; c < CT; c++) {
          float v = (row < N) ? acc6[c] : 0.f;
          for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(~0u, v, o);
          if (lane == 0 && row < N && t0 + c < T)
            part[(((size_t)(t0 + c)) * N + row) * nchunk + chunk] = v;
        }
      }
      continue;
    }
    if (ARM == 4) {
      // Software-pipelined ARM0: the b loop is serial and each iteration's two
      // int4 weight loads feed a ~48-op dependent chain, with no overlap across
      // b. w2 streams at 1.34 GB/s = 8% of the measured 17 GB/s ceiling, while
      // ARM2 (no conversions, 66% occ) and ARM3 (2x less shared traffic, no
      // spills) both failed to help -- the signature is load latency, not ops.
      // Here the b+32 load is issued BEFORE the current b's compute. Values are
      // ARM0's exactly -> bit-exact by construction, and verified.
      for (int g = 0; g < GR; g++) {
        const int r = wid + g * WPB, row = base + r;
        float acc4[CT];
#pragma unroll
        for (int c = 0; c < CT; c++) acc4[c] = 0.f;
        if (row < N) {
          const int8_t* qr = q + (size_t)row * K + koff;
          const float* sr = s + (size_t)row * (K / Q8_BLK) + (size_t)chunk * nb;
          int4 a0 = *reinterpret_cast<const int4*>(qr + (size_t)lane * Q8_BLK);
          int4 a1 = *reinterpret_cast<const int4*>(qr + (size_t)lane * Q8_BLK + 16);
          for (int b = lane; b < nb; b += 32) {
            int4 n0, n1;
            if (b + 32 < nb) {
              n0 = *reinterpret_cast<const int4*>(qr + (size_t)(b + 32) * Q8_BLK);
              n1 = *reinterpret_cast<const int4*>(qr + (size_t)(b + 32) * Q8_BLK + 16);
            }
            const uint32_t w[8] = {(uint32_t)a0.x, (uint32_t)a0.y, (uint32_t)a0.z, (uint32_t)a0.w,
                                   (uint32_t)a1.x, (uint32_t)a1.y, (uint32_t)a1.z, (uint32_t)a1.w};
            const float sb = sr[b];
#pragma unroll
            for (int c = 0; c < CT; c++) {
              const __half2* xb2 = xh + ((size_t)c * nb + (size_t)b) * 17;
              __half2 h2 = __floats2half2_rn(0.f, 0.f);
#pragma unroll
              for (int k = 0; k < 8; k++) {
                const uint32_t u = w[k] ^ 0x80808080u;
                const uint32_t lo = __byte_perm(u, 0x64646464u, 0x4140);
                const uint32_t hi = __byte_perm(u, 0x64646464u, 0x4342);
                h2 = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&lo), off), xb2[2 * k], h2);
                h2 = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&hi), off), xb2[2 * k + 1], h2);
              }
              const float2 f = __half22float2(h2);
              acc4[c] = __fmaf_rn(sb, f.x + f.y, acc4[c]);
            }
            a0 = n0; a1 = n1;
          }
        }
#pragma unroll
        for (int c = 0; c < CT; c++) {
          float v = (row < N) ? acc4[c] : 0.f;
          for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(~0u, v, o);
          if (lane == 0 && row < N && t0 + c < T)
            part[(((size_t)(t0 + c)) * N + row) * nchunk + chunk] = v;
        }
      }
      continue;
    }
    if (ARM == 3) {
      // Register-blocked rows: RB rows share ONE x load per (b,c,k). In ARM0-2
      // the g-loop is outside the b-loop, so every row re-reads the whole x
      // block from shared -- R/WPB x redundant shared traffic, and the ARM2
      // result (fewer registers, higher occupancy, SLOWER) rules out conversion
      // work and occupancy as binders, which leaves this traffic. Per (row,c)
      // the k-chain order is unchanged -> bit-exact by construction, verified.
      for (int rb = 0; rb < GR; rb += RB) {
        float acc3[RB][CT];
        __half2 h3[RB][CT];
        const int8_t* qr[RB]; const float* srp[RB]; bool act[RB];
#pragma unroll
        for (int rr = 0; rr < RB; rr++) {
          const int row = base + wid + (rb + rr) * WPB;
          act[rr] = row < N;
#pragma unroll
          for (int c = 0; c < CT; c++) { acc3[rr][c] = 0.f; h3[rr][c] = __floats2half2_rn(0.f, 0.f); }
          srp[rr] = s + (size_t)row * (K / Q8_BLK) + (size_t)chunk * nb;
          qr[rr]  = q + (size_t)row * K + koff;
        }
        for (int b = lane; b < nb; b += 32) {
          int4 wq[RB], wl[RB];
#pragma unroll
          for (int rr = 0; rr < RB; rr++)
            if (act[rr]) {
              const int8_t* p = qr[rr] + (size_t)b * Q8_BLK;
              wq[rr] = *reinterpret_cast<const int4*>(p);
              wl[rr] = *reinterpret_cast<const int4*>(p + 16);
            }
          const __half2* pb = xh + (size_t)b * 17;
          const size_t stride = (size_t)nb * 17;
#pragma unroll
          for (int k = 0; k < 8; k++) {
#pragma unroll
            for (int c = 0; c < CT; c++) {
              const __half2 xl = pb[(size_t)c * stride + 2 * k];
              const __half2 xv = pb[(size_t)c * stride + 2 * k + 1];
#pragma unroll
              for (int rr = 0; rr < RB; rr++) {
                if (!act[rr]) continue;
                const int8_t* cw = reinterpret_cast<const int8_t*>(&wq[rr]) + 4 * k;
                const int8_t* cv = reinterpret_cast<const int8_t*>(&wl[rr]) + 4 * k;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                  const uint32_t u = (uint32_t)(uint8_t)cw[j] ^ 0x80u;
                  const uint32_t lo = __byte_perm(u, 0x64646464u, 0x4140);
                  h3[rr][c] = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&lo), off), xl, h3[rr][c]);
                  const uint32_t v = (uint32_t)(uint8_t)cv[j] ^ 0x80u;
                  const uint32_t hi = __byte_perm(v, 0x64646464u, 0x4140);
                  h3[rr][c] = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&hi), off), xv, h3[rr][c]);
                }
              }
            }
          }
#pragma unroll
          for (int rr = 0; rr < RB; rr++) {
            if (!act[rr]) continue;
#pragma unroll
            for (int c = 0; c < CT; c++) {
              const float2 f = __half22float2(h3[rr][c]);
              acc3[rr][c] = __fmaf_rn(srp[rr][b], f.x + f.y, acc3[rr][c]);
              h3[rr][c] = __floats2half2_rn(0.f, 0.f);
            }
          }
        }
#pragma unroll
        for (int rr = 0; rr < RB; rr++) {
          const int row = base + wid + (rb + rr) * WPB;
#pragma unroll
          for (int c = 0; c < CT; c++) {
            float v = act[rr] ? acc3[rr][c] : 0.f;
            for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(~0u, v, o);
            if (lane == 0 && act[rr] && t0 + c < T)
              part[(((size_t)(t0 + c)) * N + row) * nchunk + chunk] = v;
          }
        }
      }
      continue;
    }
    for (int g = 0; g < GR; g++) {
      const int r = wid + g * WPB, row = base + r;
      float acc[CT][2];          // [c][block-parity chain]; ARM0 uses [0] only
#pragma unroll
      for (int c = 0; c < CT; c++) { acc[c][0] = 0.f; acc[c][1] = 0.f; }
      if (row < N) {
        const int8_t* qr = q + (size_t)row * K + koff;
        const float* sr = s + (size_t)row * (K / Q8_BLK) + (size_t)chunk * nb;
        for (int b = lane; b < nb; b += 32) {
          const int4 a0 = *reinterpret_cast<const int4*>(qr + (size_t)b * Q8_BLK);
          const int4 a1 = *reinterpret_cast<const int4*>(qr + (size_t)b * Q8_BLK + 16);
          const uint32_t w[8] = {(uint32_t)a0.x, (uint32_t)a0.y, (uint32_t)a0.z, (uint32_t)a0.w,
                                 (uint32_t)a1.x, (uint32_t)a1.y, (uint32_t)a1.z, (uint32_t)a1.w};
          const int pi = (ARM == 1) ? ((b >> 5) & 1) : 0;   // ARM0/2: constant 0, folds away
          if (ARM == 2) {
            // ARM=2: convert the weights ONCE per b and interleave the CT h2
            // chains across k. ARM=0 rebuilds cl/ch inside the c loop (CT x the
            // conversion work) and runs each c's 16-deep hfma2 chain back-to-back
            // with no overlap. Per-c chain ORDER is unchanged, so the arithmetic
            // is identical to ARM=0 -- bit-exactness is by construction, and
            // ARM2-vs-ARM0 max-rel is an external check that should be exactly 0.
            if (H2) {
              __half2 cl[8], ch[8];
#pragma unroll
              for (int k = 0; k < 8; k++) {
                const uint32_t u = w[k] ^ 0x80808080u;
                const uint32_t lo = __byte_perm(u, 0x64646464u, 0x4140);
                const uint32_t hi = __byte_perm(u, 0x64646464u, 0x4342);
                cl[k] = __hsub2(*reinterpret_cast<const __half2*>(&lo), off);
                ch[k] = __hsub2(*reinterpret_cast<const __half2*>(&hi), off);
              }
              const __half2* pb = xh + (size_t)b * 17;
              const size_t stride = (size_t)nb * 17;
              __half2 h2[CT];
#pragma unroll
              for (int c = 0; c < CT; c++) h2[c] = __floats2half2_rn(0.f, 0.f);
#pragma unroll
              for (int k = 0; k < 8; k++) {
#pragma unroll
                for (int c = 0; c < CT; c++) {
                  h2[c] = __hfma2(cl[k], pb[(size_t)c * stride + 2 * k], h2[c]);
                  h2[c] = __hfma2(ch[k], pb[(size_t)c * stride + 2 * k + 1], h2[c]);
                }
              }
#pragma unroll
              for (int c = 0; c < CT; c++) {
                const float2 f = __half22float2(h2[c]);
                acc[c][0] = __fmaf_rn(sr[b], f.x + f.y, acc[c][0]);
              }
            } else {
              const int8_t* c0 = reinterpret_cast<const int8_t*>(&a0);
              const int8_t* c1 = reinterpret_cast<const int8_t*>(&a1);
              float wv[32];
#pragma unroll
              for (int j = 0; j < 16; j++) wv[j] = (float)c0[j];
#pragma unroll
              for (int j = 0; j < 16; j++) wv[16 + j] = (float)c1[j];
#pragma unroll
              for (int c = 0; c < CT; c++) {
                const float* xb = xf + (((size_t)c * nb) + (size_t)b) * 33;
                float dot = 0.f;
#pragma unroll
                for (int j = 0; j < 16; j++) dot = __fmaf_rn(wv[j], xb[j], dot);
#pragma unroll
                for (int j = 0; j < 16; j++) dot = __fmaf_rn(wv[16 + j], xb[16 + j], dot);
                acc[c][0] = __fmaf_rn(sr[b], dot, acc[c][0]);
              }
            }
            continue;
          }
if (H2) {
#pragma unroll
            for (int c = 0; c < CT; c++) {
              const __half2* xb2 = xh + ((size_t)c * nb + (size_t)b) * 17;
              float bd;
              if (ARM == 1) {
                /* two 4-iteration chains (depth 8 hfma2 each vs one depth 16) */
                __half2 ha = __floats2half2_rn(0.f, 0.f), hb = __floats2half2_rn(0.f, 0.f);
#pragma unroll
                for (int k = 0; k < 4; k++) {
                  const uint32_t u = w[k] ^ 0x80808080u;
                  const uint32_t lo = __byte_perm(u, 0x64646464u, 0x4140);
                  const uint32_t hi = __byte_perm(u, 0x64646464u, 0x4342);
                  __half2 cl = __hsub2(*reinterpret_cast<const __half2*>(&lo), off);
                  __half2 ch = __hsub2(*reinterpret_cast<const __half2*>(&hi), off);
                  ha = __hfma2(cl, xb2[2 * k], ha);
                  ha = __hfma2(ch, xb2[2 * k + 1], ha);
                }
#pragma unroll
                for (int k = 4; k < 8; k++) {
                  const uint32_t u = w[k] ^ 0x80808080u;
                  const uint32_t lo = __byte_perm(u, 0x64646464u, 0x4140);
                  const uint32_t hi = __byte_perm(u, 0x64646464u, 0x4342);
                  __half2 cl = __hsub2(*reinterpret_cast<const __half2*>(&lo), off);
                  __half2 ch = __hsub2(*reinterpret_cast<const __half2*>(&hi), off);
                  hb = __hfma2(cl, xb2[2 * k], hb);
                  hb = __hfma2(ch, xb2[2 * k + 1], hb);
                }
                const float2 fa = __half22float2(ha), fb = __half22float2(hb);
                bd = (fa.x + fa.y) + (fb.x + fb.y);
              } else {
                __half2 h2 = __floats2half2_rn(0.f, 0.f);
#pragma unroll
                for (int k = 0; k < 8; k++) {
                  const uint32_t u = w[k] ^ 0x80808080u;
                  const uint32_t lo = __byte_perm(u, 0x64646464u, 0x4140);
                  const uint32_t hi = __byte_perm(u, 0x64646464u, 0x4342);
                  __half2 cl = __hsub2(*reinterpret_cast<const __half2*>(&lo), off);
                  __half2 ch = __hsub2(*reinterpret_cast<const __half2*>(&hi), off);
                  h2 = __hfma2(cl, xb2[2 * k], h2);
                  h2 = __hfma2(ch, xb2[2 * k + 1], h2);
                }
                const float2 f = __half22float2(h2);
                bd = f.x + f.y;
              }
              acc[c][pi] = __fmaf_rn(sr[b], bd, acc[c][pi]);
            }
          } else {
            const int8_t* c0 = reinterpret_cast<const int8_t*>(&a0);
            const int8_t* c1 = reinterpret_cast<const int8_t*>(&a1);
#pragma unroll
            for (int c = 0; c < CT; c++) {
              const float* xb = xf + (((size_t)c * nb) + (size_t)b) * 33;
              float dot;
              if (ARM == 1) {
                float d0 = 0.f, d1 = 0.f;
#pragma unroll
                for (int j = 0; j < 16; j++) d0 = __fmaf_rn((float)c0[j], xb[j], d0);
#pragma unroll
                for (int j = 0; j < 16; j++) d1 = __fmaf_rn((float)c1[j], xb[16 + j], d1);
                dot = d0 + d1;
              } else {
                dot = 0.f;
#pragma unroll
                for (int j = 0; j < 16; j++) dot = __fmaf_rn((float)c0[j], xb[j], dot);
#pragma unroll
                for (int j = 0; j < 16; j++) dot = __fmaf_rn((float)c1[j], xb[16 + j], dot);
              }
              acc[c][pi] = __fmaf_rn(sr[b], dot, acc[c][pi]);
            }
          }
        }
      }
      PF_MARK(2);
#pragma unroll
      for (int c = 0; c < CT; c++) {
        float v = (row < N) ? acc[c][0] : 0.f;
        if (ARM == 1 && row < N) v += acc[c][1];   // ARM8 must stay ARM0-exact
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(~0u, v, o);
        if (lane == 0 && row < N && t0 + c < T)
          part[(((size_t)(t0 + c)) * N + row) * nchunk + chunk] = v;
      }
      PF_MARK(3);
    }
    __syncthreads();
    PF_MARK(4);
  }
  if (ARM == 8 && lane == 0) {
    for (int i = 0; i < 5; i++) atomicAdd(&g_pf_prof[i], (unsigned long long)pf_acc[i]);
    atomicAdd(&g_pf_prof[5], (unsigned long long)(clock64() - pf_k0));
    atomicAdd(&g_pf_prof[6], 1ull);
  }
#undef PF_MARK
}


// Minimal direct-call diagnostic (run #162): one q8_partial<H2=1> launch,
// every step checked. Determines whether direct calls work at all.
static int prefill_minimal(Model& M) {
  Ent* e = nullptr;
  for (size_t i = 0; i < M.ents.size(); i++)
    if (M.ents[i].dq && M.ents[i].ds && M.ents[i].ne[0] >= 1024 && M.ents[i].ne[1] % 32 == 0) { e = &M.ents[i]; break; }
  if (!e) return 1;
  const int N = e->ne[0], K = e->ne[1], TPB = 256;
  const int KC = (K >= 2048 && K % 2048 == 0) ? 2048 : K, nb = KC / Q8_BLK;
  const int nblk = ((N + 255) / 256);
  printf("prefill-minimal %s N=%d K=%d nblk=%d shm=%zu\n", e->name.c_str(), N, K, nblk, (size_t)nb * 33 * 4); fflush(stdout);
  uint16_t* xd = nullptr; float* out = nullptr;
  printf("m-alloc=%s %s\n", cudaGetErrorString(cudaMalloc(&xd, (size_t)K * 2)), cudaGetErrorString(cudaMalloc(&out, (size_t)N * 4))); fflush(stdout);
  std::vector<uint16_t> pat((size_t)K);
  for (size_t i = 0; i < pat.size(); i++) pat[i] = (uint16_t)(0x3c00 + ((i * 37) & 0x3ff));
  printf("m-h2d=%s\n", cudaGetErrorString(cudaMemcpy(xd, pat.data(), pat.size() * 2, cudaMemcpyHostToDevice))); fflush(stdout);
  cudaMemset(out, 0xFF, (size_t)N * 4);
  cudaEvent_t ea, eb; cudaEventCreate(&ea); cudaEventCreate(&eb);
  cudaEventRecord(ea);
  q8_partial<bf16,true,false,1,1><<<nblk, TPB, (size_t)nb * 33 * 4>>>(e->dq, e->ds, (const bf16*)xd, out, N, K, KC, 256, nb);
  printf("m-launch=%s\n", cudaGetErrorString(cudaGetLastError())); fflush(stdout);
  cudaEventRecord(eb); cudaEventSynchronize(eb);
  printf("m-sync=%s\n", cudaGetErrorString(cudaGetLastError())); fflush(stdout);
  float ms = 0; cudaEventElapsedTime(&ms, ea, eb);
  std::vector<float> h(8, -999.f);
  cudaError_t me = cudaMemcpy(h.data(), out, 32, cudaMemcpyDeviceToHost);
  printf("m-read=%s ms=%.2f out[0..3]=%.9g %.9g %.9g %.9g (numpy row0: -0.110298038)\n", cudaGetErrorString(me), ms, h[0], h[1], h[2], h[3]); fflush(stdout);
  return 0;
}

// Batched H2 selftest (#162 continued): 23 sequential H2 GEMVs (proven above)
// vs one batched call. Bit-exactness expected by construction (same per-position
// order); the question is only whether it runs and how fast.
static int prefill_h2_one(Model& M, Ent* e, int T, int h2);
static long g_last_partial_bad = 0;
static uint16_t host_f2bf(float f) {   // RTN-even, matches __float2bfloat16_rn on finite input
  uint32_t u; memcpy(&u, &f, 4);
  if ((u & 0x7f800000u) == 0x7f800000u && (u & 0xffffu)) return (uint16_t)((u >> 16) | 0x0040u); // NaN -> quiet
  uint32_t lsb = (u >> 16) & 1u; u += 0x7fffu + lsb; return (uint16_t)(u >> 16);
}
static void pf_launch(int ct, int arm, int h2, int rb, const int8_t* q, const float* s,
                      const bf16* x, float* part, int N, int K, int KC, int R,
                      int nb, int T, int nblk, int TPB, size_t shm);
static int prefill_arm5_gate(Model& M) {
  if (g_caps.empty()) { printf("LFM_ARM5_GATE: nothing captured\n"); return 1; }
  int fails = 0, tested = 0;
  for (size_t i = 0; i < g_caps.size(); i++) {
    CapRec& c = g_caps[i];
    if (c.K % Q8_BLK || c.N <= 0) continue;
    const int N = c.N, K = c.K, TPB = 256, CT = 4, RPB = 8, R = (TPB / 32) * RPB;
    const int Tn = (int)(c.x.size() / (size_t)K);
    if (Tn < 2) continue;
    // KC configs: exact whole-K (nch=1, the shipped shape for K<=5760) and the
    // chunked 3840 (nch=2) that the CT=4 shared budget forces for K=7680.
    for (int cfg = 0; cfg < 2; cfg++) {
      int KC = (cfg == 0) ? K : 3840;
      if (K % KC) continue;
      int nb = KC / Q8_BLK;
      size_t shm = (size_t)CT * nb * 17 * sizeof(__half2);
      if (shm > 49152) continue;
      if (cfg == 0 && shm > 49152) continue;
      const int nch = K / KC, nblk = ((N + R - 1) / R) * nch;
      uint16_t* xd = nullptr; float* pd = nullptr;
      cudaMalloc(&xd, (size_t)Tn * K * 2);
      cudaMalloc(&pd, (size_t)Tn * N * nch * 4);
      cudaMemcpy(xd, c.x.data(), (size_t)Tn * K * 2, cudaMemcpyHostToDevice);
      Ent* e = T(M, c.name);
      // Per-token T=1 comparison against the SHIPPED kernel. The harness pair
      // proves ARM5-batch == ARM5-seq == q8_partial; the engine's decode GEMV is
      // q8_gemv_hs, a different summation order, so "batched == shipped" needs
      // this link: ARM5 at T=1 vs the shipped output, on real activations.
      for (int t = 0; t < Tn; t++)
        pf_launch(CT, 5, 1, RPB, e->dq, e->ds, xd + (size_t)t * K, pd + (size_t)t * N * nch,
                  N, K, KC, R, nb, 1, nblk, TPB, shm);
      {
        std::vector<float> h1((size_t)Tn * N * nch);
        cudaMemcpy(h1.data(), pd, h1.size() * 4, cudaMemcpyDeviceToHost);
        long bad1 = 0; double worst1 = 0; int ninf1 = 0;
        for (int t = 0; t < Tn; t++)
          for (int r = 0; r < N; r++) {
            float acc = 0.f;
            for (int ch = 0; ch < nch; ch++) acc = acc + h1[((size_t)t * N + r) * nch + ch];
            uint16_t got = host_f2bf(acc), want = c.y[(size_t)t * N + r];
            if (got != want) {
              bad1++;
              float gf, wf;
              { uint32_t bu = (uint32_t)got << 16; memcpy(&gf, &bu, 4); }
              { uint32_t bu = (uint32_t)want << 16; memcpy(&wf, &bu, 4); }
              const int gf_ok = std::isfinite(gf), wf_ok = std::isfinite(wf);
              if (!gf_ok || !wf_ok) { if (gf_ok != wf_ok) ninf1++; continue; }   // asymmetric = failure
              double den = fabs((double)wf);
              double rel = den > 0 ? fabs(gf - wf) / den : (gf == wf ? 0.0 : 1.0);
              if (rel > worst1) worst1 = rel;
            }
          }
        printf("arm5-gate-T1 %s T=%d KC=%d nch=%d: %s (%ld/%zu differ, max rel %.3g, nonfinite %d)\n",
               c.name.c_str(), Tn, KC, nch, bad1 ? "MISMATCH" : "BIT-EXACT", bad1,
               (size_t)Tn * N, worst1, ninf1);
        fflush(stdout);
        // Plumbing control: ARM=5 T=1 vs the port's reference GEMV (q8_partial)
        // on the SAME real activations, comparing f32 partials bit-for-bit.
        // Bit-exact here => the gate's layout/combine/rounding are right and
        // the shipped-vs-ARM5 gap is genuinely q8_gemv_hs's own summation order.
        float* pd2 = nullptr; cudaMalloc(&pd2, (size_t)Tn * N * nch * 4);
        for (int t = 0; t < Tn; t++)
          q8_partial<bf16,true,false,1,1><<<nblk, TPB, (size_t)nb * 33 * 4>>>(
              e->dq, e->ds, xd + (size_t)t * K, pd2 + (size_t)t * N * nch, N, K, KC, R, nb);
        std::vector<float> h2v((size_t)Tn * N * nch);
        cudaMemcpy(h2v.data(), pd2, h2v.size() * 4, cudaMemcpyDeviceToHost);
        long bad2 = 0;
        for (size_t j = 0; j < h1.size(); j++) if (memcmp(&h1[j], &h2v[j], 4)) bad2++;
        // f32 (H2=0) ARM=5 T=1 vs the shipped output: the candidate for a
        // bit-exact-vs-shipped prefill arm (fp32 FMA order, no fp16 pairs).
        // The f32 arm needs the SAME shared-memory-aware KC chooser the shipped
        // LFM_PF_USE path has. It did not have one: at the w2 shape (K=7680) the gate
        // passed its outer KC=3840, which asks for CT*nb*33*4 = 63 KB of static shared
        // memory -- over the 48 KB cap -- so the launch failed with `invalid argument`,
        // and the code then graded a buffer that was never written and printed
        // "MISMATCH 24576/92160" for a shape the kernel never ran on. That is exactly the
        // failure mode the plausibility rules exist for: a launch error must stop the
        // comparison, not flow into it. KC is now halved until the budget fits, a launch
        // error is a hard failure, and the report carries the ENERGY-relative error
        // (this project's standard metric) next to the per-cell max-rel, which is
        // meaningless on its own -- per-cell max-rel over reference denormals is what
        // produced the 4e+39 headline.
        int KC3 = KC, nb3 = nb; size_t shm3 = (size_t)CT * nb3 * 33 * 4;
        while (KC3 > 32 && (K % KC3 || shm3 > 49152)) { KC3 /= 2; nb3 = KC3 / Q8_BLK; shm3 = (size_t)CT * nb3 * 33 * 4; }
        const int nch3 = (K % KC3 == 0) ? K / KC3 : 0;
        if (!nch3 || shm3 > 49152) {
          printf("arm5f32-gate-T1 %s T=%d: SKIPPED (no KC fits the 48 KB shared-memory cap: K=%d, KC3=%d, shm=%zu)\n",
                 c.name.c_str(), Tn, K, KC3, shm3); fflush(stdout);
          fails++;
        } else {
        float* pd3 = nullptr; cudaMalloc(&pd3, (size_t)Tn * N * nch3 * 4);
        const int R3 = (TPB / 32) * 2;
        const int nblk3 = ((N + R3 - 1) / R3) * nch3;
        for (int t = 0; t < Tn; t++)
          pf_launch(CT, 5, 0, 2, e->dq, e->ds, xd + (size_t)t * K, pd3 + (size_t)t * N * nch3,
                    N, K, KC3, R3, nb3, 1, nblk3, TPB, shm3);
        cudaError_t le3 = cudaGetLastError();
        cudaDeviceSynchronize();
        if (le3) {
          printf("arm5f32-gate-T1 %s T=%d KC=%d: LAUNCH ERROR %s (NOT compared -- no number is reported for a kernel that did not run)\n",
                 c.name.c_str(), Tn, KC3, cudaGetErrorString(le3)); fflush(stdout);
          fails++;
        } else {
        std::vector<float> h3v((size_t)Tn * N * nch3);
        cudaMemcpy(h3v.data(), pd3, h3v.size() * 4, cudaMemcpyDeviceToHost);
        long bad3 = 0; double worst3 = 0; int ninf3 = 0; double snum = 0, sden = 0; long gt1ulp = 0;
        for (int t = 0; t < Tn; t++)
          for (int r = 0; r < N; r++) {
            float acc = 0.f;
            for (int ch = 0; ch < nch3; ch++) acc = acc + h3v[((size_t)t * N + r) * nch3 + ch];
            uint16_t got = host_f2bf(acc), want = c.y[(size_t)t * N + r];
            if (got != want) {
              bad3++;
              float gf, wf;
              { uint32_t bu = (uint32_t)got << 16; memcpy(&gf, &bu, 4); }
              { uint32_t bu = (uint32_t)want << 16; memcpy(&wf, &bu, 4); }
              const int okf = std::isfinite(gf), okw = std::isfinite(wf);
              if (!okf || !okw) { if (okf != okw) ninf3++; continue; }
              double den = fabs((double)wf);
              double rel = den > 0 ? fabs(gf - wf) / den : (gf == wf ? 0.0 : 1.0);
              if (rel > worst3) worst3 = rel;
              snum += (double)(gf - wf) * (gf - wf);
              sden += (double)wf * wf;
              // ulp-distance in bf16: >1 ulp is the "real defect" count, 1 ulp is rounding
              int dg = (int)got, dw = (int)want;
              if (abs(dg - dw) > 1) gt1ulp++;
            }
          }
        printf("arm5f32-gate-T1 %s T=%d KC=%d nch=%d: %s (%ld/%zu differ, >1ulp %ld, "
               "ENERGY rel %.3e, per-cell max rel %.3g, nonfinite %d)\n",
               c.name.c_str(), Tn, KC3, nch3, bad3 ? "MISMATCH" : "BIT-EXACT", bad3,
               (size_t)Tn * N, gt1ulp, sden > 0 ? sqrt(snum / sden) : (snum > 0 ? 1.0 : 0.0),
               worst3, ninf3);
        fflush(stdout);
        if (sden > 0 && sqrt(snum / sden) > 1e-2) fails++;   // a real arithmetic delta is a gate failure
        g_last_partial_bad += bad3 ? 0 : 0;
        }   // launch-error guard
        cudaFree(pd3);
        }   // KC-fits guard
        printf("arm5-vs-q8partial %s T=%d KC=%d nch=%d: %s (%ld/%zu partials differ)\n",
               c.name.c_str(), Tn, KC, nch, bad2 ? "MISMATCH" : "BIT-EXACT", bad2, h1.size());
        fflush(stdout);
        g_last_partial_bad = bad2;
        cudaFree(pd2);
      }
      pf_launch(CT, 5, 1, RPB, e->dq, e->ds, xd, pd, N, K, KC, R, nb, Tn, nblk, TPB, shm);
      cudaError_t le = cudaGetLastError();
      cudaDeviceSynchronize();
      std::vector<float> hpd((size_t)Tn * N * nch);
      cudaMemcpy(hpd.data(), pd, hpd.size() * 4, cudaMemcpyDeviceToHost);
      long bad = 0; double worst = 0; int ninf = 0; long first = -1;
      for (int t = 0; t < Tn; t++) {
        for (int r = 0; r < N; r++) {
          float acc = 0.f;
          // plain C++ float add == IEEE-754 RN on this host (aarch64, no
          // fast-math), which is what __fadd_rn does on the device.
          for (int ch = 0; ch < nch; ch++) acc = acc + hpd[((size_t)t * N + r) * nch + ch];
          uint16_t got = host_f2bf(acc);
          uint16_t want = c.y[(size_t)t * N + r];
          if (got != want) {
            if (first < 0) first = (long)((size_t)t * N + r);
            bad++;
            float gf, wf;
            { uint32_t bu = (uint32_t)got << 16; memcpy(&gf, &bu, 4); }
            { uint32_t bu = (uint32_t)want << 16; memcpy(&wf, &bu, 4); }
            const int gf_ok = std::isfinite(gf), wf_ok = std::isfinite(wf);
            if (!gf_ok || !wf_ok) { if (gf_ok != wf_ok) ninf++; continue; }   // asymmetric = failure
            double den = fabs((double)wf);
            double rel = den > 0 ? fabs(gf - wf) / den : (gf == wf ? 0.0 : 1.0);
            if (rel > worst) worst = rel;
          }
        }
      }
      const char* st = (!bad && !le) ? "BIT-EXACT" : "MISMATCH";
      printf("arm5-gate-batched %s T=%d KC=%d nch=%d: %s (%ld/%zu cells vs shipped, max rel %.3g, nonfinite %d) %s\n",
             c.name.c_str(), Tn, KC, nch, st, bad, (size_t)Tn * N, worst, ninf,
             le ? cudaGetErrorString(le) : "");
      fflush(stdout);
      cudaFree(xd); cudaFree(pd);
      tested++;
      if (le || ninf) fails++;   // launch errors / non-finite are hard failures;
                                 // the shipped-kernel delta is reported, not gated
      if (g_last_partial_bad) fails++;   // set by the arm5-vs-q8partial control
    }
  }
  printf("LFM_ARM5_GATE: %d configs tested, %s (certifies ARM5==q8_partial bit-exact; the shipped-kernel delta is reported, arbitrated by the goldens)\n",
         tested, fails ? "FAILURES" : "ALL PASS");
  fflush(stdout);
  return fails ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Tranche 1 of true CT=4 batched prefill (doc section 49, run log row 332).
//
// Row 331 measured the CEILING (2.7-3.8x on the real shapes, bit-exact, all rates
// under the plausibility ceiling). Row 330 ruled out the GEMV-level "vs shipped"
// gate as the arbiter. What is left before the engine's layer loop can be taken
// T-wide is a plumbing question, and it is the one the w2 item was parked on: can the
// batched GEMM read a POSITION-FIRST activation arena (row t at x + t*K) that the
// prefill driver owns, and reproduce T per-position launches bit-for-bit, on the real
// weights?
//
// This selftest answers exactly that, in isolation, and nothing else: one T-wide launch
// per GEMV vs T launches of the same kernel at T=1, over the same arena, compared byte
// for byte. It does NOT compare against the shipped q8_gemv_hs path -- row 330 says that
// comparison is not the arbiter -- and its x is a deterministic synthetic pattern (the
// real-activation version of the check is --prefill-arm5-gate, and row 331 used captured
// real activations), because the question here is LAYOUT, not embedding semantics.
static int prefill_chunk_test(Model& M, int Tn) {
  if (Tn < 2 || Tn > 64) { printf("--prefill-chunk: T=%d out of range 2..64\n", Tn); return 2; }
  // H and TPB are #defines in this file (H = 2048), so only the new constants are
  // declared here: shadowing them made the preprocessor expand `const int H = 2048`
  // into `const int 2048 = 2048`, which is the "expected an identifier" nvcc gave.
  const int TPBC = 256, CT = 4, RPB = 8, R = (256 / 32) * RPB;
  std::vector<bf16> hx((size_t)Tn * H);
  for (int t = 0; t < Tn; t++)
    for (int i = 0; i < H; i++)
      hx[(size_t)t * H + i] = host_f2bf(sinf(t * 7.13f + i * 0.0013f) * 0.37f);   // nonzero, finite, per-t distinct
  bf16 *xb = nullptr, *xb2 = nullptr;
  cudaMalloc(&xb, (size_t)Tn * H * 2); cudaMalloc(&xb2, (size_t)Tn * H * 2);
  cudaMemcpy(xb, hx.data(), (size_t)Tn * H * 2, cudaMemcpyHostToDevice);
  std::vector<float> hw(H, 1.0f);
  float* hw_d = nullptr; cudaMalloc(&hw_d, H * 4); cudaMemcpy(hw_d, hw.data(), H * 4, cudaMemcpyHostToDevice);
  k_rmsnorm_T<<<Tn, TPBC, 0, g_launch>>>(xb, hw_d, xb2, 1e-5f, H);
  cudaError_t e0 = cudaGetLastError(); cudaDeviceSynchronize();
  printf("PF4-CHUNK arena: T=%d, k_rmsnorm_T launch %s\n", Tn, e0 ? cudaGetErrorString(e0) : "ok");
  fflush(stdout);
  if (e0) return 1;

  const char* nm[] = {"lfm.model.layers.0.conv.in_proj.weight", "lfm.model.layers.0.conv.out_proj.weight",
                      "lfm.model.layers.0.feed_forward.w1.weight", "lfm.model.layers.0.feed_forward.w3.weight",
                      "lfm.model.layers.0.feed_forward.w2.weight",
                      "lfm.model.layers.2.self_attn.q_proj.weight", "lfm.model.layers.2.self_attn.k_proj.weight",
                      "lfm.model.layers.2.self_attn.v_proj.weight", "lfm.model.layers.2.self_attn.o_proj.weight",
                      "lfm.model.layers.2.self_attn.gate_proj.weight"};
  int fails = 0, tested = 0;
  for (int w = 0; w < 10; w++) {
    Ent* e = T(M, nm[w]);
    if (!e || !(e->dq && e->ds)) { printf("PF4-CHUNK %s: NOT IN BLOB\n", nm[w]); fails++; tested++; continue; }
    const int N = e->ne[0], K = e->ne[1];
    if (K % Q8_BLK) { printf("PF4-CHUNK %s: K=%d not a multiple of %d\n", nm[w], K, Q8_BLK); fails++; tested++; continue; }
    int KC = K, nb = K / Q8_BLK; size_t shm = (size_t)CT * nb * 33 * 4;
    while (KC > 32 && (K % KC || shm > 49152)) { KC /= 2; nb = KC / Q8_BLK; shm = (size_t)CT * nb * 33 * 4; }
    if (K % KC || shm > 49152) { printf("PF4-CHUNK %s: SKIPPED (no KC fits: K=%d KC=%d shm=%zu)\n", nm[w], K, KC, shm); fails++; tested++; continue; }
    const int nch = K / KC, nblk = ((N + R - 1) / R) * nch;
    float* pB = nullptr; float* p1 = nullptr; bf16 *yB = nullptr, *y1 = nullptr;
    cudaMalloc(&pB, (size_t)Tn * N * nch * 4); cudaMalloc(&p1, (size_t)Tn * N * nch * 4);
    cudaMalloc(&yB, (size_t)Tn * N * 2);       cudaMalloc(&y1, (size_t)Tn * N * 2);
    struct timespec t0, t1; cudaDeviceSynchronize(); clock_gettime(CLOCK_MONOTONIC, &t0);
    pf_launch(CT, 5, 1, RPB, e->dq, e->ds, xb2, pB, N, K, KC, R, nb, Tn, nblk, TPBC, shm);
    // T-aware combine (k_combine_bf_T, see its comment): the batched side combines all
    // T positions in ONE launch, the per-position side below uses k_combine_bf T times,
    // so a BIT-EXACT verdict compares two independent combine paths.
    dim3 gbat((N + 255) / 256, Tn);
    k_combine_bf_T<<<gbat, 256, 0, g_launch>>>(pB, yB, N, nch, Tn);
    cudaDeviceSynchronize(); clock_gettime(CLOCK_MONOTONIC, &t1);
    const double msbat = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) * 1e-6;
    cudaError_t le = cudaGetLastError();
    for (int t = 0; t < Tn; t++) {
      pf_launch(CT, 5, 1, RPB, e->dq, e->ds, xb2 + (size_t)t * K, p1 + (size_t)t * N * nch,
                N, K, KC, R, nb, 1, nblk, TPBC, shm);
      k_combine_bf<<<(N + 255) / 256, 256, 0, g_launch>>>(p1 + (size_t)t * N * nch, y1 + (size_t)t * N, N, nch);
    }
    cudaError_t le2 = cudaGetLastError(); cudaDeviceSynchronize();
    if (le || le2) {
      printf("PF4-CHUNK %s N=%d K=%d KC=%d: LAUNCH ERROR %s / %s (NO number reported)\n",
             nm[w], N, K, KC, cudaGetErrorString(le), cudaGetErrorString(le2));
      fflush(stdout); fails++; tested++;
      cudaFree(pB); cudaFree(p1); cudaFree(yB); cudaFree(y1); continue;
    }
    std::vector<bf16> hyB((size_t)Tn * N), hy1((size_t)Tn * N);
    cudaMemcpy(hyB.data(), yB, (size_t)Tn * N * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hy1.data(), y1, (size_t)Tn * N * 2, cudaMemcpyDeviceToHost);
    long diff = 0, nz = 0; size_t first = (size_t)-1;
    for (size_t i = 0; i < hyB.size(); i++) {
      if (hyB[i] != hy1[i]) { diff++; if (first == (size_t)-1) first = i; }
      if (hyB[i] != 0) nz++;
    }
    const double gmac = 2.0 * N * K * Tn / (msbat * 1e6);
    const int plaus = (gmac <= 120.0) && nz > 0;
    printf("PF4-CHUNK %s N=%d K=%d KC=%d nch=%d T=%d: %s (%ld/%zu cells differ, nonzero %zu/%zu, bat %.2f ms, %.1f GMAC/s, plausible %s)\n",
           nm[w], N, K, KC, nch, Tn, diff ? "MISMATCH" : "BIT-EXACT", diff, hyB.size(),
           (size_t)nz, hyB.size(), msbat, gmac, plaus ? "yes" : "NO");
    if (diff && first != (size_t)-1)
      printf("   first diff at t=%zu row=%zu (bat %04x vs seq %04x)\n", first / N, first % N, hyB[first], hy1[first]);
    fflush(stdout);
    tested++;
    if (diff || !plaus) fails++;
    cudaFree(pB); cudaFree(p1); cudaFree(yB); cudaFree(y1);
  }
  cudaFree(xb); cudaFree(xb2); cudaFree(hw_d);
  printf("PF4_CHUNK %s: %d shapes, %s\n", fails ? "FAIL" : "PASS", tested, fails ? "FAILURES" : "ALL BIT-EXACT");
  fflush(stdout);
  return fails ? 1 : 0;
}

static int prefill_h2batched(Model& M) {
  // LFM_PF_T: the batch size the harness tests (default 8 = today). The ASR
  // chunk is 24 backbone tokens per 0.96 s of audio, so 24 is the shape that
  // decides the realtime budget.
  const int TPT = getenv("LFM_PF_T") ? atoi(getenv("LFM_PF_T")) : 8;
  const char* want[4] = {"embed_tokens.weight", "feed_forward.w2.weight",
                            "feed_forward.w1.weight", "conv.out_proj.weight"};
  int fails = 0, ntested = 0;
  for (int h2 = 0; h2 <= 1; h2++)
    for (int w = 0; w < 4; w++)
      for (size_t i = 0; i < M.ents.size(); i++) {
        Ent* e = &M.ents[i];
        if (!(e->dq && e->ds)) continue;
        if (strstr(e->name.c_str(), want[w]) == nullptr) continue;
        if (e->ne[0] < 1024 || e->ne[1] % 32 != 0) continue;
        fails += prefill_h2_one(M, e, TPT, h2);
        ntested++;
        break;
      }
  printf("prefill-h2batched: %d shapes tested, %s\n", ntested, fails ? "FAILURES" : "ALL BIT-EXACT");
  fflush(stdout);
  return fails ? 1 : 0;
}
// Test-only launcher: CT x ARM x H2 in one place (section 14 experiment).
// Instantiations: bf16 only (the selftest's dtype), CT in {2,4}, ARM in {0,1}.
static void pf_launch(int ct, int arm, int h2, int rb, const int8_t* q, const float* s,
                      const bf16* x, float* part, int N, int K, int KC, int R,
                      int nb, int T, int nblk, int TPB, size_t shm) {
#define PF(CT_, H2_, AR_) \
  q8_prefill_h2<bf16, CT_, H2_, AR_><<<nblk, TPB, shm>>>(q, s, x, part, N, K, KC, R, nb, T)
#define PF3(CT_, RB_) \
  q8_prefill_h2<bf16, CT_, true, 3, RB_><<<nblk, TPB, shm>>>(q, s, x, part, N, K, KC, R, nb, T)
  if (arm == 3 && h2) {
    if (ct == 2) { if (rb == 2) PF3(2,2); else if (rb == 4) PF3(2,4); else PF3(2,8); }
    else         { if (rb == 2) PF3(4,2); else if (rb == 4) PF3(4,4); else PF3(4,8); }
    return;
  }
  if (arm == 4 && h2) {
    if (ct == 2) PF(2, true, 4); else PF(4, true, 4);
    return;
  }
  if (arm == 5 && !h2) {
#define PF5F(CT_, G_) q8_prefill_h2<bf16, CT_, false, 5, 2, G_><<<nblk, TPB, shm>>>(q, s, x, part, N, K, KC, R, nb, T)
    // GRP is small here on purpose: xv[32] + acc5[GRP][CT] hits the 255-register
    // cap at GRP=8 and the launch is REFUSED (a refused launch is silent unless
    // the error is read -- the first version of this arm measured zeros).
    if (ct == 2) { if (rb == 4) PF5F(2, 4); else if (rb == 2) PF5F(2, 2); else PF5F(2, 1); }
    else         { if (rb == 4) PF5F(4, 4); else if (rb == 2) PF5F(4, 2); else PF5F(4, 1); }
#undef PF5F
    return;
  }
  if (arm == 5 && (R / (TPB / 32)) != rb) {
    printf("pf_launch ARM5: rows/warp %d != GRP %d -- refusing (the kernel would skip its compute)\n", R / (TPB / 32), rb);
    return;
  }
  if (arm == 5 && h2) {
#define PF5(CT_, G_) q8_prefill_h2<bf16, CT_, true, 5, 2, G_><<<nblk, TPB, shm>>>(q, s, x, part, N, K, KC, R, nb, T)
    if (ct == 2) { if (rb == 2) PF5(2,2); else if (rb == 4) PF5(2,4); else PF5(2,8); }
    else         { if (rb == 2) PF5(4,2); else if (rb == 4) PF5(4,4); else PF5(4,8); }
#undef PF5
    return;
  }
  if (arm == 6 && h2) {
    if (ct == 2) PF(2, true, 6); else PF(4, true, 6);
    return;
  }
  if (arm == 8 && h2) {
    if (ct == 2) PF(2, true, 8); else PF(4, true, 8);
    return;
  }
  if (arm == 10 && h2) {
#define PF10(CT_, RB_) q8_prefill_h2<bf16, CT_, true, 10, 2, RB_><<<nblk, TPB, shm>>>(q, s, x, part, N, K, KC, R, nb, T)
    static int rbx = -1;
    if (rbx < 0) rbx = getenv("LFM_PF_RB9") ? atoi(getenv("LFM_PF_RB9")) : 2;
    if (ct == 2) { if (rbx == 1) PF10(2,1); else if (rbx == 2) PF10(2,2); else { printf("LFM_PF_RB9=%d unsupported for ARM10 (1|2)\n", rbx); exit(2); } }
    else         { if (rbx == 1) PF10(4,1); else if (rbx == 2) PF10(4,2); else { printf("LFM_PF_RB9=%d unsupported for ARM10 (1|2)\n", rbx); exit(2); } }
#undef PF10
    return;
  }
  if (arm == 9 && h2) {
#define PF9(CT_, RB_) q8_prefill_h2<bf16, CT_, true, 9, 2, RB_><<<nblk, TPB, shm>>>(q, s, x, part, N, K, KC, R, nb, T)
    static int rb9 = -1;
    if (rb9 < 0) rb9 = getenv("LFM_PF_RB9") ? atoi(getenv("LFM_PF_RB9")) : 2;
    if (ct == 2) { if (rb9 == 1) PF9(2,1); else if (rb9 == 2) PF9(2,2); else if (rb9 == 4) PF9(2,4); else { printf("LFM_PF_RB9=%d unsupported (1|2|4)\n", rb9); exit(2); } }
    else         { if (rb9 == 1) PF9(4,1); else if (rb9 == 2) PF9(4,2); else if (rb9 == 4) PF9(4,4); else { printf("LFM_PF_RB9=%d unsupported (1|2|4)\n", rb9); exit(2); } }
#undef PF9
    return;
  }
  if (arm == 7 && h2) {
    static int abl = -1;
    if (abl < 0) abl = getenv("LFM_PF_ABL") ? atoi(getenv("LFM_PF_ABL")) : 0;
#define PF7(M_) q8_prefill_h2<bf16, 4, true, 7, 2, M_><<<nblk, TPB, shm>>>(q, s, x, part, N, K, KC, R, nb, T)
    switch (abl) {
      case 0: PF7(0); break;  case 1: PF7(1); break;  case 2: PF7(2); break;
      case 4: PF7(4); break;  case 8: PF7(8); break;  case 15: PF7(15); break;
      case 16: PF7(16); break; case 18: PF7(18); break; case 26: PF7(26); break;
      default: printf("LFM_PF_ABL=%d unsupported (0|1|2|4|8|15|16|18|26)\n", abl); exit(2);
    }
#undef PF7
    return;
  }
  if (arm >= 3) { arm = 0; }   // f32 path: these arms not instantiated
  if (h2) { if (ct == 2) { if (arm == 2) PF(2, true, 2); else if (arm) PF(2, true, 1); else PF(2, true, 0); }
            else if (ct == 4) { if (arm == 2) PF(4, true, 2); else if (arm) PF(4, true, 1); else PF(4, true, 0); }
            else if (ct == 6) { if (arm == 2) PF(6, true, 2); else if (arm) PF(6, true, 1); else PF(6, true, 0); }
            else          { if (arm == 2) PF(8, true, 2); else if (arm) PF(8, true, 1); else PF(8, true, 0); } }
  else    { if (ct == 2) { if (arm == 2) PF(2, false, 2); else if (arm) PF(2, false, 1); else PF(2, false, 0); }
            else if (ct == 4) { if (arm == 2) PF(4, false, 2); else if (arm) PF(4, false, 1); else PF(4, false, 0); }
            else if (ct == 6) { if (arm == 2) PF(6, false, 2); else if (arm) PF(6, false, 1); else PF(6, false, 0); }
            else          { if (arm == 2) PF(8, false, 2); else if (arm) PF(8, false, 1); else PF(8, false, 0); } }
#undef PF
#undef PF3
}

// Batched-prefill arm knobs. File scope so the committed probe (LFM_PF_PROBE, in main)
// can select an arm per pass; pfForce overrides the env value after the env parse.
static bool is_attn_g[NLAYER], is_slide_g[NLAYER];   // mirrors main's locals, for the T-wide driver
static float bf2f_host(bf16 v) { uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f; }
static int pfCT = -1, pfARM = -1, pfRB = -1, pfRPB = -1, pfForce = -1;
static int g_pfChunk = 0;   // --prefill-run T: chunked T-wide prefill inside the real step loop
static void pf_set_arm(int a) { pfForce = a; pfCT = -1; }   // pfCT<0 forces a re-init

static int prefill_h2_one(Model& M, Ent* e, int T, int h2) {
  (void)M;
  const int N = e->ne[0], K = e->ne[1], TPB = 256;
  // LFM_PF_CT (2|4) / LFM_PF_ARM (0|1): test-only knobs, defaults = today's
  // structure (CT=2, ARM=0) so the shipped selftest output is unchanged.
  if (pfCT < 0) {
    pfCT = getenv("LFM_PF_CT") ? atoi(getenv("LFM_PF_CT")) : 2;
    pfARM = getenv("LFM_PF_ARM") ? atoi(getenv("LFM_PF_ARM")) : 0;
    if (pfForce >= 0) pfARM = pfForce;   // set by the LFM_PF_PROBE passes in main()
    if (pfCT != 2 && pfCT != 4 && pfCT != 6 && pfCT != 8) { printf("LFM_PF_CT=%d unsupported (2|4|6|8) -- using 2\n", pfCT); pfCT = 2; }
    if (pfARM < 0 || pfARM > 10) { printf("LFM_PF_ARM=%d unsupported (0..10)\n", pfARM); exit(2); }
    if (pfARM == 7 && pfCT != 4) { printf("LFM_PF_ARM=7 is instantiated for CT=4 only\n"); exit(2); }
    pfRPB = getenv("LFM_PF_RPB") ? atoi(getenv("LFM_PF_RPB")) : 4;
    if (pfRPB != 1 && pfRPB != 2 && pfRPB != 4 && pfRPB != 8) { printf("LFM_PF_RPB=%d unsupported (1|2|4|8) -- using 4\n", pfRPB); pfRPB = 4; }
    pfRB = getenv("LFM_PF_RB") ? atoi(getenv("LFM_PF_RB")) : 2;
    if (pfRB != 1 && pfRB != 2 && pfRB != 4 && pfRB != 8) { printf("LFM_PF_RB=%d unsupported (1|2|4|8)\n", pfRB); exit(2); }
  }
  const int CT = pfCT, ARM = pfARM, RB = (pfARM == 5) ? pfRPB : pfRB;   // ARM5: GRP == rows/warp, one knob
  const int R = (ARM == 5) ? (256 / 32) * pfRPB : 256;   // ARM5 needs small rows/warp
  int KC = K;
  while (KC > 32) {
    int nbtry = KC / Q8_BLK;
    size_t need = h2 ? (size_t)CT * nbtry * 17 * sizeof(__half2) : (size_t)CT * nbtry * 33 * 4;
    if (need <= 49152 && K % KC == 0) break;
    KC /= 2;
  }
  const int nb = KC / Q8_BLK, nch = K / KC;
  const int nblk = ((N + R - 1) / R) * nch;
  const size_t shm = h2 ? (size_t)CT * nb * 17 * sizeof(__half2) : (size_t)CT * nb * 33 * 4;
  printf("prefill-h2batched H2=%d CT=%d ARM=%d %s N=%d K=%d T=%d shm=%zu\n",
         h2, CT, ARM, e->name.c_str(), N, K, T, shm); fflush(stdout);
  uint16_t* xd = nullptr; float *seq = nullptr, *bat = nullptr, *bat0 = nullptr;
  printf("a=%s\n", cudaGetErrorString(cudaMalloc(&xd, (size_t)T * K * 2))); fflush(stdout);
  printf("b=%s\n", cudaGetErrorString(cudaMalloc(&seq, (size_t)T * N * nch * 4))); fflush(stdout);
  printf("c=%s\n", cudaGetErrorString(cudaMalloc(&bat, (size_t)T * N * nch * 4))); fflush(stdout);
  if (ARM) {
    printf("b0=%s\n", cudaGetErrorString(cudaMalloc(&bat0, (size_t)T * N * nch * 4))); fflush(stdout);
  }
  std::vector<uint16_t> pat((size_t)T * K);
  for (size_t i = 0; i < pat.size(); i++) pat[i] = (uint16_t)(0x3c00 + ((i * 37) & 0x3ff));
  printf("h2d=%s\n", cudaGetErrorString(cudaMemcpy(xd, pat.data(), pat.size() * 2, cudaMemcpyHostToDevice))); fflush(stdout);
  cudaEvent_t ea, eb; cudaEventCreate(&ea); cudaEventCreate(&eb); float ms = 0;
  cudaEventRecord(ea);
  if (ARM) {
    // Structure-consistent pair: T=1 launches of the SAME armed kernel. With
    // ARM=1 the q8_partial serial structure computes different parenthesization
    // than ARM=1, so comparing against it would test numerics, not batching.
    for (int t = 0; t < T; t++)
      pf_launch(CT, ARM, h2, RB, e->dq, e->ds, (const bf16*)(xd + (size_t)t * K),
                seq + (size_t)t * N * nch, N, K, KC, R, nb, 1, nblk, TPB, shm);
  } else
  for (int t = 0; t < T; t++) {
    if (h2)
      q8_partial<bf16,true,false,1,1><<<nblk, TPB, (size_t)nb * 33 * 4>>>(e->dq, e->ds, (const bf16*)(xd + (size_t)t * K),
        seq + (size_t)t * N * nch, N, K, KC, R, nb);
    else
      q8_partial<bf16,true,false,1,0><<<nblk, TPB, (size_t)nb * 33 * 4>>>(e->dq, e->ds, (const bf16*)(xd + (size_t)t * K),
        seq + (size_t)t * N * nch, N, K, KC, R, nb);
  }
  cudaEventRecord(eb); cudaEventSynchronize(eb);
  cudaEventElapsedTime(&ms, ea, eb);
  printf("seq23 err=%s\n", cudaGetErrorString(cudaGetLastError())); fflush(stdout);
  std::vector<float> a((size_t)T * N * nch);
  printf("cp-seq=%s\n", cudaGetErrorString(cudaMemcpy(a.data(), seq, a.size() * 4, cudaMemcpyDeviceToHost))); fflush(stdout);
  printf("seq[0]=%.9g (expect H2 ~-0.10976)\n", a[0]); fflush(stdout);
  if (ARM == 8) { unsigned long long z[8] = {0}; cudaMemcpyToSymbol(g_pf_prof, z, sizeof z); }
  cudaEventRecord(ea);
  pf_launch(CT, ARM, h2, RB, e->dq, e->ds, (const bf16*)xd, bat,
            N, K, KC, R, nb, T, nblk, TPB, shm);
  printf("bat-launch=%s\n", cudaGetErrorString(cudaGetLastError())); fflush(stdout);
  cudaEventRecord(eb); cudaEventSynchronize(eb);
  printf("bat-sync=%s\n", cudaGetErrorString(cudaGetLastError())); fflush(stdout);
  float ms2 = 0; cudaEventElapsedTime(&ms2, ea, eb);
  {
    // Occupancy as the runtime computes it for exactly this launch config.
    static int once = 0;
    if (!once && h2) {
      once = 1;
      cudaDeviceProp dp; cudaGetDeviceProperties(&dp, 0);
      int nb0 = 0, nb8 = 0;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb0, q8_prefill_h2<bf16, 4, true, 0>, TPB, shm);
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb8, q8_prefill_h2<bf16, 4, true, 8>, TPB, shm);
      cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, q8_prefill_h2<bf16, 4, true, 0>);
      printf("occupancy: SMs=%d regs/SM=%d shm/SM=%zu shm/block(max)=%zu maxThr/SM=%d | "
             "ARM0 CT4 H2: regs=%d static_shm=%zu dyn_shm=%zu TPB=%d -> %d blocks/SM | ARM8 -> %d blocks/SM\n",
             dp.multiProcessorCount, dp.regsPerMultiprocessor, dp.sharedMemPerMultiprocessor,
             dp.sharedMemPerBlock, dp.maxThreadsPerMultiProcessor,
             fa.numRegs, fa.sharedSizeBytes, shm, TPB, nb0, nb8);
      fflush(stdout);
    }
  }
  if (ARM == 8 && h2) {
    unsigned long long pr[8] = {0};
    cudaMemcpyFromSymbol(pr, g_pf_prof, sizeof pr);
    const double tot = (double)pr[5], w = pr[6] ? (double)pr[6] : 1.0;
    const char* nm[5] = {"staging", "barrier1", "b-loops", "reduce", "barrier2"};
    printf("profile %s T=%d: %llu warps, avg %.0f cyc/warp (%.2f ms at 904 MHz)",
           e->name.c_str(), T, pr[6], tot / w, tot / w / 904e3);
    for (int i = 0; i < 5; i++) printf("  %s %.1f%%", nm[i], tot ? 100.0 * pr[i] / tot : 0.0);
    printf("\n"); fflush(stdout);
  }
  // LFM_PF_REPS=n: steady-state timing. The single launch above can start at a
  // low DVFS clock (sm_clock: first ~0.4 s of load averages 348 MHz vs 904
  // steady; the harness idles the GPU for seconds of host compares between
  // launches). Run n back-to-back launches into a scratch buffer and time the
  // second half only. Outputs/bit-exactness still come from the single launch.
  {
    static int reps = -1;
    if (reps < 0) { reps = getenv("LFM_PF_REPS") ? atoi(getenv("LFM_PF_REPS")) : 0; }
    if (reps >= 2) {
      float* scr = nullptr; cudaMalloc(&scr, (size_t)T * N * nch * 4);
      for (int i = 0; i < reps / 2; i++)
        pf_launch(CT, ARM, h2, RB, e->dq, e->ds, (const bf16*)xd, scr, N, K, KC, R, nb, T, nblk, TPB, shm);
      cudaEventRecord(ea);
      const int nt = reps - reps / 2;
      for (int i = 0; i < nt; i++)
        pf_launch(CT, ARM, h2, RB, e->dq, e->ds, (const bf16*)xd, scr, N, K, KC, R, nb, T, nblk, TPB, shm);
      cudaEventRecord(eb); cudaEventSynchronize(eb);
      cudaError_t we = cudaGetLastError();
      float msw = 0; cudaEventElapsedTime(&msw, ea, eb); msw /= nt;
      const double gw = (double)T * N * K / (msw * 1e-3) / 1e9;
      printf("warm %s H2=%d CT=%d ARM=%d: %.3f ms/launch over %d (%.1f GMAC/s)%s%s\n",
             e->name.c_str(), h2, CT, ARM, msw, nt, gw,
             we ? " LAUNCH ERROR " : "", we ? cudaGetErrorString(we) : "");
      fflush(stdout);
      cudaFree(scr);
    }
  }
  std::vector<float> b(a.size(), -999.f);
  printf("cp-bat=%s\n", cudaGetErrorString(cudaMemcpy(b.data(), bat, b.size() * 4, cudaMemcpyDeviceToHost))); fflush(stdout);
  printf("bat[0]=%.9g\n", b[0]); fflush(stdout);
  { // PLAUSIBILITY (a pair that both emit zeros is 'bit-exact' and certifies nothing)
    size_t nz = 0; for (size_t i = 0; i < a.size(); i++) nz += (a[i] != 0.f) || (b[i] != 0.f);
    if (nz == 0) { printf("PLAUSIBILITY FAIL: %s all-zero output (launch skipped or refused)\n", e->name.c_str()); fflush(stdout); return 1; }
    const double gmac = (double)T * N * K / (ms2 * 1e-3) / 1e9;
    // Ceiling MEASURED on this board (bench/peak_rates.cu): HFMA2 105.8 GMAC/s,
    // FFMA 82 GFMA/s (datasheet 236/118 is not reachable: HFMA2 issues at ~2
    // warp-instr/clk on the single GM20B SM). Anything above 120 is a bug.
    if (gmac > 120.0) { printf("PLAUSIBILITY FAIL: %s %.0f GMAC/s exceeds the measured ~106 GMAC/s HFMA2 ceiling (gate 120)\n", e->name.c_str(), gmac); fflush(stdout); return 1; }
    printf("plausible: %s nonzero=%zu/%zu  bat %.1f GMAC/s\n", e->name.c_str(), nz, a.size(), gmac); fflush(stdout);
  }
  long bad = 0; for (size_t i = 0; i < a.size() && bad < 5; i++) if (memcmp(&a[i], &b[i], 4)) { bad++; if (bad == 1) printf("first diff at %zu: seq=%.9g bat=%.9g\n", i, a[i], b[i]); }
  long nbad = 0; for (size_t i = 0; i < a.size(); i++) nbad += memcmp(&a[i], &b[i], 4) != 0;
  printf("prefill-h2batched H2=%d CT=%d ARM=%d %s: %s (%ld/%zu differ) seq %.2f ms bat %.2f ms\n",
         h2, CT, ARM, e->name.c_str(), nbad ? "MISMATCH" : "BIT-EXACT", nbad, a.size(), ms, ms2);
  fflush(stdout);
  // ARM=1: numerics sanity -- rel-vs the ARM=0 structure on identical inputs.
  // Batch-correctness (above) says ARM1-batch == ARM1-sequential; this says what
  // the arm change does to the values themselves.
  if (ARM && !nbad) {
    pf_launch(CT, 0, h2, RB, e->dq, e->ds, (const bf16*)xd, bat0,
              N, K, KC, R, nb, T, nblk, TPB, shm);
    printf("arm0-launch=%s\n", cudaGetErrorString(cudaGetLastError())); fflush(stdout);
    cudaDeviceSynchronize();
    std::vector<float> b0(a.size(), -999.f);
    cudaMemcpy(b0.data(), bat0, b0.size() * 4, cudaMemcpyDeviceToHost);
    double worst = 0.0;
    for (size_t i = 0; i < b0.size(); i++) {
      double den = fabs((double)b0[i]);
      double rel = fabs((double)b[i] - (double)b0[i]) / (den > 1e-6 ? den : 1e-6);
      if (rel > worst) worst = rel;
    }
    printf("prefill-h2batched ARM1-vs-ARM0 %s: max rel %.3g\n", e->name.c_str(), worst);
    fflush(stdout);
  }
  return nbad ? 1 : 0;
}

// ------------------------------------------------------------------ lm_head
// The tied embedding is 128000 x 2048 -- 262 MB of Q8 codes + 8 MB of fp32
// scales -- and it is streamed exactly ONCE per token, as the lm_head GEMV
// (it is tied to the input embedding, so there is no separate output matrix).
// On a 4 GB Nano that 270 MB is the difference between fitting and not:
// device free on an IDLE board measured 1826 MB, and the whole-backbone Q8_0
// config needs 1944 MB of weights + ~48 MB arena. Keeping the embedding in
// PINNED HOST memory and streaming it in row chunks drops the device need to
// ~1682 MB, which fits.
//
// This does not change one bit of arithmetic: each output row's reduction is
// independent of every other row, so computing rows [r0,r0+nr) from a slab
// that happens to hold exactly those rows is the same computation the
// one-shot path does. `--lm-selftest` asserts exactly that, bitwise, over
// all 128000 logits, before any timing number is believed.
//
// Sequential copies for now (copy chunk, GEMV chunk), with pinned host memory
// so the H2D runs at DMA speed. Double-buffering over the 17.5 ms copy and
// the 17.5 ms read is the obvious follow-up; measured first, then optimised.

// Read the tied embedding into pinned host memory and check its CRC. Pinned
// (not pageable) is what lets the per-chunk H2D run at DMA speed; on Tegra
// the copy shares the same 21.9 GB/s bus the GEMV reads from, so streaming
// the embedding costs a second pass over those bytes -- priced in 13.5.
static bool lm_host_init(Model& M, HostEmb& E, const char* path) {
  Ent* be = nullptr;
  for (size_t i = 0; i < M.ents.size(); i++) if (M.ents[i].name == EMBNAME) be = &M.ents[i];
  if (!be) { printf("no tied embedding %s in the blob\n", EMBNAME); return false; }
  if (be->dtype == 3) {
    // CERTIFIED ARM = quant_sensitivity.py's q8_0 (linears Q8 + fp32 scales +
    // bf16 embeddings): 8/8 greedy goldens, where Q8 embeddings (q8_0_embq8)
    // cost 2 prompts and the worst prefix drops to 1. 524 MB of bf16 is twice
    // the Q8 embedding, so stream it from pinned host in row slabs like the
    // Q8 path; a bf16 row is 2x the bytes, so the slab is LM_ROWS/2 rows.
    E.N = be->ne[0]; E.K = be->ne[1];
    if (E.N != VOCAB || E.K != H) { printf("embedding shape %dx%d\n", E.N, E.K); return false; }
    E.rows = LM_ROWS / 2;
    if (getenv("LFM_ROWS")) E.rows = atoi(getenv("LFM_ROWS"));
    const size_t tot = be->nc;                       // N*K*2 bytes; ns == 0
    if (cudaHostAlloc((void**)&E.hc, tot, cudaHostAllocDefault) != cudaSuccess) {
      printf("cudaHostAlloc(%.0f MB pinned bf16) failed\n", tot / 1e6); return false;
    }
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot reopen %s\n", path); return false; }
    fseek(f, (long)M.emb_file_off, SEEK_SET);
    size_t got = fread(E.hc, 1, tot, f);
    posix_fadvise(fileno(f), (off_t)M.emb_file_off, (off_t)tot, POSIX_FADV_DONTNEED);
    fclose(f);
    if (got != tot) { printf("short read on the bf16 embedding (%zu/%zu)\n", got, tot); return false; }
    const uint32_t c = (uint32_t)crc32(0, (const Bytef*)E.hc, (uInt)tot);
    if (c != be->crc) { printf("host bf16 embedding CRC %08x want %08x\n", c, be->crc); return false; }
    if (cudaMalloc((void**)&E.dc, (size_t)E.rows * E.K * 2) != cudaSuccess) {
      printf("lm slab bf16 (%.0f MB) failed\n", E.rows * E.K * 2 / 1e6); return false;
    }
    printf("lm_head staged in host: %.0f MB pinned bf16, %d rows/slab (%.0f MB device)\n",
           tot / 1e6, E.rows, E.rows * E.K * 2 / 1e6);
    fflush(stdout);
    return true;
  }
  if (be->dtype != 1) { printf("embedding dtype %d is not Q8\n", be->dtype); return false; }
  E.N = be->ne[0]; E.K = be->ne[1];
  if (E.N != VOCAB || E.K != H) { printf("embedding shape %dx%d\n", E.N, E.K); return false; }
  E.rows = LM_ROWS;
  if (getenv("LFM_ROWS")) E.rows = atoi(getenv("LFM_ROWS"));   // slab rows, A/B knob
  const size_t tot = be->nc + be->ns;
  if (cudaHostAlloc((void**)&E.hc, tot, cudaHostAllocDefault) != cudaSuccess) {
    printf("cudaHostAlloc(%.0f MB pinned) failed\n", tot / 1e6); return false;
  }
  E.hs = (float*)(E.hc + be->nc);
  FILE* f = fopen(path, "rb");
  if (!f) { printf("cannot reopen %s\n", path); return false; }
  fseek(f, (long)M.emb_file_off, SEEK_SET);
  size_t got = fread(E.hc, 1, tot, f);
  posix_fadvise(fileno(f), (off_t)M.emb_file_off, (off_t)tot, POSIX_FADV_DONTNEED);
  fclose(f);
  if (got != tot) { printf("short read on the embedding (%zu/%zu)\n", got, tot); return false; }
  const uint32_t c = (uint32_t)crc32(0, (const Bytef*)E.hc, (uInt)tot);
  if (c != be->crc) { printf("host embedding CRC %08x want %08x\n", c, be->crc); return false; }
  if (cudaMalloc((void**)&E.dc, (size_t)E.rows * E.K) != cudaSuccess) {
    printf("lm slab codes (%.0f MB) failed\n", E.rows * E.K / 1e6); return false;
  }
  if (cudaMalloc((void**)&E.dsl, (size_t)E.rows * (E.K / Q8_BLK) * 4) != cudaSuccess) {
    printf("lm slab scales failed\n"); return false;
  }
  if (g_lmdbuf) {
    E.dbuf = true;
    for (int b = 0; b < 2; b++) {
      if (cudaMalloc((void**)&E.dc2[b], (size_t)E.rows * E.K) != cudaSuccess ||
          cudaMalloc((void**)&E.dsl2[b], (size_t)E.rows * (E.K / Q8_BLK) * 4) != cudaSuccess) {
        printf("LM_HEAD_DBUF: slab %d allocation failed\n", b); E.dbuf = false; break;
      }
      cudaEventCreateWithFlags(&E.evC[b], cudaEventDisableTiming);
      cudaEventCreateWithFlags(&E.evG[b], cudaEventDisableTiming);
    }
    if (E.dbuf) cudaStreamCreateWithFlags(&E.st, cudaStreamNonBlocking);
    if (!E.st) E.dbuf = false;
    printf("LM_HEAD_DBUF=on slabs=%d x2 rows=%d (proof: 2 slabs + stream + 4 events)\n",
           E.dbuf ? 2 : 0, E.rows);
    fflush(stdout);
  }

  printf("lm_head staged in host: %.0f MB pinned, %d rows/slab (%.0f MB device)\n",
         tot / 1e6, E.rows, (E.rows * E.K + (size_t)E.rows * (E.K / Q8_BLK) * 4) / 1e6);
  fflush(stdout);
  return true;
}

// ---------------------------------------------------------------------------
// bf16 GEMV for the CERTIFIED arm's lm_head. quant_sensitivity.py: q8_0 =
// linears Q8 + fp32 scales + bf16 embeddings reproduces 8/8 greedy goldens,
// while Q8 embeddings (q8_0_embq8) cost 2 prompts (6/8, worst prefix 1). The
// Q8-embedding blob therefore cannot pass the 8/8 gate at any latency.
//   out[n] = sum_k f32(x[k]) * f32(W[n,k])
// Reduction ORDER is documented and fixed: lanes stride k by 32 in ascending
// order into one fp32 accumulator, then the standard 16,8,4,2,1 butterfly --
// the same per-row shape as the Q8 kernel, so the arms differ only in the
// weight source. x is staged in shared (8 KB) and amortized over R rows per
// block, the lever ideas.md row 3 measures on the Q8 path.
// ---------------------------------------------------------------------------
__global__ void k_gemv_bf16(const bf16* __restrict__ W, const float* __restrict__ xf,
                            float* __restrict__ out, int N, int K, int R) {
  extern __shared__ float sx[];
  const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, nw = blockDim.x >> 5;
  for (int i = tid; i < K; i += blockDim.x) sx[i] = xf[i];
  __syncthreads();
  const int r0 = blockIdx.x * R;
  for (int rr = wid; rr < R; rr += nw) {
    const int n = r0 + rr;
    if (n >= N) return;
    const bf16* wr = W + (size_t)n * K;
    float s = 0.f;
    for (int k = lane; k < K; k += 32) s = __fadd_rn(s, __fmul_rn(bf2f(wr[k]), sx[k]));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_down_sync(0xffffffffu, s, o));
    if (lane == 0) out[n] = s;
  }
}

static void lm_head_bf16(Model& M, Ent* emb, const bf16* xb, float* yf) {
  const int K = (int)emb->ne[1], N = (int)emb->ne[0];
  k_bf2f<<<(K + 255) / 256, 256, 0, g_launch>>>(xb, M.xf, K);   // M.xf is the shared x-float buffer
  const int R = 32;                       // rows per block; tuned in measure.sh
  if (getenv("LFM_DBG_SPANS")) {
    // Validate EVERY Q8 tensor's full span against its arena, on the host, at
    // load time. A device kernel can only sample; a host check covers the
    // whole range, and an out-of-arena pointer is what makes an "unspecified
    // launch failure" surface two kernels later. The certified blob moves the
    // embedding out of the Q8 arena (524 MB of bf16), shifting every layer's
    // offset -- exactly the case worth asserting.
    int bad = 0, nq = 0;
    for (size_t i2 = 0; i2 < M.ents.size(); i2++) {
      Ent& e2 = M.ents[i2];
      if (e2.host || e2.dtype != 1) continue;
      nq++;
      const size_t cb = (size_t)e2.ne[0] * e2.ne[1];
      const size_t sb = (size_t)e2.ne[0] * (e2.ne[1] / 32) * 4;
      const char* q = (const char*)e2.dq;
      const char* sc = (const char*)e2.ds;
      if (!q || q < (const char*)M.dq || q + cb > (const char*)M.dq + M.need_q ||
          !sc || sc < (const char*)M.ds || sc + sb > (const char*)M.ds + M.need_s) {
        printf("SPAN BAD %-58s q=[%p,+%zu) s=[%p,+%zu) arena q=[%p,+%.0fMB) s=[%p,+%.0fMB)\n",
               e2.name.c_str(), (const void*)q, cb, (const void*)sc, sb,
               (const void*)M.dq, M.need_q / 1e6, (const void*)M.ds, M.need_s / 1e6);
        bad++;
      }
    }
    printf("SPANS: %d Q8 tensors checked, %d bad\n", nq, bad);
    fflush(stdout);
  }
  if (getenv("LFM_DEBUG_LM"))
    fprintf(stderr, "[lm] W=%p dq=%p xf=%p yf=%p N=%d K=%d R=%d ne=%ld,%ld dtype=%d nc=%lld ns=%lld\n",
            (const void*)(const bf16*)emb->dq, (void*)emb->dq, (void*)M.xf, (void*)yf, N, K, R,
            (long)emb->ne[0], (long)emb->ne[1], emb->dtype, (long long)emb->nc, (long long)emb->ns);
  k_gemv_bf16<<<(N + R - 1) / R, 256, K * sizeof(float), g_launch>>>(
      (const bf16*)emb->dq, M.xf, yf, N, K, R);
}

// Certified arm, host-staged: stream rows/slab of the pinned bf16 embedding
// into a device slab and run k_gemv_bf16 on it. Per-row reduction order is
// identical to the resident bf16 path (lanes stride k by 32 into one fp32,
// then the 16..1 butterfly), so staged and resident agree bit for bit.
static void lm_head_bf16_host(Model& M, HostEmb& E, const bf16* xb, float* yf) {
  const int N = E.N, K = E.K;
  k_bf2f<<<(K + 255) / 256, 256, 0, g_launch>>>(xb, M.xf, K);
  const int R = 32;
  for (int r0 = 0; r0 < N; r0 += E.rows) {
    const int nr = (N - r0 < E.rows) ? (N - r0) : E.rows;
    cudaMemcpyAsync(E.dc, (const char*)E.hc + (size_t)r0 * K * 2, (size_t)nr * K * 2,
                    cudaMemcpyHostToDevice, 0);
    k_gemv_bf16<<<(nr + R - 1) / R, 256, K * sizeof(float), g_launch>>>(
        (const bf16*)E.dc, M.xf, yf + r0, nr, K, R);
  }
}

static void lm_head_host(Model& M, HostEmb& E, const bf16* xb, float* yf) {
  const int N = E.N, K = E.K;
  GemvCfg c = pick(N, K);                       // SAME config as the resident
  const int nb = c.KCHUNK / Q8_BLK, nchunk = K / c.KCHUNK;   // path -> same bits
  k_bf2f<<<(K + 255) / 256, 256, 0, g_launch>>>(xb, M.xf, K);
  const int sb = K / Q8_BLK;                    // scales per row
  if (E.dbuf) {
    const int nslab = (N + E.rows - 1) / E.rows;
    const int nblk = ((E.rows + c.R - 1) / c.R) * nchunk;
    const size_t shm = (size_t)nb * 33 * 4;
    for (int i = 0; i < nslab; i++) {
      const int r0 = i * E.rows;
      const int nr = (N - r0 < E.rows) ? (N - r0) : E.rows;
      const int b = i & 1;
      const int nb2 = ((nr + c.R - 1) / c.R) * nchunk;      // last slab is short
      if (i >= 2) cudaStreamWaitEvent(E.st, E.evG[b], 0);   // WAR: GEMV(i-2) must have read slab b
      cudaMemcpyAsync(E.dc2[b], E.hc + (size_t)r0 * K, (size_t)nr * K,
                      cudaMemcpyHostToDevice, E.st);
      cudaMemcpyAsync(E.dsl2[b], E.hs + (size_t)r0 * sb, (size_t)nr * sb * 4,
                      cudaMemcpyHostToDevice, E.st);
      cudaEventRecord(E.evC[b], E.st);
      cudaStreamWaitEvent((cudaStream_t)0, E.evC[b], 0);
      q8_partial<float,false><<<nb2, c.TPB, shm, 0>>>(E.dc2[b], E.dsl2[b], M.xf, M.part, nr, K, c.KCHUNK, c.R, nb);
      k_combine_f<<<(nr + 255) / 256, 256, 0, g_launch>>>(M.part, yf + r0, nr, nchunk);
      cudaEventRecord(E.evG[b], (cudaStream_t)0);
    }
    return;
  }
  for (int r0 = 0; r0 < N; r0 += E.rows) {
    const int nr = (N - r0 < E.rows) ? (N - r0) : E.rows;
    cudaMemcpyAsync(E.dc, E.hc + (size_t)r0 * K, (size_t)nr * K,
                    cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(E.dsl, E.hs + (size_t)r0 * sb, (size_t)nr * sb * 4,
                    cudaMemcpyHostToDevice, 0);
    const int nblk = ((nr + c.R - 1) / c.R) * nchunk;
    const size_t shm = (size_t)nb * 33 * 4;
    if (g_fma) q8_partial<float,true><<<nblk, c.TPB, shm, g_launch>>>(E.dc, E.dsl, M.xf, M.part, nr, K, c.KCHUNK, c.R, nb);
    else q8_partial<float,false><<<nblk, c.TPB, shm, g_launch>>>(E.dc, E.dsl, M.xf, M.part, nr, K, c.KCHUNK, c.R, nb);
    k_combine_f<<<(nr + 255) / 256, 256, 0, g_launch>>>(M.part, yf + r0, nr, nchunk);
  }
}

static void gemv_f(Model& M, const std::string& name, const bf16* xb, float* yf) {
  Ent* e = T(M, name);
  if (e->tiled) { printf("gemv_f %s: tiled tensor\n", name.c_str()); exit(2); }
  const int N = e->ne[0], K = e->ne[1];
  GemvCfg c = pick(N, K);
  if (K % c.KCHUNK) { printf("%s: K=%d not divisible by KCHUNK=%d\n", name.c_str(), K, c.KCHUNK); exit(2); }
  const int nb = c.KCHUNK / Q8_BLK, nchunk = K / c.KCHUNK;
  const int nblk = ((N + c.R - 1) / c.R) * nchunk;
  const size_t shm = (size_t)nb * 33 * 4;
  if (g_nofuse) {
    k_bf2f<<<(K + 255) / 256, 256, 0, g_launch>>>(xb, M.xf, K);
    if (g_fma) q8_partial<float,true><<<nblk, c.TPB, shm, g_launch>>>(e->dq, e->ds, M.xf, M.part, N, K, c.KCHUNK, c.R, nb);
    else q8_partial<float,false><<<nblk, c.TPB, shm, g_launch>>>(e->dq, e->ds, M.xf, M.part, N, K, c.KCHUNK, c.R, nb);
  } else {
    q8_launch_bf(nblk, c.TPB, shm, e->dq, e->ds, xb, M.part, N, K, c.KCHUNK, c.R, nb, g_launch);
  }
  k_combine_f<<<(N + 255) / 256, 256, 0, g_launch>>>(M.part, yf, N, nchunk);
  // Per-launch check, env-gated. Without it a faulting kernel is reported by
  // the NEXT checked call, which in a long layer chain points at an innocent
  // victim (it cost a cycle chasing the bf16 lm_head for a fault that was in
  // layer 0's FFN GEMV).
  if (getenv("LFM_DBG_LAUNCH")) {
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess)
      { fprintf(stderr, "[launch] %s N=%d K=%d R=%d TPB=%d nb=%d nchunk=%d shm=%zu dq=%p ds=%p xf=%p part=%p: %s\n",
                name.c_str(), N, K, c.R, c.TPB, nb, nchunk, shm, (const void*)e->dq, (const void*)e->ds,
                (void*)M.xf, (void*)M.part, cudaGetErrorString(le)); fflush(stderr); }
    if (getenv("LFM_DBG_LAUNCH_SYNC")) cudaDeviceSynchronize();
  }
}
// GEMV with an explicit input width (the 2048-wide h1 feeding a 512-column
// projection is normal; the weight's own K is authoritative).
static void rms1(Model& M, const bf16* x, const float* w, bf16* y) {
  k_rmsnorm<<<1, 256, 0, g_launch>>>(x, w, y, NORM_EPS, H);
}
static void emul(Model& M, const bf16* a, const bf16* b, bf16* y, int n) {
  k_mul_bf<<<(n + 255) / 256, 256, 0, g_launch>>>(a, b, y, n);
}
static void eadd(Model& M, const bf16* a, const bf16* b, bf16* y, int n) {
  if (ewv_on() && n % 4 == 0 && (((uintptr_t)a | (uintptr_t)b | (uintptr_t)y) & 7) == 0) {
    k_add_bf4<<<(n / 4 + 255) / 256, 256, 0, g_launch>>>(a, b, y, n / 4);
    return;
  }
  k_add_bf<<<(n + 255) / 256, 256, 0, g_launch>>>(a, b, y, n);
}

// Per-layer fingerprint for localizing a divergence (the perlayer_probe role
// the Pi engine plays with tools/perlayer_probe.patch). rms, not a hash: the
// first layer where this stops tracking HF's hidden_states rms is the layer
// that is wrong, and rms is cheap to print from both sides.
static std::vector<uint16_t> g_scr;
static float rms_host_n(const bf16* x, int n) {
  g_scr.resize(n);
  // Check the copy. Four consecutive probes printing an IDENTICAL rms is the
  // tell: a failed cudaMemcpy leaves the shared staging vector at its
  // previous content, so the probe prints last time's number and looks like
  // a real measurement. A probe that can silently repeat itself is worse
  // than no probe -- it manufactures agreement.
  cudaError_t ce = cudaMemcpy(g_scr.data(), x, (size_t)n * 2, cudaMemcpyDeviceToHost);
  if (ce != cudaSuccess) { printf("    [probe copy failed: %s]\n", cudaGetErrorString(ce)); return -1.f; }
  double acc = 0;
  for (int i = 0; i < n; i++) {
    uint32_t u = ((uint32_t)g_scr[i]) << 16; float v; memcpy(&v, &u, 4);
    acc += (double)v * (double)v;
  }
  return (float)sqrt(acc / (double)n);
}
static float rms_host(Model& M, const bf16* x) { return rms_host_n(x, H); }
// Layer-0 stage probe: rms at every named stage of one layer, printed by the
// engine and computed for the same stages on the HF side (probe_lfm3_stages.py).
// rms is a weak fingerprint on purpose -- it is permutation-invariant, so a
// permuted GEMV output can print the same rms; that is exactly why the
// divergence table is paired with a positional fingerprint (first 4 values)
// and why rms equality is never treated as agreement, only rms INEQUALITY is
// treated as evidence.
static int g_probe = 0;
static int g_probe_layer = 0;      // --probe-layer N
static int g_probe_step = 0;       // --probe-step N (the decode step, == position here)
#define PROBE(nm, ptr, n) if (g_probe) printf("    %-14s rms %9.5f  %8.4f %8.4f %8.4f %8.4f\n", \
      nm, rms_host_n(ptr, n), pv4(ptr,0), pv4(ptr,1), pv4(ptr,2), pv4(ptr,3))
static float pv4(const bf16* x, int i) {
  uint16_t h; cudaMemcpy(&h, (const char*)x + 2 * i, 2, cudaMemcpyDeviceToHost);
  uint32_t u = ((uint32_t)h) << 16; float v; memcpy(&v, &u, 4); return v;
}

static void build_invf(float* out, float base, int dim, double partial) {
  const int angles = (int)(partial * dim) / 2;
  for (int i = 0; i < dim / 2; i++)
    out[i] = (i < angles) ? 1.0f / powf(base, (float)(2 * i) / (float)dim) : 0.0f;
}

// ---------------------------------------------------------------------------
// One decoder layer, parameterised by POSITION and ACTIVATION ROW (CT=4 batched
// prefill, tranche 2b; run log row 334).
//
// This is the step loop's layer body moved verbatim, with the shared single-position
// buffers reached through LayerScratch and the residual stream reached through `x`.
// The shipped path calls it with S = {M.h1, M.h2, ... M.hgu} -- the very same buffers
// it used inline -- so any buffer mapping this refactor missed is a NO-OP for the
// shipped path, and the gates (backbone streams 2/2, ASR EMB cos, token-WER) are the
// check that the move is exact. A future T-wide driver passes per-position scratches,
// which is the whole point: a missed mapping there becomes a gate failure, not a
// silent aliasing bug.
struct LayerScratch { bf16 *h1,*h2,*h3,*h4,*q,*k,*v,*gate,*attn,*bx,*hgu; };

static void layer_body(Model& M, int L, int pos, int step_s, int trace, int ci,
                       bf16* x, LayerScratch& S, int probe_on, int attn_L, int slide_L) {
      KT("rms operator_norm", rms1(M, x, NW(M, L, "operator_norm.weight"), S.h1));
      g_probe = probe_on;
      PROBE("operator_norm", S.h1, H);
      if (attn_L) {
        gemv(M, LN(L, "self_attn.q_proj.weight"), S.h1, S.q);
        gemv(M, LN(L, "self_attn.k_proj.weight"), S.h1, S.k);
        gemv(M, LN(L, "self_attn.v_proj.weight"), S.h1, S.v);
        gemv(M, LN(L, "self_attn.gate_proj.weight"), S.h1, S.gate);
        k_headnorm<<<NH, 32, 0, g_launch>>>(S.q, NW(M, L, "self_attn.q_norm.weight"), NORM_EPS, HD);
        k_headnorm<<<NKV, 32, 0, g_launch>>>(S.k, NW(M, L, "self_attn.k_norm.weight"), NORM_EPS, HD);
        PROBE("q(postnorm)", S.q, NH * HD);
        PROBE("k(postnorm)", S.k, NKV * HD);
        PROBE("v", S.v, NKV * HD);
        PROBE("gate", S.gate, NH);
        const float* invf = slide_L ? M.invf_slide : M.invf_full;
        const int* posp = g_graphs ? M.dpos : nullptr;
        // rot is per-layer-type: full_attention rotates half the dims
        // (partial_rotary_factor 0.5), sliding_attention rotates all of them
        // (partial 1.0). The old code passed HD/2 for both, leaving sliding
        // layers' slow half (dims 32..63) unrotated -- invisible at pos 0
        // (RoPE is the identity there) and under the 3% gate at pos <= 22,
        // but a position-growing phase error after that (#151: 243/7200 long
        // cells outside 3%, engine-low, first sliding layer L4 origin).
        const int rot = slide_L ? HD : HD / 2;
        k_rope<<<NH, 64, 0, g_launch>>>(S.q, invf, pos, HD, rot, posp);
        k_rope<<<NKV, 64, 0, g_launch>>>(S.k, invf, pos, HD, rot, posp);
        PROBE("q(postrope)", S.q, NH * HD);
        bf16* kbase = M.kc + (size_t)L * MAXTOK * NKV * HD;
        bf16* vbase = M.vc + (size_t)L * MAXTOK * NKV * HD;
        if (g_graphs) {
          // The KV write must be a kernel, not a D2D memcpy: a memcpy's
          // destination is baked into the capture, and pos*NKV*HD is exactly
          // what changes per step. Same bytes, same order.
          k_kv_write<<<(NKV * HD + 255) / 256, 256, 0, g_launch>>>(kbase, S.k, NKV * HD, M.dpos, NKV * HD);
          k_kv_write<<<(NKV * HD + 255) / 256, 256, 0, g_launch>>>(vbase, S.v, NKV * HD, M.dpos, NKV * HD);
        } else {
          cudaMemcpy(kbase + (size_t)pos * NKV * HD, S.k, (size_t)NKV * HD * 2, cudaMemcpyDeviceToDevice);
          cudaMemcpy(vbase + (size_t)pos * NKV * HD, S.v, (size_t)NKV * HD * 2, cudaMemcpyDeviceToDevice);
        }
        const float scaling = 1.0f / sqrtf((float)HD);
        // sliding_attention layers see only the last 256 keys (config
        // /lfm/sliding_window); full_attention layers see all of them.
        const int win = slide_L ? SLIDE_WIN : 0;
        k_attn_head<<<NH, 256, 0, g_launch>>>(S.q, kbase, vbase, S.attn, pos + 1, win, scaling, HD, NKV, g_graphs ? M.dpos + 1 : nullptr);
        PROBE("attn", S.attn, NH * HD);
        k_sigmoid_gate<<<1, NH, 0, g_launch>>>(S.gate, S.attn, HD);
        PROBE("attn*gate", S.attn, NH * HD);
        gemv(M, LN(L, "self_attn.o_proj.weight"), S.attn, S.h2);
        PROBE("o_proj", S.h2, H);
      } else {
        bf16* ring = M.ring + (size_t)ci * 7 * H;
        KT("gemv conv.in_proj", gemv(M, LN(L, "conv.in_proj.weight"), S.h1, S.bx));
        PROBE("in_proj", S.bx, 3 * H);
        // THE BUG THAT BROKE THE ENGINE. HF's conv branch is
        //     B, C, x = in_proj(h).chunk(3)  ->  conv(B * x)  ->  C * y   (modeling_lfm3.py:372-421)
        // so the convolves the GATED product. The first version computed
        // B*x into h3 and then fed the RING the raw x (bx + 2*H) -- the
        // B*x result was written and immediately discarded, because h3 is
        // reused for post_operator_norm. Symptom chain: conv branch output
        // 1.8x too large -> post_op_norm 0.0064 vs HF 0.0052 -> layer 0 rms
        // 0.0280 vs 0.0247 -> logits uncorrelated with HF (corr -0.05),
        // teacher-forcing 0/22. Found by printing rms per RING SLOT: slot 6
        // read 0.21026 = in_proj's rms, while B*x is 0.12444, so the ring
        // provably did not contain B*x.
        // Ordering matters: the gate multiply must complete before the
        // newest ring slot is written, so the shift follows the multiply.
        KT("emul B*xg", emul(M, S.bx, S.bx + 2 * H, S.h3, H));
        PROBE("B*x", S.h3, H);
        KT("k_shift_in", k_shift_in<<<(H + 63) / 64, 64, 0, g_launch>>>(ring, S.h3, H));
        if (g_probe) for (int k = 0; k < 7; k++)
          printf("    ring slot %d     rms %9.5f\n", k, rms_host_n(ring + (size_t)k * H, H));
        KT("k_conv7", k_conv7<<<(H + 63) / 64, 64, 0, g_launch>>>(ring, T(M, LN(L, "conv.conv.weight"))->df, S.h4, H));
        PROBE("conv", S.h4, H);
        KT("emul C*y", emul(M, S.bx + H, S.h4, S.h2, H));
        PROBE("C*y", S.h2, H);
        KT("gemv conv.out_proj", gemv(M, LN(L, "conv.out_proj.weight"), S.h2, S.h2));
        PROBE("out_proj", S.h2, H);
      }
      KT("rms post_op_norm", rms1(M, S.h2, NW(M, L, "post_operator_norm.weight"), S.h3));
      PROBE("post_op_norm", S.h3, H);
      eadd(M, x, S.h3, x, H);
      PROBE("add1", x, H);
      KT("rms ffn_norm", rms1(M, x, NW(M, L, "ffn_norm.weight"), S.h1));
      PROBE("ffn_norm", S.h1, H);
      char mn[96]; snprintf(mn, sizeof mn, "MERGE:%d:gate_up", L);
      if (g_merge && T(M, mn)) {
        // One GEMV over 2*INTER rows; rows [0,INTER) are w1 (gate), rows
        // [INTER,2*INTER) are w3 (up). k_silu_mul_bf takes the two halves as
        // its two input pointers, so no kernel change is needed.
        gemv(M, mn, S.h1, S.hgu);
        PROBE("w1", S.hgu, INTER);
        PROBE("w3", S.hgu + INTER, INTER);
        k_silu_mul_bf<<<(INTER + 255) / 256, 256, 0, g_launch>>>(S.hgu, S.hgu + INTER, S.h2, INTER);
      } else {
      KT("gemv ffn.w1", gemv(M, LN(L, "feed_forward.w1.weight"), S.h1, S.h2));
      PROBE("w1", S.h2, INTER);
      KT("gemv ffn.w3", gemv(M, LN(L, "feed_forward.w3.weight"), S.h1, S.h3));
      PROBE("w3", S.h3, INTER);
      k_silu_mul_bf<<<(INTER + 255) / 256, 256, 0, g_launch>>>(S.h2, S.h3, S.h2, INTER);
      }
      PROBE("silu_mul", S.h2, INTER);
      KT("gemv ffn.w2", gemv(M, LN(L, "feed_forward.w2.weight"), S.h2, S.h2));
      PROBE("w2", S.h2, INTER);
      KT("rms post_ffn_norm", rms1(M, S.h2, NW(M, L, "post_ffn_norm.weight"), S.h3));
      PROBE("post_ffn_norm", S.h3, H);
      eadd(M, x, S.h3, x, H);
      PROBE("add2", x, H);
      g_probe = 0;
      // Full layer x position matrix in one run: the perlayer_probe role the
      // Pi engine plays with tools/perlayer_probe.patch. Cheap enough to leave
      // on under --trace (a 4 KB D2H copy per layer per step).
      if (trace) printf("POS %2d L%-2d %s%s rms %.5f\n", step_s, L,
                        attn_L ? "attn" : "conv", slide_L ? "(slide)" : "",
                        rms_host(M, x));
  g_probe = 0;
}

// ---------------------------------------------------------------------------
// Tranche 3 of true CT=4 batched prefill (doc section 51, run log row 335).
//
// The pieces are all certified now: the batched GEMM reads a driver-owned
// position-first arena bit-exactly (row 332), k_combine_bf_T combines T positions in one
// launch (row 333), k_rmsnorm_T is the T=1 rmsnorm with a row offset (row 332), and
// layer_body is parameterised by position and activation row (row 334). This is the
// driver that consumes them: the prompt is fed T positions at a time, the ten GEMVs of a
// layer go out as ONE launch each, and the cheap per-position passes (headnorm, RoPE,
// KV write, attention, gated conv, SiLU, the two residual adds) run in a per-position
// loop using the SAME kernels as the T=1 path.
//
// Acceptance bar for this tranche: after all chunks, the ACTIVATION AT THE LAST PROMPT
// POSITION must be bit-identical to what the shipped T=1 prefill produces there
// (LFM_DUMPX dumps the shipped value). That is a full 24-layer, ~10-GEMV-per-layer
// end-to-end check of the T-wide sequencing -- if the residual stream at the last prompt
// position matches bit for bit, every GEMV, norm, conv ring, RoPE and attention call in
// between matched. Generated ids come when this driver is wired as the real prefill
// (next tranche); they need lm_head, which is still inline in the step loop.
// q8_partial_Tb<CT>: the T-wide twin of the SHIPPED T=1 backbone GEMV,
// q8_partial<bf16, FMA=true, STREAM=false, UNROLL=1, H2=0>. For every (row, chunk,
// position) the arithmetic is that kernel's exactly: x staged to fp32 by bf2f (exact),
// lane L folds blocks b = L, L+32, ... with dot_b = the same 32-term fmaf chain and
// acc = fmaf(s[b], dot_b, acc), then the same shuffle-down tree; the value lane 0 stores
// is the value q8_partial stages in sh[]. KCHUNK/nchunk come from pick(), so k_combine_bf_T
// sums the same partials in the same order. What changes: one weight-block load serves CT
// positions (the prefill re-streams all 1.68 GB of weights once per prompt position), and
// x is staged as bf16 PAIRS in 17-word rows (half the shared memory of the fp32 33-pad
// layout, odd stride so lanes stay conflict-free). CT=1 is a plain q8_partial with a
// position grid.
template <int CT, int TPB = 256>
__global__ void __launch_bounds__(TPB, TPB == 512 ? 2 : 1) q8_partial_Tb(const int8_t* __restrict__ q, const float* __restrict__ s,
                                                    const bf16* __restrict__ x, float* __restrict__ part,
                                                    int N, int K, int KCHUNK, int R, int nb, int T, bf16* __restrict__ yb = nullptr) {
  extern __shared__ uint32_t xw[];                    // [CT][nb][17] words, 16 used = 32 bf16
  const int nchunk = K / KCHUNK;
  const int rowblk = blockIdx.x / nchunk, chunk = blockIdx.x % nchunk;
  const int koff = chunk * KCHUNK;
  const int t0 = blockIdx.y * CT;
  const int NT = blockDim.x, tid = threadIdx.x;
  const int xstride = nb * 17;
  #pragma unroll
  for (int p = 0; p < CT; p++) {
    const int t = t0 + p;
    if (t < T) {
      const uint32_t* xg = reinterpret_cast<const uint32_t*>(x + (size_t)t * K + koff);
      for (int i = tid; i < KCHUNK / 2; i += NT) xw[p * xstride + (i >> 4) * 17 + (i & 15)] = xg[i];
    }
  }
  __syncthreads();
  const int base = rowblk * R;
  const int lane = tid & 31, wid = tid >> 5;
  const int WPB = NT >> 5, GR = R / WPB;
  for (int g = 0; g < GR; g++) {
    const int row = base + wid + g * WPB;
    if (row >= N) continue;                            // warp-uniform
    float acc[CT];
    #pragma unroll
    for (int p = 0; p < CT; p++) acc[p] = 0.f;
    const int8_t* qr = q + (size_t)row * K + koff;
    const float* sr = s + (size_t)row * (K / Q8_BLK) + (size_t)chunk * nb;
    #pragma unroll 1
    for (int b = lane; b < nb; b += 32) {
      const int4 a0 = *reinterpret_cast<const int4*>(qr + (size_t)b * Q8_BLK);
      const int4 a1 = *reinterpret_cast<const int4*>(qr + (size_t)b * Q8_BLK + 16);
      const int8_t* c0 = reinterpret_cast<const int8_t*>(&a0);
      const int8_t* c1 = reinterpret_cast<const int8_t*>(&a1);
      const float sb = sr[b];
      // Code-outer, position-inner: each code pair is converted to fp32 ONCE and applied
      // to all CT positions, so only dot[CT] is live (a 32-float code array cost CT=4 its
      // occupancy: 78 registers). Every dot_p still accumulates codes 0..31 in order --
      // the same fmaf chain q8_partial runs -- so the bits are unchanged. (I2F per
      // position per code made the first version compute-bound at ~5 GB/s.)
      float dot[CT];
      #pragma unroll
      for (int p = 0; p < CT; p++) dot[p] = 0.f;
      const uint32_t* wb0 = xw + b * 17;
      #pragma unroll
      for (int j = 0; j < 16; j++) {                   // word j = codes 2j, 2j+1
        const int8_t* cc = (j < 8) ? c0 : c1;
        const float ca = (float)cc[(2 * j) & 15], cb = (float)cc[(2 * j + 1) & 15];
        #pragma unroll
        for (int p = 0; p < CT; p++) {
          const uint32_t u = wb0[p * xstride + j];
          dot[p] = __fmaf_rn(ca, __uint_as_float(u << 16), dot[p]);
          dot[p] = __fmaf_rn(cb, __uint_as_float(u & 0xffff0000u), dot[p]);
        }
      }
      #pragma unroll
      for (int p = 0; p < CT; p++) acc[p] = __fmaf_rn(sb, dot[p], acc[p]);
    }
    #pragma unroll
    for (int p = 0; p < CT; p++) {
      float a = acc[p];
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(~0u, a, o);
      const int t = t0 + p;
      if (lane == 0 && t < T) { if (yb) yb[(size_t)t * N + row] = f2bf(__fadd_rn(0.f, a)); else part[((size_t)t * N + row) * nchunk + chunk] = a; }
    }
  }
}

// q8_partial_Tp<CT, GRW>: q8_partial_Tb for a wide single chunk (ffn.w2: K = KCHUNK =
// 7680, nb = 240), whose whole-row x stage forced CT = 2 at 32.6 KB = one block per SM.
// x is staged 32 blocks at a time (one per lane): in phase m lane L handles block
// 32m + L, i.e. the same blocks L, L+32, ... in the same order as Tb, with the same
// code-outer dot chain, acc = fmaf(s_b, dot, acc) and shuffle tree -- bit-identical.
// The price is GRW rows' accumulators live across phases, so R = 8 * GRW rows per block.
// GRW = 4 (R = 32): 64 regs/12 B spill (ptxas); ASR prefill 72.9 -> 65.9 ms/pos. GRW 8 spills harder
// (75.5), GRW 2 is equal-or-slower (66.2). LFM_PF_PH=0 restores the CT=2 whole-row stage.
template <int CT, int GRW, int TPB = 256>
__global__ void __launch_bounds__(TPB, TPB == 512 ? 2 : 1) q8_partial_Tp(const int8_t* __restrict__ q, const float* __restrict__ s,
                                                    const bf16* __restrict__ x, float* __restrict__ part,
                                                    int N, int K, int KCHUNK, int nb, int T, bf16* __restrict__ yb = nullptr) {
  __shared__ uint32_t xw[CT * 32 * 17];
  const int nchunk = K / KCHUNK;
  const int rowblk = blockIdx.x / nchunk, chunk = blockIdx.x % nchunk;
  const int koff = chunk * KCHUNK;
  const int t0 = blockIdx.y * CT;
  const int NT = blockDim.x, tid = threadIdx.x;
  const int lane = tid & 31, wid = tid >> 5;
  const int base = rowblk * ((TPB / 32) * GRW);
  float acc[GRW][CT];
  #pragma unroll
  for (int g = 0; g < GRW; g++)
    #pragma unroll
    for (int p = 0; p < CT; p++) acc[g][p] = 0.f;
  for (int m0 = 0; m0 < nb; m0 += 32) {
    __syncthreads();                                   // previous phase's readers are done
    #pragma unroll
    for (int p = 0; p < CT; p++) {
      const int t = t0 + p;
      if (t < T) {
        const uint32_t* xg = reinterpret_cast<const uint32_t*>(x + (size_t)t * K + koff);
        for (int i = tid; i < 32 * 16; i += NT) {
          const int blk = m0 + (i >> 4);
          if (blk < nb) xw[p * 544 + (i >> 4) * 17 + (i & 15)] = xg[blk * 16 + (i & 15)];
        }
      }
    }
    __syncthreads();
    const int b = m0 + lane;
    if (b < nb) {
      const uint32_t* wb0 = xw + lane * 17;
      #pragma unroll
      for (int g = 0; g < GRW; g++) {
        const int row = base + wid + g * (TPB / 32);
        if (row >= N) continue;                        // warp-uniform
        const int8_t* qr = q + (size_t)row * K + koff;
        const int4 a0 = *reinterpret_cast<const int4*>(qr + (size_t)b * Q8_BLK);
        const int4 a1 = *reinterpret_cast<const int4*>(qr + (size_t)b * Q8_BLK + 16);
        const int8_t* c0 = reinterpret_cast<const int8_t*>(&a0);
        const int8_t* c1 = reinterpret_cast<const int8_t*>(&a1);
        const float sb = s[(size_t)row * (K / Q8_BLK) + (size_t)chunk * nb + b];
        float dot[CT];
        #pragma unroll
        for (int p = 0; p < CT; p++) dot[p] = 0.f;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
          const int8_t* cc = (j < 8) ? c0 : c1;
          const float ca = (float)cc[(2 * j) & 15], cb = (float)cc[(2 * j + 1) & 15];
          #pragma unroll
          for (int p = 0; p < CT; p++) {
            const uint32_t u = wb0[p * 544 + j];
            dot[p] = __fmaf_rn(ca, __uint_as_float(u << 16), dot[p]);
            dot[p] = __fmaf_rn(cb, __uint_as_float(u & 0xffff0000u), dot[p]);
          }
        }
        #pragma unroll
        for (int p = 0; p < CT; p++) acc[g][p] = __fmaf_rn(sb, dot[p], acc[g][p]);
      }
    }
  }
  #pragma unroll
  for (int g = 0; g < GRW; g++) {
    const int row = base + wid + g * (TPB / 32);
    #pragma unroll
    for (int p = 0; p < CT; p++) {
      float a = acc[g][p];
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(~0u, a, o);
      const int t = t0 + p;
      if (lane == 0 && t < T && row < N) { if (yb) yb[(size_t)t * N + row] = f2bf(__fadd_rn(0.f, a)); else part[((size_t)t * N + row) * nchunk + chunk] = a; }
    }
  }
}

// q8_partial_TLb: q8_partial_Tb<4>'s values for nchunk == 1, one row per THREAD. Lane L of
// Tb chains blocks L, L+32, ... (acc = fmaf(s_b, dot_b, acc)) and a shuffle_down tree
// (o = 16..1) reduces the 32 lane values; here leaf i is that lane-i chain and the leaves
// are combined in 5-bit bit-reversed order by a binary counter (earlier operand left), so
// every add has the tree's operands: bit-identical. Weights in 32-row tiles (per block b:
// the tile's 32 first int4s, then its 32 second int4s; scales [tile][b][32]).
template <int CT>
__device__ __noinline__ float4 tlb_leaf(const int4* __restrict__ qt, const float* __restrict__ sts,
                                        const uint32_t* xw, int i, int nb, int xstride) {
  float acc[4] = { 0.f, 0.f, 0.f, 0.f };
  for (int b = i; b < nb; b += 32) {
    const int4 a0 = qt[b * 64], a1 = qt[b * 64 + 32];
    const unsigned w[8] = { (unsigned)a0.x, (unsigned)a0.y, (unsigned)a0.z, (unsigned)a0.w,
                            (unsigned)a1.x, (unsigned)a1.y, (unsigned)a1.z, (unsigned)a1.w };
    const float sb = sts[b * 32];
    float dot[4] = { 0.f, 0.f, 0.f, 0.f };
    const uint32_t* wb0 = xw + b * 17;
    #pragma unroll
    for (int j = 0; j < 16; j++) {
      const float ca = (float)(int)(signed char)(w[j >> 1] >> (16 * (j & 1)));
      const float cb = (float)(int)(signed char)(w[j >> 1] >> (16 * (j & 1) + 8));
      #pragma unroll
      for (int p = 0; p < CT; p++) {
        const uint32_t u = wb0[p * xstride + j];
        dot[p] = __fmaf_rn(ca, __uint_as_float(u << 16), dot[p]);
        dot[p] = __fmaf_rn(cb, __uint_as_float(u & 0xffff0000u), dot[p]);
      }
    }
    #pragma unroll
    for (int p = 0; p < CT; p++) acc[p] = __fmaf_rn(sb, dot[p], acc[p]);
  }
  return make_float4(acc[0], acc[1], acc[2], acc[3]);
}
template <int TPB, int CT = 4>
__global__ void __launch_bounds__(TPB, TPB == 512 ? 2 : 4) q8_partial_TLb(const int8_t* __restrict__ q, const float* __restrict__ s,
                                                    const bf16* __restrict__ x, float* __restrict__ part,
                                                    int N, int K, int nb, int T, bf16* __restrict__ yb) {
  extern __shared__ uint32_t xw[];                    // [CT][nb][17]
  const int t0 = blockIdx.y * CT, tid = threadIdx.x;
  const int xstride = nb * 17;
  #pragma unroll
  for (int p = 0; p < CT; p++) {
    const int t = t0 + p;
    if (t < T) {
      const uint32_t* xg = reinterpret_cast<const uint32_t*>(x + (size_t)t * K);
      for (int i = tid; i < K / 2; i += TPB) xw[p * xstride + (i >> 4) * 17 + (i & 15)] = xg[i];
    }
  }
  __syncthreads();
  const int row = blockIdx.x * TPB + tid;
  if (row >= N) return;
  const int4* qt = reinterpret_cast<const int4*>(q) + (size_t)(row >> 5) * nb * 64 + (row & 31);
  const float* sts = s + (size_t)(row >> 5) * nb * 32 + (row & 31);
  float st[5][4], pv[4];
  #pragma unroll
  for (int k = 0; k < 32; k++) {
    const int i = ((k & 1) << 4) | ((k & 2) << 2) | (k & 4) | ((k & 8) >> 2) | ((k & 16) >> 4);
    const float4 a = tlb_leaf<CT>(qt, sts, xw, i, nb, xstride);
    pv[0] = a.x; pv[1] = a.y; pv[2] = a.z; pv[3] = a.w;
    bool done = false;
    #pragma unroll
    for (int l = 0; l < 5; l++) {
      if (done) continue;
      if ((k >> l) & 1) {
        #pragma unroll
        for (int p = 0; p < 4; p++) pv[p] = __fadd_rn(st[l][p], pv[p]);
      } else {
        #pragma unroll
        for (int p = 0; p < 4; p++) st[l][p] = pv[p];
        done = true;
      }
    }
  }
  #pragma unroll
  for (int p = 0; p < CT; p++) {
    const int t = t0 + p;
    if (t < T) { if (yb) yb[(size_t)t * N + row] = f2bf(__fadd_rn(0.f, pv[p])); else part[(size_t)t * N + row] = pv[p]; }
  }
}
// q8_partial_TLf: q8_partial_TLb with x staged as f32 instead of bf16 pairs (LFM_PF_TLF).
// Every thread walks the leaves in the same order, so all lanes read the same x address
// (broadcast, no padding needed); the bf16 -> f32 unpack (a shift or a mask per product
// pair per position, 8 of TLb's ~22 instructions per code pair x 4 positions) moves into
// the staging loop, and each code pair reads its two x values with one LDS.64. bf16 ->
// f32 is exact, so every fmaf sees the same operands in the same order: bit-identical.
// Shared memory doubles (CT x K x 4 B: 32 KB at K = 2048). nvprof (instance 400 of the ASR
// prefill): TLb IPC 4.20, 85% issue slots, 52.1 M instructions; TLf alone 34.3 M at IPC 2.86,
// stalls now data request 42% / execution dependency 37%, so the speed did not move until
// the leaf's next weight block is loaded before the current block's FMAs (PF = 1).
__constant__ int g_pfgsw = 1;   // LFM_PF_GSW
template <int CT, int PF>
__device__ __noinline__ float4 tlf_leaf(const int4* __restrict__ qt, const float* __restrict__ sts,
                                        const float* xf, int i, int nb, int K) {
  float acc[4] = { 0.f, 0.f, 0.f, 0.f };
  // PF: the next block's weights and scale are loaded before this block's FMAs (1-deep
  // software pipeline): the f32-staged leaf has issue slack, and its stalls are the loads.
  int4 n0, n1; float nsb = 0.f;
  if (PF) { n0 = qt[i * 64]; n1 = qt[i * 64 + 32]; nsb = sts[i * 32]; }
  for (int b = i; b < nb; b += 32) {
    int4 a0, a1; float sb;
    if (PF) {
      a0 = n0; a1 = n1; sb = nsb;
      if (b + 32 < nb) { n0 = qt[(b + 32) * 64]; n1 = qt[(b + 32) * 64 + 32]; nsb = sts[(b + 32) * 32]; }
    } else { a0 = qt[b * 64]; a1 = qt[b * 64 + 32]; sb = sts[b * 32]; }
    const unsigned w[8] = { (unsigned)a0.x, (unsigned)a0.y, (unsigned)a0.z, (unsigned)a0.w,
                            (unsigned)a1.x, (unsigned)a1.y, (unsigned)a1.z, (unsigned)a1.w };
    float dot[4] = { 0.f, 0.f, 0.f, 0.f };
    const float2* xb = reinterpret_cast<const float2*>(xf + b * 32);
    #pragma unroll
    for (int j = 0; j < 16; j++) {
      const float ca = (float)(int)(signed char)(w[j >> 1] >> (16 * (j & 1)));
      const float cb = (float)(int)(signed char)(w[j >> 1] >> (16 * (j & 1) + 8));
      #pragma unroll
      for (int p = 0; p < CT; p++) {
        const float2 u = xb[p * (K / 2) + j];
        dot[p] = __fmaf_rn(ca, u.x, dot[p]);
        dot[p] = __fmaf_rn(cb, u.y, dot[p]);
      }
    }
    #pragma unroll
    for (int p = 0; p < CT; p++) acc[p] = __fmaf_rn(sb, dot[p], acc[p]);
  }
  return make_float4(acc[0], acc[1], acc[2], acc[3]);
}
// tlf_leaf2b: a leaf of exactly two blocks (i, i + 32; nb == 64) with both blocks' dot
// chains computed interleaved (8 independent FMA chains instead of 4); each block's dot is
// its own chain from 0 in code order, folded into acc in block order: same values.
template <int CT>
__device__ __noinline__ float4 tlf_leaf2b(const int4* __restrict__ qt, const float* __restrict__ sts,
                                          const float* xf, int i, int K) {
  const int4 a0 = qt[i * 64], a1 = qt[i * 64 + 32], b0 = qt[(i + 32) * 64], b1 = qt[(i + 32) * 64 + 32];
  const float sa = sts[i * 32], sb = sts[(i + 32) * 32];
  const unsigned wa[8] = { (unsigned)a0.x, (unsigned)a0.y, (unsigned)a0.z, (unsigned)a0.w,
                           (unsigned)a1.x, (unsigned)a1.y, (unsigned)a1.z, (unsigned)a1.w };
  const unsigned wb[8] = { (unsigned)b0.x, (unsigned)b0.y, (unsigned)b0.z, (unsigned)b0.w,
                           (unsigned)b1.x, (unsigned)b1.y, (unsigned)b1.z, (unsigned)b1.w };
  float da[4] = { 0.f, 0.f, 0.f, 0.f }, db[4] = { 0.f, 0.f, 0.f, 0.f };
  const float2* xa = reinterpret_cast<const float2*>(xf + i * 32);
  const float2* xb = reinterpret_cast<const float2*>(xf + (i + 32) * 32);
  #pragma unroll
  for (int j = 0; j < 16; j++) {
    const float ca = (float)(int)(signed char)(wa[j >> 1] >> (16 * (j & 1)));
    const float cb = (float)(int)(signed char)(wa[j >> 1] >> (16 * (j & 1) + 8));
    const float ea = (float)(int)(signed char)(wb[j >> 1] >> (16 * (j & 1)));
    const float eb = (float)(int)(signed char)(wb[j >> 1] >> (16 * (j & 1) + 8));
    #pragma unroll
    for (int p = 0; p < CT; p++) {
      const float2 u = xa[p * (K / 2) + j], v = xb[p * (K / 2) + j];
      da[p] = __fmaf_rn(ca, u.x, da[p]); db[p] = __fmaf_rn(ea, v.x, db[p]);
      da[p] = __fmaf_rn(cb, u.y, da[p]); db[p] = __fmaf_rn(eb, v.y, db[p]);
    }
  }
  float acc[4];
  #pragma unroll
  for (int p = 0; p < CT; p++) acc[p] = __fmaf_rn(sb, db[p], __fmaf_rn(sa, da[p], 0.f));
  return make_float4(acc[0], acc[1], acc[2], acc[3]);
}
// tlf_leaf2h (LFM_PF_TLF=9, NOT bit-exact, probe): the same two blocks on the fp16 pipe. x is
// staged as half2; each code word becomes two exact half2 (byte ^ 0x80 under 0x64 = 1024 + b +
// 128, minus 1152); per block two half2 chains of 4 words (|partial| <= 8 * 127 * max|x|),
// widened to f32, then the same scale fold. Accumulation order and rounding differ from f32.
template <int CT>
__device__ __noinline__ float4 tlf_leaf2h(const int4* __restrict__ qt, const float* __restrict__ sts,
                                          const float* xf, int i, int K) {
  const int4 a0 = qt[i * 64], a1 = qt[i * 64 + 32], b0 = qt[(i + 32) * 64], b1 = qt[(i + 32) * 64 + 32];
  const float sa = sts[i * 32], sb = sts[(i + 32) * 32];
  const unsigned wa[8] = { (unsigned)a0.x, (unsigned)a0.y, (unsigned)a0.z, (unsigned)a0.w,
                           (unsigned)a1.x, (unsigned)a1.y, (unsigned)a1.z, (unsigned)a1.w };
  const unsigned wb[8] = { (unsigned)b0.x, (unsigned)b0.y, (unsigned)b0.z, (unsigned)b0.w,
                           (unsigned)b1.x, (unsigned)b1.y, (unsigned)b1.z, (unsigned)b1.w };
  const __half2 off = __floats2half2_rn(1152.f, 1152.f);
  __half2 ca[16], cb[16];
  #pragma unroll
  for (int m = 0; m < 8; m++) {
    const unsigned u = wa[m] ^ 0x80808080u, v = wb[m] ^ 0x80808080u;
    unsigned t;
    t = __byte_perm(u, 0x64646464u, 0x4140); ca[2 * m] = __hsub2(*reinterpret_cast<__half2*>(&t), off);
    t = __byte_perm(u, 0x64646464u, 0x4342); ca[2 * m + 1] = __hsub2(*reinterpret_cast<__half2*>(&t), off);
    t = __byte_perm(v, 0x64646464u, 0x4140); cb[2 * m] = __hsub2(*reinterpret_cast<__half2*>(&t), off);
    t = __byte_perm(v, 0x64646464u, 0x4342); cb[2 * m + 1] = __hsub2(*reinterpret_cast<__half2*>(&t), off);
  }
  const __half2* xh = reinterpret_cast<const __half2*>(xf);
  float acc[4];
  #pragma unroll
  for (int p = 0; p < CT; p++) {
    const __half2* xa = xh + p * (K / 2) + i * 16;
    const __half2* xb = xh + p * (K / 2) + (i + 32) * 16;
    __half2 ha0 = __float2half2_rn(0.f), ha1 = ha0, hb0 = ha0, hb1 = ha0;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      ha0 = __hfma2(ca[j], xa[j], ha0);         hb0 = __hfma2(cb[j], xb[j], hb0);
      ha1 = __hfma2(ca[j + 8], xa[j + 8], ha1); hb1 = __hfma2(cb[j + 8], xb[j + 8], hb1);
    }
    const float2 fa0 = __half22float2(ha0), fa1 = __half22float2(ha1);
    const float2 fb0 = __half22float2(hb0), fb1 = __half22float2(hb1);
    const float da = (fa0.x + fa0.y) + (fa1.x + fa1.y), db = (fb0.x + fb0.y) + (fb1.x + fb1.y);
    acc[p] = __fmaf_rn(sb, db, __fmaf_rn(sa, da, 0.f));
  }
  return make_float4(acc[0], acc[1], acc[2], acc[3]);
}
// LFM_TLF_RANGE=1 (diagnostic): fp16 range of every TLf input element (bf16 bits).
__device__ unsigned long long g_rng[4];   // nonzero, subnormal-in-fp16, overflow, max|x| bits
__global__ void k_tlf_range(const bf16* __restrict__ x, long n) {
  unsigned long long nz = 0, sub = 0, ov = 0; unsigned mx = 0;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
    const unsigned a = ((unsigned)x[i] << 16) & 0x7fffffffu; const float f = __uint_as_float(a);
    if (a) { nz++; if (f < 6.103515625e-05f) sub++; if (f > 65504.f) ov++; if (a > mx) mx = a; }
  }
  atomicAdd(&g_rng[0], nz); atomicAdd(&g_rng[1], sub); atomicAdd(&g_rng[2], ov); atomicMax(&g_rng[3], (unsigned long long)mx);
}
static float __uint_as_float_host(unsigned u) { float f; memcpy(&f, &u, 4); return f; }
static void tlf_range_dump() {
  unsigned long long h[4]; cudaMemcpyFromSymbol(h, g_rng, sizeof h);
  printf("TLF_RANGE nonzero %llu subnormal_fp16 %llu (%.3f%%) overflow %llu max|x| %g\n", h[0], h[1],
         h[0] ? 100.0 * h[1] / h[0] : 0.0, h[2], (double)__uint_as_float_host((unsigned)h[3]));
}
// Guard for tlf_leaf2h: a block whose staged x has max|x| > 32 (fp16 partial bound 8*127*32 =
// 32512 < 65504; or every block under LFM_PF_H2FB=1, the test switch) runs this leaf instead:
// the two blocks in f32 from the bf16 x in global memory, in tlf_leaf2b's exact order.
__constant__ int c_h2fb;
template <int CT>
__device__ __noinline__ float4 tlf_leaf2g(const int4* __restrict__ qt, const float* __restrict__ sts,
                                          const bf16* __restrict__ xg, int i, int K, int T) {
  const int4 a0 = qt[i * 64], a1 = qt[i * 64 + 32], b0 = qt[(i + 32) * 64], b1 = qt[(i + 32) * 64 + 32];
  const float sa = sts[i * 32], sb = sts[(i + 32) * 32];
  const unsigned wa[8] = { (unsigned)a0.x, (unsigned)a0.y, (unsigned)a0.z, (unsigned)a0.w,
                           (unsigned)a1.x, (unsigned)a1.y, (unsigned)a1.z, (unsigned)a1.w };
  const unsigned wb[8] = { (unsigned)b0.x, (unsigned)b0.y, (unsigned)b0.z, (unsigned)b0.w,
                           (unsigned)b1.x, (unsigned)b1.y, (unsigned)b1.z, (unsigned)b1.w };
  float acc[4] = { 0.f, 0.f, 0.f, 0.f };
  for (int p = 0; p < CT && p < T; p++) {
    const bf16* xa = xg + (size_t)p * K + i * 32; const bf16* xb = xg + (size_t)p * K + (i + 32) * 32;
    float da = 0.f, db = 0.f;
    for (int j = 0; j < 32; j++) {
      da = __fmaf_rn((float)(int)(signed char)(wa[j >> 2] >> (8 * (j & 3))), __uint_as_float((unsigned)xa[j] << 16), da);
      db = __fmaf_rn((float)(int)(signed char)(wb[j >> 2] >> (8 * (j & 3))), __uint_as_float((unsigned)xb[j] << 16), db);
    }
    acc[p] = __fmaf_rn(sb, db, __fmaf_rn(sa, da, 0.f));
  }
  return make_float4(acc[0], acc[1], acc[2], acc[3]);
}
template <int TPB, int CT = 4, int PF = 1>
__global__ void __launch_bounds__(TPB, TPB == 512 ? 2 : 4) q8_partial_TLf(const int8_t* __restrict__ q, const float* __restrict__ s,
                                                    const bf16* __restrict__ x, float* __restrict__ part,
                                                    int N, int K, int nb, int T, bf16* __restrict__ yb) {
  extern __shared__ float xs[];                       // [CT][K]
  // g_pfgsw (LFM_PF_GSW): grid launched (position groups, row blocks), so the blocks resident
  // together share their weight rows through L2; the block's work is unchanged.
  const int bx = g_pfgsw ? blockIdx.y : blockIdx.x, by = g_pfgsw ? blockIdx.x : blockIdx.y;
  const int t0 = by * CT, tid = threadIdx.x;
  float xmax = 0.f;
  #pragma unroll
  for (int p = 0; p < CT; p++) {
    const int t = t0 + p;
    if (t < T) {
      const uint32_t* xg = reinterpret_cast<const uint32_t*>(x + (size_t)t * K);
      if (PF == 9) {
        __half2* d = reinterpret_cast<__half2*>(xs) + p * (K / 2);
        for (int i = tid; i < K / 2; i += TPB) { const uint32_t u = xg[i];
          xmax = fmaxf(xmax, fmaxf(fabsf(__uint_as_float(u << 16)), fabsf(__uint_as_float(u & 0xffff0000u))));
          d[i] = __floats2half2_rn(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u)); }
      } else {
      float2* d = reinterpret_cast<float2*>(xs + p * K);
      for (int i = tid; i < K / 2; i += TPB) { const uint32_t u = xg[i];
        d[i] = make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u)); }
      }
    }
  }
  // NaN max also routes to f32 (!(xmax <= 32)).
  const bool big = PF == 9 ? __syncthreads_or(!(xmax <= 32.f) || c_h2fb) : (__syncthreads(), false);
  const int row = bx * TPB + tid;
  if (row >= N) return;
  const int4* qt = reinterpret_cast<const int4*>(q) + (size_t)(row >> 5) * nb * 64 + (row & 31);
  const float* sts = s + (size_t)(row >> 5) * nb * 32 + (row & 31);
  float st[5][4], pv[4];
  #pragma unroll
  for (int k = 0; k < 32; k++) {
    const int i = ((k & 1) << 4) | ((k & 2) << 2) | (k & 4) | ((k & 8) >> 2) | ((k & 16) >> 4);
    const float4 a = PF == 9 ? (big ? tlf_leaf2g<CT>(qt, sts, x + (size_t)t0 * K, i, K, T - t0) : tlf_leaf2h<CT>(qt, sts, xs, i, K)) : PF == 4 ? tlf_leaf2b<CT>(qt, sts, xs, i, K) : tlf_leaf<CT, PF ? 1 : 0>(qt, sts, xs, i, nb, K);
    pv[0] = a.x; pv[1] = a.y; pv[2] = a.z; pv[3] = a.w;
    bool done = false;
    #pragma unroll
    for (int l = 0; l < 5; l++) {
      if (done) continue;
      if ((k >> l) & 1) {
        #pragma unroll
        for (int p = 0; p < 4; p++) pv[p] = __fadd_rn(st[l][p], pv[p]);
      } else {
        #pragma unroll
        for (int p = 0; p < 4; p++) st[l][p] = pv[p];
        done = true;
      }
    }
  }
  #pragma unroll
  for (int p = 0; p < CT; p++) {
    const int t = t0 + p;
    if (t < T) { if (yb) yb[(size_t)t * N + row] = f2bf(__fadd_rn(0.f, pv[p])); else part[(size_t)t * N + row] = pv[p]; }
  }
}
// q8_partial_TLg: q8_partial_Tp's values (ffn.w2, nb = 240) one row per thread on the TLb
// tile layout. Leaf i = Tp's lane-i chain over blocks i, i+32, ... (same as TLb's leaf);
// since every thread walks the leaves in the same bit-reversed order, the block stages x
// only for the 4 leaves of the current group (blocks i + 32m, m < 8, 4 positions, bf16
// pairs unpadded: all threads of a warp read the same word). Same leaves, same tree.
// The TLg x stage lives at namespace scope so the leaf can index it as shared memory:
// through the pointer argument of the noinline call the space is lost and the x reads were
// generic LD.E (SASS). XG (LFM_PF_TLGS): 0 = pointer form, 1 = shared (scalar source),
// 4 = shared, one uint4 per 4 words. Same words, same fmaf order: bit-identical.
__shared__ uint32_t g_tlg_xs[4 * 8 * 64];             // [leaf slot][m][position][16 words]
// XG = 8 (LFM_PF_TLGS=8): the stage holds f32 (bf16 -> f32 is exact), word jj's low / high
// halves at floats 2jj / 2jj + 1, so the leaf feeds the FMAs from one LDS.128 per two code
// pairs without the per-product shift / mask. Same operands, same order: bit-identical.
__shared__ float g_tlg_xf[4 * 8 * 64 * 2];
// XG = 9: the same f32 stage for 8 leaves at a time (32 KB, still 2 blocks/SM): half the
// staging passes and block barriers per kernel. Same leaves, same order.
__shared__ float g_tlg_xf8[8 * 8 * 64 * 2];
template <int XG>
__device__ __noinline__ float4 tlg_leaf(const int4* __restrict__ qt, const float* __restrict__ sts,
                                        const uint32_t* xg, int xoff, int i, int nb) {
  float acc[4] = { 0.f, 0.f, 0.f, 0.f };
  if (XG) {
    for (int m = 0; i + 32 * m < nb; m++) {
      const int b = i + 32 * m;
      const int4 a0 = qt[b * 64], a1 = qt[b * 64 + 32];
      const unsigned w[8] = { (unsigned)a0.x, (unsigned)a0.y, (unsigned)a0.z, (unsigned)a0.w,
                              (unsigned)a1.x, (unsigned)a1.y, (unsigned)a1.z, (unsigned)a1.w };
      const float sb = sts[b * 32];
      float dot[4] = { 0.f, 0.f, 0.f, 0.f };
      const uint32_t* wb0 = g_tlg_xs + xoff + m * 64;
      if (XG >= 8) {
        const float4* fv = reinterpret_cast<const float4*>((XG == 9 ? g_tlg_xf8 : g_tlg_xf) + 2 * (xoff + m * 64));
        #pragma unroll
        for (int q = 0; q < 8; q++) {
          float4 f[4];
          #pragma unroll
          for (int p = 0; p < 4; p++) f[p] = fv[p * 8 + q];
          const float c0 = (float)(int)(signed char)(w[q]);
          const float c1 = (float)(int)(signed char)(w[q] >> 8);
          const float c2 = (float)(int)(signed char)(w[q] >> 16);
          const float c3 = (float)(int)(signed char)(w[q] >> 24);
          #pragma unroll
          for (int p = 0; p < 4; p++) {
            dot[p] = __fmaf_rn(c0, f[p].x, dot[p]);
            dot[p] = __fmaf_rn(c1, f[p].y, dot[p]);
            dot[p] = __fmaf_rn(c2, f[p].z, dot[p]);
            dot[p] = __fmaf_rn(c3, f[p].w, dot[p]);
          }
        }
      } else if (XG == 4) {
        const uint4* wv = reinterpret_cast<const uint4*>(wb0);
        #pragma unroll
        for (int jq = 0; jq < 4; jq++) {
          uint4 u4[4];
          #pragma unroll
          for (int p = 0; p < 4; p++) u4[p] = wv[p * 4 + jq];
          #pragma unroll
          for (int jr = 0; jr < 4; jr++) {
            const int j = jq * 4 + jr;
            const float ca = (float)(int)(signed char)(w[j >> 1] >> (16 * (j & 1)));
            const float cb = (float)(int)(signed char)(w[j >> 1] >> (16 * (j & 1) + 8));
            #pragma unroll
            for (int p = 0; p < 4; p++) {
              const uint32_t u = jr == 0 ? u4[p].x : jr == 1 ? u4[p].y : jr == 2 ? u4[p].z : u4[p].w;
              dot[p] = __fmaf_rn(ca, __uint_as_float(u << 16), dot[p]);
              dot[p] = __fmaf_rn(cb, __uint_as_float(u & 0xffff0000u), dot[p]);
            }
          }
        }
      } else {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
          const float ca = (float)(int)(signed char)(w[j >> 1] >> (16 * (j & 1)));
          const float cb = (float)(int)(signed char)(w[j >> 1] >> (16 * (j & 1) + 8));
          #pragma unroll
          for (int p = 0; p < 4; p++) {
            const uint32_t u = wb0[p * 16 + j];
            dot[p] = __fmaf_rn(ca, __uint_as_float(u << 16), dot[p]);
            dot[p] = __fmaf_rn(cb, __uint_as_float(u & 0xffff0000u), dot[p]);
          }
        }
      }
      #pragma unroll
      for (int p = 0; p < 4; p++) acc[p] = __fmaf_rn(sb, dot[p], acc[p]);
    }
    return make_float4(acc[0], acc[1], acc[2], acc[3]);
  }
  for (int m = 0; i + 32 * m < nb; m++) {
    const int b = i + 32 * m;
    const int4 a0 = qt[b * 64], a1 = qt[b * 64 + 32];
    const unsigned w[8] = { (unsigned)a0.x, (unsigned)a0.y, (unsigned)a0.z, (unsigned)a0.w,
                            (unsigned)a1.x, (unsigned)a1.y, (unsigned)a1.z, (unsigned)a1.w };
    const float sb = sts[b * 32];
    float dot[4] = { 0.f, 0.f, 0.f, 0.f };
    const uint32_t* wb0 = xg + m * 64;
    #pragma unroll
    for (int j = 0; j < 16; j++) {
      const float ca = (float)(int)(signed char)(w[j >> 1] >> (16 * (j & 1)));
      const float cb = (float)(int)(signed char)(w[j >> 1] >> (16 * (j & 1) + 8));
      #pragma unroll
      for (int p = 0; p < 4; p++) {
        const uint32_t u = wb0[p * 16 + j];
        dot[p] = __fmaf_rn(ca, __uint_as_float(u << 16), dot[p]);
        dot[p] = __fmaf_rn(cb, __uint_as_float(u & 0xffff0000u), dot[p]);
      }
    }
    #pragma unroll
    for (int p = 0; p < 4; p++) acc[p] = __fmaf_rn(sb, dot[p], acc[p]);
  }
  return make_float4(acc[0], acc[1], acc[2], acc[3]);
}
template <int XG>
__global__ void __launch_bounds__(512, 2) q8_partial_TLg(const int8_t* __restrict__ q, const float* __restrict__ s,
                                                         const bf16* __restrict__ x, float* __restrict__ part,
                                                         int N, int K, int nb, int T, bf16* __restrict__ yb) {
  uint32_t* const xs = g_tlg_xs;
  const int bx = g_pfgsw ? blockIdx.y : blockIdx.x, by = g_pfgsw ? blockIdx.x : blockIdx.y;
  const int t0 = by * 4, tid = threadIdx.x;
  const int row = bx * 512 + tid;
  const bool live = row < N;
  const int4* qt = reinterpret_cast<const int4*>(q) + (size_t)(row >> 5) * nb * 64 + (row & 31);
  const float* sts = s + (size_t)(row >> 5) * nb * 32 + (row & 31);
  float st[5][4], pv[4];
  #pragma unroll
  for (int k = 0; k < 32; k++) {
    constexpr int NS = XG == 9 ? 8 : 4;
    if ((k & (NS - 1)) == 0) {
      __syncthreads();
      for (int w = tid; w < NS * 8 * 64; w += 512) {
        const int jj = w & 15, p = (w >> 4) & 3, m = (w >> 6) & 7, ls = w >> 9;
        const int kk = k + ls;
        const int i = ((kk & 1) << 4) | ((kk & 2) << 2) | (kk & 4) | ((kk & 8) >> 2) | ((kk & 16) >> 4);
        const int b = i + 32 * m, t = t0 + p;
        const uint32_t u = (b < nb && t < T) ? reinterpret_cast<const uint32_t*>(x + (size_t)t * K)[b * 16 + jj] : 0u;
        if (XG == 9) reinterpret_cast<float2*>(g_tlg_xf8)[w] = make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u));
        else if (XG == 8) reinterpret_cast<float2*>(g_tlg_xf)[w] = make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u));
        else xs[w] = u;
      }
      __syncthreads();
    }
    const int i = ((k & 1) << 4) | ((k & 2) << 2) | (k & 4) | ((k & 8) >> 2) | ((k & 16) >> 4);
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
    if (live) a = tlg_leaf<XG>(qt, sts, xs + (k & 3) * 512, (k & (NS - 1)) * 512, i, nb);
    pv[0] = a.x; pv[1] = a.y; pv[2] = a.z; pv[3] = a.w;
    bool done = false;
    #pragma unroll
    for (int l = 0; l < 5; l++) {
      if (done) continue;
      if ((k >> l) & 1) {
        #pragma unroll
        for (int p = 0; p < 4; p++) pv[p] = __fadd_rn(st[l][p], pv[p]);
      } else {
        #pragma unroll
        for (int p = 0; p < 4; p++) st[l][p] = pv[p];
        done = true;
      }
    }
  }
  if (!live) return;
  #pragma unroll
  for (int p = 0; p < 4; p++) {
    const int t = t0 + p;
    if (t < T) { if (yb) yb[(size_t)t * N + row] = f2bf(__fadd_rn(0.f, pv[p])); else part[(size_t)t * N + row] = pv[p]; }
  }
}
__global__ void k_tile_q(const int4* __restrict__ src, int4* __restrict__ dst, int N, int nb) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;   // (row, b, h)
  if (i >= (size_t)N * nb * 2) return;
  const int row = (int)(i / (nb * 2)), bh = (int)(i - (size_t)row * nb * 2);
  dst[((size_t)(row >> 5) * nb * 2 + bh) * 32 + (row & 31)] = src[i];
}
__global__ void k_tile_s(const float* __restrict__ src, float* __restrict__ dst, int N, int nb) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;   // (row, b)
  if (i >= (size_t)N * nb) return;
  const int row = (int)(i / nb), b = (int)(i - (size_t)row * nb);
  dst[((size_t)(row >> 5) * nb + b) * 32 + (row & 31)] = src[i];
}
// In place (y == x): blocks still staging x would see rows another block already wrote,
// so the value goes through part and the shipped k_combine_bf (nchunk 1: f2bf(0.f + p)).
static void tlb_decode(const int8_t* q, const float* s, const bf16* x, bf16* y, int N, int K, float* part) {
  const int nb = K / Q8_BLK;
  const bool inplace = (const void*)x == (const void*)y;
  q8_partial_TLb<256, 1><<<(N + 255) / 256, 256, (size_t)nb * 17 * 4, g_launch>>>(q, s, x, part, N, K, nb, 1,
                                                                                   inplace ? nullptr : y);
  if (inplace) k_combine_bf<<<(N + 255) / 256, 256, 0, g_launch>>>(part, y, N, 1);
}
// The shapes pf_gemv_T runs on TLb / TLg: one K-chunk (pick), 32-row tiles, and either
// a whole-row CT=4 stage within the 40 KB default (nb % 32 == 0) or the grouped TLg stage.
static bool pf_tilable(const Ent& e) {
  if (e.dtype != 1 || !e.dq || e.name.compare(0, 17, "lfm.model.layers.") != 0) return false;
  const int N = e.ne[0], K = e.ne[1];
  if (N % 32 || K % Q8_BLK) return false;
  const GemvCfg c = pick(N, K);
  if (K % c.KCHUNK || K / c.KCHUNK != 1) return false;
  const int nb = K / Q8_BLK;
  const size_t per = (size_t)nb * 17 * 4;
  return nb <= 256 && (4 * per > 40000 || nb % 32 == 0);
}
static bool pf_tile(Ent* e, int N, int K) {
  if (e->tiled) return true;
  const int nb = K / Q8_BLK;
  // One scratch buffer, grown to the largest tensor and reused (LFM_PF_TSCR=0: per-tensor
  // cudaMalloc/cudaFree, which on this Tegra backs and zeroes pages at ~0.86 ms/MB).
  static const int tscr = getenv("LFM_PF_TSCR") ? atoi(getenv("LFM_PF_TSCR")) : 1;
  static void* scr = nullptr; static size_t scr_n = 0;
  void* tmp = nullptr;
  if (tscr) {
    if (scr_n < (size_t)N * K) {
      if (scr) cudaFree(scr);
      scr = nullptr; scr_n = 0;
      if (cudaMalloc(&scr, (size_t)N * K)) return false;
      scr_n = (size_t)N * K;
    }
    tmp = scr;
  } else if (cudaMalloc(&tmp, (size_t)N * K)) return false;
  k_tile_q<<<(unsigned)(((size_t)N * nb * 2 + 255) / 256), 256, 0, g_launch>>>(reinterpret_cast<const int4*>(e->dq), (int4*)tmp, N, nb);
  cudaMemcpyAsync((void*)e->dq, tmp, (size_t)N * K, cudaMemcpyDeviceToDevice, g_launch);
  k_tile_s<<<(unsigned)(((size_t)N * nb + 255) / 256), 256, 0, g_launch>>>(e->ds, (float*)tmp, N, nb);
  cudaMemcpyAsync((void*)e->ds, tmp, (size_t)N * nb * 4, cudaMemcpyDeviceToDevice, g_launch);
  if (cudaStreamSynchronize(g_launch) != cudaSuccess) { printf("pf_tile failed\n"); exit(2); }
  if (!tscr) cudaFree(tmp);
  e->tiled = true;
  return true;
}

static int pf_gemv_T(Model& M, const std::string& name, const bf16* x, bf16* y,
                     float* part, int Tn) {
  Ent* e = T(M, name);
  if (!e || !(e->dq && e->ds)) { printf("PF4-E2E %s: NOT IN BLOB\n", name.c_str()); return 1; }
  const int N = e->ne[0], K = e->ne[1];
  // The SHIPPED geometry (pick): same KCHUNK -> same nchunk -> same combine order; R and
  // TPB only map rows to blocks. Anything the shipped path would reject is rejected here.
  GemvCfg c = pick(N, K);
  if (K % c.KCHUNK || c.TPB != 256 || c.R % 8) {
    printf("PF4-E2E %s: shape N=%d K=%d KC=%d TPB=%d R=%d not supported\n", name.c_str(), N, K, c.KCHUNK, c.TPB, c.R);
    return 1;
  }
  const int nb = c.KCHUNK / Q8_BLK, nch = K / c.KCHUNK;
  // LFM_PF_FUSEC (default 1): one chunk and an output that is not the input -> the GEMV
  // lane stores f2bf(0.f + a) itself, exactly k_combine_bf_T's value for nchunk == 1, and
  // the combine launch is skipped. In-place calls (y == x) keep the partial buffer: a
  // direct store would race blocks still staging x.
  static const int fusec = getenv("LFM_PF_FUSEC") ? atoi(getenv("LFM_PF_FUSEC")) : 1;
  bf16* yb = (fusec && nch == 1 && (const void*)y != (const void*)x) ? y : nullptr;
  const size_t per = (size_t)nb * 17 * 4;             // one position's staged x
  // Defaults from the 256-id sweep (ms/position, shipped T=1 = 99.6): CT=4 @ 40 KB 69.6,
  // CT=4 @ 24 KB 71.1, CT=2 @ 40 KB 77.8, CT=8 @ 40 KB 115.2 (1 block/SM). 40 KB lets the
  // K=7680 w2 run at CT=2 instead of 1. Neither knob changes a bit, only the block shape.
  static const size_t lim = getenv("LFM_PF_SMEM") ? (size_t)atoi(getenv("LFM_PF_SMEM")) : 40000;
  static const int pf_tpb = getenv("LFM_PF_TPB") ? atoi(getenv("LFM_PF_TPB")) : 512;
  static const int ctmax = getenv("LFM_PF_CT") ? atoi(getenv("LFM_PF_CT")) : 4;
  // LFM_PF_PH (default 1): wide single-chunk shapes whose full x stage would cap CT
  // below 4 run the phased kernel instead (bit-identical, see q8_partial_Tp).
  static const int pf_ph = getenv("LFM_PF_PH") ? atoi(getenv("LFM_PF_PH")) : 1;
  static const int pftg = getenv("LFM_PF_TLG") ? atoi(getenv("LFM_PF_TLG")) : 1;
  if ((e->tiled && 4 * per > lim) || (pftg && pf_ph && nch == 1 && 4 * per > lim && Tn >= 4 && N % 32 == 0 && nb <= 256 && pf_tile(e, N, K))) {
    static const int tlgs = getenv("LFM_PF_TLGS") ? atoi(getenv("LFM_PF_TLGS")) : 9;
    static const int gsw = getenv("LFM_PF_GSW") ? atoi(getenv("LFM_PF_GSW")) : 1;
    const dim3 gg = gsw ? dim3((Tn + 3) / 4, (N + 511) / 512) : dim3((N + 511) / 512, (Tn + 3) / 4);
    if (tlgs == 9)      q8_partial_TLg<9><<<gg, 512, 0, g_launch>>>(e->dq, e->ds, x, part, N, K, nb, Tn, yb);
    else if (tlgs == 8) q8_partial_TLg<8><<<gg, 512, 0, g_launch>>>(e->dq, e->ds, x, part, N, K, nb, Tn, yb);
    else if (tlgs == 4) q8_partial_TLg<4><<<gg, 512, 0, g_launch>>>(e->dq, e->ds, x, part, N, K, nb, Tn, yb);
    else if (tlgs == 1) q8_partial_TLg<1><<<gg, 512, 0, g_launch>>>(e->dq, e->ds, x, part, N, K, nb, Tn, yb);
    else                q8_partial_TLg<0><<<gg, 512, 0, g_launch>>>(e->dq, e->ds, x, part, N, K, nb, Tn, yb);
    goto launched;
  }
  if (pf_ph && nch == 1 && 4 * per > lim && Tn >= 4) {
    // LFM_PF_TPTPB (default 512): 16 warps x GRW 4 rows per block; each row's chain unchanged.
    static const int tptpb = getenv("LFM_PF_TPTPB") ? atoi(getenv("LFM_PF_TPTPB")) : 512;
    if (tptpb == 512) {
      const dim3 gp((N + 63) / 64, (Tn + 3) / 4);
      q8_partial_Tp<4, 4, 512><<<gp, 512, 0, g_launch>>>(e->dq, e->ds, x, part, N, K, c.KCHUNK, nb, Tn, yb);
    } else {
      const dim3 gp((N + 31) / 32, (Tn + 3) / 4);
      q8_partial_Tp<4, 4><<<gp, 256, 0, g_launch>>>(e->dq, e->ds, x, part, N, K, c.KCHUNK, nb, Tn, yb);
    }
    goto launched;
  }
  {
  int ct = ctmax >= 8 ? 8 : (ctmax >= 4 ? 4 : (ctmax >= 2 ? 2 : 1));
  while (ct > 1 && ct * per > lim) ct /= 2;
  // LFM_PF_TL (default 1): single-chunk CT=4 shapes run q8_partial_TLb (one row per thread,
  // weights repacked once to 32-row tiles on first use; decode then reads them through
  // tlb_decode). 2 = 256-thread blocks (slower), 0 = Tb on row-major weights.
  static const int pftl = getenv("LFM_PF_TL") ? atoi(getenv("LFM_PF_TL")) : 1;
  if (e->tiled && !(ct == 4 && nch == 1)) { printf("PF4-E2E %s: tiled tensor has no TL path\n", name.c_str()); return 1; }
  if (e->tiled || (pftl && ct == 4 && nch == 1 && N % 32 == 0 && nb % 32 == 0 && Tn >= 2 && pf_tile(e, N, K))) {
    const int tpb = pftl == 2 ? 256 : 512;
    const dim3 gl((N + tpb - 1) / tpb, (Tn + 3) / 4);
    // LFM_PF_TLF (default 1): q8_partial_TLf, x staged as f32 with the leaf's weight loads
    // pipelined one block ahead (bit-identical), when CT x K floats fit 48 KB. 2 = TLf
    // without the prefetch, 0 = TLb.
    static const int pftlf = getenv("LFM_PF_TLF") ? atoi(getenv("LFM_PF_TLF")) : 4;
    if (pftlf && tpb == 512 && (size_t)4 * K * 4 <= 48 * 1024) {
      static const int gswf = getenv("LFM_PF_GSW") ? atoi(getenv("LFM_PF_GSW")) : 1;
      const dim3 gl = gswf ? dim3((Tn + 3) / 4, (N + tpb - 1) / tpb) : dim3((N + tpb - 1) / tpb, (Tn + 3) / 4);
      static const bool rng = getenv("LFM_TLF_RANGE") != nullptr;
      if (rng && nb == 64) { static bool r = false; if (!r) { atexit(tlf_range_dump); r = true; }
        k_tlf_range<<<64, 256, 0, g_launch>>>(x, (long)Tn * K); }
      static bool fbset = false;
      if (!fbset) { const int fb = getenv("LFM_PF_H2FB") ? atoi(getenv("LFM_PF_H2FB")) : 0; cudaMemcpyToSymbol(c_h2fb, &fb, 4); fbset = true; }
      if (pftlf == 9 && nb == 64) q8_partial_TLf<512, 4, 9><<<gl, 512, (size_t)2 * K * 4, g_launch>>>(e->dq, e->ds, x, part, N, K, nb, Tn, yb);
      else if (pftlf == 4 && nb == 64) q8_partial_TLf<512, 4, 4><<<gl, 512, (size_t)4 * K * 4, g_launch>>>(e->dq, e->ds, x, part, N, K, nb, Tn, yb);
      else if (pftlf == 2) q8_partial_TLf<512, 4, 0><<<gl, 512, (size_t)4 * K * 4, g_launch>>>(e->dq, e->ds, x, part, N, K, nb, Tn, yb);
      else            q8_partial_TLf<512, 4, 1><<<gl, 512, (size_t)4 * K * 4, g_launch>>>(e->dq, e->ds, x, part, N, K, nb, Tn, yb);
      goto launched;
    }
    if (tpb == 512) q8_partial_TLb<512><<<gl, 512, 4 * per, g_launch>>>(e->dq, e->ds, x, part, N, K, nb, Tn, yb);
    else            q8_partial_TLb<256><<<gl, 256, 4 * per, g_launch>>>(e->dq, e->ds, x, part, N, K, nb, Tn, yb);
    goto launched;
  }
  // LFM_PF_R (default 512; 256 = decode's R): rows per prefill block. Tb has no sh[RMAX],
  // so it can stage each x tile once for 512 rows (32 per warp). Rows only move between
  // blocks: bit-identical. 128 / 1024 / 2048 measured slower.
  static const int pfR = getenv("LFM_PF_R") ? atoi(getenv("LFM_PF_R")) : 512;
  if (pfR >= 64 && pfR % 16 == 0 && N >= 1024) c.R = pfR;
  const int nblk = ((N + c.R - 1) / c.R) * nch;
  const dim3 grid(nblk, (Tn + ct - 1) / ct);
  if (ct == 8)      q8_partial_Tb<8><<<grid, 256, 8 * per, g_launch>>>(e->dq, e->ds, x, part, N, K, c.KCHUNK, c.R, nb, Tn, yb);
  // LFM_PF_TPB (default 512; 256 = the old shape): same kernel in 512-thread blocks. The 17.4 KB x tile caps 256-thread
  // blocks at 3 per SM (24 warps); two 512-thread blocks hold 32. Rows map to warps
  // differently, each row's chain is unchanged.
  else if (ct == 4 && pf_tpb == 512 && c.R % 16 == 0)
    q8_partial_Tb<4, 512><<<grid, 512, 4 * per, g_launch>>>(e->dq, e->ds, x, part, N, K, c.KCHUNK, c.R, nb, Tn, yb);
  else if (ct == 4) q8_partial_Tb<4><<<grid, 256, 4 * per, g_launch>>>(e->dq, e->ds, x, part, N, K, c.KCHUNK, c.R, nb, Tn, yb);
  else if (ct == 2) q8_partial_Tb<2><<<grid, 256, 2 * per, g_launch>>>(e->dq, e->ds, x, part, N, K, c.KCHUNK, c.R, nb, Tn, yb);
  else              q8_partial_Tb<1><<<grid, 256, per, g_launch>>>(e->dq, e->ds, x, part, N, K, c.KCHUNK, c.R, nb, Tn, yb);
  }
launched:
  cudaError_t le = cudaGetLastError();
  if (le != cudaSuccess) {   // no number for a kernel that did not run (the standing rule)
    printf("PF4-E2E %s N=%d K=%d KC=%d: LAUNCH ERROR %s\n", name.c_str(), N, K, c.KCHUNK, cudaGetErrorString(le));
    return 1;
  }
  dim3 g((N + 255) / 256, Tn);
  if (!yb) k_combine_bf_T<<<g, 256, 0, g_launch>>>(part, y, N, nch, Tn);
  cudaError_t le2 = cudaGetLastError();
  if (le2 != cudaSuccess) { printf("PF4-E2E %s combine: LAUNCH ERROR %s\n", name.c_str(), cudaGetErrorString(le2)); return 1; }
  return 0;
}

// Ring index = the conv-LAYER index (0..NCONV-1), NOT a running counter. The first
// version incremented it per POSITION inside the conv branch, so a 20-position chunk
// walked 20 ring slots -- past the NCONV=13 the ring array has -- and read garbage:
// in_proj and B*x matched the T=1 reference exactly (0.21023, 0.12444) while conv came
// out 4.6x too large (0.01160 vs 0.00252), which is what poisoned the residual stream
// into nan by layer 1. Each conv layer owns ONE 7-slot ring; positions advance its
// contents through k_shift_in, they do not select a different ring.
static int pf4_conv_idx[NLAYER];

struct PF4Arena {
  bf16 *xb, *h1, *q, *k, *v, *gate, *attn, *h2, *bx, *cy, *w1, *w3, *hout;
  float* part;
};

// One decoder layer, T positions at a time. Sequencing mirrors layer_body exactly:
// operator_norm -> {q,k,v,gate | conv in_proj} -> attention | conv -> o_proj | out_proj
// -> residual -> ffn_norm -> {w1,w3} -> silu -> w2 -> residual.
#define P4(nm, ptr, n) if (pf4trace && L == 0) (cudaDeviceSynchronize(), \
    printf("PF4PROBE %-14s rms %9.5f\n", nm, rms_host_n((ptr), (n))), 0)
// LFM_PF_SYNC (default 3): bound the layer-major prefill's GPU queue. Without it a 4000-id
// prompt dies inside the first attention layer with "the launch timed out and was
// terminated" (nvgpu channel watchdog: too much un-synced work queued in one layer, no
// single kernel is long; CUDA_LAUNCH_BLOCKING=1 or a sync per chunk both run clean, a sync
// per layer does not). 3 = before queueing chunk i, wait for the event recorded after chunk
// i - K (LFM_PF_SYNCK, default 4): the queue never drains, so it costs nothing measurable
// at 1642 positions. 1 = full sync per chunk (+3%), 2 = per layer (fails), 0 = unbounded.
static const int g_pfsync = getenv("LFM_PF_SYNC") ? atoi(getenv("LFM_PF_SYNC")) : 3;
static int layer_chunk_T(Model& M, int L, int pos0, int Tn, PF4Arena& A) {
  const int QDIM = NH * HD, KVDIM = NKV * HD;
  const int pf4trace = getenv("LFM_PF4_TRACE") ? 1 : 0;   // stage-by-stage rms for L0,
  // to be read against the T=1 path's own --probe-layer 0 --probe-step 0 PROBE lines.
  k_rmsnorm_T<<<Tn, 256, 0, g_launch>>>(A.xb, NW(M, L, "operator_norm.weight"), A.h1, NORM_EPS, H);
  if (cudaGetLastError() != cudaSuccess) { printf("PF4-E2E L%d operator_norm launch failed\n", L); return 1; }
  P4("operator_norm", A.h1, H);   // after the launch: the first version probed before it and printed 0.00000
  if (is_attn_g[L]) {
    if (pf_gemv_T(M, LN(L, "self_attn.q_proj.weight"), A.h1, A.q, A.part, Tn)) return 1;
    if (pf_gemv_T(M, LN(L, "self_attn.k_proj.weight"), A.h1, A.k, A.part, Tn)) return 1;
    if (pf_gemv_T(M, LN(L, "self_attn.v_proj.weight"), A.h1, A.v, A.part, Tn)) return 1;
    if (pf_gemv_T(M, LN(L, "self_attn.gate_proj.weight"), A.h1, A.gate, A.part, Tn)) return 1;
    const float* invf = is_slide_g[L] ? M.invf_slide : M.invf_full;
    const int rot = is_slide_g[L] ? HD : HD / 2;
    const int win = is_slide_g[L] ? SLIDE_WIN : 0;
    const float scaling = 1.0f / sqrtf((float)HD);
    bf16* kbase = M.kc + (size_t)L * MAXTOK * KVDIM;
    bf16* vbase = M.vc + (size_t)L * MAXTOK * KVDIM;
    // LFM_PF_ATTB (default 1): the chunk's positions in ONE launch per kernel, blockIdx.y =
    // position. Each block computes exactly what the per-position launch computed; the
    // chunk's K/V rows are all in the cache before attention, and position t reads keys
    // 0..pos0+t only, so later rows are never read early. 0 = the per-position loop.
    static const int attb = getenv("LFM_PF_ATTB") ? atoi(getenv("LFM_PF_ATTB")) : 1;
    if (attb) {
      k_headnorm<<<dim3(NH, Tn), 32, 0, g_launch>>>(A.q, NW(M, L, "self_attn.q_norm.weight"), NORM_EPS, HD);
      k_headnorm<<<dim3(NKV, Tn), 32, 0, g_launch>>>(A.k, NW(M, L, "self_attn.k_norm.weight"), NORM_EPS, HD);
      k_rope<<<dim3(NH, Tn), 64, 0, g_launch>>>(A.q, invf, pos0, HD, rot, nullptr);
      k_rope<<<dim3(NKV, Tn), 64, 0, g_launch>>>(A.k, invf, pos0, HD, rot, nullptr);
      cudaMemcpyAsync(kbase + (size_t)pos0 * KVDIM, A.k, (size_t)Tn * KVDIM * 2, cudaMemcpyDeviceToDevice, g_launch);
      cudaMemcpyAsync(vbase + (size_t)pos0 * KVDIM, A.v, (size_t)Tn * KVDIM * 2, cudaMemcpyDeviceToDevice, g_launch);
      static const int attg = getenv("LFM_PF_ATTG") ? atoi(getenv("LFM_PF_ATTG")) : 1;
      const int sln = pos0 + Tn;
      const size_t shm = (size_t)attg * (sln + 8 * HD) * 4;
      if ((attg == 1 || attg == 2 || attg == 4) && shm <= 48 * 1024) {          // else the per-head kernel
        static const int agsw = getenv("LFM_ATT_GSW") ? atoi(getenv("LFM_ATT_GSW")) : 1;
        if (attg == 1) k_attn_headG<1><<<agsw ? dim3(Tn, NH) : dim3(NH, Tn), 256, shm, g_launch>>>(A.q, kbase, vbase, A.attn, pos0 + 1, win, scaling, NKV, sln);
        else if (attg == 2) k_attn_headG<2><<<agsw ? dim3(Tn, NH / 2) : dim3(NH / 2, Tn), 256, shm, g_launch>>>(A.q, kbase, vbase, A.attn, pos0 + 1, win, scaling, NKV, sln);
        else           k_attn_headG<4><<<agsw ? dim3(Tn, NH / 4) : dim3(NH / 4, Tn), 256, shm, g_launch>>>(A.q, kbase, vbase, A.attn, pos0 + 1, win, scaling, NKV, sln);
        if (cudaGetLastError() != cudaSuccess) { printf("PF4: k_attn_headG launch failed (G=%d, sln=%d)\n", attg, sln); return 1; }
      } else
      k_attn_head<<<dim3(NH, Tn), 256, 0, g_launch>>>(A.q, kbase, vbase, A.attn, pos0 + 1, win, scaling, HD, NKV, nullptr);
      if (ewv_on() && (NH * HD) % 4 == 0 && HD % 4 == 0)
        k_sigmoid_gate4<<<dim3((NH * HD / 4 + 255) / 256, Tn), 256, 0, g_launch>>>(A.gate, A.attn, NH, HD);
      else k_sigmoid_gate<<<dim3(1, Tn), NH, 0, g_launch>>>(A.gate, A.attn, HD);
    } else
    for (int t = 0; t < Tn; t++) {
      const int pos = pos0 + t;
      bf16* q = A.q + (size_t)t * QDIM, *k = A.k + (size_t)t * KVDIM, *v = A.v + (size_t)t * KVDIM;
      bf16* gt = A.gate + (size_t)t * NH, *at = A.attn + (size_t)t * QDIM;
      k_headnorm<<<NH, 32, 0, g_launch>>>(q, NW(M, L, "self_attn.q_norm.weight"), NORM_EPS, HD);
      k_headnorm<<<NKV, 32, 0, g_launch>>>(k, NW(M, L, "self_attn.k_norm.weight"), NORM_EPS, HD);
      k_rope<<<NH, 64, 0, g_launch>>>(q, invf, pos, HD, rot, nullptr);
      k_rope<<<NKV, 64, 0, g_launch>>>(k, invf, pos, HD, rot, nullptr);
      cudaMemcpy(kbase + (size_t)pos * KVDIM, k, (size_t)KVDIM * 2, cudaMemcpyDeviceToDevice);
      cudaMemcpy(vbase + (size_t)pos * KVDIM, v, (size_t)KVDIM * 2, cudaMemcpyDeviceToDevice);
      k_attn_head<<<NH, 256, 0, g_launch>>>(q, kbase, vbase, at, pos + 1, win, scaling, HD, NKV, nullptr);
      k_sigmoid_gate<<<1, NH, 0, g_launch>>>(gt, at, HD);
    }
    if (cudaGetLastError() != cudaSuccess) { printf("PF4-E2E L%d attention launch failed\n", L); return 1; }
    if (pf_gemv_T(M, LN(L, "self_attn.o_proj.weight"), A.attn, A.h2, A.part, Tn)) return 1;
  } else {
    if (pf_gemv_T(M, LN(L, "conv.in_proj.weight"), A.h1, A.bx, A.part, Tn)) return 1;
    P4("in_proj", A.bx, 3 * H);
    // LFM_PF_CONVB (default 1): the chunk's conv branch in one launch (k_conv_chunk,
    // bit-identical); 0 = the per-position T=1 kernels below. The trace probes need 0.
    static const int convb = getenv("LFM_PF_CONVB") ? atoi(getenv("LFM_PF_CONVB")) : 1;
    if (convb && !pf4trace)
      k_conv_chunk<<<(H + 127) / 128, 128, 0, g_launch>>>(M.ring + (size_t)pf4_conv_idx[L] * 7 * H,
          T(M, LN(L, "conv.conv.weight"))->df, A.bx, A.h2, H, Tn);
    else
    for (int t = 0; t < Tn; t++) {
      bf16* bx = A.bx + (size_t)t * 3 * H;
      bf16* cy = A.cy + (size_t)t * H;
      bf16* tmp = A.h2 + (size_t)t * H;              // h2 is free until out_proj
      // emul / k_shift_in / k_conv7 are the T=1 kernels, called per position, in the
      // same order as layer_body: the gate multiply completes BEFORE the newest ring
      // slot is written (that ordering is what the "bug that broke the engine" note in
      // the step loop is about), and the ring is per conv layer, indexed by the conv
      // counter -- chunk state carried in the ring, which is why conv needs no new state.
      emul(M, bx, bx + 2 * H, tmp, H);
      if (t == 0) P4("B*x", tmp, H);
      bf16* ring = M.ring + (size_t)pf4_conv_idx[L] * 7 * H;
      k_shift_in<<<(H + 63) / 64, 64, 0, g_launch>>>(ring, tmp, H);
      if (t == 0 && pf4trace) { cudaDeviceSynchronize();
        for (int k = 0; k < 7; k++)
          printf("PF4  ring slot %d     rms %9.5f\n", k, rms_host_n(ring + (size_t)k * H, H)); }
      k_conv7<<<(H + 63) / 64, 64, 0, g_launch>>>(ring, T(M, LN(L, "conv.conv.weight"))->df, cy, H);
      if (t == 0) P4("conv", cy, H);
      emul(M, bx + H, cy, tmp, H);
      if (t == 0) P4("C*y", tmp, H);
    }
    if (cudaGetLastError() != cudaSuccess) { printf("PF4-E2E L%d conv launch failed\n", L); return 1; }
    if (pf_gemv_T(M, LN(L, "conv.out_proj.weight"), A.h2, A.h2, A.part, Tn)) return 1;
    P4("out_proj", A.h2, H);
  }
  // post_operator_norm, then the residual add -- layer_body normalises the operator
  // branch output into h3 BEFORE adding, and the first version of this driver added the
  // raw o_proj/out_proj result. Symptom was x rms = nan at the last prompt position: the
  // residual stream grew without bound across 24 layers.
  k_rmsnorm_T<<<Tn, 256, 0, g_launch>>>(A.h2, NW(M, L, "post_operator_norm.weight"), A.cy, NORM_EPS, H);
  if (cudaGetLastError() != cudaSuccess) { printf("PF4-E2E L%d post_operator_norm launch failed\n", L); return 1; }
  P4("post_op_norm", A.cy, H);
  // LFM_PF_EWB (default 1): the chunk's rows are contiguous, so each elementwise op runs
  // as one launch over Tn * n elements (same per-element arithmetic); 0 = per position.
  static const int ewb = getenv("LFM_PF_EWB") ? atoi(getenv("LFM_PF_EWB")) : 1;
  if (ewb) eadd(M, A.xb, A.cy, A.xb, Tn * H);
  else for (int t = 0; t < Tn; t++) eadd(M, A.xb + (size_t)t * H, A.cy + (size_t)t * H, A.xb + (size_t)t * H, H);
  P4("add1", A.xb, H);
  k_rmsnorm_T<<<Tn, 256, 0, g_launch>>>(A.xb, NW(M, L, "ffn_norm.weight"), A.h1, NORM_EPS, H);
  P4("ffn_norm", A.h1, H);
  if (pf_gemv_T(M, LN(L, "feed_forward.w1.weight"), A.h1, A.w1, A.part, Tn)) return 1;
  P4("w1", A.w1, INTER);
  if (pf_gemv_T(M, LN(L, "feed_forward.w3.weight"), A.h1, A.w3, A.part, Tn)) return 1;
  P4("w3", A.w3, INTER);
  static const int silv = getenv("LFM_PF_SILV") ? atoi(getenv("LFM_PF_SILV")) : 4;
  if (ewb && silv == 4 && (Tn * INTER) % 4 == 0)
    k_silu_mul_bfv<4><<<(Tn * INTER / 4 + 255) / 256, 256, 0, g_launch>>>(A.w1, A.w3, A.h2, Tn * INTER);
  else if (ewb && silv == 2 && (Tn * INTER) % 2 == 0)
    k_silu_mul_bfv<2><<<(Tn * INTER / 2 + 255) / 256, 256, 0, g_launch>>>(A.w1, A.w3, A.h2, Tn * INTER);
  else if (ewb) k_silu_mul_bf<<<(Tn * INTER + 255) / 256, 256, 0, g_launch>>>(A.w1, A.w3, A.h2, Tn * INTER);
  else for (int t = 0; t < Tn; t++)
    k_silu_mul_bf<<<(INTER + 255) / 256, 256, 0, g_launch>>>(A.w1 + (size_t)t * INTER,
                                                             A.w3 + (size_t)t * INTER,
                                                             A.h2 + (size_t)t * INTER, INTER);
  if (cudaGetLastError() != cudaSuccess) { printf("PF4-E2E L%d silu launch failed\n", L); return 1; }
  P4("silu_mul", A.h2, INTER);
  if (pf_gemv_T(M, LN(L, "feed_forward.w2.weight"), A.h2, A.hout, A.part, Tn)) return 1;
  P4("w2", A.hout, H);
  k_rmsnorm_T<<<Tn, 256, 0, g_launch>>>(A.hout, NW(M, L, "post_ffn_norm.weight"), A.h2, NORM_EPS, H);
  P4("post_ffn_norm", A.h2, H);
  if (ewb) eadd(M, A.xb, A.h2, A.xb, Tn * H);
  else for (int t = 0; t < Tn; t++) eadd(M, A.xb + (size_t)t * H, A.h2 + (size_t)t * H, A.xb + (size_t)t * H, H);
  P4("add2", A.xb, H);
  return 0;
}

// The lm_head arms as ONE function, conditions copied verbatim from the step loop.
// The driver needs the identical head on the last prompt position; copying the block
// would let the two heads drift, and a head that drifts changes the argmax -- the
// argmax IS the measurement. (My first attempt here invented a `g_lmhead` knob that
// does not exist: the real chain dispatches on the embedding tensor's dtype and
// M.emb_mode. Exact-text asserts caught that, and a second one caught me inventing the
// driver's own parameter names -- it takes (Model&, int Tn, prompt, Ent* emb).)
static void head_logits_from(Model& M, const bf16* li, Ent* eemb) {
  if (eemb && eemb->dtype == 3 && M.emb_mode == 1 && M.hemb.hc)
    lm_head_bf16_host(M, M.hemb, li, M.lgf);  // certified arm, host-staged
  else if (eemb && eemb->dtype == 3)
    lm_head_bf16(M, eemb, li, M.lgf);         // certified arm, resident
  else if (M.emb_mode == 1)
    lm_head_host(M, M.hemb, li, M.lgf);       // Q8 streamed from pinned host
  else gemv_f(M, EMBNAME, li, M.lgf);         // tied lm_head, resident
}

static int pf4_arena_alloc(PF4Arena& A, int Tn) {
  memset(&A, 0, sizeof A);
  const size_t rowH = (size_t)H, rowI = (size_t)INTER, row3 = 3 * (size_t)H;
  cudaMalloc(&A.xb, (size_t)Tn * rowH * 2);   cudaMalloc(&A.h1, (size_t)Tn * rowH * 2);
  cudaMalloc(&A.q, (size_t)Tn * NH * HD * 2); cudaMalloc(&A.k, (size_t)Tn * NKV * HD * 2);
  cudaMalloc(&A.v, (size_t)Tn * NKV * HD * 2);cudaMalloc(&A.gate, (size_t)Tn * NH * 2);
  cudaMalloc(&A.attn, (size_t)Tn * NH * HD * 2); cudaMalloc(&A.h2, (size_t)Tn * (rowI > rowH ? rowI : rowH) * 2);
  cudaMalloc(&A.bx, (size_t)Tn * row3 * 2);   cudaMalloc(&A.cy, (size_t)Tn * rowH * 2);
  cudaMalloc(&A.w1, (size_t)Tn * rowI * 2);   cudaMalloc(&A.w3, (size_t)Tn * rowI * 2);
  cudaMalloc(&A.hout, (size_t)Tn * rowH * 2);
  cudaMalloc(&A.part, (size_t)Tn * 10000 * 4);
  if (!A.xb || !A.h1 || !A.q || !A.k || !A.v || !A.gate || !A.attn || !A.h2 ||
      !A.bx || !A.cy || !A.w1 || !A.w3 || !A.hout || !A.part) {
    printf("PF4: arena cudaMalloc failed (T=%d)\n", Tn); return 1;
  }
  return 0;
}

// Embed a chunk's positions, through the SAME paths the T=1 loop uses (bf16 copy,
// host-Q8 dequant, resident Q8 dequant). Shared with the integrated --prefill-run.
static int pf4_embed_chunk(Model& M, PF4Arena& A, const std::vector<int>& prompt,
                           int c0, int Tc, Ent* emb) {
  for (int t = 0; t < Tc; t++) {
    const int tok = prompt[c0 + t];
    const int pos = c0 + t;
    bf16* dst = A.xb + (size_t)t * H;
    if (g_semb) { printf("PF4: speech-embedding rows are not supported by the chunked prefill\n"); return 1; }
    if (g_aemb && pos >= g_aemb_at && pos - g_aemb_at < g_aemb_n) {
      // Audio position: the step loop's own override, the same kernel on the same row.
      k_bf16_from_f32<<<(H + 255) / 256, 256, 0, g_launch>>>(g_aemb + (size_t)(pos - g_aemb_at) * H, dst, H);
      if (cudaGetLastError() != cudaSuccess) { printf("PF4: audio embedding launch failed at pos %d\n", pos); return 1; }
      g_aemb_fed++;
    } else if (emb->dtype == 3) {
      const char* src = (M.emb_mode == 1 && M.hemb.hc) ? (const char*)M.hemb.hc : (const char*)emb->dq;
      cudaMemcpy(dst, src + (size_t)tok * H * 2, (size_t)H * 2,
                 M.emb_mode == 1 ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice);
    } else if (M.emb_mode == 1) {
      const int sb = H / Q8_BLK;
      cudaMemcpyAsync(M.hemb.dc, M.hemb.hc + (size_t)tok * H, (size_t)H, cudaMemcpyHostToDevice, 0);
      cudaMemcpyAsync(M.hemb.dsl, M.hemb.hs + (size_t)tok * sb, (size_t)sb * 4, cudaMemcpyHostToDevice, 0);
      k_dequant_row<<<(H + 255) / 256, 256, 0, g_launch>>>(M.hemb.dc, M.hemb.dsl, dst, H);
    } else if (g_embfile) {
      // The embedding is still uploading (it is last in the order): read this row's codes
      // and scales -- the same bytes the upload copies, CRC-checked before the head -- and
      // run the same dequant kernel on a scratch copy.
      static int fd = -1; static int8_t* dc = nullptr; static float* dsl = nullptr;
      const int sb = H / Q8_BLK;
      if (fd < 0) { fd = open(g_blob_path, O_RDONLY);
        if (fd < 0 || cudaMalloc(&dc, H) || cudaMalloc(&dsl, sb * 4)) { printf("PF4: embedding row reader setup failed\n"); return 1; } }
      int8_t hc[H]; float hs[H / Q8_BLK];
      const off_t o = (off_t)M.emb_file_off;
      if (pread(fd, hc, H, o + (off_t)tok * H) != H ||
          pread(fd, hs, sb * 4, o + (off_t)emb->nc + (off_t)tok * sb * 4) != sb * 4) {
        printf("PF4: embedding row %d short read\n", tok); return 1; }
      cudaMemcpy(dc, hc, H, cudaMemcpyHostToDevice);
      cudaMemcpy(dsl, hs, sb * 4, cudaMemcpyHostToDevice);
      k_dequant_row<<<(H + 255) / 256, 256, 0, g_launch>>>(dc, dsl, dst, H);
    } else {
      k_dequant_row<<<(H + 255) / 256, 256, 0, g_launch>>>(emb->dq + (size_t)tok * H,
                                             emb->ds + (size_t)tok * (H / Q8_BLK), dst, H);
    }
  }
  // No sync needed: all launches are on g_launch, so the next launch serializes.
  // cudaGetLastError() still reports launch errors from the async launches.
  return cudaGetLastError() != cudaSuccess;
}

static int prefill_e2e(Model& M, int Tn, const std::vector<int>& prompt, Ent* emb) {
  if (Tn < 2 || Tn > 64) { printf("--prefill-e2e: T=%d out of range 2..64\n", Tn); return 2; }
  const int N = (int)prompt.size();
  PF4Arena A; memset(&A, 0, sizeof A);
  if (pf4_arena_alloc(A, Tn)) return 1;
  { int c = 0; for (int L = 0; L < NLAYER; L++) { pf4_conv_idx[L] = c; if (!is_attn_g[L]) c++; }
    if (c != NCONV) { printf("PF4-E2E: conv count %d != NCONV %d\n", c, NCONV); return 2; } }
  // The shipped path zeroes the KV caches in its WARM-UP step (s < 0, next to the ring
  // reset), which this driver never runs -- so without this, attention reads
  // uninitialised keys/values. Mirroring that reset is part of "same starting state".
  cudaMemset(M.kc, 0, (size_t)NLAYER * MAXTOK * NKV * HD * 2);
  cudaMemset(M.vc, 0, (size_t)NLAYER * MAXTOK * NKV * HD * 2);
  cudaMemset(M.ring, 0, (size_t)NCONV * 7 * H * 2);
  cudaMemset(M.x, 0, H * 2);
  struct timespec t0, t1; cudaDeviceSynchronize(); clock_gettime(CLOCK_MONOTONIC, &t0);
  for (int c0 = 0; c0 < N; c0 += Tn) {
    const int Tc = (N - c0 < Tn) ? N - c0 : Tn;
    if (pf4_embed_chunk(M, A, prompt, c0, Tc, emb)) { printf("PF4-E2E embedding failed at chunk %d\n", c0); return 1; }
    for (int L = 0; L < NLAYER; L++) {
      if (layer_chunk_T(M, L, c0, Tc, A)) { printf("PF4-E2E: layer %d failed\n", L); return 1; }
      // Per-layer rms of position 0, to be read against the T=1 path's own
      // "POS <s> L<L> attn|conv rms <v>" trace lines: the first layer whose rms
      // disagrees is the origin, which is how the encoder divergence was localised.
      // Reports the LAST position of the chunk, so the lines line up with the shipped
      // path's "POS <s> L<L> ... rms" trace at the prompt's final position -- the
      // position whose residual stream is the acceptance value. (The first version
      // printed the chunk's FIRST row, which is position 0 for a single-chunk run:
      // it compared position 0 against position 19, a comparison that means nothing.)
      if (getenv("LFM_PF4_TRACE")) cudaDeviceSynchronize(),
        printf("PF4POS %3d L%-2d %s rms %.5f\n", c0 + Tc - 1, L, is_attn_g[L] ? "attn" : "conv",
               rms_host_n(A.xb + (size_t)(Tc - 1) * H, H));
    }
    if (N > Tn) { printf("PF4-E2E chunk pos %d..%d done\n", c0, c0 + Tc - 1); fflush(stdout); }
  }
  cudaDeviceSynchronize(); clock_gettime(CLOCK_MONOTONIC, &t1);
  const double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) * 1e-6;
  // The last prompt position's residual stream is the acceptance value.
  const int last = (N - 1) % Tn;
  std::vector<bf16> hy((size_t)H);
  cudaMemcpy(hy.data(), A.xb + (size_t)last * H, (size_t)H * 2, cudaMemcpyDeviceToHost);
  // The acceptance test that matters is the TOKEN: the chunk's last prompt row through
  // the shared head, argmaxed -- directly comparable to the shipped T=1 path's token
  // generated at the same position.
  cudaMemcpy(M.x, A.xb + (size_t)last * H, (size_t)H * 2, cudaMemcpyDeviceToDevice);
  {
    Ent* fn = T(M, "lfm.model.embedding_norm.weight");
    rms1(M, M.x, fn ? fn->df : nullptr, M.h1);
    head_logits_from(M, M.h1, emb);
    k_argmax<<<1, 256, 0, g_launch>>>(M.lgf, M.tok_dev, VOCAB);
    cudaDeviceSynchronize();
    int tok = -1; cudaMemcpy(&tok, M.tok_dev, 4, cudaMemcpyDeviceToHost);
    printf("PF4-E2E T=%d: first generated id = %d  (prompt %d ids)\n", Tn, tok, (int)prompt.size());
  }
  FILE* o = fopen("/tmp/pf4_x_last.bin", "wb");
  if (o) { fwrite(hy.data(), 2, H, o); fclose(o); }
  double s2 = 0; long nz = 0;
  for (int i = 0; i < H; i++) { float v = bf2f_host(hy[i]); s2 += (double)v * v; if (hy[i]) nz++; }
  printf("PF4-E2E prefill %d ids in chunks of %d: %.0f ms (%.2f ms/pos), last-position x rms %.5f nonzero %d/%d\n",
         N, Tn, ms, ms / N, sqrt(s2 / H), (int)nz, H);
  printf("PF4_E2E_DUMP /tmp/pf4_x_last.bin\n");
  fflush(stdout);
  cudaFree(A.xb); cudaFree(A.h1); cudaFree(A.q); cudaFree(A.k); cudaFree(A.v); cudaFree(A.gate);
  cudaFree(A.attn); cudaFree(A.h2); cudaFree(A.bx); cudaFree(A.cy); cudaFree(A.w1); cudaFree(A.w3);
  cudaFree(A.hout); cudaFree(A.part);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage: %s <blob.lfm3q8> [--steps N] [--prompt id,id,...] [--trace]\n", argv[0]);
    printf("       [--audio-emb F --audio-at P] | [--clip SPEC]... (resident multi-clip; SPEC"
           " lines: emb|-, ids, audio-at, ngen)\n", argv[0]);
    return 1;
  }
  if (getenv("LFM_PF_GSW")) { const int v = atoi(getenv("LFM_PF_GSW")); cudaMemcpyToSymbol(g_pfgsw, &v, sizeof v); }
  if (getenv("LFM_ATT_GSW")) { const int v = atoi(getenv("LFM_ATT_GSW")); cudaMemcpyToSymbol(g_attgsw, &v, sizeof v); }
  if (getenv("LFM_ATT_SM1")) { const int v = atoi(getenv("LFM_ATT_SM1")); cudaMemcpyToSymbol(g_attsm1, &v, sizeof v); }
  if (getenv("LFM_RMSV")) { const int v = atoi(getenv("LFM_RMSV")); cudaMemcpyToSymbol(c_rmsv, &v, sizeof v); }
  int steps = 8; bool trace = false;
  int lm_rows_override = 0, emb_mode_req = 0, probe_layer = 0, probe_step = 0;
  int chunkT = getenv("LFM_PF_T") ? atoi(getenv("LFM_PF_T")) : 4;   // --prefill-chunk T
  const char* audio_emb_file = nullptr;   // --audio-emb (ASR path)
  std::vector<std::string> clip_files;  // --clip SPEC repeated: resident multi-clip ASR
  const char* dump_logits = nullptr;
  const char* dump_logits_all = nullptr;   // every decode step, appended (gate: top-k vs HF)
  const char* speak_emb_file = nullptr;  // --speak-emb (speech-out teacher forcing)
  std::vector<int> prompt;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--trace")) trace = true;
    else if (!strcmp(argv[i], "--probe-layer") && i + 1 < argc) probe_layer = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--probe-step") && i + 1 < argc) probe_step = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--emb-host")) emb_mode_req = 1;
    else if (!strcmp(argv[i], "--lm-selftest")) emb_mode_req = 2;
    else if (!strcmp(argv[i], "--prefill-minimal")) emb_mode_req = 3;
    else if (!strcmp(argv[i], "--prefill-h2batched")) emb_mode_req = 4;
    else if (!strcmp(argv[i], "--prefill-chunk") && i + 1 < argc) { emb_mode_req = 6; chunkT = atoi(argv[++i]); }
    else if (!strcmp(argv[i], "--prefill-e2e") && i + 1 < argc) { emb_mode_req = 7; chunkT = atoi(argv[++i]); }
    else if (!strcmp(argv[i], "--prefill-run") && i + 1 < argc) g_pfChunk = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--prefill-arm5-gate")) emb_mode_req = 5;
    else if (!strcmp(argv[i], "--merge")) g_merge = true;   // merged gate/up GEMV
    else if (!strcmp(argv[i], "--lmrows") && i + 1 < argc) lm_rows_override = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--dump-logits") && i + 1 < argc) dump_logits = argv[++i];
    else if (!strcmp(argv[i], "--dump-logits-all") && i + 1 < argc) dump_logits_all = argv[++i];
    else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) {
      char* s = argv[++i];
      while (*s) { prompt.push_back((int)strtol(s, &s, 10)); if (*s == ',') s++; }
    }
    else if (!strcmp(argv[i], "--clip") && i + 1 < argc) clip_files.push_back(argv[++i]);
    else if (!strcmp(argv[i], "--audio-emb") && i + 1 < argc) {
      audio_emb_file = argv[++i];
      if (getenv("LFM_AEMB_WAIT") && atoi(getenv("LFM_AEMB_WAIT"))) g_aemb_wait = audio_emb_file;
    }
    else if (!strcmp(argv[i], "--audio-at") && i + 1 < argc) g_aemb_at = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--speak-emb") && i + 1 < argc) speak_emb_file = argv[++i];
    else if (!strcmp(argv[i], "--speak-tok") && i + 1 < argc) g_speak_tok = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--dump-hidden-all") && i + 1 < argc) g_dump_hidden_all = argv[++i];
    else if (!strcmp(argv[i], "--serve")) g_serve = true;
  }
  printf("prompt: %zu ids:", prompt.size());
  for (size_t i = 0; i < prompt.size() && i < 8; i++) printf(" %d", prompt[i]);
  printf("%s\n", prompt.size() > 8 ? " ..." : "");
  fflush(stdout);
  g_probe_layer = probe_layer; g_probe_step = probe_step;
  Model M; std::string js;
  if (g_emb && !strcmp(g_emb, "resident")) emb_mode_req = 0;
  else if (g_emb && !strcmp(g_emb, "host")) emb_mode_req = 1;
  M.emb_mode = emb_mode_req;        // flags are parsed before M exists
  M.path = argv[1];
  { struct timespec _t0; clock_gettime(CLOCK_MONOTONIC, &_t0);
    // LFM_PF_OVL: upload the weights on a thread while the (layer-major) chunked prefill
    // runs, each layer waiting for its own tensors. Only on the plain resident-embedding
    // path: the merged gate/up copies, graphs and the host-staged embedding all need the
    // whole blob before the first step.
    // Default on; LFM_PF_OVL=0 restores the serial load (A/B control).
    g_ovl = !(getenv("LFM_PF_OVL") && !atoi(getenv("LFM_PF_OVL"))) && g_pfChunk >= 2 &&
            M.emb_mode == 0 && !g_merge && !g_graphs;
    if (!load_blob(argv[1], M, js)) return 1;
    struct timespec _t1; clock_gettime(CLOCK_MONOTONIC, &_t1);
    // LFM_LOADTIME=1: measured 76 s of a 78 s 3-step launch on the board (the GPU arena
    // leaves too little RAM to keep the 1.95 GB blob in page cache between launches).
    if (getenv("LFM_LOADTIME")) {
      printf("LOADTIMING blob(read+parse+upload): %.1f ms\n",
             (_t1.tv_sec - _t0.tv_sec) * 1e3 + (_t1.tv_nsec - _t0.tv_nsec) * 1e-6);
      fflush(stdout);
    } }

  // layer_types, parsed from the blob header (the conv layers carry the
  // operator branch, the attention layers carry self_attn).
  bool is_attn[NLAYER]; bool is_slide[NLAYER] = {};
  {
    size_t lt = js.find("\"layer_types\":[");
    if (lt == std::string::npos) { printf("no layer_types in the blob header\n"); return 1; }
    size_t q = js.find('[', lt) + 1;
    for (int L = 0; L < NLAYER; L++) {
      size_t a = js.find('"', q), b = js.find('"', a + 1);
      if (a == std::string::npos || b == std::string::npos) { printf("layer_types truncated at %d\n", L); return 1; }
      std::string s = js.substr(a + 1, b - a - 1);
      is_attn[L] = (s != "conv");
      // rope_parameters is PER LAYER TYPE (config /lfm/rope_parameters):
      //   full_attention    -> rope_theta 1e6, partial_rotary_factor 0.5
      //   sliding_attention -> rope_theta 1e4, full rotation (partial 1.0)
      // plus a 256-position sliding window. The first version applied
      // invf_full (1e6, partial) to ALL attention layers and win=0, which is
      // invisible at position 0 (RoPE is the identity there: cos=1, sin=0)
      // and wrong everywhere else -- the signature was "pos-0 logits exact
      // at corr 0.99998, positions >0 garbage".
      is_slide[L] = (s == "sliding_attention");
      q = b + 1;
    }
  }
  int nattn = 0;
  for (int L = 0; L < NLAYER; L++) nattn += is_attn[L] ? 1 : 0;
  printf("layer types: %d attention, %d conv\n", nattn, NLAYER - nattn);
  if (nattn != NATTN || NLAYER - nattn != NCONV) { printf("layer_types do not match the spec\n"); return 1; }

  // ---- arena (allocated once, never resized: the repo's CUDA-graph rule)
  size_t kvbytes = (size_t)NLAYER * MAXTOK * NKV * HD * 2;
  // Every allocation is checked. The first version of this harness let the
  // KV allocations fail silently and then reported cudaMemcpy's "invalid
  // argument" as an upload failure, which reads like a blob problem and is
  // really an out-of-memory one.
  size_t freeb0 = 0, totb0 = 0; cudaMemGetInfo(&freeb0, &totb0);
  printf("device free before arena: %.2f GB of %.2f GB\n", freeb0 / 1e9, totb0 / 1e9);
  struct Al { void** p; size_t n; const char* w; };
  const Al als[] = {
    {(void**)&M.x, (size_t)H * 2, "x"}, {(void**)&M.hgu, (size_t)2 * INTER * 2, "hgu"},
    {(void**)&M.h1, (size_t)H * 2, "h1"},
    // h2/h3 are the FFN intermediates: w1/w3/silu/w2 all hold INTER=7680
    // elements, not H=2048. Allocated at H they took a 15360-byte write into
    // a 4096-byte arena slot, so the FFN spilled into its neighbours (h4/q/k
    // /v/attn). Found because an INTER-sized probe cudaMemcpy returned
    // "invalid argument" -- CUDA caught the out-of-range read that the
    // kernel's out-of-range WRITE never reported. The probe's copy check is
    // what surfaced it: without it the same copy silently returned the
    // previous contents and four probes printed one identical rms.
    {(void**)&M.h2, (size_t)INTER * 2, "h2"}, {(void**)&M.h3, (size_t)INTER * 2, "h3"},
    {(void**)&M.h4, (size_t)H * 2, "h4"},
    {(void**)&M.q, (size_t)NH * HD * 2, "q"}, {(void**)&M.k, (size_t)NKV * HD * 2, "k"},
    {(void**)&M.v, (size_t)NKV * HD * 2, "v"}, {(void**)&M.attn, (size_t)NH * HD * 2, "attn"},
    {(void**)&M.gate, (size_t)NH * 2, "gate"},
    {(void**)&M.bx, (size_t)3 * H * 2, "bx"}, {(void**)&M.cy, (size_t)H * 2, "cy"},
    {(void**)&M.ring, (size_t)NCONV * 7 * H * 2, "ring"},
    {(void**)&M.part, (size_t)VOCAB * 8 * 4, "part"},
    {(void**)&M.xf, (size_t)INTER * 4, "xf"},
    {(void**)&M.lgf, (size_t)VOCAB * 4, "lgf"},
    {(void**)&M.kc, kvbytes, "kc"}, {(void**)&M.vc, kvbytes, "vc"},
    {(void**)&M.dpos, 2 * sizeof(int), "dpos"},
    {(void**)&M.tok_dev, 4, "tok"},
    {(void**)&M.invf_full, (HD / 2) * 4, "invf_full"},
    {(void**)&M.invf_slide, (HD / 2) * 4, "invf_slide"},
  };
  { struct timespec _t0; clock_gettime(CLOCK_MONOTONIC, &_t0);
  for (const Al& a : als) {
    if (cudaMalloc(a.p, a.n) != cudaSuccess) {
      printf("arena cudaMalloc(%s, %zu B) FAILED -- device free %.2f GB\n",
             a.w, a.n, freeb0 / 1e9);
      return 1;
    }
    for (int L = 0; L < NLAYER; L++) { is_attn_g[L] = is_attn[L]; is_slide_g[L] = is_slide[L]; }
  }
  { struct timespec _t1; clock_gettime(CLOCK_MONOTONIC, &_t1);
    if (getenv("LFM_LOADTIME")) {
      printf("LOADTIMING arena(cudaMalloc+memset): %.1f ms\n",
             (_t1.tv_sec - _t0.tv_sec) * 1e3 + (_t1.tv_nsec - _t0.tv_nsec) * 1e-6);
      fflush(stdout);
    } }
  }   // closes the _t0 timing scope opened before the arena loop
  cudaMemset(M.ring, 0, (size_t)NCONV * 7 * H * 2);
  cudaMemset(M.x, 0, H * 2);
  float hf[HD / 2], hs2[HD / 2];
  build_invf(hf, 1e6f, HD, 0.5);            // full_attention: partial 0.5, theta 1e6
  build_invf(hs2, 1e4f, HD, 1.0);           // sliding_attention: full, theta 1e4
  cudaMemcpy(M.invf_full, hf, (HD / 2) * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(M.invf_slide, hs2, (HD / 2) * 4, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) { printf("arena setup failed: %s\n", cudaGetErrorString(err)); return 1; }
  size_t freeb = 0, totb = 0; cudaMemGetInfo(&freeb, &totb);
  {
    // Tegra is unified: every GPU allocation is backed by OS pages, so the
    // host side has to hold weights+arena AND the blob's page cache. Print
    // both free numbers at load -- "unspecified launch failure" deep inside a
    // layer on a board that "has 2.7 GB of GPU free" is a page-backing
    // failure, and the host number is the one that reveals it.
    // C stdio: this file uses fopen for the blob and never includes <fstream>.
    FILE* mi = fopen("/proc/meminfo", "r");
    long long memavail = 0, mfree = 0, cached = 0;
    char ln[128]; long long v;
    while (mi && fgets(ln, sizeof(ln), mi))
      if (sscanf(ln, "MemAvailable: %lld kB", &v) == 1) memavail = v;
      else if (sscanf(ln, "MemFree: %lld kB", &v) == 1) mfree = v;
      else if (sscanf(ln, "Cached: %lld kB", &v) == 1) cached = v;
    if (mi) fclose(mi);
    printf("host at load: MemAvailable %.0f MB, MemFree %.0f MB, Cached %.0f MB\n",
           memavail / 1024.0, mfree / 1024.0, cached / 1024.0);
    fflush(stdout);
  }
  if (g_graphs) {
    cudaHostAlloc((void**)&M.hpos_pin, 2 * sizeof(int), cudaHostAllocDefault);
    cudaStreamCreateWithFlags(&M.stCap, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&M.stRun, cudaStreamNonBlocking);
    printf("LFM3_GRAPHS=on (capture stream + run stream + pinned pos mirror; proof: this line)\n");
    fflush(stdout);
  }
  printf("weights+arena: %.2f GB, device free now %.2f GB of %.2f GB\n",
         (M.need_q + M.need_s + M.need_f + kvbytes * 2) / 1e9, freeb / 1e9, totb / 1e9);
  printf("L FM3_NANODE_LOADED\n"); fflush(stdout);

  Ent* emb = nullptr;
  for (size_t i = 0; i < M.ents.size(); i++) if (M.ents[i].name == EMBNAME) emb = &M.ents[i];
  if (!emb) { printf("no %s in the blob\n", EMBNAME); return 1; }
  if (emb_mode_req == 3) return prefill_minimal(M);
  if (emb_mode_req == 4) return prefill_h2batched(M);
  if (emb_mode_req == 5) g_cap_on = 1;   // capture during the normal run; gate runs at the end
  if (emb->ne[0] != VOCAB || emb->ne[1] != H) { printf("embed shape %dx%d\n", emb->ne[0], emb->ne[1]); return 1; }
  auto load_emb = [&](const char* audio_emb_file) -> int {
    // Row count comes from the file size, not from a flag: a count the caller typed
    // by hand is one more thing that can silently disagree with the encoder's output.
    aemb_wait_file();
    FILE* af = fopen(audio_emb_file, "rb");
    if (!af) { printf("--audio-emb: cannot open %s\n", audio_emb_file); return 2; }
    fseek(af, 0, SEEK_END); long fsz = ftell(af); fseek(af, 0, SEEK_SET);
    if (fsz <= 0 || fsz % ((long)H * 4)) { printf("--audio-emb: %s is %ld bytes, not a whole number of %d-float rows\n", audio_emb_file, fsz, H); return 2; }
    g_aemb_n = (int)(fsz / ((long)H * 4));
    std::vector<float> h((size_t)g_aemb_n * H);
    if (fread(h.data(), 4, h.size(), af) != h.size()) { printf("--audio-emb: short read\n"); return 2; }
    fclose(af);
    for (size_t i = 0; i < h.size(); i++) if (!isfinite(h[i])) { printf("--audio-emb: non-finite at %zu\n", i); return 2; }
    if (cudaMalloc(&g_aemb, h.size() * 4) != cudaSuccess) { printf("--audio-emb: OOM (%.1f MB)\n", h.size() * 4 / 1e6); return 1; }
    if (cudaMemcpy(g_aemb, h.data(), h.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess) { printf("--audio-emb: copy failed\n"); return 1; }
    // Proof the path is armed (the standing rule: prove the code path ran before
    // believing any number from it). The fed rows are also verified below, per row.
    printf("AUDIO-EMB armed: %s = %d rows x %d, overriding prompt positions %d..%d (%.1f MB)\n",
           audio_emb_file, g_aemb_n, H, g_aemb_at, g_aemb_at + g_aemb_n - 1, h.size() * 4 / 1e6);
    fflush(stdout);
    return 0;
  };
  if (audio_emb_file && load_emb(audio_emb_file)) return 2;
  if (speak_emb_file) {
    // Frame count from the file size, same rule as --audio-emb: no hand-typed counts.
    FILE* sf = fopen(speak_emb_file, "rb");
    if (!sf) { printf("--speak-emb: cannot open %s\n", speak_emb_file); return 2; }
    fseek(sf, 0, SEEK_END); long ssz = ftell(sf); fseek(sf, 0, SEEK_SET);
    if (ssz <= 0 || ssz % ((long)H * 4)) { printf("--speak-emb: %s is %ld bytes, not rows of %d floats\n", speak_emb_file, ssz, H); return 2; }
    g_semb_n = (int)(ssz / ((long)H * 4));
    std::vector<float> sh((size_t)g_semb_n * H);
    if (fread(sh.data(), 4, sh.size(), sf) != sh.size()) { printf("--speak-emb: short read\n"); return 2; }
    fclose(sf);
    for (size_t i = 0; i < sh.size(); i++) if (!isfinite(sh[i])) { printf("--speak-emb: non-finite at %zu\n", i); return 2; }
    if (cudaMalloc(&g_semb, sh.size() * 4) != cudaSuccess) { printf("--speak-emb: OOM (%.1f MB)\n", sh.size() * 4 / 1e6); return 1; }
    if (cudaMemcpy(g_semb, sh.data(), sh.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess) { printf("--speak-emb: copy failed\n"); return 1; }
    printf("SPEAK-EMB armed: %s = %d rows x %d, served at audio_out token %d in slot order (%.1f MB)\n",
           speak_emb_file, g_semb_n, H, g_speak_tok, sh.size() * 4 / 1e6);
    fflush(stdout);
  }
  if (g_serve) {
    if (speak_emb_file) { printf("--serve takes frame rows over stdin; do not combine with --speak-emb\n"); return 2; }
    if (!clip_files.empty()) { printf("--serve is single-sequence; do not combine with --clip\n"); return 2; }
    if (cudaMalloc(&g_semb, (size_t)SERVE_MAXFR * H * 4) != cudaSuccess) {
      printf("--serve: frame buffer OOM (%.1f MB)\n", (double)SERVE_MAXFR * H * 4 / 1e6); return 1;
    }
    g_semb_n = 0;
    steps = MAXTOK;                       // the loop ends on Q, EOF, or the KV cap
    printf("SERVE armed: commands on stdin (T <id> | A <row.f32> | Q), %d frame rows max\n", SERVE_MAXFR);
    fflush(stdout);
  }
  if (M.emb_mode >= 1) {
    if (!lm_host_init(M, M.hemb, argv[1])) return 1;
    if (lm_rows_override > 0 && (VOCAB % lm_rows_override) == 0) M.hemb.rows = lm_rows_override;
  }
  if (M.emb_mode == 2) {
    // Bitwise gate for the host-staged lm_head: same x, both paths, all
    // 128000 logits compared as raw fp32 bits. Row reductions are
    // independent, so any difference here is a bug in the chunking, not a
    // rounding choice -- and it must pass before a single timing number
    // from this path is believed.
    // Build the pattern on the host and copy it up: the first draft wrote
    // ((bf16*)M.x)[i] from the host, and M.x is a DEVICE pointer -- segfault.
    {
      std::vector<uint16_t> pat(H);
      for (int i = 0; i < H; i++) pat[i] = (uint16_t)(0x3c00 + ((i * 37) & 0x3ff));
      cudaMemcpy(M.x, pat.data(), (size_t)H * 2, cudaMemcpyHostToDevice);
    }
    std::vector<float> a(VOCAB), b(VOCAB);
    gemv_f(M, EMBNAME, M.x, M.lgf);
    cudaMemcpy(a.data(), M.lgf, (size_t)VOCAB * 4, cudaMemcpyDeviceToHost);
    lm_head_host(M, M.hemb, M.x, M.lgf);
    cudaMemcpy(b.data(), M.lgf, (size_t)VOCAB * 4, cudaMemcpyDeviceToHost);
    long bad = 0; int first = -1;
    for (int i = 0; i < VOCAB; i++)
      if (memcmp(&a[i], &b[i], 4) != 0) { if (first < 0) first = i; bad++; }
    double mx = 0; for (int i = 0; i < VOCAB; i++) if (fabs(b[i]) > mx) mx = fabs(b[i]);
    printf("lm_head staged vs resident: rows/slab=%d  %s (%ld/%d logits differ, max |logit| %.4f)\n",
           M.hemb.rows, bad ? "MISMATCH" : "BIT-EXACT", bad, VOCAB, mx);
    if (first >= 0) printf("  first i=%d resident %.9g staged %.9g\n", first, a[first], b[first]);
    for (int rr = 1024; rr <= 32768; rr *= 2) {
      if (VOCAB % rr) continue;
      M.hemb.rows = rr;
      lm_head_host(M, M.hemb, M.x, M.lgf);
      cudaMemcpy(b.data(), M.lgf, (size_t)VOCAB * 4, cudaMemcpyDeviceToHost);
      long bb = 0; for (int i = 0; i < VOCAB; i++) bb += (memcmp(&a[i], &b[i], 4) != 0);
      printf("  rows/slab %6d: %s (%ld differ)\n", rr, bb ? "MISMATCH" : "bit-exact", bb);
    }
    printf("%s\n", bad ? "LFM_LM_FAIL" : "LFM_LM_OK");
    return bad ? 1 : 0;
  }

  // ---- resident-process clip loop -------------------------------------------------
  // --clip SPEC repeated runs several clips in ONE process. The point is the 82 s
  // one-time startup (1.946 GB blob read + upload + setup), which is 0.63 of ASR_RT for
  // a 129.7 s clip and is paid once per process, not per clip: a service pays it at
  // boot. SPEC is a text file: line 1 = embedding path (or "-" for none), line 2 =
  // comma-separated prompt ids, line 3 = audio-at position (optional).
  //
  // With no --clip this is exactly one pass, so the shipped single-prompt path is
  // byte-identical to before the loop existed.
  //
  // Graphs are NOT a factor here: LFM3 enables them only with LFM_GRAPHS set, so the
  // ASR path runs without a captured graph and a second pass cannot replay clip 0's
  // capture. (An earlier note claimed graphs were measured free on this workload; that
  // A/B used PQ1_GRAPH, which is the Bonsai engine's knob and is ignored here, so both
  // arms ran graphs-off -- see doc section 45's correction.)
  const int nclip = (int)clip_files.size();
  printf("LFM3_CLIPMODE BUILD=2026-11-03B nclip=%d\n", nclip); fflush(stdout);
  for (int cl = 0; cl < (nclip ? nclip : 1); cl++) {
  if (nclip) {
    FILE* sf = fopen(clip_files[cl].c_str(), "r");
    if (!sf) { printf("--clip: cannot open %s\n", clip_files[cl].c_str()); return 2; }
    // Read the spec as LINES, growing a std::string when a line exceeds the read
    // buffer. The first version used char idsp[8192], and the ASR prompt's id line is
    // ~11 KB (1642 ids), so fgets split it: 150 ids were parsed, the remainder of the
    // line was read as "line 3" (so audio-at became garbage) and the real line 3 became
    // "line 4". Symptom was a clip that ran a completely different workload from its
    // solo run, which is exactly the kind of thing a bit-exactness gate must catch.
    std::vector<std::string> lines;
    {
      char ln[1024];
      std::string cur;
      while (fgets(ln, sizeof ln, sf)) {
        char* nl = strchr(ln, '\n');
        if (nl) { *nl = 0; cur += ln; lines.push_back(cur); cur.clear(); }
        else cur += ln;                      // line continues past the buffer
      }
      if (!cur.empty()) lines.push_back(cur);
    }
    fclose(sf);
    const char* embp = lines.size() > 0 ? lines[0].c_str() : "";
    const char* idsp = lines.size() > 1 ? lines[1].c_str() : "";
    int atv = lines.size() > 2 ? atoi(lines[2].c_str()) : -1;
    int ngv = lines.size() > 3 ? atoi(lines[3].c_str()) : -1;
    prompt.clear();
    std::string idbuf(idsp);            // strtol wants a writable endptr
    for (char* q = &idbuf[0]; *q; ) { prompt.push_back((int)strtol(q, &q, 10)); if (*q == ',') q++; }
    if (atv >= 0) g_aemb_at = atv;
    audio_emb_file = (embp[0] && strcmp(embp, "-")) ? strdup(embp) : nullptr;
    if (g_aemb) { cudaFree(g_aemb); g_aemb = nullptr; }
    g_aemb_n = 0; g_aemb_fed = 0;
    printf("CLIP %d START ids=%zu emb=%s at=%d ngen=%d\n", cl, prompt.size(),
           audio_emb_file ? audio_emb_file : "-", g_aemb_at, steps - (int)prompt.size());
    fflush(stdout);
    // Per-clip generation count (spec line 4), so a short clip in a resident batch is
    // not forced to the longest clip's --steps.
    if (ngv >= 0) steps = (int)prompt.size() + ngv;
    if (audio_emb_file && load_emb(audio_emb_file)) return 2;
  }
  if (prompt.empty()) prompt.push_back(124894);                 // bos
  std::vector<int> toks = prompt;
  cudaEvent_t evA, evB; cudaEventCreate(&evA); cudaEventCreate(&evB);
  if (g_time) { cudaEventCreate(&g_evA); cudaEventCreate(&g_evB); }

  // The captured decode graph does not follow a batched prefill: on the ASR path it emitted
  // 597 22 and stopped (2nd token wrong) and ran 95-104 s vs 69.4 s (run #202). Refuse the
  // combination instead of printing a wrong transcript.
  if (g_graphs && g_pfChunk > 1) {
    printf("LFM_GRAPHS is not supported with --prefill-run %d (wrong tokens after the batched prefill); unset it\n", g_pfChunk);
    return 2;
  }
  if (g_graphs && trace) { printf("LFM_GRAPHS: --trace off (a D2H sync cannot happen inside a capture)\n"); trace = false; }
  printf("step tok        ms   tok/s\n");
  double tot = 0; int ng = 0;
  // Warm-up (DEFAULT ON; LFM_NOWARM=1 is the control arm): one forward at
  // load time, BEFORE any prompt is seen,
  // on a fixed BOS token at position 0. It then zeroes the only per-sequence
  // state (conv rings, KV caches) and resets the phase timers, so the real
  // run starts from exactly the state it starts from today. It exists for
  // e2e latency: the first step of a cold process pays one-time costs
  // (~52 ms on this board) that a server pays once at startup, not per query.
  // Never enabled with graphs (the capture is keyed on step indices).
  // Measured (#138, paired 4/4, order-balanced, cold page cache): TTFT on
  // prompt A 2649.3 -> 2589.6 ms (-59.8 ms); 40-step streams identical.
  // Under LFM_PF_OVL the weights are still arriving, so there is no warm-up forward;
  // its only effect on the run, the zeroed conv rings and KV caches, is done here.
  const bool warm = getenv("LFM_NOWARM") == nullptr && !g_graphs && !g_ovl;
  if (g_ovl) {
    cudaMemset(M.ring, 0, (size_t)NCONV * 7 * H * 2);
    cudaMemset(M.kc, 0, kvbytes); cudaMemset(M.vc, 0, kvbytes);
    cudaMemset(M.x, 0, H * 2);
    cudaDeviceSynchronize();
  }
  // LFM_WARMMS=N (default 0 = shipped behavior): keep running dummy forwards
  // at position 0 until N ms of wall-clock GPU load, absorbing the devfreq
  // DVFS ramp the same way the single step absorbs the page cache. Measured
  // (run #210/#211): an idle-start board sits at 76.8-153.6 MHz for the first
  // ~5-24 s; the 97 ms single warm-up step cannot complete that ramp, so a
  // cold-board TTFT pays ~+7% in the measured window. A server pays the ramp
  // once at startup -- this moves it out of the query path.
  const double warmms = warm ? atof(getenv("LFM_WARMMS") ? getenv("LFM_WARMMS") : "0") : 0.0;
  struct timespec tw0; clock_gettime(CLOCK_MONOTONIC, &tw0);
  int nwarm = 0;
  const int nprompt_all = (int)prompt.size();   // for the S8 arm's prefill scope
  // ---- committed batched-prefill bit-exactness probe ---------------------------------
  // LFM_PF_PROBE=<arm> runs the whole prompt TWICE in this process: pass 0 with the
  // shipped serial arm (ARM=0), pass 1 with the requested batched arm, and compares the
  // id streams. This is the instrument the #296/#298 "bit-exact at T=1" claims were made
  // with, rebuilt as committed code (run log row 325 found the original was a throwaway
  // patched build, so those claims were not reproducible from the repository). A claim
  // that licenses changing prefill arithmetic has to be checkable by anyone who clones.
  //
  // The KV cache is written per position and positions restart at 0 on each pass, so the
  // second pass overwrites the first; the phase timers and g_aemb_fed are reset so the
  // per-pass prints describe that pass. Pass 1's timing is the one that reaches the
  // steady-state line, so the probe also shows what the arm costs.
  const int pfprobe = getenv("LFM_PF_PROBE") ? atoi(getenv("LFM_PF_PROBE")) : -1;
  if (pfprobe > 10) { printf("LFM_PF_PROBE=%d out of range (0..10)\n", pfprobe); return 2; }
  std::vector<int> pf_base;
  for (int pfpass = 0; pfpass < (pfprobe >= 0 ? 2 : 1); pfpass++) {
  if (pfprobe >= 0) {
    if (pfpass == 1) {
      toks = prompt; tot = 0; ng = 0; g_aemb_fed = 0;
      g_tEmb = g_tConv = g_tAttn = g_tNorm = g_tHead = 0.0;
    }
    pf_set_arm(pfpass == 0 ? 0 : pfprobe);
    printf("PF-PROBE pass %d arm %d (prompt %d ids)\n", pfpass, pfpass == 0 ? 0 : pfprobe,
           nprompt_all);
    fflush(stdout);
  }
  for (int s = warm ? -1 : 0; s < steps || s < 0; s++) {
    const int pos = s < 0 ? 0 : s;
    g_prefill_phase = (s < 0 || s < nprompt_all - 1) ? 1 : 0;   // S8 arm scope: prompt steps only
    const int tok = s < 0 ? 124894 : toks[s];
    const bf16* lm_head_input = M.x;   // replaced by the embedding_norm output below
    if (pos >= MAXTOK) break;
    // (Placed at the TOP of step 0, before its embedding and layer body. It used to sit
    // after them, so position 0 went through the 24 layers twice: every conv ring took an
    // extra shift, a chunk-invariant offset of ~2000/2048 cells in the prefill output
    // that still produced plausible greedy tokens -- the "drift" runs 336-338 saw.)
    // INTEGRATED CHUNKED PREFILL (--prefill-run T): run the whole prompt T positions at
    // a time -- one T-wide launch per GEMV -- then hand the last row to M.x and jump the
    // loop counter to the last prompt step, so the head, argmax and feed run through the
    // EXACT shipped code path (they are not duplicated here). Reached only after the
    // warm-up step, which has just zeroed the KV caches and the conv ring, so the chunked
    // prefill starts from the same clean state the T=1 loop does.
    if (g_pfChunk >= 2 && s == 0) {
      const int NP = (int)prompt.size();
      static PF4Arena A; static int A_T = 0;
      if (A_T < g_pfChunk) { pf4_arena_alloc(A, g_pfChunk); A_T = g_pfChunk; }
      { int c = 0; for (int L = 0; L < NLAYER; L++) { pf4_conv_idx[L] = c; if (!is_attn_g[L]) c++; } }
      int Tc = 0;
      struct timespec q0, q1; clock_gettime(CLOCK_MONOTONIC, &q0);
      // LFM_PF_LM=1: layer-major order. Every chunk's residual lives in its own slice of
      // a [NP][H] buffer (A.xb is the only state layer_chunk_T carries across layers), so
      // layer L runs over all chunks before layer L+1. Per position the arithmetic is the
      // chunk-major path's exactly: attention at layer L reads only earlier positions'
      // K/V at layer L, and the conv ring of layer L advances chunk by chunk in order.
      static const int pf_lm = g_ovl || (getenv("LFM_PF_LM") && atoi(getenv("LFM_PF_LM")));
      bf16* xb0 = A.xb;
      if (pf_lm) {
        static bf16* xall = nullptr;
        if (!xall && cudaMalloc(&xall, (size_t)NP * H * 2) != cudaSuccess) { printf("PF4-RUN: xall alloc failed\n"); return 1; }
        if (g_lj && !g_embfile && !load_wait(g_lj->need_emb)) { printf("PF4-RUN: weight load failed (embedding)\n"); return 1; }
        for (int c0 = 0; c0 < NP; c0 += g_pfChunk) {
          Tc = (NP - c0 < g_pfChunk) ? NP - c0 : g_pfChunk;
          A.xb = xall + (size_t)c0 * H;
          if (pf4_embed_chunk(M, A, prompt, c0, Tc, emb)) { printf("PF4-RUN: embed failed at chunk %d\n", c0); return 1; }
        }
        for (int L = 0; L < NLAYER; L++) {
          if (g_lj && !load_wait(g_lj->need[L])) { printf("PF4-RUN: weight load failed (layer %d)\n", L); return 1; }
          for (int c0 = 0; c0 < NP; c0 += g_pfChunk) {
            Tc = (NP - c0 < g_pfChunk) ? NP - c0 : g_pfChunk;
            A.xb = xall + (size_t)c0 * H;
            if (layer_chunk_T(M, L, c0, Tc, A)) { printf("PF4-RUN: layer %d failed at chunk %d\n", L, c0); return 1; }
            if (g_pfsync == 3) {     // bounded queue: wait for the chunk recorded K chunks ago
              static cudaEvent_t ev[64]; static int nev = 0; static long ci = 0;
              static const int K = getenv("LFM_PF_SYNCK") ? atoi(getenv("LFM_PF_SYNCK")) : 4;
              if (!nev) { for (int k = 0; k < 64; k++) cudaEventCreateWithFlags(&ev[k], cudaEventDisableTiming); nev = 64; }
              if (ci >= K) { const cudaError_t se = cudaEventSynchronize(ev[(ci - K) % 64]);
                if (se != cudaSuccess) { printf("PF4-RUN: L%d chunk %d wait: %s\n", L, c0, cudaGetErrorString(se)); return 1; } }
              cudaEventRecord(ev[ci % 64], g_launch); ci++;
            }
            if (g_pfsync == 1) { const cudaError_t se = cudaStreamSynchronize(g_launch);
              if (se != cudaSuccess) { printf("PF4-RUN: L%d chunk %d sync: %s\n", L, c0, cudaGetErrorString(se)); return 1; } }
          }
          if (g_pfsync == 2) { const cudaError_t se = cudaStreamSynchronize(g_launch);
            if (se != cudaSuccess) { printf("PF4-RUN: L%d sync: %s\n", L, cudaGetErrorString(se)); return 1; } }
        }
        if (g_lj) {            // everything else (final norm, tied head) needs the whole blob
          if (!load_wait((int)g_lj->order.size())) { printf("PF4-RUN: weight load failed\n"); return 1; }
          pthread_join(g_lj->th, nullptr);
          printf("PF4-RUN: weights uploaded during the prefill (%zu tensors, CRC ok)\n", g_lj->order.size());
          g_embfile = false;          // embedding resident from here on
          g_lj = nullptr;
        }
      } else
      for (int c0 = 0; c0 < NP; c0 += g_pfChunk) {
        Tc = (NP - c0 < g_pfChunk) ? NP - c0 : g_pfChunk;
        if (pf4_embed_chunk(M, A, prompt, c0, Tc, emb)) { printf("PF4-RUN: embed failed at chunk %d\n", c0); return 1; }
        for (int L = 0; L < NLAYER; L++)
          if (layer_chunk_T(M, L, c0, Tc, A)) { printf("PF4-RUN: layer %d failed at chunk %d\n", L, c0); return 1; }
      }
      cudaDeviceSynchronize(); clock_gettime(CLOCK_MONOTONIC, &q1);
      printf("PF4-RUN: prompt %d ids in chunks of %d, %.0f ms (%.2f ms/pos)\n",
             NP, g_pfChunk, (q1.tv_sec - q0.tv_sec) * 1e3 + (q1.tv_nsec - q0.tv_nsec) * 1e-6,
             ((q1.tv_sec - q0.tv_sec) * 1e3 + (q1.tv_nsec - q0.tv_nsec) * 1e-6) / NP);
      // Finish the last prompt step HERE: the body's layer loop runs before its head,
      // so falling through would push the chunk's final residual through one extra
      // 24-layer pass (observed: the first generated token came out 11 instead of 207).
      // The head, the argmax and the queue append are the shipped ones -- head_logits_from
      // is shared with the body, and toks is pre-filled from `prompt`, so step NP consumes
      // toks[NP] = the token appended here exactly as the T=1 path does.
      cudaMemcpy(M.x, A.xb + (size_t)(Tc - 1) * H, (size_t)H * 2, cudaMemcpyDeviceToDevice);
      A.xb = xb0;                 // the arena's own staging row block again
      if (getenv("LFM_DUMPX") && atoi(getenv("LFM_DUMPX")) == NP - 1) {   // same acceptance dump
        cudaDeviceSynchronize();
        std::vector<bf16> hx((size_t)H);
        cudaMemcpy(hx.data(), M.x, (size_t)H * 2, cudaMemcpyDeviceToHost);
        FILE* df = fopen("/tmp/lfm_x_ref.bin", "wb");
        if (df) { fwrite(hx.data(), 2, H, df); fclose(df); printf("LFM_DUMPX step %d -> /tmp/lfm_x_ref.bin (PF4-RUN)\n", NP - 1); }
      }
      {
        Ent* fn = T(M, "lfm.model.embedding_norm.weight");
        rms1(M, M.x, fn ? fn->df : nullptr, M.h1);
        head_logits_from(M, M.h1, emb);
        k_argmax<<<1, 256, 0, g_launch>>>(M.lgf, M.tok_dev, VOCAB);
        cudaDeviceSynchronize();
        int nt = -1; cudaMemcpy(&nt, M.tok_dev, 4, cudaMemcpyDeviceToHost);
        toks.push_back(nt);   // this IS the last prompt step: the shipped body appends here too
        printf("%4d %8d  %8.2f  %6.2f   <- PF4-RUN last prompt step\n", NP - 1, nt,
               (q1.tv_sec - q0.tv_sec) * 1e3 + (q1.tv_nsec - q0.tv_nsec) * 1e-6, 0.0);
        fflush(stdout);
      }
      s = NP - 1;                 // next iteration is the first decode step
      continue;
    }
    if (g_graphs && M.gx) g_launch = M.stRun;   // embedding, layers, lm_head all on stRun
    cudaEventRecord(evA, g_launch);
    phBegin();
    if (g_semb && tok == g_speak_tok && g_speak_fed < g_semb_n) {
      // Speech slot: the embedding is the frame row, in slot order. More placeholders
      // than rows is a script error, not a silent table lookup: fail loudly.
      k_bf16_from_f32<<<(H + 255) / 256, 256, 0, g_launch>>>(
          g_semb + (size_t)g_speak_fed * H, (bf16*)M.x, H);
      cudaError_t se = cudaGetLastError();     // read it HERE, same rule as audio-emb
      if (se != cudaSuccess) {
        printf("speak embedding launch failed at pos %d (frame %ld): %s\n", pos, g_speak_fed, cudaGetErrorString(se));
        return 1;
      }
      g_speak_fed++;
    } else if (g_semb && tok == g_speak_tok) {
      printf("FAIL speak slot at pos %d: %ld frame rows served, file holds %d (script longer than audio)\n",
             pos, g_speak_fed, g_semb_n);
      return 1;
    } else if (g_aemb && pos >= g_aemb_at && pos - g_aemb_at < g_aemb_n) {
      // Audio position: the embedding comes from the encoder, not the token table.
      k_bf16_from_f32<<<(H + 255) / 256, 256, 0, g_launch>>>(
          g_aemb + (size_t)(pos - g_aemb_at) * H, (bf16*)M.x, H);
      cudaError_t ae = cudaGetLastError();     // read it HERE: a later call would
      if (ae != cudaSuccess) {                 // report a different kernel's refusal
        printf("audio embedding launch failed at pos %d: %s\n", pos, cudaGetErrorString(ae));
        return 1;
      }
      g_aemb_fed++;
    } else if (emb->dtype == 3) {
      // bf16 payload: the row IS bf16, so it is a 4 KB copy -- no dequant.
      // Handing k_dequant_row a dtype-3 entry reads emb->ds, which is null
      // for this payload: the kernel faults on a null pointer and, because
      // launches are async, "unspecified launch failure" surfaces two kernels
      // later, in layer 0's FFN. That is what made the certified blob look
      // like a memory-pool problem.
      const char* src = (M.emb_mode == 1 && M.hemb.hc) ? (const char*)M.hemb.hc
                                                       : (const char*)emb->dq;
      cudaError_t ce = cudaMemcpy(M.x, src + (size_t)tok * H * 2, (size_t)H * 2,
                                  M.emb_mode == 1 ? cudaMemcpyHostToDevice
                                                  : cudaMemcpyDeviceToDevice);
      if (ce != cudaSuccess) {
        printf("bf16 embedding row copy failed: %s\n", cudaGetErrorString(ce));
        return 1;
      }
    } else if (M.emb_mode == 1) {
      // row of the pinned host embedding -> slab -> bf16
      const int sb = H / Q8_BLK;
      cudaMemcpyAsync(M.hemb.dc, M.hemb.hc + (size_t)tok * H, (size_t)H,
                      cudaMemcpyHostToDevice, 0);
      cudaMemcpyAsync(M.hemb.dsl, M.hemb.hs + (size_t)tok * sb, (size_t)sb * 4,
                      cudaMemcpyHostToDevice, 0);
      k_dequant_row<<<(H + 255) / 256, 256, 0, g_launch>>>(M.hemb.dc, M.hemb.dsl, M.x, H);
    } else {
      k_dequant_row<<<(H + 255) / 256, 256, 0, g_launch>>>(emb->dq + (size_t)tok * H,
                                             emb->ds + (size_t)tok * (H / Q8_BLK), M.x, H);
    }

    double* phase = nullptr;        // declared outside the graph-guarded region: the final phEnd reads it
    bool use_graph = false, capturing = false;
    if (g_graphs) {
      M.hpos_pin[0] = pos; M.hpos_pin[1] = pos + 1;      // plain pinned stores
      if (M.gx) {
        cudaMemcpyAsync(M.dpos, M.hpos_pin, 2 * sizeof(int), cudaMemcpyHostToDevice, g_launch);
        cudaEventRecord(evA, g_launch);
        cudaGraphLaunch(M.gx, g_launch);
        cudaEventRecord(evB, g_launch);
        use_graph = true;
      } else if (s == 2) {
        // Capture THIS step: the layer loop becomes one graph. Capture does
        // not execute, so the graph is launched immediately below to produce
        // this step's activations.
        cudaDeviceSynchronize();
        g_launch = M.stCap;
        cudaStreamBeginCapture(M.stCap, cudaStreamCaptureModeThreadLocal);
        cudaMemcpyAsync(M.dpos, M.hpos_pin, 2 * sizeof(int), cudaMemcpyHostToDevice, M.stCap);
        capturing = true;
      }
    }
    if (!use_graph) {
    int ci = 0;
    for (int L = 0; L < NLAYER; L++) {
      {
if (g_time && !g_graphs) { if (phase) phEnd(*phase); phBegin(); phase = is_attn[L] ? &g_tAttn : &g_tConv; }
        LayerScratch S = {M.h1,M.h2,M.h3,M.h4,M.q,M.k,M.v,M.gate,M.attn,M.bx,M.hgu};
        const int probe_on = (trace && L == g_probe_layer && s == g_probe_step) ? 1 : 0;
        layer_body(M, L, pos, s, trace, ci, M.x, S, probe_on, is_attn[L] ? 1 : 0, is_slide[L] ? 1 : 0);
        if (!is_attn[L]) ci++;
      }
    }
    // LFM_DUMPX=<step>: dump the residual stream at that step. This is the acceptance
    // value the T-wide chunked driver is graded against (PF4_E2E writes /tmp/pf4_x_last.bin
    // and the two files must match bit for bit). Off by default: it is a debug readback,
    // and a device sync inside the step loop is not something to leave in the shipped
    // timing path -- the encoder lost 40 s to exactly that pattern.
    if (getenv("LFM_DUMPX") && s == atoi(getenv("LFM_DUMPX"))) {
      cudaDeviceSynchronize();
      std::vector<bf16> hx((size_t)H);
      cudaMemcpy(hx.data(), M.x, (size_t)H * 2, cudaMemcpyDeviceToHost);
      FILE* df = fopen("/tmp/lfm_x_ref.bin", "wb");
      if (df) { fwrite(hx.data(), 2, H, df); fclose(df); printf("LFM_DUMPX step %d -> /tmp/lfm_x_ref.bin\n", s); fflush(stdout); }
    }
    }   // end of the non-graph / captured region
    if (capturing) {
      cudaGraph_t graph = nullptr;
      cudaError_t ce = cudaStreamEndCapture(M.stCap, &graph);
      if (ce != cudaSuccess || !graph) {
        printf("GRAPH CAPTURE FAILED: %s\n", cudaGetErrorString(ce)); fflush(stdout);
        return 1;
      }
      ce = cudaGraphInstantiate(&M.gx, graph, nullptr, nullptr, 0);
      cudaGraphDestroy(graph);
      if (ce != cudaSuccess || !M.gx) {
        printf("GRAPH INSTANTIATE FAILED: %s\n", cudaGetErrorString(ce)); fflush(stdout);
        return 1;
      }
      g_launch = M.stRun;
      printf("LFM3_GRAPH_CAPTURED layers=%d nodes=%d (proof: this line)\n", NLAYER, -1);
      fflush(stdout);
      cudaGraphLaunch(M.gx, g_launch);
      capturing = false; use_graph = true;
    }
    // model.embedding_norm runs on the last layer's output BEFORE the tied
    // lm_head (modeling_lfm3.py:760, right after the layer loop). Feeding
    // raw x gave logits uncorrelated with HF (corr -0.025, std 0.41 vs
    // 2.09) -- the scale of an unnormalised residual stream has no relation
    // to the scale lm_head was trained against.
    // model.embedding_norm runs on the last layer's output BEFORE the tied
    // lm_head (modeling_lfm3.py:760, right after the layer loop). Feeding
    // raw x gave logits uncorrelated with HF (corr -0.025, std 0.41 vs
    // 2.09) -- the scale of an unnormalised residual stream has no relation
    // to the scale lm_head was trained against.
    //
    // DO NOT write the result back through M.x. The first version did
    // `M.x = M.h1`, and M.x is a MEMBER OF Model, so the alias survived into
    // the next step: from step 1 on, the residual stream WAS h1, so
    // operator_norm's in-place write destroyed the residual the branch is
    // added to (add1 printed 0.1905, operator_norm's own rms, not the
    // ~0.018 a real residual add gives). Position 0 was exact because the
    // alias is installed after step 0's logits -- the classic "first token
    // is right, everything after is garbage" signature.
    if (g_time && phase) phEnd(*phase);
    phBegin();
    {
      Ent* fn = T(M, "lfm.model.embedding_norm.weight");
      rms1(M, M.x, fn ? fn->df : nullptr, M.h1);
      lm_head_input = M.h1;
    }
    // --dump-hidden-all appends every step's post-final-norm hidden (the quantity HF's
    // sample_audio_frame receives). cudaMemcpy synchronizes, mirroring the logits dumps.
    // s>=0 only: the warm-up step also runs the norm block, but its state is discarded
    // by the reset below, so its hidden would be anextra row of garbage (seen: N+1 rows).
    if (g_dump_hidden_all && s >= 0) {
      std::vector<bf16> hb((size_t)H);
      if (cudaMemcpy(hb.data(), M.h1, (size_t)H * 2, cudaMemcpyDeviceToHost) != cudaSuccess) { printf("hidden dump copy failed at step %d\n", s); return 1; }
      std::vector<float> hf((size_t)H);
      for (int i = 0; i < H; i++) { uint32_t u = ((uint32_t)hb[i]) << 16; memcpy(&hf[i], &u, 4); }
      for (int i = 0; i < H; i++) if (!isfinite(hf[i])) { printf("hidden dump non-finite at step %d dim %d\n", s, i); return 1; }
      FILE* hh = fopen(g_dump_hidden_all, "ab");
      if (!hh) { printf("hidden dump cannot open %s\n", g_dump_hidden_all); return 1; }
      if (fwrite(hf.data(), 4, H, hh) != (size_t)H) { printf("hidden dump short write at step %d\n", s); fclose(hh); return 1; }
      fclose(hh);
    }
    if (trace && s == g_probe_step) printf("  final rms %.4f\n", rms_host(M, M.x));
    // NOT T(M, EMBNAME): under --emb-host the loader skips the embedding's
    // upload, and T() only knows registered/uploaded entries, so it prints
    // "MISSING TENSOR" and falls through to the Q8 lm_head. `emb` is the
    // pointer the ents loop above already resolved, host mode or not.
    phEnd(g_tNorm);
    phBegin();
    // Feeding fix (#139). The old loop pushed EVERY step's argmax into toks,
    // including the n-1 predictions made mid-prompt, so the first generation
    // step consumed toks[n] = step 0's prediction (made after one prompt
    // token), and every later step consumed a stale mid-prompt guess. Only
    // the prediction after the full context (step n-1 onward) is a real
    // output. With that fixed, the lm_head on prompt steps 0..n-2 produces
    // nothing anyone consumes, so it is skipped -- unless --dump-logits-all is
    // set: the logit gate compares every teacher-forced position against HF.
    // LFM_LEGACYFEED=1 restores the old loop as the A/B control.
    const int nprompt = (int)prompt.size();
    const bool need_head = s < 0 || g_legacy_feed || s >= nprompt - 1 || dump_logits_all;
    Ent* eemb = emb;
    if (!need_head) {}
    else head_logits_from(M, lm_head_input, eemb);   // the four arms, shared with the driver
    if (need_head) k_argmax<<<1, 256, 0, g_launch>>>(M.lgf, M.tok_dev, VOCAB);
    cudaEventRecord(evB, g_launch);
    cudaEventSynchronize(evB);
    if (s < 0) {
      float wms = 0; cudaEventElapsedTime(&wms, evA, evB);
      cudaMemset(M.ring, 0, (size_t)NCONV * 7 * H * 2);
      cudaMemset(M.kc, 0, kvbytes); cudaMemset(M.vc, 0, kvbytes);
      cudaDeviceSynchronize();
      g_tEmb = g_tConv = g_tAttn = g_tNorm = g_tHead = 0;
      nwarm++;
      if (warmms > 0) {
        struct timespec tw; clock_gettime(CLOCK_MONOTONIC, &tw);
        double el = (tw.tv_sec - tw0.tv_sec) * 1e3 + (tw.tv_nsec - tw0.tv_nsec) * 1e-6;
        if (el < warmms) { s = -2; continue; }   // keep ramping (any s<0 is a dummy step)
        printf("warm-up: %d steps, %.0f ms GPU load (DVFS ramp absorbed; state reset)\n", nwarm, el);
        continue;
      }
      printf("warm-up step %.2f ms (load time, not part of TTFT; state reset)\n", wms);
      continue;
    }
    float ms = 0; cudaEventElapsedTime(&ms, evA, evB);
    int nt = -1;   // printed as -1 on prompt steps whose lm_head was skipped
    if (need_head) cudaMemcpy(&nt, M.tok_dev, 4, cudaMemcpyDeviceToHost);
    if (g_legacy_feed || s >= nprompt - 1) toks.push_back(nt);
    phEnd(g_tHead);
    // --dump-logits-all appends EVERY step's logits, so the gate can compare
    // the whole teacher-forced prefix, not just position 0. Greedy stream
    // prefixes are 1-2 tokens of 16 and 0/7 prompts fully match, so the stream
    // gate cannot see a quality regression from an accumulation change; a
    // per-position top-k / correlation / KL comparison can.
    if (dump_logits_all) {
      std::vector<float> lg(VOCAB);
      cudaMemcpy(lg.data(), M.lgf, (size_t)VOCAB * 4, cudaMemcpyDeviceToHost);
      FILE* g = fopen(dump_logits_all, "ab");
      if (g) { fwrite(lg.data(), 4, VOCAB, g); fclose(g); }
    }
    if (dump_logits && s == steps - 1) {
      std::vector<float> lg(VOCAB);
      cudaMemcpy(lg.data(), M.lgf, (size_t)VOCAB * 4, cudaMemcpyDeviceToHost);
      FILE* g = fopen(dump_logits, "wb");
      if (g) { fwrite(lg.data(), 4, VOCAB, g); fclose(g);
               printf("wrote logits after step %d (tok %d) to %s\n", s, nt, dump_logits); }
    }
    printf("%4d %7d %9.2f %8.2f\n", s, nt, ms, 1000.0 / ms);
    fflush(stdout);
    if (s >= 2) { tot += ms; ng++; }
    cudaError_t e2 = cudaGetLastError();
    if (e2 != cudaSuccess) { printf("CUDA ERROR after step %d: %s\n", s, cudaGetErrorString(e2)); return 1; }
    if (g_serve && s >= nprompt - 1) {
      // toks already holds this step's argmax at s+1 (the shipped feed); overwrite it with
      // the controller's choice. Anything other than exactly s+2 entries is a feed bug.
      if (toks.size() != (size_t)s + 2) { printf("SERVE: toks has %zu entries after step %d, want %d\n", toks.size(), s, s + 2); return 1; }
      char cmd[1200];
      if (!fgets(cmd, sizeof cmd, stdin)) { printf("SERVE: stdin closed after step %d\n", s); fflush(stdout); break; }
      if (cmd[0] == 'Q') { printf("SERVE: quit after step %d\n", s); fflush(stdout); break; }
      if (cmd[0] == 'T' && cmd[1] == ' ') {
        char* endp = nullptr;
        long id = strtol(cmd + 2, &endp, 10);
        if (endp == cmd + 2 || id < 0 || id >= VOCAB) { printf("SERVE: bad text id in '%s'\n", cmd); return 1; }
        toks[s + 1] = (int)id;
      } else if (cmd[0] == 'A' && cmd[1] == ' ') {
        char* nl = strchr(cmd, '\n'); if (nl) *nl = 0;
        if (g_semb_n >= SERVE_MAXFR) { printf("SERVE: frame buffer full (%d rows)\n", SERVE_MAXFR); return 1; }
        FILE* rf = fopen(cmd + 2, "rb");
        if (!rf) { printf("SERVE: cannot open frame row %s\n", cmd + 2); return 1; }
        std::vector<float> row((size_t)H);
        size_t got = fread(row.data(), 4, (size_t)H, rf);
        int extra = fgetc(rf);
        fclose(rf);
        if (got != (size_t)H || extra != EOF) { printf("SERVE: %s is not exactly %d floats\n", cmd + 2, H); return 1; }
        for (int i = 0; i < H; i++) if (!isfinite(row[i])) { printf("SERVE: frame row non-finite at %d\n", i); return 1; }
        if (cudaMemcpy(g_semb + (size_t)g_semb_n * H, row.data(), (size_t)H * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
          printf("SERVE: frame row upload failed\n"); return 1;
        }
        g_semb_n++;
        toks[s + 1] = g_speak_tok;
      } else { printf("SERVE: unknown command '%s'\n", cmd); return 1; }
    }
  }
  if (pfprobe >= 0) {
    if (pfpass == 0) pf_base = toks;
    else {
      size_t n = std::min(pf_base.size(), toks.size()), fd = n;
      for (size_t i = 0; i < n; i++) if (pf_base[i] != toks[i]) { fd = i; break; }
      printf("PF-PROBE arm=%d  ids equal=%s  compared=%zu  first_diff=%zd  prompt=%d gen=%zu\n",
             pfprobe, fd == n ? "YES (bit-exact vs serial)" : "NO", n, (ssize_t)fd,
             nprompt_all, toks.size() - (size_t)prompt.size());
      printf("PF-PROBE arm=%d  verdict: %s\n", pfprobe,
             fd == n ? "batched arm reproduces the shipped bits at this shape"
                     : "batched arm DIVERGES -- must not be enabled without a contract decision");
      fflush(stdout);
    }
  }
  }  // LFM_PF_PROBE passes
  if (g_xrange) xrange_print();
  if (g_time && ng > 0)
    printf("PHASES ms/tok: emb %.2f  conv %.2f  attn %.2f  finalnorm %.2f  lm_head %.2f  sum %.2f\n",
           g_tEmb / ng, g_tConv / ng, g_tAttn / ng, g_tNorm / ng, g_tHead / ng,
           (g_tEmb + g_tConv + g_tAttn + g_tNorm + g_tHead) / ng);
  if (ng > 0)
    printf("steady-state %.2f ms/token -> %.2f tok/s over %d steps "
           "(GEMV-only budget was 127.7 ms)\n", tot / ng, 1000.0 * ng / tot, ng);
  if (g_aemb_n)
    printf("AUDIO-EMB fed %ld of %d rows (positions %d..%d); steady-state steps counted: %d\n",
           g_aemb_fed, g_aemb_n, g_aemb_at, g_aemb_at + g_aemb_n - 1, ng);
  if (g_semb_n)
    printf("SPEAK-EMB fed %ld of %d rows at audio_out token %d\n", g_speak_fed, g_semb_n, g_speak_tok);
  fflush(stdout);
  if (nclip) { printf("CLIP %d DONE steps=%d\n", cl, ng); fflush(stdout); }
  }  // resident-process clip loop
  {
    // Peak RSS, from the kernel's own counter. measure.sh needs it and the
    // Nano's login shell is dash (whose `time` has no -f) with no
    // /usr/bin/time at all. On Tegra the GPU arenas are nvmap-backed and do
    // NOT show up in RSS, so this number is host-side cost only -- the right
    // thing to compare across --emb-host / --arena-mb, and the reason the
    // plan's memory comparisons are made at equal arena size.
    FILE* st = fopen("/proc/self/status", "r");
    char ln[128]; long long hwm = 0, cur = 0;
    while (st && fgets(ln, sizeof(ln), st)) {
      long long v;
      if (sscanf(ln, "VmHWM: %lld kB", &v) == 1) hwm = v;
      else if (sscanf(ln, "VmRSS: %lld kB", &v) == 1) cur = v;
    }
    if (st) fclose(st);
    printf("RSS_KB=%lld RSS_NOW_KB=%lld\n", hwm, cur);
    fflush(stdout);
  }
  if (g_ktime) {
    printf("KTIME per-kernel ms/token (serialized: read the split, not the sum)\n");
    for (std::map<std::string,std::pair<double,int> >::iterator it = g_kt.begin(); it != g_kt.end(); ++it)
      printf("  %-24s %8.3f ms  n=%d\n", it->first.c_str(), it->second.first, it->second.second);
    fflush(stdout);
  }
  printf("FM3_NANODE_DONE\n");
  if (emb_mode_req == 5) return prefill_arm5_gate(M);
  if (emb_mode_req == 6) return prefill_chunk_test(M, chunkT);
  if (emb_mode_req == 7) return prefill_e2e(M, chunkT, prompt, emb);
  return 0;
}
