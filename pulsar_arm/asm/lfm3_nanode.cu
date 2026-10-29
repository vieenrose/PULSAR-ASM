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
#include <map>
#include <time.h>
#include <zlib.h>
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
__global__ void k_rmsnorm(const bf16* __restrict__ x, const float* __restrict__ w,
                          bf16* __restrict__ y, float eps, int H_) {
  const int tid = threadIdx.x;
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

// per-head RMSNorm over head_dim (q_norm / k_norm): one block per head,
// 32 threads, 4 elements each.
__global__ void k_headnorm(bf16* __restrict__ x, const float* __restrict__ w,
                           float eps, int hd) {
  const int tid = threadIdx.x;
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
  const int h = blockIdx.x, tid = threadIdx.x;
  const int lane = tid & 31, wid = tid >> 5;
  __shared__ float sl[MAXTOK];
  __shared__ float red[8];
  __shared__ float segred[8][HD];
  const int lo = (win > 0 && npos > win) ? npos - win : 0;
  const bf16* qr = q + (size_t)h * hd;
  const int kvh = h / (NH / nkv);
  for (int p = lo + wid; p < npos; p += 8) {
    const bf16* kr = kc + (size_t)p * nkv * hd + kvh * hd;
    float s = 0.f;
    #pragma unroll
    for (int j = 0; j < hd / 32; j++)
      s = __fadd_rn(s, __fmul_rn(bf2f(qr[lane * (hd / 32) + j]), bf2f(kr[lane * (hd / 32) + j])));
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
    acc0 = __fadd_rn(acc0, __fmul_rn(w, bf2f(vr[e])));
    acc1 = __fadd_rn(acc1, __fmul_rn(w, bf2f(vr[e + 32])));
    acc2 = __fadd_rn(acc2, __fmul_rn(w, bf2f(vr[e + 64])));
    acc3 = __fadd_rn(acc3, __fmul_rn(w, bf2f(vr[e + 96])));
  }
  segred[seg][e] = acc0;
  segred[seg][e + 32] = acc1;
  segred[seg][e + 64] = acc2;
  segred[seg][e + 96] = acc3;
  __syncthreads();
  if (tid < HD)
    out[(size_t)h * hd + tid] = f2bf(__fadd_rn(__fadd_rn(__fadd_rn(segred[0][tid], segred[1][tid]),
                                    __fadd_rn(segred[2][tid], segred[3][tid])),
                          __fadd_rn(__fadd_rn(segred[4][tid], segred[5][tid]),
                                     __fadd_rn(segred[6][tid], segred[7][tid]))));
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
__global__ void k_sigmoid_gate(const bf16* __restrict__ g, bf16* __restrict__ attn, int hd) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  const float s = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-bf2f(g[h]))));
  bf16* a = attn + (size_t)h * hd;
  #pragma unroll 4
  for (int j = 0; j < hd; j++) a[j] = f2bf(__fmul_rn(bf2f(a[j]), s));
}
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
  float *xf, *lgf, *part;          // part = GEMV per-chunk partials, f32 by contract
  bf16 *kc, *vc;
  float *invf_full, *invf_slide;
  int* tok_dev;
};

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
    if (fb >= need) {
      printf("device free %zu MB of %zu MB (need %zu MB) -- loading\n",
           fb / 1048576, tb / 1048576, need / 1048576);
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
    struct timespec ts; ts.tv_sec = 5; ts.tv_nsec = 0; nanosleep(&ts, nullptr);
  }
  if (cudaMalloc(&M.dq, M.need_q) != cudaSuccess) {
    printf("codes arena cudaMalloc(%.0f MB) failed\n", M.need_q / 1e6); fclose(f); return false;
  }
  if (cudaMalloc(&M.ds, M.need_s) != cudaSuccess) {
    printf("scales arena cudaMalloc(%.0f MB) failed\n", M.need_s / 1e6); fclose(f); return false;
  }
  if (cudaMalloc(&M.df, M.need_f) != cudaSuccess) {
    printf("f32 arena cudaMalloc(%.1f MB) failed\n", M.need_f / 1e6); fclose(f); return false;
  }
  std::vector<char> buf;
  size_t oc = 0, os = 0, of = 0;
  // f32 payloads first. At the memory gate boundary the LAST cudaMemcpy is
  // the one that starves, and in the first builds that was a norm vector 16 MB
  // into a 1.6 MB arena -- the failure then looked like a blob/CRC problem
  // while it was really an out-of-memory one. Uploading the small tensors
  // first makes starvation happen while there is still room to say so.
  for (int pass = 0; pass < 2; pass++)
  for (size_t i = 0; i < M.ents.size(); i++) {
    Ent& e = M.ents[i];
    if ((e.dtype == 2) != (pass == 0)) continue;
    fseek(f, (long)(base_data + e.file_off), SEEK_SET);
    const size_t tot = e.nc + e.ns;
    buf.resize(tot);
    if (fread(buf.data(), 1, tot, f) != tot) { printf("short read %s\n", e.name.c_str()); fclose(f); return false; }
    posix_fadvise(bfd, (off_t)(base_data + e.file_off), (off_t)tot, POSIX_FADV_DONTNEED);
    // Recomputing the exporter's per-tensor crc32 is what turns "the engine
    // runs the quantizer that was certified 8/8" into a load-time assertion.
    const uint32_t c = (uint32_t)crc32(0, (const Bytef*)buf.data(), (uInt)tot);
    if (c != e.crc) {
      printf("CRC MISMATCH %s: file %08x want %08x\n", e.name.c_str(), c, e.crc);
      fclose(f); return false;
    }
    cudaError_t ce = cudaSuccess;
    // The bf16 (dtype 3) embedding must be DEVICE-resident: lm_head_bf16
    // reads it directly, and the Q8 host-staging path has no bf16 form.
    // Under --emb-host the tied embedding lives in pinned host memory only,
    // for BOTH payload formats: Q8 (lm_head_host) and bf16 (lm_head_bf16_host).
    if (M.emb_mode == 1 && e.name == EMBNAME) continue;   // CRC checked above
    // Mode 2 (selftest) needs ONLY the embedding resident, so the other 285
    // tensors stay on disk: 262 MB + 270 MB pinned + arena fits, the full
    // 1944 MB does not, and the selftest never touches a layer weight.
    if (M.emb_mode == 2 && e.name != EMBNAME) continue;
    if (e.dtype == 3) {
      // raw bf16 payload, verbatim into the codes arena (ns == 0)
      ce = cudaMemcpy(M.dq + oc, buf.data(), e.nc, cudaMemcpyHostToDevice);
      e.dq = M.dq + oc; e.ds = nullptr;
      oc += e.nc;
    }
    else if (e.dtype == 1) {
      ce = cudaMemcpy(M.dq + oc, buf.data(), e.nc, cudaMemcpyHostToDevice);
      if (ce == cudaSuccess)
        ce = cudaMemcpy((char*)M.ds + os, buf.data() + e.nc, e.ns, cudaMemcpyHostToDevice);
      e.dq = M.dq + oc; e.ds = (float*)((char*)M.ds + os);
      oc += e.nc; os += e.ns;
    } else {
      // `M.df + of` is float* arithmetic on a BYTE offset: it advanced 4*of,
      // so the first ~48 norm vectors landed inside the arena by luck (4*of
      // < 1.55 MB) and the 49th faulted. Every "UPLOAD FAIL ... invalid
      // argument" in this session, and the memory misdiagnosis that came
      // with it, traces to this one line -- the f32 arena is the last
      // allocation, so it is where typed-pointer arithmetic first bites.
      ce = cudaMemcpy((char*)M.df + of, buf.data(), e.nc, cudaMemcpyHostToDevice);
      e.df = (float*)((char*)M.df + of);
      of += e.nc;
    }
    if (ce != cudaSuccess) {
      // The first version printed only dst, and every one of these failures
      // was blamed on memory. Print the arena window so an out-of-range
      // offset is obvious: it is the thing the gate boundary hid.
      printf("UPLOAD FAIL %s dtype=%d nd=%d ne=[%d,%d,%d,%d] nc=%zu ns=%zu src=%p\n"
             "  dst=%p  arenas: codes=%p/%zu used=%zu  scales=%p/%zu used=%zu  f32=%p/%zu used=%zu\n"
             "  -> %s\n",
             e.name.c_str(), e.dtype, e.nd, e.ne[0], e.ne[1], e.ne[2], e.ne[3], e.nc, e.ns,
             (void*)buf.data(),
             (void*)(e.dtype == 1 ? (void*)(M.dq + oc) : (void*)((char*)M.df + of)),
             (void*)M.dq, M.need_q, oc, (void*)M.ds, M.need_s, os, (void*)M.df, M.need_f, of,
             cudaGetErrorString(ce));
      fclose(f); return false;
    }
    M.by[e.name] = &M.ents[i];
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

static void gemv(Model& M, const std::string& name, const bf16* xb, bf16* yb) {
  Ent* e = T(M, name);
  const int N = e->ne[0], K = e->ne[1];
  if (g_xrange) xrange_add(name, xb, K);
  GemvCfg c = pick(N, K);
  if (K % c.KCHUNK) { printf("%s: K=%d not divisible by KCHUNK=%d\n", name.c_str(), K, c.KCHUNK); exit(2); }
  const int nb = c.KCHUNK / Q8_BLK, nchunk = K / c.KCHUNK;
  if (nchunk * c.KCHUNK != K) { printf("%s: coverage FAIL\n", name.c_str()); exit(2); }
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
}
// Batched H2 prefill, clean-room (#162 continued). VALIDATED PIECES ONLY:
// - staging layout identical to q8_partial<H2=1> (proven bit-exact by v3a);
// - per-lane register acc (shared acc was a cross-lane race -- the v1 bug);
// - shm sized exactly CT*nb*17 half2 = 8704 B (the v1 ULF was OOB shared use).
// The minimal direct-call probe just proved direct launches + shared + H2 all
// work in this context, so any failure here is in the newly added structure.
template <typename TX, int CT>
__global__ void q8_prefill_h2(const int8_t* __restrict__ q, const float* __restrict__ s,
                           const TX* __restrict__ xall, float* __restrict__ part,
                           int N, int K, int KCHUNK, int R, int nb, int T) {
  extern __shared__ float smf[];
  __half2* xh = reinterpret_cast<__half2*>(smf);
  const int nchunk = K / KCHUNK, rowblk = blockIdx.x / nchunk, chunk = blockIdx.x % nchunk;
  const int koff = chunk * KCHUNK, NT = blockDim.x, tid = threadIdx.x;
  const int lane = tid & 31, wid = tid >> 5, WPB = NT >> 5, GR = R / WPB, base = rowblk * R;
  const __half2 off = __floats2half2_rn(1152.f, 1152.f);
  for (int t0 = 0; t0 < T; t0 += CT) {
    int ct = T - t0; if (ct > CT) ct = CT;
    for (int i = tid; i < ct * nb * 16; i += NT) {
      const int c = i / (nb * 16), rem = i % (nb * 16);
      const int b = rem / 16, k = rem % 16;
      int tt = t0 + c; if (tt >= T) tt = T - 1;
      xh[((size_t)c * nb + (size_t)b) * 17 + (size_t)k] =
        __floats2half2_rn(lfm_stage(xall[((size_t)tt) * K + koff + (size_t)b * 32 + 2 * k]),
                           lfm_stage(xall[((size_t)tt) * K + koff + (size_t)b * 32 + 2 * k + 1]));
    }
    __syncthreads();
    for (int g = 0; g < GR; g++) {
      const int r = wid + g * WPB, row = base + r;
      float acc[CT];
#pragma unroll
      for (int c = 0; c < CT; c++) acc[c] = 0.f;
      if (row < N) {
        const int8_t* qr = q + (size_t)row * K + koff;
        const float* sr = s + (size_t)row * (K / Q8_BLK) + (size_t)chunk * nb;
        for (int b = lane; b < nb; b += 32) {
          const int4 a0 = *reinterpret_cast<const int4*>(qr + (size_t)b * Q8_BLK);
          const int4 a1 = *reinterpret_cast<const int4*>(qr + (size_t)b * Q8_BLK + 16);
          const uint32_t w[8] = {(uint32_t)a0.x, (uint32_t)a0.y, (uint32_t)a0.z, (uint32_t)a0.w,
                                 (uint32_t)a1.x, (uint32_t)a1.y, (uint32_t)a1.z, (uint32_t)a1.w};
#pragma unroll
          for (int c = 0; c < CT; c++) {
            const __half2* xb2 = xh + ((size_t)c * nb + (size_t)b) * 17;
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
            acc[c] = __fmaf_rn(sr[b], f.x + f.y, acc[c]);
          }
        }
      }
#pragma unroll
      for (int c = 0; c < CT; c++) {
        float v = (row < N) ? acc[c] : 0.f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(~0u, v, o);
        if (lane == 0 && row < N && t0 + c < T)
          part[(((size_t)(t0 + c)) * N + row) * nchunk + chunk] = v;
      }
    }
    __syncthreads();
  }
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
static int prefill_h2batched(Model& M) {
  Ent* e = nullptr;
  for (size_t i = 0; i < M.ents.size(); i++)
    if (M.ents[i].dq && M.ents[i].ds && M.ents[i].ne[0] >= 1024 && M.ents[i].ne[1] % 32 == 0) { e = &M.ents[i]; break; }
  if (!e) return 1;
  const int N = e->ne[0], K = e->ne[1], T = 23, R = 256, TPB = 256, CT = 2;
  const int KC = (K >= 2048 && K % 2048 == 0) ? 2048 : K, nb = KC / Q8_BLK, nch = K / KC;
  const int nblk = ((N + R - 1) / R) * nch;
  const size_t shm = (size_t)CT * nb * 17 * sizeof(__half2);
  printf("prefill-h2batched %s N=%d K=%d T=%d shm=%zu\n", e->name.c_str(), N, K, T, shm); fflush(stdout);
  uint16_t* xd = nullptr; float *seq = nullptr, *bat = nullptr;
  printf("a=%s\n", cudaGetErrorString(cudaMalloc(&xd, (size_t)T * K * 2))); fflush(stdout);
  printf("b=%s\n", cudaGetErrorString(cudaMalloc(&seq, (size_t)T * N * 4))); fflush(stdout);
  printf("c=%s\n", cudaGetErrorString(cudaMalloc(&bat, (size_t)T * N * 4))); fflush(stdout);
  std::vector<uint16_t> pat((size_t)T * K);
  for (size_t i = 0; i < pat.size(); i++) pat[i] = (uint16_t)(0x3c00 + ((i * 37) & 0x3ff));
  printf("h2d=%s\n", cudaGetErrorString(cudaMemcpy(xd, pat.data(), pat.size() * 2, cudaMemcpyHostToDevice))); fflush(stdout);
  cudaEvent_t ea, eb; cudaEventCreate(&ea); cudaEventCreate(&eb); float ms = 0;
  cudaEventRecord(ea);
  for (int t = 0; t < T; t++)
    q8_partial<bf16,true,false,1,1><<<nblk, TPB, (size_t)nb * 33 * 4>>>(e->dq, e->ds, (const bf16*)(xd + (size_t)t * K),
      seq + (size_t)t * N, N, K, KC, R, nb);
  cudaEventRecord(eb); cudaEventSynchronize(eb);
  cudaEventElapsedTime(&ms, ea, eb);
  printf("seq23 err=%s\n", cudaGetErrorString(cudaGetLastError())); fflush(stdout);
  std::vector<float> a((size_t)T * N);
  printf("cp-seq=%s\n", cudaGetErrorString(cudaMemcpy(a.data(), seq, a.size() * 4, cudaMemcpyDeviceToHost))); fflush(stdout);
  printf("seq[0]=%.9g (expect H2 ~-0.10976)\n", a[0]); fflush(stdout);
  cudaEventRecord(ea);
  q8_prefill_h2<bf16,CT><<<nblk, TPB, shm>>>(e->dq, e->ds, (const bf16*)xd, bat, N, K, KC, R, nb, T);
  printf("bat-launch=%s\n", cudaGetErrorString(cudaGetLastError())); fflush(stdout);
  cudaEventRecord(eb); cudaEventSynchronize(eb);
  printf("bat-sync=%s\n", cudaGetErrorString(cudaGetLastError())); fflush(stdout);
  float ms2 = 0; cudaEventElapsedTime(&ms2, ea, eb);
  std::vector<float> b(a.size(), -999.f);
  printf("cp-bat=%s\n", cudaGetErrorString(cudaMemcpy(b.data(), bat, b.size() * 4, cudaMemcpyDeviceToHost))); fflush(stdout);
  printf("bat[0]=%.9g\n", b[0]); fflush(stdout);
  long bad = 0; for (size_t i = 0; i < a.size() && bad < 5; i++) if (memcmp(&a[i], &b[i], 4)) { bad++; if (bad == 1) printf("first diff at %zu: seq=%.9g bat=%.9g\n", i, a[i], b[i]); }
  long nbad = 0; for (size_t i = 0; i < a.size(); i++) nbad += memcmp(&a[i], &b[i], 4) != 0;
  printf("prefill-h2batched: %s (%ld/%zu differ) seq %.2f ms bat %.2f ms\n", nbad ? "MISMATCH" : "BIT-EXACT", nbad, a.size(), ms, ms2);
  fflush(stdout);
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

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage: %s <blob.lfm3q8> [--steps N] [--prompt id,id,...] [--trace]\n", argv[0]);
    return 1;
  }
  int steps = 8; bool trace = false;
  int lm_rows_override = 0, emb_mode_req = 0, probe_layer = 0, probe_step = 0;
  const char* dump_logits = nullptr;
  const char* dump_logits_all = nullptr;   // every decode step, appended (gate: top-k vs HF)
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
    else if (!strcmp(argv[i], "--merge")) g_merge = true;   // merged gate/up GEMV
    else if (!strcmp(argv[i], "--lmrows") && i + 1 < argc) lm_rows_override = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--dump-logits") && i + 1 < argc) dump_logits = argv[++i];
    else if (!strcmp(argv[i], "--dump-logits-all") && i + 1 < argc) dump_logits_all = argv[++i];
    else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) {
      char* s = argv[++i];
      while (*s) { prompt.push_back((int)strtol(s, &s, 10)); if (*s == ',') s++; }
    }
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
  if (!load_blob(argv[1], M, js)) return 1;

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
  for (const Al& a : als) {
    if (cudaMalloc(a.p, a.n) != cudaSuccess) {
      printf("arena cudaMalloc(%s, %zu B) FAILED -- device free %.2f GB\n",
             a.w, a.n, freeb0 / 1e9);
      return 1;
    }
  }
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
  if (emb->ne[0] != VOCAB || emb->ne[1] != H) { printf("embed shape %dx%d\n", emb->ne[0], emb->ne[1]); return 1; }
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

  if (prompt.empty()) prompt.push_back(124894);                 // bos
  std::vector<int> toks = prompt;
  cudaEvent_t evA, evB; cudaEventCreate(&evA); cudaEventCreate(&evB);
  if (g_time) { cudaEventCreate(&g_evA); cudaEventCreate(&g_evB); }

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
  const bool warm = getenv("LFM_NOWARM") == nullptr && !g_graphs;
  for (int s = warm ? -1 : 0; s < steps; s++) {
    const int pos = s < 0 ? 0 : s;
    const int tok = s < 0 ? 124894 : toks[s];
    const bf16* lm_head_input = M.x;   // replaced by the embedding_norm output below
    if (pos >= MAXTOK) break;
    if (g_graphs && M.gx) g_launch = M.stRun;   // embedding, layers, lm_head all on stRun
    cudaEventRecord(evA, g_launch);
    phBegin();
    if (emb->dtype == 3) {
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
      if (g_time && !g_graphs) { if (phase) phEnd(*phase); phBegin(); phase = is_attn[L] ? &g_tAttn : &g_tConv; }
      KT("rms operator_norm", rms1(M, M.x, NW(M, L, "operator_norm.weight"), M.h1));
      g_probe = trace && L == g_probe_layer && s == g_probe_step;
      PROBE("operator_norm", M.h1, H);
      if (is_attn[L]) {
        gemv(M, LN(L, "self_attn.q_proj.weight"), M.h1, M.q);
        gemv(M, LN(L, "self_attn.k_proj.weight"), M.h1, M.k);
        gemv(M, LN(L, "self_attn.v_proj.weight"), M.h1, M.v);
        gemv(M, LN(L, "self_attn.gate_proj.weight"), M.h1, M.gate);
        k_headnorm<<<NH, 32, 0, g_launch>>>(M.q, NW(M, L, "self_attn.q_norm.weight"), NORM_EPS, HD);
        k_headnorm<<<NKV, 32, 0, g_launch>>>(M.k, NW(M, L, "self_attn.k_norm.weight"), NORM_EPS, HD);
        PROBE("q(postnorm)", M.q, NH * HD);
        PROBE("k(postnorm)", M.k, NKV * HD);
        PROBE("v", M.v, NKV * HD);
        PROBE("gate", M.gate, NH);
        const float* invf = is_slide[L] ? M.invf_slide : M.invf_full;
        const int* posp = g_graphs ? M.dpos : nullptr;
        // rot is per-layer-type: full_attention rotates half the dims
        // (partial_rotary_factor 0.5), sliding_attention rotates all of them
        // (partial 1.0). The old code passed HD/2 for both, leaving sliding
        // layers' slow half (dims 32..63) unrotated -- invisible at pos 0
        // (RoPE is the identity there) and under the 3% gate at pos <= 22,
        // but a position-growing phase error after that (#151: 243/7200 long
        // cells outside 3%, engine-low, first sliding layer L4 origin).
        const int rot = is_slide[L] ? HD : HD / 2;
        k_rope<<<NH, 64, 0, g_launch>>>(M.q, invf, pos, HD, rot, posp);
        k_rope<<<NKV, 64, 0, g_launch>>>(M.k, invf, pos, HD, rot, posp);
        PROBE("q(postrope)", M.q, NH * HD);
        bf16* kbase = M.kc + (size_t)L * MAXTOK * NKV * HD;
        bf16* vbase = M.vc + (size_t)L * MAXTOK * NKV * HD;
        if (g_graphs) {
          // The KV write must be a kernel, not a D2D memcpy: a memcpy's
          // destination is baked into the capture, and pos*NKV*HD is exactly
          // what changes per step. Same bytes, same order.
          k_kv_write<<<(NKV * HD + 255) / 256, 256, 0, g_launch>>>(kbase, M.k, NKV * HD, M.dpos, NKV * HD);
          k_kv_write<<<(NKV * HD + 255) / 256, 256, 0, g_launch>>>(vbase, M.v, NKV * HD, M.dpos, NKV * HD);
        } else {
          cudaMemcpy(kbase + (size_t)pos * NKV * HD, M.k, (size_t)NKV * HD * 2, cudaMemcpyDeviceToDevice);
          cudaMemcpy(vbase + (size_t)pos * NKV * HD, M.v, (size_t)NKV * HD * 2, cudaMemcpyDeviceToDevice);
        }
        const float scaling = 1.0f / sqrtf((float)HD);
        // sliding_attention layers see only the last 256 keys (config
        // /lfm/sliding_window); full_attention layers see all of them.
        const int win = is_slide[L] ? SLIDE_WIN : 0;
        k_attn_head<<<NH, 256, 0, g_launch>>>(M.q, kbase, vbase, M.attn, pos + 1, win, scaling, HD, NKV, g_graphs ? M.dpos + 1 : nullptr);
        PROBE("attn", M.attn, NH * HD);
        k_sigmoid_gate<<<1, NH, 0, g_launch>>>(M.gate, M.attn, HD);
        PROBE("attn*gate", M.attn, NH * HD);
        gemv(M, LN(L, "self_attn.o_proj.weight"), M.attn, M.h2);
        PROBE("o_proj", M.h2, H);
      } else {
        bf16* ring = M.ring + (size_t)ci * 7 * H;
        KT("gemv conv.in_proj", gemv(M, LN(L, "conv.in_proj.weight"), M.h1, M.bx));
        PROBE("in_proj", M.bx, 3 * H);
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
        KT("emul B*xg", emul(M, M.bx, M.bx + 2 * H, M.h3, H));
        PROBE("B*x", M.h3, H);
        KT("k_shift_in", k_shift_in<<<(H + 63) / 64, 64, 0, g_launch>>>(ring, M.h3, H));
        if (g_probe) for (int k = 0; k < 7; k++)
          printf("    ring slot %d     rms %9.5f\n", k, rms_host_n(ring + (size_t)k * H, H));
        KT("k_conv7", k_conv7<<<(H + 63) / 64, 64, 0, g_launch>>>(ring, T(M, LN(L, "conv.conv.weight"))->df, M.h4, H));
        PROBE("conv", M.h4, H);
        KT("emul C*y", emul(M, M.bx + H, M.h4, M.h2, H));
        PROBE("C*y", M.h2, H);
        KT("gemv conv.out_proj", gemv(M, LN(L, "conv.out_proj.weight"), M.h2, M.h2));
        PROBE("out_proj", M.h2, H);
        ci++;
      }
      KT("rms post_op_norm", rms1(M, M.h2, NW(M, L, "post_operator_norm.weight"), M.h3));
      PROBE("post_op_norm", M.h3, H);
      eadd(M, M.x, M.h3, M.x, H);
      PROBE("add1", M.x, H);
      KT("rms ffn_norm", rms1(M, M.x, NW(M, L, "ffn_norm.weight"), M.h1));
      PROBE("ffn_norm", M.h1, H);
      char mn[96]; snprintf(mn, sizeof mn, "MERGE:%d:gate_up", L);
      if (g_merge && T(M, mn)) {
        // One GEMV over 2*INTER rows; rows [0,INTER) are w1 (gate), rows
        // [INTER,2*INTER) are w3 (up). k_silu_mul_bf takes the two halves as
        // its two input pointers, so no kernel change is needed.
        gemv(M, mn, M.h1, M.hgu);
        PROBE("w1", M.hgu, INTER);
        PROBE("w3", M.hgu + INTER, INTER);
        k_silu_mul_bf<<<(INTER + 255) / 256, 256, 0, g_launch>>>(M.hgu, M.hgu + INTER, M.h2, INTER);
      } else {
      KT("gemv ffn.w1", gemv(M, LN(L, "feed_forward.w1.weight"), M.h1, M.h2));
      PROBE("w1", M.h2, INTER);
      KT("gemv ffn.w3", gemv(M, LN(L, "feed_forward.w3.weight"), M.h1, M.h3));
      PROBE("w3", M.h3, INTER);
      k_silu_mul_bf<<<(INTER + 255) / 256, 256, 0, g_launch>>>(M.h2, M.h3, M.h2, INTER);
      }
      PROBE("silu_mul", M.h2, INTER);
      KT("gemv ffn.w2", gemv(M, LN(L, "feed_forward.w2.weight"), M.h2, M.h2));
      PROBE("w2", M.h2, INTER);
      KT("rms post_ffn_norm", rms1(M, M.h2, NW(M, L, "post_ffn_norm.weight"), M.h3));
      PROBE("post_ffn_norm", M.h3, H);
      eadd(M, M.x, M.h3, M.x, H);
      PROBE("add2", M.x, H);
      g_probe = 0;
      // Full layer x position matrix in one run: the perlayer_probe role the
      // Pi engine plays with tools/perlayer_probe.patch. Cheap enough to leave
      // on under --trace (a 4 KB D2H copy per layer per step).
      if (trace) printf("POS %2d L%-2d %s%s rms %.5f\n", s, L,
                        is_attn[L] ? "attn" : "conv", is_slide[L] ? "(slide)" : "",
                        rms_host(M, M.x));
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
    else if (eemb && eemb->dtype == 3 && M.emb_mode == 1 && M.hemb.hc)
      lm_head_bf16_host(M, M.hemb, lm_head_input, M.lgf);  // certified arm, host-staged
    else if (eemb && eemb->dtype == 3)
      lm_head_bf16(M, eemb, lm_head_input, M.lgf);         // certified arm, resident
    else if (M.emb_mode == 1)
      lm_head_host(M, M.hemb, lm_head_input, M.lgf);     // Q8 streamed from pinned host
    else gemv_f(M, EMBNAME, lm_head_input, M.lgf);       // tied lm_head, resident
    if (need_head) k_argmax<<<1, 256, 0, g_launch>>>(M.lgf, M.tok_dev, VOCAB);
    cudaEventRecord(evB, g_launch);
    cudaEventSynchronize(evB);
    if (s < 0) {
      float wms = 0; cudaEventElapsedTime(&wms, evA, evB);
      cudaMemset(M.ring, 0, (size_t)NCONV * 7 * H * 2);
      cudaMemset(M.kc, 0, kvbytes); cudaMemset(M.vc, 0, kvbytes);
      cudaDeviceSynchronize();
      g_tEmb = g_tConv = g_tAttn = g_tNorm = g_tHead = 0;
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
  }
  if (g_xrange) xrange_print();
  if (g_time && ng > 0)
    printf("PHASES ms/tok: emb %.2f  conv %.2f  attn %.2f  finalnorm %.2f  lm_head %.2f  sum %.2f\n",
           g_tEmb / ng, g_tConv / ng, g_tAttn / ng, g_tNorm / ng, g_tHead / ng,
           (g_tEmb + g_tConv + g_tAttn + g_tNorm + g_tHead) / ng);
  if (ng > 0)
    printf("steady-state %.2f ms/token -> %.2f tok/s over %d steps "
           "(GEMV-only budget was 127.7 ms)\n", tot / ng, 1000.0 * ng / tot, ng);
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
  return 0;
}
