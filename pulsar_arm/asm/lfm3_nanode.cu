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
// 512 positions, not 4096. The KV cache is 2 x NLAYER x MAXTOK x NKV x HD x
// 2 bytes: at MAXTOK=4096 that is 1.0 GB on top of 1.95 GB of weights, which
// on a 4 GB unified-memory Nano leaves nothing and cudaMemcpy starts
// returning "invalid argument" for the allocations that silently failed.
// 512 also shrinks k_attn_head's shared logits array to 2 KB, which is what
// lets the attention block keep any occupancy at all.
// Sized to the benchmark, not to the model: MAXTOK sets the KV cache at
// 2 x NLAYER x MAXTOK x NKV x HD x 2 bytes = 25 MB at 256, 6 MB at 64. On a
// board that is 30 MB short of fitting 1.95 GB of weights, that difference
// is the difference between running and not. A 256-token context (the
// sliding window is 256) needs MAXTOK=256 and 25 MB more device memory --
// raise it together with the memory headroom, not on its own.
#define MAXTOK 64
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

// Q8_0 pass 1: R rows x one chunk, one WARP per row, lanes striding the
// chunk's blocks, x staged in shared. This is the kernel whose 14-16.5 GB/s
// rate underlies the 127.7 ms/token budget (q8_lc3.cu, commit 2b5cbe5).
template <typename TX>
__global__ void q8_partial(const int8_t* __restrict__ q, const float* __restrict__ s,
                           const TX* __restrict__ x, float* __restrict__ part,
                           int N, int K, int KCHUNK, int R, int nb) {
  extern __shared__ float xs[];
  const int nchunk = K / KCHUNK;
  const int rowblk = blockIdx.x / nchunk;
  const int chunk = blockIdx.x % nchunk;
  const int koff = chunk * KCHUNK;
  const int NT = blockDim.x, tid = threadIdx.x;
  const TX* xg = x + koff;
  for (int i = tid; i < KCHUNK; i += NT) {
    const int b = i >> 5, j = i & 31;
    xs[b * 33 + j] = lfm_stage(xg[i]);
  }
  __syncthreads();
  __shared__ float sh[RMAX];
  const int base = rowblk * R;
  const int lane = tid & 31, wid = tid >> 5;
  const int WPB = NT >> 5, GR = R / WPB;
  #pragma unroll 1
  for (int g = 0; g < GR; g++) {
    const int r = wid + g * WPB;
    const int row = base + r;
    float acc = 0.f;
    if (row < N) {
      const int8_t* qr = q + (size_t)row * K + koff;
      const float* sr = s + (size_t)row * (K / Q8_BLK) + (size_t)chunk * nb;
      #pragma unroll 1
      for (int b = lane; b < nb; b += 32) {
        const int4 a0 = *reinterpret_cast<const int4*>(qr + (size_t)b * Q8_BLK);
        const int4 a1 = *reinterpret_cast<const int4*>(qr + (size_t)b * Q8_BLK + 16);
        const int8_t* c0 = reinterpret_cast<const int8_t*>(&a0);
        const int8_t* c1 = reinterpret_cast<const int8_t*>(&a1);
        const float* xb = xs + (size_t)b * 33;
        float dot = 0.f;
        #pragma unroll
        for (int j = 0; j < 16; j++) dot = __fadd_rn(dot, __fmul_rn((float)c0[j], xb[j]));
        #pragma unroll
        for (int j = 0; j < 16; j++) dot = __fadd_rn(dot, __fmul_rn((float)c1[j], xb[16 + j]));
        acc = __fadd_rn(acc, __fmul_rn(sr[b], dot));
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
                       int pos, int hd, int rot) {
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
                            int npos, int win, float scaling, int hd, int nkv) {
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
static void gemv(Model& M, const std::string& name, const bf16* xb, bf16* yb) {
  Ent* e = T(M, name);
  const int N = e->ne[0], K = e->ne[1];
  GemvCfg c = pick(N, K);
  if (K % c.KCHUNK) { printf("%s: K=%d not divisible by KCHUNK=%d\n", name.c_str(), K, c.KCHUNK); exit(2); }
  const int nb = c.KCHUNK / Q8_BLK, nchunk = K / c.KCHUNK;
  if (nchunk * c.KCHUNK != K) { printf("%s: coverage FAIL\n", name.c_str()); exit(2); }
  const int nblk = ((N + c.R - 1) / c.R) * nchunk;
  const size_t shm = (size_t)nb * 33 * 4;
  if (g_nofuse) {
    k_bf2f<<<(K + 255) / 256, 256>>>(xb, M.xf, K);
    q8_partial<float><<<nblk, c.TPB, shm>>>(e->dq, e->ds, M.xf, M.part, N, K, c.KCHUNK, c.R, nb);
  } else {
    q8_partial<bf16><<<nblk, c.TPB, shm>>>(e->dq, e->ds, xb, M.part, N, K, c.KCHUNK, c.R, nb);
  }
  k_combine_bf<<<(N + 255) / 256, 256>>>(M.part, yb, N, nchunk);
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
  k_bf2f<<<(K + 255) / 256, 256>>>(xb, M.xf, K);   // M.xf is the shared x-float buffer
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
  k_gemv_bf16<<<(N + R - 1) / R, 256, K * sizeof(float)>>>(
      (const bf16*)emb->dq, M.xf, yf, N, K, R);
}

// Certified arm, host-staged: stream rows/slab of the pinned bf16 embedding
// into a device slab and run k_gemv_bf16 on it. Per-row reduction order is
// identical to the resident bf16 path (lanes stride k by 32 into one fp32,
// then the 16..1 butterfly), so staged and resident agree bit for bit.
static void lm_head_bf16_host(Model& M, HostEmb& E, const bf16* xb, float* yf) {
  const int N = E.N, K = E.K;
  k_bf2f<<<(K + 255) / 256, 256>>>(xb, M.xf, K);
  const int R = 32;
  for (int r0 = 0; r0 < N; r0 += E.rows) {
    const int nr = (N - r0 < E.rows) ? (N - r0) : E.rows;
    cudaMemcpyAsync(E.dc, (const char*)E.hc + (size_t)r0 * K * 2, (size_t)nr * K * 2,
                    cudaMemcpyHostToDevice, 0);
    k_gemv_bf16<<<(nr + R - 1) / R, 256, K * sizeof(float)>>>(
        (const bf16*)E.dc, M.xf, yf + r0, nr, K, R);
  }
}

static void lm_head_host(Model& M, HostEmb& E, const bf16* xb, float* yf) {
  const int N = E.N, K = E.K;
  GemvCfg c = pick(N, K);                       // SAME config as the resident
  const int nb = c.KCHUNK / Q8_BLK, nchunk = K / c.KCHUNK;   // path -> same bits
  k_bf2f<<<(K + 255) / 256, 256>>>(xb, M.xf, K);
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
      q8_partial<float><<<nb2, c.TPB, shm, 0>>>(E.dc2[b], E.dsl2[b], M.xf, M.part, nr, K, c.KCHUNK, c.R, nb);
      k_combine_f<<<(nr + 255) / 256, 256, 0>>>(M.part, yf + r0, nr, nchunk);
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
    q8_partial<float><<<nblk, c.TPB, shm>>>(E.dc, E.dsl, M.xf, M.part, nr, K, c.KCHUNK, c.R, nb);
    k_combine_f<<<(nr + 255) / 256, 256>>>(M.part, yf + r0, nr, nchunk);
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
    k_bf2f<<<(K + 255) / 256, 256>>>(xb, M.xf, K);
    q8_partial<float><<<nblk, c.TPB, shm>>>(e->dq, e->ds, M.xf, M.part, N, K, c.KCHUNK, c.R, nb);
  } else {
    q8_partial<bf16><<<nblk, c.TPB, shm>>>(e->dq, e->ds, xb, M.part, N, K, c.KCHUNK, c.R, nb);
  }
  k_combine_f<<<(N + 255) / 256, 256>>>(M.part, yf, N, nchunk);
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
  k_rmsnorm<<<1, 256>>>(x, w, y, NORM_EPS, H);
}
static void emul(Model& M, const bf16* a, const bf16* b, bf16* y, int n) {
  k_mul_bf<<<(n + 255) / 256, 256>>>(a, b, y, n);
}
static void eadd(Model& M, const bf16* a, const bf16* b, bf16* y, int n) {
  k_add_bf<<<(n + 255) / 256, 256>>>(a, b, y, n);
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
  std::vector<int> prompt;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--trace")) trace = true;
    else if (!strcmp(argv[i], "--probe-layer") && i + 1 < argc) probe_layer = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--probe-step") && i + 1 < argc) probe_step = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--emb-host")) emb_mode_req = 1;
    else if (!strcmp(argv[i], "--lm-selftest")) emb_mode_req = 2;
    else if (!strcmp(argv[i], "--merge")) g_merge = true;   // merged gate/up GEMV
    else if (!strcmp(argv[i], "--lmrows") && i + 1 < argc) lm_rows_override = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--dump-logits") && i + 1 < argc) dump_logits = argv[++i];
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
  printf("weights+arena: %.2f GB, device free now %.2f GB of %.2f GB\n",
         (M.need_q + M.need_s + M.need_f + kvbytes * 2) / 1e9, freeb / 1e9, totb / 1e9);
  printf("L FM3_NANODE_LOADED\n"); fflush(stdout);

  Ent* emb = nullptr;
  for (size_t i = 0; i < M.ents.size(); i++) if (M.ents[i].name == EMBNAME) emb = &M.ents[i];
  if (!emb) { printf("no %s in the blob\n", EMBNAME); return 1; }
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

  printf("step tok        ms   tok/s\n");
  double tot = 0; int ng = 0;
  for (int s = 0; s < steps; s++) {
    const int pos = s;
    const int tok = toks[s];
    const bf16* lm_head_input = M.x;   // replaced by the embedding_norm output below
    if (pos >= MAXTOK) break;
    cudaEventRecord(evA);
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
      k_dequant_row<<<(H + 255) / 256, 256>>>(M.hemb.dc, M.hemb.dsl, M.x, H);
    } else {
      k_dequant_row<<<(H + 255) / 256, 256>>>(emb->dq + (size_t)tok * H,
                                             emb->ds + (size_t)tok * (H / Q8_BLK), M.x, H);
    }

    int ci = 0;
    double* phase = nullptr;
    for (int L = 0; L < NLAYER; L++) {
      if (g_time) { if (phase) phEnd(*phase); phBegin(); phase = is_attn[L] ? &g_tAttn : &g_tConv; }
      rms1(M, M.x, NW(M, L, "operator_norm.weight"), M.h1);
      g_probe = trace && L == g_probe_layer && s == g_probe_step;
      PROBE("operator_norm", M.h1, H);
      if (is_attn[L]) {
        gemv(M, LN(L, "self_attn.q_proj.weight"), M.h1, M.q);
        gemv(M, LN(L, "self_attn.k_proj.weight"), M.h1, M.k);
        gemv(M, LN(L, "self_attn.v_proj.weight"), M.h1, M.v);
        gemv(M, LN(L, "self_attn.gate_proj.weight"), M.h1, M.gate);
        k_headnorm<<<NH, 32>>>(M.q, NW(M, L, "self_attn.q_norm.weight"), NORM_EPS, HD);
        k_headnorm<<<NKV, 32>>>(M.k, NW(M, L, "self_attn.k_norm.weight"), NORM_EPS, HD);
        PROBE("q(postnorm)", M.q, NH * HD);
        PROBE("k(postnorm)", M.k, NKV * HD);
        PROBE("v", M.v, NKV * HD);
        PROBE("gate", M.gate, NH);
        const float* invf = is_slide[L] ? M.invf_slide : M.invf_full;
        k_rope<<<NH, 64>>>(M.q, invf, pos, HD, HD / 2);
        k_rope<<<NKV, 64>>>(M.k, invf, pos, HD, HD / 2);
        PROBE("q(postrope)", M.q, NH * HD);
        bf16* kbase = M.kc + (size_t)L * MAXTOK * NKV * HD;
        bf16* vbase = M.vc + (size_t)L * MAXTOK * NKV * HD;
        cudaMemcpy(kbase + (size_t)pos * NKV * HD, M.k, (size_t)NKV * HD * 2, cudaMemcpyDeviceToDevice);
        cudaMemcpy(vbase + (size_t)pos * NKV * HD, M.v, (size_t)NKV * HD * 2, cudaMemcpyDeviceToDevice);
        const float scaling = 1.0f / sqrtf((float)HD);
        // sliding_attention layers see only the last 256 keys (config
        // /lfm/sliding_window); full_attention layers see all of them.
        const int win = is_slide[L] ? SLIDE_WIN : 0;
        k_attn_head<<<NH, 256>>>(M.q, kbase, vbase, M.attn, pos + 1, win, scaling, HD, NKV);
        PROBE("attn", M.attn, NH * HD);
        k_sigmoid_gate<<<1, NH>>>(M.gate, M.attn, HD);
        PROBE("attn*gate", M.attn, NH * HD);
        gemv(M, LN(L, "self_attn.o_proj.weight"), M.attn, M.h2);
        PROBE("o_proj", M.h2, H);
      } else {
        bf16* ring = M.ring + (size_t)ci * 7 * H;
        gemv(M, LN(L, "conv.in_proj.weight"), M.h1, M.bx);      // [B|C|xg], 3H rows
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
        emul(M, M.bx, M.bx + 2 * H, M.h3, H);                   // B * xg
        PROBE("B*x", M.h3, H);
        k_shift_in<<<(H + 63) / 64, 64>>>(ring, M.h3, H);
        if (g_probe) for (int k = 0; k < 7; k++)
          printf("    ring slot %d     rms %9.5f\n", k, rms_host_n(ring + (size_t)k * H, H));
        k_conv7<<<(H + 63) / 64, 64>>>(ring, T(M, LN(L, "conv.conv.weight"))->df, M.h4, H);
        PROBE("conv", M.h4, H);
        emul(M, M.bx + H, M.h4, M.h2, H);                       // C * y
        PROBE("C*y", M.h2, H);
        gemv(M, LN(L, "conv.out_proj.weight"), M.h2, M.h2);
        PROBE("out_proj", M.h2, H);
        ci++;
      }
      rms1(M, M.h2, NW(M, L, "post_operator_norm.weight"), M.h3);
      PROBE("post_op_norm", M.h3, H);
      eadd(M, M.x, M.h3, M.x, H);
      PROBE("add1", M.x, H);
      rms1(M, M.x, NW(M, L, "ffn_norm.weight"), M.h1);
      PROBE("ffn_norm", M.h1, H);
      char mn[96]; snprintf(mn, sizeof mn, "MERGE:%d:gate_up", L);
      if (g_merge && T(M, mn)) {
        // One GEMV over 2*INTER rows; rows [0,INTER) are w1 (gate), rows
        // [INTER,2*INTER) are w3 (up). k_silu_mul_bf takes the two halves as
        // its two input pointers, so no kernel change is needed.
        gemv(M, mn, M.h1, M.hgu);
        PROBE("w1", M.hgu, INTER);
        PROBE("w3", M.hgu + INTER, INTER);
        k_silu_mul_bf<<<(INTER + 255) / 256, 256>>>(M.hgu, M.hgu + INTER, M.h2, INTER);
      } else {
      gemv(M, LN(L, "feed_forward.w1.weight"), M.h1, M.h2);
      PROBE("w1", M.h2, INTER);
      gemv(M, LN(L, "feed_forward.w3.weight"), M.h1, M.h3);
      PROBE("w3", M.h3, INTER);
      k_silu_mul_bf<<<(INTER + 255) / 256, 256>>>(M.h2, M.h3, M.h2, INTER);
      }
      PROBE("silu_mul", M.h2, INTER);
      gemv(M, LN(L, "feed_forward.w2.weight"), M.h2, M.h2);
      PROBE("w2", M.h2, INTER);
      rms1(M, M.h2, NW(M, L, "post_ffn_norm.weight"), M.h3);
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
    Ent* eemb = emb;
    if (eemb && eemb->dtype == 3 && M.emb_mode == 1 && M.hemb.hc)
      lm_head_bf16_host(M, M.hemb, lm_head_input, M.lgf);  // certified arm, host-staged
    else if (eemb && eemb->dtype == 3)
      lm_head_bf16(M, eemb, lm_head_input, M.lgf);         // certified arm, resident
    else if (M.emb_mode == 1)
      lm_head_host(M, M.hemb, lm_head_input, M.lgf);     // Q8 streamed from pinned host
    else gemv_f(M, EMBNAME, lm_head_input, M.lgf);       // tied lm_head, resident
    k_argmax<<<1, 256>>>(M.lgf, M.tok_dev, VOCAB);
    cudaEventRecord(evB);
    cudaEventSynchronize(evB);
    float ms = 0; cudaEventElapsedTime(&ms, evA, evB);
    int nt = -1;
    cudaMemcpy(&nt, M.tok_dev, 4, cudaMemcpyDeviceToHost);
    toks.push_back(nt);
    phEnd(g_tHead);
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
  printf("FM3_NANODE_DONE\n");
  return 0;
}
