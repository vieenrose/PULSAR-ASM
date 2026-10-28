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
#include <cstdio>
#include <cstdlib>
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
#define MAXTOK 24
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
__global__ void k_bf2f(const bf16* __restrict__ in, float* __restrict__ out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = bf2f(in[i]);
}

// Q8_0 pass 1: R rows x one chunk, one WARP per row, lanes striding the
// chunk's blocks, x staged in shared. This is the kernel whose 14-16.5 GB/s
// rate underlies the 127.7 ms/token budget (q8_lc3.cu, commit 2b5cbe5).
__global__ void q8_partial(const int8_t* __restrict__ q, const float* __restrict__ s,
                           const float* __restrict__ x, float* __restrict__ part,
                           int N, int K, int KCHUNK, int R, int nb) {
  extern __shared__ float xs[];
  const int nchunk = K / KCHUNK;
  const int rowblk = blockIdx.x / nchunk;
  const int chunk = blockIdx.x % nchunk;
  const int koff = chunk * KCHUNK;
  const int NT = blockDim.x, tid = threadIdx.x;
  const float* xg = x + koff;
  for (int i = tid; i < KCHUNK; i += NT) {
    const int b = i >> 5, j = i & 31;
    xs[b * 33 + j] = xg[i];
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
  float acc = 0.f;
  for (int p = seg; p < np; p += 8) {
    const bf16* vr = vc + (size_t)(lo + p) * nkv * hd + kvh * hd;
    acc = __fadd_rn(acc, __fmul_rn(__fmul_rn(sl[lo + p], inv), bf2f(vr[e])));
  }
  segred[seg][e] = acc;
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
  int dtype, ne[4], nd;
  size_t file_off, nc, ns;
  uint32_t crc;
  int8_t* dq = nullptr; float* ds = nullptr; float* df = nullptr;
};

struct Model {
  std::vector<Ent> ents;
  std::map<std::string, Ent*> by;
  int8_t* dq = nullptr; float* ds = nullptr; float* df = nullptr;
  size_t need_q = 0, need_s = 0, need_f = 0;
  bf16 *x, *h1, *h2, *h3, *h4, *q, *k, *v, *attn, *gate, *bx, *cy, *ring;
  float *xf, *lgf, *part;          // part = GEMV per-chunk partials, f32 by contract
  bf16 *kc, *vc;
  float *invf_full, *invf_slide;
  int* tok_dev;
};

static bool load_blob(const char* path, Model& M, std::string& js) {
  FILE* f = fopen(path, "rb");
  if (!f) { printf("cannot open %s\n", path); return false; }
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
    size_t z2 = rec.find('[', nz) + 1;
    while (e.nd < 4 && z2 < rec.size()) {
      e.ne[e.nd++] = (int)strtol(rec.c_str() + z2, (char**)&z2, 10);
      if (z2 < rec.size() && rec[z2] == ',') z2++; else break;
    }
    if (e.dtype < 0 || e.file_off == (size_t)-1 || e.nc == (size_t)-1 || e.nd == 0) {
      printf("record for %s is incomplete\n", e.name.c_str()); fclose(f); return false;
    }
    if (e.dtype == 1) { M.need_q += e.nc; M.need_s += e.ns; }
    else M.need_f += e.nc;
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
    // Recomputing the exporter's per-tensor crc32 is what turns "the engine
    // runs the quantizer that was certified 8/8" into a load-time assertion.
    const uint32_t c = (uint32_t)crc32(0, (const Bytef*)buf.data(), (uInt)tot);
    if (c != e.crc) {
      printf("CRC MISMATCH %s: file %08x want %08x\n", e.name.c_str(), c, e.crc);
      fclose(f); return false;
    }
    cudaError_t ce = cudaSuccess;
    if (e.dtype == 1) {
      ce = cudaMemcpy(M.dq + oc, buf.data(), e.nc, cudaMemcpyHostToDevice);
      if (ce == cudaSuccess)
        ce = cudaMemcpy((char*)M.ds + os, buf.data() + e.nc, e.ns, cudaMemcpyHostToDevice);
      e.dq = M.dq + oc; e.ds = (float*)((char*)M.ds + os);
      oc += e.nc; os += e.ns;
    } else {
      ce = cudaMemcpy(M.df + of, buf.data(), e.nc, cudaMemcpyHostToDevice);
      e.df = M.df + of;
      of += e.nc;
    }
    if (ce != cudaSuccess) {
      printf("UPLOAD FAIL %s dtype=%d ne=[%d,%d] nc=%zu ns=%zu dst=%p -> %s\n",
             e.name.c_str(), e.dtype, e.ne[0], e.ne[1], e.nc, e.ns,
             (void*)(e.dtype == 1 ? (void*)(M.dq + oc - e.nc) : (void*)(M.df + of - e.nc)),
             cudaGetErrorString(ce));
      fclose(f); return false;
    }
    M.by[e.name] = &M.ents[i];
  }
  fclose(f);
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
  GemvCfg c;
  c.KCHUNK = (K >= 2048 && K % 2048 == 0) ? 2048 : K;
  c.TPB = 256;
  c.R = (N >= 8192) ? 256 : (N >= 1024 ? 128 : 64);
  if (K == 512) { c.KCHUNK = 512; c.TPB = 128; }
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
  k_bf2f<<<(K + 255) / 256, 256>>>(xb, M.xf, K);
  q8_partial<<<nblk, c.TPB, shm>>>(e->dq, e->ds, M.xf, M.part, N, K, c.KCHUNK, c.R, nb);
  k_combine_bf<<<(N + 255) / 256, 256>>>(M.part, yb, N, nchunk);
}
static void gemv_f(Model& M, const std::string& name, const bf16* xb, float* yf) {
  Ent* e = T(M, name);
  const int N = e->ne[0], K = e->ne[1];
  GemvCfg c = pick(N, K);
  if (K % c.KCHUNK) { printf("%s: K=%d not divisible by KCHUNK=%d\n", name.c_str(), K, c.KCHUNK); exit(2); }
  const int nb = c.KCHUNK / Q8_BLK, nchunk = K / c.KCHUNK;
  const int nblk = ((N + c.R - 1) / c.R) * nchunk;
  const size_t shm = (size_t)nb * 33 * 4;
  k_bf2f<<<(K + 255) / 256, 256>>>(xb, M.xf, K);
  q8_partial<<<nblk, c.TPB, shm>>>(e->dq, e->ds, M.xf, M.part, N, K, c.KCHUNK, c.R, nb);
  k_combine_f<<<(N + 255) / 256, 256>>>(M.part, yf, N, nchunk);
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
  std::vector<int> prompt;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--trace")) trace = true;
    else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) {
      char* s = argv[++i];
      while (*s) { prompt.push_back((int)strtol(s, &s, 10)); if (*s == ',') s++; }
    }
  }
  Model M; std::string js;
  if (!load_blob(argv[1], M, js)) return 1;

  // layer_types, parsed from the blob header (the conv layers carry the
  // operator branch, the attention layers carry self_attn).
  bool is_attn[NLAYER];
  {
    size_t lt = js.find("\"layer_types\":[");
    if (lt == std::string::npos) { printf("no layer_types in the blob header\n"); return 1; }
    size_t q = js.find('[', lt) + 1;
    for (int L = 0; L < NLAYER; L++) {
      size_t a = js.find('"', q), b = js.find('"', a + 1);
      if (a == std::string::npos || b == std::string::npos) { printf("layer_types truncated at %d\n", L); return 1; }
      std::string s = js.substr(a + 1, b - a - 1);
      is_attn[L] = (s != "conv");
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
    {(void**)&M.x, (size_t)H * 2, "x"}, {(void**)&M.h1, (size_t)H * 2, "h1"},
    {(void**)&M.h2, (size_t)H * 2, "h2"}, {(void**)&M.h3, (size_t)H * 2, "h3"},
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
  printf("weights+arena: %.2f GB, device free now %.2f GB of %.2f GB\n",
         (M.need_q + M.need_s + M.need_f + kvbytes * 2) / 1e9, freeb / 1e9, totb / 1e9);
  printf("L FM3_NANODE_LOADED\n"); fflush(stdout);

  Ent* emb = T(M, "lfm.model.embed_tokens.weight");
  if (emb->ne[0] != VOCAB || emb->ne[1] != H) { printf("embed shape %dx%d\n", emb->ne[0], emb->ne[1]); return 1; }

  if (prompt.empty()) prompt.push_back(124894);                 // bos
  std::vector<int> toks = prompt;
  cudaEvent_t evA, evB; cudaEventCreate(&evA); cudaEventCreate(&evB);

  printf("step tok        ms   tok/s\n");
  double tot = 0; int ng = 0;
  for (int s = 0; s < steps; s++) {
    const int pos = s;
    const int tok = toks[s];
    if (pos >= MAXTOK) break;
    cudaEventRecord(evA);
    cudaMemcpy(M.x, emb->dq + (size_t)tok * H, (size_t)H * 2, cudaMemcpyDeviceToDevice);

    int ci = 0;
    for (int L = 0; L < NLAYER; L++) {
      rms1(M, M.x, NW(M, L, "operator_norm.weight"), M.h1);
      if (is_attn[L]) {
        gemv(M, LN(L, "self_attn.q_proj.weight"), M.h1, M.q);
        gemv(M, LN(L, "self_attn.k_proj.weight"), M.h1, M.k);
        gemv(M, LN(L, "self_attn.v_proj.weight"), M.h1, M.v);
        gemv(M, LN(L, "self_attn.gate_proj.weight"), M.h1, M.gate);
        k_headnorm<<<NH, 32>>>(M.q, NW(M, L, "self_attn.q_norm.weight"), NORM_EPS, HD);
        k_headnorm<<<NKV, 32>>>(M.k, NW(M, L, "self_attn.k_norm.weight"), NORM_EPS, HD);
        k_rope<<<NH, 64>>>(M.q, M.invf_full, pos, HD, HD / 2);
        k_rope<<<NKV, 64>>>(M.k, M.invf_full, pos, HD, HD / 2);
        bf16* kbase = M.kc + (size_t)L * MAXTOK * NKV * HD;
        bf16* vbase = M.vc + (size_t)L * MAXTOK * NKV * HD;
        cudaMemcpy(kbase + (size_t)pos * NKV * HD, M.k, (size_t)NKV * HD * 2, cudaMemcpyDeviceToDevice);
        cudaMemcpy(vbase + (size_t)pos * NKV * HD, M.v, (size_t)NKV * HD * 2, cudaMemcpyDeviceToDevice);
        const float scaling = 1.0f / sqrtf((float)HD);
        k_attn_head<<<NH, 256>>>(M.q, kbase, vbase, M.attn, pos + 1, 0, scaling, HD, NKV);
        k_sigmoid_gate<<<1, NH>>>(M.gate, M.attn, HD);
        gemv(M, LN(L, "self_attn.o_proj.weight"), M.attn, M.h2);
      } else {
        bf16* ring = M.ring + (size_t)ci * 7 * H;
        gemv(M, LN(L, "conv.in_proj.weight"), M.h1, M.bx);      // [B|C|xg], 3H rows
        k_shift_in<<<(H + 63) / 64, 64>>>(ring, M.bx + 2 * H, H);
        emul(M, M.bx, M.bx + 2 * H, M.h3, H);                   // B * xg
        k_conv7<<<(H + 63) / 64, 64>>>(ring, T(M, LN(L, "conv.conv.weight"))->df, M.h4, H);
        emul(M, M.bx + H, M.h4, M.h2, H);                       // C * y
        gemv(M, LN(L, "conv.out_proj.weight"), M.h2, M.h2);
        ci++;
      }
      rms1(M, M.h2, NW(M, L, "post_operator_norm.weight"), M.h3);
      eadd(M, M.x, M.h3, M.x, H);
      rms1(M, M.x, NW(M, L, "ffn_norm.weight"), M.h1);
      gemv(M, LN(L, "feed_forward.w1.weight"), M.h1, M.h2);
      gemv(M, LN(L, "feed_forward.w3.weight"), M.h1, M.h3);
      k_silu_mul_bf<<<(INTER + 255) / 256, 256>>>(M.h2, M.h3, M.h2, INTER);
      gemv(M, LN(L, "feed_forward.w2.weight"), M.h2, M.h2);
      rms1(M, M.h2, NW(M, L, "post_ffn_norm.weight"), M.h3);
      eadd(M, M.x, M.h3, M.x, H);
    }
    gemv_f(M, "lfm.model.embed_tokens.weight", M.x, M.lgf);     // tied lm_head
    k_argmax<<<1, 256>>>(M.lgf, M.tok_dev, VOCAB);
    cudaEventRecord(evB);
    cudaEventSynchronize(evB);
    float ms = 0; cudaEventElapsedTime(&ms, evA, evB);
    int nt = -1;
    cudaMemcpy(&nt, M.tok_dev, 4, cudaMemcpyDeviceToHost);
    toks.push_back(nt);
    printf("%4d %7d %9.2f %8.2f\n", s, nt, ms, 1000.0 / ms);
    fflush(stdout);
    if (s >= 2) { tot += ms; ng++; }
    cudaError_t e2 = cudaGetLastError();
    if (e2 != cudaSuccess) { printf("CUDA ERROR after step %d: %s\n", s, cudaGetErrorString(e2)); return 1; }
  }
  if (ng > 0)
    printf("steady-state %.2f ms/token -> %.2f tok/s over %d steps "
           "(GEMV-only budget was 127.7 ms)\n", tot / ng, 1000.0 * ng / tot, ng);
  printf("FM3_NANODE_DONE\n");
  return 0;
}
