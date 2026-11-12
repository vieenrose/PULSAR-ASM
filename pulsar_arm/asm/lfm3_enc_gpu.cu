// lfm3_enc_gpu.cu -- LFM3 audio encoder on the GPU (port stage S3).
//
// Includes the text engine with its main() renamed, so the model loader, the Q8 GEMM
// arms and the combine kernels are the SHIPPED code, not copies. Activations are fp32
// here (the spec's arithmetic), so the C reference lfm3_enc.c is a tight per-stage
// oracle; LFM_ENC_H2=1 switches the linears to the batched bf16-activation half2 arm
// (the 1.9x measured in run #309), whose accuracy is gated end-to-end, not per bit.
//
// Stage order follows lfm3_enc.c exactly (which follows the checkpoint's
// Lfm3AudioDownsampler / Lfm2 block code; verified against HF to 2.2e-6 rms in
// run #304):
//   x1  = silu(conv2d(feat, conv_in,  k3 s2 p1))        (1,128,Tm)  -> (256,64,Wo)
//   x2  = conv2d(x1,  conv_mid,  k3 s2 p1, groups=256)             -> (256,32,W2)
//   x2b = silu(conv2d(x2, lin_mid, k1, groups=1))                  -> (256,32,W2)
//   x3  = conv2d(x2b, conv_end, k3 s2 p1, groups=256)              -> (256,16,W3)
//   x3b = silu(conv2d(x3, lin_end, k1, groups=1))                  -> (256,16,W3)
//   flat[t, c*EH3+h] = x3b[c,h,t]        (flatten(1,2))             -> (Tt,4096)
//   ds  = flat @ lin_out^T + b_out                                 -> (Tt,768)
// The 1x1 convs are GEMMs over H*W positions, so they run on the engine's Q8 arms;
// the 3x3 convs are direct kernels (a 9-MAC stencil, far from the GEMM threshold).
//
// Usage: lfm3_enc_gpu <enc.lfm3q8> <input_128xTm_f32.bin> <Tm> <outdir>
#define main lfm3_engine_main
#include "lfm3_nanode.cu"
#undef main
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

static const int EHID = 768, EFFN = 1792, EPROJ = 2048, ENLAYER = 17;
static const int ENH = 8, ENKV = 4, EHD = 96;
static const float EEPS = 1e-5f, ETHETA = 1e6f;
// LFM_ENC_PROBE=2: TIMING ABLATION ONLY -- the silu expf is replaced by the SFU
// __expf and the conv kernels skip their silu. The outputs of that run are NOT valid
// and the run prints a loud marker; it exists to price the expf, nothing else.
static const int g_encprobe = getenv("LFM_ENC_PROBE") ? atoi(getenv("LFM_ENC_PROBE")) : 0;
static const int g_convIn1 = getenv("LFM_ENC_CONVIN") ? atoi(getenv("LFM_ENC_CONVIN")) : 1;   // 1 shared, 0 flat
__device__ int g_probe_d;                     // device copy (a host getenv init is not usable in device code)

static int g_poison = -1;                    // read lazily by dalloc and gemm_setup
static int g_ench2 = 0;            // 0 = fp32 arms (the C-ref oracle path), 1 = batched half2
// encoder.sliding_window (section 39): 32 = HF's canonical encoder attention; 0 reproduces
// the old full-causal spec for A/B. Read from LFM_ENC_WIN, printed at start-up so a
// timing or gate run always records which mask it used.
static int g_enc_win = 32;

// ---------------------------------------------------------------- kernels
// fp32 -> bf16 (the h2 arm's input conversion), the engine's own f2bf rounding
__global__ void k_f2bf(const float* __restrict__ in, bf16* __restrict__ out, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = f2bf(in[i]);
}

// ---- 3x3 stride-2 pad-1 convs -------------------------------------------------
// Both convs use a FLAT work-item grid (one item = 4 consecutive x outputs of one
// (channel, row) pair) instead of a 3-D grid: with a 200-wide row and a 128-thread
// block, the 3-D form left 78% of the lanes idle, and that -- not the expf, which an
// ablation priced at ~10 ms -- was the cost. Out-of-range taps are SKIPPED (the C
// reference's bounds check), not zero-padded: same arithmetic, same accumulation
// order acc = b[o] + sum_i sum_j w[o,i,j] * x[iy][ix].
template <int IN_CH>                       // IN_CH=1: conv_in (mel plane); 256: depthwise
__global__ void k_conv3x3(const float* __restrict__ in, const float* __restrict__ w,
                          const float* __restrict__ b, float* __restrict__ out,
                          int nC, int nH, int nW, int nHo, int nWo, int silu) {
  const int item = blockIdx.x * blockDim.x + threadIdx.x;
  const int nxb = (nWo + 3) / 4;
  const int o = item / (nHo * nxb);                       // channel
  const int rem = item % (nHo * nxb);
  const int y = rem / nxb, x0 = (rem % nxb) * 4;
  if (o >= nC * nHo * nxb / (nHo * nxb) * nC + nC) return;   // (keeps the compiler honest)
  if (item >= nC * nHo * nxb) return;
  if (x0 >= nWo) return;
  const float bo = b[o];
  const float* wr = w + (size_t)o * 9;
  const float* base = in + (size_t)(IN_CH == 1 ? 0 : o * nH * nW);
  float acc[4] = { bo, bo, bo, bo };
  #pragma unroll
  for (int i = 0; i < 3; i++) {
    const int iy = y * 2 + i - 1;
    const bool rok = (iy >= 0 && iy < nH);
    const float* xr = base + (size_t)(rok ? iy : 0) * nW;
    float xv[9];
    #pragma unroll
    for (int j = 0; j < 9; j++) {
      const int ix = x0 * 2 - 1 + j;
      xv[j] = (rok && ix >= 0 && ix < nW) ? xr[ix] : 0.f;
    }
    #pragma unroll
    for (int q = 0; q < 4; q++) {
      if (x0 + q >= nWo) break;
      #pragma unroll
      for (int j = 0; j < 3; j++)
        acc[q] = __fadd_rn(acc[q], __fmul_rn(wr[i * 3 + j], xv[2 * q + j]));
    }
  }
  #pragma unroll
  for (int q = 0; q < 4; q++) {
    if (x0 + q >= nWo) break;
    float a = acc[q];
    if (silu && g_probe_d != 2) a = __fdiv_rn(a, __fadd_rn(1.f, expf(-a)));
    out[((size_t)o * nHo + y) * nWo + x0 + q] = a;
  }
}

// conv_in, shared-staged. The naive form re-reads the 9-tap mel window once per output
// CHANNEL: all 256 channels share the same window, so the (small) mel plane was
// generating ~256x its own traffic in L1/L2 -- 44 ms for 29.5 MMAC. One block owns a
// (row, 128-output-x) tile for ALL 256 channels: the 3x258 window is loaded once, and
// every thread walks 32 channels x 4 x-outputs. Same taps, same order, same expf.
__global__ void k_conv_in_shared(const float* __restrict__ in, const float* __restrict__ w,
                                 const float* __restrict__ b, float* __restrict__ out,
                                 int nH, int nW, int nHo, int nWo) {
  const int y = blockIdx.y;
  const int xb = blockIdx.x * 128;
  const int tid = threadIdx.x;                       // 256
  __shared__ float sx[3][258 + 3];
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    const int idx = tid + i * 256;
    if (idx < 3 * 258) {
      const int r = idx / 258, c = idx % 258;
      const int iy = y * 2 - 1 + r, ix = xb * 2 - 1 + c;
      sx[r][c] = (iy >= 0 && iy < nH && ix >= 0 && ix < nW) ? in[(size_t)iy * nW + ix] : 0.f;
    }
  }
  __syncthreads();
  const int q = tid & 31, sl = tid >> 5;              // 32 x-items x 8 channel slots
  const int x0 = xb + q * 4;
  #pragma unroll 4
  for (int oo = 0; oo < 32; oo++) {
    const int o = sl * 32 + oo;
    const float* wr = w + (size_t)o * 9;
    float acc[4] = { b[o], b[o], b[o], b[o] };
    #pragma unroll
    for (int i = 0; i < 3; i++)
      #pragma unroll
      for (int jj = 0; jj < 4; jj++) {
        const int x = x0 + jj; if (x >= nWo) break;
        const int c0 = 2 * (x - xb);                  // window column of tap ix = 2x-1
        #pragma unroll
        for (int j = 0; j < 3; j++)
          acc[jj] = __fadd_rn(acc[jj], __fmul_rn(wr[i * 3 + j], sx[i][c0 + j]));
      }
    #pragma unroll
    for (int jj = 0; jj < 4; jj++) {
      const int x = x0 + jj; if (x >= nWo) break;
      float a = acc[jj];
      if (g_probe_d != 2) a = __fdiv_rn(a, __fadd_rn(1.f, expf(-a)));
      out[((size_t)o * nHo + y) * nWo + x] = a;
    }
  }
}

// k_conv_in_shared2 (LFM_ENC_CI2, default 1): k_conv_in_shared with lane q owning outputs
// x = xb + q + 32 jj, so each store is 32 consecutive floats (the old x = xb + 4q + jj
// mapping wrote 4 floats per 16 B sector), and the thread's 3 x 3 window for its 4
// outputs is read from shared once into registers (the old loop re-read it per channel
// at a lane stride of 8 floats: 8-way bank conflicts). Per output: same bias start, same
// taps in the same order (i rows, then j), same silu.
__global__ void k_conv_in_shared2(const float* __restrict__ in, const float* __restrict__ w,
                                  const float* __restrict__ b, float* __restrict__ out,
                                  int nH, int nW, int nHo, int nWo) {
  const int y = blockIdx.y;
  const int xb = blockIdx.x * 128;
  const int tid = threadIdx.x;                       // 256
  __shared__ float sx[3][258 + 3];
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    const int idx = tid + i * 256;
    if (idx < 3 * 258) {
      const int r = idx / 258, c = idx % 258;
      const int iy = y * 2 - 1 + r, ix = xb * 2 - 1 + c;
      sx[r][c] = (iy >= 0 && iy < nH && ix >= 0 && ix < nW) ? in[(size_t)iy * nW + ix] : 0.f;
    }
  }
  __syncthreads();
  const int q = tid & 31, sl = tid >> 5;
  float xv[4][3][3];
  #pragma unroll
  for (int jj = 0; jj < 4; jj++) {
    const int c0 = 2 * (q + 32 * jj);
    #pragma unroll
    for (int i = 0; i < 3; i++)
      #pragma unroll
      for (int j = 0; j < 3; j++) xv[jj][i][j] = sx[i][c0 + j];
  }
  #pragma unroll 2
  for (int oo = 0; oo < 32; oo++) {
    const int o = sl * 32 + oo;
    const float* wr = w + (size_t)o * 9;
    float wv[9];
    #pragma unroll
    for (int k = 0; k < 9; k++) wv[k] = wr[k];
    const float bo = b[o];
    float* orow = out + ((size_t)o * nHo + y) * nWo;
    #pragma unroll
    for (int jj = 0; jj < 4; jj++) {
      const int x = xb + q + 32 * jj;
      float a = bo;
      #pragma unroll
      for (int i = 0; i < 3; i++)
        #pragma unroll
        for (int j = 0; j < 3; j++) a = __fadd_rn(a, __fmul_rn(wv[i * 3 + j], xv[jj][i][j]));
      if (g_probe_d != 2) a = __fdiv_rn(a, __fadd_rn(1.f, expf(-a)));
      if (x < nWo) orow[x] = a;
    }
  }
}

static void conv3x3(const float* in, const float* w, const float* b, float* out,
                    int C, int nH, int nW, int nHo, int nWo, int silu, int in1) {
  const int items = C * nHo * ((nWo + 3) / 4);
  const int TPB = 256;
  if (in1 && g_convIn1) {                             // shared-staged, 256 threads
    const int TPB2 = 256;
    dim3 g((nWo + 127) / 128, nHo);
    static const int ci2 = getenv("LFM_ENC_CI2") ? atoi(getenv("LFM_ENC_CI2")) : 1;
    if (ci2) k_conv_in_shared2<<<g, TPB2>>>(in, w, b, out, nH, nW, nHo, nWo);
    else     k_conv_in_shared<<<g, TPB2>>>(in, w, b, out, nH, nW, nHo, nWo);
  } else if (in1) {                                   // flat work-item grid (A/B arm)
    k_conv3x3<1><<<(C * nHo * ((nWo + 3) / 4) + 255) / 256, 256>>>(in, w, b, out, C, nH, nW, nHo, nWo, silu);
  } else {
    k_conv3x3<256><<<(items + TPB - 1) / TPB, TPB>>>(in, w, b, out, C, nH, nW, nHo, nWo, silu);
  }
}

// (C,H,W) -> (H*W, C) and back. One thread per element.
__global__ void k_chw2phw(const float* __restrict__ in, float* __restrict__ out, int C, int HW) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= C * HW) return;
  const int c = i / HW, p = i % HW;
  out[(size_t)p * C + c] = in[i];
}
// out[c][r] = in[r][c] for a rows x cols matrix, 32x32 smem tiles (coalesced both sides).
// Pure data movement: identical values to the one-thread-per-element kernels.
__global__ void k_transpose(const float* __restrict__ in, float* __restrict__ out, int rows, int cols) {
  __shared__ float t[32][33];
  const int c0 = blockIdx.x * 32, r0 = blockIdx.y * 32;
  const int tx = threadIdx.x, ty = threadIdx.y;
  for (int k = ty; k < 32; k += 8) {
    const int r = r0 + k, c = c0 + tx;
    if (r < rows && c < cols) t[k][tx] = in[(size_t)r * cols + c];
  }
  __syncthreads();
  for (int k = ty; k < 32; k += 8) {
    const int c = c0 + k, r = r0 + tx;
    if (r < rows && c < cols) out[(size_t)c * rows + r] = t[tx][k];
  }
}
static void transpose(const float* in, float* out, int rows, int cols) {
  k_transpose<<<dim3((cols + 31) / 32, (rows + 31) / 32), dim3(32, 8)>>>(in, out, rows, cols);
}
static const int g_enctr = getenv("LFM_ENC_TR") ? atoi(getenv("LFM_ENC_TR")) : 1;
__global__ void k_phw2chw(const float* __restrict__ in, float* __restrict__ out, int C, int HW) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= C * HW) return;
  const int c = i / HW, p = i % HW;
  out[i] = in[(size_t)p * C + c];
}

// y[p,o] += bias[o], then silu when asked (the C reference's order: acc + bias, then
// the activation on the whole tensor).
__global__ void k_bias_silu(float* __restrict__ y, const float* __restrict__ bias,
                            size_t n, int O, int silu) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = __fadd_rn(y[i], bias[i % O]);
  if (silu && g_probe_d != 2) v = __fdiv_rn(v, __fadd_rn(1.f, expf(-v)));
  y[i] = v;
}

// flat[t, c*EH3+h] = x[c, h, t] with x in (H*W, C) = (EH3*Tt, 256) PHW layout.
__global__ void k_flatten(const float* __restrict__ phw, float* __restrict__ flat,
                          int C, int EH3, int Tt) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= Tt * C * EH3) return;
  const int t = i / (C * EH3), rem = i % (C * EH3);
  const int c = rem / EH3, h = rem % EH3;
  flat[i] = phw[(size_t)(h * Tt + t) * C + c];
}


// Stage-level sync + error read: an async kernel fault surfaces late and points at
// the wrong launch, so every stage is checked where it is issued (run #297's lesson).
static cudaEvent_t g_seva, g_sevb;
// LFM_ENC_TIME=1 per-stage min (the bench); =2 also prints the SUM and call count per name,
// which is what attributes a whole pass (a per-layer stage otherwise reports one call only).
static const int g_enctime = getenv("LFM_ENC_TIME") ? atoi(getenv("LFM_ENC_TIME")) : 0;
static const int g_reps = getenv("LFM_ENC_REPS") ? atoi(getenv("LFM_ENC_REPS")) : 1;
// The Nano's GPU clock ramps over seconds (the backbone harness fights the same
// thing with LFM_WARMMS), so a single pass per stage is not a measurement: stages are
// run LFM_ENC_REPS times and the MIN per stage is kept, which is the same protocol the
// backbone benches use.
#include <map>
#include <string>
static std::map<std::string, double> g_best;
static std::map<std::string, double> g_sum;      // LFM_ENC_TIME=2: sum and count per stage name.
static std::map<std::string, long> g_cnt;        // The min is what the per-rep bench wants, but a
                                                 // stage that runs once per layer reports only its
                                                 // fastest call, so the table understates a 17-layer
                                                 // pass by up to 17x -- the sum is what attributes
                                                 // a served pass's GPU time.
static std::map<std::string, double> g_last;
static int g_sync = -2;   // LFM_ENC_SYNC=1 restores per-kernel device syncs (the gate path)
static void chk(const char* what) {
  if (g_enctime) {                                      // per-stage: syncs, so read the split
    cudaEventRecord(g_sevb); cudaEventSynchronize(g_sevb);
    float ms = 0.f; cudaEventElapsedTime(&ms, g_seva, g_sevb);
    std::string k(what);
    if (!g_best.count(k) || ms < g_best[k]) g_best[k] = ms;
    g_sum[k] += ms; g_cnt[k] += 1;
    g_last[k] = ms;
    cudaEventRecord(g_seva);
  }
  cudaError_t e = cudaGetLastError();
  if (e) { printf("LAUNCH ERROR at %s: %s\n", what, cudaGetErrorString(e)); exit(1); }
  // The per-kernel cudaDeviceSynchronize is a DEBUG feature and it was the encoder's
  // dominant cost: with ~200 chk() sites per pass, the 129.7 s clip's layer loop took
  // 67.5 s of process wall against 1.63 s of GPU time (measured; pooling the partial
  // buffer changed it by 1 s, so it is not the allocator). Each sync forces the Jetson
  // GPU through an idle<->busy transition, which is exactly the devfreq cost this project
  // measures elsewhere and hides behind min-of-N. Launch refusals are still caught here,
  // cheaply, by cudaGetLastError(); async EXEC faults are caught by the single sync at
  // the end of main (chk_final). The gate sets LFM_ENC_SYNC=1 to keep per-stage syncs.
  if (g_sync == -2) g_sync = getenv("LFM_ENC_SYNC") ? atoi(getenv("LFM_ENC_SYNC")) : 0;
  if (g_sync) {
    cudaDeviceSynchronize();
    e = cudaGetLastError();
    if (e) { printf("EXEC ERROR at %s: %s\n", what, cudaGetErrorString(e)); exit(1); }
  }
}
// LFM_ENC_TIME=3 additionally keys the table by SHAPE: a stage name that covers 83 launches of
// 20 different matrices says nothing about which of them cost the time, and this is how the
// encoder's GEMM budget was attributed.
static const char* gemm_tag(const char* what, int N, int K, int T, int ct, int tl, int nblk, int tpb) {
  static char buf[8][160]; static int i = 0;
  char* p = buf[i++ & 7];
  snprintf(p, 160, "%s N=%d K=%d T=%d ct=%d tl=%d blk=%d thr=%d", what, N, K, T, ct, tl, nblk, tpb);
  return p;
}
// One sync for the whole process: an async fault that no cudaGetLastError() at launch
// time could see has to be caught before any number this tool prints is believed.
static void chk_final(const char* what) {
  cudaDeviceSynchronize();
  cudaError_t e = cudaGetLastError();
  if (e) { printf("EXEC ERROR (final, at %s): %s\n", what, cudaGetErrorString(e)); exit(1); }
}

// ---------------------------------------------------------------- GEMM
// The blob keeps SOME encoder linears as f32 (lin_mid, lin_end, and the projector's
// LayerNorm weights -- the same dtype table as Liquid's mmproj GGUF, cross-checked in
// plan section 13a), so the Q8 arms cannot serve them. 16x16-tiled SGEMM: one output
// per thread, k ascending in one accumulator, which is the C reference's order.
// k_sgemm8: 128x128 output tile, 256 threads, 8x8 outputs per thread (rows ty*4+i and
// 64+ty*4+i, cols tx*4+j and 64+tx*4+j: float4 shared reads, conflict-free). Each output
// still accumulates k ascending in ONE register with fmaf, exactly as k_sgemm_f32: the
// same values. Needs K % 16 == 0.
__global__ void __launch_bounds__(256) k_sgemm8(const float* __restrict__ x, const float* __restrict__ w,
                                                float* __restrict__ y, int T, int N, int K) {
  const int t0 = blockIdx.y * 128, n0 = blockIdx.x * 128;
  __shared__ __align__(16) float sx[16][132];        // [k][t]
  __shared__ __align__(16) float sw[16][132];        // [k][n]
  const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4, tid = threadIdx.x;
  float acc[8][8];
  #pragma unroll
  for (int i = 0; i < 8; i++)
    #pragma unroll
    for (int j = 0; j < 8; j++) acc[i][j] = 0.f;
  for (int k0 = 0; k0 < K; k0 += 16) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      const int idx = tid + i * 256;                   // 2048 = 128 rows x 16 k
      const int r = idx >> 4, c = idx & 15;
      const int t = t0 + r, n = n0 + r;
      sx[c][r] = (t < T) ? x[(size_t)t * K + k0 + c] : 0.f;
      sw[c][r] = (n < N) ? w[(size_t)n * K + k0 + c] : 0.f;
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < 16; k++) {
      const float4 xa = *reinterpret_cast<const float4*>(&sx[k][ty * 4]);
      const float4 xb = *reinterpret_cast<const float4*>(&sx[k][64 + ty * 4]);
      const float4 wa = *reinterpret_cast<const float4*>(&sw[k][tx * 4]);
      const float4 wb = *reinterpret_cast<const float4*>(&sw[k][64 + tx * 4]);
      const float xv[8] = { xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w };
      const float wv[8] = { wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w };
      #pragma unroll
      for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = __fmaf_rn(xv[i], wv[j], acc[i][j]);
    }
    __syncthreads();
  }
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    const int t = t0 + (i < 4 ? ty * 4 + i : 64 + ty * 4 + i - 4);
    if (t >= T) continue;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
      const int n = n0 + (j < 4 ? tx * 4 + j : 64 + tx * 4 + j - 4);
      if (n < N) y[(size_t)t * N + n] = acc[i][j];
    }
  }
}

__global__ void k_sgemm_f32(const float* __restrict__ x, const float* __restrict__ w,
                            float* __restrict__ y, int T, int N, int K) {
  // 64x64 output tile, 256 threads, 4x4 outputs per thread. Each output accumulates
  // k ascending in ONE register, so the summation order is the C reference's and the
  // gate value is unchanged by the tiling. Pads (67) are chosen for ~2-way shared
  // conflicts: sx is [t][k] (warp-broadcast along k), sw is transposed [k][n] so the
  // per-thread w reads are a contiguous span.
  const int t0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
  __shared__ float sx[64][67];
  __shared__ float sw[64][67];
  const int ty = threadIdx.y, tx = threadIdx.x;
  const int tid = ty * 16 + tx;
  float acc[4][4];
  #pragma unroll
  for (int i = 0; i < 4; i++)
    #pragma unroll
    for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
  for (int k0 = 0; k0 < K; k0 += 64) {
    const int kk = (k0 + 64 <= K) ? k0 : (K - 64);          // K is a multiple of 32; tail reload
    #pragma unroll
    for (int i = 0; i < 16; i++) {
      const int idx = tid + i * 256;
      const int r = idx >> 6, c = idx & 63;
      const int t = t0 + r, n = n0 + r;
      sx[r][c] = (t < T) ? x[(size_t)t * K + kk + c] : 0.f;
      sw[c][r] = (n < N) ? w[(size_t)n * K + kk + c] : 0.f;
    }
    __syncthreads();
    const int kstop = (k0 + 64 <= K) ? 64 : (K - k0);
    #pragma unroll 4
    for (int k = 0; k < kstop; k++) {
      float xv[4], wv[4];
      #pragma unroll
      for (int i = 0; i < 4; i++) xv[i] = sx[ty * 4 + i][k];
      #pragma unroll
      for (int j = 0; j < 4; j++) wv[j] = sw[k][tx * 4 + j];
      #pragma unroll
      for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = __fmaf_rn(xv[i], wv[j], acc[i][j]);
    }
    __syncthreads();
    if (k0 + 64 >= K) break;
  }
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    const int t = t0 + ty * 4 + i;
    if (t >= T) continue;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      const int n = n0 + tx * 4 + j;
      if (n < N) y[(size_t)t * N + n] = acc[i][j];
    }
  }
}

// LFM_ENC_BAT32=1: use the batched fp32 GEMV arm. Read ONCE and shared by gemm_setup and
// gemm_run: setup picks KC against the arm's shared-memory budget (it stages CT
// positions), and run launches it. Reading the env in both places independently produced
// a real bug -- setup sized KC for the per-position arm, so the downsampler's K=4096
// lin_out asked the batched launch for 67 KB of shared, was refused, and silently fell
// back to 1622 per-position launches.
static int g_bat32 = -2;
static void bat32_init() { if (g_bat32 == -2) g_bat32 = getenv("LFM_ENC_BAT32") ? atoi(getenv("LFM_ENC_BAT32")) : 0; }

// q8_partial_T<CT>: q8_partial<float,FMA=true> for CT positions per block, sharing each
// weight block's registers across the CT positions. Per (row, chunk, position) the
// arithmetic is q8_partial's exactly: lane L folds blocks b = L, L+32, ..., dot_b is the
// same 32-term fmaf chain over the 33-padded staged x, acc = fmaf(s[b], dot_b, acc), then
// the same shuffle-down tree; lane 0 stores the value q8_partial would have staged in sh[].
// So partials are bit-identical, and weight traffic drops CT-fold (the encoder re-streams
// each layer's weights once per position: ~186 GB per pass at Tt=1622). No sh[] buffer
// and CT <= 4 keep shared <= 16 KB/block, so the SM still holds several blocks (CT=8 with
// 48 KB ran at 1 block/SM and lost 3.7 s). LFM_ENC_PT = 0 (per-position loop) | 2 | 4.
static int g_encpt = -2;
template <int CT>
__global__ void __launch_bounds__(256, 4) q8_partial_T(const int8_t* __restrict__ q, const float* __restrict__ s,
                                                       const float* __restrict__ x, float* __restrict__ part,
                                                       int N, int K, int KCHUNK, int R, int nb, int T) {
  extern __shared__ float xs[];                       // [CT][nb*33]
  const int nchunk = K / KCHUNK;
  const int rowblk = blockIdx.x / nchunk, chunk = blockIdx.x % nchunk;
  const int koff = chunk * KCHUNK;
  const int t0 = blockIdx.y * CT;
  const int NT = blockDim.x, tid = threadIdx.x;
  const int xstride = nb * 33;
  #pragma unroll
  for (int p = 0; p < CT; p++) {
    const int t = t0 + p;
    if (t < T) {
      const float* xg = x + (size_t)t * K + koff;
      for (int i = tid; i < KCHUNK; i += NT) xs[p * xstride + (i >> 5) * 33 + (i & 31)] = xg[i];
    }
  }
  __syncthreads();
  const int base = rowblk * R;
  const int lane = tid & 31, wid = tid >> 5;
  const int WPB = NT >> 5, GR = R / WPB;
  for (int g = 0; g < GR; g++) {
    const int row = base + wid + g * WPB;
    if (row >= N) continue;                           // warp-uniform: row depends on wid only
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
      // Code-outer, position-inner: each code converted once for all CT positions; every
      // dot[p] still runs codes 0..31 in order (same chain), so the bits are unchanged.
      const float* xb = xs + (size_t)b * 33;
      float dot[CT];
      #pragma unroll
      for (int p = 0; p < CT; p++) dot[p] = 0.f;
      #pragma unroll
      for (int j = 0; j < 32; j++) {
        const float c = (float)(j < 16 ? c0[j] : c1[j - 16]);
        #pragma unroll
        for (int p = 0; p < CT; p++) dot[p] = __fmaf_rn(c, xb[p * xstride + j], dot[p]);
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
      if (lane == 0 && t < T) part[((size_t)t * N + row) * nchunk + chunk] = a;
    }
  }
}

// q8_partial_TS<CT, RW, NJ>: q8_partial_T for nb < 32 (K = 768: nb = 24), where the
// one-block-per-lane mapping leaves lanes nb..31 idle through the whole dot phase. A warp
// takes RW rows at once: the RW*nb (row, block) items are spread NJ per lane (RW*nb ==
// 32*NJ), every lane runs the same NJ items, each item's code converted once for its CT
// positions -- exactly q8_partial_T's per-(row, block) work, so the value an item yields,
// fmaf(s_b, dot_b, 0.f), is the value lane b computed there (one block per lane: acc
// starts at 0). Per (row, position) the slot values are then gathered into lanes 0..nb-1
// (lanes >= nb hold 0.f, as before) and reduced by the same shuffle-down tree. Same
// operands, same order: bit-identical. LFM_ENC_TS=0 disables it.
static int g_encts = -2;
template <int CT, int RW, int NJ>
__global__ void __launch_bounds__(256, 4) q8_partial_TS(const int8_t* __restrict__ q, const float* __restrict__ s,
                                                        const float* __restrict__ x, float* __restrict__ part,
                                                        int N, int K, int KCHUNK, int R, int nb, int T) {
  extern __shared__ float xs[];                       // [CT][nb*33]
  const int nchunk = K / KCHUNK;
  const int rowblk = blockIdx.x / nchunk, chunk = blockIdx.x % nchunk;
  const int koff = chunk * KCHUNK;
  const int t0 = blockIdx.y * CT;
  const int NT = blockDim.x, tid = threadIdx.x;
  const int xstride = nb * 33;
  #pragma unroll
  for (int p = 0; p < CT; p++) {
    const int t = t0 + p;
    if (t < T) {
      const float* xg = x + (size_t)t * K + koff;
      for (int i = tid; i < KCHUNK; i += NT) xs[p * xstride + (i >> 5) * 33 + (i & 31)] = xg[i];
    }
  }
  __syncthreads();
  const int base = rowblk * R;
  const int lane = tid & 31, wid = tid >> 5;
  const int WPB = NT >> 5, GR = R / WPB;
  for (int g0 = 0; g0 < GR; g0 += RW) {
    float v[NJ][CT];
    #pragma unroll
    for (int j = 0; j < NJ; j++) {
      const int i = lane + 32 * j, r = i / nb, b = i - r * nb;
      const int row = base + wid + (g0 + r) * WPB;
      #pragma unroll
      for (int p = 0; p < CT; p++) v[j][p] = 0.f;
      if (row < N) {
        const int8_t* qr = q + (size_t)row * K + koff;
        const float sb = s[(size_t)row * (K / Q8_BLK) + (size_t)chunk * nb + b];
        const int4 a0 = *reinterpret_cast<const int4*>(qr + (size_t)b * Q8_BLK);
        const int4 a1 = *reinterpret_cast<const int4*>(qr + (size_t)b * Q8_BLK + 16);
        const int8_t* c0 = reinterpret_cast<const int8_t*>(&a0);
        const int8_t* c1 = reinterpret_cast<const int8_t*>(&a1);
        const float* xb = xs + (size_t)b * 33;
        float dot[CT];
        #pragma unroll
        for (int p = 0; p < CT; p++) dot[p] = 0.f;
        #pragma unroll
        for (int jj = 0; jj < 32; jj++) {                // code-outer (see q8_partial_T)
          const float c = (float)(jj < 16 ? c0[jj] : c1[jj - 16]);
          #pragma unroll
          for (int p = 0; p < CT; p++) dot[p] = __fmaf_rn(c, xb[p * xstride + jj], dot[p]);
        }
        #pragma unroll
        for (int p = 0; p < CT; p++) v[j][p] = __fmaf_rn(sb, dot[p], 0.f);
      }
    }
    #pragma unroll
    for (int r = 0; r < RW; r++) {
      const int row = base + wid + (g0 + r) * WPB;
      if (row >= N) continue;                          // warp-uniform
      const int i = r * nb + lane, srcl = i & 31, sj = i >> 5;
      #pragma unroll
      for (int p = 0; p < CT; p++) {
        float a = 0.f;
        #pragma unroll
        for (int j = 0; j < NJ; j++) { const float t = __shfl_sync(~0u, v[j][p], srcl); if (sj == j) a = t; }
        if (lane >= nb) a = 0.f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(~0u, a, o);
        const int t = t0 + p;
        if (lane == 0 && t < T) part[((size_t)t * N + row) * nchunk + chunk] = a;
      }
    }
  }
}

// q8_partial_TL<CT>: q8_partial_TS's values with one row per THREAD (nb == 24, R == 256 ==
// blockDim) and no shuffles. TS gives lane b the value v_b = fmaf(s_b, dot_b, 0.f) (lanes
// 24..31 hold 0.f) and reduces with a += shfl_down(a, o), o = 16..1, so lane 0 ends with
// the tree p_i = v_i + v_{i+16}, q_i = p_i + p_{i+8}, r_i = q_i + q_{i+4}, s_i = r_i +
// r_{i+2}, f = s_0 + s_1. Here the p_i are formed in bit-reversed i order and combined by a
// binary counter (earlier operand on the left, as the lower lane is), so every add has
// the shuffle tree's operands: bit-identical. Same staged x, same dot chains.
static int g_enctl = -2;
// LFM_ENC_RPA (default 2; 1 = sync per tensor, 0 = old malloc/free): the in-place tile repacks reuse one persistent scratch
// (grown on demand), same kernels and bytes, no per-tensor cudaMalloc / cudaFree (102
// repacks cost 0.19 s of wall). At 2 the copies are stream-ordered on the legacy stream
// with no per-tensor sync; errors surface at the next chk / chk_final. (#188's +1-3 s
// outliers were slow eMMC blob reads, seen in both arms, #189.)
static void* g_rps = nullptr; static size_t g_rpsz = 0;
static int g_rpa = -2;
static void* rp_scratch(size_t n) {
  if (g_rpa == -2) g_rpa = getenv("LFM_ENC_RPA") ? atoi(getenv("LFM_ENC_RPA")) : 2;
  if (n > g_rpsz) {
    if (g_rps) cudaFree(g_rps);
    if (cudaMalloc(&g_rps, n)) { printf("repack scratch\n"); exit(2); }
    g_rpsz = n;
  }
  return g_rps;
}
template <int CT>
__device__ __forceinline__ void tl_leaf(const int4* __restrict__ qt, const float* __restrict__ st_s,
                                        const float* xs, int b, float (&v)[CT]) {
  const int4 a0 = qt[b * 64], a1 = qt[b * 64 + 32];
  const unsigned w[8] = { (unsigned)a0.x, (unsigned)a0.y, (unsigned)a0.z, (unsigned)a0.w,
                          (unsigned)a1.x, (unsigned)a1.y, (unsigned)a1.z, (unsigned)a1.w };
  const float sb = st_s[b * 32];
  const float* xb = xs + b * 33;
  float dot[CT];
  #pragma unroll
  for (int p = 0; p < CT; p++) dot[p] = 0.f;
  #pragma unroll
  for (int jj = 0; jj < 32; jj++) {
    const float c = (float)(int)(signed char)(w[jj >> 2] >> (8 * (jj & 3)));
    #pragma unroll
    for (int p = 0; p < CT; p++) dot[p] = __fmaf_rn(c, xb[p * (24 * 33) + jj], dot[p]);
  }
  #pragma unroll
  for (int p = 0; p < CT; p++) v[p] = __fmaf_rn(sb, dot[p], 0.f);
}
__device__ __noinline__ float4 tl_leaf4(const int4* __restrict__ qt, const float* __restrict__ st_s,
                                        const float* xs, int b) {
  float v[4];
  tl_leaf<4>(qt, st_s, xs, b, v);
  return make_float4(v[0], v[1], v[2], v[3]);
}
// tl_leaf4v (LFM_ENC_XV=4): x staged unpadded (stride 32 per block: every thread of a warp
// reads the same address, so the pad against bank conflicts is not needed) and read as one
// LDS.128 per 4 codes and position instead of one LDS.32 per code and position. Each
// position's chain still runs over jj = 0..31 in order: bit-identical.
__device__ __noinline__ float4 tl_leaf4v(const int4* __restrict__ qt, const float* __restrict__ st_s,
                                         const float* xs, int b) {
  const int4 a0 = qt[b * 64], a1 = qt[b * 64 + 32];
  const unsigned w[8] = { (unsigned)a0.x, (unsigned)a0.y, (unsigned)a0.z, (unsigned)a0.w,
                          (unsigned)a1.x, (unsigned)a1.y, (unsigned)a1.z, (unsigned)a1.w };
  const float sb = st_s[b * 32];
  const float4* xb = reinterpret_cast<const float4*>(xs + b * 32);
  float dot[4] = { 0.f, 0.f, 0.f, 0.f };
  #pragma unroll
  for (int q = 0; q < 8; q++) {
    const float c0 = (float)(int)(signed char)(w[q]);
    const float c1 = (float)(int)(signed char)(w[q] >> 8);
    const float c2 = (float)(int)(signed char)(w[q] >> 16);
    const float c3 = (float)(int)(signed char)(w[q] >> 24);
    #pragma unroll
    for (int p = 0; p < 4; p++) {
      const float4 u = xb[p * (24 * 8) + q];
      dot[p] = __fmaf_rn(c0, u.x, dot[p]);
      dot[p] = __fmaf_rn(c1, u.y, dot[p]);
      dot[p] = __fmaf_rn(c2, u.z, dot[p]);
      dot[p] = __fmaf_rn(c3, u.w, dot[p]);
    }
  }
  return make_float4(__fmaf_rn(sb, dot[0], 0.f), __fmaf_rn(sb, dot[1], 0.f),
                     __fmaf_rn(sb, dot[2], 0.f), __fmaf_rn(sb, dot[3], 0.f));
}
// tl_leaf4v2 (LFM_ENC_XV=7): leaves b and b + 16 (single blocks, both live when b < 8)
// with their dot chains interleaved (8 FMA chains); each leaf's value is tl_leaf4v's.
struct tl8 { float4 lo, hi; };
__device__ __noinline__ tl8 tl_leaf4v2(const int4* __restrict__ qt, const float* __restrict__ st_s,
                                       const float* xs, int b) {
  const int c = b + 16;
  const int4 a0 = qt[b * 64], a1 = qt[b * 64 + 32], e0 = qt[c * 64], e1 = qt[c * 64 + 32];
  const unsigned w[8] = { (unsigned)a0.x, (unsigned)a0.y, (unsigned)a0.z, (unsigned)a0.w,
                          (unsigned)a1.x, (unsigned)a1.y, (unsigned)a1.z, (unsigned)a1.w };
  const unsigned v[8] = { (unsigned)e0.x, (unsigned)e0.y, (unsigned)e0.z, (unsigned)e0.w,
                          (unsigned)e1.x, (unsigned)e1.y, (unsigned)e1.z, (unsigned)e1.w };
  const float sb = st_s[b * 32], sc = st_s[c * 32];
  const float4* xb = reinterpret_cast<const float4*>(xs + b * 32);
  const float4* xc = reinterpret_cast<const float4*>(xs + c * 32);
  float d[4] = { 0.f, 0.f, 0.f, 0.f }, g[4] = { 0.f, 0.f, 0.f, 0.f };
  #pragma unroll
  for (int q = 0; q < 8; q++) {
    const float c0 = (float)(int)(signed char)(w[q]), c1 = (float)(int)(signed char)(w[q] >> 8);
    const float c2 = (float)(int)(signed char)(w[q] >> 16), c3 = (float)(int)(signed char)(w[q] >> 24);
    const float h0 = (float)(int)(signed char)(v[q]), h1 = (float)(int)(signed char)(v[q] >> 8);
    const float h2 = (float)(int)(signed char)(v[q] >> 16), h3 = (float)(int)(signed char)(v[q] >> 24);
    #pragma unroll
    for (int p = 0; p < 4; p++) {
      const float4 u = xb[p * (24 * 8) + q], z = xc[p * (24 * 8) + q];
      d[p] = __fmaf_rn(c0, u.x, d[p]); g[p] = __fmaf_rn(h0, z.x, g[p]);
      d[p] = __fmaf_rn(c1, u.y, d[p]); g[p] = __fmaf_rn(h1, z.y, g[p]);
      d[p] = __fmaf_rn(c2, u.z, d[p]); g[p] = __fmaf_rn(h2, z.z, g[p]);
      d[p] = __fmaf_rn(c3, u.w, d[p]); g[p] = __fmaf_rn(h3, z.w, g[p]);
    }
  }
  tl8 o;
  o.lo = make_float4(__fmaf_rn(sb, d[0], 0.f), __fmaf_rn(sb, d[1], 0.f), __fmaf_rn(sb, d[2], 0.f), __fmaf_rn(sb, d[3], 0.f));
  o.hi = make_float4(__fmaf_rn(sc, g[0], 0.f), __fmaf_rn(sc, g[1], 0.f), __fmaf_rn(sc, g[2], 0.f), __fmaf_rn(sc, g[3], 0.f));
  return o;
}
// tl_leaf4v3 (LFM_ENC_XV=8): blocks b, b + 16 and b + 8 (b < 8) with interleaved chains;
// lo / hi are leaf b / b + 16 (tl_leaf4v2's), nx is leaf b + 8, used at the next k.
struct tl12 { float4 lo, hi, nx; };
__device__ __noinline__ tl12 tl_leaf4v3(const int4* __restrict__ qt, const float* __restrict__ st_s,
                                        const float* xs, int b) {
  const int bk[3] = { b, b + 16, b + 8 };
  unsigned w[3][8]; float sc[3];
  #pragma unroll
  for (int r = 0; r < 3; r++) {
    const int4 u0 = qt[bk[r] * 64], u1 = qt[bk[r] * 64 + 32];
    w[r][0] = u0.x; w[r][1] = u0.y; w[r][2] = u0.z; w[r][3] = u0.w;
    w[r][4] = u1.x; w[r][5] = u1.y; w[r][6] = u1.z; w[r][7] = u1.w;
    sc[r] = st_s[bk[r] * 32];
  }
  float d[3][4];
  #pragma unroll
  for (int r = 0; r < 3; r++)
    #pragma unroll
    for (int p = 0; p < 4; p++) d[r][p] = 0.f;
  #pragma unroll
  for (int q = 0; q < 8; q++) {
    #pragma unroll
    for (int r = 0; r < 3; r++) {
      const float c0 = (float)(int)(signed char)(w[r][q]), c1 = (float)(int)(signed char)(w[r][q] >> 8);
      const float c2 = (float)(int)(signed char)(w[r][q] >> 16), c3 = (float)(int)(signed char)(w[r][q] >> 24);
      const float4* xb = reinterpret_cast<const float4*>(xs + bk[r] * 32);
      #pragma unroll
      for (int p = 0; p < 4; p++) {
        const float4 u = xb[p * (24 * 8) + q];
        d[r][p] = __fmaf_rn(c0, u.x, d[r][p]); d[r][p] = __fmaf_rn(c1, u.y, d[r][p]);
        d[r][p] = __fmaf_rn(c2, u.z, d[r][p]); d[r][p] = __fmaf_rn(c3, u.w, d[r][p]);
      }
    }
  }
  tl12 o;
  o.lo = make_float4(__fmaf_rn(sc[0], d[0][0], 0.f), __fmaf_rn(sc[0], d[0][1], 0.f), __fmaf_rn(sc[0], d[0][2], 0.f), __fmaf_rn(sc[0], d[0][3], 0.f));
  o.hi = make_float4(__fmaf_rn(sc[1], d[1][0], 0.f), __fmaf_rn(sc[1], d[1][1], 0.f), __fmaf_rn(sc[1], d[1][2], 0.f), __fmaf_rn(sc[1], d[1][3], 0.f));
  o.nx = make_float4(__fmaf_rn(sc[2], d[2][0], 0.f), __fmaf_rn(sc[2], d[2][1], 0.f), __fmaf_rn(sc[2], d[2][2], 0.f), __fmaf_rn(sc[2], d[2][3], 0.f));
  return o;
}
static const int g_tlni = getenv("LFM_ENC_TLNI") ? atoi(getenv("LFM_ENC_TLNI")) : 1;
__constant__ int g_encgsw = 1;   // LFM_ENC_GSW
template <int CT, bool NI = false, int XV = 1>
__global__ void __launch_bounds__(256, 4) q8_partial_TL(const int8_t* __restrict__ q, const float* __restrict__ s,
                                                        const float* __restrict__ x, float* __restrict__ part,
                                                        int N, int K, int KCHUNK, int R, int nb, int T) {
  extern __shared__ float xs[];                       // [CT][nb*33]
  const int nchunk = K / KCHUNK;
  // g_encgsw (LFM_ENC_GSW): grid launched (position groups, row blocks); the block's work is
  // unchanged, consecutive blocks share one row block's weights through L2.
  const int gbx = (g_encgsw && XV >= 4) ? blockIdx.y : blockIdx.x, gby = (g_encgsw && XV >= 4) ? blockIdx.x : blockIdx.y;
  const int rowblk = gbx / nchunk, chunk = gbx % nchunk;
  const int koff = chunk * KCHUNK;
  const int t0 = gby * CT;
  const int NT = blockDim.x, tid = threadIdx.x;
  const int xstride = XV >= 4 ? 24 * 32 : 24 * 33;
  #pragma unroll
  for (int p = 0; p < CT; p++) {
    const int t = t0 + p;
    if (t < T) {
      const float* xg = x + (size_t)t * K + koff;
      for (int i = tid; i < KCHUNK; i += NT) xs[p * xstride + (i >> 5) * (XV >= 4 ? 32 : 33) + (i & 31)] = xg[i];
    }
  }
  __syncthreads();
  const int row = rowblk * R + tid;
  if (row >= N) return;
  // Tiled weights (q8_tile_768): per 32-row tile and block b, the 32 rows' first int4,
  // then their second int4, then (separately) their 32 scales -- one 512 B line per
  // warp load instead of 32 row-strided sectors.
  const int4* qt = reinterpret_cast<const int4*>(q) + (size_t)(row >> 5) * 24 * 64 + (row & 31);
  const float* st_s = s + (size_t)(row >> 5) * 24 * 32 + (row & 31);
  float st[4][CT];
  float pv[CT];
  float4 nx4;
  #pragma unroll
  for (int k = 0; k < 16; k++) {
    const int i = ((k & 1) << 3) | ((k & 2) << 1) | ((k & 4) >> 1) | ((k & 8) >> 3);
    float lo[CT], hi[CT];
    if (XV == 8 && i < 8) {
      const tl12 r = tl_leaf4v3(qt, st_s, xs, i);
      lo[0] = r.lo.x; lo[1 % CT] = r.lo.y; lo[2 % CT] = r.lo.z; lo[3 % CT] = r.lo.w;
      hi[0] = r.hi.x; hi[1 % CT] = r.hi.y; hi[2 % CT] = r.hi.z; hi[3 % CT] = r.hi.w;
      nx4 = r.nx;
    } else if (XV == 8) {
      lo[0] = nx4.x; lo[1 % CT] = nx4.y; lo[2 % CT] = nx4.z; lo[3 % CT] = nx4.w;
    } else if (XV == 7 && i < 8) {
      const tl8 r = tl_leaf4v2(qt, st_s, xs, i);
      lo[0] = r.lo.x; lo[1 % CT] = r.lo.y; lo[2 % CT] = r.lo.z; lo[3 % CT] = r.lo.w;
      hi[0] = r.hi.x; hi[1 % CT] = r.hi.y; hi[2 % CT] = r.hi.z; hi[3 % CT] = r.hi.w;
    } else if (NI) {
      const float4 a = XV >= 4 ? tl_leaf4v(qt, st_s, xs, i) : tl_leaf4(qt, st_s, xs, i);
      lo[0] = a.x; lo[1 % CT] = a.y; lo[2 % CT] = a.z; lo[3 % CT] = a.w;
    } else tl_leaf<CT>(qt, st_s, xs, i, lo);
    if (XV >= 7 && i < 8) {
    } else if (i + 16 < 24) {
      if (NI) {
        const float4 a = XV >= 4 ? tl_leaf4v(qt, st_s, xs, i + 16) : tl_leaf4(qt, st_s, xs, i + 16);
        hi[0] = a.x; hi[1 % CT] = a.y; hi[2 % CT] = a.z; hi[3 % CT] = a.w;
      } else tl_leaf<CT>(qt, st_s, xs, i + 16, hi);
    } else {
      #pragma unroll
      for (int p = 0; p < CT; p++) hi[p] = 0.f;
    }
    #pragma unroll
    for (int p = 0; p < CT; p++) pv[p] = __fadd_rn(lo[p], hi[p]);
    bool done = false;
    #pragma unroll
    for (int l = 0; l < 4; l++) {
      if (done) continue;
      if ((k >> l) & 1) {
        #pragma unroll
        for (int p = 0; p < CT; p++) pv[p] = __fadd_rn(st[l][p], pv[p]);
      } else {
        #pragma unroll
        for (int p = 0; p < CT; p++) st[l][p] = pv[p];
        done = true;
      }
    }
  }
  #pragma unroll
  for (int p = 0; p < CT; p++) {
    const int t = t0 + p;
    // Stored as 0.f + v, what k_combine_f produces for one chunk, so the caller may point
    // part at y and skip the combine (LFM_ENC_FUSEC). If a combine does run, its sum
    // starts at +0 and is never -0, so acc + (-0) == acc + (+0): unchanged either way.
    if (t < T) part[((size_t)t * N + row) * nchunk + chunk] = __fadd_rn(0.f, pv[p]);
  }
}

__global__ void k_tile768_q(const int4* __restrict__ src, int4* __restrict__ dst, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;   // (row, b, h)
  if (i >= N * 48) return;
  const int row = i / 48, bh = i - row * 48;
  dst[((size_t)(row >> 5) * 48 + bh) * 32 + (row & 31)] = src[i];
}
__global__ void k_tile768_s(const float* __restrict__ src, float* __restrict__ dst, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;   // (row, b)
  if (i >= N * 24) return;
  const int row = i / 24, b = i - row * 24;
  dst[((size_t)(row >> 5) * 24 + b) * 32 + (row & 31)] = src[i];
}
// q8_partial_TLe: q8_partial_T's values for 24 < nb <= 256 (K = 1792: nb = 56), one row
// per thread on 32-row tiles, CT = 4 regardless of nb. Lane L of q8_partial_T chains blocks
// L, L+32, ... (acc = fmaf(s_b, dot_b, acc)) and reduces with shfl_down o = 16..1; leaf i
// here is that chain, leaves combined in 5-bit bit-reversed order by a binary counter
// (earlier operand left): the same adds. x is staged 4 leaves at a time (blocks i + 32m,
// fp32, unpadded: warp reads are broadcasts). Positions are independent, so CT = 4 gives
// each position exactly the CT = 2 kernel's value.
// XE (LFM_ENC_XE): 0 = x through the pointer argument (the shared space is lost at the
// noinline boundary: generic LD.E); 1 = the dynamic shared array addressed by offset
// (LDS.32); 4 = same, one LDS.128 per 4 codes and position. Every position's chain still
// runs jj = 0..31 in order: bit-identical.
template <int XE>
__device__ __noinline__ float4 tle_leaf(const int4* __restrict__ qt, const float* __restrict__ sts,
                                        const float* xg, int xoff, int i, int nb) {
  extern __shared__ float xe_dyn[];
  float acc[4] = { 0.f, 0.f, 0.f, 0.f };
  if (XE) {
    for (int m = 0; i + 32 * m < nb; m++) {
      const int b = i + 32 * m;
      const int4 a0 = qt[b * 64], a1 = qt[b * 64 + 32];
      const unsigned w[8] = { (unsigned)a0.x, (unsigned)a0.y, (unsigned)a0.z, (unsigned)a0.w,
                              (unsigned)a1.x, (unsigned)a1.y, (unsigned)a1.z, (unsigned)a1.w };
      const float sb = sts[b * 32];
      const float* xb = xe_dyn + xoff + m * 128;
      float dot[4] = { 0.f, 0.f, 0.f, 0.f };
      if (XE == 4) {
        const float4* xv = reinterpret_cast<const float4*>(xb);
        #pragma unroll
        for (int q = 0; q < 8; q++) {
          const float c0 = (float)(int)(signed char)(w[q]);
          const float c1 = (float)(int)(signed char)(w[q] >> 8);
          const float c2 = (float)(int)(signed char)(w[q] >> 16);
          const float c3 = (float)(int)(signed char)(w[q] >> 24);
          #pragma unroll
          for (int p = 0; p < 4; p++) {
            const float4 u = xv[p * 8 + q];
            dot[p] = __fmaf_rn(c0, u.x, dot[p]);
            dot[p] = __fmaf_rn(c1, u.y, dot[p]);
            dot[p] = __fmaf_rn(c2, u.z, dot[p]);
            dot[p] = __fmaf_rn(c3, u.w, dot[p]);
          }
        }
      } else {
        #pragma unroll
        for (int jj = 0; jj < 32; jj++) {
          const float c = (float)(int)(signed char)(w[jj >> 2] >> (8 * (jj & 3)));
          #pragma unroll
          for (int p = 0; p < 4; p++) dot[p] = __fmaf_rn(c, xb[p * 32 + jj], dot[p]);
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
    const float* xb = xg + m * 128;
    float dot[4] = { 0.f, 0.f, 0.f, 0.f };
    #pragma unroll
    for (int jj = 0; jj < 32; jj++) {
      const float c = (float)(int)(signed char)(w[jj >> 2] >> (8 * (jj & 3)));
      #pragma unroll
      for (int p = 0; p < 4; p++) dot[p] = __fmaf_rn(c, xb[p * 32 + jj], dot[p]);
    }
    #pragma unroll
    for (int p = 0; p < 4; p++) acc[p] = __fmaf_rn(sb, dot[p], acc[p]);
  }
  return make_float4(acc[0], acc[1], acc[2], acc[3]);
}
template <int XE>
__global__ void __launch_bounds__(256, 4) q8_partial_TLe(const int8_t* __restrict__ q, const float* __restrict__ s,
                                                         const float* __restrict__ x, float* __restrict__ part,
                                                         int N, int K, int nb, int T) {
  extern __shared__ float xe[];                       // [4 leaf slots][mm][4 positions][32]
  const int mm = (nb + 31) / 32, slot = mm * 128;
  const int gbx = g_encgsw ? blockIdx.y : blockIdx.x, gby = g_encgsw ? blockIdx.x : blockIdx.y;
  const int t0 = gby * 4, tid = threadIdx.x;
  const int row = gbx * 256 + tid;
  const bool live = row < N;
  const int4* qt = reinterpret_cast<const int4*>(q) + (size_t)(row >> 5) * nb * 64 + (row & 31);
  const float* sts = s + (size_t)(row >> 5) * nb * 32 + (row & 31);
  float st[5][4], pv[4];
  #pragma unroll
  for (int k = 0; k < 32; k++) {
    if ((k & 3) == 0) {
      __syncthreads();
      for (int w = tid; w < 4 * slot; w += 256) {
        const int jj = w & 31, p = (w >> 5) & 3, m = (w >> 7) % mm, ls = w / slot;
        const int kk = k + ls;
        const int i = ((kk & 1) << 4) | ((kk & 2) << 2) | (kk & 4) | ((kk & 8) >> 2) | ((kk & 16) >> 4);
        const int b = i + 32 * m, t = t0 + p;
        xe[w] = (b < nb && t < T) ? x[(size_t)t * K + b * 32 + jj] : 0.f;
      }
      __syncthreads();
    }
    const int i = ((k & 1) << 4) | ((k & 2) << 2) | (k & 4) | ((k & 8) >> 2) | ((k & 16) >> 4);
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
    if (live) a = tle_leaf<XE>(qt, sts, xe + (k & 3) * slot, (k & 3) * slot, i, nb);
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
  for (int p = 0; p < 4; p++) { const int t = t0 + p; if (t < T) part[(size_t)t * N + row] = __fadd_rn(0.f, pv[p]); }
}
__global__ void k_tile_qg(const int4* __restrict__ src, int4* __restrict__ dst, int N, int nb) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)N * nb * 2) return;
  const int row = (int)(i / (nb * 2)), bh = (int)(i - (size_t)row * nb * 2);
  dst[((size_t)(row >> 5) * nb * 2 + bh) * 32 + (row & 31)] = src[i];
}
__global__ void k_tile_sg(const float* __restrict__ src, float* __restrict__ dst, int N, int nb) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)N * nb) return;
  const int row = (int)(i / nb), b = (int)(i - (size_t)row * nb);
  dst[((size_t)(row >> 5) * nb + b) * 32 + (row & 31)] = src[i];
}
static int g_enctle = -2;
#include <set>
static std::set<const void*> g_tiled;
struct Gemm {
  const int8_t* q; const float* s; const float* wf;   // wf != null = f32 weights
  int N, K, KC, nb, nch, R, nblk, nblk256;
  bool tl = false;                  // weights repacked to the q8_partial_TL tile layout
  bool tle = false;                 // weights repacked to the q8_partial_TLe tile layout
  float* part;                      // [T][N][nch] partials
  bool pooled;                      // part came from the process-wide pool
};
// The partial buffer is allocated ONCE and reused, grow-only. Per-GEMM
// cudaMalloc/cudaMemset/cudaFree of an 11 MB buffer, ~100 times per pass, was measured
// at 68.5 s of the encoder's 75.1 s process wall for a 129.7 s clip, against 1.63 s of
// GPU time: on Tegra the allocations are nvmap operations, so the layer loop was paying
// tens of seconds of allocator work that never shows up in a kernel timer. This is the
// same thing the backbone already does with M.part.
static float* g_part_buf = nullptr;
static size_t g_part_cap = 0;
static float* part_get(size_t nfloats) {
  if (nfloats <= g_part_cap) return g_part_buf;
  if (g_part_buf) { cudaFree(g_part_buf); g_part_buf = nullptr; g_part_cap = 0; }
  if (cudaMalloc(&g_part_buf, nfloats * 4)) {
    printf("gemm part cudaMalloc %.1f MB failed\n", nfloats * 4 / 1e6); exit(2);
  }
  g_part_cap = nfloats;
  return g_part_buf;
}
static void gemm_release(Gemm& g) { if (g.part && !g.pooled) cudaFree(g.part); g.part = nullptr; }
static void gemm_setup_f32(Gemm& g, const float* wf, int N, int K, int T) {
  g.q = nullptr; g.s = nullptr; g.wf = wf; g.N = N; g.K = K;
  g.KC = K; g.nb = K / Q8_BLK; g.nch = 1; g.R = 256; g.nblk = 0; g.nblk256 = 0;
  g.part = nullptr; g.pooled = false;                // the SGEMM writes y directly
  if (K % Q8_BLK) { printf("f32 GEMM needs K a multiple of 32 (K=%d)\n", K); exit(2); }
}
static void gemm_setup(Gemm& g, const int8_t* q, const float* s, int N, int K, int T, int h2, int CT, int bat = 0) {
  g.q = q; g.s = s; g.N = N; g.K = K; g.wf = nullptr;
  if (!q) { printf("gemm_setup: Q8 weights requested but q is null\n"); exit(2); }
  bat32_init();
  // Size KC against the arm that will ACTUALLY run. gemm_run chooses h2-vs-fp32 from
  // LFM_ENC_H2 (and the batched fp32 arm from LFM_ENC_BAT32) at launch time, and several
  // call sites pass h2=1 meaning "the h2 sizing", so gating the bound on the h2 argument
  // let the downsampler's K=4096 lin_out keep KC=4096 (nb=128) and then get a batched
  // launch refused for 67 KB of shared -- silently falling back to 1622 per-position
  // launches. When the batched arm is enabled, respect its budget regardless of h2 (the
  // h2 arm needs less shared, so a KC chosen this way is valid for both).
  const int bat_eff = bat || g_bat32;
  int KC = K;
  while (KC > 32) {
    const int nbtry = KC / Q8_BLK;
    // The batched fp32 arm stages CT positions, so its shared footprint carries the CT
    // factor -- KC has to be chosen against THAT bound, not the per-position one, or a
    // K as wide as the downsampler's lin_out (K=4096, nb=128) asks for 67 KB of shared
    // and the launch is refused.
    const size_t need = (h2 ? (size_t)CT * nbtry * 17 : (bat_eff ? (size_t)CT * nbtry * 33 : (size_t)nbtry * 33)) * 4;
    if (need <= 49152 && K % KC == 0) break;
    KC /= 2;
  }
  g.KC = KC; g.nb = KC / Q8_BLK; g.nch = K / KC;
  g.R = 256;                        // the fp32 per-position arm
  g.nblk = ((N + g.R - 1) / g.R) * g.nch;
  g.nblk256 = g.nblk;               // the batched h2 arm also runs R=256
  // LFM_ENC_TL (default 1): K = 768 tensors are repacked in place, once, to the tile
  // layout q8_partial_TL reads; every later use of the tensor goes through that kernel
  // (gemm_run refuses any other path for a tiled tensor).
  if (g_enctl == -2) g_enctl = getenv("LFM_ENC_TL") ? atoi(getenv("LFM_ENC_TL")) : 1;
  g.tl = false;
  if (g_enctl && K == 768 && g.nch == 1 && N % 32 == 0 && !g_ench2 && !g_bat32 && T > 1 && (!getenv("LFM_ENC_PT") || atoi(getenv("LFM_ENC_PT")) >= 4)) {
    if (!g_tiled.count(q)) {
      void* tmp = nullptr;
      if (rp_scratch(0), g_rpa) tmp = rp_scratch((size_t)N * K);
      else if (cudaMalloc(&tmp, (size_t)N * K)) { printf("tile768 scratch\n"); exit(2); }
      k_tile768_q<<<(N * 48 + 255) / 256, 256>>>(reinterpret_cast<const int4*>(q), (int4*)tmp, N);
      if (g_rpa == 2) cudaMemcpyAsync((void*)q, tmp, (size_t)N * K, cudaMemcpyDeviceToDevice, 0);
      else cudaMemcpy((void*)q, tmp, (size_t)N * K, cudaMemcpyDeviceToDevice);
      k_tile768_s<<<(N * 24 + 255) / 256, 256>>>(s, (float*)tmp, N);
      if (g_rpa == 2) cudaMemcpyAsync((void*)s, tmp, (size_t)N * 24 * 4, cudaMemcpyDeviceToDevice, 0);
      else cudaMemcpy((void*)s, tmp, (size_t)N * 24 * 4, cudaMemcpyDeviceToDevice);
      if (g_rpa != 2 && cudaDeviceSynchronize() != cudaSuccess) { printf("tile768 repack failed\n"); exit(2); }
      if (!g_rpa) cudaFree(tmp);
      g_tiled.insert(q);
    }
    g.tl = true;
  }
  // LFM_ENC_TLE (default 1): 24 < nb <= 62 single-chunk shapes (the ones q8_partial_T runs
  // at CT = 2) go to q8_partial_TLe on generic 32-row tiles, repacked once in place.
  if (g_enctle == -2) g_enctle = getenv("LFM_ENC_TLE") ? atoi(getenv("LFM_ENC_TLE")) : 1;
  g.tle = false;
  if (!g.tl && g_enctle && g.nch == 1 && g.nb > 24 && g.nb <= (getenv("LFM_ENC_TLEMAX") ? atoi(getenv("LFM_ENC_TLEMAX")) : 256) && N % 32 == 0 &&
      !g_ench2 && !g_bat32 && T > 1 && (!getenv("LFM_ENC_PT") || atoi(getenv("LFM_ENC_PT")) >= 4)) {
    if (!g_tiled.count(q)) {
      const int nbg = g.nb;
      void* tmp = nullptr;
      if (rp_scratch(0), g_rpa) tmp = rp_scratch((size_t)N * K);
      else if (cudaMalloc(&tmp, (size_t)N * K)) { printf("tile scratch\n"); exit(2); }
      k_tile_qg<<<(unsigned)(((size_t)N * nbg * 2 + 255) / 256), 256>>>(reinterpret_cast<const int4*>(q), (int4*)tmp, N, nbg);
      if (g_rpa == 2) cudaMemcpyAsync((void*)q, tmp, (size_t)N * K, cudaMemcpyDeviceToDevice, 0);
      else cudaMemcpy((void*)q, tmp, (size_t)N * K, cudaMemcpyDeviceToDevice);
      k_tile_sg<<<(unsigned)(((size_t)N * nbg + 255) / 256), 256>>>(s, (float*)tmp, N, nbg);
      if (g_rpa == 2) cudaMemcpyAsync((void*)s, tmp, (size_t)N * nbg * 4, cudaMemcpyDeviceToDevice, 0);
      else cudaMemcpy((void*)s, tmp, (size_t)N * nbg * 4, cudaMemcpyDeviceToDevice);
      if (g_rpa != 2 && cudaDeviceSynchronize() != cudaSuccess) { printf("tile repack failed\n"); exit(2); }
      if (!g_rpa) cudaFree(tmp);
      g_tiled.insert(q);
    }
    g.tle = true;
  }
  const size_t need_f = (size_t)T * N * g.nch;
  g.part = part_get(need_f); g.pooled = true;
  if (g_poison && cudaMemset(g.part, 0xff, need_f * 4)) { printf("memset part\n"); exit(2); }
  // NOT zeroed on the shipping path, deliberately, and this is a measured fact rather
  // than a style choice:
  // zeroing g.part made the encoder WORSE (layer_01 vs the C reference 2.86e-06 -> 2.12e-04,
  // and the run-to-run spread rose from ~1e-4 to 1.7e-3 energy), i.e. the slots the
  // combine kernel reads are all written by this GEMM's launches and zeroing them only
  // destroys what the previous occupant of that memory put there. The encoder's residual
  // run-to-run spread therefore does NOT come from g.part (see doc section 29); the
  // run-to-run spread itself was fixed in section 34 (the k_rmsnorm_f32 barrier).
}
// y[T][N] = x[T][K] @ W^T. fp32 path = per-position q8_partial<float> (the fastest
// measured fp32 arm, run #309); h2 path = batched q8_prefill_h2<bf16,ARM10,RB2>.
static void gemm_run(Gemm& g, const float* x, float* y, int T, bf16* xb) {
  const int TPB = 256;
  // LFM_ENC_FUSEC (default 1): with one K-chunk the thread-per-row kernels (TL, TLe) write
  // 0.f + v straight into y and the k_combine_f launch is skipped. Not when y overlaps x:
  // a block would overwrite rows other blocks are still staging.
  static const int fusec = getenv("LFM_ENC_FUSEC") ? atoi(getenv("LFM_ENC_FUSEC")) : 1;
  const bool disj = (const char*)(y + (size_t)T * g.N) <= (const char*)x || (const char*)(x + (size_t)T * g.K) <= (const char*)y;
  const bool fuse = fusec && g.nch == 1 && disj;
  if (g.wf) {                                        // f32-weight path (no half2 arm)
    static const int sg8 = getenv("LFM_ENC_SG8") ? atoi(getenv("LFM_ENC_SG8")) : 1;
    if (sg8 && g.K % 16 == 0) {
      k_sgemm8<<<dim3((g.N + 127) / 128, (T + 127) / 128), 256>>>(x, g.wf, y, T, g.N, g.K);
    } else {
    const dim3 grid((g.N + 63) / 64, (T + 63) / 64), block(16, 16);
    k_sgemm_f32<<<grid, block>>>(x, g.wf, y, T, g.N, g.K);
    }
    return;
  }
  if (g_ench2 && xb) {
    k_f2bf<<<(unsigned)((size_t)T * g.K / 256 + 1), 256>>>(x, xb, (size_t)T * g.K);
    const size_t shm = (size_t)4 * g.nb * 17 * 4;
    q8_prefill_h2<bf16, 4, true, 10, 2, 2><<<g.nblk256, TPB, shm>>>(g.q, g.s, xb, g.part,
        g.N, g.K, g.KC, 256, g.nb, T);
    cudaError_t e = cudaGetLastError();                      // a refused launch is silent
    if (e) { printf("gemm h2 launch refused (%s) at N=%d K=%d -- falling back to fp32\n",
                    cudaGetErrorString(e), g.N, g.K); }
    // k_combine_f indexes rows flat, so T*N rows with nchunk cols works in one launch
    else { k_combine_f<<<(unsigned)((T * g.N + 255) / 256), 256>>>(g.part, y, T * g.N, g.nch); return; }
  }
  const size_t shm = (size_t)g.nb * 33 * 4;
  // LFM_ENC_BAT32=1: the batched fp32 arm. q8_partial is launched ONCE PER POSITION in
  // the loop below, which at Tt=1622 and ~100 Q8 GEMMs per pass is ~162k launches -- the
  // encoder's dominant wall cost (61.5 s of host time against 1.68 s of GPU time, §37).
  // q8_prefill_h2<float, CT=4, H2=false, ARM=5, RB=2, GRP=2> was measured bit-identical
  // to this per-position arm on the real encoder activations in run #309 at 4.6 vs
  // 10.4 GMAC/s: GPU time roughly doubles, host launches drop 4x, and it is the same
  // arithmetic order, so the accuracy gate must stay green -- that is the acceptance bar
  // for this arm, not a timing number. Falls back to the loop below if the launch is
  // refused (shared budget, R not dividing the shape), so the shipping path never
  // depends on it succeeding.
  if (g_bat32 && T > 4) {
    const int RB = 2, GRP = 2, Rb = 64;                       // (256/32)/RB = 4 rows per warp -> R=64
    const size_t shmB = (size_t)4 * g.nb * 33 * 4;
    const int nblkB = ((g.N + Rb - 1) / Rb) * g.nch;
    q8_prefill_h2<float, 4, false, 5, RB, GRP><<<nblkB, TPB, shmB>>>(
        g.q, g.s, x, g.part, g.N, g.K, g.KC, Rb, g.nb, T);
    const cudaError_t be = cudaGetLastError();
    if (be == cudaSuccess) {
      k_combine_f<<<(unsigned)((T * g.N + 255) / 256), 256>>>(g.part, y, T * g.N, g.nch);
      const cudaError_t ce = cudaGetLastError();
      if (ce) { printf("combine (bat32) refused: %s\n", cudaGetErrorString(ce)); exit(3); }
      return;
    }
    printf("bat32 launch refused at N=%d K=%d KC=%d nb=%d nch=%d shm=%zu: %s -- per-position loop\n",
           g.N, g.K, g.KC, g.nb, g.nch, shmB, cudaGetErrorString(be));
  }
  if (g.tle) {
    const size_t shm = (size_t)4 * ((g.nb + 31) / 32) * 128 * 4;
    for (int t0 = 0; t0 < T; t0 += 65535 * 4) {
      const int Tl = (T - t0 < 65535 * 4) ? T - t0 : 65535 * 4;
      static const int xe = getenv("LFM_ENC_XE") ? atoi(getenv("LFM_ENC_XE")) : 4;
      static const int egsw = getenv("LFM_ENC_GSW") ? atoi(getenv("LFM_ENC_GSW")) : 1;
      static bool egsw_set = false;
      if (!egsw_set) { cudaMemcpyToSymbol(g_encgsw, &egsw, sizeof egsw); egsw_set = true; }
      const dim3 ge = egsw ? dim3((Tl + 3) / 4, (g.N + 255) / 256) : dim3((g.N + 255) / 256, (Tl + 3) / 4);
      float* const po = (fuse ? y : g.part) + (size_t)t0 * g.N;
      if (xe == 4)      q8_partial_TLe<4><<<ge, 256, shm>>>(g.q, g.s, x + (size_t)t0 * g.K, po, g.N, g.K, g.nb, Tl);
      else if (xe == 1) q8_partial_TLe<1><<<ge, 256, shm>>>(g.q, g.s, x + (size_t)t0 * g.K, po, g.N, g.K, g.nb, Tl);
      else              q8_partial_TLe<0><<<ge, 256, shm>>>(g.q, g.s, x + (size_t)t0 * g.K, po, g.N, g.K, g.nb, Tl);
    }
    if (!fuse) k_combine_f<<<(unsigned)(((size_t)T * g.N + 255) / 256), 256>>>(g.part, y, T * g.N, 1);
    const cudaError_t ce = cudaGetLastError();
    if (ce) { printf("q8_partial_TLe launch failed: %s\n", cudaGetErrorString(ce)); exit(3); }
    return;
  }
  if (g_encpt == -2) g_encpt = getenv("LFM_ENC_PT") ? atoi(getenv("LFM_ENC_PT")) : 4;
  if (g_encpt && T > 1 && g.R == 256) {
    const size_t per = (size_t)g.nb * 33 * 4;         // one position's staged x
    int ct = g_encpt >= 4 ? 4 : 2;
    while (ct > 1 && ct * per > 16384) ct /= 2;
    if (ct >= 2) {
      for (int t0 = 0; t0 < T; t0 += 65535 * ct) {    // gridDim.y <= 65535
        const int Tl = (T - t0 < 65535 * ct) ? T - t0 : 65535 * ct;
        const dim3 grid(g.nblk, (Tl + ct - 1) / ct);
        if (g_encts == -2) g_encts = getenv("LFM_ENC_TS") ? atoi(getenv("LFM_ENC_TS")) : 1;
        if (g.tl && !(ct == 4 && g.R == TPB)) { printf("tiled tensor needs q8_partial_TL\n"); exit(2); }
        static const int xv = getenv("LFM_ENC_XV") ? atoi(getenv("LFM_ENC_XV")) : 8;
        float* const tlo = (fuse && g.tl) ? y + (size_t)t0 * g.N : g.part + (size_t)t0 * g.N * g.nch;
        static const int egsw2 = getenv("LFM_ENC_GSW") ? atoi(getenv("LFM_ENC_GSW")) : 1;
        static bool egsw2_set = false;
        if (!egsw2_set) { cudaMemcpyToSymbol(g_encgsw, &egsw2, sizeof egsw2); egsw2_set = true; }
        if (g.tl && g_tlni && xv == 8)
          q8_partial_TL<4, true, 8><<<egsw2 ? dim3(grid.y, grid.x) : grid, TPB, 4 * per>>>(g.q, g.s, x + (size_t)t0 * g.K,
                         tlo, g.N, g.K, g.KC, g.R, g.nb, Tl);
        else if (g.tl && g_tlni && xv == 7)
          q8_partial_TL<4, true, 7><<<egsw2 ? dim3(grid.y, grid.x) : grid, TPB, 4 * per>>>(g.q, g.s, x + (size_t)t0 * g.K,
                         tlo, g.N, g.K, g.KC, g.R, g.nb, Tl);
        else if (g.tl && g_tlni && xv == 4)
          q8_partial_TL<4, true, 4><<<egsw2 ? dim3(grid.y, grid.x) : grid, TPB, 4 * per>>>(g.q, g.s, x + (size_t)t0 * g.K,
                         tlo, g.N, g.K, g.KC, g.R, g.nb, Tl);
        else if (g.tl && g_tlni)
          q8_partial_TL<4, true><<<grid, TPB, 4 * per>>>(g.q, g.s, x + (size_t)t0 * g.K,
                         tlo, g.N, g.K, g.KC, g.R, g.nb, Tl);
        else if (g.tl)   // 4 blocks/SM: 64 regs with ~580 B of spills beats 2 blocks at 128 regs (17.4 vs 15.0 s)
          q8_partial_TL<4><<<grid, TPB, 4 * per>>>(g.q, g.s, x + (size_t)t0 * g.K,
                         tlo, g.N, g.K, g.KC, g.R, g.nb, Tl);
        else if (ct == 4 && g_encts && g.nb == 24 && (g.R / 8) % 4 == 0)
          q8_partial_TS<4, 4, 3><<<grid, TPB, 4 * per>>>(g.q, g.s, x + (size_t)t0 * g.K,
                         g.part + (size_t)t0 * g.N * g.nch, g.N, g.K, g.KC, g.R, g.nb, Tl);
        else if (ct == 4) q8_partial_T<4><<<grid, TPB, 4 * per>>>(g.q, g.s, x + (size_t)t0 * g.K,
                         g.part + (size_t)t0 * g.N * g.nch, g.N, g.K, g.KC, g.R, g.nb, Tl);
        else         q8_partial_T<2><<<grid, TPB, 2 * per>>>(g.q, g.s, x + (size_t)t0 * g.K,
                         g.part + (size_t)t0 * g.N * g.nch, g.N, g.K, g.KC, g.R, g.nb, Tl);
        const cudaError_t le = cudaGetLastError();
        if (le) { printf("q8_partial_T<%d> launch REFUSED at N=%d K=%d KC=%d nb=%d: %s\n",
                         ct, g.N, g.K, g.KC, g.nb, cudaGetErrorString(le)); exit(3); }
      }
      chk(g_enctime >= 3 ? gemm_tag("gemm fp32 T-wide", g.N, g.K, T, ct, (int)g.tl, g.nblk, TPB)
                         : "gemm fp32 T-wide");
      if (!(fuse && g.tl)) k_combine_f<<<(unsigned)((T * g.N + 255) / 256), 256>>>(g.part, y, T * g.N, g.nch);
      chk("gemm combine");
      return;
    }
  }
  for (int t = 0; t < T; t++) {
    q8_partial<float, true, false, 1, 0><<<g.nblk, TPB, shm>>>(g.q, g.s, x + (size_t)t * g.K,
        g.part + (size_t)t * g.N * g.nch, g.N, g.K, g.KC, g.R, g.nb);
    // Read the error IMMEDIATELY (run #307's lesson): a refused launch leaves that
    // position's partial slots holding whatever the previous occupant of that device
    // memory had, which is exactly the self-referential, two-stable-state, history
    // dependent signature of the section 29 defect. Silent here = unfalsifiable there.
    const cudaError_t le = cudaGetLastError();
    if (le) { printf("q8_partial launch REFUSED at N=%d K=%d KC=%d R=%d nb=%d nblk=%d shm=%zu t=%d: %s\n",
                     g.N, g.K, g.KC, g.R, g.nb, g.nblk, shm, t, cudaGetErrorString(le)); exit(3); }
  }
  chk("gemm fp32 loop");
  k_combine_f<<<(unsigned)((T * g.N + 255) / 256), 256>>>(g.part, y, T * g.N, g.nch);
  chk("gemm combine");
}

// ---------------------------------------------------------------- layer kernels
// All of these copy lfm3_enc.c's arithmetic, including its fp64 sum-of-squares in
// rmsnorm (the C reference accumulates the variance in double, so the GPU does too).
// RMSNorm over one row, in place. THE BARRIER BELOW IS LOAD-BEARING.
// Every thread computes the whole-row fp64 sum of squares itself (ascending i, the C
// reference's order), and then all threads write the row back IN PLACE. Without a
// barrier between the sum phase and the write phase, an ahead warp scales r[i] while a
// lagging warp is still inside its sum loop, so the lagging warp's sum of squares is
// built from a mix of pre-scale and post-scale values, its `inv` is wrong, and the row
// ends up with elements scaled by several different inversions. blockDim is 256 = 8
// warps, D is 768, so the sum loop is long enough for the interleaving to happen.
// That is the root cause of the encoder's run-to-run nonreproducibility that ran
// through six runs of the defect hunt: the same binary, same input, disagreed with
// itself at Tt=1000 by max abs 2.2e-01 in layer_00 (a CONV block, which is why the
// attention-race fix of run #313 did not touch it), in a sparse row pattern (rows
// 40,44,68,72-78,96,... of 1000) across all 768 channels, with rows 0-39 bit-identical
// -- collision-dependent, hence sparse, hence amplitude continuous, not bimodal, in the
// 1.4e-06 .. 2.9e-03 range the 4 s fixture showed, and rarer at Tt=50 simply because
// 50 blocks collide less often than 1000. ds_out was always bit-identical because the
// downsampler has no in-place norm.
// k_headnorm (96 threads) and k_layernorm_f32 (256 threads) do not have this bug: the
// former has its barrier, the latter writes to a separate output so x is read-only.
// Row sums of squares, one row per LANE: each row's serial fp64 sum is exactly the norm
// kernels' thread-0 loop (same ascending order, same exact double products), but one fp64
// warp instruction now serves 32 rows instead of one (fp64 is 1/32 rate on this GPU).
__global__ void k_sumsq_rows(const float* __restrict__ x, double* __restrict__ ss, int nrows, int D) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= nrows) return;
  const float* r = x + (size_t)row * D;
  double v = 0;
  for (int i = 0; i < D; i++) v += (double)r[i] * r[i];
  ss[row] = v;
}
static double* g_ss = nullptr;
static const int g_encss = getenv("LFM_ENC_SS") ? atoi(getenv("LFM_ENC_SS")) : 1;
static const double* sumsq_rows(const float* x, int nrows, int D) {
  if (!g_encss) return nullptr;
  static int cap = 0;
  if (nrows > cap) { if (g_ss) cudaFree(g_ss); cap = nrows; if (cudaMalloc(&g_ss, (size_t)cap * 8)) { printf("ss alloc\n"); exit(3); } }
  k_sumsq_rows<<<(nrows + 127) / 128, 128>>>(x, g_ss, nrows, D);
  return g_ss;
}
__global__ void k_rmsnorm_f32(float* __restrict__ x, const float* __restrict__ w, int D,
                              const double* __restrict__ ss = nullptr) {
  const int t = blockIdx.x;
  float* r = x + (size_t)t * D;
  // Every thread used to run this same serial fp64 sum (256x redundant fp64 on a GPU with
  // 1/32-rate fp64: 266 ms per call at Tt=1622, 9.3 s of the encoder). Thread 0 now runs
  // it once, in the same order, and shares it: the value every thread used is unchanged.
  __shared__ double sv;
  if (ss) { if (threadIdx.x == 0) sv = ss[t]; }
  else if (threadIdx.x == 0) {
    double v = 0;
    for (int i = 0; i < D; i++) v += (double)r[i] * r[i];
    sv = v;
  }
  __syncthreads();                      // nobody writes until the sum is done (and shared)
  const double v = sv;
  const float inv = 1.0f / sqrtf((float)(v / D) + EEPS);
  for (int i = threadIdx.x; i < D; i += blockDim.x) r[i] = __fmul_rn(w[i], __fmul_rn(r[i], inv));
}

// in_proj output [T,3C] -> B*X into bx, C gate into cg (the C reference's split)
__global__ void k_bcx_split(const float* __restrict__ bc, float* __restrict__ bx,
                            float* __restrict__ cg, int T, int C) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= T * C) return;
  const int t = i / C, c = i % C;
  const float* row = bc + (size_t)t * 3 * C;
  bx[i] = __fmul_rn(row[c], row[2 * C + c]);
  cg[i] = row[C + c];
}

// causal depthwise conv width 3 over time, then the C gate multiply:
//   y[t,c] = cg[t,c] * sum_k cw[c,k] * bx[t-2+k,c], t-2+k < 0 skipped
__global__ void k_conv3_causal(const float* __restrict__ bx, const float* __restrict__ cw,
                               const float* __restrict__ cg, float* __restrict__ y, int T, int C) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= T * C) return;
  const int t = i / C, c = i % C;
  float acc = 0.f;
  #pragma unroll
  for (int k = 0; k < 3; k++) {
    const int tt = t - 2 + k;
    if (tt < 0) continue;
    acc = __fadd_rn(acc, __fmul_rn(cw[(size_t)c * 3 + k], bx[(size_t)tt * C + c]));
  }
  y[i] = __fmul_rn(cg[i], acc);
}

__global__ void k_silu_mul(float* __restrict__ g, const float* __restrict__ u, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float v = g[i];
  g[i] = __fmul_rn(__fdiv_rn(v, __fadd_rn(1.f, expf(-v))), u[i]);
}

__global__ void k_add(float* __restrict__ a, const float* __restrict__ b, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = __fadd_rn(a[i], b[i]);
}

// per-head rmsnorm over HD dims: one block per (t, head) row, matching
// rmsnorm(q + (t*NH+h)*HD, 1, HD, qn) in lfm3_enc.c -- fp64 sum of squares, summed
// SERIALLY in ascending d. (A shared ps[tid&31] tree was tried first: with blockDim 96
// eight to twelve threads alias each slot, last writer wins, and 7/8 of the partials
// vanish. The gate saw it as layer_02 rms rel 40 vs the C reference.)
__global__ void k_headnorm(float* __restrict__ x, const float* __restrict__ w,
                           const double* __restrict__ ss = nullptr) {
  const int row = blockIdx.x;
  float* r = x + (size_t)row * EHD;
  __shared__ double v;
  if (ss) { if (threadIdx.x == 0) v = ss[row]; }
  else if (threadIdx.x == 0) {
    double acc = 0;
    for (int i = 0; i < EHD; i++) acc += (double)r[i] * r[i];
    v = acc;
  }
  __syncthreads();
  const float inv = 1.0f / sqrtf((float)(v / EHD) + EEPS);
  for (int i = threadIdx.x; i < EHD; i += blockDim.x) r[i] = __fmul_rn(w[i], __fmul_rn(r[i], inv));
}

// RoPE rotate-half over all HD dims, pairs (d, d+HD/2), theta 1e6, position = t.
// One block per (t, d) pair-column, covering every q head and every kv head.
__global__ void k_rope(float* __restrict__ q, float* __restrict__ k, int T) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= T * (EHD / 2)) return;
  const int t = i / (EHD / 2), d = i % (EHD / 2);
  const double inv = pow((double)ETHETA, -(double)(2 * d) / EHD);
  const double ang = (double)t * inv;
  const float c = (float)cos(ang), sn = (float)sin(ang);
  for (int h = 0; h < ENH; h++) {
    float* v = q + ((size_t)t * ENH + h) * EHD;
    const float a = v[d], b = v[d + EHD / 2];
    v[d] = __fadd_rn(__fmul_rn(a, c), -__fmul_rn(b, sn));
    v[d + EHD / 2] = __fadd_rn(__fmul_rn(b, c), __fmul_rn(a, sn));
  }
  for (int h = 0; h < ENKV; h++) {
    float* v = k + ((size_t)t * ENKV + h) * EHD;
    const float a = v[d], b = v[d + EHD / 2];
    v[d] = __fadd_rn(__fmul_rn(a, c), -__fmul_rn(b, sn));
    v[d + EHD / 2] = __fadd_rn(__fmul_rn(b, c), __fmul_rn(a, sn));
  }
}

// Full causal attention, one block per (query position i, q head hh); thread d owns
// out[d] in a register, so the accumulation over j runs in ascending j exactly as
// lfm3_enc.c does it: s = sum_d q[d]*k[j][d]; s*scale; max; expf(s-mx) ascending; den;
// p = sc[j]/den (a division, not a reciprocal multiply); o[d] += p*v[j][d].
// T-any-length prefill attention. There is deliberately NO score buffer: the scores
// are recomputed from q/k on each of the three passes (max, denominator, weighted sum).
// The earlier version staged sc[1024] in shared, which capped Tt at 1024 audio tokens
// = 81.9 s of speech (the encoder refused the 129.7 s gate fixture at Tt=1622). Scoring
// is deterministic, so the recomputed value is the SAME bits as the staged one, and
// every loop keeps the C reference's ascending-j order and the same per-pass sequence
// (thread 0 reduces the 96 products ascending, barrier, everyone reads). Measured
// cost of the recompute: see the 80 s / 129.7 s rows in the port doc (the QK dot runs
// 3x instead of 1x); the encoder stays well inside its realtime budget.
//
// The exp'd weight is written by THREAD 0 ONLY, with a barrier before every read and
// before the next write. Having all 96 threads redundantly write the value and read it
// back with no barrier is a race: 96 threads is three warps, warps are not synchronized
// with each other, and a warp that ran ahead divided by the PRE-EXP score. That showed
// up as ao max abs 1.0e-01 (rms 6.8e-3) against an fp64 referee while the scores
// themselves matched fp64 to 3e-6. With one shared slot the same discipline is now
// load-bearing for every j, so the barriers are written to be obviously correct:
// barrier before thread 0 writes, barrier before anyone reads.
__global__ void k_attn_prefill(const float* __restrict__ q, const float* __restrict__ k,
                               const float* __restrict__ v, float* __restrict__ ao,
                               int T, float scale, float* dbg, int ts, int hs, int win) {
  const int i = blockIdx.x, hh = blockIdx.y, kv = hh / (ENH / ENKV);
  // win = 32 for the encoder's sliding-window attention layers, 0 = full causal. The
  // checkpoint's encoder sub-config says encoder.sliding_window = 32 and HF builds the
  // mask with create_sliding_window_causal_mask(sliding_window=32); see the section 39
  // note at the layer loop for the measurement and for why the LAST layer stays full.
  const int jlo = (i >= win) ? (i - win + 1) : 0;
  __shared__ float prod[EHD];
  __shared__ float sc;
  const int d = threadIdx.x;
  const float qi = q[((size_t)i * ENH + hh) * EHD + d];
  // one score, in the C reference's order: 96 products, summed ascending by thread 0,
  // then scaled. dbg captures pass A's scores for the probe layer.
 #define QKSCORE(j) do { \
    prod[d] = __fmul_rn(qi, k[((size_t)(j) * ENKV + kv) * EHD + d]); \
    __syncthreads(); \
    if (d == 0) { \
      float s2 = 0.f; \
      for (int dd = 0; dd < EHD; dd++) s2 = __fadd_rn(s2, prod[dd]); \
      sc = __fmul_rn(s2, scale); \
      if (dbg && i == ts && hh == hs) dbg[j] = sc; \
    } \
    __syncthreads(); \
  } while (0)
  float mx = -INFINITY;
  for (int j = jlo; j <= i; j++) { QKSCORE(j); if (sc > mx) mx = sc; }
  float den = 0.f;
  for (int j = jlo; j <= i; j++) {
    QKSCORE(j);
    __syncthreads();                      // the exp write below must not race a read
    if (d == 0) sc = expf(__fadd_rn(sc, -mx));
    __syncthreads();
    den = __fadd_rn(den, sc);
  }
  float o = 0.f;
  for (int j = jlo; j <= i; j++) {
    QKSCORE(j);
    __syncthreads();
    if (d == 0) sc = expf(__fadd_rn(sc, -mx));
    __syncthreads();
    o = __fadd_rn(o, __fmul_rn(__fdiv_rn(sc, den), v[((size_t)j * ENKV + kv) * EHD + d]));
  }
#undef QKSCORE
  ao[((size_t)i * ENH + hh) * EHD + d] = o;
}

// k_attn_prefill2: k_attn_prefill computing each score ONCE, one score per thread. Thread
// t forms score j with the same 96 products summed in ascending dd order (the reference's
// thread-0 loop), then scales; scores live in dynamic shared memory. Thread 0 takes the
// max and the denominator in ascending j exactly as before; exps are formed per score with
// the same expf(sc - mx); the output loop per d is unchanged. Bit-identical.
__global__ void k_attn_prefill2(const float* __restrict__ q, const float* __restrict__ k,
                                const float* __restrict__ v, float* __restrict__ ao,
                                int T, float scale, float* dbg, int ts, int hs, int win) {
  extern __shared__ float es[];                  // [i - jlo + 1]
  __shared__ float qs[EHD];
  __shared__ float mxs, dens;
  const int i = blockIdx.x, hh = blockIdx.y, kv = hh / (ENH / ENKV);
  const int jlo = (i >= win) ? (i - win + 1) : 0;
  const int n = i - jlo + 1;
  const int d = threadIdx.x;
  qs[d] = q[((size_t)i * ENH + hh) * EHD + d];
  __syncthreads();
  for (int jj = d; jj < n; jj += EHD) {
    const float* kr = k + ((size_t)(jlo + jj) * ENKV + kv) * EHD;
    float s2 = 0.f;
    for (int dd = 0; dd < EHD; dd++) s2 = __fadd_rn(s2, __fmul_rn(qs[dd], kr[dd]));
    const float sc = __fmul_rn(s2, scale);
    es[jj] = sc;
    if (dbg && i == ts && hh == hs) dbg[jlo + jj] = sc;
  }
  __syncthreads();
  if (d == 0) { float mx = -INFINITY; for (int jj = 0; jj < n; jj++) if (es[jj] > mx) mx = es[jj]; mxs = mx; }
  __syncthreads();
  const float mx = mxs;
  for (int jj = d; jj < n; jj += EHD) es[jj] = expf(__fadd_rn(es[jj], -mx));
  __syncthreads();
  if (d == 0) { float den = 0.f; for (int jj = 0; jj < n; jj++) den = __fadd_rn(den, es[jj]); dens = den; }
  __syncthreads();
  const float den = dens;
  float o = 0.f;
  for (int jj = 0; jj < n; jj++)
    o = __fadd_rn(o, __fmul_rn(__fdiv_rn(es[jj], den), v[((size_t)(jlo + jj) * ENKV + kv) * EHD + d]));
  ao[((size_t)i * ENH + hh) * EHD + d] = o;
}

// LayerNorm(768) for the projector, mean and variance accumulated in double over
// ascending i exactly as lfm3_enc.c does. Serial in thread 0: the shared ps[tid&31]
// tree aliased 8 threads per slot at blockDim 256 and corrupted both moments.
// k_moments_rows: the same two ascending double loops, one row per LANE (the #75 fix:
// a 1-lane fp64 loop holds a full warp slot of the 1/32-rate DP pipe).
__global__ void k_moments_rows(const float* __restrict__ x, double* __restrict__ mv, int nrows) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= nrows) return;
  const float* r = x + (size_t)row * EHID;
  double mu = 0, var = 0;
  for (int i = 0; i < EHID; i++) mu += (double)r[i];
  mu /= EHID;
  for (int i = 0; i < EHID; i++) { const double dv = (double)r[i] - mu; var += dv * dv; }
  mv[2 * row] = mu; mv[2 * row + 1] = var / EHID;
}
__global__ void k_layernorm_f32(const float* __restrict__ x, const float* __restrict__ g,
                               const float* __restrict__ b, float* __restrict__ y,
                               const double* __restrict__ mv = nullptr) {
  const int t = blockIdx.x;
  const float* r = x + (size_t)t * EHID;
  __shared__ double sh[2];
  if (mv) { if (threadIdx.x == 0) { sh[0] = mv[2 * t]; sh[1] = mv[2 * t + 1]; } }
  else if (threadIdx.x == 0) {
    double mu = 0, var = 0;
    for (int i = 0; i < EHID; i++) mu += (double)r[i];
    mu /= EHID;
    for (int i = 0; i < EHID; i++) { const double dv = (double)r[i] - mu; var += dv * dv; }
    sh[0] = mu; sh[1] = var / EHID;
  }
  __syncthreads();
  const float muf = (float)sh[0];
  const float inv = 1.0f / sqrtf((float)sh[1] + 1e-5f);
  for (int i = threadIdx.x; i < EHID; i += blockDim.x)
    y[(size_t)t * EHID + i] = __fadd_rn(__fmul_rn(__fmul_rn(__fadd_rn(r[i], -muf), inv), g[i]), b[i]);
}

// GELU tanh-exact form used by the projector: 0.5*x*(1+erf(x/sqrt(2)))
__global__ void k_gelu_erf(float* __restrict__ x, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { const float v = x[i]; x[i] = __fmul_rn(__fmul_rn(0.5f, v), __fadd_rn(1.f, erff(__fdiv_rn(v, 1.41421356237f)))); }
}

static void gemm_to(Ent* e, const float* x, float* y, int T, bf16* xb, const char* what) {
  if (e->dtype == 2) {
    Gemm g; gemm_setup_f32(g, e->df, (int)e->ne[0], (int)e->ne[1], T);
    gemm_run(g, x, y, T, nullptr); chk(what); return;
  }
  if (e->dtype != 1) { printf("%s: unsupported dtype %d\n", what, e->dtype); exit(2); }
  Gemm g; gemm_setup(g, e->dq, e->ds, (int)e->ne[0], (int)e->ne[1], T, 1, 4);
  gemm_run(g, x, y, T, xb); chk(what);
}

// ---------------------------------------------------------------- helpers
// Per-request device allocations. In a one-shot process these are cudaMalloc'd once and
// reclaimed at exit; a resident service that repeated them per turn both leaked (fixed by
// tracking and freeing) and paid the allocator every turn: a served request costs ~0.48 s of
// wall for ~0.12 s of GPU, and the difference is cudaMalloc/cudaFree + the zero fill over the
// pass's activation set. So the buffers are POOLED by call index and reused across passes
// (grow-only: a slot is reallocated only when the request needs more bytes than it holds --
// the same code path with a shorter mel reuses the bigger buffer). LFM_ENC_POOL=0 restores the
// free-after-pass behaviour (the A/B arm).
struct ReqAlloc { void* p; size_t bytes; };
static std::vector<ReqAlloc> g_pool;
static std::vector<void*> g_reqallocs;   // LFM_ENC_POOL=0 arm: freed at the end of each pass
static size_t g_slot = 0;
static int g_pool_on = -1;
static void* req_alloc(size_t bytes, int zero, const char* what) {
  if (g_pool_on < 0) g_pool_on = getenv("LFM_ENC_POOL") ? atoi(getenv("LFM_ENC_POOL")) : 1;
  void* p = nullptr;
  if (g_pool_on && g_slot < g_pool.size() && g_pool[g_slot].bytes >= bytes) {
    p = g_pool[g_slot].p;                       // reuse: same slot, big enough
  } else {
    if (cudaMalloc(&p, bytes)) { printf("OOM (%s, %.1f MB)\n", what, bytes / 1e6); exit(2); }
    if (g_pool_on) {
      if (g_slot < g_pool.size()) { cudaFree(g_pool[g_slot].p); g_pool[g_slot] = { p, bytes }; }
      else g_pool.push_back({ p, bytes });
    } else g_reqallocs.push_back(p);
  }
  g_slot++;
  // cudaMalloc hands back stale device memory -- and so does a reused slot. Reading any element
  // before it is written is therefore run-history dependent: four back-to-back runs of the same
  // binary agreed with each other to the bit and disagreed with a run that followed a fresh
  // build (layer_01 max abs 2.9e-06 vs 9.1e-03). Zero every allocation, and LFM_ENC_POISON=1
  // fills with 0xff bytes = NaN so any such read turns the stage's output non-finite and the
  // gate's finite check fails LOUDLY.
  if (zero && cudaMemset(p, g_poison ? 0xff : 0, bytes)) { printf("memset failed (%s)\n", what); exit(2); }
  return p;
}
static float* dalloc(size_t n, const char* what) {
  if (g_poison < 0) g_poison = getenv("LFM_ENC_POISON") ? atoi(getenv("LFM_ENC_POISON")) : 0;
  return (float*)req_alloc(n * 4, 1, what);
}
// dalloc_w: for buffers that the next kernel writes in full before anything reads them
// (conv_in -> x1, conv_mid -> x2), the zero fill is skipped (LFM_ENC_NZ, default 1);
// LFM_ENC_POISON=1 still fills them with NaN, which is how full coverage is checked.
static float* dalloc_w(size_t n, const char* what) {
  static const int nz = getenv("LFM_ENC_NZ") ? atoi(getenv("LFM_ENC_NZ")) : 1;
  if (g_poison < 0) g_poison = getenv("LFM_ENC_POISON") ? atoi(getenv("LFM_ENC_POISON")) : 0;
  // Both arms take a pool slot, so the slot sequence does not depend on LFM_ENC_NZ.
  if (!nz || g_poison) return (float*)req_alloc(n * 4, 1, what);
  return (float*)req_alloc(n * 4, 0, what);
}
// LFM_ENC_OVL (default 1): the blob upload continues on the loader thread (the backbone's
// LFM_PF_OVL machinery: per-chunk stream sync + per-tensor CRC before `done` is published)
// while the encoder starts; each stage waits for its own tensors (enc_need = 1 + the last
// upload-order index of a name prefix, 0 when loading synchronously).
static int enc_need(const char* prefix) {
  if (!g_lj) return 0;
  int n = 0; const size_t pl = strlen(prefix);
  for (size_t oi = 0; oi < g_lj->order.size(); oi++)
    if (!g_lj->M->ents[g_lj->order[oi]].name.compare(0, pl, prefix)) n = (int)oi + 1;
  return n;
}
static bool enc_wait(const char* prefix) {
  const int n = prefix ? enc_need(prefix) : (g_lj ? (int)g_lj->order.size() : 0);
  if (!load_wait(n)) { printf("encoder: weight load failed (%s)\n", prefix ? prefix : "all"); return false; }
  return true;
}
static int g_nodump = -2;
// Wall-clock breakdown for the ASR path. The encoder's GPU time is 0.013 s per audio
// second but its process wall time is ~0.6 s per audio second, and ASR_RT is charged
// with the wall time, so the difference (blob load, CRC, uploads, teardown) has to be
// attributable to a phase or it cannot be optimized. LFM_ENC_WALL=1 prints each phase.
static double g_tw_prev = 0, g_tw_all = 0; static int g_tw_on = -2;
static void wallmark(const char* what) {
  if (g_tw_on == -2) g_tw_on = getenv("LFM_ENC_WALL") ? atoi(getenv("LFM_ENC_WALL")) : 0;
  if (!g_tw_on) return;
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  const double now = t.tv_sec + 1e-9 * t.tv_nsec;
  printf("WALL %-22s %7.2f s (since start %7.2f s)\n", what, now - g_tw_prev, now - g_tw_all);
  fflush(stdout);
  g_tw_prev = now;
}
static const char* g_only = NULL;
static void dump(const char* dir, const char* name, const float* dptr, size_t n) {
  // LFM_ENC_NODUMP=1 is the shipping path: the 20 stage dumps are the GATE's
  // instrumentation (~100 MB of device-to-host copies and file writes on a 4 GB Tegra
  // with eMMC), and leaving them in an ASR_RT measurement charges the gate's cost to the
  // product. The dump's non-zero plausibility check is also skipped, so this is for
  // timing runs only -- a gate run must never set it.
  if (g_nodump == -2) g_nodump = getenv("LFM_ENC_NODUMP") ? atoi(getenv("LFM_ENC_NODUMP")) : 0;
  if (g_nodump) return;
  // `--only <stage>` (argv 5-6): the ASR shell consumes proj_out alone, so every other
  // stage dump is skipped; the kept one still gets the plausibility check below.
  if (g_only && strcmp(name, g_only) != 0) return;
  std::vector<float> h(n);
  cudaMemcpy(h.data(), dptr, n * 4, cudaMemcpyDeviceToHost);
  char path[512]; snprintf(path, sizeof path, "%s/%s.bin", dir, name);
  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) { printf("cannot write %s\n", path); exit(2); }
  if (write(fd, h.data(), n * 4) != (ssize_t)(n * 4)) { printf("short write %s\n", path); exit(2); }
  close(fd);
  long nz = 0; double mx = 0;
  for (size_t i = 0; i < n; i++) { nz += (h[i] != 0.f); if (fabs(h[i]) > mx) mx = fabs(h[i]); }
  printf("dump %-12s n=%zu nonzero=%ld max=%.4f\n", name, n, nz, mx);
  if (nz == 0) { printf("PLAUSIBILITY FAIL: %s is all zeros\n", name); exit(1); }
}
static Ent* need(Model& M, const char* name) {
  Ent* e = ::T(M, name);
  if (!e) { printf("tensor missing from the blob: %s\n", name); exit(2); }
  return e;
}

// Read one request line from the serve fifo without ever blocking on a writer (the same shape
// as lfm3_nanode's clip_read, which had to solve the identical problem: a per-request client
// means the read side sees EOF between requests). Returns 0 only when the fifo cannot be opened.
static int enc_read_request(const char* path, char* out, size_t cap) {
  int fd = open(path, O_RDONLY | O_NONBLOCK);
  if (fd < 0) { printf("--serve-fifo: cannot open %s (%s)\n", path, strerror(errno)); return 0; }
  size_t n = 0; bool got_nl = false; int idle = 0;
  for (;;) {
    char c;
    ssize_t r = read(fd, &c, 1);
    if (r == 1) {
      idle = 0;
      if (c == '\n') { got_nl = true; break; }
      if (n + 1 < cap) out[n++] = c;
      continue;
    }
    if (r < 0 && errno != EAGAIN && errno != EINTR) { close(fd); return 0; }
    // Both no-data cases land here: r == 0 (no writer has the fifo open) and EAGAIN (a writer is
    // open but idle) -- a non-blocking fifo read returns EAGAIN, not 0, while a writer exists, and
    // the client holds one open between its own write and close.
    if (got_nl) break;
    // A line with no newline is accepted only after ~100 ms of silence: the client writes the
    // whole request in one write() (atomic on a fifo below PIPE_BUF), so an idle read with a
    // partial buffer means the writer is done, while accepting it immediately would be free to
    // split a request that is still arriving.
    if (n && ++idle > 5) break;
    if (!n) {                                  // writer closed between requests: reopen
      close(fd);
      fd = open(path, O_RDONLY | O_NONBLOCK);
      if (fd < 0) return 0;
    }
    struct timespec ts = { 0, 20000000 };   // 20 ms poll; a turn is seconds away
    nanosleep(&ts, nullptr);
  }
  out[n] = 0;
  close(fd);
  return 1;
}

int main(int argc, char** argv) {
  const char* serve_fifo = nullptr;   // --serve-fifo PATH: resident encoder, one pass per request line
  // The two optional switches always follow the four positionals, so they are scanned from argv[5]:
  // in --serve-fifo mode argv[2]/argv[4] are placeholders (the mel path and the output dir arrive
  // per request), which is why the positional count cannot be validated the same way there.
  for (int i = 5; i < argc; i++) {
    if (!strcmp(argv[i], "--serve-fifo") && i + 1 < argc) serve_fifo = argv[++i];
    else if (!strcmp(argv[i], "--only") && i + 1 < argc) g_only = argv[++i];
  }
  if (argc == 7 && !strcmp(argv[5], "--only")) g_only = argv[6];
  else if (!serve_fifo && argc != 5) { printf("usage: %s <enc blob> <input.bin> <Tm> <outdir> [--only <stage> | --serve-fifo <fifo> --only <stage>]\n", argv[0]); return 2; }
  g_ench2 = getenv("LFM_ENC_H2") ? atoi(getenv("LFM_ENC_H2")) : 0;
  g_enc_win = getenv("LFM_ENC_WIN") ? atoi(getenv("LFM_ENC_WIN")) : 32;
  printf("ENC-ATTN layers=2,5,8,11,14,15,16 window=%d (encoder.sliding_window, all "
         "attention layers; section 42)\n", g_enc_win); fflush(stdout);
  // Tm = "auto": the ASR shell starts this process before the mel front end has run, so
  // the CUDA init + blob load overlap WAV+mel. The input is then waited for after the load
  // (the shell publishes it with an atomic rename once lfm3_mel succeeded and the file is
  // validated) and Tm is its size / (128 x 4 B).
  const bool tm_auto = !strcmp(argv[3], "auto");
  int Tm = tm_auto ? 8 : atoi(argv[3]);
  // The 8192 that used to sit here was a bring-up guard, not a structural limit: every
  // per-position buffer is a dynamic cudaMalloc (dalloc) and the downsampler's strides
  // are computed from Tm, so the real constraint is device memory, which the loader's
  // free/need gate below measures. 32768 mel frames = 327.7 s at the 100 frames/s mel
  // rate = 4096 audio tokens; the footprint scales linearly at ~52 KB/token, so that
  // needs ~215 MB of activation buffers on top of the 130 MB blob -- inside the ~1.9 GB
  // this board leaves free, and the gate below refuses before any kernel runs if not.
  if (Tm < 8 || Tm > 32768) { printf("Tm out of range: %d (8..32768 mel frames = 0.08..327.7 s)\n", Tm); return 2; }
  if (!serve_fifo) mkdir(argv[4], 0777);   // in serve mode the output dir arrives per request
  { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); g_tw_all = g_tw_prev = t.tv_sec + 1e-9 * t.tv_nsec; }
  Model M; std::string js;
  { const char* ov = getenv("LFM_ENC_OVL");
    if (!ov || atoi(ov)) { g_ovl = 1; setenv("LFM_PF_LAZY", "0", 1); } }
  if (!load_blob(argv[1], M, js)) return 1;
  wallmark("blob read+CRC+upload");
  { const char* pz = getenv("LFM_ENC_POISON_ARENA");        // hunt, not a shipping path
    if (pz && atoi(pz)) {
      if (!enc_wait(nullptr)) return 1;
      if (M.df && M.need_f) cudaMemset(M.df, 0xff, M.need_f);
      if (M.ds && M.need_s) cudaMemset(M.ds, 0xff, M.need_s);
      printf("poisoned arenas: f32 %.1f MB scales %.1f MB\n", M.need_f / 1e6, M.need_s / 1e6);
    } }

  // ---- one encoder pass ---------------------------------------------------------------
  // Extracted verbatim from main so a resident process (--serve-fifo) can run it per turn.
  // The one-shot path calls it exactly once with the argv values, which is the behaviour every
  // gate has always exercised; nothing in the body changed except where the mel path, the
  // output dir and Tm come from.
  auto run_pass = [&](const char* melp, const char* odir_in, int Tm_in) -> int {
  int Tm = Tm_in;
  const bool pass_auto = (Tm_in <= 0);
  mkdir(odir_in, 0777);
  g_slot = 0;                 // the pass's allocations come from the pooled slots, in call order
  g_reqallocs.clear();
  g_best.clear(); g_sum.clear(); g_cnt.clear();   // per-pass attribution (see the stage table)
  if (pass_auto) {
    struct stat stt; int waited_ms = 0;
    while (stat(melp, &stt) != 0) {
      if (getppid() == 1) { printf("parent gone while waiting for %s\n", melp); return 2; }
      if (waited_ms > 600000) { printf("timed out waiting for %s\n", melp); return 2; }
      usleep(2000); waited_ms += 2;
    }
    if (stt.st_size % (128 * 4) != 0) { printf("%s: %ld bytes, not whole 128-row frames\n", melp, (long)stt.st_size); return 2; }
    Tm = (int)(stt.st_size / (128 * 4));
    if (Tm < 8 || Tm > 32768) { printf("Tm out of range: %d\n", Tm); return 2; }
    printf("Tm=auto: %d mel frames (waited %d ms after the blob load)\n", Tm, waited_ms);
    wallmark("mel wait");
  }
  std::vector<float> hfeat((size_t)128 * Tm);
  int fd = open(melp, O_RDONLY);
  if (fd < 0) { printf("cannot open %s\n", melp); return 2; }
  if (read(fd, &hfeat[0], hfeat.size() * 4) != (ssize_t)(hfeat.size() * 4)) { printf("input short\n"); return 2; }
  close(fd);
  for (size_t i = 0; i < hfeat.size(); i++) if (!isfinite(hfeat[i])) { printf("input non-finite at %zu\n", i); return 2; }
  wallmark("mel read+validated");

  const int EH1 = 64, EWo1 = (Tm + 2 - 3) / 2 + 1;
  const int EH2 = 32, EWo2 = (EWo1 + 2 - 3) / 2 + 1;
  const int EH3 = 16, EWo3 = (EWo2 + 2 - 3) / 2 + 1;
  const int Tt = EWo3;
  printf("Tm=%d -> EWo1=%d EWo2=%d tokens Tt=%d (expect ceil(Tm/8)); arms=%s\n",
         Tm, EWo1, EWo2, Tt, g_ench2 ? "batched half2" : "fp32");
  if (g_encprobe == 2) { int one = 2; cudaMemcpyToSymbol(g_probe_d, &one, sizeof(int));
    printf("*** PROBE RUN: silu disabled, OUTPUTS ARE NOT VALID (timing only) ***\n"); }
  fflush(stdout);
  const size_t n1 = (size_t)256 * EH1 * EWo1, n2 = (size_t)256 * EH2 * EWo2, n3 = (size_t)256 * EH3 * EWo3;
  float *feat, *x1, *x2, *x2b, *x3, *x3b, *perm, *flat, *ds, *xb = nullptr;
  feat = dalloc(hfeat.size(), "feat"); cudaMemcpy(feat, &hfeat[0], hfeat.size() * 4, cudaMemcpyHostToDevice);
  x1 = dalloc_w(n1, "x1"); x2 = dalloc_w(n2, "x2"); x2b = dalloc(n2, "x2b");
  x3 = dalloc(n3, "x3"); x3b = dalloc(n3, "x3b"); perm = dalloc(n2, "perm");
  flat = dalloc((size_t)Tt * 4096, "flat"); ds = dalloc((size_t)Tt * EHID, "ds");
  if (g_ench2) {                                   // the biggest GEMM input in bf16
    size_t xbn = (size_t)EH2 * EWo2 * 256;
    xbn = std::max(xbn, (size_t)EH3 * EWo3 * 256); xbn = std::max(xbn, (size_t)Tt * 4096);
    if (cudaMalloc(&xb, xbn * 2)) { printf("OOM xb\n"); return 2; }
  }

  Ent* w_in = need(M, "encoder.downsample.conv_in.weight");     // [256,1,3,3] f32
  Ent* b_in = need(M, "encoder.downsample.conv_in.bias");
  Ent* w_mid = need(M, "encoder.downsample.conv_mid.weight");   // [256,1,3,3] f32
  Ent* b_mid = need(M, "encoder.downsample.conv_mid.bias");
  Ent* w_lmid = need(M, "encoder.downsample.lin_mid.weight");   // Q8 [256,256]
  Ent* b_lmid = need(M, "encoder.downsample.lin_mid.bias");
  Ent* w_end = need(M, "encoder.downsample.conv_end.weight");
  Ent* b_end = need(M, "encoder.downsample.conv_end.bias");
  Ent* w_lend = need(M, "encoder.downsample.lin_end.weight");
  Ent* b_lend = need(M, "encoder.downsample.lin_end.bias");
  Ent* w_out = need(M, "encoder.downsample.lin_out.weight");
  Ent* b_out = need(M, "encoder.downsample.lin_out.bias");
  // the engine's convention: ne[0] = N (rows), ne[1] = K
  if (w_out->dtype != 1 || w_out->ne[0] != EHID || w_out->ne[1] != 4096) {
    printf("lin_out is not Q8 N=%d K=4096 (dtype %d %dx%d)\n", EHID, w_out->dtype, w_out->ne[0], w_out->ne[1]); return 2;
  }

  cudaEvent_t ev0, ev1; cudaEventCreate(&ev0); cudaEventCreate(&ev1);
  cudaEventCreate(&g_seva); cudaEventCreate(&g_sevb);
  cudaEventRecord(ev0); cudaEventRecord(g_seva);
  const int nreps = g_enctime ? (g_reps < 1 ? 1 : g_reps) : 1;
  float best_ms = 1e30f;
  int nlay_last = 0;

  if (!enc_wait("encoder.downsample.")) return 1;
  // 1) conv_in + silu
  conv3x3(feat, w_in->df, b_in->df, x1, 256, 128, Tm, EH1, EWo1, 1, 1);
  chk("conv_in");
  // 2) conv_mid (depthwise)
  conv3x3(x1, w_mid->df, b_mid->df, x2, 256, EH1, EWo1, EH2, EWo2, 0, 0);
  chk("conv_mid");
  // 3) lin_mid as a GEMM over P2 = EH2*EWo2 positions, then silu
  { const int P2 = EH2 * EWo2;
    if (g_enctr) transpose(x2, perm, 256, P2);
    else k_chw2phw<<<(unsigned)((n2 + 255) / 256), 256>>>(x2, perm, 256, P2);
    chk("chw2phw");
    Gemm g;
    if (w_lmid->dtype == 2) gemm_setup_f32(g, w_lmid->df, 256, 256, P2);
    else gemm_setup(g, w_lmid->dq, w_lmid->ds, 256, 256, P2, 1, 4);
    printf("  lin_mid N=%d K=%d T=%d %s weights\n", g.N, g.K, P2, g.wf ? "f32" : "Q8");
    gemm_run(g, perm, x2b, P2, g_ench2 ? (bf16*)xb : nullptr);
    gemm_release(g);
    k_bias_silu<<<(unsigned)((n2 + 255) / 256), 256>>>(x2b, b_lmid->df, n2, 256, 1); }
  chk("lin_mid gemm");
  // 4) conv_end (depthwise) on the CHW view of x2b
  if (g_enctr) transpose(x2b, perm, EH2 * EWo2, 256);
  else k_phw2chw<<<(unsigned)((n2 + 255) / 256), 256>>>(x2b, perm, 256, EH2 * EWo2);
  conv3x3(perm, w_end->df, b_end->df, x3, 256, EH2, EWo2, EH3, EWo3, 0, 0);
  chk("conv_end");
  // 5) lin_end as a GEMM over P3 = EH3*EWo3 positions, then silu
  { const int P3 = EH3 * EWo3;
    if (g_enctr) transpose(x3, x2b, 256, P3);   // reuse x2b as the PHW source
    else k_chw2phw<<<(unsigned)((n3 + 255) / 256), 256>>>(x3, x2b, 256, P3);
    Gemm g;
    if (w_lend->dtype == 2) gemm_setup_f32(g, w_lend->df, 256, 256, P3);
    else gemm_setup(g, w_lend->dq, w_lend->ds, 256, 256, P3, 1, 4);
    gemm_run(g, x2b, x3b, P3, g_ench2 ? (bf16*)xb : nullptr);
    gemm_release(g);
    k_bias_silu<<<(unsigned)((n3 + 255) / 256), 256>>>(x3b, b_lend->df, n3, 256, 1); }
  chk("lin_end gemm");
  // 6) flatten(1,2) -> (Tt, 4096)
  k_flatten<<<(unsigned)((Tt * 4096 + 255) / 256), 256>>>(x3b, flat, 256, EH3, Tt);
  chk("flatten");
  // 7) lin_out + bias -> ds_out
  { Gemm g; if (w_out->dtype != 1) { printf("lin_out must be Q8\n"); return 2; }
    gemm_setup(g, w_out->dq, w_out->ds, EHID, 4096, Tt, 1, 4);
    gemm_run(g, flat, ds, Tt, g_ench2 ? (bf16*)xb : nullptr);
    gemm_release(g);
    k_bias_silu<<<(unsigned)((Tt * EHID + 255) / 256), 256>>>(ds, b_out->df, (size_t)Tt * EHID, EHID, 0); }
  chk("lin_out");

  cudaEventRecord(ev1); cudaEventSynchronize(ev1);
  float ms = 0; cudaEventElapsedTime(&ms, ev0, ev1);
  cudaError_t e = cudaGetLastError();
  if (e) { printf("CUDA ERROR: %s\n", cudaGetErrorString(e)); return 1; }
  const double macs = 256.0 * 9 * EH1 * EWo1                    // conv_in
                    + 256.0 * 9 * EH2 * EWo2                    // conv_mid  (depthwise)
                    + 256.0 * 256.0 * EH2 * EWo2                // lin_mid
                    + 256.0 * 9 * EH3 * EWo3                    // conv_end  (depthwise)
                    + 256.0 * 256.0 * EH3 * EWo3                // lin_end
                    + (double)EHID * 4096 * Tt;                // lin_out
  printf("ENCODER total %.2f ms (min of %d) = %.4f s GPU per audio second; downsampler alone %.0f MMAC, %.1f GMAC/s at that rate\n",
         ms, nreps, ms / 1000.0 / (Tm / 100.0), macs / 1e6, macs / (ms * 1e-3) / 1e9);
  char odir[600]; snprintf(odir, sizeof odir, "%s", odir_in);
  const int dumprep = getenv("LFM_ENC_DUMP_REP") ? atoi(getenv("LFM_ENC_DUMP_REP")) : 0;
  char repdir[640];
  for (int rep = 0; rep < nreps; rep++) {
  if (dumprep) { snprintf(repdir, sizeof repdir, "%s_r%d", odir, rep); mkdir(repdir, 0755); }
  const char* OD = dumprep ? repdir : odir;
  cudaEventRecord(ev0);
  dump(OD, "ds_out", ds, (size_t)Tt * EHID);

  // ---------------- the 17 layers ----------------
  // Both layer types are ported: conv blocks (types[L]==0 in lfm3_enc.c) and the
  // attention blocks (full causal attention, 8 q heads / 4 kv heads, per-head rmsnorm
  // over 96, RoPE rotate-half theta 1e6). A layer whose tensors are requested but
  // missing aborts, so a partial port cannot silently skip work.
  static const char ETYPES[ENLAYER] = {0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,1,1};
  // The residual stream rotates between TWO dedicated buffers; ds is read-only.
  // Rotating xr/hout with only one scratch buffer let odd layers write into ds, so
  // every rep after the first started from layer 16's output (gate caught it: rep 2's
  // ds_out max was 5683, layer 16's, not 1.78).
  float* hpool[2] = { dalloc((size_t)Tt * EHID, "pool0"), dalloc((size_t)Tt * EHID, "pool1") };
  float *xr = ds;                                   // the residual stream
  float *hout = hpool[0];
  int pool_i = 0;
  float *hn = dalloc((size_t)Tt * EHID, "hn");
  float *hbcx = dalloc((size_t)Tt * 3 * EHID, "bcx");
  float *hbx = dalloc((size_t)Tt * EHID, "bx");
  float *hcg = dalloc((size_t)Tt * EHID, "cg");
  float *hcy = dalloc((size_t)Tt * EHID, "cy");
  float *hffn = dalloc((size_t)Tt * EFFN, "ffn");
  float *hffn2 = dalloc((size_t)Tt * EFFN, "ffn2");
  float *hffno = dalloc((size_t)Tt * EHID, "ffno");   // w2 output: hout already holds x+conv
  float *hq = dalloc((size_t)Tt * ENH * EHD, "q");
  float *hk = dalloc((size_t)Tt * ENKV * EHD, "k");
  float *hv = dalloc((size_t)Tt * ENKV * EHD, "v");
  float *hao = dalloc((size_t)Tt * ENH * EHD, "ao");
  float *hp1 = dalloc((size_t)Tt * EHID, "pln");
  float *hp2 = dalloc((size_t)Tt * EPROJ, "pp1");
  float *hp3 = dalloc((size_t)Tt * EPROJ, "pp2");
  bf16* xb2 = nullptr;
  if (g_ench2) {                                    // the GEMM arm's bf16 input buffer
    size_t n = (size_t)Tt * 4096;                    // the widest GEMM input in the layers
    n = std::max(n, (size_t)Tt * EFFN);
    if (cudaMalloc(&xb2, n * 2)) { printf("OOM xb2\n"); return 2; }
  }
  int nlayer_done = 0;
    for (int L = 0; L < ENLAYER; L++) {
    char nm[160];
    snprintf(nm, sizeof nm, "encoder.lfm.lfm.layers.%d.", L);
    if (!enc_wait(nm)) return 1;
    snprintf(nm, sizeof nm, "encoder.lfm.lfm.layers.%d.operator_norm.weight", L);
    Ent* opn = need(M, nm);
    cudaMemcpy(hn, xr, (size_t)Tt * EHID * 4, cudaMemcpyDeviceToDevice);
    { const double* ss = sumsq_rows(hn, Tt, EHID);
      k_rmsnorm_f32<<<Tt, 256>>>(hn, opn->df, EHID, ss); } chk("op_norm");
    if (ETYPES[L]) {
      // q/k/v, per-head rmsnorm, RoPE, causal attention, out_proj
      snprintf(nm, sizeof nm, "encoder.lfm.lfm.layers.%d.self_attn.q_proj.weight", L);
      Ent* wq = need(M, nm);
      snprintf(nm, sizeof nm, "encoder.lfm.lfm.layers.%d.self_attn.k_proj.weight", L);
      Ent* wk = need(M, nm);
      snprintf(nm, sizeof nm, "encoder.lfm.lfm.layers.%d.self_attn.v_proj.weight", L);
      Ent* wv = need(M, nm);
      snprintf(nm, sizeof nm, "encoder.lfm.lfm.layers.%d.self_attn.out_proj.weight", L);
      Ent* wo = need(M, nm);
      snprintf(nm, sizeof nm, "encoder.lfm.lfm.layers.%d.self_attn.q_layernorm.weight", L);
      Ent* qn = need(M, nm);
      snprintf(nm, sizeof nm, "encoder.lfm.lfm.layers.%d.self_attn.k_layernorm.weight", L);
      Ent* kn = need(M, nm);
      gemm_to(wq, hn, hq, Tt, xb2, "q_proj");
      gemm_to(wk, hn, hk, Tt, xb2, "k_proj");
      gemm_to(wv, hn, hv, Tt, xb2, "v_proj");
      { const double* ss = sumsq_rows(hq, Tt * ENH, EHD);
        k_headnorm<<<Tt * ENH, 96>>>(hq, qn->df, ss); } chk("q headnorm");
      { const double* ss = sumsq_rows(hk, Tt * ENKV, EHD);
        k_headnorm<<<Tt * ENKV, 96>>>(hk, kn->df, ss); } chk("k headnorm");
      k_rope<<<(unsigned)((Tt * (EHD / 2) + 255) / 256), 256>>>(hq, hk, Tt); chk("rope");
      dim3 gA(Tt, ENH);
      float* dbg = nullptr; int ts = -1, hs = -1;
      { const char* sd = getenv("LFM_ENC_PROBE_SCORES");
        if (sd && L == atoi(sd)) {
          if (cudaMalloc((void**)&dbg, (size_t)Tt * 4) || cudaMemset(dbg, 0, (size_t)Tt * 4)) { printf("probe alloc\n"); return 3; }
          ts = atoi(strchr(sd, ',') ? strchr(sd, ',') + 1 : "0"); hs = atoi(strrchr(sd, ',') ? strrchr(sd, ',') + 1 : "0");
        } }
      { static const int att2 = getenv("LFM_ENC_ATT2") ? atoi(getenv("LFM_ENC_ATT2")) : 1;
        const size_t smx = (size_t)((g_enc_win > 0 && g_enc_win < Tt) ? g_enc_win : Tt) * 4;
        if (att2 && smx <= 40 * 1024)
          k_attn_prefill2<<<gA, 96, smx>>>(hq, hk, hv, hao, Tt, 1.0f / sqrtf((float)EHD), dbg, ts, hs, g_enc_win);
        else
          k_attn_prefill<<<gA, 96>>>(hq, hk, hv, hao, Tt, 1.0f / sqrtf((float)EHD), dbg, ts, hs, g_enc_win);
        chk("attention"); }
      if (dbg) { static float* hb = nullptr; if (!hb) hb = (float*)malloc((size_t)Tt * 4);
        cudaMemcpy(hb, dbg, (size_t)Tt * 4, cudaMemcpyDeviceToHost);
        printf("scores t=%d h=%d:", ts, hs); for (int j = 0; j <= ts; j++) printf(" %.6f", hb[j]); printf("\n"); }
      { static const int pl = getenv("LFM_ENC_PROBE_LAYER") ? atoi(getenv("LFM_ENC_PROBE_LAYER")) : -1;
        if (pl == L) {
          char pp[40];
          snprintf(pp, sizeof pp, "probe_q_%02d", L); dump(OD, pp, hq, (size_t)Tt * ENH * EHD);
          snprintf(pp, sizeof pp, "probe_k_%02d", L); dump(OD, pp, hk, (size_t)Tt * ENKV * EHD);
          snprintf(pp, sizeof pp, "probe_v_%02d", L); dump(OD, pp, hv, (size_t)Tt * ENKV * EHD);
          snprintf(pp, sizeof pp, "probe_ao_%02d", L); dump(OD, pp, hao, (size_t)Tt * ENH * EHD);
        } }
      gemm_to(wo, hao, hout, Tt, xb2, "attn out_proj");
      k_add<<<(unsigned)((Tt * EHID + 255) / 256), 256>>>(hout, xr, (size_t)Tt * EHID); chk("res1");
    } else {
    snprintf(nm, sizeof nm, "encoder.lfm.lfm.layers.%d.conv.in_proj.weight", L);
    Ent* win = need(M, nm);
    gemm_to(win, hn, hbcx, Tt, xb2, "in_proj");
    k_bcx_split<<<(unsigned)((Tt * EHID + 255) / 256), 256>>>(hbcx, hbx, hcg, Tt, EHID); chk("bcx split");
    snprintf(nm, sizeof nm, "encoder.lfm.lfm.layers.%d.conv.conv.weight", L);
    Ent* wc = need(M, nm);
    k_conv3_causal<<<(unsigned)((Tt * EHID + 255) / 256), 256>>>(hbx, wc->df, hcg, hcy, Tt, EHID); chk("conv3");
    snprintf(nm, sizeof nm, "encoder.lfm.lfm.layers.%d.conv.out_proj.weight", L);
    Ent* wout = need(M, nm);
    gemm_to(wout, hcy, hout, Tt, xb2, "out_proj");
    k_add<<<(unsigned)((Tt * EHID + 255) / 256), 256>>>(hout, xr, (size_t)Tt * EHID); chk("res1");
    }   // end conv-block branch; the FFN tail below is shared by both layer types
    snprintf(nm, sizeof nm, "encoder.lfm.lfm.layers.%d.ffn_norm.weight", L);
    Ent* fn = need(M, nm);
    cudaMemcpy(hn, hout, (size_t)Tt * EHID * 4, cudaMemcpyDeviceToDevice);
    { const double* ss = sumsq_rows(hn, Tt, EHID);
      k_rmsnorm_f32<<<Tt, 256>>>(hn, fn->df, EHID, ss); } chk("ffn_norm");
    snprintf(nm, sizeof nm, "encoder.lfm.lfm.layers.%d.feed_forward.w1.weight", L);
    Ent* w1 = need(M, nm);
    snprintf(nm, sizeof nm, "encoder.lfm.lfm.layers.%d.feed_forward.w3.weight", L);
    Ent* w3 = need(M, nm);
    snprintf(nm, sizeof nm, "encoder.lfm.lfm.layers.%d.feed_forward.w2.weight", L);
    Ent* w2 = need(M, nm);
    gemm_to(w1, hn, hffn, Tt, xb2, "ffn w1");
    gemm_to(w3, hn, hffn2, Tt, xb2, "ffn w3");
    k_silu_mul<<<(unsigned)((Tt * EFFN + 255) / 256), 256>>>(hffn, hffn2, (size_t)Tt * EFFN); chk("silu_mul");
    // out = (x + conv) so far; the FFN result goes in its OWN buffer and is ADDED.
    // Writing w2 over hout here silently dropped the conv branch from the residual
    // (gate caught it at rms rel 24 vs the C reference).
    gemm_to(w2, hffn, hffno, Tt, xb2, "ffn w2");
    k_add<<<(unsigned)((Tt * EHID + 255) / 256), 256>>>(hout, hffno, (size_t)Tt * EHID); chk("res2");
    char ln[32]; snprintf(ln, sizeof ln, "layer_%02d", L);
    dump(OD, ln, hout, (size_t)Tt * EHID);
    xr = hout; pool_i ^= 1; hout = hpool[pool_i];                      // residual rotates
    nlayer_done++;
    nlay_last = nlayer_done;
  }
  // embedding_norm -> enc_out, then the projector
  if (!enc_wait(nullptr)) return 1;
  {
    Ent* en = need(M, "encoder.lfm.lfm.embedding_norm.weight");
    cudaMemcpy(hn, xr, (size_t)Tt * EHID * 4, cudaMemcpyDeviceToDevice);
    { const double* ss = sumsq_rows(hn, Tt, EHID);
      k_rmsnorm_f32<<<Tt, 256>>>(hn, en->df, EHID, ss); } chk("embedding_norm");
    dump(OD, "enc_out", hn, (size_t)Tt * EHID);
    Ent* pw0 = need(M, "mlp.model.0.weight"), *pb0 = need(M, "mlp.model.0.bias");
    Ent* pw1 = need(M, "mlp.model.1.weight"), *pb1 = need(M, "mlp.model.1.bias");
    Ent* pw3 = need(M, "mlp.model.3.weight"), *pb3 = need(M, "mlp.model.3.bias");
    wallmark("17 layers");
    // LFM_ENC_LNM (default 1): moments one row per lane (k_moments_rows); 0 = thread 0.
    static double* mv = nullptr;
    const bool lnm = !getenv("LFM_ENC_LNM") || atoi(getenv("LFM_ENC_LNM"));
    if (lnm && !mv && cudaMalloc(&mv, (size_t)Tt * 16)) { printf("moments alloc\n"); exit(3); }
    if (lnm) k_moments_rows<<<(Tt + 127) / 128, 128>>>(hn, mv, Tt);
    k_layernorm_f32<<<Tt, 256>>>(hn, pw0->df, pb0->df, hp1, lnm ? mv : nullptr); chk("proj ln");
    gemm_to(pw1, hp1, hp2, Tt, xb2, "proj lin1");
    k_bias_silu<<<(unsigned)((Tt * EPROJ + 255) / 256), 256>>>(hp2, pb1->df, (size_t)Tt * EPROJ, EPROJ, 0); chk("proj b1");
    wallmark("proj ln + gemms");
    k_gelu_erf<<<(unsigned)((Tt * EPROJ + 255) / 256), 256>>>(hp2, (size_t)Tt * EPROJ); chk("gelu");
    wallmark("p: gelu issued");
    gemm_to(pw3, hp2, hp3, Tt, xb2, "proj lin2");
    wallmark("p: lin2 issued");
    k_bias_silu<<<(unsigned)((Tt * EPROJ + 255) / 256), 256>>>(hp3, pb3->df, (size_t)Tt * EPROJ, EPROJ, 0); chk("proj b3");
    wallmark("p: b3 issued");
    dump(OD, "proj_out", hp3, (size_t)Tt * EPROJ);
    wallmark("p: dumped");
  }
  cudaEventRecord(ev1); cudaEventSynchronize(ev1);
  float m1 = 0; cudaEventElapsedTime(&m1, ev0, ev1);
  if (m1 < best_ms) best_ms = m1;
  }
  ms = best_ms;
  wallmark("p: gpu drained");
  printf("layers ported: %d of %d\n", nlay_last, ENLAYER);
  if (g_enctime) for (std::map<std::string,double>::iterator it = g_best.begin(); it != g_best.end(); ++it) {
    const char* n = it->first.c_str();
    if (g_enctime >= 2)
      printf("  stage %-14s %8.2f ms (min of %d)  SUM %8.2f ms in %ld calls\n",
             n, it->second, nreps, g_sum[n], g_cnt[n]);
    else
      printf("  stage %-14s %8.2f ms (min of %d)\n", n, it->second, nreps);
  }
  printf("FM3_ENC_GPU_DONE Tm=%d Tt=%d\n", Tm, Tt);
  chk_final("end of pass");
  fflush(stdout);
  wallmark("layers+projector+dumps");
  // With the pool on, the pass's buffers stay allocated for the next turn (that is the whole
  // point); with LFM_ENC_POOL=0 they go back to the driver here, which is the pre-pool behaviour
  // and the A/B arm. Nothing is ever leaked: the pool is freed when the process exits.
  if (!g_pool_on) { for (size_t i = 0; i < g_reqallocs.size(); i++) cudaFree(g_reqallocs[i]); }
  g_reqallocs.clear();
  return 0;
  };

  if (serve_fifo) {
    // Resident encoder: the ASR shell starts ONE of these per session, so the ~1.9 s of CUDA
    // init + blob CRC/upload is paid at boot instead of on every turn (the pass itself is
    // ~0.2 s of GPU for a 9 s clip). Request = one line "<mel> <outdir> <Tm|auto>"; the reply
    // is "ENC_DONE <n> rc=..", which is what the client waits for. Errors exit: a service that
    // has lost its context cannot serve the next turn meaningfully.
    // The fifo is opened O_RDONLY|O_NONBLOCK and reopened on EOF, exactly as the backbone
    // service does: each ASR turn is its own client process, so the last writer closing is the
    // end of a turn's writer, not of the session. Blocking fopen would wait for a writer that
    // only shows up at the first turn -- after the boot check that waits for us.
    for (int req = 0; ; req++) {
      char rl[1400];
      printf("ENC_AWAIT %d\n", req); fflush(stdout);
      if (!enc_read_request(serve_fifo, rl, sizeof rl)) break;
      if (!rl[0]) { req--; continue; }
      if (!strcmp(rl, "Q")) break;
      char* s1 = strchr(rl, ' '); if (!s1) { printf("ENC: bad request '%s'\n", rl); return 2; }
      *s1++ = 0; char* s2 = strchr(s1, ' '); if (!s2) { printf("ENC: bad request '%s'\n", rl); return 2; }
      *s2++ = 0;
      while (*s2 == ' ') s2++;
      size_t sl = strlen(s2); while (sl && (s2[sl-1] == '\n' || s2[sl-1] == '\r')) s2[--sl] = 0;
      const int rq_tm = strcmp(s2, "auto") ? atoi(s2) : 0;
      const int rc = run_pass(rl, s1, rq_tm);
      printf("ENC_DONE %d rc=%d\n", req, rc); fflush(stdout);
      if (rc) return rc;
    }
    return 0;
  }
  { int rc = run_pass(argv[2], argv[4], tm_auto ? 0 : Tm); return rc; }
}
