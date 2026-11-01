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

static int g_ench2 = 0;            // 0 = fp32 arms (the C-ref oracle path), 1 = batched half2

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

static void conv3x3(const float* in, const float* w, const float* b, float* out,
                    int C, int nH, int nW, int nHo, int nWo, int silu, int in1) {
  const int items = C * nHo * ((nWo + 3) / 4);
  const int TPB = 256;
  if (in1 && g_convIn1) {                             // shared-staged, 256 threads
    const int TPB2 = 256;
    dim3 g((nWo + 127) / 128, nHo);
    k_conv_in_shared<<<g, TPB2>>>(in, w, b, out, nH, nW, nHo, nWo);
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
static const bool g_enctime = getenv("LFM_ENC_TIME") != nullptr;
static const int g_reps = getenv("LFM_ENC_REPS") ? atoi(getenv("LFM_ENC_REPS")) : 1;
// The Nano's GPU clock ramps over seconds (the backbone harness fights the same
// thing with LFM_WARMMS), so a single pass per stage is not a measurement: stages are
// run LFM_ENC_REPS times and the MIN per stage is kept, which is the same protocol the
// backbone benches use.
#include <map>
#include <string>
static std::map<std::string, double> g_best;
static std::map<std::string, double> g_last;
static void chk(const char* what) {
  if (g_enctime) {                                      // per-stage: syncs, so read the split
    cudaEventRecord(g_sevb); cudaEventSynchronize(g_sevb);
    float ms = 0.f; cudaEventElapsedTime(&ms, g_seva, g_sevb);
    std::string k(what);
    if (!g_best.count(k) || ms < g_best[k]) g_best[k] = ms;
    g_last[k] = ms;
    cudaEventRecord(g_seva);
  }
  cudaError_t e = cudaGetLastError();
  if (e) { printf("LAUNCH ERROR at %s: %s\n", what, cudaGetErrorString(e)); exit(1); }
  cudaDeviceSynchronize();
  e = cudaGetLastError();
  if (e) { printf("EXEC ERROR at %s: %s\n", what, cudaGetErrorString(e)); exit(1); }
}

// ---------------------------------------------------------------- GEMM
// The blob keeps SOME encoder linears as f32 (lin_mid, lin_end, and the projector's
// LayerNorm weights -- the same dtype table as Liquid's mmproj GGUF, cross-checked in
// plan section 13a), so the Q8 arms cannot serve them. 16x16-tiled SGEMM: one output
// per thread, k ascending in one accumulator, which is the C reference's order.
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

struct Gemm {
  const int8_t* q; const float* s; const float* wf;   // wf != null = f32 weights
  int N, K, KC, nb, nch, R, nblk, nblk256;
  float* part;                      // [T][N][nch] partials
};
static void gemm_setup_f32(Gemm& g, const float* wf, int N, int K, int T) {
  g.q = nullptr; g.s = nullptr; g.wf = wf; g.N = N; g.K = K;
  g.KC = K; g.nb = K / Q8_BLK; g.nch = 1; g.R = 256; g.nblk = 0; g.nblk256 = 0;
  g.part = nullptr;                                  // the SGEMM writes y directly
  if (K % Q8_BLK) { printf("f32 GEMM needs K a multiple of 32 (K=%d)\n", K); exit(2); }
}
static void gemm_setup(Gemm& g, const int8_t* q, const float* s, int N, int K, int T, int h2, int CT) {
  g.q = q; g.s = s; g.N = N; g.K = K; g.wf = nullptr;
  if (!q) { printf("gemm_setup: Q8 weights requested but q is null\n"); exit(2); }
  int KC = K;
  while (KC > 32) {
    const int nbtry = KC / Q8_BLK;
    const size_t need = h2 ? (size_t)CT * nbtry * 17 * 4 : (size_t)CT * nbtry * 33 * 4;
    if (need <= 49152 && K % KC == 0) break;
    KC /= 2;
  }
  g.KC = KC; g.nb = KC / Q8_BLK; g.nch = K / KC;
  g.R = 256;                        // the fp32 per-position arm
  g.nblk = ((N + g.R - 1) / g.R) * g.nch;
  g.nblk256 = g.nblk;               // the batched h2 arm also runs R=256
  const cudaError_t pe = cudaMalloc(&g.part, (size_t)T * N * g.nch * 4);
  if (pe) { printf("gemm part cudaMalloc %.1f MB failed: %s (N=%d K=%d T=%d nch=%d)\n",
                   (size_t)T * N * g.nch * 4 / 1e6, cudaGetErrorString(pe), N, K, T, g.nch); exit(2); }
}
// y[T][N] = x[T][K] @ W^T. fp32 path = per-position q8_partial<float> (the fastest
// measured fp32 arm, run #309); h2 path = batched q8_prefill_h2<bf16,ARM10,RB2>.
static void gemm_run(Gemm& g, const float* x, float* y, int T, bf16* xb) {
  const int TPB = 256;
  if (g.wf) {                                        // f32-weight path (no half2 arm)
    const dim3 grid((g.N + 63) / 64, (T + 63) / 64), block(16, 16);
    k_sgemm_f32<<<grid, block>>>(x, g.wf, y, T, g.N, g.K);
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
  for (int t = 0; t < T; t++)
    q8_partial<float, true, false, 1, 0><<<g.nblk, TPB, shm>>>(g.q, g.s, x + (size_t)t * g.K,
        g.part + (size_t)t * g.N * g.nch, g.N, g.K, g.KC, g.R, g.nb);
  chk("gemm fp32 loop");
  k_combine_f<<<(unsigned)((T * g.N + 255) / 256), 256>>>(g.part, y, T * g.N, g.nch);
  chk("gemm combine");
}

// ---------------------------------------------------------------- helpers
static float* dalloc(size_t n, const char* what) {
  float* p = nullptr;
  if (cudaMalloc(&p, n * 4)) { printf("OOM (%s, %.1f MB)\n", what, n * 4 / 1e6); exit(2); }
  return p;
}
static void dump(const char* dir, const char* name, const float* dptr, size_t n) {
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

int main(int argc, char** argv) {
  if (argc != 5) { printf("usage: %s <enc blob> <input.bin> <Tm> <outdir>\n", argv[0]); return 2; }
  g_ench2 = getenv("LFM_ENC_H2") ? atoi(getenv("LFM_ENC_H2")) : 0;
  const int Tm = atoi(argv[3]);
  if (Tm < 8 || Tm > 8192) { printf("Tm out of range: %d\n", Tm); return 2; }
  mkdir(argv[4], 0777);
  Model M; std::string js;
  if (!load_blob(argv[1], M, js)) return 1;

  std::vector<float> hfeat((size_t)128 * Tm);
  int fd = open(argv[2], O_RDONLY);
  if (fd < 0) { printf("cannot open %s\n", argv[2]); return 2; }
  if (read(fd, &hfeat[0], hfeat.size() * 4) != (ssize_t)(hfeat.size() * 4)) { printf("input short\n"); return 2; }
  close(fd);
  for (size_t i = 0; i < hfeat.size(); i++) if (!isfinite(hfeat[i])) { printf("input non-finite at %zu\n", i); return 2; }

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
  x1 = dalloc(n1, "x1"); x2 = dalloc(n2, "x2"); x2b = dalloc(n2, "x2b");
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
  for (int rep = 0; rep < nreps; rep++) {

  // 1) conv_in + silu
  conv3x3(feat, w_in->df, b_in->df, x1, 256, 128, Tm, EH1, EWo1, 1, 1);
  chk("conv_in");
  // 2) conv_mid (depthwise)
  conv3x3(x1, w_mid->df, b_mid->df, x2, 256, EH1, EWo1, EH2, EWo2, 0, 0);
  chk("conv_mid");
  // 3) lin_mid as a GEMM over P2 = EH2*EWo2 positions, then silu
  { const int P2 = EH2 * EWo2;
    k_chw2phw<<<(unsigned)((n2 + 255) / 256), 256>>>(x2, perm, 256, P2);
    chk("chw2phw");
    Gemm g;
    if (w_lmid->dtype == 2) gemm_setup_f32(g, w_lmid->df, 256, 256, P2);
    else gemm_setup(g, w_lmid->dq, w_lmid->ds, 256, 256, P2, 1, 4);
    printf("  lin_mid N=%d K=%d T=%d %s weights\n", g.N, g.K, P2, g.wf ? "f32" : "Q8");
    gemm_run(g, perm, x2b, P2, g_ench2 ? (bf16*)xb : nullptr);
    cudaFree(g.part);
    k_bias_silu<<<(unsigned)((n2 + 255) / 256), 256>>>(x2b, b_lmid->df, n2, 256, 1); }
  chk("lin_mid gemm");
  // 4) conv_end (depthwise) on the CHW view of x2b
  k_phw2chw<<<(unsigned)((n2 + 255) / 256), 256>>>(x2b, perm, 256, EH2 * EWo2);
  conv3x3(perm, w_end->df, b_end->df, x3, 256, EH2, EWo2, EH3, EWo3, 0, 0);
  chk("conv_end");
  // 5) lin_end as a GEMM over P3 = EH3*EWo3 positions, then silu
  { const int P3 = EH3 * EWo3;
    k_chw2phw<<<(unsigned)((n3 + 255) / 256), 256>>>(x3, x2b, 256, P3);   // reuse x2b as the PHW source
    Gemm g;
    if (w_lend->dtype == 2) gemm_setup_f32(g, w_lend->df, 256, 256, P3);
    else gemm_setup(g, w_lend->dq, w_lend->ds, 256, 256, P3, 1, 4);
    gemm_run(g, x2b, x3b, P3, g_ench2 ? (bf16*)xb : nullptr);
    cudaFree(g.part);
    k_bias_silu<<<(unsigned)((n3 + 255) / 256), 256>>>(x3b, b_lend->df, n3, 256, 1); }
  chk("lin_end gemm");
  // 6) flatten(1,2) -> (Tt, 4096)
  k_flatten<<<(unsigned)((Tt * 4096 + 255) / 256), 256>>>(x3b, flat, 256, EH3, Tt);
  chk("flatten");
  // 7) lin_out + bias -> ds_out
  { Gemm g; if (w_out->dtype != 1) { printf("lin_out must be Q8\n"); return 2; }
    gemm_setup(g, w_out->dq, w_out->ds, EHID, 4096, Tt, 1, 4);
    gemm_run(g, flat, ds, Tt, g_ench2 ? (bf16*)xb : nullptr);
    cudaFree(g.part);
    k_bias_silu<<<(unsigned)((Tt * EHID + 255) / 256), 256>>>(ds, b_out->df, (size_t)Tt * EHID, EHID, 0); }
  chk("lin_out");

  }
  cudaEventRecord(ev1); cudaEventSynchronize(ev1);
  float ms = 0; cudaEventElapsedTime(&ms, ev0, ev1);
  ms /= nreps;
  if (g_enctime) for (std::map<std::string,double>::iterator it = g_best.begin(); it != g_best.end(); ++it)
    printf("  stage %-14s %8.2f ms (min of %d)\n", it->first.c_str(), it->second, nreps);
  cudaError_t e = cudaGetLastError();
  if (e) { printf("CUDA ERROR: %s\n", cudaGetErrorString(e)); return 1; }
  const double macs = 256.0 * 9 * EH1 * EWo1                    // conv_in
                    + 256.0 * 9 * EH2 * EWo2                    // conv_mid  (depthwise)
                    + 256.0 * 256.0 * EH2 * EWo2                // lin_mid
                    + 256.0 * 9 * EH3 * EWo3                    // conv_end  (depthwise)
                    + 256.0 * 256.0 * EH3 * EWo3                // lin_end
                    + (double)EHID * 4096 * Tt;                // lin_out
  printf("downsampler %.2f ms = %.4f s GPU per audio second (%.0f MMAC, %.1f GMAC/s)\n",
         ms, ms / 1000.0 / (Tm / 100.0), macs / 1e6, macs / (ms * 1e-3) / 1e9);
  dump(argv[4], "ds_out", ds, (size_t)Tt * EHID);
  printf("FM3_ENC_GPU_DONE Tm=%d Tt=%d\n", Tm, Tt);
  return 0;
}
