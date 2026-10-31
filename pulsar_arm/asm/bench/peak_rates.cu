// Calibrate the Nano GPU's achievable per-instruction rates (root-free).
// HFMA2 / FFMA: 8 independent register chains per thread (enough ILP to cover
// latency), grid sized to fill the SM. LDS: conflict-free 32-bit shared reads.
// Also: HFMA2 with the SAME chain shape as q8_prefill_h2 (4 chains, 16 deep)
// at the engine's occupancy (2 blocks x 256 threads per SM), to see what the
// inner-loop structure itself can reach without any memory traffic.
// Every kernel writes a result that depends on the work (no dead-code
// elimination); plausibility: nonzero result, rate below the datasheet peak.
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>

__global__ void k_hfma2(__half2* out, int n) {
  __half2 a[8], m = __floats2half2_rn(0.999f, 1.001f), c = __floats2half2_rn(1e-3f, 2e-3f);
  for (int j = 0; j < 8; j++) a[j] = __floats2half2_rn(threadIdx.x * 1e-4f + j, j * 1e-3f);
  for (int i = 0; i < n; i++)
#pragma unroll
    for (int j = 0; j < 8; j++) a[j] = __hfma2(a[j], m, c);
  __half2 r = a[0];
  for (int j = 1; j < 8; j++) r = __hadd2(r, a[j]);
  out[blockIdx.x * blockDim.x + threadIdx.x] = r;
}
__global__ void k_ffma(float* out, int n) {
  float a[8];
  for (int j = 0; j < 8; j++) a[j] = threadIdx.x * 1e-4f + j;
  for (int i = 0; i < n; i++)
#pragma unroll
    for (int j = 0; j < 8; j++) a[j] = __fmaf_rn(a[j], 0.9999f, 1e-4f);
  float r = 0; for (int j = 0; j < 8; j++) r += a[j];
  out[blockIdx.x * blockDim.x + threadIdx.x] = r;
}
__global__ void k_lds(float* out, int n) {
  __shared__ float sh[256 * 17];
  for (int i = threadIdx.x; i < 256 * 17; i += blockDim.x) sh[i] = i * 1e-3f;
  __syncthreads();
  float r = 0; int idx = threadIdx.x * 17;
  for (int i = 0; i < n; i++) {
#pragma unroll
    for (int k = 0; k < 16; k++) r += sh[idx + k];
    idx = (idx + 1) % (256 * 17 - 16);
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = r;
}
// Engine-shaped: per iteration, 4 independent chains (c) of 16 dependent HFMA2,
// operands from registers -- the q8_prefill_h2 arithmetic with memory removed.
__global__ void k_engine_shape(__half2* out, int n) {
  __half2 w[16], x[4][16];
  for (int k = 0; k < 16; k++) {
    w[k] = __floats2half2_rn(1e-3f * (k + threadIdx.x % 7), 2e-3f * k);
    for (int c = 0; c < 4; c++) x[c][k] = __floats2half2_rn(1e-2f * (c + 1), 1e-2f * k);
  }
  __half2 acc = __floats2half2_rn(0.f, 0.f);
  for (int i = 0; i < n; i++) {
    __half2 h[4];
#pragma unroll
    for (int c = 0; c < 4; c++) h[c] = __floats2half2_rn(0.f, 0.f);
#pragma unroll
    for (int k = 0; k < 16; k++)
#pragma unroll
      for (int c = 0; c < 4; c++) h[c] = __hfma2(w[k], x[c][k], h[c]);
#pragma unroll
    for (int c = 0; c < 4; c++) acc = __hadd2(acc, h[c]);
    w[i & 15] = __hadd2(w[i & 15], acc);   // loop-carried, defeats hoisting
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template <typename F>
static void time_it(const char* name, F launch, double ops_per_launch, const char* unit, double peak) {
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  launch(); cudaDeviceSynchronize();           // warm + DVFS ramp
  launch(); cudaDeviceSynchronize();
  cudaEventRecord(a); launch(); cudaEventRecord(b); cudaEventSynchronize(b);
  cudaError_t e = cudaGetLastError();
  if (e) { printf("%-34s LAUNCH ERROR %s\n", name, cudaGetErrorString(e)); exit(2); }
  float ms = 0; cudaEventElapsedTime(&ms, a, b);
  const double rate = ops_per_launch / (ms * 1e-3) / 1e9;
  printf("%-34s %8.2f ms  %7.1f %s  (%.0f%% of %.0f)%s\n", name, ms, rate, unit,
         100.0 * rate / peak, peak, rate > peak * 1.02 ? "  PLAUSIBILITY FAIL" : "");
}

int main() {
  cudaDeviceProp dp; cudaGetDeviceProperties(&dp, 0);
  const double clk = 0.9216;   // GHz, max DVFS step (sm_clock measured 904 under load)
  const int sms = dp.multiProcessorCount;
  printf("SMs=%d  clock(max)=%.1f MHz  -> FFMA peak %.1f G/s, HFMA2 peak %.1f G lane-instr/s (= %.1f GMAC/s)\n",
         sms, clk * 1e3, sms * 128 * clk, sms * 128 * clk, 2 * sms * 128 * clk);
  void* buf; cudaMalloc(&buf, 64 << 20);
  const int blocks = sms * 32, tpb = 256, n = 20000;
  const double lanes = (double)blocks * tpb;
  time_it("HFMA2 8 chains (full occupancy)", [&] { k_hfma2<<<blocks, tpb>>>((__half2*)buf, n); },
          lanes * n * 8 * 2, "GMAC/s", 2 * sms * 128 * clk);
  time_it("FFMA 8 chains (full occupancy)", [&] { k_ffma<<<blocks, tpb>>>((float*)buf, n); },
          lanes * n * 8, "GFMA/s", sms * 128 * clk);
  time_it("LDS.32 conflict-free", [&] { k_lds<<<blocks, tpb>>>((float*)buf, n / 4); },
          lanes * (n / 4) * 16 * 4, "GB/s", sms * 128 * clk);
  // engine shape: 2 blocks/SM x 256 threads, as the occupancy API reports for ARM0 CT4
  const int eb = sms * 2;
  time_it("engine-shaped HFMA2, 16 warps/SM", [&] { k_engine_shape<<<eb, 256>>>((__half2*)buf, n); },
          (double)eb * 256 * n * 64 * 2, "GMAC/s", 2 * sms * 128 * clk);
  time_it("engine-shaped HFMA2, 64 warps/SM", [&] { k_engine_shape<<<eb * 4, 256>>>((__half2*)buf, n); },
          (double)eb * 4 * 256 * n * 64 * 2, "GMAC/s", 2 * sms * 128 * clk);
  return 0;
}
