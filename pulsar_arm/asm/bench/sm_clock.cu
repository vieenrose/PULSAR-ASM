// Root-free SM clock measurement: SM cycles (clock64) over wall time (cudaEvent).
// Runs a dependent FFMA chain long enough (~0.3-1 s) that event resolution is
// irrelevant, several times back to back so a DVFS ramp shows up as a trend.
// Usage: ./sm_clock [iters]   (default 6)
#include <cstdio>
#include <cstdlib>

__global__ void spin(long long* cyc, float* sink, int n) {
  long long t0 = clock64();
  float a = threadIdx.x * 1e-3f, b = 1.0000001f, c = 1e-7f;
  for (int i = 0; i < n; i++) a = __fmaf_rn(a, b, c);
  long long t1 = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) *cyc = t1 - t0;
  if (a == 12345.f) sink[threadIdx.x] = a;   // keep the chain alive
}

int main(int argc, char** argv) {
  const int iters = argc > 1 ? atoi(argv[1]) : 6;
  long long* dc; float* ds; cudaMalloc(&dc, 8); cudaMalloc(&ds, 4096);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  const int n = 20000000;   // 20M dependent FFMA per thread
  for (int it = 0; it < iters; it++) {
    cudaEventRecord(a);
    spin<<<2, 128>>>(dc, ds, n);     // one block per SM, latency-bound chain
    cudaEventRecord(b); cudaEventSynchronize(b);
    cudaError_t e = cudaGetLastError();
    if (e) { printf("LAUNCH ERROR %s\n", cudaGetErrorString(e)); return 2; }
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    long long cyc = 0; cudaMemcpy(&cyc, dc, 8, cudaMemcpyDeviceToHost);
    if (cyc <= 0 || ms <= 0) { printf("PLAUSIBILITY FAIL cyc=%lld ms=%.3f\n", cyc, ms); return 1; }
    printf("iter %d: %lld SM cycles in %.2f ms -> %.1f MHz (cycles per FFMA %.2f)\n",
           it, cyc, ms, cyc / (ms * 1e3), (double)cyc / n);
  }
  return 0;
}
