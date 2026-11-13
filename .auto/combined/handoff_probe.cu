// .auto/combined/handoff_probe.cu -- the per-split GPU->CPU->GPU hand-off latency on this board.
//
// Decides the CPU row share of the batched ASR prefill (run #293): the CPU path of a split GEMV is
//   x produced on the GPU -> D2H (pinned) -> host sees it -> [CPU rows] -> H2D of y -> consumer
// and its fixed latency L (everything except the CPU rows themselves) is paid ~576 times a turn.
// Sizes are a real 32-position prefill chunk: x = 32 x 2048 fp32 (256 KB), y = 32 x 640 fp32 (80 KB).
// Arms, each N iterations, median reported:
//   base     producer kernel; consumer kernel; sync                    (no hand-off)
//   handoff  producer; D2H x; stream sync; H2D y; consumer; sync     (CPU work = 0)
//   busy     the handoff chain on stream A while stream B runs a ~2 ms spin kernel per iteration
//            standing in for the GPU's share of the rows (one SM: the copy engines are separate,
//            the question is whether the chain's latency grows when the SM is occupied)
// L = handoff - base. busy reports the chain's own latency measured on the host while B runs.
// Build: nvcc -gencode arch=compute_53,code=sm_53 -O3 -o handoff_probe handoff_probe.cu
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <time.h>
#include <cuda_runtime.h>

static double now_ms() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec * 1e-6; }
#define CK(x) do { cudaError_t e = (x); if (e) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void produce(float* x, int n, float v) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] = v + i; }
__global__ void consume(const float* y, float* z, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) z[i] = y[i] * 2.f; }
__global__ void spin(long long cycles, int* sink) {
  long long t0 = clock64(); int a = threadIdx.x;
  while (clock64() - t0 < cycles) a = a * 1664525 + 1013904223;
  if (a == 42) *sink = a;
}

static double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
  const int N = argc > 1 ? atoi(argv[1]) : 200;
  const int nx = 32 * 2048, ny = 32 * 640;
  float *dx, *dy, *dz, *hx, *hy; int* sink;
  CK(cudaMalloc(&dx, nx * 4)); CK(cudaMalloc(&dy, ny * 4)); CK(cudaMalloc(&dz, ny * 4)); CK(cudaMalloc(&sink, 4));
  CK(cudaHostAlloc(&hx, nx * 4, cudaHostAllocDefault)); CK(cudaHostAlloc(&hy, ny * 4, cudaHostAllocDefault));
  for (int i = 0; i < ny; i++) hy[i] = (float)i;
  cudaStream_t A, B; CK(cudaStreamCreateWithFlags(&A, cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&B, cudaStreamNonBlocking));
  int clk_khz = 0; CK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0));
  const long long spin_cycles = (long long)clk_khz * 2;   // ~2 ms at the reported clock
  // warm up (clock ramp, first-touch, context)
  for (int i = 0; i < 300; i++) { spin<<<1, 32, 0, B>>>(spin_cycles / 4, sink); }
  CK(cudaDeviceSynchronize());

  std::vector<double> tb, th, tbusy;
  for (int it = 0; it < N; it++) {
    double t0 = now_ms();
    produce<<<(nx + 255) / 256, 256, 0, A>>>(dx, nx, (float)it);
    consume<<<(ny + 255) / 256, 256, 0, A>>>(dy, dz, ny);
    CK(cudaStreamSynchronize(A));
    tb.push_back(now_ms() - t0);

    t0 = now_ms();
    produce<<<(nx + 255) / 256, 256, 0, A>>>(dx, nx, (float)it);
    CK(cudaMemcpyAsync(hx, dx, nx * 4, cudaMemcpyDeviceToHost, A));
    CK(cudaStreamSynchronize(A));
    hy[0] = hx[0];                                   // the CPU "sees" x (rows would run here)
    CK(cudaMemcpyAsync(dy, hy, ny * 4, cudaMemcpyHostToDevice, A));
    consume<<<(ny + 255) / 256, 256, 0, A>>>(dy, dz, ny);
    CK(cudaStreamSynchronize(A));
    th.push_back(now_ms() - t0);

    spin<<<4, 256, 0, B>>>(spin_cycles, sink);       // the GPU's own share of the rows
    t0 = now_ms();
    produce<<<(nx + 255) / 256, 256, 0, A>>>(dx, nx, (float)it);
    CK(cudaMemcpyAsync(hx, dx, nx * 4, cudaMemcpyDeviceToHost, A));
    CK(cudaStreamSynchronize(A));
    hy[0] = hx[0];
    CK(cudaMemcpyAsync(dy, hy, ny * 4, cudaMemcpyHostToDevice, A));
    consume<<<(ny + 255) / 256, 256, 0, A>>>(dy, dz, ny);
    CK(cudaStreamSynchronize(A));
    tbusy.push_back(now_ms() - t0);
    CK(cudaStreamSynchronize(B));
  }
  const double b = med(tb), h = med(th), bu = med(tbusy);
  printf("N=%d  base %.3f ms  handoff %.3f ms  L = %.3f ms  | chain while SM busy %.3f ms (L_busy = %.3f ms)\n",
         N, b, h, h - b, bu, bu - b);
  printf("saving model (3.89 s prefill, 576 splits, CPU/GPU 7.5/50): L=%.3f -> %.0f ms/turn ; L_busy -> %.0f ms/turn\n",
         h - b, (3890.0 - 576 * (h - b)) / 7.67, (3890.0 - 576 * (bu - b)) / 7.67);
  return 0;
}
