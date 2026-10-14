/* tools/graph_probe.cu - what does a CUDA graph save on a Jetson Nano?
 *
 * The engine issues ~145 GEMV launches per token, each with the same API pattern:
 * H2D of a 4.7KB activation vector, one kernel, D2H of the output, then a sync.
 * Run #53's phase timing put that pattern at ~190us per call, i.e. 27.5ms per
 * token = 7.8% of the 352ms decode step, which is pure API cost, not compute.
 * A CUDA graph collapses memcpy+kernel+memcpy into one cudaGraphLaunch.
 *
 * This probe measures the three patterns on the SAME tiny kernel, in one process,
 * interleaved (A/B) so board drift cannot be blamed for the result:
 *   plain  : cudaMemcpy H2D, launch, cudaMemcpy D2H, cudaDeviceSynchronize
 *   graph  : cudaGraphLaunch of a captured {H2D, kernel, D2H}, cudaDeviceSynchronize
 *   nograph: a no-op kernel launch + sync (lower bound: launch + sync latency)
 *
 * Build (on the Nano, CUDA 10.2):
 *   nvcc -gencode arch=compute_53,code=sm_53 -O3 -std=c++11 graph_probe.cu -o graph_probe
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cuda_runtime.h>

#define NOUT 1024

__global__ void tiny(const float * __restrict__ x, float * __restrict__ y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < NOUT) y[i] = x[i] * 2.0f + 1.0f;
}

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e6 + t.tv_nsec / 1e3; /* us */
}

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA %s -> %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

int main(int argc, char **argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 2000;
  int reps = argc > 2 ? atoi(argv[2]) : 5;
  float *hx = NULL, *hy = NULL, *dx = NULL, *dy = NULL;
  CK(cudaHostAlloc(&hx, NOUT * 4, cudaHostAllocDefault));
  CK(cudaHostAlloc(&hy, NOUT * 4, cudaHostAllocDefault));
  CK(cudaMalloc(&dx, NOUT * 4));
  CK(cudaMalloc(&dy, NOUT * 4));
  for (int i = 0; i < NOUT; i++) hx[i] = (float)i;
  CK(cudaMemcpy(dx, hx, NOUT * 4, cudaMemcpyHostToDevice));

  /* --- capture the graph: H2D, kernel, D2H, exactly the engine's pattern --- */
  cudaStream_t cs;
  CK(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking));
  cudaGraph_t g = NULL;
  cudaGraphExec_t ge = NULL;
  CK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal));
  cudaMemcpyAsync(dy, dx, NOUT * 4, cudaMemcpyDeviceToDevice, cs);
  tiny<<<8, 128, 0, cs>>>(dx, dy);
  cudaMemcpyAsync(hy, dy, NOUT * 4, cudaMemcpyDeviceToHost, cs);
  CK(cudaStreamEndCapture(cs, &g));
  CK(cudaGraphInstantiate(&ge, g, NULL, NULL, 0));
  printf("graph capture: OK\n");

  double tp = 0, tg = 0, tn = 0;
  for (int r = 0; r < reps; r++) {
    /* plain */
    double t0 = now();
    for (int i = 0; i < iters; i++) {
      CK(cudaMemcpy(dx, hx, NOUT * 4, cudaMemcpyHostToDevice));
      tiny<<<8, 128>>>(dx, dy);
      CK(cudaMemcpy(hy, dy, NOUT * 4, cudaMemcpyDeviceToHost));
      CK(cudaDeviceSynchronize());
    }
    double t1 = now();
    /* graph */
    double t2 = now();
    for (int i = 0; i < iters; i++) {
      CK(cudaGraphLaunch(ge, 0));
      CK(cudaDeviceSynchronize());
    }
    double t3 = now();
    /* lower bound: kernel + sync only */
    double t4 = now();
    for (int i = 0; i < iters; i++) {
      tiny<<<8, 128>>>(dx, dy);
      CK(cudaDeviceSynchronize());
    }
    double t5 = now();
    double a = (t1 - t0) / iters, b = (t3 - t2) / iters, c = (t5 - t4) / iters;
    tp += a; tg += b; tn += c;
    printf("rep%d plain=%7.2fus graph=%7.2fus launch+sync=%6.2fus  ratio plain/graph=%5.2f\n",
           r + 1, a, b, c, a / b);
  }
  printf("MEAN plain=%.2fus graph=%.2fus launch+sync=%.2fus  => saving %.2fus/call (%.1f%%)\n",
         tp / reps, tg / reps, tn / reps, (tp - tg) / reps,
         100.0 * (tp - tg) / tp);
  printf("at 145 calls/token and a 352us/token step: %.2f ms/token = %.1f%% of decode\n",
         145.0 * (tp - tg) / reps / 1000.0,
         100.0 * 145.0 * (tp - tg) / reps / 352000.0);
  CK(cudaGraphExecDestroy(ge));
  CK(cudaGraphDestroy(g));
  CK(cudaStreamDestroy(cs));
  return 0;
}
