// Q8_0 GEMV bandwidth probe at the LFM3-Audio shapes, Jetson Nano (sm_53, 1 SM).
//
// Why: pq1_swar_sm53.cu documents its tuned kernels as "K in {4096, 12288}" and
// the dispatch gate is `N < 16 || (K != 4096 && K != 12288)`. Every LFM3 GEMV is
// K=2048 (conv in/out, q, k, v, o, gate, w1, w3) or K=7680 (w2) -- none match, so
// all of them take the fallback path and none of the 16.11 tok/s applies. Before
// spending days retuning for K=2048, measure whether a Q8_0 GEMV at K=2048 can
// reach the bus ceiling at all. If it can, the gap is tuning; if it can't, the
// LFM3 memory story is dead on arrival.
//
// Q8_0 block = 32 int8 codes + 1 f16 scale (34 bytes per 32 weights), the same
// layout the engine repacks into. Reported GB/s is per weight-byte streamed
// (1 byte/weight), so it is directly comparable to the repo's 21.9 GB/s ceiling.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define WK 256

// One warp per row, K/32 blocks per warp-chunk; classic Q8_0 dequant-dot.
__global__ void q8_gemv(const int8_t* __restrict__ w, const __half* __restrict__ sc,
                        const float* __restrict__ x, float* __restrict__ out,
                        int N, int K) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int nw = blockDim.x >> 5;
  const int wid = threadIdx.x >> 5;
  if (row >= N) return;
  const int nblk = K >> 5;                       // 32 weights per Q8_0 block
  float sum = 0.f;
  for (int b = wid; b < nblk; b += nw) {
    // Each lane owns 4 CONSECUTIVE weights, so the int load is 4-byte aligned.
    // Indexing with +lane (stride 1) makes it misaligned and Maxwell takes the
    // whole launch down with "unspecified launch failure".
    const int8_t* qw = w + (size_t)row * K + (size_t)b * 32 + lane * 4;
    const float* qx = x + (size_t)b * 32 + lane * 4;
    float s = __half2float(sc[(size_t)row * nblk + b]);
    int q = *reinterpret_cast<const int*>(qw);
    float d = (float)(signed char)(q & 0xff)         * qx[0]
            + (float)(signed char)((q >> 8) & 0xff)  * qx[1]
            + (float)(signed char)((q >> 16) & 0xff) * qx[2]
            + (float)(signed char)((q >> 24) & 0xff) * qx[3];
    sum += s * d;
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(~0u, sum, o);
  if (lane == 0) atomicAdd(&out[row], sum);
}

// Pure read-bandwidth ceiling: stream the same weight array, no math.
__global__ void readbw(const int8_t* __restrict__ w, const __half* __restrict__ sc,
                       size_t nbyte, float* __restrict__ sink) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  float acc = 0.f;
  for (; i * 4 < nbyte; i += stride)
    acc += (float)(w[i * 4] + w[i * 4 + 1] + w[i * 4 + 2] + w[i * 4 + 3]);
  if (acc == 12345.67f) *sink = acc;               // keep the loads live
}

static float bench(void (*launch)(void*), void* ctx, int iters) {
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  launch(ctx);
  // Prove the kernel actually launched. A silent launch failure measures 0 ms
  // and prints an absurd GB/s, which is how this probe first presented.
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    printf("  LAUNCH ERROR: %s\n", cudaGetErrorString(e));
    cudaGetLastError();
  }
  cudaEventRecord(a);
  for (int i = 0; i < iters; i++) launch(ctx);
  cudaEventRecord(b); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms, a, b);
  cudaEventDestroy(a); cudaEventDestroy(b);
  return ms / iters;
}

struct Ctx { const int8_t* w; const __half* sc; const float* x; float* out; int N, K;
             size_t nbyte; float* sink; int grid; int block; };
static void l_gemv(void* p){ Ctx& c=*(Ctx*)p; q8_gemv<<<c.N, c.block>>>(c.w,c.sc,c.x,c.out,c.N,c.K); }
static void l_read(void* p){ Ctx& c=*(Ctx*)p; readbw<<<c.grid, c.block>>>(c.w,c.sc,c.nbyte,c.sink); }

int main() {
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  printf("%s  cc%d.%d  SMs=%d  mem=%.2f GB\n", p.name, p.major, p.minor,
         p.multiProcessorCount, p.totalGlobalMem / 1e9);

  // The real LFM3 projection shapes (N rows, K reduction).
  struct S { const char* n; int N, K; };
  S shapes[] = {
    {"conv_in_proj  13x", 6144, 2048}, {"conv_out_proj 13x", 2048, 2048},
    {"w_q           11x", 2048, 2048}, {"w_k            11x",  512, 2048},
    {"w_o           11x", 2048, 2048}, {"w1/w3          24x", 7680, 2048},
    {"w2            24x", 2048, 7680},
    {"REF K=4096 (bonsai)",         4096, 4096},
    {"REF K=12288 (bonsai)",        4096, 12288},
  };
  const int MAXN = 8192, MAXK = 12288;
  int8_t* w; __half* sc; float *x, *out, *sink;
  // Every allocation checked: a NULL pointer here shows up as a 0 ms kernel
  // in the table, not as a crash.
  struct { void** p; size_t n; const char* w; } allocs[] = {
    {(void**)&w, (size_t)MAXN * MAXK, "w"},
    {(void**)&sc, (size_t)MAXN * (MAXK / 32) * 2, "sc"},
    {(void**)&x, (size_t)MAXK * 4, "x"},
    {(void**)&out, (size_t)MAXN * 4, "out"},
    {(void**)&sink, 4, "sink"},
  };
  for (auto& A : allocs) {
    cudaError_t e = cudaMalloc(A.p, A.n);
    if (e != cudaSuccess) { printf("cudaMalloc %s (%zu B) FAILED: %s\n",
                                  A.w, A.n, cudaGetErrorString(e)); return 1; }
  }
  size_t wsz = (size_t)MAXN < (size_t)8192 * 12288 ? (size_t)MAXN * MAXK : (size_t)MAXN * MAXK;
  cudaMemset(w, 0x11, wsz);
  cudaMemset(sc, 0x3c, (size_t)MAXN * (MAXK / 32) * 2);
  cudaMemset(x, 0x3c, (size_t)MAXK * 4);
  cudaMemset(out, 0, (size_t)MAXN * 4);
  cudaError_t le = cudaGetLastError();
  if (le != cudaSuccess) { printf("setup error: %s\n", cudaGetErrorString(le)); return 1; }

  printf("\n%-22s %6s %6s %9s %9s %9s\n",
         "shape", "N", "K", "MB/tok", "ms/call", "GB/s");
  double tot_ms = 0;
  for (S& s : shapes) {
    size_t nbyte = (size_t)s.N * s.K;
    Ctx c{w, sc, x, out, s.N, s.K, nbyte, sink,
          128 * p.multiProcessorCount, 256};
    float ms_g = bench(l_gemv, &c, 20);
    float ms_r = bench(l_read, &c, 20);
    if (s.N < 8192) tot_ms += ms_g;                 // skip the two reference rows
    printf("%-22s %6d %6d %9.1f %9.3f %9.2f   (read ceiling %.2f GB/s)\n",
           s.n, s.N, s.K, nbyte / 1e6, ms_g, nbyte / (ms_g * 1e6),
           nbyte / (ms_r * 1e6));
  }
  // Per-token total for the 24-layer backbone, weights only.
  printf("\nper-token GEMV total (24 layers, weights only): %.2f ms -> %.1f tok/s\n",
         tot_ms, 1000.0 / tot_ms);
  printf("backbone weights 1728.8 M: Q8_0 = 2.082 GB/token, Q4 = 1.041 GB/token\n");
  cudaFree(w); cudaFree(sc); cudaFree(x); cudaFree(out); cudaFree(sink);
  return 0;
}
