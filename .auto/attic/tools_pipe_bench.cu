// tools_pipe_bench.cu -- which PIPE binds the batched Q8 prefill GEMV on sm_53?
//
// Why this exists: the ASR batched prefill (lfm3_nanode --prefill-run) runs 24 layers x
// 118 prompt positions at ~105 GFLOP/s, which is 44% of this part's fp32 peak (128 cores
// x 2 flop x 0.921 GHz = 236 GFLOP/s) and 76% of its measured single-SM streaming ceiling.
// Neither pipe looks saturated, so the arithmetic says the third one does: the int8 -> float
// CONVERSION. Per code the kernel does one cvt and one fma; if cvt runs at 1/4 rate, the
// conversion pipe alone accounts for the missing 2x and no amount of tiling, launch geometry
// or cache policy can move it (which is what the sweeps of the prefill knobs found).
//
// The candidate replacement: build the float from the byte's BITS instead of converting it.
//   fl(b) == __uint_as_float(0x4B000000u | b) - 8388608.0f     for 0 <= b <= 255
// (the "magic constant" trick: 2^23's bit pattern ORed with the byte holds the integer
// exactly in the mantissa). That is 2 full-rate integer ops (SHF + LOP3) on the ALU pipe
// instead of one quarter-rate conversion. It is not a numerical trick -- it produces the
// same float the cvt would -- but folding the -2^23 correction into the existing chain needs
// care (it is a per-block constant times sum(x), which is the form the Q8 epilogue already
// has), so this file measures the OP-MIX upside only. Run it before writing any kernel.
//
// Build exactly like the engine (the -fmad=false matters: it is what keeps the engine's
// chains uncontracted, and contraction would change the answer here):
//   nvcc $FLAGS -gencode arch=compute_53,code=sm_53 -O3 -fmad=false -o tools_pipe_bench tools_pipe_bench.cu
// Run: ./tools_pipe_bench [reps]
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define Q8_BLK 32
#define TPB 512
#define NROWS 4096            // rows in the synthetic tensor
#define KC 2048               // codes per row (one chunk: nb = 64 blocks of 32)
#define NPOS 4                // positions per block, exactly q8_partial_TLg's tile

#define CK(x) do { cudaError_t e = (x); if (e) { printf("CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)

// MODE 0: fma only          -- the FMA pipe's ceiling for this op mix
// MODE 1: cvt + fma         -- what the engine does today (int8 code -> float)
// MODE 2: bits + fma        -- the magic-constant form (2 ALU ops instead of 1 cvt)
// MODE 3: cvt only          -- price the conversion pipe by itself
// MODE 4: bits only         -- price the ALU form by itself
// MODE 5: u8 cvt + fma      -- the unsigned form (the bias is what MODE 2 folds away)
template <int MODE>
__global__ void __launch_bounds__(TPB, 2) gemv_like(const int8_t* __restrict__ q, const float* __restrict__ xb,
                                                    float* __restrict__ out, int nb, int reps) {
  __shared__ float xs[NPOS * KC];
  for (int i = threadIdx.x; i < NPOS * KC; i += TPB) xs[i] = xb[i];
  __syncthreads();
  const int row = blockIdx.x * TPB + threadIdx.x;
  // the engine's tiled layout: a warp's 32 loads are one contiguous 512 B line
  const int4* qt = reinterpret_cast<const int4*>(q) + (size_t)(row >> 5) * nb * 64 + (row & 31);
  float acc[NPOS];
  #pragma unroll
  for (int p = 0; p < NPOS; p++) acc[p] = 0.f;
  for (int r = 0; r < reps; r++) {
    for (int k = 0; k < 32; k++) {                  // 32 codes = one Q8_K block
      // the rotation by r keeps the loads out of the compiler's hands: without it the whole
      // k loop is invariant across reps and gets hoisted, and the benchmark times nothing
      // (which the first version did -- it reported 3x the hardware's fp32 peak)
      const int kk = (k + r) & 31;
      const int i = ((kk & 1) << 4) | ((kk & 2) << 2) | (kk & 4) | ((kk & 8) >> 2) | ((kk & 16) >> 4);
      const int4 cw = qt[(size_t)i * 32];           // 4 codes per lane, the engine's tile read
      const uint32_t w[4] = {(uint32_t)cw.x, (uint32_t)cw.y, (uint32_t)cw.z, (uint32_t)cw.w};
      #pragma unroll
      for (int p = 0; p < NPOS; p++) {
        const float4 x0 = *reinterpret_cast<const float4*>(xs + (size_t)p * KC + i * 8);
        const float4 x1 = *reinterpret_cast<const float4*>(xs + (size_t)p * KC + i * 8 + 4);
        const float xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          #pragma unroll
          for (int by = 0; by < 2; by++) {
            const int sh = j * 8;                    // two codes per word touched per step
            const uint32_t b = (w[j] >> (sh + 8 * by)) & 0xffu;
            if (MODE == 0) {                            // 1 FFMA, no extract, no cvt
              acc[p] = __fmaf_rn(xv[2 * j + by], 1.0f, acc[p]);
            } else if (MODE == 1) {                     // the engine's form
              acc[p] = __fmaf_rn((float)(int)(signed char)b - 128.f, xv[2 * j + by], acc[p]);
            } else if (MODE == 2) {                     // bits instead of cvt
              acc[p] = __fmaf_rn(__uint_as_float(0x4B000000u | b), xv[2 * j + by], acc[p]);
            } else if (MODE == 3) {                     // cvt, add instead of fma
              acc[p] = __fadd_rn(acc[p], (float)(int)(signed char)b);
            } else if (MODE == 4) {                     // bits, add instead of fma
              acc[p] = __fadd_rn(acc[p], __uint_as_float(0x4B000000u | b));
            } else {                                    // 5: unsigned cvt + fma
              acc[p] = __fmaf_rn((float)(int)(unsigned char)b, xv[2 * j + by], acc[p]);
            }
          }
        }
      }
    }
  }
  float s = 0.f;
  #pragma unroll
  for (int p = 0; p < NPOS; p++) s += acc[p];
  if (s == 1234.5f) out[row] = s;                     // keep the work alive
}

int main(int argc, char** argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 40;
  int dev = 0; cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
  printf("%s: %d SMs, clock %.0f MHz -> fp32 peak %.0f GFLOP/s\n", p.name, p.multiProcessorCount,
         p.clockRate / 1000.0, p.multiProcessorCount * 128.0 * 2 * p.clockRate / 1e6);
  int8_t* q; float* x; float* out;
  const size_t qn = (size_t)NROWS * KC, xn = (size_t)NPOS * KC;
  CK(cudaMalloc(&q, qn)); CK(cudaMalloc(&x, xn * 4)); CK(cudaMalloc(&out, NROWS * 4));
  {                                  // deterministic, non-degenerate codes and activations
    int8_t* hq = (int8_t*)malloc(qn);
    unsigned s = 12345;
    for (size_t i = 0; i < qn; i++) { s = s * 1103515245u + 12345u; hq[i] = (int8_t)((s >> 16) & 0xff); }
    CK(cudaMemcpy(q, hq, qn, cudaMemcpyHostToDevice)); free(hq);
    float* hx = (float*)malloc(xn * 4);
    for (size_t i = 0; i < xn; i++) hx[i] = (float)((i * 37) % 251) * 0.01f - 1.2f;
    CK(cudaMemcpy(x, hx, xn * 4, cudaMemcpyHostToDevice)); free(hx);
  }
  const int nb = KC / Q8_BLK;
  const int nblk = 8;                                 // one SM: 2 blocks x 512 threads
  // Clock ramp first, or every number is the devfreq's, not the kernel's: this part parks at
  // 76.8 MHz when idle and the whole binary finishes in a few hundred ms, so the modes that
  // happen to run first look 10x slower than the ones that run last. The first version of
  // this file reported three times the hardware's fp32 peak for exactly that reason. Spin the
  // conversion kernel for ~2 s before any timing, and take the best of several passes.
  { cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); float el = 0.f;
    for (int t = 0; t < 40 && el < 2000.f; t++) {
      cudaEventRecord(a); gemv_like<1><<<nblk, TPB>>>(q, x, out, nb, reps);
      cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&el, a, b);
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    printf("warm-up done (%.0f ms per launch at the ramped clock)\n", el);
  }
  const double macs = (double)nblk * TPB * KC * NPOS;   // MACs per rep of the grid
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  const char* nm[6] = {"fma only (no extract, no cvt)", "extract + cvt s8 + fma (ENGINE TODAY)",
                       "extract + bits + fma (magic const)", "extract + cvt s8 + add",
                       "extract + bits + add", "extract + cvt u8 + fma"};
  for (int m = 0; m < 6; m++) {
    float best = 1e30f;
    for (int t = 0; t < 7; t++) {
      cudaEvent_t c0, c1; cudaEventCreate(&c0); cudaEventCreate(&c1);
      cudaEventRecord(c0);
      switch (m) {
        case 0: gemv_like<0><<<nblk, TPB>>>(q, x, out, nb, reps); break;
        case 1: gemv_like<1><<<nblk, TPB>>>(q, x, out, nb, reps); break;
        case 2: gemv_like<2><<<nblk, TPB>>>(q, x, out, nb, reps); break;
        case 3: gemv_like<3><<<nblk, TPB>>>(q, x, out, nb, reps); break;
        case 4: gemv_like<4><<<nblk, TPB>>>(q, x, out, nb, reps); break;
        default: gemv_like<5><<<nblk, TPB>>>(q, x, out, nb, reps); break;
      }
      cudaEventRecord(c1); cudaEventSynchronize(c1);
      float ms; cudaEventElapsedTime(&ms, c0, c1); cudaEventDestroy(c0); cudaEventDestroy(c1);
      if (ms < best) best = ms;
    }
    printf("%-28s %8.2f ms  %6.1f GFLOP/s  %5.1f%% of fp32 peak\n", nm[m], best,
           2 * macs * reps / (best * 1e6), 2 * macs * reps / (best * 1e6) /
           (p.multiProcessorCount * 128.0 * 2 * p.clockRate / 1e6) * 100.0);
    CK(cudaGetLastError());
  }
  CK(cudaDeviceSynchronize());
  cudaFree(q); cudaFree(x); cudaFree(out);
  return 0;
}
