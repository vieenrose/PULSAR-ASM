// Standalone microbenchmark: what limits the H2 inner loop of q8_prefill_h2?
// Variants (same arithmetic per mode, so modes must agree bit-for-bit):
//   LOAD 0: each lane loads its own 32-byte Q8 block directly (current pattern)
//   LOAD 1: warp loads the row coalesced into shared memory, lanes read back
//   CHAIN 1: one hfma2 chain of 16 per (b,c)   CHAIN 2: two chains of 8 (ARM1-style)
// Plausibility gate: every output nonzero; achieved weight bandwidth < 25.6 GB/s.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda_fp16.h>

#define Q8_BLK 32
#define CT 4

template <int LOAD, int CHAIN>
__global__ void h2_bench(const int8_t* __restrict__ q, const __half2* __restrict__ xh,
                         float* __restrict__ out, int N, int K, int nb, int reps) {
  __shared__ uint32_t stage[8][32 * 8 * 4];   // per-warp staging: 32 blocks x 32 B = 1 KB (LOAD 1)
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, WPB = blockDim.x >> 5;
  const int row = blockIdx.x * WPB + wid;
  if (row >= N) return;
  const __half2 off = __floats2half2_rn(1152.f, 1152.f);
  float acc[CT];
  for (int c = 0; c < CT; c++) acc[c] = 0.f;
  const int8_t* qr = q + (size_t)row * K;
  for (int rep = 0; rep < reps; rep++) {
    for (int b0 = 0; b0 < nb; b0 += 32) {
      const int b = b0 + lane;
      uint32_t w[8];
      if (LOAD == 0) {
        if (b < nb) {
          const int4 a0 = *reinterpret_cast<const int4*>(qr + (size_t)b * Q8_BLK);
          const int4 a1 = *reinterpret_cast<const int4*>(qr + (size_t)b * Q8_BLK + 16);
          w[0]=a0.x; w[1]=a0.y; w[2]=a0.z; w[3]=a0.w; w[4]=a1.x; w[5]=a1.y; w[6]=a1.z; w[7]=a1.w;
        }
      } else {
        // coalesced: 32 lanes x 32 B = 1 KB contiguous; 4 int4 per lane, staged per warp
        const int nbl = (nb - b0 < 32) ? (nb - b0) : 32;
        const int8_t* base = qr + (size_t)b0 * Q8_BLK;
        for (int i = lane; i < nbl * 2; i += 32) {   // 2 int4 per 32-byte block
          const int4 v = *reinterpret_cast<const int4*>(base + (size_t)i * 16);
          stage[wid][i * 4 + 0] = v.x; stage[wid][i * 4 + 1] = v.y;
          stage[wid][i * 4 + 2] = v.z; stage[wid][i * 4 + 3] = v.w;
        }
        __syncwarp();
        if (b < nb) {
          const uint32_t* s = &stage[wid][lane * 8];   // block `lane` = 8 words
          for (int k = 0; k < 8; k++) w[k] = s[k];
        }
        __syncwarp();
      }
      if (b < nb) {
#pragma unroll
        for (int c = 0; c < CT; c++) {
          const __half2* xb2 = xh + ((size_t)c * nb + b) * 17;
          __half2 h = __floats2half2_rn(0.f, 0.f), h2 = h;
#pragma unroll
          for (int k = 0; k < 8; k++) {
            const uint32_t u = w[k] ^ 0x80808080u;
            const uint32_t lo = __byte_perm(u, 0x64646464u, 0x4140);
            const uint32_t hi = __byte_perm(u, 0x64646464u, 0x4342);
            const __half2 cl = __hsub2(*reinterpret_cast<const __half2*>(&lo), off);
            const __half2 ch = __hsub2(*reinterpret_cast<const __half2*>(&hi), off);
            if (CHAIN == 1) {
              h = __hfma2(cl, xb2[2 * k], h);
              h = __hfma2(ch, xb2[2 * k + 1], h);
            } else {
              h = __hfma2(cl, xb2[2 * k], h);
              h2 = __hfma2(ch, xb2[2 * k + 1], h2);
            }
          }
          const float2 f = __half22float2(__hadd2(h, h2));
          acc[c] += f.x + f.y;
        }
      }
    }
  }
  if (lane == 0)
    for (int c = 0; c < CT; c++) out[(size_t)row * CT + c] = acc[c];
}

static double run(int which, const int8_t* q, const __half2* xh, float* out, int N, int K, int nb,
                  int reps, float* ms_out) {
  const int TPB = 256, WPB = TPB / 32, grid = (N + WPB - 1) / WPB;
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  // warm-up, then time
  switch (which) {
    case 0: h2_bench<0,1><<<grid,TPB>>>(q,xh,out,N,K,nb,1); break;
    case 1: h2_bench<1,1><<<grid,TPB>>>(q,xh,out,N,K,nb,1); break;
    case 2: h2_bench<0,2><<<grid,TPB>>>(q,xh,out,N,K,nb,1); break;
    case 3: h2_bench<1,2><<<grid,TPB>>>(q,xh,out,N,K,nb,1); break;
  }
  cudaDeviceSynchronize();
  cudaEventRecord(a);
  switch (which) {
    case 0: h2_bench<0,1><<<grid,TPB>>>(q,xh,out,N,K,nb,reps); break;
    case 1: h2_bench<1,1><<<grid,TPB>>>(q,xh,out,N,K,nb,reps); break;
    case 2: h2_bench<0,2><<<grid,TPB>>>(q,xh,out,N,K,nb,reps); break;
    case 3: h2_bench<1,2><<<grid,TPB>>>(q,xh,out,N,K,nb,reps); break;
  }
  cudaEventRecord(b); cudaEventSynchronize(b);
  cudaError_t le = cudaGetLastError();
  float ms = 0; cudaEventElapsedTime(&ms, a, b);
  if (le) { printf("LAUNCH ERROR %s\n", cudaGetErrorString(le)); exit(2); }
  *ms_out = ms;
  return 0;
}

int main(int argc, char** argv) {
  const int N = 12288, K = 7680, nb = K / Q8_BLK, reps = 20;   // w2-like shape, KC = K
  std::vector<int8_t> hq((size_t)N * K);
  for (size_t i = 0; i < hq.size(); i++) hq[i] = (int8_t)((i * 73 + 11) % 255 - 127);
  std::vector<__half2> hx((size_t)CT * nb * 17);
  for (size_t i = 0; i < hx.size(); i++) hx[i] = __floats2half2_rn(0.001f * ((i * 37) % 101), 0.002f * ((i * 11) % 53));
  int8_t* dq; __half2* dx; float* dout;
  cudaMalloc(&dq, hq.size()); cudaMalloc(&dx, hx.size() * sizeof(__half2)); cudaMalloc(&dout, (size_t)N * CT * 4);
  cudaMemcpy(dq, hq.data(), hq.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dx, hx.data(), hx.size() * sizeof(__half2), cudaMemcpyHostToDevice);
  const char* names[4] = {"LOAD0 direct  CHAIN1", "LOAD1 coalesced CHAIN1",
                          "LOAD0 direct  CHAIN2", "LOAD1 coalesced CHAIN2"};
  std::vector<float> ref(N * CT), got(N * CT);
  int fails = 0;
  for (int v = 0; v < 4; v++) {
    float ms = 0; run(v, dq, dx, dout, N, K, nb, reps, &ms);
    cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost);
    long nz = 0; for (float f : got) nz += (f != 0.f);
    const double macs = (double)N * K * CT * reps;               // weights x positions
    const double gmac = macs / (ms * 1e-3) / 1e9;
    const double wbytes = (double)N * K * 1.125 * reps;          // Q8 weight stream
    const double gbs = wbytes / (ms * 1e-3) / 1e9;
    const char* plaus = (nz == 0) ? "PLAUSIBILITY FAIL (all-zero)" :
                        (gmac > 236.0) ? "PLAUSIBILITY FAIL (>236 GMAC/s)" : "ok";
    printf("%-24s %8.2f ms  %6.1f GMAC/s  %5.2f GB/s  nonzero %ld/%d  %s\n",
           names[v], ms, gmac, gbs, nz, N * CT, plaus);
    if (nz == 0 || gmac > 236.0) fails++;
    if (v == 0) ref = got;
    else {
      // LOAD modes with the same CHAIN must agree bit-for-bit
      if (v == 1 && memcmp(ref.data(), got.data(), got.size() * 4)) { printf("  LOAD1 vs LOAD0 CHAIN1: MISMATCH\n"); fails++; }
      else if (v == 1) printf("  LOAD1 vs LOAD0 CHAIN1: bit-exact\n");
    }
  }
  printf(fails ? "BENCH FAILED\n" : "BENCH OK\n");
  return fails ? 1 : 0;
}
