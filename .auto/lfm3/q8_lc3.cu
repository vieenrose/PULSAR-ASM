// q8_lc.cu -- "lane-strided chunked" Q8_0 GEMV: the first kernel in this
// chain that can cover K=7680 AT ALL, plus a sweep over its configs.
//
// WHY A NEW DESIGN WAS NEEDED.
// f878571 established the structural blocker. 5fd7e79's split-K puts nb
// threads on one row and reduces with a 32-lane butterfly, so nb must be a
// multiple of 32, so KCHUNK must be a multiple of 1024 -- and no multiple of
// 1024 divides 7680 (7680 = 32 x 240, and 240 % 32 = 16). Its "12.88 GB/s"
// was a kernel that read 6144 of 7680 columns. down w2 is 22% of decode, so
// this is not a corner case.
//
// WHAT CHANGED HERE.
// Take the row's reduction from split-K's nb-threads-per-row back to kg's
// WARP per row with lanes striding over the row's blocks. A warp is 32 lanes
// by construction, so the butterfly is always well-formed and nb can be ANY
// integer -- no multiple-of-32 requirement, and 7680/32 = 240 is a legal
// block count. Then keep split-K's real win: the x vector is staged once in
// shared and amortized over R rows, which is exactly what kg could not do
// (kg re-reads x from L1 per row and runs at 5.96 GB/s).
//
// SCALE PRECISION IS A CORRECTNESS CHOICE, NOT A DETAIL.
// The certified 8/8 arm of quant_sensitivity.py ("q8_0", groups=all) computes
//   s = max|w| / 127  in fp32,  codes = round(w/s) clamped to +-128.
// The pq1/ggml Q8_0 convention stores s as __half. quant_sensitivity.py now
// also has that arm ("q8_0h", scale16=True) and it is NOT the same result:
// 7/8 exact, prompt F diverging at token 6 of 24 (mean prefix 20.9), because
// fp16 has 11 mantissa bits and the scale is then also used to derive the
// codes. So an fp16-scale engine is not the numerics that were certified.
// This kernel therefore takes const float* scales and clamps codes to
// -128..127, i.e. it reproduces the certified arm element for element. Cost:
// 1.0625 -> 1.125 bytes/param (+5.9%, 91 -> 96 MB) and one extra 4-byte load
// per 32 codes, which is bandwidth-negligible next to the codes themselves.
//
//   kg:         warp/row, lanes stride, no x staging            5.96 GB/s
//   split-K:    nb threads/row, x staged, needs nb%32==0        n/a for 7680
//   this:       warp/row, lanes stride, x staged, nb arbitrary  (target 10+)
//
// BIT-EXACTNESS CONTRACT.
// The kernel defines the order; the reference mirrors it per element:
//   per row, per chunk c (ascending):
//     per lane l (0..31): acc_l = 0
//       for b = l, l+32, l+64, ... < nb (ascending):
//         dot = sum_{j<32} code[j]*x[b*32+j]   (j ascending, __fadd_rn, no FMA)
//         acc_l = __fadd_rn(acc_l, __fmul_rn(scale[b], dot))
//     32-lane butterfly: v += shfl_down(v,16), then 8,4,2,1
//     chunk_c = butterfly result in lane 0
//   y = sum over chunks ascending (__fadd_rn)
// The lane-to-block map is lane-strided, NOT contiguous: lane l owns blocks
// l, l+32, ... A contiguous split (lane owns blocks l*nb/32..) has a
// DIFFERENT rounding and is not bit-exact to this.
//
// Build on the Nano (CUDA 10.2):
//   nvcc -gencode arch=compute_53,code=sm_53 -O3 -fmad=false \
//        -Xptxas -O3,-dlcm=ca -o q8_lc q8_lc.cu
//   ./q8_lc
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cuda_fp16.h>

#define Q8_BLK 32
#define RMAX 256
#define WPBMAX 16          // warps per block <= 16 (TPB <= 512)

// Mirrors quant_sensitivity.py's CERTIFIED arm exactly: s = max|w|/127 in
// fp32 (no fp16 rounding), codes = round(w/s) clamped to -128..127.
static void quant_row(const float* w, int K, int8_t* q, float* s) {
  const int nb = K / Q8_BLK;
  for (int b = 0; b < nb; b++) {
    float mx = 0.f;
    for (int j = 0; j < Q8_BLK; j++) mx = fmaxf(mx, fabsf(w[b * Q8_BLK + j]));
    float sc = mx / 127.f;                       // fp32, as certified
    if (sc == 0.f) { memset(q + (size_t)b * Q8_BLK, 0, Q8_BLK); s[b] = 0.f; continue; }
    for (int j = 0; j < Q8_BLK; j++) {
      int iv = (int)lrintf(w[b * Q8_BLK + j] / sc);   // RN, == torch.round
      if (iv > 127) iv = 127; if (iv < -128) iv = -128;
      q[(size_t)b * Q8_BLK + j] = (int8_t)iv;
    }
    s[b] = sc;
  }
}

// pass 1: R rows x one chunk of nb blocks. One WARP per row, lanes stride
// over the chunk's blocks. blockDim.x = 32*WPB, R must be a multiple of WPB.
__global__ void q8_lc(const int8_t* __restrict__ q, const float* __restrict__ s,
                      const float* __restrict__ x, float* __restrict__ part,
                      int N, int K, int KCHUNK, int R, int nb) {
  extern __shared__ float xs[];                  // nb * 33
  const int nchunk = K / KCHUNK;
  const int rowblk = blockIdx.x / nchunk;
  const int chunk  = blockIdx.x % nchunk;
  const int koff   = chunk * KCHUNK;
  const int NT     = blockDim.x;
  const int tid    = threadIdx.x;
  const int WPB    = NT >> 5;
  const float* xg = x + koff;
  for (int i = tid; i < KCHUNK; i += NT) {
    int b = i >> 5, j = i & 31;
    xs[b * 33 + j] = xg[i];
  }
  __syncthreads();

  __shared__ float sh[RMAX];                     // one partial per row of this block
  const int base = rowblk * R;
  const int lane = tid & 31, wid = tid >> 5;
  const int GR = R / WPB;
  #pragma unroll 1
  for (int g = 0; g < GR; g++) {
    const int r = wid + g * WPB;                 // one warp per row
    const int row = base + r;
    float acc = 0.f;
    if (row < N) {
      const int8_t* qr = q + (size_t)row * K + koff;
      const float* sr = s + (size_t)row * (K / Q8_BLK) + (size_t)chunk * nb;
      #pragma unroll 1
      for (int b = lane; b < nb; b += 32) {      // lane-strided, ascending
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
  if (tid < R && base + tid < N)
    part[(size_t)(base + tid) * nchunk + chunk] = sh[tid];
}

__global__ void q8_combine(const float* __restrict__ part, float* __restrict__ out,
                           int N, int nchunk) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= N) return;
  float acc = 0.f;
  for (int c = 0; c < nchunk; c++) acc = __fadd_rn(acc, part[(size_t)r * nchunk + c]);
  out[r] = acc;
}

// bit-exact reference: mirrors the contract in the file header exactly.
// One thread per (row, chunk); the 32-lane state is a local array so the
// butterfly is literal.
__global__ void ref_lc(const int8_t* q, const float* s, const float* x,
                       float* part, int N, int K, int KCHUNK, int nb) {
  const int nchunk = K / KCHUNK;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = idx / nchunk, c = idx % nchunk;
  if (r >= N) return;
  const int koff = c * KCHUNK;
  float v[32];
  for (int l = 0; l < 32; l++) v[l] = 0.f;
  for (int l = 0; l < 32; l++) {
    for (int b = l; b < nb; b += 32) {
      float dot = 0.f;
      for (int j = 0; j < Q8_BLK; j++)
        dot = __fadd_rn(dot, __fmul_rn((float)q[(size_t)r * K + koff + (size_t)b * Q8_BLK + j],
                                       x[koff + (size_t)b * Q8_BLK + j]));
      v[l] = __fadd_rn(v[l], __fmul_rn(s[(size_t)r * (K / Q8_BLK) + (size_t)c * nb + b], dot));
    }
  }
  for (int o = 16; o > 0; o >>= 1) {
    float t[32];
    for (int l = 0; l < 32; l++) t[l] = __fadd_rn(v[l], v[l + o < 32 ? l + o : l]);
    for (int l = 0; l < 32; l++) v[l] = t[l];
  }
  part[(size_t)r * nchunk + c] = v[0];
}

int main(void) {
  struct Shape { int N, K; const char* name; int ncalls; };
  // lm_head is TIED to embed_tokens (config tie_word_embeddings=true, and
  // the safetensors census lists no lm_head), so every token streams the
  // whole 128000x2048 table. It was absent from every earlier budget.
  Shape shapes[] = {
    {128000, 2048, "lm_head(tied embed)", 1},
    {2048, 7680, "down w2",      24},
    {7680, 2048, "gate+up w1,w3", 48},
    {6144, 2048, "conv_in_proj",  13},
    {2048, 2048, "conv_out/w_q/gate_proj/w_o", 11},
    {2048,  512, "w_k/w_v",       11},
  };
  const int NSH = (int)(sizeof shapes / sizeof *shapes);
  const int KCH[] = {2048, 1024, 2560, 3840, 7680, 1536, 512};
  const int NKCH = (int)(sizeof KCH / sizeof *KCH);
  const int TPB[] = {128, 256, 512};
  const int NTPB = (int)(sizeof TPB / sizeof *TPB);
  const int RR[] = {32, 64, 128, 256};
  const int NRR = (int)(sizeof RR / sizeof *RR);

  // max over shapes of N*K, NOT maxN*maxK: the latter multiplies the
  // embedding's 128000 rows by the FFN's 7680 columns = 983 M elements, a
  // 3.9 GB host allocation on a 4 GB board.
  int maxN = 0, maxK = 0; size_t wmax = 0; int maxnb = 0;
  for (int i = 0; i < NSH; i++) {
    if (shapes[i].N > maxN) maxN = shapes[i].N;
    if (shapes[i].K > maxK) maxK = shapes[i].K;
    size_t wk = (size_t)shapes[i].N * shapes[i].K;
    if (wk > wmax) wmax = wk;
    if (shapes[i].K / Q8_BLK > maxnb) maxnb = shapes[i].K / Q8_BLK;
  }
  float* hw = (float*)malloc(wmax * 4);
  int8_t* hq = (int8_t*)malloc(wmax);
  float* hs = (float*)malloc((size_t)maxN * maxnb * 4);
  float* hx = (float*)malloc((size_t)maxK * 4);
  float* ay = (float*)malloc((size_t)maxN * 4);
  // ar holds the per-CHUNK partials (N * nchunk), not the final y: with
  // KCHUNK=128 on K=2048, nchunk=16, so a maxN*4 allocation overflowed the
  // heap and the process dumped core before printing anything.
  float* ar = (float*)malloc((size_t)maxN * 64 * 4);
  int8_t *dq; float *ds, *dx, *dy, *dpart, *dref;
  for (size_t i = 0; i < wmax; i++) hw[i] = (float)(((int)(i * 2654435761u) >> 8) % 4001 - 2000) / 500.f;
  cudaMalloc(&dq, wmax);
  cudaMalloc(&ds, (size_t)maxN * maxnb * 4);
  cudaMalloc(&dx, (size_t)maxK * 4);
  cudaMalloc(&dy, (size_t)maxN * 4);
  cudaMalloc(&dpart, (size_t)maxN * 64 * 4);
  cudaMalloc(&dref, (size_t)maxN * 64 * 4);
  int dev; cudaGetDevice(&dev);
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
  printf("GPU: %s  SMs=%d  maxThreadsPerSM=%d  sharedPerBlock=%zu  sharedPerSM=%zu\n\n",
         prop.name, prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor,
         prop.sharedMemPerBlock, prop.sharedMemPerMultiprocessor);

  double tot_new = 0;
  for (int si = 0; si < NSH; si++) {
    const int N = shapes[si].N, K = shapes[si].K, NC = shapes[si].ncalls;
    for (int r = 0; r < N; r++)
      quant_row(hw + (size_t)r * K, K, hq + (size_t)r * K, hs + (size_t)r * (K / Q8_BLK));
    cudaMemcpy(dq, hq, (size_t)N * K, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs, (size_t)N * (K / Q8_BLK) * 4, cudaMemcpyHostToDevice);
    for (int i = 0; i < K; i++) hx[i] = (float)(((int)(i * 40503u) >> 4) % 2001 - 1000) / 700.f;
    cudaMemcpy(dx, hx, (size_t)K * 4, cudaMemcpyHostToDevice);
    const double mb = (N * (double)K + (size_t)N * (K / Q8_BLK) * 4 + (size_t)K * 4 + (size_t)N * 4) / 1e6;
    printf("=== %s  N=%d K=%d  %d/token  (%.1f MB) ===\n", shapes[si].name, N, K, NC, mb);
    printf("%8s %4s %4s %7s %9s %8s  %s\n", "KCHUNK", "TPB", "R", "blocks", "ms", "GB/s", "bit-exact");
    double bestms = 1e30; int bk = 0, bt = 0, br = 0;
    int nchecked = 0, nskip = 0, nfail = 0;

    for (int ki = 0; ki < NKCH; ki++) {
      const int KCHUNK = KCH[ki];
      // THE ASSERTION THAT WAS MISSING IN 5fd7e79: a chunk width that does
      // not divide K silently truncates K (integer nchunk) and both kernel
      // and mirrored reference agree on the truncated answer. Check
      // coverage, do not compute a number.
      if (KCHUNK > K || K % KCHUNK != 0) continue;
      const int nb = KCHUNK / Q8_BLK;
      const int nchunk = K / KCHUNK;
      if (nchunk * KCHUNK != K) { printf("  coverage FAIL KCHUNK=%d\n", KCHUNK); continue; }
      if (nchunk > 64 || nb < 1) continue;
      ref_lc<<<((size_t)N * nchunk + 255) / 256, 256>>>(dq, ds, dx, dref, N, K, KCHUNK, nb);
      cudaMemcpy(ar, dref, (size_t)N * nchunk * 4, cudaMemcpyDeviceToHost);

      for (int ti = 0; ti < NTPB; ti++) {
        for (int ri = 0; ri < NRR; ri++) {
          const int NT = TPB[ti], R = RR[ri];
          const int WPB = NT / 32;
          if (WPB > WPBMAX || R % WPB) { nskip++; continue; }   // one warp per row
          if (R > RMAX) { nskip++; continue; }
          const size_t shm = (size_t)nb * 33 * 4;
          if (shm > prop.sharedMemPerBlock) { nskip++; continue; }
          const int nblk = ((N + R - 1) / R) * nchunk;
          const int reps = (int)(1.5e8 / (double)((size_t)N * K)) + 2;

          q8_lc<<<nblk, NT, shm>>>(dq, ds, dx, dpart, N, K, KCHUNK, R, nb);
          q8_combine<<<(N + 255) / 256, 256>>>(dpart, dy, N, nchunk);
          if (cudaGetLastError() != cudaSuccess) { nfail++; continue; }
          cudaDeviceSynchronize();
          if (cudaGetLastError() != cudaSuccess) { nfail++; continue; }
          for (int w = 0; w < 10; w++) {
            q8_lc<<<nblk, NT, shm>>>(dq, ds, dx, dpart, N, K, KCHUNK, R, nb);
            q8_combine<<<(N + 255) / 256, 256>>>(dpart, dy, N, nchunk);
          }
          cudaDeviceSynchronize();
          cudaEvent_t ev0, ev1;
          cudaEventCreate(&ev0); cudaEventCreate(&ev1);
          double best = 1e30;
          for (int t = 0; t < 3; t++) {
            cudaEventRecord(ev0);
            for (int r2 = 0; r2 < reps; r2++) {
              q8_lc<<<nblk, NT, shm>>>(dq, ds, dx, dpart, N, K, KCHUNK, R, nb);
              q8_combine<<<(N + 255) / 256, 256>>>(dpart, dy, N, nchunk);
            }
            cudaEventRecord(ev1);
            cudaEventSynchronize(ev1);
            float msf = 0.f; cudaEventElapsedTime(&msf, ev0, ev1);
            if (msf / reps < best) best = msf / reps;
          }
          cudaEventDestroy(ev0); cudaEventDestroy(ev1);

          cudaMemcpy(ay, dy, (size_t)N * 4, cudaMemcpyDeviceToHost);
          // combine the reference the same way the kernel does, then compare
          float bad = 0; long nbad = 0;
          for (int i = 0; i < N; i++) {
            float acc = 0.f;
            // plain host adds: a loop-carried sum has no fusion opportunity,
            // so aarch64 gives IEEE RN add == __fadd_rn bit for bit (and
            // __fadd_rn is __device__-only, so it cannot be called here).
            for (int c = 0; c < nchunk; c++) acc = acc + ar[(size_t)i * nchunk + c];
            if (acc != ay[i]) { nbad++; if (fabsf(acc) > bad) bad = fabsf(acc); }
          }
          const double gb = mb / (best * 1e-3) / 1e6 * 1e3;
          printf("%8d %4d %4d %7d %9.3f %8.2f  %s\n", KCHUNK, NT, R, nblk, best, gb,
                 nbad ? "NO" : "YES");
          nchecked++;
          if (!nbad && best < bestms) { bestms = best; bk = KCHUNK; bt = NT; br = R; }
        }
      }
    }
    if (bestms < 1e29) {
      printf("--> BEST %s: KCHUNK=%d TPB=%d R=%d  %.3f ms  %.2f GB/s  x%d = %.1f ms/token  (%d cfg, %d skip, %d fail)\n\n",
             shapes[si].name, bk, bt, br, bestms, mb / (bestms * 1e-3) / 1e6 * 1e3,
             NC, NC * bestms, nchecked, nskip, nfail);
      tot_new += NC * bestms;
    } else {
      printf("--> %s: NO VALID CONFIG (checked %d, skipped %d, failed %d)\n\n",
             shapes[si].name, nchecked, nskip, nfail);
    }
  }
  printf("all-shape decode GEMV total (this kernel, best cfg each) = %.1f ms -> %.2f tok/s\n",
         tot_new, 1000.0 / tot_new);
  printf("LC_DONE\n");
  return 0;
}
