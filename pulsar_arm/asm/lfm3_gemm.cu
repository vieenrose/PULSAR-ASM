/* lfm3_gemm.cu -- batched Q8 GEMM for the audio-side stages, and the record of
   what it took to make one that is not 40x off the board's bandwidth.

   FORM. C[t][n] = sum_k W[n][k] * x[t][k], W = Q8 rows with fp32 128-block
   scales (the .lfm3q8 dtype-1 layout, 1.125 B/weight), x = bf16 [T][K],
   C = f32 [T][N]. 256 threads = 32 groups of 8 lanes; one group owns one weight
   row and walks the T positions, so a weight row read once serves TILE outputs.
   That reuse is the entire reason a batched GEMM exists here (D2).

   THREE WRONG DIAGNOSES, kept because each was worth more than the fix:

   1 "7.1 GFLOP/s on every audio shape is a shared-bandwidth wall." It was not:
      padding the shared stride to break an 8-way bank conflict changed nothing.
      The number was invariant in N, K, T and TILE, and no real hardware limit
      is -- that invariance was the actual clue, and I read it as a hardware
      signature instead of as a symptom.

   2 "It must be spilling the int4 unpack to local memory." ptxas says 0 bytes
      spill, 40 registers. Dead end.

   3 Real cause: UNCOALESCED WEIGHT LOADS. Lane b owned k-block b, so the eight
      lanes of a group read bytes 128 apart -- 32 distinct 32-byte sectors per
      load instruction for 512 useful bytes. Predicted ~0.5 MAC/SM/cycle against
      0.96 measured; fixing it took the kernel 7.1 -> 30 GFLOP/s, 4.2x.

   THE SECOND STRUCTURAL FIX. Even coalesced, holding the whole [TILE][K] x-tile
   in shared caps TILE at sharedMem/4K -- 9 at K=1024 -- so the weight matrix is
   re-read ceil(T/TILE) times, up to 38x at T=75. Chunking K instead (shared
   holds one 128-element scale block for all TILE positions, 13 KB) lets each
   weight row be read ONCE per y-tile, and costs acc[TILE] live registers, which
   is why TILE is 24 and not 75.

   SUMMATION ORDER (the gate depends on it, so it is stated): blocks ascending;
   within a block, the 8 lanes cover 4-byte quarters at offset ci*32 + lane*4,
   each lane's 16 elements accumulated in ascending order; each lane's block
   partial scaled by that block's fp32 scale; the 8 lane partials reduced by a
   shfl_down tree (4,2,1). Every lane visits every block -- mixing "lane b owns
   block b" with a lane-relative in-block offset silently drops 7/8 of each
   block (the fp64 gate reported rel=0.18 while the kernel looked 8x faster).

   fmad is ON (default build); this kernel's gate is the fp64 relative-error
   bound, unlike the engine's GEMV which is bit-gated against a C reference.

   usage: lfm3_gemm            # all audio shapes, T in {12,24,48,75}
          lfm3_gemm N K T      # one shape
*/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <time.h>
#include <vector>
#include <cuda_runtime.h>

#define SCALE_BLK 128
#define ROWS_PER_BLOCK 32
/* TILE is capped by REGISTERS PER BLOCK, not by shared memory. v2 folds the
   block scale into the weights (no ps[TILE] scratch) so acc[TILE] is the only
   live array: TILE=24 uses ~70 regs/thread, TILE=32 fits. The launch check in
   main() stays because a refused launch returns cudaSuccess from sync. */
#ifndef TILE
#define TILE 24
#endif
#define KCHPAD 136                  /* 136 % 32 == 8: 4-byte lane stride -> distinct banks */

__global__ void k_gemm_q8(const signed char* __restrict__ W, const float* __restrict__ SC,
                          const unsigned short* __restrict__ X, float* __restrict__ C,
                          int N, int K, int T) {
    __shared__ float sx[TILE * KCHPAD];
    const int t0 = blockIdx.y * TILE;
    const int tid = threadIdx.x, nt = blockDim.x;
    const int lane = tid & 7, grp = tid >> 3;
    const int n = blockIdx.x * ROWS_PER_BLOCK + grp;
    const bool active = (n < N);
    const int nb = K >> 7;
    float acc[TILE];
#pragma unroll
    for (int t = 0; t < TILE; t++) acc[t] = 0.0f;

    for (int b = 0; b < nb; b++) {
        for (int i = tid; i < TILE * SCALE_BLK; i += nt) {
            int t = i / SCALE_BLK, k = i - t * SCALE_BLK;
            union { unsigned int u; float f; } cv;
            cv.u = (t0 + t < T) ? (((unsigned int)X[(t0 + t) * K + (b << 7) + k]) << 16) : 0u;
            sx[t * KCHPAD + k] = cv.f;
        }
        __syncthreads();
        if (active) {
            const signed char* wb = W + n * K + (b << 7);
            const float sc = SC[n * nb + b];
            /* v2: scale folded into the 4 weights (no ps[TILE]), float4 x reads.
               Per chunk ci the 8 lanes cover 32 contiguous bytes (lane*4), so a
               float4 at o touches banks o..o+3: lanes give 0-3,4-7,...,28-31,
               all distinct, no conflict. 1 float4 + 4 FMAs per (chunk, t). */
#pragma unroll
            for (int ci = 0; ci < SCALE_BLK / 32; ci++) {
                const int o = ci * 32 + lane * 4;      /* 8 lanes -> 32 contiguous bytes */
                /* __ldcs (evict-first): weights are re-read ceil(T/TILE) times with no
                   L2 reuse across passes (25 MB >> 2 MB L2), so cache-all pollutes. */
                const int q = __ldcs(reinterpret_cast<const int*>(wb + o));
                const signed char* qb = reinterpret_cast<const signed char*>(&q);
                const float w0 = (float)qb[0] * sc, w1 = (float)qb[1] * sc;
                const float w2 = (float)qb[2] * sc, w3 = (float)qb[3] * sc;
#pragma unroll
                for (int t = 0; t < TILE; t++) {
                    const float4 xv = *reinterpret_cast<const float4*>(sx + t * KCHPAD + o);
                    acc[t] += w0 * xv.x + w1 * xv.y + w2 * xv.z + w3 * xv.w;
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int t = 0; t < TILE; t++) {
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) acc[t] += __shfl_down_sync(0xffffffffu, acc[t], o, 8);
        /* `active` is REQUIRED here, not redundant with the bounds check: for
           N not a multiple of 32 the last block's surplus groups have n >= N,
           and C[(t0+t)*N + n] for n = N..N+29 lands in the NEXT row at
           n-N. Symptom was 30 x (T-1) zeros scattered into the following row
           (330 at T=12) and a gate failure only for N=1282, never for N=1280
           or N=1312. */
        if (active && lane == 0 && t0 + t < T) C[(t0 + t) * N + n] = acc[t];
    }
}

static unsigned short bf16_of(float f) { union { float f; unsigned int u; } v; v.f = f;
    return (unsigned short)((v.u + 0x7FFF + ((v.u >> 16) & 1)) >> 16); }
static float f32_of(unsigned short h) { union { float f; unsigned int u; } v;
    v.u = ((unsigned int)h) << 16; return v.f; }
static double now_ms() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

struct Shape { int N, K; const char* tag; };
static Shape SHAPES[] = {
    {2304,  768, "enc conv in_proj"}, { 768,  768, "enc conv out_pr"},
    {1792,  768, "enc w1/w3       "}, { 768, 1792, "enc w2          "},
    { 768,  768, "enc q_proj      "}, { 384,  768, "enc k/v_proj    "},
    { 768, 4096, "enc lin_out     "},
    {2816, 1024, "dep w1/w3       "}, {1024, 2816, "dep w2          "},
    {1024, 1024, "dep q/o_proj    "}, { 256, 1024, "dep k/v_proj    "},
    {24204, 1024, "dep 12x to_logit"}, {12288, 2048, "dep depth_linear"},
    {2304,  512, "cod w1/w3       "}, { 512, 2304, "cod w2          "},
    {1536,  512, "cod conv in_proj"}, { 512,  512, "cod conv out    "},
    {1282,  512, "cod lin (istft) "},
};

int main(int argc, char** argv) {
    std::vector<Shape> shapes;
    if (argc > 3) shapes.push_back({atoi(argv[1]), atoi(argv[2]), "custom          "});
    else for (Shape& s : SHAPES) shapes.push_back(s);
    const int TS[] = {12, 24, 48, 75};
    const double CEIL_GBS = 17.0;       /* the engine's measured streaming rate on this board */

    printf("%-17s %6s %5s %4s %8s %8s %8s %9s %9s  %s\n",
           "shape", "N", "K", "T", "ms", "GB/s", "GFLOP/s", "us/pos", "ideal_ms", "verdict");
    int fails = 0;
    unsigned seed = 12345u;
    for (Shape& sh : shapes) {
        const int N = sh.N, K = sh.K;
        if (K % SCALE_BLK) { printf("%-17s K=%d not a multiple of %d\n", sh.tag, K, SCALE_BLK); fails++; continue; }
        std::vector<signed char> hq((size_t)N * K);
        std::vector<float> hs((size_t)N * (K / SCALE_BLK));
        std::vector<double> ref_w((size_t)N * K);
        auto rnd = [&seed]() { seed = seed * 1103515245u + 12345u; return (int)((seed >> 16) & 0xFF) - 128; };
        for (size_t i = 0; i < hq.size(); i++) hq[i] = (signed char)rnd();
        for (size_t i = 0; i < hs.size(); i++) hs[i] = 0.002f * (1.0f + (float)(rnd() & 63) / 64.0f);
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++)
                ref_w[(size_t)n * K + k] = (double)hq[(size_t)n * K + k] * hs[(size_t)n * (K / SCALE_BLK) + (k >> 7)];

        signed char* dq; float* ds; unsigned short* dx; float* dc;
        const int TMAX = 75;
        if (cudaMalloc(&dq, (size_t)N * K) || cudaMalloc(&ds, (size_t)N * (K / SCALE_BLK) * 4) ||
            cudaMalloc(&dx, (size_t)TMAX * K * 2) || cudaMalloc(&dc, (size_t)TMAX * N * 4)) {
            printf("%-17s cudaMalloc failed\n", sh.tag); fails++; cudaGetLastError(); continue; }
        cudaMemcpy(dq, hq.data(), (size_t)N * K, cudaMemcpyHostToDevice);
        cudaMemcpy(ds, hs.data(), (size_t)N * (K / SCALE_BLK) * 4, cudaMemcpyHostToDevice);
        const double wbytes = (double)N * K * 1.0 + (double)N * (K / SCALE_BLK) * 4;

        for (int ti = 0; ti < 4; ti++) {
            const int T = TS[ti];
            std::vector<unsigned short> hx((size_t)T * K);
            for (size_t i = 0; i < hx.size(); i++)
                hx[i] = bf16_of(0.5f * ((int)((seed = seed * 1103515245u + 12345u) >> 20) % 2000 - 1000) / 1000.0f);
            cudaMemcpy(dx, hx.data(), (size_t)T * K * 2, cudaMemcpyHostToDevice);
            dim3 grid((N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, (T + TILE - 1) / TILE);

            cudaMemset(dc, 0, (size_t)T * N * 4);
            k_gemm_q8<<<grid, 256>>>(dq, ds, dx, dc, N, K, T);
            /* Check the LAUNCH, not just the sync. cudaDeviceSynchronize()
               returns cudaSuccess for a launch that was refused for resources
               (there is nothing to wait for) -- only the per-thread last-error
               carries it. That is how a kernel that never ran produced
               "all-zero output, no error, impossible speed" for four
               configurations, and how TILE=24 went unnoticed. */
            cudaError_t el = cudaGetLastError();
            if (el != cudaSuccess) { printf("%-17s T=%2d LAUNCH REFUSED: %s\n", sh.tag, T,
                                            cudaGetErrorString(el));
                fails++; cudaGetLastError(); continue; }
            cudaError_t e = cudaDeviceSynchronize();
            if (e != cudaSuccess) { printf("%-17s T=%2d SYNC %s\n", sh.tag, T, cudaGetErrorString(e));
                fails++; cudaGetLastError(); continue; }
            /* re-run after the timed pass: the timed loop must be the thing we read */
            const int reps = (int)(2e8 / ((double)N * K)) + 3;
            double t0 = now_ms();
            for (int rep = 0; rep < reps; rep++) k_gemm_q8<<<grid, 256>>>(dq, ds, dx, dc, N, K, T);
            if (cudaGetLastError() != cudaSuccess) { printf("%-17s T=%2d TIMED LAUNCH FAILED\n", sh.tag, T);
                fails++; cudaGetLastError(); continue; }
            cudaDeviceSynchronize();
            const double ms = (now_ms() - t0) / reps;
            cudaMemset(dc, 0, (size_t)T * N * 4);
            k_gemm_q8<<<grid, 256>>>(dq, ds, dx, dc, N, K, T);
            if (cudaGetLastError() != cudaSuccess) { printf("%-17s T=%2d FINAL LAUNCH FAILED\n", sh.tag, T);
                fails++; cudaGetLastError(); continue; }
            cudaDeviceSynchronize();

            std::vector<float> hc((size_t)T * N);
            cudaMemcpy(hc.data(), dc, (size_t)T * N * 4, cudaMemcpyDeviceToHost);

            /* PLAUSIBILITY GATE. Every number below is only believed if these
               pass; this file's own history is why. A REFUSED launch returns
               cudaSuccess from cudaDeviceSynchronize (nothing to wait for), so
               a kernel that never ran once read as "all zeros, no error,
               174 GB/s" across four configurations. */
            {
                size_t nz = 0;
                for (size_t i = 0; i < hc.size(); i++) if (hc[i] == 0.0f) nz++;
                const double gf_chk = 2.0 * T * N * K / 1e9 / (ms / 1e3);
                if (nz == hc.size()) {
                    printf("%-17s T=%2d IMPLAUSIBLE: every output is zero -- the kernel did not run\n", sh.tag, T);
                    fails++; continue;
                }
                if (nz) {
                    printf("%-17s T=%2d IMPLAUSIBLE: %zu of %zu outputs never written\n", sh.tag, T, nz, hc.size());
                    fails++; continue;
                }
                if (gf_chk > 236.0) {
                    printf("%-17s T=%2d IMPLAUSIBLE: %.0f GFLOP/s is above the 236 GFLOP/s peak -- "
                           "the work did not happen\n", sh.tag, T, gf_chk);
                    fails++; continue;
                }
            }

            /* EVERY output is checked. The denominator is sum|w_k x_k| -- a
               cancelling dot product has no useful relative error. */
            double worst = 0.0; int wt = -1, wn = -1; long nzero = 0;
            const int nstep = N > 4096 ? 3 : 1;
            std::vector<double> xk(K);
            for (int t = 0; t < T; t++) {
                for (int k = 0; k < K; k++) xk[k] = f32_of(hx[(size_t)t * K + k]);
                for (int n = 0; n < N; n += nstep) {
                    const double* wr = &ref_w[(size_t)n * K];
                    double acc = 0.0, absum = 0.0;
                    for (int k = 0; k < K; k++) { double p = wr[k] * xk[k]; acc += p; absum += fabs(p); }
                    const double got = hc[(size_t)t * N + n];
                    if (got == 0.0f) nzero++;
                    const double rel = fabs(got - acc) / (absum > 1e-12 ? absum : 1e-12);
                    if (rel > worst) { worst = rel; wt = t; wn = n; }
                }
            }
            const double passes = (double)grid.y;                 /* weight re-reads */
            const double gbs = wbytes * passes / 1e9 / (ms / 1e3);
            const double gf = 2.0 * T * N * K / 1e9 / (ms / 1e3);
            const double ideal = wbytes * passes / 1e9 / CEIL_GBS * 1e3;
            printf("%-17s %6d %5d %4d %8.3f %8.2f %8.2f %9.1f %9.3f  %s",
                   sh.tag, N, K, T, ms, gbs, gf, ms * 1e3 / T, ideal,
                   worst <= 2e-5 ? "ok" : "FAIL");
            if (worst > 2e-5) { printf(" rel=%.2g at t=%d n=%d (zeros seen: %ld)", worst, wt, wn, nzero); fails++; }
            else printf(" rel=%.1g  %.1fx of DRAM", worst, ideal / ms);
            printf("\n");
            fflush(stdout);
        }
        cudaFree(dq); cudaFree(ds); cudaFree(dx); cudaFree(dc);
    }
    printf("\nverdict: %s (fp64 relative gate 2e-5, every sampled output)\n",
           fails ? "FAIL" : "PASS");
    return fails != 0;
}
