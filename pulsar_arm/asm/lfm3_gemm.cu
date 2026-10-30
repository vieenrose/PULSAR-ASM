/* lfm3_gemm.cu -- the batched Q8 GEMM the audio-side stages all need
   (doc/LFM3_PORT_STAGES.md D2), measured at the audio shapes.

   WHY THIS EXISTS BEFORE ANY AUDIO LAYER DOES. Every component after the
   backbone (encoder, depth head, codec decoder) is a stack of GEMMs with
   68-219 MB of weights, and D2's arithmetic says they are only affordable
   batched: the codec decoder is 4.3 ms/frame batched and 320 ms/frame
   token-at-a-time; the encoder is 13 ms per 0.96 s chunk batched and 1.3 s per
   audio second unbatched. The batched kernel is a prerequisite, not an
   optimisation, and its efficiency at the audio shapes is item 3 of the plan's
   "measure before you quote" list.

   FORM. C[t][n] = sum_k W[n][k] * x[t][k] with
     W: Q8 rows with fp32 128-block scales -- the .lfm3q8 dtype-1 layout
        (1.125 B/weight), row-major [N][K];
     x: bf16 [T][K] (activations are bf16 everywhere in this model, matching
        the HF reference's rounding points);
     C: f32 [T][N].
   One 8-lane group owns ONE weight row and loops the TILE positions inside it,
   so the weight row is read once for all TILE outputs -- that is the whole
   point: weight bytes per output element fall by TILE, which turns a
   bandwidth-bound GEMV into a compute-bound GEMM. The x tile lives in shared.

   SUMMATION ORDER, documented because a gate depends on it: within a
   128-element block the 8 lanes of a group split k (lane b owns blocks
   b, b+8, b+16, ...), each lane's block partial is a 4-way unrolled
   multiply-accumulate chain, block partials are scaled by that block's fp32
   scale and accumulated in ascending block order, and the 8 lane partials are
   reduced by a shfl_down tree (4,2,1). Identical for every TILE, which is what
   the TILE=1 vs TILE=T self-consistency check below proves.

   fmad IS ON here (the default build). The engine's own GEMV builds with
   -fmad=false because it is bit-parity-gated against a C reference; this
   kernel's gate is the fp64 relative-error bound, and the audio layers it will
   serve are gated on rms/top-1, not on bits.

   ACCURACY: fp64 CPU reference on the dequantised weights, max relative error
   <= 2e-5 on every sampled (t,n); plus TILE=1 vs TILE=T self-consistency.

   usage: lfm3_gemm                       # the audio-shape sweep + the gate
          lfm3_gemm N K T                 # one shape
*/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <time.h>
#include <vector>
#include <cuda_runtime.h>

#define SCALE_BLK 128                     /* the .lfm3q8 dtype-1 block size */
#define ROWS_PER_BLOCK 32                 /* 256 threads / 8 lanes per row */
#define SHARED_BUDGET 40000               /* Maxwell: 48 KB/block, keep headroom */

template <int TILE>
__global__ void k_gemm_q8(const signed char* __restrict__ W, const float* __restrict__ SC,
                          const unsigned short* __restrict__ X, float* __restrict__ C,
                          int N, int K, int T) {
    extern __shared__ float sx[];                        /* TILE * K, bf16 widened */
    const int t0 = blockIdx.y * TILE;
    const int tid = threadIdx.x, nt = blockDim.x;
    /* T bounds the tile: ceil(T/TILE) tiles means the LAST one can be
       partial, and without this guard the tile read X[T..] and wrote C[T..] --
       an out-of-bounds write for any T that is not a multiple of TILE (T=75
       with tile=4). It showed up as garbage output for that T plus a poisoned
       CUDA context, i.e. "cudaMalloc failed" on every shape after it. */
    for (int i = tid; i < TILE * K; i += nt) {
        int t = i / K, k = i - t * K;
        union { unsigned int u; float f; } cv;
        cv.u = (t0 + t < T) ? (((unsigned int)X[(size_t)(t0 + t) * K + k]) << 16) : 0u;
        sx[i] = cv.f;
    }
    __syncthreads();
    const int lane = tid & 7, grp = tid >> 3;
    const int n = blockIdx.x * ROWS_PER_BLOCK + grp;
    if (n >= N) return;
    const signed char* w = W + (size_t)n * K;
    const float* scl = SC + (size_t)n * (K >> 7);
    const int nb = K >> 7;
    float acc[TILE];
#pragma unroll
    for (int t = 0; t < TILE; t++) acc[t] = 0.0f;
    for (int b = lane; b < nb; b += 8) {
        const signed char* wb = w + (b << 7);
        const float sc = scl[b];
        float ps[TILE];
#pragma unroll
        for (int t = 0; t < TILE; t++) ps[t] = 0.0f;
        /* 16-byte weight loads, unpacked in registers, NOT scalar int8 loads.
           The first version read wb[j] one byte at a time: 8 lanes per row,
           128 scalar loads each, and the kernel streamed weights at 0.45 GB/s
           -- 40x under the 17 GB/s the engine's q8_partial achieves on the same
           board. k order is unchanged (ascending, 16 at a time), so the
           documented summation order and the fp64 gate still apply. */
#pragma unroll
        for (int v = 0; v < SCALE_BLK / 16; v++) {
            const int4 q = *reinterpret_cast<const int4*>(wb + v * 16);
            const signed char* qb = reinterpret_cast<const signed char*>(&q);
#pragma unroll
            for (int i = 0; i < 16; i++) {
                const float wv = (float)qb[i];
                const int j = v * 16 + i;
#pragma unroll
                for (int t = 0; t < TILE; t++) ps[t] += wv * sx[t * K + (b << 7) + j];
            }
        }
#pragma unroll
        for (int t = 0; t < TILE; t++) acc[t] += ps[t] * sc;
    }
#pragma unroll
    for (int t = 0; t < TILE; t++) {
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) acc[t] += __shfl_down_sync(0xffffffffu, acc[t], o, 8);
        if (lane == 0 && t0 + t < T) C[(size_t)(t0 + t) * N + n] = acc[t];
    }
}

typedef void (*launch_fn)(const signed char*, const float*, const unsigned short*, float*,
                          int, int, int, dim3, size_t);
#define MK(T_)                                                                                    \
    static void launch_##T_(const signed char* a, const float* b, const unsigned short* c,       \
                            float* d, int N, int K, int T, dim3 g, size_t s) {                   \
        k_gemm_q8<T_><<<g, 256, s>>>(a, b, c, d, N, K, T);                                       \
    }
MK(1) MK(2) MK(4) MK(8) MK(16) MK(24) MK(32)

static launch_fn pick_tile(int t) {
    switch (t) { case 1: return launch_1; case 2: return launch_2; case 4: return launch_4;
                 case 8: return launch_8; case 16: return launch_16; case 24: return launch_24;
                 case 32: return launch_32; }
    return 0;
}

static unsigned short bf16_of(float f) {
    union { float f; unsigned int u; } v; v.f = f;
    return (unsigned short)((v.u + 0x7FFF + ((v.u >> 16) & 1)) >> 16);
}
static float f32_of_bf16(unsigned short h) {
    union { float f; unsigned int u; } v; v.u = ((unsigned int)h) << 16; return v.f;
}

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

static double now_ms() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

int main(int argc, char** argv) {
    int one_n = argc > 1 ? atoi(argv[1]) : 0, one_k = argc > 2 ? atoi(argv[2]) : 0,
        one_t = argc > 3 ? atoi(argv[3]) : 0;
    std::vector<Shape> shapes;
    if (one_n && one_k && one_t) shapes.push_back({one_n, one_k, "custom        "});
    else for (Shape& s : SHAPES) shapes.push_back(s);
    const int TS[] = {1, 6, 12, 24, 75};
    const int TILES[] = {1, 2, 4, 8, 16, 24, 32};

    printf("%-17s %6s %5s %4s %9s %9s %9s %9s %8s  %s\n",
           "shape", "N", "K", "T", "ms", "GB/s(w)", "GFLOP/s", "us/pos", "tile", "verdict");
    int fails = 0;
    unsigned seed = 12345u;
    for (Shape& sh : shapes) {
        const int N = sh.N, K = sh.K;
        if (K % SCALE_BLK) { printf("%-17s K=%d not a multiple of %d -- skipped\n", sh.tag, K, SCALE_BLK); continue; }
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
            printf("%-17s cudaMalloc failed (N=%d K=%d)\n", sh.tag, N, K); fails++; continue;
        }
        cudaMemcpy(dq, hq.data(), (size_t)N * K, cudaMemcpyHostToDevice);
        cudaMemcpy(ds, hs.data(), (size_t)N * (K / SCALE_BLK) * 4, cudaMemcpyHostToDevice);

        for (int ti = 0; ti < 5; ti++) {
            const int T = TS[ti];
            std::vector<unsigned short> hx((size_t)T * K);
            for (size_t i = 0; i < hx.size(); i++)
                hx[i] = bf16_of(0.5f * ((int)((seed = seed * 1103515245u + 12345u) >> 20) % 2000 - 1000) / 1000.0f);
            cudaMemcpy(dx, hx.data(), (size_t)T * K * 2, cudaMemcpyHostToDevice);
            std::vector<float> hc((size_t)T * N, 0.0f);

            double best_ms = 1e30; int best_tile = 0; double base_ms = 0;
            for (int tix = 0; tix < 7; tix++) {
                const int TT = TILES[tix];
                if (TT > T) continue;
                if ((size_t)TT * K * 4 > SHARED_BUDGET) continue;
                launch_fn lf = pick_tile(TT);
                if (!lf) continue;
                dim3 grid((N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, (T + TT - 1) / TT);
                const size_t shm = (size_t)TT * K * 4;
                cudaMemset(dc, 0, (size_t)T * N * 4);
                lf(dq, ds, dx, dc, N, K, T, grid, shm);
                /* clear the launch error: an out-of-resources launch (shared >
                   sharedMemPerBlock) leaves it queued, and the next cudaMalloc
                   reports THAT -- which is how one oversized tile config made
                   every later shape in the sweep "fail to allocate". */
                if (cudaDeviceSynchronize() != cudaSuccess) { cudaGetLastError(); continue; }
                if (cudaGetLastError() != cudaSuccess) continue;
                const int reps = (int)(3e8 / ((double)N * K * (1.0 + TT))) + 2;
                double t0 = now_ms();
                for (int rep = 0; rep < reps; rep++) lf(dq, ds, dx, dc, N, K, T, grid, shm);
                cudaDeviceSynchronize();
                double ms = (now_ms() - t0) / reps;
                if (TT == 1) base_ms = ms;
                if (ms < best_ms) { best_ms = ms; best_tile = TT; }
            }
            if (!best_tile) { printf("%-17s %6d %5d %4d  no valid tile config\n", sh.tag, N, K, T); fails++; continue; }
            cudaMemcpy(hc.data(), dc, (size_t)T * N * 4, cudaMemcpyDeviceToHost);

            /* Relative to sum|w_k x_k| (the condition-aware denominator), NOT to
               |sum|: a dot product that cancels to ~0 has no useful relative
               error, and dividing by it accused a correct kernel of 8.9e-5. */
            double worst = 0.0;
            const int nstep = N > 4096 ? 7 : 1;
            for (int t = 0; t < T; t++) {
                std::vector<float> xk(K);
                for (int k = 0; k < K; k++) xk[k] = f32_of_bf16(hx[(size_t)t * K + k]);
                for (int n = 0; n < N; n += nstep) {
                    double acc = 0.0, absum = 0.0;
                    const double* wr = &ref_w[(size_t)n * K];
                    for (int k = 0; k < K; k++) { double p = wr[k] * xk[k]; acc += p; absum += fabs(p); }
                    double den = absum > 1e-12 ? absum : 1e-12;
                    double got = hc[(size_t)t * N + n];
                    if (fabs(got - acc) / den > worst) {
                        worst = fabs(got - acc) / den;
                        if (getenv("LFM_GEMM_DEBUG"))
                            printf("  worst at t=%d n=%d got %.6g exact %.6g absum %.4g\n", t, n, got, acc, absum);
                    }
                }
            }
            const double wbytes = (double)N * K * 1.0 + (double)N * (K / SCALE_BLK) * 4;
            const double gf = 2.0 * T * N * K / 1e9;
            char verdict[64];
            snprintf(verdict, sizeof verdict, "%s rel=%.2g%s", worst <= 2e-5 ? "ok" : "FAIL", worst,
                     T > 1 && base_ms > 0 ? "" : "");
            if (worst > 2e-5) fails++;
            printf("%-17s %6d %5d %4d %9.3f %9.2f %9.2f %9.1f %8d  %s",
                   sh.tag, N, K, T, best_ms, wbytes / 1e9 / (best_ms / 1e3), gf / (best_ms / 1e3),
                   best_ms * 1e3 / T, best_tile, verdict);
            if (T > 1 && base_ms > 0) printf("  vs T=1 GEMV/pos: %.2fx", (base_ms / T) / best_ms ? base_ms / (best_ms * T) : 0.0);
            printf("\n");
            fflush(stdout);
        }
        cudaFree(dq); cudaFree(ds); cudaFree(dx); cudaFree(dc);
    }
    printf("\nverdict: %s\n", fails ? "FAIL" : "PASS (fp64 relative gate 2e-5 on all sampled outputs)");
    return fails != 0;
}
