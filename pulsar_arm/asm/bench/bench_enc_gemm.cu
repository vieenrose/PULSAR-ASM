// Encoder linear shapes on the engine's own GEMM kernels (S3 decision bench).
// Includes lfm3_nanode.cu with its main() renamed, so the kernels under test are
// the shipped templates, not copies. Two arms, both fp32 activations (the encoder
// spec is fp32, no bf16 rounding):
//   B  q8_partial<float>        per position (T launches, the decode-style GEMV)
//   A  q8_prefill_h2<float,ARM5> batched over T positions (ARM5 f32 form, GRP=8)
// Plausibility gate: outputs nonzero; rate below the measured ~106 GMAC/s HFMA2 ceiling.
// NOTE: the batched f32 arm is GRP=2 (GRP=8 is refused at launch on this build).
// Usage: bench_enc_gemm [T]   (T = tokens per 4 s clip; default 50)
#define main lfm3_engine_main
#include "../lfm3_nanode.cu"
#undef main
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

struct Shape { const char* name; int N, K; };

static float time_ms(void (*fn)(void*), void* ctx, int reps) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    fn(ctx); cudaDeviceSynchronize();                 // warm
    float best = 1e30f;
    for (int r = 0; r < reps; r++) {
        cudaEventRecord(a); fn(ctx); cudaEventRecord(b); cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b); best = std::min(best, ms);
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    return best;
}

struct Ctx {
    const int8_t* q; const float* s; const float* x; float* part; float* out;
    int N, K, KC, R, nb, nch, nblk, T;
};
static void run_B(void* p) {                          // per-position q8_partial
    Ctx& c = *(Ctx*)p;
    const int TPB = 256;
    const size_t shm = (size_t)c.nb * 33 * 4;
    for (int t = 0; t < c.T; t++)
        q8_partial<float, true><<<c.nblk, TPB, shm>>>(c.q, c.s, c.x + (size_t)t * c.K,
            c.part + (size_t)t * c.N * c.nch, c.N, c.K, c.KC, c.R, c.nb);
}
static void run_A(void* p) {                          // batched ARM5 f32, GRP=8
    Ctx& c = *(Ctx*)p;
    const int TPB = 256;
    const size_t shm = (size_t)4 * c.nb * 33 * 4;     // CT=4, f32 staging
    // f32 ARM5 needs GRP=2 at CT=4: GRP=8 is refused (too many resources, >255 regs)
    q8_prefill_h2<float, 4, false, 5, 2, 2><<<c.nblk, TPB, shm>>>(c.q, c.s, c.x, c.part,
        c.N, c.K, c.KC, c.R, c.nb, c.T);
}

int main(int argc, char** argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 50;
    const Shape shapes[] = {
        {"enc in_proj 2304x768", 2304, 768}, {"enc out_proj 768x768", 768, 768},
        {"enc w1 1792x768", 1792, 768}, {"enc w2 768x1792", 768, 1792},
        {"enc lin_out 768x4096", 768, 4096}, {"proj1 2048x768", 2048, 768},
        {"proj2 2048x2048", 2048, 2048},
    };
    double total_macs = 0, sumB = 0, sumA = 0;
    int fails = 0;
    printf("encoder GEMM bench, T=%d positions (fp32 activations), reps 5, min time\n", T);
    printf("%-24s %9s %9s %9s %9s  %s\n", "shape", "B ms", "B GMAC/s", "A ms", "A GMAC/s", "A==B(max rel)");
    for (const Shape& sh : shapes) {
        const int N = sh.N, K = sh.K;
        int KC = std::min(K, 2048); while (K % KC) KC /= 2;
        const int nb = KC / Q8_BLK, nch = K / KC;
        const int R = (N >= 1024) ? 256 : 64;
        const int nblk = ((N + R - 1) / R) * nch;
        // A needs R = WPB*GRP = 64 rows per block (ARM5 contract); B uses its own R
        const int RA = 16;   // GRP=2 x 8 warps
        const int nblkA = ((N + RA - 1) / RA) * nch;
        std::vector<int8_t> hq((size_t)N * K); std::vector<float> hs((size_t)N * (K / Q8_BLK)), hx((size_t)T * K);
        for (size_t i = 0; i < hq.size(); i++) hq[i] = (int8_t)((i * 73 + 11) % 255 - 127);
        for (size_t i = 0; i < hs.size(); i++) hs[i] = 0.002f + 0.0001f * (i % 17);
        for (size_t i = 0; i < hx.size(); i++) hx[i] = 0.01f * (float)((int)((i * 37) % 201) - 100) / 100.f;
        int8_t* dq; float *ds, *dx, *pa, *pb;
        cudaMalloc(&dq, hq.size()); cudaMalloc(&ds, hs.size() * 4); cudaMalloc(&dx, hx.size() * 4);
        cudaMalloc(&pa, (size_t)T * N * nch * 4); cudaMalloc(&pb, (size_t)T * N * nch * 4);
        cudaMemcpy(dq, hq.data(), hq.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(ds, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
        Ctx cb{dq, ds, dx, pb, nullptr, N, K, KC, R, nb, nch, nblk, T};
        Ctx ca{dq, ds, dx, pa, nullptr, N, K, KC, RA, nb, nch, nblkA, T};
        float msB = time_ms(run_B, &cb, 5);
        float msA = time_ms(run_A, &ca, 5);
        // a refused launch is silent until the error is READ (run #297 lesson): check
        // the launch error right after the launches, then again after the sync
        cudaError_t e = cudaGetLastError();
        if (e) { printf("%-24s LAUNCH ERROR %s\n", sh.name, cudaGetErrorString(e)); return 2; }
        cudaDeviceSynchronize();
        e = cudaGetLastError();
        if (e) { printf("%-24s EXEC ERROR %s\n", sh.name, cudaGetErrorString(e)); return 2; }
        // partial sums -> per-position outputs (ascending chunk order), compare A vs B
        std::vector<float> ob((size_t)T * N * nch), oa((size_t)T * N * nch);
        cudaMemcpy(ob.data(), pb, ob.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(oa.data(), pa, oa.size() * 4, cudaMemcpyDeviceToHost);
        long nz = 0; double worst = 0;
        for (size_t i = 0; i < ob.size(); i++) {
            nz += (ob[i] != 0.f);
            double d = fabs(ob[i] - oa[i]) / (fabs(ob[i]) + 1e-30);
            if (fabs(ob[i]) > 1e-6 && d > worst) worst = d;
        }
        const double macs = (double)T * N * K;
        const double gB = macs / (msB * 1e-3) / 1e9, gA = macs / (msA * 1e-3) / 1e9;
        const bool plaus = nz > 0 && gB < 120 && gA < 120;
        if (!plaus) fails++;
        printf("%-24s %9.3f %9.1f %9.3f %9.1f  %.2e %s\n", sh.name, msB, gB, msA, gA, worst,
               plaus ? "" : "PLAUSIBILITY FAIL");
        total_macs += macs; sumB += msB; sumA += msA;
        cudaFree(dq); cudaFree(ds); cudaFree(dx); cudaFree(pa); cudaFree(pb);
    }
    printf("sum over shapes (one encoder pass of linears per %d tokens): B %.2f ms (%.1f GMAC/s), A %.2f ms (%.1f GMAC/s)\n",
           T, sumB, total_macs / (sumB * 1e-3) / 1e9, sumA, total_macs / (sumA * 1e-3) / 1e9);
    printf("%s\n", fails ? "BENCH FAILED (plausibility)" : "BENCH OK");
    return fails ? 1 : 0;
}
