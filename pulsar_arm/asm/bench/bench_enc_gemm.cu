// Encoder linear shapes on the engine's own GEMM kernels (S3 decision bench, #307/#308).
// Includes lfm3_nanode.cu with its main() renamed, so the kernels under test are the
// shipped templates, not copies. Each arm uses the SAME launch contract the engine uses
// (KC from the engine's 49152-byte shared loop, R, nblk, shm), so the rates are the rates
// the engine would get.
//
//   B  q8_partial<float,true,false,1,0>   per position, fp32 math (the spec's arithmetic)
//   A  q8_prefill_h2<float,4,false,5,2,2> batched T, fp32 math (GRP=2: GRP=8 refused)
//   E  q8_partial<bf16,true,false,1,1>    per position, half2 (decode-style)
//   C  q8_prefill_h2<bf16,4,true,0>       batched T, half2  (#162-#165 arm)
//   D  q8_prefill_h2<bf16,4,true,10,2,2>  batched T, half2  (#302's best arm)
//
// The E/C/D arms are the LFM_H2 arms: int8 codes converted to half by the 0x64-magic
// trick (exact) and accumulated in FP16 inside each 32-code block. They are the arms
// the handoff's open decision #1 is about, so they are measured, not assumed.
//
// Modes
//   bench_enc_gemm [T]                         synthetic codes/scales/activations
//   bench_enc_gemm real <enc.lfm3q8> <x.bin> [T]
//         weights = the REAL encoder tensors from the blob (layer 0 + downsampler +
//         projector), activations = the real HF layer-0 input (enc_ref/ref/hidden_00,
//         tiled to each shape's K when that shape's own activation was not dumped).
// The x for the half2 arms is the fp32 x rounded with the engine's own f2bf, so a
// half2-vs-fp32 delta isolates the ARITHMETIC, not the input quantization.
// Plausibility gate per arm: launch error read, output nonzero, rate < the measured
// ~106 GMAC/s HFMA2 ceiling.
#define main lfm3_engine_main
#include "../lfm3_nanode.cu"
#undef main
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>

struct Shape { const char* name; const char* blob; int N, K; int kreal; };
// kreal = the width of the dumped real activation to tile from (768 = hidden)

struct Ctx {
    const int8_t* q; const float* s; const float* xf; const bf16* xb;
    float* part; int N, K, KC, R, nb, nch, nblk, T, CT;
};

static void run_B(void* p) {
    Ctx& c = *(Ctx*)p;
    const int TPB = 256; const size_t shm = (size_t)c.nb * 33 * 4;
    for (int t = 0; t < c.T; t++)
        q8_partial<float, true, false, 1, 0><<<c.nblk, TPB, shm>>>(c.q, c.s,
            c.xf + (size_t)t * c.K, c.part + (size_t)t * c.N * c.nch, c.N, c.K, c.KC, c.R, c.nb);
}
static void run_A(void* p) {
    Ctx& c = *(Ctx*)p;
    const int TPB = 256; const size_t shm = (size_t)c.CT * c.nb * 33 * 4;
    q8_prefill_h2<float, 4, false, 5, 2, 2><<<c.nblk, TPB, shm>>>(c.q, c.s, c.xf, c.part,
        c.N, c.K, c.KC, c.R, c.nb, c.T);
}
static void run_E(void* p) {
    Ctx& c = *(Ctx*)p;
    const int TPB = 256; const size_t shm = (size_t)c.nb * 33 * 4;
    for (int t = 0; t < c.T; t++)
        q8_partial<bf16, true, false, 1, 1><<<c.nblk, TPB, shm>>>(c.q, c.s,
            c.xb + (size_t)t * c.K, c.part + (size_t)t * c.N * c.nch, c.N, c.K, c.KC, c.R, c.nb);
}
static void run_C(void* p) {
    Ctx& c = *(Ctx*)p;
    const int TPB = 256; const size_t shm = (size_t)c.CT * c.nb * 17 * 4;
    q8_prefill_h2<bf16, 4, true, 0><<<c.nblk, TPB, shm>>>(c.q, c.s, c.xb, c.part,
        c.N, c.K, c.KC, c.R, c.nb, c.T);
}
static void run_D(void* p) {
    Ctx& c = *(Ctx*)p;
    const int TPB = 256; const size_t shm = (size_t)c.CT * c.nb * 17 * 4;
    q8_prefill_h2<bf16, 4, true, 10, 2, 2><<<c.nblk, TPB, shm>>>(c.q, c.s, c.xb, c.part,
        c.N, c.K, c.KC, c.R, c.nb, c.T);
}

static void run_F(void* p) {                                 // bf16 x, fp32 math = the shipped GEMV
    Ctx& c = *(Ctx*)p;
    const int TPB = 256; const size_t shm = (size_t)c.nb * 33 * 4;
    for (int t = 0; t < c.T; t++)
        q8_partial<bf16, true, false, 1, 0><<<c.nblk, TPB, shm>>>(c.q, c.s,
            c.xb + (size_t)t * c.K, c.part + (size_t)t * c.N * c.nch, c.N, c.K, c.KC, c.R, c.nb);
}

typedef void (*ArmFn)(void*);
struct Arm { const char* n; ArmFn fn; int CT, R, h2; };
static const Arm arms[] = {
    {"B fp32/pos", run_B, 1, 256, 0},
    {"A fp32/bat", run_A, 4, 16, 0},        // ARM5 f32: R = WPB*GRP = 8*2
    {"E h2 /pos", run_E, 1, 256, 1},
    {"C h2 /bat", run_C, 4, 256, 1},
    {"D h2/bat10", run_D, 4, 256, 1},
    {"F bf16/pos", run_F, 1, 256, 0},        // bf16 activations, fp32 math (shipped GEMV)
};
static const int NA = 6;

static float time_ms(ArmFn fn, void* ctx, int reps) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    fn(ctx); cudaDeviceSynchronize();
    float best = 1e30f;
    for (int r = 0; r < reps; r++) {
        cudaEventRecord(a); fn(ctx); cudaEventRecord(b); cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b); best = std::min(best, ms);
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    return best;
}

// The engine's own KC rule (prefill_h2_one): the largest power-of-2 divisor of K whose
// staging fits 49152 B of dynamic shared for this arm's CT and element type.
static void contract(Ctx& c, int N, int K, int CT, int h2, int R, int T) {
    int KC = K;
    while (KC > 32) {
        const int nbtry = KC / Q8_BLK;
        const size_t need = h2 ? (size_t)CT * nbtry * 17 * 4 : (size_t)CT * nbtry * 33 * 4;
        if (need <= 49152 && K % KC == 0) break;
        KC /= 2;
    }
    c.KC = KC; c.nb = KC / Q8_BLK; c.nch = K / KC; c.R = R; c.T = T; c.CT = CT; c.N = N; c.K = K;
    c.nblk = ((N + R - 1) / R) * c.nch;
}

struct Totals { double ms[NA]; double wmax[NA]; double wsum[NA]; long ncell[NA]; long nzmin[NA]; int fails;
                double f64max[NA], f64sum[NA]; long f64n[NA]; };

// One shape, all arms. q/s are device weights (real or synthetic), xh host fp32 [T*K].
static int run_shape(const Shape& sh, const int8_t* q, const float* s, const float* xh,
                     int T, Totals& tot, bool verbose, const int8_t* qh, const float* shh) {
    int8_t* dq; float* ds; float* dxf; bf16* dxb;
    if (cudaMalloc(&dq, (size_t)sh.N * sh.K) || cudaMalloc(&ds, (size_t)sh.N * (sh.K / Q8_BLK) * 4)
        || cudaMalloc(&dxf, (size_t)T * sh.K * 4) || cudaMalloc(&dxb, (size_t)T * sh.K * 2)) {
        printf("%-22s OOM\n", sh.name); return 1;
    }
    cudaMemcpy(dq, q, (size_t)sh.N * sh.K, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, s, (size_t)sh.N * (sh.K / Q8_BLK) * 4, cudaMemcpyHostToDevice);
    (void)qh; (void)shh;
    cudaMemcpy(dxf, xh, (size_t)T * sh.K * 4, cudaMemcpyHostToDevice);
    std::vector<uint16_t> hb((size_t)T * sh.K);
    for (size_t i = 0; i < hb.size(); i++) {                       // f2bf, RN-even
        const unsigned u = *reinterpret_cast<const unsigned*>(&xh[i]);
        hb[i] = (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    }
    cudaMemcpy(dxb, hb.data(), hb.size() * 2, cudaMemcpyHostToDevice);
    if (verbose) printf("%-22s", sh.name);
    std::vector<float> ref;
    std::vector<std::vector<float> > arms_out(NA);
    for (int a = 0; a < NA; a++) {
        Ctx c; c.q = dq; c.s = ds; c.xf = dxf; c.xb = dxb;
        contract(c, sh.N, sh.K, arms[a].CT, arms[a].h2, arms[a].R, T);
        if (cudaMalloc(&c.part, (size_t)T * sh.N * c.nch * 4)) { printf("OOM\n"); return 1; }
        cudaMemset(c.part, 0, (size_t)T * sh.N * c.nch * 4);
        const float ms = time_ms(arms[a].fn, &c, 5);
        cudaError_t e = cudaGetLastError();                        // read BEFORE the sync
        if (e) { printf(" %-13s LAUNCH ERROR %s\n", arms[a].n, cudaGetErrorString(e)); tot.fails++; return 1; }
        cudaDeviceSynchronize();
        e = cudaGetLastError();
        if (e) { printf(" %-13s EXEC ERROR %s\n", arms[a].n, cudaGetErrorString(e)); tot.fails++; return 1; }
        std::vector<float> raw((size_t)T * sh.N * c.nch);
        cudaMemcpy(raw.data(), c.part, raw.size() * 4, cudaMemcpyDeviceToHost);
        std::vector<float> out((size_t)T * sh.N);                  // ascending chunk order
        for (size_t i = 0; i < out.size(); i++) {
            float v = 0; for (int ch = 0; ch < c.nch; ch++) v += raw[i * c.nch + ch];
            out[i] = v;
        }
        arms_out[a] = out;
        long nz = 0; for (float v : out) nz += (v != 0.f);
        const double macs = (double)T * sh.N * sh.K;
        const double g = macs / (ms * 1e-3) / 1e9;
        if (nz == 0 || g > 120) { tot.fails++; if (verbose) printf(" %6.3f/%5.0f!PLAUS", ms, g); }
        else if (verbose) printf(" %6.3f/%5.1f", ms, g);
        tot.ms[a] += ms; tot.nzmin[a] = std::min(tot.nzmin[a], nz);
        if (a == 0) ref = out;
        else {
            double w = 0, se = 0; long big = 0, n = 0;
            for (size_t i = 0; i < out.size(); i++) {
                if (fabs(ref[i]) <= 1e-9) continue;
                const double d = fabs(out[i] - ref[i]) / fabs(ref[i]);
                n++; se += d * d; if (d > 1e-2) big++; if (d > w) w = d;
            }
            tot.wmax[a] = std::max(tot.wmax[a], w); tot.wsum[a] += se; tot.ncell[a] += n;
            if (verbose) printf(" [rel max %.1e rms %.1e >1e-2 %ld/%ld]", w, sqrt(se / (double)(n ? n : 1)), big, n);
        }
        cudaFree(c.part);
    }
    // fp64 referee on a row subset: says which arm is RIGHT, not just which two agree.
    // y = sum_b s[r][b] * (sum_{j in block b} c_j * x_j), blocks in ascending order.
    if (qh && shh) {
        const int NREF = std::min(sh.N, 48);
        // x rounded to bf16 on the HOST, fp64-dot'd: separates "the bf16 input
        // quantization is that lossy on this data" from "the kernel's bf16 read
        // is broken". refb = fp64 dot with bf16 x; refd = fp64 dot with fp32 x.
        std::vector<float> xbf((size_t)T * sh.K);
        for (size_t i = 0; i < xbf.size(); i++) {
            const unsigned u = *reinterpret_cast<const unsigned*>(&xh[i]);
            const unsigned short h = (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
            const unsigned uu = ((unsigned)h) << 16;
            xbf[i] = *reinterpret_cast<const float*>(&uu);
        }
        double sbw = 0, sbmax = 0, sda = 0, damax = 0; long nbw = 0;
        for (int r = 0; r < NREF; r++) for (int t = 0; t < T; t++) {
            long double ab = 0, ad = 0;
            for (int b = 0; b < sh.K / Q8_BLK; b++) {
                long double bb = 0, bd2 = 0;
                for (int j = 0; j < Q8_BLK; j++) {
                    const long double c = (long double)qh[(size_t)r * sh.K + b * Q8_BLK + j];
                    bb += c * (long double)xbf[(size_t)t * sh.K + b * Q8_BLK + j];
                    bd2 += c * (long double)xh[(size_t)t * sh.K + b * Q8_BLK + j];
                }
                const long double sc = (long double)shh[(size_t)r * (sh.K / Q8_BLK) + b];
                ab += sc * bb; ad += sc * bd2;
            }
            if (fabs((double)ad) <= 1e-9) continue;
            const double db = fabs((double)(ab - ad) / ad);          // input rounding alone
            sbw += db * db; nbw++; if (db > sbmax) sbmax = db;
            const double da = fabs(arms_out[5][(size_t)t * sh.N + r] - (double)ab) / fabs((double)ab);
            sda += da * da; if (da > damax) damax = da;               // arm F vs bf16-input fp64
        }
        if (verbose) printf("   [input bf16 rounding alone: max %.2e rms %.2e over %ld | arm F vs fp64-on-bf16-x: max %.2e rms %.2e]\n",
                           sbmax, sqrt(sbw / (nbw ? nbw : 1)), nbw, damax, sqrt(sda / (nbw ? nbw : 1)));
        for (int r = 0; r < NREF; r++) {
            for (int t = 0; t < T; t++) {
                long double acc = 0;
                for (int b = 0; b < sh.K / Q8_BLK; b++) {
                    long double bd = 0;
                    for (int j = 0; j < Q8_BLK; j++)
                        bd += (long double)qh[(size_t)r * sh.K + b * Q8_BLK + j] *
                              (long double)xh[(size_t)t * sh.K + b * Q8_BLK + j];
                    acc += (long double)shh[(size_t)r * (sh.K / Q8_BLK) + b] * bd;
                }
                const double refd = (double)acc;
                if (fabs(refd) <= 1e-9) continue;
                for (int a = 0; a < NA; a++) {
                    const double d = fabs(arms_out[a][(size_t)t * sh.N + r] - refd) / fabs(refd);
                    tot.f64max[a] = std::max(tot.f64max[a], d);
                    tot.f64sum[a] += d * d; tot.f64n[a]++;
                }
            }
        }
    }
    if (verbose) printf("\n");
    cudaFree(dq); cudaFree(ds); cudaFree(dxf); cudaFree(dxb);
    return 0;
}

static int tile(const std::vector<float>& src, int K, int T, std::vector<float>& dst) {
    const int sw = 768;                                  // the dumped activation's width
    dst.resize((size_t)T * K);
    if (src.size() < (size_t)T * sw) return 0;
    for (int t = 0; t < T; t++)
        for (int k = 0; k < K; k++) dst[(size_t)t * K + k] = src[(size_t)t * sw + (k % sw)];
    return 1;
}

int main(int argc, char** argv) {
    const bool real = argc > 1 && !strcmp(argv[1], "real");
    // usage: bench_enc_gemm real <blob> <x.bin> [T]  -> T is argv[4]; the argv[3] is the path
    const int T = real ? (argc > 4 ? atoi(argv[4]) : 50) : (argc > 1 ? atoi(argv[1]) : 50);
    if (T <= 0) { printf("T must be > 0\n"); return 2; }
    const Shape shapes[] = {
        {"enc in_proj 2304x768", "encoder.lfm.lfm.layers.0.conv.in_proj.weight", 2304, 768, 768},
        {"enc out_proj 768x768", "encoder.lfm.lfm.layers.0.conv.out_proj.weight", 768, 768, 768},
        {"enc w1 1792x768", "encoder.lfm.lfm.layers.0.feed_forward.w1.weight", 1792, 768, 768},
        {"enc w2 768x1792", "encoder.lfm.lfm.layers.0.feed_forward.w2.weight", 768, 1792, 768},
        {"enc lin_out 768x4096", "encoder.downsample.lin_out.weight", 768, 4096, 768},
        {"proj1 2048x768", "mlp.model.1.weight", 2048, 768, 768},
        {"proj2 2048x2048", "mlp.model.3.weight", 2048, 2048, 768},
    };
    const int NS = 7;
    Totals tot;
    for (int a = 0; a < NA; a++) { tot.ms[a] = 0; tot.wmax[a] = 0; tot.wsum[a] = 0; tot.ncell[a] = 0; tot.nzmin[a] = 1L << 40;
                                   tot.f64max[a] = 0; tot.f64sum[a] = 0; tot.f64n[a] = 0; }
    tot.fails = 0;

    Model M; std::string js;
    std::vector<float> act;
    if (real) {
        if (argc < 4) { printf("usage: bench_enc_gemm real <blob> <x.bin> [T]\n"); return 2; }
        if (!load_blob(argv[2], M, js)) return 1;
        int fd = open(argv[3], O_RDONLY);
        if (fd < 0) { printf("cannot open %s\n", argv[3]); return 2; }
        act.resize((size_t)T * 768);
        ssize_t got = read(fd, &act[0], act.size() * 4);
        close(fd);
        if (got != (ssize_t)(act.size() * 4)) { printf("x.bin short: %lld of %zu bytes\n", (long long)got, act.size() * 4); return 2; }
        double nrm = 0; long nn = 0;
        for (size_t i = 0; i < act.size(); i++) { nrm += (double)act[i] * act[i]; nn++; if (!isfinite(act[i])) { printf("x.bin non-finite at %zu\n", i); return 2; } }
        printf("real activations %s: T=%d rms %.5f\n", argv[3], T, sqrt(nrm / nn));
    }
    printf("encoder GEMM bench (%s), T=%d, reps 5, min time; KC/R per the engine's own contract\n",
           real ? "REAL weights+activations" : "synthetic", T);
    printf("%-22s", "shape");
    for (int a = 0; a < NA; a++) printf(" %13s", arms[a].n);
    printf("\n");
    double total_macs = 0;
    for (int i = 0; i < NS; i++) {
        const Shape& sh = shapes[i];
        std::vector<int8_t> hq; std::vector<float> hs, xh;
        const int8_t *dq_src, *dq_real = nullptr; const float *ds_src, *ds_real = nullptr;
        if (real) {
            Ent* e = ::T(M, sh.blob);   // the engine's lookup; T() is shadowed by the local T
            if (!e) { printf("%-22s NOT IN BLOB: %s\n", sh.name, sh.blob); tot.fails++; continue; }
            if (e->dtype != 1 || e->ne[0] != sh.N || e->ne[1] != sh.K) {
                printf("%-22s SHAPE/DTYPE MISMATCH %s: blob %d %dx%d, bench %dx%d\n", sh.name,
                       sh.blob, e->dtype, (long)e->ne[0], (long)e->ne[1], sh.N, sh.K);
                tot.fails++; continue;
            }
            hq.resize((size_t)sh.N * sh.K); hs.resize((size_t)sh.N * (sh.K / Q8_BLK));
            cudaMemcpy(hq.data(), e->dq, hq.size(), cudaMemcpyDeviceToHost);
            cudaMemcpy(hs.data(), e->ds, hs.size() * 4, cudaMemcpyDeviceToHost);
            if (!tile(act, sh.K, T, xh)) { printf("%-22s activation too short\n", sh.name); tot.fails++; continue; }
            long nzq = 0; for (size_t k = 0; k < hq.size(); k++) nzq += (hq[k] != 0);
            if (nzq == 0) { printf("%-22s ALL-ZERO WEIGHTS\n", sh.name); tot.fails++; continue; }
        } else {
            hq.resize((size_t)sh.N * sh.K); hs.resize((size_t)sh.N * (sh.K / Q8_BLK));
            for (size_t k = 0; k < hq.size(); k++) hq[k] = (int8_t)((int)((k * 73) % 255) - 127);
            for (size_t k = 0; k < hs.size(); k++) hs[k] = 0.002f + 0.0001f * (float)(k % 17);
            xh.resize((size_t)T * sh.K);
            for (size_t k = 0; k < xh.size(); k++)
                xh[k] = 0.01f * (float)((int)((k * 37) % 201) - 100) / 100.f;
        }
        run_shape(sh, hq.data(), hs.data(), &xh[0], T, tot, true, hq.data(), hs.data());
        total_macs += (double)T * sh.N * sh.K;
    }
    printf("\nsum over the 7 encoder linears (%.1f MMAC per position-set): cost at 1.45 GMAC per audio second\n",
           total_macs / 1e6);
    for (int a = 0; a < NA; a++) {
        const double g = total_macs / (tot.ms[a] * 1e-3) / 1e9;
        printf("  %-13s %8.2f ms  %6.1f GMAC/s  -> %6.3f s GPU per audio second  (nonzero min %ld)\n",
               arms[a].n, tot.ms[a], g, 1.45 / g, tot.nzmin[a]);
    }
    printf("vs the fp32 arm (same bf16-rounded inputs):\n");
    for (int a = 1; a < NA; a++)
        printf("  %-13s max rel %8.2e   rms rel %8.2e over %ld cells\n", arms[a].n, tot.wmax[a],
               sqrt(tot.wsum[a] / (double)(tot.ncell[a] ? tot.ncell[a] : 1)), tot.ncell[a]);
    printf("vs the fp64 referee (48 rows x T per shape, real weights and activations):\n");
    for (int a = 0; a < NA; a++)
        printf("  %-13s max rel %8.2e   rms rel %8.2e over %ld cells\n", arms[a].n, tot.f64max[a],
               sqrt(tot.f64sum[a] / (double)(tot.f64n[a] ? tot.f64n[a] : 1)), tot.f64n[a]);
    printf("%s\n", tot.fails ? "BENCH FAILED (plausibility)" : "BENCH OK");
    return tot.fails ? 1 : 0;
}
