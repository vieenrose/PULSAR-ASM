// fwd35_sm53.cu: Ternary-Bonsai-2 (arch qwen35, PTQ1_0 weights) on the Jetson Nano GPU (sm_53).
//
// Bit-exact port of fwd35_ref.c (itself bit-exact vs the PrismML fork's x86 CPU build): every
// op keeps the fork's evaluation order, so the greedy stream and every logit equal the fork's.
//
//   build: nvcc -O3 -gencode arch=compute_53,code=sm_53 --fmad=false -Xcompiler -fopenmp,-ffp-contract=off
//          -o fwd35 fwd35_sm53.cu -lgomp
//   usage: fwd35 model.gguf N id id ...   (prints N greedy ids, one per line; speeds on stderr)
//   env:   E_DUMP=f     raw f32 logits of every step appended to f (same format as R_DUMP)
//          E_TDUMP=f E_TSTEP=s   named intermediates of the forward that yields step s (no graph)
//          E_CTX=n      K/V cache positions (default 4096)   E_GRAPH=0  no CUDA graph
//
// Weights: PTQ1_0 (base-3 packed trits, 28 bytes per 128) is re-encoded at load into two bit
// planes per 128-weight group, b0 and b1 with code = b0 + 2 b1 = trit + 1, so a 32-weight
// integer sum is a handful of LOP3 + POPC against the bit-sliced Q8_0 activation (the 2 SMs
// cannot afford a base-3 decode per weight). The integer sums are encoding-independent, the
// float chain after them is the fork's: sumi = fma(d8_k, s_k, sumi) over the 4 sub-blocks,
// sumf = fma(d, sumi, sumf) over the blocks in order. 34 bytes per 128 weights (+21 %).
//
// Long double sums (rms_norm, l2_norm: sequential double adds in the fork) run one warp
// wide and stay exact: inside one binade of the running sum every add rounds the term to the
// same ulp grid independently, so a chunk of 32 is an integer prefix scan; binade crossings
// and exact ties (whose rounding depends on the state) fall back to one real double add.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <omp.h>
#include "glibc_mathf.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)
#define FULL 0xffffffffu
static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float bitsf(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

/* ---------------- fp16 (IEEE binary16, round to nearest even) ---------------- */
static float f16f(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e == 0) {
        if (!m) return bitsf(s);
        return bitsf(fbits(ldexpf((float)m, -24)) | s);
    }
    if (e == 31) return bitsf(s | 0x7f800000u | (m << 13));
    return bitsf(s | ((e + 112) << 23) | (m << 13));
}
static uint16_t ff16(float ff) {
    const uint32_t f32infty = 255u << 23, f16max = (127u + 16) << 23;
    const uint32_t denorm_magic = ((127u - 15) + (23 - 10) + 1) << 23;
    uint32_t f = fbits(ff), sign = f & 0x80000000u;
    uint16_t o;
    f ^= sign;
    if (f >= f16max) o = (f > f32infty) ? 0x7e00 : 0x7c00;
    else if (f < (113u << 23)) {
        float t = bitsf(f) + bitsf(denorm_magic);
        o = (uint16_t)(fbits(t) - denorm_magic);
    } else {
        uint32_t mant_odd = (f >> 13) & 1;
        f += ((uint32_t)(15 - 127) << 23) + 0xfff;
        f += mant_odd;
        o = (uint16_t)(f >> 13);
    }
    return o | (uint16_t)(sign >> 16);
}

/* ---------------- GGUF ---------------- */
typedef struct { const char *k; uint64_t kl; uint32_t t; const uint8_t *v; } KV;
typedef struct { const char *n; uint64_t nl; uint32_t nd, type; int64_t ne[4]; uint64_t off; const uint8_t *d; } TI;
static const uint8_t *G; static size_t GN;
static KV *kvs; static uint64_t nkv;
static TI *tis; static uint64_t nti;

static uint64_t rd64(const uint8_t **p) { uint64_t v; memcpy(&v, *p, 8); *p += 8; return v; }
static uint32_t rd32(const uint8_t **p) { uint32_t v; memcpy(&v, *p, 4); *p += 4; return v; }
static const size_t gsz[13] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
static void skipv(const uint8_t **p, uint32_t t) {
    if (t == 8) { uint64_t l = rd64(p); *p += l; return; }
    if (t == 9) { uint32_t at = rd32(p); uint64_t n = rd64(p);
        for (uint64_t i = 0; i < n; i++) skipv(p, at);
        return; }
    *p += gsz[t];
}
static void gguf_open(const char *path) {
    int fd = open(path, O_RDONLY); struct stat st;
    if (fd < 0 || fstat(fd, &st)) { perror(path); exit(1); }
    GN = st.st_size;
    G = (const uint8_t *)mmap(NULL, GN, PROT_READ, MAP_PRIVATE, fd, 0);
    if (G == MAP_FAILED) { perror("mmap"); exit(1); }
    const uint8_t *p = G;
    if (memcmp(p, "GGUF", 4)) { fprintf(stderr, "not gguf\n"); exit(1); }
    p += 8;
    nti = rd64(&p); nkv = rd64(&p);
    kvs = (KV *)calloc(nkv, sizeof(KV)); tis = (TI *)calloc(nti, sizeof(TI));
    for (uint64_t i = 0; i < nkv; i++) {
        kvs[i].kl = rd64(&p); kvs[i].k = (const char *)p; p += kvs[i].kl;
        kvs[i].t = rd32(&p); kvs[i].v = p; skipv(&p, kvs[i].t);
    }
    for (uint64_t i = 0; i < nti; i++) {
        TI *t = &tis[i];
        t->nl = rd64(&p); t->n = (const char *)p; p += t->nl;
        t->nd = rd32(&p);
        for (int j = 0; j < 4; j++) t->ne[j] = 1;
        for (uint32_t j = 0; j < t->nd; j++) t->ne[j] = (int64_t)rd64(&p);
        t->type = rd32(&p); t->off = rd64(&p);
    }
    uint64_t align = 32;
    for (uint64_t i = 0; i < nkv; i++)
        if (kvs[i].kl == 17 && !memcmp(kvs[i].k, "general.alignment", 17)) align = *(const uint32_t *)kvs[i].v;
    size_t base = ((size_t)(p - G) + align - 1) / align * align;
    for (uint64_t i = 0; i < nti; i++) tis[i].d = G + base + tis[i].off;
}
static const KV *kv_get(const char *k) {
    size_t l = strlen(k);
    for (uint64_t i = 0; i < nkv; i++) if (kvs[i].kl == l && !memcmp(kvs[i].k, k, l)) return &kvs[i];
    return NULL;
}
static int64_t kv_int(const char *k, int64_t def) {
    const KV *e = kv_get(k); if (!e) return def;
    switch (e->t) {
        case 0: return *(const uint8_t *)e->v;  case 1: return *(const int8_t *)e->v;
        case 4: return *(const uint32_t *)e->v; case 5: return *(const int32_t *)e->v;
        case 7: return *(const uint8_t *)e->v;
        case 10: return (int64_t)*(const uint64_t *)e->v; case 11: return *(const int64_t *)e->v;
    }
    fprintf(stderr, "kv %s: type %u\n", k, e->t); exit(1);
}
static float kv_f32(const char *k, float def) {
    const KV *e = kv_get(k); if (!e) return def;
    if (e->t != 6) { fprintf(stderr, "kv %s not f32\n", k); exit(1); }
    return *(const float *)e->v;
}
// int32 array: returns element pointer and count
static const int32_t *kv_i32arr(const char *k, uint64_t *n) {
    const KV *e = kv_get(k); if (!e || e->t != 9) return NULL;
    const uint8_t *p = e->v; uint32_t at = rd32(&p); *n = rd64(&p);
    if (at != 5 && at != 4) { fprintf(stderr, "kv %s: array type %u\n", k, at); exit(1); }
    return (const int32_t *)p;
}
static const TI *tget(const char *n, int must) {
    size_t l = strlen(n);
    for (uint64_t i = 0; i < nti; i++) if (tis[i].nl == l && !memcmp(tis[i].n, n, l)) return &tis[i];
    if (must) { fprintf(stderr, "missing tensor %s\n", n); exit(1); }
    return NULL;
}

/* ---------------- model ---------------- */
/* ======================= device arithmetic (fork x86 order) ======================= */
__device__ __forceinline__ float f16r(float x) { return __half2float(__float2half_rn(x)); }
// ggml_v_expf (AVX2+FMA), one lane
__device__ __forceinline__ float d_vexpf(float x) {
    const float r = 0x1.8p23f;
    const float z = fmaf(x, 0x1.715476p+0f, r);
    const float n = z - r;
    const float b = fmaf(-n, 0x1.7f7d1cp-20f, fmaf(-n, 0x1.62e4p-1f, x));
    const uint32_t e = __float_as_uint(z) << 23;
    const float k = __uint_as_float(e + __float_as_uint(1.0f));
    const float u = b * b;
    const float j = fmaf(fmaf(fmaf(0x1.0e4020p-7f, b, 0x1.573e2ep-5f), u,
                              fmaf(0x1.555e66p-3f, b, 0x1.fffdb6p-2f)),
                         u, 0x1.ffffecp-1f * b);
    if (!(fabsf(n) > 126.0f)) return fmaf(j, k, k);
    const uint32_t g = (n <= 0.0f) ? 0x82000000u : 0u;
    const float s1 = __uint_as_float(g + 0x7f000000u);
    const float s2 = __uint_as_float(e - g);
    if (fabsf(n) > 192.0f) return s1 * s1;
    return fmaf(s2, j, s2) * s1;
}
__device__ __forceinline__ float d_silu(float x) { return x / (1.0f + d_vexpf(0.0f - x)); }
__device__ __forceinline__ float d_sigm(float x) { return 1.0f / (1.0f + gm_expf(-x)); }

/* Exact value of   double s = 0; for (i < n) s += (double)(x[i] * x[i]);   on one warp.
 * The terms t are floats >= 0, s only grows. Inside the binade [2^e, 2^(e+1)) of s every
 * add rounds the exact sum s + t on the ulp grid u = 2^(e-52), and since s is a multiple of
 * u, RN(s + t) = s + RN_u(t) as long as the result stays in the binade and t is not an exact
 * tie (a tie rounds to even, which depends on s). So a chunk of 32 terms becomes an integer
 * prefix scan of r_i = RN_u(t_i); at the first lane whose step leaves the binade or ties,
 * the committed prefix is turned back into s and that one step is a real double add, then the
 * rest of the chunk is rescanned. While s == 0 the steps are taken one by one. */
__device__ __forceinline__ double d_mk(int e, long long S) {
    return __longlong_as_double(((long long)(e + 1023) << 52) | (S & ((1ll << 52) - 1)));
}
__device__ double wsum_sq(const float *x, int n, int lane) {
    double s = 0.0;
    for (int base = 0; base < n; base += 32) {
        const int i = base + lane;
        const float xv = i < n ? x[i] : 0.0f;
        const float t = xv * xv;
        const uint32_t tb = __float_as_uint(t);
        const int tex = (int)(tb >> 23);
        const uint32_t tm = tex ? ((tb & 0x7fffffu) | 0x800000u) : (tb & 0x7fffffu);
        const int te = tex ? tex - 150 : -149;
        unsigned todo = n - base >= 32 ? FULL : ((1u << (n - base)) - 1u);
        while (todo) {
            if (s == 0.0) {
                const int f = __ffs(todo) - 1;
                s = s + (double)__shfl_sync(FULL, t, f);
                todo &= todo - 1;
                continue;
            }
            const unsigned long long sb = (unsigned long long)__double_as_longlong(s);
            const int e = (int)(sb >> 52) - 1023;
            const long long S0 = (long long)((sb & ((1ull << 52) - 1)) | (1ull << 52));
            const int sh = te - (e - 52);
            long long r = 0;
            bool ev = false;
            if (tm) {
                if (sh >= 0) { if (sh > 29) ev = true; else r = (long long)tm << sh; }
                else {
                    const int k = -sh;
                    if (k <= 25) {
                        const uint32_t half = 1u << (k - 1), rem = tm & ((half << 1) - 1u);
                        if (rem == half) ev = true;
                        r = (long long)(tm >> k) + (rem > half ? 1 : 0);
                    }
                }
            }
            const bool act = (todo >> lane) & 1u;
            if (!act) { r = 0; ev = false; }
            long long P = r;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const long long v = __shfl_up_sync(FULL, P, o);
                if (lane >= o) P += v;
            }
            if (act && S0 + P >= (1ll << 53)) ev = true;
            const unsigned evm = __ballot_sync(FULL, ev);
            if (!evm) {
                s = d_mk(e, S0 + __shfl_sync(FULL, P, 31));
                todo = 0;
            } else {
                const int f = __ffs(evm) - 1;
                const long long Pb = __shfl_sync(FULL, P, f > 0 ? f - 1 : 0);
                const double tf = (double)__shfl_sync(FULL, t, f);
                s = d_mk(e, S0 + (f > 0 ? Pb : 0)) + tf;
                todo &= f == 31 ? 0u : (FULL << (f + 1));
            }
        }
    }
    return s;
}
__device__ __forceinline__ float rms_scale(double s, int n, float eps) {
    const float mean = (float)(s / n);
    return 1.0f / sqrtf(mean + eps);
}
// ggml_vec_dot_f32 AVX2 reduction: lane 8j+l holds accumulator a[j][l]
__device__ __forceinline__ float dotg_reduce(float a, int lane) {
    float o = __shfl_down_sync(FULL, a, 16); if (lane < 16) a = a + o;
    o = __shfl_down_sync(FULL, a, 8); if (lane < 8) a = a + o;
    o = __shfl_down_sync(FULL, a, 4); if (lane < 4) a = a + o;
    o = __shfl_down_sync(FULL, a, 1); if (!(lane & 1)) a = a + o;
    o = __shfl_down_sync(FULL, a, 2); if (lane == 0) a = a + o;
    return __shfl_sync(FULL, a, 0);
}
// tinyBLAS<8> hsum: lanes 0..7 hold c[i]
__device__ __forceinline__ float dott_reduce(float c, int lane) {
    float o = __shfl_down_sync(FULL, c, 4); if (lane < 4) c = o + c;
    o = __shfl_down_sync(FULL, c, 2); if (lane < 2) c = c + o;
    o = __shfl_down_sync(FULL, c, 1); if (lane == 0) c = c + o;
    return __shfl_sync(FULL, c, 0);
}

/* ---- Q8_0 activation, bit-sliced: per 128-group 40 words [d8 x4][N x4][Q0 x4]..[Q6 x4][cg x4],
 * word k of a plane = elements 32k..32k+31 of the group (bit = element), N = sign, Q_b =
 * bit b of |q|, cg = -(S + S_N) of the GEMV identity below (S = sum |q|, S_N = sum over q < 0).
 * quantize_row_q8_0: d = amax/127 stored fp16, q = rint(x * 127/amax). */
__device__ __forceinline__ void q8word(float xv, int lane, uint32_t *pl, int word) {
    float amax = fabsf(xv);
#pragma unroll
    for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(FULL, amax, o));
    const float d = amax / 127.0f;
    const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
    const int q = (int)rintf(xv * id);
    const uint32_t N = __ballot_sync(FULL, q < 0);
    const int a = q < 0 ? -q : q;
    uint32_t *g = pl + (word >> 2) * 40;
    const int k = word & 3;
    if (lane == 0) g[k] = __float_as_uint(f16r(d));
    if (lane == 1) g[4 + k] = N;
#pragma unroll
    for (int b = 0; b < 7; b++) {
        const uint32_t Q = __ballot_sync(FULL, (a >> b) & 1);
        if (lane == 2 + b) g[8 + 4 * b + k] = Q;
    }
    int c = q < 0 ? 2 * a : a;
#pragma unroll
    for (int o = 16; o; o >>= 1) c += __shfl_xor_sync(FULL, c, o);
    if (lane == 9) g[36 + k] = (uint32_t)(-c);
}
// v[0..bs) in shared: scale, Sylvester butterflies, then Q8_0 planes of words b0/32 + ...
__device__ void fwht_q8(float *v, int bs, float fsc, uint32_t *pl, int b0) {
    const int T = blockDim.x, tid = threadIdx.x;
    for (int i = tid; i < bs; i += T) v[i] = v[i] * fsc;
    __syncthreads();
    for (int len = 1; len < bs; len <<= 1) {
        for (int p = tid; p < bs / 2; p += T) {
            const int i = ((p & ~(len - 1)) << 1) | (p & (len - 1));
            const float u = v[i], w = v[i + len];
            v[i] = u + w; v[i + len] = u - w;
        }
        __syncthreads();
    }
    const int lane = tid & 31, wid = tid >> 5, nw = T >> 5;
    for (int wd = wid; wd < bs / 32; wd += nw) q8word(v[wd * 32 + lane], lane, pl, (b0 >> 5) + wd);
}
// rms_norm(x) * w -> xn (optional), then signs, FWHT, Q8_0. One block per Hadamard block;
// warp 0 of every block computes the (exact, cheap) full-width sum.
__global__ void k_norm_xq(const float *x, const float *w, int n, float eps, float *xn,
                          const float *sg, int bs, float fsc, uint32_t *pl, int pls) {
    extern __shared__ float sv[];
    __shared__ float s_sc;
    const int b0 = blockIdx.x * bs, tid = threadIdx.x;
    x += (size_t)blockIdx.y * n; if (xn) xn += (size_t)blockIdx.y * n; pl += (size_t)blockIdx.y * pls;
    if (tid < 32) {
        const double s = wsum_sq(x, n, tid);
        if (tid == 0) s_sc = rms_scale(s, n, eps);
    }
    __syncthreads();
    const float sc = s_sc;
    for (int i = tid; i < bs; i += blockDim.x) {
        const float y = (x[b0 + i] * sc) * w[b0 + i];
        if (xn) xn[b0 + i] = y;
        sv[i] = y * sg[b0 + i];
    }
    fwht_q8(sv, bs, fsc, pl, b0);
}
// xform + Q8_0 of a: mode 0 plain, 1 GDN head regroup (rep = NV/NK), 2 silu(a) * b
__global__ void k_xq(const float *a, const float *b, int mode, const float *sg, int bs, float fsc,
                     uint32_t *pl, int nk, int rep, int as, int pls) {
    extern __shared__ float sv[];
    const int b0 = blockIdx.x * bs, tid = threadIdx.x;
    a += (size_t)blockIdx.y * as; if (b) b += (size_t)blockIdx.y * as; pl += (size_t)blockIdx.y * pls;
    for (int i = tid; i < bs; i += blockDim.x) {
        const int ix = b0 + i;
        float v;
        if (mode == 0) v = a[ix];
        else if (mode == 1) {
            const int hd = ix >> 7, j = ix & 127, r = hd % rep, k = hd / rep;
            v = a[128 * (k + nk * r) + j];
        } else v = d_silu(a[ix]) * b[ix];
        sv[i] = v * sg[ix];
    }
    fwht_q8(sv, bs, fsc, pl, b0);
}

/* ---- ternary GEMV. Weight row r: G uint4 of plane b0, then G uint4 of plane b1 (word k of
 * group g = weights 128g + 32k ..), scale half per group. With the sign-magnitude activation
 * (N, |q| planes Q_b) and T(m) = sum_b 2^b popc((m ^ N) & Q_b) = sum m.x + S_N,
 *   sum (b0 + 2 b1 - 1) x = T(b0) + 2 T(b1) - S - S_N,  S = sum |x|, S_N = sum_N |x|.
 * T(b0) + 2 T(b1) is one ripple of carry-save adders over the 8 weight levels (b1 plane b
 * sits at level b+1): 9 POPC per 32 weights for both planes together. */
__device__ __forceinline__ uint32_t lop_mk(uint32_t m, uint32_t n, uint32_t q) {
    uint32_t r; asm("lop3.b32 %0, %1, %2, %3, 0x28;" : "=r"(r) : "r"(m), "r"(n), "r"(q)); return r;
}
__device__ __forceinline__ uint32_t lop_x3(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r; asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r;
}
__device__ __forceinline__ uint32_t lop_mj(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r; asm("lop3.b32 %0, %1, %2, %3, 0xE8;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r;
}
__device__ __forceinline__ int tw(const uint32_t (&nq)[8][4], int k, uint32_t m0, uint32_t m1) {
    const uint32_t N = nq[0][k];
    int acc = __popc(lop_mk(m0, N, nq[1][k]));
    uint32_t a = lop_mk(m0, N, nq[2][k]), b = lop_mk(m1, N, nq[1][k]);
    uint32_t s = a ^ b, c = a & b;
    acc += __popc(s) << 1;
#pragma unroll
    for (int q = 2; q < 7; q++) {
        a = lop_mk(m0, N, nq[1 + q][k]); b = lop_mk(m1, N, nq[q][k]);
        s = lop_x3(a, b, c); c = lop_mj(a, b, c);
        acc += __popc(s) << q;
    }
    b = lop_mk(m1, N, nq[7][k]);
    s = b ^ c; c = b & c;
    return acc + (__popc(s) << 7) + (__popc(c) << 8);
}
/* A team of WR warps holds R = 32 WR / G rows per pass (lane = slot * G + group); a batch is
 * RB passes = RB * R rows. Each lane leaves (sumi, d) of its group in a shared tile, lane i of
 * the team then runs row i's block chain sumf = fma(d, sumi, sumf) in group order. PD rows of
 * weights are in flight per lane. resid: y[r] = sumf + y[r] (the residual add, fused). */
template <int G, int WR, int TPB, int RB, int PD, int AD>
__global__ __launch_bounds__(32 * WR * TPB)
void k_gemv(const uint4 *__restrict__ wc, const __half *__restrict__ ws, const uint32_t *__restrict__ pl,
            float *__restrict__ y, int N, int resid) {
    constexpr int L = 32 * WR, R = L / G, RBR = RB * R, PS = G + 1;
    constexpr int NBUF = WR > 1 ? 2 : 1;
    static_assert(R >= 1 && RBR <= L && RB % PD == 0, "shape");
    __shared__ float2 tile[TPB][NBUF][RBR * PS];
    const int warp = (int)threadIdx.x >> 5, lane = (int)threadIdx.x & 31;
    const int team = warp / WR, wt = warp - team * WR, lid = wt * 32 + lane;
    const int slot = lid / G, g = lid - slot * G;
    const bool act = slot < R;
    uint32_t nq[8][4];
    float d8[4];
    int cg[4];
    if (act) {
        const uint32_t *pb = pl + g * 40;
#pragma unroll
        for (int k = 0; k < 4; k++) d8[k] = __uint_as_float(__ldg(pb + k));
#pragma unroll
        for (int q = 0; q < 8; q++)
#pragma unroll
            for (int k = 0; k < 4; k++) nq[q][k] = __ldg(pb + 4 + 4 * q + k);
#pragma unroll
        for (int k = 0; k < 4; k++) cg[k] = (int)__ldg(pb + 36 + k);
    }
    const int nbatch = (N + RBR - 1) / RBR;
    const int stride = (int)gridDim.x * TPB;
    int bi = (int)blockIdx.x * TPB + team;
    if (bi >= nbatch) return; /* team-uniform */
    /* AD 1: rows past N read the next tensor or the arena tail pad (never stored), so the
     * row addresses of a batch are one base plus immediate offsets. AD 0: clamped rows. */
    const uint4 *wb = wc + g + (size_t)(bi * RBR + slot) * (2 * G);
    const __half *sb = ws + g + (size_t)(bi * RBR + slot) * G;
    const size_t wstep = (size_t)stride * RBR * 2 * G, sstep = (size_t)stride * RBR * G;
    uint4 c0r[PD], c1r[PD];
    __half sr[PD];
    if (act) {
#pragma unroll
        for (int i = 0; i < PD; i++) {
            if (AD) { c0r[i] = __ldg(wb + i * R * 2 * G); c1r[i] = __ldg(wb + i * R * 2 * G + G); sr[i] = __ldg(sb + i * R * G); }
            else {
                const size_t r = (size_t)min(bi * RBR + i * R + slot, N - 1);
                c0r[i] = __ldg(wc + g + r * 2 * G); c1r[i] = __ldg(wc + g + r * 2 * G + G); sr[i] = ws[g + r * G];
            }
        }
    }
    int buf = 0;
    for (;;) {
        const int bnext = bi + stride;
        const bool more = bnext < nbatch;
        const uint4 *wn = more ? wb + wstep : wb;
        const __half *sn = more ? sb + sstep : sb;
        float2 *P = tile[team][buf];
        if (act) {
#pragma unroll
            for (int rr = 0; rr < RB; rr++) {
                const uint4 c0 = c0r[rr % PD], c1 = c1r[rr % PD];
                const __half wsh = sr[rr % PD];
                if (AD) {
                    const uint4 *wl = rr + PD < RB ? wb + (rr + PD) * R * 2 * G : wn + (rr + PD - RB) * R * 2 * G;
                    const __half *sl = rr + PD < RB ? sb + (rr + PD) * R * G : sn + (rr + PD - RB) * R * G;
                    c0r[rr % PD] = __ldg(wl); c1r[rr % PD] = __ldg(wl + G); sr[rr % PD] = __ldg(sl);
                } else {
                    const int nb = rr + PD < RB ? bi : (more ? bnext : bi), nr = rr + PD < RB ? rr + PD : rr + PD - RB;
                    const size_t r = (size_t)min(nb * RBR + nr * R + slot, N - 1);
                    c0r[rr % PD] = __ldg(wc + g + r * 2 * G); c1r[rr % PD] = __ldg(wc + g + r * 2 * G + G);
                    sr[rr % PD] = ws[g + r * G];
                }
                const int s0 = tw(nq, 0, c0.x, c1.x) + cg[0];
                const int s1 = tw(nq, 1, c0.y, c1.y) + cg[1];
                const int s2 = tw(nq, 2, c0.z, c1.z) + cg[2];
                const int s3 = tw(nq, 3, c0.w, c1.w) + cg[3];
                float si = fmaf(d8[0], (float)s0, 0.0f);
                si = fmaf(d8[1], (float)s1, si);
                si = fmaf(d8[2], (float)s2, si);
                si = fmaf(d8[3], (float)s3, si);
                P[(rr * R + slot) * PS + g] = make_float2(si, __half2float(wsh));
            }
        }
        if (WR == 1) __syncwarp();
        else asm volatile("bar.sync %0, %1;" ::"r"(team + 1), "r"(L));
        if (lid < RBR) {
            const int r = bi * RBR + lid;
            const float2 *src = P + lid * PS;
            float acc = 0.0f;
#pragma unroll 8
            for (int k = 0; k < G; k++) { const float2 v = src[k]; acc = fmaf(v.y, v.x, acc); }
            if (r < N) y[r] = resid ? acc + y[r] : acc;
        }
        if (!more) break;
        if (WR == 1) __syncwarp();
        else buf ^= 1;
        bi = bnext; wb = wn; sb = sn;
    }
}

/* Batched form for nt <= NTM tokens (prefill chunks, speculative verify): the RB rows of a batch
 * stay in registers while the lane walks the tokens, so a weight row is read once for all of
 * them. Per (row, token) the arithmetic is the single-token kernel's, so the bits are equal. */
template <int G, int WR, int TPB, int RB, int NTM>
__global__ __launch_bounds__(32 * WR * TPB)
void k_gemvN(const uint4 *__restrict__ wc, const __half *__restrict__ ws, const uint32_t *__restrict__ pl, int pls,
             float *__restrict__ y, int ys, int N, int resid, int nt) {
    constexpr int L = 32 * WR, R = L / G, RBR = RB * R, PS = G + 1;
    static_assert(R >= 1 && RBR <= L, "shape");
    __shared__ float2 tile[TPB][NTM][RBR * PS];
    const int warp = (int)threadIdx.x >> 5, lane = (int)threadIdx.x & 31;
    const int team = warp / WR, wt = warp - team * WR, lid = wt * 32 + lane;
    const int slot = lid / G, g = lid - slot * G;
    const bool act = slot < R;
    const int nbatch = (N + RBR - 1) / RBR;
    const int stride = (int)gridDim.x * TPB;
    const uint4 *wq = wc + g;
    const __half *sq = ws + g;
    const uint4 *pa = (const uint4 *)pl + g * 10;
    for (int bi = (int)blockIdx.x * TPB + team; bi < nbatch; bi += stride) {
        if (act) {
            uint4 c0r[RB], c1r[RB];
            float sr[RB];
#pragma unroll
            for (int i = 0; i < RB; i++) {
                const size_t r = (size_t)min(bi * RBR + i * R + slot, N - 1);
                c0r[i] = __ldg(wq + r * 2 * G); c1r[i] = __ldg(wq + r * 2 * G + G); sr[i] = __half2float(sq[r * G]);
            }
            for (int t = 0; t < nt; t++) {
                const uint4 *pt = pa + (size_t)t * (pls >> 2);
                const uint4 dv = __ldg(pt), cv = __ldg(pt + 9);
                uint32_t nq[8][4];
#pragma unroll
                for (int q = 0; q < 8; q++) {
                    const uint4 v = __ldg(pt + 1 + q);
                    nq[q][0] = v.x; nq[q][1] = v.y; nq[q][2] = v.z; nq[q][3] = v.w;
                }
                float2 *P = tile[team][t];
#pragma unroll
                for (int rr = 0; rr < RB; rr++) {
                    const uint4 c0 = c0r[rr], c1 = c1r[rr];
                    const int s0 = tw(nq, 0, c0.x, c1.x) + (int)cv.x;
                    const int s1 = tw(nq, 1, c0.y, c1.y) + (int)cv.y;
                    const int s2 = tw(nq, 2, c0.z, c1.z) + (int)cv.z;
                    const int s3 = tw(nq, 3, c0.w, c1.w) + (int)cv.w;
                    float si = fmaf(__uint_as_float(dv.x), (float)s0, 0.0f);
                    si = fmaf(__uint_as_float(dv.y), (float)s1, si);
                    si = fmaf(__uint_as_float(dv.z), (float)s2, si);
                    si = fmaf(__uint_as_float(dv.w), (float)s3, si);
                    P[(rr * R + slot) * PS + g] = make_float2(si, sr[rr]);
                }
            }
        }
        if (WR == 1) __syncwarp();
        else asm volatile("bar.sync %0, %1;" ::"r"(team + 1), "r"(L));
        if (lid < RBR) {
            const int r = bi * RBR + lid;
            for (int t = 0; t < nt; t++) {
                const float2 *src = tile[team][t] + lid * PS;
                float acc = 0.0f;
#pragma unroll 8
                for (int k = 0; k < G; k++) { const float2 v = src[k]; acc = fmaf(v.y, v.x, acc); }
                if (r < N) { float *yy = y + (size_t)t * ys + r; *yy = resid ? acc + *yy : acc; }
            }
        }
        if (WR == 1) __syncwarp();
        else asm volatile("bar.sync %0, %1;" ::"r"(team + 1), "r"(L));
    }
}

/* ---- GDN (gated delta net) layer pieces ---- */
// ssm_beta / ssm_alpha F32 projections of xn: ggml_vec_dot_f32 order (tiny = 0, one token) or
// tinyBLAS order (tiny = 1, the prompt batch)
__global__ void k_ab(const float *wb, const float *wa, const float *xn, int E, int NV, int tiny, float *out) {
    const int lane = threadIdx.x & 31;
    xn += (size_t)blockIdx.y * E; out += (size_t)blockIdx.y * 2 * NV;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= 2 * NV) return;
    const float *w = row < NV ? wb + (size_t)row * E : wa + (size_t)(row - NV) * E;
    float r;
    if (!tiny) {
        float a = 0.0f;
        int i = 0;
        for (; i + 256 <= E; i += 256) { /* loads first, then the chain in the same order */
            float wv[8], xv[8];
#pragma unroll
            for (int u = 0; u < 8; u++) { wv[u] = __ldg(w + i + 32 * u + lane); xv[u] = xn[i + 32 * u + lane]; }
#pragma unroll
            for (int u = 0; u < 8; u++) a = fmaf(wv[u], xv[u], a);
        }
        for (; i < E; i += 32) a = fmaf(w[i + lane], xn[i + lane], a);
        r = dotg_reduce(a, lane);
    } else {
        float c = 0.0f;
        if (lane < 8) for (int l = 0; l < E; l += 8) c = fmaf(w[l + lane], xn[l + lane], c);
        r = dott_reduce(c, lane);
    }
    if (lane == 0) out[row] = r;
}
// causal conv over [s0 s1 s2 x] (products rounded, summed in order), silu; state shifts
__global__ void k_conv(const float *qkv, const float *cw, float *st, int CH, float *cv, int nt, int qs) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= CH) return;
    const float *k = cw + 4 * c;
    float w0 = st[c], w1 = st[CH + c], w2 = st[2 * CH + c];
    for (int t = 0; t < nt; t++) {
        const float w3 = qkv[(size_t)t * qs + c];
        float s = 0.0f;
        s += w0 * k[0]; s += w1 * k[1]; s += w2 * k[2]; s += w3 * k[3];
        cv[(size_t)t * CH + c] = d_silu(s);
        w0 = w1; w1 = w2; w2 = w3;
    }
    st[c] = w0; st[CH + c] = w1; st[2 * CH + c] = w2;
}
// l2_norm per 128-wide head, one warp per head, in place
__global__ void k_l2(float *x, int nh, float eps, int xs) {
    const int lane = threadIdx.x & 31, h = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    x += (size_t)blockIdx.y * xs;
    if (h >= nh) return;
    float *v = x + 128 * h;
    const double s = wsum_sq(v, 128, lane);
    const float sc = 1.0f / fmaxf(sqrtf((float)s), eps);
    for (int i = lane; i < 128; i += 32) v[i] = v[i] * sc;
}
// one block per v head; warp per state row j (row j of M is the ref's M + j*DS). A warp runs
// RW rows j, j + nw, .. side by side (independent chains, same per-row arithmetic).
template <int RW>
__global__ void k_gdn(const float *cv0, const float *ab0, const float *ssm_a, const float *dtb, float *st,
                      float *o0, int NV, int NK, float scale, int nt, int CH, int INNER) {
    const int h = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  for (int t = 0; t < nt; t++) {
    const float *cv = cv0 + (size_t)t * CH, *ab = ab0 + (size_t)t * 2 * NV;
    float *o = o0 + (size_t)t * INNER;
    const float *q = cv + 128 * (h % NK), *k = cv + NK * 128 + 128 * (h % NK), *v = cv + 2 * NK * 128 + 128 * h;
    __shared__ float s_be[2];
    if (threadIdx.x == 0) { /* the glibc-exact exp/log run in double (1/32 rate): once per block */
        const float xg = ab[NV + h] + dtb[h];
        const float g = ssm_a[h] * ((xg > 20.0f) ? xg : gm_logf(1.0f + gm_expf(xg)));
        s_be[0] = 1.0f / (1.0f + gm_expf(-ab[h]));
        s_be[1] = gm_expf(g);
    }
    __syncthreads();
    const float bt = s_be[0], eg = s_be[1];
    float kr[4], qr[4];
#pragma unroll
    for (int c = 0; c < 4; c++) { kr[c] = k[32 * c + lane]; qr[c] = q[32 * c + lane]; }
    float *M = st + (size_t)h * 128 * 128;
    for (int j0 = warp; j0 < 128; j0 += RW * nw) {
        float m[RW][4], a[RW], dt[RW], a2[RW];
#pragma unroll
        for (int u = 0; u < RW; u++) {
            a[u] = 0.0f;
#pragma unroll
            for (int c = 0; c < 4; c++) { m[u][c] = M[(j0 + u * nw) * 128 + 32 * c + lane] * eg; a[u] = fmaf(m[u][c], kr[c], a[u]); }
        }
#pragma unroll
        for (int u = 0; u < RW; u++) dt[u] = dotg_reduce(a[u], lane);
#pragma unroll
        for (int u = 0; u < RW; u++) {
            const int j = j0 + u * nw;
            const float delta = (v[j] - dt[u]) * bt;
            a2[u] = 0.0f;
#pragma unroll
            for (int c = 0; c < 4; c++) {
                m[u][c] = fmaf(kr[c], delta, m[u][c]);
                a2[u] = fmaf(m[u][c], qr[c], a2[u]);
                M[j * 128 + 32 * c + lane] = m[u][c];
            }
        }
#pragma unroll
        for (int u = 0; u < RW; u++) {
            const float oo = dotg_reduce(a2[u], lane) * scale;
            if (lane == 0) o[h * 128 + j0 + u * nw] = oo;
        }
    }
    __syncthreads(); /* s_be is rewritten for the next token */
  }
}
// gated norm: silu(z) * rms_norm(o_head) * ssm_norm, warp per head
__global__ void k_gnorm(const float *o, const float *z, const float *w, int nh, float eps, float *out, int zs) {
    const int lane = threadIdx.x & 31, h = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    o += (size_t)blockIdx.y * nh * 128; out += (size_t)blockIdx.y * nh * 128; z += (size_t)blockIdx.y * zs;
    if (h >= nh) return;
    const double s = wsum_sq(o + 128 * h, 128, lane);
    const float sc = rms_scale(s, 128, eps);
    for (int i = lane; i < 128; i += 32) {
        const float nrm = (o[128 * h + i] * sc) * w[i];
        out[128 * h + i] = d_silu(z[128 * h + i]) * nrm;
    }
}

/* ---- full attention layer pieces (HD 256) ---- */
// q/k rms_norm + imrope on the first nrot dims; k and v go to the f16 cache at pos
__global__ void k_qk(const float *buf, const float *qn, const float *kn, const int *posp, float *qo,
                     __half *kc, __half *vc, int NH, int NKV, int nrot, float ts, float eps, int bs_) {
    __shared__ float sh[8][256];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int hh = blockIdx.x * (blockDim.x >> 5) + warp;
    if (hh >= NH + NKV) return;
    const int pos = *posp + blockIdx.y;
    buf += (size_t)blockIdx.y * bs_; qo += (size_t)blockIdx.y * NH * 256;
    const bool isq = hh < NH;
    const float *x = isq ? buf + hh * 512 : buf + NH * 512 + (hh - NH) * 256;
    const float *w = isq ? qn : kn;
    const double s = wsum_sq(x, 256, lane);
    const float sc = rms_scale(s, 256, eps);
    float *y = sh[warp];
    for (int i = lane; i < 256; i += 32) y[i] = (x[i] * sc) * w[i];
    __syncwarp();
    const int half = nrot / 2;
    if (lane < half) {
        float th = (float)pos;
        for (int t = 0; t < lane; t++) th *= ts;
        const float c = gm_cosf(th), sn = gm_sinf(th);
        const float x0 = y[lane], x1 = y[lane + half];
        y[lane] = fmaf(x0, c, -(x1 * sn));
        y[lane + half] = fmaf(x0, sn, x1 * c);
    }
    __syncwarp();
    if (isq) {
        for (int i = lane; i < 256; i += 32) qo[hh * 256 + i] = y[i];
    } else {
        const int kh = hh - NH;
        const float *vv = buf + NH * 512 + NKV * 256 + kh * 256;
        __half *kd = kc + (size_t)pos * NKV * 256 + kh * 256, *vd = vc + (size_t)pos * NKV * 256 + kh * 256;
        for (int i = lane; i < 256; i += 32) { kd[i] = __float2half_rn(y[i]); vd[i] = __float2half_rn(vv[i]); }
    }
}
/* flash attention, f16 K/V, the fork's one_chunk online softmax: per position
 * s = dot(k, f16(q)) * scale; if s > M: M = s, ms = exp(Mold - M), VKQ = f16(VKQ * ms)
 * else vs = exp(s - M); VKQ = f16(fma(v, vs, VKQ)); S = fma(S, ms, vs). The scores and the
 * (ms | vs) of every position only depend on the running max, a prefix max, so they are
 * computed in parallel; the f16 VKQ chain stays sequential per output dim. Then
 * out = VKQ / S * sigmoid(gate). One block of 256 threads per q head. */
#ifndef FAU
#define FAU 16
#endif
#ifndef FAK
#define FAK 4
#endif
__global__ void k_fa(const float *q, const float *buf, const __half *kc, const __half *vc, const int *posp,
                     float *att, float *fao, int NH, int NKV, int ctx, float scale, int bs_) {
    extern __shared__ float fsh[];
    __shared__ float qs[256], cm[256];
    /* one float per position: the score, then in place the weight (ms | vs, >= +0) with the
       sign bit set where the running max moved (ms); 4 B instead of 9 B per position */
    float *sc = fsh;
    const int h = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n = *posp + (int)blockIdx.y + 1, g = h / (NH / NKV);
    q += (size_t)blockIdx.y * NH * 256; buf += (size_t)blockIdx.y * bs_; att += (size_t)blockIdx.y * NH * 256;
    if (fao) fao += (size_t)blockIdx.y * NH * 256;
    qs[tid] = f16r(q[h * 256 + tid]);
    __syncthreads();
    float qr[8];
#pragma unroll
    for (int c = 0; c < 8; c++) qr[c] = qs[32 * c + lane];
    /* FAK positions per warp pass, their K loads issued together */
    for (int ic0 = warp * FAK; ic0 < n; ic0 += 8 * FAK) {
        __half kh[FAK][8];
#pragma unroll
        for (int u = 0; u < FAK; u++) {
            const __half *kr = kc + (size_t)min(ic0 + u, n - 1) * NKV * 256 + g * 256;
#pragma unroll
            for (int c = 0; c < 8; c++) kh[u][c] = kr[32 * c + lane];
        }
#pragma unroll
        for (int u = 0; u < FAK; u++) {
            float a = 0.0f;
#pragma unroll
            for (int c = 0; c < 8; c++) a = fmaf(__half2float(kh[u][c]), qr[c], a);
            const float s = dotg_reduce(a, lane) * scale;
            if (lane == 0 && ic0 + u < n) sc[ic0 + u] = s;
        }
    }
    __syncthreads();
    const int chunk = (n + 255) / 256, lo = min(tid * chunk, n), hi = min(lo + chunk, n);
    float m = -INFINITY;
    for (int i = lo; i < hi; i++) m = fmaxf(m, sc[i]);
    cm[tid] = m;
    __syncthreads();
    if (tid == 0) {
        float run = -INFINITY;
        for (int t = 0; t < 256; t++) { const float v = cm[t]; cm[t] = run; run = fmaxf(run, v); }
    }
    __syncthreads();
    float run = cm[tid];
    for (int i = lo; i < hi; i++) {
        const float s = sc[i];
        if (s > run) { sc[i] = -gm_expf(run - s); run = s; }
        else sc[i] = gm_expf(s - run);
    }
    __syncthreads();
    float vk = 0.0f, S = 0.0f;
    const __half *vp = vc + g * 256 + tid;
    /* the V rows are independent of the f16 chain: issue FAU loads at once so one memory
       latency covers FAU positions instead of one */
    for (int ic0 = 0; ic0 < n; ic0 += FAU) {
        __half vh[FAU];
#pragma unroll
        for (int u = 0; u < FAU; u++) if (ic0 + u < n) vh[u] = vp[(size_t)(ic0 + u) * NKV * 256];
#pragma unroll
        for (int u = 0; u < FAU; u++) {
            if (ic0 + u >= n) break;
            const float sw = sc[ic0 + u], w = fabsf(sw);
            const float vv = __half2float(vh[u]);
            if (__float_as_uint(sw) >> 31) { vk = f16r(vk * w); vk = f16r(fmaf(vv, 1.0f, vk)); S = fmaf(S, w, 1.0f); }
            else { vk = f16r(fmaf(vv, w, vk)); S = fmaf(S, 1.0f, w); }
        }
    }
    const float Si = S == 0.0f ? 0.0f : 1.0f / S;
    const float acc = vk * Si;
    if (fao) fao[h * 256 + tid] = acc;
    att[h * 256 + tid] = acc * d_sigm(buf[h * 512 + 256 + tid]);
}

/* ---- argmax: first index of the maximum ---- */
__device__ __forceinline__ void amx(float &v, int &i, float v2, int i2) {
    if (v2 > v || (v2 == v && i2 < i)) { v = v2; i = i2; }
}
__global__ void k_argmax1(const float *x, int n, float *pv, int *pi) {
    __shared__ float sv[32]; __shared__ int si[32];
    x += (size_t)blockIdx.y * n; pv += blockIdx.y * gridDim.x; pi += blockIdx.y * gridDim.x;
    float v = -INFINITY; int ix = 0x7fffffff;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) amx(v, ix, x[i], i);
    for (int o = 16; o; o >>= 1) amx(v, ix, __shfl_xor_sync(FULL, v, o), __shfl_xor_sync(FULL, ix, o));
    if ((threadIdx.x & 31) == 0) { sv[threadIdx.x >> 5] = v; si[threadIdx.x >> 5] = ix; }
    __syncthreads();
    if (threadIdx.x < 32) {
        v = threadIdx.x < (blockDim.x >> 5) ? sv[threadIdx.x] : -INFINITY;
        ix = threadIdx.x < (blockDim.x >> 5) ? si[threadIdx.x] : 0x7fffffff;
        for (int o = 16; o; o >>= 1) amx(v, ix, __shfl_xor_sync(FULL, v, o), __shfl_xor_sync(FULL, ix, o));
        if (threadIdx.x == 0) { pv[blockIdx.x] = v; pi[blockIdx.x] = ix; }
    }
}
__global__ void k_argmax2(const float *pv, const int *pi, int n, int *out) {
    float v = -INFINITY; int ix = 0x7fffffff;
    pv += blockIdx.x * n; pi += blockIdx.x * n; out += blockIdx.x;
    for (int i = threadIdx.x; i < n; i += 32) amx(v, ix, pv[i], pi[i]);
    for (int o = 16; o; o >>= 1) amx(v, ix, __shfl_xor_sync(FULL, v, o), __shfl_xor_sync(FULL, ix, o));
    if (threadIdx.x == 0) *out = ix;
}

/* ======================= host: model ======================= */
#define T_F32 0
#define T_PTQ1 143
#define PTQ_BYTES 28   /* qs[24] qh[2] d(fp16) per 128 weights */
typedef struct { const TI *t; int K, N; size_t co, so; } DT;   /* device tensor: uint4 / half offsets */
typedef struct {
    int full;
    float *attn_norm, *post_norm, *ssm_a, *ssm_alpha, *ssm_beta, *conv1d, *dt_bias, *ssm_norm, *q_norm, *k_norm;
    DT in, out, ffu, ffd;   /* in: qkv+gate | q+k+v merged; out: ssm_out | wo; ffu: gate+up; ffd: down */
} Layer;

static int NL, E, FF, NH, NKV, HD = 256, NV, NK, DS = 128, INNER, CH, NROT, VOCAB, BS, GROUPED, TIED, CTX = 4096;
static float EPS, ROPE_BASE, ROPE_TS, FSC;
static Layer *Ls;
static const TI *tok_t;
static DT head;
static float *d_out_norm_w;
static int nsign; static int sign_w[8]; static float *sign_h[8], *sign_d[8];
static int8_t LUT[256][5];
static uint4 *d_codes; static __half *d_scales;
static size_t n_codes, n_scales;

static const TI *ptq(const char *n) {
    const TI *t = tget(n, 1);
    if (t->type != T_PTQ1) { fprintf(stderr, "%s: type %u, want PTQ1_0\n", n, t->type); exit(1); }
    return t;
}
static float *f32d(const char *n) {
    const TI *t = tget(n, 1);
    if (t->type != T_F32) { fprintf(stderr, "%s: type %u, want F32\n", n, t->type); exit(1); }
    size_t c = 1; for (int j = 0; j < 4; j++) c *= (size_t)t->ne[j];
    float *d; CK(cudaMalloc(&d, c * 4)); CK(cudaMemcpy(d, t->d, c * 4, cudaMemcpyHostToDevice));
    return d;
}
static int sign_ix(int w) {
    for (int i = 0; i < nsign; i++) if (sign_w[i] == w) return i;
    fprintf(stderr, "no sign vector for width %d\n", w); exit(1);
}
/* merged device tensor: the listed PTQ1 tensors' rows back to back (same K) */
static DT dt_alloc(const TI **ts, int nt) {
    DT d; d.t = NULL; d.K = (int)ts[0]->ne[0]; d.N = 0; d.co = n_codes; d.so = n_scales;
    for (int i = 0; i < nt; i++) {
        if ((int)ts[i]->ne[0] != d.K) { fprintf(stderr, "merge: K mismatch\n"); exit(1); }
        d.N += (int)ts[i]->ne[1];
    }
    const size_t G = d.K / 128;
    n_codes += (size_t)d.N * 2 * G; n_scales += (size_t)d.N * G;
    n_scales = (n_scales + 127) & ~(size_t)127;
    return d;
}
/* PTQ1_0 rows -> b0/b1 planes. Trit n of qs[m] is weight 16n + m (m < 16), of qs[16 + m]
 * weight 80 + 8n + m, of qh[h] weight 120 + 2n + h. */
static void repack_rows(const uint8_t *src, int K, int nrows, uint32_t *codes, uint16_t *scales) {
    const int G = K / 128;
#pragma omp parallel for schedule(static)
    for (int r = 0; r < nrows; r++) {
        const uint8_t *row = src + (size_t)r * G * PTQ_BYTES;
        uint32_t *c = codes + (size_t)r * 8 * G;
        for (int g = 0; g < G; g++) {
            const uint8_t *b = row + PTQ_BYTES * g;
            uint32_t p0[4] = {0, 0, 0, 0}, p1[4] = {0, 0, 0, 0};
            int8_t t[128];
            for (int n = 0; n < 5; n++) for (int m = 0; m < 16; m++) t[n * 16 + m] = LUT[b[m]][n];
            for (int n = 0; n < 5; n++) for (int m = 0; m < 8; m++) t[80 + n * 8 + m] = LUT[b[16 + m]][n];
            for (int n = 0; n < 4; n++) for (int h = 0; h < 2; h++) t[120 + n * 2 + h] = LUT[b[24 + h]][n];
            for (int j = 0; j < 128; j++) {
                const int code = t[j] + 1;
                p0[j >> 5] |= (uint32_t)(code & 1) << (j & 31);
                p1[j >> 5] |= (uint32_t)(code >> 1) << (j & 31);
            }
            for (int k = 0; k < 4; k++) { c[4 * g + k] = p0[k]; c[4 * G + 4 * g + k] = p1[k]; }
            memcpy(scales + (size_t)r * G + g, b + 26, 2);
        }
    }
}
static void dt_fill(const DT *d, const TI **ts, int nt) {
    const int G = d->K / 128, CHUNK = 8192;
    uint32_t *hc = (uint32_t *)malloc((size_t)CHUNK * 8 * G * 4);
    uint16_t *hs = (uint16_t *)malloc((size_t)CHUNK * G * 2);
    size_t row = 0;
    for (int i = 0; i < nt; i++) {
        const int N = (int)ts[i]->ne[1];
        for (int r0 = 0; r0 < N; r0 += CHUNK) {
            const int nr = N - r0 < CHUNK ? N - r0 : CHUNK;
            const uint8_t *src = ts[i]->d + (size_t)r0 * G * PTQ_BYTES;
            repack_rows(src, d->K, nr, hc, hs);
            CK(cudaMemcpy(d_codes + d->co + (row + r0) * 2 * G, hc, (size_t)nr * 8 * G * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_scales + d->so + (row + r0) * G, hs, (size_t)nr * G * 2, cudaMemcpyHostToDevice));
            /* the source pages are not needed again (embedding rows aside) */
            uintptr_t a = (uintptr_t)src & ~(uintptr_t)4095, e = (uintptr_t)(src + (size_t)nr * G * PTQ_BYTES) & ~(uintptr_t)4095;
            if (ts[i] != tok_t && e > a) madvise((void *)a, e - a, MADV_DONTNEED);
        }
        row += N;
    }
    free(hc); free(hs);
}

static void load(void) {
    char a[64];
    NL = kv_int("qwen35.block_count", 0); E = kv_int("qwen35.embedding_length", 0);
    FF = kv_int("qwen35.feed_forward_length", 0);
    NH = kv_int("qwen35.attention.head_count", 0); NKV = kv_int("qwen35.attention.head_count_kv", 0);
    HD = kv_int("qwen35.attention.key_length", 256);
    NK = kv_int("qwen35.ssm.group_count", 0); NV = kv_int("qwen35.ssm.time_step_rank", 0);
    DS = kv_int("qwen35.ssm.state_size", 128); INNER = kv_int("qwen35.ssm.inner_size", 0);
    CH = INNER + 2 * NK * DS;
    NROT = kv_int("qwen35.rope.dimension_count", 64);
    EPS = kv_f32("qwen35.attention.layer_norm_rms_epsilon", 1e-6f);
    ROPE_BASE = kv_f32("qwen35.rope.freq_base", 1e7f);
    const int fai = kv_int("qwen35.full_attention_interval", 4);
    BS = kv_int("prism.hadamard.block_size", 0);
    GROUPED = kv_int("prism.hadamard.gdn_v_grouped", 0);
    TIED = kv_int("prism.hadamard.tied_output", 0);
    if (!BS || kv_int("qwen35.ssm.conv_kernel", 4) != 4 || HD != 256 || DS != 128 || INNER != NV * DS ||
        NROT > 64 || NH % NKV || BS > 4096) {
        fprintf(stderr, "unsupported config\n"); exit(1);
    }
    uint64_t nw, nv; const int32_t *sw = kv_i32arr("prism.hadamard.sign_widths", &nw);
    const int32_t *sv = kv_i32arr("prism.hadamard.sign_values", &nv);
    uint64_t o = 0;
    for (uint64_t i = 0; i < nw; i++) {
        int32_t swi; memcpy(&swi, sw + i, 4); sign_w[nsign] = swi; sign_h[nsign] = (float *)malloc(sizeof(float) * swi);
        for (int j = 0; j < swi; j++) { int32_t t; memcpy(&t, sv + o + j, 4); sign_h[nsign][j] = (float)t; }
        CK(cudaMalloc(&sign_d[nsign], sizeof(float) * swi));
        CK(cudaMemcpy(sign_d[nsign], sign_h[nsign], sizeof(float) * swi, cudaMemcpyHostToDevice));
        o += swi; nsign++;
    }
    if (o != nv) { fprintf(stderr, "sign_values length\n"); exit(1); }
    static const uint8_t pow3[5] = {1, 3, 9, 27, 81};
    for (int b = 0; b < 256; b++)
        for (int n = 0; n < 5; n++) {
            uint8_t v = (uint8_t)(b * pow3[n]);
            LUT[b][n] = (int8_t)((((uint16_t)v * 3) >> 8) - 1);
        }
    tok_t = ptq("token_embd.weight"); VOCAB = (int)tok_t->ne[1];
    Ls = (Layer *)calloc(NL, sizeof(Layer));
    /* pass 1: layout; pass 2: fill */
    const TI *lt[64][4][3]; int lnt[64][4];
    for (int l = 0; l < NL; l++) {
        Layer *L = &Ls[l];
        L->full = ((l + 1) % fai) == 0;
#define N_(s) (snprintf(a, sizeof a, "blk.%d." s, l), a)
        if (L->full) {
            lt[l][0][0] = ptq(N_("attn_q.weight")); lt[l][0][1] = ptq(N_("attn_k.weight"));
            lt[l][0][2] = ptq(N_("attn_v.weight")); lnt[l][0] = 3;
            lt[l][1][0] = ptq(N_("attn_output.weight")); lnt[l][1] = 1;
        } else {
            lt[l][0][0] = ptq(N_("attn_qkv.weight")); lt[l][0][1] = ptq(N_("attn_gate.weight")); lnt[l][0] = 2;
            lt[l][1][0] = ptq(N_("ssm_out.weight")); lnt[l][1] = 1;
        }
        lt[l][2][0] = ptq(N_("ffn_gate.weight")); lt[l][2][1] = ptq(N_("ffn_up.weight")); lnt[l][2] = 2;
        lt[l][3][0] = ptq(N_("ffn_down.weight")); lnt[l][3] = 1;
        L->in = dt_alloc(lt[l][0], lnt[l][0]); L->out = dt_alloc(lt[l][1], lnt[l][1]);
        L->ffu = dt_alloc(lt[l][2], lnt[l][2]); L->ffd = dt_alloc(lt[l][3], lnt[l][3]);
        L->attn_norm = f32d(N_("attn_norm.weight")); L->post_norm = f32d(N_("post_attention_norm.weight"));
        if (L->full) { L->q_norm = f32d(N_("attn_q_norm.weight")); L->k_norm = f32d(N_("attn_k_norm.weight")); }
        else {
            L->ssm_a = f32d(N_("ssm_a")); L->ssm_alpha = f32d(N_("ssm_alpha.weight"));
            L->ssm_beta = f32d(N_("ssm_beta.weight")); L->conv1d = f32d(N_("ssm_conv1d.weight"));
            L->dt_bias = f32d(N_("ssm_dt.bias")); L->ssm_norm = f32d(N_("ssm_norm.weight"));
        }
#undef N_
    }
    d_out_norm_w = f32d("output_norm.weight");
    const TI *ht = TIED ? tok_t : ptq("output.weight");
    head = dt_alloc(&ht, 1);
    fprintf(stderr, "fwd35: weights %.1f MB planes + %.1f MB scales on the GPU\n", n_codes * 16 / 1048576.0, n_scales * 2 / 1048576.0);
    CK(cudaMalloc(&d_codes, n_codes * 16 + (2 << 20))); CK(cudaMalloc(&d_scales, n_scales * 2 + (1 << 20))); /* tail pad */
    const double t0 = omp_get_wtime();
    for (int l = 0; l < NL; l++) {
        Layer *L = &Ls[l];
        dt_fill(&L->in, lt[l][0], lnt[l][0]); dt_fill(&L->out, lt[l][1], lnt[l][1]);
        dt_fill(&L->ffu, lt[l][2], lnt[l][2]); dt_fill(&L->ffd, lt[l][3], lnt[l][3]);
    }
    dt_fill(&head, &ht, 1);
    fprintf(stderr, "fwd35: repack %.1f s\n", omp_get_wtime() - t0);
}

/* host embedding: dequantized row, FWHT, signs (fwd35_ref embed) */
static void embed(int id, float *x) {
    const int nb = E / 128;
    const uint8_t *row = tok_t->d + (size_t)id * nb * PTQ_BYTES;
    for (int i = 0; i < nb; i++) {
        const uint8_t *b = row + PTQ_BYTES * i;
        uint16_t dh; memcpy(&dh, b + 26, 2);
        const float d = f16f(dh);
        float *y = x + 128 * i; int o = 0;
        for (int n = 0; n < 5; n++) for (int m = 0; m < 16; m++) y[o++] = (float)LUT[b[m]][n] * d;
        for (int n = 0; n < 5; n++) for (int m = 0; m < 8; m++) y[o++] = (float)LUT[b[16 + m]][n] * d;
        for (int n = 0; n < 4; n++) for (int h = 0; h < 2; h++) y[o++] = (float)LUT[b[24 + h]][n] * d;
    }
    for (int b0 = 0; b0 < E; b0 += BS) {
        float *v = x + b0;
        for (int j = 0; j < BS; j++) v[j] = v[j] * FSC;
        for (int len = 1; len < BS; len <<= 1)
            for (int i = 0; i < BS; i += 2 * len)
                for (int j = 0; j < len; j++) {
                    const float u = v[i + j], w = v[i + len + j];
                    v[i + j] = u + w; v[i + len + j] = u - w;
                }
    }
    const float *s = sign_h[sign_ix(E)];
    for (int i = 0; i < E; i++) x[i] = x[i] * s[i];
}

/* ======================= host: forward ======================= */
static float *d_x, *d_xn, *d_buf, *d_cv, *d_ab, *d_o, *d_g, *d_q, *d_att, *d_fao, *d_logits, *d_pv;
static uint32_t *d_pl;
static int *d_pos, *d_pi, *d_out;
static float **d_conv, **d_ssm;
static __half **d_kc, **d_vc;
static cudaStream_t st;
#define NTM 8          /* most tokens per batched forward */
static int BIG, PLS;   /* d_buf / d_pl per-token strides (floats / words) */

template <int G, int WR, int TPB, int RB, int PD, int AD = 1>
static void gemv_t(const DT *d, float *y, int resid) {
    static int nres = 0;
    if (!nres) {
        int per = 0, dev = 0, nsm = 1;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k_gemv<G, WR, TPB, RB, PD, AD>, 32 * WR * TPB, 0));
        cudaGetDevice(&dev); cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        nres = (per > 0 ? per : 1) * nsm;
    }
    constexpr int RBR = RB * (32 * WR / G);
    const int nbatch = (d->N + RBR - 1) / RBR;
    int nblk = (nbatch + TPB - 1) / TPB;
    if (nblk > nres) nblk = nres;
    k_gemv<G, WR, TPB, RB, PD, AD><<<nblk, 32 * WR * TPB, 0, st>>>(d_codes + d->co, d_scales + d->so, d_pl, y, d->N, resid);
}
template <int G, int WR, int TPB, int RB>
static void gemvN_t(const DT *d, float *y, int resid, int nt, int ys) {
    static int nres = 0;
    if (!nres) {
        int per = 0, dev = 0, nsm = 1;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, k_gemvN<G, WR, TPB, RB, NTM>, 32 * WR * TPB, 0));
        cudaGetDevice(&dev); cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        nres = (per > 0 ? per : 1) * nsm;
    }
    constexpr int RBR = RB * (32 * WR / G);
    const int nbatch = (d->N + RBR - 1) / RBR;
    int nblk = (nbatch + TPB - 1) / TPB;
    if (nblk > nres) nblk = nres;
    k_gemvN<G, WR, TPB, RB, NTM><<<nblk, 32 * WR * TPB, 0, st>>>(d_codes + d->co, d_scales + d->so, d_pl, PLS, y, ys,
                                                                  d->N, resid, nt);
}
static void gemv(const DT *d, float *y, int resid);
static void gemvn(const DT *d, float *y, int resid, int nt, int ys) {
    if (nt == 1) { gemv(d, y, resid); return; }
    switch (d->K / 128) {
    case 16: gemvN_t<16, 1, 8, 2>(d, y, resid, nt, ys); break;
    case 20: gemvN_t<20, 5, 2, 2>(d, y, resid, nt, ys); break;
    case 32: gemvN_t<32, 1, 8, 2>(d, y, resid, nt, ys); break;
    case 48: gemvN_t<48, 3, 2, 2>(d, y, resid, nt, ys); break;
    case 72: gemvN_t<72, 9, 1, 2>(d, y, resid, nt, ys); break;
    case 96: gemvN_t<96, 3, 2, 2>(d, y, resid, nt, ys); break;
    default: fprintf(stderr, "gemv: K %d unsupported\n", d->K); exit(1);
    }
}
static void gemv(const DT *d, float *y, int resid) {
    static int v20 = -1, v72 = -1, v32 = -1, v16 = -1, v48 = -1, v96 = -1;
    if (v20 < 0) {
        const char *e16 = getenv("E_V16"), *e48 = getenv("E_V48");
        v16 = e16 ? atoi(e16) : 0; v48 = e48 ? atoi(e48) : 0;
        const char *e = getenv("E_V20"); v20 = e ? atoi(e) : 0;
        e = getenv("E_V72"); v72 = e ? atoi(e) : 0; e = getenv("E_V32"); v32 = e ? atoi(e) : 0;
        e = getenv("E_V96"); v96 = e ? atoi(e) : 0;
    }
    if (d->K == 12288 && v96) {
        switch (v96) {
        case 1: gemv_t<96, 3, 2, 4, 2>(d, y, resid); return;
        case 2: gemv_t<96, 3, 2, 3, 3, 0>(d, y, resid); return;
        case 3: gemv_t<96, 6, 1, 2, 2, 0>(d, y, resid); return;
        case 4: gemv_t<96, 3, 1, 4, 2, 0>(d, y, resid); return;
        case 5: gemv_t<96, 3, 4, 2, 2, 0>(d, y, resid); return;
        }
    }
    if (d->K == 2048 && v16) {
        switch (v16) {
        case 1: gemv_t<16, 1, 8, 2, 2, 0>(d, y, resid); return;
        case 2: gemv_t<16, 1, 4, 2, 2, 0>(d, y, resid); return;
        }
    }
    if (d->K == 6144 && v48) {
        switch (v48) {
        case 1: gemv_t<48, 3, 2, 4, 2, 0>(d, y, resid); return;
        case 2: gemv_t<48, 3, 2, 3, 3, 0>(d, y, resid); return;
        case 3: gemv_t<48, 3, 2, 3, 3>(d, y, resid); return;
        case 4: gemv_t<48, 3, 1, 3, 3, 0>(d, y, resid); return;
        }
    }
    if (d->K == 2560 && v20) {
        switch (v20) {
        case 1: gemv_t<20, 5, 1, 3, 3, 0>(d, y, resid); return;
        }
    }
    if (d->K == 9216 && v72) {
        switch (v72) {
        case 1: gemv_t<72, 9, 1, 2, 1, 0>(d, y, resid); return;
        }
    }
    if (d->K == 4096 && v32) {
        switch (v32) {
        case 1: gemv_t<32, 1, 8, 4, 2, 0>(d, y, resid); return;
        case 2: gemv_t<32, 1, 8, 3, 3>(d, y, resid); return;
        case 3: gemv_t<32, 1, 4, 4, 2>(d, y, resid); return;
        case 4: gemv_t<32, 2, 4, 4, 2>(d, y, resid); return;
        case 5: gemv_t<32, 1, 8, 6, 3>(d, y, resid); return;
        }
    }
    switch (d->K / 128) {
    case 16: gemv_t<16, 1, 8, 2, 2, 0>(d, y, resid); break;
    case 20: gemv_t<20, 5, 1, 3, 3>(d, y, resid); break;
    case 32: gemv_t<32, 1, 8, 4, 2>(d, y, resid); break;
    case 48: gemv_t<48, 3, 2, 4, 2, 0>(d, y, resid); break;
    case 72: gemv_t<72, 9, 1, 2, 1, 0>(d, y, resid); break;
    case 96: gemv_t<96, 3, 2, 4, 2, 0>(d, y, resid); break;
    default: fprintf(stderr, "gemv: K %d unsupported\n", d->K); exit(1);
    }
}
static void norm_xq(const float *x, const float *w, float *xn, int nt) {
    static int nt1 = -1;
    if (nt1 < 0) nt1 = getenv("E_NT") ? atoi(getenv("E_NT")) : 0;
    const int T = nt1 ? (BS < 1024 ? BS : 1024) : (BS / 2 < 1024 ? BS / 2 : 1024); /* E/BS blocks of BS threads overflow 2048/SM at E 2560 */
    k_norm_xq<<<dim3(E / BS, nt), T, BS * 4, st>>>(x, w, E, EPS, xn, sign_d[sign_ix(E)], BS, FSC, d_pl, PLS);
}
static void xq(const float *a, const float *b, int mode, int n, int as, int nt) {
    const int T = BS < 1024 ? BS : 1024;
    k_xq<<<dim3(n / BS, nt), T, BS * 4, st>>>(a, b, mode, sign_d[sign_ix(n)], BS, FSC, d_pl, NK, NV / NK, as, PLS);
}

/* E_TDUMP: named intermediates (fork_greedy FG_TDUMP / fwd35_ref R_TDUMP record format) */
static FILE *tdf; static int tcur = -1, tstep = -2;
static void tdump(const char *fmt, int il, const float *dv, int n) {
    if (!tdf || tcur != tstep) return;
    CK(cudaStreamSynchronize(st));
    float *h = (float *)malloc(4 * (size_t)n);
    CK(cudaMemcpy(h, dv, 4 * (size_t)n, cudaMemcpyDeviceToHost));
    char nm[64]; snprintf(nm, sizeof nm, fmt, il);
    uint32_t nl = strlen(nm); uint64_t ne[4] = {(uint64_t)n, 1, 1, 1}, nn = n;
    fwrite(&nl, 4, 1, tdf); fwrite(nm, 1, nl, tdf); fwrite(ne, 8, 4, tdf); fwrite(&nn, 8, 1, tdf); fwrite(h, 4, n, tdf);
    free(h);
}

/* nt consecutive tokens at positions *d_pos .. + nt - 1 (inputs d_x[t]). tiny: the prompt-batch
 * order of the F32 projections. want: 0 no head, 1 logits of the last token, 2 of every token. */
static int gdn_rw = 1, gdn_nw = 4; /* rows per warp chain, warps per block (128 % (rw * nw) == 0) */
static void enqueue(int nt, int tiny, int want) {
    const float gscale = 1.0f / sqrtf((float)DS), ascale = 1.0f / sqrtf((float)HD);
    for (int l = 0; l < NL; l++) {
        Layer *L = &Ls[l];
        norm_xq(d_x, L->attn_norm, d_xn, nt);
        gemvn(&L->in, d_buf, 0, nt, BIG);
        if (!L->full) {
            k_ab<<<dim3((2 * NV + 3) / 4, nt), 128, 0, st>>>(L->ssm_beta, L->ssm_alpha, d_xn, E, NV, tiny, d_ab);
            k_conv<<<(CH + 255) / 256, 256, 0, st>>>(d_buf, L->conv1d, d_conv[l], CH, d_cv, nt, BIG);
            tdump("conv_output_silu-%d", l, d_cv, CH);
            tdump("alpha-%d", l, d_ab + NV, NV); tdump("beta-%d", l, d_ab, NV);
            k_l2<<<dim3((2 * NK + 3) / 4, nt), 128, 0, st>>>(d_cv, 2 * NK, EPS, CH);
            switch (gdn_rw) {
            case 1: k_gdn<1><<<NV, 32 * gdn_nw, 0, st>>>(d_cv, d_ab, L->ssm_a, L->dt_bias, d_ssm[l], d_o, NV, NK, gscale, nt, CH, INNER); break;
            case 2: k_gdn<2><<<NV, 32 * gdn_nw, 0, st>>>(d_cv, d_ab, L->ssm_a, L->dt_bias, d_ssm[l], d_o, NV, NK, gscale, nt, CH, INNER); break;
            default: k_gdn<4><<<NV, 32 * gdn_nw, 0, st>>>(d_cv, d_ab, L->ssm_a, L->dt_bias, d_ssm[l], d_o, NV, NK, gscale, nt, CH, INNER); break;
            }
            tdump("attn_output-%d", l, d_o, INNER);
            k_gnorm<<<dim3((NV + 3) / 4, nt), 128, 0, st>>>(d_o, d_buf + CH, L->ssm_norm, NV, EPS, d_g, BIG);
            tdump("final_output-%d", l, d_g, INNER);
            xq(d_g, NULL, GROUPED && NV > NK ? 1 : 0, INNER, INNER, nt);
            gemvn(&L->out, d_x, 1, nt, E);
        } else {
            const int fi = l / 4;
            k_qk<<<dim3((NH + NKV + 3) / 4, nt), 128, 0, st>>>(d_buf, L->q_norm, L->k_norm, d_pos, d_q, d_kc[fi], d_vc[fi],
                                                             NH, NKV, NROT, ROPE_TS, EPS, BIG);
            tdump("Qcur-%d", l, d_q, NH * HD);
            k_fa<<<dim3(NH, nt), 256, (size_t)CTX * 4, st>>>(d_q, d_buf, d_kc[fi], d_vc[fi], d_pos, d_att,
                                                             tdf ? d_fao : NULL, NH, NKV, CTX, ascale, BIG);
            tdump("fa_out-%d", l, d_fao, NH * HD);
            tdump("attn_gated-%d", l, d_att, NH * HD);
            xq(d_att, NULL, 0, NH * HD, NH * HD, nt);
            gemvn(&L->out, d_x, 1, nt, E);
        }
        tdump("attn_residual-%d", l, d_x, E);
        norm_xq(d_x, L->post_norm, d_xn, nt);
        gemvn(&L->ffu, d_buf, 0, nt, BIG);
        xq(d_buf, d_buf + FF, 2, FF, BIG, nt);
        gemvn(&L->ffd, d_x, 1, nt, E);
        tdump("l_out-%d", l, d_x, E);
    }
    if (!want) return;
    const int nh = want == 2 ? nt : 1;
    norm_xq(d_x + (want == 2 ? 0 : (size_t)(nt - 1) * E), d_out_norm_w, NULL, nh);
    gemvn(&head, d_logits, 0, nh, VOCAB);
    k_argmax1<<<dim3(64, nh), 1024, 0, st>>>(d_logits, VOCAB, d_pv, d_pi);
    k_argmax2<<<nh, 32, 0, st>>>(d_pv, d_pi, 64, d_out);
}

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static cudaGraphExec_t gx[NTM + 1][2][3];
static int use_graph = 1;
static int EOS_ID;
static void run(int nt, int tiny, int want) {
    if (!use_graph || tdf) { enqueue(nt, tiny, want); return; }
    cudaGraphExec_t *e = &gx[nt][tiny][want];
    if (!*e) {
        cudaGraph_t g;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
        enqueue(nt, tiny, want);
        CK(cudaStreamEndCapture(st, &g));
        CK(cudaGraphInstantiate(e, g, NULL, NULL, 0));
        CK(cudaGraphDestroy(g));
    }
    CK(cudaGraphLaunch(*e, st));
}

int main(int argc, char **argv) {
    const double tstart = now();
    if (argc < 4) { fprintf(stderr, "usage: %s model.gguf N id id ...\n", argv[0]); return 1; }
    if (getenv("E_CTX")) CTX = atoi(getenv("E_CTX"));
    if (getenv("E_GRAPH")) use_graph = atoi(getenv("E_GRAPH"));
    CK(cudaFree(0));
    gguf_open(argv[1]);
    load();
    int N = atoi(argv[2]); const int np = argc - 3;
    EOS_ID = (int)kv_int("tokenizer.ggml.eos_token_id", 248046);
    ROPE_TS = powf(ROPE_BASE, -2.0f / NROT);
    FSC = 1.0f / sqrtf((float)BS);
    fprintf(stderr, "fwd35: rope theta_scale %a\n", ROPE_TS);
    fprintf(stderr, "fwd35: L=%d E=%d FF=%d H=%d/%d NV=%d NK=%d BS=%d grouped=%d tied=%d vocab=%d ctx=%d\n",
            NL, E, FF, NH, NKV, NV, NK, BS, GROUPED, TIED, VOCAB, CTX);
    if (np + N > CTX) { fprintf(stderr, "prompt + generation exceeds ctx %d\n", CTX); return 1; }
    if ((size_t)CTX * 4 > 48 * 1024 - 2048) { fprintf(stderr, "ctx too large for the attention kernel\n"); return 1; }
    CK(cudaFuncSetAttribute(k_fa, cudaFuncAttributeMaxDynamicSharedMemorySize, CTX * 4));
    int big = FF * 2; if (CH + INNER > big) big = CH + INNER; if (2 * NH * HD + 2 * NKV * HD > big) big = 2 * NH * HD + 2 * NKV * HD;
    int kmax = E; if (FF > kmax) kmax = FF; if (INNER > kmax) kmax = INNER; if (NH * HD > kmax) kmax = NH * HD;
    BIG = big; PLS = kmax / 32 * 10;
    CK(cudaMalloc(&d_x, 4 * (size_t)NTM * E)); CK(cudaMalloc(&d_xn, 4 * (size_t)NTM * E));
    CK(cudaMalloc(&d_buf, 4 * (size_t)NTM * big));
    CK(cudaMalloc(&d_cv, 4 * (size_t)NTM * CH)); CK(cudaMalloc(&d_ab, 4 * (size_t)NTM * 2 * NV));
    CK(cudaMalloc(&d_o, 4 * (size_t)NTM * INNER)); CK(cudaMalloc(&d_g, 4 * (size_t)NTM * INNER));
    CK(cudaMalloc(&d_q, 4 * (size_t)NTM * NH * HD)); CK(cudaMalloc(&d_att, 4 * (size_t)NTM * NH * HD));
    CK(cudaMalloc(&d_fao, 4 * (size_t)NTM * NH * HD)); CK(cudaMalloc(&d_logits, 4 * (size_t)NTM * VOCAB));
    CK(cudaMalloc(&d_pv, 4 * 64 * NTM)); CK(cudaMalloc(&d_pi, 4 * 64 * NTM)); CK(cudaMalloc(&d_out, 4 * NTM));
    CK(cudaMalloc(&d_pos, 4));
    CK(cudaMalloc(&d_pl, 4 * (size_t)NTM * PLS));
    d_conv = (float **)calloc(NL, sizeof(float *)); d_ssm = (float **)calloc(NL, sizeof(float *));
    d_kc = (__half **)calloc(NL, sizeof(__half *)); d_vc = (__half **)calloc(NL, sizeof(__half *));
    size_t stbytes = 0;
    for (int l = 0; l < NL; l++) {
        if (Ls[l].full) {
            const size_t b = (size_t)CTX * NKV * HD * 2;
            CK(cudaMalloc(&d_kc[l / 4], b)); CK(cudaMalloc(&d_vc[l / 4], b));
            CK(cudaMemset(d_kc[l / 4], 0, b)); CK(cudaMemset(d_vc[l / 4], 0, b)); stbytes += 2 * b;
        } else {
            CK(cudaMalloc(&d_conv[l], 4 * (size_t)3 * CH)); CK(cudaMemset(d_conv[l], 0, 4 * (size_t)3 * CH));
            CK(cudaMalloc(&d_ssm[l], 4 * (size_t)NV * DS * DS)); CK(cudaMemset(d_ssm[l], 0, 4 * (size_t)NV * DS * DS));
            stbytes += 4 * ((size_t)3 * CH + (size_t)NV * DS * DS);
        }
    }
    fprintf(stderr, "fwd35: state %.1f MB (K/V f16 at ctx %d, GDN, conv)\n", stbytes / 1048576.0, CTX);
    CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    float *hx; CK(cudaMallocHost(&hx, sizeof(float) * E * NTM));
    int *hpos; CK(cudaMallocHost(&hpos, sizeof(int) * (np + N + 1)));
    int *hid; CK(cudaMallocHost(&hid, sizeof(int) * NTM));
    float *hl = NULL;
    const char *dump = getenv("E_DUMP");
    FILE *df = dump ? fopen(dump, "wb") : NULL;
    if (df) hl = (float *)malloc(4 * (size_t)VOCAB);
    if (getenv("E_TDUMP")) { tdf = fopen(getenv("E_TDUMP"), "wb"); tstep = atoi(getenv("E_TSTEP")); }
    /* batched prefill: the ternary GEMV is ALU-bound per (row, token) on sm_53, where nt = 2..8
     * runs 1.36-1.81x slower per token than nt = 1 (Nano, 4B); on sm_61 nt = 8 is 3.3x faster */
    int ccmaj = 0, ccmin = 0, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&ccmaj, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&ccmin, cudaDevAttrComputeCapabilityMinor, dev);
    int pch = getenv("E_PCHUNK") ? atoi(getenv("E_PCHUNK")) : (ccmaj * 10 + ccmin >= 61 ? NTM : 1);
    if (pch < 1) pch = 1;
    if (pch > NTM) pch = NTM;
    if (tdf) pch = 1;
    const int tiny = np >= 2;
    if (getenv("E_GRW")) gdn_rw = atoi(getenv("E_GRW"));
    if (getenv("E_GNW")) gdn_nw = atoi(getenv("E_GNW"));
    /* capture outside the timed region */
    if (use_graph && !tdf) {
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal)); /* warm the occupancy queries */
        cudaGraph_t g; enqueue(2, tiny, 2); enqueue(1, tiny, 1); CK(cudaStreamEndCapture(st, &g)); CK(cudaGraphDestroy(g));
    }
    int pos = 0;
    tcur = 0;
    /* demo switches, stream unchanged: E_STEPMS=1 prints each token's step time, E_STOP=1 ends
       after the end-of-turn id (it is printed; later steps are not run) */
    const int stepms = getenv("E_STEPMS") ? atoi(getenv("E_STEPMS")) : 0;
    const int stop = getenv("E_STOP") ? atoi(getenv("E_STOP")) : 0;
    const double t0 = now();
    if (stepms) fprintf(stderr, "fwd35: load %.3f s\n", t0 - tstart);
    for (int i = 0; i < np;) {
        const int nt = np - i < pch ? np - i : pch;
        for (int t = 0; t < nt; t++) embed(atoi(argv[3 + i + t]), hx + (size_t)E * t);
        hpos[pos] = pos;
        CK(cudaMemcpyAsync(d_x, hx, 4 * (size_t)E * nt, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_pos, hpos + pos, 4, cudaMemcpyHostToDevice, st));
        i += nt; pos += nt;
        run(nt, tiny, i == np ? 1 : 0);
        if (i < np) CK(cudaStreamSynchronize(st)); /* hx is reused */
    }
    CK(cudaMemcpyAsync(hid, d_out, 4, cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    const double t1 = now();
    double t2 = t1;
    for (int s = 0; s < N; s++) {
        const int a = *hid;
        if (df) { CK(cudaMemcpy(hl, d_logits, 4 * (size_t)VOCAB, cudaMemcpyDeviceToHost)); fwrite(hl, 4, VOCAB, df); }
        printf("%d\n", a); fflush(stdout);
        tcur = s + 1;
        if (stop && a == EOS_ID) { N = s + 1; break; }
        if (s + 1 < N) {
            const double ts = now();
            embed(a, hx);
            hpos[pos] = pos;
            CK(cudaMemcpyAsync(d_x, hx, 4 * E, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(d_pos, hpos + pos, 4, cudaMemcpyHostToDevice, st));
            pos++;
            run(1, 0, 1);
            CK(cudaMemcpyAsync(hid, d_out, 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            if (stepms) fprintf(stderr, "STEP %d ms=%.2f\n", s + 1, 1000 * (now() - ts));
        }
        t2 = now();
    }
    CK(cudaGetLastError());
    fprintf(stderr, "fwd35: prefill %d tok %.3f s = %.2f tok/s | decode %d tok %.3f s = %.2f tok/s\n",
            np, t1 - t0, np / (t1 - t0), N - 1, t2 - t1, N > 1 ? (N - 1) / (t2 - t1) : 0.0);
    if (df) fclose(df);
    if (tdf) fclose(tdf);
    return 0;
}
