/* lfm3_codec.cu -- S5: LiquidCodec DECODER on the Jetson Nano (Tegra X1).

   Chain (modeling_liquid_codec.py, HF-as-contract):
     codes [12,T] -> FSQ dequant (levels [8,7,6,6], basis [1,8,56,336])
       -> quantizer.project_out [256,48] + bias
       -> from_quantizer [512,256] + bias
       -> nearest x6 upsample (each frame repeated 6 times)
       -> 8-layer LFM3: conv layers 0,1,3,5,7 (packed B*x -> causal depthwise k=7 -> out_proj),
          sliding attention layers 2,4,6 (window 30, causal, q/k RMSNorm per head, full RoPE
          theta 1e6, per-head sigmoid gate, o_proj); sandwich norms on every layer
       -> embedding_norm -> decoder.lin [1282,512] -> log_magnitude | angle
       -> polar(exp(log_mag), angle) -> iSTFT (irfft 1280 * hann, fold hop 320, envelope)
       -> trim 480, divide by envelope  -> waveform, 320 samples per STFT frame

   STAGE: correctness reference. Every GEMV is a scalar fp32 loop on host arrays (the same
   style that gated the depth engine); the CUDA GEMV/GEMM kernels come after the gate is
   green. The iSTFT keeps the documented accumulation order (k ascending, DC and Nyquist
   terms last); parity vs torch is an SNR gate, never a bit claim.

   usage: lfm3_codec <codec.lfm3q8> <codes.i32> <T> <out.f32>
     codes.i32: 12*T int32, codebook-major ([c*T + t])
     out.f32  : 6*T*320 float32 waveform at 24 kHz
*/
#include <cstdio>
#include <unistd.h>
#include <sys/stat.h>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <time.h>
// LFM_CODEC_TIME=1: wall time per section, accumulated over the 8 layers and printed once.
// The codec is a CPU reference, so a host clock is the profiler that applies here.
static double ck_now() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec * 1e-6; }
static double ck_t[8];
static const char* ck_n[8] = {"op_norm", "qkv_gate", "qknorm_rope", "attention", "o_proj", "conv_op", "ffn", "post_ffn"};
static const bool ck_on = getenv("LFM_CODEC_TIME") != nullptr;
#define CK(i) do { if (ck_on) { double _n = ck_now(); ck_t[i] += _n - _ck; _ck = _n; } } while (0)

#include <string>
#include <map>

#define Q8_BLK 32
#define NFFT 1280
#define NHOP 320
#define NBIN 641
#define NQ 12
#define FSTACK 6

struct Ent {
  std::string name;
  int dtype, ne[4], nd;
  size_t file_off, nc, ns;
  int8_t* dq; float* ds; float* df;
  int8_t* gq = nullptr; float* gs = nullptr;   // device copies (GPU GEMM path)
};

static bool load_blob(const char* path, std::vector<Ent>& ents, std::map<std::string, Ent*>& by) {
  FILE* f = fopen(path, "rb");
  if (!f) { printf("cannot open %s\n", path); return false; }
  char magic[8];
  if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "LFM3Q8\x01\x00", 8) != 0) {
    printf("bad magic\n"); fclose(f); return false;
  }
  uint32_t n = 0;
  if (fread(&n, 4, 1, f) != 1) { fclose(f); return false; }
  char* js = (char*)malloc(n + 1);
  if (fread(js, 1, n, f) != n) { fclose(f); return false; }
  js[n] = 0;
  const size_t base = 8 + 4 + n;
  ents.reserve(256);
  size_t p = 0;
  while (true) {
    char* found = strstr(js + p, "{\"name\":");
    if (!found) break;
    p = found - js;
    char* e1p = strstr(js + p, "},");
    if (!e1p) e1p = strstr(js + p, "}");
    size_t e1 = e1p - js;
    std::string rec(js + p, e1 - p + 1);
    auto get_int = [&](const char* k, long def) -> long {
      size_t z = rec.find(std::string("\"") + k + "\":");
      return z == std::string::npos ? def : atol(rec.c_str() + z + strlen(k) + 3);
    };
    Ent e;
    size_t q1 = rec.find('"', rec.find("\"name\":") + 7);
    size_t q2 = rec.find('"', q1 + 1);
    e.name = rec.substr(q1 + 1, q2 - q1 - 1);
    e.dtype = (int)get_int("dtype", -1);
    e.file_off = (size_t)get_int("off", -1);
    e.nc = (size_t)get_int("ncodes", -1);
    e.ns = (size_t)get_int("nscales", 0);
    e.nd = 0;
    memset(e.ne, 0, sizeof e.ne);
    e.dq = nullptr; e.ds = nullptr; e.df = nullptr;
    size_t nz = rec.find("\"ne\":[");
    size_t z2 = rec.find('[', nz) + 1;
    const char* cp = rec.c_str() + z2;
    while (e.nd < 4) {
      char* endp = nullptr;
      long v = strtol(cp, &endp, 10);
      if (endp == cp) break;
      e.ne[e.nd++] = (int)v;
      cp = endp;
      if (*cp == ',') cp++; else break;
    }
    fseek(f, base + e.file_off, SEEK_SET);
    if (e.dtype == 1) {
      e.dq = (int8_t*)malloc(e.nc);
      e.ds = (float*)malloc(e.ns * 4);
      if (fread(e.dq, 1, e.nc, f) != e.nc) { printf("short read %s\n", e.name.c_str()); return false; }
      if (fread(e.ds, 4, e.ns, f) != e.ns) { printf("short read %s scales\n", e.name.c_str()); return false; }
    } else {
      e.df = (float*)malloc(e.nc);
      if (fread(e.df, 1, e.nc, f) != e.nc) { printf("short read %s\n", e.name.c_str()); return false; }
    }
    ents.push_back(e);
    p = e1 + 1;
  }
  // rebuild the index after all push_backs: vector growth moves the elements
  for (size_t i = 0; i < ents.size(); i++) by[ents[i].name] = &ents[i];
  fclose(f);
  return true;
}

static void rmsnorm(const float* x, const float* w, float* y, int D, float eps) {
  float acc = 0;
  for (int i = 0; i < D; i++) acc += x[i] * x[i];
  float r = 1.0f / sqrtf(acc / D + eps);
  for (int i = 0; i < D; i++) y[i] = x[i] * r * w[i];
}

static void q8_gemv(const int8_t* q, const float* s, const float* x, float* y, int N, int K) {
  for (int n = 0; n < N; n++) {
    float acc = 0;
    const int8_t* row = q + (size_t)n * K;
    const float* sc = s + (size_t)n * (K / Q8_BLK);
    for (int k = 0; k < K; k++) acc += ((float)row[k] * sc[k / Q8_BLK]) * x[k];
    y[n] = acc;
  }
}

// Batched over passes. The codec runs T6 passes through the same weights, and the per-pass
// GEMV streamed every weight row T6 times, each output one dependent fp32 chain (measured
// ~37 cycles/MAC). Here 12 passes go at once: each weight row is streamed once per 12
// passes, through the same NEON kernel the depth engine was gated with. Per output the
// arithmetic is unchanged (fmul by the block scale, then fmadd, k ascending), so the
// samples must come out byte-identical -- checked against the pre-change binary.
extern "C" void lfm3_q8_gemm12_kmajor(const int8_t* q, const float* s, const float* XT,
                                      float* Y, int ldy, int N, int K);
#include "lfm3_q8g.cuh"
// GPU arm (LFM_CODEC_GPU, default 1): one launch per GEMM, grid y = 12-position tiles
// (tile_mode 1), RR=2 rows per thread when N >= 1024. Same contract as the NEON kernel, so
// the same bytes. The tail tile is zero-padded; padding rows have their own accumulators
// and are dropped, exactly as the NEON path drops its padded rows.
static std::vector<Ent>* g_cents = nullptr;
static int g_cgpu = 0;
static bool gpu_gemm_tiles(const int8_t* q, const float* s, const float* X, int ldx,
                           float* Y, int ldy, int N, int K, int T) {
  const Ent* e = nullptr;
  for (auto& x : *g_cents) if (x.dtype == 1 && x.dq == q && x.gq) { e = &x; break; }
  if (!e || s != e->ds || K % 32) return false;
  const int tiles = (T + 11) / 12, TT = tiles * 12;
  static float *dX = nullptr, *dY = nullptr; static size_t capX = 0, capY = 0;
  static std::vector<float> hx, hy;
  if ((size_t)TT * K > capX) { if (dX) cudaFree(dX); capX = (size_t)TT * K; if (cudaMalloc(&dX, capX * 4)) { printf("codec gpu: cudaMalloc\n"); exit(3); } }
  if ((size_t)TT * N > capY) { if (dY) cudaFree(dY); capY = (size_t)TT * N; if (cudaMalloc(&dY, capY * 4)) { printf("codec gpu: cudaMalloc\n"); exit(3); } }
  hx.assign((size_t)TT * K, 0.f); hy.resize((size_t)TT * N);
  for (int t = 0; t < T; t++) memcpy(&hx[(size_t)t * K], X + (size_t)t * ldx, (size_t)K * 4);
  cudaMemcpy(dX, hx.data(), (size_t)TT * K * 4, cudaMemcpyHostToDevice);
  if (N >= 1024) k_q8g<12, 2><<<dim3((unsigned)((N + 255) / 256), (unsigned)tiles), 128>>>(e->gq, e->gs, dX, dY, N, K, 1);
  else           k_q8g<12, 1><<<dim3((unsigned)((N + 127) / 128), (unsigned)tiles), 128>>>(e->gq, e->gs, dX, dY, N, K, 1);
  cudaMemcpy(hy.data(), dY, (size_t)TT * N * 4, cudaMemcpyDeviceToHost);
  cudaError_t ce = cudaGetLastError();
  if (ce) { printf("codec gpu: %s\n", cudaGetErrorString(ce)); exit(3); }
  for (int t = 0; t < T; t++) memcpy(Y + (size_t)t * ldy, &hy[(size_t)t * N], (size_t)N * 4);
  return true;
}
static void q8_gemm_tiles(const int8_t* q, const float* s, const float* X, int ldx,
                          float* Y, int ldy, int N, int K, int T) {
  if (g_cgpu && gpu_gemm_tiles(q, s, X, ldx, Y, ldy, N, K, T)) return;
  static std::vector<float> xt;
  xt.resize((size_t)K * 12);
  std::vector<float> tail;
  for (int base = 0; base < T; base += 12) {
    const int cnt = (T - base < 12) ? (T - base) : 12;
    for (int i = 0; i < 12; i++) {
      const float* srow = X + (size_t)(base + (i < cnt ? i : cnt - 1)) * ldx;
      for (int k = 0; k < K; k++) xt[(size_t)k * 12 + i] = srow[k];
    }
    if (cnt == 12) {
      lfm3_q8_gemm12_kmajor(q, s, xt.data(), Y + (size_t)base * ldy, ldy, N, K);
    } else {                       // pad the tail tile, keep only the real rows
      tail.assign((size_t)12 * ldy, 0.f);
      lfm3_q8_gemm12_kmajor(q, s, xt.data(), tail.data(), ldy, N, K);
      for (int i = 0; i < cnt; i++)
        for (int n = 0; n < N; n++) Y[(size_t)(base + i) * ldy + n] = tail[(size_t)i * ldy + n];
    }
  }
}

static void f32_gemv(const float* w, const float* x, float* y, int N, int K) {
  for (int n = 0; n < N; n++) {
    float acc = 0;
    const float* row = w + (size_t)n * K;
    for (int k = 0; k < K; k++) acc += row[k] * x[k];
    y[n] = acc;
  }
}

// ---- GPU iSTFT (LFM_CODEC_GISTFT, default 1; 0 = the host DFT below). The kernels are
// lfm3_istft.cu's, copied verbatim: the gated S5 form (direct inverse, k ascending, the
// __sincosf arm that shipped at 103-130 dB SNR vs torch over 9 cases, fixed 4-term fold
// order). The waveform contract is an SNR gate, never bits (the host DFT accumulates in
// double, this one in float).
__global__ void k_irfft_window(const float* logmag, const float* angle, const float* win,
                               float* frames, int F, int use_table, const float* table) {
    int f = blockIdx.x;                                    /* one block per frame */
    __shared__ float sa[NBIN], sb[NBIN];
    for (int k = threadIdx.x; k < NBIN; k += blockDim.x) {
        float m = expf(logmag[(size_t)f * NBIN + k]);
        float c, s;
        __sincosf(angle[(size_t)f * NBIN + k], &s, &c);
        sa[k] = m * c;
        sb[k] = m * s;
    }
    __syncthreads();
    float a0 = sa[0], ah = sa[NFFT / 2];
    for (int n = blockIdx.x * 0 + threadIdx.x; n < NFFT; n += blockDim.x) {
        float acc = 0.0f;
        for (int k = 1; k < NFFT / 2; k++) {
            float c, s;
            if (use_table) {
                c = table[(size_t)n * (NFFT / 2 - 1) + (k - 1)];
                s = table[(size_t)(NFFT + n) * (NFFT / 2 - 1) + (k - 1)];
            } else {
                /* 32-bit: n,k < 1280 so n*k < 1.64e6 fits; a 64-bit % here is a software
                   divide per tap and made the kernel 100x slower than its FLOP count. */
                __sincosf(2.0f * 3.14159265358979f * (float)(((unsigned)n * (unsigned)k) % NFFT) / NFFT, &s, &c);
            }
            acc += sa[k] * c - sb[k] * s;
        }
        float x = (a0 + ((n & 1) ? -ah : ah) + 2.0f * acc) / NFFT;
        frames[(size_t)f * NFFT + n] = x * win[n];
    }
}

// k_irfft_window2: same frames, bit for bit. The per-tap twiddle depends only on
// m = (n*k) % NFFT, so each block builds the 1280-entry (c, s) table with exactly the
// __sincosf expression above and the tap loop reads it (m advanced by n mod NFFT instead of a
// modulo per tap). acc's expression and k order are unchanged. LFM_CODEC_IRF2=0 = old kernel.
__global__ void k_irfft_window2(const float* logmag, const float* angle, const float* win,
                                float* frames, int F) {
    int f = blockIdx.x;
    __shared__ float sa[NBIN], sb[NBIN], tc[NFFT], ts[NFFT];
    for (int k = threadIdx.x; k < NBIN; k += blockDim.x) {
        float m = expf(logmag[(size_t)f * NBIN + k]);
        float c, s;
        __sincosf(angle[(size_t)f * NBIN + k], &s, &c);
        sa[k] = m * c;
        sb[k] = m * s;
    }
    for (int m = threadIdx.x; m < NFFT; m += blockDim.x) {
        float c, s;
        __sincosf(2.0f * 3.14159265358979f * (float)(unsigned)m / NFFT, &s, &c);
        tc[m] = c; ts[m] = s;
    }
    __syncthreads();
    float a0 = sa[0], ah = sa[NFFT / 2];
    for (int n = threadIdx.x; n < NFFT; n += blockDim.x) {
        float acc = 0.0f;
        unsigned m = n;                                    /* (n * 1) % NFFT */
        for (int k = 1; k < NFFT / 2; k++) {
            acc += sa[k] * tc[m] - sb[k] * ts[m];
            m += n; if (m >= NFFT) m -= NFFT;              /* n < NFFT: one subtraction */
        }
        float x = (a0 + ((n & 1) ? -ah : ah) + 2.0f * acc) / NFFT;
        frames[(size_t)f * NFFT + n] = x * win[n];
    }
}

/* fold with a fixed contribution order: output t sums frames f ascending,
   f in [ceil((t-1279)/320), floor(t/320)] -- 4 terms, no atomics, so the
   result is reproducible run to run (atomics would make the sum order vary). */
__global__ void k_fold(const float* frames, const float* win, float* out, int F, int divide) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int L = (F - 1) * NHOP + NFFT;
    if (t >= L) return;
    float a = 0.0f, e = 0.0f;
    int f0 = (t - (NFFT - 1) + NHOP - 1) / NHOP; if (f0 < 0) f0 = 0;
    int f1 = t / NHOP; if (f1 > F - 1) f1 = F - 1;
    for (int f = f0; f <= f1; f++) {
        int j = t - f * NHOP;
        a += frames[(size_t)f * NFFT + j];
        e += win[j] * win[j];
    }
    /* divide=1 -> store the ENVELOPE sum; divide=0 -> store the audio sum.
       The first version wrote `a / e` here, so the envelope buffer held the
       normalised waveform and main's division returned the envelope itself
       (out == env, and NaN where the envelope was 0). Caught by the SNR gate:
       the symptom was out == the reference's envelope column. */
    out[t] = divide ? e : a;
}

// LFM_CODEC_OMP (default 1): the per-position host loops (q/k norm + RoPE, attention, gate,
// the conv branch, silu) split positions over the cores. Each position's chain is unchanged,
// so the output is the same bytes; 0 = serial.
static const bool g_comp = !(getenv("LFM_CODEC_OMP") && !atoi(getenv("LFM_CODEC_OMP")));
static float silu(float v) { return v / (1.0f + expf(-v)); }

// HF rotate_half: element i pairs with i + HD/2 (the lesson from the depth engine)
static void rope_head(float* v, int pos, int HD, float theta) {
  const int half = HD / 2;
  for (int i = 0; i < half; i++) {
    float freq = 1.0f / powf(theta, (float)(2 * i) / (float)HD);
    float c = cosf(pos * freq), s = sinf(pos * freq);
    float a = v[i], b = v[i + half];
    v[i] = a * c - b * s;
    v[i + half] = b * c + a * s;
  }
}

int main(int argc, char** argv) {
  if (argc < 5) { printf("usage: %s <codec.lfm3q8> <codes.i32> <T|wait> <out.f32>\n", argv[0]); return 2; }
  // T = "wait": load and upload first, then wait for <codes.i32> to appear (the caller
  // publishes it by rename, so it is complete when seen) and take T from its size. Lets the
  // speech shell start the codec at the beginning of the turn instead of after the last frame.
  const bool waitmode = !strcmp(argv[3], "wait");
  int T = waitmode ? 0 : atoi(argv[3]);
  const int HID = 512, INTER = 2304, NH = 8, NKV = 2, HD = 64, NL = 8;
  const float EPS = 1e-5f, THETA = 1e6f;
  const int LEVELS[4] = {8, 7, 6, 6}, BASIS[4] = {1, 8, 56, 336};

  std::vector<Ent> ents;
  std::map<std::string, Ent*> by;
  if (!load_blob(argv[1], ents, by)) return 1;
  g_cents = &ents;
  g_cgpu = getenv("LFM_CODEC_GPU") ? atoi(getenv("LFM_CODEC_GPU")) : 1;
  if (g_cgpu) {
    for (auto& e : ents) if (e.dtype == 1 && e.dq && e.ds) {
      if (cudaMalloc(&e.gq, e.nc) || cudaMalloc(&e.gs, e.ns * 4)) { printf("codec gpu: weight cudaMalloc failed\n"); return 3; }
      cudaMemcpy(e.gq, e.dq, e.nc, cudaMemcpyHostToDevice);
      cudaMemcpy(e.gs, e.ds, e.ns * 4, cudaMemcpyHostToDevice);
    }
    if (cudaGetLastError()) { printf("codec gpu: upload failed\n"); return 3; }
  }
  auto get = [&](const char* n) -> Ent* {
    auto it = by.find(n);
    if (it == by.end()) { printf("MISSING TENSOR: %s\n", n); exit(2); }
    return it->second;
  };

  if (waitmode) {
    struct stat st; time_t t0 = time(nullptr);
    while (stat(argv[2], &st) != 0) {
      if (getppid() == 1) { printf("codec wait: parent exited\n"); return 3; }
      if (time(nullptr) - t0 > 1800) { printf("codec wait: %s did not appear\n", argv[2]); return 3; }
      usleep(2000);
    }
    if (st.st_size <= 0 || st.st_size % (4 * NQ)) { printf("codec wait: %s is %ld bytes\n", argv[2], (long)st.st_size); return 2; }
    T = (int)(st.st_size / (4 * NQ));
    printf("codec wait: %s ready, T=%d\n", argv[2], T); fflush(stdout);
  }
  const int T6 = T * FSTACK;
  std::vector<int32_t> codes((size_t)NQ * T);
  FILE* fc = fopen(argv[2], "rb");
  if (!fc) { printf("cannot open %s\n", argv[2]); return 2; }
  if (fread(codes.data(), 4, codes.size(), fc) != codes.size()) { printf("short codes\n"); return 2; }
  fclose(fc);
  for (size_t i = 0; i < codes.size(); i++)
    if (codes[i] < 0 || codes[i] >= 2016) { printf("code out of range: %d\n", codes[i]); return 2; }

  // ---- FSQ dequant: code -> 4 levels -> latents [T,48]
  std::vector<float> lat((size_t)T * 48);
  for (int t = 0; t < T; t++)
    for (int c = 0; c < NQ; c++) {
      int idx = codes[(size_t)c * T + t];
      for (int d = 0; d < 4; d++) {
        int lv = (idx / BASIS[d]) % LEVELS[d];
        int hw = LEVELS[d] / 2;
        lat[(size_t)t * 48 + c * 4 + d] = (float)(lv - hw) / (float)hw;
      }
    }

  // ---- project_out [256,48] (f32) + bias, then from_quantizer [512,256] (Q8) + bias
  std::vector<float> p256((size_t)T * 256), emb((size_t)T * HID);
  Ent* wpo = get("quantizer.project_out.weight");
  float* bpo = get("quantizer.project_out.bias")->df;
  Ent* wfq = get("from_quantizer.weight");
  float* bfq = get("from_quantizer.bias")->df;
  for (int t = 0; t < T; t++) {
    f32_gemv(wpo->df, &lat[(size_t)t * 48], &p256[(size_t)t * 256], 256, 48);
    for (int i = 0; i < 256; i++) p256[(size_t)t * 256 + i] += bpo[i];
    q8_gemv(wfq->dq, wfq->ds, &p256[(size_t)t * 256], &emb[(size_t)t * HID], HID, 256);
    for (int i = 0; i < HID; i++) emb[(size_t)t * HID + i] += bfq[i];
  }

  if (const char* d = getenv("LFM_CODEC_DUMP")) {   // staged gate vs the HF oracle
    char p[512];
    snprintf(p, sizeof p, "%s/fsq_latents.f32", d); FILE* g = fopen(p, "wb"); if (g) { fwrite(lat.data(), 4, lat.size(), g); fclose(g); }
    snprintf(p, sizeof p, "%s/embeddings.f32", d);  FILE* h2 = fopen(p, "wb"); if (h2) { fwrite(emb.data(), 4, emb.size(), h2); fclose(h2); }
  }

  // ---- nearest x6: each frame repeated 6 times (torch nearest-exact at an integer factor)
  std::vector<float> X((size_t)T6 * HID);
  for (int t = 0; t < T6; t++)
    memcpy(&X[(size_t)t * HID], &emb[(size_t)(t / FSTACK) * HID], HID * 4);
  if (const char* d = getenv("LFM_CODEC_DUMP")) {
    char p[512]; snprintf(p, sizeof p, "%s/upsample.f32", d);
    FILE* g = fopen(p, "wb"); if (g) { fwrite(X.data(), 4, X.size(), g); fclose(g); }
  }

  // ---- 8 LFM3 layers
  std::vector<float> h((size_t)T6 * HID), tmp((size_t)T6 * HID), proj((size_t)T6 * 3 * HID);
  std::vector<float> qh(NH * HD), kh(NKV * HD), vh(NKV * HD), attn(NH * HD);
  std::vector<float> scores(T6);
  for (int L = 0; L < NL; L++) {
    char pfx[128]; snprintf(pfx, sizeof pfx, "decoder.lfm.layers.%d", L);
    std::string P(pfx);
    Ent* on = get((P + ".operator_norm.weight").c_str());
    Ent* pon = get((P + ".post_operator_norm.weight").c_str());
    Ent* fn = get((P + ".ffn_norm.weight").c_str());
    Ent* pfn = get((P + ".post_ffn_norm.weight").c_str());
    bool is_attn = (L == 2 || L == 4 || L == 6);

    double _ck = ck_now();
    for (int t = 0; t < T6; t++) rmsnorm(&X[(size_t)t * HID], on->df, &h[(size_t)t * HID], HID, EPS);
    CK(0);

    if (is_attn) {
      Ent* wq = get((P + ".self_attn.q_proj.weight").c_str());
      Ent* wk = get((P + ".self_attn.k_proj.weight").c_str());
      Ent* wv = get((P + ".self_attn.v_proj.weight").c_str());
      Ent* wo = get((P + ".self_attn.o_proj.weight").c_str());
      Ent* wg = get((P + ".self_attn.gate_proj.weight").c_str());
      float* qnw = get((P + ".self_attn.q_norm.weight").c_str())->df;
      float* knw = get((P + ".self_attn.k_norm.weight").c_str())->df;
      std::vector<float> K((size_t)T6 * NKV * HD), V((size_t)T6 * NKV * HD), G((size_t)T6 * NH);
      std::vector<float> Q((size_t)T6 * NH * HD), A((size_t)T6 * NH * HD);
      q8_gemm_tiles(wq->dq, wq->ds, h.data(), HID, Q.data(), NH * HD, NH * HD, HID, T6);
      q8_gemm_tiles(wk->dq, wk->ds, h.data(), HID, K.data(), NKV * HD, NKV * HD, HID, T6);
      q8_gemm_tiles(wv->dq, wv->ds, h.data(), HID, V.data(), NKV * HD, NKV * HD, HID, T6);
      q8_gemm_tiles(wg->dq, wg->ds, h.data(), HID, G.data(), NH, NH, HID, T6);
      CK(1);
      #pragma omp parallel for schedule(static) if (g_comp)
      for (int t = 0; t < T6; t++) {
        // rmsnorm per head into a scratch row, copied back INSIDE the same iteration:
        // a split loop left the last head's values in every head slot (found by staged dump).
        std::vector<float> tn(HD);
        for (int hh = 0; hh < NH; hh++) {
          rmsnorm(&Q[(size_t)t * NH * HD + hh * HD], qnw, tn.data(), HD, EPS);
          memcpy(&Q[(size_t)t * NH * HD + hh * HD], tn.data(), HD * 4);
        }
        for (int kv = 0; kv < NKV; kv++) {
          rmsnorm(&K[(size_t)t * NKV * HD + kv * HD], knw, tn.data(), HD, EPS);
          memcpy(&K[(size_t)t * NKV * HD + kv * HD], tn.data(), HD * 4);
        }
        for (int hh = 0; hh < NH; hh++) rope_head(&Q[(size_t)t * NH * HD + hh * HD], t, HD, THETA);
        for (int kv = 0; kv < NKV; kv++) rope_head(&K[(size_t)t * NKV * HD + kv * HD], t, HD, THETA);
      }
      if (const char* d = getenv("LFM_CODEC_DUMP")) {
        char p[512];
        snprintf(p, sizeof p, "%s/L%d_Q.f32", d, L); FILE* g = fopen(p, "wb"); if (g) { fwrite(Q.data(), 4, Q.size(), g); fclose(g); }
        snprintf(p, sizeof p, "%s/L%d_K.f32", d, L); g = fopen(p, "wb"); if (g) { fwrite(K.data(), 4, K.size(), g); fclose(g); }
        snprintf(p, sizeof p, "%s/L%d_V.f32", d, L); g = fopen(p, "wb"); if (g) { fwrite(V.data(), 4, V.size(), g); fclose(g); }
        snprintf(p, sizeof p, "%s/L%d_G.f32", d, L); g = fopen(p, "wb"); if (g) { fwrite(G.data(), 4, G.size(), g); fclose(g); }
        snprintf(p, sizeof p, "%s/L%d_h.f32", d, L); g = fopen(p, "wb"); if (g) { fwrite(h.data(), 4, h.size(), g); fclose(g); }
      }
      CK(2);
      const float scale = 1.0f / sqrtf((float)HD);
      const int WIN = 30;                              // causal + sliding window of 30
      #pragma omp parallel for schedule(static) if (g_comp)
      for (int t = 0; t < T6; t++) {
        std::vector<float> scores(T6), attn(NH * HD);   // per position (were shared)
        for (int hh = 0; hh < NH; hh++) {
          int kvh = hh / (NH / NKV);
          int lo = t - WIN + 1; if (lo < 0) lo = 0;
          float mx = -1e30f;
          for (int j = lo; j <= t; j++) {
            float acc = 0;
            const float* qq = &Q[(size_t)t * NH * HD + hh * HD];
            const float* kk = &K[(size_t)j * NKV * HD + kvh * HD];
            for (int d = 0; d < HD; d++) acc += qq[d] * kk[d];
            scores[j] = acc * scale;
            if (scores[j] > mx) mx = scores[j];
          }
          float sum = 0;
          for (int j = lo; j <= t; j++) { scores[j] = expf(scores[j] - mx); sum += scores[j]; }
          for (int d = 0; d < HD; d++) {
            float acc = 0;
            for (int j = lo; j <= t; j++) acc += scores[j] * V[(size_t)j * NKV * HD + kvh * HD + d];
            attn[hh * HD + d] = acc / sum;
          }
        }
        for (int hh = 0; hh < NH; hh++) {                             // per-head sigmoid gate
          float g = 1.0f / (1.0f + expf(-G[(size_t)t * NH + hh]));
          for (int d = 0; d < HD; d++) attn[hh * HD + d] *= g;
        }
        memcpy(&A[(size_t)t * NH * HD], attn.data(), (size_t)NH * HD * 4);
      }
      CK(3);
      q8_gemm_tiles(wo->dq, wo->ds, A.data(), NH * HD, tmp.data(), HID, HID, NH * HD, T6);
      CK(4);
      if (const char* d = getenv("LFM_CODEC_DUMP")) {
        char p[512]; snprintf(p, sizeof p, "%s/L%d_attn.f32", d, L);
        FILE* g = fopen(p, "wb"); if (g) { fwrite(tmp.data(), 4, tmp.size(), g); fclose(g); }
      }
    } else {
      CK(1); CK(2); CK(3); CK(4);
      Ent* win_ = get((P + ".conv.in_proj.weight").c_str());
      Ent* wout = get((P + ".conv.out_proj.weight").c_str());
      Ent* wc = get((P + ".conv.conv.weight").c_str());           // [512,1,7] f32
      const float* cw = wc->df;
      std::vector<float> z((size_t)T6 * HID), conv((size_t)T6 * HID);
      q8_gemm_tiles(win_->dq, win_->ds, h.data(), HID, proj.data(), 3 * HID, 3 * HID, HID, T6);
      for (int t = 0; t < T6; t++) {
        const float* B = &proj[(size_t)t * 3 * HID];
        const float* XG = &proj[(size_t)t * 3 * HID + 2 * HID];   // BCx chunk order: B, C, x
        for (int i = 0; i < HID; i++) z[(size_t)t * HID + i] = B[i] * XG[i];
      }
      #pragma omp parallel for schedule(static) if (g_comp)
      for (int t = 0; t < T6; t++)                                  // causal depthwise k=7
        for (int c = 0; c < HID; c++) {
          float acc = 0;
          for (int j = 0; j < 7; j++) {
            int s = t - 6 + j;
            if (s >= 0) acc += cw[c * 7 + j] * z[(size_t)s * HID + c];
          }
          conv[(size_t)t * HID + c] = acc;
        }
      for (int t = 0; t < T6; t++) {
        const float* Cg = &proj[(size_t)t * 3 * HID + HID];       // y = C * conv(z): the middle chunk gates
        for (int i = 0; i < HID; i++) conv[(size_t)t * HID + i] *= Cg[i];
      }
      q8_gemm_tiles(wout->dq, wout->ds, conv.data(), HID, tmp.data(), HID, HID, HID, T6);
      CK(5);
    }

    CK(5); CK(6); CK(7);
    for (int t = 0; t < T6; t++) {                                  // sandwich: post_operator_norm
      rmsnorm(&tmp[(size_t)t * HID], pon->df, h.data(), HID, EPS);
      for (int i = 0; i < HID; i++) X[(size_t)t * HID + i] += h[i];
    }

    Ent* w1 = get((P + ".feed_forward.w1.weight").c_str());
    Ent* w2 = get((P + ".feed_forward.w2.weight").c_str());
    Ent* w3 = get((P + ".feed_forward.w3.weight").c_str());
    std::vector<float> U((size_t)T6 * INTER), GA((size_t)T6 * INTER);
    CK(6);
    for (int t = 0; t < T6; t++) rmsnorm(&X[(size_t)t * HID], fn->df, &h[(size_t)t * HID], HID, EPS);
    q8_gemm_tiles(w1->dq, w1->ds, h.data(), HID, U.data(), INTER, INTER, HID, T6);
    q8_gemm_tiles(w3->dq, w3->ds, h.data(), HID, GA.data(), INTER, INTER, HID, T6);
    #pragma omp parallel for schedule(static) if (g_comp)
    for (long i = 0; i < (long)T6 * INTER; i++) U[i] = silu(U[i]) * GA[i];
    q8_gemm_tiles(w2->dq, w2->ds, U.data(), INTER, tmp.data(), HID, HID, INTER, T6);
    CK(7);
    for (int t = 0; t < T6; t++) {
      rmsnorm(&tmp[(size_t)t * HID], pfn->df, &h[(size_t)t * HID], HID, EPS);
      for (int i = 0; i < HID; i++) X[(size_t)t * HID + i] += h[(size_t)t * HID + i];
    }
    if (const char* d = getenv("LFM_CODEC_DUMP")) {
      char p[512]; snprintf(p, sizeof p, "%s/layer%d.f32", d, L);
      FILE* g = fopen(p, "wb"); if (g) { fwrite(X.data(), 4, X.size(), g); fclose(g); }
    }
    if (getenv("LFM_CODEC_TRACE")) {
      double acc = 0; for (size_t i = 0; i < X.size(); i++) acc += (double)X[i] * X[i];
      printf("layer%d rms=%.4f first4=[%.4f,%.4f,%.4f,%.4f]\n", L, sqrt(acc / X.size()),
             X[0], X[1], X[2], X[3]);
      fflush(stdout);
    }
  }

  // ---- embedding_norm -> lin -> polar -> iSTFT
  float* enorm = get("decoder.lfm.embedding_norm.weight")->df;
  std::vector<float> lin((size_t)T6 * 2 * NBIN);
  Ent* wlin = get("decoder.lin.weight");
  float* blin = get("decoder.lin.bias")->df;
  std::vector<float> HN((size_t)T6 * HID);
  for (int t = 0; t < T6; t++) rmsnorm(&X[(size_t)t * HID], enorm, &HN[(size_t)t * HID], HID, EPS);
  q8_gemm_tiles(wlin->dq, wlin->ds, HN.data(), HID, lin.data(), 2 * NBIN, 2 * NBIN, HID, T6);
  for (int t = 0; t < T6; t++)
    for (int i = 0; i < 2 * NBIN; i++) lin[(size_t)t * 2 * NBIN + i] += blin[i];
  if (ck_on) {
    double tot = 0; for (int i = 0; i < 8; i++) tot += ck_t[i];
    printf("CODEC_TIME ms (T=%d, %d passes):", T, T6);
    for (int i = 0; i < 8; i++) printf(" %s=%.0f", ck_n[i], ck_t[i]);
    printf("  sum=%.0f\n", tot); fflush(stdout);
  }

  if (const char* d = getenv("LFM_CODEC_DUMP")) {
    char p[512]; snprintf(p, sizeof p, "%s/lin.f32", d);
    FILE* g = fopen(p, "wb"); if (g) { fwrite(lin.data(), 4, lin.size(), g); fclose(g); }
  }
  const size_t L = (size_t)(T6 - 1) * NHOP + NFFT;
  const int trim = (NFFT - NHOP) / 2;
  static const int gistft = getenv("LFM_CODEC_GISTFT") ? atoi(getenv("LFM_CODEC_GISTFT")) : 1;
  if (gistft) {
    std::vector<float> in((size_t)2 * NBIN * T6), wv(NFFT);
    for (int f = 0; f < T6; f++) {
      memcpy(&in[(size_t)f * NBIN], &lin[(size_t)f * 2 * NBIN], NBIN * 4);                       // log-magnitudes
      memcpy(&in[(size_t)(T6 + f) * NBIN], &lin[(size_t)f * 2 * NBIN + NBIN], NBIN * 4);          // angles
    }
    for (int n = 0; n < NFFT; n++) wv[n] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * n / (NFFT - 1));
    float *d_win, *d_in, *d_frames, *d_audio, *d_env;
    if (cudaMalloc(&d_win, NFFT * 4) || cudaMalloc(&d_in, in.size() * 4) || cudaMalloc(&d_frames, (size_t)T6 * NFFT * 4) ||
        cudaMalloc(&d_audio, L * 4) || cudaMalloc(&d_env, L * 4)) { printf("codec gpu istft: cudaMalloc failed\n"); return 3; }
    cudaMemcpy(d_win, wv.data(), NFFT * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in, in.data(), in.size() * 4, cudaMemcpyHostToDevice);
    const int thr = 256;
    static const int irf2 = getenv("LFM_CODEC_IRF2") ? atoi(getenv("LFM_CODEC_IRF2")) : 1;
    if (irf2) k_irfft_window2<<<T6, thr>>>(d_in, d_in + (size_t)NBIN * T6, d_win, d_frames, T6);
    else k_irfft_window<<<T6, thr>>>(d_in, d_in + (size_t)NBIN * T6, d_win, d_frames, T6, 0, nullptr);
    k_fold<<<(int)((L + thr - 1) / thr), thr>>>(d_frames, d_win, d_audio, T6, 0);
    k_fold<<<(int)((L + thr - 1) / thr), thr>>>(d_frames, d_win, d_env, T6, 1);
    std::vector<float> au(L), ev(L);
    cudaMemcpy(au.data(), d_audio, L * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(ev.data(), d_env, L * 4, cudaMemcpyDeviceToHost);
    if (cudaGetLastError()) { printf("codec gpu istft: kernel failed\n"); return 3; }
    std::vector<float> o((size_t)T6 * NHOP);
    for (size_t t = 0; t < o.size(); t++) o[t] = au[t + trim] / ev[t + trim];
    FILE* fo = fopen(argv[4], "wb");
    if (!fo || fwrite(o.data(), 4, o.size(), fo) != o.size()) { printf("cannot write %s\n", argv[4]); return 2; }
    fclose(fo);
    printf("done: wrote %d samples (%.2f s at 24 kHz) [gpu istft]\n", T6 * NHOP, T6 * NHOP / 24000.0);
    return 0;
  }
  std::vector<float> audio(L, 0.f), envelope(L, 0.f);
  std::vector<float> win(NFFT);
  for (int n = 0; n < NFFT; n++) win[n] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * n / (NFFT - 1));
  // The irfft below is a direct DFT: O(NFFT*NBIN) per frame, 1280*639 = 818k terms, and the
  // k-only work (expf, the magnitude's cos/sin, the DC and Nyquist bins) was recomputed for
  // every n -- 295M transcendental calls per run, which had become the codec's whole cost
  // once the GEMVs were batched. Hoisting them into per-frame tables and splitting frames
  // across cores changes no arithmetic: every output element is computed by the same
  // expression in the same order (verified byte-identical against the pre-change binary).
  // Frames are computed in parallel but overlap-added SERIALLY: hop 320 < frame 1280, so
  // consecutive frames write the same samples and a parallel += there is a data race (it
  // produced a wrong first sample on the first attempt -- caught by the byte comparison).
  std::vector<float> frames((size_t)T6 * NFFT);
  #pragma omp parallel
  {
    std::vector<float> av(NBIN), bv(NBIN);      // per-thread: shared tables raced here
    #pragma omp for schedule(dynamic, 1)
    for (int f = 0; f < T6; f++) {
      const float* lm = &lin[(size_t)f * 2 * NBIN];
      const float* an = lm + NBIN;
      for (int k = 1; k < NBIN - 1; k++) {
        float mag = expf(lm[k]);
        av[k] = mag * cosf(an[k]);
        bv[k] = mag * sinf(an[k]);
      }
      const float dc = expf(lm[0]), ny = expf(lm[NBIN - 1]);
      for (int n = 0; n < NFFT; n++) {                                // irfft, k ascending, DC/Nyquist last
        double acc = 0;
        for (int k = 1; k < NBIN - 1; k++) {
          float th = 2.0f * (float)M_PI * k * n / (float)NFFT;
          acc += 2.0 * ((double)av[k] * cosf(th) - (double)bv[k] * sinf(th));
        }
        acc += (double)dc;
        acc += (double)((n & 1) ? -1.0f : 1.0f) * ny;                 // (-1)^n * a_{N/2}
        frames[(size_t)f * NFFT + n] = (float)(acc / NFFT) * win[n];
      }
    }
  }
  for (int f = 0; f < T6; f++)
    for (int n = 0; n < NFFT; n++) {
      audio[(size_t)f * NHOP + n] += frames[(size_t)f * NFFT + n];
      envelope[(size_t)f * NHOP + n] += win[n] * win[n];
    }
  FILE* fo = fopen(argv[4], "wb");
  if (!fo) { printf("cannot write %s\n", argv[4]); return 2; }
  for (int i = 0; i < T6 * NHOP; i++) {
    float v = audio[trim + i] / envelope[trim + i];
    fwrite(&v, 4, 1, fo);
  }
  fclose(fo);
  printf("done: wrote %d samples (%.2f s at 24 kHz)\n", T6 * NHOP, T6 * NHOP / 24000.0);
  return 0;
}
