/* lfm3_depth.cu -- S4 depth decoder: depth_linear + 3-layer bidirectional Lfm3
   trunk (all full_attention, no conv, no sandwich) + 12 codebook heads.

   Input: hidden vector [2048] (backbone hidden state) + codes [12] + masked [12]
   Output: logits [12, 2017] (all codebooks, one pass)

   Usage: lfm3_depth <depth.lfm3q8> <hidden.f32> <codes.i32> <masked.u8> <out.f32>
     hidden.f32: 2048 floats
     codes.i32: 12 ints
     masked.u8: 12 bytes (0/1)
     out.f32: 12*2017 floats

   For multi-pass MaskGIT, the caller runs this repeatedly with updated codes/masked.
*/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <cublas_v2.h>

// reuse the backbone's conventions
#define Q8_BLK 32
typedef uint16_t bf16;

struct Ent {
  std::string name;
  int dtype, ne[4] = {0,0,0,0}, nd = 0;
  size_t file_off, nc, ns;
  uint32_t crc;
  int8_t* dq = nullptr; float* ds = nullptr; float* df = nullptr;
};

// (loader copied from lfm3_nanode.cu pattern - simplified)
static bool load_blob(const char* path, std::vector<Ent>& ents, std::map<std::string,Ent*>& by, char*& js) {
  FILE* f = fopen(path, "rb");
  if (!f) { printf("cannot open %s\n", path); return false; }
  char magic[8];
  if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "LFM3Q8\x01\x00", 8) != 0) {
    printf("bad magic\n"); fclose(f); return false;
  }
  uint32_t n = 0;
  if (fread(&n, 4, 1, f) != 1) { fclose(f); return false; }
  js = (char*)malloc(n+1);
  if (fread(js, 1, n, f) != n) { fclose(f); return false; }
  js[n] = 0;
  const size_t base = 8 + 4 + n;
  ents.reserve(256);
  size_t p = 0;
  while (true) {
    char* found = strstr(js + p, "{\"name\":");
    if (!found) break;
    p = found - js;
    if (js[p] == 0) break;
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
    e.crc = (uint32_t)get_int("crc", 0);
    e.nd = 0;
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
      fread(e.dq, 1, e.nc, f);
      fread(e.ds, 4, e.ns, f);
    } else if (e.dtype == 2) {
      e.df = (float*)malloc(e.nc);
      fread(e.df, 1, e.nc, f);
    } else if (e.dtype == 3) {
      e.dq = (int8_t*)malloc(e.nc);
      fread(e.dq, 1, e.nc, f);
    }
    ents.push_back(e);
    p = e1 + 1;
  }
  for (auto& e : ents) by[e.name] = &e;
  fclose(f);
  return true;
}

// bf16 -> f32 conversion
static float bf16_to_f32(uint16_t h) {
  uint32_t u = ((uint32_t)h) << 16;
  float v; memcpy(&v, &u, 4); return v;
}
static uint16_t f32_to_bf16(float v) {
  uint32_t u; memcpy(&u, &v, 4); return (uint16_t)(u >> 16);
}

// RMSNorm
static void rmsnorm(const float* x, const float* w, float* y, int D, float eps) {
  float acc = 0;
  for (int i = 0; i < D; i++) acc += x[i]*x[i];
  float rms = 1.0f / sqrtf(acc / D + eps);
  for (int i = 0; i < D; i++) y[i] = x[i] * rms * w[i];
}

// Q8 GEMM over T positions: Y[t*ldy + n] = sum_k dequant(q[n,k], s[n,k/32]) * X[t*ldx + k].
// The kernels live in lfm3_depth_neon.c (NEON + OpenMP; nvcc 10.2 cannot parse arm_neon.h).
// Per output the arithmetic is the original per-position loop's (fmadd, k ascending), so the
// logits are bit-identical to the pre-change engine. Build:
//   gcc -O2 -fopenmp -c lfm3_depth_neon.c -o lfm3_depth_neon.o
//   nvcc -std=c++11 -gencode arch=compute_53,code=sm_53 -O2 -Xcompiler -fopenmp \
//        -o lfm3_depth lfm3_depth.cu lfm3_depth_neon.o -lgomp
extern "C" void lfm3_q8_gemm12_kmajor(const int8_t* q, const float* s, const float* XT,
                                      float* Y, int ldy, int N, int K);
extern "C" void lfm3_q8_gemv1(const int8_t* q, const float* s, const float* x, float* y, int N, int K);
static void q8_gemm(const int8_t* q, const float* s, const float* X, int ldx,
                    float* Y, int ldy, int N, int K, int T) {
  if (T == 1) { lfm3_q8_gemv1(q, s, X, Y, N, K); return; }
  if (T != 12) { printf("q8_gemm: T=%d unsupported (1 or 12)\n", T); exit(2); }
  static std::vector<float> xt;
  xt.resize((size_t)K * 12);
  for (int t = 0; t < 12; t++)
    for (int k = 0; k < K; k++) xt[(size_t)k * 12 + t] = X[(size_t)t * ldx + k];
  lfm3_q8_gemm12_kmajor(q, s, xt.data(), Y, ldy, N, K);
}
// Q8 GEMV: one position (the head slices and depth_linear).
static void q8_gemv(const int8_t* q, const float* s, const float* x, float* y, int N, int K) {
  q8_gemm(q, s, x, K, y, N, N, K, 1);
}

// SiLU
static float silu(float x) { return x / (1.0f + expf(-x)); }

// GELU (erf approximation)
static float gelu(float x) {
  return 0.5f * x * (1.0f + erff(x / 1.41421356f));
}

// RoPE
// HF apply_rotary_pos_emb: rotate_half pairs element i with i + HD/2 (NOT 2i with 2i+1).
// q_embed = q*cos + rotate_half(q)*sin, with cos/sin built from cat(freqs, freqs).
static void rope(float* q, float* k, int pos, int HD, float theta, float partial) {
  int angles = (int)(partial * HD) / 2;
  const int half = HD / 2;
  for (int i = 0; i < half; i++) {
    float freq = (i < angles) ? 1.0f / powf(theta, (float)(2*i) / (float)HD) : 0.0f;
    float cosv = cosf(pos * freq), sinv = sinf(pos * freq);
    if (q) {
      float a = q[i], b = q[i + half];
      q[i]        = a*cosv - b*sinv;
      q[i + half] = b*cosv + a*sinv;
    }
    if (k) {
      float a = k[i], b = k[i + half];
      k[i]        = a*cosv - b*sinv;
      k[i + half] = b*cosv + a*sinv;
    }
  }
}

// Attention (bidirectional, all positions)
static void attention(float* q, float* k, float* v, float* out, int pos, int NQ, int NKV, int HD, int T, int* kv_lens) {
  // q: [NQ, HD], k: [NKV, T, HD] (all positions), v: [NKV, T, HD]
  int ngrp = NQ / NKV;
  float scale = 1.0f / sqrtf((float)HD);
  for (int h = 0; h < NQ; h++) {
    int kvh = h / ngrp;
    // compute attention scores over all positions
    std::vector<float> scores(T);
    float maxs = -1e9f;
    for (int t = 0; t < T; t++) {
      float acc = 0;
      for (int d = 0; d < HD; d++) acc += q[h*HD+d] * k[t*NKV*HD + kvh*HD + d];
      scores[t] = acc * scale;
      if (scores[t] > maxs) maxs = scores[t];
    }
    // softmax
    float sumexp = 0;
    for (int t = 0; t < T; t++) { scores[t] = expf(scores[t] - maxs); sumexp += scores[t]; }
    for (int t = 0; t < T; t++) scores[t] /= sumexp;
    // weighted sum of values
    for (int d = 0; d < HD; d++) {
      float acc = 0;
      for (int t = 0; t < T; t++) acc += scores[t] * v[t*NKV*HD + kvh*HD + d];
      out[h*HD+d] = acc;
    }
  }
}

// LFM_DEPTH_TIME=1: wall time per forward section (host clock), printed per call.
#include <time.h>
static double dt_now() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec * 1e-6; }
static double g_dt[8];
static const char* g_dtn[8] = {"depth_linear", "embed", "norm+qkv", "qk_norm+rope+attn", "gate+o", "ffn", "final_norm", "head"};
#define DT_MARK(i) do { double _n = dt_now(); g_dt[i] += _n - _dt_last; _dt_last = _n; } while (0)
static void depth_forward(std::vector<Ent>& ents, std::map<std::string, Ent*>& by,
                      const float* hidden, const int* codes, const unsigned char* masked,
                      float* out_logits) {
  const int DH = 1024;  // depth hidden
  const int SH = 2048;  // src hidden
  const int NCB = 12;
  const int CBS = 2017;
  const int NHEADS = 32;
  const int NKV = 8;
  const int HD = 32;
  const int INTER = 2816;
  const int NLAYERS = 3;

  // find tensors
  auto get = [&](const char* name) -> Ent* {
    auto it = by.find(name);
    // A missing tensor is fatal: returning nullptr here segfaulted the caller two
    // sessions ago, and a silently skipped weight would run the trunk without it.
    if (it == by.end()) { printf("MISSING TENSOR: %s\n", name); exit(2); }
    return it->second;
  };

  // depth_linear: [12288, 2048] Q8 -> f32 [12, 1024, 2048]
  Ent* dl = get("depth_linear.weight");
  Ent* dl_bias = get("depth_linear.bias");

  // depth_linear forward: out[12, 1024] = W[12*1024, 2048] @ hidden[2048] + bias
  for (int i = 0; i < 8; i++) g_dt[i] = 0;
  double _dt_last = dt_now();
  float depth_in[12 * 1024];
  q8_gemv(dl->dq, dl->ds, hidden, depth_in, 12 * DH, SH);
  for (int i = 0; i < 12 * DH; i++) depth_in[i] += dl_bias->df[i];
  DT_MARK(0);

  // add mask/codebook embeddings
  // depth_in = depth_in + where(masked, mask_embedding, codebook_embedding(codes))
  Ent* mask_emb = get("mask_embedding");
  // codebook embeddings: depth_embeddings.{i}.embedding.weight [2017, 1024] bf16
  for (int c = 0; c < NCB; c++) {
    for (int d = 0; d < DH; d++) {
      float ce;
      if (masked[c]) {
        ce = mask_emb->df[d];
      } else {
        int code = codes[c];
        Ent* emb = nullptr;
        char name[128]; snprintf(name, sizeof name, "depth_embeddings.%d.embedding.weight", c);
        emb = get(name);
        ce = bf16_to_f32(((uint16_t*)emb->dq)[code * DH + d]);
      }
      depth_in[c * DH + d] += ce;
    }
  }

  DT_MARK(1);
  // reshape to [12, 1024] for the trunk (bidirectional attention over 12 positions)
  // trunk: 3 layers, all full_attention
  float* x = depth_in;  // [12, 1024] in-place
  float* h1 = (float*)malloc(12 * DH * 4);
  float* h2 = (float*)malloc(12 * INTER * 4);
  float* h3 = (float*)malloc(12 * INTER * 4);
  float* h4 = (float*)malloc(12 * DH * 4);
  float* q = (float*)malloc(12 * NHEADS * HD * 4);
  float* k = (float*)malloc(12 * NKV * HD * 4);
  float* v = (float*)malloc(12 * NKV * HD * 4);
  float* attn = (float*)malloc(12 * NHEADS * HD * 4);

  for (int L = 0; L < NLAYERS; L++) {
    char prefix[128]; snprintf(prefix, sizeof prefix, "depthformer.model.layers.%d", L);
    // rms norm (operator_norm, no sandwich)
    for (int t = 0; t < 12; t++) {
      char name[256]; snprintf(name, sizeof name, "%s.operator_norm.weight", prefix);
      rmsnorm(x + t*DH, get(name)->df, h1 + t*DH, DH, 1e-5f);
    }
      // q, k, v projections
    char wname[256];
    snprintf(wname, sizeof wname, "%s.self_attn.q_proj.weight", prefix);
    q8_gemm(get(wname)->dq, get(wname)->ds, h1, DH, q, NHEADS*HD, NHEADS*HD, DH, 12);
      snprintf(wname, sizeof wname, "%s.self_attn.k_proj.weight", prefix);
    q8_gemm(get(wname)->dq, get(wname)->ds, h1, DH, k, NKV*HD, NKV*HD, DH, 12);
      snprintf(wname, sizeof wname, "%s.self_attn.v_proj.weight", prefix);
    q8_gemm(get(wname)->dq, get(wname)->ds, h1, DH, v, NKV*HD, NKV*HD, DH, 12);
    DT_MARK(2);
      // q_norm, k_norm (per head)
    for (int t = 0; t < 12; t++) {
      for (int h = 0; h < NHEADS; h++) {
        char qn[256]; snprintf(qn, sizeof qn, "%s.self_attn.q_norm.weight", prefix);
        float tmp[HD];
        for (int d = 0; d < HD; d++) tmp[d] = q[t*NHEADS*HD + h*HD + d];
        rmsnorm(tmp, get(qn)->df, tmp, HD, 1e-5f);
        for (int d = 0; d < HD; d++) q[t*NHEADS*HD + h*HD + d] = tmp[d];
      }
    }
      for (int t = 0; t < 12; t++) {
      for (int h = 0; h < NHEADS; h++) {
        // placeholder
      }
      for (int h = 0; h < NKV; h++) {
        char kn[256]; snprintf(kn, sizeof kn, "%s.self_attn.k_norm.weight", prefix);
        float tmp[HD];
        for (int d = 0; d < HD; d++) tmp[d] = k[t*NKV*HD + h*HD + d];
        rmsnorm(tmp, get(kn)->df, tmp, HD, 1e-5f);
        for (int d = 0; d < HD; d++) k[t*NKV*HD + h*HD + d] = tmp[d];
      }
    }
        // RoPE (full rotation, theta 1e6) - apply once per q head and once per kv head
    for (int t = 0; t < 12; t++) {
      for (int h = 0; h < NHEADS; h++) {
        rope(q + t*NHEADS*HD + h*HD, nullptr, t, HD, 1e6f, 1.0f);
      }
      for (int h = 0; h < NKV; h++) {
        rope(nullptr, k + t*NKV*HD + h*HD, t, HD, 1e6f, 1.0f);
      }
    }
    // attention (bidirectional: all 12 positions)
    // rearrange: q[t] is at position t, k/v[t] at position t
    // attention function expects k/v as [NKV, T, HD]
    for (int t = 0; t < 12; t++) {
      attention(q + t*NHEADS*HD, k, v, attn + t*NHEADS*HD, t, NHEADS, NKV, HD, 12, nullptr);
    }
    DT_MARK(3);
    // gate (Q8). A leftover first loop here ran bf16_gemv over this Q8 tensor -- reading
    // 64 KB from its 32 KB code buffer, out of bounds -- and discarded the result; removed.
    {
      char gn[256]; snprintf(gn, sizeof gn, "%s.self_attn.gate_proj.weight", prefix);
      float gate[12 * NHEADS];
      q8_gemm(get(gn)->dq, get(gn)->ds, h1, DH, gate, NHEADS, NHEADS, DH, 12);
      for (int t = 0; t < 12; t++)
        for (int h = 0; h < NHEADS; h++) {
          float g = 1.0f / (1.0f + expf(-gate[t*NHEADS + h]));
          for (int d = 0; d < HD; d++) attn[t*NHEADS*HD + h*HD + d] *= g;
        }
    }
    // o_proj
    {
      char on[256]; snprintf(on, sizeof on, "%s.self_attn.o_proj.weight", prefix);
      q8_gemm(get(on)->dq, get(on)->ds, attn, NHEADS*HD, h4, DH, DH, NHEADS*HD, 12);
    }
    // residual add
    for (int i = 0; i < 12*DH; i++) x[i] += h4[i];
    DT_MARK(4);
    // ffn
    for (int t = 0; t < 12; t++) {
      char fn[256]; snprintf(fn, sizeof fn, "%s.operator_norm.weight", prefix);
      // operator_norm already done above, but we need ffn_norm now
    }
    // ffn_norm
    for (int t = 0; t < 12; t++) {
      char fn[256]; snprintf(fn, sizeof fn, "%s.ffn_norm.weight", prefix);
      rmsnorm(x + t*DH, get(fn)->df, h1 + t*DH, DH, 1e-5f);
    }
    // w1, w3
    {
      char w1n[256], w3n[256];
      snprintf(w1n, sizeof w1n, "%s.feed_forward.w1.weight", prefix);
      snprintf(w3n, sizeof w3n, "%s.feed_forward.w3.weight", prefix);
      q8_gemm(get(w1n)->dq, get(w1n)->ds, h1, DH, h2, INTER, INTER, DH, 12);
      q8_gemm(get(w3n)->dq, get(w3n)->ds, h1, DH, h3, INTER, INTER, DH, 12);
    }
    // silu_mul
    for (int i = 0; i < 12*INTER; i++) h2[i] = silu(h2[i]) * h3[i];
    // w2
    {
      char w2n[256]; snprintf(w2n, sizeof w2n, "%s.feed_forward.w2.weight", prefix);
      q8_gemm(get(w2n)->dq, get(w2n)->ds, h2, INTER, h4, DH, DH, INTER, 12);
    }
    // post_ffn_norm (no sandwich, so just ffn_norm? Actually no post_ffn_norm in non-sandwich)
    // residual add
    for (int i = 0; i < 12*DH; i++) x[i] += h4[i];
    DT_MARK(5);
    // per-layer rms
    // LFM_DEPTH_TRACE=1: per-layer rms of the 12-position trunk state, the same quantity
    // the oracle's forward hooks record (tf/layers_*.json), so a layer can be localised.
    if (getenv("LFM_DEPTH_TRACE")) {
      double acc = 0; for (int i = 0; i < 12*DH; i++) acc += (double)x[i]*x[i];
      printf("layer%d rms=%.4f first4=[%.4f,%.4f,%.4f,%.4f]\n", L, sqrt(acc/(12*DH)), x[0], x[1], x[2], x[3]);
    }
  }
  // final embedding_norm (once, after all layers)
  {
    char en[256]; snprintf(en, sizeof en, "depthformer.model.embedding_norm.weight");
    for (int t = 0; t < 12; t++) {
      rmsnorm(x + t*DH, get(en)->df, h1 + t*DH, DH, 1e-5f);
      memcpy(x + t*DH, h1 + t*DH, DH*4);
    }
  }

  // Now compute logits for each codebook using the stacked head
  // logits[c, v] = stacked_head[c*2017 + v] @ x[c]  (each codebook has its own position)
  // Actually: the trunk outputs x[c] for codebook c, and head c computes logits
  // out[c, v] = W_head[c*CBS + v] . x[c]  (W_head is [24204, 1024])
  // HF Lfm3AudioCodebookEmbedding.logits(): F.linear(embedding_norm(h), embedding.weight).
  // Each codebook has its OWN embedding_norm before the tied head. Missing this step was
  // the cause of the final-logit error (layer rms matched; logits did not).
  DT_MARK(6);
  Ent* head = get("depth_head_stacked.weight");
  float xn[DH];
  for (int c = 0; c < NCB; c++) {
    char en[256]; snprintf(en, sizeof en, "depth_embeddings.%d.embedding_norm.weight", c);
    rmsnorm(x + c*DH, get(en)->df, xn, DH, 1e-5f);
    q8_gemv(head->dq + (size_t)(c*CBS)*DH,
            head->ds + (size_t)(c*CBS)*(DH/Q8_BLK),
            xn, out_logits + c*CBS, CBS, DH);
  }
  DT_MARK(7);
  if (getenv("LFM_DEPTH_TIME")) {
    printf("DEPTH_TIME ms:");
    for (int i = 0; i < 8; i++) printf(" %s=%.1f", g_dtn[i], g_dt[i]);
    printf("\n"); fflush(stdout);
  }

}

// One MaskGIT frame (shared by --maskgit and --serve): fills codes[12]. Returns 0 or an
// exit code. The body is the original --maskgit loop, moved verbatim.
static int maskgit_frame(std::vector<Ent>& ents, std::map<std::string, Ent*>& by,
                         const float* hidden, const std::vector<int>& order, long no,
                         const std::vector<int>& hist, int n_hist, int lo, float PEN,
                         int* codes_out) {
    int codes[12] = {0};
    unsigned char revealed[12] = {0};
    float logits[12 * 2017];
    for (long p = 0; p + 4 <= no; p += 4) {
      unsigned char masked[12];
      for (int c = 0; c < 12; c++) masked[c] = (unsigned char)!revealed[c];
      depth_forward(ents, by, hidden, codes, masked, logits);
      for (int j = 0; j < 4; j++) {
        int idx = order[p + j];
        if (idx < 0 || idx >= 12) { printf("bad order entry %d\n", idx); return 2; }
        float s[2017];
        memcpy(s, logits + idx * 2017, sizeof s);
        // HF RepetitionPenaltyLogitsProcessor, 2-D path: it GATHERS the history's scores
        // from the unmodified row and scatters the penalised values back, so a code that
        // appears k times in the window is penalised ONCE. Penalising per occurrence
        // over-penalised repeated codes (found on fixture p3, frames 11/14/15).
        static unsigned char seen[2017];
        memset(seen, 0, sizeof seen);
        for (int i = lo; i < n_hist; i++) {
          int v = hist[(size_t)i * 12 + idx];                   // per-codebook history, window 64
          if (v < 0 || v >= 2017) { printf("history code out of range %d\n", v); return 2; }
          if (seen[v]) continue;
          seen[v] = 1;
          float g = s[v];
          s[v] = (g < 0) ? g * PEN : g / PEN;
        }
        int best = 0;
        for (int v = 1; v < 2017; v++) if (s[v] > s[best]) best = v;
        codes[idx] = best;
        revealed[idx] = 1;
      }
      if (getenv("LFM_DEPTH_TRACE")) {
        printf("pass %ld codes = [", p);
        for (int c = 0; c < 12; c++) printf("%d%s", revealed[c] ? codes[c] : -1, c == 11 ? "]" : ",");
        printf("  order = [");
        for (int j = 0; j < 4; j++) printf("%d%s", order[p + j], j == 3 ? "]" : ",");
        printf("  n_hist=%d\n", n_hist);
        fflush(stdout);
      }
    }
    for (int c = 0; c < 12; c++)
      if (!revealed[c]) { printf("codebook %d never scheduled by the reveal order\n", c); return 2; }
    memcpy(codes_out, codes, sizeof codes);
    return 0;
}
// The frame's backbone input embedding (the gated --embeds reduction), written to eo.
static int frame_embed_out(std::map<std::string, Ent*>& by, const int* codes, const char* eo) {
    {
      const int NCB2 = 12, CBS2 = 2017, SH2 = 2048;
      Ent* cb2 = by.count("codebook_embedding.weight") ? by["codebook_embedding.weight"] : nullptr;
      if (!cb2) { printf("MISSING TENSOR: codebook_embedding.weight\n"); return 3; }
      const uint16_t* E = (const uint16_t*)cb2->dq;
      std::vector<float> o((size_t)SH2);
      for (int d = 0; d < SH2; d++) {
        float acc = 0.f;
        for (int i = 0; i < NCB2; i++) {
          int code = codes[i];
          if (code < 0 || code >= CBS2) { printf("FAIL code %d outside [0,%d)\n", code, CBS2); return 3; }
          acc += bf16_to_f32(E[((size_t)code + (size_t)i * CBS2) * SH2 + d]);
        }
        o[d] = acc;
      }
      double rms = 0;
      for (int d = 0; d < SH2; d++) rms += (double)o[d] * o[d];
      if (!(rms > 0.0) || !(rms == rms)) { printf("FAIL maskgit embedding rms %.6f\n", rms); return 3; }
      FILE* fe = fopen(eo, "wb");
      if (!fe) { printf("cannot write %s\n", eo); return 2; }
      if (fwrite(o.data(), 4, o.size(), fe) != o.size()) { printf("short write\n"); fclose(fe); return 2; }
      fclose(fe);
      printf("maskgit: embedding rms %.6f -> %s\n", sqrt(rms), eo);
      fflush(stdout);
    }
    return 0;
}

int main(int argc, char** argv) {
  // single pass:  lfm3_depth <blob> <hidden.f32> <codes.i32> <masked.u8> <out.f32>
  // MaskGIT:      lfm3_depth <blob> <hidden.f32> --maskgit <order.i32> <hist.i32> <out_codes.i32> [penalty] [window]
  //   order.i32: n_pass*4 int32, pass-major reveal order (the oracle's order.<f>.i32, fed as data)
  //   hist.i32 : n_hist*12 int32, frame-major per-codebook history (may be empty); the
  //              repetition penalty is applied over the last `window` frames, default 64, penalty 1.1
  if (argc < 6 && !(argc >= 5 && !strcmp(argv[2], "--embeds")) && !(argc >= 3 && !strcmp(argv[2], "--serve"))) {
    printf("usage: %s <depth.lfm3q8> <hidden.f32> <codes.i32> <masked.u8> <out.f32>\n", argv[0]);
    printf("       %s <depth.lfm3q8> <hidden.f32> --maskgit <order.i32> <hist.i32> <out_codes.i32> [penalty] [window]\n", argv[0]);
    printf("       %s <depth.lfm3q8> --embeds <codes.i32> <out.f32> [nframes]\n", argv[0]);
    return 2;
  }
  std::vector<Ent> ents;
  std::map<std::string, Ent*> by;
  char* js = nullptr;
  if (!load_blob(argv[1], ents, by, js)) return 1;

  /* Resident mode: lfm3_depth <blob> --serve. The blob is loaded once; each stdin line
       M <hidden.f32> <order.i32> <hist.i32> <codes_out.i32> <emb_out.f32> <penalty> <window>
     runs exactly the --maskgit frame (maskgit_frame) and the LFM_DEPTH_EMBEDS_OUT embedding
     (frame_embed_out), then prints "DEPTH_DONE <rc>". "Q" or EOF ends it. Per frame the
     inputs, code and order of operations are the per-process path's, so codes and
     embeddings are the same bytes; what goes away is the per-frame process start and blob
     read (2.52 s per frame in the speech loop vs ~0.45 s of compute). */
  if (argc >= 3 && !strcmp(argv[2], "--serve")) {
    auto rd = [](const char* path, std::vector<int>& v) -> bool {
      v.clear(); FILE* f = fopen(path, "rb"); if (!f) return false;
      fseek(f, 0, SEEK_END); long n = ftell(f) / 4; fseek(f, 0, SEEK_SET);
      v.resize((size_t)n); bool ok = n == 0 || fread(v.data(), 4, (size_t)n, f) == (size_t)n;
      fclose(f); return ok;
    };
    printf("DEPTH_READY\n"); fflush(stdout);
    char line[8192];
    while (fgets(line, sizeof line, stdin)) {
      if (line[0] == 'Q') break;
      char hp[2048], op[2048], hsp[2048], cp[2048], ep[2048]; float PEN; int WIN;
      int rc = 0;
      if (sscanf(line, "M %2047s %2047s %2047s %2047s %2047s %f %d", hp, op, hsp, cp, ep, &PEN, &WIN) != 7) rc = 2;
      float hidden[2048];
      std::vector<int> order, hist;
      if (!rc) { FILE* fh = fopen(hp, "rb"); if (!fh || fread(hidden, 4, 2048, fh) != 2048) rc = 2; if (fh) fclose(fh); }
      if (!rc && (!rd(op, order) || order.empty() || order.size() % 4)) rc = 2;
      if (!rc) { FILE* t = fopen(hsp, "rb"); if (t) { fclose(t); if (!rd(hsp, hist)) rc = 2; } else hist.clear(); }
      if (!rc && hist.size() % 12) rc = 2;
      int codes[12];
      if (!rc) {
        const int n_hist = (int)(hist.size() / 12);
        int lo = n_hist - WIN; if (lo < 0) lo = 0;
        rc = maskgit_frame(ents, by, hidden, order, (long)order.size(), hist, n_hist, lo, PEN, codes);
      }
      if (!rc) {
        FILE* foc = fopen(cp, "wb");
        if (!foc || fwrite(codes, 4, 12, foc) != 12) rc = 2;
        if (foc) fclose(foc);
        printf("maskgit: codes = [");
        for (int c = 0; c < 12; c++) printf("%d%s", codes[c], c == 11 ? "]" : ",");
        printf("\n");
      }
      if (!rc) rc = frame_embed_out(by, codes, ep);
      printf("DEPTH_DONE %d\n", rc); fflush(stdout);
    }
    return 0;
  }

  /* code -> backbone input embedding (the S6 speech-out loop feeds this at each
     audio_out_token_id position):
       lfm3_depth <blob> --embeds <codes.i32> <out.f32> [nframes]
     codes.i32 is frame-major n*NCB (the oracle's per-frame codes concatenated); out.f32 is
     n*SRC_H. Mirrors HF get_audio_out_embeds: codebook_embedding(codes + arange(NCB)*CBS).sum(-2),
     codebooks summed in ascending order in f32, exactly as HF reduces dim -2. */
  if (argc >= 5 && !strcmp(argv[2], "--embeds")) {
    const int NCB = 12, CBS = 2017, SH = 2048;   // same values depth_forward uses
    Ent* cb = by.count("codebook_embedding.weight") ? by["codebook_embedding.weight"] : nullptr;
    if (!cb) { printf("MISSING TENSOR: codebook_embedding.weight\n"); return 3; }
    if (cb->dtype != 3 || cb->ne[0] != NCB * CBS || cb->ne[1] != SH) {
      printf("FAIL codebook_embedding.weight: want bf16 [%d,%d], got dtype %d [%d,%d]\n",
             NCB * CBS, SH, cb->dtype, cb->ne[0], cb->ne[1]);
      return 3;
    }
    FILE* fc = fopen(argv[3], "rb");
    if (!fc) { printf("cannot open %s\n", argv[3]); return 2; }
    fseek(fc, 0, SEEK_END); long nc = ftell(fc) / 4; fseek(fc, 0, SEEK_SET);
    std::vector<int> codes((size_t)nc);
    if (nc <= 0 || fread(codes.data(), 4, (size_t)nc, fc) != (size_t)nc) { printf("short codes\n"); fclose(fc); return 2; }
    fclose(fc);
    if (nc % NCB) { printf("FAIL codes count %ld not a multiple of %d\n", nc, NCB); return 3; }
    long n = nc / NCB;
    if (argc > 5) { long want = atol(argv[5]); if (want != n) { printf("FAIL nframes given %ld != %ld from file\n", want, n); return 3; } }
    const uint16_t* E = (const uint16_t*)cb->dq;
    std::vector<float> out((size_t)n * SH);
    for (long j = 0; j < n; j++) {
      for (int d = 0; d < SH; d++) {
        float acc = 0.f;
        for (int i = 0; i < NCB; i++) {
          long code = codes[(size_t)j * NCB + i];
          if (code < 0 || code >= CBS) {
            printf("FAIL code %ld outside [0,%d) at frame %ld codebook %d\n", code, CBS, j, i);
            return 3;
          }
          acc += bf16_to_f32(E[((size_t)code + (size_t)i * CBS) * SH + d]);
        }
        out[(size_t)j * SH + d] = acc;
      }
      // plausibility: a real embedding is never the zero vector (HF's own table is random-init-free)
      double rms = 0;
      for (int d = 0; d < SH; d++) rms += (double)out[(size_t)j * SH + d] * out[(size_t)j * SH + d];
      if (!(rms > 0.0)) { printf("FAIL frame %ld embedding rms is zero\n", j); return 3; }
      if (!(rms == rms)) { printf("FAIL frame %ld embedding not finite\n", j); return 3; }
      if (j == 0) printf("EMBEDS frame0 rms %.6f\n", sqrt(rms));
    }
    FILE* fw = fopen(argv[4], "wb");
    if (!fw) { printf("cannot write %s\n", argv[4]); return 2; }
    if (fwrite(out.data(), 4, out.size(), fw) != out.size()) { printf("short write\n"); fclose(fw); return 2; }
    fclose(fw);
    printf("LFM3DEPTH-EMBEDS-1 frames=%ld dim=%d table=%dx%d bf16\n", n, SH, NCB * CBS, SH);
    return 0;
  }

  float hidden[2048];
  FILE* fh = fopen(argv[2], "rb");
  if (!fh) { printf("cannot open %s\n", argv[2]); return 2; }
  if (fread(hidden, 4, 2048, fh) != 2048) { printf("short hidden\n"); fclose(fh); return 2; }
  fclose(fh);

  if (!strcmp(argv[3], "--maskgit")) {
    if (argc < 7) { printf("maskgit needs order, history, out\n"); return 2; }
    const float PEN = (argc > 7) ? (float)atof(argv[7]) : 1.1f;
    const int WIN = (argc > 8) ? atoi(argv[8]) : 64;
    FILE* fo1 = fopen(argv[4], "rb");
    if (!fo1) { printf("cannot open %s\n", argv[4]); return 2; }
    fseek(fo1, 0, SEEK_END); long no = ftell(fo1) / 4; fseek(fo1, 0, SEEK_SET);
    std::vector<int> order(no);
    if (fread(order.data(), 4, no, fo1) != (size_t)no) { printf("short order\n"); return 2; }
    fclose(fo1);
    if (no % 4) { printf("order length %ld not a multiple of 4 (schedule [4,4,4])\n", no); return 2; }
    FILE* fhist = fopen(argv[5], "rb");
    std::vector<int> hist;
    if (fhist) {
      fseek(fhist, 0, SEEK_END); long nh = ftell(fhist) / 4; fseek(fhist, 0, SEEK_SET);
      hist.resize(nh);
      if (nh && fread(hist.data(), 4, nh, fhist) != (size_t)nh) { printf("short hist\n"); fclose(fhist); return 2; }
      fclose(fhist);
    }
    if (hist.size() % 12) { printf("history length %zu not a multiple of 12\n", hist.size()); return 2; }
    int n_hist = (int)(hist.size() / 12);
    int lo = n_hist - WIN; if (lo < 0) lo = 0;

    int codes[12];
    if (int rc = maskgit_frame(ents, by, hidden, order, no, hist, n_hist, lo, PEN, codes)) return rc;
    FILE* foc = fopen(argv[6], "wb");
    if (!foc) { printf("cannot write %s\n", argv[6]); return 2; }
    fwrite(codes, 4, 12, foc);
    fclose(foc);
    printf("maskgit: codes = [");
    for (int c = 0; c < 12; c++) printf("%d%s", codes[c], c == 11 ? "]" : ",");
    printf("\n");
    /* LFM_DEPTH_EMBEDS_OUT=<file>: also emit this frame's backbone input embedding, in the
       same form --embeds produces. The speech loop used to fork lfm3_depth twice per frame
       (maskgit, then --embeds), paying the 1.3 s blob load twice; the computation is tiny
       next to the maskgit passes, so one process does both. Same table, same reduction
       order as the gated --embeds path. */
    if (const char* eo = getenv("LFM_DEPTH_EMBEDS_OUT"))
      if (int rc = frame_embed_out(by, codes, eo)) return rc;
    return 0;
  }

  int codes[12];
  FILE* fc = fopen(argv[3], "rb");
  if (!fc) { printf("cannot open %s\n", argv[3]); return 2; }
  if (fread(codes, 4, 12, fc) != 12) { printf("short codes\n"); fclose(fc); return 2; }
  fclose(fc);
  unsigned char masked[12];
  FILE* fm = fopen(argv[4], "rb");
  if (!fm) { printf("cannot open %s\n", argv[4]); return 2; }
  if (fread(masked, 1, 12, fm) != 12) { printf("short masked\n"); fclose(fm); return 2; }
  fclose(fm);

  float out_logits[12 * 2017];
  depth_forward(ents, by, hidden, codes, masked, out_logits);
  FILE* fo = fopen(argv[5], "wb");
  if (!fo) { printf("cannot write %s\n", argv[5]); return 2; }
  fwrite(out_logits, 4, 12 * 2017, fo);
  fclose(fo);
  printf("done: wrote %d floats\n", 12 * 2017);
  return 0;
}
