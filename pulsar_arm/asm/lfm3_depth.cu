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

// Q8 GEMV: y[n] = sum_k dequant(q[n,k], s[n,k/32]) * x[k]
static void q8_gemv(const int8_t* q, const float* s, const float* x, float* y, int N, int K) {
  for (int n = 0; n < N; n++) {
    float acc = 0;
    for (int k = 0; k < K; k++) {
      float w = (float)q[n*K + k] * s[n*(K/32) + k/32];
      acc += w * x[k];
    }
    y[n] = acc;
  }
}

// bf16 GEMV (for bf16 weights)
static void bf16_gemv(const uint16_t* w, const float* x, float* y, int N, int K) {
  for (int n = 0; n < N; n++) {
    float acc = 0;
    for (int k = 0; k < K; k++) {
      acc += bf16_to_f32(w[n*K + k]) * x[k];
    }
    y[n] = acc;
  }
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

int main(int argc, char** argv) {
  if (argc < 6) {
    printf("usage: %s <depth.lfm3q8> <hidden.f32> <codes.i32> <masked.u8> <out.f32>\n", argv[0]);
    return 2;
  }
  std::vector<Ent> ents;
  std::map<std::string, Ent*> by;
  char* js = nullptr;
  if (!load_blob(argv[1], ents, by, js)) return 1;

  // read hidden
  FILE* fh = fopen(argv[2], "rb");
  float hidden[2048];
  fread(hidden, 4, 2048, fh); fclose(fh);
  // read codes
  FILE* fc = fopen(argv[3], "rb");
  int codes[12];
  fread(codes, 4, 12, fc); fclose(fc);
  // read masked
  FILE* fm = fopen(argv[4], "rb");
  unsigned char masked[12];
  fread(masked, 1, 12, fm); fclose(fm);

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
  float depth_in[12 * 1024];
  q8_gemv(dl->dq, dl->ds, hidden, depth_in, 12 * DH, SH);
  for (int i = 0; i < 12 * DH; i++) depth_in[i] += dl_bias->df[i];

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
      for (int t = 0; t < 12; t++) {
      q8_gemv(get(wname)->dq, get(wname)->ds, h1 + t*DH, q + t*NHEADS*HD, NHEADS*HD, DH);
    }
      snprintf(wname, sizeof wname, "%s.self_attn.k_proj.weight", prefix);
    for (int t = 0; t < 12; t++) {
      q8_gemv(get(wname)->dq, get(wname)->ds, h1 + t*DH, k + t*NKV*HD, NKV*HD, DH);
    }
      snprintf(wname, sizeof wname, "%s.self_attn.v_proj.weight", prefix);
    for (int t = 0; t < 12; t++) {
      q8_gemv(get(wname)->dq, get(wname)->ds, h1 + t*DH, v + t*NKV*HD, NKV*HD, DH);
    }
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
    // gate
    for (int t = 0; t < 12; t++) {
      char gn[256]; snprintf(gn, sizeof gn, "%s.self_attn.gate_proj.weight", prefix);
      float gate[NHEADS];
      bf16_gemv((uint16_t*)get(gn)->dq, h1 + t*DH, gate, NHEADS, DH);  // wait, gate_proj is Q8 not bf16
      // actually gate_proj is Q8
    }
    // redo gate properly with Q8
    for (int t = 0; t < 12; t++) {
      char gn[256]; snprintf(gn, sizeof gn, "%s.self_attn.gate_proj.weight", prefix);
      float gate[NHEADS];
      q8_gemv(get(gn)->dq, get(gn)->ds, h1 + t*DH, gate, NHEADS, DH);
      for (int h = 0; h < NHEADS; h++) {
        float g = 1.0f / (1.0f + expf(-gate[h]));
        for (int d = 0; d < HD; d++) attn[t*NHEADS*HD + h*HD + d] *= g;
      }
    }
    // o_proj
    for (int t = 0; t < 12; t++) {
      char on[256]; snprintf(on, sizeof on, "%s.self_attn.o_proj.weight", prefix);
      q8_gemv(get(on)->dq, get(on)->ds, attn + t*NHEADS*HD, h4 + t*DH, DH, NHEADS*HD);
    }
    // residual add
    for (int i = 0; i < 12*DH; i++) x[i] += h4[i];
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
    for (int t = 0; t < 12; t++) {
      char w1n[256], w3n[256];
      snprintf(w1n, sizeof w1n, "%s.feed_forward.w1.weight", prefix);
      snprintf(w3n, sizeof w3n, "%s.feed_forward.w3.weight", prefix);
      q8_gemv(get(w1n)->dq, get(w1n)->ds, h1 + t*DH, h2 + t*INTER, INTER, DH);
      q8_gemv(get(w3n)->dq, get(w3n)->ds, h1 + t*DH, h3 + t*INTER, INTER, DH);
    }
    // silu_mul
    for (int i = 0; i < 12*INTER; i++) h2[i] = silu(h2[i]) * h3[i];
    // w2
    for (int t = 0; t < 12; t++) {
      char w2n[256]; snprintf(w2n, sizeof w2n, "%s.feed_forward.w2.weight", prefix);
      q8_gemv(get(w2n)->dq, get(w2n)->ds, h2 + t*INTER, h4 + t*DH, DH, INTER);
    }
    // post_ffn_norm (no sandwich, so just ffn_norm? Actually no post_ffn_norm in non-sandwich)
    // residual add
    for (int i = 0; i < 12*DH; i++) x[i] += h4[i];
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
  Ent* head = get("depth_head_stacked.weight");
  float out_logits[12 * 2017];
  float xn[DH];
  for (int c = 0; c < NCB; c++) {
    char en[256]; snprintf(en, sizeof en, "depth_embeddings.%d.embedding_norm.weight", c);
    rmsnorm(x + c*DH, get(en)->df, xn, DH, 1e-5f);
    q8_gemv(head->dq + (size_t)(c*CBS)*DH,
            head->ds + (size_t)(c*CBS)*(DH/Q8_BLK),
            xn, out_logits + c*CBS, CBS, DH);
  }

  // write output
  FILE* fo = fopen(argv[5], "wb");
  fwrite(out_logits, 4, 12 * CBS, fo);
  fclose(fo);
  printf("done: wrote %d floats\n", 12 * CBS);
  return 0;
}
