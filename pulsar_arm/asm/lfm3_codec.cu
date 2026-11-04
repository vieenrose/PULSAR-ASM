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
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
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

static void f32_gemv(const float* w, const float* x, float* y, int N, int K) {
  for (int n = 0; n < N; n++) {
    float acc = 0;
    const float* row = w + (size_t)n * K;
    for (int k = 0; k < K; k++) acc += row[k] * x[k];
    y[n] = acc;
  }
}

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
  if (argc < 5) { printf("usage: %s <codec.lfm3q8> <codes.i32> <T> <out.f32>\n", argv[0]); return 2; }
  int T = atoi(argv[3]);
  const int T6 = T * FSTACK;
  const int HID = 512, INTER = 2304, NH = 8, NKV = 2, HD = 64, NL = 8;
  const float EPS = 1e-5f, THETA = 1e6f;
  const int LEVELS[4] = {8, 7, 6, 6}, BASIS[4] = {1, 8, 56, 336};

  std::vector<Ent> ents;
  std::map<std::string, Ent*> by;
  if (!load_blob(argv[1], ents, by)) return 1;
  auto get = [&](const char* n) -> Ent* {
    auto it = by.find(n);
    if (it == by.end()) { printf("MISSING TENSOR: %s\n", n); exit(2); }
    return it->second;
  };

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

    for (int t = 0; t < T6; t++) rmsnorm(&X[(size_t)t * HID], on->df, &h[(size_t)t * HID], HID, EPS);

    if (is_attn) {
      Ent* wq = get((P + ".self_attn.q_proj.weight").c_str());
      Ent* wk = get((P + ".self_attn.k_proj.weight").c_str());
      Ent* wv = get((P + ".self_attn.v_proj.weight").c_str());
      Ent* wo = get((P + ".self_attn.o_proj.weight").c_str());
      Ent* wg = get((P + ".self_attn.gate_proj.weight").c_str());
      float* qnw = get((P + ".self_attn.q_norm.weight").c_str())->df;
      float* knw = get((P + ".self_attn.k_norm.weight").c_str())->df;
      std::vector<float> K((size_t)T6 * NKV * HD), V((size_t)T6 * NKV * HD), G((size_t)T6 * NH);
      std::vector<float> Q((size_t)T6 * NH * HD);
      for (int t = 0; t < T6; t++) {
        q8_gemv(wq->dq, wq->ds, &h[(size_t)t * HID], &Q[(size_t)t * NH * HD], NH * HD, HID);
        q8_gemv(wk->dq, wk->ds, &h[(size_t)t * HID], &K[(size_t)t * NKV * HD], NKV * HD, HID);
        q8_gemv(wv->dq, wv->ds, &h[(size_t)t * HID], &V[(size_t)t * NKV * HD], NKV * HD, HID);
        q8_gemv(wg->dq, wg->ds, &h[(size_t)t * HID], &G[(size_t)t * NH], NH, HID);
        for (int hh = 0; hh < NH; hh++) rmsnorm(&Q[(size_t)t * NH * HD + hh * HD], qnw, qh.data(), HD, EPS);
        for (int hh = 0; hh < NH; hh++) memcpy(&Q[(size_t)t * NH * HD + hh * HD], qh.data(), HD * 4);
        for (int kv = 0; kv < NKV; kv++) rmsnorm(&K[(size_t)t * NKV * HD + kv * HD], knw, kh.data(), HD, EPS);
        for (int kv = 0; kv < NKV; kv++) memcpy(&K[(size_t)t * NKV * HD + kv * HD], kh.data(), HD * 4);
        for (int hh = 0; hh < NH; hh++) rope_head(&Q[(size_t)t * NH * HD + hh * HD], t, HD, THETA);
        for (int kv = 0; kv < NKV; kv++) rope_head(&K[(size_t)t * NKV * HD + kv * HD], t, HD, THETA);
      }
      const float scale = 1.0f / sqrtf((float)HD);
      const int WIN = 30;                              // causal + sliding window of 30
      for (int t = 0; t < T6; t++) {
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
        q8_gemv(wo->dq, wo->ds, attn.data(), &tmp[(size_t)t * HID], HID, NH * HD);
      }
    } else {
      Ent* win_ = get((P + ".conv.in_proj.weight").c_str());
      Ent* wout = get((P + ".conv.out_proj.weight").c_str());
      Ent* wc = get((P + ".conv.conv.weight").c_str());           // [512,1,7] f32
      const float* cw = wc->df;
      std::vector<float> z((size_t)T6 * HID), conv((size_t)T6 * HID);
      for (int t = 0; t < T6; t++) {
        q8_gemv(win_->dq, win_->ds, &h[(size_t)t * HID], &proj[(size_t)t * 3 * HID], 3 * HID, HID);
        const float* B = &proj[(size_t)t * 3 * HID];
        const float* XG = &proj[(size_t)t * 3 * HID + 2 * HID];   // BCx chunk order: B, C, x
        for (int i = 0; i < HID; i++) z[(size_t)t * HID + i] = B[i] * XG[i];
      }
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
        q8_gemv(wout->dq, wout->ds, &conv[(size_t)t * HID], &tmp[(size_t)t * HID], HID, HID);
      }
    }

    for (int t = 0; t < T6; t++) {                                  // sandwich: post_operator_norm
      rmsnorm(&tmp[(size_t)t * HID], pon->df, h.data(), HID, EPS);
      for (int i = 0; i < HID; i++) X[(size_t)t * HID + i] += h[i];
    }

    Ent* w1 = get((P + ".feed_forward.w1.weight").c_str());
    Ent* w2 = get((P + ".feed_forward.w2.weight").c_str());
    Ent* w3 = get((P + ".feed_forward.w3.weight").c_str());
    std::vector<float> u(INTER), gate(INTER), ffn(HID);
    for (int t = 0; t < T6; t++) {
      rmsnorm(&X[(size_t)t * HID], fn->df, h.data(), HID, EPS);
      q8_gemv(w1->dq, w1->ds, h.data(), u.data(), INTER, HID);
      q8_gemv(w3->dq, w3->ds, h.data(), gate.data(), INTER, HID);
      for (int i = 0; i < INTER; i++) u[i] = silu(u[i]) * gate[i];
      q8_gemv(w2->dq, w2->ds, u.data(), ffn.data(), HID, INTER);
      rmsnorm(ffn.data(), pfn->df, h.data(), HID, EPS);
      for (int i = 0; i < HID; i++) X[(size_t)t * HID + i] += h[i];
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
  std::vector<float> hn(HID);
  for (int t = 0; t < T6; t++) {
    rmsnorm(&X[(size_t)t * HID], enorm, hn.data(), HID, EPS);
    q8_gemv(wlin->dq, wlin->ds, hn.data(), &lin[(size_t)t * 2 * NBIN], 2 * NBIN, HID);
    for (int i = 0; i < 2 * NBIN; i++) lin[(size_t)t * 2 * NBIN + i] += blin[i];
  }

  if (const char* d = getenv("LFM_CODEC_DUMP")) {
    char p[512]; snprintf(p, sizeof p, "%s/lin.f32", d);
    FILE* g = fopen(p, "wb"); if (g) { fwrite(lin.data(), 4, lin.size(), g); fclose(g); }
  }
  const size_t L = (size_t)(T6 - 1) * NHOP + NFFT;
  const int trim = (NFFT - NHOP) / 2;
  std::vector<float> audio(L, 0.f), envelope(L, 0.f);
  std::vector<float> win(NFFT);
  for (int n = 0; n < NFFT; n++) win[n] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * n / (NFFT - 1));
  std::vector<float> frame(NFFT);
  for (int f = 0; f < T6; f++) {
    const float* lm = &lin[(size_t)f * 2 * NBIN];
    const float* an = lm + NBIN;
    for (int n = 0; n < NFFT; n++) {                                  // irfft, k ascending, DC/Nyquist last
      double acc = 0;
      for (int k = 1; k < NBIN - 1; k++) {
        float mag = expf(lm[k]);
        float a = mag * cosf(an[k]), b = mag * sinf(an[k]);
        float th = 2.0f * (float)M_PI * k * n / (float)NFFT;
        acc += 2.0 * ((double)a * cosf(th) - (double)b * sinf(th));
      }
      acc += (double)expf(lm[0]);
      acc += (double)((n & 1) ? -1.0f : 1.0f) * expf(lm[NBIN - 1]);  // (-1)^n * a_{N/2}
      frame[n] = (float)(acc / NFFT) * win[n];
    }
    for (int n = 0; n < NFFT; n++) {
      audio[(size_t)f * NHOP + n] += frame[n];
      envelope[(size_t)f * NHOP + n] += win[n] * win[n];
    }
    if (getenv("LFM_CODEC_TRACE") && f % 90 == 0) { printf("istft frame %d\n", f); fflush(stdout); }
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
