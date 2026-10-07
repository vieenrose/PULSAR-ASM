// dump_stage_rms: run 1 token through the fork, print rms of every F32
// vector-shaped intermediate in execution order. Aligns with fwd.c MAG trace.
// Build: g++ -O2 -o dump_stage_rms dump_stage_rms.cpp -I$F/include
//   -I$F/ggml/include -L$B/bin -lllama -lggml -lggml-base -Wl,-rpath,$B/bin -lm
// Run: ./dump_stage_rms <model.gguf> <tokid>
#include "llama.h"
#include "ggml.h"
#include <cmath>
#include <cstdio>
#include <cstring>

static bool want_shape(const ggml_tensor * t) {
    if (t->type != GGML_TYPE_F32) return false;
    if (t->ne[3] > 1) return false;
    long n = t->ne[0] * t->ne[1] * t->ne[2];
    if (n > 65536) return false;
    long n0 = t->ne[0];
    return n0 == 5120 || n0 == 6144 || n0 == 10240 || n0 == 17408 ||
           n0 == 12288 || n0 == 248320 || n0 == 2048 || n0 == 4096 ||
           n0 == 128 || n0 == 256 || n0 == 1024;
}

static bool key_tensor(const char *nm) {
    if (strstr(nm, "reshaped") || strstr(nm, "(view)")) return false;
    return strstr(nm, "input_embed") || strstr(nm, "qkv_mixed") ||
           strstr(nm, "conv_output_silu") || strstr(nm, "z-0") ||
           strstr(nm, "final_output") || strstr(nm, "linear_attn_out") ||
           strstr(nm, "ffn_gate") || strstr(nm, "ffn_up") ||
           strstr(nm, "ffn_out") || strstr(nm, "result_output") ||
           strstr(nm, "attn_output") || strstr(nm, "Qcur_full") ||
           strstr(nm, "Kcur") || strstr(nm, "Vcur") ||
           strstr(nm, "attn_gated") || strstr(nm, "l_out") ||
           strstr(nm, "attn_post_norm") || strstr(nm, "attn_norm-0") ||
           strstr(nm, "attn_residual") || strstr(nm, "normed") ||
           strstr(nm, "embd");
}

static void v8r(const char *tag, const float *d, int s) {
    printf("  %s [%d]:", tag, s);
    for (int i = 0; i < 8; i++) printf(" %.6g", d[s + i]);
    printf("\n");
}

static bool dump_cb(ggml_tensor * t, bool ask, void * ud) {
    (void)ud;
    if (ask) return true;
    if (!want_shape(t) || !t->data) return true;
    long n = t->ne[0] * t->ne[1];
    double s = 0;
    float *d = (float *)t->data;
    for (long i = 0; i < n; i++) s += (double)d[i] * d[i];
    char nm[80] = "?";
    // ggml tensor name lives in t->name when set by the graph cb()
    memcpy(nm, t->name, sizeof(nm) - 1);
    printf("T %-24.24s [%6ld] rms=%.5g\n", nm, n, sqrt(s / n));
    if (key_tensor(nm)) {
        printf("  v8:");
        for (int i = 0; i < 8 && i < n; i++) printf(" %.6g", d[i]);
        printf("\n");
        if (n > 1008) v8r("v1000", d, 1000);
        if (n > 8008 && n <= 20000) v8r("v8000", d, 8000);
    }
    return true;
}

int main(int argc, char **argv) {
    llama_model_params mp = llama_model_default_params();
    llama_model *m = llama_model_load_from_file(argv[1], mp);
    if (!m) { printf("model load fail\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 64;
    cp.n_batch = 64;
    cp.n_threads = 20;
    cp.cb_eval = dump_cb;
    cp.cb_eval_user_data = nullptr;
    // capture everything: decode must produce logits so the full graph runs
    llama_context *ctx = llama_init_from_model(m, cp);
    if (!ctx) { printf("ctx fail\n"); return 1; }
    llama_token tok = atoi(argv[2]);
    llama_batch b = llama_batch_init(1, 0, 1);
    b.n_tokens = 1;
    b.token[0] = tok;
    b.pos[0] = 0;
    b.n_seq_id[0] = 1;
    b.seq_id[0][0] = 0;
    b.logits[0] = 1;
    if (llama_decode(ctx, b) != 0) { printf("decode fail\n"); return 1; }
    const float *lg = llama_get_logits_ith(ctx, 0);
    int nv = llama_vocab_n_tokens(llama_model_get_vocab(m));
    int bi = 0;
    for (int i = 1; i < nv; i++) if (lg[i] > lg[bi]) bi = i;
    printf("TOP %d logit=%.4f (vocab %d)\n", bi, lg[bi], nv);
    // rms of the raw embedding of tok for the emb check
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(m);
    return 0;
}
