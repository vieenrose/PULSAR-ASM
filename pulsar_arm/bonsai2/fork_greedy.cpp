// fork_greedy: greedy stream (argmax of every prefix) from the PrismML fork, CPU,
// f16 K/V cache, flash attention on, one thread. usage: fork_greedy model.gguf N id id ...
// prints one id per line, then "# margin" lines (top1-top2 logit) on stderr.
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include "ggml-backend.h"
// FG_TDUMP=file FG_TSTEP=s: at greedy step s (0 = the prompt batch), append every F32
// graph tensor as {u32 namelen, name, u64 n, f32[n]} (only the last token's column).
static FILE *tdf; static int tcur = -1, tstep = -1;
static bool tcb(struct ggml_tensor *t, bool ask, void *) {
    if (tcur != tstep || !tdf) return false;
    if (ask) return t->type == GGML_TYPE_F32;
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) return true;
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * 4);
    uint32_t nl = strlen(t->name); uint64_t n = v.size();
    // keep the last row group along ne1 when the tensor spans the batch
    fwrite(&nl, 4, 1, tdf); fwrite(t->name, 1, nl, tdf);
    uint64_t ne[4] = {(uint64_t)t->ne[0], (uint64_t)t->ne[1], (uint64_t)t->ne[2], (uint64_t)t->ne[3]};
    fwrite(ne, 8, 4, tdf); fwrite(&n, 8, 1, tdf); fwrite(v.data(), 4, n, tdf);
    return true;
}
int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage\n"); return 1; }
    const char *path = argv[1]; int N = atoi(argv[2]);
    std::vector<llama_token> ids;
    for (int i = 3; i < argc; i++) ids.push_back(atoi(argv[i]));
    const char *dump = getenv("FG_DUMP");   // write all logits of every step to this file
    int nth = getenv("FG_THREADS") ? atoi(getenv("FG_THREADS")) : 1;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model *m = llama_model_load_from_file(path, mp);
    if (!m) return 2;
    llama_context_params cp = llama_context_default_params();
    // FG_BATCH: n_batch = n_ubatch (default 512); a longer prompt is decoded in chunks of it
    const int nb = getenv("FG_BATCH") ? atoi(getenv("FG_BATCH")) : 512;
    cp.n_ctx = 4096; cp.n_batch = nb; cp.n_ubatch = nb;
    cp.n_threads = nth; cp.n_threads_batch = nth;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
    if (getenv("FG_TDUMP")) { tdf = fopen(getenv("FG_TDUMP"), "wb"); tstep = atoi(getenv("FG_TSTEP")); cp.cb_eval = tcb; cp.cb_eval_user_data = nullptr; }
    llama_context *c = llama_init_from_model(m, cp);
    if (!c) return 3;
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(m));
    FILE *df = dump ? fopen(dump, "wb") : nullptr;
    // the last prompt chunk is step 0
    size_t off = 0;
    while (ids.size() - off > (size_t)nb) {
        llama_batch pb = llama_batch_get_one(ids.data() + off, nb);
        if (llama_decode(c, pb)) { fprintf(stderr, "decode fail\n"); return 4; }
        off += nb;
    }
    llama_batch b = llama_batch_get_one(ids.data() + off, ids.size() - off);
    for (int s = 0; s < N; s++) {
        tcur = s;
        if (llama_decode(c, b)) { fprintf(stderr, "decode fail\n"); return 4; }
        const float *lg = llama_get_logits_ith(c, -1);
        if (df) fwrite(lg, 4, nv, df);
        int a = 0; for (int i = 1; i < nv; i++) if (lg[i] > lg[a]) a = i;
        int a2 = a == 0 ? 1 : 0; for (int i = 0; i < nv; i++) if (i != a && lg[i] > lg[a2]) a2 = i;
        printf("%d\n", a); fflush(stdout);
        fprintf(stderr, "# step %d id %d logit %.4f margin %.4f (2nd %d)\n", s, a, lg[a], lg[a] - lg[a2], a2);
        static llama_token t; t = a;
        b = llama_batch_get_one(&t, 1);
    }
    if (df) fclose(df);
    if (tdf) fclose(tdf);
    llama_free(c); llama_model_free(m);
    return 0;
}
