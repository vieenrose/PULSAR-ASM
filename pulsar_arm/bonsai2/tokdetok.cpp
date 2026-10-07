// tokdetok: fork-API tokenizer tools for the engine pipeline (no server needed).
//   tokdetok tok <model.gguf> <id> [id...]   -> detokenized text
//   tokdetok tokstr <model.gguf> <text...>    -> token ids (add_bos=false)
// Build: g++ -O2 -o tokdetok tokdetok.cpp -I$F/include -I$F/ggml/include
//   -L$B/bin -lllama -lggml -lggml-base -Wl,-rpath,$B/bin -lm
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    const char *mode = argv[1];
    llama_model_params mp = llama_model_default_params();
    llama_model *m = llama_model_load_from_file(argv[2], mp);
    if (!m) { printf("model load fail\n"); return 1; }
    const llama_vocab *voc = llama_model_get_vocab(m);
    if (!strcmp(mode, "tok")) {
        std::vector<llama_token> ids;
        for (int a = 3; a < argc; a++) ids.push_back(atoi(argv[a]));
        // whole-tail single call (correct BPE handling):
        int need = llama_detokenize(voc, ids.data(), ids.size(), nullptr, 0, false, true);
        std::string big(need > 0 ? need : 4096, '\0');
        int got = llama_detokenize(voc, ids.data(), ids.size(), big.data(), big.size(), false, true);
        big.resize(got > 0 ? got : 0);
        fwrite(big.data(), 1, big.size(), stdout);
        printf("\n");
    } else if (!strcmp(mode, "tokstr")) {
        std::string text;
        for (int a = 3; a < argc; a++) {
            if (a > 3) text += " ";
            text += argv[a];
        }
        std::vector<llama_token> out(text.size() + 16);
        int n = llama_tokenize(voc, text.c_str(), text.size(), out.data(), out.size(), false, false);
        if (n < 0) { printf("tokenize fail\n"); return 1; }
        for (int i = 0; i < n; i++) printf("%d%c", out[i], i + 1 < n ? ' ' : '\n');
    } else { printf("usage: tok|tokstr\n"); return 1; }
    llama_model_free(m);
    return 0;
}
