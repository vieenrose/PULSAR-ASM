/* lfm3_tok.h -- BPE tokenizer port (S1). See doc/LFM3_PORT_STAGES.md section 12.
   Pipeline: added-token match (longest-first, raw bytes) -> Split regex (hand-rolled
   against shipped L/N/S ranges) -> ByteLevel (byte map) -> BPE with ignore_merges.
   No normalizer. byte_fallback=False, no unk: every final symbol must be in vocab. */
#ifndef LFM3_TOK_H
#define LFM3_TOK_H
#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint8_t *bytes; uint16_t len; uint32_t id;   /* vocab entry (sorted by bytes for search) */
} TokV;
typedef struct {
    uint8_t *key; uint32_t klen; uint32_t rank;   /* merge key = left NUL right, sorted */
} TokM;

typedef struct {
    uint32_t n_vocab, n_merges, n_added;
    uint8_t **vbytes; uint16_t *vlen;             /* by id: vocab strings */
    TokV *vsorted;                                /* by bytes: for id lookup */
    TokM *msorted;                                /* by key: for rank lookup */
    uint8_t **add_bytes; uint16_t *add_len; uint32_t *add_id;  /* longest-first */
    uint32_t (*L)[2]; uint32_t nL;
    uint32_t (*N)[2]; uint32_t nN;
    uint32_t (*S)[2]; uint32_t nS;
    uint16_t bmap[256];                           /* byte -> visible codepoint */
} LTok;

int ltok_load(LTok *t, const char *path);
void ltok_free(LTok *t);
/* Encode UTF-8 text (bytes, len) to ids. Returns count, or -1 on error. ids must hold
   at least (len*2 + 16) entries (fuzz-tested worst case is far below). */
int ltok_encode(const LTok *t, const uint8_t *text, size_t n, uint32_t *ids, int cap);
#endif
