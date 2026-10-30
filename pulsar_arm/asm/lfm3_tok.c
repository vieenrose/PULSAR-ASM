/* lfm3_tok.c -- S1 BPE port. The algorithm was pinned by mktok.py's Python replay
   (531/531 strings id-for-id vs HF) before any C was written; this file reproduces
   exactly that replay. The risky part is the Split regex, so its structure is
   documented alternative by alternative below -- see plan section 12 for why
   tokenizer_config.json's pre_tokenizer:null is wrong and what ignore_merges does.

   Split pattern (verbatim from tokenizer.json):
     (?i:'s|'t|'re|'ve|'m|'ll|'d)
     |[^\\r\\n\\p{L}\\p{N}]?\\p{L}+
     |\\p{N}{1,3}
     | ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*
     |\\s*[\\r\\n]+
     |\\s+(?!\\S)
     |\\s+
   Semantics = regex.finditer left-to-right, first alternative that matches anchored
   at each position wins (greedy quantifiers). Every input position matches something:
   L via A2, N via A3, symbols via A4, newlines via A5, spaces via A6/A7.
*/
#include "lfm3_tok.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int cmp_v(const void *a, const void *b) {
    const TokV *x = a, *y = b;
    size_t n = x->len < y->len ? x->len : y->len;
    int c = memcmp(x->bytes, y->bytes, n);
    return c ? c : (int)x->len - (int)y->len;
}
static int cmp_m(const void *a, const void *b) {
    const TokM *x = a, *y = b;
    size_t n = x->klen < y->klen ? x->klen : y->klen;
    int c = memcmp(x->key, y->key, n);
    return c ? c : (int)x->klen - (int)y->klen;
}

int ltok_load(LTok *t, const char *path) {
    memset(t, 0, sizeof *t);
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "LFM3TOK1", 8)) { fclose(f); return -1; }
    uint32_t jl = 0;
    if (fread(&jl, 4, 1, f) != 1) { fclose(f); return -1; }
    char *js = malloc(jl + 1);
    if (!js || fread(js, 1, jl, f) != jl) { free(js); fclose(f); return -1; }
    js[jl] = 0;
    /* header counts are re-derived from the sections; the JSON is informational */
    free(js);
    uint32_t n;
    /* vocab by id */
    if (fread(&n, 4, 1, f) != 1) { fclose(f); return -1; }
    t->n_vocab = n;
    t->vbytes = calloc(n, sizeof(uint8_t*)); t->vlen = calloc(n, sizeof(uint16_t));
    t->vsorted = calloc(n, sizeof(TokV));
    for (uint32_t i = 0; i < n; i++) {
        uint16_t l; if (fread(&l, 2, 1, f) != 1) { fclose(f); return -1; }
        uint8_t *b = malloc(l ? l : 1);
        if (l && fread(b, 1, l, f) != l) { fclose(f); return -1; }
        t->vbytes[i] = b; t->vlen[i] = l;
        t->vsorted[i].bytes = b; t->vsorted[i].len = l; t->vsorted[i].id = i;
    }
    qsort(t->vsorted, n, sizeof(TokV), cmp_v);
    /* merges in rank order */
    if (fread(&n, 4, 1, f) != 1) { fclose(f); return -1; }
    t->n_merges = n;
    uint8_t **ml = calloc(n, sizeof(uint8_t*)), **mr = calloc(n, sizeof(uint8_t*));
    uint16_t *mll = calloc(n, sizeof(uint16_t)), *mrl = calloc(n, sizeof(uint16_t));
    for (uint32_t i = 0; i < n; i++) {
        uint16_t al, bl;
        if (fread(&al, 2, 1, f) != 1) { fclose(f); return -1; }
        ml[i] = malloc(al ? al : 1); mll[i] = al;
        if (al && fread(ml[i], 1, al, f) != al) { fclose(f); return -1; }
        if (fread(&bl, 2, 1, f) != 1) { fclose(f); return -1; }
        mr[i] = malloc(bl ? bl : 1); mrl[i] = bl;
        if (bl && fread(mr[i], 1, bl, f) != bl) { fclose(f); return -1; }
    }
    t->msorted = calloc(n, sizeof(TokM));
    for (uint32_t i = 0; i < n; i++) {
        uint32_t kl = (uint32_t)mll[i] + 1 + mrl[i];
        uint8_t *k = malloc(kl);
        memcpy(k, ml[i], mll[i]); k[mll[i]] = 0; memcpy(k + mll[i] + 1, mr[i], mrl[i]);
        t->msorted[i].key = k; t->msorted[i].klen = kl; t->msorted[i].rank = i;
        free(ml[i]); free(mr[i]);
    }
    free(ml); free(mr); free(mll); free(mrl);
    qsort(t->msorted, n, sizeof(TokM), cmp_m);
    /* added, longest-first for raw-byte matching */
    if (fread(&n, 4, 1, f) != 1) { fclose(f); return -1; }
    t->n_added = n;
    t->add_bytes = calloc(n, sizeof(uint8_t*)); t->add_len = calloc(n, sizeof(uint16_t));
    t->add_id = calloc(n, sizeof(uint32_t));
    for (uint32_t i = 0; i < n; i++) {
        uint32_t id; uint16_t l;
        if (fread(&id, 4, 1, f) != 1 || fread(&l, 2, 1, f) != 1) { fclose(f); return -1; }
        uint8_t *b = malloc(l ? l : 1);
        if (l && fread(b, 1, l, f) != l) { fclose(f); return -1; }
        t->add_bytes[i] = b; t->add_len[i] = l; t->add_id[i] = id;
    }
    /* sort added by length descending (blob is by id; matching needs longest-first) */
    for (uint32_t i = 0; i < n; i++)
        for (uint32_t j = i + 1; j < n; j++)
            if (t->add_len[j] > t->add_len[i]) {
                uint8_t *tb = t->add_bytes[i]; t->add_bytes[i] = t->add_bytes[j]; t->add_bytes[j] = tb;
                uint16_t tl = t->add_len[i]; t->add_len[i] = t->add_len[j]; t->add_len[j] = tl;
                uint32_t ti = t->add_id[i]; t->add_id[i] = t->add_id[j]; t->add_id[j] = ti;
            }
    uint32_t nn;
#define RD_RNG(dst, cnt) do { if (fread(&nn, 4, 1, f) != 1) { fclose(f); return -1; } \
        (cnt) = nn; (dst) = malloc(nn * sizeof(uint32_t[2])); \
        for (uint32_t i = 0; i < nn; i++) \
            if (fread((dst)[i], 4, 2, f) != 2) { fclose(f); return -1; } } while (0)
    RD_RNG(t->L, t->nL); RD_RNG(t->N, t->nN); RD_RNG(t->S, t->nS);
#undef RD_RNG
    /* byte map: rebuild the canonical ByteLevel alphabet exactly like mktok.py, so
       the blob cannot disagree with the code (bijectivity is asserted by the gate) */
    {
        int bs[256], cs[256], nb = 0;
        for (int c = 0x21; c <= 0x7E; c++) { bs[nb] = c; cs[nb] = c; nb++; }
        for (int c = 0xA1; c <= 0xAC; c++) { bs[nb] = c; cs[nb] = c; nb++; }
        for (int c = 0xAE; c <= 0xFF; c++) { bs[nb] = c; cs[nb] = c; nb++; }
        int m = 0;
        for (int b = 0; b < 256; b++) {
            int f = 0;
            for (int i = 0; i < nb; i++) if (bs[i] == b) { f = 1; break; }
            if (!f) { bs[nb] = b; cs[nb] = 256 + m; nb++; m++; }
        }
        for (int i = 0; i < 256; i++) t->bmap[bs[i]] = (uint16_t)cs[i];
    }
    fclose(f);
    return 0;
}

void ltok_free(LTok *t) {
    if (!t) return;
    for (uint32_t i = 0; i < t->n_vocab; i++) free(t->vbytes[i]);
    free(t->vbytes); free(t->vlen); free(t->vsorted);
    for (uint32_t i = 0; i < t->n_merges; i++) free(t->msorted[i].key);
    free(t->msorted);
    for (uint32_t i = 0; i < t->n_added; i++) free(t->add_bytes[i]);
    free(t->add_bytes); free(t->add_len); free(t->add_id);
    free(t->L); free(t->N); free(t->S);
    memset(t, 0, sizeof *t);
}

/* ---- codepoint helpers ---- */
static int in_rng(uint32_t (*R)[2], uint32_t n, uint32_t cp) {
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (cp < R[mid][0]) hi = mid;
        else if (cp > R[mid][1]) lo = mid + 1;
        else return 1;
    }
    return 0;
}
static int is_L(const LTok *t, uint32_t cp) { return in_rng(t->L, t->nL, cp); }
static int is_N(const LTok *t, uint32_t cp) { return in_rng(t->N, t->nN, cp); }
static int is_S(const LTok *t, uint32_t cp) { return in_rng(t->S, t->nS, cp); }

static int utf8_dec(const uint8_t *s, size_t n, uint32_t *cp) {
    if (!n) return 0;
    uint8_t a = s[0];
    if (a < 0x80) { *cp = a; return 1; }
    if ((a & 0xE0) == 0xC0) {
        if (n < 2) return 0;
        *cp = ((uint32_t)(a & 0x1F) << 6) | (s[1] & 0x3F); return 2;
    }
    if ((a & 0xF0) == 0xE0) {
        if (n < 3) return 0;
        *cp = ((uint32_t)(a & 0x0F) << 12) | ((uint32_t)(s[1] & 0x3F) << 6) | (s[2] & 0x3F); return 3;
    }
    if ((a & 0xF8) == 0xF0) {
        if (n < 4) return 0;
        *cp = ((uint32_t)(a & 0x07) << 18) | ((uint32_t)(s[1] & 0x3F) << 12) |
              ((uint32_t)(s[2] & 0x3F) << 6) | (s[3] & 0x3F); return 4;
    }
    return 0;
}
static int utf8_enc(uint32_t cp, uint8_t *o) {
    if (cp < 0x80) { o[0] = (uint8_t)cp; return 1; }
    if (cp < 0x800) { o[0] = (uint8_t)(0xC0 | (cp >> 6)); o[1] = (uint8_t)(0x80 | (cp & 0x3F)); return 2; }
    o[0] = (uint8_t)(0xE0 | (cp >> 12)); o[1] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
    o[2] = (uint8_t)(0x80 | (cp & 0x3F)); return 3;
}
static uint32_t ascii_low(uint32_t cp) { return (cp >= 'A' && cp <= 'Z') ? cp + 32 : cp; }

static int32_t vocab_id(const LTok *t, const uint8_t *b, size_t n) {
    if (n > 0xFFFF) return -1;
    TokV k; k.bytes = (uint8_t*)b; k.len = (uint16_t)n;
    TokV *f = bsearch(&k, t->vsorted, t->n_vocab, sizeof(TokV), cmp_v);
    return f ? (int32_t)f->id : -1;
}
static int32_t merge_rank(const LTok *t, const uint8_t *l, size_t ln, const uint8_t *r, size_t rn) {
    /* key on stack: merges keys are <= ~1000 bytes; pieces here are small */
    if (ln + 1 + rn > 2048) return -1;
    uint8_t k[2048];
    memcpy(k, l, ln); k[ln] = 0; memcpy(k + ln + 1, r, rn);
    TokM key; key.key = k; key.klen = (uint32_t)(ln + 1 + rn);
    TokM *f = bsearch(&key, t->msorted, t->n_merges, sizeof(TokM), cmp_m);
    return f ? (int32_t)f->rank : -1;
}

typedef struct { const LTok *t; uint32_t *ids; int n, cap; int err; } EncSt;

/* BPE a visible key (UTF-8, n bytes) to ids */
static void bpe_key(const LTok *t, const uint8_t *key, size_t n, EncSt *e) {
    int32_t id = vocab_id(t, key, n);
    if (id >= 0) {   /* ignore_merges: whole pre-token already in vocab */
        if (e->n < e->cap) e->ids[e->n++] = (uint32_t)id; else e->err = 1;
        return;
    }
    /* split into single-visible-char symbols: need codepoint boundaries of key.
       key is UTF-8 of visible codepoints (each 1-3 bytes); decode to get starts. */
    size_t scap = 64, sn = 0;
    const uint8_t **ss = malloc(scap * sizeof *ss);
    size_t *sl = malloc(scap * sizeof *sl);
    for (size_t p = 0; p < n;) {
        uint32_t cp; int bl = utf8_dec(key + p, n - p, &cp);
        if (!bl) { e->err = 1; break; }
        if (sn == scap) { scap *= 2; ss = realloc(ss, scap * sizeof *ss); sl = realloc(sl, scap * sizeof *sl); }
        ss[sn] = key + p; sl[sn] = (size_t)bl; sn++;
        p += (size_t)bl;
    }
    while (sn > 1 && !e->err) {
        int32_t best = -1; size_t bi = 0;
        for (size_t i = 0; i + 1 < sn; i++) {
            int32_t r = merge_rank(t, ss[i], sl[i], ss[i + 1], sl[i + 1]);
            if (r >= 0 && (best < 0 || r < best)) { best = r; bi = i; }
        }
        if (best < 0) break;
        size_t nl = sl[bi] + sl[bi + 1];
        uint8_t *nb = malloc(nl);
        memcpy(nb, ss[bi], sl[bi]); memcpy(nb + sl[bi], ss[bi + 1], sl[bi + 1]);
        /* free merged children if owned (owned = not pointing into key) */
        if (ss[bi] < key || ss[bi] >= key + n) free((void*)ss[bi]);
        if (ss[bi + 1] < key || ss[bi + 1] >= key + n) free((void*)ss[bi + 1]);
        ss[bi] = nb; sl[bi] = nl;
        memmove(ss + bi + 1, ss + bi + 2, (sn - bi - 2) * sizeof *ss);
        memmove(sl + bi + 1, sl + bi + 2, (sn - bi - 2) * sizeof *sl);
        sn--;
    }
    for (size_t i = 0; i < sn && !e->err; i++) {
        int32_t v = vocab_id(t, ss[i], sl[i]);
        if (v < 0) { e->err = 1; break; }
        if (e->n < e->cap) e->ids[e->n++] = (uint32_t)v; else e->err = 1;
        if (ss[i] < key || ss[i] >= key + n) free((void*)ss[i]);
    }
    free(ss); free(sl);
}

int ltok_encode(const LTok *t, const uint8_t *text, size_t n, uint32_t *ids, int cap) {
    /* decode to codepoints with byte offsets */
    size_t ccap = n + 1, cn = 0;
    uint32_t *cps = malloc(ccap * sizeof *cps);
    size_t *boff = malloc((ccap + 1) * sizeof *boff);
    for (size_t p = 0; p < n;) {
        uint32_t cp; int bl = utf8_dec(text + p, n - p, &cp);
        if (!bl) { free(cps); free(boff); return -1; }
        cps[cn] = cp; boff[cn] = p; cn++;
        p += (size_t)bl;
    }
    boff[cn] = n;
    EncSt e; e.t = t; e.ids = ids; e.n = 0; e.cap = cap; e.err = 0;
    /* walk raw bytes for added tokens (all ASCII: byte match = exact) */
    size_t pos = 0;
    /* codepoint index of pos */
    size_t ci = 0;
    while (pos < n && !e.err) {
        /* longest added match at pos */
        uint32_t amatch = 0; size_t amlen = 0;
        for (uint32_t a = 0; a < t->n_added; a++) {
            size_t l = t->add_len[a];
            if (l <= n - pos && memcmp(text + pos, t->add_bytes[a], l) == 0) {
                amatch = t->add_id[a]; amlen = l; break;   /* sorted longest-first */
            }
        }
        if (amlen) {
            if (e.n < cap) ids[e.n++] = amatch; else e.err = 1;
            while (ci < cn && boff[ci] < pos + amlen) ci++;
            pos += amlen;
            continue;
        }
        /* next added occurrence (byte search) */
        size_t next = n; 
        for (uint32_t a = 0; a < t->n_added && next == n; a++) {
            /* naive search is fine: 140 short ASCII needles, inputs are short */
            for (size_t p = pos; p + t->add_len[a] <= n; p++) {
                if (memcmp(text + p, t->add_bytes[a], t->add_len[a]) == 0) {
                    if (p < next) next = p;
                    break;
                }
            }
        }
        size_t cend = ci;
        while (cend < cn && boff[cend] < next) cend++;
        /* split codepoints [ci, cend): hand-rolled Split regex, anchored at each
           position, first matching alternative wins (see file header) */
        size_t pcap = 16, pn = 0;
        size_t *ps0 = malloc(pcap * sizeof *ps0), *ps1 = malloc(pcap * sizeof *ps1);
        {
            size_t i = ci;
            while (i < cend && !e.err) {
                size_t adv = 0;
                uint32_t c = cps[i];
                if (c == 39 && i + 1 < cend)  /* '\'' */ {
                    uint32_t d = ascii_low(cps[i + 1]);
                    if (d == 's' || d == 't' || d == 'm' || d == 'd') adv = 2;
                    else if (i + 2 < cend) {
                        uint32_t e1 = ascii_low(cps[i + 1]), e2 = ascii_low(cps[i + 2]);
                        if ((e1 == 'r' && e2 == 'e') || (e1 == 'v' && e2 == 'e') ||
                            (e1 == 'l' && e2 == 'l')) adv = 3;
                    }
                }
                if (!adv) {
                    size_t p = i;
                    if (c != 0x0D && c != 0x0A && !is_L(t, c) && !is_N(t, c)) p++;
                    size_t q = p;
                    while (q < cend && is_L(t, cps[q])) q++;
                    if (q > p) adv = q - i;
                }
                if (!adv && is_N(t, c)) {
                    size_t q = i;
                    while (q < cend && q < i + 3 && is_N(t, cps[q])) q++;
                    adv = q - i;
                }
                if (!adv) {
                    size_t p = (c == 0x20) ? i + 1 : i;
                    size_t q = p;
                    while (q < cend && !is_S(t, cps[q]) && !is_L(t, cps[q]) && !is_N(t, cps[q])) q++;
                    if (q > p) {
                        size_t r = q;
                        while (r < cend && (cps[r] == 0x0D || cps[r] == 0x0A)) r++;
                        adv = r - i;
                    }
                }
                if (!adv) {
                    size_t j = i, last = 0, have = 0;
                    while (j < cend && is_S(t, cps[j])) {
                        if (cps[j] == 0x0D || cps[j] == 0x0A) { have = 1; last = j; }
                        j++;
                    }
                    if (have) adv = last + 1 - i;
                }
                /* A6: \\s+ (?!\\S). Greedy \\s+ takes the LONGEST whitespace prefix
                   whose next char is not non-space. Inside a maximal spaces-only
                   run of length L (A5 already claimed any run with CR/LF), any
                   proper prefix qualifies (next is space); the full run qualifies
                   only at end of input. So: trailing -> all L; followed by
                   non-space -> L-1 (which fails for L==1, falling to A7); that
                   leftover space is what lets '  b' split as ' '+' b' and single
                   space before a digit fall to A7. Verified against Python regex:
                   'a  b' -> ['a',' ',' b'], 'a  1' -> ['a',' ',' ','1']. */
                if (!adv) {
                    size_t j = i;
                    while (j < cend && is_S(t, cps[j]) && cps[j] != 0x0D && cps[j] != 0x0A) j++;
                    if (j > i) {
                        if (j == cn) adv = j - i;             /* true end: trailing */
                        else if (j - i >= 2) adv = j - i - 1; /* leave one space */
                    }
                }
                if (!adv) {
                    size_t j = i;
                    while (j < cend && is_S(t, cps[j])) j++;
                    if (j > i) adv = j - i;
                }
                if (!adv) adv = 1;
                if (pn == pcap) { pcap *= 2; ps0 = realloc(ps0, pcap * sizeof *ps0); ps1 = realloc(ps1, pcap * sizeof *ps1); }
                size_t bs = boff[i], be = boff[i + adv];
                ps0[pn] = bs; ps1[pn] = be - bs; pn++;
                i += adv;
            }
        }
        for (size_t k = 0; k < pn && !e.err; k++) {
            /* byte-level: UTF-8 bytes of piece -> visible codepoints -> UTF-8 key */
            const uint8_t *pb = text + ps0[k];
            size_t pl = ps1[k];
            uint8_t key[4096]; size_t kn = 0;
            for (size_t b = 0; b < pl; b++) {
                uint8_t tmp[3]; int bl = utf8_enc(t->bmap[pb[b]], tmp);
                if (kn + (size_t)bl > sizeof key) { e.err = 1; break; }
                memcpy(key + kn, tmp, (size_t)bl); kn += (size_t)bl;
            }
            if (!e.err) {
                /* stage through temp ids to keep pairs (start,len) out of output */
                int save = e.n;
                bpe_key(t, key, kn, &e);
                (void)save;
            }
        }
        free(ps0); free(ps1);
        pos = next;
        while (ci < cn && boff[ci] < pos) ci++;
    }
    free(cps); free(boff);
    if (e.err) return -1;
    return e.n;
}
