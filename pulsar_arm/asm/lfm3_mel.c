/* lfm3_mel.c -- log-mel front-end driver for the LFM3-Audio runtime (S2).

   Reads raw float32 mono PCM at 16 kHz, right-pads it exactly the way
   Lfm3AudioFeatureExtractor.__call__ does, runs the NEON kernels in
   lfm3_mel.S per frame, applies log10 (libm log10f) and the clamped affine
   tail, and writes the (128, T) float32 feature matrix the audio encoder
   consumes -- the same bytes HF's `input_features` holds.

     usage: lfm3_mel <mel.bin> <pcm.f32> <mel.f32>

   The padding rule is reproduced from the reference, not re-derived:
     pad_length(N) = 400                              for N < 400
                   = N + (-(N - 400) mod 160)         for N >= 400
     frames        = 1 + (padded - 400) / 160
   (feature_extraction_lfm3_audio.py:pad_length / num_frames).

   BUILD (Nano, gcc 9.4):
     as -o lfm3_mel_o lfm3_mel.S
     gcc -O2 -ffp-contract=off -o lfm3_mel lfm3_mel.c lfm3_mel_o -lm
   -ffp-contract=off is required: a fused a*b+c is a different real number than
   the kernel's fmul-then-fadd, and lfm3_mel_test.c asserts BIT equality with
   the kernel.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define MEL_MAGIC "LFM3MEL1"

typedef struct {
    int n_fft, hop, win, n_mels, n_bins, bins_pad;
    float log_floor, offset, divisor;
    float *window, *twc, *tws, *filt;
} mel_t;

static int rdint(FILE *f) { int v; if (fread(&v, 4, 1, f) != 1) exit(2); return v; }
static float rdf(FILE *f) { float v; if (fread(&v, 4, 1, f) != 1) exit(2); return v; }
static float *rdarr(FILE *f, size_t n) {
    float *p = malloc(n * sizeof(float));
    if (!p || fread(p, sizeof(float), n, f) != n) { fprintf(stderr, "mel.bin short\n"); exit(2); }
    return p;
}

static int mel_load(const char *path, mel_t *m) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s\n", path); return 0; }
    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, MEL_MAGIC, 8)) {
        fprintf(stderr, "%s: not a LFM3MEL1 blob\n", path); exit(2);
    }
    m->n_fft = rdint(f); m->hop = rdint(f); m->win = rdint(f);
    m->n_mels = rdint(f); m->n_bins = rdint(f); m->bins_pad = rdint(f);
    rdint(f); rdint(f);                          /* reserved */
    m->log_floor = rdf(f); m->offset = rdf(f); m->divisor = rdf(f); rdf(f);
    if (m->n_fft != 400 || m->win != 400 || m->hop != 160 || m->n_bins != 201 ||
        m->bins_pad != 208 || m->n_mels != 128) {
        fprintf(stderr, "this driver is written for n_fft=400 hop=160 bins=201(208) mels=128;"
                        " blob has %d/%d/%d(%d)/%d -- rebuild mel.bin with mkmel.py\n",
                m->n_fft, m->hop, m->n_bins, m->bins_pad, m->n_mels);
        exit(2);
    }
    m->window = rdarr(f, m->win);
    m->twc = rdarr(f, (size_t)m->n_bins * (m->n_fft / 2));
    m->tws = rdarr(f, (size_t)m->n_bins * (m->n_fft / 2));
    m->filt = rdarr(f, (size_t)m->n_mels * m->bins_pad);
    fclose(f);
    return 1;
}

void lfm3_mel_power(const float *x, const float *win, const float *twc,
                    const float *tws, float *pw, int n_bins);
void lfm3_mel_matmul(const float *pw, const float *filt, float *mel, int n_mels);
void lfm3_mel_norm(const float *logmel, float *out, int n, float floor_v, float offset,
                   float divisor);

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <mel.bin> <pcm.f32> <mel.f32>\n", argv[0]); return 2; }
    mel_t m;
    if (!mel_load(argv[1], &m)) return 2;

    FILE *f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "open %s\n", argv[2]); return 2; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f) / 4;
    fseek(f, 0, SEEK_SET);
    float *x = calloc((size_t)n + m.n_fft, sizeof(float));       /* slack = the zero pad */
    if (!x || fread(x, sizeof(float), n, f) != (size_t)n) { fprintf(stderr, "pcm read\n"); return 2; }
    fclose(f);

    /* right zero padding to a whole number of frames, exactly as the reference */
    int pad = (n < m.n_fft) ? m.n_fft
                            : (int)n + (int)(((-(n - m.n_fft) % m.hop) + m.hop) % m.hop);
    long frames = 1 + (pad - m.n_fft) / m.hop;

    float *pw = calloc(m.bins_pad, sizeof(float));                /* 208; tail stays zero */
    float *mel = malloc(m.n_mels * sizeof(float));
    float *out = malloc((size_t)m.n_mels * frames * sizeof(float));
    if (!pw || !mel || !out) { fprintf(stderr, "oom\n"); return 2; }

    for (long t = 0; t < frames; t++) {
        lfm3_mel_power(x + t * m.hop, m.window, m.twc, m.tws, pw, m.n_bins);
        lfm3_mel_matmul(pw, m.filt, mel, m.n_mels);
        for (int i = 0; i < m.n_mels; i++)
            mel[i] = log10f(mel[i]);          /* log10(0) = -inf; the clamp below maps it to -12 */
        lfm3_mel_norm(mel, out + t * m.n_mels, m.n_mels, m.log_floor, m.offset, m.divisor);
    }

    FILE *o = fopen(argv[3], "wb");
    if (!o) { fprintf(stderr, "open %s\n", argv[3]); return 2; }
    /* row-major (m, t) so the gate can reshape (128, T) directly */
    float *rows = malloc((size_t)m.n_mels * frames * sizeof(float));
    for (int mi = 0; mi < m.n_mels; mi++)
        for (long t = 0; t < frames; t++) rows[(size_t)mi * frames + t] = out[t * m.n_mels + mi];
    fwrite(rows, sizeof(float), (size_t)m.n_mels * frames, o);
    fclose(o);
    printf("MEL n=%ld pad=%d frames=%ld mels=%d out=%s\n", n, pad, frames, m.n_mels, argv[3]);
    return 0;
}
