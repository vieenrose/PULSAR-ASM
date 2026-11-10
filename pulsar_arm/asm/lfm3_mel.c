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
#define _DEFAULT_SOURCE
#include <stdio.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
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
void lfm3_mel_power2(const float *xa, const float *win, const float *twc,
                     const float *tws, float *pwa, int n_bins, const float *xb, float *pwb);
void lfm3_mel_power4(const float *x, const float *win, const float *twc,
                     const float *tws, float *pw, int n_bins, long hop);
void lfm3_mel_matmul(const float *pw, const float *filt, float *mel, int n_mels);
void lfm3_mel_norm(const float *logmel, float *out, int n, float floor_v, float offset,
                   float divisor);

/* Frames are independent (each reads only its own n_fft samples), so they are split into
   contiguous ranges over LFM_MEL_THREADS forked workers (default 4; fork, not pthreads,
   so the build line needs no libpthread) writing into one MAP_SHARED buffer;
   every frame runs the same routines on the same inputs, so the output is byte-identical
   to the single-threaded loop. Each worker writes its frames straight into the (m, t)
   row-major output. */
typedef struct { const mel_t *m; const float *x; float *rows; long frames, t0, t1; int ok; } mel_job;
static void *mel_worker(void *a) {
    mel_job *j = a; const mel_t *m = j->m;
    float *pw = calloc(m->bins_pad, sizeof(float));               /* 208; tail stays zero */
    float *pw2 = calloc(m->bins_pad, sizeof(float));
    float *pw4 = calloc((size_t)4 * m->bins_pad, sizeof(float));   /* 4 rows of 208 */
    /* LFM_MEL_P2 (default 1): frames in pairs through lfm3_mel_power2, which reads each
       twiddle row once for both frames (bit-identical to two lfm3_mel_power calls). */
    /* 4 = groups of four through lfm3_mel_power4, the tail in pairs. */
    const int p2 = getenv("LFM_MEL_P2") ? atoi(getenv("LFM_MEL_P2")) : 4;
    float *mel = malloc(m->n_mels * sizeof(float)), *o = malloc(m->n_mels * sizeof(float));
    if (!pw || !pw2 || !pw4 || !mel || !o) { j->ok = 0; return NULL; }
    for (long t = j->t0; t < j->t1; ) {
        if (p2 == 4 && t + 3 < j->t1) {
            lfm3_mel_power4(j->x + t * m->hop, m->window, m->twc, m->tws, pw4, m->n_bins, m->hop);
            for (int u = 0; u < 4; u++, t++) {
                lfm3_mel_matmul(pw4 + (size_t)u * m->bins_pad, m->filt, mel, m->n_mels);
                for (int i = 0; i < m->n_mels; i++) mel[i] = log10f(mel[i]);
                lfm3_mel_norm(mel, o, m->n_mels, m->log_floor, m->offset, m->divisor);
                for (int mi = 0; mi < m->n_mels; mi++) j->rows[(size_t)mi * j->frames + t] = o[mi];
            }
            continue;
        }
        const int two = p2 && t + 1 < j->t1;
        if (two) lfm3_mel_power2(j->x + t * m->hop, m->window, m->twc, m->tws, pw, m->n_bins,
                                 j->x + (t + 1) * m->hop, pw2);
        else lfm3_mel_power(j->x + t * m->hop, m->window, m->twc, m->tws, pw, m->n_bins);
        for (int u = 0; u <= two; u++, t++) {
            lfm3_mel_matmul(u ? pw2 : pw, m->filt, mel, m->n_mels);
            for (int i = 0; i < m->n_mels; i++)
                mel[i] = log10f(mel[i]);      /* log10(0) = -inf; the clamp below maps it to -12 */
            lfm3_mel_norm(mel, o, m->n_mels, m->log_floor, m->offset, m->divisor);
            for (int mi = 0; mi < m->n_mels; mi++) j->rows[(size_t)mi * j->frames + t] = o[mi];
        }
    }
    free(pw); free(pw2); free(pw4); free(mel); free(o); j->ok = 1;
    return NULL;
}

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

    float *rows = mmap(NULL, (size_t)m.n_mels * frames * sizeof(float), PROT_READ | PROT_WRITE,
                       MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (rows == MAP_FAILED) { fprintf(stderr, "oom\n"); return 2; }
    int nth = getenv("LFM_MEL_THREADS") ? atoi(getenv("LFM_MEL_THREADS")) : 4;
    if (nth < 1) nth = 1;
    if (nth > 16) nth = 16;
    if (nth > frames) nth = (int)frames;
    pid_t pid[16]; mel_job jb[16];
    for (int i = 0; i < nth; i++)
        jb[i] = (mel_job){ &m, x, rows, frames, frames * i / nth, frames * (i + 1) / nth, 0 };
    for (int i = 1; i < nth; i++) {
        fflush(stdout);
        pid[i] = fork();
        if (pid[i] < 0) { fprintf(stderr, "fork\n"); return 2; }
        if (pid[i] == 0) { mel_worker(&jb[i]); _exit(jb[i].ok ? 0 : 3); }
    }
    mel_worker(&jb[0]);
    int bad = !jb[0].ok;
    for (int i = 1; i < nth; i++) {
        int st = 0;
        if (waitpid(pid[i], &st, 0) != pid[i] || !WIFEXITED(st) || WEXITSTATUS(st) != 0) bad = 1;
    }
    if (bad) { fprintf(stderr, "mel worker failed\n"); return 2; }

    FILE *o = fopen(argv[3], "wb");
    if (!o) { fprintf(stderr, "open %s\n", argv[3]); return 2; }
    /* row-major (m, t) so the gate can reshape (128, T) directly */
    fwrite(rows, sizeof(float), (size_t)m.n_mels * frames, o);
    fclose(o);
    printf("MEL n=%ld pad=%d frames=%ld mels=%d out=%s\n", n, pad, frames, m.n_mels, argv[3]);
    return 0;
}
