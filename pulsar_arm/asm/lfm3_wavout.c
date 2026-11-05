/* lfm3_wavout.c -- f32 mono PCM -> RIFF/WAVE (16-bit or float32). No libraries.
 *
 *   lfm3_wavout <in.f32> <out.wav> [rate] [fmt]      fmt: s16 (default) | f32
 *
 * The codec writes raw f32 samples at 24 kHz; a WAV wrapper is the difference between
 * "a file" and something a person can play. Input must be a whole number of samples;
 * non-finite input is refused rather than written (a NaN in a WAV plays as a click and
 * then reads as a mystery downstream).
 *
 * s16 uses round-half-away-from-zero on a 32767 scale with a hard clip: the codec's own
 * iSTFT output is bounded well inside [-1,1] (measured peak 0.61), so the clip is a guard,
 * not a normal path -- and it is counted and reported.
 *
 * Build: gcc -O2 -o lfm3_wavout lfm3_wavout.c -lm   (sqrt for the rms line; without
 * -lm the link fails with "undefined reference to sqrt" -- the first board build did).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static void w32(FILE *f, uint32_t v) { fputc(v & 0xff, f); fputc((v >> 8) & 0xff, f); fputc((v >> 16) & 0xff, f); fputc((v >> 24) & 0xff, f); }
static void w16(FILE *f, uint16_t v) { fputc(v & 0xff, f); fputc((v >> 8) & 0xff, f); }

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <in.f32> <out.wav> [rate] [s16|f32]\n", argv[0]);
    return 2;
  }
  const char *in = argv[1], *out = argv[2];
  int rate = (argc > 3) ? atoi(argv[3]) : 24000;
  int is_f32 = (argc > 4) && !strcmp(argv[4], "f32");
  if (rate < 4000 || rate > 192000) { fprintf(stderr, "lfm3_wavout: rate %d out of range\n", rate); return 2; }

  FILE *fi = fopen(in, "rb");
  if (!fi) { fprintf(stderr, "lfm3_wavout: cannot open %s\n", in); return 2; }
  fseek(fi, 0, SEEK_END); long bytes = ftell(fi); fseek(fi, 0, SEEK_SET);
  if (bytes <= 0 || bytes % 4) { fprintf(stderr, "lfm3_wavout: %s is %ld bytes, not f32 samples\n", in, bytes); fclose(fi); return 2; }
  long n = bytes / 4;
  float *pcm = malloc((size_t)bytes);
  if (!pcm) { fprintf(stderr, "lfm3_wavout: out of memory\n"); fclose(fi); return 2; }
  if (fread(pcm, 4, (size_t)n, fi) != (size_t)n) { fprintf(stderr, "lfm3_wavout: short read\n"); fclose(fi); free(pcm); return 2; }
  fclose(fi);

  double peak = 0, sumsq = 0;
  long clipped = 0, quiet = 0;
  for (long i = 0; i < n; i++) {
    if (!isfinite(pcm[i])) { fprintf(stderr, "lfm3_wavout: sample %ld is not finite (%g) -- refusing to write\n", i, pcm[i]); free(pcm); return 2; }
    double a = fabs((double)pcm[i]);
    if (a > peak) peak = a;
    sumsq += (double)pcm[i] * pcm[i];
  }
  /* nothing to say about an all-zero file except that it IS all zero: report, do not hide */
  for (long i = 0; i < n; i++) if (pcm[i] == 0.0f) quiet++;

  int bits = is_f32 ? 32 : 16;
  int channels = 1;
  uint32_t data_bytes = (uint32_t)(n * (bits / 8));
  uint16_t fmt_tag = is_f32 ? 3 : 1;   /* 3 = IEEE float, 1 = PCM */
  uint32_t byte_rate = (uint32_t)rate * channels * (bits / 8);
  uint16_t block_align = (uint16_t)(channels * (bits / 8));

  FILE *fo = fopen(out, "wb");
  if (!fo) { fprintf(stderr, "lfm3_wavout: cannot write %s\n", out); free(pcm); return 2; }
  fwrite("RIFF", 1, 4, fo); w32(fo, 36 + data_bytes); fwrite("WAVE", 1, 4, fo);
  fwrite("fmt ", 1, 4, fo); w32(fo, 16); w16(fo, fmt_tag); w16(fo, (uint16_t)channels);
  w32(fo, (uint32_t)rate); w32(fo, byte_rate); w16(fo, block_align); w16(fo, (uint16_t)bits);
  fwrite("data", 1, 4, fo); w32(fo, data_bytes);
  if (is_f32) {
    if (fwrite(pcm, 4, (size_t)n, fo) != (size_t)n) { fprintf(stderr, "lfm3_wavout: short write\n"); fclose(fo); free(pcm); return 2; }
  } else {
    for (long i = 0; i < n; i++) {
      double v = (double)pcm[i] * 32767.0;
      long s;
      if (v > 32767.0) { s = 32767; clipped++; }
      else if (v < -32768.0) { s = -32768; clipped++; }
      else s = (long)(v >= 0 ? v + 0.5 : v - 0.5);
      w16(fo, (uint16_t)(int16_t)s);
    }
  }
  if (fclose(fo) != 0) { fprintf(stderr, "lfm3_wavout: close failed on %s\n", out); free(pcm); return 2; }
  free(pcm);

  printf("WAVOUT %s: %ld samples %.3f s @ %d Hz %s, peak %.4f rms %.4f, clipped %ld, zeros %ld\n",
         out, n, (double)n / rate, rate, is_f32 ? "f32" : "s16", peak,
         sqrt(sumsq / (double)(n ? n : 1)), clipped, quiet);
  if (peak == 0.0) printf("WAVOUT WARNING: signal is all zeros -- the codes decoded to silence\n");
  fflush(stdout);
  return 0;
}
