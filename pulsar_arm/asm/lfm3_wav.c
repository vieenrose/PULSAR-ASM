/* lfm3_wav.c -- the WAV front of the ASR path: RIFF/WAVE -> interleaved-free f32 mono.
 *
 *   lfm3_wav in.wav out.f32          (writes n float32 samples, one per frame row)
 *
 * This is the only place in the runtime that parses a container. Everything downstream
 * (lfm3_mel.c -> lfm3_enc_gpu.cu) takes raw float32, and the S2 gate validated the mel
 * driver on exactly that format, so the WAV reader is the whole remaining gap between
 * "a recording" and "audio embeddings".
 *
 * Accepted, and only these, because the reference feature extractor is defined at
 * 16 kHz and resampling here would put an unvalidated DSP step inside a gated path:
 *   - PCM 16-bit signed little-endian (format 1), scaled by 1/32768 (torch/soundfile's
 *     f32 convention: x/32768, NOT x/32767 -- checked bit-for-bit against numpy),
 *   - IEEE float32 (format 3) and WAVE_FORMAT_EXTENSIBLE (0xFFFE) whose SubFormat GUID
 *     says PCM16 or IEEE float,
 *   - sample rate exactly 16000 Hz (the model's feature extractor rate),
 *   - 1 channel, or 2 channels averaged as (l + r) * 0.5f, evaluated per sample in
 *     that exact form,
 *   - chunks in any order, junk chunks ignored (LIST/fact/bext/...), 'data' may come
 *     before or after 'fmt ', odd chunk sizes are honoured with their pad byte.
 *
 * Everything else is REFUSED with the reason on stderr and a non-zero exit. A silent
 * guess here (assume 16 kHz, linearly interpolate, take the first channel) is the kind
 * of bug that shows up as a transcript that is subtly wrong and a mel gate that is
 * green, because the mel gate never sees the resampler.
 *
 * No libc dependencies beyond stdio/stdlib: this links into the asm runtime.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WANT_RATE 16000u

static unsigned int rd_u32(const unsigned char* p) {
  return (unsigned)p[0] | ((unsigned)p[1] << 8) | ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24);
}
static unsigned int rd_u16(const unsigned char* p) { return (unsigned)p[0] | ((unsigned)p[1] << 8); }

static int rd_le(FILE* f, void* p, size_t n) { return fread(p, 1, n, f) == n ? 1 : 0; }

/* WAVEFORMATEXTENSIBLE SubFormat GUIDs, little-endian byte order as stored. */
static const unsigned char GUID_PCM16[16] = {
  0x01,0x00,0x00,0x00,0x00,0x00,0x10,0x00,0x80,0x00,0x00,0xaa,0x00,0x38,0x9b,0x71};
static const unsigned char GUID_FLOAT[16] = {
  0x03,0x00,0x00,0x00,0x00,0x00,0x10,0x00,0x80,0x00,0x00,0xaa,0x00,0x38,0x9b,0x71};

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s in.wav out.f32\n", argv[0]); return 2; }
  FILE* f = fopen(argv[1], "rb");
  if (!f) { fprintf(stderr, "lfm3_wav: cannot open %s\n", argv[1]); return 2; }

  unsigned char hdr[12];
  if (!rd_le(f, hdr, 12) || memcmp(hdr, "RIFF", 4) || memcmp(hdr + 8, "WAVE", 4)) {
    fprintf(stderr, "lfm3_wav: %s is not RIFF/WAVE\n", argv[1]); fclose(f); return 2;
  }

  unsigned format = 0, channels = 0, rate = 0, bits = 0, blockalign = 0;
  int have_fmt = 0, have_data = 0;
  unsigned data_len = 0;
  long data_off = 0;
  long file_size = 0;

  /* the RIFF size field is a lie in streaming encoders; measure the file ourselves */
  { long cur = ftell(f); fseek(f, 0, SEEK_END); file_size = ftell(f); fseek(f, cur, SEEK_SET); }

  for (;;) {
    unsigned char ch[8];
    long pos = ftell(f);
    if (fread(ch, 1, 8, f) != 8) break;
    unsigned size = rd_u32(ch + 4);
    long body = ftell(f);
    if (body + (long)size > file_size) {
      fprintf(stderr, "lfm3_wav: chunk '%c%c%c%c' at %ld claims %u bytes, file has %ld\n",
              ch[0], ch[1], ch[2], ch[3], pos, size, file_size - body);
      fclose(f); return 2;
    }
    if (!memcmp(ch, "fmt ", 4)) {
      if (size < 16) { fprintf(stderr, "lfm3_wav: fmt chunk too small (%u)\n", size); fclose(f); return 2; }
      unsigned char fmt[40];
      size_t want = size < 40 ? size : 40;
      if (fread(fmt, 1, want, f) != want) { fprintf(stderr, "lfm3_wav: short fmt\n"); fclose(f); return 2; }
      format = rd_u16(fmt); channels = rd_u16(fmt + 2); rate = rd_u32(fmt + 4);
      blockalign = rd_u16(fmt + 12); bits = rd_u16(fmt + 14);
      if (format == 0xFFFE) {
        if (size < 40 || want < 40) {
          fprintf(stderr, "lfm3_wav: WAVE_FORMAT_EXTENSIBLE needs a 40-byte fmt (got %u)\n", size);
          fclose(f); return 2;
        }
        if (memcmp(fmt + 24, GUID_PCM16, 16) == 0) format = 1;
        else if (memcmp(fmt + 24, GUID_FLOAT, 16) == 0) format = 3;
        else {
          fprintf(stderr, "lfm3_wav: extensible SubFormat is neither PCM16 nor IEEE float\n");
          fclose(f); return 2;
        }
      }
      have_fmt = 1;
    } else if (!memcmp(ch, "data", 4)) {
      have_data = 1; data_off = body; data_len = size;
    }
    /* chunks are word-aligned: an odd size carries a pad byte */
    if (fseek(f, body + (long)(size + (size & 1u)), SEEK_SET) != 0) {
      fprintf(stderr, "lfm3_wav: seek failed after chunk\n"); fclose(f); return 2;
    }
  }
  fclose(f);
  if (!have_fmt)  { fprintf(stderr, "lfm3_wav: no fmt chunk\n"); return 2; }
  if (!have_data) { fprintf(stderr, "lfm3_wav: no data chunk\n"); return 2; }

  if (rate != WANT_RATE) {
    fprintf(stderr, "lfm3_wav: sample rate %u Hz, this runtime is 16000 Hz only.\n"
                    "        Resample upstream (sox/libswresample) -- a resampler inside a\n"
                    "        gated path is DSP nobody has validated.\n", rate);
    return 3;
  }
  if (channels != 1u && channels != 2u) {
    fprintf(stderr, "lfm3_wav: %u channels; mono and stereo only (stereo is (l+r)*0.5f).\n", channels);
    return 3;
  }
  if (format == 1) { if (bits != 16) { fprintf(stderr, "lfm3_wav: PCM with %u bits; only 16-bit PCM.\n", bits); return 3; } }
  else if (format == 3) { if (bits != 32) { fprintf(stderr, "lfm3_wav: float with %u bits; only IEEE float32.\n", bits); return 3; } }
  else { fprintf(stderr, "lfm3_wav: format %u (want PCM 1, IEEE float 3, or extensible).\n", format); return 3; }

  const unsigned bytesps = bits / 8u;
  const unsigned bpf = blockalign ? blockalign : channels * bytesps;
  if (bpf == 0 || data_len % bpf) {
    fprintf(stderr, "lfm3_wav: data length %u is not a whole number of %u-byte frames\n", data_len, bpf);
    return 3;
  }
  const long nframes = (long)(data_len / bpf);

  FILE* o = fopen(argv[2], "wb");
  if (!o) { fprintf(stderr, "lfm3_wav: cannot write %s\n", argv[2]); return 2; }
  f = fopen(argv[1], "rb");
  if (!f || fseek(f, data_off, SEEK_SET) != 0) { fprintf(stderr, "lfm3_wav: reopen data failed\n"); return 2; }

  unsigned char* buf = (unsigned char*)malloc(bpf);
  if (!buf) { fprintf(stderr, "lfm3_wav: OOM\n"); return 2; }
  /* One f32 write per sample, buffered by the stdio layer: a 60 s clip is 96000
   * samples = 384 KB, so no need for a second buffer. */
  for (long i = 0; i < nframes; i++) {
    if (fread(buf, 1, bpf, f) != bpf) { fprintf(stderr, "lfm3_wav: truncated data at frame %ld\n", i); return 2; }
    float v;
    if (format == 1) {
      const short* s = (const short*)buf;
      v = (channels == 2u) ? ((float)s[0] + (float)s[1]) * 0.5f / 32768.0f
                           : (float)s[0] / 32768.0f;
    } else {
      float l, r;
      memcpy(&l, buf, 4);
      if (channels == 2u) { memcpy(&r, buf + 4, 4); v = (l + r) * 0.5f; } else v = l;
    }
    if (fwrite(&v, 4, 1, o) != 1) { fprintf(stderr, "lfm3_wav: write failed\n"); return 2; }
  }
  fclose(o); fclose(f); free(buf);
  printf("lfm3_wav: %s -> %s  rate=%u ch=%u bits=%u format=%u samples=%ld\n",
         argv[1], argv[2], rate, channels, bits, format, nframes);
  return 0;
}
