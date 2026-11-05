/* lfm3_m3prompt.c -- M3 hand-off: transcript ids (from lfm3_asr --out-ids) -> speech prompt.
 *
 *   lfm3_m3prompt <tok.bin> <asr_ids.i32> <out_prompt.i32> [system_text]
 *
 * Decodes the ids with the S1 tokenizer (the same blob lfm3_asr loads), wraps the text in
 * the chat template the speech prompts use (speak_hf_oracle.py PROMPTS), and encodes the
 * whole rendered prompt back to ids. Writes int32 ids for lfm3_speak. No Python.
 *
 * This is the CASCADED speech->speech path: ASR text, then TTS. It is not native M3 (audio
 * rows fed into the speech prompt); that needs lfm3_speak to accept audio-embedding rows.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "lfm3_tok.h"

static void die(const char *m) { fprintf(stderr, "lfm3_m3prompt: %s\n", m); exit(1); }

int main(int argc, char **argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s <tok.bin> <asr_ids.i32> <out_prompt.i32> [system_text]\n", argv[0]); return 2; }
  const char *system_text = (argc > 4) ? argv[4] : "You are a helpful assistant.";

  FILE *f = fopen(argv[2], "rb");
  if (!f) die("cannot open ids");
  fseek(f, 0, SEEK_END); long nb = ftell(f); fseek(f, 0, SEEK_SET);
  if (nb <= 0 || nb % 4) die("ids file is empty or not int32");
  uint32_t *ids = malloc((size_t)nb);
  if (fread(ids, 1, (size_t)nb, f) != (size_t)nb) die("short read of ids");
  fclose(f);
  size_t nids = (size_t)nb / 4;

  LTok tok; memset(&tok, 0, sizeof tok);
  if (ltok_load(&tok, argv[1]) != 0) die("cannot load tokenizer blob");

  size_t dcap = nids * 16 + 64;
  uint8_t *txt = malloc(dcap);
  int dn = ltok_decode(&tok, ids, nids, txt, dcap);
  if (dn < 0) die("detokenize failed");
  txt[dn] = 0;

  /* rendered prompt: the same template as the speech gate prompts */
  size_t rcap = (size_t)dn + strlen(system_text) + 256;
  char *rendered = malloc(rcap);
  int rn = snprintf(rendered, rcap,
                    "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n"
                    "<|im_start|>assistant\n", system_text, (char *)txt);
  if (rn < 0 || (size_t)rn >= rcap) die("rendered prompt overflow");

  int cap = (int)((size_t)rn * 2 + 16);
  uint32_t *out = malloc((size_t)cap * 4);
  int n = ltok_encode(&tok, (const uint8_t *)rendered, (size_t)rn, out, cap);
  if (n < 0) die("encode failed");

  FILE *g = fopen(argv[3], "wb");
  if (!g) die("cannot write prompt");
  if (fwrite(out, 4, (size_t)n, g) != (size_t)n) die("short write of prompt");
  fclose(g);

  printf("M3 prompt: %zu ASR ids -> %d chars of text -> %d speech-prompt ids -> %s\n",
         nids, dn, n, argv[3]);
  printf("transcript: %s\n", (char *)txt);
  ltok_free(&tok);
  free(ids); free(txt); free(rendered); free(out);
  return 0;
}
