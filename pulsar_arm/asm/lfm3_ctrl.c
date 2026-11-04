/* lfm3_ctrl.c -- S6 mode-machine checker for interleaved speech output.

   Replays HF's Lfm3AudioForConditionalGeneration._generate slot rules (interleaved
   mode, one row) against a slot trace recorded from HF's own generate():

       T <id>      a text token (id from the text vocabulary)
       A <eos>     a speech frame; eos = 1 if the frame carries the end-of-audio code

   The machine state is HF's: mode (INTERLEAVED | TEXT), speaking, text_done, slots_left.
   Each slot must be the kind the machine expects at that point; a speech slot where the
   machine wants text (or the reverse) fails the gate, as does a trace that does not end
   with <|im_end|> after the EOS frame.

   The sampler (which token / code is chosen) is NOT replayed here: the trace supplies the
   choices, and this checks only that the choices are legal under the mode rules and that
   the rule-driven transitions (block counts, text_end, EOS) happen where HF makes them.

   usage: lfm3_ctrl <slots.txt> [n_text=6] [n_audio=12]
   constants from config.json: interleaved_n_text 6, interleaved_n_audio 12;
   token ids: text_end 125014, im_end 124900, mixed_start 125015. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEXT_END 125014
#define IM_END 124900

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <slots.txt> [n_text] [n_audio]\n", argv[0]); return 2; }
  int n_text = argc > 2 ? atoi(argv[2]) : 6;
  int n_audio = argc > 3 ? atoi(argv[3]) : 12;
  FILE *f = fopen(argv[1], "r");
  if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }

  /* machine state, HF names */
  int interleaving = 1;      /* modes[row] is INTERLEAVED */
  int speaking = 0;          /* speaking[row] */
  int text_done = 0;         /* text_done[row] */
  int slots_left = n_text;   /* slots_left[row] */
  int finished = 0;
  int slot = 0, n_a = 0, n_t = 0, eos_at = -1;
  char line[256];

  while (fgets(line, sizeof line, f)) {
    char kind;
    long v;
    if (sscanf(line, "%c %ld", &kind, &v) != 2) { printf("FAIL slot %d: unparsable '%s'", slot, line); return 1; }
    if (finished) { printf("FAIL slot %d: slot after <|im_end|>\n", slot); return 1; }

    if (kind == 'A') {
      /* a speech slot is legal only while the machine is speaking inside an interleaved block */
      if (!interleaving || !speaking) {
        printf("FAIL slot %d: speech frame while the machine wants text (interleaving=%d speaking=%d)\n",
               slot, interleaving, speaking);
        return 1;
      }
      n_a++;
      if (v == 1) {               /* EOS frame: HF sets the mode back to TEXT */
        eos_at = slot;
        interleaving = 0;
      }
    } else if (kind == 'T') {
      if (interleaving && speaking && !text_done) {
        /* speaking only flips with slots_left, so this state is reached only at a block end.
           HF samples a speech frame here, never text. */
        printf("FAIL slot %d: text token %ld where the machine wants a speech frame\n", slot, v);
        return 1;
      }
      if (interleaving && speaking && text_done) {
        /* the audio tail after <|text_end|>: HF samples only frames here */
        printf("FAIL slot %d: text token %ld in the audio tail after <|text_end|>\n", slot, v);
        return 1;
      }
      n_t++;
      if (!interleaving || (!speaking)) {
        /* text is legal in TEXT mode and inside a text block. <|im_end|> ends the turn anywhere. */
      }
      if (v == IM_END) { finished = 1; }
      else if (interleaving && v == TEXT_END) {
        text_done = 1;
        speaking = 1;
      }
    } else {
      printf("FAIL slot %d: unknown kind '%c'\n", slot, kind);
      return 1;
    }

    /* HF: a slot in an interleaved block, before text_done, counts down; at 0 it flips */
    if (interleaving && !text_done) {
      slots_left--;
      if (slots_left == 0) {
        speaking = !speaking;
        slots_left = speaking ? n_audio : n_text;
      }
    }
    slot++;
  }
  fclose(f);

  if (!finished) { printf("FAIL: trace ended without <|im_end|>\n"); return 1; }
  if (n_a == 0) { printf("FAIL: no speech frames in the trace\n"); return 1; }
  if (eos_at < 0) { printf("FAIL: no end-of-audio frame in the trace\n"); return 1; }
  printf("PASS slots=%d text=%d audio=%d eos_at_slot=%d (interleaved machine, n_text=%d n_audio=%d)\n",
         slot, n_t, n_a, eos_at, n_text, n_audio);
  return 0;
}
