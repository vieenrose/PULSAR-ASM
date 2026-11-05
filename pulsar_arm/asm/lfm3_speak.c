/* lfm3_speak.c -- M2 shell: text prompt in, 24 kHz speech out. No Python, no HF, no /bin/sh.
 *
 *   lfm3_speak <dir> <prompt_ids.i32> <mode> <out_prefix> [max_slots] [order_dir] [seed]
 *
 * <dir> holds the gated pieces: lfm3_nanode lfm3_depth lfm3_codec (executables) plus
 *   lfm3-audio-2b-q8.lfm3q8 lfm3-audio-2b-depth.lfm3q8 lfm3-audio-2b-codec.lfm3q8
 * <prompt_ids.i32> is the raw prompt (the mode marker is appended here, not handed in:
 *   audio -> <|audio_start|> 125012, interleaved -> <|mixed_start|> 125015, text -> none).
 * <mode> is audio | interleaved | text.
 * [order_dir] prescribes the MaskGIT reveal order per frame (order.<f>.i32, 12 ints):
 *   the GATED path -- the oracle's recorded orders, so the live stream is comparable.
 *   Without it the shell draws its own randperm per frame (seeded by [seed], default
 *   1234): a free run, whose codes legitimately differ from any HF stream.
 * [max_slots] caps the slot loop (default 150).
 *
 * Slot loop (one row, HF _generate rules -- see lfm3_ctrl.c, which gates the mode
 * machine on recorded traces):
 *   - the CONTROLLER decides the slot kind (text block 6 / audio block 12 /
 *     <|text_end|> starts the audio tail / EOS frame returns to text /
 *     <|im_end|> ends the turn). Sampling alone would emit <|text_end|> where the
 *     interleaved machine demands audio (measured: greedy argmax at the 6th text
 *     slot is 125014, not audio) -- the mode machine is not optional.
 *   - text slot: backbone forward over the whole prefix (one nanode launch, greedy
 *     argmax of the last logits row; HF runs do_sample=False with text penalty 1.0,
 *     a no-op, so raw argmax is the comparison).
 *   - audio slot: backbone forward, last hidden row -> depth --maskgit with the
 *     frame's order and the previous frames as penalty history (1.1, window 4096 =
 *     HF's audio_repetition_penalty/window) -> 12 codes; any code == 2016 (audio EOS)
 *     overrides the WHOLE frame to 2016 before feedback (HF's rule, found when the
 *     raw-vs-returned mismatch gave cos 0.529) -> depth --embeds row appended to the
 *     frame-embedding file that the NEXT backbone launch consumes at the audio_out
 *     placeholder (the --speak-emb path gate_speak.py certifies).
 * Every stage is the gated binary, run by execv; outputs are size-checked before use.
 * Cost: one nanode launch per slot (full prefix prefill each time -- the blob stays
 * in page cache after the first launch, so relaunches cost seconds, not the load).
 * A resident backbone (one process, KV kept) is the S7 optimization; this shell proves
 * the loop first.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <stdarg.h>
#include <time.h>

/* token ids (model config + frames.meta.json; nanode's --speak-tok defaults match) */
#define AUDIO_START 125012
#define MIXED_START 125015
#define AUDIO_OUT 125032
#define TEXT_END 125014
#define IM_END 124900
#define AUDIO_EOS 2016
#define NCB 12
#define CBS 2017
#define HDIM 2048
#define NTEXT 6
#define NAUDIO 12

static char DIR[1024];
static char TMPD[1024];
static int stage_no = 0;

static void die(const char *fmt, ...) __attribute__((noreturn));
static void die(const char *fmt, ...) {
  va_list ap; va_start(ap, fmt);
  fprintf(stderr, "lfm3_speak: "); vfprintf(stderr, fmt, ap); fputc('\n', stderr); va_end(ap);
  exit(2);
}
static void* xmalloc(size_t n) { void *p = malloc(n); if (!p) die("out of memory (%zu bytes)", n); return p; }
static void path_of(char *dst, size_t dn, const char *name) { snprintf(dst, dn, "%s/%s", DIR, name); }
static long file_size(const char *p) {
  struct stat st; if (stat(p, &st) != 0) return -1; return (long)st.st_size;
}
static void need_file(const char *p, long min_bytes, const char *what) {
  long n = file_size(p);
  if (n < min_bytes) die("%s produced %ld bytes at %s (expected >= %ld)", what, n, p, min_bytes);
}
static void stage(const char *what) { printf("STAGE %d  %s\n", ++stage_no, what); fflush(stdout); }
static void run_cap(char *const argv[], const char *out) {
  pid_t pid = fork();
  if (pid < 0) die("fork failed");
  if (pid == 0) {
    int fd = open(out, O_WRONLY|O_CREAT|O_TRUNC, 0644);
    if (fd < 0) { fprintf(stderr, "lfm3_speak: cannot write %s\n", out); _exit(127); }
    dup2(fd, 1); dup2(fd, 2);
    char prog[1200]; path_of(prog, sizeof prog, argv[0]);
    execv(prog, argv);
    fprintf(stderr, "lfm3_speak: cannot exec %s\n", prog); _exit(127);
  }
  int st = 0;
  if (waitpid(pid, &st, 0) < 0) die("waitpid failed");
  if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) die("%s failed (exit %d) -- see %s",
      argv[0], WIFEXITED(st) ? WEXITSTATUS(st) : -1, out);
}
/* last row of a frame-major f32 file -> dst (proves the file has >= 1 row) */
static void tail_row_f32(const char *p, float *dst, int cols) {
  long n = file_size(p);
  if (n < (long)cols * 4 || n % ((long)cols * 4) != 0)
    die("%s is %ld bytes, not rows of %d floats", p, n, cols);
  FILE *f = fopen(p, "rb"); if (!f) die("cannot open %s", p);
  fseek(f, n - (long)cols * 4, SEEK_SET);
  if (fread(dst, 4, (size_t)cols, f) != (size_t)cols) die("short read of %s tail", p);
  fclose(f);
}
static int argmax_f32(const float *v, int n) {
  int b = 0; for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i; return b;
}
static uint32_t lcg_state;
static uint32_t lcg_next(void) { lcg_state = lcg_state * 1103515245u + 12345u; return (lcg_state >> 16) & 0x7fffu; }

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s <dir> <prompt_ids.i32> <mode> <out_prefix> [max_slots] [order_dir] [seed]\n", argv[0]);
    return 2;
  }
  snprintf(DIR, sizeof DIR, "%s", argv[1]);
  const char *prompt_path = argv[2];
  const char *mode = argv[3];
  const char *outp = argv[4];
  int max_slots = (argc > 5) ? atoi(argv[5]) : 150;
  const char *order_dir = (argc > 6) ? argv[6] : NULL;
  lcg_state = (uint32_t)((argc > 7) ? atol(argv[7]) : 1234);
  int is_audio = !strcmp(mode, "audio"), is_il = !strcmp(mode, "interleaved"), is_tx = !strcmp(mode, "text");
  if (!is_audio && !is_il && !is_tx) die("mode must be audio, interleaved or text (got %s)", mode);
  if (max_slots < 1 || max_slots > 4096) die("max_slots %d out of range", max_slots);
  char BB[1024], DB[1024], CB[1024];
  path_of(BB, sizeof BB, "lfm3-audio-2b-q8.lfm3q8");
  path_of(DB, sizeof DB, "lfm3-audio-2b-depth.lfm3q8");
  path_of(CB, sizeof CB, "lfm3-audio-2b-codec.lfm3q8");
  if (file_size(BB) <= 0) die("missing %s", BB);
  if (file_size(DB) <= 0) die("missing %s", DB);
  long pn = file_size(prompt_path);
  if (pn <= 0 || pn % 4 != 0) die("%s is %ld bytes, not int32 ids", prompt_path, pn);
  int nprompt = (int)(pn / 4);

  snprintf(TMPD, sizeof TMPD, "/tmp/lfm3_speak.%ld.d", (long)getpid());
  if (mkdir(TMPD, 0755) != 0) die("cannot create %s", TMPD);

  /* ---- token buffer: prompt + marker + slots -------------------------------- */
  int cap = nprompt + 8 + max_slots + 8;
  int32_t *toks = xmalloc((size_t)cap * 4);
  FILE *fp = fopen(prompt_path, "rb"); if (!fp) die("cannot open %s", prompt_path);
  if (fread(toks, 4, (size_t)nprompt, fp) != (size_t)nprompt) die("short read of %s", prompt_path);
  fclose(fp);
  int ntoks = nprompt;
  if (is_audio) toks[ntoks++] = AUDIO_START;
  else if (is_il) toks[ntoks++] = MIXED_START;
  printf("prompt: %d ids + %s marker -> %d prefix tokens\n",
         nprompt, is_tx ? "no" : (is_audio ? "audio_start" : "mixed_start"), ntoks);

  /* ---- mode machine (HF _generate, one row; see lfm3_ctrl.c) ----------------- */
  int interleaving = is_il, speaking = is_audio, text_done = 0, slots_left = NTEXT;
  int nframes = 0, ntext = 0, eos_seen = 0;
  int32_t *frames = xmalloc((size_t)max_slots * NCB * 4);
  int32_t *texts = xmalloc((size_t)max_slots * 4);
  char embfile[1100]; snprintf(embfile, sizeof embfile, "%s.embeds.f32", TMPD);
  { FILE *e = fopen(embfile, "wb"); if (!e) die("cannot create %s", embfile); fclose(e); }

  char csv[1100 * 64], lgf[1100], hdf[1100], hid1[1100], ordf[1100],
       histf[1100], codef[1100], embf[1100], nanolog[1100];
  snprintf(nanolog, sizeof nanolog, "%s.nanode.log", TMPD);

  for (int slot = 0; slot < max_slots; slot++) {
    int want_audio = interleaving ? speaking : is_audio;
    if (is_tx) want_audio = 0;
    if (want_audio && text_done && !interleaving) want_audio = 0;  /* unreachable guard */

    /* prefix csv for this slot's backbone launch */
    size_t used = 0;
    for (int i = 0; i < ntoks; i++)
      used += (size_t)snprintf(csv + used, sizeof csv - used, "%s%d", i ? "," : "", toks[i]);
    if (used >= sizeof csv - 64) die("prefix csv overflow at %d tokens", ntoks);
    char steps[32]; snprintf(steps, sizeof steps, "%d", ntoks);
    snprintf(lgf, sizeof lgf, "%s.last.f32", TMPD);
    snprintf(hdf, sizeof hdf, "%s.hidden.f32", TMPD);

    /* frame feedback for this launch: all previous frames, in slot order (the file
       only ever holds previous frames -- the current slot's token is not in toks yet) */
    char *av_semb[] = { (char*)"--speak-emb", embfile, NULL };
    int have_emb = nframes > 0;
    if (!want_audio) {
      /* ---- text slot: greedy argmax of the last logits row ------------------ */
      char *av[] = { (char*)"lfm3_nanode", BB, (char*)"--prompt", csv,
                     (char*)"--steps", steps,
                     have_emb ? av_semb[0] : (char*)"--dump-logits", have_emb ? av_semb[1] : lgf,
                     have_emb ? (char*)"--dump-logits" : NULL, have_emb ? lgf : NULL, NULL };
      run_cap(av, nanolog);
      need_file(lgf, 4, "last-step logits");
      long ln = file_size(lgf);
      int V = (int)(ln / 4);
      float *lv = xmalloc((size_t)ln);
      FILE *lf = fopen(lgf, "rb");
      if (fread(lv, 1, (size_t)ln, lf) != (size_t)ln) die("short read of %s", lgf);
      fclose(lf);
      int tok = argmax_f32(lv, V);
      free(lv);
      /* plausibility: sampled id inside the text vocabulary, never a codebook id */
      if (tok < 0 || tok >= 128000) die("slot %d sampled out-of-range id %d", slot, tok);
      toks[ntoks++] = tok;
      texts[ntext++] = tok;
      printf("slot %d T %d%s\n", slot, tok,
             tok == IM_END ? " (im_end, turn ends)" : (tok == TEXT_END ? " (text_end)" : ""));
      fflush(stdout);
      if (tok == IM_END) break;
      if (interleaving && tok == TEXT_END) { text_done = 1; speaking = 1; }
    } else {
      /* ---- audio slot: last hidden -> MaskGIT -> codes -> embed feedback ---- */
      char *av[] = { (char*)"lfm3_nanode", BB, (char*)"--prompt", csv,
                     (char*)"--steps", steps,
                     have_emb ? av_semb[0] : (char*)"--dump-hidden-all", have_emb ? av_semb[1] : hdf,
                     have_emb ? (char*)"--dump-hidden-all" : NULL, have_emb ? hdf : NULL, NULL };
      run_cap(av, nanolog);
      need_file(hdf, (long)HDIM * 4, "hidden dump");
      float *h = xmalloc((size_t)HDIM * 4);
      tail_row_f32(hdf, h, HDIM);
      for (int i = 0; i < HDIM; i++)
        if (!(h[i] == h[i])) die("slot %d hidden not finite at dim %d", slot, i);
      snprintf(hid1, sizeof hid1, "%s.hid.f32", TMPD);
      FILE *hf = fopen(hid1, "wb");
      if (!hf || fwrite(h, 4, HDIM, hf) != HDIM) die("cannot write %s", hid1);
      fclose(hf);
      free(h);
      /* order: prescribed (gated) or drawn (free) */
      if (order_dir) {
        snprintf(ordf, sizeof ordf, "%s/order.%d.i32", order_dir, nframes);
        if (file_size(ordf) != NCB * 4)
          die("slot %d frame %d: %s is %ld bytes, want %d (schedule [4,4,4])",
              slot, nframes, ordf, file_size(ordf), NCB * 4);
      } else {
        int perm[NCB]; for (int i = 0; i < NCB; i++) perm[i] = i;
        for (int i = NCB - 1; i > 0; i--) { int j = (int)(lcg_next() % (uint32_t)(i + 1)); int t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
        snprintf(ordf, sizeof ordf, "%s.ord.i32", TMPD);
        FILE *of = fopen(ordf, "wb");
        if (!of || fwrite(perm, 4, NCB, of) != NCB) die("cannot write %s", ordf);
        fclose(of);
      }
      /* penalty history: previous frames, oldest dropped past the 4096 window */
      int hn = nframes > 4096 ? 4096 : nframes;
      snprintf(histf, sizeof histf, "%s.hist.i32", TMPD);
      FILE *hf2 = fopen(histf, "wb");
      if (!hf2) die("cannot create %s", histf);
      if (hn > 0 && fwrite(frames + (nframes - hn) * NCB, 4, (size_t)hn * NCB, hf2) != (size_t)hn * NCB)
        die("cannot write %s", histf);
      fclose(hf2);
      snprintf(codef, sizeof codef, "%s.codes.i32", TMPD);
      char pen[32], win[32];
      snprintf(pen, sizeof pen, "1.1"); snprintf(win, sizeof win, "4096");
      char *av2[] = { (char*)"lfm3_depth", DB, hid1, (char*)"--maskgit",
                      ordf, histf, codef, pen, win, NULL };
      run_cap(av2, nanolog);
      need_file(codef, (long)NCB * 4, "frame codes");
      int32_t codes[NCB];
      FILE *cf = fopen(codef, "rb");
      if (fread(codes, 4, NCB, cf) != NCB) die("short read of %s", codef);
      fclose(cf);
      int iseos = 0;
      for (int i = 0; i < NCB; i++) {
        if (codes[i] < 0 || codes[i] >= CBS) die("slot %d frame %d codebook %d code %d out of range",
                                                 slot, nframes, i, codes[i]);
        if (codes[i] == AUDIO_EOS) iseos = 1;
      }
      /* HF's rule: any codebook hitting EOS overrides the WHOLE frame (found when
         raw-vs-returned codes gave cos 0.529 in the --embeds gate). */
      if (iseos) for (int i = 0; i < NCB; i++) codes[i] = AUDIO_EOS;
      /* embed feedback row for the next backbone launch */
      snprintf(embf, sizeof embf, "%s.emb1.f32", TMPD);
      char one[32]; snprintf(one, sizeof one, "1");
      char *av3[] = { (char*)"lfm3_depth", DB, (char*)"--embeds", codef, embf, one, NULL };
      run_cap(av3, nanolog);
      need_file(embf, (long)HDIM * 4, "frame embedding");
      FILE *e1 = fopen(embf, "rb"), *ea = fopen(embfile, "ab");
      if (!e1 || !ea) die("cannot append frame embedding");
      float row[HDIM];
      if (fread(row, 4, HDIM, e1) != HDIM) die("short read of %s", embf);
      if (fwrite(row, 4, HDIM, ea) != HDIM) die("cannot append to %s", embfile);
      fclose(e1); fclose(ea);
      memcpy(frames + nframes * NCB, codes, sizeof codes);
      printf("slot %d A frame %d codes = [%d,%d,%d,%d,...] %s\n",
             slot, nframes, codes[0], codes[1], codes[2], codes[3],
             iseos ? "(EOS FRAME, mode -> text)" : "");
      fflush(stdout);
      { /* per-frame record for first-divergence forensics */
        char rec[1100]; snprintf(rec, sizeof rec, "%s.frame%d.codes.i32", outp, nframes);
        FILE *rf = fopen(rec, "wb");
        if (rf) { fwrite(codes, 4, NCB, rf); fclose(rf); }
      }
      toks[ntoks++] = AUDIO_OUT;
      nframes++;
      if (iseos) { eos_seen = 1; interleaving = 0; }
    }

    /* HF: a slot in an interleaved block, before text_done, counts down; at 0 it flips */
    if (interleaving && !text_done) {
      if (--slots_left == 0) {
        speaking = !speaking;
        slots_left = speaking ? NAUDIO : NTEXT;
      }
    }
    if (ntoks + 2 > cap) die("slot %d: token buffer exhausted", slot);
  }

  /* ---- records ---------------------------------------------------------------- */
  char sp[1100]; snprintf(sp, sizeof sp, "%s.slots.txt", outp);
  FILE *sf = fopen(sp, "w"); if (!sf) die("cannot write %s", sp);
  /* replay the slot kinds from the token stream: AUDIO_OUT marks frames */
  {
    int fi = 0, ti = 0;
    for (int i = nprompt + (is_tx ? 0 : 1); i < ntoks; i++) {
      if (toks[i] == AUDIO_OUT) {
        int iseos = 1;
        for (int c = 0; c < NCB; c++) if (frames[fi * NCB + c] != AUDIO_EOS) { iseos = 0; break; }
        fprintf(sf, "A %d\n", iseos); fi++;
      } else { fprintf(sf, "T %d\n", toks[i]); ti++; }
    }
    if (fi != nframes) die("slot record mismatch: %d audio slots, %d frames", fi, nframes);
  }
  fclose(sf);
  char tc[1100]; snprintf(tc, sizeof tc, "%s.text.i32", outp);
  FILE *tf = fopen(tc, "wb");
  if (!tf || (ntext > 0 && fwrite(texts, 4, (size_t)ntext, tf) != (size_t)ntext)) die("cannot write %s", tc);
  fclose(tf);
  char fr[1100]; snprintf(fr, sizeof fr, "%s.frames.i32", outp);
  FILE *ff = fopen(fr, "wb");
  if (!ff || (nframes > 0 && fwrite(frames, 4, (size_t)nframes * NCB, ff) != (size_t)nframes * NCB))
    die("cannot write %s", fr);
  fclose(ff);
  printf("SPEAK DONE slots=%d text=%d audio=%d eos=%d\n", ntext + nframes, ntext, nframes, eos_seen);
  printf("records: %s.{slots.txt,text.i32,frames.i32}\n", outp);
  fflush(stdout);
  free(toks); free(frames); free(texts);
  return 0;
}
