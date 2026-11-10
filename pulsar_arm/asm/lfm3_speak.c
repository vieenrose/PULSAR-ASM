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
 * After the last slot: the non-EOS frames go through lfm3_codec (24 kHz f32) and
 * lfm3_wavout (RIFF/WAVE), so <out_prefix>.wav is the deliverable. Trailing all-EOS
 * frames carry no audio and HF's decode rejects codes >= 2016, so they are dropped.
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
#include <signal.h>
#include <glob.h>

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
/* Stage timing (printed as STIME lines at exit): wall seconds per stage kind, keyed by
   program and first flag, plus the backbone server step waits. Timing only. */
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
#define NTK 16
static char tk_name[NTK][64]; static double tk_s[NTK]; static int tk_n[NTK], ntk = 0;
static void tk_add(const char *name, double s) {
  int i; for (i = 0; i < ntk; i++) if (!strcmp(tk_name[i], name)) break;
  if (i == ntk) { if (ntk == NTK) return; snprintf(tk_name[ntk++], 64, "%s", name); }
  tk_s[i] += s; tk_n[i]++;
}
static double t_start;
static void tk_dump(void) {
  for (int i = 0; i < ntk; i++) printf("STIME %-28s calls %4d  total %9.3f s  mean %8.3f s\n", tk_name[i], tk_n[i], tk_s[i], tk_s[i] / tk_n[i]);
  printf("STIME wall %.3f s\n", now_s() - t_start);
}
static void run_cap_t(char *const argv[], const char *out);
static void run_cap(char *const argv[], const char *out) {
  const double t0 = now_s();
  run_cap_t(argv, out);
  char k[64]; snprintf(k, sizeof k, "%s %s", argv[0], argv[1] && argv[1][0] == '-' ? argv[1] : (argv[2] && argv[2][0] == '-' ? argv[2] : ""));
  tk_add(k, now_s() - t0);
}
static void run_cap_t(char *const argv[], const char *out) {
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
/* ---- resident backbone (nanode --serve) --------------------------------------------
 * One nanode process for the whole turn: it loads the blob once, keeps KV/conv state,
 * and after each step reads the next input from its stdin (T <id> | A <row.f32> | Q).
 * Every line it prints goes to the nanode log as well, for forensics. */
static pid_t srv_pid = -1;
static FILE *srv_in = NULL, *srv_out = NULL, *srv_log = NULL;
static void serve_start(char *const argv[], const char *logpath) {
  int pin[2], pout[2];
  if (pipe(pin) != 0 || pipe(pout) != 0) die("pipe failed");
  srv_pid = fork();
  if (srv_pid < 0) die("fork failed");
  if (srv_pid == 0) {
    dup2(pin[0], 0); dup2(pout[1], 1); dup2(pout[1], 2);
    close(pin[0]); close(pin[1]); close(pout[0]); close(pout[1]);
    char prog[1200]; path_of(prog, sizeof prog, argv[0]);
    execv(prog, argv);
    fprintf(stderr, "lfm3_speak: cannot exec %s\n", prog); _exit(127);
  }
  close(pin[0]); close(pout[1]);
  srv_in = fdopen(pin[1], "w");
  srv_out = fdopen(pout[0], "r");
  srv_log = fopen(logpath, "w");
  if (!srv_in || !srv_out || !srv_log) die("cannot set up the backbone server pipes");
}
/* Read server lines until the step line for `want` (exactly "step tok ms tps"); returns
   that step's argmax token. Dies on server exit or on a step past `want` (lost sync). */
static int serve_wait_step_t(int want);
static int serve_wait_step(int want) {
  const double t0 = now_s(); const int r = serve_wait_step_t(want); tk_add("backbone step (serve)", now_s() - t0); return r;
}
static int serve_wait_step_t(int want) {
  char line[4096];
  while (fgets(line, sizeof line, srv_out)) {
    fputs(line, srv_log); fflush(srv_log);
    int st, tk; double ms, tps; char extra;
    if (sscanf(line, "%d %d %lf %lf %c", &st, &tk, &ms, &tps, &extra) == 4) {
      if (st == want) return tk;
      if (st > want) die("backbone server is at step %d, expected %d (lost sync)", st, want);
    }
    if (!strncmp(line, "SERVE: ", 7) && strncmp(line, "SERVE: quit", 11))
      die("backbone server refused input: %s", line);
  }
  die("backbone server exited before step %d (see the nanode log)", want);
}
static void serve_send(const char *cmd) {
  if (fputs(cmd, srv_in) < 0 || fflush(srv_in) != 0) die("cannot write to the backbone server");
}
static void serve_stop(void) {
  if (srv_pid <= 0) return;
  serve_send("Q\n");
  fclose(srv_in); srv_in = NULL;
  char line[4096];
  while (fgets(line, sizeof line, srv_out)) fputs(line, srv_log);
  fclose(srv_out); fclose(srv_log);
  int st = 0;
  if (waitpid(srv_pid, &st, 0) < 0) die("waitpid on the backbone server failed");
  if (!WIFEXITED(st) || WEXITSTATUS(st) != 0)
    die("backbone server ended badly (exit %d)", WIFEXITED(st) ? WEXITSTATUS(st) : -1);
  srv_pid = -1;
}

/* ---- resident depth head (lfm3_depth --serve) ---------------------------------------
 * One depth process for the turn (LFM_SPK_DEPTHSRV, default 1; 0 = one lfm3_depth
 * --maskgit process per frame). Same frame computation, same files: codes and frame
 * embedding are the same bytes; the per-frame process start and blob read go away. */
/* ---- early codec (LFM_SPK_CODECEARLY, default 1) -------------------------------------
 * lfm3_codec <blob> <codes>.ready wait <out> starts with the turn: blob read, CUDA init and
 * weight upload overlap the slot loop; at the end the codes are written and published by
 * rename, and the process is reaped like a normal run. */
static pid_t codec_pid = -1;
static char codec_ready[1200], codec_out[1200], codec_log[1200];
static int codec_pub = 0; static double codec_pub_t = 0;   /* frames published at the EOS frame */
static void codec_early_start(const char *blob) {
  if (getenv("LFM_SPK_CODECEARLY") && !atoi(getenv("LFM_SPK_CODECEARLY"))) return;
  snprintf(codec_ready, sizeof codec_ready, "%s.codes_audio.i32.ready", TMPD);
  snprintf(codec_out, sizeof codec_out, "%s.audio24.f32", TMPD);
  snprintf(codec_log, sizeof codec_log, "%s.codec.log", TMPD);
  unlink(codec_ready);
  pid_t pid = fork();
  if (pid < 0) die("fork failed");
  if (pid == 0) {
    int fd = open(codec_log, O_WRONLY|O_CREAT|O_TRUNC, 0644);
    if (fd < 0) _exit(127);
    dup2(fd, 1); dup2(fd, 2);
    char prog[1200]; path_of(prog, sizeof prog, "lfm3_codec");
    char *av[] = { (char*)"lfm3_codec", (char*)blob, codec_ready, (char*)"wait", codec_out, NULL };
    execv(prog, av); _exit(127);
  }
  codec_pid = pid;
}
static pid_t dsrv_pid = -1;
static FILE *dsrv_in = NULL, *dsrv_out = NULL, *dsrv_log = NULL;
static void dsrv_start(const char *blob, const char *logpath) {
  int pin[2], pout[2];
  if (pipe(pin) != 0 || pipe(pout) != 0) die("pipe failed");
  dsrv_pid = fork();
  if (dsrv_pid < 0) die("fork failed");
  if (dsrv_pid == 0) {
    dup2(pin[0], 0); dup2(pout[1], 1); dup2(pout[1], 2);
    close(pin[0]); close(pin[1]); close(pout[0]); close(pout[1]);
    char prog[1200]; path_of(prog, sizeof prog, "lfm3_depth");
    char *av[] = { (char*)"lfm3_depth", (char*)blob, (char*)"--serve", NULL };
    execv(prog, av);
    fprintf(stderr, "lfm3_speak: cannot exec %s\n", prog); _exit(127);
  }
  close(pin[0]); close(pout[1]);
  dsrv_in = fdopen(pin[1], "w"); dsrv_out = fdopen(pout[0], "r"); dsrv_log = fopen(logpath, "w");
  if (!dsrv_in || !dsrv_out || !dsrv_log) die("cannot set up the depth server pipes");
}
static void dsrv_frame(const char *hid, const char *ord, const char *hist, const char *codes,
                       const char *emb, const char *pen, const char *win) {
  const double t0 = now_s();
  if (fprintf(dsrv_in, "M %s %s %s %s %s %s %s\n", hid, ord, hist, codes, emb, pen, win) < 0 ||
      fflush(dsrv_in) != 0) die("cannot write to the depth server");
  char line[4096];
  while (fgets(line, sizeof line, dsrv_out)) {
    fputs(line, dsrv_log); fflush(dsrv_log);
    int rc;
    if (sscanf(line, "DEPTH_DONE %d", &rc) == 1) {
      if (rc) die("depth server frame failed (rc %d) -- see its log", rc);
      tk_add("lfm3_depth (serve)", now_s() - t0);
      return;
    }
  }
  die("depth server ended unexpectedly -- see its log");
}
static void dsrv_stop(void) {
  if (dsrv_pid <= 0) return;
  fputs("Q\n", dsrv_in); fclose(dsrv_in); dsrv_in = NULL;
  char line[4096];
  while (fgets(line, sizeof line, dsrv_out)) fputs(line, dsrv_log);
  fclose(dsrv_out); fclose(dsrv_log);
  int st = 0;
  if (waitpid(dsrv_pid, &st, 0) < 0) die("waitpid on the depth server failed");
  if (!WIFEXITED(st) || WEXITSTATUS(st) != 0)
    die("depth server ended badly (exit %d)", WIFEXITED(st) ? WEXITSTATUS(st) : -1);
  dsrv_pid = -1;
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
  t_start = now_s(); atexit(tk_dump);
  const int depthsrv = getenv("LFM_SPK_DEPTHSRV") ? atoi(getenv("LFM_SPK_DEPTHSRV")) : 1;
  if (argc >= 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) {
    fprintf(stderr, "usage: %s <dir> <prompt_ids.i32> <mode> <out_prefix> [max_slots] [order_dir] [seed]\n", argv[0]);
    fprintf(stderr, "  <dir>         directory with the lfm3 blobs\n");
    fprintf(stderr, "  <prompt_ids.i32> int32 token ids (from lfm3_asr/m3prompt or a file)\n");
    fprintf(stderr, "  <mode>        audio | interleaved | text\n");
    fprintf(stderr, "  <out_prefix>  output path stem (slots.txt, text.i32, frames.i32, .wav)\n");
    fprintf(stderr, "  [max_slots]   slot budget (default 150)\n");
    fprintf(stderr, "  [order_dir]   dir with order.<f>.i32 reveal schedules (default: random)\n");
    fprintf(stderr, "  [seed]        RNG seed (default 1234)\n");
    fprintf(stderr, "  env: LFM_SPEAK_RELAUNCH=1  relaunch backbone per slot (A/B control arm)\n");
    return 0;
  }
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

  /* scratch for the per-slot hand-off files (hidden rows, orders, codes, embeddings):
     tmpfs when present (LFM_SPK_TMP overrides). On the Nano /tmp is ext4 on the eMMC, where
     the per-frame create/write/read cost ~4 ms of journaled metadata per frame. */
  { const char *tb = getenv("LFM_SPK_TMP");
    struct stat sb;
    if (!tb) tb = (stat("/dev/shm", &sb) == 0 && S_ISDIR(sb.st_mode) && access("/dev/shm", W_OK) == 0) ? "/dev/shm" : "/tmp";
    snprintf(TMPD, sizeof TMPD, "%s/lfm3_speak.%ld.d", tb, (long)getpid()); }
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
  snprintf(lgf, sizeof lgf, "%s.last.f32", TMPD);
  snprintf(hdf, sizeof hdf, "%s.hidden.f32", TMPD);

  /* Resident by default; LFM_SPEAK_RELAUNCH=1 is the A/B control arm (one nanode launch
     per slot, the path the first live gate ran). The two must produce identical records. */
  const int resident = getenv("LFM_SPEAK_RELAUNCH") == NULL;
  int last_nt = -1;
  char srvlog[1100]; snprintf(srvlog, sizeof srvlog, "%s.serve.log", TMPD);
  if (resident) {
    unlink(hdf);
    size_t u0 = 0;
    for (int i = 0; i < ntoks; i++)
      u0 += (size_t)snprintf(csv + u0, sizeof csv - u0, "%s%d", i ? "," : "", toks[i]);
    if (u0 >= sizeof csv - 64) die("prefix csv overflow at %d tokens", ntoks);
    char *avs[] = { (char*)"lfm3_nanode", BB, (char*)"--prompt", csv, (char*)"--serve",
                    (char*)"--dump-hidden-all", hdf, NULL };
    serve_start(avs, srvlog);
    /* the depth server loads while the backbone loads and prefills (its first frame used to
       pay the 169 MB read + 120 MB upload); same server, same frames */
    if (depthsrv && strcmp(mode, "text") != 0) { char dl[1100]; snprintf(dl, sizeof dl, "%s.depth.log", outp); dsrv_start(DB, dl); }
    if (strcmp(mode, "text") != 0) codec_early_start(CB);
    last_nt = serve_wait_step(ntoks - 1);
    printf("backbone: resident (nanode --serve, one load for the turn)\n");
  } else {
    printf("backbone: relaunch per slot (LFM_SPEAK_RELAUNCH=1)\n");
  }
  fflush(stdout);
  char feed[1300];

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
      int tok;
      if (resident) {
        tok = last_nt;    /* the server's argmax after the last fed step */
      } else {
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
      tok = argmax_f32(lv, V);
      free(lv);
      }
      /* plausibility: sampled id inside the text vocabulary, never a codebook id */
      if (tok < 0 || tok >= 128000) die("slot %d sampled out-of-range id %d", slot, tok);
      toks[ntoks++] = tok;
      texts[ntext++] = tok;
      snprintf(feed, sizeof feed, "T %d\n", tok);
      printf("slot %d T %d%s\n", slot, tok,
             tok == IM_END ? " (im_end, turn ends)" : (tok == TEXT_END ? " (text_end)" : ""));
      fflush(stdout);
      if (tok == IM_END) break;
      if (interleaving && tok == TEXT_END) { text_done = 1; speaking = 1; }
      /* HF: outside an interleaved block <|text_end|> ends the turn (it is in eos_token_id) */
      else if (!interleaving && tok == TEXT_END) break;
    } else {
      /* ---- audio slot: last hidden -> MaskGIT -> codes -> embed feedback ---- */
      if (!resident) {
      char *av[] = { (char*)"lfm3_nanode", BB, (char*)"--prompt", csv,
                     (char*)"--steps", steps,
                     have_emb ? av_semb[0] : (char*)"--dump-hidden-all", have_emb ? av_semb[1] : hdf,
                     have_emb ? (char*)"--dump-hidden-all" : NULL, have_emb ? hdf : NULL, NULL };
      run_cap(av, nanolog);
      }
      /* resident: the server appended this step's hidden before printing its step line */
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
      snprintf(embf, sizeof embf, "%s.emb1.f32", TMPD);
      char pen[32], win[32];
      snprintf(pen, sizeof pen, "1.1"); snprintf(win, sizeof win, "4096");
      /* One process per frame, not two: the maskgit call also emits the frame embedding
         (LFM_DEPTH_EMBEDS_OUT). Forking lfm3_depth twice paid the 1.3 s blob load twice
         per frame; the embedding itself is a 24k-entry table reduction. */
      if (depthsrv) {
        if (dsrv_pid <= 0) { char dl[1100]; snprintf(dl, sizeof dl, "%s.depth.log", outp); dsrv_start(DB, dl); }
        dsrv_frame(hid1, ordf, histf, codef, embf, pen, win);
      } else {
      setenv("LFM_DEPTH_EMBEDS_OUT", embf, 1);
      char *av2[] = { (char*)"lfm3_depth", DB, hid1, (char*)"--maskgit",
                      ordf, histf, codef, pen, win, NULL };
      run_cap(av2, nanolog);
      }
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
      /* the embedding came out of the maskgit process above */
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
      snprintf(feed, sizeof feed, "A %s\n", embf);
      if (iseos) { eos_seen = 1; interleaving = 0; }
      /* after the EOS frame of an interleaved turn no audio slot can follow (the loop-top
         kind is then is_audio = 0), so the frames are final: publish them to the waiting
         codec now, overlapping the remaining text slot(s) (LFM_SPK_CODECEOS, default 1). */
      if (iseos && !is_audio && !interleaving && codec_pid > 0 && !codec_pub &&
          !(getenv("LFM_SPK_CODECEOS") && !atoi(getenv("LFM_SPK_CODECEOS")))) {
        int na = nframes;
        while (na > 0) {
          int ae = 1;
          for (int c = 0; c < NCB; c++) if (frames[(na - 1) * NCB + c] != AUDIO_EOS) { ae = 0; break; }
          if (!ae) break;
          na--;
        }
        if (na > 0) {
          char cf[1100]; snprintf(cf, sizeof cf, "%s.codes_audio.i32", TMPD);
          FILE *cd = fopen(cf, "wb");
          if (!cd || fwrite(frames, 4, (size_t)na * NCB, cd) != (size_t)na * NCB) die("cannot write %s", cf);
          fclose(cd);
          if (rename(cf, codec_ready) != 0) die("cannot publish %s", codec_ready);
          codec_pub = na; codec_pub_t = now_s();
        }
      }
    }

    /* HF: a slot in an interleaved block, before text_done, counts down; at 0 it flips */
    if (interleaving && !text_done) {
      if (--slots_left == 0) {
        speaking = !speaking;
        slots_left = speaking ? NAUDIO : NTEXT;
      }
    }
    if (ntoks + 2 > cap) die("slot %d: token buffer exhausted", slot);
    /* resident: hand this slot's input to the server and collect the next step. Skipped
       after the final slot -- nothing would consume that step. */
    if (resident && slot + 1 < max_slots) {
      /* the next slot's kind, by the expression at the loop top: if it is audio, only the
         hidden row of this step is read, so the server may skip its lm_head (lowercase
         command). LFM_SPK_NOHEAD=0 always asks for the head. */
      int nxt = interleaving ? speaking : is_audio;
      if (is_tx) nxt = 0;
      if (nxt && text_done && !interleaving) nxt = 0;
      if (nxt && !(getenv("LFM_SPK_NOHEAD") && !atoi(getenv("LFM_SPK_NOHEAD")))) feed[0] = (char)(feed[0] + 32);
      serve_send(feed);
      last_nt = serve_wait_step(ntoks - 1);
    }
  }

  if (resident) serve_stop();
  dsrv_stop();

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

  /* ---- audio: codes -> codec (24 kHz f32) -> WAV ---------------------------------- */
  int n_audio_frames = nframes;
  while (n_audio_frames > 0) {
    int all_eos = 1;
    for (int c = 0; c < NCB; c++)
      if (frames[(n_audio_frames - 1) * NCB + c] != AUDIO_EOS) { all_eos = 0; break; }
    if (!all_eos) break;
    n_audio_frames--;
  }
  if (n_audio_frames <= 0 && codec_pid > 0) { kill(codec_pid, SIGTERM); waitpid(codec_pid, NULL, 0); codec_pid = -1; }
  if (n_audio_frames > 0) {
    stage("codec decode -> 24 kHz");
    char cf[1100], af[1100];
    snprintf(cf, sizeof cf, "%s.codes_audio.i32", TMPD);
    FILE *cd = fopen(cf, "wb");
    if (!cd) die("cannot write %s", cf);
    if (fwrite(frames, 4, (size_t)n_audio_frames * NCB, cd) != (size_t)n_audio_frames * NCB)
      die("cannot write %s", cf);
    fclose(cd);
    snprintf(af, sizeof af, "%s.audio24.f32", TMPD);
    char nf2[32]; snprintf(nf2, sizeof nf2, "%d", n_audio_frames);
    if (codec_pid > 0) {
      double t0 = now_s();
      if (codec_pub) {
        if (codec_pub != n_audio_frames) die("early codec got %d frames, turn has %d", codec_pub, n_audio_frames);
        unlink(cf); t0 = codec_pub_t;
      } else
      if (rename(cf, codec_ready) != 0) die("cannot publish %s", codec_ready);
      int st = 0;
      if (waitpid(codec_pid, &st, 0) < 0) die("waitpid codec failed");
      codec_pid = -1;
      if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) die("lfm3_codec failed (exit %d) -- see %s",
          WIFEXITED(st) ? WEXITSTATUS(st) : -1, codec_log);
      tk_add("lfm3_codec (early)", now_s() - t0);
      snprintf(af, sizeof af, "%s", codec_out);
    } else {
    char *avc[] = { (char*)"lfm3_codec", CB, cf, nf2, af, NULL };
    run_cap(avc, nanolog);
    }
    need_file(af, (long)n_audio_frames * 1920 * 4, "codec decode");
    /* the codec emits frame_samples=1920 per frame at 24 kHz; a short file means a
       partial decode, which would play as a truncated clip rather than an error */
    { long got = file_size(af);
      long want = (long)n_audio_frames * 1920 * 4;
      if (got != want) die("codec wrote %ld bytes, expected %ld (%d frames x 1920 x 4)",
                           got, want, n_audio_frames); }
    stage("WAV 24 kHz");
    char wv[1200], rt[32];
    snprintf(wv, sizeof wv, "%s.wav", outp);
    snprintf(rt, sizeof rt, "24000");
    char *avw[] = { (char*)"lfm3_wavout", af, wv, rt, NULL };
    run_cap(avw, nanolog);
    need_file(wv, 44 + 100, "wav writer");
    printf("AUDIO %s.wav from %d frames (%.2f s at 24 kHz, %d EOS frame(s) dropped)\n",
           outp, n_audio_frames, n_audio_frames * 1920.0 / 24000.0, nframes - n_audio_frames);
    fflush(stdout);
  } else {
    printf("AUDIO: no non-EOS frames -- no WAV written\n");
  }
  free(toks); free(frames); free(texts);
  /* a successful turn removes its scratch (on tmpfs it is RAM); failures die() before this
     and keep it for forensics; LFM_SPK_KEEP=1 keeps it always */
  if (!(getenv("LFM_SPK_KEEP") && atoi(getenv("LFM_SPK_KEEP")))) {
    /* the scratch is the directory TMPD plus the sibling files TMPD.<name> */
    char pat[1300]; glob_t gl;
    snprintf(pat, sizeof pat, "%s/*", TMPD);
    if (glob(pat, 0, NULL, &gl) == 0) { for (size_t i = 0; i < gl.gl_pathc; i++) unlink(gl.gl_pathv[i]); globfree(&gl); }
    rmdir(TMPD);
    snprintf(pat, sizeof pat, "%s.*", TMPD);
    if (glob(pat, 0, NULL, &gl) == 0) { for (size_t i = 0; i < gl.gl_pathc; i++) unlink(gl.gl_pathv[i]); globfree(&gl); }
  }
  return 0;
}
