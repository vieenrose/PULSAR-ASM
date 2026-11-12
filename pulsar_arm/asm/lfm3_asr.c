/* lfm3_asr.c -- S7-lite: one command from WAV to transcript, no Python, no HF, no /bin/sh.
 *
 *   lfm3_asr <dir> <clip.wav> [max_gen]
 *
 * <dir> holds the gated pieces, exactly as shipped:
 *   lfm3_wav lfm3_mel lfm3_enc_gpu lfm3_nanode          (executables)
 *   mel.bin lfm3-audio-2b-enc.lfm3q8 lfm3-audio-2b-q8.lfm3q8 tok.bin
 *   asr_prompt.txt asr_manifest.txt                      (from asr_manifest.py)
 *
 * Stages: WAV -> PCM -> log-mel -> GPU encoder -> prompt ids built natively (sha256
 * verified, tokenizer from the S1 blob) -> backbone with the audio-embedding override ->
 * detokenized transcript.
 *
 * Every stage is the SAME binary the gates certify, run by execv with an argv array:
 * no shell, so no quoting, no word splitting, no globbing -- the clip path is passed as
 * one argument. Stages stay separate processes on purpose: collapsing the four mains
 * into one binary is S7 proper, and doing it by editing gated sources would trade a
 * verified pipeline for an unverified one. The shell prints a "STAGE n" line per stage
 * and stops on the first non-zero exit -- a stage that silently produced nothing is the
 * failure mode this pipeline has already been bitten by, so each stage's output file is
 * size-checked before the next stage runs.
 *
 * The prompt is built here, not handed in: text from asr_prompt.txt, sha256 checked
 * against asr_manifest.txt, encoded by the S1 tokenizer, split on the audio placeholder,
 * and the placeholder expanded to prompt_audio_placeholder_tokens copies of
 * audio_token_id. The totals are checked against the manifest (1642 ids, 1622 audio) and
 * the encoder's row count must equal 1622 -- an id list one token off shifts every audio
 * embedding by a position, which is the bug class that produces a subtly wrong transcript.
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
#include <signal.h>
#include <errno.h>
#include <time.h>
#include <pthread.h>
#include "lfm3_tok.h"

/* The tokenizer off the turn's critical path. Loading tok.bin (5.4 MB: a malloc per vocab
   entry plus two sorts) and encoding the prompt text costs 0.85 s on the A57, against 0.45 s
   for the resident encoder's request that it used to overlap -- so the tokenizer, not the
   encoder, decided when the backbone could start (measured by the "TIME tokenizer" line). It
   now runs on a thread started before stage 1. Its two products are needed at different times:
   the prompt ids before the backbone starts, the vocabulary only for the detokenizer ~5 s
   later. The ids are a pure function of (prompt text, tok.bin), so they are cached on disk
   keyed by the prompt's sha256 and tok.bin's size + mtime; a turn that hits the cache does not
   wait for the thread, and the thread's own encode is compared with the cached ids before the
   transcript is printed, so a stale cache cannot produce a transcript -- it dies and is
   removed. LFM_ASR_IDCACHE=0 ignores the cache (the A/B control: join, then use the ids). */
typedef struct {
  const char *path; LTok tok; const uint8_t *prompt; size_t pre_len, post_off, post_len;
  uint32_t *pre, *post; int cap, npre, npost, rc;
} TokJob;
static void *tok_job(void *a) {
  TokJob *j = a;
  memset(&j->tok, 0, sizeof j->tok);
  if (ltok_load(&j->tok, j->path) != 0) { j->rc = 1; return NULL; }
  j->npre = ltok_encode(&j->tok, j->prompt, j->pre_len, j->pre, j->cap);
  j->npost = ltok_encode(&j->tok, j->prompt + j->post_off, j->post_len, j->post, j->cap);
  j->rc = (j->npre < 0 || j->npost < 0) ? 2 : 0;
  return NULL;
}
typedef struct { char magic[8]; char sha[72]; int64_t size, mt_s, mt_ns; int32_t npre, npost; } IdCacheHdr;

/* ------------------------------------------------------------------ sha256 (FIPS 180-4)
 * The manifest carries the HF golden's sha256 for the prompt text, so verifying it here
 * is what stops a stale, truncated or hand-edited prompt file from reaching the model.
 */
static const uint32_t K256[64] = {
  0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
  0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
  0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
  0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
  0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
  0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
  0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
  0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u };
#define ROT(x,n) (((x) >> (n)) | ((x) << (32 - (n))))
static void sha256_block(uint32_t *h, const uint8_t *p) {
  uint32_t w[64], a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
  for (int i = 0; i < 16; i++)
    w[i] = ((uint32_t)p[i*4]<<24)|((uint32_t)p[i*4+1]<<16)|((uint32_t)p[i*4+2]<<8)|p[i*4+3];
  for (int i = 16; i < 64; i++) {
    uint32_t s0 = ROT(w[i-15],7)^ROT(w[i-15],18)^(w[i-15]>>3);
    uint32_t s1 = ROT(w[i-2],17)^ROT(w[i-2],19)^(w[i-2]>>10);
    w[i] = w[i-16] + s0 + w[i-7] + s1;
  }
  for (int i = 0; i < 64; i++) {
    uint32_t S1 = ROT(e,6)^ROT(e,11)^ROT(e,25);
    uint32_t ch = (e & f) ^ ((~e) & g);
    uint32_t t1 = hh + S1 + ch + K256[i] + w[i];
    uint32_t S0 = ROT(a,2)^ROT(a,13)^ROT(a,22);
    uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
    uint32_t t2 = S0 + mj;
    hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
  }
  h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
}
static void sha256_hex(const uint8_t *msg, size_t n, char out[65]) {
  uint32_t h[8] = {0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
                   0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
  size_t full = n / 64, i;
  for (i = 0; i < full; i++) sha256_block(h, msg + i*64);
  uint8_t tailb[128]; memset(tailb, 0, sizeof tailb);
  size_t rem = n - full*64;
  memcpy(tailb, msg + full*64, rem);
  tailb[rem] = 0x80;
  size_t pad = (rem < 56) ? 64 : 128;
  uint64_t bits = (uint64_t)n * 8;
  for (int k = 0; k < 8; k++) tailb[pad - 1 - k] = (uint8_t)(bits >> (8*k));
  for (i = 0; i < pad/64; i++) sha256_block(h, tailb + i*64);
  for (i = 0; i < 8; i++) sprintf(out + i*8, "%08x", h[i]);
  out[64] = 0;
}

/* ------------------------------------------------------------------ helpers */
static char DIR[1024];
static char TMP[1024], TMPD[1024];
static int stage_no = 0;

static char WORK_MEL[1100], WORK_OUT[1100], WORK_MELP[1100];
static pid_t g_bb_pid = 0;    /* backbone child (started early), killed the same way */
static pid_t g_enc_pid = 0;   /* encoder child, killed if this shell dies before reaping it */
/* atexit: remove the per-run work files (PCM, mel, proj_out and its dir). Each run left
   ~115 MB under /tmp and 132 of them filled picard's root fs. The engine log (2 KB) stays.
   LFM_ASR_KEEP=1 keeps everything for debugging. */
static void cleanup_work(void) {
  if (getenv("LFM_ASR_KEEP")) return;
  if (TMP[0]) unlink(TMP);
  if (WORK_MEL[0]) unlink(WORK_MEL);
  if (WORK_MELP[0]) unlink(WORK_MELP);
  if (WORK_OUT[0]) unlink(WORK_OUT);
  if (TMPD[0]) rmdir(TMPD);
}
static void die(const char *fmt, ...) __attribute__((noreturn));
#include <stdarg.h>
static void die(const char *fmt, ...) {
  va_list ap; va_start(ap, fmt);
  fprintf(stderr, "lfm3_asr: "); vfprintf(stderr, fmt, ap); fputc('\n', stderr); va_end(ap);
  if (g_enc_pid > 0) { kill(g_enc_pid, SIGKILL); waitpid(g_enc_pid, NULL, 0); }
  if (g_bb_pid > 0) { kill(g_bb_pid, SIGKILL); waitpid(g_bb_pid, NULL, 0); }
  exit(2);
}
static void* xmalloc(size_t n) { void *p = malloc(n); if (!p) die("out of memory (%zu bytes)", n); return p; }

static void path_of(char *dst, size_t dn, const char *name) {
  snprintf(dst, dn, "%s/%s", DIR, name);
}
static long file_size(const char *p) {
  struct stat st; if (stat(p, &st) != 0) return -1; return (long)st.st_size;
}
static uint8_t* read_file(const char *p, size_t *len) {
  FILE *f = fopen(p, "rb"); if (!f) die("cannot open %s", p);
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  if (n < 0) die("cannot size %s", p);
  uint8_t *b = xmalloc((size_t)n + 1);
  if (fread(b, 1, (size_t)n, f) != (size_t)n) die("short read of %s", p);
  fclose(f); b[n] = 0; *len = (size_t)n; return b;
}
/* argv-only child: no shell, so the clip path is one argument and nothing is expanded.
   spawn + reap split so independent host work can run while a child does (run = both). */
static pid_t spawn(char *const argv[]) {
  fflush(stdout); fflush(stderr);
  pid_t pid = fork();
  if (pid < 0) die("fork failed");
  if (pid == 0) {
    char prog[1200]; path_of(prog, sizeof prog, argv[0]);
    execv(prog, argv);
    fprintf(stderr, "lfm3_asr: cannot exec %s\n", prog); _exit(127);
  }
  return pid;
}
static void reap(pid_t pid, const char *name) {
  int st = 0;
  if (waitpid(pid, &st, 0) < 0) die("waitpid failed");
  if (!WIFEXITED(st)) die("%s terminated by signal %d", name, WIFSIGNALED(st) ? WTERMSIG(st) : -1);
  if (WEXITSTATUS(st) != 0) die("%s exited %d", name, WEXITSTATUS(st));
}
static void run(char *const argv[]) { reap(spawn(argv), argv[0]); }
static pid_t spawn_capture(char *const argv[], const char *out) {
  fflush(stdout); fflush(stderr);
  pid_t pid = fork();
  if (pid < 0) die("fork failed");
  if (pid == 0) {
    int fd = open(out, O_WRONLY|O_CREAT|O_TRUNC, 0644);
    if (fd < 0) { fprintf(stderr, "lfm3_asr: cannot write %s\n", out); _exit(127); }
    dup2(fd, 1); dup2(fd, 2);
    char prog[1200]; path_of(prog, sizeof prog, argv[0]);
    execv(prog, argv);
    fprintf(stderr, "lfm3_asr: cannot exec %s\n", prog); _exit(127);
  }
  return pid;
}
static void reap_capture(pid_t pid, char *const argv[], const char *out) {
  int st = 0;
  if (waitpid(pid, &st, 0) < 0) die("waitpid failed");
  if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) die("%s failed (exit %d) -- see %s",
                                                  argv[0], WIFEXITED(st) ? WEXITSTATUS(st) : -1, out);
}
static double now_s(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
static long g_enc_n = -1, g_enc_off = 0;   // resident-encoder request index and its log offset
static double g_stage_t = -1.0;
/* Poll a resident-service log from a byte offset until one of its lines starts with `prefix`.
   The services print one flushed line per completed request (ENC_DONE n / CLIP n DONE), and the
   client is a different process each turn, so the log -- not a socket -- is the rendezvous. */
static int srv_wait(const char *logp, long from, const char *prefix, FILE *sink, double timeout_s) {
  int rf = open(logp, O_RDONLY);
  if (rf < 0) die("cannot open %s (%s)", logp, strerror(errno));
  if (lseek(rf, (off_t)from, SEEK_SET) != (off_t)from) { close(rf); die("%s seek to %ld", logp, from); }
  char lb[8192]; size_t lh = 0; int hit = 0;
  double t0 = now_s();
  for (;;) {
    ssize_t got = read(rf, lb + lh, sizeof lb - lh - 1);
    if (got <= 0) {
      if (now_s() - t0 > timeout_s) break;
      struct timespec ts = {0, 10000000}; nanosleep(&ts, NULL);
      continue;
    }
    lh += (size_t)got;
    char *nl;
    while ((nl = memchr(lb, '\n', lh))) {
      *nl = 0;
      if (sink) { fputs(lb, sink); fputc('\n', sink); }
      if (!strncmp(lb, prefix, strlen(prefix))) hit = 1;
      lh -= (size_t)(nl - lb) + 1;
      memmove(lb, nl + 1, lh);
    }
    if (hit) break;
  }
  close(rf);
  return hit;
}
static void stage(const char *what) {
  // Per-stage wall, so a turn's cost can be attributed without guessing: the ASR path is now
  // ~8 s and the question is always "how much of that is the encoder's process start".
  double t = now_s();
  double dt = (g_stage_t < 0) ? 0.0 : t - g_stage_t;   // same numbering as before the timer existed
  printf("STAGE %d  %s  (+%.3f s)\n", ++stage_no, what, dt);
  g_stage_t = t; fflush(stdout);
}

/* One turn's encoder, either way: a child process reaped, or a request to the resident encoder
   and a wait for its reply line. Both end with TMPD/proj_out.bin written and validated. */
static void enc_collect(const char *enc_log, long off, long idx) {
  if (enc_log) {
    char pfx[32]; snprintf(pfx, sizeof pfx, "ENC_DONE %ld ", idx);
    if (!srv_wait(enc_log, off, pfx, NULL, 900.0))
      die("the resident encoder never answered request %ld (see %s)", idx, enc_log);
  } else {
    pid_t ep = g_enc_pid; g_enc_pid = 0; reap(ep, "lfm3_enc_gpu");
  }
}

static void need_file(const char *p, long min_bytes, const char *what) {
  long n = file_size(p);
  if (n < min_bytes) die("%s produced %ld bytes at %s (expected >= %ld)", what, n, p, min_bytes);
}

/* manifest: key=value lines */
static const char* man_get(char *man, const char *key, char *buf, size_t bn) {
  size_t kl = strlen(key);
  char *p = man;
  while (p && *p) {
    if (!strncmp(p, key, kl) && p[kl] == '=') {
      char *e = strchr(p + kl + 1, '\n'); size_t n = e ? (size_t)(e - p - kl - 1) : strlen(p + kl + 1);
      if (n >= bn) n = bn - 1;
      memcpy(buf, p + kl + 1, n); buf[n] = 0;
      return buf;
    }
    p = strchr(p, '\n'); while (p && *p == '\n') p++;
  }
  return NULL;
}
static long man_long(char *man, const char *key) {
  char b[64]; if (!man_get(man, key, b, sizeof b)) die("manifest has no %s", key);
  return strtol(b, NULL, 10);
}

int main(int argc, char **argv) {
  const double t_main0 = now_s();
  if (argc >= 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) {
    fprintf(stderr, "usage: %s <dir> <clip.wav> [max_gen] [--out-ids FILE] [--tokens-only]\n", argv[0]);
    fprintf(stderr, "  <dir>         directory with the lfm3 blobs (backbone, enc, depth, codec, tok, mel)\n");
    fprintf(stderr, "  <clip.wav>    16 kHz mono WAV input (refused otherwise)\n");
    fprintf(stderr, "  [max_gen]     generated tokens (default 256)\n");
    fprintf(stderr, "  --out-ids FILE write generated ids as int32 (feeds lfm3_m3prompt -> lfm3_speak)\n");
    fprintf(stderr, "  --tokens-only  print IDS_PLAIN instead of the transcript\n");
    return 0;
  }
  if (argc < 3) {
    fprintf(stderr, "usage: %s <dir> <clip.wav> [max_gen] [--out-ids FILE] [--tokens-only]\n", argv[0]);
    fprintf(stderr, "  --out-ids FILE  write the generated ids as int32 (feeds lfm3_speak: M3)\n");
    fprintf(stderr, "  --tokens-only   print ids, skip the detokenized transcript\n");
    return 2;
  }
  snprintf(DIR, sizeof DIR, "%s", argv[1]);
  const char *wav = argv[2];
  int max_gen = (argc > 3) ? atoi(argv[3]) : 256;
  if (max_gen < 1) { fprintf(stderr, "max_gen must be >= 1\n"); return 2; }
  /* --out-ids / --tokens-only complete the speech->speech chain: the transcript ids this
     shell generates are exactly the prompt ids lfm3_speak consumes, so M3 is this program
     followed by that one, with no Python in between. */
  const char *out_ids = NULL;
  int tokens_only = 0;
  /* Resident-backbone service client. --serve-in/--serve-log name the fifo and the log file of
     a `lfm3_nanode <blob> --clip-stdin` process that is ALREADY running (the harness starts it
     once). Everything else about the turn is unchanged -- mel, encoder, prompt build, collector,
     detokenizer all run exactly as in the one-shot path -- so the per-turn ids are the same bits
     the cold path produces, and the goldens gate the two against each other. It exists because
     the 1.946 GB backbone costs 21.2 s of eMMC read (92 MB/s at any thread count: device-bound,
     run #257) plus ~8 s of parse/CRC/upload per process, against ~5 s of actual compute. */
  const char *srv_in = NULL, *srv_log = NULL;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--serve-in") && i + 1 < argc) srv_in = argv[++i];
    else if (!strcmp(argv[i], "--serve-log") && i + 1 < argc) srv_log = argv[++i];
  }
  if ((srv_in == NULL) != (srv_log == NULL)) die("--serve-in and --serve-log come as a pair");
  /* Same deal for the audio encoder: its process costs ~1.9 s of CUDA init + blob CRC/upload
     against a ~0.2 s pass, so a session keeps one lfm3_enc_gpu --serve-fifo alive and hands it a
     request per turn. --enc-req is a file holding the request counter (each turn is its own
     client process, so the number cannot live in this process's memory). */
  const char *enc_in = NULL, *enc_log = NULL;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--enc-in") && i + 1 < argc) enc_in = argv[++i];
    else if (!strcmp(argv[i], "--enc-log") && i + 1 < argc) enc_log = argv[++i];
  }
  if ((enc_in == NULL) != (enc_log == NULL)) die("--enc-in and --enc-log come as a pair");
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--out-ids") && i + 1 < argc) out_ids = argv[++i];
    else if (!strcmp(argv[i], "--tokens-only")) tokens_only = 1;
  }
  if (max_gen < 1 || max_gen > 4096) die("max_gen %d out of range 1..4096", max_gen);
  if (file_size(wav) < 44) die("%s is missing or smaller than a WAV header", wav);

  snprintf(TMP, sizeof TMP, "/tmp/lfm3_asr.%ld.pcm.f32", (long)getpid());
  char MEL[1024], OUT[1024], PROMPT[1024], MANI[1024], TOKB[1024], MELB[1024],
       ENCB[1024], BB[1024];
  snprintf(MEL, sizeof MEL, "%s", TMP); strcat(MEL, ".mel.f32");
  snprintf(TMPD, sizeof TMPD, "/tmp/lfm3_asr.%ld.d", (long)getpid());
  snprintf(OUT, sizeof OUT, "%s/proj_out.bin", TMPD);
  path_of(PROMPT, sizeof PROMPT, "asr_prompt.txt");
  path_of(MANI, sizeof MANI, "asr_manifest.txt");
  path_of(TOKB, sizeof TOKB, "tok.bin");
  path_of(MELB, sizeof MELB, "mel.bin");
  path_of(ENCB, sizeof ENCB, "lfm3-audio-2b-enc.lfm3q8");
  path_of(BB, sizeof BB, "lfm3-audio-2b-q8.lfm3q8");
  for (int i = 0; i < 4; i++) {
    const char *need = (i == 0) ? PROMPT : (i == 1) ? MANI : (i == 2) ? TOKB : MELB;
    if (file_size(need) <= 0) die("missing %s in %s (run .auto/lfm3/asr_manifest.py; mel.bin/tok.bin ship with the model dir)", need, DIR);
  }
  if (mkdir(TMPD, 0755) != 0 && file_size(TMPD) < 0) die("cannot create %s", TMPD);
  snprintf(WORK_MEL, sizeof WORK_MEL, "%s", MEL); snprintf(WORK_OUT, sizeof WORK_OUT, "%s", OUT);
  atexit(cleanup_work);

  /* ---- manifest + prompt text, sha256 verified ------------------------------ */
  size_t mlen; uint8_t *manb = read_file(MANI, &mlen);
  char *man = (char*)manb;
  char want_sha[128], got_sha[65];
  if (!man_get(man, "sha256", want_sha, sizeof want_sha)) die("manifest has no sha256");
  size_t plen; uint8_t *prompt = read_file(PROMPT, &plen);
  sha256_hex(prompt, plen, got_sha);
  if (strcmp(want_sha, got_sha) != 0)
    die("%s sha256 is %s, manifest demands %s -- refusing to run on an unverified prompt",
        PROMPT, got_sha, want_sha);
  long text_len = man_long(man, "prompt_text_len");
  if ((long)plen != text_len) die("%s is %ld bytes, manifest says %ld", PROMPT, (long)plen, text_len);
  char ph[64]; if (!man_get(man, "audio_placeholder", ph, sizeof ph)) die("manifest has no audio_placeholder");
  long audio_id = man_long(man, "audio_token_id");
  long n_audio = man_long(man, "prompt_audio_placeholder_tokens");
  long n_total = man_long(man, "prompt_tokens_total");
  long audio_at = man_long(man, "audio_at");
  long eos = man_long(man, "eos_token_id");
  char *split = strstr((char*)prompt, ph);
  if (!split) die("placeholder %s not found in %s", ph, PROMPT);
  size_t pre_len = (size_t)(split - (char*)prompt);
  size_t post_off = pre_len + strlen(ph);
  size_t post_len = plen - post_off;
  if (strstr(split + 1, ph)) die("placeholder %s occurs more than once; the manifest describes a single-slot prompt", ph);
  printf("prompt: %ld bytes, sha256 verified, placeholder at %zu, %ld audio rows at position %ld\n",
         (long)plen, pre_len, n_audio, audio_at);

  /* tokenizer thread: started here so that it overlaps stages 1-3 (see TokJob) */
  static TokJob tj; pthread_t tok_th;
  tj.path = TOKB; tj.prompt = prompt; tj.pre_len = pre_len; tj.post_off = post_off; tj.post_len = post_len;
  tj.cap = (int)(plen * 2 + 16);
  tj.pre = xmalloc((size_t)tj.cap * 4); tj.post = xmalloc((size_t)tj.cap * 4);
  if (pthread_create(&tok_th, NULL, tok_job, &tj) != 0) die("cannot start the tokenizer thread");
  int tok_joined = 0;

  /* ---- stage 1: WAV -> PCM ---------------------------------------------------- */
  /* The encoder starts first with Tm = auto: its CUDA init + blob load (~1.9 s) overlap
     stages 1-2, and it waits for MEL, which appears only by rename after lfm3_mel
     succeeded and the frame count was validated below. */
  unlink(MEL);   /* a leftover from a killed run with a recycled pid must not be read as this clip's mel */
  if (!enc_in) {
    char *av[] = { (char*)"lfm3_enc_gpu", ENCB, MEL, (char*)"auto", TMPD, (char*)"--only", (char*)"proj_out", NULL };
    g_enc_pid = spawn(av);
  }
  char MELP[1100]; snprintf(MELP, sizeof MELP, "%s.part", MEL);   /* MEL was unlinked before the spawn */
  snprintf(WORK_MELP, sizeof WORK_MELP, "%s", MELP);

  stage("WAV -> PCM16/float32 mono 16 kHz");
  { char *av[] = { (char*)"lfm3_wav", (char*)wav, TMP, NULL }; run(av); }
  need_file(TMP, 4, "lfm3_wav");

  /* ---- stage 2: PCM -> log-mel ------------------------------------------------ */
  stage("log-mel front end");
  { char *av[] = { (char*)"lfm3_mel", MELB, TMP, MELP, NULL }; run(av); }
  need_file(MELP, 128 * 4, "lfm3_mel");
  long mel_bytes = file_size(MELP);
  if (mel_bytes % (128 * 4) != 0)
    die("%s is %ld bytes, not a multiple of the 128-row f32 frame (%ld)", MELP, mel_bytes, (long)(128*4));
  if (rename(MELP, MEL) != 0) die("cannot publish %s", MEL);
  if (enc_in) {
    /* Hand the mel to the resident encoder (the one-shot path pays ~1.9 s of CUDA init + blob
       CRC/upload per turn for a pass that itself takes ~0.2 s). The request index is the number
       of ENC_DONE lines already in the log: turns are serialised, so that count is this turn's
       index and waiting for "ENC_DONE <n> " cannot match an earlier turn's reply. */
    long nenc = 0;
    { int rf = open(enc_log, O_RDONLY);
      if (rf >= 0) {
        char b[4096]; ssize_t g; long prev = 0;
        while ((g = pread(rf, b, sizeof b, prev)) > 0) {
          prev += g;
          for (ssize_t i = 0; i + 9 <= g; i++) if (!memcmp(b + i, "ENC_DONE ", 9)) nenc++;
        }
        close(rf);
      } }
    g_enc_off = file_size(enc_log);
    int wf = open(enc_in, O_WRONLY | O_NONBLOCK);
    if (wf < 0) die("--enc-in %s: cannot open for writing (%s) -- is the resident encoder running?", enc_in, strerror(errno));
    char rq[2400];
    int rl = snprintf(rq, sizeof rq, "%s %s auto\n", MEL, TMPD);
    if (write(wf, rq, (size_t)rl) != rl) die("--enc-in write failed");
    close(wf);
    g_enc_n = nenc;
  }
  long Tm = mel_bytes / (128 * 4);
  printf("mel frames Tm = %ld (%.3f audio seconds; %ld audio tokens = ceil(Tm/8))\n",
         Tm, (double)Tm / 100.0, (Tm + 7) / 8);
  /* The audio slot holds one row per encoder output, ceil(Tm/8), for THIS clip. The
     manifest's prompt_audio_placeholder_tokens is the golden fixture's count: it still
     pins the text part (n_total - n_audio ids around the slot), so a different clip
     length changes only the slot, never the text ids. */
  const long n_text = n_total - n_audio;
  n_audio = (Tm + 7) / 8;
  n_total = n_text + n_audio;

  /* ---- stage 3: encoder tower ------------------------------------------------ */
  stage("audio encoder tower (GPU) -> 2048-d rows");
  /* already running (spawned before stage 1). --only proj_out: the 19 other stage dumps
     (~100 MB of D2H + eMMC writes) are gate instrumentation; this shell reads proj_out
     alone. checks_enc.sh keeps the full set. */
  /* Stage 4 needs only the prompt text and the manifest, not the encoder's rows, so it
     runs here while the encoder child works; the encoder is reaped and its output checked
     before stage 5. */

  /* ---- stage 4: prompt ids, built natively ----------------------------------- */
  stage("ASR prompt ids (S1 tokenizer, manifest-expanded audio slot)");
  int cap = tj.cap;
  uint32_t *pre_ids = xmalloc((size_t)cap * 4), *post_ids = xmalloc((size_t)cap * 4);
  int npre = -1, npost = -1, id_cached = 0;
  char IDC[1100]; path_of(IDC, sizeof IDC, "asr_prompt.idcache");
  IdCacheHdr want; memset(&want, 0, sizeof want);
  { struct stat ts; if (stat(TOKB, &ts) != 0) die("cannot stat %s", TOKB);
    memcpy(want.magic, "LFMIDC01", 8); snprintf(want.sha, sizeof want.sha, "%s", got_sha);
    want.size = (int64_t)ts.st_size; want.mt_s = (int64_t)ts.st_mtim.tv_sec; want.mt_ns = (int64_t)ts.st_mtim.tv_nsec; }
  if (!(getenv("LFM_ASR_IDCACHE") && !atoi(getenv("LFM_ASR_IDCACHE")))) {
    FILE *cf = fopen(IDC, "rb");
    if (cf) {
      IdCacheHdr h;
      if (fread(&h, sizeof h, 1, cf) == 1 && !memcmp(h.magic, want.magic, 8) && !strcmp(h.sha, want.sha) &&
          h.size == want.size && h.mt_s == want.mt_s && h.mt_ns == want.mt_ns &&
          h.npre >= 0 && h.npost >= 0 && h.npre <= cap && h.npost <= cap &&
          fread(pre_ids, 4, (size_t)h.npre, cf) == (size_t)h.npre &&
          fread(post_ids, 4, (size_t)h.npost, cf) == (size_t)h.npost) {
        npre = h.npre; npost = h.npost; id_cached = 1;
      }
      fclose(cf);
    }
  }
  if (!id_cached) {
    pthread_join(tok_th, NULL); tok_joined = 1;
    if (tj.rc == 1) die("cannot load tokenizer blob %s", TOKB);
    npre = tj.npre; npost = tj.npost;
    if (npre < 0 || npost < 0) die("tokenizer failed on the prompt (pre %d, post %d)", npre, npost);
    memcpy(pre_ids, tj.pre, (size_t)npre * 4); memcpy(post_ids, tj.post, (size_t)npost * 4);
    /* publish by rename, so a concurrent turn never reads a half-written cache; a model
       directory that is not writable just means no cache */
    char tmpc[1200]; snprintf(tmpc, sizeof tmpc, "%s.%ld", IDC, (long)getpid());
    FILE *cf = fopen(tmpc, "wb");
    if (cf) {
      want.npre = npre; want.npost = npost;
      int okw = fwrite(&want, sizeof want, 1, cf) == 1 &&
                fwrite(pre_ids, 4, (size_t)npre, cf) == (size_t)npre &&
                fwrite(post_ids, 4, (size_t)npost, cf) == (size_t)npost;
      okw = (fclose(cf) == 0) && okw;
      if (!okw || rename(tmpc, IDC) != 0) unlink(tmpc);
    }
  }
  printf("prompt ids from %s\n", id_cached ? "the id cache (tokenizer still loading; verified before the transcript)" : "the tokenizer");
  long n_ids = (long)npre + n_audio + npost;
  if (n_ids != n_total)
    die("built %ld ids (%d + %ld audio + %d), manifest says %ld -- refusing",
        n_ids, npre, n_audio, npost, n_total);
  if (npre != audio_at)
    die("%d ids precede the audio slot, manifest says audio starts at %ld", npre, audio_at);
  uint32_t *ids = xmalloc((size_t)n_ids * 4);
  memcpy(ids, pre_ids, (size_t)npre * 4);
  for (long i = 0; i < n_audio; i++) ids[npre + i] = (uint32_t)audio_id;
  memcpy(ids + npre + n_audio, post_ids, (size_t)npost * 4);
  char *idstr = xmalloc((size_t)n_ids * 12);
  size_t used = 0;
  for (long i = 0; i < n_ids; i++)
    used += (size_t)sprintf(idstr + used, "%s%u", i ? "," : "", ids[i]);
  printf("prompt ids: %ld built, %ld audio rows, matches the manifest totals\n", n_ids, n_audio);
  if (getenv("LFM_ASR_PRINT_IDS")) printf("IDS %s\n", idstr);

  /* ---- stage 5: backbone ----------------------------------------------------- */
  char embp[1100]; snprintf(embp, sizeof embp, "--audio-emb");
  char ata[64], atv[32], stq[64], stv[32], pp[64];
  snprintf(ata, sizeof ata, "--audio-at"); snprintf(atv, sizeof atv, "%ld", audio_at);
  /* --steps is the TOTAL step count, prompt steps included (checks_asr.py uses
     n_prompt + len(want) + 2): the head only runs once s >= nprompt-1, so passing
     the generation budget alone makes every step a prompt step and every printed
     token a -1. Cost a full run to learn; the engine prints -1 rather than a fake id. */
  /* The last prompt step (index n_ids - 1) already predicts generated id 1, so max_gen ids
     need n_ids + max_gen - 1 steps; one more step only computed an id that stage 6 drops. */
  snprintf(stq, sizeof stq, "--steps"); snprintf(stv, sizeof stv, "%ld", n_ids + (long)max_gen - 1);
  printf("steps: %ld prompt + %d generation = %s (the last prompt step yields id 1)\n", n_ids, max_gen, stv);
  /* lfm3_nanode's context cap (MAXTOK): refuse here, before 20+ s of encoder work is
     wasted on a run the engine would truncate. */
  if (n_ids + (long)max_gen > 4096)
    die("%ld prompt ids + %d generation exceed the engine's 4096-position context "
        "(%.0f s of audio max at this prompt)", n_ids, max_gen, (4096.0 - max_gen - n_text) * 8 / 100.0);
  snprintf(pp, sizeof pp, "--prompt");
  char eng[1100]; snprintf(eng, sizeof eng, "/tmp/lfm3_asr.%ld.engine.txt", (long)getpid());
  /* --prefill-run T: the prompt goes through the backbone T positions at a time (one
     T-wide launch per GEMV, sharing each weight load across positions; the per-position
     passes run the T=1 kernels in the T=1 order). Bit-identical to the T=1 prefill on the
     full ASR prompt (all generated ids and the final-step logits, byte for byte), ~30%
     faster per prompt position. LFM_ASR_PFRUN=0 runs the T=1 path (A/B control). */
  char pfr[32], pfv[32];
  const int pfrun = getenv("LFM_ASR_PFRUN") ? atoi(getenv("LFM_ASR_PFRUN")) : 32;
  snprintf(pfr, sizeof pfr, "--prefill-run"); snprintf(pfv, sizeof pfv, "%d", pfrun);
  /* Stop the greedy loop at the transcript's eos. Stage 6 has always broken its collection
     there, so the steps between the eos and max_gen computed ids that no one read: on the 9 s
     fixtures max_gen=96 against transcripts of 9 (zh) / 18 (en) ids, ~8-9 s of the turn.
     LFM_ASR_STOPID=0 sends -1 (run to max_gen, the A/B control arm). */
  char siq[32], siv[32];
  snprintf(siq, sizeof siq, "--stop-id");
  snprintf(siv, sizeof siv, "%ld",
           (getenv("LFM_ASR_STOPID") && !atoi(getenv("LFM_ASR_STOPID"))) ? -1L : eos);
  /* LFM_ASR_EARLYBB (default 1): the backbone starts now, while the encoder still runs.
     Its CUDA init, blob parse and the upload of layers 0-1 overlap the encoder; it then
     waits (LFM_AEMB_WAIT) for PUB, which appears by rename only after the encoder was
     reaped and its row count validated below, and uploads nothing past layer 1 until then
     (memory: the encoder's activations plus the whole backbone do not fit, run 281). */
  const int early = !(getenv("LFM_ASR_EARLYBB") && !atoi(getenv("LFM_ASR_EARLYBB")));
  char PUB[1100]; snprintf(PUB, sizeof PUB, "%s.ready", OUT);
  unlink(PUB);
  char *av[] = { (char*)"lfm3_nanode", BB, embp, early ? PUB : OUT, ata, atv, pp, idstr, stq, stv,
                 siq, siv,
                 pfrun >= 2 ? pfr : NULL, pfv, NULL };
  if (early && !srv_in) { setenv("LFM_AEMB_WAIT", "1", 1); g_bb_pid = spawn_capture(av, eng); unsetenv("LFM_AEMB_WAIT"); }
  { double t = now_s(); printf("TIME prompt ids ready %.3f s after the encoder request\n", t - g_stage_t); }
  enc_collect(enc_log, g_enc_off, g_enc_n);
  stage("encoder reply");   // splits the pre-backbone window: request+GPU pass vs everything else
  need_file(OUT, n_audio * 2048 * 4, "lfm3_enc_gpu");
  { long rows = file_size(OUT) / (2048 * 4);
    if (rows != n_audio) die("encoder produced %ld rows, the prompt expands the placeholder %ld times",
                             rows, n_audio); }
  stage("backbone prefill + greedy decode");
  if (srv_in) {
    /* Resident-backbone arm: hand this clip to the live service and collect its answer.
       The spec carries emb, ids, audio-at, ngen and the stop id; only its PATH goes down the
       fifo, then the log is read from the size it had before the write -- the service is blocked
       in fgets() right now (the previous turn's client consumed its DONE line), so nothing it
       writes can be missed and nothing from an earlier turn is re-read. Everything downstream
       (stage 6's collector, the detokenizer, the goldens) is the one-shot path's, unchanged. */
    char specf[1100]; snprintf(specf, sizeof specf, "/tmp/lfm3_asr.%ld.spec.txt", (long)getpid());
    unlink(specf);
    FILE* sf = fopen(specf, "w");
    if (!sf) die("cannot write %s", specf);
    fprintf(sf, "%s\n%s\n%ld\n%d\n%s\n", OUT, idstr, audio_at, max_gen,
            (getenv("LFM_ASR_STOPID") && !atoi(getenv("LFM_ASR_STOPID"))) ? "-1" : siv);
    fclose(sf);
    long s0 = file_size(srv_log);
    int wf = open(srv_in, O_WRONLY | O_NONBLOCK);
    if (wf < 0) die("--serve-in %s: cannot open for writing (%s) -- is the resident backbone running?", srv_in, strerror(errno));
    size_t sl = strlen(specf);
    if (write(wf, specf, sl) != (ssize_t)sl || write(wf, "\n", 1) != 1) die("--serve-in write failed");
    close(wf);
    int rf = open(srv_log, O_RDONLY);
    if (rf < 0) die("cannot open %s", srv_log);
    if (lseek(rf, (off_t)s0, SEEK_SET) != (off_t)s0) die("--serve-log seek");
    FILE* ef = fopen(eng, "w");
    if (!ef) die("cannot write %s", eng);
    char lb[8192]; size_t lh = 0; int done = 0; double wt0 = now_s();
    while (!done) {
      ssize_t got = read(rf, lb + lh, sizeof lb - lh - 1);
      if (got <= 0) {
        if (now_s() - wt0 > 900.0) die("the resident backbone produced nothing for 900 s (see %s)", srv_log);
        struct timespec ts = {0, 20000000}; nanosleep(&ts, NULL);
        continue;
      }
      lh += (size_t)got;
      char* nl;
      while ((nl = memchr(lb, '\n', lh))) {
        *nl = 0;
        fputs(lb, ef); fputc('\n', ef);
        if (strncmp(lb, "CLIP ", 5) == 0 && strstr(lb, "DONE steps=")) done = 1;
        lh -= (size_t)(nl - lb) + 1;
        memmove(lb, nl + 1, lh);
      }
    }
    fclose(ef); close(rf);
  } else if (early) {
    if (rename(OUT, PUB) != 0) die("cannot publish %s", PUB);
    snprintf(WORK_OUT, sizeof WORK_OUT, "%s", PUB);
    pid_t bp = g_bb_pid; g_bb_pid = 0; reap_capture(bp, av, eng);
  } else {
    reap_capture(spawn_capture(av, eng), av, eng);
  }
  size_t elen; uint8_t *eb = read_file(eng, &elen);

  /* ---- stage 6: collect the generated ids, detokenize ------------------------ */
  stage("detokenize");
  uint32_t *gen = xmalloc((size_t)(max_gen + 8) * 4);
  int ngen = 0;
  char *line = strtok((char*)eb, "\n");
  while (line) {
    long step, tok_id; double ms, tps;
    /* the engine's step lines are "  <step>  <token>  <ms>  <tps>"; gen steps are the
       ones whose index has passed the prompt, and token -1 marks a skipped head. */
    if (sscanf(line, "%ld %ld %lf %lf", &step, &tok_id, &ms, &tps) == 4 &&
        step >= n_total - 1 && tok_id >= 0) {
      if (tok_id == eos) break;
      if (ngen < max_gen) gen[ngen++] = (uint32_t)tok_id;
    }
    line = strtok(NULL, "\n");
  }
  if (ngen == 0) die("no generated tokens in %s -- refusing to print an empty transcript", eng);
  size_t dcap = (size_t)ngen * 16 + 64;
  uint8_t *txt = xmalloc(dcap);
  if (!tok_joined) { pthread_join(tok_th, NULL); tok_joined = 1; }
  if (tj.rc == 1) die("cannot load tokenizer blob %s", TOKB);
  if (id_cached && (tj.rc != 0 || tj.npre != npre || tj.npost != npost ||
                    memcmp(tj.pre, pre_ids, (size_t)npre * 4) || memcmp(tj.post, post_ids, (size_t)npost * 4))) {
    unlink(IDC);
    die("%s disagreed with the tokenizer's own encode of the prompt -- removed it; no transcript "
        "is printed from ids the tokenizer did not produce", IDC);
  }
  LTok tok = tj.tok;
  int dn = ltok_decode(&tok, gen, (size_t)ngen, txt, dcap);
  if (dn < 0) die("detokenizer failed on %d ids", ngen);
  txt[dn] = 0;
  if (out_ids) {
    FILE *f = fopen(out_ids, "wb");
    if (!f) die("cannot write %s", out_ids);
    if (fwrite(gen, 4, (size_t)ngen, f) != (size_t)ngen) die("short write to %s", out_ids);
    fclose(f);
    printf("IDS -> %s (%d ids, %ld bytes)\n", out_ids, ngen, (long)ngen * 4);
  }
  if (tokens_only) {
    printf("IDS_PLAIN");
    for (int i = 0; i < ngen; i++) printf(" %u", gen[i]);
    printf("\n");
  } else {
    printf("TRANSCRIPT (%d tokens, %.1f chars/KB): %s\n", ngen, (double)dn, txt);
  }
  free(eb); free(txt); free(gen); free(ids); free(pre_ids); free(post_ids);
  free(prompt); free(manb); ltok_free(&tok);
  printf("TIME client total %.3f s (process start to exit)\n", now_s() - t_main0);
  return 0;
}
