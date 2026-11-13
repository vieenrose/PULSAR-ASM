// memstream TH SECS MB -- TH threads each sum a private MB-megabyte buffer in a loop for SECS seconds
// (read-only streaming, the traffic shape of a CPU GEMV share at one position). Prints total GB/s.
// Probe for run #312: does CPU memory traffic slow the bus-bound GPU decode step?
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static double secs; static size_t mb; static double bytes[8]; static uint64_t sink[8];
static void* run(void* a) {
  long id = (long)a; size_t n = mb << 20; uint64_t* p = malloc(n); memset(p, 1, n);
  double t0 = now(); uint64_t s = 0; double b = 0;
  while (now() - t0 < secs) { for (size_t i = 0; i < n / 8; i++) s += p[i]; b += n; }
  bytes[id] = b / (now() - t0); sink[id] = s; free(p); return 0;
}
int main(int c, char** v) {
  int th = c > 1 ? atoi(v[1]) : 3; secs = c > 2 ? atof(v[2]) : 5; mb = c > 3 ? atoi(v[3]) : 96;
  pthread_t t[8]; for (long i = 0; i < th; i++) pthread_create(&t[i], 0, run, (void*)i);
  double tot = 0; for (int i = 0; i < th; i++) { pthread_join(t[i], 0); tot += bytes[i]; }
  printf("memstream %d threads x %zu MB: %.2f GB/s (sink %llu)\n", th, mb, tot / 1e9, (unsigned long long)sink[0]);
  return 0;
}
