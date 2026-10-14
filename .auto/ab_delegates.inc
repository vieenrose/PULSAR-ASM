/* Appended to an OLD kernel TU by ab_kernel.sh when the working-tree glue calls
 * kernel entry points that do not exist in the old TU (run #47 added r256, #45
 * r128, #42 tm). Each delegates to the oldest kernel so the old binary behaves
 * exactly like the baseline: every GEMV goes through the baseline kernel.
 * Without this the audit link fails with undefined symbols, which is why #33's
 * audit (GITREF=902dabc) worked before these entry points existed. */
extern "C" int pq1_gemv_sm53_launch(const unsigned char *, const unsigned short *,
                                    const unsigned char *, float *, int, int, void *);
extern "C" int pq1_gemv_r128_launch(const unsigned char *a, const unsigned short *b,
                                    const unsigned char *c, float *d, int e, int f,
                                    void *g) {
  return pq1_gemv_sm53_launch(a, b, c, d, e, f, g);
}
extern "C" int pq1_gemv_r256_launch(const unsigned char *a, const unsigned short *b,
                                    const unsigned char *c, float *d, int e, int f,
                                    void *g) {
  return pq1_gemv_sm53_launch(a, b, c, d, e, f, g);
}
extern "C" int pq1_gemv_tm_launch(const unsigned char *a, const unsigned short *b,
                                  const unsigned int *c, const int *d, float *e,
                                  int f, int g, void *h) {
  (void)c; (void)d;
  return pq1_gemv_sm53_launch(a, b, (const unsigned char *)c, e, f, g, h);
}
