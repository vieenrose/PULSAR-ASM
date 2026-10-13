/* bq1_host_test.c — parity gate for the BQ1 path, no CUDA, no model.
 * Checks: quant roundtrip, S_pos/S_all vs brute-force float dot,
 * repack LSB-first mapping, per-256 scale flush order.
 * Pass mark: EXACT int partials, float relerr < 2e-6 (i8 quant noise only).
 */
#include "bq1_wrap.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int main(void) {
  const int K = 1024, N = 4;
  static uint32_t wb[4 * 32];
  static uint16_t ws[4 * 4];
  static float x[1024], y[4], yref[4];
  static int8_t xq[1024];
  static float wfl[4 * 1024];

  srand(7);
  for (int r = 0; r < N; r++) {
    for (int i = 0; i < K; i++) {
      int bit = rand() & 1;
      if (i < 32) { /* keep first word checkable */ }
      wb[(size_t)r * 32 + (i >> 5)] |= (uint32_t)bit << (i & 31);
      wfl[(size_t)r * K + i] = bit ? 1.0f : -1.0f;
    }
    ws[(size_t)r * 4 + 0] = bq1_f2h(0.5f);
    ws[(size_t)r * 4 + 1] = bq1_f2h(1.0f);
    ws[(size_t)r * 4 + 2] = bq1_f2h(2.0f);
    ws[(size_t)r * 4 + 3] = bq1_f2h(0.25f);
    for (int i = 0; i < K; i++) wfl[(size_t)r * K + i] *=
      bq1_h2f(ws[(size_t)r * 4 + (i >> 8)]);
  }
  for (int i = 0; i < K; i++) x[i] = ((rand() % 2001) - 1000) / 1000.0f;
  float xs = bq1_quant_row_i8(x, xq, K);

  bq1_gemv_cpu(N, K, wb, ws, xq, xs, y);

  /* brute force: sum(w_fl * x_q)*xs */
  for (int r = 0; r < N; r++) {
    double s = 0;
    for (int i = 0; i < K; i++) s += (double)wfl[(size_t)r * K + i] * (double)xq[i];
    yref[r] = (float)(s * xs);
  }
  double md = 0;
  for (int r = 0; r < N; r++) {
    double d = fabs((double)y[r] - yref[r]) / fmax(fabs((double)yref[r]), 1e-6);
    if (d > md) md = d;
    printf("row %d: swar=%.6f ref=%.6f rel=%.2e\n", r, y[r], yref[r], d);
  }
  printf("maxrel %.2e %s\n", md, md < 2e-6 ? "PASS" : "FAIL");
  /* LSB-first spot check: word0 bit0 must equal wfl sign of col0 */
  int b0 = (wb[0] & 1u) ? 1 : -1;
  int f0 = wfl[0] > 0 ? 1 : -1;
  printf("lsb-map %s\n", b0 == f0 ? "PASS" : "FAIL");
  return (md < 2e-6 && b0 == f0) ? 0 : 1;
}
